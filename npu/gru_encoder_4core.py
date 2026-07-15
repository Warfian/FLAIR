#
# gru_encoder_4core.py
#
# 4-core DATA-PARALLEL FLAIR encoder in one NPU column. Same kernel
# (gru_encoder_bf16) as the single-core gru_encoder.py, but replicated across
# 4 compute tiles that each process an INDEPENDENT slice of the window batch.
# Windows are independent (the GRU recurrence is sequential only WITHIN a
# window), so this is pure data parallelism -> ~4x throughput.
#
# Column data flow (memtile = L2 does all scatter/gather/broadcast):
#     shim --x_windows--> memtile --split--> 4 cores        (scatter windows)
#     shim --params-----> memtile --forward/bcast--> 4 cores (shared weights)
#     4 cores --latents--> memtile --join--> shim           (gather outputs)
#
# `batch` here is windows PER CORE per dispatch; one dispatch processes
# 4*batch windows total. The host (batch_infer.exe) just provides one
# contiguous 4*batch-window input buffer and reads one 4*batch-window output
# buffer -- the memtile split/join handle the per-core distribution, so no
# host-side changes are needed vs the single-core flow (only the volumes
# change: in1_vol = 4*batch*SEQ*INPUT_DIM, out_vol = 4*batch*HIDDEN).
#
# Patterns mirror mlir-aie programming_examples: reduce_max memtile
# (split/join) and matmul whole_array (forward + multi-.cons broadcast).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNELS_DIR = Path(__file__).parent / "kernels"
_KERNEL_SRC = _KERNELS_DIR / "gru_encoder.cc"

INPUT_DIM = 48
HIDDEN_DIM = 64
SEQ_LEN = 10
BATCH = 1        # windows PER CORE per dispatch
N_CORES = 4


def _make_encoder_kernel(arg_types, compile_flags):
    header_base = Path(config.cxx_header_path())
    runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
    lut_cpp = runtime_dir / "lut_based_ops.cpp"

    include_dirs = [
        str(header_base),
        str(header_base / "aie_kernels"),
        str(runtime_dir),
        str(_KERNELS_DIR),
    ]
    source = f'#include "{_KERNEL_SRC}"\n#include "{lut_cpp}"\n'

    return ExternalFunction(
        "gru_encoder_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )


@iron.jit
def gru_encoder_4core(
    x_windows: In,
    params: In,
    latents: Out,
    *,
    input_dim: CompileTime[int] = INPUT_DIM,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
    batch: CompileTime[int] = BATCH,
):
    h3 = 3 * hidden_dim
    n_params = h3 * input_dim + h3 * hidden_dim + h3 + h3

    per_core_win = batch * seq_len * input_dim   # x_window for one core's batch
    per_core_lat = batch * hidden_dim            # latent for one core's batch
    total_win = N_CORES * per_core_win
    total_lat = N_CORES * per_core_lat

    dtype = np.dtype[bfloat16]
    x_all_ty = np.ndarray[(total_win,), dtype]        # shim<->memtile (all cores)
    x_core_ty = np.ndarray[(per_core_win,), dtype]    # memtile->one core
    params_ty = np.ndarray[(n_params,), dtype]
    lat_all_ty = np.ndarray[(total_lat,), dtype]      # memtile<->shim (all cores)
    lat_core_ty = np.ndarray[(per_core_lat,), dtype]  # one core->memtile

    # The kernel each core runs is the UNCHANGED single-core encoder, sized to
    # `batch` windows per core.
    kernel = _make_encoder_kernel(
        arg_types=[x_core_ty, params_ty, lat_core_ty],
        compile_flags=[
            f"-DINPUT_DIM={input_dim}",
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
            f"-DBATCH={batch}",
        ],
    )

    # --- x_windows: shim -> memtile -> split to N cores (scatter) ---
    x_fifo = ObjectFifo(x_all_ty, name="x_windows")
    x_offsets = [per_core_win * i for i in range(N_CORES)]
    x_core_fifos = x_fifo.cons().split(
        x_offsets,
        obj_types=[x_core_ty] * N_CORES,
        names=[f"x_core{i}" for i in range(N_CORES)],
    )

    # --- params: shim -> memtile -> broadcast to all N cores (shared) ---
    params_fifo = ObjectFifo(params_ty, name="params")
    params_bcast = params_fifo.cons().forward(
        obj_type=params_ty, name="params_bcast"
    )

    # --- latents: N cores -> memtile -> join -> shim (gather) ---
    lat_fifo = ObjectFifo(lat_all_ty, name="latents")
    lat_offsets = [per_core_lat * i for i in range(N_CORES)]
    lat_core_fifos = lat_fifo.prod().join(
        lat_offsets,
        obj_types=[lat_core_ty] * N_CORES,
        names=[f"lat_core{i}" for i in range(N_CORES)],
    )

    def core_fn(x_c, params_c, lat_p, k):
        ew = x_c.acquire(1)
        ep = params_c.acquire(1)
        el = lat_p.acquire(1)
        k(ew, ep, el)
        x_c.release(1)
        params_c.release(1)
        lat_p.release(1)

    workers = []
    for i in range(N_CORES):
        workers.append(
            Worker(
                core_fn,
                [
                    x_core_fifos[i].cons(),
                    params_bcast.cons(),   # same fifo -> broadcast to every core
                    lat_core_fifos[i].prod(),
                    kernel,
                ],
            )
        )

    rt = Runtime()
    with rt.sequence(x_all_ty, params_ty, lat_all_ty) as (x_arg, params_arg, lat_arg):
        rt.start(*workers)
        rt.fill(x_fifo.prod(), x_arg)
        rt.fill(params_fifo.prod(), params_arg)
        rt.drain(lat_fifo.cons(), lat_arg, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="FLAIR GRU encoder (4-core data-parallel)")
    add_compile_args(p)
    p.add_argument("--input-dim", type=int, default=INPUT_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch", type=int, default=BATCH,
                   help="windows PER CORE per dispatch (total = 4*batch)")
    return p


def _compile_kwargs(opts):
    return dict(
        input_dim=opts.input_dim,
        hidden_dim=opts.hidden_dim,
        seq_len=opts.seq_len,
        batch=opts.batch,
    )


def _run_and_verify(opts):
    raise SystemExit(
        "Compile-only design (WSL). Run via batch_infer.exe on Windows."
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gru_encoder_4core,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        # One column = 1 shim + 1 memtile + 4 compute tiles -- exactly what the
        # 4-core data-parallel design needs.
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
