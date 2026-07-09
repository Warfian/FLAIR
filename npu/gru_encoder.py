#
# gru_encoder.py
#
# IRON driver for the fused FLAIR encoder kernel (npu/kernels/gru_encoder.cc):
# a full SEQ_LEN-timestep GRU encode in ONE kernel invocation, weights
# resident, hidden state carried on-core. Outputs the latent (last hidden).
#
# Buffers: x_window (SEQ_LEN*INPUT_DIM) + params (encoder GRU weights) -> 2
# inputs; latent (HIDDEN_DIM) -> 1 output. Fits the core's 2-in/2-out DMA
# budget and 64 KB L1 (params 42624 B + small buffers).
#
# Compile-only (WSL):
#   python3 gru_encoder.py --dev npu --xclbin-path build/gru.xclbin \
#           --insts-path build/insts.bin
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

# Dims from the trained checkpoint (encoder.gru.weight_ih_l0: (192, 45)).
INPUT_DIM = 45
HIDDEN_DIM = 64
SEQ_LEN = 10  # preprocess.window_size in config.yaml


def _make_encoder_kernel(arg_types, compile_flags):
    """Build the gru_encoder ExternalFunction with the include wiring the aie2
    LUT kernels rely on, plus the kernels dir (so gru_encoder.cc can
    #include "gru_common.h"). Compiles lut_based_ops.cpp into the same TU for
    getExpBf16's tables. Mirrors gru_cell_encoder._make_gru_kernel."""
    header_base = Path(config.cxx_header_path())
    runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
    lut_cpp = runtime_dir / "lut_based_ops.cpp"

    include_dirs = [
        str(header_base),                  # aie_api/aie.hpp etc.
        str(header_base / "aie_kernels"),  # aie_kernel_utils.h
        str(runtime_dir),                  # lut_based_ops.h
        str(_KERNELS_DIR),                 # gru_common.h
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
def gru_encoder(
    x_window: In,
    params: In,
    latent: Out,
    *,
    input_dim: CompileTime[int] = INPUT_DIM,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
):
    h3 = 3 * hidden_dim
    n_params = h3 * input_dim + h3 * hidden_dim + h3 + h3
    win_len = seq_len * input_dim  # 10*45 = 450 (even -> 4-byte DMA aligned)
    dtype = np.dtype[bfloat16]

    win_ty = np.ndarray[(win_len,), dtype]
    params_ty = np.ndarray[(n_params,), dtype]
    h_ty = np.ndarray[(hidden_dim,), dtype]

    kernel = _make_encoder_kernel(
        arg_types=[win_ty, params_ty, h_ty],
        compile_flags=[
            f"-DINPUT_DIM={input_dim}",
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
        ],
    )

    win_fifo = ObjectFifo(win_ty, depth=1, name="x_window")
    params_fifo = ObjectFifo(params_ty, depth=1, name="params")
    latent_fifo = ObjectFifo(h_ty, depth=1, name="latent")

    def core_fn(win_c, params_c, latent_p, k):
        ew = win_c.acquire(1)
        ep = params_c.acquire(1)
        el = latent_p.acquire(1)
        k(ew, ep, el)
        win_c.release(1)
        params_c.release(1)
        latent_p.release(1)

    worker = Worker(
        core_fn,
        [win_fifo.cons(), params_fifo.cons(), latent_fifo.prod(), kernel],
    )

    rt = Runtime()
    with rt.sequence(win_ty, params_ty, h_ty) as (win_arg, params_arg, latent_arg):
        rt.start(worker)
        rt.fill(win_fifo.prod(), win_arg)
        rt.fill(params_fifo.prod(), params_arg)
        rt.drain(latent_fifo.cons(), latent_arg, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="FLAIR GRU encoder (AIE)")
    add_compile_args(p)
    p.add_argument("--input-dim", type=int, default=INPUT_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    return p


def _compile_kwargs(opts):
    return dict(
        input_dim=opts.input_dim,
        hidden_dim=opts.hidden_dim,
        seq_len=opts.seq_len,
    )


def _run_and_verify(opts):
    raise SystemExit(
        "This design is intended for the WSL compile-only + Windows host flow "
        "(make run). Direct NPU execution from Python isn't supported here "
        "(WSL has no NPU)."
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gru_encoder,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
