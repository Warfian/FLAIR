#
# gru_encoder_noop.py  -- DIAGNOSTIC ONLY
#
# IRON wrapper for kernels/gru_encoder.cc's gru_encoder_noop_bf16: SAME
# (x_window, params, latent) buffer signature/sizes and SAME ObjectFifo/DMA
# wiring as gru_encoder.py, but the kernel body does no gru_step (no real
# compute).
#
# Purpose: localize the encoder's ~145us unexplained per-window overhead. If
# this shows ~the decoder-noop floor (~32us/window), the overhead is in the
# gru_step/compute path (a codegen problem). If it stays high, the overhead is
# dispatch/DMA of the encoder's large x_window input (15x the decoder's per
# dispatch). Not used for scoring -- driven by diag_encoder_timing.py.
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
BATCH = 1


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
        "gru_encoder_noop_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )


@iron.jit
def gru_encoder_noop(
    x_window: In,
    params: In,
    latent: Out,
    *,
    input_dim: CompileTime[int] = INPUT_DIM,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
    batch: CompileTime[int] = BATCH,
):
    h3 = 3 * hidden_dim
    n_params = h3 * input_dim + h3 * hidden_dim + h3 + h3
    win_len = batch * seq_len * input_dim
    latent_len = batch * hidden_dim
    dtype = np.dtype[bfloat16]

    win_ty = np.ndarray[(win_len,), dtype]
    params_ty = np.ndarray[(n_params,), dtype]
    h_ty = np.ndarray[(latent_len,), dtype]

    kernel = _make_encoder_kernel(
        arg_types=[win_ty, params_ty, h_ty],
        compile_flags=[
            f"-DINPUT_DIM={input_dim}",
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
            f"-DBATCH={batch}",
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
    p = argparse.ArgumentParser(prog="FLAIR GRU encoder (AIE, no-op diagnostic)")
    add_compile_args(p)
    p.add_argument("--input-dim", type=int, default=INPUT_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch", type=int, default=BATCH,
                   help="windows processed per kernel invocation")
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
        gru_encoder_noop,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
