#
# gru_decoder_fused.py
#
# IRON wrapper for npu/kernels/gru_decoder.cc's gru_decoder_fused_bf16 -- the
# decoder GRU sequence PLUS the hidden_to_output linear layer, fused on-core.
#
# Runs:
#   h0_vec + decoder GRU params + W_out/b_out -> recon (final reconstruction)
#
# Separate driver from gru_decoder.py (which returns the raw hidden_seq) so
# the single-window live-demo/verify flow stays on the unfused kernel,
# unaffected. This fused variant exists purely to shrink the decoder's
# per-window output footprint (BATCH*SEQ_LEN*OUTPUT_DIM instead of
# BATCH*SEQ_LEN*HIDDEN_DIM, ~3x smaller at OUTPUT_DIM=21), freeing L1 budget
# for a larger BATCH in the dataset-scale inference pipeline
# (run_dataset_inference.py).
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


NPU_DIR = Path(__file__).resolve().parent
KERNELS_DIR = NPU_DIR / "kernels"
KERNEL_SRC = KERNELS_DIR / "gru_decoder.cc"

HIDDEN_DIM = 64
SEQ_LEN = 10
OUTPUT_DIM = 21
# Windows processed per kernel invocation. params (weights, now including
# W_out/b_out) stay a single resident copy -- only h0_vec/recon scale with
# BATCH. Default 1 for parity with the unfused driver's default.
BATCH = 1


def _make_decoder_kernel(arg_types, compile_flags):
    header_base = Path(config.cxx_header_path())
    runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
    lut_cpp = runtime_dir / "lut_based_ops.cpp"

    include_dirs = [
        str(header_base),
        str(header_base / "aie_kernels"),
        str(runtime_dir),
        str(KERNELS_DIR),
    ]

    source = f'#include "{KERNEL_SRC}"\n#include "{lut_cpp}"\n'

    return ExternalFunction(
        "gru_decoder_fused_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )


@iron.jit
def gru_decoder_fused(
    h0_vec: In,
    params: In,
    recon: Out,
    *,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
    output_dim: CompileTime[int] = OUTPUT_DIM,
    batch: CompileTime[int] = BATCH,
):
    h3 = 3 * hidden_dim

    # decoder GRU input_dim = hidden_dim, because x_t = h0_vec.
    # params layout: [w_ih | w_hh | b_ih | b_hh | w_out | b_out]
    n_params = (
        h3 * hidden_dim  # w_ih
        + h3 * hidden_dim  # w_hh
        + h3  # b_ih
        + h3  # b_hh
        + output_dim * hidden_dim  # w_out
        + output_dim  # b_out
    )

    dtype = np.dtype[bfloat16]

    h0_ty = np.ndarray[(batch * hidden_dim,), dtype]
    params_ty = np.ndarray[(n_params,), dtype]
    recon_ty = np.ndarray[(batch * seq_len * output_dim,), dtype]

    kernel = _make_decoder_kernel(
        arg_types=[h0_ty, params_ty, recon_ty],
        compile_flags=[
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
            f"-DOUTPUT_DIM={output_dim}",
            f"-DBATCH={batch}",
        ],
    )

    h0_fifo = ObjectFifo(h0_ty, depth=1, name="decoder_h0")
    params_fifo = ObjectFifo(params_ty, depth=1, name="decoder_params")
    recon_fifo = ObjectFifo(recon_ty, depth=1, name="decoder_recon")

    def core_fn(h0_c, params_c, recon_p, k):
        eh0 = h0_c.acquire(1)
        ep = params_c.acquire(1)
        erec = recon_p.acquire(1)

        k(eh0, ep, erec)

        h0_c.release(1)
        params_c.release(1)
        recon_p.release(1)

    worker = Worker(
        core_fn,
        [h0_fifo.cons(), params_fifo.cons(), recon_fifo.prod(), kernel],
    )

    rt = Runtime()

    with rt.sequence(h0_ty, params_ty, recon_ty) as (
        h0_arg,
        params_arg,
        recon_arg,
    ):
        rt.start(worker)
        rt.fill(h0_fifo.prod(), h0_arg)
        rt.fill(params_fifo.prod(), params_arg)
        rt.drain(recon_fifo.cons(), recon_arg, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="FLAIR decoder GRU sequence + fused hidden_to_output")
    add_compile_args(p)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--output-dim", type=int, default=OUTPUT_DIM)
    p.add_argument("--batch", type=int, default=BATCH,
                   help="windows processed per kernel invocation")
    return p


def _compile_kwargs(opts):
    return dict(
        hidden_dim=opts.hidden_dim,
        seq_len=opts.seq_len,
        output_dim=opts.output_dim,
        batch=opts.batch,
    )


def _run_and_verify(opts):
    raise SystemExit(
        "This design is intended for the WSL compile-only + Windows host flow "
        "(batch_infer.exe). Direct NPU execution from Python isn't supported "
        "here (WSL has no NPU)."
    )


def main():
    opts = _make_argparser().parse_args()

    run_design_cli(
        gru_decoder_fused,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
