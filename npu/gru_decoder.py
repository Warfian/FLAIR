#
# gru_decoder.py
#
# IRON wrapper for npu/kernels/gru_decoder.cc
#
# Runs:
#   h0_vec + decoder GRU params -> hidden_seq
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
        "gru_decoder_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )


@iron.jit
def gru_decoder(
    h0_vec: In,
    params: In,
    hidden_seq: Out,
    *,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
):
    h3 = 3 * hidden_dim

    # decoder GRU input_dim = hidden_dim, because x_t = h0_vec
    n_params = h3 * hidden_dim + h3 * hidden_dim + h3 + h3

    dtype = np.dtype[bfloat16]

    h0_ty = np.ndarray[(hidden_dim,), dtype]
    params_ty = np.ndarray[(n_params,), dtype]
    hidden_seq_ty = np.ndarray[(seq_len * hidden_dim,), dtype]

    kernel = _make_decoder_kernel(
        arg_types=[h0_ty, params_ty, hidden_seq_ty],
        compile_flags=[
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
        ],
    )

    h0_fifo = ObjectFifo(h0_ty, depth=1, name="decoder_h0")
    params_fifo = ObjectFifo(params_ty, depth=1, name="decoder_params")
    hidden_fifo = ObjectFifo(hidden_seq_ty, depth=1, name="decoder_hidden_seq")

    def core_fn(h0_c, params_c, hidden_p, k):
        eh0 = h0_c.acquire(1)
        ep = params_c.acquire(1)
        ehid = hidden_p.acquire(1)

        k(eh0, ep, ehid)

        h0_c.release(1)
        params_c.release(1)
        hidden_p.release(1)

    worker = Worker(
        core_fn,
        [h0_fifo.cons(), params_fifo.cons(), hidden_fifo.prod(), kernel],
    )

    rt = Runtime()

    with rt.sequence(h0_ty, params_ty, hidden_seq_ty) as (
        h0_arg,
        params_arg,
        hidden_arg,
    ):
        rt.start(worker)
        rt.fill(h0_fifo.prod(), h0_arg)
        rt.fill(params_fifo.prod(), params_arg)
        rt.drain(hidden_fifo.cons(), hidden_arg, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _load_decoder_test_data():
    h0 = np.fromfile(NPU_DIR / "decoder_h0.bin", dtype=bfloat16)
    params = np.fromfile(NPU_DIR / "decoder_gru_params.bin", dtype=bfloat16)
    golden = np.fromfile(NPU_DIR / "decoder_hidden_golden.bin", dtype=np.float32)

    return h0, params, golden


def _make_argparser():
    p = argparse.ArgumentParser(prog="FLAIR decoder GRU sequence")
    add_compile_args(p)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    return p


def _compile_kwargs(opts):
    return dict(
        hidden_dim=opts.hidden_dim,
        seq_len=opts.seq_len,
    )


def _run_and_verify(opts):
    h0, params, golden = _load_decoder_test_data()

    h0_t = iron.tensor(h0, dtype=bfloat16, device="npu")
    params_t = iron.tensor(params, dtype=bfloat16, device="npu")
    hidden_t = iron.zeros(opts.seq_len * opts.hidden_dim, dtype=bfloat16, device="npu")

    gru_decoder(
        h0_t,
        params_t,
        hidden_t,
        hidden_dim=opts.hidden_dim,
        seq_len=opts.seq_len,
    )

    got = hidden_t.numpy().astype(np.float32)
    golden = golden.reshape(-1)

    diff = np.abs(got - golden)

    print("decoder hidden_seq NPU result:")
    print("shape:", got.shape)
    print("max abs diff:", diff.max())
    print("mean abs diff:", diff.mean())

    if diff.max() < 0.10:
        print("PASS: decoder GRU hidden sequence is close to golden.")
    else:
        print("FAIL: decoder GRU hidden sequence differs too much.")


def main():
    opts = _make_argparser().parse_args()

    run_design_cli(
        gru_decoder,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()