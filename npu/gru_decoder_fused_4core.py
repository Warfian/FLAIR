# gru_decoder_fused_4core.py
#
# Four-core data-parallel FLAIR fused decoder.
#
# Each compute tile runs the existing gru_decoder_fused_bf16 kernel on an
# independent subset of decoder windows:
#
#   shim --h0-------> memtile --split------> 4 compute tiles
#   shim --params---> memtile --broadcast--> 4 compute tiles
#   4 tiles --recon-> memtile --join-------> shim
#
# `batch` means windows PER CORE. Therefore, one dispatch processes
# N_CORES * batch windows.

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli


NPU_DIR = Path(__file__).resolve().parent
KERNELS_DIR = NPU_DIR / "kernels"
KERNEL_SRC = KERNELS_DIR / "gru_decoder.cc"

HIDDEN_DIM = 64
SEQ_LEN = 10
OUTPUT_DIM = 21

# Windows per compute tile per dispatch.
BATCH = 1
N_CORES = 4


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
def gru_decoder_fused_4core(
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

    # Packed fused-decoder parameter layout:
    #
    #   w_ih | w_hh | b_ih | b_hh | w_out | b_out
    n_params = (
        h3 * hidden_dim
        + h3 * hidden_dim
        + h3
        + h3
        + output_dim * hidden_dim
        + output_dim
    )

    # DMA transfer lengths must be multiples of four bytes. BF16 is two bytes,
    # so pad an odd number of BF16 parameter elements by one.
    if n_params % 2 != 0:
        n_params += 1

    per_core_h0 = batch * hidden_dim
    per_core_recon = batch * seq_len * output_dim

    total_h0 = N_CORES * per_core_h0
    total_recon = N_CORES * per_core_recon

    dtype = np.dtype[bfloat16]

    # Shim-to-memory-tile aggregate buffers.
    h0_all_ty = np.ndarray[(total_h0,), dtype]
    recon_all_ty = np.ndarray[(total_recon,), dtype]

    # Per-compute-tile buffers.
    h0_core_ty = np.ndarray[(per_core_h0,), dtype]
    recon_core_ty = np.ndarray[(per_core_recon,), dtype]

    # Shared parameter buffer.
    params_ty = np.ndarray[(n_params,), dtype]

    kernel = _make_decoder_kernel(
        arg_types=[h0_core_ty, params_ty, recon_core_ty],
        compile_flags=[
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
            f"-DOUTPUT_DIM={output_dim}",
            f"-DBATCH={batch}",
        ],
    )

    # Initial hidden states:
    # shim -> memory tile -> four compute-tile slices.
    h0_fifo = ObjectFifo(h0_all_ty, name="decoder_h0")

    h0_offsets = [
        per_core_h0 * core_index for core_index in range(N_CORES)
    ]

    h0_core_fifos = h0_fifo.cons().split(
        h0_offsets,
        obj_types=[h0_core_ty] * N_CORES,
        names=[
            f"decoder_h0_core{core_index}"
            for core_index in range(N_CORES)
        ],
    )

    # Parameters:
    # shim -> memory tile -> broadcast to every compute tile.
    #
    # Depth one is intentional. The fused parameter buffer is approximately
    # 52.7 KB, so double buffering would exceed a compute tile's L1 capacity.
    params_fifo = ObjectFifo(
        params_ty,
        depth=1,
        name="decoder_params",
    )

    params_bcast = params_fifo.cons().forward(
        obj_type=params_ty,
        depth=1,
        name="decoder_params_bcast",
    )

    # Reconstructions:
    # four compute tiles -> memory tile -> aggregate shim output.
    recon_fifo = ObjectFifo(recon_all_ty, name="decoder_recon")

    recon_offsets = [
        per_core_recon * core_index
        for core_index in range(N_CORES)
    ]

    recon_core_fifos = recon_fifo.prod().join(
        recon_offsets,
        obj_types=[recon_core_ty] * N_CORES,
        names=[
            f"decoder_recon_core{core_index}"
            for core_index in range(N_CORES)
        ],
    )

    def core_fn(h0_consumer, params_consumer, recon_producer, kernel_fn):
        h0_element = h0_consumer.acquire(1)
        params_element = params_consumer.acquire(1)
        recon_element = recon_producer.acquire(1)

        kernel_fn(h0_element, params_element, recon_element)

        h0_consumer.release(1)
        params_consumer.release(1)
        recon_producer.release(1)

    workers = []

    for core_index in range(N_CORES):
        workers.append(
            Worker(
                core_fn,
                [
                    h0_core_fifos[core_index].cons(),
                    params_bcast.cons(),
                    recon_core_fifos[core_index].prod(),
                    kernel,
                ],
            )
        )

    runtime = Runtime()

    with runtime.sequence(
        h0_all_ty,
        params_ty,
        recon_all_ty,
    ) as (
        h0_arg,
        params_arg,
        recon_arg,
    ):
        runtime.start(*workers)

        runtime.fill(h0_fifo.prod(), h0_arg)
        runtime.fill(params_fifo.prod(), params_arg)
        runtime.drain(recon_fifo.cons(), recon_arg, wait=True)

    return Program(
        iron.get_current_device(),
        runtime,
    ).resolve_program()


def _make_argparser():
    parser = argparse.ArgumentParser(
        prog="FLAIR fused decoder, four-core data-parallel"
    )

    add_compile_args(parser)

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=OUTPUT_DIM,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help=(
            "Windows per compute tile per dispatch; "
            "total windows per dispatch = 4 * batch"
        ),
    )

    return parser


def _compile_kwargs(options):
    return {
        "hidden_dim": options.hidden_dim,
        "seq_len": options.seq_len,
        "output_dim": options.output_dim,
        "batch": options.batch,
    }


def _run_and_verify(options):
    raise SystemExit(
        "Compile-only design under WSL. Run the generated xclbin with "
        "batch_infer.exe on Windows."
    )


def main():
    options = _make_argparser().parse_args()

    run_design_cli(
        gru_decoder_fused_4core,
        options,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        # One NPU column provides one shim tile, one memory tile, and four
        # compute tiles.
        device=lambda opts: device_from_args(opts, n_cols=1),
    )


if __name__ == "__main__":
    main()
