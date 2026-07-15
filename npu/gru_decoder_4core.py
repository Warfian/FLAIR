# gru_decoder_4core.py
#
# Four-core data-parallel FLAIR unfused decoder.
#
# Each compute tile runs gru_decoder_bf16 on an independent subset:
#
#   shim --h0-------> memtile --split------> 4 compute tiles
#   shim --params---> memtile --broadcast--> 4 compute tiles
#   4 tiles --hidden-> memtile --join-------> shim
#
# `batch` is windows PER CORE. One dispatch processes 4 * batch windows.

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
        "gru_decoder_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )


@iron.jit
def gru_decoder_4core(
    h0_vec: In,
    params: In,
    hidden_seq: Out,
    *,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
    batch: CompileTime[int] = BATCH,
):
    h3 = 3 * hidden_dim

    # Packed unfused decoder parameters:
    #
    #   w_ih | w_hh | b_ih | b_hh
    #
    # For H=64:
    #   192*64 + 192*64 + 192 + 192 = 24,960 bf16
    n_params = (
        h3 * hidden_dim
        + h3 * hidden_dim
        + h3
        + h3
    )

    per_core_h0 = batch * hidden_dim
    per_core_hidden = batch * seq_len * hidden_dim

    total_h0 = N_CORES * per_core_h0
    total_hidden = N_CORES * per_core_hidden

    dtype = np.dtype[bfloat16]

    # Aggregate memory-tile buffers.
    h0_all_ty = np.ndarray[(total_h0,), dtype]
    hidden_all_ty = np.ndarray[(total_hidden,), dtype]

    # Per-compute-tile buffers.
    h0_core_ty = np.ndarray[(per_core_h0,), dtype]
    hidden_core_ty = np.ndarray[(per_core_hidden,), dtype]

    # Shared parameter buffer.
    params_ty = np.ndarray[(n_params,), dtype]

    kernel = _make_decoder_kernel(
        arg_types=[h0_core_ty, params_ty, hidden_core_ty],
        compile_flags=[
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
            f"-DBATCH={batch}",
        ],
    )

    # h0: shim -> memory tile -> four compute-tile slices.
    h0_fifo = ObjectFifo(
        h0_all_ty,
        name="decoder_h0",
    )

    h0_offsets = [
        core_index * per_core_h0
        for core_index in range(N_CORES)
    ]

    h0_core_fifos = h0_fifo.cons().split(
        h0_offsets,
        obj_types=[h0_core_ty] * N_CORES,
        names=[
            f"decoder_h0_core{core_index}"
            for core_index in range(N_CORES)
        ],
    )

    # Parameters: shim -> memory tile -> broadcast to all compute tiles.
    #
    # Depth one avoids duplicating the approximately 50 KB parameter buffer
    # inside each compute tile's local memory.
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

    # Hidden sequences: four compute tiles -> memory tile -> shim.
    hidden_fifo = ObjectFifo(
        hidden_all_ty,
        name="decoder_hidden_seq",
    )

    hidden_offsets = [
        core_index * per_core_hidden
        for core_index in range(N_CORES)
    ]

    hidden_core_fifos = hidden_fifo.prod().join(
        hidden_offsets,
        obj_types=[hidden_core_ty] * N_CORES,
        names=[
            f"decoder_hidden_core{core_index}"
            for core_index in range(N_CORES)
        ],
    )

    def core_fn(
        h0_consumer,
        params_consumer,
        hidden_producer,
        kernel_fn,
    ):
        h0_element = h0_consumer.acquire(1)
        params_element = params_consumer.acquire(1)
        hidden_element = hidden_producer.acquire(1)

        kernel_fn(
            h0_element,
            params_element,
            hidden_element,
        )

        h0_consumer.release(1)
        params_consumer.release(1)
        hidden_producer.release(1)

    workers = []

    for core_index in range(N_CORES):
        workers.append(
            Worker(
                core_fn,
                [
                    h0_core_fifos[core_index].cons(),
                    params_bcast.cons(),
                    hidden_core_fifos[core_index].prod(),
                    kernel,
                ],
            )
        )

    runtime = Runtime()

    with runtime.sequence(
        h0_all_ty,
        params_ty,
        hidden_all_ty,
    ) as (
        h0_arg,
        params_arg,
        hidden_arg,
    ):
        runtime.start(*workers)

        runtime.fill(h0_fifo.prod(), h0_arg)
        runtime.fill(params_fifo.prod(), params_arg)
        runtime.drain(
            hidden_fifo.cons(),
            hidden_arg,
            wait=True,
        )

    return Program(
        iron.get_current_device(),
        runtime,
    ).resolve_program()


def _make_argparser():
    parser = argparse.ArgumentParser(
        prog="FLAIR unfused decoder, four-core data-parallel"
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
        "batch": options.batch,
    }


def _run_and_verify(options):
    raise SystemExit(
        "Compile-only design under WSL. Run the generated xclbin "
        "with batch_infer.exe on Windows."
    )


def main():
    options = _make_argparser().parse_args()

    run_design_cli(
        gru_decoder_4core,
        options,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda opts: device_from_args(opts, n_cols=1),
    )


if __name__ == "__main__":
    main()
