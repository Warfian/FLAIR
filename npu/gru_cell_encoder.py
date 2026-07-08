#
# gru_cell_encoder.py
#
# IRON driver for the FLAIR encoder GRU-cell kernel (npu/kernels/gru_cell.cc).
#
# One AIE core computes a single GRU-cell timestep:
#   h_next = GRUCell(x_in, h_prev; W_ih, W_hh, b_ih, b_hh)
#
# Modeled on mlir-aie's programming_examples/basic/matrix_multiplication/
# matrix_vector/matrix_vector.py -- same @iron.jit / ObjectFifo / Worker /
# Runtime structure, but wired to a single custom fused kernel instead of
# the built-in kernels.mv (which only supports int16/int32, not bf16).
#
# The four weight/bias tensors (w_ih, w_hh, b_ih, b_hh) are bundled into one
# flat `params` buffer -- see gru_cell.cc's header comment for the exact
# layout. This matches the convention in programming_examples/ml (conv2d.py,
# scale_shift.py) of bundling constant weights into a single tensor, and
# keeps rt.sequence()'s arity in line with what's used elsewhere in the
# codebase (nothing there goes past 4 args).
#
# NOT YET RUN ON HARDWARE. This is a first draft written without access to
# a machine with the IRON toolchain installed; run/debug it on the NPU
# machine where mlir-aie is built. Formula correctness is separately
# validated (independent of this driver / the AIE toolchain) in
# npu/verify_gru_cell_math.py against a real trained checkpoint.
#
# Usage (on a machine with the IRON toolchain + NPU):
#   python3 gru_cell_encoder.py
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

_KERNEL_SRC = Path(__file__).parent / "kernels" / "gru_cell.cc"


def _make_gru_kernel(arg_types, compile_flags):
    """Build the gru_cell ExternalFunction with the include wiring the aie2
    LUT kernels rely on.

    The JIT copies the kernel source into build/<name>.prj/ before compiling,
    so relative includes in gru_cell.cc break. Instead we:
      * pass explicit -I dirs (wheel's aie_kernels for aie_kernel_utils.h,
        aie_runtime_lib/AIE2 for lut_based_ops.h), and
      * wrap the kernel in a TU that also #includes lut_based_ops.cpp by
        absolute path, so getTanhBf16's tanh_lut_ab/cd tables are defined
        (otherwise: undefined-symbol link error).

    This mirrors aie.iron.kernels.activation._create_lut_kernel's aie2 path.
    Hardcoded to AIE2 because this design targets Phoenix (XDNA1) and the
    kernel uses the AIE2 getTanhBf16 primitive.
    """
    header_base = Path(config.cxx_header_path())
    runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
    lut_cpp = runtime_dir / "lut_based_ops.cpp"

    include_dirs = [
        str(header_base),                    # aie_api/aie.hpp etc.
        str(header_base / "aie_kernels"),    # aie_kernel_utils.h
        str(runtime_dir),                    # lut_based_ops.h
    ]
    source = f'#include "{_KERNEL_SRC}"\n#include "{lut_cpp}"\n'

    return ExternalFunction(
        "gru_cell_encoder_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )

# Dims from the trained checkpoint (experiments/results/flair_minimal.pt):
# encoder.gru.weight_ih_l0: (192, 45) -> INPUT_DIM=45, HIDDEN_DIM=64
INPUT_DIM = 45
HIDDEN_DIM = 64


@iron.jit
def gru_cell_encoder(
    state: In,
    params: In,
    h_next: Out,
    *,
    input_dim: CompileTime[int] = INPUT_DIM,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
):
    # state = [x_in (input_dim) | h_prev (hidden_dim)] concatenated into one
    # input buffer. x_in and h_prev are bundled so the compute tile stays
    # within its 2-input DMA-channel budget: state + params = 2 inputs,
    # h_next = 1 output (each AIE core tile has only 2 in / 2 out channels).
    h3 = 3 * hidden_dim
    n_params = h3 * input_dim + h3 * hidden_dim + h3 + h3
    state_len = input_dim + hidden_dim
    dtype = np.dtype[bfloat16]

    state_ty = np.ndarray[(state_len,), dtype]
    h_ty = np.ndarray[(hidden_dim,), dtype]
    params_ty = np.ndarray[(n_params,), dtype]

    gru_kernel = _make_gru_kernel(
        arg_types=[state_ty, params_ty, h_ty],
        compile_flags=[f"-DINPUT_DIM={input_dim}", f"-DHIDDEN_DIM={hidden_dim}"],
    )

    state_fifo = ObjectFifo(state_ty, name="state")
    params_fifo = ObjectFifo(params_ty, name="params")
    hnext_fifo = ObjectFifo(h_ty, name="h_next")

    def core_fn(state_c, params_c, hnext_p, kernel):
        elem_state = state_c.acquire(1)
        elem_params = params_c.acquire(1)
        elem_out = hnext_p.acquire(1)

        kernel(elem_state, elem_params, elem_out)

        state_c.release(1)
        params_c.release(1)
        hnext_p.release(1)

    worker = Worker(
        core_fn,
        [
            state_fifo.cons(),
            params_fifo.cons(),
            hnext_fifo.prod(),
            gru_kernel,
        ],
    )

    rt = Runtime()
    with rt.sequence(state_ty, params_ty, h_ty) as (
        state_arg,
        params_arg,
        hnext_arg,
    ):
        rt.start(worker)
        rt.fill(state_fifo.prod(), state_arg)
        rt.fill(params_fifo.prod(), params_arg)
        rt.drain(hnext_fifo.cons(), hnext_arg, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _load_inputs_from_checkpoint():
    """Loads real encoder GRU weights + one real timestep's input from the
    trained FLAIR checkpoint, matching npu/verify_gru_cell_math.py exactly.

    params layout must match gru_cell.cc's header comment: w_ih, w_hh,
    b_ih, b_hh concatenated back to back, all flattened row-major.
    """
    import torch

    ckpt = torch.load("../experiments/results/flair_minimal.pt", map_location="cpu")
    sd = ckpt["model_state_dict"]

    w_ih = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16).reshape(-1)
    w_hh = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16).reshape(-1)
    b_ih = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    params = np.concatenate([w_ih, w_hh, b_ih, b_hh])

    bundle = np.load("../data/processed/preprocessed.npz", allow_pickle=True)
    x_num0 = bundle["X_num"][0, 0].astype(np.float32)
    x_cat0 = bundle["X_cat"][0, 0].astype(np.int64)
    sport_e = sd["sport_emb.weight"].numpy()[x_cat0[0]]
    dport_e = sd["dport_emb.weight"].numpy()[x_cat0[1]]
    proto_e = sd["proto_emb.weight"].numpy()[x_cat0[2]]
    x_in = np.concatenate([x_num0, sport_e, dport_e, proto_e]).astype(bfloat16)

    h_prev = np.zeros(HIDDEN_DIM, dtype=bfloat16)

    # Concatenate into the single `state` buffer the kernel expects:
    # [x_in (INPUT_DIM) | h_prev (HIDDEN_DIM)].
    state = np.concatenate([x_in, h_prev]).astype(bfloat16)

    return state, params


def _make_argparser():
    p = argparse.ArgumentParser(prog="FLAIR GRU-cell encoder (AIE)")
    # add_compile_args gives us -d/--dev, --xclbin-path, --insts-path.
    # Compile-only mode (what the WSL Makefile / buildHostWin flow uses)
    # is triggered by passing --xclbin-path; it needs --dev because WSL
    # has no attached NPU to auto-detect. This machine is a Ryzen 7940HS
    # (Phoenix / XDNA1 / AIE2), so the device is "npu" (not "npu2").
    add_compile_args(p)
    p.add_argument("--input-dim", type=int, default=INPUT_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    return p


def _compile_kwargs(opts):
    return dict(input_dim=opts.input_dim, hidden_dim=opts.hidden_dim)


def _run_and_verify(opts):
    """Pure-Python JIT-and-run path. Only usable where an NPU is visible to
    Python (native Windows / native Linux) -- NOT in WSL, which cannot see
    the NPU. In the WSL compile-only flow this function is never called;
    run_design_cli dispatches to compile-only whenever --xclbin-path is set.
    """
    state, params = _load_inputs_from_checkpoint()

    state_t = iron.tensor(state, dtype=bfloat16, device="npu")
    params_t = iron.tensor(params, dtype=bfloat16, device="npu")
    hnext_t = iron.zeros(HIDDEN_DIM, dtype=bfloat16, device="npu")

    gru_cell_encoder(
        state_t, params_t, hnext_t,
        input_dim=opts.input_dim, hidden_dim=opts.hidden_dim,
    )

    print("h_next (NPU):", hnext_t.numpy())
    # Compare against npu/verify_gru_cell_math.py's golden output for the
    # same checkpoint + first timestep to confirm end-to-end correctness.


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gru_cell_encoder,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
