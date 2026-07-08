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
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker

_KERNEL_SRC = str(Path(__file__).parent / "kernels" / "gru_cell.cc")

# Dims from the trained checkpoint (experiments/results/flair_minimal.pt):
# encoder.gru.weight_ih_l0: (192, 45) -> INPUT_DIM=45, HIDDEN_DIM=64
INPUT_DIM = 45
HIDDEN_DIM = 64


@iron.jit
def gru_cell_encoder(
    x_in: In,
    h_prev: In,
    params: In,
    h_next: Out,
    *,
    input_dim: CompileTime[int] = INPUT_DIM,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
):
    h3 = 3 * hidden_dim
    n_params = h3 * input_dim + h3 * hidden_dim + h3 + h3
    dtype = np.dtype[bfloat16]

    x_ty = np.ndarray[(input_dim,), dtype]
    h_ty = np.ndarray[(hidden_dim,), dtype]
    params_ty = np.ndarray[(n_params,), dtype]

    gru_kernel = ExternalFunction(
        "gru_cell_encoder_bf16",
        source_file=_KERNEL_SRC,
        arg_types=[x_ty, h_ty, params_ty, h_ty],
        compile_flags=[f"-DINPUT_DIM={input_dim}", f"-DHIDDEN_DIM={hidden_dim}"],
    )

    x_fifo = ObjectFifo(x_ty, name="x_in")
    h_fifo = ObjectFifo(h_ty, name="h_prev")
    params_fifo = ObjectFifo(params_ty, name="params")
    hnext_fifo = ObjectFifo(h_ty, name="h_next")

    def core_fn(x_c, h_c, params_c, hnext_p, kernel):
        elem_x = x_c.acquire(1)
        elem_h = h_c.acquire(1)
        elem_params = params_c.acquire(1)
        elem_out = hnext_p.acquire(1)

        kernel(elem_x, elem_h, elem_params, elem_out)

        x_c.release(1)
        h_c.release(1)
        params_c.release(1)
        hnext_p.release(1)

    worker = Worker(
        core_fn,
        [
            x_fifo.cons(),
            h_fifo.cons(),
            params_fifo.cons(),
            hnext_fifo.prod(),
            gru_kernel,
        ],
    )

    rt = Runtime()
    with rt.sequence(x_ty, h_ty, params_ty, h_ty) as (
        x_arg,
        h_arg,
        params_arg,
        hnext_arg,
    ):
        rt.start(worker)
        rt.fill(x_fifo.prod(), x_arg)
        rt.fill(h_fifo.prod(), h_arg)
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

    return x_in, h_prev, params


def main():
    x_in, h_prev, params = _load_inputs_from_checkpoint()

    x_t = iron.tensor(x_in, dtype=bfloat16, device="npu")
    h_t = iron.tensor(h_prev, dtype=bfloat16, device="npu")
    params_t = iron.tensor(params, dtype=bfloat16, device="npu")
    hnext_t = iron.zeros(HIDDEN_DIM, dtype=bfloat16, device="npu")

    gru_cell_encoder(x_t, h_t, params_t, hnext_t)

    print("h_next (NPU):", hnext_t.numpy())
    # Compare against npu/verify_gru_cell_math.py's golden output for the
    # same checkpoint + first timestep to confirm end-to-end correctness.


if __name__ == "__main__":
    main()
