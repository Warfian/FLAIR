"""
gen_test_data.py

Writes the binary input + golden-output files that the C++ host harness
(test.cpp) loads to run the FLAIR GRU-cell kernel on the NPU and verify it.

Outputs (raw little-endian, written next to this script):
  gru_state.bin   : (STATE_LEN,)  bf16   [x_in (INPUT_DIM) | h_prev (HIDDEN_DIM) | pad]
  gru_params.bin  : (N_PARAMS,)   bf16   [w_ih | w_hh | b_ih | b_hh]
  gru_golden.bin  : (HIDDEN_DIM,) float32  reference h_next

The state/params layout + padding here must match npu/gru_cell_encoder.py
and npu/kernels/gru_cell.cc exactly. The golden is computed in float from
the SAME bf16-quantized inputs the NPU receives, so the only expected
divergence at verify time is bf16 rounding inside the kernel + the LUT
tanh/sigmoid approximation (hence the host uses a tolerance, not equality).

Usage (from npu/):
    python -m gen_test_data          # if run as a module from repo root
    python gen_test_data.py          # or directly from npu/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Same dims as the checkpoint / driver.
INPUT_DIM = 45
HIDDEN_DIM = 64

# Resolve paths relative to the repo root regardless of CWD.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CKPT = _REPO / "experiments" / "results" / "flair_minimal.pt"
_NPZ = _REPO / "data" / "processed" / "preprocessed.npz"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell_golden(x, h_prev, w_ih, w_hh, b_ih, b_hh):
    """float reference for one GRU cell timestep (PyTorch nn.GRUCell math).
    Gate order [reset, update, new]; reset gates the hidden-side candidate
    contribution before combining with the input side."""
    hidden = h_prev.shape[0]
    gi = w_ih @ x + b_ih
    gh = w_hh @ h_prev + b_hh
    gi_r, gi_z, gi_n = gi[:hidden], gi[hidden:2 * hidden], gi[2 * hidden:]
    gh_r, gh_z, gh_n = gh[:hidden], gh[hidden:2 * hidden], gh[2 * hidden:]
    r = sigmoid(gi_r + gh_r)
    z = sigmoid(gi_z + gh_z)
    n = np.tanh(gi_n + r * gh_n)
    return (1.0 - z) * n + z * h_prev


def main() -> None:
    import torch

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]

    # Weights, bf16 (what the NPU gets), row-major flatten.
    w_ih_bf = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_bf = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_bf = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_bf = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    params = np.concatenate(
        [w_ih_bf.reshape(-1), w_hh_bf.reshape(-1), b_ih_bf, b_hh_bf]
    ).astype(bfloat16)

    # One real timestep's combined input, bf16.
    bundle = np.load(str(_NPZ), allow_pickle=True)
    x_num0 = bundle["X_num"][0, 0].astype(np.float32)
    x_cat0 = bundle["X_cat"][0, 0].astype(np.int64)
    sport_e = sd["sport_emb.weight"].numpy()[x_cat0[0]]
    dport_e = sd["dport_emb.weight"].numpy()[x_cat0[1]]
    proto_e = sd["proto_emb.weight"].numpy()[x_cat0[2]]
    x_in_bf = np.concatenate([x_num0, sport_e, dport_e, proto_e]).astype(bfloat16)
    h_prev_bf = np.zeros(HIDDEN_DIM, dtype=bfloat16)

    # state = [x_in | h_prev], padded to an even length (4-byte DMA alignment).
    state = np.concatenate([x_in_bf, h_prev_bf]).astype(bfloat16)
    if state.size % 2 != 0:
        state = np.concatenate([state, np.zeros(1, dtype=bfloat16)])

    # Golden: float reference from the bf16-quantized inputs (float32 view).
    golden = gru_cell_golden(
        x_in_bf.astype(np.float32),
        h_prev_bf.astype(np.float32),
        w_ih_bf.astype(np.float32),
        w_hh_bf.astype(np.float32),
        b_ih_bf.astype(np.float32),
        b_hh_bf.astype(np.float32),
    ).astype(np.float32)

    (_HERE / "gru_state.bin").write_bytes(state.tobytes())
    (_HERE / "gru_params.bin").write_bytes(params.tobytes())
    (_HERE / "gru_golden.bin").write_bytes(golden.tobytes())

    print(f"state:  {state.shape} bf16  -> gru_state.bin ({state.nbytes} bytes)")
    print(f"params: {params.shape} bf16  -> gru_params.bin ({params.nbytes} bytes)")
    print(f"golden: {golden.shape} f32   -> gru_golden.bin ({golden.nbytes} bytes)")
    print(f"golden h_next[:8] = {golden[:8]}")


if __name__ == "__main__":
    main()
