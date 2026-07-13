"""
gen_encoder_data.py

Writes the binary inputs + golden output for the fused encoder host harness
(reuses test.cpp, which loads gru_state.bin / gru_params.bin / gru_golden.bin):

  gru_state.bin  : (SEQ_LEN*INPUT_DIM,) bf16   x_window (10 timesteps x 45)
  gru_params.bin : (N_PARAMS,)          bf16   encoder GRU [w_ih|w_hh|b_ih|b_hh]
  gru_golden.bin : (HIDDEN_DIM,)        f32    reference latent (last hidden)

Golden runs the full SEQ_LEN-step GRU encode in float from the SAME
bf16-quantized inputs the NPU sees, so verify differences are only bf16 +
LUT-approximation. Layout matches gru_encoder.py / gru_encoder.cc.

Usage (from npu/):  python gen_encoder_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

INPUT_DIM = 45          # real feature count (21 numeric + 3x8 embeddings)
INPUT_DIM_PADDED = 48   # padded to a multiple of 16 so the w_ih matvec vectorizes
HIDDEN_DIM = 64
SEQ_LEN = 10  # default; override with --seq-len (must match the kernel's SEQ_LEN)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CKPT = _REPO / "experiments" / "results" / "flair_80_10_10.pt"
_NPZ = _REPO / "data" / "processed" / "preprocessed.npz"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_step_golden(x, h, w_ih, w_hh, b_ih, b_hh):
    H = h.shape[0]
    gi = w_ih @ x + b_ih
    gh = w_hh @ h + b_hh
    r = sigmoid(gi[:H] + gh[:H])
    z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return (1.0 - z) * n + z * h


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = p.parse_args()
    seq_len = args.seq_len

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]

    w_ih_bf = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_bf = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_bf = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_bf = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    # Pad w_ih rows from INPUT_DIM (45) to INPUT_DIM_PADDED (48) with zeros so
    # each row is 32-byte aligned and cols is a multiple of 16 -> the w_ih
    # matvec takes the vector path. The 3 padded weights are 0, so gi is
    # unchanged. w_hh (64 cols) is already a multiple of 16.
    w_ih_pad = np.zeros((3 * HIDDEN_DIM, INPUT_DIM_PADDED), dtype=bfloat16)
    w_ih_pad[:, :INPUT_DIM] = w_ih_bf
    params = np.concatenate(
        [w_ih_pad.reshape(-1), w_hh_bf.reshape(-1), b_ih_bf, b_hh_bf]
    ).astype(bfloat16)

    # First window's 10 timesteps: x_in[t] = [x_num | sport_e | dport_e | proto_e].
    bundle = np.load(str(_NPZ), allow_pickle=True)
    x_num = bundle["X_num"][0].astype(np.float32)   # (T, 21)
    x_cat = bundle["X_cat"][0].astype(np.int64)     # (T, 3)
    sport_w = sd["sport_emb.weight"].numpy()
    dport_w = sd["dport_emb.weight"].numpy()
    proto_w = sd["proto_emb.weight"].numpy()

    x_in_steps = []
    for t in range(seq_len):
        xin = np.concatenate([
            x_num[t],
            sport_w[x_cat[t, 0]],
            dport_w[x_cat[t, 1]],
            proto_w[x_cat[t, 2]],
        ]).astype(bfloat16)
        x_in_steps.append(xin)
    # Pad each timestep's x_in from INPUT_DIM to INPUT_DIM_PADDED with zeros
    # (matches the padded w_ih layout). x_in_steps stay unpadded for the golden.
    _pad = np.zeros(INPUT_DIM_PADDED - INPUT_DIM, dtype=bfloat16)
    x_window = np.concatenate(
        [np.concatenate([s, _pad]) for s in x_in_steps]
    ).astype(bfloat16)  # (seq_len*INPUT_DIM_PADDED,)

    # Golden latent: run the encode in float from the bf16-quantized inputs.
    w_ih = w_ih_bf.astype(np.float32)
    w_hh = w_hh_bf.astype(np.float32)
    b_ih = b_ih_bf.astype(np.float32)
    b_hh = b_hh_bf.astype(np.float32)
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    for t in range(seq_len):
        h = gru_step_golden(x_in_steps[t].astype(np.float32), h, w_ih, w_hh, b_ih, b_hh)
    latent = h.astype(np.float32)

    (_HERE / "gru_state.bin").write_bytes(x_window.tobytes())
    (_HERE / "gru_params.bin").write_bytes(params.tobytes())
    (_HERE / "gru_golden.bin").write_bytes(latent.tobytes())

    print(f"x_window: {x_window.shape} bf16 -> gru_state.bin ({x_window.nbytes} B)")
    print(f"params:   {params.shape} bf16 -> gru_params.bin ({params.nbytes} B)")
    print(f"latent:   {latent.shape} f32  -> gru_golden.bin ({latent.nbytes} B)")
    print(f"latent[:8] = {latent[:8]}")


if __name__ == "__main__":
    main()
