#!/usr/bin/env python3
"""
compare_anomaly_score.py

Closes the FLAIR anomaly-detection loop end-to-end: takes the decoder's actual
NPU output (decoder_npu_hidden.bin, dumped by test_decoder.cpp), runs the
remaining host-side steps (hidden_to_output -> reconstruction -> MSE), and
compares the resulting NPU anomaly score against:

  * the float / PyTorch-precision reference (same decoder path, unquantized), and
  * the bf16 golden score gen_decoder_data.py computed (decoder_golden_score.bin).

This answers the real question: does the NPU's bf16 + LUT-gate drift actually
move the anomaly score vs PyTorch? Run it after the decoder stage (same
--seq-len / --window-index as the decoder run).

Usage (from npu/):
    python3 compare_anomaly_score.py --seq-len 5 --window-index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig  # noqa: E402

HIDDEN_DIM = 64
OUTPUT_DIM = 21

_CKPT = _REPO / "experiments" / "results" / "flair_80_10_10.pt"
_NPZ = _REPO / "data" / "processed" / "preprocessed.npz"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_step_float(x, h, w_ih, w_hh, b_ih, b_hh):
    """One float GRU-cell step (PyTorch nn.GRUCell math)."""
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
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--window-index", type=int, default=0)
    args = p.parse_args()
    T = args.seq_len

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd, strict=False)  # checkpoint may include unused cat-loss heads
    model.eval()

    bundle = np.load(str(_NPZ), allow_pickle=True)
    wi = args.window_index
    x_num_np = bundle["X_num"][wi:wi + 1].astype(np.float32)
    x_cat_np = bundle["X_cat"][wi:wi + 1].astype(np.int64)
    x_num_window = x_num_np[0, :T].astype(np.float32)  # (T, 21)

    # Latent for this window (float, from the trained model).
    with torch.no_grad():
        latent = model(torch.from_numpy(x_num_np), torch.from_numpy(x_cat_np))[
            "latent"
        ].squeeze(0).numpy().astype(np.float32)

    # Decoder weights (full float precision).
    W_lh = sd["decoder.latent_to_hidden.weight"].numpy().astype(np.float32)
    b_lh = sd["decoder.latent_to_hidden.bias"].numpy().astype(np.float32)
    w_ih = sd["decoder.gru.weight_ih_l0"].numpy().astype(np.float32)
    w_hh = sd["decoder.gru.weight_hh_l0"].numpy().astype(np.float32)
    b_ih = sd["decoder.gru.bias_ih_l0"].numpy().astype(np.float32)
    b_hh = sd["decoder.gru.bias_hh_l0"].numpy().astype(np.float32)
    W_out = sd["decoder.hidden_to_output.weight"].numpy().astype(np.float32)
    b_out = sd["decoder.hidden_to_output.bias"].numpy().astype(np.float32)

    def score_from_hidden(hidden_seq):
        recon = hidden_seq @ W_out.T + b_out            # (T, 21)
        return float(np.mean((recon - x_num_window) ** 2))

    # --- Float / PyTorch-precision reference: same decoder path, unquantized ---
    h0 = np.tanh(W_lh @ latent + b_lh).astype(np.float32)
    h = h0.copy()
    hidden_ref = []
    for _ in range(T):
        h = gru_step_float(h0, h, w_ih, w_hh, b_ih, b_hh)
        hidden_ref.append(h.copy())
    hidden_ref = np.stack(hidden_ref, axis=0).astype(np.float32)
    score_ref = score_from_hidden(hidden_ref)

    # --- NPU: the actual decoder hidden_seq off the chip ---
    npu_path = _HERE / "decoder_npu_hidden.bin"
    if not npu_path.exists():
        sys.exit(
            "decoder_npu_hidden.bin not found -- run the decoder stage first "
            "(make -f Makefile.decoder run)."
        )
    hidden_npu = np.frombuffer(npu_path.read_bytes(), dtype=np.float32)
    if hidden_npu.size != T * HIDDEN_DIM:
        sys.exit(
            f"decoder_npu_hidden.bin has {hidden_npu.size} floats; expected "
            f"{T * HIDDEN_DIM} (T={T}). Re-run the decoder with matching --seq-len."
        )
    hidden_npu = hidden_npu.reshape(T, HIDDEN_DIM)
    score_npu = score_from_hidden(hidden_npu)

    # --- bf16 golden score gen_decoder_data.py wrote (optional cross-check) ---
    golden_path = _HERE / "decoder_golden_score.bin"
    score_golden = (
        float(np.frombuffer(golden_path.read_bytes(), dtype=np.float32)[0])
        if golden_path.exists()
        else None
    )

    print("=" * 68)
    print(f"FLAIR anomaly score - NPU vs reference  (window {wi}, T={T})")
    print("=" * 68)
    print(f"  NPU-derived score (from NPU hidden_seq): {score_npu:.6f}")
    print(f"  Float reference  (PyTorch-precision)   : {score_ref:.6f}")
    if score_golden is not None:
        print(f"  bf16 golden score (gen_decoder_data)   : {score_golden:.6f}")
    print("-" * 68)
    abs_err = abs(score_npu - score_ref)
    rel_err = abs_err / abs(score_ref) if score_ref != 0 else float("nan")
    print(f"  |NPU - float ref|      : {abs_err:.6f}")
    print(f"  relative error         : {rel_err * 100:.2f}%")
    print("=" * 68)
    if rel_err < 0.05:
        print("NPU anomaly score matches PyTorch to within 5% - drift does not "
              "meaningfully move the score.")
    elif rel_err < 0.15:
        print("NPU anomaly score within ~15% of PyTorch - usable; the gate drift "
              "has a small effect.")
    else:
        print("NPU anomaly score differs >15% from PyTorch - the gate drift is "
              "material; consider the vectorized-Pade gate fix.")


if __name__ == "__main__":
    main()
