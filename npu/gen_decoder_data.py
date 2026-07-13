from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import sys
import torch



_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig


LATENT_DIM = 64
HIDDEN_DIM = 64
SEQ_LEN = 10
OUTPUT_DIM = 21

_CKPT = _REPO / "experiments" / "results" / "flair_80_10_10.pt"
_NPZ = _REPO / "data" / "processed" / "preprocessed.npz"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_step_golden(x, h, w_ih, w_hh, b_ih, b_hh):
    H = h.shape[0]

    gi = w_ih @ x + b_ih
    gh = w_hh @ h + b_hh

    gi_r, gi_z, gi_n = gi[:H], gi[H:2 * H], gi[2 * H:]
    gh_r, gh_z, gh_n = gh[:H], gh[H:2 * H], gh[2 * H:]

    r = sigmoid(gi_r + gh_r)
    z = sigmoid(gi_z + gh_z)
    n = np.tanh(gi_n + r * gh_n)

    return (1.0 - z) * n + z * h


def pad_even_bf16(arr):
    arr = arr.astype(bfloat16)
    if arr.size % 2 != 0:
        arr = np.concatenate([arr, np.zeros(1, dtype=bfloat16)])
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = p.parse_args()

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = FLAIRConfig(**ckpt["model_cfg"])

    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd, strict=False)  # checkpoint may include unused cat-loss heads
    model.eval()

    bundle = np.load(str(_NPZ), allow_pickle=True)

    x_num_np = bundle["X_num"][args.window_index : args.window_index + 1].astype(np.float32)
    x_cat_np = bundle["X_cat"][args.window_index : args.window_index + 1].astype(np.int64)

    x_num = torch.from_numpy(x_num_np)
    x_cat = torch.from_numpy(x_cat_np)

    with torch.no_grad():
        out = model(x_num, x_cat)
        latent = out["latent"].squeeze(0).numpy().astype(np.float32)

    # Quantize the decoder input and weights to bf16, matching what the NPU sees.
    latent_bf = latent.astype(bfloat16)

    W_lh_bf = sd["decoder.latent_to_hidden.weight"].numpy().astype(bfloat16)
    b_lh_bf = sd["decoder.latent_to_hidden.bias"].numpy().astype(bfloat16)

    w_ih_bf = sd["decoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_bf = sd["decoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_bf = sd["decoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_bf = sd["decoder.gru.bias_hh_l0"].numpy().astype(bfloat16)

    W_out_bf = sd["decoder.hidden_to_output.weight"].numpy().astype(bfloat16)
    b_out_bf = sd["decoder.hidden_to_output.bias"].numpy().astype(bfloat16)

    # Float golden computed from bf16-quantized values.
    latent_f = latent_bf.astype(np.float32)

    W_lh = W_lh_bf.astype(np.float32)
    b_lh = b_lh_bf.astype(np.float32)

    w_ih = w_ih_bf.astype(np.float32)
    w_hh = w_hh_bf.astype(np.float32)
    b_ih = b_ih_bf.astype(np.float32)
    b_hh = b_hh_bf.astype(np.float32)

    W_out = W_out_bf.astype(np.float32)
    b_out = b_out_bf.astype(np.float32)

    # Decoder:
    # h0_vec = tanh(latent_to_hidden(latent))
    h0 = np.tanh(W_lh @ latent_f + b_lh).astype(np.float32)

    # GRU decoder loop:
    # x_t is always h0
    # h starts as h0
    h = h0.copy()
    hidden_seq = []

    for _ in range(args.seq_len):
        h = gru_step_golden(h0, h, w_ih, w_hh, b_ih, b_hh)
        hidden_seq.append(h.copy())

    hidden_seq = np.stack(hidden_seq, axis=0).astype(np.float32)

    # hidden_to_output
    recon = hidden_seq @ W_out.T + b_out
    recon = recon.astype(np.float32)

    # MSE anomaly score
    x_num_window = x_num_np[0, : args.seq_len].astype(np.float32)
    score = np.array([np.mean((recon - x_num_window) ** 2)], dtype=np.float32)

    # Pack params.
    decoder_gru_params = np.concatenate([
        w_ih_bf.reshape(-1),
        w_hh_bf.reshape(-1),
        b_ih_bf.reshape(-1),
        b_hh_bf.reshape(-1),
    ]).astype(bfloat16)

    # Full decoder params:
    # [W_lh | b_lh | W_ih | W_hh | b_ih | b_hh | W_out | b_out]
    decoder_params = np.concatenate([
        W_lh_bf.reshape(-1),
        b_lh_bf.reshape(-1),
        w_ih_bf.reshape(-1),
        w_hh_bf.reshape(-1),
        b_ih_bf.reshape(-1),
        b_hh_bf.reshape(-1),
        W_out_bf.reshape(-1),
        b_out_bf.reshape(-1),
    ])

    decoder_params = pad_even_bf16(decoder_params)

    # Save files.
    (_HERE / "decoder_latent.bin").write_bytes(latent_bf.tobytes())
    (_HERE / "decoder_h0.bin").write_bytes(h0.astype(bfloat16).tobytes())
    (_HERE / "decoder_gru_params.bin").write_bytes(decoder_gru_params.tobytes())
    (_HERE / "decoder_params.bin").write_bytes(decoder_params.tobytes())
    (_HERE / "decoder_x_num.bin").write_bytes(x_num_window.astype(bfloat16).tobytes())
    (_HERE / "decoder_hidden_golden.bin").write_bytes(hidden_seq.tobytes())
    (_HERE / "decoder_golden_recon.bin").write_bytes(recon.tobytes())
    (_HERE / "decoder_golden_score.bin").write_bytes(score.tobytes())
    
    # Show saves!
    print(f"latent:        {latent_bf.shape} bf16 -> decoder_latent.bin ({latent_bf.nbytes} B)")
    print(f"h0:            {h0.shape} bf16 -> decoder_h0.bin ({h0.astype(bfloat16).nbytes} B)")
    print(f"gru params:    {decoder_gru_params.shape} bf16 -> decoder_gru_params.bin ({decoder_gru_params.nbytes} B)")
    print(f"full params:   {decoder_params.shape} bf16 -> decoder_params.bin ({decoder_params.nbytes} B)")
    print(f"x_num:         {x_num_window.shape} bf16 -> decoder_x_num.bin ({x_num_window.astype(bfloat16).nbytes} B)")
    print(f"hidden_seq:    {hidden_seq.shape} f32 -> decoder_hidden_golden.bin ({hidden_seq.nbytes} B)")
    print(f"recon:         {recon.shape} f32 -> decoder_golden_recon.bin ({recon.nbytes} B)")
    print(f"MSE score:     {score[0]:.8f} -> decoder_golden_score.bin")
    print(f"recon[0, :5] = {recon[0, :5]}")


if __name__ == "__main__":
    main()