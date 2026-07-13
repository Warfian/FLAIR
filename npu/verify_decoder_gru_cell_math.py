"""
verify_decoder_gru_cell_math.py

Verifies the decoder GRU-cell math against PyTorch before moving it to IRON/NPU.

This mirrors the encoder verification flow:
  - load trained FLAIR checkpoint
  - load one real preprocessed window
  - run the real model encoder to get a real latent vector
  - compute decoder h0_vec = tanh(latent_to_hidden(latent))
  - verify one decoder GRUCell timestep against PyTorch nn.GRUCell
  - verify timestep 2 with nonzero h_prev
  - verify the full decoder sequence against model.decoder(...)

Run from the FLAIR repo root:

    python -m npu.verify_decoder_gru_cell_math

or, if this file is in scripts/:

    python scripts/verify_decoder_gru_cell_math.py
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig


CHECKPOINT_PATH = Path("experiments/results/flair_80_10_10.pt")
NPZ_PATH = Path("data/processed/preprocessed.npz")
WINDOW_INDEX = 0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell_golden(
    x: np.ndarray,
    h_prev: np.ndarray,
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
) -> np.ndarray:
    """Numpy reference for PyTorch GRUCell gate math.

    x:      (input_dim,)
    h_prev: (hidden_dim,)
    w_ih:   (3*hidden_dim, input_dim)
    w_hh:   (3*hidden_dim, hidden_dim)
    b_ih:   (3*hidden_dim,)
    b_hh:   (3*hidden_dim,)

    PyTorch GRU gate order is [reset, update, new].
    """
    hidden = h_prev.shape[0]

    gi = w_ih @ x + b_ih
    gh = w_hh @ h_prev + b_hh

    gi_r, gi_z, gi_n = gi[:hidden], gi[hidden : 2 * hidden], gi[2 * hidden :]
    gh_r, gh_z, gh_n = gh[:hidden], gh[hidden : 2 * hidden], gh[2 * hidden :]

    r = sigmoid(gi_r + gh_r)
    z = sigmoid(gi_z + gh_z)
    n = np.tanh(gi_n + r * gh_n)

    h_next = (1.0 - z) * n + z * h_prev
    return h_next


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if any(str(k).startswith(("encoder.", "decoder.", "sport_emb.")) for k in ckpt.keys()):
            return ckpt
    raise ValueError("Could not find model state_dict in checkpoint.")


def build_model_cfg(ckpt: Any, sd: Dict[str, torch.Tensor], data: np.lib.npyio.NpzFile) -> FLAIRConfig:
    """Build FLAIRConfig using checkpoint config when available, otherwise infer from tensors."""
    numeric_dim = int(data["X_num"].shape[-1])
    sport_vocab_size = int(sd["sport_emb.weight"].shape[0])
    dport_vocab_size = int(sd["dport_emb.weight"].shape[0])
    proto_vocab_size = int(sd["proto_emb.weight"].shape[0])
    embed_dim = int(sd["sport_emb.weight"].shape[1])
    hidden_dim = int(sd["decoder.gru.weight_hh_l0"].shape[1])

    defaults = dict(
        numeric_dim=numeric_dim,
        sport_vocab_size=sport_vocab_size,
        dport_vocab_size=dport_vocab_size,
        proto_vocab_size=proto_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    )

    if isinstance(ckpt, dict) and isinstance(ckpt.get("model_cfg"), dict):
        allowed = {f.name for f in fields(FLAIRConfig)}
        for k, v in ckpt["model_cfg"].items():
            if k in allowed:
                defaults[k] = v

        # Keep dimensions tied to the actual loaded data/weights.
        defaults["numeric_dim"] = numeric_dim
        defaults["sport_vocab_size"] = sport_vocab_size
        defaults["dport_vocab_size"] = dport_vocab_size
        defaults["proto_vocab_size"] = proto_vocab_size

    return FLAIRConfig(**defaults)


def main() -> None:
    print("Loading checkpoint:", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    sd = extract_state_dict(ckpt)

    print("Loading preprocessed data:", NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)

    cfg = build_model_cfg(ckpt, sd, data)
    print(f"model cfg: {cfg}")

    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd, strict=False)  # checkpoint may include unused cat-loss heads
    model.eval()

    x_num = torch.tensor(data["X_num"][WINDOW_INDEX : WINDOW_INDEX + 1], dtype=torch.float32)
    x_cat = torch.tensor(data["X_cat"][WINDOW_INDEX : WINDOW_INDEX + 1], dtype=torch.long)
    seq_len = int(x_num.shape[1])

    with torch.no_grad():
        full_out = model(x_num, x_cat)
        latent_t = full_out["latent"]  # (1, latent_dim)
        decoder_ref_t = model.decoder(latent_t, seq_len=seq_len)  # (1, T, numeric_dim)

    latent = latent_t.squeeze(0).numpy().astype(np.float64)
    print(f"latent shape: {latent.shape}")
    print(f"seq_len: {seq_len}")

    # --- Decoder weights ---
    W_lh = sd["decoder.latent_to_hidden.weight"].numpy().astype(np.float64)
    b_lh = sd["decoder.latent_to_hidden.bias"].numpy().astype(np.float64)

    w_ih = sd["decoder.gru.weight_ih_l0"].numpy().astype(np.float64)
    w_hh = sd["decoder.gru.weight_hh_l0"].numpy().astype(np.float64)
    b_ih = sd["decoder.gru.bias_ih_l0"].numpy().astype(np.float64)
    b_hh = sd["decoder.gru.bias_hh_l0"].numpy().astype(np.float64)

    W_out = sd["decoder.hidden_to_output.weight"].numpy().astype(np.float64)
    b_out = sd["decoder.hidden_to_output.bias"].numpy().astype(np.float64)

    hidden_dim = w_hh.shape[1]
    print(f"W_lh {W_lh.shape}  b_lh {b_lh.shape}")
    print(f"w_ih {w_ih.shape}  w_hh {w_hh.shape}  b_ih {b_ih.shape}  b_hh {b_hh.shape}")
    print(f"W_out {W_out.shape}  b_out {b_out.shape}")

    # This matches decoder.py:
    # h0_vec = torch.tanh(self.latent_to_hidden(latent))
    h0_vec = np.tanh(W_lh @ latent + b_lh)  # (hidden_dim,)

    # Decoder.py uses the same h0_vec as:
    #   1. repeated decoder input at every timestep
    #   2. initial hidden state h0
    x_t = h0_vec.copy()
    h_prev = h0_vec.copy()

    # --- PyTorch GRUCell reference for timestep 1 ---
    cell = torch.nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.from_numpy(w_ih).float())
        cell.weight_hh.copy_(torch.from_numpy(w_hh).float())
        cell.bias_ih.copy_(torch.from_numpy(b_ih).float())
        cell.bias_hh.copy_(torch.from_numpy(b_hh).float())

        h_next_torch = cell(
            torch.from_numpy(x_t).float().unsqueeze(0),
            torch.from_numpy(h_prev).float().unsqueeze(0),
        ).squeeze(0).numpy().astype(np.float64)

    # --- Numpy golden for timestep 1 ---
    h_next_golden = gru_cell_golden(x_t, h_prev, w_ih, w_hh, b_ih, b_hh)

    diff = np.abs(h_next_torch - h_next_golden)
    print(f"\ntimestep 1 max abs diff vs nn.GRUCell:  {diff.max():.3e}")
    print(f"timestep 1 mean abs diff vs nn.GRUCell: {diff.mean():.3e}")

    if not np.allclose(h_next_torch, h_next_golden, atol=1e-5):
        raise SystemExit("FAIL: timestep 1 golden formula does not match nn.GRUCell.")
    print("PASS: timestep 1 matches nn.GRUCell.")

    # --- Timestep 2 check with nonzero h_prev ---
    with torch.no_grad():
        h_next2_torch = cell(
            torch.from_numpy(x_t).float().unsqueeze(0),
            torch.from_numpy(h_next_torch).float().unsqueeze(0),
        ).squeeze(0).numpy().astype(np.float64)

    h_next2_golden = gru_cell_golden(x_t, h_next_golden, w_ih, w_hh, b_ih, b_hh)
    diff2 = np.abs(h_next2_torch - h_next2_golden)

    print(f"\ntimestep 2 max abs diff vs nn.GRUCell:  {diff2.max():.3e}")
    print(f"timestep 2 mean abs diff vs nn.GRUCell: {diff2.mean():.3e}")

    if not np.allclose(h_next2_torch, h_next2_golden, atol=1e-5):
        raise SystemExit("FAIL: timestep 2 golden formula does not match nn.GRUCell.")
    print("PASS: timestep 2 matches nn.GRUCell.")

    # --- Full decoder sequence check ---
    h = h0_vec.copy()
    manual_hidden_states = []

    for _ in range(seq_len):
        h = gru_cell_golden(h0_vec, h, w_ih, w_hh, b_ih, b_hh)
        manual_hidden_states.append(h.copy())

    manual_hidden_states = np.stack(manual_hidden_states, axis=0)  # (T, hidden_dim)
    manual_recon = manual_hidden_states @ W_out.T + b_out          # (T, numeric_dim)

    decoder_ref = decoder_ref_t.squeeze(0).numpy().astype(np.float64)
    full_diff = np.abs(decoder_ref - manual_recon)

    print(f"\nfull decoder max abs diff:  {full_diff.max():.3e}")
    print(f"full decoder mean abs diff: {full_diff.mean():.3e}")

    if not np.allclose(decoder_ref, manual_recon, atol=1e-5):
        raise SystemExit("FAIL: full manual decoder does not match model.decoder.")
    print("PASS: full manual decoder matches model.decoder within 1e-5.")


if __name__ == "__main__":
    main()
