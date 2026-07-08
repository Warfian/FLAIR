"""
flair_model.py

FLAIR autoencoder with categorical embeddings.

Inputs:
  x_num: (B, T, D_num) float
  x_cat: (B, T, D_cat) long (IDs)

We embed categorical features and concatenate with numeric:
  x_in = concat(x_num, embed(x_cat))

Encoder GRU -> latent
Decoder GRU -> reconstruct numeric only:
  x_hat_num: (B, T, D_num)

Loss/anomaly score computed on numeric reconstruction only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .encoder import GRUEncoder, EncoderConfig
from .decoder import GRUDecoder, DecoderConfig


@dataclass
class FLAIRConfig:
    # numeric input dimension (21 for your setup)
    numeric_dim: int

    # categorical embedding settings
    sport_vocab_size: int
    dport_vocab_size: int
    proto_vocab_size: int
    embed_dim: int = 8

    # GRU settings
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False


class FLAIRAutoencoder(nn.Module):
    def __init__(self, cfg: FLAIRConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings (ID=0 is UNK; fine)
        self.sport_emb = nn.Embedding(cfg.sport_vocab_size, cfg.embed_dim)
        self.dport_emb = nn.Embedding(cfg.dport_vocab_size, cfg.embed_dim)
        self.proto_emb = nn.Embedding(cfg.proto_vocab_size, cfg.embed_dim)

        # Combined input dim to encoder GRU
        # x_num + [sport_emb, dport_emb, proto_emb]
        combined_dim = cfg.numeric_dim + (3 * cfg.embed_dim)

        enc_cfg = EncoderConfig(
            input_dim=combined_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional
        )
        self.encoder = GRUEncoder(enc_cfg)

        latent_dim = self.encoder.output_dim

        # Decoder reconstructs NUMERIC features only
        dec_cfg = DecoderConfig(
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_dim=cfg.numeric_dim
        )
        self.decoder = GRUDecoder(dec_cfg)

        self.mse = nn.MSELoss(reduction="mean")

    def _combine_inputs(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B,T,D_num) float
        x_cat: (B,T,3) long  [Sport_id, Dport_id, Proto_id]
        returns: (B,T,D_num+3*embed_dim)
        """
        if x_cat.shape[-1] != 3:
            raise ValueError(f"Expected x_cat last dim=3 (Sport,Dport,Proto), got {x_cat.shape[-1]}")

        sport_id = x_cat[..., 0]
        dport_id = x_cat[..., 1]
        proto_id = x_cat[..., 2]

        sport_e = self.sport_emb(sport_id)  # (B,T,E)
        dport_e = self.dport_emb(dport_id)  # (B,T,E)
        proto_e = self.proto_emb(proto_id)  # (B,T,E)

        return torch.cat([x_num, sport_e, dport_e, proto_e], dim=-1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x_num.ndim != 3 or x_cat.ndim != 3:
            raise ValueError("Expected x_num and x_cat with shape (batch, seq_len, dim).")
        if x_num.shape[0] != x_cat.shape[0] or x_num.shape[1] != x_cat.shape[1]:
            raise ValueError("x_num and x_cat must match on (batch, seq_len).")
        if x_num.shape[-1] != self.cfg.numeric_dim:
            raise ValueError(f"Expected numeric_dim={self.cfg.numeric_dim}, got {x_num.shape[-1]}")

        B, T, _ = x_num.shape
        x_in = self._combine_inputs(x_num, x_cat)  # (B,T,combined_dim)

        latent, _ = self.encoder(x_in)
        x_hat_num = self.decoder(latent, seq_len=T)  # (B,T,D_num)

        return {"x_hat_num": x_hat_num, "latent": latent}

    def reconstruction_loss(self, x_num: torch.Tensor, x_hat_num: torch.Tensor) -> torch.Tensor:
        return self.mse(x_hat_num, x_num)

    @torch.no_grad()
    def anomaly_score(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        out = self.forward(x_num, x_cat)
        x_hat = out["x_hat_num"]
        return torch.mean((x_hat - x_num) ** 2, dim=(1, 2))  # (B,)