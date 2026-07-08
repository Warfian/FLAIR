"""
encoder.py

This module defines the GRU-based encoder for FLAIR.

Plain-English purpose:
- The encoder reads a *sequence of flows* (a time-ordered list).
- It learns the "normal rhythm" of communication in ICS networks.
- It compresses the entire sequence into a compact summary called
  a latent representation (sometimes called a "context vector").

What goes in:
  x with shape (batch_size, seq_len, input_dim)

What comes out:
  latent with shape (batch_size, hidden_dim)  [or hidden_dim*2 if bidirectional]
  final_hidden with shape (num_layers * num_directions, batch_size, hidden_dim)

No attention and no decoder here yet — this is intentionally minimal
so we can test it in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    """
    Configuration for the GRU encoder.

    input_dim:
        Number of features per flow (e.g., 10 numeric features).
    hidden_dim:
        Size of the GRU hidden state (bigger = more capacity).
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout between GRU layers (only used if num_layers > 1).
    bidirectional:
        If True, encoder reads the sequence forward and backward.
        For now, keep False for simplicity and interpretability.
    """
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False


class GRUEncoder(nn.Module):
    """
    GRU-based encoder.

    Input:
        x: (batch, seq_len, input_dim)

    Output:
        latent: (batch, hidden_dim * num_directions)
        hidden: (num_layers * num_directions, batch, hidden_dim)
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        num_directions = 2 if cfg.bidirectional else 1

        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional
        )

        self.output_dim = cfg.hidden_dim * num_directions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            latent: last GRU output for the sequence
            hidden: final hidden state tensor
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (batch, seq_len, input_dim), got ndim={x.ndim}")
        if x.shape[-1] != self.cfg.input_dim:
            raise ValueError(
                f"Expected input_dim={self.cfg.input_dim}, but got x.shape[-1]={x.shape[-1]}"
            )

        outputs, hidden = self.gru(x)

        # outputs: (batch, seq_len, hidden_dim * num_directions)
        # Use the last timestep output as the sequence summary (latent vector)
        latent = outputs[:, -1, :]

        return latent, hidden
