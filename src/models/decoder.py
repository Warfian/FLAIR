"""
decoder.py

This module defines the GRU-based decoder for FLAIR.

Plain-English purpose:
- The decoder takes the latent summary produced by the encoder
  (a compact "fingerprint" of the sequence).
- It attempts to reconstruct the original sequence of flows.
- If the input behavior is normal, reconstruction should be good.
- If behavior is unusual, reconstruction error becomes large.

Design choice for incremental testing:
- We feed the decoder a simple repeated input at each timestep.
  This avoids teacher forcing and keeps behavior easy to test.

Shapes:
  latent: (batch, latent_dim)
  output: (batch, seq_len, output_dim)

Later, we will set output_dim == input_dim so the decoder reconstructs
the same feature vector per timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    """
    Configuration for the GRU decoder.

    latent_dim:
        Dimension of the latent vector coming from the encoder.
        (Typically equals encoder hidden_dim * num_directions.)
    hidden_dim:
        GRU hidden state size inside the decoder.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout between layers (only used if num_layers > 1).
    output_dim:
        Number of features per flow we want to reconstruct.
        In an autoencoder, this should match the encoder input_dim.
    """
    latent_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    output_dim: int = 10


class GRUDecoder(nn.Module):
    """
    GRU-based decoder.

    Input:
        latent: (batch, latent_dim)
        seq_len: int

    Output:
        recon: (batch, seq_len, output_dim)
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Project latent vector into an initial hidden state space
        # We will use this to initialize the GRU hidden state.
        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)

        self.gru = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False
        )

        # Map decoder GRU outputs back to the reconstructed feature space
        self.hidden_to_output = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Forward pass.

        We generate a constant decoder input at each timestep by repeating
        a transformed version of the latent vector. This is a simple and
        stable starting point for reconstruction-based models.

        Returns:
            recon: (batch, seq_len, output_dim)
        """
        if latent.ndim != 2:
            raise ValueError(f"Expected latent with shape (batch, latent_dim), got ndim={latent.ndim}")
        if latent.shape[-1] != self.cfg.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.cfg.latent_dim}, but got latent.shape[-1]={latent.shape[-1]}"
            )
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        batch_size = latent.shape[0]

        # Create an initial hidden input vector from latent
        h0_vec = torch.tanh(self.latent_to_hidden(latent))  # (batch, hidden_dim)

        # Build a repeated input sequence for the GRU
        # decoder_inputs: (batch, seq_len, hidden_dim)
        decoder_inputs = h0_vec.unsqueeze(1).repeat(1, seq_len, 1)

        # Initialize GRU hidden state:
        # h0: (num_layers, batch, hidden_dim)
        h0 = h0_vec.unsqueeze(0).repeat(self.cfg.num_layers, 1, 1)

        dec_outputs, _ = self.gru(decoder_inputs, h0)

        # Map GRU hidden outputs to reconstructed features
        recon = self.hidden_to_output(dec_outputs)  # (batch, seq_len, output_dim)

        return recon
