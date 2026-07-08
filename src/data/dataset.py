"""
dataset.py

PyTorch Dataset for FLAIR with:
- numeric sequences (float32): X_num  (N, T, D_num)
- categorical sequences (int64): X_cat (N, T, D_cat)

Returns:
  ((x_num, x_cat), y_num)

Where:
- y_num is the numeric target for reconstruction (autoencoder): y_num == x_num
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    dtype_num: torch.dtype = torch.float32
    dtype_cat: torch.dtype = torch.long
    return_targets: bool = True


class FLAIRDataset(Dataset):
    """
    Expected shapes:
      X_num: (N, T, D_num)
      X_cat: (N, T, D_cat)
    """

    def __init__(
        self,
        X_num: Union[np.ndarray, torch.Tensor],
        X_cat: Union[np.ndarray, torch.Tensor],
        config: Optional[DatasetConfig] = None
    ) -> None:
        super().__init__()
        self.config = config or DatasetConfig()

        # Numeric
        if isinstance(X_num, np.ndarray):
            if X_num.ndim != 3:
                raise ValueError(f"X_num expected ndim=3, got {X_num.ndim}")
            self.X_num = torch.tensor(X_num, dtype=self.config.dtype_num)
        elif isinstance(X_num, torch.Tensor):
            if X_num.ndim != 3:
                raise ValueError(f"X_num expected ndim=3, got {X_num.ndim}")
            self.X_num = X_num.to(dtype=self.config.dtype_num)
        else:
            raise TypeError("X_num must be a NumPy array or torch.Tensor")

        # Categorical
        if isinstance(X_cat, np.ndarray):
            if X_cat.ndim != 3:
                raise ValueError(f"X_cat expected ndim=3, got {X_cat.ndim}")
            self.X_cat = torch.tensor(X_cat, dtype=self.config.dtype_cat)
        elif isinstance(X_cat, torch.Tensor):
            if X_cat.ndim != 3:
                raise ValueError(f"X_cat expected ndim=3, got {X_cat.ndim}")
            self.X_cat = X_cat.to(dtype=self.config.dtype_cat)
        else:
            raise TypeError("X_cat must be a NumPy array or torch.Tensor")

        if self.X_num.shape[0] != self.X_cat.shape[0] or self.X_num.shape[1] != self.X_cat.shape[1]:
            raise ValueError("X_num and X_cat must match on (N, T).")

    def __len__(self) -> int:
        return self.X_num.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x_num = self.X_num[idx]  # (T, D_num)
        x_cat = self.X_cat[idx]  # (T, D_cat)

        if self.config.return_targets:
            return (x_num, x_cat), x_num
        return (x_num, x_cat), torch.empty(0)