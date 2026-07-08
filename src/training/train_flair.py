"""
train_flair.py

Train FLAIR (GRU autoencoder) with:
- numeric inputs normalized (X_num)
- categorical embeddings for Sport/Dport (X_cat)
- reconstruct numeric only

Preprocess must produce:
  data/processed/preprocessed.npz with X_num, X_cat, y_seq, and vocabs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FLAIRDataset, DatasetConfig
from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 30
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: str = "experiments/results/flair_minimal.pt"
    val_split: float = 0.1
    patience: Optional[int] = 5


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val_normal(Xn: np.ndarray, Xc: np.ndarray, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be between 0 and 1.")
    n = len(Xn)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_split))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    return Xn[tr_idx], Xc[tr_idx], Xn[val_idx], Xc[val_idx]


def train_one_epoch(model: FLAIRAutoencoder, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    batches = 0
    for (x_num, x_cat), y_num in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_num = y_num.to(device)

        out = model(x_num, x_cat)
        x_hat = out["x_hat_num"]
        loss = model.reconstruction_loss(y_num, x_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def eval_one_epoch(model: FLAIRAutoencoder, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for (x_num, x_cat), y_num in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_num = y_num.to(device)

        out = model(x_num, x_cat)
        x_hat = out["x_hat_num"]
        loss = model.reconstruction_loss(y_num, x_hat)

        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def train_from_preprocessed(
    npz_path: str = "data/processed/preprocessed.npz",
    train_cfg: Optional[TrainConfig] = None,
    config_path: Optional[str] = None,
) -> Dict[str, object]:
    if train_cfg is None:
        train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)  # (N,T,D_num)
    X_cat = bundle["X_cat"].astype(np.int64)    # (N,T,3)
    y_seq = bundle["y_seq"].astype(np.int64)    # (N,)

    sport_vocab = bundle["sport_vocab"][0]  # dict
    dport_vocab = bundle["dport_vocab"][0]  # dict
    proto_vocab = bundle["proto_vocab"][0]  # dict

    print(f"[train] Loaded: {npz_path}")
    print(f"[train] X_num: {X_num.shape}  X_cat: {X_cat.shape}  y_seq: {y_seq.shape}")
    print(f"[train] attack windows: {int(y_seq.sum())}/{len(y_seq)}")

    # normal-only windows
    normal_mask = (y_seq == 0)
    Xn = X_num[normal_mask]
    Xc = X_cat[normal_mask]
    print(f"[train] normal windows used: {len(Xn)}")

    if len(Xn) < 10:
        raise ValueError("Not enough normal windows to train.")

    Xn_tr, Xc_tr, Xn_val, Xc_val = split_train_val_normal(Xn, Xc, train_cfg.val_split, train_cfg.seed)
    print(f"[train] train normal: {len(Xn_tr)}  val normal: {len(Xn_val)}")

    # Load model hyperparameters from config if provided, else use defaults
    cfg_model: Dict[str, object] = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f)
        cfg_model = _cfg.get("model", {})

    model_cfg = FLAIRConfig(
        numeric_dim=int(X_num.shape[-1]),
        sport_vocab_size=len(sport_vocab) + 1,  # +UNK
        dport_vocab_size=len(dport_vocab) + 1,
        proto_vocab_size=len(proto_vocab) + 1,
        embed_dim=int(cfg_model.get("embed_dim", 8)),
        hidden_dim=int(cfg_model.get("hidden_dim", 64)),
        num_layers=int(cfg_model.get("num_layers", 1)),
        dropout=float(cfg_model.get("dropout", 0.0)),
        bidirectional=bool(cfg_model.get("bidirectional", False)),
    )

    train_ds = FLAIRDataset(Xn_tr, Xc_tr, config=DatasetConfig(return_targets=True))
    val_ds = FLAIRDataset(Xn_val, Xc_val, config=DatasetConfig(return_targets=True))

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = FLAIRAutoencoder(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_left = train_cfg.patience if train_cfg.patience is not None else None

    for epoch in range(1, train_cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = eval_one_epoch(model, val_loader, device)

        train_losses.append(tr)
        val_losses.append(va)

        print(f"Epoch {epoch}/{train_cfg.epochs} - train loss: {tr:.6f}  val loss: {va:.6f}")

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if patience_left is not None:
                patience_left = train_cfg.patience
        else:
            if patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[train] Early stopping at epoch {epoch} (best epoch {best_epoch}, best val {best_val:.6f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = Path(train_cfg.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
        },
        ckpt_path
    )

    print(f"\nSaved checkpoint to: {ckpt_path}")
    print(f"[train] Best val loss: {best_val:.6f} at epoch {best_epoch}")

    return {
        "checkpoint_path": str(ckpt_path),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    _config_path = "config.yaml"
    with open(_config_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)

    _t = _cfg.get("training", {})
    _p = _cfg.get("paths", {})

    cfg = TrainConfig(
        batch_size=int(_t.get("batch_size", 32)),
        learning_rate=float(_t.get("learning_rate", 1e-3)),
        epochs=int(_t.get("epochs", 30)),
        seed=int(_t.get("seed", 42)),
        device=str(_t.get("device", "cpu")),
        checkpoint_path=str(_t.get("checkpoint_path", "experiments/results/flair_minimal.pt")),
        val_split=float(_t.get("val_split", 0.1)),
        patience=_t.get("patience", 5),
    )
    npz_path = str(_p.get("processed_npz", "data/processed/preprocessed.npz"))
    train_from_preprocessed(npz_path, train_cfg=cfg, config_path=_config_path)