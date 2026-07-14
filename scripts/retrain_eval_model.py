"""
retrain_eval_model.py

Train a proper NPU-sized (hidden=64) FLAIR autoencoder on the thesis-style eval
split (scripts/preprocess_data.py --split-eval). Normal-only training on the
full dataset's train partition, matching Dumond's FLAIR thesis methodology (Sec
3.6) but at the hidden dim that fits the AIE tile (64, vs the thesis's 128).

Saves to experiments/results/flair_h64_full.pt so the sample-trained
flair_minimal.pt (NPU-validated) is left untouched. The new weights are a
drop-in for the NPU pipeline: run_dataset_inference.py bakes them into the
param .bin files at runtime, so no xclbin rebuild is needed.

Usage:  python scripts/retrain_eval_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_flair import TrainConfig, train_from_preprocessed

if __name__ == "__main__":
    cfg = TrainConfig(
        batch_size=512,       # thesis batch size; far faster than the default 32
        learning_rate=1e-3,
        epochs=32,
        seed=42,
        device="cpu",
        checkpoint_path="experiments/results/flair_h64_full.pt",
        val_split=0.1,        # internal random val split for early stopping
        patience=5,
    )
    # config.yaml model.hidden_dim=64 -> NPU-sized model.
    train_from_preprocessed(
        "data/processed/retrain_train.npz",
        train_cfg=cfg,
        config_path="config.yaml",
    )
