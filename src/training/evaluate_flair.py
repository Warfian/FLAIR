"""
evaluate_flair.py

Evaluate a trained FLAIR model (with embeddings) by computing anomaly scores + full metrics.

Loads:
- checkpoint: experiments/results/flair_minimal.pt
- preprocessed bundle: data/processed/preprocessed.npz

Computes:
- anomaly scores per window
- threshold from NORMAL windows only (percentile)
- full metrics at that threshold:
    confusion matrix, accuracy, precision, recall, F1, TPR, FPR
- ROC AUC and PR AUC (threshold-independent)
- optional: best-F1 threshold (useful for reporting upper-bound performance)
- saves CSV with scores + predicted label columns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FLAIRDataset, DatasetConfig
from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig


@dataclass
class EvalConfig:
    batch_size: int = 128
    device: str = "cpu"
    threshold_percentile: float = 99.0  # operational threshold uses normal-only percentile
    output_csv: str = "experiments/results/anomaly_scores.csv"


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[FLAIRAutoencoder, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def compute_scores(
    model: FLAIRAutoencoder,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    batch_size: int,
    device: torch.device
) -> np.ndarray:
    ds = FLAIRDataset(X_num, X_cat, config=DatasetConfig(return_targets=True))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores_all = []
    for (x_num, x_cat), _ in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        s = model.anomaly_score(x_num, x_cat)  # (B,)
        scores_all.append(s.cpu().numpy())

    return np.concatenate(scores_all, axis=0)


def compute_threshold(scores_normal: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores_normal, percentile))


def confusion_from_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, int]:
    """
    y_true: 0/1
    predicted anomaly if score > threshold
    """
    y_pred = (scores > threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Returns common classification metrics.
    """
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # TPR
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(recall),
        "fpr": float(fpr),
    }


def roc_pr_curves(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute ROC and PR curves without sklearn.

    Uses the standard technique:
    - sort by descending score
    - walk thresholds implicitly
    - compute cumulative TP/FP
    """
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)

    order = np.argsort(-scores)  # descending
    y = y_true[order]
    s = scores[order]

    P = int((y == 1).sum())
    N = int((y == 0).sum())

    if P == 0 or N == 0:
        # Degenerate case: can't compute ROC/PR meaningfully
        return {
            "fpr": np.array([0.0, 1.0], dtype=np.float64),
            "tpr": np.array([0.0, 1.0], dtype=np.float64),
            "precision": np.array([1.0, 0.0], dtype=np.float64),
            "recall": np.array([0.0, 1.0], dtype=np.float64),
            "thresholds": np.array([np.inf, -np.inf], dtype=np.float64),
        }

    # Cumulative counts at each position
    tp_cum = np.cumsum(y == 1)
    fp_cum = np.cumsum(y == 0)

    # We only want points when the score changes (unique thresholds)
    score_change = np.r_[True, s[1:] != s[:-1]]
    idx = np.where(score_change)[0]

    tp = tp_cum[idx].astype(np.float64)
    fp = fp_cum[idx].astype(np.float64)

    # ROC
    tpr = tp / P
    fpr = fp / N

    # PR
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tpr  # tp/P

    thresholds = s[idx]

    # Add (0,0) start for ROC, and (recall=0, precision=1) start for PR
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    thresholds = np.r_[np.inf, thresholds]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute area under curve using trapezoidal rule.
    Assumes x is increasing.
    """
    return float(np.trapezoid(y, x))


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that maximizes F1 on the provided y_true/scores.
    This is NOT an operational threshold (it uses labels), but useful as a reporting upper bound.
    """
    curves = roc_pr_curves(y_true, scores)
    thresholds = curves["thresholds"]

    best_thr = thresholds[0]
    best_metrics = {"f1": -1.0}

    for thr in thresholds:
        cm = confusion_from_threshold(y_true, scores, float(thr))
        m = metrics_from_confusion(**cm)
        if m["f1"] > best_metrics["f1"]:
            best_metrics = m
            best_thr = float(thr)

    return best_thr, best_metrics


def save_scores_csv(scores: np.ndarray, threshold: float, y_true: np.ndarray, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sequence_id": np.arange(len(scores)),
        "anomaly_score": scores,
        "threshold": np.full(len(scores), threshold, dtype=np.float64),
        "is_anomalous": (scores > threshold).astype(int),
        "target": y_true.astype(int),
    })
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as _f:
        _yaml = yaml.safe_load(_f)

    _ev = _yaml.get("evaluation", {})
    _t = _yaml.get("training", {})
    _p = _yaml.get("paths", {})

    cfg = EvalConfig(
        threshold_percentile=float(_ev.get("threshold_percentile", 99.0)),
        output_csv=str(_ev.get("output_csv", "experiments/results/anomaly_scores.csv")),
    )

    device = torch.device(cfg.device)
    checkpoint_path = str(_t.get("checkpoint_path", "experiments/results/flair_minimal.pt"))
    model, _ = load_checkpoint(checkpoint_path, device)

    npz_path = str(_p.get("processed_npz", "data/processed/preprocessed.npz"))
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y_seq = bundle["y_seq"].astype(np.int64)

    scores = compute_scores(model, X_num, X_cat, cfg.batch_size, device)

    # Operational threshold computed from NORMAL windows only
    normal_scores = scores[y_seq == 0]
    threshold = compute_threshold(normal_scores, cfg.threshold_percentile)

    print(f"Threshold p{cfg.threshold_percentile} (normal-only): {threshold:.6f}")
    print(f"Scores (all):    mean={scores.mean():.6f}, max={scores.max():.6f}")
    print(f"Scores (normal): mean={normal_scores.mean():.6f}, max={normal_scores.max():.6f}")

    # Metrics at the operational threshold
    cm = confusion_from_threshold(y_seq, scores, threshold)
    m = metrics_from_confusion(**cm)

    print("\n=== Metrics @ normal-only percentile threshold ===")
    print(f"Confusion: TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    print(f"Accuracy:  {m['accuracy']:.6f}")
    print(f"Precision: {m['precision']:.6f}")
    print(f"Recall:    {m['recall']:.6f}  (TPR)")
    print(f"F1:        {m['f1']:.6f}")
    print(f"FPR:       {m['fpr']:.6f}")

    # Threshold-independent metrics (ROC AUC, PR AUC)
    curves = roc_pr_curves(y_seq, scores)

    # ROC AUC
    roc_auc = auc_trapz(curves["fpr"], curves["tpr"])

    # PR AUC: integrate precision over recall
    # Ensure recall increases (it does by construction)
    pr_auc = auc_trapz(curves["recall"], curves["precision"])

    print("\n=== Threshold-independent metrics ===")
    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"PR  AUC: {pr_auc:.6f}")

    # Optional: best-F1 threshold (uses labels; not for real deployment)
    best_thr, best_m = best_f1_threshold(y_seq, scores)
    best_cm = confusion_from_threshold(y_seq, scores, best_thr)
    print("\n=== Best-F1 threshold (label-informed, upper-bound) ===")
    print(f"Best threshold: {best_thr:.6f}")
    print(f"Confusion: TP={best_cm['tp']}  FP={best_cm['fp']}  TN={best_cm['tn']}  FN={best_cm['fn']}")
    print(f"Precision: {best_m['precision']:.6f}  Recall: {best_m['recall']:.6f}  F1: {best_m['f1']:.6f}")

    # Save CSV (operational threshold predictions)
    save_scores_csv(scores, threshold, y_seq, cfg.output_csv)
    print(f"\nSaved scores to: {cfg.output_csv}")