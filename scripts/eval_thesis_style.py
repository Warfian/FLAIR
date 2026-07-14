"""
eval_thesis_style.py

Thesis-style FLAIR evaluation (Dumond thesis Sec 3.6 / Chap 4):
  - operational threshold tau = p99 of VALIDATION-normal anomaly scores
    (label-free; no attack labels consulted to pick it)
  - apply tau to the mixed TEST set -> confusion matrix + metrics
  - also report ROC-AUC, PR-AUC, and the label-informed best-F1 upper bound

Runs on the --split-eval bundles (retrain_{val,test}.npz). Default checkpoint is
the retrained NPU-sized model (hidden=64).

Usage:
  python scripts/eval_thesis_style.py [--ckpt PATH] [--val NPZ] [--test NPZ]
                                      [--percentile 99]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig
from src.training.evaluate_flair import (
    confusion_from_threshold, metrics_from_confusion, roc_pr_curves,
    auc_trapz, best_f1_threshold,
)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = FLAIRAutoencoder(FLAIRConfig(**ck["model_cfg"])).to(device).eval()
    model.load_state_dict(ck["model_state_dict"], strict=False)
    return model, ck


@torch.no_grad()
def score(model, npz_path, device, bs=8192):
    b = np.load(npz_path, allow_pickle=True)
    Xn = b["X_num"].astype(np.float32)
    Xc = b["X_cat"].astype(np.int64)
    y = b["y_seq"].astype(np.int64)
    out = np.empty(len(Xn), np.float64)
    for s in range(0, len(Xn), bs):
        e = min(s + bs, len(Xn))
        out[s:e] = model.anomaly_score(
            torch.from_numpy(Xn[s:e]).to(device),
            torch.from_numpy(Xc[s:e]).to(device)).cpu().numpy()
    return out, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="experiments/results/flair_h64_full.pt")
    p.add_argument("--val", default="data/processed/retrain_val.npz")
    p.add_argument("--test", default="data/processed/retrain_test.npz")
    p.add_argument("--percentile", type=float, default=99.0)
    args = p.parse_args()

    device = torch.device("cpu")
    model, ck = load_model(args.ckpt, device)
    hid = ck["model_cfg"]["hidden_dim"]

    val_scores, val_y = score(model, args.val, device)
    test_scores, test_y = score(model, args.test, device)
    assert (val_y == 0).all(), "validation set must be normal-only"

    tau = float(np.percentile(val_scores, args.percentile))
    cm = confusion_from_threshold(test_y, test_scores, tau)
    m = metrics_from_confusion(**cm)
    curves = roc_pr_curves(test_y, test_scores)
    roc = auc_trapz(curves["fpr"], curves["tpr"])
    pr = auc_trapz(curves["recall"], curves["precision"])
    bthr, bm = best_f1_threshold(test_y, test_scores)
    bcm = confusion_from_threshold(test_y, test_scores, bthr)

    print("=" * 66)
    print(f"Thesis-style eval  ckpt={args.ckpt} (hidden={hid})")
    print(f"  test: {len(test_y)} windows, {int(test_y.sum())} attacks "
          f"({100*test_y.mean():.2f}%)")
    print("=" * 66)
    print(f"  operational threshold tau = p{args.percentile:g}(val-normal) = {tau:.6f}")
    print(f"  Confusion: TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")
    print(f"  Precision {m['precision']:.4f}  Recall {m['recall']:.4f}  "
          f"F1 {m['f1']:.4f}  FPR {m['fpr']:.4f}")
    print("-" * 66)
    print(f"  ROC-AUC {roc:.4f}   PR-AUC {pr:.4f}")
    print("-" * 66)
    print(f"  best-F1 (label-informed upper bound) tau={bthr:.6f}")
    print(f"    TN={bcm['tn']} FP={bcm['fp']} FN={bcm['fn']} TP={bcm['tp']}  "
          f"P {bm['precision']:.4f} R {bm['recall']:.4f} F1 {bm['f1']:.4f}")
    print("=" * 66)
    print("  thesis hidden=128 ref (80/10/10): F1=0.9320  ROC-AUC=0.9994  tau=0.3226")
    print("=" * 66)


if __name__ == "__main__":
    main()
