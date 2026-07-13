#!/usr/bin/env python3
"""
cpu_baseline.py

Runs the same FLAIR autoencoder over the same dataset windows as
run_dataset_inference.py, but on the CPU via PyTorch eager mode -- one window
at a time (batch=1), single thread. That mirrors exactly how the NPU side
streams windows through batch_infer.exe, so the two wall-clock totals are a
fair, apples-to-apples comparison for the "how much faster is the NPU" number
on the demo website.

Single thread (torch.set_num_threads(1)) is used deliberately: intra-op
parallelism overhead dominates for a model this small at batch=1 (see
scripts/benchmark_inference.py), and it matches a realistic single-stream
edge-deployment scenario -- the same scenario the NPU kernel targets.

Usage (from npu/):
    python3 cpu_baseline.py --limit 990
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))  # so `import run_dataset_inference` resolves
                                # even when this module is imported (not run
                                # directly) from outside npu/, e.g. by
                                # webdemo/server.py.

from run_dataset_inference import (  # noqa: E402
    f1_at_percentile, load_model_and_data, roc_auc,
)

# Progress callback: (windows_done, windows_total) -> None.
ProgressCB = Optional[Callable[[int, int], None]]


def run_cpu_pipeline(
    npz_path: str,
    seq_len: int = 10,
    limit: int = 990,
    on_progress: ProgressCB = None,
    progress_every: int = 10,
) -> dict:
    """Runs PyTorch FLAIR inference window-by-window (batch=1, 1 thread) over
    `limit` windows of `npz_path` and returns:

      scores   : (N,) PyTorch anomaly scores
      labels   : (N,) ground-truth labels
      timings  : {"cpu_inference_ms", "cpu_us_per_window"} -- pure per-window
                 inference time, same shape as run_npu_pipeline()'s timings.
      metrics  : {"auc", "f1"}
    """
    import torch

    model, _sd, X_num, X_cat, y, N = load_model_and_data(npz_path, limit)
    X_num, X_cat = X_num[:, :seq_len], X_cat[:, :seq_len]

    default_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    x_num_t = torch.from_numpy(X_num)
    x_cat_t = torch.from_numpy(X_cat)

    scores = np.empty(N, dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N):
            s = model.anomaly_score(x_num_t[i:i + 1], x_cat_t[i:i + 1])
            scores[i] = float(s.item())
            if on_progress and ((i + 1) % progress_every == 0 or i + 1 == N):
                on_progress(i + 1, N)
    t1 = time.perf_counter()

    torch.set_num_threads(default_threads)

    total_ms = (t1 - t0) * 1000.0
    auc = roc_auc(scores, y)
    f1, _ = f1_at_percentile(scores, y)

    return {
        "scores": scores,
        "labels": y,
        "n_windows": N,
        "timings": {
            "cpu_inference_ms": total_ms,
            "cpu_us_per_window": (total_ms * 1000.0 / N) if N else 0.0,
        },
        "metrics": {"auc": auc, "f1": f1},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str,
                   default=str(_REPO / "data" / "processed" / "preprocessed.npz"))
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="max windows (0 = all)")
    args = p.parse_args()

    result = run_cpu_pipeline(npz_path=args.npz, seq_len=args.seq_len, limit=args.limit)
    t, m, N = result["timings"], result["metrics"], result["n_windows"]

    print("=" * 64)
    print(f"FLAIR on CPU (PyTorch, 1 thread, batch=1)  ({N} windows)")
    print("=" * 64)
    print(f"  ROC-AUC {m['auc']:.4f}   F1@p99 {m['f1']:.4f}")
    print(f"  CPU inference time: {t['cpu_inference_ms']:.2f} ms total "
          f"({t['cpu_us_per_window']:.1f} us/window)")
    print("=" * 64)


if __name__ == "__main__":
    main()
