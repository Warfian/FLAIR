"""
benchmark_inference.py

Establishes a fair CPU baseline for single-window (batch=1) FLAIR inference
latency, to compare against a future IRON/NPU implementation.

Measures three variants, since naive eager-mode Python is not a fair baseline:
  1. Eager mode  (model.anomaly_score call, as evaluate_flair.py does it)
  2. TorchScript (torch.jit.trace'd module, removes Python dispatch overhead)
  3. Eager mode, single-thread (torch.set_num_threads(1))

For each variant reports mean/median/p95/p99 latency in microseconds over
many repeated single-window calls, after a warmup period.

Usage:
    python -m scripts.benchmark_inference
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import yaml

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

WARMUP_ITERS = 200
TIMED_ITERS = 2000


class AnomalyScoreWrapper(torch.nn.Module):
    """Wraps FLAIRAutoencoder.anomaly_score so it returns a plain tensor
    (not a dict), which torch.jit.trace requires for a stable output structure."""

    def __init__(self, model: FLAIRAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        out = self.model.forward(x_num, x_cat)
        x_hat = out["x_hat_num"]
        return torch.mean((x_hat - x_num) ** 2, dim=(1, 2))


@dataclass
class BenchConfig:
    checkpoint_path: str
    npz_path: str
    device: str = "cpu"


def load_checkpoint(checkpoint_path: str, device: torch.device) -> FLAIRAutoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_single_window(npz_path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bundle = np.load(npz_path, allow_pickle=True)
    x_num = torch.from_numpy(bundle["X_num"][:1].astype(np.float32)).to(device)
    x_cat = torch.from_numpy(bundle["X_cat"][:1].astype(np.int64)).to(device)
    return x_num, x_cat


def time_calls(fn: Callable[[], None], warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()

    samples_us: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        samples_us.append((end - start) * 1e6)
    return samples_us


def summarize(name: str, samples_us: List[float]) -> Dict[str, float]:
    arr = np.array(samples_us)
    stats = {
        "mean_us": float(arr.mean()),
        "median_us": float(np.median(arr)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
    }
    print(f"\n=== {name} ===")
    for k, v in stats.items():
        print(f"  {k:10s}: {v:9.2f} us")
    return stats


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    training_cfg = cfg_yaml.get("training", {})
    paths_cfg = cfg_yaml.get("paths", {})

    cfg = BenchConfig(
        checkpoint_path=str(training_cfg.get("checkpoint_path", "experiments/results/flair_minimal.pt")),
        npz_path=str(paths_cfg.get("processed_npz", "data/processed/preprocessed.npz")),
    )

    device = torch.device(cfg.device)
    default_threads = torch.get_num_threads()

    model = load_checkpoint(cfg.checkpoint_path, device)
    x_num, x_cat = load_single_window(cfg.npz_path, device)
    print(f"Benchmarking single-window inference: x_num={tuple(x_num.shape)} x_cat={tuple(x_cat.shape)}")
    print(f"Default torch intra-op threads: {default_threads}")

    results: Dict[str, Dict[str, float]] = {}

    # 1. Eager mode, default thread count (mirrors evaluate_flair.py as-is)
    def eager_call() -> None:
        with torch.no_grad():
            model.anomaly_score(x_num, x_cat)

    results["eager_default_threads"] = summarize(
        f"Eager mode (default threads={default_threads})",
        time_calls(eager_call, WARMUP_ITERS, TIMED_ITERS),
    )

    # 2. Eager mode, single thread (removes intra-op parallelism overhead,
    #    which dominates for a model this small and matches a realistic
    #    single-stream edge deployment scenario)
    torch.set_num_threads(1)

    results["eager_single_thread"] = summarize(
        "Eager mode (1 thread)",
        time_calls(eager_call, WARMUP_ITERS, TIMED_ITERS),
    )

    # 3. TorchScript traced, single thread (closest fair CPU baseline:
    #    static graph, no Python dispatch per-op)
    wrapper = AnomalyScoreWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (x_num, x_cat))
    traced.eval()

    def traced_call() -> None:
        with torch.no_grad():
            traced(x_num, x_cat)

    results["torchscript_single_thread"] = summarize(
        "TorchScript traced (1 thread)",
        time_calls(traced_call, WARMUP_ITERS, TIMED_ITERS),
    )

    torch.set_num_threads(default_threads)

    print("\n=== Summary (median latency, single window / batch=1) ===")
    for name, stats in results.items():
        print(f"  {name:28s}: {stats['median_us']:9.2f} us  (p99={stats['p99_us']:9.2f} us)")

    print(
        "\nUse 'torchscript_single_thread' median as the fair CPU baseline "
        "to compare against a future IRON/NPU implementation."
    )


if __name__ == "__main__":
    main()
