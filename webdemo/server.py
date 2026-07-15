#!/usr/bin/env python3
"""
webdemo/server.py

Local web server for the FLAIR NPU demo site. Runs the CPU (PyTorch, batch=1,
1 thread) and the validated 4-CORE NPU pipeline concurrently over the full
thesis-style test set, streams live progress to a browser dashboard, and
reports the comparative speedup + accuracy once both finish.

Deliberately stdlib-only (http.server + threading, no Flask/FastAPI): this
project pins exact versions of a fragile IRON/XRT toolchain (see npu/*.py's
header comments), so the demo server avoids adding any new pip dependency to
that environment.

Real inference lives in webdemo/demo_backend.py (run_npu_pipeline /
run_cpu_pipeline). It needs torch/numpy/ml_dtypes, the hidden=64 model
(experiments/results/flair_h64_full.pt), the full dataset
(data/processed/retrain_test.npz), and the PREBUILT 4-core xclbins
(build/gru_4core.xclbin, build/decoder_4core.xclbin). Build those once, from
npu/ in the WSL IRON env:
    python3 run_dataset_inference.py --npz ../data/processed/retrain_test.npz \\
        --ckpt ../experiments/results/flair_h64_full.pt \\
        --batch-encoder 8 --batch-decoder 8 \\
        --encoder-4core --decoder-4core --decoder-mode unfused \\
        --sample 8 --skip-cpu-baseline
Then run this server (--skip-build is the default, so a click never recompiles).

For UI development without any of that installed, run with --mock: it
fabricates plausible progress/timing/accuracy numbers using only the stdlib.

Usage:
    python3 webdemo/server.py                          # real run, prebuilt 4-core xclbins
    python3 webdemo/server.py --mock                   # UI dev, no NPU/torch needed
"""

from __future__ import annotations

import argparse
import copy
import http.server
import json
import random
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_NPU_DIR = _REPO / "npu"
_STATIC_DIR = _HERE / "static"

# The real pipelines live in webdemo/demo_backend.py (run_npu_pipeline /
# run_cpu_pipeline for the validated 4-core NPU designs). demo_backend imports
# src.models (needs _REPO on path) and shells npu/batch_infer.exe from _NPU_DIR;
# it's in this server's own dir, so sys.path[0] already resolves it. Add the
# repo root + npu/ so its `src.models` / bare npu imports work when loaded here.
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_NPU_DIR))
sys.path.insert(0, str(_HERE))  # webdemo/ -> `import demo_backend`

# Full thesis-style test set (real attacks, z-clamped) + the properly-trained
# hidden=64 model, run through the validated 4-core NPU pipeline (see
# webdemo/demo_backend.py). Larger default window count than the old sample
# demo so the live run lasts several seconds and the speedup is visible.
DEFAULT_NPZ = str(_REPO / "data" / "processed" / "retrain_test.npz")
DEFAULT_SEQ_LEN = 10
DEFAULT_LIMIT = 10000

# ---------------------------------------------------------------------------
# Shared run state (guarded by _lock), polled by the frontend via /api/status.
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state: dict = {
    "state": "idle",  # idle | running | done | error
    "config": {"npz": DEFAULT_NPZ, "seq_len": DEFAULT_SEQ_LEN, "limit": DEFAULT_LIMIT},
    "cpu": {"done": 0, "total": 0, "finished": False, "elapsed_ms": 0.0},
    "npu": {"stage": None, "done": 0, "total": 0, "finished": False, "elapsed_ms": 0.0,
            "encoder_done": False},
    "error": None,
    "result": None,
    "mock": False,
}


def _reset_state(npz: str, seq_len: int, limit: int, mock: bool) -> None:
    with _lock:
        _state.update({
            "state": "running",
            "config": {"npz": npz, "seq_len": seq_len, "limit": limit},
            "cpu": {"done": 0, "total": limit, "finished": False, "elapsed_ms": 0.0},
            "npu": {"stage": "encoder", "done": 0, "total": limit, "finished": False,
                    "elapsed_ms": 0.0, "encoder_done": False},
            "error": None,
            "result": None,
            "mock": mock,
        })


def _snapshot() -> dict:
    with _lock:
        s = copy.deepcopy(_state)
    return s


def _json_fallback(o):
    """json.dumps(default=...) hook: numpy scalar types (float32, int64, ...)
    all expose .item() to convert to the equivalent native Python type."""
    item = getattr(o, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Mock pipelines (stdlib only) -- used for frontend development / demoing the
# UI on a machine without the NPU/torch toolchain installed.
# ---------------------------------------------------------------------------

def _mock_dataset(N: int):
    """Fixed seed so the CPU and NPU mock passes agree on which windows are
    "anomalies" and roughly how anomalous each one is -- real runs get this
    for free by reading the same npz. Each side still adds its own
    independent noise on top, so scores are correlated but not identical
    (matching what the real bf16 NPU vs float32 CPU comparison looks like)."""
    rng = random.Random(1234)
    labels, severity = [], []
    for _ in range(N):
        is_attack = rng.random() < 0.06
        labels.append(1 if is_attack else 0)
        severity.append(rng.uniform(0.6, 1.4) if is_attack else 0.0)
    return labels, severity


def _mock_cpu_pipeline(npz_path, seq_len, limit, on_progress=None, **_kw) -> dict:
    N = limit
    us_per_window = 115.0
    labels, severity = _mock_dataset(N)
    scores = []
    for i in range(N):
        base = random.uniform(0.15, 0.35)
        scores.append(base + severity[i])
        time.sleep(us_per_window / 1e6)
        if on_progress and ((i + 1) % max(1, N // 40) == 0 or i + 1 == N):
            on_progress(i + 1, N)
    total_ms = us_per_window * N / 1000.0
    m = _mock_metrics(scores, labels)
    return {
        "scores": scores, "labels": labels, "n_windows": N,
        "timings": {"cpu_inference_ms": total_ms, "cpu_us_per_window": us_per_window},
        "metrics": m,
    }


def _mock_npu_pipeline(npz_path, seq_len, limit, on_progress=None, **_kw) -> dict:
    N = limit
    us_per_window = 14.0
    labels, severity = _mock_dataset(N)
    scores = []
    half = max(1, N // 40)
    for stage in ("encoder", "decoder"):
        for i in range(N):
            time.sleep(us_per_window / 1e6)
            if on_progress and ((i + 1) % half == 0 or i + 1 == N):
                on_progress(stage, i + 1, N)
    for i in range(N):
        base = random.uniform(0.15, 0.35)
        jitter = random.uniform(-0.03, 0.03)  # bf16 + LUT-gate drift vs float32
        scores.append(base + severity[i] + jitter)
    total_ms = us_per_window * N * 2 / 1000.0
    m = _mock_metrics(scores, labels)
    return {
        "scores": scores, "pt_scores": scores, "labels": labels, "n_windows": N,
        "timings": {
            "encoder_ms": total_ms / 2, "decoder_ms": total_ms / 2,
            "npu_inference_ms": total_ms,
            "npu_us_per_window": (total_ms * 1000.0 / N) if N else 0.0,
        },
        "metrics": {**m, "corr": 0.993, "mean_rel_err": 0.021, "median_rel_err": 0.014},
    }


def _mock_metrics(scores, labels) -> dict:
    normals = sorted(s for s, l in zip(scores, labels) if l == 0)
    thr = normals[int(0.99 * (len(normals) - 1))] if normals else 0.5
    tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s > thr)
    fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s > thr)
    fn = sum(1 for s, l in zip(scores, labels) if l == 1 and s <= thr)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    auc = 0.985 + random.uniform(-0.01, 0.01)
    return {"auc": auc, "f1": f1, "npu_auc": auc, "pt_auc": auc, "npu_f1": f1, "pt_f1": f1}


# ---------------------------------------------------------------------------
# Orchestration: run CPU + NPU pipelines concurrently, combine results.
# ---------------------------------------------------------------------------

def _combine_results(cpu_result: dict, npu_result: dict) -> dict:
    cpu_ms = cpu_result["timings"]["cpu_inference_ms"]
    npu_ms = npu_result["timings"]["npu_inference_ms"]
    speedup = (cpu_ms / npu_ms) if npu_ms > 0 else float("nan")

    # Real pipelines return numpy arrays (float32/int64 scalars aren't
    # JSON-serializable), so cast to native Python types here.
    cpu_scores = [float(s) for s in cpu_result["scores"]]
    npu_scores = [float(s) for s in npu_result["scores"]]
    labels = [int(l) for l in npu_result["labels"]]

    examples = []
    if len(cpu_scores) == len(npu_scores) == len(labels):
        anomaly_idx = sorted(
            (i for i, l in enumerate(labels) if l == 1),
            key=lambda i: -npu_scores[i],
        )[:4]
        normal_idx = sorted(
            (i for i, l in enumerate(labels) if l == 0),
            key=lambda i: npu_scores[i],
        )[:2]
        for i in sorted(anomaly_idx + normal_idx):
            examples.append({
                "window": i,
                "label": "attack" if labels[i] == 1 else "normal",
                "cpu_score": cpu_scores[i],
                "npu_score": npu_scores[i],
            })

    return {
        "speedup": speedup,
        "cpu": {
            "inference_ms": cpu_ms,
            "us_per_window": cpu_result["timings"]["cpu_us_per_window"],
            "auc": cpu_result["metrics"]["auc"],
            "f1": cpu_result["metrics"]["f1"],
        },
        "npu": {
            "inference_ms": npu_ms,
            "us_per_window": npu_result["timings"]["npu_us_per_window"],
            "encoder_ms": npu_result["timings"]["encoder_ms"],
            "decoder_ms": npu_result["timings"]["decoder_ms"],
            "auc": npu_result["metrics"].get("npu_auc"),
            "f1": npu_result["metrics"].get("npu_f1"),
            "corr_vs_pytorch": npu_result["metrics"].get("corr"),
            "mean_rel_err": npu_result["metrics"].get("mean_rel_err"),
        },
        "n_windows": npu_result.get("n_windows"),
        "examples": examples,
    }


def _run_demo(npz: str, seq_len: int, limit: int, mock: bool,
              xrt_inc_dir: Optional[str], xrt_lib_dir: Optional[str]) -> None:
    _reset_state(npz, seq_len, limit, mock)

    cpu_result: dict = {}
    npu_result: dict = {}
    errors: list[tuple[str, str]] = []

    def cpu_progress(done, total):
        with _lock:
            _state["cpu"]["done"] = done
            _state["cpu"]["total"] = total

    def npu_progress(stage, done, total):
        with _lock:
            _state["npu"]["stage"] = stage
            _state["npu"]["done"] = done
            _state["npu"]["total"] = total
            if stage == "decoder":
                _state["npu"]["encoder_done"] = True

    def cpu_worker():
        start = time.perf_counter()
        try:
            if mock:
                cpu_result.update(_mock_cpu_pipeline(npz, seq_len, limit, on_progress=cpu_progress))
            else:
                from demo_backend import run_cpu_pipeline
                cpu_result.update(run_cpu_pipeline(
                    npz_path=npz, seq_len=seq_len, limit=limit, on_progress=cpu_progress,
                ))
        except Exception as e:  # noqa: BLE001
            errors.append(("cpu", str(e)))
        finally:
            with _lock:
                _state["cpu"]["finished"] = True
                _state["cpu"]["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
                if cpu_result:
                    _state["cpu"]["done"] = _state["cpu"]["total"]

    def npu_worker():
        start = time.perf_counter()
        try:
            if mock:
                npu_result.update(_mock_npu_pipeline(npz, seq_len, limit, on_progress=npu_progress))
            else:
                from demo_backend import run_npu_pipeline
                npu_result.update(run_npu_pipeline(
                    npz_path=npz, seq_len=seq_len, limit=limit, skip_build=True,
                    xrt_inc_dir=xrt_inc_dir, xrt_lib_dir=xrt_lib_dir,
                    on_progress=npu_progress,
                ))
        except Exception as e:  # noqa: BLE001
            errors.append(("npu", str(e)))
        finally:
            with _lock:
                _state["npu"]["finished"] = True
                _state["npu"]["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
                if npu_result:
                    _state["npu"]["done"] = _state["npu"]["total"]

    t_cpu = threading.Thread(target=cpu_worker, daemon=True)
    t_npu = threading.Thread(target=npu_worker, daemon=True)
    t_cpu.start()
    t_npu.start()
    t_cpu.join()
    t_npu.join()

    with _lock:
        if errors:
            _state["state"] = "error"
            _state["error"] = "; ".join(f"{who}: {msg}" for who, msg in errors)
            return
        try:
            combined = _combine_results(cpu_result, npu_result)
        except Exception as e:  # noqa: BLE001
            _state["state"] = "error"
            _state["error"] = f"combine: {e}"
            return
        _state["state"] = "done"
        _state["result"] = combined


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class DemoHandler(http.server.BaseHTTPRequestHandler):
    server_version = "FLAIRDemo/1.0"
    args: argparse.Namespace  # set on the class by main()

    def log_message(self, fmt, *a):  # quieter default logging
        print("[http] " + (fmt % a))

    def _send_json(self, obj: dict, status: int = 200) -> None:
        # Safety net: state/result dicts are built from real numpy pipeline
        # output (float32/int64 scalars aren't JSON-serializable on their
        # own). Known spots already cast explicitly; this catches anything
        # missed instead of crashing the whole request.
        body = json.dumps(obj, default=_json_fallback).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        if rel_path in ("", "/"):
            rel_path = "index.html"
        rel_path = rel_path.lstrip("/")
        target = (_STATIC_DIR / rel_path).resolve()
        if _STATIC_DIR not in target.parents and target != _STATIC_DIR:
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(_snapshot())
            return
        if parsed.path.startswith("/api/"):
            self.send_error(404)
            return
        self._send_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(404)
            return
        with _lock:
            already_running = _state["state"] == "running"
        if already_running:
            self._send_json({"error": "a run is already in progress"}, status=409)
            return

        qs = urllib.parse.parse_qs(parsed.query)
        npz = qs.get("npz", [self.args.npz])[0]
        seq_len = int(qs.get("seq_len", [self.args.seq_len])[0])
        limit = int(qs.get("limit", [self.args.limit])[0])

        t = threading.Thread(
            target=_run_demo,
            args=(npz, seq_len, limit, self.args.mock, self.args.xrt_inc_dir, self.args.xrt_lib_dir),
            daemon=True,
        )
        t.start()
        self._send_json({"status": "started"})


def main() -> None:
    p = argparse.ArgumentParser(description="FLAIR NPU-vs-CPU demo website")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="number of dataset windows to run per demo click")
    p.add_argument("--xrt-inc-dir", default=None)
    p.add_argument("--xrt-lib-dir", default=None)
    p.add_argument("--mock", action="store_true",
                   help="fabricate results with the stdlib only; for UI dev "
                        "without torch/ml_dtypes/NPU installed")
    args = p.parse_args()

    with _lock:
        _state["config"] = {"npz": args.npz, "seq_len": args.seq_len, "limit": args.limit}
        _state["mock"] = args.mock

    DemoHandler.args = args
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), DemoHandler)
    mode = "MOCK (fabricated data)" if args.mock else "REAL (CPU + NPU hardware)"
    print(f"FLAIR demo site [{mode}] -> http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
