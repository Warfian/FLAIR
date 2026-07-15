#!/usr/bin/env python3
"""
webdemo/demo_backend.py

Pipeline functions the demo server (webdemo/server.py) imports:
  * run_npu_pipeline(...)  -- the validated 4-CORE NPU pipeline
  * run_cpu_pipeline(...)  -- PyTorch, one window at a time (batch=1, 1 thread)

This is a self-contained copy of the pipeline glue (kept OUT of the validated
CLI npu/run_dataset_inference.py so that file stays untouched). It targets:
  * checkpoint : experiments/results/flair_h64_full.pt (properly-trained hidden=64)
  * dataset    : data/processed/retrain_test.npz (full thesis-style test set,
                 already z-clamped, contains real attacks)
  * NPU code   : the 4-core data-parallel encoder + decoder xclbins
                 (build/gru_4core.xclbin, build/decoder_4core.xclbin), i.e. the
                 same designs run_dataset_inference.py validated at F1 0.9349.

The 4-core designs process 8 windows per dispatch (2 per core across the
column); batch_infer.exe presents them as batch=8 and the memtile scatters/
gathers internally, so N must be padded to a multiple of 8 (extra rows are
discarded before scoring).

Build the xclbins once (from npu/, WSL IRON env) before running the demo:
    python3 run_dataset_inference.py --npz ../data/processed/retrain_test.npz \\
        --ckpt ../experiments/results/flair_h64_full.pt \\
        --batch-encoder 8 --batch-decoder 8 \\
        --encoder-4core --decoder-4core --decoder-mode unfused \\
        --sample 8 --skip-cpu-baseline
Then the demo runs with --skip-build (default) and never recompiles on a click.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_NPU_DIR = _REPO / "npu"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_NPU_DIR))

INPUT_DIM = 45
INPUT_DIM_PADDED = 48
HIDDEN_DIM = 64
OUTPUT_DIM = 21
SEQ_LEN = 10
BATCH_4CORE = 8  # windows per dispatch for the 4-core designs (2 per core x 4)

# Defaults: the properly-trained hidden=64 model + the full thesis-style test
# set (real attacks, already z-clamped). Overridable by the server.
_CKPT = _REPO / "experiments" / "results" / "flair_h64_full.pt"
_NPZ = _REPO / "data" / "processed" / "retrain_test.npz"

_PROGRESS_RE = re.compile(r"processed\s+(\d+)/(\d+)\s+windows")
_TIMING_RE = re.compile(r"(\d+)\s+windows in .*?([\d.]+)\s+ms total")

# Progress callbacks.
NpuProgressCB = Optional[Callable[[str, int, int], None]]   # (stage, done, total)
CpuProgressCB = Optional[Callable[[int, int], None]]         # (done, total)


def f1_at_percentile(scores, labels, pct=99.0):
    thr = float(np.percentile(scores[labels == 0], pct))
    pred = (scores > thr).astype(np.int64)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, thr


def roc_auc(scores, labels):
    order = np.argsort(-scores)
    y = labels[order]
    P = int((y == 1).sum())
    Nn = int((y == 0).sum())
    if P == 0 or Nn == 0:
        return float("nan")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / Nn]
    return float(np.trapezoid(tpr, fpr))


def load_model_and_data(npz_path: str, limit: int, ckpt_path: Optional[str] = None):
    import torch
    from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

    ckpt = torch.load(str(ckpt_path or _CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd, strict=False)  # flair_h64_full has unused cat heads
    model.eval()

    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y = bundle["y_seq"].astype(np.int64)
    N = X_num.shape[0] if limit == 0 else min(limit, X_num.shape[0])
    X_num, X_cat, y = X_num[:N], X_cat[:N], y[:N]
    return model, sd, X_num, X_cat, y, N


def sh_stream(cmd, stage: str, on_progress: NpuProgressCB = None):
    """Run cmd, forwarding output live (batch_infer overwrites its progress
    line with '\\r'). Returns (text, timing_ms) where timing_ms is parsed from
    the 'N windows in ... MS ms total' line."""
    print("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=_NPU_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    lines, timing_ms, buf = [], None, []

    def flush():
        nonlocal timing_ms
        if not buf:
            return
        line = "".join(buf)
        buf.clear()
        lines.append(line)
        m = _PROGRESS_RE.search(line)
        if m and on_progress:
            on_progress(stage, int(m.group(1)), int(m.group(2)))
        m = _TIMING_RE.search(line)
        if m:
            timing_ms = float(m.group(2))

    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        if ch in ("\r", "\n"):
            flush()
        else:
            buf.append(ch)
    flush()
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, output="\n".join(lines))
    return "\n".join(lines), timing_ms


def _prepare_inputs(sd, X_num, X_cat, N: int, N_pad: int, T: int):
    """Write the padded (to N_pad, a multiple of BATCH_4CORE) encoder input +
    both param blobs the 4-core kernels consume."""
    # Vectorized embedding lookup + concat (the batch CLI uses a Python
    # double-loop; the demo vectorizes it so the progress bar isn't stalled
    # for seconds at 10k windows). Identical result: [x_num | sport_e |
    # dport_e | proto_e] per timestep, padded to INPUT_DIM_PADDED.
    sport_w = sd["sport_emb.weight"].numpy()
    dport_w = sd["dport_emb.weight"].numpy()
    proto_w = sd["proto_emb.weight"].numpy()
    sport_e = sport_w[X_cat[:, :, 0]]   # (N, T, 8)
    dport_e = dport_w[X_cat[:, :, 1]]
    proto_e = proto_w[X_cat[:, :, 2]]
    xcat = np.concatenate([X_num, sport_e, dport_e, proto_e], axis=-1)  # (N,T,45)
    x = np.zeros((N_pad, T, INPUT_DIM_PADDED), dtype=bfloat16)
    x[:N, :, :INPUT_DIM] = xcat.astype(bfloat16)
    (_NPU_DIR / "all_x_windows.bin").write_bytes(x.reshape(N_pad, -1).tobytes())

    w_ih_e = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_e = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_e = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_e = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    w_ih_e_pad = np.zeros((3 * HIDDEN_DIM, INPUT_DIM_PADDED), dtype=bfloat16)
    w_ih_e_pad[:, :INPUT_DIM] = w_ih_e
    enc_params = np.concatenate(
        [w_ih_e_pad.reshape(-1), w_hh_e.reshape(-1), b_ih_e, b_hh_e]).astype(bfloat16)
    (_NPU_DIR / "enc_params.bin").write_bytes(enc_params.tobytes())

    w_ih_d = sd["decoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_d = sd["decoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_d = sd["decoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_d = sd["decoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    dec_params = np.concatenate(
        [w_ih_d.reshape(-1), w_hh_d.reshape(-1), b_ih_d, b_hh_d]).astype(bfloat16)
    (_NPU_DIR / "dec_params.bin").write_bytes(dec_params.tobytes())

    return enc_params.size, dec_params.size, T * INPUT_DIM_PADDED


def run_npu_pipeline(
    npz_path: str = str(_NPZ),
    seq_len: int = SEQ_LEN,
    limit: int = 5000,
    skip_build: bool = True,
    ckpt_path: Optional[str] = None,
    on_progress: NpuProgressCB = None,
    **_kw,
) -> dict:
    """Full 4-core NPU pipeline over `limit` windows. Returns scores, pt_scores,
    labels, timings, metrics (same shape the demo server expects)."""
    T = seq_len
    B = BATCH_4CORE
    model, sd, X_num, X_cat, y, N = load_model_and_data(npz_path, limit, ckpt_path)
    N_pad = ((N + B - 1) // B) * B  # pad up to a multiple of 8 for the 4-core designs
    print(f"Demo NPU: {N} windows ({int(y.sum())} anomalies), padded to {N_pad}")

    n_enc_params, n_dec_params, enc_in1_vol = _prepare_inputs(sd, X_num, X_cat, N, N_pad, T)
    ps = "powershell.exe"

    print("\n[encoder] 4-core NPU pass")
    _, encoder_ms = sh_stream(
        [ps, "./batch_infer.exe", "build/gru_4core.xclbin", "build/gru_4core_insts.bin",
         "all_x_windows.bin", "enc_params.bin", "all_latents.bin",
         str(N_pad), str(B), str(enc_in1_vol), str(n_enc_params), str(HIDDEN_DIM)],
        stage="encoder", on_progress=on_progress)

    latents = np.frombuffer((_NPU_DIR / "all_latents.bin").read_bytes(),
                            dtype=bfloat16).reshape(N_pad, HIDDEN_DIM).astype(np.float32)
    W_lh = sd["decoder.latent_to_hidden.weight"].numpy().astype(np.float32)
    b_lh = sd["decoder.latent_to_hidden.bias"].numpy().astype(np.float32)
    h0 = np.tanh(latents @ W_lh.T + b_lh).astype(bfloat16)
    (_NPU_DIR / "all_h0.bin").write_bytes(h0.tobytes())

    print("\n[decoder] 4-core NPU pass (unfused)")
    _, decoder_ms = sh_stream(
        [ps, "./batch_infer.exe", "build/decoder_4core.xclbin", "build/decoder_4core_insts.bin",
         "all_h0.bin", "dec_params.bin", "all_hidden.bin",
         str(N_pad), str(B), str(HIDDEN_DIM), str(n_dec_params), str(T * HIDDEN_DIM)],
        stage="decoder", on_progress=on_progress)

    # Discard the N_pad-N padding rows before scoring.
    hidden = np.frombuffer((_NPU_DIR / "all_hidden.bin").read_bytes(),
                           dtype=bfloat16).reshape(N_pad, T, HIDDEN_DIM).astype(np.float32)[:N]
    W_out = sd["decoder.hidden_to_output.weight"].numpy().astype(np.float32)
    b_out = sd["decoder.hidden_to_output.bias"].numpy().astype(np.float32)
    recon = hidden @ W_out.T + b_out
    npu_scores = np.mean((recon - X_num[:, :T]) ** 2, axis=(1, 2))

    import torch
    with torch.no_grad():
        pt_scores = model.anomaly_score(
            torch.from_numpy(X_num), torch.from_numpy(X_cat)).numpy()

    npu_f1, _ = f1_at_percentile(npu_scores, y)
    pt_f1, _ = f1_at_percentile(pt_scores, y)
    corr = float(np.corrcoef(npu_scores, pt_scores)[0, 1])
    rel = np.abs(npu_scores - pt_scores) / (np.abs(pt_scores) + 1e-9)
    encoder_ms = encoder_ms or 0.0
    decoder_ms = decoder_ms or 0.0
    total_ms = encoder_ms + decoder_ms

    return {
        "scores": npu_scores, "pt_scores": pt_scores, "labels": y, "n_windows": N,
        "timings": {
            "encoder_ms": encoder_ms, "decoder_ms": decoder_ms,
            "npu_inference_ms": total_ms,
            "npu_us_per_window": (total_ms * 1000.0 / N) if N else 0.0,
        },
        "metrics": {
            "npu_auc": roc_auc(npu_scores, y), "pt_auc": roc_auc(pt_scores, y),
            "npu_f1": npu_f1, "pt_f1": pt_f1, "corr": corr,
            "mean_rel_err": float(rel.mean()), "median_rel_err": float(np.median(rel)),
        },
    }


def run_cpu_pipeline(
    npz_path: str = str(_NPZ),
    seq_len: int = SEQ_LEN,
    limit: int = 5000,
    ckpt_path: Optional[str] = None,
    on_progress: CpuProgressCB = None,
    progress_every: int = 50,
    **_kw,
) -> dict:
    """PyTorch FLAIR inference window-by-window (batch=1, 1 thread) -- the fair
    single-stream edge baseline the NPU kernel targets."""
    import torch
    model, _sd, X_num, X_cat, y, N = load_model_and_data(npz_path, limit, ckpt_path)
    X_num, X_cat = X_num[:, :seq_len], X_cat[:, :seq_len]

    default_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x_num_t, x_cat_t = torch.from_numpy(X_num), torch.from_numpy(X_cat)

    scores = np.empty(N, dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N):
            scores[i] = float(model.anomaly_score(x_num_t[i:i + 1], x_cat_t[i:i + 1]).item())
            if on_progress and ((i + 1) % progress_every == 0 or i + 1 == N):
                on_progress(i + 1, N)
    total_ms = (time.perf_counter() - t0) * 1000.0
    torch.set_num_threads(default_threads)

    return {
        "scores": scores, "labels": y, "n_windows": N,
        "timings": {
            "cpu_inference_ms": total_ms,
            "cpu_us_per_window": (total_ms * 1000.0 / N) if N else 0.0,
        },
        "metrics": {"auc": roc_auc(scores, y), "f1": f1_at_percentile(scores, y)[0]},
    }
