#!/usr/bin/env python3
"""
run_dataset_inference.py

Runs the full FLAIR autoencoder over an entire dataset on the NPU and compares
the resulting anomaly scores + detection metrics against the PyTorch baseline.

Pipeline (two batched NPU passes with numpy glue in between):
  1. host : embeddings + pad -> all_x_windows.bin           (N x SEQ*48 bf16)
  2. NPU  : batch_infer.exe (encoder xclbin, loaded once)   -> all_latents.bin
  3. host : latent -> h0 = tanh(latent_to_hidden(latent))   -> all_h0.bin
  4. NPU  : batch_infer.exe (decoder xclbin, loaded once)   -> all_hidden.bin
  5. host : hidden_to_output -> MSE -> NPU anomaly scores
  6. host : PyTorch scores + threshold -> F1 / ROC-AUC for both, compared

Requires: the WSL IRON env sourced (incl. XRT setup.sh so xclbinutil is on
PATH), and the NPU visible to the Windows-side .exe. On native Windows set
XRT paths for the host build via --xrt-inc-dir / --xrt-lib-dir.

Usage (from npu/):
    python3 run_dataset_inference.py --limit 990        # full sample dataset
    python3 run_dataset_inference.py --npz /path/to/other.npz --limit 0

This module is also imported by webdemo/server.py (see run_npu_pipeline())
to drive the live comparison website -- the CLI below is just a thin wrapper
around the same function.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

INPUT_DIM = 45
INPUT_DIM_PADDED = 48
HIDDEN_DIM = 64
OUTPUT_DIM = 21

_CKPT = _REPO / "experiments" / "results" / "flair_minimal.pt"

_PROGRESS_RE = re.compile(r"processed\s+(\d+)/(\d+)\s+windows")
_TIMING_RE = re.compile(r"(\d+)\s+windows in\s+([\d.]+)\s+ms")

# Type of a progress callback: (stage, windows_done, windows_total) -> None.
# stage is one of "encoder", "decoder".
ProgressCB = Optional[Callable[[str, int, int], None]]


def sh(cmd: list[str], **kw) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=_HERE, check=True, **kw)


def sh_stream(cmd: list[str], stage: str, on_progress: ProgressCB = None) -> tuple[str, Optional[float]]:
    """Run cmd, forwarding its output line-by-line (splitting on '\\r' as well
    as '\\n', since batch_infer.exe overwrites its progress line with '\\r').

    Returns (full_captured_text, timing_ms) where timing_ms is parsed from a
    "<N> windows in <ms> ms" line if present (None otherwise). Raises
    CalledProcessError on nonzero exit, same as sh().
    """
    print("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=_HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    timing_ms: Optional[float] = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal timing_ms
        if not buf:
            return
        line = "".join(buf)
        buf.clear()
        lines.append(line)
        print(line)
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
    text = "\n".join(lines)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, output=text)
    return text, timing_ms


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_step_float(x, h, w_ih, w_hh, b_ih, b_hh):
    H = h.shape[0]
    gi = w_ih @ x + b_ih
    gh = w_hh @ h + b_hh
    r = sigmoid(gi[:H] + gh[:H])
    z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return (1.0 - z) * n + z * h


def f1_at_percentile(scores, labels, pct=99.0):
    """Threshold = percentile of NORMAL scores; return (f1, threshold)."""
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


def load_model_and_data(npz_path: str, limit: int):
    import torch
    from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd)
    model.eval()

    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)   # (N, T, 21)
    X_cat = bundle["X_cat"].astype(np.int64)     # (N, T, 3)
    y = bundle["y_seq"].astype(np.int64)
    N = X_num.shape[0] if limit == 0 else min(limit, X_num.shape[0])
    X_num, X_cat, y = X_num[:N], X_cat[:N], y[:N]
    return model, sd, X_num, X_cat, y, N


def prepare_encoder_inputs(sd, X_num, X_cat, N: int, T: int):
    sport_w = sd["sport_emb.weight"].numpy()
    dport_w = sd["dport_emb.weight"].numpy()
    proto_w = sd["proto_emb.weight"].numpy()
    x_windows = np.zeros((N, T, INPUT_DIM_PADDED), dtype=bfloat16)
    for w in range(N):
        for t in range(T):
            xin = np.concatenate([
                X_num[w, t],
                sport_w[X_cat[w, t, 0]],
                dport_w[X_cat[w, t, 1]],
                proto_w[X_cat[w, t, 2]],
            ]).astype(bfloat16)
            x_windows[w, t, :INPUT_DIM] = xin  # last 3 stay 0 (pad)
    (_HERE / "all_x_windows.bin").write_bytes(x_windows.reshape(N, -1).tobytes())

    w_ih_e = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_e = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_e = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_e = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    w_ih_e_pad = np.zeros((3 * HIDDEN_DIM, INPUT_DIM_PADDED), dtype=bfloat16)
    w_ih_e_pad[:, :INPUT_DIM] = w_ih_e
    enc_params = np.concatenate(
        [w_ih_e_pad.reshape(-1), w_hh_e.reshape(-1), b_ih_e, b_hh_e]
    ).astype(bfloat16)
    (_HERE / "enc_params.bin").write_bytes(enc_params.tobytes())
    n_enc_params = enc_params.size
    enc_in1_vol = T * INPUT_DIM_PADDED
    return n_enc_params, enc_in1_vol


def prepare_decoder_params(sd):
    w_ih_d = sd["decoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_d = sd["decoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_d = sd["decoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_d = sd["decoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    dec_params = np.concatenate(
        [w_ih_d.reshape(-1), w_hh_d.reshape(-1), b_ih_d, b_hh_d]
    ).astype(bfloat16)
    (_HERE / "dec_params.bin").write_bytes(dec_params.tobytes())
    return dec_params.size


def build_designs(seq_len: int, xrt_inc_dir: Optional[str], xrt_lib_dir: Optional[str]) -> None:
    xf = []
    if xrt_inc_dir:
        xf.append(f"XRT_INC_DIR={xrt_inc_dir}")
    if xrt_lib_dir:
        xf.append(f"XRT_LIB_DIR={xrt_lib_dir}")
    sh(["python3", "gru_encoder.py", "--dev", "npu", "--input-dim",
        str(INPUT_DIM_PADDED), "--hidden-dim", str(HIDDEN_DIM), "--seq-len",
        str(seq_len), "--xclbin-path", "build/gru.xclbin", "--insts-path",
        "build/insts.bin"])
    sh(["python3", "gru_decoder.py", "--dev", "npu", "--hidden-dim",
        str(HIDDEN_DIM), "--seq-len", str(seq_len), "--xclbin-path",
        "build/decoder.xclbin", "--insts-path", "build/decoder_insts.bin"])
    sh(["make", "-f", "Makefile.batch"] + xf)


def run_npu_pipeline(
    npz_path: str,
    seq_len: int = 10,
    limit: int = 990,
    skip_build: bool = False,
    xrt_inc_dir: Optional[str] = None,
    xrt_lib_dir: Optional[str] = None,
    on_progress: ProgressCB = None,
) -> dict:
    """Runs the full NPU pipeline (encoder pass + decoder pass + host glue)
    over `limit` windows of `npz_path` and returns a result dict:

      scores        : (N,) NPU-derived anomaly scores
      pt_scores     : (N,) PyTorch anomaly scores (same windows)
      labels        : (N,) ground-truth labels
      timings       : {"encoder_ms", "decoder_ms", "npu_inference_ms",
                        "npu_us_per_window"} -- pure streamed-inference time,
                        i.e. excludes xclbin build + one-time load.
      metrics       : {"npu_auc", "pt_auc", "npu_f1", "pt_f1", "corr",
                        "mean_rel_err"}

    Requires build_designs() (or an equivalent `make`) to have produced
    build/gru.xclbin, build/decoder.xclbin and batch_infer.exe already when
    skip_build=True -- that's the expected mode for the live web demo, so a
    slow AIE recompile never happens on the user-facing "Run Demo" click.
    """
    T = seq_len
    model, sd, X_num, X_cat, y, N = load_model_and_data(npz_path, limit)
    print(f"Dataset: {N} windows, {int(y.sum())} anomalies, T={T}")

    n_enc_params, enc_in1_vol = prepare_encoder_inputs(sd, X_num, X_cat, N, T)
    n_dec_params = prepare_decoder_params(sd)

    if not skip_build:
        build_designs(T, xrt_inc_dir, xrt_lib_dir)

    ps = "powershell.exe"

    print("\n[encoder] batched NPU pass")
    _, encoder_ms = sh_stream(
        [ps, "./batch_infer.exe", "build/gru.xclbin", "build/insts.bin",
         "all_x_windows.bin", "enc_params.bin", "all_latents.bin",
         str(N), str(enc_in1_vol), str(n_enc_params), str(HIDDEN_DIM)],
        stage="encoder", on_progress=on_progress,
    )

    latents = np.frombuffer((_HERE / "all_latents.bin").read_bytes(),
                            dtype=bfloat16).reshape(N, HIDDEN_DIM).astype(np.float32)
    W_lh = sd["decoder.latent_to_hidden.weight"].numpy().astype(np.float32)
    b_lh = sd["decoder.latent_to_hidden.bias"].numpy().astype(np.float32)
    h0 = np.tanh(latents @ W_lh.T + b_lh).astype(bfloat16)  # (N, 64)
    (_HERE / "all_h0.bin").write_bytes(h0.tobytes())

    print("\n[decoder] batched NPU pass")
    _, decoder_ms = sh_stream(
        [ps, "./batch_infer.exe", "build/decoder.xclbin",
         "build/decoder_insts.bin", "all_h0.bin", "dec_params.bin",
         "all_hidden.bin", str(N), str(HIDDEN_DIM), str(n_dec_params),
         str(T * HIDDEN_DIM)],
        stage="decoder", on_progress=on_progress,
    )

    hidden = np.frombuffer((_HERE / "all_hidden.bin").read_bytes(),
                           dtype=bfloat16).reshape(N, T, HIDDEN_DIM).astype(np.float32)
    W_out = sd["decoder.hidden_to_output.weight"].numpy().astype(np.float32)
    b_out = sd["decoder.hidden_to_output.bias"].numpy().astype(np.float32)
    recon = hidden @ W_out.T + b_out                       # (N, T, 21)
    npu_scores = np.mean((recon - X_num[:, :T]) ** 2, axis=(1, 2))

    import torch
    with torch.no_grad():
        pt_scores = model.anomaly_score(
            torch.from_numpy(X_num), torch.from_numpy(X_cat)
        ).numpy()

    npu_f1, _ = f1_at_percentile(npu_scores, y)
    pt_f1, _ = f1_at_percentile(pt_scores, y)
    npu_auc = roc_auc(npu_scores, y)
    pt_auc = roc_auc(pt_scores, y)
    corr = float(np.corrcoef(npu_scores, pt_scores)[0, 1])
    rel = np.abs(npu_scores - pt_scores) / (np.abs(pt_scores) + 1e-9)

    encoder_ms = encoder_ms or 0.0
    decoder_ms = decoder_ms or 0.0
    total_ms = encoder_ms + decoder_ms

    return {
        "scores": npu_scores,
        "pt_scores": pt_scores,
        "labels": y,
        "n_windows": N,
        "timings": {
            "encoder_ms": encoder_ms,
            "decoder_ms": decoder_ms,
            "npu_inference_ms": total_ms,
            "npu_us_per_window": (total_ms * 1000.0 / N) if N else 0.0,
        },
        "metrics": {
            "npu_auc": npu_auc,
            "pt_auc": pt_auc,
            "npu_f1": npu_f1,
            "pt_f1": pt_f1,
            "corr": corr,
            "mean_rel_err": float(rel.mean()),
            "median_rel_err": float(np.median(rel)),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str,
                   default=str(_REPO / "data" / "processed" / "preprocessed.npz"))
    p.add_argument("--seq-len", type=int, default=10,
                   help="encoder AND decoder sequence length (= window size)")
    p.add_argument("--limit", type=int, default=0,
                   help="max windows (0 = all)")
    p.add_argument("--xrt-inc-dir", type=str, default=None)
    p.add_argument("--xrt-lib-dir", type=str, default=None)
    p.add_argument("--skip-build", action="store_true",
                   help="reuse existing xclbins + batch_infer.exe")
    args = p.parse_args()

    result = run_npu_pipeline(
        npz_path=args.npz, seq_len=args.seq_len, limit=args.limit,
        skip_build=args.skip_build, xrt_inc_dir=args.xrt_inc_dir,
        xrt_lib_dir=args.xrt_lib_dir,
    )

    npu_scores, pt_scores, y = result["scores"], result["pt_scores"], result["labels"]
    N = result["n_windows"]
    m = result["metrics"]
    t = result["timings"]

    print("\n" + "=" * 64)
    print(f"FLAIR on NPU vs PyTorch  ({N} windows)")
    print("=" * 64)
    print(f"  per-window score: mean rel err {m['mean_rel_err']*100:.2f}%  "
          f"median {m['median_rel_err']*100:.2f}%")
    print(f"  score correlation (Pearson r) : {m['corr']:.4f}")
    print("-" * 64)
    print(f"  ROC-AUC   NPU {m['npu_auc']:.4f}   PyTorch {m['pt_auc']:.4f}")
    print(f"  F1 @p99   NPU {m['npu_f1']:.4f}   PyTorch {m['pt_f1']:.4f}")
    print("-" * 64)
    print(f"  NPU inference time: {t['npu_inference_ms']:.2f} ms total "
          f"({t['npu_us_per_window']:.1f} us/window) "
          f"[encoder {t['encoder_ms']:.2f} ms, decoder {t['decoder_ms']:.2f} ms]")
    print("=" * 64)

    out_csv = _HERE / "npu_vs_pytorch_scores.csv"
    with open(out_csv, "w") as f:
        f.write("window,label,npu_score,pytorch_score\n")
        for i in range(N):
            f.write(f"{i},{int(y[i])},{npu_scores[i]:.6f},{pt_scores[i]:.6f}\n")
    print(f"per-window scores -> {out_csv}")


if __name__ == "__main__":
    main()
