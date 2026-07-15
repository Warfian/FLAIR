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
    python3 run_dataset_inference.py --batch-encoder 8 --batch-decoder 6 --limit 0
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

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


def sh(cmd: list[str], **kw) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=_HERE, check=True, **kw)


def sh_capture(cmd: list[str], **kw) -> str:
    """Like sh(), but also returns captured stdout (still echoed live-ish after the call)."""
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=_HERE, check=True, capture_output=True, text=True, **kw)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def parse_us_per_window(stdout: str) -> float | None:
    m = re.search(r"([\d.]+)\s*us/window", stdout)
    return float(m.group(1)) if m else None


def cpu_single_window_latency(
    model, X_num: np.ndarray, X_cat: np.ndarray, *,
    threads: int, use_torchscript: bool, warmup: int, iters: int,
) -> float:
    """Fair single-window (batch=1) CPU latency in us/window, cycling through
    real dataset windows. Mirrors scripts/benchmark_inference.py's methodology
    so this is comparable to the previously-established CPU baseline."""
    import torch
    from scripts.benchmark_inference import AnomalyScoreWrapper

    default_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    N = X_num.shape[0]

    if use_torchscript:
        wrapper = AnomalyScoreWrapper(model).eval()
        x0 = torch.from_numpy(X_num[:1])
        c0 = torch.from_numpy(X_cat[:1])
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (x0, c0))
        traced.eval()

        def call(i: int) -> None:
            with torch.no_grad():
                traced(torch.from_numpy(X_num[i:i + 1]), torch.from_numpy(X_cat[i:i + 1]))
    else:
        def call(i: int) -> None:
            with torch.no_grad():
                model.anomaly_score(
                    torch.from_numpy(X_num[i:i + 1]), torch.from_numpy(X_cat[i:i + 1])
                )

    for i in range(min(warmup, N)):
        call(i % N)

    t0 = time.perf_counter()
    for i in range(iters):
        call(i % N)
    t1 = time.perf_counter()

    torch.set_num_threads(default_threads)
    return (t1 - t0) * 1e6 / iters


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


def detection_metrics(scores, labels, pct=99.0):
    """Full per-path detection breakdown at a NORMAL-only pth-percentile
    threshold. Each path (NPU vs PyTorch) is calibrated on its OWN score
    distribution -- NPU bf16/LUT scores are on a slightly different scale than
    PyTorch fp32, so a shared/absolute threshold is misleading; the calibrated
    detection metrics are what should be compared. Returns a dict."""
    thr = float(np.percentile(scores[labels == 0], pct))
    pred = (scores > thr).astype(np.int64)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    tn = int(((labels == 0) & (pred == 0)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(thr=thr, tp=tp, fp=fp, tn=tn, fn=fn,
                prec=prec, rec=rec, f1=f1, fpr=fpr)


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


def main() -> None:
    import torch
    from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str,
                   default=str(_REPO / "data" / "processed" / "preprocessed.npz"))
    p.add_argument("--seq-len", type=int, default=10,
                   help="encoder AND decoder sequence length (= window size)")
    p.add_argument("--limit", type=int, default=0,
                   help="max windows (0 = all)")
    p.add_argument("--sample", type=int, default=0,
                   help="if >0, randomly sample this many windows (after "
                        "--limit) with a fixed seed. Use for a BALANCED labeled "
                        "subset: the train split's anomalies are chronologically "
                        "clustered (none before window ~223k), so a --limit "
                        "prefix is all-normal and yields undefined ROC-AUC; a "
                        "random --sample of train gives ~13.5%% anomalies.")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for --sample (reproducible subset).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint to load (default: flair_minimal.pt). Use "
                        "experiments/results/flair_h64_full.pt with the "
                        "--split-eval retrain_test.npz to validate the properly "
                        "trained NPU-sized model on hardware. Same hidden=64, so "
                        "weights drop into the param .bin -- no xclbin rebuild.")
    p.add_argument("--xrt-inc-dir", type=str, default=None)
    p.add_argument("--xrt-lib-dir", type=str, default=None)
    p.add_argument("--skip-build", action="store_true",
                   help="reuse existing xclbins + batch_infer.exe")
    p.add_argument("--skip-cpu-baseline", action="store_true",
                   help="skip the CPU single-window latency comparison")
    p.add_argument("--cpu-warmup-iters", type=int, default=50)
    p.add_argument("--cpu-timed-iters", type=int, default=500)
    p.add_argument("--batch-encoder", type=int, default=8,
                   help="windows processed per encoder NPU kernel dispatch. "
                        "Encoder's tile has more L1 headroom than the "
                        "decoder's, so it can run a larger batch (8 "
                        "compiled successfully on real hardware).")
    p.add_argument("--batch-decoder", type=int, default=6,
                   help="windows processed per decoder NPU kernel dispatch. "
                        "Uses the unfused decoder (hidden_to_output done on "
                        "host); the fused-on-core variant was measured SLOWER "
                        "on real hardware (added ~1900us fixed per-dispatch "
                        "cost for reasons in the compiled design, not the "
                        "matvec compute), so it's not used here. Decoder tile "
                        "budget caps this near 6-7 (1408B/window output).")
    p.add_argument("--encoder-4core", action="store_true",
                   help="use the 4-core data-parallel encoder (one NPU column, "
                        "memtile weight broadcast) instead of the single-core "
                        "encoder. batch-encoder must be a multiple of 4; each "
                        "core runs batch-encoder/4 windows. ~3.5x encoder "
                        "throughput. Output is bf16-noise-close (~0.003) to the "
                        "single-core -- validated by F1, not bit-equality.")
    args = p.parse_args()
    T = args.seq_len
    B_enc = args.batch_encoder
    B_dec = args.batch_decoder
    if args.encoder_4core and B_enc % 4 != 0:
        sys.exit(f"--encoder-4core requires --batch-encoder a multiple of 4 "
                 f"(got {B_enc})")
    B_lcm = math.lcm(B_enc, B_dec)

    ckpt_path = Path(args.ckpt) if args.ckpt else _CKPT
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = FLAIRConfig(**ckpt["model_cfg"])
    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd, strict=False)  # checkpoint may include unused cat-loss heads
    model.eval()

    bundle = np.load(args.npz, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)   # (N, T, 21)
    X_cat = bundle["X_cat"].astype(np.int64)     # (N, T, 3)
    y = bundle["y_seq"].astype(np.int64)
    N = X_num.shape[0] if args.limit == 0 else min(args.limit, X_num.shape[0])
    X_num, X_cat, y = X_num[:N], X_cat[:N], y[:N]
    if args.sample and args.sample < N:
        rng = np.random.default_rng(args.sample_seed)
        sel = np.sort(rng.choice(N, size=args.sample, replace=False))
        X_num, X_cat, y = X_num[sel], X_cat[sel], y[sel]
        N = args.sample
        print(f"Sampled {N} windows (seed={args.sample_seed}) from the split")
    print(f"Dataset: {N} windows, {int(y.sum())} anomalies, T={T}")

    # NPU kernels process B_enc/B_dec windows per dispatch (see --batch-*);
    # pad the window count up to a multiple of BOTH (their LCM) so the same
    # padded window set is a whole number of dispatches for both passes. The
    # extra rows are all-zero inputs; their outputs are discarded before
    # scoring, below.
    N_pad = ((N + B_lcm - 1) // B_lcm) * B_lcm
    if N_pad != N:
        print(f"Padding {N} -> {N_pad} windows to fill batch-encoder={B_enc} "
              f"/ batch-decoder={B_dec} (lcm={B_lcm}) "
              f"(extra {N_pad - N} rows discarded before scoring)")

    # --- 1. Encoder inputs: embeddings + pad to 48 per timestep ---
    sport_w = sd["sport_emb.weight"].numpy()
    dport_w = sd["dport_emb.weight"].numpy()
    proto_w = sd["proto_emb.weight"].numpy()
    x_windows = np.zeros((N_pad, T, INPUT_DIM_PADDED), dtype=bfloat16)
    for w in range(N):
        for t in range(T):
            xin = np.concatenate([
                X_num[w, t],
                sport_w[X_cat[w, t, 0]],
                dport_w[X_cat[w, t, 1]],
                proto_w[X_cat[w, t, 2]],
            ]).astype(bfloat16)
            x_windows[w, t, :INPUT_DIM] = xin  # last 3 stay 0 (pad)
    (_HERE / "all_x_windows.bin").write_bytes(x_windows.reshape(N_pad, -1).tobytes())

    # Encoder params (padded w_ih), matching gen_encoder_data.py.
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

    # Decoder GRU params: [w_ih | w_hh | b_ih | b_hh]. hidden_to_output is
    # applied on the host (below), not fused into the kernel -- the fused
    # variant was measured slower on hardware (see --batch-decoder help).
    w_ih_d = sd["decoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_d = sd["decoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_d = sd["decoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_d = sd["decoder.gru.bias_hh_l0"].numpy().astype(bfloat16)
    dec_params = np.concatenate(
        [w_ih_d.reshape(-1), w_hh_d.reshape(-1), b_ih_d, b_hh_d]
    ).astype(bfloat16)
    (_HERE / "dec_params.bin").write_bytes(dec_params.tobytes())
    n_dec_params = dec_params.size

    # --- Build xclbins + batch host (once) ---
    xf = []
    if args.xrt_inc_dir:
        xf.append(f"XRT_INC_DIR={args.xrt_inc_dir}")
    if args.xrt_lib_dir:
        xf.append(f"XRT_LIB_DIR={args.xrt_lib_dir}")
    if not args.skip_build:
        # ALWAYS delete .prj dirs before rebuilding. IRON/aiecc's own
        # ExternalFunction build cache does not reliably invalidate on
        # changes to included headers like gru_common.h (source_string is
        # just two fixed #include lines, unaffected by what's inside them --
        # see flair-npu-iron-kernel-gotchas memory item 10). Without this, a
        # kernel edit can silently test stale, unchanged compiled code.
        shutil.rmtree(_HERE / "build" / "gru.prj", ignore_errors=True)
        shutil.rmtree(_HERE / "build" / "gru_4core.prj", ignore_errors=True)
        shutil.rmtree(_HERE / "build" / "decoder.prj", ignore_errors=True)
        if args.encoder_4core:
            # 4-core: each core runs B_enc//4 windows; one dispatch = B_enc
            # windows across the column (memtile scatter/broadcast/gather).
            sh(["python3", "gru_encoder_4core.py", "--dev", "npu", "--input-dim",
                str(INPUT_DIM_PADDED), "--hidden-dim", str(HIDDEN_DIM), "--seq-len",
                str(T), "--batch", str(B_enc // 4), "--xclbin-path",
                "build/gru_4core.xclbin", "--insts-path", "build/gru_4core_insts.bin"])
        else:
            sh(["python3", "gru_encoder.py", "--dev", "npu", "--input-dim",
                str(INPUT_DIM_PADDED), "--hidden-dim", str(HIDDEN_DIM), "--seq-len",
                str(T), "--batch", str(B_enc), "--xclbin-path", "build/gru.xclbin",
                "--insts-path", "build/insts.bin"])
        sh(["python3", "gru_decoder.py", "--dev", "npu", "--hidden-dim",
            str(HIDDEN_DIM), "--seq-len", str(T), "--batch", str(B_dec),
            "--xclbin-path", "build/decoder.xclbin", "--insts-path",
            "build/decoder_insts.bin"])
        sh(["make", "-f", "Makefile.batch"] + xf)

    ps = "powershell.exe"

    # --- 2. NPU encoder pass ---
    # Both encoders present B_enc windows/dispatch to batch_infer; the 4-core
    # design splits them across 4 cores internally via the memtile.
    if args.encoder_4core:
        enc_xclbin, enc_insts = "build/gru_4core.xclbin", "build/gru_4core_insts.bin"
        print("\n[encoder] batched NPU pass (4-core data-parallel)")
    else:
        enc_xclbin, enc_insts = "build/gru.xclbin", "build/insts.bin"
        print("\n[encoder] batched NPU pass")
    enc_stdout = sh_capture([ps, "./batch_infer.exe", enc_xclbin, enc_insts,
        "all_x_windows.bin", "enc_params.bin", "all_latents.bin",
        str(N_pad), str(B_enc), str(enc_in1_vol), str(n_enc_params), str(HIDDEN_DIM)])
    enc_us_per_window = parse_us_per_window(enc_stdout)

    # --- 3. latent -> h0 (host) ---
    latents = np.frombuffer((_HERE / "all_latents.bin").read_bytes(),
                            dtype=bfloat16).reshape(N_pad, HIDDEN_DIM).astype(np.float32)
    W_lh = sd["decoder.latent_to_hidden.weight"].numpy().astype(np.float32)
    b_lh = sd["decoder.latent_to_hidden.bias"].numpy().astype(np.float32)
    h0 = np.tanh(latents @ W_lh.T + b_lh).astype(bfloat16)  # (N_pad, 64)
    (_HERE / "all_h0.bin").write_bytes(h0.tobytes())

    # --- 4. NPU decoder pass ---
    print("\n[decoder] batched NPU pass")
    dec_stdout = sh_capture([ps, "./batch_infer.exe", "build/decoder.xclbin",
        "build/decoder_insts.bin", "all_h0.bin", "dec_params.bin",
        "all_hidden.bin", str(N_pad), str(B_dec), str(HIDDEN_DIM), str(n_dec_params),
        str(T * HIDDEN_DIM)])
    dec_us_per_window = parse_us_per_window(dec_stdout)

    # --- 5. hidden -> recon -> NPU MSE scores (host) ---
    # Discard the N_pad-N padding rows before scoring.
    hidden = np.frombuffer((_HERE / "all_hidden.bin").read_bytes(),
                           dtype=bfloat16).reshape(N_pad, T, HIDDEN_DIM).astype(np.float32)[:N]
    W_out = sd["decoder.hidden_to_output.weight"].numpy().astype(np.float32)
    b_out = sd["decoder.hidden_to_output.bias"].numpy().astype(np.float32)
    recon = hidden @ W_out.T + b_out                       # (N, T, 21)
    npu_scores = np.mean((recon - X_num[:, :T]) ** 2, axis=(1, 2))

    # --- 6. PyTorch scores + metrics ---
    with torch.no_grad():
        pt_scores = model.anomaly_score(
            torch.from_numpy(X_num), torch.from_numpy(X_cat)
        ).numpy()

    npu_auc = roc_auc(npu_scores, y)
    pt_auc = roc_auc(pt_scores, y)
    corr = float(np.corrcoef(npu_scores, pt_scores)[0, 1])
    rel = np.abs(npu_scores - pt_scores) / (np.abs(pt_scores) + 1e-9)
    n_anom = int(y.sum())

    print("\n" + "=" * 64)
    print(f"FLAIR on NPU vs PyTorch  ({N} windows, {n_anom} anomalies)")
    print("=" * 64)
    print(f"  score correlation (Pearson r) : {corr:.4f}")
    print(f"  per-window rel err (INFO ONLY): mean {rel.mean()*100:.2f}%  "
          f"median {np.median(rel)*100:.2f}%")
    print("  (NPU bf16/LUT scores are on a different scale than PyTorch fp32;")
    print("   rel err is expected to be nonzero. Judge detection by the")
    print("   per-path calibrated metrics below, not by score identity.)")
    print("-" * 64)
    if n_anom == 0:
        print("  No anomalies in this window set -> ROC-AUC / F1 undefined.")
        print("  (eval & inference splits are all-normal; use --sample on the")
        print("   train split for a labeled subset, e.g. --npz ...train.npz")
        print("   --sample 20000.)")
    else:
        nm = detection_metrics(npu_scores, y)
        pm = detection_metrics(pt_scores, y)
        print(f"  ROC-AUC     NPU {npu_auc:.4f}     PyTorch {pt_auc:.4f}")
        print("  Detection @ normal-p99 threshold (each path self-calibrated):")
        print(f"    {'':10s} {'thr':>10s} {'TP':>5s} {'FP':>4s} {'TN':>6s} "
              f"{'FN':>4s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'FPR':>7s}")
        for tag, m in (("NPU", nm), ("PyTorch", pm)):
            print(f"    {tag:10s} {m['thr']:>10.5f} {m['tp']:>5d} {m['fp']:>4d} "
                  f"{m['tn']:>6d} {m['fn']:>4d} {m['prec']:>7.4f} {m['rec']:>7.4f} "
                  f"{m['f1']:>7.4f} {m['fpr']:>7.4f}")
    print("=" * 64)

    # Save per-window scores for plotting / the poster.
    out_csv = _HERE / "npu_vs_pytorch_scores.csv"
    with open(out_csv, "w") as f:
        f.write("window,label,npu_score,pytorch_score\n")
        for i in range(N):
            f.write(f"{i},{int(y[i])},{npu_scores[i]:.6f},{pt_scores[i]:.6f}\n")
    print(f"per-window scores -> {out_csv}")

    # --- 7. NPU vs CPU speed comparison ---
    if not args.skip_cpu_baseline:
        import torch

        default_threads = torch.get_num_threads()
        print(f"\n[cpu baseline] single-window (batch=1) latency, "
              f"{args.cpu_warmup_iters} warmup + {args.cpu_timed_iters} timed calls")
        cpu_eager_us = cpu_single_window_latency(
            model, X_num, X_cat, threads=default_threads, use_torchscript=False,
            warmup=args.cpu_warmup_iters, iters=args.cpu_timed_iters,
        )
        cpu_ts_us = cpu_single_window_latency(
            model, X_num, X_cat, threads=1, use_torchscript=True,
            warmup=args.cpu_warmup_iters, iters=args.cpu_timed_iters,
        )

        print("\n" + "=" * 64)
        print(f"NPU vs CPU inference speed  (per window, full encoder+decoder, "
              f"batch-encoder={B_enc} batch-decoder={B_dec})")
        print("=" * 64)
        if enc_us_per_window is not None and dec_us_per_window is not None:
            npu_us = enc_us_per_window + dec_us_per_window
            print(f"  NPU   (encoder {enc_us_per_window:.1f} + decoder "
                  f"{dec_us_per_window:.1f}, incl. host sync) : {npu_us:8.1f} us/window")
        else:
            npu_us = None
            print("  NPU   : could not parse per-window timing from batch_infer.exe output")
        print(f"  CPU   eager       (default threads={default_threads})   : {cpu_eager_us:8.1f} us/window")
        print(f"  CPU   TorchScript (1 thread, fair baseline) : {cpu_ts_us:8.1f} us/window")
        if npu_us is not None:
            print("-" * 64)
            print(f"  Speedup vs CPU eager (default threads) : {cpu_eager_us / npu_us:.2f}x")
            print(f"  Speedup vs CPU fair baseline (TS/1thr)  : {cpu_ts_us / npu_us:.2f}x")
        print("=" * 64)


if __name__ == "__main__":
    main()
