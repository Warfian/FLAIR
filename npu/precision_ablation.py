#!/usr/bin/env python3
"""
precision_ablation.py

Answers: WHERE does the NPU's accuracy loss come from -- bf16 rounding, or the
exp-LUT sigmoid/tanh?

Simulates run_dataset_inference.py's exact pipeline in numpy with each precision
source toggled independently, and reports the metric that actually matters: the
inflation of the normal-p99 detection threshold vs PyTorch fp32 (the NPU's
inflated threshold is what costs recall -- see ACCURACY_HANDOFF.md).

Findings (hidden=64 model, thesis-style test set; real NPU = 6.06x inflation):
  * bf16 rounding of gates/hidden/MAC operands, with EXACT sigmoid/tanh -> 1.00x
    inflation, F1 identical to PyTorch. bf16 is NOT the problem.
  * Injecting relative error into the nonlinearity reproduces the NPU:
    2% -> 1.05x,  4% -> 1.27x,  8% -> 4.67x,  10% -> 18.3x.
    The real NPU (6.06x) implies the LUT sigmoid/tanh is only ~8-9% accurate.
  => The ENTIRE loss is the nonlinearity. Getting it under ~4% relative error
     restores essentially full PyTorch F1 (0.894 -> ~0.937).

Usage:  python npu/precision_ablation.py [--ckpt PATH] [--npz PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

H, T = 64, 10
b16 = lambda a: a.astype(bfloat16).astype(np.float32)
ident = lambda a: a.astype(np.float32)
sig = lambda x: 1.0 / (1.0 + np.exp(-x))


def build(ckpt, npz, limit):
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck["model_state_dict"]
    model = FLAIRAutoencoder(FLAIRConfig(**ck["model_cfg"])).eval()
    model.load_state_dict(sd, strict=False)
    W = {k: v.numpy().astype(np.float32) for k, v in sd.items()}
    d = np.load(npz, allow_pickle=True)
    Xn = d["X_num"].astype(np.float32)
    Xc = d["X_cat"].astype(np.int64)
    y = d["y_seq"].astype(np.int64)
    if limit and limit < len(y):
        sel = np.sort(np.random.default_rng(0).choice(len(y), limit, replace=False))
        Xn, Xc, y = Xn[sel], Xc[sel], y[sel]
    return model, W, Xn, Xc, y


def simulate(W, Xn, Xc, *, qg, qh, qm, eps=0.0, mode="rand", seed=1):
    """qg/qh/qm: rounding for gates / hidden / MAC operands. eps: relative error
    injected into sigmoid+tanh (mode 'rand' = zero-mean, 'bias' = systematic)."""
    rng = np.random.default_rng(seed)
    if eps <= 0:
        pert = lambda v: v
    elif mode == "bias":
        pert = lambda v: v * (1.0 + eps)
    else:
        pert = lambda v: v * (1.0 + eps * rng.uniform(-1, 1, v.shape).astype(np.float32))
    S = lambda x: pert(sig(x))
    TH = lambda x: pert(np.tanh(x))

    g = W.__getitem__
    sp, dp, pr = g("sport_emb.weight"), g("dport_emb.weight"), g("proto_emb.weight")
    Wie, Whe = qm(g("encoder.gru.weight_ih_l0")), qm(g("encoder.gru.weight_hh_l0"))
    bie, bhe = g("encoder.gru.bias_ih_l0"), g("encoder.gru.bias_hh_l0")
    Wlh, blh = g("decoder.latent_to_hidden.weight"), g("decoder.latent_to_hidden.bias")
    Wid, Whd = qm(g("decoder.gru.weight_ih_l0")), qm(g("decoder.gru.weight_hh_l0"))
    bid, bhd = g("decoder.gru.bias_ih_l0"), g("decoder.gru.bias_hh_l0")
    Wo, bo = g("decoder.hidden_to_output.weight"), g("decoder.hidden_to_output.bias")

    N = len(Xn)
    out = np.empty(N, np.float64)
    for s in range(0, N, 8192):
        e = min(s + 8192, N)
        B = e - s
        xin = qm(np.concatenate([Xn[s:e], sp[Xc[s:e, :, 0]], dp[Xc[s:e, :, 1]],
                                 pr[Xc[s:e, :, 2]]], axis=-1))
        h = np.zeros((B, H), np.float32)
        for t in range(T):                                   # encoder
            gi = qg(xin[:, t] @ Wie.T + bie)
            gh = qg(qm(h) @ Whe.T + bhe)
            r = S(gi[:, :H] + gh[:, :H])
            z = S(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = TH(gi[:, 2 * H:] + r * gh[:, 2 * H:])
            h = qh((1 - z) * n + z * h)
        h0 = qm(TH(h @ Wlh.T + blh))                         # host fp32 linear
        gid = qg(h0 @ Wid.T + bid)                           # decoder gi (invariant)
        hd = qh(h0.copy())    # decoder init hidden = h0_vec (decoder.py / gru_decoder.cc)
        hid = np.empty((B, T, H), np.float32)
        for t in range(T):                                   # decoder
            gh = qg(qm(hd) @ Whd.T + bhd)
            r = S(gid[:, :H] + gh[:, :H])
            z = S(gid[:, H:2 * H] + gh[:, H:2 * H])
            n = TH(gid[:, 2 * H:] + r * gh[:, 2 * H:])
            hd = qh((1 - z) * n + z * hd)
            hid[:, t] = hd
        out[s:e] = np.mean((hid @ Wo.T + bo - Xn[s:e]) ** 2, axis=(1, 2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(_REPO / "experiments/results/flair_h64_full.pt"))
    p.add_argument("--npz", default=str(_REPO / "data/processed/retrain_test.npz"))
    p.add_argument("--limit", type=int, default=40000, help="0 = all windows")
    args = p.parse_args()

    model, W, Xn, Xc, y = build(args.ckpt, args.npz, args.limit)
    norm, att = y == 0, y == 1
    with torch.no_grad():
        pt = np.concatenate([
            model.anomaly_score(torch.from_numpy(Xn[s:s + 8192]),
                                torch.from_numpy(Xc[s:s + 8192])).numpy()
            for s in range(0, len(y), 8192)])
    ref = float(np.percentile(pt[norm], 99))

    def rep(tag, sc):
        thr = float(np.percentile(sc[norm], 99))
        pred = sc > thr
        tp = int((pred & att).sum()); fp = int((pred & norm).sum()); fn = int((~pred & att).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {tag:32s} normal_p99={thr:.4f}  inflation={thr/ref:6.2f}x  R={rec:.4f}  F1={f1:.4f}")

    print(f"Precision ablation  ({len(y)} windows, {int(y.sum())} attacks)")
    print(f"  {'PyTorch fp32 (reference)':32s} normal_p99={ref:.4f}  inflation=  1.00x\n")
    print("-- bf16 rounding only (EXACT sigmoid/tanh) --")
    rep("all fp32 (sim sanity check)", simulate(W, Xn, Xc, qg=ident, qh=ident, qm=ident))
    rep("bf16 gates+hidden+MAC (= NPU)", simulate(W, Xn, Xc, qg=b16, qh=b16, qm=b16))
    print("\n-- bf16 + relative error injected into sigmoid/tanh --")
    for eps in (0.02, 0.04, 0.08, 0.10):
        rep(f"nonlinearity rel err {eps:.0%}",
            simulate(W, Xn, Xc, qg=b16, qh=b16, qm=b16, eps=eps))
    print("\n  Real NPU measured: inflation=6.06x, R=0.9112, F1=0.8940")
    print("  => bf16 is exact-equivalent; the LUT sigmoid/tanh (~8-9% error) is")
    print("     the entire loss. Target <4% rel err to recover PyTorch-level F1.")


if __name__ == "__main__":
    main()
