#!/usr/bin/env python3
"""
diag_decoder_timing.py  -- DIAGNOSTIC ONLY

Isolates the source of the decoder's large fixed per-dispatch cost (~5.5x the
encoder's, and which the fused variant made worse). Builds two decoder xclbins
that do IDENTICAL compute (the full GRU sequence) and differ ONLY in output
footprint, then times both through the same batch_infer.exe dispatch path:

  * unfused  : writes the whole hidden_seq   (batch * SEQ_LEN * HIDDEN_DIM)
  * final    : writes only the final hidden  (batch * HIDDEN_DIM)

Interpretation of the reported us/dispatch:
  * if `final` collapses toward the encoder's ~600us while `unfused` stays
    ~3300us  -> the decoder floor is the per-timestep output writes / the
    larger output DMA. Fix: restructure the output.
  * if `final` stays ~3300us too -> the floor is the gru_step compilation in
    the decoder design itself, independent of output size. Look there instead.

Inputs are synthetic (small random) -- only timing matters here, not values.

Usage (from npu/, WSL IRON env sourced):
    python3 diag_decoder_timing.py --batch 6 --windows 600
    python3 diag_decoder_timing.py --batch 6 --windows 600 --skip-build
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent

HIDDEN_DIM = 64
SEQ_LEN = 10


def sh(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=_HERE, check=True)


def sh_capture(cmd: list[str]) -> str:
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=_HERE, check=True, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    return r.stdout


def parse_us_per_window(stdout: str) -> float | None:
    m = re.search(r"([\d.]+)\s*us/window", stdout)
    return float(m.group(1)) if m else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--windows", type=int, default=600,
                   help="total windows to stream (rounded up to a multiple of batch)")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()

    H = HIDDEN_DIM
    T = args.seq_len
    B = args.batch
    N = ((args.windows + B - 1) // B) * B
    h3 = 3 * H
    n_params = h3 * H + h3 * H + h3 + h3  # unfused decoder params (24960)

    rng = np.random.default_rng(0)
    # Synthetic inputs (values irrelevant to timing).
    h0 = (rng.standard_normal((N, H)) * 0.1).astype(bfloat16)
    params = (rng.standard_normal(n_params) * 0.1).astype(bfloat16)
    (_HERE / "diag_h0.bin").write_bytes(h0.tobytes())
    (_HERE / "diag_params.bin").write_bytes(params.tobytes())

    ps = "powershell.exe"

    if not args.skip_build:
        sh(["python3", "gru_decoder.py", "--dev", "npu", "--hidden-dim", str(H),
            "--seq-len", str(T), "--batch", str(B), "--xclbin-path",
            "build/decoder.xclbin", "--insts-path", "build/decoder_insts.bin"])
        sh(["python3", "gru_decoder_final.py", "--dev", "npu", "--hidden-dim",
            str(H), "--seq-len", str(T), "--batch", str(B), "--xclbin-path",
            "build/decoder_final.xclbin", "--insts-path",
            "build/decoder_final_insts.bin"])
        sh(["make", "-f", "Makefile.batch"])

    results = {}

    print("\n[unfused] full hidden_seq output (batch*SEQ_LEN*HIDDEN_DIM)")
    out = sh_capture([ps, "./batch_infer.exe", "build/decoder.xclbin",
        "build/decoder_insts.bin", "diag_h0.bin", "diag_params.bin",
        "diag_out_unfused.bin", str(N), str(B), str(H), str(n_params),
        str(T * H)])
    results["unfused"] = parse_us_per_window(out)

    print("\n[final] final-hidden-only output (batch*HIDDEN_DIM)")
    out = sh_capture([ps, "./batch_infer.exe", "build/decoder_final.xclbin",
        "build/decoder_final_insts.bin", "diag_h0.bin", "diag_params.bin",
        "diag_out_final.bin", str(N), str(B), str(H), str(n_params),
        str(H)])
    results["final"] = parse_us_per_window(out)

    print("\n" + "=" * 64)
    print(f"Decoder fixed-cost isolation  (batch={B}, N={N})")
    print("=" * 64)
    for name, us_win in results.items():
        if us_win is None:
            print(f"  {name:8s}: (could not parse timing)")
        else:
            print(f"  {name:8s}: {us_win:8.1f} us/window  ->  "
                  f"{us_win * B:8.1f} us/dispatch")
    print("-" * 64)
    if results.get("unfused") and results.get("final"):
        drop = (results["unfused"] - results["final"]) / results["unfused"] * 100
        print(f"  final vs unfused per-dispatch: {drop:+.1f}% "
              f"(large drop -> output writes are the floor; "
              f"~0 -> gru_step compilation is)")
    print("=" * 64)


if __name__ == "__main__":
    main()
