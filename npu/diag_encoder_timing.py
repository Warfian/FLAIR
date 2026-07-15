#!/usr/bin/env python3
"""
diag_encoder_timing.py  -- DIAGNOSTIC ONLY

Localizes the encoder's ~145us unexplained per-window overhead (the encoder
measures ~435us/window but a floor+matvec+gate model predicts ~290us). Builds
TWO encoder xclbins with identical (x_window, params, latent) buffer
signature/sizes and identical ObjectFifo/DMA wiring, differing only in the
kernel body:

  * real : full T-timestep GRU encode (gru_encoder_bf16)
  * noop : NO gru_step calls at all -- same buffers/wiring, ~no compute
           (gru_encoder_noop_bf16)

Interpretation of the reported us/dispatch:
  * if `noop` shows ~the decoder-noop floor (~32us/window, ~190us/dispatch at
    batch=8) -> the ~145us overhead is in the gru_step/compute path (codegen);
    next step is disassembly / comparing to the decoder's compute path.
  * if `noop` stays high -> the overhead is dispatch/DMA of the encoder's large
    x_window input (batch*480 bf16, 15x the decoder-noop's batch*64 input);
    next step is restructuring how x_window is moved.

Inputs are synthetic (small random) -- only timing matters here, not values.

Usage (from npu/, WSL IRON env sourced):
    python3 diag_encoder_timing.py --batch 8 --windows 800
    python3 diag_encoder_timing.py --batch 8 --windows 800 --skip-build
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent

INPUT_DIM = 48   # padded encoder input length
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
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--windows", type=int, default=800,
                   help="total windows to stream (rounded up to a multiple of batch)")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()

    ID = INPUT_DIM
    H = HIDDEN_DIM
    T = args.seq_len
    B = args.batch
    N = ((args.windows + B - 1) // B) * B
    h3 = 3 * H
    n_params = h3 * ID + h3 * H + h3 + h3  # encoder params (21888 at ID=48)
    in1_vol = T * ID                       # x_window per window (480)
    out_vol = H                            # latent per window (64)

    rng = np.random.default_rng(0)
    # Synthetic inputs (values irrelevant to timing).
    x = (rng.standard_normal((N, in1_vol)) * 0.1).astype(bfloat16)
    params = (rng.standard_normal(n_params) * 0.1).astype(bfloat16)
    (_HERE / "diag_enc_x.bin").write_bytes(x.tobytes())
    (_HERE / "diag_enc_params.bin").write_bytes(params.tobytes())

    ps = "powershell.exe"

    if not args.skip_build:
        # Always clean first -- the IRON build cache does not invalidate on
        # gru_common.h / gru_encoder.cc edits (see kernel-gotchas memory #10).
        for prj in ("gru", "gru_noop"):
            shutil.rmtree(_HERE / "build" / f"{prj}.prj", ignore_errors=True)
        sh(["python3", "gru_encoder.py", "--dev", "npu", "--input-dim", str(ID),
            "--hidden-dim", str(H), "--seq-len", str(T), "--batch", str(B),
            "--xclbin-path", "build/gru.xclbin", "--insts-path", "build/insts.bin"])
        sh(["python3", "gru_encoder_noop.py", "--dev", "npu", "--input-dim", str(ID),
            "--hidden-dim", str(H), "--seq-len", str(T), "--batch", str(B),
            "--xclbin-path", "build/gru_noop.xclbin",
            "--insts-path", "build/gru_noop_insts.bin"])
        sh(["make", "-f", "Makefile.batch"])

    results = {}

    print("\n[real] full GRU encode (gru_encoder_bf16)")
    out = sh_capture([ps, "./batch_infer.exe", "build/gru.xclbin", "build/insts.bin",
        "diag_enc_x.bin", "diag_enc_params.bin", "diag_enc_out_real.bin",
        str(N), str(B), str(in1_vol), str(n_params), str(out_vol)])
    results["real"] = parse_us_per_window(out)

    print("\n[noop] no gru_step calls, same buffers/wiring as real")
    out = sh_capture([ps, "./batch_infer.exe", "build/gru_noop.xclbin",
        "build/gru_noop_insts.bin", "diag_enc_x.bin", "diag_enc_params.bin",
        "diag_enc_out_noop.bin", str(N), str(B), str(in1_vol), str(n_params),
        str(out_vol)])
    results["noop"] = parse_us_per_window(out)

    print("\n" + "=" * 64)
    print(f"Encoder overhead isolation  (batch={B}, N={N})")
    print("=" * 64)
    for name, us_win in results.items():
        if us_win is None:
            print(f"  {name:6s}: (could not parse timing)")
        else:
            print(f"  {name:6s}: {us_win:8.1f} us/window  ->  {us_win * B:8.1f} us/dispatch")
    print("-" * 64)
    if results.get("real") and results.get("noop"):
        r, n = results["real"], results["noop"]
        print(f"  compute (real - noop): {r - n:.1f} us/window")
        print(f"  noop floor vs decoder-noop (~32us/window): "
              f"{'DISPATCH/DMA bound (encoder input is 15x decoder)' if n > 80 else 'low, like decoder -> overhead is in COMPUTE path'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
