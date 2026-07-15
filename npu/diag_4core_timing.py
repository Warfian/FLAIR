#!/usr/bin/env python3
"""
diag_4core_timing.py

Measures the throughput of the 4-core data-parallel encoder (gru_encoder_4core)
vs the single-core encoder (gru_encoder), both via batch_infer.exe. Windows are
independent, so 4 cores in one column should give ~4x throughput (~4x lower
us/window).

Comparison is at the SAME windows-per-dispatch so it's apples-to-apples:
  * single-core: --batch = W   -> W windows/dispatch on 1 core
  * 4-core:      --batch-per-core = W//4 -> W windows/dispatch across 4 cores

batch_infer.exe treats the 4-core design as a single kernel with
(4*batch_per_core) windows per dispatch: the memtile split/join distribute the
windows across cores internally, so the host just uses a bigger per-dispatch
"batch" and the same per-window in1_vol/out_vol.

Usage (from npu/, WSL IRON env sourced):
    python3 diag_4core_timing.py --windows-per-dispatch 8 --windows 800
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

INPUT_DIM = 48
HIDDEN_DIM = 64
SEQ_LEN = 10
N_CORES = 4


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
    p.add_argument("--windows-per-dispatch", type=int, default=8,
                   help="windows per dispatch (must be a multiple of 4 for the "
                        "4-core design). single-core uses batch=this; 4-core "
                        "uses batch-per-core=this/4.")
    p.add_argument("--windows", type=int, default=800)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--only-4core", action="store_true",
                   help="skip the single-core build/run (e.g. while debugging "
                        "the 4-core compile)")
    args = p.parse_args()

    ID, H, T = INPUT_DIM, HIDDEN_DIM, args.seq_len
    WPD = args.windows_per_dispatch
    if WPD % N_CORES != 0:
        sys.exit(f"--windows-per-dispatch {WPD} must be a multiple of {N_CORES}")
    per_core = WPD // N_CORES
    N = ((args.windows + WPD - 1) // WPD) * WPD
    h3 = 3 * H
    n_params = h3 * ID + h3 * H + h3 + h3
    in1_vol = T * ID   # per window
    out_vol = H        # per window

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((N, in1_vol)) * 0.1).astype(bfloat16)
    params = (rng.standard_normal(n_params) * 0.1).astype(bfloat16)
    (_HERE / "diag_enc_x.bin").write_bytes(x.tobytes())
    (_HERE / "diag_enc_params.bin").write_bytes(params.tobytes())

    ps = "powershell.exe"
    results = {}

    if not args.skip_build:
        for prj in ("gru", "gru_4core"):
            shutil.rmtree(_HERE / "build" / f"{prj}.prj", ignore_errors=True)

    # --- single-core baseline (batch = WPD windows on one core) ---
    if not args.only_4core:
        if not args.skip_build:
            sh(["python3", "gru_encoder.py", "--dev", "npu", "--input-dim", str(ID),
                "--hidden-dim", str(H), "--seq-len", str(T), "--batch", str(WPD),
                "--xclbin-path", "build/gru.xclbin", "--insts-path", "build/insts.bin"])
        print(f"\n[single-core] batch={WPD} windows/dispatch on 1 core")
        out = sh_capture([ps, "./batch_infer.exe", "build/gru.xclbin", "build/insts.bin",
            "diag_enc_x.bin", "diag_enc_params.bin", "diag_enc_out_1core.bin",
            str(N), str(WPD), str(in1_vol), str(n_params), str(out_vol)])
        results["1-core"] = parse_us_per_window(out)

    # --- 4-core (batch-per-core = WPD/4; total WPD windows/dispatch across 4) ---
    if not args.skip_build:
        sh(["python3", "gru_encoder_4core.py", "--dev", "npu", "--input-dim", str(ID),
            "--hidden-dim", str(H), "--seq-len", str(T), "--batch", str(per_core),
            "--xclbin-path", "build/gru_4core.xclbin",
            "--insts-path", "build/gru_4core_insts.bin"])
    print(f"\n[4-core] batch-per-core={per_core} (={WPD} windows/dispatch across 4 cores)")
    # batch_infer sees WPD windows/dispatch; memtile split/join distribute them.
    out = sh_capture([ps, "./batch_infer.exe", "build/gru_4core.xclbin",
        "build/gru_4core_insts.bin", "diag_enc_x.bin", "diag_enc_params.bin",
        "diag_enc_out_4core.bin", str(N), str(WPD), str(in1_vol), str(n_params),
        str(out_vol)])
    results["4-core"] = parse_us_per_window(out)

    print("\n" + "=" * 64)
    print(f"Encoder throughput: 1-core vs 4-core  (windows/dispatch={WPD}, N={N})")
    print("=" * 64)
    for name, us_win in results.items():
        if us_win is None:
            print(f"  {name:8s}: (could not parse timing)")
        else:
            print(f"  {name:8s}: {us_win:8.1f} us/window")
    print("-" * 64)
    if results.get("1-core") and results.get("4-core"):
        print(f"  4-core speedup: {results['1-core'] / results['4-core']:.2f}x "
              f"(ideal ~4x)")
    print("=" * 64)


if __name__ == "__main__":
    main()
