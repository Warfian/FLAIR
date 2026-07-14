# FLAIR-on-NPU Accuracy Handoff

**Mission:** make the NPU implementation of FLAIR *numerically correct* at full
dataset scale. Two distinct problems (details below): (A) a **hard NaN** that
appears when scoring the full WUSTL-IIoT dataset, and (B) a **soft accuracy
drift** (~13–22% per-window score error vs PyTorch) from the bf16 + LUT-based
nonlinearity math.

This document is a starting-context dump. **Speed/latency is explicitly OUT OF
SCOPE here** — it is being handled in a separate conversation. Do not optimize
for performance; optimize for correctness.

---

## 1. What FLAIR is (just enough to work on accuracy)

FLAIR is a GRU autoencoder for unsupervised network-intrusion detection. It is
trained on normal traffic only and flags anomalies by reconstruction error.

Pipeline (all bf16 on the NPU; PyTorch fp32 is the reference):

```
x_num (21 numeric, z-score normalized) + x_cat (Sport/Dport/Proto embeddings, 3×8)
  -> concat -> 45-dim input (padded to 48 for vectorization)
  -> ENCODER GRU (hidden=64), run SEQ_LEN=10 timesteps -> latent = last hidden (64)
  -> h0 = tanh(latent_to_hidden(latent))            [host-side fp32 linear]
  -> DECODER GRU (hidden=64): x_t = h0 EVERY timestep (repeated-input design,
     no autoregression), run 10 steps -> hidden_seq (10×64)
  -> recon = hidden_to_output(hidden_seq)           [linear, 10×21]
  -> anomaly score = MSE(recon, x_num)
```

**Checkpoint:** `experiments/results/flair_minimal.pt` (hidden_dim=64, trained on
the small 1000-row sample; its vocab + mu/sigma are sample-derived).
Do NOT use `flair_80_10_10.pt` — it is hidden_dim=128 and does not fit the AIE
tile's 64KB L1; that's a separate, out-of-scope problem.

---

## 2. Current accuracy status

- **Small sample dataset (`data/processed/preprocessed.npz`, 990 windows): WORKS.**
  No NaN. Single-window relative score error was ~4.3%; median over 990 windows
  ~41% (drift, but finite and rank-preserving).
- **Full dataset inference split (119,437 windows): NaN.** Scoring produces
  `RuntimeWarning: invalid value encountered in matmul/subtract`, and the
  resulting scores/metrics are all `nan`. This is the primary bug to fix.
- On a clean 3,000-window prefix of the inference split (all-normal, no
  anomalies), scores were finite with **Pearson r = 1.0000** vs PyTorch but
  **mean rel err ~22% / median ~14%** — i.e. the soft drift (problem B) is real
  even where there's no NaN, though ranking is preserved.

Note: `ROC-AUC = nan` on the inference split is NOT a bug — that split has 0
labeled anomalies, so AUC is undefined by construction. Don't chase that.

---

## 3. The two problems

### Problem A — the hard NaN (priority)

**When:** full dataset only, not the sample. The full dataset has extreme
real-world flows (huge byte/packet counts) that, z-scored against the *sample's*
mu/sigma, land far outside the range the sample ever produced.

**Leading hypothesis (UNCONFIRMED — first job is to confirm/refute it):**
bf16 overflow in `matvec_bias` on an extreme normalized input produces `inf`,
which then becomes `NaN` via an `inf − inf` or `0 × inf` in the gate combine
(e.g. `pre_r = gi_r + gh_r` with opposite-sign infs, or `n_pre = gi_n + r*gh_n`
with `r=0` and `gh_n=inf`). `getExpBf16` itself has a truncate out-of-range
policy and is believed to never emit NaN directly, so a lone `sigmoid16` call
shouldn't NaN — the NaN most likely originates *upstream* in the matvec, then
propagates through the gate math.

**First diagnostic to run (cheap, decisive):**
1. In `run_dataset_inference.py`, find the first window whose NPU `hidden`/`recon`
   contains NaN. Dump that window's `x_num`/`x_cat` (and its intermediate
   `latents`, `h0`).
2. Ask: does the **PyTorch fp32 path also produce NaN/inf on that same window**?
   - If PyTorch is *fine* but the NPU NaNs → it's a bf16/LUT/kernel problem
     (clamp inputs or intermediates; investigate `matvec_bias` overflow and the
     gate-combine inf paths in `gru_common.h`).
   - If PyTorch *also* NaNs/infs → it's an upstream data/normalization problem
     (extreme z-scores); fix in preprocessing (`scripts/preprocess_data.py`) or
     the embedding/normalization step, e.g. clamp normalized features to a sane
     range like [−10, 10] before they ever reach the kernel.

**Likely fix directions** (pick based on the diagnostic):
- Clamp z-scored numeric inputs to a bounded range in preprocessing / in
  `run_dataset_inference.py`'s embedding step.
- Add defensive clamping of `gi`/`gh` (or the matvec output) inside
  `matvec_bias` / `gru_step` in `gru_common.h`.
- Both — input clamping for correctness parity with a (clamped) PyTorch
  reference, plus in-kernel clamping as a NaN backstop.

### Problem B — the soft drift (secondary)

Even without NaN, per-window scores drift ~13–22% from PyTorch because the
bf16 + exp-LUT sigmoid/tanh (`sigmoid(x)=1/(1+exp(-x))` via `getExpBf16` +
per-lane `getInvBf16`; `tanh(x)=2·sigmoid(2x)−1`) has limited accuracy that
compounds through the 10-step GRU recurrence. Correlation stays ~1.0, so
detection *ranking* survives, but absolute scores diverge.

If you improve nonlinearity accuracy, validate it doesn't reintroduce NaN
(see gotchas). A previously-abandoned idea was a rational Padé[7/6] tanh
(numpy-validated ~6e-4, unbiased) — it was dropped for *compiler/stack* reasons,
not accuracy; revisiting it in a purely vectorized form (staying in vector
registers, never large scalar stack arrays) is a plausible path.

---

## 4. Key files

Kernel math (this is where NaN and drift live):
- **`npu/kernels/gru_common.h`** — THE core. Contains `sigmoid16`, `tanh16`,
  `matvec_bias`, `gru_step` (encoder), `gru_step_with_gi` (decoder). Shared by
  both encoder and decoder, so a fix here affects both.
- `npu/kernels/gru_encoder.cc`, `npu/kernels/gru_decoder.cc` — the kernels that
  call the above.

Drivers / pipeline:
- `npu/run_dataset_inference.py` — dataset-scale run where the NaN manifests
  (step 5, `recon = hidden @ W_out.T + b_out`). Best place to add first-NaN
  instrumentation. Loads `flair_minimal.pt`; `--npz <path> --limit 0` scores a
  whole split.
- `npu/gru_encoder.py`, `npu/gru_decoder.py` — IRON drivers (compile the kernels
  to xclbins).

Validation tools (USE THESE — they isolate LUT/bf16 error cleanly):
- **`npu/gen_encoder_data.py`, `npu/gen_decoder_data.py`** — write a *float
  golden* computed from the SAME bf16-quantized inputs the NPU sees. So a
  golden-vs-NPU diff isolates exactly the kernel's bf16+LUT error (not
  input-quantization error). This is your ground truth for single-window checks.
- `npu/verify_decoder_gru_cell_math.py` — cross-checks the decoder cell math
  against PyTorch nn.GRUCell (validated to ~4e-8 in float).
- `npu/compare_anomaly_score.py` — single-window NPU-vs-PyTorch score compare.

Data / preprocessing:
- `scripts/preprocess_data.py` — normalization (mu/sigma) lives here. `--split`
  mode makes the 80/10/10 train/eval/inference split from
  `src/data/wustl_iiot_2021.csv`. It reuses `flair_minimal.pt`'s vocab + mu/sigma
  via `paths.vocab_reference_npz` in `config.yaml` (so extreme/unseen values map
  to UNK and are scaled by the sample's stats — which is exactly why extremes
  blow up). This is a prime suspect / fix site for Problem A.
- Datasets: sample = `data/processed/preprocessed.npz` (no NaN); full inference
  split = `data/processed/preprocessed_inference.npz` (NaNs; regenerate with
  `python scripts/preprocess_data.py --split`).

---

## 5. Environment & workflow (read before building anything)

- **Hardware:** AMD Ryzen 9 7940HS (Phoenix / XDNA1 / AIE2), NPU device name
  `"npu"`, hidden_dim=64. L1 per tile = 64KB; core stack ≈ 1KB (separate, tiny).
- **Compile-in-WSL, run-on-native-Windows hybrid.** WSL cannot see the NPU
  (no `/dev/accel/accel0`). You compile the xclbin in WSL, then a Windows-side
  `batch_infer.exe` (invoked via `powershell.exe` from WSL) runs it on the NPU.
- **XRT setup gotcha:** in every new WSL shell, `source
  ~/xrt_work/XRT/build/Debug/opt/xilinx/xrt/setup.sh` or `xclbinutil`/`pyxrt`
  aren't found and builds fail silently ("exit 0 but no xclbin").
- **⚠️ STALE BUILD CACHE — THE most important workflow gotcha ⚠️**
  IRON/aiecc's ExternalFunction build cache does NOT reliably invalidate when
  you edit `gru_common.h` (the kernel source is included via a fixed
  `source_string`, so the cache key doesn't see the header content change).
  **After ANY kernel edit, `rm -rf build/<name>.prj` before rebuilding**, or you
  will silently test old, unchanged code. This already caused a full day of
  false "the fix didn't work" / "it's mysteriously slow" results in the speed
  investigation. `run_dataset_inference.py` and `diag_decoder_timing.py` now
  auto-`rm -rf`, but any manual `python3 gru_*.py` build does not — clean first.
  For accuracy work this is critical: a stale build means you're validating the
  wrong binary.

---

## 6. Critical do's and don'ts

- **DON'T reintroduce `getTanhBf16`** for general inputs — it had a deterministic
  NaN on an interior (in-range) value. That's why the code uses exp-LUT-based
  sigmoid/tanh. If you change the nonlinearity, prove it's NaN-free across the
  full input range, not just the sample.
- **Distinguish two kinds of NaN:**
  - *Numerical NaN* (Problem A) — data-dependent, reproducible for a given
    input, same index every run.
  - *Corruption NaN* (stack overflow) — the ~1KB core stack overflows if you add
    large local/scratch arrays; the tell is a NaN whose **index moves** when you
    change unrelated code. `gru_step` already carries `gi[192]+gh[192]+h_prev[64]`
    ≈ 896B of stack; adding more scratch can tip it over. If you see a
    moving-index NaN, it's this, not the math — move big scratch to `static`
    (L1/BSS) or shrink it.
- **Validate against the float golden, not just PyTorch.** The golden (from
  `gen_*_data.py`) uses the same bf16 inputs, so it isolates kernel error. A
  PyTorch fp32 comparison mixes in input-quantization error too.
- **Match the reference to any input change.** If you clamp inputs on the NPU
  side, clamp them in the PyTorch reference too, or the comparison is apples to
  oranges.
- **Don't touch speed.** No batching/latency changes here.

---

## 7. Suggested order of attack

1. Reproduce the NaN: `python run_dataset_inference.py --npz
   ../data/processed/preprocessed_inference.npz --limit 0` (from `npu/`). Confirm
   the `invalid value` warning + nan scores.
2. Instrument to find the first NaN window + dump its inputs/intermediates.
3. Run that window through the PyTorch fp32 path → is it also non-finite?
   (This is the fork in the road: data problem vs kernel problem.)
4. Based on (3), apply input clamping (preprocessing) and/or kernel-level
   clamping (`gru_common.h` / `matvec_bias`), keeping the PyTorch reference
   consistent.
5. Re-run full dataset; confirm NaN gone and scores finite. Check ROC-AUC/F1
   against PyTorch on the *eval* split (which HAS anomalies, unlike the
   all-normal inference split) — that's the real accuracy metric.
6. (Secondary) Attack Problem B drift if the NaN fix alone doesn't get metrics
   close enough to the PyTorch baseline.

---

## 8. Git

Latest work is on branch **`flair-speedup`** (forked from `flair-merge`).
Recommend branching a fresh **`flair-accuracy`** from `flair-speedup` for this
work so accuracy and speed histories stay separate. Commit + push at each
working step (the repo is on GitHub: `Warfian/FLAIR`).

Repo also has a `.gitattributes` forcing LF for `npu/**` — if a shell script
ever fails with `$'\r': command not found` on a fresh checkout, force a
re-checkout of that file so the LF rule applies.
