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

---

## 9. PROGRESS LOG (branch `flair-accuracy`, 2026-07-13)

### Problem A NaN — root cause CONFIRMED, fix APPLIED (input clamp)

The fork-in-the-road (§3/§7 step 3) is resolved: it is a **data-extreme
problem that only the bf16/LUT path chokes on**, not a PyTorch/fp32 problem.

Evidence (full inference split, 119,437 windows):
- **PyTorch fp32 is 100% finite** on the whole split (scores up to 2.7e8). So
  fp32's range absorbs the extremes; the NaN is specific to the NPU bf16 path.
- Inputs are pathologically extreme: reusing the 1000-row **sample's mu/sigma**
  makes full-dataset `TotBytes` z-score to **max|z| ≈ 2.36e5** (train reaches
  **4.8e6**), vs the sample's own **max|z| = 33.4** (the range the model was
  trained within). p99.9 ≈ 15 everywhere, so it's a razor-thin artifact tail.
- These extremes are the shared precondition for the hypothesized
  matvec/LUT overflow → NaN. The sample (max|z|=33) never NaNs; the full data
  does — the only difference is input magnitude.

**Fix (committed):** clamp z-scored numeric features to
`[-clip_zscore, +clip_zscore]` (`config.yaml` → `preprocess.clip_zscore`,
default **10.0**) right after normalization, in both `preprocess_data.py`
paths (`main` and `main_split`). Because the clamp lives in preprocessing, the
NPU embedding path AND the PyTorch reference both consume the same clamped
`X_num` (satisfies §6 "match the reference"), and **no kernel rebuild is
needed** — the existing prebuilt xclbins are still valid.

clip=10 was chosen from data (not guessed): on a 200k-window labeled TRAIN
subset it maximized PyTorch detection accuracy —
**ROC-AUC 0.985 → 0.994, F1@p99 0.831 → 0.900** vs unclamped — and caps
max score at ~27 (was 1.2e11). clip=5 over-clips (AUC drops to 0.986); 20/40
are finite but less accurate. Regenerated splits now report `max|z| = 10` and
PyTorch scores are finite (range 0.002–16.5 on inference).

### CONFIRMED ON REAL NPU HARDWARE (AMD 7940HS)
Both problems are now validated on-device with the clamped path:

- **Problem A (NaN): FIXED.** Full 119,437-window inference split ran to
  completion, **all scores finite**, no "invalid value" warning (was 100% nan
  before). `Pearson r = 0.9642` vs PyTorch. ROC-AUC/F1 = nan/0 there only
  because that split has 0 anomalies (both paths identical).
- **Problem B (drift): NON-ISSUE for detection**, exactly as predicted by the
  per-path-calibration approach. On a balanced 20,000-window TRAIN subset
  (`--sample 20000`, 2703 anomalies), with each path self-calibrated to its own
  normal-p99 threshold, NPU and PyTorch flag the **identical** windows:

  ```
            thr      TP   FP    TN   FN   Prec    Rec     F1     FPR
  NPU     3.45778   2390  173 17124  313 0.9325 0.8842 0.9077 0.0100
  PyTorch 3.58067   2390  173 17124  313 0.9325 0.8842 0.9077 0.0100
  ```
  `r = 0.9991`, ROC-AUC 0.9935 (NPU) vs 0.9943 (PyTorch). Only the threshold
  differs (bf16 scores on a slightly different scale); calibration absorbs it.
  CAVEAT: this subset is from TRAIN (seen in training), so F1=0.9077 is a
  fidelity/parity number, not a held-out generalization estimate — the current
  chronological split has no held-out anomalies to give the latter.

The commands that produced the above (reproduce with prebuilt binaries in
`Code/xcl/`: `batch_infer.exe`, `gru.xclbin`+`insts.bin`,
`decoder.xclbin`+`decoder_insts.bin`):

```
# 1. regenerate the CLAMPED split with the committed config (clip_zscore: 10.0)
python scripts/preprocess_data.py --split
# 2. stage prebuilt binaries into npu/ and npu/build/ (batch_infer.exe + 4 build files)
# 3. from npu/, run the full split reusing the prebuilt xclbins (no WSL build):
python3 run_dataset_inference.py --npz ../data/processed/preprocessed_inference.npz \
        --limit 0 --skip-build --skip-cpu-baseline        # NaN-gone check (all-normal)
# Labeled detection metrics (balanced subset from TRAIN, which has the anomalies;
# a --limit prefix would be all-normal since train anomalies start ~window 223k):
python3 run_dataset_inference.py --npz ../data/processed/preprocessed_train.npz \
        --sample 20000 --skip-build --skip-cpu-baseline
# To reproduce the ORIGINAL NaN first, set clip_zscore: null, re-run step 1, run step 3.
```

Optional in-kernel backstop (defense-in-depth, NOT shipped — needs a rebuild,
which reverts the "no rebuild" advantage): clamp the argument inside
`sigmoid16` in `gru_common.h` to a saturation-safe range, e.g.
`x = aie::min(aie::max(x, -20), 20)` before `getExpBf16(-x)`. Prefer the
vector `aie::min/max` (no stack cost) over a scalar loop (adds ~32B to the
~1KB core stack → corruption-NaN risk per §6). With inputs clamped to |z|≤10
this path is never exercised, so it is purely belt-and-suspenders.

### CORRECTION to §2/§7 — eval split has NO anomalies either
The handoff assumed "the eval split HAS anomalies." It does **not**: this
dataset's attacks are chronologically concentrated, so the 80/10/10
time-ordered split puts **all 128,581 anomaly windows in TRAIN**; eval and
inference are both 0-anomaly. So ROC-AUC/F1 must be measured on a TRAIN subset
(as done above), or the split methodology changed, not on eval.

### Problem B (soft drift) — RESOLVED as a detection non-issue
The ~13–22% per-window rel err is real but does NOT harm detection: with each
path calibrated to its own normal-p99 threshold, NPU and PyTorch produce the
identical confusion matrix on the 20k labeled subset (see the hardware result
above). So do not chase the rel-err number — it is expected from bf16/LUT and
is absorbed by per-path threshold calibration. `run_dataset_inference.py` now
prints these calibrated metrics and labels rel-err "INFO ONLY".

### Proper (non-toy) accuracy — thesis-style split + retrained hidden=64
`flair_minimal.pt` is a TOY (hidden=64 trained on the 1000-row sample), so its
metrics only measure NPU↔PyTorch fidelity, not real detection quality. Per
Dumond's FLAIR thesis Sec 3.6, proper testing needs normal-only train/val and a
test set with attacks *sampled in* (a pure chronological slice is all-normal —
that is why our eval/inference splits had 0 anomalies).

Implemented:
- `preprocess_data.py --split-eval` → `retrain_{train,val,test}.npz`. Fresh
  mu/sigma on train-normal rows; test = newest normal + attacks sampled to
  7.28%. Reproduces the thesis test set (106,587 normal + 8,369 attack =
  114,956; thesis: 114,957 / 8,369).
- `scripts/retrain_eval_model.py` → `experiments/results/flair_h64_full.pt`
  (hidden=64, normal-only, 200k subsampled train windows, best val 0.0246).
- `scripts/eval_thesis_style.py` (operational p99-of-val-normal threshold +
  best-F1 + ROC/PR-AUC).

Result (held-out test, operational τ = 0.1115):
```
Precision 0.8973  Recall 0.9909  F1 0.9418  FPR 0.0089   ROC-AUC 0.9988
best-F1 upper bound 0.9683
```
This NPU-sized model **matches/beats the thesis hidden=128** (F1 0.9320,
ROC-AUC 0.9994) and is a genuine generalization number (attacks fully excluded
from train/val). Note vocabs are large (sport 51,058) from the full data — fine
for the NPU (host-side embedding lookup; the tile sees only the 45-dim input).

To validate THIS model on the NPU (no rebuild — hidden=64):
```
python scripts/preprocess_data.py --split-eval    # build retrain_test.npz
python3 run_dataset_inference.py --npz ../data/processed/retrain_test.npz \
        --ckpt ../experiments/results/flair_h64_full.pt \
        --skip-build --skip-cpu-baseline
```

### NPU precision floor on the GOOD model — and the clip lever
Running the retrained hidden=64 model on the NPU (thesis-style test set) gave
NPU F1=0.894 vs PyTorch 0.936 (ROC-AUC 0.992 vs 0.999). This is NOT the toy
result (where NPU≡PyTorch) — because the good model reconstructs normal traffic
so well (scores ~0.1) that the NPU's numerical error becomes significant.

Diagnosed from the per-window scores CSV:
- The NPU error is **input-magnitude-dependent**, not a constant floor:
  corr(NPU−PyTorch score, mean|z|) = 0.70 on normal windows. bf16 has fixed
  ~0.4% *relative* precision, so *absolute* error grows with activation
  magnitude and compounds through the 10-step recurrence.
- Attacks (large scores) are barely affected. The damage is a heavy right tail
  on ~2% of NORMAL windows (those with peak|z|≈9), which inflates the normal-p99
  threshold 0.103→0.626 (6×) and thereby misses 676 mild attacks (PyTorch score
  0.1–0.6). No global additive/multiplicative correction recovers it (tested) —
  it is added variance, not offset; even label-informed best-F1 caps NPU at 0.899.

**Lever (validated PyTorch-side, pending NPU confirmation):** clip tighter.
Since the noise scales with |z|, capping |z| at 6 instead of 10 shrinks exactly
the activations driving the error. Retrained at clip=6, PyTorch cost is
negligible (F1 0.9418→0.9399, ROC-AUC unchanged) — so it should lower the NPU
threshold and recover recall at almost no accuracy cost. Checkpoint committed:
`experiments/results/flair_h64_clip6.pt`. To test on the NPU:
```
# set preprocess.clip_zscore: 6.0 in config.yaml, then:
python scripts/preprocess_data.py --split-eval          # regen retrain_test.npz at clip=6
python3 run_dataset_inference.py --npz ../data/processed/retrain_test.npz \
        --ckpt ../experiments/results/flair_h64_clip6.pt --skip-build --skip-cpu-baseline
# MUST use the clip=6 npz with the clip=6 ckpt (train/inference clip must match).
```
If confirmed, clip=6 (or a swept optimum) becomes the NPU-deployment default.
The deeper fix is higher-precision LUT nonlinearity (Problem B proper), but the
clip lever is far cheaper if it lands.

### Remaining / optional
- Held-out generalization metric: needs a split where held-out data contains
  anomalies (current chronological split is train-only for anomalies). Change
  `preprocess.split_ratios` / split strategy if a clean eval-set F1 is wanted.
- Threshold is path-specific: the clamped path's normal-p99 (~3.46 NPU on the
  20k subset) differs from the old unclamped `3.189216`. Re-derive per final
  execution path (unfused vs fused/batched decoder); never carry a locked
  threshold across paths.
- In-kernel `sigmoid16` clamp backstop still optional (see above); not needed
  while inputs are clamped to |z|≤10.
