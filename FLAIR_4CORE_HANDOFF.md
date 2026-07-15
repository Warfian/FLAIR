# FLAIR Four-Core NPU Final Handoff

## Status

This branch contains the validated four-core FLAIR GRU autoencoder pipeline.

The final configuration was validated end to end on:

- Dataset: `data/processed/retrain_test.npz`
- Checkpoint: `experiments/results/flair_h64_full.pt`
- Windows: 114,956
- Anomalies: 8,369
- Sequence length: 10
- Encoder host batch: 8
- Decoder host batch: 8
- Four NPU workers, two windows per worker per dispatch

The full validation completed successfully with exit status 0.

## Final architecture

```text
CPU input preparation and categorical embeddings
    ↓
Four-core NPU GRU encoder
    ↓
CPU latent_to_hidden projection
    ↓
Four-core NPU unfused GRU decoder
    ↓
CPU hidden_to_output reconstruction
    ↓
CPU MSE anomaly scoring and threshold evaluation
```

“Unfused” does not omit reconstruction. The NPU decoder returns the complete hidden sequence, and the final decoder output projection is performed on the CPU in FP32. This path was faster and more accurate than the fused NPU decoder.

## Final validated accuracy

| Metric | NPU | PyTorch |
|---|---:|---:|
| ROC-AUC | 0.9987 | 0.9989 |
| F1 | 0.9368 | 0.9370 |
| Recall | 0.9933 | 0.9938 |
| Precision | 0.8863 | 0.8864 |
| False positives | 1,066 | 1,066 |
| False negatives | 56 | 52 |
| True positives | 8,313 | 8,317 |
| True negatives | 105,521 | 105,521 |
| Score correlation | 0.9990 | — |
| Evaluation threshold | 0.11483 | 0.09696 |

The NPU missed only four more anomalies than PyTorch over 8,369 anomalies.

### Threshold warning

`run_dataset_inference.py` self-calibrates each path by taking the configured percentile of the normal scores in the evaluated labeled split. The threshold `0.11483` is therefore an evaluation result, not automatically a production threshold.

For deployment:

1. Run the final selected pipeline on a separate representative normal-only calibration set.
2. Calculate and save the chosen normal-score percentile.
3. Reuse that fixed threshold on unknown traffic.
4. Do not recalibrate using labels from the deployment/test stream.

## Final validated speed

### Full-window run

- Four-core encoder: **90.3185 µs/window**
- Four-core unfused decoder: **78.2175 µs/window**
- Combined NPU-stage time: **168.5360 µs/window**

These values include the host dispatch overhead reported by `batch_infer.exe`, but they do not separately time CPU preprocessing, `latent_to_hidden`, `hidden_to_output`, reconstruction MSE, or CSV generation.

### Speedups

Encoder comparison:

- Earlier one-core encoder diagnostic: approximately **501.984 µs/window**
- Final four-core encoder: **90.3185 µs/window**
- Approximate encoder speedup: **5.56×**

Decoder comparison:

- Matching one-core unfused decoder at batch 2: **303.382 µs/window**
- Final four-core unfused decoder: **78.2175 µs/window**
- Decoder speedup: **3.88×**
- The four-core unfused decoder was bit-exact against the matching one-core batch-2 decoder.

The isolated four-core decoder diagnostic measured approximately 80.636 µs/window, consistent with the 78.2175 µs/window full-window result.

## Important numerical decision

The final reciprocal implementation is:

```text
scalar getInvBf16 seed
+ two Newton-Raphson refinements
```

The encoder retains the compiler/code-generation optimization:

```text
matvec_bias(...)
+ gru_step_with_gi(...)
```

Do not re-enable the vectorized `0x7EE6` reciprocal seed without repeating full accuracy validation.

That vectorized seed improved execution speed but caused a significant regression:

| Metric | Accurate final | Vectorized seed |
|---|---:|---:|
| False negatives | 56 | 221 |
| F1 | 0.9368 | 0.9268 |
| ROC-AUC | 0.9987 | 0.9981 |
| Score correlation | 0.9990 | 0.9972 |
| NPU threshold | 0.11483 | 0.24978 |

The threshold was recalibrated in both runs, so the extra false negatives were a true score-separation regression rather than an old-threshold issue.

## Main changes

### `npu/gru_encoder_4core.py`

Introduces the four-worker encoder architecture:

- Four compute-tile workers
- Memory-tile input split
- Shared parameter forwarding/broadcast
- Output join
- `--batch` means batch per core
- Host-visible batch is four times the compile-time per-core batch

For the validated pipeline:

```text
host batch = 8
compile batch = 2 per core
```

### `npu/gru_decoder_4core.py`

Introduces the four-worker unfused decoder architecture:

- Kernel symbol: `gru_decoder_bf16`
- Four compute workers
- Memory-tile input split
- Decoder parameter forwarding/broadcast
- Hidden-sequence output join
- Per-window input: 64 BF16 values
- Per-window output: 10 × 64 = 640 BF16 values
- Parameters: 24,960 BF16 values

This wrapper was validated bit-exact against a matching one-core batch-2 decoder.

### `npu/gru_decoder_fused_4core.py`

Four-core fused decoder implementation retained for comparison and future experiments.

It is not the recommended final path:

- Four-core fused: approximately 140.895 µs/window
- Four-core unfused: approximately 80.636 µs/window in the isolated matched test
- Fused was about 1.75× slower and also gave worse full-model accuracy in earlier validation

### `npu/kernels/gru_encoder.cc`

The encoder was changed from the monolithic:

```cpp
gru_step(...)
```

to:

```cpp
matvec_bias(...)
gru_step_with_gi(...)
```

The mathematical operation is unchanged. Separating the input matvec from the recurrent gate step generated substantially better encoder code and reduced full-run encoder time from roughly 131 to 90 µs/window.

### `npu/kernels/gru_common.h`

Keep the accurate scalar reciprocal seed followed by two Newton refinements.

Do not replace it wholesale with the vectorized-seed upstream version unless intentionally repeating the numerical experiment.

### `npu/run_dataset_inference.py`

Added:

- `--encoder-4core`
- `--decoder-4core`
- Four-core build-script selection
- Correct compile-batch conversion (`host_batch // 4`)
- Host-visible batch remains unchanged for `batch_infer.exe`
- Batch-divisibility checks
- Four-core decoder restricted to `--decoder-mode unfused`
- Batch-specific xclbin and project selection
- Correct padding to the LCM of encoder and decoder host batches

Important batch rule:

```text
IRON four-core --batch argument = windows per core
batch_infer.exe batch argument = total windows per dispatch
```

For the validated configuration:

```text
IRON compile batch: 2
batch_infer.exe batch: 8
```

## Build-cache warning

IRON/aiecc does not always invalidate external kernel objects when included headers change. A stale project can silently link an old kernel.

Before testing any kernel/header change, delete the exact selected project directories and xclbins. For the final four-core path:

```bash
cd npu

rm -rf \
  build/gru_4core.prj \
  build/decoder_4core.prj

rm -f \
  build/gru_4core.xclbin \
  build/gru_4core_insts.bin \
  build/decoder_4core.xclbin \
  build/decoder_4core_insts.bin
```

The first run after a kernel change must not use `--skip-build`.

## Reproduce the final smoke test

From `npu/`:

```bash
set -o pipefail

python3 run_dataset_inference.py \
  --npz ../data/processed/retrain_test.npz \
  --ckpt ../experiments/results/flair_h64_full.pt \
  --sample 240 \
  --sample-seed 0 \
  --batch-encoder 8 \
  --batch-decoder 8 \
  --encoder-4core \
  --decoder-4core \
  --decoder-mode unfused \
  --skip-cpu-baseline \
  2>&1 | tee results/benchmarks_4core/smoke_final_4core.txt

echo "Smoke status: ${PIPESTATUS[0]}"
```

Expected smoke behavior:

- Exit status 0
- No NaNs
- ROC-AUC approximately 1.0 on the fixed 240-window sample
- Encoder around 90–100 µs/window on the small sample, subject to host variance
- Decoder around 80–85 µs/window

## Reproduce the full validation

After a successful build/smoke run:

```bash
set -o pipefail

python3 run_dataset_inference.py \
  --npz ../data/processed/retrain_test.npz \
  --ckpt ../experiments/results/flair_h64_full.pt \
  --batch-encoder 8 \
  --batch-decoder 8 \
  --encoder-4core \
  --decoder-4core \
  --decoder-mode unfused \
  --skip-build \
  --skip-cpu-baseline \
  2>&1 | tee results/benchmarks_4core/full_final_4core.txt

echo "Full status: ${PIPESTATUS[0]}"

cp npu_vs_pytorch_scores.csv \
  results/benchmarks_4core/full_final_4core_scores.csv
```

Expected full-run acceptance range:

- NPU ROC-AUC: approximately 0.9987
- NPU F1: approximately 0.9368
- False negatives: approximately 56
- Score correlation: approximately 0.9990
- Encoder: approximately 90.3 µs/window
- Decoder: approximately 78.2 µs/window

Minor timing variation is expected. Material accuracy differences require investigation.

## Files/results worth preserving

Recommended files in the handoff package:

```text
FLAIR_4CORE_HANDOFF.md
npu/gru_encoder_4core.py
npu/gru_decoder_4core.py
npu/gru_decoder_fused_4core.py
npu/diag_4core_timing.py
npu/run_dataset_inference.py
npu/kernels/gru_common.h
npu/kernels/gru_encoder.cc
npu/kernels/gru_decoder.cc
npu/results/benchmarks_4core/full_codegen_only.txt
npu/results/benchmarks_4core/full_codegen_only_scores.csv
experiments/results/flair_h64_full.pt
```

The following built files are useful for convenience but should not be treated as portable source artifacts:

```text
npu/build/gru_4core.xclbin
npu/build/gru_4core_insts.bin
npu/build/decoder_4core.xclbin
npu/build/decoder_4core_insts.bin
npu/batch_infer.exe
```

Rebuilding is preferred when the teammate's MLIR-AIE, XRT, compiler, or hardware environment differs.

## Recommended Git handoff

The preferred handoff is a committed branch plus an annotated tag.

Suggested tag:

```text
flair-4core-validated
```

Suggested commit message if the final state is not committed:

```text
Finalize accurate four-core FLAIR pipeline
```

The package script supplied with this handoff creates:

- A Git bundle containing the current branch and validation tag
- A committed-source archive
- The checkpoint, benchmark logs, score CSV, and final xclbins when present
- A manifest and SHA-256 checksums
- A final compressed handoff archive

## Receiver instructions

To clone the complete Git history from the bundle:

```bash
git clone flair-4core-validated.bundle FLAIR_fused
cd FLAIR_fused
git checkout flair-4core-validated
```

If the tag checkout is detached, create a local branch:

```bash
git switch -c fourcore-final
```

Then read this file, activate the expected IRON environment, clear/rebuild the four-core caches, run the smoke test, and finally run the full validation.

## Suggested next work

The model implementation and full validation are complete. The highest-value remaining engineering task is true end-to-end profiling, including:

- CPU feature/embedding preparation
- NPU encoder
- CPU `latent_to_hidden`
- NPU decoder
- CPU `hidden_to_output`
- Reconstruction MSE and threshold comparison
- Host-device transfer and file-I/O effects

This would distinguish pure NPU-stage latency from deployable end-to-end latency and identify whether the CPU projection or transport becomes the next bottleneck.
