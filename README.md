# FLAIR — Flow-Level Autoencoder for Intrusion Recognition

FLAIR is a GRU-based autoencoder for unsupervised anomaly detection on network flow data. It is trained exclusively on normal traffic and flags anomalies by measuring how poorly it reconstructs a window of flows. This repository implements FLAIR for the **WUSTL-IIoT** dataset using the 24 selected features from Table III of the paper (3 categorical + 21 numeric).

---

## Repository Structure

```
FLAIR/
├── config.yaml                        # Single source of truth for all settings
├── requirements.txt                   # Python dependencies
│
├── scripts/
│   └── preprocess_data.py             # Step 1: raw CSV/XLSX → preprocessed.npz
│
├── src/
│   ├── data/
│   │   ├── feature_definitions.py     # Feature name lists (categorical + numeric)
│   │   ├── dataset.py                 # PyTorch Dataset (FLAIRDataset)
│   │   ├── flow_extractor.py          # Raw flow parsing utilities
│   │   ├── flow_window_builder.py     # Sliding-window construction helpers
│   │   └── normalization.py           # Z-score normalization utilities
│   │
│   ├── models/
│   │   ├── flair_model.py             # Top-level FLAIRAutoencoder + FLAIRConfig
│   │   ├── encoder.py                 # GRUEncoder
│   │   ├── decoder.py                 # GRUDecoder
│   │   └── attention.py               # (Optional) attention module
│   │
│   ├── training/
│   │   ├── train_flair.py             # Step 2: train on normal windows
│   │   ├── evaluate_flair.py          # Step 3: anomaly scores + metrics
│   │   └── thresholding.py            # Threshold selection utilities
│   │
│   └── analysis/
│       ├── anomaly_analysis.py        # Post-hoc analysis helpers
│       ├── metrics.py                 # Evaluation metrics (F1, ROC, PR)
│       └── plots.py                   # Plotting utilities
│
├── data/
│   └── processed/
│       └── preprocessed.npz           # Output of Step 1 (generated, not committed)
│
└── experiments/
    └── results/
        ├── flair_minimal.pt           # Saved model checkpoint (generated)
        └── anomaly_scores.csv         # Per-window anomaly scores (generated)
```

---

## Features

FLAIR uses **24 flow-level features** (matching Table III of the paper):

| Type | Features |
|------|----------|
| **Categorical** (embedded) | `Sport`, `Dport`, `Proto` |
| **Numeric** (z-score normalized) | `Mean`, `SrcPkts`, `DstPkts`, `TotPkts`, `SrcBytes`, `DstBytes`, `TotBytes`, `SrcLoad`, `DstLoad`, `Load`, `SrcRate`, `DstRate`, `Rate`, `SrcLoss`, `DstLoss`, `Loss`, `pLoss`, `SrcJitter`, `DstJitter`, `SIntPkt`, `DIntPkt` |

Categorical features are mapped to integer IDs and passed through learned `nn.Embedding` layers. Numeric features are z-score normalized using **normal-traffic rows only** (label = 0). The model reconstructs numeric features only; anomaly score is per-window MSE reconstruction error.

---

## Configuration

All pipeline settings live in [`config.yaml`](config.yaml). Edit this file to change paths, hyperparameters, or feature lists — no code changes needed.

Key sections:

```yaml
features:          # Which columns are categorical vs numeric
preprocess:        # window_size, stride, sort_time, dropna
paths:             # Input dataset path and output .npz path
model:             # hidden_dim, num_layers, dropout, bidirectional
training:          # batch_size, learning_rate, epochs, seed, device, checkpoint_path
evaluation:        # threshold_percentile, output_csv
```

---

## Environment Setup

### Requirements

- Python 3.10 or 3.11 (recommended)
- PyTorch 2.10.0
- See [`requirements.txt`](requirements.txt) for the full pinned dependency list

### Create and activate a virtual environment

**Windows (PowerShell / CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU training:** If you have a CUDA-capable GPU, install the matching CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`. Then set `device: "cuda"` in `config.yaml`.

---

## Pipeline

All three steps are driven by `config.yaml` and should be run from the **repository root**.

### Step 1 — Preprocess

Reads the raw WUSTL-IIoT dataset (`.xlsx` or `.csv`), builds vocabularies for the categorical features, z-score normalizes numeric features on normal rows, and produces sliding-window sequences saved as a single `.npz` bundle.

```bash
python -m scripts.preprocess_data
```

Output: `data/processed/preprocessed.npz` containing:
- `X_num` — `(N, T, 21)` normalized numeric windows
- `X_cat` — `(N, T, 3)` categorical ID windows (Sport, Dport, Proto)
- `y_seq` — `(N,)` window-level labels
- `sport_vocab`, `dport_vocab`, `proto_vocab` — value-to-ID mappings
- `mu`, `sigma` — normalization statistics (computed on normal rows only)

### Step 2 — Train

Loads the `.npz` bundle, filters to normal-only windows, splits into train/val, and trains the GRU autoencoder with early stopping. Saves the best checkpoint.

```bash
python -m src.training.train_flair
```

Output: `experiments/results/flair_minimal.pt`

### Step 3 — Evaluate

Loads the saved checkpoint and `.npz` bundle, computes per-window reconstruction error (anomaly score), applies a percentile threshold derived from normal windows, and reports full metrics.

```bash
python -m src.training.evaluate_flair
```

Output: `experiments/results/anomaly_scores.csv` and printed metrics:
- Confusion matrix, Accuracy, Precision, Recall, F1, FPR
- ROC AUC and PR AUC (threshold-independent)
- Best-F1 threshold (label-informed upper bound for reporting)

---

## Model Architecture

```
x_num (B, T, 21)  ──────────────────────────────────────────┐
x_cat (B, T, 3)   → Embedding(Sport) ┐                      │
                  → Embedding(Dport) ├─ concat → x_in (B,T,D)─→ GRUEncoder → latent
                  → Embedding(Proto) ┘                              │
                                                             GRUDecoder
                                                                    │
                                                          x_hat_num (B, T, 21)
                                                                    │
                                              Anomaly score = MSE(x_num, x_hat_num)
```

- **Encoder:** multi-layer GRU, optionally bidirectional
- **Decoder:** GRU that expands the final latent state back to sequence length
- **Loss:** mean squared error on numeric reconstruction (categorical features are not reconstructed)
- **Anomaly score:** mean per-timestep MSE across all 21 numeric features, giving one scalar per window

---

## NPU Demo Site

`webdemo/` is a small local website that runs FLAIR on the CPU (PyTorch) and
on the Ryzen AI NPU at the same time over the sample dataset, and displays
the speedup and accuracy side by side. It's meant to run on the machine that
actually has the NPU attached.

It's stdlib-only on the server side (`http.server` + `threading`, no Flask/
FastAPI) so it doesn't add a new pip dependency to the pinned IRON/XRT
toolchain environment.

### One-time setup (build the NPU designs)

From the WSL IRON environment, in `npu/`:

```bash
make -f Makefile.encoder
make -f Makefile.decoder
make -f Makefile.batch
```

This produces `build/gru.xclbin`, `build/decoder.xclbin`, and
`batch_infer.exe` — the demo server always runs with `--skip-build` (the
default) so a slow AIE recompile never happens on a "Run the demo" click.
Re-run the relevant `make` command whenever a kernel `.cc` file changes.

### Running the site

From the WSL IRON environment (same one `run_dataset_inference.py` normally
runs in — it needs torch, numpy, ml_dtypes, and `powershell.exe` reachable
for the NPU-side `.exe`):

```bash
python3 webdemo/server.py --limit 200
```

Then open `http://127.0.0.1:8765` in a browser on the same machine. Each
"Run the demo" click streams `--limit` windows through both the CPU and NPU
pipelines concurrently and reports:

- the NPU speedup multiplier (pure per-window streamed inference time, i.e.
  excluding one-time xclbin load)
- ROC-AUC and F1 for both, and how closely NPU scores track the CPU's
- a few example traffic windows with both scores side by side

Useful flags: `--npz` / `--seq-len` to point at a different dataset/window
size, `--xrt-inc-dir` / `--xrt-lib-dir` if XRT isn't auto-detected, `--port`
to change the port.

### UI development without hardware

`python3 webdemo/server.py --mock` fabricates plausible progress/timing/
accuracy numbers using only the Python standard library (no torch, no NPU)
so the frontend can be iterated on anywhere. The page shows a "demo mode"
banner whenever mock data is being displayed.
