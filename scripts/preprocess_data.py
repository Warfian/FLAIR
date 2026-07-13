"""
preprocess_data.py

Preprocesses a FLOW-LEVEL dataset (WUSTL-IIoT style) into sliding-window sequences
for a GRU autoencoder with categorical embeddings.

Outputs (single bundle):
  data/processed/preprocessed.npz containing:
    - X_num:       (N, T, D_num) float32  normalized numeric features
    - X_cat:       (N, T, D_cat) int64    categorical IDs (Sport, Dport, Proto)
    - y_seq:       (N,)          int64    window label (1 if any Target==1 in window else 0)
    - mu:          (D_num,)      float32  normalization mean (computed on normal rows only)
    - sigma:       (D_num,)      float32  normalization std (computed on normal rows only)
    - num_features (D_num,)      object   list of numeric feature names in order
    - cat_features (D_cat,)      object   list of categorical feature names in order
    - sport_vocab / dport_vocab / proto_vocab  object   dict: value -> id

Design choices:
- We normalize ONLY numeric features using normal rows (Target==0).
- We treat Sport/Dport/Proto as categorical IDs and DO NOT normalize them.
- We reserve ID=0 for UNK (unseen values).
- Feature lists are read from config.yaml (features.categorical / features.numeric).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml


# Opens config.yaml and parses it with yaml.safe_load
# This is the single dictionary that drives every setting
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Makes sure the specified folder exists before we try to write into it
def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# Looks at the file extension and dispatches to either pd.read_excel or pd.read_csv
def read_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

# Takes the StartTime column and makes sure it's an actual datetime type
def to_datetime_safe(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

# Turns a categorical column into small integer IDs a neural net can embed.
def build_vocab(values: pd.Series) -> Dict[int, int]:
    """
    Build mapping from raw port value -> integer ID.
    Reserve 0 for UNK.
    """
    # Drop NaNs, convert to int safely
    v = values.dropna().astype(int).unique().tolist()
    v_sorted = sorted(v)
    vocab = {port: (i + 1) for i, port in enumerate(v_sorted)}  # start at 1
    return vocab

# Takes the raw column plus the vocab dict built above, and produces the actual integer array to feed the model
def encode_with_vocab(values: pd.Series, vocab: Dict[int, int]) -> np.ndarray:
    """
    Encode ports with vocab; unknown -> 0
    """
    # Convert to numeric, coerce bad -> NaN, fill -> -1 -> UNK
    vals = pd.to_numeric(values, errors="coerce")
    out = np.zeros(len(vals), dtype=np.int64)

    # Where we have valid numbers, map
    valid = vals.notna()
    vals_i = vals[valid].astype(int).to_numpy()
    mapped = np.array([vocab.get(p, 0) for p in vals_i], dtype=np.int64)
    out[valid.to_numpy()] = mapped
    return out

# Makes columns with wildly different scales comparable to a neural net
def zscore_normalize_numeric(X_num: np.ndarray, y_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mu/sigma on NORMAL rows only (y_row==0), then scale all rows.
    """
    normal_mask = (y_row == 0)
    if normal_mask.sum() < 10:
        raise ValueError("Not enough normal rows to compute normalization stats.")

    mu = X_num[normal_mask].mean(axis=0).astype(np.float32)
    sigma = X_num[normal_mask].std(axis=0).astype(np.float32)
    sigma = np.where(sigma < 1e-8, 1.0, sigma).astype(np.float32)

    X_scaled = ((X_num - mu) / sigma).astype(np.float32)
    return X_scaled, mu, sigma

# Turns a flat table of flows into the (N, T, D) sequence format the GRU model expects
def build_sliding_windows(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y_row: np.ndarray,
    window_size: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X_num: (M, D_num)
    X_cat: (M, D_cat)
    y_row: (M,) row labels
    Returns:
      X_num_win: (N, T, D_num)
      X_cat_win: (N, T, D_cat)
      y_seq:     (N,)   any(y_row==1 in window)
    """
    M = len(y_row)
    if M < window_size:
        raise ValueError(f"Not enough rows ({M}) for window_size={window_size}")

    num_windows = 1 + (M - window_size) // stride
    Xn = np.zeros((num_windows, window_size, X_num.shape[1]), dtype=np.float32)
    Xc = np.zeros((num_windows, window_size, X_cat.shape[1]), dtype=np.int64)
    ys = np.zeros((num_windows,), dtype=np.int64)

    w = 0
    for start in range(0, M - window_size + 1, stride):
        end = start + window_size
        Xn[w] = X_num[start:end]
        Xc[w] = X_cat[start:end]
        ys[w] = 1 if y_row[start:end].max() > 0 else 0
        w += 1

    return Xn, Xc, ys


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)

    CATEGORICAL_FEATURES: List[str] = cfg["features"]["categorical"]
    NUMERIC_FEATURES: List[str] = cfg["features"]["numeric"]

    time_col = cfg["data"]["time_column"]
    label_col = cfg["data"]["label_column"]

    window_size = int(cfg["preprocess"]["window_size"])
    stride = int(cfg["preprocess"].get("stride", 1))
    sort_time = bool(cfg["preprocess"].get("sort_time", True))
    dropna = bool(cfg["preprocess"].get("dropna", True))

    paths_cfg = cfg.get("paths", {})
    sample_xlsx = paths_cfg.get("sample_xlsx")
    full_csv = paths_cfg.get("full_csv")
    input_path = full_csv or sample_xlsx
    if not input_path:
        raise ValueError("No input dataset path provided (paths.full_csv or paths.sample_xlsx).")

    out_npz = paths_cfg.get("processed_npz", "data/processed/preprocessed.npz")
    ensure_parent_dir(out_npz)

    print(f"[preprocess] Reading dataset: {input_path}")
    df = read_dataset(input_path)
    print(f"[preprocess] Loaded shape: {df.shape}")

    required = [time_col, label_col] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df[required].copy()

    # Parse time for sorting (NOT used as feature)
    work[time_col] = to_datetime_safe(work[time_col])

    if dropna:
        # We allow categorical to have NaNs (will become UNK=0), but time+numeric+label must be present
        work = work.dropna(subset=[time_col, label_col] + NUMERIC_FEATURES).copy()

    if sort_time:
        work = work.sort_values(by=time_col).reset_index(drop=True)

    # Row labels (0/1)
    y_row = work[label_col].astype(int).to_numpy(dtype=np.int64)

    # ---------------------------
    # Categorical: build vocabs and encode
    # ---------------------------
    # Build vocab from ALL rows in this dataset file (label-agnostic)
    sport_vocab = build_vocab(work["Sport"])
    dport_vocab = build_vocab(work["Dport"])
    proto_vocab = build_vocab(work["Proto"])

    sport_ids = encode_with_vocab(work["Sport"], sport_vocab)
    dport_ids = encode_with_vocab(work["Dport"], dport_vocab)
    proto_ids = encode_with_vocab(work["Proto"], proto_vocab)

    X_cat = np.stack([sport_ids, dport_ids, proto_ids], axis=1).astype(np.int64)  # (M, 3)

    # ---------------------------
    # Numeric: z-score normalize using NORMAL rows only
    # ---------------------------
    X_num_raw = work[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    X_num, mu, sigma = zscore_normalize_numeric(X_num_raw, y_row)

    # ---------------------------
    # Windows
    # ---------------------------
    X_num_win, X_cat_win, y_seq = build_sliding_windows(
        X_num=X_num,
        X_cat=X_cat,
        y_row=y_row,
        window_size=window_size,
        stride=stride
    )

    np.savez(
        out_npz,
        X_num=X_num_win,
        X_cat=X_cat_win,
        y_seq=y_seq,
        mu=mu,
        sigma=sigma,
        num_features=np.array(NUMERIC_FEATURES, dtype=object),
        cat_features=np.array(CATEGORICAL_FEATURES, dtype=object),
        sport_vocab=np.array([sport_vocab], dtype=object),
        dport_vocab=np.array([dport_vocab], dtype=object),
        proto_vocab=np.array([proto_vocab], dtype=object),
    )

    print(f"[preprocess] saved: {out_npz}")
    print(f"[preprocess] X_num shape: {X_num_win.shape}  (N, T, D_num)")
    print(f"[preprocess] X_cat shape: {X_cat_win.shape}  (N, T, D_cat)")
    print(f"[preprocess] y_seq shape: {y_seq.shape}  attack_windows={int(y_seq.sum())}/{len(y_seq)}")
    print(f"[preprocess] Sport vocab size: {len(sport_vocab)+1} (including UNK)")
    print(f"[preprocess] Dport vocab size: {len(dport_vocab)+1} (including UNK)")
    print(f"[preprocess] Proto vocab size: {len(proto_vocab)+1} (including UNK)")


def chronological_split_ranges(M: int, ratios: List[float]) -> List[Tuple[int, int]]:
    """Contiguous, time-ordered row ranges (train, eval, inference) summing to M."""
    if len(ratios) != 3:
        raise ValueError("split_ratios must have exactly 3 values (train, eval, inference).")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {ratios}")

    bounds = [0]
    acc = 0.0
    for r in ratios[:-1]:
        acc += r
        bounds.append(int(round(M * acc)))
    bounds.append(M)
    return [(bounds[i], bounds[i + 1]) for i in range(3)]


def main_split(config_path: str = "config.yaml") -> None:
    """
    Preprocess paths.full_csv_split into a chronological 80/10/10 train/eval/inference
    split (see preprocess.split_ratios), each windowed independently so no window
    spans a split boundary. Normalization stats (mu/sigma) are computed from the
    TRAIN split's normal rows only and reused for eval/inference, matching how the
    model was trained.
    """
    cfg = load_config(config_path)

    CATEGORICAL_FEATURES: List[str] = cfg["features"]["categorical"]
    NUMERIC_FEATURES: List[str] = cfg["features"]["numeric"]

    time_col = cfg["data"]["time_column"]
    label_col = cfg["data"]["label_column"]

    window_size = int(cfg["preprocess"]["window_size"])
    stride = int(cfg["preprocess"].get("stride", 1))
    sort_time = bool(cfg["preprocess"].get("sort_time", True))
    dropna = bool(cfg["preprocess"].get("dropna", True))
    ratios = cfg["preprocess"].get("split_ratios", [0.8, 0.1, 0.1])

    paths_cfg = cfg.get("paths", {})
    input_path = paths_cfg.get("full_csv_split") or paths_cfg.get("full_csv")
    if not input_path:
        raise ValueError("paths.full_csv_split must be set in config.yaml for --split mode.")

    out_train = paths_cfg.get("processed_train_npz", "data/processed/preprocessed_train.npz")
    out_eval = paths_cfg.get("processed_eval_npz", "data/processed/preprocessed_eval.npz")
    out_inference = paths_cfg.get("processed_inference_npz", "data/processed/preprocessed_inference.npz")
    for p in (out_train, out_eval, out_inference):
        ensure_parent_dir(p)

    print(f"[preprocess-split] Reading dataset: {input_path}")
    df = read_dataset(input_path)
    print(f"[preprocess-split] Loaded shape: {df.shape}")

    required = [time_col, label_col] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = df[required].copy()
    work[time_col] = to_datetime_safe(work[time_col])

    if dropna:
        work = work.dropna(subset=[time_col, label_col] + NUMERIC_FEATURES).copy()
    if sort_time:
        work = work.sort_values(by=time_col).reset_index(drop=True)

    y_row = work[label_col].astype(int).to_numpy(dtype=np.int64)

    # Vocabs from the full dataset so category IDs are consistent across splits.
    sport_vocab = build_vocab(work["Sport"])
    dport_vocab = build_vocab(work["Dport"])
    proto_vocab = build_vocab(work["Proto"])

    X_cat = np.stack([
        encode_with_vocab(work["Sport"], sport_vocab),
        encode_with_vocab(work["Dport"], dport_vocab),
        encode_with_vocab(work["Proto"], proto_vocab),
    ], axis=1).astype(np.int64)

    X_num_raw = work[NUMERIC_FEATURES].to_numpy(dtype=np.float32)

    M = len(work)
    (tr_start, tr_end), (ev_start, ev_end), (inf_start, inf_end) = chronological_split_ranges(M, ratios)
    print(f"[preprocess-split] rows: train={tr_end - tr_start}  eval={ev_end - ev_start}  "
          f"inference={inf_end - inf_start}")

    # Normalize using TRAIN-split normal rows only; apply the same mu/sigma to all splits.
    _, mu, sigma = zscore_normalize_numeric(X_num_raw[tr_start:tr_end], y_row[tr_start:tr_end])
    X_num = ((X_num_raw - mu) / sigma).astype(np.float32)

    splits = {
        "train": (tr_start, tr_end, out_train),
        "eval": (ev_start, ev_end, out_eval),
        "inference": (inf_start, inf_end, out_inference),
    }

    for name, (start, end, out_path) in splits.items():
        Xn_win, Xc_win, y_seq = build_sliding_windows(
            X_num=X_num[start:end],
            X_cat=X_cat[start:end],
            y_row=y_row[start:end],
            window_size=window_size,
            stride=stride,
        )
        np.savez(
            out_path,
            X_num=Xn_win,
            X_cat=Xc_win,
            y_seq=y_seq,
            mu=mu,
            sigma=sigma,
            num_features=np.array(NUMERIC_FEATURES, dtype=object),
            cat_features=np.array(CATEGORICAL_FEATURES, dtype=object),
            sport_vocab=np.array([sport_vocab], dtype=object),
            dport_vocab=np.array([dport_vocab], dtype=object),
            proto_vocab=np.array([proto_vocab], dtype=object),
        )
        print(f"[preprocess-split] {name}: saved {out_path}  X_num={Xn_win.shape}  "
              f"attack_windows={int(y_seq.sum())}/{len(y_seq)}")

    print(f"[preprocess-split] Sport vocab size: {len(sport_vocab) + 1} (including UNK)")
    print(f"[preprocess-split] Dport vocab size: {len(dport_vocab) + 1} (including UNK)")
    print(f"[preprocess-split] Proto vocab size: {len(proto_vocab) + 1} (including UNK)")


if __name__ == "__main__":
    if "--split" in sys.argv:
        main_split()
    else:
        main()