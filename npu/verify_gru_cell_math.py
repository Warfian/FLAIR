"""
verify_gru_cell_math.py

De-risks the AIE gru_cell kernel *before* writing any C++: validates the
exact gate-split formula the kernel will implement against PyTorch's real
nn.GRUCell, using the actual trained FLAIR encoder weights and a real
preprocessed window's first timestep.

This is a pure numpy re-implementation of the formula intended for
npu/kernels/gru_cell.cc -- if this doesn't match PyTorch, the kernel won't
either, so fix it here first where iteration is instant.

Usage:
    python -m npu.verify_gru_cell_math
"""

from __future__ import annotations

import numpy as np
import torch

CHECKPOINT_PATH = "experiments/results/flair_minimal.pt"
NPZ_PATH = "data/processed/preprocessed.npz"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell_golden(
    x: np.ndarray,
    h_prev: np.ndarray,
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
) -> np.ndarray:
    """Numpy reference for the exact formula gru_cell.cc will implement.

    x: (input_dim,)   h_prev: (hidden,)
    w_ih: (3*hidden, input_dim)   w_hh: (3*hidden, hidden)
    b_ih, b_hh: (3*hidden,)

    Gate order along the 3*hidden axis is [reset, update, new] -- PyTorch's
    GRU convention. The 'new' gate's hidden-side contribution must stay
    separate from the input-side contribution until after the reset gate
    is applied (gh_n is gated by r before being combined with gi_n) --
    this is the one place a naive "sum everything then split" kernel
    implementation would silently produce the wrong answer.
    """
    hidden = h_prev.shape[0]

    gi = w_ih @ x + b_ih  # (3*hidden,)
    gh = w_hh @ h_prev + b_hh  # (3*hidden,)

    gi_r, gi_z, gi_n = gi[:hidden], gi[hidden : 2 * hidden], gi[2 * hidden :]
    gh_r, gh_z, gh_n = gh[:hidden], gh[hidden : 2 * hidden], gh[2 * hidden :]

    r = sigmoid(gi_r + gh_r)
    z = sigmoid(gi_z + gh_z)
    n = np.tanh(gi_n + r * gh_n)

    h_next = (1.0 - z) * n + z * h_prev
    return h_next


def main() -> None:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = ckpt["model_cfg"]
    print(f"model_cfg: {cfg}")

    w_ih = sd["encoder.gru.weight_ih_l0"].numpy().astype(np.float64)
    w_hh = sd["encoder.gru.weight_hh_l0"].numpy().astype(np.float64)
    b_ih = sd["encoder.gru.bias_ih_l0"].numpy().astype(np.float64)
    b_hh = sd["encoder.gru.bias_hh_l0"].numpy().astype(np.float64)
    hidden = cfg["hidden_dim"]
    print(f"w_ih {w_ih.shape}  w_hh {w_hh.shape}  b_ih {b_ih.shape}  b_hh {b_hh.shape}")

    # Build the real combined input for one timestep: 21 numeric features +
    # 3x8-dim categorical embeddings, exactly as FLAIRAutoencoder._combine_inputs does.
    bundle = np.load(NPZ_PATH, allow_pickle=True)
    x_num0 = bundle["X_num"][0, 0].astype(np.float64)  # (21,)
    x_cat0 = bundle["X_cat"][0, 0].astype(np.int64)  # (3,) = [sport_id, dport_id, proto_id]

    sport_e = sd["sport_emb.weight"].numpy().astype(np.float64)[x_cat0[0]]
    dport_e = sd["dport_emb.weight"].numpy().astype(np.float64)[x_cat0[1]]
    proto_e = sd["proto_emb.weight"].numpy().astype(np.float64)[x_cat0[2]]
    x_in = np.concatenate([x_num0, sport_e, dport_e, proto_e])  # (45,)
    print(f"x_in shape: {x_in.shape}")

    h_prev = np.zeros(hidden, dtype=np.float64)

    # --- PyTorch reference ---
    cell = torch.nn.GRUCell(input_size=x_in.shape[0], hidden_size=hidden)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.from_numpy(w_ih))
        cell.weight_hh.copy_(torch.from_numpy(w_hh))
        cell.bias_ih.copy_(torch.from_numpy(b_ih))
        cell.bias_hh.copy_(torch.from_numpy(b_hh))
        h_next_torch = cell(
            torch.from_numpy(x_in).float().unsqueeze(0),
            torch.from_numpy(h_prev).float().unsqueeze(0),
        ).squeeze(0).numpy().astype(np.float64)

    # --- numpy golden (== intended AIE kernel formula) ---
    h_next_golden = gru_cell_golden(x_in, h_prev, w_ih, w_hh, b_ih, b_hh)

    diff = np.abs(h_next_torch - h_next_golden)
    print(f"\nmax abs diff vs nn.GRUCell: {diff.max():.3e}")
    print(f"mean abs diff vs nn.GRUCell: {diff.mean():.3e}")

    if np.allclose(h_next_torch, h_next_golden, atol=1e-5):
        print("PASS: golden formula matches nn.GRUCell within 1e-5.")
    else:
        print("FAIL: golden formula diverges from nn.GRUCell -- fix formula before writing C++.")
        raise SystemExit(1)

    # Second step, feeding h_next back in as h_prev, to also exercise the
    # r-gates-hidden-contribution path with a nonzero h_prev (h_prev=0 on
    # step 1 makes some terms trivially zero and could hide a bug).
    x_in1_num = bundle["X_num"][0, 1].astype(np.float64)
    x_cat1 = bundle["X_cat"][0, 1].astype(np.int64)
    x_in1 = np.concatenate(
        [
            x_in1_num,
            sd["sport_emb.weight"].numpy().astype(np.float64)[x_cat1[0]],
            sd["dport_emb.weight"].numpy().astype(np.float64)[x_cat1[1]],
            sd["proto_emb.weight"].numpy().astype(np.float64)[x_cat1[2]],
        ]
    )
    with torch.no_grad():
        h_next2_torch = cell(
            torch.from_numpy(x_in1).float().unsqueeze(0),
            torch.from_numpy(h_next_torch).float().unsqueeze(0),
        ).squeeze(0).numpy().astype(np.float64)
    h_next2_golden = gru_cell_golden(x_in1, h_next_golden, w_ih, w_hh, b_ih, b_hh)
    diff2 = np.abs(h_next2_torch - h_next2_golden).max()
    print(f"\ntimestep 2 (nonzero h_prev) max abs diff: {diff2:.3e}")
    assert np.allclose(h_next2_torch, h_next2_golden, atol=1e-5), "timestep 2 mismatch"
    print("PASS: timestep 2 also matches.")


if __name__ == "__main__":
    main()
