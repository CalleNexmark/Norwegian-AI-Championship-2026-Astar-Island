"""
Data-driven per-cell predictor trained on historical ground-truth distributions.

The organiser pre-computes ground truth by running the true simulator hundreds
of times.  Via GET /analysis/{round_id}/{seed_index} we get the full 40×40×6
ground-truth probability tensor alongside the initial grid — for every completed
round.  With 13 rounds × 5 seeds × 1600 cells ≈ 104 000 labelled examples we
train a feature-rich per-terrain-code ridge regression that captures:

  • Spatial correlations  — isolated settlement vs. dense cluster behave very
                            differently; captured by neighbourhood settlement
                            counts at radii 1, 3, 7
  • Coastal dynamics      — Port formation probability depends on ocean proximity
  • Forest interaction    — adjacent forests feed settlements, slowing collapse
  • Round-level survival  — survival_rate fed as a feature makes the model
                            adapt to each round's hidden difficulty parameter

Build (one-time, run whenever new rounds complete):
    python learned_predictor.py --token YOUR_JWT_TOKEN

Load & use:
    model = load_model()
    probs  = predict_grid(initial_state, survival_rate, model)   # → (H,W,6)
"""

import argparse
import json
import os
import time

import numpy as np
import requests

BASE_URL        = "https://api.ainm.no/astar-island"
MODEL_FILE      = "learned_model.json"
CALIBRATION_FILE = "calibration.json"

N_CLASSES   = 6
N_FEATURES  = 7        # see compute_feature_grid docstring
PROB_FLOOR  = 0.01
RIDGE_ALPHA = 1.0      # L2 regularisation strength

# Map each terrain code to a named model group
TERRAIN_GROUP = {
    0:  "empty",      # Empty
    11: "empty",      # Plains  (same dynamics as empty)
    1:  "settle",     # Settlement
    2:  "port",       # Port
    4:  "forest",     # Forest
    5:  "mountain",   # static — always Mountain
    10: "ocean",      # static — always Ocean/Empty
    3:  "ruin",       # no initial Ruin cells in historical data → static fallback
}
STATIC_CODES  = {5, 10}
NO_DATA_CODES = {3}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _box_sum_excl(mask: np.ndarray, r: int) -> np.ndarray:
    """
    For every cell (y, x), sum mask over the (2r+1)×(2r+1) Moore neighbourhood,
    excluding the centre cell itself.  O(H·W) via 2-D prefix sums.
    Out-of-bounds treated as 0.
    """
    H, W = mask.shape
    padded = np.pad(mask.astype(np.float32), r, constant_values=0.0)
    # Build prefix-sum with a leading zero row+col so we can index cleanly
    cs = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float64)
    cs[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    y = np.arange(H)[:, None]   # (H, 1)
    x = np.arange(W)[None, :]   # (1, W)
    box = (cs[y + 2*r + 1, x + 2*r + 1]
           - cs[y,          x + 2*r + 1]
           - cs[y + 2*r + 1, x         ]
           + cs[y,           x         ])
    return (box - mask.astype(np.float64)).astype(np.float32)


def compute_feature_grid(initial_grid: list, survival_rate: float) -> np.ndarray:
    """
    Returns (H, W, N_FEATURES) float32 array.

    Feature index  Description
    ─────────────  ────────────────────────────────────────────────────────
    0              n_adj_settle_r1   settlements / ports in 1-cell ring (max 8)
    1              n_adj_settle_r3   settlements / ports in 3-cell ring
    2              n_adj_settle_r7   settlements / ports in 7-cell ring
    3              is_coastal        any ocean cell in 1-ring (0 or 1)
    4              n_adj_forest_r1   forest cells in 1-ring
    5              n_adj_ocean_r3    ocean cells in 3-ring
    6              survival_rate     round-level estimated 50yr settlement survival
    """
    grid = np.array(initial_grid, dtype=np.int8)
    H, W = grid.shape

    is_settle = np.isin(grid, [1, 2]).astype(np.float32)
    is_forest = (grid == 4).astype(np.float32)
    is_ocean  = (grid == 10).astype(np.float32)

    f0 = _box_sum_excl(is_settle, 1)
    f1 = _box_sum_excl(is_settle, 3)
    f2 = _box_sum_excl(is_settle, 7)
    f3 = (_box_sum_excl(is_ocean, 1) > 0).astype(np.float32)
    f4 = _box_sum_excl(is_forest, 1)
    f5 = _box_sum_excl(is_ocean, 3)
    f6 = np.full((H, W), survival_rate, dtype=np.float32)

    return np.stack([f0, f1, f2, f3, f4, f5, f6], axis=2)   # (H, W, 7)


# ---------------------------------------------------------------------------
# Ridge regression (numpy-only, no scikit-learn)
# ---------------------------------------------------------------------------

def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = RIDGE_ALPHA):
    """
    Closed-form ridge regression including bias term.
    X: (n, p)  Y: (n, k)  →  W: (p+1, k)   [last row = bias]
    """
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])   # (n, p+1)
    p  = Xb.shape[1]
    A  = (Xb.T @ Xb).astype(np.float64) + alpha * np.eye(p, dtype=np.float64)
    B  = (Xb.T @ Y.astype(np.float32)).astype(np.float64)
    return np.linalg.solve(A, B).astype(np.float32)           # (p+1, k)


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply ridge weights (W includes bias row) to X."""
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
    return Xb @ W   # (n, 6)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    feat_by_group: dict,     # group → list[np.ndarray (n, 7)]
    gt_by_group:   dict,     # group → list[np.ndarray (n, 6)]
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """Train per-group ridge regression. Returns {group: W (p+1, 6)} dict."""
    model = {}
    for group in feat_by_group:
        X = np.vstack(feat_by_group[group]).astype(np.float32)
        Y = np.vstack(gt_by_group[group]).astype(np.float32)
        W = _ridge_fit(X, Y, alpha=alpha)
        model[group] = W
        print(f"  [{group}]  n={len(X):>7}  W.shape={W.shape}")
    return model


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def save_model(model: dict, path: str = MODEL_FILE):
    serialisable = {k: v.tolist() for k, v in model.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f)
    print(f"Saved → {path}")


def load_model(path: str = MODEL_FILE) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Static fallback distributions for codes without a trained model
_STATIC_FALLBACK = {
    5:  [0.01, 0.01, 0.01, 0.01, 0.01, 0.95],   # Mountain
    10: [0.97, 0.01, 0.005, 0.005, 0.005, 0.005], # Ocean
    3:  [0.20, 0.10, 0.03, 0.45, 0.15, 0.07],   # Ruin (no training data)
}


def predict_grid(
    initial_state: dict,
    survival_rate: float,
    model: dict,
) -> np.ndarray:
    """
    Predict H×W×6 probability distribution for every cell using the learned model.

    • Per-group ridge regression for trained terrain codes
    • Static fallback for mountain, ocean, ruin

    Returns float32 (H, W, 6) with PROB_FLOOR applied and rows summing to 1.
    """
    initial_grid = initial_state["grid"]
    grid_arr = np.array(initial_grid, dtype=np.int8)
    H, W = grid_arr.shape

    feat_flat  = compute_feature_grid(initial_grid, survival_rate).reshape(-1, N_FEATURES)
    codes_flat = grid_arr.reshape(-1)
    probs      = np.zeros((H * W, N_CLASSES), dtype=np.float32)

    # Apply learned model per terrain group
    for group, W_weights in model.items():
        mask = np.array([TERRAIN_GROUP.get(int(c), "empty") == group for c in codes_flat])
        if mask.sum() == 0:
            continue
        raw = _ridge_predict(feat_flat[mask], W_weights)   # (n, 6)
        probs[mask] = raw.astype(np.float32)

    # Fill any cells whose group has no model entry
    for code, fallback in _STATIC_FALLBACK.items():
        mask = codes_flat == code
        if probs[mask].sum() == 0:   # not already filled
            probs[mask] = np.array(fallback, dtype=np.float32)

    probs = probs.reshape(H, W, N_CLASSES)
    probs = np.maximum(probs, PROB_FLOOR)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs   # (H, W, 6)


def blend_with_empirical(
    model_probs: np.ndarray,
    observations: dict,
    width: int,
    height: int,
    n0: float = 3.0,
) -> np.ndarray:
    """
    Bayesian blend of learned model (prior) + empirical observations.

    For n observations at a cell:
        final = (n0 / (n + n0)) * model_probs  +  (n / (n + n0)) * empirical

    n0=3 means 1 observation → 75% model / 25% empirical,
              3 observations → 50% / 50%,
              ∞ observations → 100% empirical.

    This avoids the overconfidence of pure Laplace empirical with n=1.
    """
    ALPHA = 0.1   # Laplace smoothing for empirical
    probs = model_probs.copy()

    for (x, y), class_list in observations.items():
        if x >= width or y >= height or len(class_list) == 0:
            continue
        counts = np.zeros(N_CLASSES, dtype=np.float32)
        for cls in class_list:
            counts[cls] += 1.0
        n = float(len(class_list))
        empirical = (counts + ALPHA) / (n + ALPHA * N_CLASSES)
        w_model   = n0 / (n + n0)
        w_empirical = n / (n + n0)
        p = w_model * model_probs[y, x] + w_empirical * empirical
        p = np.maximum(p, PROB_FLOOR)
        probs[y, x] = p / p.sum()

    return probs


# ---------------------------------------------------------------------------
# Build model from API
# ---------------------------------------------------------------------------

def _make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def build_and_save(token: str, out_file: str = MODEL_FILE):
    """
    Fetch all completed rounds' ground-truth data from /analysis,
    accumulate (feature, ground_truth) pairs, train model, save.
    """
    session = _make_session(token)

    # Load per-round survival rates (needed as a feature during training)
    per_round_survival: dict = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}
        print(f"Loaded per-round survival for {len(per_round_survival)} rounds from {CALIBRATION_FILE}")

    rounds = session.get(f"{BASE_URL}/rounds", timeout=15).json()
    completed = sorted([r for r in rounds if r["status"] == "completed"],
                       key=lambda r: r["round_number"])
    print(f"Found {len(completed)} completed rounds\n")

    feat_by_group: dict = {}
    gt_by_group:   dict = {}
    total_cells = 0

    for r in completed:
        rnum = r["round_number"]
        rid  = r["id"]
        detail = session.get(f"{BASE_URL}/rounds/{rid}", timeout=15).json()
        n_seeds = detail.get("seeds_count", 5)

        survival_rate = per_round_survival.get(rnum)
        surv_str = f"{survival_rate:.3f}" if survival_rate is not None else "unknown"
        print(f"Round #{rnum}  survival={surv_str}")

        for seed_idx in range(n_seeds):
            resp = session.get(f"{BASE_URL}/analysis/{rid}/{seed_idx}", timeout=30)
            if resp.status_code != 200:
                print(f"  seed {seed_idx}: HTTP {resp.status_code} — skipping")
                continue
            data = resp.json()
            ig = data.get("initial_grid")
            gt = data.get("ground_truth")
            if ig is None or gt is None:
                continue

            H, W = len(ig), len(ig[0]) if ig else 0

            # Estimate survival from this seed if not in calibration
            s_rate = survival_rate
            if s_rate is None:
                vals = [gt[y][x][1] + gt[y][x][2]
                        for y in range(H) for x in range(W) if ig[y][x] in (1, 2)]
                s_rate = float(np.mean(vals)) if vals else 0.30

            feat  = compute_feature_grid(ig, s_rate).reshape(-1, N_FEATURES)  # (H*W, 7)
            gt_arr = np.array(gt, dtype=np.float32).reshape(-1, N_CLASSES)    # (H*W, 6)
            codes  = np.array(ig, dtype=np.int8).reshape(-1)                  # (H*W,)

            for group in set(TERRAIN_GROUP.values()):
                if group in ("mountain", "ocean", "ruin"):
                    continue   # static / no training data
                mask = np.array([TERRAIN_GROUP.get(int(c), "empty") == group for c in codes])
                if mask.sum() == 0:
                    continue
                feat_by_group.setdefault(group, []).append(feat[mask])
                gt_by_group.setdefault(group,   []).append(gt_arr[mask])

            total_cells += H * W
            print(f"  seed {seed_idx}: {H}×{W}={H*W} cells  survival={s_rate:.3f}")
            time.sleep(0.10)

    print(f"\nTotal cells accumulated: {total_cells}")
    print("Training models...")
    model = train_model(feat_by_group, gt_by_group)
    save_model(model, out_file)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train learned cell predictor from historical ground-truth data."
    )
    parser.add_argument("--token", required=True, help="JWT from app.ainm.no cookies")
    parser.add_argument("--out",   default=MODEL_FILE)
    args = parser.parse_args()
    build_and_save(args.token, args.out)


if __name__ == "__main__":
    main()
