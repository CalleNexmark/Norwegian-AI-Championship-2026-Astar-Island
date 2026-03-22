#!/usr/bin/env python3
"""
GBM (Gradient Boosted Machine) predictor for Astar Island.

Replaces the ridge regression in learned_predictor.py with
HistGradientBoostingRegressor from sklearn, which captures nonlinear
feature interactions that linear models miss.

Key improvements over ridge_v1:
  - Nonlinear survival × density interactions
  - Better on high-survival rounds (R12 +6, R14 +10 in LOOCV backtest)
  - Uses 13 features (v2): 7 base + 6 interaction terms

Build (run whenever new rounds complete, after calibrate_from_history.py):
    python gbm_predictor.py --token YOUR_JWT_TOKEN

Load & use:
    model = load_model()
    probs  = predict_grid(initial_state, survival_rate, model)   # → (H,W,6)
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import requests

BASE_URL         = "https://api.ainm.no/astar-island"
MODEL_FILE       = "gbm_model.pkl"
CALIBRATION_FILE = "calibration.json"

N_CLASSES   = 6
N_FEATURES  = 13       # v2 features (7 base + 6 interactions)
PROB_FLOOR  = 0.01

TERRAIN_GROUP = {
    0:  "empty", 11: "empty",
    1:  "settle", 2:  "port",
    4:  "forest",
    5:  "mountain", 10: "ocean",
    3:  "ruin",
}
TRAINABLE_GROUPS = {"empty", "forest", "settle", "port"}
STATIC_CODES     = {5, 10}


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — v2 (13 features)
# Must stay in sync with backtest.py compute_features_v2
# ─────────────────────────────────────────────────────────────────────────────

def _box_sum_excl(mask: np.ndarray, r: int) -> np.ndarray:
    """O(H·W) prefix-sum neighborhood count, excludes centre cell."""
    H, W = mask.shape
    padded = np.pad(mask.astype(np.float32), r, constant_values=0.0)
    cs = np.zeros((H + 2*r + 1, W + 2*r + 1), dtype=np.float64)
    cs[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    box = (cs[y+2*r+1, x+2*r+1] - cs[y, x+2*r+1]
           - cs[y+2*r+1, x] + cs[y, x])
    return (box - mask.astype(np.float64)).astype(np.float32)


def compute_feature_grid(initial_grid: list, survival_rate: float) -> np.ndarray:
    """
    Returns (H, W, 13) float32 feature array.

    Base features (0-6):
      0  n_adj_settle_r1   settlement/port neighbors within radius 1
      1  n_adj_settle_r3   settlement/port neighbors within radius 3
      2  n_adj_settle_r7   settlement/port neighbors within radius 7
      3  is_coastal        any ocean cell within radius 1
      4  n_adj_forest_r1   forest neighbors within radius 1
      5  n_adj_ocean_r3    ocean cells within radius 3
      6  survival_rate     round-level estimated 50yr settlement survival

    Interaction features (7-12):
      7  survival_rate * n_adj_settle_r3   (density matters more at high survival)
      8  survival_rate * n_adj_forest_r1   (food access matters at high survival)
      9  is_coastal * n_adj_settle_r1      (coastal settlement dynamics differ)
     10  n_adj_forest_r1 * n_adj_settle_r1 (food-to-settlement local ratio)
     11  survival_rate^2                   (nonlinear survival effect)
     12  n_adj_settle_r3 * n_adj_settle_r7 (local vs regional density)
    """
    grid = np.array(initial_grid, dtype=np.int8)
    H, W = grid.shape
    is_settle = np.isin(grid, [1, 2]).astype(np.float32)
    is_forest = (grid == 4).astype(np.float32)
    is_ocean  = (grid == 10).astype(np.float32)
    sr = np.float32(survival_rate)

    f0 = _box_sum_excl(is_settle, 1)
    f1 = _box_sum_excl(is_settle, 3)
    f2 = _box_sum_excl(is_settle, 7)
    f3 = (_box_sum_excl(is_ocean, 1) > 0).astype(np.float32)
    f4 = _box_sum_excl(is_forest, 1)
    f5 = _box_sum_excl(is_ocean, 3)
    f6 = np.full((H, W), sr, dtype=np.float32)
    # interactions
    f7  = sr * f1
    f8  = sr * f4
    f9  = f3 * f0
    f10 = f4 * f0
    f11 = np.full((H, W), sr ** 2, dtype=np.float32)
    f12 = f1 * f2

    return np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12], axis=2)


# ─────────────────────────────────────────────────────────────────────────────
# GBM training / inference
# ─────────────────────────────────────────────────────────────────────────────

def _train_group(X: np.ndarray, Y: np.ndarray) -> list:
    """
    Train one HistGradientBoostingRegressor per output class (6 total).
    Returns list of 6 fitted models.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = []
    for k in range(N_CLASSES):
        m = HistGradientBoostingRegressor(
            max_depth=4, max_iter=100, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        )
        m.fit(X, Y[:, k])
        models.append(m)
    return models


def _predict_group(X: np.ndarray, models: list) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: dict, path: str = MODEL_FILE):
    joblib.dump(model, path)
    print(f"Saved → {path}")


def load_model(path: str = MODEL_FILE) -> dict | None:
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

_STATIC_FALLBACK = {
    5:  np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95], dtype=np.float32),
    10: np.array([0.97, 0.01, 0.005, 0.005, 0.005, 0.005], dtype=np.float32),
    3:  np.array([0.20, 0.10, 0.03, 0.45, 0.15, 0.07], dtype=np.float32),
}


def predict_grid(
    initial_state: dict,
    survival_rate: float,
    model: dict,
) -> np.ndarray:
    """
    Predict H×W×6 probability distribution using the GBM model.
    Same interface as learned_predictor.predict_grid.
    Returns float32 (H, W, 6) with PROB_FLOOR applied and rows summing to 1.
    """
    initial_grid = initial_state["grid"]
    grid_arr = np.array(initial_grid, dtype=np.int8)
    H, W = grid_arr.shape

    feat_flat  = compute_feature_grid(initial_grid, survival_rate).reshape(-1, N_FEATURES)
    codes_flat = grid_arr.reshape(-1)
    probs      = np.zeros((H * W, N_CLASSES), dtype=np.float32)

    for group, gbm_models in model.items():
        if group in ("_meta",):
            continue
        mask = np.array([TERRAIN_GROUP.get(int(c), "empty") == group for c in codes_flat])
        if mask.sum() == 0:
            continue
        raw = _predict_group(feat_flat[mask], gbm_models)
        probs[mask] = raw

    # Static fallbacks for mountain/ocean/ruin
    for code, fallback in _STATIC_FALLBACK.items():
        mask = codes_flat == code
        probs[mask] = fallback

    probs = probs.reshape(H, W, N_CLASSES)
    probs = np.maximum(probs, PROB_FLOOR)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs


def blend_with_empirical(
    model_probs: np.ndarray,
    observations: dict,
    width: int,
    height: int,
    n0: float = 5.0,
) -> np.ndarray:
    """
    Bayesian blend: final = (n0/(n+n0)) * model + (n/(n+n0)) * empirical.
    Default n0=5 (backtest-validated improvement over n0=3).
    """
    ALPHA = 0.1
    probs = model_probs.copy()
    for (x, y), class_list in observations.items():
        if x >= width or y >= height or len(class_list) == 0:
            continue
        counts = np.zeros(N_CLASSES, dtype=np.float32)
        for cls in class_list:
            counts[cls] += 1.0
        n = float(len(class_list))
        empirical  = (counts + ALPHA) / (n + ALPHA * N_CLASSES)
        w_model    = n0 / (n + n0)
        w_emp      = n  / (n + n0)
        p = w_model * model_probs[y, x] + w_emp * empirical
        p = np.maximum(p, PROB_FLOOR)
        probs[y, x] = p / p.sum()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Build model from API
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save(token: str, out_file: str = MODEL_FILE):
    """
    Fetch all completed rounds' ground-truth data from /analysis,
    accumulate (feature, ground_truth) pairs, train GBM, save.
    """
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    per_round_survival: dict = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}
        print(f"Loaded per-round survival for {len(per_round_survival)} rounds")

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
            s_rate = survival_rate
            if s_rate is None:
                vals = [gt[y][x][1] + gt[y][x][2]
                        for y in range(H) for x in range(W) if ig[y][x] in (1, 2)]
                s_rate = float(np.mean(vals)) if vals else 0.30

            feat   = compute_feature_grid(ig, s_rate).reshape(-1, N_FEATURES)
            gt_arr = np.array(gt, dtype=np.float32).reshape(-1, N_CLASSES)
            codes  = np.array(ig, dtype=np.int8).reshape(-1)

            for group in TRAINABLE_GROUPS:
                mask = np.array([TERRAIN_GROUP.get(int(c), "empty") == group for c in codes])
                if mask.sum() == 0:
                    continue
                feat_by_group.setdefault(group, []).append(feat[mask])
                gt_by_group.setdefault(group,   []).append(gt_arr[mask])

            total_cells += H * W
            print(f"  seed {seed_idx}: {H}×{W}={H*W} cells  survival={s_rate:.3f}")
            time.sleep(0.10)

    print(f"\nTotal cells accumulated: {total_cells}")
    print("Training GBM models (this takes a few minutes)...")

    model = {}
    for group in TRAINABLE_GROUPS:
        if group not in feat_by_group:
            continue
        X = np.vstack(feat_by_group[group]).astype(np.float32)
        Y = np.vstack(gt_by_group[group]).astype(np.float32)
        print(f"  [{group}]  n={len(X):>7}  training 6 GBMs...")
        model[group] = _train_group(X, Y)

    save_model(model, out_file)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train GBM cell predictor from historical ground-truth data."
    )
    parser.add_argument("--token", required=True, help="JWT from app.ainm.no cookies")
    parser.add_argument("--out",   default=MODEL_FILE)
    args = parser.parse_args()
    build_and_save(args.token, args.out)


if __name__ == "__main__":
    main()
