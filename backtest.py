#!/usr/bin/env python3
"""
Backtesting harness for Astar Island prediction models.

Scores prediction models against ALL historical ground truth before any change
goes live. Every recommendation must show improvement here first.

Ground truth is fetched once from the API and cached locally.

Usage:
    # Step 1 — baseline scores for current model on all rounds
    python backtest.py --token TOKEN

    # Step 2 — sweep n0 blending values (uses actual observations from R14/R15)
    python backtest.py --token TOKEN --n0-sweep

    # Step 3 — compare ridge_v1 vs ridge_v2 (interaction features)
    python backtest.py --token TOKEN --model ridge_v2 --loocv

    # Step 4 — compare ridge vs GBM
    python backtest.py --token TOKEN --model gbm --loocv

    # General: compare two models side-by-side
    python backtest.py --token TOKEN --compare ridge_v1 ridge_v2 --loocv

    # Single model with custom n0
    python backtest.py --token TOKEN --model ridge_v1 --n0 1.0
"""

import argparse
import json
import os
import time

import numpy as np
import requests

BASE_URL         = "https://api.ainm.no/astar-island"
GT_CACHE_FILE    = "ground_truth_cache.json"
CALIBRATION_FILE = "calibration.json"
OBS_FILE         = "observations_history.json"
MODEL_FILE       = "learned_model.json"

N_CLASSES  = 6
PROB_FLOOR = 0.01
RIDGE_ALPHA = 1.0
STATIC_CODES = {5, 10}   # Mountain, Ocean — excluded from scoring

TERRAIN_GROUP = {
    0: "empty", 11: "empty",
    1: "settle", 2: "port",
    4: "forest",
    5: "mountain", 10: "ocean",
    3: "ruin",
}
TRAINABLE_GROUPS = {"empty", "forest", "settle", "port"}


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth cache
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ground_truth(token: str, force_refresh: bool = False) -> dict:
    """
    Fetch /analysis for every completed round+seed, cache to GT_CACHE_FILE.
    Cache format: {round_id: {round_number, survival, seeds: {seed_idx: {initial_grid, ground_truth}}}}
    Returns the loaded cache.
    """
    if os.path.exists(GT_CACHE_FILE) and not force_refresh:
        with open(GT_CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Loaded ground truth cache: {len(cache)} rounds from {GT_CACHE_FILE}")
        return cache

    print("Fetching ground truth from API (this may take a minute)...")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    # Load per-round survival from calibration.json
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    rounds = session.get(f"{BASE_URL}/rounds", timeout=15).json()
    completed = sorted([r for r in rounds if r["status"] == "completed"],
                       key=lambda r: r["round_number"])

    cache = {}
    for r in completed:
        rnum = r["round_number"]
        rid  = r["id"]
        detail = session.get(f"{BASE_URL}/rounds/{rid}", timeout=15).json()
        n_seeds = detail.get("seeds_count", 5)
        survival = per_round_survival.get(rnum)

        surv_str = f"{survival:.3f}" if survival is not None else "?"
        print(f"  R{rnum} (survival={surv_str})...", end=" ", flush=True)
        seeds = {}
        for seed_idx in range(n_seeds):
            resp = session.get(f"{BASE_URL}/analysis/{rid}/{seed_idx}", timeout=30)
            if resp.status_code != 200:
                print(f"seed{seed_idx}:HTTP{resp.status_code} ", end="")
                continue
            data = resp.json()
            ig = data.get("initial_grid")
            gt = data.get("ground_truth")
            if ig is None or gt is None:
                continue
            seeds[str(seed_idx)] = {"initial_grid": ig, "ground_truth": gt}
            time.sleep(0.1)
        print(f"{len(seeds)} seeds OK")

        cache[rid] = {
            "round_number": rnum,
            "survival": survival,
            "seeds": seeds,
        }

    with open(GT_CACHE_FILE, "w") as f:
        json.dump(cache, f)
    print(f"Saved ground truth cache → {GT_CACHE_FILE}")
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Exact competition scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_prediction(pred: np.ndarray, ground_truth: np.ndarray,
                     initial_grid: np.ndarray) -> float:
    """
    score = 100 * exp(-3 * entropy_weighted_KL)

    Only dynamic cells (initial terrain not in {5=Mountain, 10=Ocean}) count.
    KL  = KL(ground_truth || prediction) per cell
    H   = entropy of ground_truth per cell
    weighted_KL = sum(H * KL) / sum(H)

    Returns 0.0 if all dynamic cells have zero entropy (degenerate case).
    """
    pred = np.array(pred, dtype=np.float64)
    gt   = np.array(ground_truth, dtype=np.float64)
    grid = np.array(initial_grid, dtype=np.int32)

    H_map, W_map = grid.shape
    dynamic = np.array([[grid[y, x] not in STATIC_CODES
                          for x in range(W_map)] for y in range(H_map)])  # (H, W) bool

    # Clip to avoid log(0); gt==0 terms handled by where()
    pred_safe = np.clip(pred, 1e-15, 1.0)
    gt_safe   = np.clip(gt,   1e-15, 1.0)

    # Per-cell KL(gt||pred) = sum_k gt[k] * log(gt[k]/pred[k])
    # When gt[k]==0, contribution is 0 (0*log(0/pred)=0)
    kl = np.where(gt > 1e-15,
                  gt_safe * np.log(gt_safe / pred_safe),
                  0.0).sum(axis=2)   # (H, W)

    # Per-cell entropy of ground truth: H = -sum_k gt[k]*log(gt[k])
    entropy = -np.where(gt > 1e-15, gt_safe * np.log(gt_safe), 0.0).sum(axis=2)

    kl_dyn      = kl[dynamic]
    entropy_dyn = entropy[dynamic]

    total_entropy = entropy_dyn.sum()
    if total_entropy < 1e-12:
        return 100.0   # perfectly deterministic ground truth → trivial

    weighted_kl = np.dot(entropy_dyn, kl_dyn) / total_entropy
    return float(100.0 * np.exp(-3.0 * weighted_kl))


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — v1 (current) and v2 (with interactions)
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


def compute_features_v1(initial_grid: list, survival_rate: float) -> np.ndarray:
    """
    7 features: n_adj_settle_r1, r3, r7, is_coastal, n_adj_forest_r1,
                n_adj_ocean_r3, survival_rate
    Returns (H, W, 7) float32.
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
    return np.stack([f0, f1, f2, f3, f4, f5, f6], axis=2)


def compute_features_v2(initial_grid: list, survival_rate: float) -> np.ndarray:
    """
    13 features: 7 base + 6 interaction/new features.

    New features (indices 7-12):
      7  survival_rate * n_adj_settle_r3      (density matters more when survival high)
      8  survival_rate * n_adj_forest_r1      (food access matters more when survival high)
      9  is_coastal * n_adj_settle_r1         (coastal settlements behave differently)
     10  n_adj_forest_r1 * n_adj_settle_r1    (food-to-settlement local ratio)
     11  survival_rate^2                      (nonlinear survival effect)
     12  n_adj_settle_r3 * n_adj_settle_r7    (local vs regional density interaction)

    Returns (H, W, 13) float32.
    """
    base = compute_features_v1(initial_grid, survival_rate)   # (H, W, 7)
    H, W = base.shape[:2]
    sr = np.float32(survival_rate)

    f7  = sr * base[:, :, 1]                  # sr * r3
    f8  = sr * base[:, :, 4]                  # sr * forest_r1
    f9  = base[:, :, 3] * base[:, :, 0]       # coastal * settle_r1
    f10 = base[:, :, 4] * base[:, :, 0]       # forest_r1 * settle_r1
    f11 = np.full((H, W), sr ** 2, dtype=np.float32)
    f12 = base[:, :, 1] * base[:, :, 2]       # settle_r3 * settle_r7

    return np.concatenate([base,
                           np.stack([f7, f8, f9, f10, f11, f12], axis=2)], axis=2)


def compute_features_v3(initial_grid: list, survival_rate: float) -> np.ndarray:
    """
    15 features: 13 from v2 + 2 new distance/isolation features.

    New features (indices 13-14):
     13  dist_to_nearest_settle   Euclidean distance to nearest settlement/port cell
                                  (0.0 for settlement cells themselves; capped at 40)
     14  settle_isolation         n_settle_r3 / (n_settle_r1 + 1)
                                  High value = surrounded at medium range but isolated locally
                                  (captures forest refugia inside large settlement zones)

    Returns (H, W, 15) float32.
    """
    base = compute_features_v2(initial_grid, survival_rate)   # (H, W, 13)
    H, W = base.shape[:2]
    grid = np.array(initial_grid, dtype=np.int8)
    is_settle = np.isin(grid, [1, 2]).astype(bool)

    # --- f13: distance transform to nearest settlement cell ---
    try:
        from scipy.ndimage import distance_transform_edt
        # edt gives distance to nearest True cell; we want distance to nearest settle
        # If cell IS a settlement, distance = 0
        dist = distance_transform_edt(~is_settle).astype(np.float32)
    except ImportError:
        # Fallback: approximate via brute-force (slow but correct)
        ys_s, xs_s = np.where(is_settle)
        dist = np.full((H, W), float(H + W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                if ys_s.size > 0:
                    d = np.min(np.hypot(xs_s - x, ys_s - y))
                    dist[y, x] = float(d)
    dist = np.clip(dist, 0.0, 40.0)   # cap at map diagonal-ish

    # --- f14: settlement isolation index ---
    # base[:,:,1] = settle_r3, base[:,:,0] = settle_r1
    f14 = base[:, :, 1] / (base[:, :, 0] + 1.0)   # r3 / (r1+1)

    return np.concatenate([base,
                           np.stack([dist, f14], axis=2)], axis=2)


FEATURE_FN = {
    "ridge_v1": compute_features_v1,
    "ridge_v2": compute_features_v2,
    "gbm":      compute_features_v2,   # GBM uses v2 features
    "gbm_v3":   compute_features_v3,   # GBM with distance + isolation features
}


# ─────────────────────────────────────────────────────────────────────────────
# Ridge regression (numpy-only)
# ─────────────────────────────────────────────────────────────────────────────

def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """Closed-form ridge with bias. Returns W (p+1, k)."""
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    p  = Xb.shape[1]
    A  = (Xb.T @ Xb).astype(np.float64) + alpha * np.eye(p, dtype=np.float64)
    B  = (Xb.T @ Y.astype(np.float32)).astype(np.float64)
    return np.linalg.solve(A, B).astype(np.float32)


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
    return Xb @ W


# ─────────────────────────────────────────────────────────────────────────────
# GBM model (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_gbm():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor
    except ImportError:
        return None


# Mutable GBM hyperparameters — overridden by --gbm-depth / --gbm-iter / --gbm-lr flags
_GBM_PARAMS: dict = {"max_depth": 4, "max_iter": 100, "learning_rate": 0.1}


def _gbm_fit(X: np.ndarray, Y: np.ndarray) -> list:
    """
    Train one HistGradientBoostingRegressor per output class.
    Hyperparameters taken from _GBM_PARAMS (overridden by CLI flags).
    Returns list of 6 fitted models.
    """
    HGBR = _try_import_gbm()
    if HGBR is None:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    models = []
    for k in range(N_CLASSES):
        m = HGBR(max_depth=_GBM_PARAMS["max_depth"],
                  max_iter=_GBM_PARAMS["max_iter"],
                  learning_rate=_GBM_PARAMS["learning_rate"],
                  random_state=42)
        m.fit(X, Y[:, k])
        models.append(m)
    return models


def _gbm_predict(X: np.ndarray, models: list) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model training (for LOOCV and model comparison)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(gt_cache: dict, model_name: str,
                exclude_round: int | None = None) -> dict:
    """
    Train a model on all rounds in gt_cache, optionally excluding one round
    (for leave-one-out cross-validation).

    Returns model dict: {group: W} for ridge, {group: [gbm_models]} for GBM.
    """
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    _model_key = model_name if model_name not in ("gbm", "gbm_v3") else model_name
    feat_fn = FEATURE_FN[_model_key]

    feat_by_group: dict = {}
    gt_by_group:   dict = {}

    for rid, rdata in gt_cache.items():
        rnum = rdata["round_number"]
        if exclude_round is not None and rnum == exclude_round:
            continue
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        for seed_str, sdata in rdata["seeds"].items():
            ig = sdata["initial_grid"]
            gt = sdata["ground_truth"]
            H, W = len(ig), len(ig[0])

            feat  = feat_fn(ig, survival).reshape(-1, feat_fn(ig, survival).shape[2])
            gt_arr = np.array(gt, dtype=np.float32).reshape(-1, N_CLASSES)
            codes  = np.array(ig, dtype=np.int8).reshape(-1)

            for group in TRAINABLE_GROUPS:
                mask = np.array([TERRAIN_GROUP.get(int(c), "empty") == group
                                  for c in codes])
                if mask.sum() == 0:
                    continue
                feat_by_group.setdefault(group, []).append(feat[mask])
                gt_by_group.setdefault(group,   []).append(gt_arr[mask])

    # Stack and train
    model = {}
    for group in TRAINABLE_GROUPS:
        if group not in feat_by_group:
            continue
        X = np.vstack(feat_by_group[group]).astype(np.float32)
        Y = np.vstack(gt_by_group[group]).astype(np.float32)
        if model_name in ("ridge_v1", "ridge_v2"):
            model[group] = _ridge_fit(X, Y)
        elif model_name in ("gbm", "gbm_v3"):
            model[group] = _gbm_fit(X, Y)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

_STATIC_FALLBACK = {
    5:  np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95], dtype=np.float32),
    10: np.array([0.97, 0.01, 0.005, 0.005, 0.005, 0.005], dtype=np.float32),
    3:  np.array([0.20, 0.10, 0.03, 0.45, 0.15, 0.07], dtype=np.float32),
}


def predict_grid(initial_grid: list, survival_rate: float,
                 model: dict, model_name: str) -> np.ndarray:
    """Returns (H, W, 6) float32 with PROB_FLOOR applied and rows summing to 1."""
    grid_arr  = np.array(initial_grid, dtype=np.int8)
    H, W      = grid_arr.shape
    feat_fn   = FEATURE_FN.get(model_name, FEATURE_FN["gbm"])

    feat_2d   = feat_fn(initial_grid, survival_rate)        # (H, W, n_feat)
    n_feat    = feat_2d.shape[2]
    feat_flat = feat_2d.reshape(-1, n_feat)
    codes     = grid_arr.reshape(-1)
    probs     = np.zeros((H * W, N_CLASSES), dtype=np.float32)

    for group, W_or_models in model.items():
        mask = np.array([TERRAIN_GROUP.get(int(c), "empty") == group for c in codes])
        if mask.sum() == 0:
            continue
        if model_name in ("ridge_v1", "ridge_v2"):
            raw = _ridge_predict(feat_flat[mask], W_or_models)
        else:  # gbm
            raw = _gbm_predict(feat_flat[mask], W_or_models)
        probs[mask] = raw.astype(np.float32)

    # Static fallbacks for mountain/ocean/ruin
    for code, fallback in _STATIC_FALLBACK.items():
        mask = codes == code
        probs[mask] = fallback

    probs = probs.reshape(H, W, N_CLASSES)
    probs = np.maximum(probs, PROB_FLOOR)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulator prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_grid_mc(initial_grid: list, survival_rate: float,
                    n_runs: int = 300) -> np.ndarray:
    """
    Run Monte Carlo simulation, return (H, W, 6) float32.

    Loads calibrated structural params from calibration.json and overrides
    p_annual_survive / p_port_survive to match the round's known survival rate.
    """
    from simulator import run_monte_carlo, DEFAULT_PARAMS

    params = DEFAULT_PARAMS.copy()
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        params.update(cal.get("params", {}))

    # Override survival to match this round's calibrated rate
    p_survive = float(np.clip(survival_rate ** (1.0 / 50), 0.80, 0.999))
    params["p_annual_survive"] = p_survive
    params["p_port_survive"]   = min(p_survive + 0.003, 0.999)

    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0
    return run_monte_carlo({"grid": initial_grid}, W, H, params, n_runs=n_runs)


def predict_grid_mc_ensemble(initial_grid: list, survival_rate: float,
                              model: dict, model_name: str,
                              n_runs: int = 300,
                              mc_weight: float = 0.5) -> np.ndarray:
    """
    Ensemble: mc_weight * MC + (1 - mc_weight) * GBM, then floor + renormalise.
    """
    mc_probs  = predict_grid_mc(initial_grid, survival_rate, n_runs=n_runs)
    gbm_probs = predict_grid(initial_grid, survival_rate, model, model_name)
    blended   = mc_weight * mc_probs + (1.0 - mc_weight) * gbm_probs
    blended   = np.maximum(blended, PROB_FLOOR)
    blended  /= blended.sum(axis=2, keepdims=True)
    return blended


def blend_with_empirical(model_probs: np.ndarray, observations: dict,
                          width: int, height: int, n0: float = 3.0) -> np.ndarray:
    """Bayesian blend: final = (n0/(n+n0))*model + (n/(n+n0))*empirical."""
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


def load_observations() -> dict:
    """Load cached observations keyed by round_id → {seed_idx: {(x,y): [cls,...]}}"""
    if not os.path.exists(OBS_FILE):
        return {}
    with open(OBS_FILE) as f:
        raw = json.load(f)
    result = {}
    for rid, seeds in raw.items():
        result[rid] = {
            int(s): {tuple(map(int, k.split(","))): v for k, v in obs.items()}
            for s, obs in seeds.items()
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing transforms (temperature scaling, spatial smooth, global mult)
# Applied after model blend, before or after observation blending as noted.
# ─────────────────────────────────────────────────────────────────────────────

def apply_temperature_scaling(probs: np.ndarray, T_low: float = 0.5, T_high: float = 2.0,
                               initial_grid: list | None = None) -> np.ndarray:
    """
    Entropy-weighted temperature scaling per cell.
    High-entropy (uncertain) cells → T_high (flatten toward uniform).
    Low-entropy (confident) cells → T_low (sharpen).
    Applied BEFORE Bayesian blending with observations.
    T=1.0 is a no-op.  T<1 sharpens, T>1 flattens.
    """
    p = probs.astype(np.float64)
    p_safe = np.clip(p, 1e-15, 1.0)

    # Per-cell entropy (H, W)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=2)
    norm_ent = np.clip(entropy / np.log(N_CLASSES), 0.0, 1.0)  # 0..1

    # Temperature per cell broadcast to (H, W, 1)
    T = (T_low + (T_high - T_low) * norm_ent)[:, :, np.newaxis]

    scaled = np.power(p_safe, 1.0 / T)
    scaled = scaled / scaled.sum(axis=2, keepdims=True)

    # Restore static cells unchanged
    if initial_grid is not None:
        grid   = np.array(initial_grid, dtype=np.int8)
        static = np.isin(grid, list(STATIC_CODES))
        scaled[static] = probs[static]

    result = np.maximum(scaled, PROB_FLOOR).astype(np.float32)
    result /= result.sum(axis=2, keepdims=True)
    return result


def apply_spatial_smooth(probs: np.ndarray, sigma: float = 0.15,
                          initial_grid: list | None = None) -> np.ndarray:
    """
    Gaussian blur per class channel on dynamic cells, static cells restored unchanged.
    Applied BEFORE Bayesian blending with observations.
    """
    from scipy.ndimage import gaussian_filter

    H, W = probs.shape[:2]
    dynamic_mask = np.ones((H, W), dtype=bool)
    if initial_grid is not None:
        grid = np.array(initial_grid, dtype=np.int8)
        dynamic_mask = ~np.isin(grid, list(STATIC_CODES))

    smoothed = probs.astype(np.float32).copy()
    for c in range(N_CLASSES):
        channel = probs[:, :, c].astype(np.float64)
        channel[~dynamic_mask] = 0.0
        smoothed[:, :, c] = gaussian_filter(channel, sigma=sigma).astype(np.float32)

    # Restore static cells and renormalize dynamic cells
    smoothed[~dynamic_mask] = probs[~dynamic_mask]
    dyn_total = smoothed[dynamic_mask].sum(axis=1, keepdims=True)
    smoothed[dynamic_mask] /= np.maximum(dyn_total, 1e-12)

    result = np.maximum(smoothed, PROB_FLOOR).astype(np.float32)
    result /= result.sum(axis=2, keepdims=True)
    return result


def apply_global_multiplier(probs: np.ndarray, observations: dict,
                              initial_grid: list,
                              clip_low: float = 0.5, clip_high: float = 2.0) -> np.ndarray:
    """
    Compute observed/expected frequency ratio per class from observed cells,
    then apply as a global correction to ALL cells.
    Applied AFTER Bayesian blending with observations.
    """
    if not observations:
        return probs

    H, W = probs.shape[:2]
    expected_counts = np.zeros(N_CLASSES, dtype=np.float64)
    observed_counts = np.zeros(N_CLASSES, dtype=np.float64)

    for (x, y), obs_list in observations.items():
        if x >= W or y >= H:
            continue
        expected_counts += probs[y, x].astype(np.float64)
        for cls in obs_list:
            if 0 <= cls < N_CLASSES:
                observed_counts[cls] += 1.0

    if observed_counts.sum() < 1.0 or expected_counts.sum() < 1e-12:
        return probs

    expected_freq = expected_counts / expected_counts.sum()
    observed_freq  = observed_counts / observed_counts.sum()
    ratio = np.clip(observed_freq / (expected_freq + 1e-6), clip_low, clip_high)

    grid    = np.array(initial_grid, dtype=np.int8)
    dynamic = ~np.isin(grid, list(STATIC_CODES))

    corrected = probs.astype(np.float64).copy()
    corrected[dynamic] = corrected[dynamic] * ratio[np.newaxis, :]

    # Renormalize dynamic cells
    dyn_total = corrected[dynamic].sum(axis=1, keepdims=True)
    corrected[dynamic] /= np.maximum(dyn_total, 1e-12)

    result = np.maximum(corrected, PROB_FLOOR).astype(np.float32)
    result /= result.sum(axis=2, keepdims=True)
    return result


def apply_static_floor(probs: np.ndarray, initial_grid: list,
                        static_floor: float = 0.005) -> np.ndarray:
    """
    Use a lower probability floor for static cells (ocean/mountain), regular floor for rest.
    Applied as the final step before scoring.
    """
    grid   = np.array(initial_grid, dtype=np.int8)
    static = np.isin(grid, list(STATIC_CODES))

    result = np.maximum(probs, PROB_FLOOR)            # regular floor for dynamic cells
    result[static] = np.maximum(probs[static], static_floor)  # lower floor for static

    result /= result.sum(axis=2, keepdims=True)
    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical empirical prior
# 4-level lookup: Global → per Terrain → per Terrain×SurvBucket
#                → per Terrain×SurvBucket×DistBand
# ─────────────────────────────────────────────────────────────────────────────

def _surv_bucket(survival: float) -> int:
    """0=catastrophic(<15%), 1=hard-medium(15-35%), 2=normal(35-55%), 3=high(55%+)"""
    if survival < 0.15: return 0
    if survival < 0.35: return 1
    if survival < 0.55: return 2
    return 3


def _dist_band_arr(dist: np.ndarray) -> np.ndarray:
    """Vectorized distance-to-nearest-settlement → band 0-3."""
    db = np.full(dist.shape, 3, dtype=np.int8)
    db[dist < 8.0] = 2
    db[dist < 4.0] = 1
    db[dist < 1.5] = 0
    return db


def build_hierarchical_prior(gt_cache: dict,
                               exclude_round: int | None = None) -> dict:
    """
    Build lookup tables at 4 granularity levels from ground truth.

    Returns dict mapping key → (mean_distribution, count):
      ("G",)                   — global
      ("T",  group)            — per terrain group
      ("TS", group, sb)        — per terrain × survival bucket
      ("TSD",group, sb, db)    — per terrain × survival × distance band
    """
    from collections import defaultdict
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("scipy required for hierarchical prior")

    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    accum  = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
    counts = defaultdict(int)

    for rid, rdata in gt_cache.items():
        rnum     = rdata["round_number"]
        if exclude_round is not None and rnum == exclude_round:
            continue
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)
        sb = _surv_bucket(survival)

        for seed_str, sdata in rdata["seeds"].items():
            ig  = np.array(sdata["initial_grid"], dtype=np.int8)
            gt  = np.array(sdata["ground_truth"],  dtype=np.float32)
            H, W = ig.shape

            is_settle = np.isin(ig, [1, 2]).astype(bool)
            dist  = distance_transform_edt(~is_settle).astype(np.float32)
            db_arr = _dist_band_arr(dist)

            gt_flat = gt.reshape(-1, N_CLASSES)
            ig_flat = ig.reshape(-1)
            db_flat = db_arr.reshape(-1)

            # Global accumulation (all dynamic cells)
            dyn = ~np.isin(ig_flat, list(STATIC_CODES))
            accum[("G",)] += gt_flat[dyn].sum(axis=0)
            counts[("G",)] += int(dyn.sum())

            # Per trainable terrain group
            for group in TRAINABLE_GROUPS:
                group_codes = [c for c, g in TERRAIN_GROUP.items() if g == group]
                gm = np.isin(ig_flat, group_codes)
                if not gm.any():
                    continue
                gt_g = gt_flat[gm]
                db_g = db_flat[gm]

                accum[("T", group)]     += gt_g.sum(axis=0)
                counts[("T", group)]    += len(gt_g)
                accum[("TS", group, sb)] += gt_g.sum(axis=0)
                counts[("TS", group, sb)] += len(gt_g)

                for db in range(4):
                    dm = db_g == db
                    if not dm.any():
                        continue
                    accum[("TSD", group, sb, db)]  += gt_g[dm].sum(axis=0)
                    counts[("TSD", group, sb, db)] += int(dm.sum())

    priors: dict = {}
    for key, total in accum.items():
        c = counts[key]
        if c > 0:
            p = total / c
            p = np.maximum(p, PROB_FLOOR)
            p /= p.sum()
            priors[key] = (p.astype(np.float32), c)
    return priors


def predict_grid_hier(initial_grid: list, survival_rate: float,
                       prior: dict, min_count: int = 100) -> np.ndarray:
    """
    Returns (H, W, 6) probs using finest hierarchical prior available per cell.
    Falls back to coarser levels when fine-grained count < min_count.
    """
    from scipy.ndimage import distance_transform_edt

    grid = np.array(initial_grid, dtype=np.int8)
    H, W = grid.shape

    is_settle = np.isin(grid, [1, 2]).astype(bool)
    dist   = distance_transform_edt(~is_settle).astype(np.float32)
    db_arr = _dist_band_arr(dist)
    sb = _surv_bucket(survival_rate)

    probs = np.zeros((H, W, N_CLASSES), dtype=np.float32)

    # Static cells
    for code, fallback in _STATIC_FALLBACK.items():
        probs[grid == code] = fallback

    ig_flat = grid.reshape(-1)
    db_flat = db_arr.reshape(-1)

    # Trainable terrain groups
    for group in TRAINABLE_GROUPS:
        group_codes = [c for c, g in TERRAIN_GROUP.items() if g == group]
        gm = np.isin(ig_flat, group_codes)
        if not gm.any():
            continue

        for db in range(4):
            cell_mask = gm & (db_flat == db)
            if not cell_mask.any():
                continue

            p = None
            for key in [("TSD", group, sb, db),
                        ("TS",  group, sb),
                        ("T",   group),
                        ("G",)]:
                entry = prior.get(key)
                if entry is not None and entry[1] >= min_count:
                    p = entry[0]
                    break

            if p is None:
                p = np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES
            probs.reshape(-1, N_CLASSES)[cell_mask] = p

    # Ruin cells (no training data — use static fallback)
    ruin_mask = ig_flat == 3
    if ruin_mask.any():
        probs.reshape(-1, N_CLASSES)[ruin_mask] = _STATIC_FALLBACK.get(
            3, np.ones(N_CLASSES) / N_CLASSES)

    probs = np.maximum(probs, PROB_FLOOR)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs


def run_hier_blend_sweep(gt_cache: dict, observations: dict,
                          n0: float, mc_weight: float) -> None:
    """
    Sweep hier_weight for blending hierarchical prior with gbm_ridge ensemble.
    Pre-computes LOOCV models once; also applies best post-processing (global multiplier).

    hier_weight=0.0 → pure gbm_ridge (baseline)
    hier_weight=1.0 → pure hierarchical prior
    """
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    sorted_rounds = sorted(gt_cache.items(), key=lambda kv: kv[1]["round_number"])

    print("  Pre-computing LOOCV models (GBM+ridge + hierarchical prior)...")
    round_cache = []

    for rid, rdata in sorted_rounds:
        rnum     = rdata["round_number"]
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        print(f"    R{rnum}: training GBM+ridge...", end=" ", flush=True)
        gbm_ridge_model = (train_model(gt_cache, "gbm",      exclude_round=rnum),
                           train_model(gt_cache, "ridge_v1", exclude_round=rnum))
        print("building hier prior...", end=" ", flush=True)
        hier_prior = build_hierarchical_prior(gt_cache, exclude_round=rnum)
        print("predictions...", end=" ", flush=True)

        seeds = []
        for seed_str, sdata in rdata["seeds"].items():
            seed_idx = int(seed_str)
            ig   = sdata["initial_grid"]
            gt   = np.array(sdata["ground_truth"], dtype=np.float32)
            H, W = len(ig), len(ig[0])

            # GBM+ridge raw probs
            gbm_m, ridge_m = gbm_ridge_model
            p_gbm   = predict_grid(ig, survival, gbm_m,   "gbm")
            p_ridge = predict_grid(ig, survival, ridge_m, "ridge_v1")
            blended = mc_weight * p_gbm + (1.0 - mc_weight) * p_ridge
            blended = np.maximum(blended, PROB_FLOOR)
            p_gbr   = blended / blended.sum(axis=2, keepdims=True)

            # Hierarchical probs
            p_hier = predict_grid_hier(ig, survival, hier_prior)

            obs = None
            if observations and rid in observations and seed_idx in observations[rid]:
                obs = observations[rid][seed_idx]

            seeds.append((ig, gt, H, W, obs, p_gbr, p_hier))

        print(f"done ({len(seeds)} seeds)")
        round_cache.append((rnum, survival, seeds))

    print()

    hier_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    baseline_avg = None

    print(f"  {'hier_weight':>12} {'LOOCV Avg':>10} {'vs baseline':>12}")
    print("  " + "─" * 40)

    for hw in hier_weights:
        round_avgs = []
        for rnum, survival, seeds in round_cache:
            seed_scores = []
            for (ig, gt, H, W, obs, p_gbr, p_hier) in seeds:
                # Blend
                blended = hw * p_hier + (1.0 - hw) * p_gbr
                blended = np.maximum(blended, PROB_FLOOR)
                probs   = blended / blended.sum(axis=2, keepdims=True)

                # Apply spatial smooth + obs blend + global multiplier
                probs = apply_spatial_smooth(probs, sigma=0.3, initial_grid=ig)
                if obs is not None:
                    probs = blend_with_empirical(probs, obs, W, H, n0=n0)
                    probs = apply_global_multiplier(probs, obs, ig,
                                                     clip_low=0.5, clip_high=2.0)
                seed_scores.append(score_prediction(probs, gt, np.array(ig)))
            round_avgs.append(float(np.mean(seed_scores)))

        avg = float(np.mean(round_avgs))

        if baseline_avg is None:
            baseline_avg = avg
            delta_str = "(baseline)"
        else:
            delta = avg - baseline_avg
            flag  = " ▲" if delta > 0.2 else (" ▼" if delta < -0.2 else " ~")
            delta_str = f"{delta:+.2f}{flag}"
        print(f"  {hw:>12.1f} {avg:10.2f} {delta_str:>12}")

    print()


def _get_raw_probs(ig: list, survival: float, model, model_name: str,
                   mc_weight: float = 0.5, n_runs: int = 300) -> np.ndarray:
    """Compute base model probs for one seed (no post-processing, no obs blending)."""
    if model_name == "mc":
        return predict_grid_mc(ig, survival, n_runs=n_runs)
    elif model_name == "mc_ensemble":
        return predict_grid_mc_ensemble(ig, survival, model, "gbm",
                                         n_runs=n_runs, mc_weight=mc_weight)
    elif model_name == "gbm_ridge":
        gbm_m, ridge_m = model
        p_gbm   = predict_grid(ig, survival, gbm_m,   "gbm")
        p_ridge = predict_grid(ig, survival, ridge_m, "ridge_v1")
        blended = mc_weight * p_gbm + (1.0 - mc_weight) * p_ridge
        blended = np.maximum(blended, PROB_FLOOR)
        return blended / blended.sum(axis=2, keepdims=True)
    else:
        return predict_grid(ig, survival, model, model_name)


def _apply_postproc(probs: np.ndarray, ig: list, H: int, W: int,
                    obs,
                    temp_T_low: float = 1.0, temp_T_high: float = 1.0,
                    spatial_sigma: float = 0.0,
                    global_mult_clip=None,
                    static_floor_val: float = PROB_FLOOR,
                    n0: float = 10.0) -> np.ndarray:
    """Apply full post-processing pipeline to raw model probs."""
    # Temperature scaling (before obs)
    if temp_T_low != 1.0 or temp_T_high != 1.0:
        probs = apply_temperature_scaling(probs, temp_T_low, temp_T_high, ig)

    # Spatial smoothing (before obs)
    if spatial_sigma > 0.0:
        probs = apply_spatial_smooth(probs, spatial_sigma, ig)

    # Bayesian blend with observations
    if obs is not None:
        probs = blend_with_empirical(probs, obs, W, H, n0=n0)
        # Global multiplier (after obs, uses obs to compute correction)
        if global_mult_clip is not None:
            probs = apply_global_multiplier(probs, obs, ig,
                                             global_mult_clip[0], global_mult_clip[1])

    # Lower floor for static cells
    if static_floor_val < PROB_FLOOR:
        probs = apply_static_floor(probs, ig, static_floor_val)

    return probs


def run_postproc_loocv_sweep(gt_cache: dict, model_name: str,
                              observations: dict, n0: float, mc_weight: float,
                              sweep_configs: list,
                              quiet_rounds: bool = True) -> list:
    """
    Efficient post-processing parameter sweep using LOOCV.

    Pre-computes LOOCV models and raw predictions once per round, then sweeps
    over post-processing configurations without retraining.

    sweep_configs: list of (label, dict) where dict has optional keys:
        temp_T_low, temp_T_high, spatial_sigma, global_mult_clip, static_floor_val
    Returns list of (label, round_avgs, overall_avg) tuples.
    """
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    sorted_rounds = sorted(gt_cache.items(), key=lambda kv: kv[1]["round_number"])

    # ── Pre-compute LOOCV models + raw predictions for each round ────────────
    print("  Pre-computing LOOCV models and raw predictions...")
    round_cache = []  # [(rnum, survival, [(ig, gt_arr, H, W, obs, raw_probs), ...])]

    for rid, rdata in sorted_rounds:
        rnum     = rdata["round_number"]
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        print(f"    R{rnum}: training LOOCV...", end=" ", flush=True)
        if model_name == "gbm_ridge":
            model = (train_model(gt_cache, "gbm",      exclude_round=rnum),
                     train_model(gt_cache, "ridge_v1", exclude_round=rnum))
        elif model_name not in MC_MODELS:
            model = train_model(gt_cache, model_name, exclude_round=rnum)
        else:
            model = None
        print("done", end="  predictions...", flush=True)

        seeds = []
        for seed_str, sdata in rdata["seeds"].items():
            seed_idx = int(seed_str)
            ig   = sdata["initial_grid"]
            gt   = np.array(sdata["ground_truth"], dtype=np.float32)
            H, W = len(ig), len(ig[0])

            raw_probs = _get_raw_probs(ig, survival, model, model_name,
                                        mc_weight=mc_weight)

            obs = None
            if observations and rid in observations and seed_idx in observations[rid]:
                obs = observations[rid][seed_idx]

            seeds.append((ig, gt, H, W, obs, raw_probs))

        print(f"done ({len(seeds)} seeds)")
        round_cache.append((rnum, survival, seeds))

    print()

    # ── Sweep configurations ─────────────────────────────────────────────────
    baseline_avg = None
    all_results  = []

    print(f"  {'Config':<38} {'LOOCV Avg':>10} {'vs baseline':>12}")
    print("  " + "─" * 65)

    for label, cfg in sweep_configs:
        T_low   = cfg.get("temp_T_low",    1.0)
        T_high  = cfg.get("temp_T_high",   1.0)
        sigma   = cfg.get("spatial_sigma", 0.0)
        mult    = cfg.get("global_mult_clip", None)
        floor_v = cfg.get("static_floor_val", PROB_FLOOR)

        round_avgs = []
        for rnum, survival, seeds in round_cache:
            seed_scores = []
            for (ig, gt, H, W, obs, raw_probs) in seeds:
                probs = _apply_postproc(
                    raw_probs.copy(), ig, H, W, obs,
                    temp_T_low=T_low, temp_T_high=T_high,
                    spatial_sigma=sigma, global_mult_clip=mult,
                    static_floor_val=floor_v, n0=n0,
                )
                seed_scores.append(score_prediction(probs, gt, np.array(ig)))
            round_avgs.append(float(np.mean(seed_scores)))

        avg = float(np.mean(round_avgs))
        all_results.append((label, round_avgs, avg))

        if baseline_avg is None:
            baseline_avg = avg
            delta_str = " (baseline)"
        else:
            delta = avg - baseline_avg
            flag  = " ▲" if delta > 0.2 else (" ▼" if delta < -0.2 else " ~")
            delta_str = f"{delta:+.2f}{flag}"
        print(f"  {label:<38} {avg:10.2f} {delta_str:>12}")

    print()
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ─────────────────────────────────────────────────────────────────────────────

MC_MODELS = {"mc", "mc_ensemble"}
ENSEMBLE_MODELS = {"gbm_ridge"}  # models that combine two base models


def run_backtest(gt_cache: dict, model_name: str,
                 preloaded_model: dict | None = None,
                 n0: float = 3.0,
                 loocv: bool = False,
                 observations: dict | None = None,
                 n_runs: int = 300,
                 mc_weight: float = 0.5,
                 temp_T_low: float = 1.0,
                 temp_T_high: float = 1.0,
                 spatial_sigma: float = 0.0,
                 global_mult_clip=None,
                 static_floor_val: float = PROB_FLOOR) -> dict:
    """
    Score model on every round in gt_cache.

    loocv=True: for each round, retrain excluding that round (fair generalization test).
                For "mc" pure-MC models, loocv skips retraining (no trained model).
                For "mc_ensemble", loocv retrains the GBM component only.
    loocv=False: use preloaded_model (in-sample; valid for relative model comparison).

    Returns {round_number: {"survival": float, "score": float, "seed_scores": [...]}}
    """
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    results = {}
    sorted_rounds = sorted(gt_cache.items(), key=lambda kv: kv[1]["round_number"])

    for rid, rdata in sorted_rounds:
        rnum     = rdata["round_number"]
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        if model_name == "mc":
            model = None  # MC needs no trained model
            print(f"  R{rnum}: MC ({n_runs} runs)...", end=" ", flush=True)
        elif model_name == "mc_ensemble":
            if loocv:
                print(f"  R{rnum}: retraining GBM (excluding R{rnum}) + MC ({n_runs} runs)...",
                      end=" ", flush=True)
                model = train_model(gt_cache, "gbm", exclude_round=rnum)
            else:
                model = preloaded_model
        elif model_name == "gbm_ridge":
            if loocv:
                print(f"  R{rnum}: retraining GBM+ridge (excluding R{rnum})...",
                      end=" ", flush=True)
                model = (train_model(gt_cache, "gbm",      exclude_round=rnum),
                         train_model(gt_cache, "ridge_v1", exclude_round=rnum))
            else:
                model = preloaded_model  # expects (gbm_model, ridge_model) tuple
        elif loocv:
            print(f"  R{rnum}: retraining (excluding R{rnum})...", end=" ", flush=True)
            model = train_model(gt_cache, model_name, exclude_round=rnum)
        else:
            model = preloaded_model

        seed_scores = []
        for seed_str, sdata in rdata["seeds"].items():
            seed_idx = int(seed_str)
            ig = sdata["initial_grid"]
            gt = sdata["ground_truth"]
            H, W = len(ig), len(ig[0])

            probs = _get_raw_probs(ig, survival, model, model_name,
                                    mc_weight=mc_weight, n_runs=n_runs)

            obs = None
            if observations and rid in observations and seed_idx in observations[rid]:
                obs = observations[rid][seed_idx]

            probs = _apply_postproc(
                probs, ig, H, W, obs,
                temp_T_low=temp_T_low, temp_T_high=temp_T_high,
                spatial_sigma=spatial_sigma, global_mult_clip=global_mult_clip,
                static_floor_val=static_floor_val, n0=n0,
            )

            s = score_prediction(probs, gt, np.array(ig))
            seed_scores.append(s)

        round_score = float(np.mean(seed_scores))
        results[rnum] = {"survival": survival, "score": round_score,
                          "seed_scores": seed_scores}
        print(f"  R{rnum:2d} | survival={survival:.3f} | score={round_score:.2f} "
              f"| seeds={[round(s,1) for s in seed_scores]}")

    return results


def print_comparison_table(label_a: str, results_a: dict,
                            label_b: str | None = None,
                            results_b: dict | None = None):
    """Print formatted comparison table."""
    print()
    if label_b:
        print(f"{'Round':>6} {'Survival':>9} {label_a:>12} {label_b:>12} {'Delta':>8}")
        print("─" * 55)
    else:
        print(f"{'Round':>6} {'Survival':>9} {label_a:>12}")
        print("─" * 35)

    rnums = sorted(set(results_a) | (set(results_b) if results_b else set()))
    scores_a, scores_b = [], []
    for rnum in rnums:
        sa = results_a.get(rnum, {}).get("score")
        sb = results_b.get(rnum, {}).get("score") if results_b else None
        surv = (results_a.get(rnum) or {}).get("survival", 0)
        if label_b and sa is not None and sb is not None:
            delta = sb - sa
            flag  = " ▲" if delta > 0.5 else (" ▼" if delta < -0.5 else "")
            print(f"  R{rnum:2d} | {surv:7.1%} | {sa:10.2f} | {sb:10.2f} | {delta:+8.2f}{flag}")
            scores_a.append(sa); scores_b.append(sb)
        elif sa is not None:
            print(f"  R{rnum:2d} | {surv:7.1%} | {sa:10.2f}")
            scores_a.append(sa)

    print("─" * (55 if label_b else 35))
    avg_a = np.mean(scores_a) if scores_a else 0
    if label_b and scores_b:
        avg_b = np.mean(scores_b)
        delta = avg_b - avg_a
        flag  = " ▲ IMPROVEMENT" if delta > 0.2 else (" ▼ REGRESSION" if delta < -0.2 else " ~ TIE")
        print(f"  {'AVG':>2} | {'':>7} | {avg_a:10.2f} | {avg_b:10.2f} | {delta:+8.2f}{flag}")
    else:
        print(f"  {'AVG':>2} | {'':>7} | {avg_a:10.2f}")
    print()


def adaptive_n0_value(survival: float, strategy: str) -> float:
    """
    Compute n0 based on survival rate.

    Strategy A: fixed n0=10 (baseline)
    Strategy B: tiered — <20%→15, 20-35%→10, 35-50%→5, >50%→3
    Strategy C: continuous — max(2, 20 - 35 × survival)
    """
    if strategy == "A":
        return 10.0
    elif strategy == "B":
        if survival < 0.20:   return 15.0
        elif survival < 0.35: return 10.0
        elif survival < 0.50: return 5.0
        else:                 return 3.0
    elif strategy == "C":
        return max(2.0, 20.0 - 35.0 * survival)
    return 10.0


def run_adaptive_n0_comparison(gt_cache: dict, model_name: str,
                                observations: dict, loocv: bool = True) -> None:
    """
    Compare fixed n0=10 vs survival-adaptive n0 on rounds with real observations.
    Uses LOOCV by default for fair comparison.
    """
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    rid_to_rnum = {rid: rdata["round_number"] for rid, rdata in gt_cache.items()}
    obs_round_ids = sorted(set(observations.keys()) & set(gt_cache.keys()),
                           key=lambda r: rid_to_rnum[r])
    if not obs_round_ids:
        print("No rounds with cached observations found.")
        return

    strategies = ["A", "B", "C"]
    labels = {"A": "Fixed(n0=10)", "B": "Tiered(B)", "C": "Cont(C)"}
    mode = "LOOCV" if loocv else "in-sample"
    print(f"\n{'='*65}")
    print(f"  Survival-adaptive n0 comparison ({mode}, {model_name})")
    print(f"  Strategy A: fixed n0=10   B: <20%→15, 20-35%→10, 35-50%→5, >50%→3"
          f"   C: max(2, 20-35×surv)")
    print(f"{'='*65}")
    print(f"  {'Round':>5} {'Surv':>5} {'n0_B':>5} {'n0_C':>5} | "
          + " | ".join(f"{labels[s]:>14}" for s in strategies))
    print("  " + "─" * 63)

    loocv_models: dict = {}
    if loocv:
        for rid in obs_round_ids:
            rnum = rid_to_rnum[rid]
            print(f"  LOOCV: retraining excluding R{rnum}...", end=" ", flush=True)
            loocv_models[rid] = train_model(gt_cache, model_name, exclude_round=rnum)
            print("done")
        print()

    all_scores = {s: [] for s in strategies}
    for rid in obs_round_ids:
        rdata    = gt_cache[rid]
        rnum     = rdata["round_number"]
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)
        model    = loocv_models[rid] if loocv else train_model(gt_cache, model_name)

        n0_b = adaptive_n0_value(survival, "B")
        n0_c = adaptive_n0_value(survival, "C")

        strat_scores: dict = {s: [] for s in strategies}
        for seed_str, sdata in rdata["seeds"].items():
            seed_idx = int(seed_str)
            ig = sdata["initial_grid"]
            gt = sdata["ground_truth"]
            H, W = len(ig), len(ig[0])
            base_probs = predict_grid(ig, survival, model, model_name)
            for strat in strategies:
                n0 = adaptive_n0_value(survival, strat)
                if seed_idx in observations[rid]:
                    probs = blend_with_empirical(base_probs, observations[rid][seed_idx],
                                                  W, H, n0=n0)
                else:
                    probs = base_probs
                strat_scores[strat].append(score_prediction(probs, gt, np.array(ig)))

        row = {s: float(np.mean(strat_scores[s])) for s in strategies}
        for s in strategies:
            all_scores[s].append(row[s])
        best = max(strategies, key=lambda s: row[s])
        scores_str = " | ".join(
            f"{'>>'+f'{row[s]:.2f}':>14}" if s == best else f"{row[s]:>14.2f}"
            for s in strategies
        )
        print(f"  R{rnum:2d} | {survival:4.0%} | {n0_b:>4.1f} | {n0_c:>4.1f} | {scores_str}")

    print("  " + "─" * 63)
    avgs = {s: float(np.mean(all_scores[s])) for s in strategies}
    best = max(strategies, key=lambda s: avgs[s])
    avg_str = " | ".join(
        f"{'>>'+f'{avgs[s]:.2f}':>14}" if s == best else f"{avgs[s]:>14.2f}"
        for s in strategies
    )
    print(f"  {'AVG':>5} {'':>5} {'':>5} {'':>5} | {avg_str}")
    delta_b = avgs["B"] - avgs["A"]
    delta_c = avgs["C"] - avgs["A"]
    flag_b = " ▲ IMPROVEMENT" if delta_b > 0.2 else (" ▼ REGRESSION" if delta_b < -0.2 else " ~ TIE")
    flag_c = " ▲ IMPROVEMENT" if delta_c > 0.2 else (" ▼ REGRESSION" if delta_c < -0.2 else " ~ TIE")
    print(f"  vs A (fixed n0=10): B={delta_b:+.2f}{flag_b}   C={delta_c:+.2f}{flag_c}")
    print()


def n0_sweep(gt_cache: dict, model: dict, model_name: str,
             observations: dict, n0_values: list, loocv: bool = False):
    """
    Test different n0 blending values on rounds where we have real observations.
    Only rounds present in observations dict are tested.

    loocv=True: retrain model excluding each test round for fair comparison.
    loocv=False: use the provided pre-trained model (in-sample — biased toward high n0).
    """
    # Reverse map: round_id → round_number
    rid_to_rnum = {rid: rdata["round_number"] for rid, rdata in gt_cache.items()}

    obs_round_ids = set(observations.keys()) & set(gt_cache.keys())
    if not obs_round_ids:
        print("No rounds with cached observations found. n0 sweep skipped.")
        return

    obs_rnums = [rid_to_rnum[rid] for rid in obs_round_ids]
    mode_label = "LOOCV" if loocv else "in-sample"
    print(f"n0 sweep ({mode_label}) on rounds with actual observations: R{sorted(obs_rnums)}")
    print()
    print(f"{'n0':>6} | " + " | ".join(f"R{rnum:2d}" for rnum in sorted(obs_rnums))
          + " | Average")
    print("─" * (10 + 10 * len(obs_rnums)))

    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    # Pre-compute LOOCV models once per round (not once per n0)
    loocv_models = {}
    if loocv:
        for rid in sorted(obs_round_ids, key=lambda r: rid_to_rnum[r]):
            rnum = rid_to_rnum[rid]
            print(f"  LOOCV: retraining excluding R{rnum}...", end=" ", flush=True)
            loocv_models[rid] = train_model(gt_cache, model_name, exclude_round=rnum)
            print("done")
        print()

    for n0 in n0_values:
        row_scores = []
        for rid in sorted(obs_round_ids, key=lambda r: rid_to_rnum[r]):
            rdata = gt_cache[rid]
            rnum  = rdata["round_number"]
            surv  = rdata.get("survival") or per_round_survival.get(rnum, 0.30)
            active_model = loocv_models[rid] if loocv else model
            seed_scores = []
            for seed_str, sdata in rdata["seeds"].items():
                seed_idx = int(seed_str)
                ig = sdata["initial_grid"]
                gt = sdata["ground_truth"]
                H, W = len(ig), len(ig[0])
                probs = predict_grid(ig, surv, active_model, model_name)
                if seed_idx in observations[rid]:
                    obs = observations[rid][seed_idx]
                    probs = blend_with_empirical(probs, obs, W, H, n0=n0)
                seed_scores.append(score_prediction(probs, gt, np.array(ig)))
            row_scores.append(np.mean(seed_scores))

        avg = np.mean(row_scores)
        scores_str = " | ".join(f"{s:8.2f}" for s in row_scores)
        print(f"  {n0:4.1f} | {scores_str} | {avg:7.2f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive viewport selection (entropy-driven greedy set cover)
# ─────────────────────────────────────────────────────────────────────────────

# 9 fixed non-overlapping 15×15 tiles covering the full 40×40 map
UNIFORM_VIEWPORTS = [
    (vx, vy)
    for vy in (0, 15, 25)
    for vx in (0, 15, 25)
]


def select_adaptive_viewports(entropy_map: np.ndarray, n_viewports: int = 9,
                               vp_size: int = 15) -> list:
    """
    Greedily select viewports that maximise total entropy of uncovered cells.
    Uses O(n_vp × H × W) prefix-sum approach for efficiency.

    Returns list of (vx, vy) top-left corners.
    """
    H, W = entropy_map.shape
    remaining = entropy_map.copy()
    viewports = []

    for _ in range(n_viewports):
        # Prefix sum of remaining entropy
        cs = np.zeros((H + 1, W + 1), dtype=np.float64)
        cs[1:, 1:] = remaining.cumsum(axis=0).cumsum(axis=1)

        best_score = -1.0
        best_vp = (0, 0)
        for vy in range(H - vp_size + 1):
            for vx in range(W - vp_size + 1):
                score = (cs[vy + vp_size, vx + vp_size]
                         - cs[vy, vx + vp_size]
                         - cs[vy + vp_size, vx]
                         + cs[vy, vx])
                if score > best_score:
                    best_score = score
                    best_vp = (vx, vy)

        viewports.append(best_vp)
        vx, vy = best_vp
        remaining[vy:vy + vp_size, vx:vx + vp_size] = 0.0  # zero covered cells

    return viewports


def simulate_obs_from_gt(ground_truth: list, viewports: list,
                          vp_size: int = 15) -> dict:
    """
    Simulate what we'd observe at each viewport cell by taking argmax of the
    ground-truth distribution (most likely final class per cell).

    Returns {(x, y): [cls]} compatible with blend_with_empirical.
    """
    gt_arr = np.array(ground_truth, dtype=np.float32)
    H, W = gt_arr.shape[:2]
    obs: dict = {}
    for vx, vy in viewports:
        for dy in range(vp_size):
            for dx in range(vp_size):
                y, x = vy + dy, vx + dx
                if 0 <= y < H and 0 <= x < W:
                    cls = int(np.argmax(gt_arr[y, x]))
                    obs[(x, y)] = [cls]
    return obs


def compute_entropy_map(probs: np.ndarray, initial_grid: list) -> np.ndarray:
    """
    Per-cell prediction entropy H = -sum p*log(p).
    Static cells (ocean/mountain) get zero entropy (they don't score anyway).
    Returns (H, W) float32.
    """
    grid = np.array(initial_grid, dtype=np.int32)
    p = np.clip(probs, 1e-15, 1.0)
    entropy = -np.sum(p * np.log(p), axis=2).astype(np.float32)
    # Zero out static terrain (ocean=10, mountain=5)
    entropy[np.isin(grid, list(STATIC_CODES))] = 0.0
    return entropy


def run_coverage_comparison(gt_cache: dict, model_name: str,
                             n0: float = 10.0, loocv: bool = True,
                             n_phase1: int = 2, n_total: int = 9) -> None:
    """
    Compare uniform (9 fixed tiles) vs adaptive (entropy-driven) viewport
    selection across all historical rounds.

    Observations are simulated from ground truth (argmax of each cell's
    distribution). Phase-1 shared viewports are skipped — survival is assumed
    known (ground truth) so that we isolate the viewport placement question.

    n_phase1: viewports reserved for phase-1 survival estimation (not re-selected)
    n_total:  total viewports per seed (phase1 + phase2)
    """
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    print(f"\n{'Round':>6} {'Surv':>6} {'Uniform':>9} {'Adaptive':>10} {'Delta':>7}")
    print("─" * 50)

    all_uniform, all_adaptive = [], []
    sorted_rounds = sorted(gt_cache.items(), key=lambda kv: kv[1]["round_number"])

    for rid, rdata in sorted_rounds:
        rnum     = rdata["round_number"]
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        if loocv:
            model = train_model(gt_cache, model_name, exclude_round=rnum)
        else:
            model = train_model(gt_cache, model_name)

        uni_scores, ada_scores = [], []

        for seed_str, sdata in rdata["seeds"].items():
            ig = sdata["initial_grid"]
            gt = sdata["ground_truth"]
            H, W = len(ig), len(ig[0])

            # Base GBM prediction (no observations)
            base_probs = predict_grid(ig, survival, model, model_name)

            # ── Uniform coverage ──────────────────────────────────────────
            obs_uni = simulate_obs_from_gt(gt, UNIFORM_VIEWPORTS[:n_total])
            probs_uni = blend_with_empirical(base_probs, obs_uni, W, H, n0=n0)
            uni_scores.append(score_prediction(probs_uni, gt, np.array(ig)))

            # ── Adaptive coverage ─────────────────────────────────────────
            entropy = compute_entropy_map(base_probs, ig)
            ada_vps = select_adaptive_viewports(entropy, n_viewports=n_total)
            obs_ada = simulate_obs_from_gt(gt, ada_vps)
            probs_ada = blend_with_empirical(base_probs, obs_ada, W, H, n0=n0)
            ada_scores.append(score_prediction(probs_ada, gt, np.array(ig)))

        uni_avg = float(np.mean(uni_scores))
        ada_avg = float(np.mean(ada_scores))
        delta   = ada_avg - uni_avg
        flag    = " ▲" if delta > 0.5 else (" ▼" if delta < -0.5 else "")
        print(f"  R{rnum:2d} | {survival:5.1%} | {uni_avg:9.2f} | {ada_avg:10.2f} | {delta:+7.2f}{flag}")
        all_uniform.append(uni_avg)
        all_adaptive.append(ada_avg)

    print("─" * 50)
    avg_u = float(np.mean(all_uniform))
    avg_a = float(np.mean(all_adaptive))
    delta = avg_a - avg_u
    flag  = " ▲ IMPROVEMENT" if delta > 0.2 else (" ▼ REGRESSION" if delta < -0.2 else " ~ TIE")
    print(f"  {'AVG':>2} | {'':>5} | {avg_u:9.2f} | {avg_a:10.2f} | {delta:+7.2f}{flag}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Astar Island backtesting harness")
    parser.add_argument("--token",       required=True)
    parser.add_argument("--model",       default="ridge_v1",
                        choices=["ridge_v1", "ridge_v2", "gbm", "gbm_v3", "mc", "mc_ensemble",
                                 "gbm_ridge"],
                        help="Prediction model to test (gbm_ridge=50/50 GBM+ridge ensemble)")
    parser.add_argument("--compare",     nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="Compare two models side-by-side")
    parser.add_argument("--n-runs",      type=int, default=300,
                        help="Monte Carlo simulation runs per seed (default 300, MC models only)")
    parser.add_argument("--mc-weight",   type=float, default=0.5,
                        help="MC weight in mc_ensemble blend [0..1] (default 0.5)")
    parser.add_argument("--n0",          type=float, default=3.0,
                        help="Bayesian blending strength (default 3.0)")
    parser.add_argument("--n0-sweep",    action="store_true",
                        help="Test n0 in [0.25, 0.5, 1, 2, 3, 5, 10] on rounds with obs")
    parser.add_argument("--loocv",            action="store_true",
                        help="Leave-one-round-out CV (retrains model for each round)")
    parser.add_argument("--adaptive-coverage", action="store_true",
                        help="Compare uniform vs entropy-driven viewport selection")
    parser.add_argument("--adaptive-n0",       action="store_true",
                        help="Compare fixed n0=10 vs survival-adaptive n0 strategies B and C")
    parser.add_argument("--refresh",          action="store_true",
                        help="Force re-fetch ground truth from API")
    parser.add_argument("--gbm-depth",        type=int, default=None,
                        help="Override GBM max_depth (default 4)")
    parser.add_argument("--gbm-iter",         type=int, default=None,
                        help="Override GBM max_iter (default 100)")
    parser.add_argument("--gbm-lr",           type=float, default=None,
                        help="Override GBM learning_rate (default 0.1)")
    parser.add_argument("--gbm-sweep",        action="store_true",
                        help="Sweep GBM hyperparameter configs via LOOCV")
    # Post-processing flags
    parser.add_argument("--temp-sweep",  action="store_true",
                        help="Sweep T_low × T_high for temperature scaling (16 combos)")
    parser.add_argument("--sigma-sweep", action="store_true",
                        help="Sweep Gaussian smoothing sigma values")
    parser.add_argument("--mult-sweep",  action="store_true",
                        help="Sweep global observed/expected multiplier clip ranges")
    parser.add_argument("--floor-sweep", action="store_true",
                        help="Test lower probability floor for static cells")
    parser.add_argument("--postproc-stack", action="store_true",
                        help="Test each post-processing step individually then all combined")
    parser.add_argument("--all-sweeps", action="store_true",
                        help="Run all post-processing sweeps in one LOOCV pass (efficient)")
    parser.add_argument("--hier-sweep", action="store_true",
                        help="Sweep hierarchical prior blend weight vs gbm_ridge")
    # Individual post-processing params (for final combined run)
    parser.add_argument("--temp-T-low",  type=float, default=1.0)
    parser.add_argument("--temp-T-high", type=float, default=1.0)
    parser.add_argument("--sigma",       type=float, default=0.0)
    parser.add_argument("--mult-clip",   nargs=2, type=float, default=None,
                        metavar=("CLIP_LOW", "CLIP_HIGH"))
    parser.add_argument("--static-floor", type=float, default=None,
                        help="Lower probability floor for static cells (e.g. 0.005)")
    args = parser.parse_args()

    # Apply GBM hyperparameter overrides
    if args.gbm_depth is not None: _GBM_PARAMS["max_depth"]      = args.gbm_depth
    if args.gbm_iter  is not None: _GBM_PARAMS["max_iter"]       = args.gbm_iter
    if args.gbm_lr    is not None: _GBM_PARAMS["learning_rate"]  = args.gbm_lr

    # 1. Ground truth
    gt_cache = fetch_ground_truth(args.token, force_refresh=args.refresh)
    observations = load_observations()
    print(f"Observations available for {len(observations)} round(s): "
          f"{[gt_cache[r]['round_number'] for r in observations if r in gt_cache]}\n")

    # 1b. GBM hyperparameter sweep
    if args.gbm_sweep:
        configs = [
            {"max_depth": 4, "max_iter": 100,  "learning_rate": 0.10},   # baseline
            {"max_depth": 5, "max_iter": 100,  "learning_rate": 0.10},
            {"max_depth": 6, "max_iter": 100,  "learning_rate": 0.10},
            {"max_depth": 4, "max_iter": 200,  "learning_rate": 0.10},
            {"max_depth": 5, "max_iter": 200,  "learning_rate": 0.10},
            {"max_depth": 4, "max_iter": 200,  "learning_rate": 0.05},
            {"max_depth": 5, "max_iter": 200,  "learning_rate": 0.05},
        ]
        print(f"{'='*70}")
        print(f"  GBM Hyperparameter Sweep (LOOCV, n0={args.n0}, {len(gt_cache)} rounds)")
        print(f"{'='*70}")
        print(f"  {'Config':>30}  {'Avg':>7}  {'vs baseline':>12}")
        print("  " + "─" * 55)
        baseline_avg = None
        for cfg in configs:
            _GBM_PARAMS.update(cfg)
            label = f"d={cfg['max_depth']} n={cfg['max_iter']} lr={cfg['learning_rate']}"
            print(f"  {label:>30}  ", end="", flush=True)
            res = run_backtest(gt_cache, "gbm", loocv=True, n0=args.n0,
                                observations=observations)
            avg = float(np.mean([v["score"] for v in res.values()]))
            if baseline_avg is None:
                baseline_avg = avg
                delta_str = "(baseline)"
            else:
                delta = avg - baseline_avg
                flag = " ▲" if delta > 0.2 else (" ▼" if delta < -0.2 else " ~")
                delta_str = f"{delta:+.2f}{flag}"
            print(f"{avg:7.2f}  {delta_str:>12}")
        print()
        return

    # 1c. Temperature scaling sweep
    if args.temp_sweep:
        T_low_vals  = [0.3, 0.5, 0.7, 1.0]
        T_high_vals = [1.5, 2.0, 2.5, 3.0]
        configs = [("no scaling (baseline)", {"temp_T_low": 1.0, "temp_T_high": 1.0})]
        for tl in T_low_vals:
            for th in T_high_vals:
                if tl == 1.0 and th == 1.0:
                    continue
                configs.append((f"T_low={tl}  T_high={th}", {"temp_T_low": tl, "temp_T_high": th}))
        print(f"\n{'='*70}")
        print(f"  Temperature Scaling Sweep (LOOCV, model={args.model}, n0={args.n0})")
        print(f"{'='*70}")
        run_postproc_loocv_sweep(gt_cache, args.model, observations, args.n0,
                                  args.mc_weight, configs)
        return

    # 1d. Spatial smoothing sweep
    if args.sigma_sweep:
        sigma_vals = [0.0, 0.10, 0.15, 0.20, 0.30, 0.50]
        configs = [(f"sigma={s}", {"spatial_sigma": s}) for s in sigma_vals]
        print(f"\n{'='*70}")
        print(f"  Spatial Smoothing Sweep (LOOCV, model={args.model}, n0={args.n0})")
        print(f"{'='*70}")
        run_postproc_loocv_sweep(gt_cache, args.model, observations, args.n0,
                                  args.mc_weight, configs)
        return

    # 1e. Global multiplier sweep
    if args.mult_sweep:
        configs = [
            ("no multiplier (baseline)",      {"global_mult_clip": None}),
            ("clip [0.7, 1.5]",               {"global_mult_clip": (0.7, 1.5)}),
            ("clip [0.5, 2.0]",               {"global_mult_clip": (0.5, 2.0)}),
            ("clip [0.3, 3.0]",               {"global_mult_clip": (0.3, 3.0)}),
        ]
        print(f"\n{'='*70}")
        print(f"  Global Multiplier Sweep (LOOCV, model={args.model}, n0={args.n0})")
        print(f"  Note: multiplier applied only on rounds with cached observations")
        print(f"{'='*70}")
        run_postproc_loocv_sweep(gt_cache, args.model, observations, args.n0,
                                  args.mc_weight, configs)
        return

    # 1f. Static floor sweep
    if args.floor_sweep:
        configs = [
            ("floor=0.010 (baseline)",  {"static_floor_val": PROB_FLOOR}),
            ("floor=0.008 static",      {"static_floor_val": 0.008}),
            ("floor=0.005 static",      {"static_floor_val": 0.005}),
            ("floor=0.002 static",      {"static_floor_val": 0.002}),
        ]
        print(f"\n{'='*70}")
        print(f"  Static Cell Floor Sweep (LOOCV, model={args.model}, n0={args.n0})")
        print(f"{'='*70}")
        run_postproc_loocv_sweep(gt_cache, args.model, observations, args.n0,
                                  args.mc_weight, configs)
        return

    # 1g. Full post-processing stack comparison
    if args.postproc_stack:
        # Use best known values from individual sweeps (override via CLI if needed)
        best_T_low  = args.temp_T_low
        best_T_high = args.temp_T_high
        best_sigma  = args.sigma
        best_mult   = tuple(args.mult_clip) if args.mult_clip else None
        best_floor  = args.static_floor if args.static_floor else PROB_FLOOR

        configs = [
            ("baseline (no post-proc)",    {}),
            ("+ temperature scaling only", {"temp_T_low": best_T_low, "temp_T_high": best_T_high}),
            ("+ spatial smoothing only",   {"spatial_sigma": best_sigma}),
            ("+ global multiplier only",   {"global_mult_clip": best_mult}),
            ("+ lower static floor only",  {"static_floor_val": best_floor}),
            ("+ all combined",             {
                "temp_T_low":       best_T_low,
                "temp_T_high":      best_T_high,
                "spatial_sigma":    best_sigma,
                "global_mult_clip": best_mult,
                "static_floor_val": best_floor,
            }),
        ]
        print(f"\n{'='*70}")
        print(f"  Post-processing Stack Test (LOOCV, model={args.model}, n0={args.n0})")
        print(f"  T_low={best_T_low}, T_high={best_T_high}, sigma={best_sigma}, "
              f"mult={best_mult}, floor={best_floor}")
        print(f"{'='*70}")
        run_postproc_loocv_sweep(gt_cache, args.model, observations, args.n0,
                                  args.mc_weight, configs)
        return

    # 1h. All sweeps in one LOOCV pass (efficient — precomputes models once)
    if args.all_sweeps:
        T_low_vals  = [0.3, 0.5, 0.7, 1.0]
        T_high_vals = [1.5, 2.0, 2.5, 3.0]
        sigma_vals  = [0.0, 0.10, 0.15, 0.20, 0.30, 0.50]
        floor_vals  = [PROB_FLOOR, 0.008, 0.005, 0.002]

        all_configs = [("baseline (no post-proc)", {})]
        for tl in T_low_vals:
            for th in T_high_vals:
                all_configs.append(
                    (f"temp T_low={tl}  T_high={th}", {"temp_T_low": tl, "temp_T_high": th})
                )
        for s in sigma_vals[1:]:  # skip 0.0 (same as baseline)
            all_configs.append((f"sigma={s}", {"spatial_sigma": s}))
        for (cl, ch) in [(0.7, 1.5), (0.5, 2.0), (0.3, 3.0)]:
            all_configs.append((f"mult clip=[{cl},{ch}]", {"global_mult_clip": (cl, ch)}))
        for fv in floor_vals[1:]:  # skip baseline floor
            all_configs.append((f"static_floor={fv}", {"static_floor_val": fv}))

        print(f"\n{'='*70}")
        print(f"  All Post-Processing Sweeps (LOOCV, model={args.model}, n0={args.n0})")
        print(f"  LOOCV models precomputed ONCE for all {len(all_configs)} configs")
        print(f"{'='*70}")
        run_postproc_loocv_sweep(gt_cache, args.model, observations, args.n0,
                                  args.mc_weight, all_configs)
        return

    # 1i. Hierarchical prior blend sweep
    if args.hier_sweep:
        print(f"\n{'='*70}")
        print(f"  Hierarchical Prior Blend Sweep (LOOCV, gbm_ridge+hier, n0={args.n0})")
        print(f"  Baseline includes: spatial smooth σ=0.3 + global mult [0.5,2.0]")
        print(f"{'='*70}")
        run_hier_blend_sweep(gt_cache, observations, args.n0, args.mc_weight)
        return

    # 2. Compare two models
    if args.compare:
        model_a_name, model_b_name = args.compare
        print(f"=== Comparing {model_a_name} vs {model_b_name} "
              f"({'LOOCV' if args.loocv else 'in-sample'}) ===\n")

        if args.loocv:
            print(f"--- {model_a_name} (LOOCV) ---")
            results_a = run_backtest(gt_cache, model_a_name, loocv=True,
                                      n0=args.n0, observations=observations,
                                      n_runs=args.n_runs, mc_weight=args.mc_weight)
            print(f"\n--- {model_b_name} (LOOCV) ---")
            results_b = run_backtest(gt_cache, model_b_name, loocv=True,
                                      n0=args.n0, observations=observations,
                                      n_runs=args.n_runs, mc_weight=args.mc_weight)
        else:
            if model_a_name not in MC_MODELS:
                print(f"Training {model_a_name}...")
                m_a = train_model(gt_cache, model_a_name)
            else:
                m_a = None
            if model_b_name not in MC_MODELS:
                print(f"Training {model_b_name}...")
                m_b = train_model(gt_cache, model_b_name)
            else:
                m_b = None
            print(f"\n--- {model_a_name} ---")
            results_a = run_backtest(gt_cache, model_a_name, preloaded_model=m_a,
                                      n0=args.n0, observations=observations,
                                      n_runs=args.n_runs, mc_weight=args.mc_weight)
            print(f"\n--- {model_b_name} ---")
            results_b = run_backtest(gt_cache, model_b_name, preloaded_model=m_b,
                                      n0=args.n0, observations=observations,
                                      n_runs=args.n_runs, mc_weight=args.mc_weight)

        print_comparison_table(model_a_name, results_a, model_b_name, results_b)
        return

    # 2b. Adaptive n0 comparison
    if args.adaptive_n0:
        run_adaptive_n0_comparison(gt_cache, args.model, observations, loocv=args.loocv)
        return

    # 2c. Adaptive coverage comparison
    if args.adaptive_coverage:
        mode = "LOOCV" if args.loocv else "in-sample"
        print(f"=== Coverage Strategy Comparison: Uniform vs Adaptive ({mode}, n0={args.n0}) ===")
        print("  Observations simulated from ground truth (argmax per cell).")
        print("  Survival from calibration.json (ground truth — isolates viewport strategy).\n")
        run_coverage_comparison(gt_cache, args.model, n0=args.n0, loocv=args.loocv)
        return

    # 3. n0 sweep
    if args.n0_sweep:
        mode = "LOOCV" if args.loocv else "in-sample"
        print(f"=== n0 Sweep (testing Bayesian blending strength, {mode}) ===\n")
        if not args.loocv:
            print("Training model for n0 sweep...")
            model = train_model(gt_cache, args.model)
        else:
            model = None  # loocv_models built inside n0_sweep
        n0_sweep(gt_cache, model, args.model, observations,
                  n0_values=[1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
                  loocv=args.loocv)
        return

    # 4. Single model baseline or LOOCV
    mode_str = "LOOCV" if args.loocv else "in-sample"
    print(f"=== {args.model} baseline ({mode_str}, n0={args.n0}) ===\n")

    static_floor_arg = args.static_floor if args.static_floor else PROB_FLOOR
    mult_clip_arg    = tuple(args.mult_clip) if args.mult_clip else None

    if args.loocv:
        results = run_backtest(gt_cache, args.model, loocv=True,
                                n0=args.n0, observations=observations,
                                n_runs=args.n_runs, mc_weight=args.mc_weight,
                                temp_T_low=args.temp_T_low, temp_T_high=args.temp_T_high,
                                spatial_sigma=args.sigma, global_mult_clip=mult_clip_arg,
                                static_floor_val=static_floor_arg)
    else:
        if args.model in MC_MODELS or args.model in ENSEMBLE_MODELS:
            model = None  # will be built per-round in LOOCV, or must be pre-supplied
        else:
            print(f"Training {args.model} on all {len(gt_cache)} rounds...")
            model = train_model(gt_cache, args.model)
            print()
        results = run_backtest(gt_cache, args.model, preloaded_model=model,
                                n0=args.n0, observations=observations,
                                n_runs=args.n_runs, mc_weight=args.mc_weight,
                                temp_T_low=args.temp_T_low, temp_T_high=args.temp_T_high,
                                spatial_sigma=args.sigma, global_mult_clip=mult_clip_arg,
                                static_floor_val=static_floor_arg)

    print_comparison_table(f"{args.model} (n0={args.n0})", results)


if __name__ == "__main__":
    main()
