#!/usr/bin/env python3
"""
Part B: Coverage vs Depth tradeoff — viewport strategy comparison.

Tests four strategies on all historical rounds:
  S1  Full coverage    — 9 unique 15×15 tiles (current)
  S2  Deep zone        — 6 unique + 3 repeats on densest tiles
  S3  Hybrid           — 7 unique + 2 repeats on densest tiles
  S4  Survival-adaptive — S1 if survival<0.20, S3 if 0.20-0.40, S2 if >0.40

For cells observed N times: blend weight = N/(N+n0) (Bayesian formula).
For unobserved cells: pure model prior (weight 0).

Observations are simulated from GT argmax (same approach as --adaptive-coverage).
Survival is taken from calibration.json (ground truth) to isolate the viewport question.

Usage:
    python analyze_viewports.py --token TOKEN [--n0 10]
"""

import argparse
import json
import os

import numpy as np

from backtest import (
    fetch_ground_truth,
    train_model,
    predict_grid,
    score_prediction,
    STATIC_CODES,
    CALIBRATION_FILE,
    UNIFORM_VIEWPORTS,
)

N_CLASSES = 6
PROB_FLOOR = 0.01
VP_SIZE    = 15


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box_sum_excl(mask: np.ndarray, r: int) -> np.ndarray:
    H, W = mask.shape
    padded = np.pad(mask.astype(np.float32), r, constant_values=0.0)
    cs = np.zeros((H + 2*r + 1, W + 2*r + 1), dtype=np.float64)
    cs[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    box = (cs[y+2*r+1, x+2*r+1] - cs[y, x+2*r+1]
           - cs[y+2*r+1, x] + cs[y, x])
    return (box - mask.astype(np.float64)).astype(np.float32)


def settlement_density_per_tile(ig: np.ndarray, vp_size: int = VP_SIZE) -> dict:
    """
    Returns {(vx, vy): settlement_count} for all possible tile origins.
    """
    H, W = ig.shape
    is_settle = np.isin(ig, [1, 2]).astype(np.float32)
    # prefix sum for fast tile queries
    cs = np.zeros((H+1, W+1), dtype=np.float64)
    cs[1:, 1:] = is_settle.cumsum(axis=0).cumsum(axis=1)

    tiles = {}
    for vy in range(H - vp_size + 1):
        for vx in range(W - vp_size + 1):
            count = (cs[vy+vp_size, vx+vp_size]
                     - cs[vy, vx+vp_size]
                     - cs[vy+vp_size, vx]
                     + cs[vy, vx])
            tiles[(vx, vy)] = int(count)
    return tiles


def top_settle_tiles(ig: np.ndarray, n: int, exclude: set | None = None,
                     vp_size: int = VP_SIZE) -> list:
    """Top-n tiles by settlement count, excluding already-selected ones."""
    exclude = exclude or set()
    tile_scores = settlement_density_per_tile(ig, vp_size)
    ranked = sorted(tile_scores.items(), key=lambda kv: -kv[1])
    result = []
    for vp, _ in ranked:
        if vp not in exclude:
            result.append(vp)
        if len(result) == n:
            break
    return result


def simulate_obs_multi(gt: np.ndarray, viewports: list, vp_size: int = VP_SIZE) -> dict:
    """
    Simulate observations from ground truth (argmax).
    Returns {(x,y): [cls, cls, ...]} — one entry per visit to that cell.
    Multiple viewports covering the same cell append multiple observations.
    """
    H, W = gt.shape[:2]
    obs: dict = {}
    for vx, vy in viewports:
        for dy in range(vp_size):
            for dx in range(vp_size):
                y, x = vy+dy, vx+dx
                if 0 <= y < H and 0 <= x < W:
                    cls = int(np.argmax(gt[y, x]))
                    obs.setdefault((x, y), []).append(cls)
    return obs


def blend(model_probs: np.ndarray, observations: dict,
          width: int, height: int, n0: float) -> np.ndarray:
    """Bayesian blend — supports multiple observations per cell."""
    ALPHA = 0.1
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
        w_emp     = n  / (n + n0)
        p = w_model * model_probs[y, x] + w_emp * empirical
        p = np.maximum(p, PROB_FLOOR)
        probs[y, x] = p / p.sum()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Strategy definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_strategy_viewports(ig: np.ndarray, survival: float,
                              strategy: str) -> list:
    """
    Returns a list of (vx, vy) viewport positions (with repeats for multi-visit).

    S1: 9 uniform tiles (full coverage)
    S2: 6 uniform tiles + 3 repeats on densest tiles
    S3: 7 uniform tiles + 2 repeats on densest tiles  (= 7 unique + 2 extra visits)
    S4: survival-dependent:
        survival < 0.20 → S1
        0.20–0.40       → S3
        > 0.40          → S2
    """
    if strategy == "S4":
        if survival < 0.20:
            strategy = "S1"
        elif survival < 0.40:
            strategy = "S3"
        else:
            strategy = "S2"

    uniform_9 = list(UNIFORM_VIEWPORTS)   # 9 fixed tiles

    if strategy == "S1":
        return uniform_9                   # 9 unique tiles

    if strategy == "S2":
        # 6 uniform + 3 repeat visits on densest tiles
        base6  = uniform_9[:6]
        dense3 = top_settle_tiles(ig, 3, exclude=set())
        return base6 + dense3              # 9 total (6 unique, 3 may overlap)

    if strategy == "S3":
        # 7 uniform + 2 repeats on densest tiles
        base7  = uniform_9[:7]
        dense2 = top_settle_tiles(ig, 2, exclude=set())
        return base7 + dense2              # 9 total (7 unique, 2 may overlap)

    raise ValueError(f"Unknown strategy: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_viewport_comparison(gt_cache: dict, n0: float = 10.0) -> None:
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    strategies = ["S1", "S2", "S3", "S4"]
    labels = {
        "S1": "Full(9)",
        "S2": "Deep(6+3r)",
        "S3": "Hybrid(7+2r)",
        "S4": "Adaptive",
    }

    header = f"  {'Round':>6} {'Surv':>5} | " + " | ".join(f"{labels[s]:>12}" for s in strategies)
    print()
    print(f"{'='*70}")
    print("  Viewport Strategy Comparison (n0={:.0f}, LOOCV, simulated obs from GT)".format(n0))
    print(f"{'='*70}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    all_scores = {s: [] for s in strategies}
    sorted_rounds = sorted(gt_cache.items(), key=lambda kv: kv[1]["round_number"])

    for rid, rdata in sorted_rounds:
        rnum     = rdata["round_number"]
        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        # LOOCV model
        print(f"  R{rnum}: training LOOCV model...", end=" ", flush=True)
        model = train_model(gt_cache, "gbm", exclude_round=rnum)
        print("done", end="  ", flush=True)

        round_scores = {s: [] for s in strategies}

        for seed_str, sdata in rdata["seeds"].items():
            ig_list = sdata["initial_grid"]
            gt_list = sdata["ground_truth"]
            ig      = np.array(ig_list, dtype=np.int8)
            gt      = np.array(gt_list, dtype=np.float32)
            H, W    = ig.shape

            base_probs = predict_grid(ig_list, survival, model, "gbm")

            for strat in strategies:
                vps  = build_strategy_viewports(ig, survival, strat)
                obs  = simulate_obs_multi(gt, vps)
                pred = blend(base_probs, obs, W, H, n0=n0)
                s    = score_prediction(pred, gt, ig)
                round_scores[strat].append(s)

        row_avgs = {s: float(np.mean(round_scores[s])) for s in strategies}
        for s in strategies:
            all_scores[s].append(row_avgs[s])

        # Best strategy highlight
        best_s = max(strategies, key=lambda s: row_avgs[s])
        scores_str = " | ".join(
            f"{'>>'+f'{row_avgs[s]:.2f}':>12}" if s == best_s else f"{row_avgs[s]:>12.2f}"
            for s in strategies
        )
        print(f"  R{rnum:2d} | {survival:4.0%} | {scores_str}")

    print("  " + "─" * (len(header) - 2))
    avgs = {s: float(np.mean(all_scores[s])) for s in strategies}
    best_s = max(strategies, key=lambda s: avgs[s])
    avg_str = " | ".join(
        f"{'>>'+f'{avgs[s]:.2f}':>12}" if s == best_s else f"{avgs[s]:>12.2f}"
        for s in strategies
    )
    print(f"  {'AVG':>6} {'':>5} | {avg_str}")

    # Delta vs S1 (current baseline)
    delta_str = " | ".join(
        f"{avgs[s]-avgs['S1']:>+12.2f}" for s in strategies
    )
    print(f"  {'vs S1':>6} {'':>5} | {delta_str}")
    print()

    # Per-survival-tier breakdown
    print("  Per-survival-tier breakdown:")
    tiers = [
        ("Catastrophic (<15%)", lambda surv: surv < 0.15),
        ("Hard (15-30%)",       lambda surv: 0.15 <= surv < 0.30),
        ("Medium (30-45%)",     lambda surv: 0.30 <= surv < 0.45),
        ("Thriving (>45%)",     lambda surv: surv >= 0.45),
    ]
    for tier_name, tier_fn in tiers:
        tier_scores = {s: [] for s in strategies}
        for i, (rid, rdata) in enumerate(sorted_rounds):
            rnum     = rdata["round_number"]
            survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)
            if tier_fn(survival):
                for s in strategies:
                    tier_scores[s].append(all_scores[s][i])
        if not tier_scores["S1"]:
            continue
        tier_avgs = {s: float(np.mean(tier_scores[s])) for s in strategies}
        best = max(strategies, key=lambda s: tier_avgs[s])
        scores_str = " | ".join(
            f"*{tier_avgs[s]:.1f}*" if s == best else f"{tier_avgs[s]:.1f}"
            for s in strategies
        )
        n_rounds = len(tier_scores["S1"])
        print(f"    {tier_name:<25} (n={n_rounds}): " + scores_str)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Viewport strategy comparison")
    parser.add_argument("--token", required=True)
    parser.add_argument("--n0",    type=float, default=10.0)
    args = parser.parse_args()

    gt_cache = fetch_ground_truth(args.token)
    run_viewport_comparison(gt_cache, n0=args.n0)


if __name__ == "__main__":
    main()
