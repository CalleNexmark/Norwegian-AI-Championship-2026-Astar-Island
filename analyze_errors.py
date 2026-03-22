#!/usr/bin/env python3
"""
Part A: Deep error analysis on weak rounds (R1, R7, R12).

Steps:
  A1 — per-cell KL divergence, top-50 worst cells per round
  A2 — categorize error types, KL contribution by category
  A3 — spatial pattern analysis (clustered vs scattered, terrain context)
  A4 — feature gap analysis (what our features DON'T capture)

Usage:
    python analyze_errors.py --token TOKEN [--rounds 1,7,12]
"""

import argparse
import json
import os

import numpy as np

# ─── reuse backtest infrastructure ───────────────────────────────────────────
from backtest import (
    fetch_ground_truth,
    train_model,
    predict_grid,
    compute_features_v2,
    STATIC_CODES,
    TERRAIN_GROUP,
    N_CLASSES,
    CALIBRATION_FILE,
)

TERRAIN_NAMES = {0: "Empty", 1: "Settle", 2: "Port", 3: "Ruin",
                 4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}
CLASS_NAMES   = ["Empty", "Settle", "Port", "Ruin", "Forest", "Mountain"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell KL
# ─────────────────────────────────────────────────────────────────────────────

def per_cell_kl(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """KL(gt||pred) per cell. Shape (H, W)."""
    p = np.clip(pred, 1e-15, 1.0)
    g = np.clip(gt,   1e-15, 1.0)
    return np.where(gt > 1e-15, g * np.log(g / p), 0.0).sum(axis=2)


def per_cell_entropy(gt: np.ndarray) -> np.ndarray:
    """H(gt) per cell. Shape (H, W)."""
    g = np.clip(gt, 1e-15, 1.0)
    return -np.where(gt > 1e-15, g * np.log(g), 0.0).sum(axis=2)


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Error categorisation
# ─────────────────────────────────────────────────────────────────────────────

def categorize_error(init_code: int, gt_row: np.ndarray, pred_row: np.ndarray) -> str:
    """
    Assign one error category to a cell.
    Compares dominant GT class vs dominant predicted class.
    """
    gt_cls   = int(np.argmax(gt_row))
    pred_cls = int(np.argmax(pred_row))
    init_grp = TERRAIN_GROUP.get(init_code, "empty")

    # Settlement / port survival
    if init_grp in ("settle", "port"):
        if gt_cls == 0 and pred_cls in (1, 2):
            return "over_predict_settle"    # model says survive, GT says collapse
        if gt_cls in (1, 2) and pred_cls == 0:
            return "under_predict_settle"   # model says collapse, GT says survive

    # Expansion into empty/forest
    if init_grp in ("empty", "forest"):
        if gt_cls in (1, 2) and pred_cls not in (1, 2):
            return "missed_expansion"       # settlement expanded here, model didn't catch it
        if pred_cls in (1, 2) and gt_cls not in (1, 2):
            return "ghost_expansion"        # model predicts expansion that didn't happen

    # Port formation (settlement → port)
    if init_grp == "settle":
        if gt_cls == 2 and pred_cls != 2:
            return "missed_port"
        if pred_cls == 2 and gt_cls != 2:
            return "ghost_port"

    # Forest dynamics
    if init_grp == "forest":
        if gt_cls != 4 and pred_cls == 4:
            return "forest_over"
        if gt_cls == 4 and pred_cls != 4:
            return "forest_under"

    # Ruin
    if gt_cls == 3 or pred_cls == 3:
        return "ruin_error"

    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Top-50 worst cells per seed, diagnostic table
# ─────────────────────────────────────────────────────────────────────────────

def analyze_round(rnum: int, rdata: dict, model: dict, survival: float,
                  top_n: int = 50, verbose_n: int = 10):
    """
    Full analysis for one round: A1 (worst cells), A2 (categories), A3 (spatial).
    Returns per-seed analysis dicts.
    """
    print(f"\n{'='*70}")
    print(f"  ROUND {rnum}  |  survival={survival:.1%}")
    print(f"{'='*70}")

    round_results = []

    for seed_str, sdata in sorted(rdata["seeds"].items(), key=lambda kv: int(kv[0])):
        seed_idx = int(seed_str)
        ig_list  = sdata["initial_grid"]
        gt_list  = sdata["ground_truth"]
        ig       = np.array(ig_list, dtype=np.int8)
        gt       = np.array(gt_list, dtype=np.float32)
        H, W     = ig.shape

        probs    = predict_grid(ig_list, survival, model, "gbm")

        kl_map   = per_cell_kl(probs, gt)
        ent_map  = per_cell_entropy(gt)

        # Only dynamic cells
        dynamic_mask = ~np.isin(ig, list(STATIC_CODES))
        kl_flat  = kl_map.copy()
        kl_flat[~dynamic_mask] = -1.0

        # Top-N worst cells
        flat_idx = np.argsort(kl_flat.reshape(-1))[::-1][:top_n]
        worst_ys, worst_xs = np.unravel_index(flat_idx, (H, W))

        # ── A1: diagnostic table (top verbose_n rows) ──────────────────────
        print(f"\n  Seed {seed_idx} — top {verbose_n} worst cells "
              f"(mean KL={kl_map[dynamic_mask].mean():.4f}, "
              f"total dynamic={dynamic_mask.sum()}):")
        print(f"  {'Cell':>8} {'Init':>8} {'GT top-2':>22} {'Pred top-2':>22} {'KL':>7} {'Category'}")
        print("  " + "─" * 82)

        categories = {}
        kl_by_cat  = {}
        worst_cells_data = []

        for iy, ix in zip(worst_ys, worst_xs):
            if kl_flat[iy, ix] < 0:
                break
            init_code = int(ig[iy, ix])
            gt_row    = gt[iy, ix]
            pred_row  = probs[iy, ix]

            cat = categorize_error(init_code, gt_row, pred_row)
            categories[cat]  = categories.get(cat, 0) + 1
            kl_by_cat[cat]   = kl_by_cat.get(cat, 0.0) + float(kl_flat[iy, ix])

            worst_cells_data.append({
                "x": int(ix), "y": int(iy),
                "init": init_code,
                "gt": gt_row.tolist(),
                "pred": pred_row.tolist(),
                "kl": float(kl_flat[iy, ix]),
                "cat": cat,
            })

            if len(worst_cells_data) <= verbose_n:
                # Format top-2 GT and pred
                gt_top2 = sorted(range(6), key=lambda k: -gt_row[k])[:2]
                pr_top2 = sorted(range(6), key=lambda k: -pred_row[k])[:2]
                gt_str  = " ".join(f"{CLASS_NAMES[k]}:{gt_row[k]:.2f}" for k in gt_top2)
                pr_str  = " ".join(f"{CLASS_NAMES[k]}:{pred_row[k]:.2f}" for k in pr_top2)
                init_name = TERRAIN_NAMES.get(init_code, str(init_code))
                print(f"  ({ix:2d},{iy:2d})    {init_name:>7}  {gt_str:>22}  {pr_str:>22}  "
                      f"{kl_flat[iy, ix]:6.3f}  {cat}")

        # ── A2: error category summary ──────────────────────────────────────
        total_kl_top = sum(kl_by_cat.values())
        print(f"\n  Error categories (top {top_n} cells, total KL={total_kl_top:.2f}):")
        for cat, cnt in sorted(categories.items(), key=lambda kv: -kv[1]):
            frac = kl_by_cat[cat] / total_kl_top if total_kl_top > 0 else 0
            print(f"    {cat:<30}  count={cnt:3d}  KL_share={frac:.1%}")

        # ── A3: spatial analysis ─────────────────────────────────────────────
        wx = np.array([d["x"] for d in worst_cells_data[:top_n]])
        wy = np.array([d["y"] for d in worst_cells_data[:top_n]])

        # Nearest-neighbour distance among worst cells
        if len(wx) > 1:
            dists = []
            for i in range(len(wx)):
                for j in range(i+1, min(len(wx), 20)):  # sample pairs for speed
                    dists.append(np.hypot(wx[i]-wx[j], wy[i]-wy[j]))
            mean_nn = np.mean(dists) if dists else 0.0
            # If random scatter in 40x40 -> expected pairwise ~ sqrt(1600/50)*something
            # A rough heuristic: clustered if mean_nn < 8, scattered if > 15
            cluster_label = "CLUSTERED" if mean_nn < 8 else ("SCATTERED" if mean_nn > 15 else "MIXED")
            print(f"\n  Spatial: mean pairwise dist among worst-{min(len(wx),20)} = {mean_nn:.1f}  [{cluster_label}]")

        # Terrain context breakdown
        init_counts = {}
        for d in worst_cells_data[:top_n]:
            nm = TERRAIN_NAMES.get(d["init"], str(d["init"]))
            init_counts[nm] = init_counts.get(nm, 0) + 1
        print(f"  Initial terrain of worst cells: "
              + ", ".join(f"{k}:{v}" for k, v in sorted(init_counts.items(), key=lambda kv: -kv[1])))

        # Coastal vs inland
        is_coastal = np.isin(ig, [10]).astype(np.float32)
        from backtest import _box_sum_excl
        coastal_adj = (_box_sum_excl(is_coastal, 1) > 0)
        worst_coastal = sum(1 for d in worst_cells_data[:top_n]
                             if coastal_adj[d["y"], d["x"]])
        print(f"  Coastal worst cells: {worst_coastal}/{len(worst_cells_data[:top_n])}")

        # ── A4: feature gap analysis ─────────────────────────────────────────
        feat = compute_features_v2(ig_list, survival)   # (H, W, 13)
        FEAT_NAMES = ["settle_r1", "settle_r3", "settle_r7", "is_coastal",
                      "forest_r1", "ocean_r3", "survival",
                      "sr*settle_r3", "sr*forest_r1", "coastal*settle_r1",
                      "forest*settle_r1", "survival^2", "settle_r3*r7"]

        # Compare feature distributions: worst-50 vs best-50 (highest-KL vs lowest-KL dynamic)
        dynamic_ys, dynamic_xs = np.where(dynamic_mask)
        dynamic_kl = kl_map[dynamic_ys, dynamic_xs]
        order = np.argsort(dynamic_kl)
        best50_ys  = dynamic_ys[order[:50]]
        best50_xs  = dynamic_xs[order[:50]]

        worst50_feats = feat[worst_ys[:top_n], worst_xs[:top_n]]     # (50, 13)
        best50_feats  = feat[best50_ys, best50_xs]                    # (50, 13)

        print(f"\n  Feature analysis — mean difference (worst50 - best50):")
        print(f"  {'Feature':>20}  {'worst50':>9}  {'best50':>9}  {'diff':>9}")
        print("  " + "─" * 52)
        for fi, fname in enumerate(FEAT_NAMES):
            wm = float(np.mean(worst50_feats[:, fi]))
            bm = float(np.mean(best50_feats[:, fi]))
            diff = wm - bm
            flag = " <<" if abs(diff) > 0.5 else ""
            print(f"  {fname:>20}  {wm:9.3f}  {bm:9.3f}  {diff:+9.3f}{flag}")

        round_results.append({
            "seed": seed_idx,
            "worst_cells": worst_cells_data[:top_n],
            "categories": categories,
            "kl_by_cat": kl_by_cat,
        })

    return round_results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-round summary
# ─────────────────────────────────────────────────────────────────────────────

def cross_round_summary(all_results: dict):
    """Aggregate error categories across all analyzed rounds."""
    print(f"\n{'='*70}")
    print("  CROSS-ROUND SUMMARY — Error categories")
    print(f"{'='*70}")

    cat_total   = {}
    cat_kl      = {}
    cat_rounds  = {}  # which rounds each category appears in

    for rnum, seeds in all_results.items():
        round_cats = set()
        for s in seeds:
            for cat, cnt in s["categories"].items():
                cat_total[cat]  = cat_total.get(cat, 0) + cnt
                cat_kl[cat]     = cat_kl.get(cat, 0.0) + s["kl_by_cat"].get(cat, 0.0)
                round_cats.add(cat)
        for cat in round_cats:
            cat_rounds.setdefault(cat, set()).add(rnum)

    total_kl = sum(cat_kl.values())
    print(f"\n  {'Category':<30} {'Count':>7} {'KL_share':>9} {'Rounds'}")
    print("  " + "─" * 65)
    for cat, cnt in sorted(cat_total.items(), key=lambda kv: -kv[1]):
        frac   = cat_kl[cat] / total_kl if total_kl > 0 else 0
        rounds = sorted(cat_rounds.get(cat, set()))
        print(f"  {cat:<30} {cnt:7d}  {frac:8.1%}  R{rounds}")

    print(f"\n  Total KL in top-50 cells across all analyzed rounds: {total_kl:.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Error analysis for weak rounds")
    parser.add_argument("--token",   required=True)
    parser.add_argument("--rounds",  default="1,7,12",
                        help="Comma-separated round numbers to analyze (default: 1,7,12)")
    parser.add_argument("--top",     type=int, default=50,
                        help="Worst cells per seed to examine (default 50)")
    parser.add_argument("--verbose", type=int, default=10,
                        help="Rows to print in diagnostic table per seed (default 10)")
    args = parser.parse_args()

    target_rounds = [int(r) for r in args.rounds.split(",")]
    print(f"Analyzing rounds: {target_rounds}")

    # Load ground truth
    gt_cache = fetch_ground_truth(args.token)

    # Load per-round survival
    per_round_survival = {}
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        per_round_survival = {int(k): float(v)
                               for k, v in cal.get("per_round_survival_50yr", {}).items()}

    all_results = {}

    for rid, rdata in sorted(gt_cache.items(), key=lambda kv: kv[1]["round_number"]):
        rnum = rdata["round_number"]
        if rnum not in target_rounds:
            continue

        survival = rdata.get("survival") or per_round_survival.get(rnum, 0.30)

        # LOOCV: retrain excluding this round for fair evaluation
        print(f"\nTraining LOOCV model (excluding R{rnum})...", end=" ", flush=True)
        model = train_model(gt_cache, "gbm", exclude_round=rnum)
        print("done")

        seeds_results = analyze_round(rnum, rdata, model, survival,
                                      top_n=args.top, verbose_n=args.verbose)
        all_results[rnum] = seeds_results

    cross_round_summary(all_results)


if __name__ == "__main__":
    main()
