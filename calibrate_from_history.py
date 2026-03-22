"""
Calibrate all simulator parameters from historical ground-truth distributions.

For each completed round, fetches /analysis/{round_id}/{seed_index} which
returns the organiser-computed ground-truth (H×W×6 probability tensor built
from hundreds of true-parameter simulation runs) alongside the initial grid.

From (initial_grid, ground_truth) pairs we derive per-year transition rates
analytically:

  initial Settlement → GT[y,x][1+2]  →  p_annual_survive
  initial Port       → GT[y,x][1+2]  →  p_port_survive
  coastal Settlement → GT[y,x][2] / survival  →  p_port_form
  initial Ruin       → GT[y,x][3]    →  p_ruin_rebuild, p_ruin_forest, p_ruin_empty
  adjacent Empty     → GT[y,x][1+2]  →  p_annual_expand
  Forest near settle → GT[y,x][0]    →  p_forest_clear

Saves calibration.json which main.py loads at startup as the base_params for
the simulator (current-round observations then refine only p_annual_survive and
p_annual_expand on top of these stable structural parameters).

Usage:
    python calibrate_from_history.py --token YOUR_JWT_TOKEN
"""

import argparse
import json
import time

import numpy as np
import requests

BASE_URL = "https://api.ainm.no/astar-island"
DELTAS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def fetch_rounds(session) -> list:
    return session.get(f"{BASE_URL}/rounds", timeout=15).json()


def fetch_round_detail(session, round_id: str) -> dict:
    return session.get(f"{BASE_URL}/rounds/{round_id}", timeout=15).json()


def fetch_analysis(session, round_id: str, seed_index: int) -> dict | None:
    r = session.get(f"{BASE_URL}/analysis/{round_id}/{seed_index}", timeout=30)
    if r.status_code == 200:
        return r.json()
    print(f"    /analysis/{round_id[:8]}/{seed_index}: HTTP {r.status_code}")
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_coastal(grid, x, y, H, W) -> bool:
    for dy, dx in DELTAS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 10:
            return True
    return False


def adj_settle_count(settle_set: set, x: int, y: int) -> int:
    return sum(1 for dy, dx in DELTAS if (x + dx, y + dy) in settle_set)


# ---------------------------------------------------------------------------
# Statistics accumulation
# ---------------------------------------------------------------------------

def accumulate_stats(stats: dict, initial_grid: list, ground_truth: list):
    """
    Pool raw statistics from one (initial_grid, ground_truth) pair.
    All values are 50-year probabilities extracted per cell.
    """
    H = len(initial_grid)
    W = len(initial_grid[0]) if H > 0 else 0

    initial_settle = {
        (x, y)
        for y in range(H)
        for x in range(W)
        if initial_grid[y][x] in (1, 2)
    }

    for y in range(H):
        for x in range(W):
            code = initial_grid[y][x]
            gt = ground_truth[y][x]       # 6-float list

            p_active  = gt[1] + gt[2]     # P(Settlement or Port at t=50)
            p_port    = gt[2]
            p_ruin    = gt[3]
            p_forest  = gt[4]
            p_empty   = gt[0]

            if code == 1:   # initial Settlement
                stats["settle_survival_50yr"].append(p_active)
                if is_coastal(initial_grid, x, y, H, W) and p_active > 0.02:
                    stats["port_given_survive"].append(p_port / p_active)

            elif code == 2:  # initial Port
                stats["port_survival_50yr"].append(p_active)

            elif code == 3:  # initial Ruin
                stats["ruin_still_50yr"].append(p_ruin)
                stats["ruin_to_active_50yr"].append(p_active)
                stats["ruin_to_forest_50yr"].append(p_forest)
                stats["ruin_to_empty_50yr"].append(p_empty)

            elif code in (0, 11):   # initial Empty / Plains
                n_adj = adj_settle_count(initial_settle, x, y)
                if n_adj > 0:
                    stats["expand_adj_count"].append(n_adj)
                    stats["expand_prob_50yr"].append(p_active)

            elif code == 4:   # initial Forest
                n_adj = adj_settle_count(initial_settle, x, y)
                if n_adj > 0:
                    stats["forest_clear_50yr_near"].append(p_empty)


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

def fit_params(stats: dict) -> dict:
    """
    Derive annual simulator parameters from pooled 50-year statistics.

    Key identity: for a cell that transitions at constant annual rate r,
        P(not yet transitioned at t=50) = (1-r)^50
    so  r = 1 - P(not_transitioned)^(1/50)

    For cells with multiple destination states we first estimate the total
    escape rate, then split by observed proportions.
    """
    params = {}

    def smean(lst):
        return float(np.mean(lst)) if lst else None

    def annual_from_50yr_survive(p50, lo=0.80, hi=0.999):
        p50 = float(np.clip(p50, 0.001, 0.999))
        return float(np.clip(p50 ** (1 / 50), lo, hi))

    # --- Settlement survival ---
    if stats["settle_survival_50yr"]:
        s50 = smean(stats["settle_survival_50yr"])
        params["p_annual_survive"] = annual_from_50yr_survive(s50)
        params["_settle_survival_50yr_mean"] = round(s50, 4)
        print(f"  Settlement: 50yr survival={s50:.3f}  →  p_annual_survive={params['p_annual_survive']:.4f}"
              f"  (n={len(stats['settle_survival_50yr'])})")

    # --- Port survival ---
    if stats["port_survival_50yr"]:
        p50 = smean(stats["port_survival_50yr"])
        params["p_port_survive"] = annual_from_50yr_survive(p50)
        print(f"  Port:       50yr survival={p50:.3f}  →  p_port_survive={params['p_port_survive']:.4f}"
              f"  (n={len(stats['port_survival_50yr'])})")

    # --- Port formation ---
    if stats["port_given_survive"]:
        p_port_cond = smean(stats["port_given_survive"])
        # P(port|survive) = 1 - (1-p_port_form)^50
        p_port_form = float(1.0 - (1.0 - np.clip(p_port_cond, 0.001, 0.999)) ** (1 / 50))
        params["p_port_form"] = float(np.clip(p_port_form, 0.001, 0.30))
        print(f"  Port form:  P(port|survive)={p_port_cond:.3f}  →  p_port_form={params['p_port_form']:.4f}"
              f"  (n={len(stats['port_given_survive'])})")

    # --- Ruin dynamics ---
    if stats["ruin_still_50yr"]:
        still50 = smean(stats["ruin_still_50yr"])
        # Total annual escape rate from Ruin state
        q = float(1.0 - np.clip(still50, 0.001, 0.999) ** (1 / 50))
        # Split by observed 50yr proportions
        to_active = smean(stats["ruin_to_active_50yr"]) or 0.0
        to_forest = smean(stats["ruin_to_forest_50yr"]) or 0.0
        to_empty  = smean(stats["ruin_to_empty_50yr"]) or 0.0
        total_out = to_active + to_forest + to_empty
        if total_out > 0.001:
            params["p_ruin_rebuild"] = float(np.clip(q * to_active / total_out, 0.001, 0.10))
            params["p_ruin_forest"]  = float(np.clip(q * to_forest / total_out, 0.001, 0.10))
            params["p_ruin_empty"]   = float(np.clip(q * to_empty  / total_out, 0.001, 0.10))
        print(
            f"  Ruin: still_50yr={still50:.3f} q={q:.4f}  →  "
            f"rebuild={params.get('p_ruin_rebuild',0):.4f}  "
            f"forest={params.get('p_ruin_forest',0):.4f}  "
            f"empty={params.get('p_ruin_empty',0):.4f}"
            f"  (n={len(stats['ruin_still_50yr'])})"
        )

    # --- Expansion ---
    if stats["expand_adj_count"]:
        adj  = np.array(stats["expand_adj_count"], dtype=float)
        prob = np.clip(stats["expand_prob_50yr"], 0.001, 0.999)
        # Exact per-cell: p_expand = 1 - (1 - prob)^(1 / (adj * 50))
        per_cell = 1.0 - (1.0 - np.array(prob)) ** (1.0 / (adj * 50))
        p_expand = float(np.clip(np.mean(per_cell), 0.0005, 0.05))
        params["p_annual_expand"] = p_expand
        print(f"  Expansion:  mean_p50yr={np.mean(prob):.3f}  →  p_annual_expand={p_expand:.5f}"
              f"  (n={len(stats['expand_adj_count'])})")

    # --- Forest clearing near settlements ---
    if stats["forest_clear_50yr_near"]:
        p_clear50 = smean(stats["forest_clear_50yr_near"])
        p_fc = float(np.clip(1.0 - (1.0 - p_clear50) ** (1 / 50), 0.0001, 0.05))
        params["p_forest_clear"] = p_fc
        print(f"  Forest:     p_clear_50yr={p_clear50:.4f}  →  p_forest_clear={p_fc:.5f}"
              f"  (n={len(stats['forest_clear_50yr_near'])})")

    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate simulator parameters from all historical ground-truth data."
    )
    parser.add_argument("--token", required=True, help="JWT access_token from app.ainm.no")
    parser.add_argument("--out", default="calibration.json", help="Output file (default calibration.json)")
    args = parser.parse_args()

    session = make_session(args.token)

    rounds = fetch_rounds(session)
    completed = sorted(
        [r for r in rounds if r["status"] == "completed"],
        key=lambda r: r["round_number"],
    )
    print(f"Found {len(completed)} completed rounds\n")

    stats: dict = {
        "settle_survival_50yr": [],
        "port_survival_50yr": [],
        "port_given_survive": [],
        "ruin_still_50yr": [],
        "ruin_to_active_50yr": [],
        "ruin_to_forest_50yr": [],
        "ruin_to_empty_50yr": [],
        "expand_adj_count": [],
        "expand_prob_50yr": [],
        "forest_clear_50yr_near": [],
    }

    per_round_survival: dict = {}
    rounds_used: list = []

    for r in completed:
        rnum = r["round_number"]
        rid  = r["id"]
        detail = fetch_round_detail(session, rid)
        n_seeds = detail.get("seeds_count", 5)
        print(f"Round #{rnum}  ({rid[:8]}...)  seeds={n_seeds}")

        round_survivals = []
        for seed_idx in range(n_seeds):
            analysis = fetch_analysis(session, rid, seed_idx)
            if not analysis:
                continue
            ig = analysis.get("initial_grid")
            gt = analysis.get("ground_truth")
            if ig is None or gt is None:
                print(f"  seed {seed_idx}: missing initial_grid or ground_truth")
                continue

            H, W = len(ig), (len(ig[0]) if ig else 0)
            settle_surv = [
                gt[y][x][1] + gt[y][x][2]
                for y in range(H) for x in range(W)
                if ig[y][x] in (1, 2)
            ]
            if settle_surv:
                s = float(np.mean(settle_surv))
                round_survivals.append(s)
                print(f"  seed {seed_idx}: survival={s:.3f}  ({len(settle_surv)} settle cells)")

            accumulate_stats(stats, ig, gt)
            time.sleep(0.15)

        if round_survivals:
            per_round_survival[str(rnum)] = round(float(np.mean(round_survivals)), 4)
            rounds_used.append(rnum)

    print(f"\nPooled data — fitting parameters from {len(rounds_used)} rounds...")
    params = fit_params(stats)

    calibration = {
        "fitted_from_rounds": rounds_used,
        "params": params,
        "per_round_survival_50yr": per_round_survival,
        "n_cells_per_stat": {k: len(v) for k, v in stats.items()},
    }

    with open(args.out, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nSaved → {args.out}")
    survival_vals = list(per_round_survival.values())
    if survival_vals:
        print(f"Survival across rounds: min={min(survival_vals):.3f}  max={max(survival_vals):.3f}  "
              f"mean={float(np.mean(survival_vals)):.3f}")


if __name__ == "__main__":
    main()
