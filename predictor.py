"""
Build H x W x 6 probability tensors from initial state + observations.
"""

import numpy as np
from typing import Dict, List, Tuple

# Class indices
C_EMPTY = 0
C_SETTLEMENT = 1
C_PORT = 2
C_RUIN = 3
C_FOREST = 4
C_MOUNTAIN = 5
N_CLASSES = 6

TERRAIN_TO_CLASS = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0,
}

# Base probability templates for unobserved cells (before floor+renorm)
# Shape: [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
# Calibrated from R5+R6+R7 ground truth:
#   - Survival rate varies per round (29%–59%) — these are midpoint estimates
#   - Empty cell prior is accurate — don't change it
#   - Forest is mostly static
BASE_PROBS = {
    0:  [0.82, 0.05, 0.01, 0.03, 0.08, 0.01],  # Empty/Plains
    1:  [0.50, 0.40, 0.05, 0.03, 0.01, 0.01],  # Settlement — midpoint 29–59% survival
    2:  [0.45, 0.15, 0.35, 0.03, 0.01, 0.01],  # Port
    3:  [0.25, 0.10, 0.03, 0.45, 0.15, 0.02],  # Ruin
    4:  [0.05, 0.02, 0.01, 0.02, 0.88, 0.02],  # Forest — mostly static
    5:  [0.01, 0.01, 0.01, 0.01, 0.01, 0.95],  # Mountain — static
    10: [0.97, 0.01, 0.005, 0.005, 0.005, 0.005], # Ocean — static
    11: [0.82, 0.05, 0.01, 0.03, 0.08, 0.01],  # Plains
}

PROB_FLOOR = 0.01  # never assign 0 to any class — KL divergence goes infinite


def estimate_survival_rate(
    all_observations: dict,
    all_initial_states: list,
) -> float | None:
    """
    Estimate this round's settlement survival rate from cross-seed observations.

    For each seed, look at cells that were initially Settlement/Port and check
    what they became. Pool all seeds together for a single estimate.

    Returns float in [0, 1], or None if no settlement cells were observed.
    """
    survived = 0
    total = 0

    for seed_idx, observations in all_observations.items():
        if seed_idx >= len(all_initial_states):
            continue
        grid = all_initial_states[seed_idx]["grid"]
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0

        for (x, y), class_list in observations.items():
            if x >= width or y >= height:
                continue
            terrain = grid[y][x]
            if terrain not in (1, 2):  # only initial settlements / ports
                continue
            for cls in class_list:
                total += 1
                if cls in (C_SETTLEMENT, C_PORT):
                    survived += 1

    if total == 0:
        return None

    rate = survived / total
    print(f"  Estimated survival rate: {rate:.3f} ({survived}/{total} obs)")
    return rate


def _near_coast(grid: list, x: int, y: int, height: int, width: int, radius: int = 3) -> bool:
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny][nx] == 10:
                    return True
    return False


def _base_probs_for_cell(terrain_code: int, near_settlement: bool, near_coast: bool, base: dict = BASE_PROBS) -> np.ndarray:
    p = np.array(base.get(terrain_code, base[0]), dtype=float)
    if near_settlement and terrain_code in (0, 11, 3):
        p[C_SETTLEMENT] += 0.10
        if near_coast:
            p[C_PORT] += 0.05
        p[C_RUIN] += 0.05
        p[C_EMPTY] -= 0.15
    return p


def build_prediction(
    initial_state: dict,
    observations: Dict[Tuple[int, int], List[int]],
    width: int,
    height: int,
    survival_rate: float | None = None,
    simulator_probs: np.ndarray | None = None,
) -> list:
    """
    Build H x W x 6 probability tensor.

    For observed cells: Laplace-smoothed empirical frequencies (alpha=0.1).
    For unobserved cells:
      - If simulator_probs provided (H×W×6 ndarray from Monte Carlo sim):
        use simulator output as the prior — captures spatial expansion chains
        and coastal dynamics that static terrain priors miss.
      - Otherwise: terrain-based prior + proximity heuristics (legacy fallback).

    If survival_rate is provided without simulator_probs, it overrides the
    hardcoded 50% midpoint for unobserved settlement cells (R9/R10 behaviour).

    Returns nested list for JSON serialization.
    """
    grid = initial_state["grid"]

    # Build static base probs table (used only when simulator_probs is None)
    base = BASE_PROBS.copy()
    if survival_rate is not None and simulator_probs is None:
        s = survival_rate
        p_s = np.array([max(1 - s - 0.10, 0.05), s, 0.05, 0.03, 0.01, 0.01], dtype=float)
        base[1] = (p_s / p_s.sum()).tolist()
        print(f"  Survival override: Settlement prior = {[round(v,3) for v in base[1]]}")

    settlements = [(s["x"], s["y"]) for s in initial_state["settlements"] if s.get("alive", True)]

    settlement_proximity = np.zeros((height, width), dtype=bool)
    for sx, sy in settlements:
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    settlement_proximity[ny, nx] = True

    prediction = np.zeros((height, width, N_CLASSES), dtype=float)

    for y in range(height):
        for x in range(width):
            terrain_code = grid[y][x]
            key = (x, y)

            if key in observations and len(observations[key]) > 0:
                # Observed cell: Laplace-smoothed empirical frequencies
                counts = np.zeros(N_CLASSES, dtype=float)
                for cls in observations[key]:
                    counts[cls] += 1.0
                n = len(observations[key])
                alpha = 0.1
                p = (counts + alpha) / (n + alpha * N_CLASSES)
            elif simulator_probs is not None:
                # Unobserved cell: use Monte Carlo simulator distribution
                p = simulator_probs[y, x].astype(float)
            else:
                # Fallback: static terrain-based prior + proximity heuristics
                near_s = bool(settlement_proximity[y, x])
                near_c = _near_coast(grid, x, y, height, width)
                p = _base_probs_for_cell(terrain_code, near_s, near_c, base)

            p = np.maximum(p, PROB_FLOOR)
            p = p / p.sum()
            prediction[y, x] = p

    return prediction.tolist()
