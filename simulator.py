"""
Forward stochastic cellular automaton simulator for Astar Island.

Models Norse civilization dynamics as a parameterized CA:
  - Settlement/Port: survive each year or collapse → Ruin
  - Settlements expand into adjacent empty cells
  - Coastal settlements may form Ports
  - Ruins rebuild (near active settlements) or decay to Forest/Plains
  - Forest mostly stable, rarely cleared near settlements

Run N Monte Carlo simulations → per-cell probability distributions that
capture spatial structure (expansion chains, coastal dynamics, ruin cascades)
far better than static terrain-based priors.
"""

import numpy as np

# Internal terrain codes
T_EMPTY      = 0
T_SETTLEMENT = 1
T_PORT       = 2
T_RUIN       = 3
T_FOREST     = 4
T_MOUNTAIN   = 5
T_OCEAN      = 10
T_PLAINS     = 11

# Prediction class indices
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
N_CLASSES = 6
PROB_FLOOR = 0.01

# Default parameters — calibrated offline from R5/R6/R7 ground truth.
# p_annual_survive^50 ≈ 50yr survival rate. ~0.982^50 ≈ 0.40 (midpoint).
DEFAULT_PARAMS: dict = {
    "p_annual_survive": 0.982,   # P(settlement survives 1 year)
    "p_port_survive":   0.985,   # Ports slightly more stable (trade income)
    "p_annual_expand":  0.012,   # P(empty cell settled per adj active per year)
    "p_port_form":      0.08,    # P(coastal surviving settlement → Port per year)
    "p_ruin_rebuild":   0.015,   # P(Ruin → Settlement per year if adj active)
    "p_ruin_forest":    0.008,   # P(Ruin → Forest per year)
    "p_ruin_empty":     0.012,   # P(Ruin → Plains per year)
    "p_forest_clear":   0.003,   # P(Forest → Plains per year near active settlement)
}


def _adj8(mask: np.ndarray) -> np.ndarray:
    """Sum of 8-connected neighbours where mask is True. Returns float32 (H, W)."""
    m = mask.astype(np.float32)
    out = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out += np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def run_one(
    initial_grid: np.ndarray,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run one 50-year stochastic simulation.

    Args:
        initial_grid: (H, W) int8 terrain codes
        params:       simulator parameter dict (see DEFAULT_PARAMS)
        rng:          numpy Generator

    Returns:
        Final (H, W) int8 terrain grid after 50 years
    """
    grid = initial_grid.copy()
    H, W = grid.shape

    for _ in range(50):
        is_settlement = grid == T_SETTLEMENT
        is_port       = grid == T_PORT
        is_active     = is_settlement | is_port
        is_ruin       = grid == T_RUIN
        is_forest     = grid == T_FOREST
        is_ocean      = grid == T_OCEAN
        is_mountain   = grid == T_MOUNTAIN
        is_empty      = (grid == T_EMPTY) | (grid == T_PLAINS)

        adj_active = _adj8(is_active)
        adj_ocean  = _adj8(is_ocean)

        new_grid = grid.copy()

        # Settlements collapse → Ruin
        r = rng.random((H, W), dtype=np.float32)
        collapse = is_settlement & (r < (1.0 - params["p_annual_survive"]))
        new_grid[collapse] = T_RUIN

        # Ports collapse → Ruin
        r = rng.random((H, W), dtype=np.float32)
        collapse_port = is_port & (r < (1.0 - params["p_port_survive"]))
        new_grid[collapse_port] = T_RUIN

        # Surviving coastal settlements form Ports
        surviving_settle = is_settlement & ~collapse
        r = rng.random((H, W), dtype=np.float32)
        new_grid[surviving_settle & (adj_ocean > 0) & (r < params["p_port_form"])] = T_PORT

        # Active cells expand into adjacent empty cells
        # P(empty cell gets settled) = 1 - (1 - p_expand)^adj_active
        p_settle = 1.0 - (1.0 - params["p_annual_expand"]) ** adj_active
        r = rng.random((H, W), dtype=np.float32)
        new_grid[is_empty & (r < p_settle)] = T_SETTLEMENT

        # Ruins: rebuild / grow forest / become plains
        r_rebuild = rng.random((H, W), dtype=np.float32)
        r_forest  = rng.random((H, W), dtype=np.float32)
        r_plains  = rng.random((H, W), dtype=np.float32)

        rebuild   = is_ruin & (adj_active > 0) & (r_rebuild < params["p_ruin_rebuild"])
        to_forest = is_ruin & ~rebuild & (r_forest < params["p_ruin_forest"])
        to_plains = is_ruin & ~rebuild & ~to_forest & (r_plains < params["p_ruin_empty"])

        new_grid[rebuild]   = T_SETTLEMENT
        new_grid[to_forest] = T_FOREST
        new_grid[to_plains] = T_PLAINS

        # Forest cleared near active settlements
        r = rng.random((H, W), dtype=np.float32)
        new_grid[is_forest & (adj_active > 0) & (r < params["p_forest_clear"])] = T_PLAINS

        # Static terrain
        new_grid[is_ocean]    = T_OCEAN
        new_grid[is_mountain] = T_MOUNTAIN

        grid = new_grid

    return grid


def run_monte_carlo(
    initial_state: dict,
    width: int,
    height: int,
    params: dict,
    n_runs: int = 200,
    rng_seed: int | None = None,
) -> np.ndarray:
    """
    Run n_runs simulations → H×W×6 float32 probability array.

    Applies PROB_FLOOR (0.01) per class before renormalizing so no
    KL-divergence term blows up from a zero prediction.
    """
    rng = np.random.default_rng(rng_seed)
    initial_grid = np.array(initial_state["grid"], dtype=np.int8)

    counts = np.zeros((height, width, N_CLASSES), dtype=np.float32)
    for _ in range(n_runs):
        final = run_one(initial_grid, params, rng)
        for code, cls in TERRAIN_TO_CLASS.items():
            counts[:, :, cls] += (final == code).astype(np.float32)

    probs = counts / n_runs
    probs = np.maximum(probs, PROB_FLOOR)
    probs /= probs.sum(axis=2, keepdims=True)
    return probs  # H×W×6


def calibrate_params(
    all_observations: dict,
    all_initial_states: list,
    base_params: dict | None = None,
) -> dict:
    """
    Estimate simulator parameters from cross-seed observations.

    Calibrates from observations of cells at t=50:
      - p_annual_survive: derived from observed 50yr settlement survival rate
      - p_annual_expand:  derived from new settlements detected in empty cells

    All other parameters fall back to base_params / DEFAULT_PARAMS.

    Returns updated params dict.
    """
    params = (base_params or DEFAULT_PARAMS).copy()

    survived       = 0
    total_settle   = 0
    new_settle_obs = 0
    adj_pressure   = 0.0   # sum of adj initial-settlement counts for observed empty cells

    for seed_idx, observations in all_observations.items():
        if seed_idx >= len(all_initial_states):
            continue
        grid = all_initial_states[seed_idx]["grid"]
        H, W = len(grid), (len(grid[0]) if grid else 0)

        initial_settle = {
            (x, y)
            for y in range(H)
            for x in range(W)
            if grid[y][x] in (T_SETTLEMENT, T_PORT)
        }

        for (x, y), class_list in observations.items():
            if x >= W or y >= H:
                continue
            terrain = grid[y][x]

            if terrain in (T_SETTLEMENT, T_PORT):
                for cls in class_list:
                    total_settle += 1
                    if cls in (1, 2):
                        survived += 1

            elif terrain in (T_EMPTY, T_PLAINS):
                # Check if this empty cell had initial-settlement neighbours
                n_adj = sum(
                    1
                    for dy in range(-1, 2)
                    for dx in range(-1, 2)
                    if (x + dx, y + dy) in initial_settle
                )
                if n_adj > 0:
                    for cls in class_list:
                        if cls in (1, 2):
                            new_settle_obs += 1
                    adj_pressure += n_adj

    if total_settle > 10:
        survival_50yr = survived / total_settle
        p_survive = float(np.clip(survival_50yr ** (1.0 / 50), 0.80, 0.999))
        params["p_annual_survive"] = p_survive
        params["p_port_survive"]   = min(p_survive + 0.003, 0.999)
        print(
            f"  Calibrated p_annual_survive={p_survive:.4f}  "
            f"(50yr={survival_50yr:.3f}, n_settle_obs={total_settle})"
        )

    if adj_pressure > 20:
        p_expand = float(np.clip(new_settle_obs / (adj_pressure * 50), 0.001, 0.05))
        params["p_annual_expand"] = p_expand
        print(
            f"  Calibrated p_annual_expand={p_expand:.5f}  "
            f"(new_settle={new_settle_obs}, adj_pressure={adj_pressure:.0f})"
        )

    return params
