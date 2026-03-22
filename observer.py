"""
Strategic viewport placement and query management for Astar Island.
"""

import time
from typing import List, Dict, Tuple
import numpy as np

# Map terrain codes to prediction class indices
TERRAIN_TO_CLASS = {
    0: 0,   # Empty   -> class 0 (Empty)
    1: 1,   # Settlement -> class 1
    2: 2,   # Port    -> class 2
    3: 3,   # Ruin    -> class 3
    4: 4,   # Forest  -> class 4
    5: 5,   # Mountain -> class 5
    10: 0,  # Ocean   -> class 0 (Empty)
    11: 0,  # Plains  -> class 0 (Empty)
}

# Static terrains that never change
STATIC_TERRAINS = {10, 5}  # Ocean, Mountain


def plan_full_coverage(
    initial_state: dict,
    width: int,
    height: int,
    n_queries: int,
    exclude: set | None = None,
) -> List[Tuple[int, int]]:
    """
    Return up to n_queries non-overlapping 15×15 viewport positions that
    tile the full map, sorted by settlement density (most important first).

    A 40×40 map needs exactly 9 tiles for full coverage:
        x ∈ {0, 15, 25},  y ∈ {0, 15, 25}

    Pass `exclude` (set of (x,y) tuples) to skip viewports already observed
    in an earlier phase so the two phases complement each other perfectly.
    """
    VP = 15
    grid = initial_state["grid"]

    # Build the minimal set of anchors that cover [0, dim) with VP-wide tiles
    def anchors(dim: int) -> List[int]:
        pts: List[int] = []
        p = 0
        while p + VP <= dim:
            pts.append(p)
            p += VP
        if not pts or pts[-1] + VP < dim:
            pts.append(dim - VP)
        return sorted(set(pts))

    all_tiles = [(ax, ay) for ax in anchors(width) for ay in anchors(height)]

    if exclude:
        all_tiles = [t for t in all_tiles if t not in exclude]

    # Score by initial settlement / port density inside tile
    def settle_count(ax: int, ay: int) -> int:
        return sum(
            1 for dy in range(VP) for dx in range(VP)
            if ax + dx < width and ay + dy < height
            and grid[ay + dy][ax + dx] in (1, 2)
        )

    all_tiles.sort(key=lambda t: settle_count(*t), reverse=True)

    # If fewer distinct tiles than budget, cycle from the top
    if len(all_tiles) == 0:
        return []
    plan: List[Tuple[int, int]] = []
    while len(plan) < n_queries:
        plan.extend(all_tiles[: n_queries - len(plan)])
    return plan[:n_queries]


def find_shared_viewports(
    all_initial_states: list,
    width: int,
    height: int,
    n_viewports: int,
) -> List[Tuple[int, int]]:
    """
    Find viewport positions that maximise total initial settlement coverage
    across ALL seeds combined. Used for R9 parameter hunting — same viewport
    positions are queried in every seed so cross-seed outcomes can be compared.
    """
    VP = 15
    x_anchors = list(range(0, width - VP + 1, VP))
    if not x_anchors or x_anchors[-1] + VP < width:
        x_anchors.append(width - VP)
    y_anchors = list(range(0, height - VP + 1, VP))
    if not y_anchors or y_anchors[-1] + VP < height:
        y_anchors.append(height - VP)

    def cross_seed_score(ax, ay):
        total = 0
        for state in all_initial_states:
            grid = state["grid"]
            for dy in range(VP):
                for dx in range(VP):
                    gx, gy = ax + dx, ay + dy
                    if gx < width and gy < height and grid[gy][gx] in (1, 2):
                        total += 1
        return total

    all_tiles = [(ax, ay) for ax in x_anchors for ay in y_anchors]
    all_tiles.sort(key=lambda t: cross_seed_score(*t), reverse=True)
    chosen = all_tiles[:n_viewports]
    print(f"  Shared viewports (cross-seed): {chosen}")
    return chosen


def find_densest_zone(initial_state: dict, width: int, height: int) -> Tuple[int, int]:
    """
    Find the 15x15 viewport position with the most initial settlements.
    Used for Round 8 deep narrow sampling experiment.
    """
    grid = initial_state["grid"]
    VP = 15
    best_pos, best_count = (0, 0), 0

    x_anchors = list(range(0, width - VP + 1, VP))
    if not x_anchors or x_anchors[-1] + VP < width:
        x_anchors.append(width - VP)
    y_anchors = list(range(0, height - VP + 1, VP))
    if not y_anchors or y_anchors[-1] + VP < height:
        y_anchors.append(height - VP)

    for ax in x_anchors:
        for ay in y_anchors:
            count = sum(
                1 for dy in range(VP) for dx in range(VP)
                if grid[ay + dy][ax + dx] in (1, 2)
            )
            if count > best_count:
                best_count, best_pos = count, (ax, ay)

    print(f"  Densest zone: ({best_pos[0]},{best_pos[1]}) with {best_count} initial settlements")
    return best_pos


def plan_viewports(
    initial_state: dict,
    width: int,
    height: int,
    n_queries: int,
    deep_zone: bool = False,
) -> List[Tuple[int, int]]:
    """
    Plan viewport (x, y) positions for a single seed.

    Strategy:
    - Pass 1: Cover all land area with 15x15 tiles (avoid pure-ocean tiles)
    - Pass 2: Re-observe settlement-adjacent tiles for probability estimates
    - Stop when n_queries reached
    """
    grid = initial_state["grid"]  # grid[y][x]
    settlements = [(s["x"], s["y"]) for s in initial_state["settlements"] if s.get("alive", True)]

    VP = 15  # viewport size

    # Tile anchors that cover the full map
    x_anchors = []
    x = 0
    while x < width:
        x_anchors.append(x)
        x += VP
    # Ensure last tile doesn't overshoot: clamp to map edge
    if x_anchors[-1] + VP > width:
        x_anchors[-1] = width - VP

    y_anchors = []
    y = 0
    while y < height:
        y_anchors.append(y)
        y += VP
    if y_anchors[-1] + VP > height:
        y_anchors[-1] = height - VP

    # Round 8 mode: repeat the same densest zone for all queries
    if deep_zone:
        vx, vy = find_densest_zone(initial_state, width, height)
        return [(vx, vy)] * n_queries

    # Score each tile by land/settlement content (skip pure-ocean tiles)
    def tile_score(ax, ay):
        land = 0
        near_settlement = 0
        for dy in range(VP):
            for dx in range(VP):
                gx, gy = ax + dx, ay + dy
                if gx >= width or gy >= height:
                    continue
                cell = grid[gy][gx]
                if cell not in STATIC_TERRAINS:
                    land += 1
                # Count proximity to settlements
                for sx, sy in settlements:
                    if abs(gx - sx) <= 7 and abs(gy - sy) <= 7:
                        near_settlement += 1
                        break
        return land + near_settlement * 2  # weight settlement proximity

    # Sort tiles by score descending (skip pure-ocean tiles)
    all_tiles = [(ax, ay) for ax in x_anchors for ay in y_anchors]
    all_tiles.sort(key=lambda t: tile_score(*t), reverse=True)

    # Filter out tiles that are entirely ocean (score == 0) — only skip if map large enough
    scored = [(t, tile_score(*t)) for t in all_tiles]
    land_tiles = [(t, s) for t, s in scored if s > 0]
    if not land_tiles:
        land_tiles = scored  # fallback: use all tiles

    # Pass 1: full coverage — take top tiles until we've seen every land cell
    plan = []
    covered = set()
    tile_list = [t for t, _ in land_tiles]

    for tx, ty in tile_list:
        if len(plan) >= n_queries:
            break
        plan.append((tx, ty))
        for dy in range(VP):
            for dx in range(VP):
                covered.add((tx + dx, ty + dy))

    # Pass 2: fill remaining budget with settlement-dense tiles (repeated observations)
    if len(plan) < n_queries:
        # Settlement-focused tiles: center viewport on each settlement cluster
        for sx, sy in settlements:
            if len(plan) >= n_queries:
                break
            # Clamp viewport so settlement is roughly centered
            vx = max(0, min(sx - VP // 2, width - VP))
            vy = max(0, min(sy - VP // 2, height - VP))
            plan.append((vx, vy))

    # Fill any remaining budget with high-score tiles (repeat)
    idx = 0
    while len(plan) < n_queries and tile_list:
        plan.append(tile_list[idx % len(tile_list)])
        idx += 1

    return plan[:n_queries]


def run_observations(
    session,
    base_url: str,
    round_id: str,
    seed_index: int,
    viewport_plan: List[Tuple[int, int]],
    delay: float = 0.25,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Execute viewport queries for one seed.

    Returns:
        observations: dict mapping (x, y) -> list of observed class indices
    """
    observations: Dict[Tuple[int, int], List[int]] = {}

    for i, (vx, vy) in enumerate(viewport_plan):
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": vx,
            "viewport_y": vy,
            "viewport_w": 15,
            "viewport_h": 15,
        }

        try:
            resp = session.post(f"{base_url}/simulate", json=payload, timeout=30)
        except Exception as e:
            print(f"    [seed {seed_index}] query {i+1} error: {e}")
            time.sleep(1.0)
            continue

        if resp.status_code == 429:
            print(f"    [seed {seed_index}] rate limited / budget exhausted, stopping.")
            break
        if resp.status_code != 200:
            print(f"    [seed {seed_index}] query {i+1} HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(0.5)
            continue

        data = resp.json()
        grid = data["grid"]       # viewport_h x viewport_w
        viewport = data["viewport"]  # {x, y, w, h}
        used = data.get("queries_used", "?")
        max_q = data.get("queries_max", "?")

        print(f"    [seed {seed_index}] query {i+1}/{len(viewport_plan)} vp=({vx},{vy}) budget={used}/{max_q}")

        # Record each observed cell
        for dy, row in enumerate(grid):
            for dx, cell_code in enumerate(row):
                gx = viewport["x"] + dx
                gy = viewport["y"] + dy
                cls = TERRAIN_TO_CLASS.get(cell_code, 0)
                key = (gx, gy)
                if key not in observations:
                    observations[key] = []
                observations[key].append(cls)

        # Respect rate limit: 5 req/sec max
        time.sleep(delay)

    return observations
