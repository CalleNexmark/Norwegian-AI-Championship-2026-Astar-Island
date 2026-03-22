#!/usr/bin/env python3
"""
Astar Island pipeline — observe, predict, submit.

Usage:
    python main.py --token YOUR_JWT_TOKEN [--queries 50] [--dry-run]

Get token: log in at app.ainm.no → browser DevTools → Cookies → access_token
"""

import argparse
import json
import os
import sys
import time
import requests
import numpy as np

from observer import plan_viewports, find_shared_viewports, plan_full_coverage, run_observations
from predictor import build_prediction, estimate_survival_rate
from simulator import run_monte_carlo, calibrate_params, DEFAULT_PARAMS
from learned_predictor import load_model as load_ridge_model, predict_grid as predict_grid_ridge, blend_with_empirical as blend_ridge
from gbm_predictor import load_model as load_gbm_model, predict_grid as predict_grid_gbm, blend_with_empirical as blend_gbm
from backtest import apply_spatial_smooth, apply_global_multiplier

HISTORY_FILE   = "observations_history.json"
CALIBRATION_FILE = "calibration.json"


def load_cache(round_id: str) -> dict:
    """Load cached observations for this round from the history file."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE) as f:
        history = json.load(f)
    round_data = history.get(round_id, {})
    result = {}
    for seed_str, obs in round_data.items():
        result[int(seed_str)] = {
            tuple(map(int, k.split(","))): v
            for k, v in obs.items()
        }
    return result


def save_cache(round_id: str, all_observations: dict):
    """Append/update observations for this round in the multi-round history file."""
    history = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    history[round_id] = {
        str(seed_idx): {
            f"{x},{y}": classes
            for (x, y), classes in obs.items()
        }
        for seed_idx, obs in all_observations.items()
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)
    n_rounds = len(history)
    print(f"  Saved to {HISTORY_FILE} ({n_rounds} rounds in history)")


def load_calibration() -> dict | None:
    """Load simulator parameters from calibration.json if it exists."""
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE) as f:
        cal = json.load(f)
    params = cal.get("params", {})
    rounds = cal.get("fitted_from_rounds", [])
    survival = cal.get("per_round_survival_50yr", {})
    if params:
        # Merge: DEFAULT_PARAMS as base so any uncalibrated keys (e.g. ruin params,
        # which can't be estimated without initial Ruin cells) get sensible defaults.
        merged = DEFAULT_PARAMS.copy()
        merged.update({k: v for k, v in params.items() if not k.startswith("_")})
        print(f"  Loaded calibration from {len(rounds)} rounds: {rounds}")
        surv_vals = list(survival.values())
        if surv_vals:
            import numpy as _np
            print(f"  Historical survival range: {min(surv_vals):.3f}–{max(surv_vals):.3f} "
                  f"(mean={_np.mean(surv_vals):.3f})")
        return merged
    return None

BASE_URL = "https://api.ainm.no/astar-island"
TOTAL_BUDGET = 50
N_SEEDS = 5


def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def get_active_round(session: requests.Session) -> dict | None:
    resp = session.get(f"{BASE_URL}/rounds", timeout=15)
    resp.raise_for_status()
    rounds = resp.json()
    for r in rounds:
        if r["status"] == "active":
            return r
    return None


def get_round_detail(session: requests.Session, round_id: str) -> dict:
    resp = session.get(f"{BASE_URL}/rounds/{round_id}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_budget(session: requests.Session) -> dict:
    resp = session.get(f"{BASE_URL}/budget", timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return {"queries_used": 0, "queries_max": TOTAL_BUDGET}


def submit_prediction(
    session: requests.Session,
    round_id: str,
    seed_index: int,
    prediction: list,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        print(f"  [DRY RUN] would submit seed {seed_index}")
        return True

    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction,
    }
    resp = session.post(f"{BASE_URL}/submit", json=payload, timeout=60)
    if resp.status_code == 200:
        print(f"  seed {seed_index}: submitted OK")
        return True
    else:
        print(f"  seed {seed_index}: submit failed HTTP {resp.status_code}: {resp.text[:300]}")
        return False


def run(token: str, total_budget: int = TOTAL_BUDGET, dry_run: bool = False, prior_only: bool = False, deep_zone: bool = False, param_hunt: bool = False, adaptive: bool = False, estimation_queries: int = 2, simulate: bool = False, n_runs: int = 200, full_coverage: bool = False, learned: bool = False, model_name: str = "gbm", n0: float = 10.0):
    session = make_session(token)

    # 1. Find active round
    print("Checking for active round...")
    active = get_active_round(session)
    if not active:
        print("No active round found. Check back when a round is open.")
        sys.exit(0)

    round_id = active["id"]
    print(f"Active round #{active['round_number']} — id={round_id}")
    print(f"  Closes at: {active.get('closes_at', 'unknown')}")

    # 2. Check budget
    budget = get_budget(session)
    queries_used = budget.get("queries_used", 0)
    queries_max = budget.get("queries_max", TOTAL_BUDGET)
    queries_remaining = min(queries_max - queries_used, total_budget)
    print(f"Budget: {queries_used}/{queries_max} used, {queries_remaining} available (cap={total_budget})")

    if queries_remaining <= 0:
        print("No queries remaining. Submitting predictions from initial state only.")

    # 3. Get round details (initial states for all seeds)
    print("Fetching round details + initial states...")
    detail = get_round_detail(session, round_id)
    width = detail["map_width"]
    height = detail["map_height"]
    n_seeds = detail.get("seeds_count", N_SEEDS)
    initial_states = detail["initial_states"]
    print(f"Map: {width}x{height}, {n_seeds} seeds, {len(initial_states)} initial states")

    # 4. Distribute query budget across seeds
    queries_per_seed = queries_remaining // n_seeds if queries_remaining > 0 else 0
    leftover = queries_remaining - queries_per_seed * n_seeds

    print(f"Queries per seed: {queries_per_seed} (+{leftover} leftover to seed 0)")

    # Load prediction model
    # model_name: "gbm" (default), "ridge_v1" (legacy --learned), "ridge_v2"
    # --learned flag is a legacy alias for --model ridge_v1
    if learned and model_name == "gbm":
        model_name = "ridge_v1"   # backward compat: --learned without --model means ridge_v1

    learned_model = None
    ridge_model_for_ensemble = None  # only used when model_name == "gbm_ridge"
    use_learned = model_name in ("gbm", "ridge_v1", "ridge_v2", "gbm_ridge")
    if use_learned:
        print(f"Loading model ({model_name})...")
        if model_name in ("gbm", "gbm_ridge"):
            learned_model = load_gbm_model()
            if learned_model is None:
                print("  No gbm_model.pkl found — falling back to ridge_v1. "
                      "Run: python gbm_predictor.py --token TOKEN")
                model_name = "ridge_v1"
                learned_model = load_ridge_model()
            elif model_name == "gbm_ridge":
                ridge_model_for_ensemble = load_ridge_model()
                if ridge_model_for_ensemble is None:
                    print("  No learned_model.json found for ridge component — using GBM only.")
                    model_name = "gbm"
                else:
                    print(f"  Ensemble: GBM + ridge (50/50 blend)")
        else:
            learned_model = load_ridge_model()
        if learned_model is None:
            print("  No model file found — falling back to simulator/prior.")
            use_learned = False
        else:
            print(f"  Loaded model groups: {list(learned_model.keys())}")

    # Load calibrated simulator parameters (from calibrate_from_history.py)
    historical_params = None
    if simulate and not learned:
        print("Loading simulator calibration...")
        historical_params = load_calibration()
        if historical_params is None:
            print("  No calibration.json found — using DEFAULT_PARAMS. "
                  "Run calibrate_from_history.py --token TOKEN to build it.")

    # Load any cached observations from a previous run in this round
    # (skipped in prior_only mode — we want truly no observations)
    cached = {} if prior_only else load_cache(round_id)
    all_observations: dict = {}

    # R9 param-hunt: pick shared viewport positions across all seeds upfront
    shared_viewports = None
    if param_hunt and queries_remaining > 0:
        shared_viewports = find_shared_viewports(
            initial_states, width, height, n_viewports=queries_per_seed
        )
        print(f"Param-hunt mode: {len(shared_viewports)} shared viewports per seed")

    # R10 adaptive: phase 1 — shared viewports for survival rate estimation
    if adaptive and queries_remaining > 0:
        est_total = estimation_queries * n_seeds
        deep_total = queries_remaining - est_total
        deep_per_seed = deep_total // n_seeds
        print(f"Adaptive mode: {estimation_queries} estimation queries/seed ({est_total} total) "
              f"+ {deep_per_seed} deep zone queries/seed ({deep_total} total)")
        shared_viewports = find_shared_viewports(
            initial_states, width, height, n_viewports=estimation_queries
        )

    # 5. For each seed: phase 1 observations (estimation or standard)
    for seed_idx in range(n_seeds):
        print(f"\n--- Seed {seed_idx}/{n_seeds-1} ---")
        initial_state = initial_states[seed_idx]
        settlements = initial_state.get("settlements", [])
        print(f"  Initial settlements: {len(settlements)}")

        observations = cached.get(seed_idx, {})
        if observations:
            print(f"  Loaded {len(observations)} cached observations")

        seed_budget = queries_per_seed + (leftover if seed_idx == 0 else 0)
        if adaptive:
            seed_budget = estimation_queries  # phase 1 only for now

        if seed_budget > 0 and not dry_run:
            if shared_viewports is not None:
                vp_plan = shared_viewports
                label = "[PARAM HUNT]" if param_hunt else "[ESTIMATION]"
                print(f"  Using {len(vp_plan)} shared viewports {label}")
            else:
                vp_plan = plan_viewports(initial_state, width, height, seed_budget, deep_zone=deep_zone)
                print(f"  Planned {len(vp_plan)} viewport queries" + (" [DEEP ZONE]" if deep_zone else ""))

            new_obs = run_observations(
                session, BASE_URL, round_id, seed_idx, vp_plan, delay=0.25
            )
            for k, v in new_obs.items():
                if k in observations:
                    observations[k].extend(v)
                else:
                    observations[k] = v
            print(f"  Observed {len(observations)} unique cells total")
        elif dry_run and seed_budget > 0:
            vp_plan = shared_viewports if shared_viewports is not None else plan_viewports(initial_state, width, height, seed_budget, deep_zone=deep_zone)
            print(f"  [DRY RUN] would run {len(vp_plan)} viewport queries (skipping API calls)")
        else:
            print("  No budget — using initial state only")

        all_observations[seed_idx] = observations

        if not dry_run and not prior_only and observations:
            save_cache(round_id, all_observations)

        if seed_idx < n_seeds - 1:
            time.sleep(0.5)

    # Estimate survival rate from phase 1 observations
    survival_rate = None
    if (param_hunt or adaptive) and all_observations and not simulate:
        print("\nEstimating survival rate from phase 1 observations...")
        survival_rate = estimate_survival_rate(all_observations, initial_states)

    # Calibrate simulator parameters: start from historical calibration (if available),
    # then refine p_annual_survive + p_annual_expand from this round's observations
    sim_params = None
    if simulate and all_observations:
        base = (historical_params or DEFAULT_PARAMS).copy()
        print("\nCalibrating simulator parameters from current-round observations...")
        sim_params = calibrate_params(all_observations, initial_states, base)

    # Phase 2 — full map coverage OR deep zone (depending on --full-coverage flag)
    if adaptive and not dry_run and queries_remaining > 0:
        deep_per_seed = (queries_remaining - estimation_queries * n_seeds) // n_seeds
        phase2_label = "full coverage" if full_coverage else "deep zone"
        if deep_per_seed > 0:
            print(f"\n--- Phase 2: {phase2_label} ({deep_per_seed} queries/seed) ---")
            for seed_idx in range(n_seeds):
                initial_state = initial_states[seed_idx]
                observations = all_observations[seed_idx]
                if full_coverage:
                    # Cover tiles NOT visited in phase 1 (shared_viewports)
                    visited = set(shared_viewports) if shared_viewports else set()
                    vp_plan = plan_full_coverage(
                        initial_state, width, height, deep_per_seed, exclude=visited
                    )
                    print(f"\n  Seed {seed_idx}: {len(vp_plan)} full-coverage tiles "
                          f"(excluding {len(visited)} phase-1 tiles)")
                else:
                    vp_plan = plan_viewports(initial_state, width, height, deep_per_seed, deep_zone=True)
                print(f"\n  Seed {seed_idx}: {len(vp_plan)} deep zone queries")
                new_obs = run_observations(
                    session, BASE_URL, round_id, seed_idx, vp_plan, delay=0.25
                )
                for k, v in new_obs.items():
                    if k in observations:
                        observations[k].extend(v)
                    else:
                        observations[k] = v
                print(f"  Total observed cells: {len(observations)}")
                all_observations[seed_idx] = observations
                if not prior_only:
                    save_cache(round_id, all_observations)
                if seed_idx < n_seeds - 1:
                    time.sleep(0.5)

    # Re-estimate survival rate from ALL observations (phase 1 + phase 2 combined).
    # Phase 2 adds ~7× more settlement observations → much less noisy estimate.
    if adaptive and all_observations and not simulate:
        print("\nRe-estimating survival rate from all observations (phase 1 + phase 2)...")
        better_rate = estimate_survival_rate(all_observations, initial_states)
        if better_rate is not None:
            print(f"  Updated survival: {survival_rate:.3f} (phase 1) → {better_rate:.3f} (all obs)")
            survival_rate = better_rate

    # Build predictions and submit for all seeds
    print("\nBuilding and submitting predictions...")
    for seed_idx in range(n_seeds):
        print(f"\n--- Seed {seed_idx}/{n_seeds-1} ---")
        initial_state = initial_states[seed_idx]
        observations = all_observations.get(seed_idx, {})

        # --- Learned / GBM model path (preferred) ---
        if use_learned and learned_model is not None:
            sr = survival_rate if survival_rate is not None else 0.323  # historical mean fallback
            print(f"  Running {model_name} predictor (survival_rate={sr:.3f}, n0={n0})...")
            if model_name == "gbm_ridge":
                p_gbm   = predict_grid_gbm(initial_state, sr, learned_model)
                p_ridge = predict_grid_ridge(initial_state, sr, ridge_model_for_ensemble)
                blended = 0.5 * p_gbm + 0.5 * p_ridge
                blended = np.maximum(blended, 0.01)
                model_probs = blended / blended.sum(axis=2, keepdims=True)
                # Spatial smoothing on model prior (sigma=0.3, +0.28 LOOCV)
                ig_grid = initial_state["grid"]
                model_probs = apply_spatial_smooth(model_probs, sigma=0.3,
                                                    initial_grid=ig_grid)
                # Bayesian blend with observations
                pred_arr = blend_gbm(model_probs, observations, width, height, n0=n0)
                # Global multiplier: correct class-frequency bias from observations (+1.25 LOOCV)
                if observations:
                    pred_arr = apply_global_multiplier(pred_arr, observations, ig_grid,
                                                        clip_low=0.5, clip_high=2.0)
                prediction = pred_arr.tolist()
            elif model_name == "gbm":
                model_probs = predict_grid_gbm(initial_state, sr, learned_model)
                prediction  = blend_gbm(model_probs, observations, width, height, n0=n0).tolist()
            else:
                model_probs = predict_grid_ridge(initial_state, sr, learned_model)
                prediction  = blend_ridge(model_probs, observations, width, height, n0=n0).tolist()
            arr = np.array(prediction)
            print(f"  Prediction: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, "
                  f"empirical cells={len(observations)}")

        # --- Simulator path (fallback when --simulate without --learned) ---
        elif simulate and sim_params is not None:
            print(f"  Running {n_runs} Monte Carlo simulations...")
            simulator_probs = run_monte_carlo(
                initial_state, width, height, sim_params, n_runs=n_runs, rng_seed=seed_idx
            )
            n_obs = len(observations)
            total_cells = width * height
            print(f"  Simulator done. Observed {n_obs}/{total_cells} cells "
                  f"({100*n_obs/total_cells:.1f}%) will use empirical; rest use simulator.")
            prediction = build_prediction(
                initial_state, observations, width, height,
                survival_rate=survival_rate,
                simulator_probs=simulator_probs,
            )

        # --- Static prior path (legacy) ---
        else:
            prediction = build_prediction(
                initial_state, observations, width, height,
                survival_rate=survival_rate,
            )

        arr = np.array(prediction)
        row_sums = arr.reshape(-1, 6).sum(axis=1)
        assert arr.shape == (height, width, 6), f"Wrong shape: {arr.shape}"
        assert np.allclose(row_sums, 1.0, atol=0.01), f"Rows don't sum to 1: min={row_sums.min():.4f}"
        assert arr.min() >= 0, f"Negative probability: {arr.min()}"
        print(f"  Prediction validated: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

        submit_prediction(session, round_id, seed_idx, prediction, dry_run=dry_run)

        if seed_idx < n_seeds - 1:
            time.sleep(0.5)

    print("\nDone! All seeds submitted.")


def main():
    parser = argparse.ArgumentParser(description="Astar Island prediction pipeline")
    parser.add_argument("--token", required=True, help="JWT access_token from app.ainm.no cookies")
    parser.add_argument("--queries", type=int, default=TOTAL_BUDGET, help="Total query budget (default 50)")
    parser.add_argument("--dry-run", action="store_true", help="Plan viewports but don't call API or submit")
    parser.add_argument("--prior-only", action="store_true", help="Skip all queries, submit calibrated-prior predictions only (Round 7 experiment)")
    parser.add_argument("--deep-zone", action="store_true", help="Spend all queries on densest 15x15 settlement zone (Round 8 experiment)")
    parser.add_argument("--param-hunt", action="store_true", help="Same viewports across all seeds, estimate survival rate cross-seed (Round 9 experiment)")
    parser.add_argument("--adaptive", action="store_true", help="Phase 1: estimate survival rate, Phase 2: deep zone (Round 10 experiment)")
    parser.add_argument("--estimation-queries", type=int, default=2, help="Queries per seed for survival rate estimation in adaptive mode (default 2)")
    parser.add_argument("--simulate", action="store_true", help="Run Monte Carlo forward simulator for unobserved cells")
    parser.add_argument("--n-runs", type=int, default=200, help="Monte Carlo simulations per seed (default 200)")
    parser.add_argument("--full-coverage", action="store_true", help="Phase 2: cover whole map with non-overlapping tiles instead of deep zone (Round 15+)")
    parser.add_argument("--learned", action="store_true", help="[legacy] Use ridge_v1 predictor. Prefer --model ridge_v1")
    parser.add_argument("--model", default="gbm",
                        choices=["gbm", "ridge_v1", "ridge_v2", "gbm_ridge"],
                        help="Prediction model: gbm (default), gbm_ridge (50/50 GBM+ridge ensemble), "
                             "ridge_v1 (legacy), ridge_v2 (interactions)")
    parser.add_argument("--n0", type=float, default=10.0,
                        help="Bayesian blending strength: higher = trust model more vs observations (default 5.0)")
    args = parser.parse_args()

    run(token=args.token, total_budget=0 if args.prior_only else args.queries,
        dry_run=args.dry_run, prior_only=args.prior_only, deep_zone=args.deep_zone,
        param_hunt=args.param_hunt, adaptive=args.adaptive,
        estimation_queries=args.estimation_queries, simulate=args.simulate,
        n_runs=args.n_runs, full_coverage=args.full_coverage, learned=args.learned,
        model_name=args.model, n0=args.n0)


if __name__ == "__main__":
    main()
