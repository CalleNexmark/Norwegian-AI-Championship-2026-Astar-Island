# Astar Island — NM i AI 2026 Competition

## Challenge Summary
- **Organiser:** app.ainm.no / api.ainm.no/astar-island
- **Task:** Observe a stochastic Norse-civilization simulator through limited viewports, predict final terrain probability distributions across the full map
- **Map:** 40×40 grid, 5 random seeds per round, 50 total queries (10/seed), 50-year simulation
- **Viewport:** 15×15 cells per query
- **Prediction format:** H×W×6 float tensor, each cell sums to 1.0, floor 0.01 (never assign 0)
- **Scoring:** `score = 100 × exp(-3 × entropy_weighted_KL)` — only dynamic cells count; static ocean/mountain excluded
- **Competition dates:** March 19–22, 2026

## Terrain Codes → Prediction Classes
| Code | Terrain | Class |
|---|---|---|
| 10 | Ocean | 0 (static) |
| 5 | Mountain | 5 (static) |
| 0/11 | Empty/Plains | 0 |
| 1 | Settlement | 1 |
| 2 | Port | 2 |
| 3 | Ruin | 3 |
| 4 | Forest | 4 |

## Files
| File | Purpose |
|---|---|
| `main.py` | Orchestration — observations, calibration, prediction, submission |
| `observer.py` | Viewport strategy: `plan_viewports`, `plan_full_coverage`, `find_shared_viewports`, `run_observations` |
| `predictor.py` | Legacy static-prior predictor (fallback) |
| `simulator.py` | Stochastic CA forward simulator — `run_monte_carlo`, `calibrate_params` |
| `learned_predictor.py` | Ridge regression predictor — `build_and_save`, `load_model`, `predict_grid`, `blend_with_empirical` |
| `calibrate_from_history.py` | Fetch all round ground truths, fit all simulator params → `calibration.json` |
| `calibration.json` | Fitted simulator params + per-round survival rates (13 rounds) |
| `learned_model.json` | Trained ridge regression weights (4 terrain groups: empty, settle, port, forest) |
| `observations_history.json` | Multi-round observation cache keyed by round_id |

## Recommended Command (R15+)
```bash
python main.py --token TOKEN \
  --adaptive --estimation-queries 2 \
  --full-coverage \
  --learned
```

### What each flag does
- `--adaptive`: Phase 1 (2 queries/seed shared) → estimate survival rate; Phase 2 (8 queries/seed) → coverage
- `--full-coverage`: Phase 2 uses non-overlapping tiles to cover the WHOLE MAP instead of repeating one deep zone
- `--learned`: Uses `learned_model.json` ridge regression for unobserved cells + Bayesian blend for observed cells

### Coverage arithmetic
- 40×40 map needs 9 non-overlapping 15×15 tiles: x ∈ {0,15,25}, y ∈ {0,15,25}
- Phase 1 (2 shared tiles) + Phase 2 (7 distinct tiles) = 9 tiles = **full map coverage**
- One remaining query repeats the densest tile

## After Each Completed Round — Refresh Historical Data
```bash
python calibrate_from_history.py --token TOKEN   # updates calibration.json
python learned_predictor.py --token TOKEN        # retrains learned_model.json
```

## Key Findings from Historical Data (R1–R13)
### Per-round 50yr survival rates
| Round | Survival | Type |
|---|---|---|
| R3 | 1.8% | Catastrophic |
| R8 | 6.6% | Catastrophic |
| R10 | 5.8% | Catastrophic |
| R4, R9, R13 | 23–27% | Hard |
| R5 | 33% | Medium |
| R1, R2, R6, R7 | 42–43% | Normal |
| R11 | 49.6% | Thriving |
| R12 | 60.1% | Very thriving |
| R14 | ~47% (estimated) | Thriving |

**Survival variance is extreme (2–60%) — always estimate from phase 1 observations before predicting.**

### Calibrated simulator parameters (from 13 rounds)
| Parameter | Calibrated | Old hardcoded |
|---|---|---|
| p_annual_survive | 0.9767 | 0.982 |
| p_port_survive | 0.9731 | 0.985 |
| p_port_form | **0.0059** | **0.08** (13× wrong) |
| p_annual_expand | 0.00495 | 0.012 |
| p_forest_clear | 0.00257 | 0.003 |
| p_ruin_* | uncalibrated (no initial Ruin cells) | hardcoded defaults |

### Learned model inputs/outputs
- **Features (7):** n_adj_settle_r1, n_adj_settle_r3, n_adj_settle_r7, is_coastal, n_adj_forest_r1, n_adj_ocean_r3, survival_rate
- **Training:** 104,000 cells (13 rounds × 5 seeds × 1600 cells), per-group ridge regression (α=1.0)
- **Groups trained:** empty (63k), forest (22k), settle (2.9k), port (119)
- **Ruin/Mountain/Ocean:** static fallback (no initial examples)

### Model sensitivity check (R12 real map, settlement cells)
| survival_rate | P(Empty) | P(Settle) | P(Port) |
|---|---|---|---|
| 0.05 | 0.611 | 0.059 | 0.014 |
| 0.25 | 0.481 | 0.244 | 0.014 |
| 0.60 | 0.248 | 0.578 | 0.014 |

Correctly learned: high survival → settlements survive; low survival → collapse to Empty.

## Round Strategy History
| Round | Mode | Key experiment |
|---|---|---|
| R1–R6 | `standard` | Greedy viewport coverage + static terrain priors |
| R5/R6 | — | Calibrated BASE_PROBS from ground truth |
| R7 | `--prior-only` | Ablation: 0 queries, pure calibrated prior |
| R8 | `--deep-zone` | 10× same densest 15×15 zone per seed |
| R9 | `--param-hunt` | Shared viewports cross-seed → survival rate MLE |
| R10 | `--adaptive` | 2 estimation + 8 deep zone queries/seed |
| R11–R13 | `--adaptive --simulate` | CA forward simulator (calibrated) |
| R14 | `--adaptive --simulate` | First run with historical calibration (13 rounds) |
| R15+ | `--adaptive --full-coverage --learned` | **Full map coverage + learned model** |

## API Endpoints
- Base: `https://api.ainm.no/astar-island`
- Auth: `Bearer {JWT}` — get token from app.ainm.no cookies → `access_token`
- `GET /rounds` — list all rounds with status
- `GET /rounds/{id}` — round details + initial states for all seeds
- `GET /budget` — remaining query budget
- `POST /simulate` — observe viewport (costs 1 query)
- `POST /submit` — submit H×W×6 prediction tensor
- `GET /analysis/{round_id}/{seed_index}` — post-round ground truth + your prediction
- `GET /my-rounds` — your scores across all rounds

## Important Rules
- Never assign 0.0 probability to any class (KL divergence → ∞); floor = 0.01
- Only dynamic cells (non-ocean, non-mountain) contribute to score
- Later rounds weighted more: `best_score × 1.05^round_number`
- Budget: 50 queries/round shared across 5 seeds (10/seed)
