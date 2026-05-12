"""
03_optimize_simulated.py: MIQP Portfolio Optimisation (Simulated Dataset)
===========================================================================
ESG-Constrained Portfolio Optimisation - Independent Research Project

PIPELINE POSITION:
  02_preprocess_simulated.py → data/clean/meta_preprocessed.csv
                                data/clean/sigma_*_annual.npy
  03_optimize_simulated.py (this file) → data/results/frontier_*.csv
                                          data/results/frontier_ledoit_wolf_esg55.csv
                                          data/results/esg_sensitivity.csv
                                          data/results/baseline_weights.csv
                                          data/results/cov_comparison.csv
                                          data/results/selected_vol_cap.txt
                                          data/results/optimisation_log.txt
                                          figures/step3_frontier.png
                                          figures/step3_esg_sensitivity.png
                                          figures/step3_cov_comparison.png

MODEL: Mixed-Integer Quadratic Programme (MIQP)
  Objective:  maximise  w' mu
  s.t.        w' Sigma w  <= vol_cap^2          (vol-cap, swept for frontier)
              sum(w)       = 1                   (budget)
              w' esg       >= esg_floor          (ESG constraint)
              w_i          <= W_MAX * z_i        (upper bound per stock)
              w_i          >= W_MIN * z_i        (lower bound if selected)
              sum(z_i)     <= K_MAX              (max holdings)
              sum(z_i)     >= K_MIN              (min holdings, enforces diversification)
              w_i          >= 0                  (long-only)
              z_i          in {0,1}              (binary selection)
              sum_j(w_j)   <= SECTOR_CAP  forall sector  (sector diversification)

FRONTIER METHOD: vol-cap approach (not quadratic objective sweep)
  vol_caps swept on a geometric grid (geomspace) — denser at low vol
  where the frontier curves, sparser at high vol where it is flat.

VOL CAP SELECTION (data-driven):
  Max-Sharpe point in the central 60% of the feasible unconstrained Ledoit-Wolf
  frontier, after trimming the bottom 20% and top 20% of the feasible vol range.

TASKS:
  1a. Frontier — 3 covariance estimators, NO ESG constraint (true unconstrained)
  1b. Frontier — Ledoit-Wolf only, ESG floor = 55  (constrained)
  2.  ESG floor sweep — all 3 estimators, 6 floors [40,45,50,55,60,65], fixed vol cap
  3.  Baseline portfolio — LW, ESG floor = 55
  4.  Covariance comparison table (unconstrained frontiers only)

THREE COVARIANCE ESTIMATORS: sample, Ledoit-Wolf, OAS
ESG FLOOR SWEEP: 40, 45, 50, 55, 60, 65
SOLVER: Gurobi 13.0 via cvxpy (academic licence)

Run:
    python src/03_optimize_simulated.py

Requires:
    pip install cvxpy gurobipy numpy pandas matplotlib
    Gurobi academic licence activated
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# PARAMETERS — change only here, never inline
# ═════════════════════════════════════════════════════════════

CLEAN_DIR   = "data/clean"
RESULTS_DIR = "data/results"
FIGURES_DIR = "figures"

# ── Portfolio constraints ──────────────────────────────────
W_MAX   = 0.20      # max weight per stock (20%)
W_MIN   = 0.01      # min weight if stock is selected (1%)
K_MAX   = 100        # max number of holdings
K_MIN   = 50        # min number of holdings
                    # NOTE: K_MIN >= ceil(1/W_MAX) = 5 is the mathematical minimum
                    # for feasibility, but 15 is used to enforce meaningful diversification.
                    # A portfolio of 5 stocks at 20% each is technically valid but
                    # not a credible institutional portfolio.

# ── Sector constraint ──────────────────────────────────────
SECTOR_CAP = 0.25   # max total weight in any single sector (40%)
                    # set to 1.0 to disable

# ── Frontier: vol-cap sweep ────────────────────────────────
# Volatility caps in annualised terms.
# Simulated data has avg ann vol ~1.46, so caps go up to 3.0.
# geomspace used instead of linspace: denser grid at the low end
# where the frontier curves sharply, sparser at the top where
# the frontier is flat. This gives better coverage of the
# efficient part of the frontier.
N_FRONTIER_POINTS = 30          # number of frontier points
VOL_CAP_MIN       = 0.30        # min annualised vol cap (30%)
                                 # set above the minimum feasible vol
                                 # to avoid wasting solves on infeasible points
VOL_CAP_MAX       = 3.00        # max annualised vol cap (300%)

# ── ESG floor sweep ───────────────────────────────────────
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# ── Which mu to use ───────────────────────────────────────
# Simulated main specification:      mu_native  = original FICO metadata mu
# Simulated robustness check:        mu_winsor  = winsorised mu_native (p1/p99)
# mu_empirical_log available in meta for symmetry check only.
MU_COLUMN            = "mu_native"  # simulated main specification (FICO metadata mu)
MU_ROBUSTNESS_COLUMN = "mu_winsor"  # robustness: winsorised FICO mu
USE_ROBUSTNESS_MU    = False        # set True to run robustness check with mu_winsor

# ── Covariance estimators to run ──────────────────────────
COV_ESTIMATORS = {
    "sample":      "sigma_sample_annual.npy",
    "ledoit_wolf": "sigma_lw_annual.npy",
    "oas":         "sigma_oas_annual.npy",
}

# ── Gurobi solver settings ────────────────────────────────
SOLVER_VERBOSE     = False
SOLVER_TIME_LIMIT  = 240        # seconds per solve
SOLVER_MIP_GAP     = 1e-4       # relative MIP optimality gap

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# OPTIMISATION LOG
# ─────────────────────────────────────────────────────────────
log_lines = []

def log(msg: str) -> None:
    """Print to console and append to the optimisation log buffer."""
    print(msg)
    log_lines.append(msg)

# ═════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════
log("=" * 65)
log("STEP 3 — MIQP PORTFOLIO OPTIMISATION")
log("=" * 65)

meta = pd.read_csv(f"{CLEAN_DIR}/meta_preprocessed.csv", index_col=0)
tickers = list(meta.index)
N       = len(tickers)

# Select which mu column to use — decision stays here, not inside the optimizer logic
active_mu_col = MU_ROBUSTNESS_COLUMN if USE_ROBUSTNESS_MU else MU_COLUMN
mu_vec  = meta[active_mu_col].values.astype(float)
esg_vec = meta["esg"].values.astype(float)   # common 0-100 scale (Option A)

log(f"\n  Universe:       {N} stocks")
log(f"  Mu column:      '{active_mu_col}'  "
    f"({'robustness run — mu_winsor' if USE_ROBUSTNESS_MU else 'main specification — FICO metadata mu'})  "
    f"mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}")
log(f"  ESG column:     esg  (common 0-100 scale, Option A)  "
    f"mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
log(f"  Constraints:    W_MAX={W_MAX}  W_MIN={W_MIN}  "
    f"K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
log(f"  ESG floors:     {ESG_FLOORS}")
log(f"  Frontier pts:   {N_FRONTIER_POINTS}  "
    f"vol_cap=[{VOL_CAP_MIN}, {VOL_CAP_MAX}]")

# Sector index map: {sector_name: [list of stock indices]}
sector_map = {}
if "sector" in meta.columns:
    for sec, grp in meta.groupby("sector"):
        sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
    log(f"  Sectors:        {list(sector_map.keys())}")
else:
    log("  No sector column — sector cap constraint disabled.")

# ═════════════════════════════════════════════════════════════
# CORE OPTIMISATION FUNCTION
# ═════════════════════════════════════════════════════════════

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=0.0, apply_esg=True,
               sector_map=None,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN,
               sector_cap=SECTOR_CAP):
    """
    Solve one MIQP instance via cvxpy + Gurobi.

    Objective:  maximise  w' mu
    Subject to: w' Sigma w  <= vol_cap^2     (vol-cap frontier)
                sum(w)       = 1              (budget)
                w' esg       >= esg_floor     (ESG floor — only if apply_esg=True)
                w_i          <= w_max * z_i   (upper bound)
                w_i          >= w_min * z_i   (lower bound if selected)
                sum(z_i)     <= k_max         (max holdings)
                sum(z_i)     >= k_min         (min holdings)
                w_i          >= 0             (long-only)
                z_i          in {0,1}         (binary selection)
                sector weight <= sector_cap   (sector diversification)

    Parameters
    ----------
    Sigma     : (N, N) annualised covariance matrix
    mu        : (N,) expected returns
    esg       : (N,) ESG scores on 0-100 scale
    vol_cap   : annualised volatility cap (scalar)
    esg_floor : minimum portfolio ESG score (used only if apply_esg=True)
    apply_esg : if False, the ESG constraint is omitted entirely.
                This gives the TRUE unconstrained benchmark, not a weak floor.
    sector_map: dict {sector_name: [stock indices]} — None to disable

    Returns
    -------
    dict with keys: status, weights, ret, vol, sharpe, esg_score, n_holdings
    """
    n = len(mu)

    w = cp.Variable(n, nonneg=True)      # continuous weights
    z = cp.Variable(n, boolean=True)     # binary selection

    # ── Objective: maximise expected return ──────────────────
    objective = cp.Maximize(mu @ w)

    # ── Constraints ──────────────────────────────────────────
    constraints = [
        cp.sum(w) == 1,                                 # budget
        cp.quad_form(w, Sigma) <= vol_cap ** 2,         # vol cap
        w <= w_max * z,                                 # upper bound
        w >= w_min * z,                                 # lower bound (if selected)
        cp.sum(z) <= k_max,                             # max holdings
        cp.sum(z) >= k_min,                             # min holdings
    ]

    # ESG floor — omitted entirely when apply_esg=False
    if apply_esg:
        constraints.append(esg @ w >= esg_floor)        # ESG floor

    # Sector cap constraints
    if sector_map:
        for sec_name, idx in sector_map.items():
            if len(idx) > 0:
                constraints.append(cp.sum(w[idx]) <= sector_cap)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(
            solver=cp.GUROBI,
            verbose=SOLVER_VERBOSE,
            TimeLimit=SOLVER_TIME_LIMIT,
            MIPGap=SOLVER_MIP_GAP,
        )
    except Exception as e:
        return {"status": f"solver_error: {e}", "weights": None}

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "weights": None}

    w_val = np.array(w.value).flatten()
    w_val = np.clip(w_val, 0, None)        # numerical safety: no negative weights
    w_val /= w_val.sum()                   # renormalise to exactly 1

    port_ret   = float(mu @ w_val)
    port_var   = float(w_val @ Sigma @ w_val)
    port_vol   = float(np.sqrt(max(port_var, 0)))
    port_sharpe= port_ret / port_vol if port_vol > 1e-8 else np.nan
    port_esg   = float(esg @ w_val)
    n_holdings = int((w_val > 1e-4).sum())

    return {
        "status":    prob.status,
        "weights":   w_val,
        "ret":       port_ret,
        "vol":       port_vol,
        "sharpe":    port_sharpe,
        "esg_score": port_esg,
        "n_holdings":n_holdings,
    }


# ═════════════════════════════════════════════════════════════
# TASK 1 — EFFICIENT FRONTIER
#   1a. TRUE UNCONSTRAINED (no ESG constraint at all) — 3 covariance estimators
#       This is the genuine benchmark. ESG floor = 50 is still a constraint
#       and would artificially inflate the benchmark performance vs floor=55.
#   1b. ESG-CONSTRAINED (floor = 55) — Ledoit-Wolf only
#       The main result. Compared against 1a to show the true ESG cost.
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("TASK 1a — True Unconstrained Frontier  (no ESG constraint, 3 estimators)")
log("=" * 65)

vol_caps = np.geomspace(VOL_CAP_MIN, VOL_CAP_MAX, N_FRONTIER_POINTS)
log(f"  Vol caps (geomspace): min={VOL_CAP_MIN}  max={VOL_CAP_MAX}  n={N_FRONTIER_POINTS}")
log(f"  First 5: {np.round(vol_caps[:5], 3).tolist()}")
frontier_results = {}   # {cov_name: DataFrame}
sigma_cache      = {}   # {cov_name: processed Sigma} — reused in all tasks

for cov_name, cov_file in COV_ESTIMATORS.items():
    Sigma = np.load(f"{CLEAN_DIR}/{cov_file}")

    # Force symmetry and PSD (numerical safety) — done once per estimator
    Sigma = (Sigma + Sigma.T) / 2
    Sigma += np.eye(N) * 1e-8
    sigma_cache[cov_name] = Sigma   # store for reuse

    log(f"\n  [{cov_name}]  sigma file: {cov_file}")
    rows = []
    t0   = time.time()

    for vc in vol_caps:
        result = solve_miqp(
            Sigma, mu_vec, esg_vec,
            vol_cap=vc,
            apply_esg=False,          # TRUE unconstrained — no ESG floor at all
            sector_map=sector_map if sector_map else None,
        )
        if result["weights"] is not None:
            rows.append({
                "vol_cap":    vc,
                "ret":        result["ret"],
                "vol":        result["vol"],
                "sharpe":     result["sharpe"],
                "esg_score":  result["esg_score"],
                "n_holdings": result["n_holdings"],
                "status":     result["status"],
                "cov":        cov_name,
                "mu_source":  active_mu_col,
            })
            log(f"    vc={vc:.3f}  ret={result['ret']:.4f}  "
                f"vol={result['vol']:.4f}  sharpe={result['sharpe']:.3f}  "
                f"ESG={result['esg_score']:.1f}  n={result['n_holdings']}")
        else:
            log(f"    vc={vc:.3f}  INFEASIBLE / ERROR: {result['status']}")

    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s — {len(rows)}/{N_FRONTIER_POINTS} points feasible")

    df_frontier = pd.DataFrame(rows)
    frontier_results[cov_name] = df_frontier
    df_frontier.to_csv(f"{RESULTS_DIR}/frontier_{cov_name}_unconstrained.csv", index=False)
    log(f"  Saved: {RESULTS_DIR}/frontier_{cov_name}_unconstrained.csv")

# ── Task 1b: ESG-constrained frontier (floor=55, LW only) ──────────────
# True constrained frontier. Compared against Task 1a (LW unconstrained)
# to show the exact ESG performance cost.
log("\n" + "=" * 65)
log("TASK 1b — ESG-Constrained Frontier  (floor = 55, Ledoit-Wolf only)")
log("=" * 65)

rows_55 = []
t0 = time.time()
for vc in vol_caps:
    result = solve_miqp(
        sigma_cache["ledoit_wolf"], mu_vec, esg_vec,
        vol_cap=vc,
        apply_esg=True,
        esg_floor=55.0,
        sector_map=sector_map if sector_map else None,
    )
    if result["weights"] is not None:
        rows_55.append({
            "vol_cap":    vc,
            "ret":        result["ret"],
            "vol":        result["vol"],
            "sharpe":     result["sharpe"],
            "esg_score":  result["esg_score"],
            "n_holdings": result["n_holdings"],
            "status":     result["status"],
            "cov":        "ledoit_wolf_esg55",
            "mu_source":  active_mu_col,
        })
        log(f"    vc={vc:.3f}  ret={result['ret']:.4f}  "
            f"vol={result['vol']:.4f}  sharpe={result['sharpe']:.3f}  "
            f"ESG={result['esg_score']:.1f}  n={result['n_holdings']}")
    else:
        log(f"    vc={vc:.3f}  INFEASIBLE: {result['status']}")

df_frontier_55 = pd.DataFrame(rows_55)
frontier_results["ledoit_wolf_esg55"] = df_frontier_55
df_frontier_55.to_csv(f"{RESULTS_DIR}/frontier_ledoit_wolf_esg55.csv", index=False)
elapsed = time.time() - t0
log(f"  Done in {elapsed:.1f}s — {len(rows_55)}/{N_FRONTIER_POINTS} points feasible")
log(f"  Saved: {RESULTS_DIR}/frontier_ledoit_wolf_esg55.csv")


# ═════════════════════════════════════════════════════════════
# VOL CAP SELECTION — data-driven from the unconstrained LW frontier
#
# Procedure:
#   1. Keep only feasible points from the LW unconstrained frontier.
#   2. Trim the bottom 20% of the feasible vol range (near-infeasibility
#      boundary — these points are too close to structural infeasibility).
#   3. Trim the top 20% (overly aggressive extreme of the frontier).
#   4. Within the central 60%, pick the max-Sharpe point — the tangency
#      portfolio, economically the most representative choice.
#
# This selected_vol_cap is reported here and used by step3b_esg_sweep.py
# (ESG floor sweep + baseline portfolio).  Step 3 itself does NOT run
# those tasks — it only builds the frontier and identifies the cap.
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("VOL CAP SELECTION — data-driven from unconstrained LW frontier")
log("=" * 65)

_TRIM_LOW  = 0.20
_TRIM_HIGH = 0.20
_feasible_lw = frontier_results["ledoit_wolf"]
_feasible_lw = _feasible_lw[_feasible_lw["status"].isin(["optimal", "optimal_inaccurate"])].copy()
_feasible_lw = _feasible_lw.sort_values("vol_cap").reset_index(drop=True)

if _feasible_lw.empty:
    raise RuntimeError("LW unconstrained frontier has no feasible points — cannot select vol cap.")

_vol_min  = _feasible_lw["vol_cap"].min()
_vol_max  = _feasible_lw["vol_cap"].max()
_vol_rng  = _vol_max - _vol_min
_low_cut  = _vol_min + _TRIM_LOW  * _vol_rng
_high_cut = _vol_max - _TRIM_HIGH * _vol_rng
_central  = _feasible_lw[(_feasible_lw["vol_cap"] >= _low_cut) & (_feasible_lw["vol_cap"] <= _high_cut)]
if _central.empty:
    _central = _feasible_lw   # fallback: use full feasible set

_best_row        = _central.loc[_central["sharpe"].idxmax()]
selected_vol_cap = float(_best_row["vol_cap"])

log(f"  Feasible frontier: {len(_feasible_lw)} points  "
    f"vol range [{_vol_min:.3f}, {_vol_max:.3f}]")
log(f"  Central band (trim {_TRIM_LOW:.0%}/{_TRIM_HIGH:.0%}): "
    f"[{_low_cut:.3f}, {_high_cut:.3f}]  →  {len(_central)} candidates")
log(f"  Selected vol cap : {selected_vol_cap:.4f}  "
    f"(max-Sharpe = {_best_row['sharpe']:.4f},  "
    f"ret = {_best_row['ret']:.4f},  vol = {_best_row['vol']:.4f})")
log(f"  → This value is also used for Tasks 2 and 3 below, and by step3b_esg_sweep.py.")
with open(f"{RESULTS_DIR}/selected_vol_cap.txt", "w") as f:
    f.write(str(selected_vol_cap))


# ═════════════════════════════════════════════════════════════
# TASK 2 — ESG FLOOR SWEEP
#           All 3 estimators, 6 floors [40,45,50,55,60,65], data-driven vol cap
#           Mirrors Step 7 structure exactly for cross-dataset comparison
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log(f"TASK 2 — ESG Floor Sweep  (all 3 estimators, vol cap = {selected_vol_cap:.4f})")
log("=" * 65)

_idx_esg_desc = np.argsort(esg_vec)[::-1]
_ub_weights   = np.zeros(len(esg_vec))
_budget       = 1.0
for _i in _idx_esg_desc:
    _alloc = min(W_MAX, _budget)
    _ub_weights[_i] = _alloc
    _budget -= _alloc
    if _budget <= 1e-9:
        break
max_achievable_esg = float(esg_vec @ _ub_weights)
log(f"  Max achievable portfolio ESG (upper bound, greedy): {max_achievable_esg:.2f}")

esg_rows = []
for cov_name, Sigma in sigma_cache.items():
    log(f"\n  [{cov_name}]")
    t0 = time.time()
    for floor in ESG_FLOORS:
        if max_achievable_esg < floor:
            log(f"    floor={floor}  SKIPPED — max achievable ESG={max_achievable_esg:.1f} < {floor}.")
            continue
        log(f"    floor={floor}  solving...")
        result = solve_miqp(
            Sigma, mu_vec, esg_vec,
            vol_cap=selected_vol_cap,
            apply_esg=True,
            esg_floor=float(floor),
            sector_map=sector_map if sector_map else None,
        )
        if result["weights"] is not None:
            esg_rows.append({
                "cov_estimator": cov_name,
                "esg_floor":     floor,
                "ret":           result["ret"],
                "vol":           result["vol"],
                "sharpe":        result["sharpe"],
                "esg":           result["esg_score"],
                "n_holdings":    result["n_holdings"],
                "status":        result["status"],
                "mu_source":     active_mu_col,
            })
            log(f"    floor={floor}  ret={result['ret']:.4f}  "
                f"vol={result['vol']:.4f}  sharpe={result['sharpe']:.3f}  "
                f"ESG={result['esg_score']:.1f}  n={result['n_holdings']}")
        else:
            log(f"    floor={floor}  INFEASIBLE: {result['status']}")
    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s")

df_esg = pd.DataFrame(esg_rows)
df_esg.to_csv(f"{RESULTS_DIR}/esg_sensitivity.csv", index=False)
log(f"\n  Saved: {RESULTS_DIR}/esg_sensitivity.csv")


# ═════════════════════════════════════════════════════════════
# TASK 3 — BASELINE PORTFOLIO
#           Ledoit-Wolf, ESG floor = 55, data-driven vol cap
#           Used as the "main result" portfolio throughout the analysis
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log(f"TASK 3 — Baseline Portfolio  (LW, ESG floor=55, vol cap={selected_vol_cap:.4f})")
log("=" * 65)

n_eligible_base = (esg_vec >= 55).sum()
log(f"  Vol cap: {selected_vol_cap:.4f}   ESG floor: 55.0")
log(f"  Eligible stocks (ESG >= 55): {n_eligible_base}  (K_MIN={K_MIN})")
if n_eligible_base < K_MIN:
    log(f"  *** WARNING: fewer eligible stocks than K_MIN — baseline likely infeasible ***")

baseline_result = solve_miqp(
    sigma_cache["ledoit_wolf"], mu_vec, esg_vec,
    vol_cap=selected_vol_cap,
    apply_esg=True,
    esg_floor=55.0,
    sector_map=sector_map if sector_map else None,
)

if baseline_result["weights"] is not None:
    w_base = baseline_result["weights"]
    log(f"\n  Baseline portfolio:")
    log(f"    Return:     {baseline_result['ret']:.4f}")
    log(f"    Volatility: {baseline_result['vol']:.4f}")
    log(f"    Sharpe:     {baseline_result['sharpe']:.4f}")
    log(f"    ESG score:  {baseline_result['esg_score']:.2f}")
    log(f"    Holdings:   {baseline_result['n_holdings']}")

    baseline_weights = pd.DataFrame({
        "ticker":    tickers,
        "weight":    w_base,
        "mu":        mu_vec,
        "esg":       esg_vec,
        "sector":    meta["sector"].values if "sector" in meta.columns else "N/A",
        "mu_source": active_mu_col,
    })
    baseline_weights = baseline_weights[baseline_weights["weight"] > 1e-4].copy()
    baseline_weights = baseline_weights.sort_values("weight", ascending=False)
    baseline_weights.to_csv(f"{RESULTS_DIR}/baseline_weights.csv", index=False)
    log(f"\n  Saved: {RESULTS_DIR}/baseline_weights.csv  "
        f"({len(baseline_weights)} stocks with w > 0.01%)")

    log(f"\n  Top 10 holdings:")
    log(f"  {'Ticker':<12}  {'Weight':>8}  {'mu':>8}  {'ESG':>8}  {'Sector'}")
    for _, row in baseline_weights.head(10).iterrows():
        log(f"  {row['ticker']:<12}  {row['weight']:>8.4f}  "
            f"{row['mu']:>8.4f}  {row['esg']:>8.1f}  {row['sector']}")
else:
    log(f"  Baseline INFEASIBLE: {baseline_result['status']}")
    log("  *** Try lowering K_MIN, relaxing W_MIN, or reducing ESG floor ***")


# ═════════════════════════════════════════════════════════════
# TASK 4 — COVARIANCE COMPARISON TABLE
#           For each estimator: frontier stats at matched vol levels
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("TASK 4 — Covariance estimator comparison")
log("=" * 65)

cov_rows = []
for cov_name, df_fr in frontier_results.items():
    if len(df_fr) == 0:
        continue
    # Only compare the three estimators on the unconstrained frontier.
    # The esg55 frontier is a different scenario, not a different estimator.
    if cov_name == "ledoit_wolf_esg55":
        continue
    cov_rows.append({
        "cov_estimator":      cov_name,
        "frontier_type":      "unconstrained",
        "n_feasible_points":  len(df_fr),
        "max_return":         df_fr["ret"].max(),
        "min_vol":            df_fr["vol"].min(),
        "max_sharpe":         df_fr["sharpe"].max(),
        "avg_esg":            df_fr["esg_score"].mean(),
        "avg_holdings":       df_fr["n_holdings"].mean(),
    })
    log(f"  {cov_name:<15}  feasible={len(df_fr)}  "
        f"max_ret={df_fr['ret'].max():.4f}  "
        f"max_sharpe={df_fr['sharpe'].max():.4f}  "
        f"avg_holdings={df_fr['n_holdings'].mean():.1f}")

df_cov = pd.DataFrame(cov_rows)
df_cov.to_csv(f"{RESULTS_DIR}/cov_comparison.csv", index=False)
log(f"\n  Saved: {RESULTS_DIR}/cov_comparison.csv")


# ═════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════
log("\n[Plots] Generating figures...")

# ── Figure 1: Efficient Frontier ──────────────────────────────────────
# Two panels:
#   Left:  3 covariance estimators — all unconstrained (no ESG floor)
#          Shows model risk: how much the frontier shifts across estimators
#   Right: LW only — unconstrained vs ESG floor=55
#          Shows the true ESG cost: downward shift of the frontier
fig1, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Efficient Frontier — MIQP  (Ledoit-Wolf)", fontsize=13)

colors_cov = {
    "sample":      "#4C9BE8",
    "ledoit_wolf": "#E87C4C",
    "oas":         "#5DBE8A",
}
labels_cov = {
    "sample":      "Sample covariance (unconstrained)",
    "ledoit_wolf": "Ledoit-Wolf (unconstrained)",
    "oas":         "OAS (unconstrained)",
}

# Left panel: 3 estimators, no ESG constraint
for cov_name in ["sample", "ledoit_wolf", "oas"]:
    df_fr = frontier_results.get(cov_name, pd.DataFrame())
    if len(df_fr) == 0:
        continue
    df_plot = df_fr.sort_values("vol")
    ax_left.plot(df_plot["vol"], df_plot["ret"],
                 "o-", color=colors_cov[cov_name],
                 linewidth=2, markersize=5,
                 label=labels_cov[cov_name])
ax_left.set_xlabel("Annualised Volatility", fontsize=11)
ax_left.set_ylabel("Annualised Expected Return", fontsize=11)
ax_left.set_title("Model Risk: Three Covariance Estimators\n(No ESG constraint)", fontsize=11)
ax_left.legend(fontsize=9)
ax_left.grid(True, alpha=0.3)

# Right panel: LW — true unconstrained vs ESG floor=55
# The vertical distance between the two curves at any vol level
# is the return cost of imposing the ESG floor = 55 constraint.
df_lw_free = frontier_results.get("ledoit_wolf", pd.DataFrame())
df_lw_55   = frontier_results.get("ledoit_wolf_esg55", pd.DataFrame())

if len(df_lw_free) > 0:
    df_plot = df_lw_free.sort_values("vol")
    ax_right.plot(df_plot["vol"], df_plot["ret"],
                  "o-", color="#4C9BE8", linewidth=2, markersize=5,
                  label="Unconstrained  (no ESG floor)")
if len(df_lw_55) > 0:
    df_plot = df_lw_55.sort_values("vol")
    ax_right.plot(df_plot["vol"], df_plot["ret"],
                  "s--", color="#E87C4C", linewidth=2, markersize=5,
                  label="ESG floor = 55  (constrained)")

if baseline_result["weights"] is not None:
    ax_right.scatter(baseline_result["vol"], baseline_result["ret"],
                     color="red", zorder=6, s=120, marker="*",
                     label=f"Baseline  (LW, ESG≥55, vc={selected_vol_cap:.3f})")

ax_right.set_xlabel("Annualised Volatility", fontsize=11)
ax_right.set_ylabel("Annualised Expected Return", fontsize=11)
ax_right.set_title("ESG Constraint Cost  (Ledoit-Wolf)\n"
                   "Unconstrained vs ESG floor = 55", fontsize=11)
ax_right.legend(fontsize=9)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(f"{FIGURES_DIR}/step3_frontier.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
log(f"  Saved: {FIGURES_DIR}/step3_frontier.png")

# ── Figure 2: ESG Sensitivity ──
if len(df_esg) > 0:
    colors_esg = {"sample": "#4C9BE8", "ledoit_wolf": "#E87C4C", "oas": "#5DBE8A"}
    labels_esg = {"sample": "Sample", "ledoit_wolf": "Ledoit-Wolf", "oas": "OAS"}

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f"ESG Floor Sensitivity  (all estimators, vol cap = {selected_vol_cap:.4f})",
                  fontsize=13)

    # axes[0]: Return cost — Ledoit-Wolf only for clarity
    df_esg_lw = df_esg[df_esg["cov_estimator"] == "ledoit_wolf"].sort_values("esg_floor")
    if len(df_esg_lw) > 0:
        ret_max = df_esg_lw["ret"].max()
        axes2[0].bar(df_esg_lw["esg_floor"], ret_max - df_esg_lw["ret"],
                     color="#E87C4C", width=3.5, edgecolor="white")
    axes2[0].set_xlabel("ESG Floor (0-100 scale)")
    axes2[0].set_ylabel("Return Cost vs ESG floor 40 (pp)")
    axes2[0].set_title("Return Cost of ESG Constraint\n(Ledoit-Wolf)")
    axes2[0].grid(True, alpha=0.3)

    # axes[1]: Sharpe ratio — all 3 estimators as separate lines
    for cov_name in ["sample", "ledoit_wolf", "oas"]:
        df_sub = df_esg[df_esg["cov_estimator"] == cov_name].sort_values("esg_floor")
        if len(df_sub) > 0:
            axes2[1].plot(df_sub["esg_floor"], df_sub["sharpe"],
                          "o-", color=colors_esg[cov_name], linewidth=2, markersize=7,
                          label=labels_esg[cov_name])
    axes2[1].set_xlabel("ESG Floor (0-100 scale)")
    axes2[1].set_ylabel("Sharpe Ratio")
    axes2[1].set_title("Sharpe Ratio vs ESG Floor")
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    # axes[2]: Realised ESG vs floor — Ledoit-Wolf only
    if len(df_esg_lw) > 0:
        axes2[2].plot(df_esg_lw["esg_floor"], df_esg_lw["esg"],
                      "o-", color="#5DBE8A", linewidth=2, markersize=7,
                      label="Realised ESG")
        axes2[2].plot(df_esg_lw["esg_floor"], df_esg_lw["esg_floor"],
                      "--", color="grey", linewidth=1.5, label="45° line (binding)")
    axes2[2].set_xlabel("ESG Floor (0-100 scale)")
    axes2[2].set_ylabel("Portfolio ESG Score")
    axes2[2].set_title("Realised ESG vs Floor\n(Ledoit-Wolf)")
    axes2[2].legend(fontsize=9)
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(f"{FIGURES_DIR}/step3_esg_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log(f"  Saved: {FIGURES_DIR}/step3_esg_sensitivity.png")

# ── Figure 3: Covariance estimator comparison ──
if len(df_cov) > 1:
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
    fig3.suptitle("Covariance Estimator Comparison", fontsize=13)

    # Max return per estimator
    axes3[0].bar(df_cov["cov_estimator"], df_cov["max_return"],
                 color=["#4C9BE8", "#E87C4C", "#5DBE8A"])
    axes3[0].set_title("Max Frontier Return")
    axes3[0].set_ylabel("Annualised Return")

    # Max Sharpe per estimator
    axes3[1].bar(df_cov["cov_estimator"], df_cov["max_sharpe"],
                 color=["#4C9BE8", "#E87C4C", "#5DBE8A"])
    axes3[1].set_title("Max Frontier Sharpe")
    axes3[1].set_ylabel("Sharpe Ratio")

    # Avg number of holdings per estimator
    axes3[2].bar(df_cov["cov_estimator"], df_cov["avg_holdings"],
                 color=["#4C9BE8", "#E87C4C", "#5DBE8A"])
    axes3[2].set_title("Avg Holdings per Frontier Point")
    axes3[2].set_ylabel("Number of Stocks")

    plt.tight_layout()
    fig3.savefig(f"{FIGURES_DIR}/step3_cov_comparison.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)
    log(f"  Saved: {FIGURES_DIR}/step3_cov_comparison.png")

# ═════════════════════════════════════════════════════════════
# SAVE OPTIMISATION LOG
# ═════════════════════════════════════════════════════════════
log_path = f"{RESULTS_DIR}/optimisation_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\n[Log] Saved: {log_path}")

# ═════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Mu specification  : '{active_mu_col}'  "
      f"({'robustness run — mu_winsor' if USE_ROBUSTNESS_MU else 'main specification — FICO metadata mu'})")
print(f"  ESG convention    : common 0-100 scale (Option A)")
for cov_name, df_fr in frontier_results.items():
    print(f"  Frontier [{cov_name:<15}]:  "
          f"{len(df_fr)}/{N_FRONTIER_POINTS} points feasible")
print(f"  Selected vol cap  : {selected_vol_cap:.4f}  "
      f"(max-Sharpe in central 60% of feasible LW frontier)")
print(f"  ESG sensitivity:  {len(df_esg)}/{len(ESG_FLOORS)} floors feasible")
if baseline_result["weights"] is not None:
    print(f"  Baseline portfolio:  "
          f"ret={baseline_result['ret']:.4f}  "
          f"vol={baseline_result['vol']:.4f}  "
          f"sharpe={baseline_result['sharpe']:.4f}  "
          f"ESG={baseline_result['esg_score']:.1f}  "
          f"n={baseline_result['n_holdings']}")
else:
    print(f"  Baseline portfolio:  INFEASIBLE")
print(f"\nOutputs in {RESULTS_DIR}/:")
print(f"  frontier_sample_unconstrained.csv")
print(f"  frontier_ledoit_wolf_unconstrained.csv")
print(f"  frontier_oas_unconstrained.csv")
print(f"  frontier_ledoit_wolf_esg55.csv")
print(f"  esg_sensitivity.csv, baseline_weights.csv, cov_comparison.csv")
print(f"\nFigures in {FIGURES_DIR}/:")
print(f"  step3_frontier.png, step3_esg_sensitivity.png, step3_cov_comparison.png")
print(f"\nReady for Step 4 (out-of-sample analysis).")
print("=" * 65)