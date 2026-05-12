"""
03b_esg_sweep_simulated.py: ESG Floor Sweep and Baseline Portfolio
==================================================================
ESG-Constrained Portfolio Optimisation - Independent Research Project

PIPELINE POSITION:
  02_preprocess_simulated.py → data/clean/meta_preprocessed.csv
                                data/clean/sigma_*_annual.npy
  03_optimize_simulated.py   → data/results/frontier_ledoit_wolf_unconstrained.csv
  03b_esg_sweep_simulated.py (this file) → data/results/esg_sensitivity.csv
                                            data/results/baseline_weights.csv
                                            figures/step3_esg_sensitivity.png

VOL CAP SELECTION (data-driven, rigorous):
  1. Load the unconstrained LW frontier from step 3.
  2. Keep only feasible points.
  3. Trim the bottom 20% (near-infeasibility boundary) and top 20% (most
     aggressive) of the feasible vol range.
  4. Within the central 60%, pick the point with the highest Sharpe ratio
     (tangency portfolio — economically the most representative choice).
  This ensures the ESG sensitivity is evaluated at a point that is:
  - strictly interior to the feasible region,
  - not arbitrarily conservative or aggressive, and
  - optimal in risk-adjusted return terms.

TASKS:
  2.  ESG floor sweep — all 3 covariance estimators, floors [40,45,50,55,60,65],
      data-driven vol cap selected above
      Output: esg_sensitivity.csv with 'cov_estimator' column
  3.  Baseline portfolio — LW only, ESG floor=55, same vol cap
      Output: baseline_weights.csv

SOLVER: Gurobi via cvxpy (academic licence)
  TimeLimit = 300 s per solve
  MIPGap    = 1e-4
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════
# PARAMETERS — change only here, never inline
# ═════════════════════════════════════════════════════════════

CLEAN_DIR   = "data/clean"
RESULTS_DIR = "data/results"
FIGURES_DIR = "figures"

# ── Portfolio constraints ──────────────────────────────────
W_MAX      = 0.20       # max weight per stock (20%)
W_MIN      = 0.01       # min weight if stock is selected (1%)
K_MAX      = 100        # max number of holdings
K_MIN      = 50         # min number of holdings
SECTOR_CAP = 0.25       # max total weight in any single sector

# ── ESG sweep parameters ────────────────────────────────────
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# Vol cap trimming rule: exclude the bottom and top fraction of the feasible
# vol range before picking the max-Sharpe point.
FRONTIER_TRIM_LOW  = 0.20   # drop bottom 20% of feasible vol range
FRONTIER_TRIM_HIGH = 0.20   # drop top    20% of feasible vol range
FRONTIER_LW_CSV    = "data/results/frontier_ledoit_wolf_unconstrained.csv"

# ── Mu column ───────────────────────────────────────────────
MU_COLUMN = "mu_native"

# ── Covariance estimators ────────────────────────────────────
COV_ESTIMATORS = {
    "sample":      "sigma_sample_annual.npy",
    "ledoit_wolf": "sigma_lw_annual.npy",
    "oas":         "sigma_oas_annual.npy",
}

# ── Gurobi solver settings ────────────────────────────────
SOLVER_VERBOSE    = False
SOLVER_TIME_LIMIT = 300      # seconds per solve
SOLVER_MIP_GAP    = 1e-4     # relative MIP optimality gap

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
log("STEP 3B — ESG FLOOR SWEEP + BASELINE PORTFOLIO")
log("=" * 65)

meta    = pd.read_csv(f"{CLEAN_DIR}/meta_preprocessed.csv", index_col=0)
tickers = list(meta.index)
N       = len(tickers)

mu_vec  = meta[MU_COLUMN].values.astype(float)
esg_vec = meta["esg"].values.astype(float)

log(f"\n  Universe:      {N} stocks")
log(f"  Mu column:     '{MU_COLUMN}'  mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}")
log(f"  ESG column:    esg  mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
log(f"  Constraints:   W_MAX={W_MAX}  W_MIN={W_MIN}  K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
log(f"  ESG floors:    {ESG_FLOORS}")
log(f"  Vol cap:       data-driven (selected from LW unconstrained frontier)")

# Sector index map
sector_map = {}
if "sector" in meta.columns:
    for sec, grp in meta.groupby("sector"):
        sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
    log(f"  Sectors:       {list(sector_map.keys())}")
else:
    log("  No sector column — sector cap constraint disabled.")
    sector_map = None

# Load and regularise all sigma matrices once
sigma_cache = {}
for cov_name, cov_file in COV_ESTIMATORS.items():
    Sigma = np.load(f"{CLEAN_DIR}/{cov_file}")
    Sigma = (Sigma + Sigma.T) / 2       # force symmetry
    Sigma += np.eye(N) * 1e-8           # PSD regularisation
    sigma_cache[cov_name] = Sigma
    log(f"  Loaded [{cov_name}] from {cov_file}")

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
    Subject to: w' Sigma w  <= vol_cap^2     (vol-cap)
                sum(w)       = 1              (budget)
                w' esg       >= esg_floor     (ESG floor, if apply_esg=True)
                w_i          <= w_max * z_i   (upper bound)
                w_i          >= w_min * z_i   (lower bound if selected)
                sum(z_i)     <= k_max         (max holdings)
                sum(z_i)     >= k_min         (min holdings)
                w_i          >= 0             (long-only)
                z_i          in {0,1}         (binary selection)
                sector weight <= sector_cap   (sector diversification)
    """
    n = len(mu)
    w = cp.Variable(n, nonneg=True)
    z = cp.Variable(n, boolean=True)

    objective = cp.Maximize(mu @ w)

    constraints = [
        cp.sum(w) == 1,
        cp.quad_form(w, Sigma) <= vol_cap ** 2,
        w <= w_max * z,
        w >= w_min * z,
        cp.sum(z) <= k_max,
        cp.sum(z) >= k_min,
    ]

    if apply_esg:
        constraints.append(esg @ w >= esg_floor)

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

    w_val  = np.clip(np.array(w.value).flatten(), 0, None)
    w_val /= w_val.sum()

    port_ret    = float(mu @ w_val)
    port_var    = float(w_val @ Sigma @ w_val)
    port_vol    = float(np.sqrt(max(port_var, 0)))
    port_sharpe = port_ret / port_vol if port_vol > 1e-8 else np.nan
    port_esg    = float(esg @ w_val)
    n_holdings  = int((w_val > 1e-4).sum())

    return {
        "status":     prob.status,
        "weights":    w_val,
        "ret":        port_ret,
        "vol":        port_vol,
        "sharpe":     port_sharpe,
        "esg_score":  port_esg,
        "n_holdings": n_holdings,
    }

# ═════════════════════════════════════════════════════════════
# VOL CAP SELECTION — data-driven from unconstrained LW frontier
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log("VOL CAP SELECTION — from unconstrained LW frontier")
log("=" * 65)

frontier_df = pd.read_csv(FRONTIER_LW_CSV)
feasible    = frontier_df[frontier_df["status"].isin(["optimal", "optimal_inaccurate"])].copy()
feasible    = feasible.sort_values("vol_cap").reset_index(drop=True)

if feasible.empty:
    raise RuntimeError(
        f"No feasible points found in {FRONTIER_LW_CSV}. "
        "Run step3_optimize.py first."
    )

vol_min = feasible["vol_cap"].min()
vol_max = feasible["vol_cap"].max()
vol_rng = vol_max - vol_min

low_cut  = vol_min + FRONTIER_TRIM_LOW  * vol_rng
high_cut = vol_max - FRONTIER_TRIM_HIGH * vol_rng
central  = feasible[(feasible["vol_cap"] >= low_cut) & (feasible["vol_cap"] <= high_cut)]

if central.empty:
    # Fallback: if trimming leaves nothing, use the full feasible set
    log("  WARNING: trimming left no points — using full feasible set as fallback.")
    central = feasible

best_row    = central.loc[central["sharpe"].idxmax()]
VOL_CAP_ESG = float(best_row["vol_cap"])

log(f"  Feasible frontier: {len(feasible)} points  "
    f"vol range [{vol_min:.3f}, {vol_max:.3f}]")
log(f"  Central band (trim {FRONTIER_TRIM_LOW:.0%}/{FRONTIER_TRIM_HIGH:.0%}): "
    f"[{low_cut:.3f}, {high_cut:.3f}]  →  {len(central)} candidates")
log(f"  Selected vol cap : {VOL_CAP_ESG:.4f}  "
    f"(max Sharpe = {best_row['sharpe']:.4f},  "
    f"ret = {best_row['ret']:.4f},  "
    f"vol = {best_row['vol']:.4f})")

# ═════════════════════════════════════════════════════════════
# TASK 2 — ESG FLOOR SWEEP  (all 3 estimators, fixed vol cap)
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log(f"TASK 2 — ESG Floor Sweep  (all 3 estimators, vol cap = {VOL_CAP_ESG:.4f})")
log("=" * 65)

# Upper bound on achievable portfolio ESG (greedy, W_MAX blocks)
_idx_desc   = np.argsort(esg_vec)[::-1]
_ub_w       = np.zeros(N)
_budget     = 1.0
for _i in _idx_desc:
    _alloc      = min(W_MAX, _budget)
    _ub_w[_i]   = _alloc
    _budget    -= _alloc
    if _budget <= 1e-9:
        break
max_achievable_esg = float(esg_vec @ _ub_w)
log(f"  Max achievable portfolio ESG (greedy upper bound): {max_achievable_esg:.2f}")
log(f"  Fixed vol cap: {VOL_CAP_ESG}")

esg_rows = []

for cov_name, Sigma in sigma_cache.items():
    log(f"\n  [{cov_name}]")
    t0 = time.time()

    for floor in ESG_FLOORS:
        if max_achievable_esg < floor:
            log(f"    floor={floor}  SKIPPED — max achievable ESG={max_achievable_esg:.1f} < {floor}")
            continue

        result = solve_miqp(
            Sigma, mu_vec, esg_vec,
            vol_cap=VOL_CAP_ESG,
            apply_esg=True,
            esg_floor=float(floor),
            sector_map=sector_map,
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
                "mu_source":     MU_COLUMN,
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
log(f"\n  Saved: {RESULTS_DIR}/esg_sensitivity.csv  ({len(df_esg)} rows)")

# ═════════════════════════════════════════════════════════════
# TASK 3 — BASELINE PORTFOLIO  (LW, ESG floor=55, vol cap=1.069)
# ═════════════════════════════════════════════════════════════
log("\n" + "=" * 65)
log(f"TASK 3 — Baseline Portfolio  (LW, ESG floor=55, vol cap={VOL_CAP_ESG:.4f})")
log("=" * 65)

n_eligible = (esg_vec >= 55).sum()
log(f"  Eligible stocks (ESG >= 55): {n_eligible}  (K_MIN={K_MIN})")
if n_eligible < K_MIN:
    log("  *** WARNING: fewer eligible stocks than K_MIN — baseline may be infeasible ***")

baseline_result = solve_miqp(
    sigma_cache["ledoit_wolf"], mu_vec, esg_vec,
    vol_cap=VOL_CAP_ESG,
    apply_esg=True,
    esg_floor=55.0,
    sector_map=sector_map,
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
        "mu_source": MU_COLUMN,
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
# FIGURE — ESG Sensitivity  (all 3 estimators)
# ═════════════════════════════════════════════════════════════
log("\n[Plots] Generating ESG sensitivity figure...")

if len(df_esg) > 0:
    colors = {
        "sample":      "#4C9BE8",
        "ledoit_wolf": "#E87C4C",
        "oas":         "#5DBE8A",
    }
    labels = {
        "sample":      "Sample",
        "ledoit_wolf": "Ledoit-Wolf",
        "oas":         "OAS",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"ESG Floor Sensitivity  (all 3 estimators, vol cap = {VOL_CAP_ESG:.4f})",
        fontsize=13,
    )

    for cov_name in ["sample", "ledoit_wolf", "oas"]:
        df_sub = df_esg[df_esg["cov_estimator"] == cov_name].sort_values("esg_floor")
        if df_sub.empty:
            continue
        col = colors[cov_name]
        lbl = labels[cov_name]

        # Panel 0: Return cost vs lowest-floor return
        ret_ref = df_sub["ret"].max()
        axes[0].plot(df_sub["esg_floor"], ret_ref - df_sub["ret"],
                     "o-", color=col, linewidth=2, markersize=6, label=lbl)

        # Panel 1: Sharpe ratio
        axes[1].plot(df_sub["esg_floor"], df_sub["sharpe"],
                     "o-", color=col, linewidth=2, markersize=6, label=lbl)

        # Panel 2: Realised ESG vs floor
        axes[2].plot(df_sub["esg_floor"], df_sub["esg"],
                     "o-", color=col, linewidth=2, markersize=6, label=lbl)

    # 45-degree reference line on panel 2
    floor_range = sorted(df_esg["esg_floor"].unique())
    axes[2].plot(floor_range, floor_range,
                 "--", color="grey", linewidth=1.5, label="45° (binding)")

    axes[0].set_xlabel("ESG Floor")
    axes[0].set_ylabel("Return Cost vs Lowest Floor")
    axes[0].set_title("Return Cost of ESG Constraint")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("ESG Floor")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Sharpe Ratio vs ESG Floor")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("ESG Floor")
    axes[2].set_ylabel("Portfolio ESG Score")
    axes[2].set_title("Realised ESG vs Floor")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{FIGURES_DIR}/step3_esg_sensitivity.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {fig_path}")
else:
    log("  No ESG sweep results — figure skipped.")

# ═════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3B COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Mu specification  : '{MU_COLUMN}'")
print(f"  Vol cap (fixed)   : {VOL_CAP_ESG}")
print(f"  ESG sweep rows    : {len(df_esg)}  ({len(ESG_FLOORS)} floors × 3 estimators)")
for cov_name in ["sample", "ledoit_wolf", "oas"]:
    n_ok = len(df_esg[df_esg["cov_estimator"] == cov_name])
    print(f"    [{cov_name:<15}]: {n_ok}/{len(ESG_FLOORS)} floors feasible")
if baseline_result["weights"] is not None:
    print(f"  Baseline portfolio:  "
          f"ret={baseline_result['ret']:.4f}  "
          f"vol={baseline_result['vol']:.4f}  "
          f"sharpe={baseline_result['sharpe']:.4f}  "
          f"ESG={baseline_result['esg_score']:.1f}  "
          f"n={baseline_result['n_holdings']}")
else:
    print("  Baseline portfolio:  INFEASIBLE")
print(f"\nOutputs in {RESULTS_DIR}/:")
print(f"  esg_sensitivity.csv   (Task 2 — all 3 estimators, cov_estimator column)")
print(f"  baseline_weights.csv  (Task 3 — LW, floor=55)")
print(f"\nFigures in {FIGURES_DIR}/:")
print(f"  step3_esg_sensitivity.png")
print("=" * 65)
