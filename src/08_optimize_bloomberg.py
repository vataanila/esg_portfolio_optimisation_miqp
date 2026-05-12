"""
08_optimize_bloomberg.py: Bloomberg MIQP Portfolio Optimisation
================================================================
Bloomberg EURO STOXX 600 - ESG-Constrained Portfolio Optimisation
Independent Research Project

PIPELINE POSITION:
  07_preprocess_bloomberg.py → data/clean_bloomberg/bloomberg_meta_analytical.csv
                                data/clean_bloomberg/bloomberg_Sigma_sample.csv
                                data/clean_bloomberg/bloomberg_Sigma_lw.csv
                                data/clean_bloomberg/bloomberg_Sigma_oas.csv
  08_optimize_bloomberg.py (this file) → data/results_bloomberg/bloomberg_frontier_*.csv
                                          data/results_bloomberg/bloomberg_esg_sensitivity.csv
                                          data/results_bloomberg/bloomberg_baseline_weights.csv
                                          data/results_bloomberg/bloomberg_cov_comparison.csv
                                          data/results_bloomberg/bloomberg_optimisation_log.txt
                                          figures/step7_frontier.png
                                          figures/step7_esg_sensitivity.png
                                          figures/step7_cov_comparison.png

MODEL: Mixed-Integer Quadratic Programme (MIQP)
  Objective:  maximise  w' mu
  s.t.        w' Sigma w  <= vol_cap^2          (vol-cap, swept for frontier)
              sum(w)       = 1                   (budget)
              w' esg       >= esg_floor          (weighted-avg ESG constraint)
              w_i          <= W_MAX * z_i        (upper bound per stock)
              w_i          >= W_MIN * z_i        (lower bound if selected)
              sum(z_i)     <= K_MAX              (max holdings)
              sum(z_i)     >= K_MIN              (min holdings)
              w_i          >= 0                  (long-only)
              z_i          in {0,1}              (binary selection)
              sum_j(w_j)   <= SECTOR_CAP  forall sector (sector cap)

VOL CAP SELECTION (Task 2 and 3):
  The fixed vol cap for ESG sweep and baseline is selected data-driven:
  max-Sharpe point in the central 60% of the feasible unconstrained Ledoit-Wolf
  frontier (trim bottom 20% + top 20% of feasible vol range).

TASKS (mirrors 03_optimize_simulated.py structure exactly):
  1a. Frontier — 3 covariance estimators, NO ESG constraint (true unconstrained)
  1b. Frontier — Ledoit-Wolf only, ESG floor = 55 (main constrained result)
  2.  ESG floor sweep — all 3 estimators, 6 floors [40,45,50,55,60,65], fixed vol cap
  3.  Baseline portfolio — LW, ESG floor = 55
  4.  Covariance comparison table

KEY DIFFERENCES FROM 03_optimize_simulated.py:
  - Covariance files are .csv (not .npy) — as saved by step 07
  - mu = mu_trailing_winsor (3-year trailing winsorised) — Bloomberg main specification
    (no native FICO mu exists for Bloomberg data)
  - ESG floors: same range [40,45,50,55,60,65] as the simulated pipeline
    (both datasets use the same grid; Bloomberg ESG mean = 44.1 vs simulated mean = 50.5)
  - Vol cap range: [0.10, 0.80] calibrated to real equity vol (mean 30%, max 72%)
    instead of [0.30, 3.00] used for synthetic data with inflated volatility

SOLVER: Gurobi 13.0 via cvxpy (academic licence)

Run:
    python src/08_optimize_bloomberg.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETERS — change only here, never inline
# =============================================================================

CLEAN_DIR   = "data/clean_bloomberg"
RESULTS_DIR = "data/results_bloomberg"
FIGURES_DIR = "figures"

# Portfolio constraints (identical to Step 3)
W_MAX      = 0.20
W_MIN      = 0.01
K_MAX      = 100
K_MIN      = 50
SECTOR_CAP = 0.25

# Frontier sweep
N_FRONTIER_POINTS = 30          # 30 for smooth frontier curves (Step 3 used 12)

# Vol cap range calibrated to real Bloomberg equity data
# (annualised vol range: 14%-72%, mean 30%)
VOL_CAP_MIN = 0.10
VOL_CAP_MAX = 0.80

# ESG floors — shifted down vs simulated to match Bloomberg ESG distribution
# Bloomberg ESG mean = 44.1 on 0-100 scale (vs simulated mean 50.5)
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# mu specification
# Bloomberg has no native "given mu" — mu is estimated from historical returns.
# Bloomberg main specification:      mu_trailing_winsor = 3-year trailing winsorised mu
# Bloomberg robustness check (1):    mu_winsor          = full-sample winsorised mu
# Bloomberg robustness check (2):    mu                 = full-sample raw mu
MU_COLUMN            = "mu_trailing_winsor" # Bloomberg main specification (3-yr trailing)
MU_ROBUSTNESS_COLUMN = "mu_winsor"          # robustness: full-sample winsorised mu
MU_TRAILING_COLUMN   = "mu"                 # robustness: raw unwinsorised full-sample mu
USE_ROBUSTNESS_MU    = False                # set True to use mu_winsor (full-sample winsorised)
USE_TRAILING_MU      = False                # reserved for future robustness checks

# Covariance files (as saved by step6 — .csv format)
COV_ESTIMATORS = {
    "sample":      "bloomberg_Sigma_sample.csv",
    "ledoit_wolf": "bloomberg_Sigma_lw.csv",
    "oas":         "bloomberg_Sigma_oas.csv",
}

# Gurobi settings
SOLVER_VERBOSE    = False
SOLVER_TIME_LIMIT = 120
SOLVER_MIP_GAP    = 1e-4

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# OPTIMISATION LOG
# =============================================================================
log_lines = []

def log(msg: str) -> None:
    """Print to console and append to the optimisation log buffer."""
    print(msg)
    log_lines.append(msg)

# =============================================================================
# LOAD DATA
# =============================================================================
log("=" * 65)
log("STEP 7 — BLOOMBERG MIQP PORTFOLIO OPTIMISATION")
log("=" * 65)

meta    = pd.read_csv(f"{CLEAN_DIR}/bloomberg_meta_analytical.csv", index_col=0)
tickers = list(meta.index)
N       = len(tickers)

if USE_ROBUSTNESS_MU:
    active_mu_col = MU_ROBUSTNESS_COLUMN
elif USE_TRAILING_MU:
    active_mu_col = MU_TRAILING_COLUMN
else:
    active_mu_col = MU_COLUMN

mu_vec  = meta[active_mu_col].values.astype(float)
esg_vec = meta["esg"].values.astype(float)

if USE_ROBUSTNESS_MU:
    mu_spec_label = "robustness — full-sample winsorised mu"
elif USE_TRAILING_MU:
    mu_spec_label = "robustness — raw full-sample mu"
else:
    mu_spec_label = "main specification — 3-year trailing winsorised mu"

log(f"\n  Dataset         : Bloomberg EURO STOXX 600")
log(f"  Universe        : {N} stocks")
log(f"  Mu column       : '{active_mu_col}'  ({mu_spec_label})"
    f"  mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}")
log(f"  ESG column      : 'esg'  (0-100 scale, mapped from Bloomberg BESG 1-10)"
    f"  mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
log(f"  Constraints     : W_MAX={W_MAX}  W_MIN={W_MIN}  "
    f"K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
log(f"  ESG floors      : {ESG_FLOORS}")
log(f"  Frontier pts    : {N_FRONTIER_POINTS}  vol_cap=[{VOL_CAP_MIN},{VOL_CAP_MAX}]")

# Load covariance matrices (.csv)
sigma_cache = {}
for cov_name, cov_file in COV_ESTIMATORS.items():
    path = f"{CLEAN_DIR}/{cov_file}"
    S_df = pd.read_csv(path, index_col=0)
    S_df = S_df.loc[tickers, tickers]
    S    = S_df.values.astype(float)
    S    = (S + S.T) / 2
    S   += np.eye(N) * 1e-8
    sigma_cache[cov_name] = S
    min_eig = np.linalg.eigvalsh(S).min()
    log(f"  Sigma ({cov_name:12s}) loaded  shape={S.shape}  min_eig={min_eig:.6f}")

# Sector map
sector_col = "sector"
sector_map = {}
if sector_col in meta.columns:
    for sec, grp in meta.groupby(sector_col):
        sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
    log(f"  Sectors         : {list(sector_map.keys())}")
else:
    log("  WARNING: no 'sector' column — sector cap disabled.")

# =============================================================================
# CORE OPTIMISATION FUNCTION  (identical to Step 3)
# =============================================================================

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=0.0, apply_esg=True,
               sector_map=None,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN,
               sector_cap=SECTOR_CAP):
    """
    Solve one MIQP instance via cvxpy + Gurobi.
    Identical formulation to Step 3 (simulated dataset).

    apply_esg=False → TRUE unconstrained (no ESG floor at all).
    This is the correct benchmark. Using ESG floor=40 as "unconstrained"
    would still restrict the feasible set — wrong comparison.
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


# =============================================================================
# TASK 1a — TRUE UNCONSTRAINED FRONTIER  (3 estimators, no ESG)
# =============================================================================
log("\n" + "=" * 65)
log("TASK 1a — True Unconstrained Frontier  (no ESG constraint, 3 estimators)")
log("=" * 65)

vol_caps = np.geomspace(VOL_CAP_MIN, VOL_CAP_MAX, N_FRONTIER_POINTS)
log(f"  Vol caps (geomspace): min={VOL_CAP_MIN}  max={VOL_CAP_MAX}  n={N_FRONTIER_POINTS}")
log(f"  First 5: {np.round(vol_caps[:5], 3).tolist()}")

frontier_results = {}

for cov_name, Sigma in sigma_cache.items():
    log(f"\n  [{cov_name}]")
    rows = []
    t0   = time.time()

    for vc in vol_caps:
        result = solve_miqp(
            Sigma, mu_vec, esg_vec,
            vol_cap=vc,
            apply_esg=False,
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
            log(f"    vc={vc:.3f}  INFEASIBLE: {result['status']}")

    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s — {len(rows)}/{N_FRONTIER_POINTS} feasible")

    df_fr = pd.DataFrame(rows)
    frontier_results[cov_name] = df_fr
    df_fr.to_csv(
        f"{RESULTS_DIR}/bloomberg_frontier_{cov_name}_unconstrained.csv",
        index=False)
    log(f"  Saved: bloomberg_frontier_{cov_name}_unconstrained.csv")

    # Crash protection — save partial after each estimator
    pd.concat(list(frontier_results.values())).to_csv(
        f"{RESULTS_DIR}/bloomberg_frontier_partial.csv", index=False)


# =============================================================================
# TASK 1b — ESG-CONSTRAINED FRONTIER  (LW only, floor = 55)
# =============================================================================
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
df_frontier_55.to_csv(
    f"{RESULTS_DIR}/bloomberg_frontier_ledoit_wolf_esg55.csv", index=False)
elapsed = time.time() - t0
log(f"  Done in {elapsed:.1f}s — {len(rows_55)}/{N_FRONTIER_POINTS} feasible")
log(f"  Saved: bloomberg_frontier_ledoit_wolf_esg55.csv")


# =============================================================================
# TASK 2 — ESG FLOOR SWEEP  (all 3 estimators, fixed vol cap at 40th percentile)
# =============================================================================
log("\n" + "=" * 65)
log("TASK 2 — ESG Floor Sweep  (sample / ledoit_wolf / oas, fixed vol cap)")
log("=" * 65)

# --- Tangency vol cap: max-Sharpe in central 60% of unconstrained LW frontier ---
_fr_lw = pd.read_csv(f"{RESULTS_DIR}/bloomberg_frontier_ledoit_wolf_unconstrained.csv")
_fr_lw = _fr_lw[_fr_lw["status"].isin(["optimal", "optimal_inaccurate"])].copy()
_fr_lw = _fr_lw.sort_values("vol_cap").reset_index(drop=True)
if _fr_lw.empty:
    raise RuntimeError("LW unconstrained frontier has no feasible points — cannot select vol cap.")
_vc_min   = _fr_lw["vol_cap"].min()
_vc_max   = _fr_lw["vol_cap"].max()
_vc_rng   = _vc_max - _vc_min
_low_cut  = _vc_min + 0.20 * _vc_rng
_high_cut = _vc_max - 0.20 * _vc_rng
_central  = _fr_lw[(_fr_lw["vol_cap"] >= _low_cut) & (_fr_lw["vol_cap"] <= _high_cut)]
if _central.empty:
    _central = _fr_lw
_best_row   = _central.loc[_central["sharpe"].idxmax()]
vol_cap_esg = float(_best_row["vol_cap"])
log(f"  Vol cap for ESG sweep : {vol_cap_esg:.4f}  "
    f"(tangency/max-Sharpe in central 60% of LW frontier, "
    f"Sharpe={_best_row['sharpe']:.4f}, ret={_best_row['ret']:.4f}, vol={_best_row['vol']:.4f})")
with open(f"{RESULTS_DIR}/bloomberg_selected_vol_cap.txt", "w") as _f:
    _f.write(str(vol_cap_esg))
log(f"  Bloomberg ESG mean    : {esg_vec.mean():.2f}  "
    f"(both datasets use floors [40-65]; Bloomberg mean is lower at {esg_vec.mean():.1f} vs simulated ~50.5)")

esg_rows = []

# Upper bound on achievable portfolio ESG: greedily assign W_MAX to the
# highest-ESG stocks until budget is exhausted (true upper bound — ignores K_MIN).
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

for cov_name, Sigma in sigma_cache.items():
    log(f"\n  [{cov_name}]")
    for floor in ESG_FLOORS:
        if max_achievable_esg < floor:
            log(f"  floor={floor}  SKIPPED — max achievable ESG={max_achievable_esg:.1f} < {floor}. "
                f"Structurally infeasible.")
            continue
        log(f"  floor={floor}  solving...")
        result = solve_miqp(
            Sigma, mu_vec, esg_vec,
            vol_cap=vol_cap_esg,
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
            log(f"    ret={result['ret']:.4f}  vol={result['vol']:.4f}  "
                f"sharpe={result['sharpe']:.3f}  ESG={result['esg_score']:.1f}  "
                f"n={result['n_holdings']}")
        else:
            log(f"    INFEASIBLE: {result['status']}")

df_esg = pd.DataFrame(esg_rows)
df_esg.to_csv(f"{RESULTS_DIR}/bloomberg_esg_sensitivity.csv", index=False)
log(f"\n  Saved: bloomberg_esg_sensitivity.csv")


# =============================================================================
# TASK 3 -- BASELINE PORTFOLIO  (LW, ESG floor = 55)
# =============================================================================
log("\n" + "=" * 65)
log("TASK 3 — Baseline Portfolio  (LW, ESG floor=55, fixed vol cap)")
log("=" * 65)

log(f"  Vol cap: {vol_cap_esg:.4f}   ESG floor: 55.0")

Sigma_lw = sigma_cache["ledoit_wolf"]
baseline_result = solve_miqp(
    Sigma_lw, mu_vec, esg_vec,
    vol_cap=vol_cap_esg,
    apply_esg=True,
    esg_floor=55.0,
    sector_map=sector_map if sector_map else None,
)

if baseline_result["weights"] is not None:
    w_base = baseline_result["weights"]
    log(f"\n  Baseline portfolio:")
    log(f"    Return     : {baseline_result['ret']:.4f}")
    log(f"    Volatility : {baseline_result['vol']:.4f}")
    log(f"    Sharpe     : {baseline_result['sharpe']:.4f}")
    log(f"    ESG score  : {baseline_result['esg_score']:.2f}")
    log(f"    Holdings   : {baseline_result['n_holdings']}")

    baseline_weights = pd.DataFrame({
        "ticker":    tickers,
        "weight":    w_base,
        "mu":        mu_vec,
        "esg":       esg_vec,
        "sector":    meta[sector_col].values if sector_col in meta.columns else "N/A",
        "country":   meta["country"].values  if "country"  in meta.columns else "N/A",
        "mu_source": active_mu_col,
    })
    baseline_weights = (baseline_weights[baseline_weights["weight"] > 1e-4]
                        .sort_values("weight", ascending=False))
    baseline_weights.to_csv(
        f"{RESULTS_DIR}/bloomberg_baseline_weights.csv", index=False)
    log(f"\n  Saved: bloomberg_baseline_weights.csv  "
        f"({len(baseline_weights)} stocks)")

    log(f"\n  Top 10 holdings:")
    log(f"  {'Ticker':<12}  {'Weight':>8}  {'mu':>8}  {'ESG':>6}  "
        f"{'Sector':<28}  Country")
    for _, row in baseline_weights.head(10).iterrows():
        log(f"  {str(row['ticker']):<12}  {row['weight']:>8.4f}  "
            f"{row['mu']:>8.4f}  {row['esg']:>6.1f}  "
            f"{str(row['sector']):<28}  {row['country']}")
else:
    log(f"  Baseline INFEASIBLE: {baseline_result['status']}")
    log("  *** Try reducing ESG floor, K_MIN, or relaxing W_MIN ***")


# =============================================================================
# TASK 4 — COVARIANCE COMPARISON TABLE
# =============================================================================
log("\n" + "=" * 65)
log("TASK 4 — Covariance estimator comparison (unconstrained frontiers)")
log("=" * 65)

cov_rows = []
for cov_name, df_fr in frontier_results.items():
    if cov_name == "ledoit_wolf_esg55":
        continue
    if len(df_fr) == 0:
        continue
    cov_rows.append({
        "cov_estimator":     cov_name,
        "frontier_type":     "unconstrained",
        "n_feasible_points": len(df_fr),
        "max_return":        df_fr["ret"].max(),
        "min_vol":           df_fr["vol"].min(),
        "max_sharpe":        df_fr["sharpe"].max(),
        "avg_esg":           df_fr["esg_score"].mean(),
        "avg_holdings":      df_fr["n_holdings"].mean(),
    })
    log(f"  {cov_name:<15}  feasible={len(df_fr)}  "
        f"max_ret={df_fr['ret'].max():.4f}  "
        f"max_sharpe={df_fr['sharpe'].max():.4f}  "
        f"avg_holdings={df_fr['n_holdings'].mean():.1f}")

df_cov = pd.DataFrame(cov_rows)
df_cov.to_csv(f"{RESULTS_DIR}/bloomberg_cov_comparison.csv", index=False)
log(f"\n  Saved: bloomberg_cov_comparison.csv")


# =============================================================================
# FIGURES  (mirror Step 3 layout exactly)
# =============================================================================
log("\n[Plots] Generating figures...")

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

# Figure 1: Frontier — left: 3 estimators | right: LW constrained vs unconstrained
fig1, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle(
    "Bloomberg EURO STOXX 600 — Efficient Frontier (MIQP)", fontsize=13)

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
ax_left.set_title(
    "Model Risk: Three Covariance Estimators\n(No ESG constraint)", fontsize=11)
ax_left.legend(fontsize=9)
ax_left.grid(True, alpha=0.3)

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
    ax_right.scatter(
        baseline_result["vol"], baseline_result["ret"],
        color="red", zorder=6, s=120, marker="*",
        label="Baseline portfolio  (LW, ESG ≥ 55)")

ax_right.set_xlabel("Annualised Volatility", fontsize=11)
ax_right.set_ylabel("Annualised Expected Return", fontsize=11)
ax_right.set_title(
    "ESG Constraint Cost  (Ledoit-Wolf)\nUnconstrained vs ESG floor = 55",
    fontsize=11)
ax_right.legend(fontsize=9)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(f"{FIGURES_DIR}/step7_frontier.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
log(f"  Saved: step7_frontier.png")

# Figure 2: ESG sensitivity
if len(df_esg) > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(
        "Bloomberg — ESG Floor Sensitivity  (fixed vol cap, all 3 estimators)",
        fontsize=13)

    colors_esg = {"sample": "#4C9BE8", "ledoit_wolf": "#E87C4C", "oas": "#5DBE8A"}
    labels_esg = {"sample": "Sample", "ledoit_wolf": "Ledoit-Wolf", "oas": "OAS"}

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
    fig2.savefig(
        f"{FIGURES_DIR}/step7_esg_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log(f"  Saved: step7_esg_sensitivity.png")

# Figure 3: Covariance comparison
if len(df_cov) > 1:
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
    fig3.suptitle(
        "Bloomberg — Covariance Estimator Comparison (Unconstrained Frontier)",
        fontsize=13)

    bar_colors = [colors_cov.get(n, "#888") for n in df_cov["cov_estimator"]]

    axes3[0].bar(df_cov["cov_estimator"], df_cov["max_return"], color=bar_colors)
    axes3[0].set_title("Max Frontier Return")
    axes3[0].set_ylabel("Annualised Return")

    axes3[1].bar(df_cov["cov_estimator"], df_cov["max_sharpe"], color=bar_colors)
    axes3[1].set_title("Max Frontier Sharpe")
    axes3[1].set_ylabel("Sharpe Ratio")

    axes3[2].bar(df_cov["cov_estimator"], df_cov["avg_holdings"], color=bar_colors)
    axes3[2].set_title("Avg Holdings per Frontier Point")
    axes3[2].set_ylabel("Number of Stocks")

    plt.tight_layout()
    fig3.savefig(
        f"{FIGURES_DIR}/step7_cov_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    log(f"  Saved: step7_cov_comparison.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
log("\n" + "=" * 65)
log("STEP 8 COMPLETE — SUMMARY")
log("=" * 65)
log(f"  Dataset           : Bloomberg EURO STOXX 600")
log(f"  Mu specification  : '{active_mu_col}'  "
    f"({'robustness — full-sample winsorised' if USE_ROBUSTNESS_MU else 'main specification — 3-year trailing winsorised'})")
log(f"  ESG convention    : 0-100 scale  (mapped from Bloomberg BESG 1-10 via (BESG-1)/9*100)")
for cov_name, df_fr in frontier_results.items():
    log(f"  Frontier [{cov_name:<22}]:  "
        f"{len(df_fr)}/{N_FRONTIER_POINTS} points feasible")
log(f"  ESG sensitivity   : {len(df_esg)}/{3 * len(ESG_FLOORS)} solver calls feasible"
    f"  (3 estimators × {len(ESG_FLOORS)} floors)")
if baseline_result["weights"] is not None:
    log(f"  Baseline portfolio:  "
        f"ret={baseline_result['ret']:.4f}  "
        f"vol={baseline_result['vol']:.4f}  "
        f"sharpe={baseline_result['sharpe']:.4f}  "
        f"ESG={baseline_result['esg_score']:.1f}  "
        f"n={baseline_result['n_holdings']}")
log(f"\nOutputs in {RESULTS_DIR}/:")
log(f"  bloomberg_frontier_sample_unconstrained.csv")
log(f"  bloomberg_frontier_ledoit_wolf_unconstrained.csv")
log(f"  bloomberg_frontier_oas_unconstrained.csv")
log(f"  bloomberg_frontier_ledoit_wolf_esg55.csv")
log(f"  bloomberg_esg_sensitivity.csv")
log(f"  bloomberg_baseline_weights.csv")
log(f"  bloomberg_cov_comparison.csv")
log(f"\nFigures in {FIGURES_DIR}/:")
log(f"  step7_frontier.png  step7_esg_sensitivity.png  step7_cov_comparison.png")
log(f"\nReady for Step 9 — Out-of-sample analysis.")
log("=" * 65)

# =============================================================================
# SAVE LOG
# =============================================================================
log_path = f"{RESULTS_DIR}/bloomberg_optimisation_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\n[Log] Saved: {log_path}")