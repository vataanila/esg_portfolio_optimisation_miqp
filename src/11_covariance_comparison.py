"""
11_covariance_comparison.py: Covariance Estimator Comparison
=============================================================
Sample vs Ledoit-Wolf vs OAS
ESG-Constrained Portfolio Optimisation - Independent Research Project

PIPELINE POSITION:
  02_preprocess_simulated.py → data/clean/meta_preprocessed.csv
                                data/clean/sigma_*.npy
                                data/clean/log_returns.csv
  07_preprocess_bloomberg.py → data/clean_bloomberg/bloomberg_meta_analytical.csv
                                data/clean_bloomberg/bloomberg_Sigma_*.csv
                                data/clean_bloomberg/bloomberg_returns.csv
  11_covariance_comparison.py (this file) →
      data/results/covariance_comparison_results.csv   (all frontier rows)
      data/results/covariance_comparison_table.csv     (summary per combo)
      figures/fig_cov_comparison_simulated.png
      figures/fig_cov_comparison_bloomberg.png

PURPOSE:
  Compare three covariance estimators (Sample, Ledoit-Wolf, OAS) on the
  portfolio optimisation problem using BOTH the simulated (FICO) and
  Bloomberg datasets.  For each dataset × estimator combination (6 total)
  we compute the ESG-constrained efficient frontier via a volatility-cap
  sweep and assess whether the choice of estimator materially affects the
  portfolio's risk-return trade-off.

MODEL (identical to 03_optimize_simulated.py and 08_optimize_bloomberg.py):
  Objective:  maximise  w'mu
  s.t.        w'Sigma w  <= vol_cap^2    (vol-cap sweep for frontier)
              sum(w)      = 1            (budget)
              w'esg      >= ESG_FLOOR    (ESG floor, fixed at 55)
              w_i        <= W_MAX * z_i  (upper bound)
              w_i        >= W_MIN * z_i  (lower bound if selected)
              sum(z_i)   <= K            (max cardinality = 10)
              sum(z_i)   >= K_MIN        (min holdings)
              w_i        >= 0            (long-only)
              z_i in {0,1}              (binary selection)

KEY DIFFERENCES FROM MAIN PIPELINE (steps 03 and 08):
  - Cardinality capped at K_MAX=10: smaller portfolios amplify differences
    between covariance estimators, which is the focus of this step.
  - K_MIN=5 enforced (prevents degenerate 1-stock solutions).
  - ESG floor fixed at 55 for all runs (not swept here).
  - Sector cap not applied to keep the model lean for this comparison.

Run:
    python src/11_covariance_comparison.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe on Windows
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS

warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETERS -- change only here, never inline
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -- Portfolio constraints -----------------------------------------------------
W_MAX  = 0.20   # max weight per stock (20%)
W_MIN  = 0.01   # min weight if a stock is selected (1%)
K_MAX  = 10     # cardinality cap: at most 10 stocks
K_MIN  = 5      # min holdings (prevents degenerate 1-stock solutions)

# -- ESG constraint (fixed for this comparison step) --------------------------
ESG_FLOOR = 55.0

# -- Frontier vol-cap sweep ---------------------------------------------------
N_FRONTIER_POINTS = 30   # number of vol-cap grid points per estimator

# Per-dataset vol-cap ranges (calibrated to each dataset's return/vol scale)
VOL_CAPS = {
    "simulated": (0.30, 3.00),   # FICO synthetic data  -- inflated vol scale
    "bloomberg": (0.10, 0.80),   # Real equity data     -- typical 10-80% ann vol
}

# -- Gurobi solver settings ---------------------------------------------------
SOLVER_VERBOSE    = False
SOLVER_TIME_LIMIT = 60      # seconds per solve (as specified)
SOLVER_MIP_GAP    = 0.01    # 1% relative MIP optimality gap (as specified)

# =============================================================================
# DATASET CONFIGURATIONS
#
# mu_col choices (match the main spec used in steps 3/7):
#   simulated -> "mu_native"          : native FICO mu (main spec, per step3)
#   bloomberg -> "mu_trailing_winsor" : 3-yr trailing winsorised mu
#                                       (main spec, per step7 USE_TRAILING_MU=True)
#
# sigma_fmt:
#   "npy" for simulated (step2 saves numpy binary .npy files)
#   "csv" for Bloomberg  (step6 saves labelled CSV files)
# =============================================================================

DATASETS = {
    "simulated": {
        "meta_path":    os.path.join(BASE_DIR, "data", "clean",
                                     "meta_preprocessed.csv"),
        "returns_path": os.path.join(BASE_DIR, "data", "clean",
                                     "log_returns.csv"),
        "mu_col":       "mu_native",
        "esg_col":      "esg",
        "sigma_files": {
            "sample":      os.path.join(BASE_DIR, "data", "clean",
                                        "sigma_sample_annual.npy"),
            "ledoit_wolf": os.path.join(BASE_DIR, "data", "clean",
                                        "sigma_lw_annual.npy"),
            "oas":         os.path.join(BASE_DIR, "data", "clean",
                                        "sigma_oas_annual.npy"),
        },
        "sigma_fmt": "npy",
    },
    "bloomberg": {
        "meta_path":    os.path.join(BASE_DIR, "data", "clean_bloomberg",
                                     "bloomberg_meta_analytical.csv"),
        "returns_path": os.path.join(BASE_DIR, "data", "clean_bloomberg",
                                     "bloomberg_returns.csv"),
        "mu_col":       "mu_trailing_winsor",
        "esg_col":      "esg",
        "sigma_files": {
            "sample":      os.path.join(BASE_DIR, "data", "clean_bloomberg",
                                        "bloomberg_Sigma_sample.csv"),
            "ledoit_wolf": os.path.join(BASE_DIR, "data", "clean_bloomberg",
                                        "bloomberg_Sigma_lw.csv"),
            "oas":         os.path.join(BASE_DIR, "data", "clean_bloomberg",
                                        "bloomberg_Sigma_oas.csv"),
        },
        "sigma_fmt": "csv",
    },
}

# -- Display names and colours (consistent across both figures) ---------------
ESTIMATOR_LABELS = {
    "sample":      "Sample",
    "ledoit_wolf": "Ledoit-Wolf",
    "oas":         "OAS",
}
ESTIMATOR_COLORS = {
    "sample":      "#1f77b4",   # blue
    "ledoit_wolf": "#ff7f0e",   # orange
    "oas":         "#2ca02c",   # green
}

# =============================================================================
# LOGGING UTILITY
# =============================================================================

log_lines = []

def log(msg: str = "") -> None:
    """Print to console and accumulate for end-of-run reference."""
    print(msg)
    log_lines.append(str(msg))


# =============================================================================
# CORE OPTIMISATION FUNCTION
# =============================================================================

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=ESG_FLOOR,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN):
    """
    Solve one ESG-constrained MIQP instance via cvxpy + Gurobi.

    Objective:  maximise  w'mu
    Subject to:
        w'Sigma w  <= vol_cap^2      (volatility cap -- swept to trace frontier)
        sum(w)      = 1              (fully invested)
        w'esg      >= esg_floor      (ESG floor constraint)
        w          <= w_max * z      (upper-bound links weights to selection)
        w          >= w_min * z      (lower-bound if selected)
        sum(z)     <= k_max          (cardinality upper bound)
        sum(z)     >= k_min          (cardinality lower bound)
        w          >= 0              (long-only, via nonneg=True in cp.Variable)
        z in {0,1}                   (binary selection indicators)

    Parameters
    ----------
    Sigma    : (N,N) annualised covariance matrix -- must be PSD
    mu       : (N,) annualised expected return vector
    esg      : (N,) ESG scores on 0-100 scale
    vol_cap  : annualised volatility cap (scalar)
    esg_floor: minimum portfolio-average ESG score
    w_max    : max weight per stock
    w_min    : min weight if selected
    k_max    : max number of stocks held
    k_min    : min number of stocks held

    Returns
    -------
    dict with keys:
        status     : Gurobi/cvxpy status string
        weights    : (N,) weight array, or None on failure
        ret        : annualised portfolio return
        vol        : annualised portfolio volatility
        sharpe     : Sharpe ratio (ret/vol, risk-free = 0)
        esg_score  : portfolio-average ESG score
        n_holdings : number of stocks with weight > 1e-4
    """
    n = len(mu)
    w = cp.Variable(n, nonneg=True)    # continuous weights (long-only)
    z = cp.Variable(n, boolean=True)   # binary selection indicators

    objective = cp.Maximize(mu @ w)

    constraints = [
        cp.sum(w) == 1,                          # fully invested
        cp.quad_form(w, Sigma) <= vol_cap ** 2,  # volatility cap
        w <= w_max * z,                          # upper bound
        w >= w_min * z,                          # lower bound if selected
        cp.sum(z) <= k_max,                      # cardinality cap
        cp.sum(z) >= k_min,                      # min holdings
        esg @ w >= esg_floor,                    # ESG floor
    ]

    prob = cp.Problem(objective, constraints)

    # Wrap solve in try/except: a single infeasible or failed point should NOT
    # abort the entire frontier sweep -- just skip that point and continue.
    try:
        prob.solve(
            solver=cp.GUROBI,
            verbose=SOLVER_VERBOSE,
            TimeLimit=SOLVER_TIME_LIMIT,
            MIPGap=SOLVER_MIP_GAP,
        )
    except Exception as exc:
        return {"status": f"solver_error: {exc}", "weights": None}

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "weights": None}

    # Sanitise weights: clip tiny negatives from numerical noise, renormalise
    w_val = np.array(w.value).flatten()
    w_val = np.clip(w_val, 0.0, None)
    w_val /= w_val.sum()

    port_ret    = float(mu @ w_val)
    port_var    = float(w_val @ Sigma @ w_val)
    port_vol    = float(np.sqrt(max(port_var, 0.0)))
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
# SHRINKAGE COEFFICIENT HELPER
# =============================================================================

def compute_shrinkage(returns_df, tickers, estimator_name):
    """
    Re-fit the sklearn estimator on daily log-returns to extract the scalar
    shrinkage coefficient (alpha).

    The definitive covariance matrices were already saved by step2/step6 and
    are loaded directly for optimisation.  This function is only needed to
    fill the Shrinkage_Coefficient column in the summary table.

    Parameters
    ----------
    returns_df     : DataFrame  (index=dates, columns=tickers)
    tickers        : list of tickers in the optimisation universe
    estimator_name : "sample", "ledoit_wolf", or "oas"

    Returns
    -------
    float or None -- None for sample covariance (no shrinkage)
    """
    if estimator_name == "sample":
        return None   # classical sample covariance has no shrinkage parameter

    # Restrict to tickers present in both the universe and the returns file
    cols = [t for t in tickers if t in returns_df.columns]
    if not cols:
        return None

    R = returns_df[cols].dropna().values   # shape (T, N_available)

    try:
        if estimator_name == "ledoit_wolf":
            return float(LedoitWolf().fit(R).shrinkage_)
        elif estimator_name == "oas":
            return float(OAS().fit(R).shrinkage_)
    except Exception as exc:
        log(f"     WARNING: shrinkage re-fit failed -- {exc}")
    return None


# =============================================================================
# MAIN LOOP  --  iterate over datasets, then over estimators
# =============================================================================

log("=" * 70)
log("  STEP 10 -- COVARIANCE ESTIMATOR COMPARISON")
log("  Sample  vs  Ledoit-Wolf  vs  OAS")
log("  Datasets: Simulated (FICO)  and  Bloomberg")
log("=" * 70)
log(f"\n  ESG floor (fixed) : {ESG_FLOOR}")
log(f"  Cardinality cap   : K_MAX = {K_MAX}")
log(f"  W_MAX / W_MIN     : {W_MAX} / {W_MIN}")
log(f"  Frontier points   : {N_FRONTIER_POINTS}")
log(f"  Solver time limit : {SOLVER_TIME_LIMIT}s  |  MIP gap: {SOLVER_MIP_GAP*100:.0f}%")

all_rows     = []   # every frontier row  ->  covariance_comparison_results.csv
summary_rows = []   # one row per combo   ->  covariance_comparison_table.csv

for ds_name, ds_cfg in DATASETS.items():

    log("\n" + "=" * 70)
    log(f"  DATASET: {ds_name.upper()}")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------------------
    meta = pd.read_csv(ds_cfg["meta_path"], index_col=0)
    meta.index   = meta.index.astype(str).str.strip()
    meta.columns = meta.columns.str.strip()
    tickers = list(meta.index)
    N       = len(tickers)

    mu_col  = ds_cfg["mu_col"]
    esg_col = ds_cfg["esg_col"]

    # Validate that the expected mu column actually exists
    if mu_col not in meta.columns:
        log(f"  ERROR: mu column '{mu_col}' not found in meta. "
            f"Available columns: {list(meta.columns)}")
        log("  Skipping this dataset.")
        continue

    mu_vec  = meta[mu_col].values.astype(float)
    esg_vec = meta[esg_col].values.astype(float)

    log(f"\n  Universe   : {N} stocks")
    log(f"  Mu column  : '{mu_col}'  "
        f"mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}")
    log(f"  ESG column : '{esg_col}'  "
        f"mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")

    # -------------------------------------------------------------------------
    # Geomspace vol-cap grid (denser at low vol where frontier curves sharply)
    # -------------------------------------------------------------------------
    vc_min, vc_max = VOL_CAPS[ds_name]
    vol_caps = np.geomspace(vc_min, vc_max, N_FRONTIER_POINTS)
    log(f"  Vol caps   : geomspace({vc_min}, {vc_max}, n={N_FRONTIER_POINTS})")
    log(f"  First 5    : {np.round(vol_caps[:5], 4).tolist()}")

    # -------------------------------------------------------------------------
    # Load daily returns (used ONLY for shrinkage extraction -- NOT optimisation)
    # -------------------------------------------------------------------------
    returns_df = None
    try:
        returns_df = pd.read_csv(ds_cfg["returns_path"], index_col=0)
        # Orientation check: expect (T_days, N_stocks). If rows < cols and cols == N,
        # the file was saved transposed (stocks × dates) — correct it.
        if returns_df.shape[0] < returns_df.shape[1] and returns_df.shape[0] == N:
            returns_df = returns_df.T
            log(f"  Returns    : transposed to (dates x stocks) — was stored as (stocks x dates)")
        log(f"  Returns    : {returns_df.shape}  (dates x stocks, shrinkage only)")
    except Exception as exc:
        log(f"  WARNING: returns file not loaded -- Shrinkage_Coefficient will be NaN. ({exc})")

    # -------------------------------------------------------------------------
    # Loop over the three covariance estimators
    # -------------------------------------------------------------------------
    for est_name in ESTIMATOR_LABELS:

        sigma_path = ds_cfg["sigma_files"][est_name]
        label      = ESTIMATOR_LABELS[est_name]

        log(f"\n  -- Estimator: {label} " + "-" * max(1, 22 - len(label)))
        log(f"     File    : {os.path.basename(sigma_path)}")

        # Load covariance matrix from the format saved by step 02 / step 07
        if ds_cfg["sigma_fmt"] == "npy":
            # Simulated: saved as (N,N) numpy binary by step2
            Sigma = np.load(sigma_path)
        else:
            # Bloomberg: saved as labelled CSV (tickers as index/columns) by step6
            S_df  = pd.read_csv(sigma_path, index_col=0)
            S_df  = S_df.loc[tickers, tickers]   # align to universe order
            Sigma = S_df.values.astype(float)

        # Enforce exact symmetry and add small diagonal regulariser for PSD
        Sigma  = (Sigma + Sigma.T) / 2
        Sigma += np.eye(N) * 1e-8

        log(f"     Shape   : {Sigma.shape}")

        # Re-fit sklearn estimator on raw returns to get the shrinkage scalar
        shrinkage = None
        if returns_df is not None:
            shrinkage = compute_shrinkage(returns_df, tickers, est_name)
            if shrinkage is not None:
                log(f"     Shrinkage (refitted on returns): {shrinkage:.4f}")
            elif est_name != "sample":
                log(f"     Shrinkage: could not compute (ticker mismatch or fit error)")

        # -- Frontier sweep ---------------------------------------------------
        rows     = []
        n_solved = 0
        t0       = time.time()

        for i, vc in enumerate(vol_caps):

            # solve_miqp returns {"weights": None} on infeasible/error -- skip
            result = solve_miqp(Sigma, mu_vec, esg_vec, vol_cap=vc)

            if result["weights"] is not None:
                sharpe_val = result["sharpe"]
                rows.append({
                    "dataset":    ds_name,
                    "estimator":  est_name,
                    "vol_cap":    round(float(vc), 6),
                    "ret":        round(result["ret"],      6),
                    "vol":        round(result["vol"],      6),
                    "sharpe":     round(sharpe_val, 4) if not np.isnan(sharpe_val) else np.nan,
                    "esg_score":  round(result["esg_score"], 4),
                    "n_holdings": result["n_holdings"],
                    "status":     result["status"],
                })
                n_solved += 1
                log(f"     [{i+1:02d}/{N_FRONTIER_POINTS}]  vc={vc:.4f}  "
                    f"ret={result['ret']:.4f}  vol={result['vol']:.4f}  "
                    f"Sharpe={result['sharpe']:.3f}  "
                    f"ESG={result['esg_score']:.1f}  "
                    f"n_held={result['n_holdings']}  [{result['status']}]")
            else:
                log(f"     [{i+1:02d}/{N_FRONTIER_POINTS}]  vc={vc:.4f}  "
                    f"SKIPPED -- {result['status']}")

        elapsed = time.time() - t0
        log(f"\n     Feasible : {n_solved}/{N_FRONTIER_POINTS}  ({elapsed:.1f}s)")

        # Accumulate this estimator's frontier rows into the global list
        all_rows.extend(rows)

        # -- Summary statistics for this (dataset, estimator) combo ----------
        min_vol, max_sr_ret, max_sr_vol, max_sr_ratio = np.nan, np.nan, np.nan, np.nan
        if rows:
            df_f     = pd.DataFrame(rows)
            df_valid = df_f.dropna(subset=["sharpe"])
            min_vol_row = df_f.loc[df_f["vol"].idxmin()]
            max_sr_row  = (df_valid.loc[df_valid["sharpe"].idxmax()]
                           if not df_valid.empty else df_f.iloc[0])
            min_vol      = round(float(min_vol_row["vol"]),    4)
            max_sr_ret   = round(float(max_sr_row["ret"]),     4)
            max_sr_vol   = round(float(max_sr_row["vol"]),     4)
            max_sr_ratio = round(float(max_sr_row["sharpe"]),  4)

        summary_rows.append({
            "Dataset":                ds_name,
            "Estimator":              label,
            "Min_Vol":                min_vol,
            "Max_Sharpe_Return":      max_sr_ret,
            "Max_Sharpe_Vol":         max_sr_vol,
            "Max_Sharpe_Ratio":       max_sr_ratio,
            "Shrinkage_Coefficient":  round(shrinkage, 4) if shrinkage is not None else "N/A",
        })


# =============================================================================
# SAVE RESULTS
# =============================================================================

log("\n" + "=" * 70)
log("  SAVING OUTPUTS")
log("=" * 70)

# 1. Full frontier results -- one row per (dataset, estimator, vol_cap point)
df_all = pd.DataFrame(all_rows)
path_results = os.path.join(RESULTS_DIR, "covariance_comparison_results.csv")
df_all.to_csv(path_results, index=False)
log(f"\n  [1] Frontier results  : {path_results}")
log(f"      Total rows        : {len(df_all)}")

# 2. Summary table -- one row per (dataset, estimator) combination
df_summary = pd.DataFrame(summary_rows, columns=[
    "Dataset", "Estimator",
    "Min_Vol", "Max_Sharpe_Return", "Max_Sharpe_Vol",
    "Max_Sharpe_Ratio", "Shrinkage_Coefficient",
])
path_table = os.path.join(RESULTS_DIR, "covariance_comparison_table.csv")
df_summary.to_csv(path_table, index=False)
log(f"\n  [2] Summary table     : {path_table}")
log("\n" + df_summary.to_string(index=False))


# =============================================================================
# FIGURES
# =============================================================================

def make_frontier_figure(df_all, dataset_name, save_path):
    """
    Draw one academic-style figure with 3 efficient frontiers (one per
    covariance estimator) for the given dataset.

    X-axis : annualised volatility (%)
    Y-axis : annualised expected return (%)
    Legend : estimator name + average Sharpe ratio computed over the frontier
    """
    df = df_all[df_all["dataset"] == dataset_name].copy()
    if df.empty:
        log(f"  WARNING: no data for dataset '{dataset_name}' -- figure skipped.")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for est_name in ESTIMATOR_LABELS:
        sub = df[df["estimator"] == est_name].sort_values("vol")
        if sub.empty:
            continue

        avg_sr       = sub["sharpe"].dropna().mean()
        legend_label = (f"{ESTIMATOR_LABELS[est_name]}"
                        f"  (avg Sharpe = {avg_sr:.2f})")

        ax.plot(
            sub["vol"] * 100,   # annualised vol in % for readability
            sub["ret"] * 100,   # annualised return in %
            color=ESTIMATOR_COLORS[est_name],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=legend_label,
        )

    dataset_title = "Simulated (FICO)" if dataset_name == "simulated" else "Bloomberg"
    ax.set_title(
        f"Efficient Frontier -- {dataset_title}\n"
        f"Covariance Estimator Comparison  "
        f"(ESG floor = {ESG_FLOOR:.0f},  K <= {K_MAX})",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualised Return (%)",     fontsize=11)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved : {save_path}")


log("\n" + "-" * 70)
log("  GENERATING FIGURES")
log("-" * 70)

make_frontier_figure(
    df_all,
    dataset_name="simulated",
    save_path=os.path.join(FIGURES_DIR, "fig_cov_comparison_simulated.png"),
)

make_frontier_figure(
    df_all,
    dataset_name="bloomberg",
    save_path=os.path.join(FIGURES_DIR, "fig_cov_comparison_bloomberg.png"),
)

# =============================================================================
# DONE
# =============================================================================

log("\n" + "=" * 70)
log("  STEP 11 COMPLETE")
log("=" * 70)
log(f"  Frontier results : data/results/covariance_comparison_results.csv")
log(f"  Summary table    : data/results/covariance_comparison_table.csv")
log(f"  Figure (sim)     : figures/fig_cov_comparison_simulated.png")
log(f"  Figure (bbg)     : figures/fig_cov_comparison_bloomberg.png")
