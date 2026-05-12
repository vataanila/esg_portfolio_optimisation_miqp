"""
05_bootstrap_simulated.py: Stationary Block Bootstrap for OOS Sharpe Ratios
=============================================================================
ESG-Constrained Portfolio Optimisation - Independent Research Project

PIPELINE POSITION:
  04_oos_simulated.py → data/results_simulated/simulated_oos_fixedvol_summary.csv
  05_bootstrap_simulated.py (this file) → data/results_simulated/simulated_bootstrap_summary.csv
                                           data/results_simulated/simulated_bootstrap_log.txt
                                           figures/step11_bootstrap_ci.png

DESIGN
  1. Re-run the 21 MIQP solves from step 04 to recover frozen portfolio weights
     (weights are NOT stored in simulated_oos_fixedvol_summary.csv).
  2. For each of the 21 portfolios, apply stationary block bootstrap
     (Politis & Romano, 1994) to the OOS test returns.
  3. Report 95% bootstrap confidence intervals for the annualised Sharpe ratio.

KEY DIFFERENCES vs 10_bootstrap_bloomberg.py

  Difference 1 -- mu specification:
    mu is loaded ONCE from metadata as meta["mu_winsor"] (winsorised FICO mu).
    It is NEVER re-estimated on the train window.
    Reason: in the simulated dataset the true expected returns are the
    pre-assigned FICO values. Re-estimating from returns would discard
    this information.

  Difference 2 -- data loading:
    Load from data/clean/ not data/clean_bloomberg/.
    Files: meta_preprocessed.csv and log_returns.csv.
    Universe size is read from meta at runtime (NOT hardcoded).

  Difference 3 -- output paths:
    All outputs go to data/results_simulated/ and figures/.
    File names use simulated_ prefix.

SOLVER: Gurobi 13.0 via cvxpy (academic licence)

Run:
    python src/05_bootstrap_simulated.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS
from arch.bootstrap import StationaryBootstrap

warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETERS  -- change only here, never inline
# =============================================================================

CLEAN_DIR   = "data/clean"
RESULTS_DIR = "data/results_simulated"
FIGURES_DIR = "figures"

# -- Portfolio constraints: identical to Step 4a / Step 3 ---------------------
W_MAX      = 0.20
W_MIN      = 0.01
K_MAX      = 100
K_MIN      = 50
SECTOR_CAP = 0.25

# -- Mu specification: FICO metadata (simulated data convention) ---------------
MU_COLUMN = "mu_winsor"   # loaded once from meta; never re-estimated on train

# -- Train/test split ----------------------------------------------------------
TRAIN_RATIO  = 0.70
ANNUAL_DAYS  = 252

# -- Covariance estimators (same naming as Step 4a / Step 3) ------------------
COV_METHODS = ["sample", "ledoit_wolf", "oas"]

# -- ESG floors (0-100 scale, same as Step 4a / Step 3) -----------------------
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# -- Vol-cap grid: identical to Step 4a / Step 3 ------------------------------
VOL_CAP_MIN       = 0.30
VOL_CAP_MAX       = 3.00
N_FRONTIER_POINTS = 30

# Step 4b fixed vol cap: loaded from Step 3 optimisation output
with open("data/results/selected_vol_cap.txt") as f:
    VOL_CAP_BOOTSTRAP = float(f.read().strip())
print(f"  Vol cap loaded from selected_vol_cap.txt: {VOL_CAP_BOOTSTRAP:.4f}")

# -- Solver settings: identical to Step 4a ------------------------------------
SOLVER_VERBOSE    = False
SOLVER_TIME_LIMIT = 120
SOLVER_MIP_GAP    = 1e-4

# -- Bootstrap settings --------------------------------------------------------
B          = 1000    # number of bootstrap resamples
CI_LEVEL   = 0.95   # confidence level for the interval
CI_LOW     = (1 - CI_LEVEL) / 2 * 100         # 2.5
CI_HIGH    = (1 - (1 - CI_LEVEL) / 2) * 100   # 97.5
BLOCK_LENGTH = 21   # fixed block length (~1 trading month); same as step9

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

_log_lines: list[str] = []
_log_path = f"{RESULTS_DIR}/simulated_bootstrap_log.txt"

def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)

def log_section(title: str) -> None:
    bar = "=" * 65
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

def save_log() -> None:
    with open(_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))
    log(f"\n  Log saved -> {_log_path}")

# =============================================================================
# LOAD AND ALIGN DATA  (simulated-data variant of Step 9)
# =============================================================================

def load_and_align_data():
    """
    Load simulated metadata and log-returns, apply the same train/test split
    as Step 4a.

    DIFFERENCE 1: mu loaded from meta["mu_winsor"] -- never re-estimated.
    DIFFERENCE 2: files from data/clean/, no EXPECTED_N assertion.

    Returns
    -------
    tickers    : list[str]
    mu_vec     : ndarray  (N,)  fixed from FICO metadata
    esg_vec    : ndarray  (N,)
    sector_map : dict[str, list[int]] or None
    n_train    : int
    r_train    : ndarray  (n_train x N)
    r_test     : ndarray  (n_test  x N)
    """
    log_section("LOADING DATA  --  Simulated dataset (meta_preprocessed + log_returns)")

    # -- Metadata --------------------------------------------------------------
    meta    = pd.read_csv(f"{CLEAN_DIR}/meta_preprocessed.csv", index_col=0)
    tickers = list(meta.index)
    N       = len(tickers)

    # -- Mu: loaded from FICO metadata -- NOT re-estimated on train window -----
    mu_vec  = meta[MU_COLUMN].values.astype(float)
    esg_vec = meta["esg"].values.astype(float)

    log(f"\n  Dataset         : Simulated (~{N} stocks, FICO metadata)")
    log(f"  Universe        : {N} stocks  (read from meta at runtime)")
    log(f"  Mu column       : '{MU_COLUMN}'  (FICO metadata -- NOT re-estimated)")
    log(f"  mu loaded from metadata (FICO) -- fixed, not re-estimated")
    log(f"  mu_winsor       : mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}"
        f"  range=[{mu_vec.min():.4f}, {mu_vec.max():.4f}]")
    log(f"  ESG column      : 'esg'  mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
    log(f"  Constraints     : W_MAX={W_MAX}  W_MIN={W_MIN}  "
        f"K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
    log(f"  ESG floors      : {ESG_FLOORS}")
    log(f"  Fixed vol cap   : {VOL_CAP_BOOTSTRAP:.4f}  "
        f"(max-Sharpe in central 60% of feasible unconstrained LW frontier)")

    # -- Returns ---------------------------------------------------------------
    returns_raw = pd.read_csv(
        f"{CLEAN_DIR}/log_returns.csv", index_col=0, parse_dates=True
    )
    if returns_raw.shape[0] < returns_raw.shape[1]:
        returns_raw = returns_raw.T
        log("  log_returns transposed -> (days x stocks)")
    try:
        returns_raw.index = pd.to_datetime(returns_raw.index)
    except Exception:
        pass  # synthetic labels (e.g. "Day_2") -- keep as strings

    common     = [t for t in tickers if t in returns_raw.columns]
    returns_df = returns_raw[common].copy()

    assert list(returns_df.columns) == list(meta.index), \
        "Final aligned returns columns must exactly equal meta.index."

    def _idx_str(idx_val):
        return idx_val.date() if hasattr(idx_val, "date") else str(idx_val)

    log(f"  Date range      : {_idx_str(returns_df.index[0])} "
        f"-> {_idx_str(returns_df.index[-1])}")

    # -- Sector map ------------------------------------------------------------
    sector_map = None
    if "sector" in meta.columns:
        sector_map = {}
        for sec, grp in meta.groupby("sector"):
            sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
        log(f"  Sectors         : {list(sector_map.keys())}")
    else:
        log("  WARNING: no 'sector' column - sector cap constraint disabled.")

    # -- Train / test split ----------------------------------------------------
    log_section("TRAIN / TEST SPLIT")
    n_days  = len(returns_df)
    n_train = int(n_days * TRAIN_RATIO)
    n_test  = n_days - n_train

    r_train = np.nan_to_num(
        returns_df.iloc[:n_train].values.astype(float), nan=0.0)
    r_test  = np.nan_to_num(
        returns_df.iloc[n_train:].values.astype(float), nan=0.0)

    log(f"  Total days  : {n_days}")
    log(f"  Train days  : {n_train}  "
        f"({_idx_str(returns_df.index[0])} -> {_idx_str(returns_df.index[n_train-1])})")
    log(f"  Test  days  : {n_test}   "
        f"({_idx_str(returns_df.index[n_train])} -> {_idx_str(returns_df.index[-1])})")

    return tickers, mu_vec, esg_vec, sector_map, n_train, r_train, r_test

# =============================================================================
# COVARIANCE ESTIMATION ON TRAIN WINDOW  (identical to Step 9)
# =============================================================================

def estimate_covariances(r_train: np.ndarray, N: int) -> dict:
    """
    Sample, Ledoit-Wolf, and OAS covariance matrices on the train window only.
    Same PSD regularisation as Step 4a / Step 3.
    """
    log_section("COVARIANCE ESTIMATION  (train window only - no look-ahead)")
    sigma_cache = {}

    # Sample covariance
    S = np.cov(r_train.T) * ANNUAL_DAYS
    S = (S + S.T) / 2
    S += np.eye(N) * 1e-8
    sigma_cache["sample"] = S
    log(f"  sample      : shape={S.shape}  "
        f"min_eig={np.linalg.eigvalsh(S).min():.6f}")

    # Ledoit-Wolf shrinkage
    lw   = LedoitWolf().fit(r_train)
    S_lw = lw.covariance_ * ANNUAL_DAYS
    S_lw = (S_lw + S_lw.T) / 2
    S_lw += np.eye(N) * 1e-8
    sigma_cache["ledoit_wolf"] = S_lw
    log(f"  ledoit_wolf : shrinkage={lw.shrinkage_:.4f}  "
        f"min_eig={np.linalg.eigvalsh(S_lw).min():.6f}")

    # Oracle Approximating Shrinkage
    oas_est = OAS().fit(r_train)
    S_oas   = oas_est.covariance_ * ANNUAL_DAYS
    S_oas   = (S_oas + S_oas.T) / 2
    S_oas   += np.eye(N) * 1e-8
    sigma_cache["oas"] = S_oas
    log(f"  oas         : shrinkage={oas_est.shrinkage_:.4f}  "
        f"min_eig={np.linalg.eigvalsh(S_oas).min():.6f}")

    return sigma_cache

# =============================================================================
# MIQP SOLVER  (identical formulation to Step 4a / Step 3)
# =============================================================================

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=0.0, apply_esg=True,
               sector_map=None,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN,
               sector_cap=SECTOR_CAP):
    """
    Solve one MIQP instance via cvxpy + Gurobi.
    Identical formulation to Step 4a / Step 3.
    apply_esg=False -> unconstrained (no ESG floor).

    Returns
    -------
    dict with keys: status, weights (None if infeasible)
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

    return {"status": prob.status, "weights": w_val}

# =============================================================================
# BOOTSTRAP SHARPE  (Politis & Romano, 1994)
# =============================================================================

def bootstrap_sharpe(r_test: np.ndarray,
                     weights: np.ndarray,
                     block_length: int,
                     B: int = 1000,
                     seed: int = 42) -> np.ndarray:
    """
    Stationary block bootstrap of the annualised Sharpe ratio.

    Parameters
    ----------
    r_test       : ndarray  (T x N)  OOS daily returns matrix
    weights      : ndarray  (N,)     frozen portfolio weights
    block_length : int               block length (fixed at 21 days)
    B            : int               number of bootstrap resamples
    seed         : int               random seed for reproducibility

    Returns
    -------
    sharpe_boot : ndarray  (B,)  bootstrapped Sharpe ratios
    """
    np.random.seed(seed)
    port_test  = r_test @ weights    # (T,) portfolio daily return series

    bs         = StationaryBootstrap(block_length, port_test)
    sharpe_boot = np.empty(B)

    for i, (boot_data, _) in enumerate(bs.bootstrap(B)):
        r_b = np.asarray(boot_data[0]).flatten()    # bootstrapped return series
        mu_b = r_b.mean() * ANNUAL_DAYS
        sd_b = r_b.std()  * np.sqrt(ANNUAL_DAYS)
        sharpe_boot[i] = mu_b / sd_b if sd_b > 1e-10 else np.nan

    return sharpe_boot


def annualised_sharpe(r_test: np.ndarray, weights: np.ndarray) -> float:
    """Point estimate of annualised Sharpe on the OOS window."""
    port    = r_test @ weights
    ann_ret = float(port.mean() * ANNUAL_DAYS)
    ann_vol = float(port.std()  * np.sqrt(ANNUAL_DAYS))
    return ann_ret / ann_vol if ann_vol > 1e-8 else np.nan

# =============================================================================
# MAIN: RECOVER WEIGHTS + RUN BOOTSTRAP FOR ALL 21 COMBINATIONS
# =============================================================================

def run_bootstrap_analysis(sigma_cache, mu_vec, esg_vec, sector_map,
                            r_train, r_test) -> pd.DataFrame:
    """
    3 estimators x (unconstrained + 6 ESG floors) = 21 MIQP solves.
    For each solved portfolio, apply stationary block bootstrap to the
    OOS test returns to quantify uncertainty around the Sharpe ratio.

    mu_vec is fixed from FICO metadata -- same vector passed to every solve.

    Returns
    -------
    results_df : DataFrame  (21 rows)
    """
    log_section(
        f"BOOTSTRAP ANALYSIS  (B={B},  CI={int(CI_LEVEL*100)}%,  "
        f"method=stationary block)"
    )
    log(f"  Fixed vol cap : {VOL_CAP_BOOTSTRAP:.4f}")
    log(f"  Grid          : {len(COV_METHODS)} estimators x "
        f"(1 unconstrained + {len(ESG_FLOORS)} ESG floors) "
        f"= {len(COV_METHODS) * (1 + len(ESG_FLOORS))} combinations\n")

    # -- Block length: fixed at 21 days (same as step9) ----------------------
    bl = BLOCK_LENGTH
    log(f"  Block length : {bl} days  (fixed, ~1 trading month)")
    log(f"  Rationale    : standard choice in empirical finance; "
        f"sqrt(T={r_test.shape[0]}) = {int(np.sqrt(r_test.shape[0]))}")

    rows   = []
    t_total = time.time()
    combo_idx = 0

    for cov in COV_METHODS:
        Sigma = sigma_cache[cov]

        # Build list of (case_label, esg_floor_value, apply_esg)
        combos = [("unconstrained", None, False)]
        for fl in ESG_FLOORS:
            combos.append(("esg", float(fl), True))

        for case_label, esg_floor_val, apply_esg in combos:
            combo_idx += 1
            fl_str = f"{esg_floor_val:.0f}" if esg_floor_val is not None else "-"
            tag    = f"[{combo_idx:2d}/21]  {cov:12s}  {case_label:14s}  esg>={fl_str}"

            # -- MIQP solve --------------------------------------------------
            t0  = time.time()
            res = solve_miqp(
                Sigma, mu_vec, esg_vec,
                vol_cap    = VOL_CAP_BOOTSTRAP,
                esg_floor  = esg_floor_val if esg_floor_val is not None else 0.0,
                apply_esg  = apply_esg,
                sector_map = sector_map,
            )
            t_solve = time.time() - t0

            if res["weights"] is None:
                log(f"  {tag}  ->  INFEASIBLE / ERROR: {res['status']}")
                rows.append({
                    "cov_method":    cov,
                    "case":          case_label,
                    "esg_floor":     esg_floor_val,
                    "oos_sharpe":    np.nan,
                    "ci_lower":      np.nan,
                    "ci_upper":      np.nan,
                    "ci_width":      np.nan,
                    "sig_from_zero": False,
                    "block_length":  bl,
                })
                continue

            weights = res["weights"]

            # -- Point-estimate Sharpe on OOS window -------------------------
            oos_sr = annualised_sharpe(r_test, weights)

            # -- Stationary block bootstrap ----------------------------------
            t1          = time.time()
            sr_boot     = bootstrap_sharpe(r_test, weights, bl, B=B)
            t_boot      = time.time() - t1

            ci_lo  = float(np.nanpercentile(sr_boot, CI_LOW))
            ci_hi  = float(np.nanpercentile(sr_boot, CI_HIGH))
            ci_w   = ci_hi - ci_lo
            sig    = bool(ci_lo > 0)

            log(f"  {tag}  OOS_SR={oos_sr:+.3f}  "
                f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  w={ci_w:.3f}  "
                f"sig={'YES' if sig else 'no '}  "
                f"solve={t_solve:.1f}s  boot={t_boot:.1f}s")

            rows.append({
                "cov_method":    cov,
                "case":          case_label,
                "esg_floor":     esg_floor_val,
                "oos_sharpe":    round(oos_sr,  6),
                "ci_lower":      round(ci_lo,   6),
                "ci_upper":      round(ci_hi,   6),
                "ci_width":      round(ci_w,    6),
                "sig_from_zero": sig,
                "block_length":  bl,
            })

    log(f"\n  Total elapsed : {time.time() - t_total:.1f}s")
    return pd.DataFrame(rows)

# =============================================================================
# FIGURE: HORIZONTAL CI CHART
# =============================================================================

def plot_bootstrap_ci(df: pd.DataFrame, path: str) -> None:
    """
    Horizontal bar chart with point estimates (dots) and 95% CI (lines).
    Red = CI includes zero; green = CI strictly above zero.
    """
    valid = df.dropna(subset=["oos_sharpe"]).copy()

    # Build a readable label for each row
    def make_label(row):
        if row["case"] == "unconstrained":
            return f"{row['cov_method']}  | unconstrained"
        return f"{row['cov_method']}  | ESG>={int(row['esg_floor'])}"

    valid["label"] = valid.apply(make_label, axis=1)
    valid = valid.reset_index(drop=True)

    n_rows   = len(valid)
    fig_h    = max(6, n_rows * 0.45 + 2)
    fig, ax  = plt.subplots(figsize=(10, fig_h))

    colours = ["#d62728" if not s else "#2ca02c"
               for s in valid["sig_from_zero"]]

    y_pos = np.arange(n_rows)

    for i, row in valid.iterrows():
        col = colours[i]
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i],
                color=col, linewidth=2.5, solid_capstyle="round", zorder=2)
        ax.plot(row["oos_sharpe"], i, "o",
                color=col, markersize=7, zorder=3)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=1, label="Sharpe = 0")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid["label"], fontsize=8.5)
    ax.invert_yaxis()

    ax.set_xlabel("Annualised Sharpe Ratio", fontsize=11)
    ax.set_title(
        f"Simulated Dataset \u2014 OOS Sharpe 95% Bootstrap CI\n"
        f"(B = {B} resamples, block length = {valid['block_length'].iloc[0]} days)",
        fontsize=12, fontweight="bold",
    )

    # Legend patches
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ca02c", linewidth=2.5, marker="o",
               markersize=7, label="CI > 0 entirely (significant)"),
        Line2D([0], [0], color="#d62728", linewidth=2.5, marker="o",
               markersize=7, label="CI includes 0 (not significant)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
               label="Sharpe = 0"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.85)

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Figure saved -> {path}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    log_section("SUMMARY TABLE")
    hdr = (f"  {'cov_method':12s}  {'case':14s}  {'esg_fl':6s}  "
           f"{'OOS_SR':7s}  {'CI_lo':7s}  {'CI_hi':7s}  "
           f"{'width':6s}  {'sig':3s}  {'bl':3s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))

    for _, row in df.iterrows():
        fl_str = f"{int(row['esg_floor'])}" if pd.notna(row["esg_floor"]) else "-"
        sr_str = f"{row['oos_sharpe']:+.3f}" if pd.notna(row["oos_sharpe"]) else " n/a "
        lo_str = f"{row['ci_lower']:+.3f}"  if pd.notna(row["ci_lower"])  else " n/a "
        hi_str = f"{row['ci_upper']:+.3f}"  if pd.notna(row["ci_upper"])  else " n/a "
        wd_str = f"{row['ci_width']:.3f}"   if pd.notna(row["ci_width"])  else " n/a "
        sig_str = "YES" if row["sig_from_zero"] else "no"
        bl_str  = str(int(row["block_length"])) if pd.notna(row["block_length"]) else "-"

        log(f"  {row['cov_method']:12s}  {row['case']:14s}  {fl_str:6s}  "
            f"{sr_str:7s}  {lo_str:7s}  {hi_str:7s}  "
            f"{wd_str:6s}  {sig_str:3s}  {bl_str:3s}")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    t_pipeline = time.time()

    log("=" * 65)
    log("  STEP 4B - STATIONARY BLOCK BOOTSTRAP FOR OOS SHARPE RATIOS")
    log("  Simulated Dataset  |  ESG-Constrained Portfolio Optimisation")
    log("=" * 65)
    log(f"  Bootstrap resamples : B = {B}")
    log(f"  CI level            : {int(CI_LEVEL*100)}%  (percentile method)")
    log(f"  Block length        : {BLOCK_LENGTH} days  (fixed -- same as step9)")
    log(f"  Solver              : Gurobi 13.0 via cvxpy")
    log(f"  Vol cap             : {VOL_CAP_BOOTSTRAP:.4f}  (Step 3 / 4a Task 2 rule)")

    # -- Load data -------------------------------------------------------------
    tickers, mu_vec, esg_vec, sector_map, n_train, r_train, r_test = \
        load_and_align_data()
    N = len(tickers)

    # -- Covariance on train window --------------------------------------------
    sigma_cache = estimate_covariances(r_train, N)

    # -- Mu: fixed from metadata (FICO) -- NOT re-estimated --------------------
    log_section("MU SPECIFICATION  (FICO metadata -- fixed, not re-estimated)")
    log(f"  mu loaded from metadata (FICO) -- fixed, not re-estimated")
    log(f"  mu_winsor : mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}  "
        f"range=[{mu_vec.min():.4f}, {mu_vec.max():.4f}]")
    log(f"  Same mu passed to every MIQP solve regardless of train/test split")

    # -- 21 MIQP solves + bootstrap -------------------------------------------
    results_df = run_bootstrap_analysis(
        sigma_cache, mu_vec, esg_vec, sector_map, r_train, r_test
    )

    # -- Save CSV --------------------------------------------------------------
    log_section("SAVING OUTPUTS")
    csv_path = f"{RESULTS_DIR}/simulated_bootstrap_summary.csv"
    results_df.to_csv(csv_path, index=False)
    log(f"  CSV saved  -> {csv_path}  ({len(results_df)} rows)")

    # -- Figure ----------------------------------------------------------------
    fig_path = f"{FIGURES_DIR}/step5_bootstrap_ci.png"
    plot_bootstrap_ci(results_df, fig_path)

    # -- Print summary ---------------------------------------------------------
    print_summary(results_df)

    log_section("PIPELINE COMPLETE")
    log(f"  Total wall time : {time.time() - t_pipeline:.1f}s")
    log(f"  Significant (CI > 0): "
        f"{results_df['sig_from_zero'].sum()} / {len(results_df)} portfolios")

    # -- Save log --------------------------------------------------------------
    save_log()


if __name__ == "__main__":
    main()
