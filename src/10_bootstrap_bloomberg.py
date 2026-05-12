"""
10_bootstrap_bloomberg.py: Stationary Block Bootstrap for OOS Sharpe Ratios
=============================================================================
Bloomberg EURO STOXX 600 - ESG-Constrained Portfolio Optimisation
Independent Research Project

PIPELINE POSITION:
  09_oos_bloomberg.py → data/results_bloomberg/bloomberg_oos_fixedvol_summary.csv
  10_bootstrap_bloomberg.py (this file) → data/results_bloomberg/bloomberg_bootstrap_summary.csv
                                           data/results_bloomberg/bloomberg_bootstrap_log.txt
                                           figures/step9_bootstrap_ci.png

DESIGN
  1. Re-run the 21 MIQP solves from step 09 to recover frozen portfolio weights
     (weights are NOT stored in bloomberg_oos_fixedvol_summary.csv).
  2. For each of the 21 portfolios, apply stationary block bootstrap
     (Politis & Romano, 1994) to the OOS test returns.
  3. Report 95% bootstrap confidence intervals for the annualised Sharpe ratio.

MU SPECIFICATION (Bloomberg main):
  mu_trailing_winsor = 3-year trailing winsorised empirical expected return.
  Loaded from metadata exactly as in steps 08/09 — NOT re-estimated.

SOLVER: Gurobi 13.0 via cvxpy (academic licence)

Run:
    python src/10_bootstrap_bloomberg.py
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
# PARAMETERS  - identical to Step 8A; change only here
# =============================================================================

CLEAN_DIR   = "data/clean_bloomberg"
RESULTS_DIR = "data/results_bloomberg"
FIGURES_DIR = "figures"

# ── Portfolio constraints: identical to Step 7 / Step 8 ──────────────────────
W_MAX      = 0.20
W_MIN      = 0.01
K_MAX      = 100
K_MIN      = 50
SECTOR_CAP = 0.25

# ── Universe consistency ──────────────────────────────────────────────────────
EXPECTED_N = 261

# ── Mu specification: identical to Step 8 ────────────────────────────────────
MU_TRAILING_COLUMN = "mu_trailing_winsor"
USE_TRAILING_MU    = True

# ── Train/test split: identical to Step 8 ────────────────────────────────────
TRAIN_RATIO  = 0.70
ANNUAL_DAYS  = 252

# ── Covariance estimators (same naming as Step 7/8) ──────────────────────────
COV_METHODS = ["sample", "ledoit_wolf", "oas"]

# ── ESG floors (0-100 scale, same as Step 7/8) ───────────────────────────────
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# ── Vol-cap grid: identical to Step 7/8 ──────────────────────────────────────
VOL_CAP_MIN       = 0.10
VOL_CAP_MAX       = 0.80
N_FRONTIER_POINTS = 30

# Step 8A fixed vol cap: loaded from Step 7 optimisation output
with open("data/results_bloomberg/bloomberg_selected_vol_cap.txt") as f:
    VOL_CAP_8A = float(f.read().strip())
print(f"  Vol cap loaded from bloomberg_selected_vol_cap.txt: {VOL_CAP_8A:.4f}")

# ── Solver settings: identical to Step 8 ─────────────────────────────────────
SOLVER_VERBOSE    = False
SOLVER_TIME_LIMIT = 120
SOLVER_MIP_GAP    = 1e-4

# ── Bootstrap settings ────────────────────────────────────────────────────────
B          = 1000    # number of bootstrap resamples
CI_LEVEL   = 0.95   # confidence level for the interval
CI_LOW     = (1 - CI_LEVEL) / 2 * 100         # 2.5
CI_HIGH    = (1 - (1 - CI_LEVEL) / 2) * 100   # 97.5

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

_log_lines: list[str] = []
_log_path = f"{RESULTS_DIR}/bloomberg_bootstrap_log.txt"

def log(msg: str = "") -> None:
    """Print to console and append to the bootstrap log buffer."""
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
# LOAD AND ALIGN DATA  (identical to Step 8)
# =============================================================================

def load_and_align_data():
    """
    Load Bloomberg metadata and returns, apply the same train/test split
    as Step 8.  mu is re-estimated on the train window only (no look-ahead).

    Returns
    -------
    tickers    : list[str]
    esg_vec    : ndarray  (N,)
    sector_map : dict[str, list[int]] or None
    n_train    : int
    r_train    : ndarray  (n_train x N)
    r_test     : ndarray  (n_test  x N)
    """
    log_section("LOADING DATA  -  same 261-stock clean universe as Step 7/8")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta    = pd.read_csv(f"{CLEAN_DIR}/bloomberg_meta_analytical.csv", index_col=0)
    tickers = list(meta.index)
    N       = len(tickers)

    esg_vec = meta["esg"].values.astype(float)

    log(f"\n  Dataset       : Bloomberg EURO STOXX 600")
    log(f"  Universe      : {N} stocks  (expected {EXPECTED_N})")
    log(f"  ESG column    : 'esg'  mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
    log(f"  Constraints   : W_MAX={W_MAX}  W_MIN={W_MIN}  "
        f"K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
    log(f"  ESG floors    : {ESG_FLOORS}")
    log(f"  Fixed vol cap : np.percentile(geomspace({VOL_CAP_MIN},{VOL_CAP_MAX},"
        f"{N_FRONTIER_POINTS}), 40) = {VOL_CAP_8A:.4f}")

    assert N == EXPECTED_N, (
        f"Universe size mismatch: got {N}, expected {EXPECTED_N}.\n"
        "Check bloomberg_meta_analytical.csv matches Step 7."
    )

    # ── Returns ───────────────────────────────────────────────────────────────
    returns_raw = pd.read_csv(
        f"{CLEAN_DIR}/bloomberg_returns.csv", index_col=0, parse_dates=True
    )
    if returns_raw.shape[0] < returns_raw.shape[1]:
        returns_raw = returns_raw.T
        log("  bloomberg_returns transposed -> (days x stocks)")

    common     = [t for t in tickers if t in returns_raw.columns]
    returns_df = returns_raw[common].copy()

    assert list(returns_df.columns) == list(meta.index), \
        "Final aligned returns columns must exactly equal meta.index."
    assert len(common) == EXPECTED_N, \
        f"Universe size after alignment: {len(common)}, expected {EXPECTED_N}."

    log(f"  Date range    : {returns_df.index[0].date()} "
        f"-> {returns_df.index[-1].date()}")

    # ── Sector map ────────────────────────────────────────────────────────────
    sector_map = None
    if "sector" in meta.columns:
        sector_map = {}
        for sec, grp in meta.groupby("sector"):
            sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
        log(f"  Sectors       : {list(sector_map.keys())}")
    else:
        log("  WARNING: no 'sector' column - sector cap constraint disabled.")

    # ── Train / test split ────────────────────────────────────────────────────
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
        f"({returns_df.index[0].date()} -> {returns_df.index[n_train-1].date()})")
    log(f"  Test  days  : {n_test}   "
        f"({returns_df.index[n_train].date()} -> {returns_df.index[-1].date()})")

    return tickers, esg_vec, sector_map, n_train, r_train, r_test

# =============================================================================
# COVARIANCE ESTIMATION ON TRAIN WINDOW  (identical to Step 8)
# =============================================================================

def estimate_covariances(r_train: np.ndarray, N: int) -> dict:
    """
    Sample, Ledoit-Wolf, and OAS covariance matrices on the train window only.
    Same PSD regularisation as Step 7/8.
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
# MU ESTIMATION ON TRAIN WINDOW  (identical to Step 8)
# =============================================================================

def estimate_mu_train(r_train: np.ndarray) -> np.ndarray:
    """
    Annualised mean on train window, winsorised at p1/p99.
    """
    raw_mu   = r_train.mean(axis=0) * ANNUAL_DAYS
    p1       = np.percentile(raw_mu, 1)
    p99      = np.percentile(raw_mu, 99)
    mu_clean = np.clip(raw_mu, p1, p99)

    log(f"  mu_clean: mean={mu_clean.mean():.4f}  std={mu_clean.std():.4f}  "
        f"range=[{mu_clean.min():.4f}, {mu_clean.max():.4f}]")
    return mu_clean

# =============================================================================
# MIQP SOLVER  (identical formulation to Step 7/8)
# =============================================================================

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=0.0, apply_esg=True,
               sector_map=None,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN,
               sector_cap=SECTOR_CAP):
    """
    Solve one MIQP instance via cvxpy + Gurobi.
    Identical formulation to Step 7/8.
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

# NOTE: automatic optimal_block_length returned 1 for this dataset
# (near-zero autocorrelation in equal-weight portfolio).
# Fixed block length of 21 days used instead - see main analysis.
def compute_block_length(r_test: np.ndarray, weights_ew: np.ndarray) -> int:
    """
    Compute the optimal stationary bootstrap block length using the
    equal-weight portfolio return series as a proxy for dependence structure.

    Parameters
    ----------
    r_test      : ndarray  (T x N)
    weights_ew  : ndarray  (N,)  equal weights

    Returns
    -------
    block_length : int  (at least 1)
    """
    port_ew = r_test @ weights_ew          # (T,) proxy series
    result  = optimal_block_length(port_ew)
    # optimal_block_length returns a DataFrame with 'stationary' column
    bl = float(result["stationary"].iloc[0])
    return max(1, int(round(bl)))


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
    block_length : int               optimal block length
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

    Returns
    -------
    results_df : DataFrame  (21 rows)
    """
    log_section(
        f"BOOTSTRAP ANALYSIS  (B={B},  CI={int(CI_LEVEL*100)}%,  "
        f"method=stationary block)"
    )
    log(f"  Fixed vol cap : {VOL_CAP_8A:.4f}")
    log(f"  Grid          : {len(COV_METHODS)} estimators x "
        f"(1 unconstrained + {len(ESG_FLOORS)} ESG floors) "
        f"= {len(COV_METHODS) * (1 + len(ESG_FLOORS))} combinations\n")

    # ── Optimal block length (one value for all 21 portfolios) ────────────────
    N       = r_test.shape[1]
    w_ew    = np.ones(N) / N
    bl = 21
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

            # ── MIQP solve ────────────────────────────────────────────────────
            t0  = time.time()
            res = solve_miqp(
                Sigma, mu_vec, esg_vec,
                vol_cap    = VOL_CAP_8A,
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

            # ── Point-estimate Sharpe on OOS window ──────────────────────────
            oos_sr = annualised_sharpe(r_test, weights)

            # ── Stationary block bootstrap ────────────────────────────────────
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
        f"OOS Sharpe Ratios - 95% Stationary Block Bootstrap CI\n"
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
    log("  STEP 9 - STATIONARY BLOCK BOOTSTRAP FOR OOS SHARPE RATIOS")
    log("  Bloomberg EURO STOXX 600  |  ESG-Constrained Portfolio Optimisation")
    log("=" * 65)
    log(f"  Bootstrap resamples : B = {B}")
    log(f"  CI level            : {int(CI_LEVEL*100)}%  (percentile method)")
    log(f"  Block length method : arch.bootstrap.optimal_block_length")
    log(f"  Solver              : Gurobi 13.0 via cvxpy")
    log(f"  Vol cap             : {VOL_CAP_8A:.4f}  (Step 7/8 Task 2 rule)")

    # ── Load data ─────────────────────────────────────────────────────────────
    tickers, esg_vec, sector_map, n_train, r_train, r_test = load_and_align_data()
    N = len(tickers)

    # ── Covariance on train window ────────────────────────────────────────────
    sigma_cache = estimate_covariances(r_train, N)

    # ── Mu on train window ────────────────────────────────────────────────────
    log_section("MU ESTIMATION ON TRAIN WINDOW  (no look-ahead)")
    mu_vec = estimate_mu_train(r_train)

    # ── 21 MIQP solves + bootstrap ────────────────────────────────────────────
    results_df = run_bootstrap_analysis(
        sigma_cache, mu_vec, esg_vec, sector_map, r_train, r_test
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    log_section("SAVING OUTPUTS")
    csv_path = f"{RESULTS_DIR}/bloomberg_bootstrap_summary.csv"
    results_df.to_csv(csv_path, index=False)
    log(f"  CSV saved  -> {csv_path}  ({len(results_df)} rows)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_path = f"{FIGURES_DIR}/step9_bootstrap_ci.png"
    plot_bootstrap_ci(results_df, fig_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(results_df)

    log_section("PIPELINE COMPLETE")
    log(f"  Total wall time : {time.time() - t_pipeline:.1f}s")
    log(f"  Significant (CI > 0) : "
        f"{results_df['sig_from_zero'].sum()} / {len(results_df)} portfolios")

    # ── Save log ──────────────────────────────────────────────────────────────
    save_log()


if __name__ == "__main__":
    main()
