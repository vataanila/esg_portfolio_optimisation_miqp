"""
04_oos_simulated.py: Out-of-Sample Analysis (Simulated Dataset)
=================================================================
ESG-Constrained Portfolio Optimisation - Independent Research Project

PIPELINE POSITION:
  03_optimize_simulated.py → data/results/frontier_*.csv
                              data/results/selected_vol_cap.txt
  04_oos_simulated.py (this file) → data/results_simulated/simulated_oos_fixedvol_summary.csv
                                     data/results_simulated/simulated_oos_frontier_main.csv
                                     data/results_simulated/simulated_oos_log.txt
                                     figures/step4a_heatmaps.png
                                     figures/step4a_bars.png
                                     figures/step4b_frontier.png
                                     figures/step4b_sharpe.png
                                     figures/step4b_degradation.png
                                     figures/step4b_return_scatter.png
                                     figures/step4b_vol_scatter.png

DESIGN
  Step 4A -- Fixed-vol ESG sensitivity
    3 covariance estimators x 6 ESG floors + 3 unconstrained = 21 MIQP solves.
    Fixed vol cap = data-driven vol cap from 03_optimize_simulated.py
    (max-Sharpe point in central 60% of feasible unconstrained Ledoit-Wolf frontier).

  Step 4B -- Main OOS frontier
    Ledoit-Wolf only, unconstrained + ESG >= 55.
    Identical vol-cap grid to Step 03.  IS optimiser metrics vs OOS realised metrics.

KEY DIFFERENCES vs 09_oos_bloomberg.py

  Difference 1 -- mu specification:
    mu is loaded ONCE from metadata as meta["mu_winsor"].values (winsorised FICO mu).
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

SOLVER: Gurobi 13.0 via cvxpy (academic licence)

Run:
    python src/04_oos_simulated.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf, OAS

warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETERS  -- change only here, never inline
# =============================================================================

CLEAN_DIR   = "data/clean"
RESULTS_DIR = "data/results_simulated"
FIGURES_DIR = "figures"

# -- Portfolio constraints: identical to Step 3 --------------------------------
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

# -- Covariance estimators (same naming as Step 3) -----------------------------
COV_METHODS = ["sample", "ledoit_wolf", "oas"]

# -- ESG floors (0-100 scale, same as Step 3) ----------------------------------
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# -- Vol-cap grid: identical to Step 3 -----------------------------------------
VOL_CAP_MIN       = 0.30
VOL_CAP_MAX       = 3.00
N_FRONTIER_POINTS = 30

# Step 4A fixed vol cap: loaded from Step 3 optimisation output
with open("data/results/selected_vol_cap.txt") as f:
    VOL_CAP_4A = float(f.read().strip())
print(f"  Vol cap loaded from selected_vol_cap.txt: {VOL_CAP_4A:.4f}")

# Step 4B: same full grid -- same range, same number of points as Step 3
ESG_FLOOR_4B = 55.0

# -- Solver settings: identical to Step 3 --------------------------------------
SOLVER_VERBOSE    = False
SOLVER_TIME_LIMIT = 120
SOLVER_MIP_GAP    = 1e-4

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

_log_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)

def log_section(title: str) -> None:
    bar = "=" * 65
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

# =============================================================================
# LOAD AND ALIGN DATA
# =============================================================================

def load_and_align_data():
    """
    Load simulated metadata and log-returns.
    Universe size is read from meta at runtime (NOT hardcoded).

    DIFFERENCE 1: mu loaded from meta["mu_winsor"] -- never re-estimated.
    DIFFERENCE 2: files from data/clean/, no EXPECTED_N assertion.

    Returns
    -------
    returns_df  : DataFrame   (days x N)  daily log-returns
    tickers     : list[str]   ordered as in meta.index
    mu_vec      : ndarray     (N,)  mu_winsor from FICO metadata
    esg_vec     : ndarray     (N,)  ESG scores 0-100
    sector_map  : dict[str, list[int]]  or None
    n_train     : int
    n_test      : int
    r_train     : ndarray  (n_train x N)
    r_test      : ndarray  (n_test  x N)
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
    log(f"  mu loaded from metadata (FICO assigned) -- "
        f"NOT re-estimated on train window (simulated data convention)")
    log(f"  mu_winsor       : mean={mu_vec.mean():.4f}  std={mu_vec.std():.4f}"
        f"  range=[{mu_vec.min():.4f}, {mu_vec.max():.4f}]")
    log(f"  ESG column      : 'esg'  (0-100 scale)"
        f"  mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
    log(f"  Constraints     : W_MAX={W_MAX}  W_MIN={W_MIN}  "
        f"K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
    log(f"  ESG floors      : {ESG_FLOORS}")

    # -- Returns ---------------------------------------------------------------
    returns_raw = pd.read_csv(
        f"{CLEAN_DIR}/log_returns.csv", index_col=0, parse_dates=True
    )
    if returns_raw.shape[0] < returns_raw.shape[1]:
        returns_raw = returns_raw.T
        log("  log_returns transposed -> (days x stocks)")

    # Align to meta ticker order
    common     = [t for t in tickers if t in returns_raw.columns]
    returns_df = returns_raw[common].copy()

    assert list(returns_df.columns) == common, \
        "Column order mismatch after alignment."
    assert common == [t for t in meta.index if t in returns_raw.columns], \
        "Ticker order in returns does not match meta.index order."
    assert list(returns_df.columns) == list(meta.index), \
        "Final aligned returns columns must exactly equal meta.index."

    log(f"  Universe confirmed: {len(common)} stocks -- order consistent with Step 3")
    idx0 = returns_df.index[0]
    idx1 = returns_df.index[-1]
    try:
        idx0 = idx0.date()
        idx1 = idx1.date()
    except AttributeError:
        pass   # non-datetime index (e.g. Day_1 ... Day_N)
    log(f"  Index range      : {idx0} -> {idx1}")

    # -- Sector map ------------------------------------------------------------
    sector_map = None
    if "sector" in meta.columns:
        sector_map = {}
        for sec, grp in meta.groupby("sector"):
            sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
        log(f"  Sectors          : {list(sector_map.keys())}")
    else:
        log("  WARNING: no 'sector' column -- sector cap constraint disabled.")

    # -- Train / test split ----------------------------------------------------
    log_section("TRAIN / TEST SPLIT")
    n_days  = len(returns_df)
    n_train = int(n_days * TRAIN_RATIO)
    n_test  = n_days - n_train

    r_train = np.nan_to_num(
        returns_df.iloc[:n_train].values.astype(float), nan=0.0)
    r_test  = np.nan_to_num(
        returns_df.iloc[n_train:].values.astype(float), nan=0.0)

    def _idx(i):
        v = returns_df.index[i]
        try:
            return v.date()
        except AttributeError:
            return v

    log(f"  Total days  : {n_days}")
    log(f"  Train days  : {n_train}  ({_idx(0)} -> {_idx(n_train-1)})")
    log(f"  Test  days  : {n_test}   ({_idx(n_train)} -> {_idx(-1)})")

    return (returns_df, tickers, mu_vec, esg_vec, sector_map,
            n_train, n_test, r_train, r_test)


# =============================================================================
# COVARIANCE ESTIMATION ON TRAIN WINDOW
# =============================================================================

def estimate_covariances(r_train: np.ndarray, N: int) -> dict:
    """
    Estimate sample, Ledoit-Wolf, and OAS covariance matrices on the
    train window only (no look-ahead into the test period).
    Forces symmetry and adds PSD regularisation identical to Step 3.

    Returns
    -------
    sigma_cache : dict[str, ndarray]  keys = COV_METHODS
    """
    log_section("COVARIANCE ESTIMATION  (train window only -- no look-ahead)")
    sigma_cache = {}

    # Sample covariance
    S = np.cov(r_train.T) * ANNUAL_DAYS
    S = (S + S.T) / 2
    S += np.eye(N) * 1e-8
    sigma_cache["sample"] = S
    log(f"  sample      : shape={S.shape}  min_eig={np.linalg.eigvalsh(S).min():.6f}")

    # Ledoit-Wolf shrinkage
    lw   = LedoitWolf().fit(r_train)
    S_lw = lw.covariance_ * ANNUAL_DAYS
    S_lw = (S_lw + S_lw.T) / 2
    S_lw += np.eye(N) * 1e-8
    sigma_cache["ledoit_wolf"] = S_lw
    log(f"  ledoit_wolf : shrinkage={lw.shrinkage_:.4f}  "
        f"min_eig={np.linalg.eigvalsh(S_lw).min():.6f}")

    # Oracle Approximating Shrinkage
    oas   = OAS().fit(r_train)
    S_oas = oas.covariance_ * ANNUAL_DAYS
    S_oas = (S_oas + S_oas.T) / 2
    S_oas += np.eye(N) * 1e-8
    sigma_cache["oas"] = S_oas
    log(f"  oas         : shrinkage={oas.shrinkage_:.4f}  "
        f"min_eig={np.linalg.eigvalsh(S_oas).min():.6f}")

    return sigma_cache


# =============================================================================
# MIQP SOLVER  (identical formulation to Step 3)
# =============================================================================

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=0.0, apply_esg=True,
               sector_map=None,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN,
               sector_cap=SECTOR_CAP):
    """
    Solve one MIQP instance via cvxpy + Gurobi.
    Identical formulation to Step 3.

    apply_esg=False -> true unconstrained (no ESG floor at all).

    Returns
    -------
    dict with keys: status, weights (None if infeasible),
                    ret, vol, sharpe, esg_score, n_holdings
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
# REALISED PERFORMANCE
# =============================================================================

def compute_realised(weights: np.ndarray, r_mat: np.ndarray) -> tuple:
    """
    Annualised realised return, volatility, Sharpe on a (T x N) return matrix.
    Weights are frozen (no rebalancing).

    Returns
    -------
    (ann_ret, ann_vol, sharpe)
    """
    port    = r_mat @ weights
    ann_ret = float(port.mean() * ANNUAL_DAYS)
    ann_vol = float(port.std()  * np.sqrt(ANNUAL_DAYS))
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-8 else np.nan
    return ann_ret, ann_vol, sharpe


# =============================================================================
# STEP 4A -- FIXED-VOL ESG SENSITIVITY
# =============================================================================

def run_step4a(sigma_cache, mu_vec, esg_vec, sector_map,
               r_train, r_test) -> pd.DataFrame:
    """
    3 estimators x (unconstrained + 6 ESG floors) at the fixed vol cap
    defined by the exact same percentile rule as Step 3 Task 2.

    Stores both optimiser-implied metrics (opt_*) and realised IS/OOS metrics.
    """
    log_section(
        f"STEP 4A -- FIXED-VOL ESG SENSITIVITY  "
        f"(vol_cap = {VOL_CAP_4A:.4f},  Step 3 Task 2 percentile rule)"
    )
    log(f"  Vol-cap grid    : geomspace({VOL_CAP_MIN}, {VOL_CAP_MAX}, "
        f"{N_FRONTIER_POINTS})")
    log(f"  Fixed vol cap   : {VOL_CAP_4A:.4f}  (max-Sharpe in central 60% of feasible LW frontier)")
    log(f"  Grid            : {len(COV_METHODS)} estimators x "
        f"(1 unconstrained + {len(ESG_FLOORS)} ESG floors) "
        f"= {len(COV_METHODS) * (1 + len(ESG_FLOORS))} combinations")

    hdr = (f"  {'cov_method':12s}  {'case':14s}  {'esg_fl':6s}  "
           f"{'IS_SR':6s}  {'TR_SR':6s}  {'TE_SR':6s}  "
           f"{'gap':6s}  {'n':3s}  {'esg':5s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))

    rows = []
    t0   = time.time()

    for cov in COV_METHODS:
        Sigma = sigma_cache[cov]

        # Unconstrained case
        cases = [("unconstrained", False, float("nan"))]
        # ESG-constrained cases
        cases += [("esg", True, float(f)) for f in ESG_FLOORS]

        for case_label, apply_esg, esg_fl in cases:
            esg_floor_val = 0.0 if not apply_esg else esg_fl
            res = solve_miqp(
                Sigma, mu_vec, esg_vec,
                vol_cap=VOL_CAP_4A,
                esg_floor=esg_floor_val,
                apply_esg=apply_esg,
                sector_map=sector_map,
            )

            row = dict(
                cov_method=cov,
                case=case_label,
                esg_floor=(np.nan if not apply_esg else esg_fl),
                status=res["status"],
                n_holdings=np.nan,
                realised_esg_in_sample=np.nan,
                opt_return=np.nan, opt_vol=np.nan, opt_sharpe=np.nan,
                train_return=np.nan, train_vol=np.nan, train_sharpe=np.nan,
                test_return=np.nan,  test_vol=np.nan,  test_sharpe=np.nan,
                sharpe_gap=np.nan,
            )

            if res["weights"] is None:
                esg_tag = f"{esg_fl:.0f}" if apply_esg else "-"
                log(f"  {cov:12s}  {case_label:14s}  {esg_tag:>6s}  INFEASIBLE")
            else:
                w = res["weights"]
                tr_r, tr_v, tr_s = compute_realised(w, r_train)
                te_r, te_v, te_s = compute_realised(w, r_test)
                gap = res["sharpe"] - te_s   # IS optimiser Sharpe - OOS realised Sharpe
                esg_tag = f"{esg_fl:.0f}" if apply_esg else "-"

                row.update(
                    n_holdings=res["n_holdings"],
                    realised_esg_in_sample=res["esg_score"],
                    opt_return=res["ret"],
                    opt_vol=res["vol"],
                    opt_sharpe=res["sharpe"],
                    train_return=tr_r, train_vol=tr_v, train_sharpe=tr_s,
                    test_return=te_r,  test_vol=te_v,  test_sharpe=te_s,
                    sharpe_gap=gap,
                )
                log(f"  {cov:12s}  {case_label:14s}  {esg_tag:>6s}  "
                    f"{res['sharpe']:6.3f}  {tr_s:6.3f}  {te_s:6.3f}  "
                    f"{gap:6.3f}  {res['n_holdings']:3d}  {res['esg_score']:5.1f}")

            rows.append(row)

    n_feasible = sum(pd.notna(r["n_holdings"]) for r in rows)
    log(f"\n  Step 4A done in {time.time() - t0:.1f}s  --  "
        f"{n_feasible}/{len(rows)} feasible")
    return pd.DataFrame(rows)


# =============================================================================
# STEP 4B -- MAIN OOS FRONTIER
# =============================================================================

def run_step4b(sigma_lw, mu_vec, esg_vec, sector_map,
               r_train, r_test) -> pd.DataFrame:
    """
    Full vol-cap sweep (exact Step 3 grid) for the main specification:
    Ledoit-Wolf, unconstrained + ESG >= 55.

    Stores both optimiser-implied metrics (opt_*) and realised IS/OOS metrics.
    """
    log_section(
        f"STEP 4B -- MAIN OOS FRONTIER  "
        f"(Ledoit-Wolf  |  unconstrained + ESG>={int(ESG_FLOOR_4B)})"
    )
    vol_caps = np.geomspace(VOL_CAP_MIN, VOL_CAP_MAX, N_FRONTIER_POINTS)
    log(f"  Vol-cap grid : geomspace({VOL_CAP_MIN}, {VOL_CAP_MAX}, "
        f"{N_FRONTIER_POINTS})  -- identical to Step 3")
    log(f"  First 5      : {np.round(vol_caps[:5], 3).tolist()}")

    cases = [
        ("unconstrained", False, 0.0),
        (f"esg{int(ESG_FLOOR_4B)}", True, ESG_FLOOR_4B),
    ]

    rows = []
    t0   = time.time()

    for case_label, apply_esg, esg_fl in cases:
        log(f"\n  [{case_label}]")
        hdr = (f"  {'vol_cap':8s}  "
               f"{'IS_ret':7s}  {'IS_vol':7s}  {'IS_SR':6s}  "
               f"{'TR_ret':7s}  {'TR_vol':7s}  {'TR_SR':6s}  "
               f"{'TE_ret':7s}  {'TE_vol':7s}  {'TE_SR':6s}  "
               f"{'gap':6s}  n")
        log(hdr)
        log("  " + "-" * (len(hdr) - 2))

        for vc in vol_caps:
            res = solve_miqp(
                sigma_lw, mu_vec, esg_vec,
                vol_cap=vc,
                esg_floor=esg_fl,
                apply_esg=apply_esg,
                sector_map=sector_map,
            )

            if res["weights"] is None:
                log(f"  {vc:.3f}     INFEASIBLE ({res['status']})")
                continue

            w = res["weights"]
            tr_r, tr_v, tr_s = compute_realised(w, r_train)
            te_r, te_v, te_s = compute_realised(w, r_test)
            gap = res["sharpe"] - te_s   # IS optimiser Sharpe - OOS realised Sharpe

            log(f"  {vc:.3f}     "
                f"{res['ret']:7.4f}  {res['vol']:7.4f}  {res['sharpe']:6.3f}  "
                f"{tr_r:7.4f}  {tr_v:7.4f}  {tr_s:6.3f}  "
                f"{te_r:7.4f}  {te_v:7.4f}  {te_s:6.3f}  "
                f"{gap:6.3f}  {res['n_holdings']}")

            rows.append(dict(
                case=case_label,
                vol_cap=vc,
                status=res["status"],
                n_holdings=res["n_holdings"],
                realised_esg_in_sample=res["esg_score"],
                opt_return=res["ret"],
                opt_vol=res["vol"],
                opt_sharpe=res["sharpe"],
                train_return=tr_r, train_vol=tr_v, train_sharpe=tr_s,
                test_return=te_r,  test_vol=te_v,  test_sharpe=te_s,
                sharpe_gap=gap,
            ))

    log(f"\n  Step 4B done in {time.time() - t0:.1f}s  --  {len(rows)} feasible points")
    return pd.DataFrame(rows)


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_summary_tables(df4a: pd.DataFrame, df4b: pd.DataFrame) -> None:
    p4a = f"{RESULTS_DIR}/simulated_oos_fixedvol_summary.csv"
    p4b = f"{RESULTS_DIR}/simulated_oos_frontier_main.csv"
    df4a.to_csv(p4a, index=False)
    df4b.to_csv(p4b, index=False)
    log(f"  Saved: {p4a}")
    log(f"  Saved: {p4b}")


# =============================================================================
# FIGURES -- STEP 4A
# =============================================================================

def plot_fixedvol_heatmaps(df4a: pd.DataFrame) -> None:
    """Three heatmaps for ESG cases only: IS Sharpe, OOS Sharpe, Sharpe gap."""
    df_esg = df4a[df4a["case"] == "esg"].copy()
    if df_esg.empty:
        log("  plot_fixedvol_heatmaps: no ESG rows -- skipped.")
        return

    metrics = [
        ("opt_sharpe",    "In-Sample Sharpe (optimiser)",    "Blues"),
        ("test_sharpe",   "Out-of-Sample Sharpe",            "Oranges"),
        ("sharpe_gap",    "Sharpe Gap  (IS opt - OOS real)", "RdYlGn_r"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Step 4A -- Fixed-Vol ESG Sensitivity  "
        f"(vol cap = {VOL_CAP_4A:.3f},  data-driven from unconstrained LW frontier)\n"
        "Simulated Dataset  |  70/30 Train/Test",
        fontsize=12,
    )

    for ax, (metric, title, cmap) in zip(axes, metrics):
        pivot = (
            df_esg.pivot(index="cov_method", columns="esg_floor", values=metric)
                  .reindex(COV_METHODS)
        )
        sns.heatmap(
            pivot, ax=ax, cmap=cmap, annot=True, fmt=".3f",
            linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ESG Floor", fontsize=10)
        ax.set_ylabel("Covariance Estimator", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    _savefig(fig, "step4a_heatmaps.png")


def plot_fixedvol_bars(df4a: pd.DataFrame) -> None:
    """IS vs OOS Sharpe bar chart, one panel per covariance estimator."""
    df_esg = df4a[(df4a["case"] == "esg") & df4a["opt_sharpe"].notna()].copy()
    if df_esg.empty:
        log("  plot_fixedvol_bars: no feasible ESG rows -- skipped.")
        return

    fig, axes = plt.subplots(1, len(COV_METHODS), figsize=(18, 5), sharey=True)
    fig.suptitle(
        "Step 4A -- IS vs OOS Sharpe by ESG Floor\n"
        "Simulated Dataset  |  70/30 Train/Test",
        fontsize=12,
    )

    x     = np.arange(len(ESG_FLOORS))
    width = 0.30
    C_IS  = "#4C9BE8"
    C_TR  = "#5DBE8A"
    C_OOS = "#E87C4C"

    for ax, cov in zip(axes, COV_METHODS):
        sub = df_esg[df_esg["cov_method"] == cov].set_index("esg_floor")
        is_vals  = [sub.loc[f, "opt_sharpe"]   if f in sub.index else np.nan for f in ESG_FLOORS]
        tr_vals  = [sub.loc[f, "train_sharpe"] if f in sub.index else np.nan for f in ESG_FLOORS]
        oos_vals = [sub.loc[f, "test_sharpe"]  if f in sub.index else np.nan for f in ESG_FLOORS]

        ax.bar(x - width,     is_vals,  width, label="IS (opt)",    color=C_IS,  alpha=0.85)
        ax.bar(x,             tr_vals,  width, label="Train real.",  color=C_TR,  alpha=0.85)
        ax.bar(x + width,     oos_vals, width, label="OOS real.",    color=C_OOS, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(cov.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ESG_FLOORS, fontsize=9)
        ax.set_xlabel("ESG Floor", fontsize=9)
        ax.set_ylabel("Sharpe Ratio", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, "step4a_bars.png")


# =============================================================================
# FIGURES -- STEP 4B
# =============================================================================

def plot_main_oos_frontiers(df4b: pd.DataFrame) -> None:
    """Five Step 4B figures."""
    unc = df4b[df4b["case"] == "unconstrained"].copy()
    esg = df4b[df4b["case"] == f"esg{int(ESG_FLOOR_4B)}"].copy()

    C_IS  = "#4C9BE8"
    C_OOS = "#E87C4C"

    # -- Figure 1: IS optimiser frontier vs OOS realised (vol-ret space) -------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Step 4B -- IS Frontier vs Out-of-Sample Realised Performance\n"
        f"Ledoit-Wolf  |  Simulated Dataset  |  70/30 Train/Test",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG >= {int(ESG_FLOOR_4B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        df_s = df_fr.sort_values("opt_vol")
        ax.plot(df_s["opt_vol"], df_s["opt_return"],
                "o-", color=C_IS, lw=2, ms=5, label="IS frontier (optimiser)")
        ax.scatter(df_s["test_vol"], df_s["test_return"],
                   color=C_OOS, s=55, zorder=5, marker="s", label="OOS realised")
        for _, r in df_s.iterrows():
            ax.annotate(
                "", xy=(r["test_vol"], r["test_return"]),
                xytext=(r["opt_vol"], r["opt_return"]),
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.6, alpha=0.4),
            )
        ax.set(xlabel="Annualised Volatility",
               ylabel="Annualised Return", title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step4b_frontier.png")

    # -- Figure 2: IS vs OOS Sharpe along the frontier -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Step 4B -- IS vs OOS Sharpe Along the Frontier\n"
        "Ledoit-Wolf  |  Simulated Dataset",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG >= {int(ESG_FLOOR_4B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        df_s = df_fr.sort_values("vol_cap")
        ax.plot(df_s["vol_cap"], df_s["opt_sharpe"],
                "o-", color=C_IS, lw=2, ms=5, label="IS Sharpe (optimiser)")
        ax.plot(df_s["vol_cap"], df_s["test_sharpe"],
                "s--", color=C_OOS, lw=2, ms=5, label="OOS Sharpe")
        ax.fill_between(df_s["vol_cap"],
                        df_s["test_sharpe"], df_s["opt_sharpe"],
                        alpha=0.12, color=C_OOS, label="IS - OOS gap")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set(xlabel="Vol Cap", ylabel="Sharpe Ratio", title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step4b_sharpe.png")

    # -- Figure 3: Sharpe degradation bars -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Step 4B -- Sharpe Degradation (IS - OOS) Along the Frontier\n"
        "Positive = in-sample optimism  |  Simulated Dataset",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG >= {int(ESG_FLOOR_4B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        df_s    = df_fr.sort_values("vol_cap").reset_index(drop=True)
        x       = np.arange(len(df_s))
        colours = ["#E87C4C" if v >= 0 else "#5DBE8A" for v in df_s["sharpe_gap"]]
        ax.bar(x, df_s["sharpe_gap"], color=colours, edgecolor="white", width=0.7)
        ax.axhline(0, color="black", lw=1.0)
        mean_gap = df_s["sharpe_gap"].mean()
        ax.axhline(mean_gap, color="navy", lw=1.2, ls="--",
                   label=f"Mean = {mean_gap:.3f}")
        ax.set_xticks(x[::4])
        ax.set_xticklabels(
            [f"{v:.2f}" for v in df_s["vol_cap"].iloc[::4]],
            fontsize=8, rotation=30,
        )
        ax.set(xlabel="Vol Cap", ylabel="IS Sharpe - OOS Sharpe", title=title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step4b_degradation.png")

    # -- Figure 4: IS vs OOS return scatter ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Step 4B -- IS vs OOS Return\n"
        "Points above 45 degree line: OOS return exceeds IS estimate",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG >= {int(ESG_FLOOR_4B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        ax.scatter(df_fr["opt_return"], df_fr["test_return"],
                   color=C_OOS, s=60, zorder=5, alpha=0.8)
        lo = min(df_fr["opt_return"].min(), df_fr["test_return"].min()) - 0.02
        hi = max(df_fr["opt_return"].max(), df_fr["test_return"].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", label="45 degree line")
        ax.set(xlabel="IS Return (optimiser)", ylabel="OOS Realised Return",
               title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step4b_return_scatter.png")

    # -- Figure 5: IS vs OOS volatility scatter --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Step 4B -- IS vs OOS Volatility\n"
        "Points above 45 degree line: OOS vol exceeds IS estimate",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG >= {int(ESG_FLOOR_4B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        ax.scatter(df_fr["opt_vol"], df_fr["test_vol"],
                   color=C_IS, s=60, zorder=5, alpha=0.8)
        lo = min(df_fr["opt_vol"].min(), df_fr["test_vol"].min()) - 0.005
        hi = max(df_fr["opt_vol"].max(), df_fr["test_vol"].max()) + 0.005
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", label="45 degree line")
        ax.set(xlabel="IS Volatility (optimiser)", ylabel="OOS Realised Volatility",
               title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step4b_vol_scatter.png")


def _savefig(fig, filename: str) -> None:
    path = f"{FIGURES_DIR}/{filename}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t_start = time.time()

    log_section("STEP 4A -- OUT-OF-SAMPLE ANALYSIS  (Simulated Dataset)")
    log(f"  Dataset   : Simulated (FICO metadata, universe size read at runtime)")
    log(f"  Split     : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}  train/test")
    log(f"  Mu        : '{MU_COLUMN}'  loaded from metadata (FICO assigned values)")
    log(f"              NOT re-estimated on train window -- simulated data convention")
    log(f"  Sigma     : re-estimated on train window only  (3 estimators)")
    log(f"  Vol grid  : geomspace({VOL_CAP_MIN}, {VOL_CAP_MAX}, {N_FRONTIER_POINTS})  "
        f"-- identical to Step 3")
    log(f"  4A vol cap: {VOL_CAP_4A:.4f}  (max-Sharpe in central 60% of feasible LW frontier)")
    log(f"  4B grid   : same full Step 3 vol-cap grid")

    # -- Data and split --------------------------------------------------------
    (returns_df, tickers, mu_vec, esg_vec, sector_map,
     n_train, n_test, r_train, r_test) = load_and_align_data()

    N = len(tickers)

    # -- Covariance: train window only -----------------------------------------
    sigma_cache = estimate_covariances(r_train, N)

    # -- Mu: from FICO metadata -- identical vector used for all solves --------
    log_section("MU SPECIFICATION  (FICO metadata -- no train-window re-estimation)")
    log(f"  mu_winsor (FICO metadata): mean = {mu_vec.mean():.4f}  "
        f"std = {mu_vec.std():.4f}  range = [{mu_vec.min():.4f}, {mu_vec.max():.4f}]")
    log(f"  Sigma: re-estimated on train window only")
    log(f"  Methodological asymmetry: mu is fixed (FICO), Sigma is OOS-clean (train only)")

    # -- Step 4A ---------------------------------------------------------------
    df4a = run_step4a(
        sigma_cache, mu_vec, esg_vec, sector_map, r_train, r_test
    )

    # -- Step 4B ---------------------------------------------------------------
    df4b = run_step4b(
        sigma_cache["ledoit_wolf"], mu_vec, esg_vec, sector_map, r_train, r_test
    )

    # -- Save tables -----------------------------------------------------------
    log_section("SAVING OUTPUT TABLES")
    save_summary_tables(df4a, df4b)

    # -- Figures ---------------------------------------------------------------
    log_section("GENERATING FIGURES")
    plot_fixedvol_heatmaps(df4a)
    plot_fixedvol_bars(df4a)
    plot_main_oos_frontiers(df4b)

    # -- Final summary ---------------------------------------------------------
    log_section("STEP 4A COMPLETE -- SUMMARY")
    log(f"  Dataset          : Simulated  ({N} stocks)")
    log(f"  mu_winsor (FICO metadata): mean = {mu_vec.mean():.4f}")
    log(f"  Sigma: re-estimated on train window only")
    log(f"  Train window     : {n_train} days")
    log(f"  Test  window     : {n_test} days")

    n4a_feasible = sum(pd.notna(r) for r in df4a["n_holdings"])
    log(f"  4A combinations  : {n4a_feasible} / {len(df4a)} feasible")
    log(f"  4B frontier pts  : {len(df4b)} total "
        f"(unconstrained + ESG>={int(ESG_FLOOR_4B)})")

    df4a_esg = df4a[df4a["case"] == "esg"]
    if not df4a_esg.empty and df4a_esg["test_sharpe"].notna().any():
        best4a = df4a_esg.loc[df4a_esg["test_sharpe"].idxmax()]
        log(f"  Best OOS Sharpe 4A : {best4a['test_sharpe']:.4f}  "
            f"({best4a['cov_method']}, ESG>={best4a['esg_floor']:.0f})")

    if len(df4b) > 0:
        best4b = df4b.loc[df4b["test_sharpe"].idxmax()]
        log(f"  Best OOS Sharpe 4B : {best4b['test_sharpe']:.4f}  "
            f"(case={best4b['case']}, vc={best4b['vol_cap']:.3f})")

    log(f"\n  Total runtime    : {time.time() - t_start:.1f}s")
    log(f"  Tables  -> {RESULTS_DIR}/")
    log(f"  Figures -> {FIGURES_DIR}/")
    log(f"\nReady for Step 4b (bootstrap) -- Simulated vs Bloomberg comparison.")

    # -- Save log --------------------------------------------------------------
    log_path = f"{RESULTS_DIR}/simulated_oos_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log_lines))
    print(f"\n  Log saved: {log_path}")


if __name__ == "__main__":
    main()
