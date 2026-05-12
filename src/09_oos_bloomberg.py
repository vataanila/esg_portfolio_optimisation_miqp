"""
09_oos_bloomberg.py: Bloomberg Out-of-Sample Analysis
======================================================
Bloomberg EURO STOXX 600 - ESG-Constrained Portfolio Optimisation
Independent Research Project

PIPELINE POSITION:
  08_optimize_bloomberg.py → data/results_bloomberg/bloomberg_frontier_*.csv
  09_oos_bloomberg.py (this file) → data/results_bloomberg/bloomberg_oos_fixedvol_summary.csv
                                     data/results_bloomberg/bloomberg_oos_frontier_main.csv
                                     data/results_bloomberg/bloomberg_oos_log.txt
                                     figures/step8a_*.png
                                     figures/step8b_*.png

DESIGN
  Step 9A — Fixed-vol ESG sensitivity
    3 covariance estimators × 6 ESG floors + 3 unconstrained = 21 MIQP solves
    at one fixed vol cap identical to Step 8 Task 2.
    Shows whether covariance choice matters and quantifies OOS ESG cost.

  Step 9B — Main OOS frontier
    Ledoit-Wolf only, unconstrained + ESG ≥ 55.
    Identical vol-cap grid to Step 8.  IS optimiser metrics vs OOS realised metrics.

STRICT STEP 8 CONSISTENCY
  - Same 261-stock universe: bloomberg_meta_analytical.csv, same ticker order.
  - Same mu specification: mu_trailing_winsor (3-year trailing winsorised).
    mu is loaded from metadata exactly as in Step 8, not re-estimated from returns.
  - Same constraints: W_MAX, W_MIN, K_MIN, K_MAX, SECTOR_CAP.
  - Same ESG floors: [40, 45, 50, 55, 60, 65].
  - Same vol-cap grid: np.geomspace(0.10, 0.80, 30).
  - Same fixed vol cap for Step 9A: max-Sharpe in central 60% of feasible LW frontier
    (loaded from bloomberg_selected_vol_cap.txt written by Step 8).

OOS EVALUATION
  Covariance is re-estimated on the TRAIN window only (no look-ahead).
  Realised train/test performance is computed by applying frozen weights to
  the respective return sub-matrices.

SOLVER: Gurobi 13.0 via cvxpy (academic licence)

Run:
    python src/09_oos_bloomberg.py
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
# PARAMETERS  — change only here, never inline
# =============================================================================

CLEAN_DIR   = "data/clean_bloomberg"
RESULTS_DIR = "data/results_bloomberg"
FIGURES_DIR = "figures"

# ── Portfolio constraints: identical to Step 7 ────────────────────────────────
W_MAX      = 0.20
W_MIN      = 0.01
K_MAX      = 100
K_MIN      = 50
SECTOR_CAP = 0.25

# ── Universe consistency ──────────────────────────────────────────────────────
EXPECTED_N = 261

# ── Mu specification: identical to Step 8 ────────────────────────────────────
# Bloomberg main specification:   mu_trailing_winsor = 3-year trailing winsorised mu
# Bloomberg robustness check (1): mu_winsor          = full-sample winsorised mu
# Bloomberg robustness check (2): mu                 = full-sample raw mu
MU_COLUMN            = "mu_trailing_winsor"  # Bloomberg main specification
MU_ROBUSTNESS_COLUMN = "mu_winsor"           # robustness: full-sample winsorised mu
MU_TRAILING_COLUMN   = "mu"                  # robustness: raw full-sample mu
USE_ROBUSTNESS_MU    = False                 # set True to use mu_winsor
USE_TRAILING_MU      = False                 # reserved for future robustness checks

# ── Train/test split ──────────────────────────────────────────────────────────
TRAIN_RATIO  = 0.70
ANNUAL_DAYS  = 252

# ── Covariance estimators (same naming as Step 7) ────────────────────────────
COV_METHODS = ["sample", "ledoit_wolf", "oas"]

# ── ESG floors (0-100 scale, same as Step 7) ─────────────────────────────────
ESG_FLOORS = [40, 45, 50, 55, 60, 65]

# ── Vol-cap grid: identical to Step 7 ────────────────────────────────────────
VOL_CAP_MIN       = 0.10
VOL_CAP_MAX       = 0.80
N_FRONTIER_POINTS = 30

# Step 8A fixed vol cap: loaded from Step 7 optimisation output
with open("data/results_bloomberg/bloomberg_selected_vol_cap.txt") as f:
    VOL_CAP_8A = float(f.read().strip())
print(f"  Vol cap loaded from bloomberg_selected_vol_cap.txt: {VOL_CAP_8A:.4f}")

# Step 8B: use the same full grid — same range, same number of points as Step 7
ESG_FLOOR_8B = 55.0

# ── Solver settings: identical to Step 7 ─────────────────────────────────────
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
    """Print to console and append to the OOS analysis log buffer."""
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
    Load Bloomberg metadata and returns.
    Universe and ticker ORDER must match Step 7 exactly.

    mu is loaded from metadata using the same active_mu_col logic as Step 7:
      USE_TRAILING_MU → mu_trailing_winsor
      USE_ROBUSTNESS_MU → mu (raw)
      else            → mu_winsor

    Returns
    -------
    returns_df  : DataFrame   (days × N)  daily log-returns
    tickers     : list[str]   ordered as in meta.index
    mu_vec      : ndarray     (N,)  annualised expected returns
    esg_vec     : ndarray     (N,)  ESG scores 0-100
    sector_map  : dict[str, list[int]]  or None
    n_train     : int
    n_test      : int
    r_train     : ndarray  (n_train × N)
    r_test      : ndarray  (n_test  × N)
    """
    log_section("LOADING DATA  —  same 261-stock clean universe as Step 7")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta    = pd.read_csv(f"{CLEAN_DIR}/bloomberg_meta_analytical.csv", index_col=0)
    tickers = list(meta.index)
    N       = len(tickers)

    # ── Mu: exact same logic as Step 8 ───────────────────────────────────────
    if USE_ROBUSTNESS_MU:
        active_mu_col = MU_ROBUSTNESS_COLUMN
        mu_spec_label = "robustness — full-sample winsorised mu"
    elif USE_TRAILING_MU:
        active_mu_col = MU_TRAILING_COLUMN
        mu_spec_label = "robustness — raw full-sample mu"
    else:
        active_mu_col = MU_COLUMN
        mu_spec_label = "main specification — 3-year trailing winsorised mu"

    mu_trailing_ref = meta[active_mu_col].values.astype(float)   # reference only
    esg_vec = meta["esg"].values.astype(float)

    log(f"\n  Dataset         : Bloomberg EURO STOXX 600")
    log(f"  Universe        : {N} stocks  (must equal {EXPECTED_N})")
    log(f"  Mu column       : '{active_mu_col}'  ({mu_spec_label})"
        f"  mean={mu_trailing_ref.mean():.4f}  std={mu_trailing_ref.std():.4f}")
    log(f"  mu_trailing_winsor loaded for reference only — "
        f"OOS mu will be re-estimated on train window (no look-ahead)")
    log(f"  ESG column      : 'esg'  (0-100 scale, mapped from Bloomberg BESG 1-10)"
        f"  mean={esg_vec.mean():.2f}  std={esg_vec.std():.2f}")
    log(f"  Constraints     : W_MAX={W_MAX}  W_MIN={W_MIN}  "
        f"K=[{K_MIN},{K_MAX}]  sector_cap={SECTOR_CAP}")
    log(f"  ESG floors      : {ESG_FLOORS}")

    # ── Consistency checks ────────────────────────────────────────────────────
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
        log("  bloomberg_returns transposed → (days × stocks)")

    # Align to meta ticker order — ONLY keep tickers present in both
    common = [t for t in tickers if t in returns_raw.columns]
    returns_df = returns_raw[common].copy()

    # After alignment, verify order is still consistent with meta
    assert list(returns_df.columns) == common, \
        "Column order mismatch after alignment."
    assert common == [t for t in meta.index if t in returns_raw.columns], \
        "Ticker order in returns does not match meta.index order."
    assert list(returns_df.columns) == list(meta.index), \
        "Final aligned returns columns must exactly equal meta.index."

    assert len(common) == EXPECTED_N, (
        f"Universe size after alignment: {len(common)}, expected {EXPECTED_N}.\n"
        "Some tickers in bloomberg_meta_analytical.csv are missing from bloomberg_returns.csv.\n"
        "Ensure both files are generated by the same Step 6 run."
    )
    log(f"  ✓ Universe confirmed: {N} stocks — order consistent with Step 7")
    log(f"  Date range       : {returns_df.index[0].date()} → "
        f"{returns_df.index[-1].date()}")

    # ── Sector map ────────────────────────────────────────────────────────────
    sector_map = None
    if "sector" in meta.columns:
        sector_map = {}
        for sec, grp in meta.groupby("sector"):
            sector_map[sec] = [tickers.index(t) for t in grp.index if t in tickers]
        log(f"  Sectors          : {list(sector_map.keys())}")
    else:
        log("  WARNING: no 'sector' column — sector cap constraint disabled.")

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
        f"({returns_df.index[0].date()} → {returns_df.index[n_train-1].date()})")
    log(f"  Test  days  : {n_test}   "
        f"({returns_df.index[n_train].date()} → {returns_df.index[-1].date()})")

    return (returns_df, tickers, mu_trailing_ref, esg_vec, sector_map,
            n_train, n_test, r_train, r_test, active_mu_col)


# =============================================================================
# COVARIANCE ESTIMATION ON TRAIN WINDOW
# =============================================================================

def estimate_covariances(r_train: np.ndarray, N: int) -> dict:
    """
    Estimate sample, Ledoit-Wolf, and OAS covariance matrices on the
    train window only (no look-ahead into the test period).
    Forces symmetry and adds PSD regularisation identical to Step 7.

    Returns
    -------
    sigma_cache : dict[str, ndarray]  keys = COV_METHODS
    """
    log_section("COVARIANCE ESTIMATION  (train window only — no look-ahead)")
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
# MU ESTIMATION ON TRAIN WINDOW  (OOS-clean — no look-ahead)
# =============================================================================

def estimate_mu_train(r_train: np.ndarray) -> np.ndarray:
    """
    Estimate expected returns from the TRAIN window only (no look-ahead).
    Mirrors the mu_trailing_winsor recipe but applied strictly to r_train:
      1. Annualised daily mean across training days.
      2. Cross-sectional winsorisation at p1/p99.

    This replaces mu_trailing_winsor (which was computed in Step 6 on the
    last 756 days of the FULL dataset, overlapping with the test period).

    Returns
    -------
    mu_clean : ndarray  (N,)  annualised, winsorised expected returns
    """
    log_section("MU ESTIMATION ON TRAIN WINDOW  (no look-ahead)")

    raw_mu  = r_train.mean(axis=0) * ANNUAL_DAYS   # annualised daily mean

    p1  = np.percentile(raw_mu, 1)
    p99 = np.percentile(raw_mu, 99)
    mu_clean = np.clip(raw_mu, p1, p99)

    log(f"  raw_mu   : mean={raw_mu.mean():.4f}  std={raw_mu.std():.4f}"
        f"  range=[{raw_mu.min():.4f}, {raw_mu.max():.4f}]")
    log(f"  mu_clean : mean={mu_clean.mean():.4f}  std={mu_clean.std():.4f}"
        f"  range=[{mu_clean.min():.4f}, {mu_clean.max():.4f}]"
        f"  (clipped at p1={p1:.4f}, p99={p99:.4f})")
    log(f"  mu re-estimated on train window only — theoretically clean OOS")

    return mu_clean


# =============================================================================
# MIQP SOLVER  (identical formulation to Step 7)
# =============================================================================

def solve_miqp(Sigma, mu, esg, vol_cap,
               esg_floor=0.0, apply_esg=True,
               sector_map=None,
               w_max=W_MAX, w_min=W_MIN,
               k_max=K_MAX, k_min=K_MIN,
               sector_cap=SECTOR_CAP):
    """
    Solve one MIQP instance via cvxpy + Gurobi.
    Identical formulation to Step 7.

    apply_esg=False → true unconstrained (no ESG floor at all).

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
    Annualised realised return, volatility, Sharpe on a (T × N) return matrix.
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
# STEP 8A — FIXED-VOL ESG SENSITIVITY
# =============================================================================

def run_step8a(sigma_cache, mu_vec, esg_vec, sector_map,
               r_train, r_test) -> pd.DataFrame:
    """
    3 estimators × (unconstrained + 6 ESG floors) at the fixed vol cap
    defined by the same vol cap as Step 8 Task 2 (max-Sharpe in central 60% of feasible LW frontier).

    Stores both optimiser-implied metrics (opt_*) and realised IS/OOS metrics.
    """
    log_section(
        f"STEP 8A — FIXED-VOL ESG SENSITIVITY  "
        f"(vol_cap = {VOL_CAP_8A:.4f},  max-Sharpe in central 60% of feasible LW frontier)"
    )
    log(f"  Vol-cap grid    : geomspace({VOL_CAP_MIN}, {VOL_CAP_MAX}, "
        f"{N_FRONTIER_POINTS})")
    log(f"  Fixed vol cap   : {VOL_CAP_8A:.4f}  (max-Sharpe in central 60% of feasible LW frontier)")
    log(f"  Grid            : {len(COV_METHODS)} estimators × "
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
                vol_cap=VOL_CAP_8A,
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
                esg_tag = f"{esg_fl:.0f}" if apply_esg else "—"
                log(f"  {cov:12s}  {case_label:14s}  {esg_tag:>6s}  INFEASIBLE")
            else:
                w = res["weights"]
                tr_r, tr_v, tr_s = compute_realised(w, r_train)
                te_r, te_v, te_s = compute_realised(w, r_test)
                gap = res["sharpe"] - te_s   # IS optimiser Sharpe − OOS realised Sharpe
                esg_tag = f"{esg_fl:.0f}" if apply_esg else "—"

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
    log(f"\n  Step 8A done in {time.time() - t0:.1f}s  —  "
        f"{n_feasible}/{len(rows)} feasible")
    return pd.DataFrame(rows)


# =============================================================================
# STEP 8B — MAIN OOS FRONTIER
# =============================================================================

def run_step8b(sigma_lw, mu_vec, esg_vec, sector_map,
               r_train, r_test) -> pd.DataFrame:
    """
    Full vol-cap sweep (exact Step 7 grid) for the main specification:
    Ledoit-Wolf, unconstrained + ESG ≥ 55.

    Stores both optimiser-implied metrics (opt_*) and realised IS/OOS metrics.
    """
    log_section(
        f"STEP 8B — MAIN OOS FRONTIER  "
        f"(Ledoit-Wolf  |  unconstrained + ESG≥{int(ESG_FLOOR_8B)})"
    )
    vol_caps = np.geomspace(VOL_CAP_MIN, VOL_CAP_MAX, N_FRONTIER_POINTS)
    log(f"  Vol-cap grid : geomspace({VOL_CAP_MIN}, {VOL_CAP_MAX}, "
        f"{N_FRONTIER_POINTS})  — identical to Step 7")
    log(f"  First 5      : {np.round(vol_caps[:5], 3).tolist()}")

    cases = [
        ("unconstrained", False, 0.0),
        (f"esg{int(ESG_FLOOR_8B)}", True, ESG_FLOOR_8B),
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
            gap = res["sharpe"] - te_s   # IS optimiser Sharpe − OOS realised Sharpe

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

    log(f"\n  Step 8B done in {time.time() - t0:.1f}s  —  {len(rows)} feasible points")
    return pd.DataFrame(rows)


# =============================================================================
# SAVE TABLES
# =============================================================================

def save_summary_tables(df8a: pd.DataFrame, df8b: pd.DataFrame) -> None:
    p8a = f"{RESULTS_DIR}/bloomberg_oos_fixedvol_summary.csv"
    p8b = f"{RESULTS_DIR}/bloomberg_oos_frontier_main.csv"
    df8a.to_csv(p8a, index=False)
    df8b.to_csv(p8b, index=False)
    log(f"  Saved: {p8a}")
    log(f"  Saved: {p8b}")


# =============================================================================
# FIGURES — STEP 8A
# =============================================================================

def plot_fixedvol_heatmaps(df8a: pd.DataFrame) -> None:
    """Three heatmaps for ESG cases only: IS Sharpe, OOS Sharpe, Sharpe gap."""
    df_esg = df8a[df8a["case"] == "esg"].copy()
    if df_esg.empty:
        log("  plot_fixedvol_heatmaps: no ESG rows — skipped.")
        return

    metrics = [
        ("opt_sharpe",    "In-Sample Sharpe (optimiser)",   "Blues"),
        ("test_sharpe",   "Out-of-Sample Sharpe",           "Oranges"),
        ("sharpe_gap",    "Sharpe Gap  (IS opt − OOS real)", "RdYlGn_r"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Step 8A — Fixed-Vol ESG Sensitivity  "
        f"(vol cap = {VOL_CAP_8A:.3f},  max-Sharpe in central 60% of feasible LW frontier)\n"
        "Bloomberg EURO STOXX 600  |  70/30 Train/Test",
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
    _savefig(fig, "step8a_heatmaps.png")


def plot_fixedvol_bars(df8a: pd.DataFrame) -> None:
    """IS vs OOS Sharpe bar chart, one panel per covariance estimator."""
    df_esg = df8a[(df8a["case"] == "esg") & df8a["opt_sharpe"].notna()].copy()
    if df_esg.empty:
        log("  plot_fixedvol_bars: no feasible ESG rows — skipped.")
        return

    fig, axes = plt.subplots(1, len(COV_METHODS), figsize=(18, 5), sharey=True)
    fig.suptitle(
        "Step 8A — IS vs OOS Sharpe by ESG Floor\n"
        "Bloomberg EURO STOXX 600  |  70/30 Train/Test",
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
    _savefig(fig, "step8a_bars.png")


# =============================================================================
# FIGURES — STEP 8B
# =============================================================================

def plot_main_oos_frontiers(df8b: pd.DataFrame) -> None:
    """Five Step 8B figures."""
    unc = df8b[df8b["case"] == "unconstrained"].copy()
    esg = df8b[df8b["case"] == f"esg{int(ESG_FLOOR_8B)}"].copy()

    C_IS  = "#4C9BE8"
    C_OOS = "#E87C4C"

    # ── Figure 1: IS optimiser frontier vs OOS realised (vol-ret space) ──────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Step 8B — IS Frontier vs Out-of-Sample Realised Performance\n"
        "Ledoit-Wolf  |  Bloomberg EURO STOXX 600  |  70/30 Train/Test",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG ≥ {int(ESG_FLOOR_8B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        df_s = df_fr.sort_values("opt_vol")
        # IS optimiser frontier
        ax.plot(df_s["opt_vol"], df_s["opt_return"],
                "o-", color=C_IS, lw=2, ms=5, label="IS frontier (optimiser)")
        # OOS realised scatter
        ax.scatter(df_s["test_vol"], df_s["test_return"],
                   color=C_OOS, s=55, zorder=5, marker="s", label="OOS realised")
        # Arrows IS → OOS
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
    _savefig(fig, "step8b_frontier.png")

    # ── Figure 2: IS vs OOS Sharpe along the frontier ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Step 8B — IS vs OOS Sharpe Along the Frontier\n"
        "Ledoit-Wolf  |  Bloomberg EURO STOXX 600",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG ≥ {int(ESG_FLOOR_8B)}"),
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
                        alpha=0.12, color=C_OOS, label="IS − OOS gap")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set(xlabel="Vol Cap", ylabel="Sharpe Ratio", title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step8b_sharpe.png")

    # ── Figure 3: Sharpe degradation bars ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Step 8B — Sharpe Degradation (IS − OOS) Along the Frontier\n"
        "Positive = in-sample optimism  |  Bloomberg EURO STOXX 600",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG ≥ {int(ESG_FLOOR_8B)}"),
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
        ax.set(xlabel="Vol Cap", ylabel="IS Sharpe − OOS Sharpe", title=title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step8b_degradation.png")

    # ── Figure 4: IS vs OOS return scatter (45° line) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Step 8B — IS vs OOS Return\n"
        "Points above 45° line: OOS return exceeds IS estimate",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG ≥ {int(ESG_FLOOR_8B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        ax.scatter(df_fr["opt_return"], df_fr["test_return"],
                   color=C_OOS, s=60, zorder=5, alpha=0.8)
        lo = min(df_fr["opt_return"].min(), df_fr["test_return"].min()) - 0.02
        hi = max(df_fr["opt_return"].max(), df_fr["test_return"].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", label="45° line")
        ax.set(xlabel="IS Return (optimiser)", ylabel="OOS Realised Return",
               title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step8b_return_scatter.png")

    # ── Figure 5: IS vs OOS volatility scatter (45° line) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Step 8B — IS vs OOS Volatility\n"
        "Points above 45° line: OOS vol exceeds IS estimate",
        fontsize=12,
    )
    for ax, df_fr, title in [
        (axes[0], unc, "Unconstrained"),
        (axes[1], esg, f"ESG ≥ {int(ESG_FLOOR_8B)}"),
    ]:
        if df_fr.empty:
            ax.set_title(f"{title}\n(no feasible points)")
            continue
        ax.scatter(df_fr["opt_vol"], df_fr["test_vol"],
                   color=C_IS, s=60, zorder=5, alpha=0.8)
        lo = min(df_fr["opt_vol"].min(), df_fr["test_vol"].min()) - 0.005
        hi = max(df_fr["opt_vol"].max(), df_fr["test_vol"].max()) + 0.005
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", label="45° line")
        ax.set(xlabel="IS Volatility (optimiser)", ylabel="OOS Realised Volatility",
               title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "step8b_vol_scatter.png")


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

    log_section("STEP 8 — OUT-OF-SAMPLE ANALYSIS  (Bloomberg EURO STOXX 600)")
    log(f"  Universe  : exactly {EXPECTED_N} stocks  (same clean universe as Step 7)")
    log(f"  Split     : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}  train/test")
    log(f"  Mu        : '{MU_TRAILING_COLUMN if USE_TRAILING_MU else (MU_ROBUSTNESS_COLUMN if USE_ROBUSTNESS_MU else MU_COLUMN)}'  "
        f"(loaded from metadata, same active mu logic as Step 7)")
    log(f"  Sigma     : re-estimated on train window only  (3 estimators)")
    log(f"  Vol grid  : geomspace({VOL_CAP_MIN}, {VOL_CAP_MAX}, {N_FRONTIER_POINTS})  "
        f"— identical to Step 7")
    log(f"  9A vol cap: {VOL_CAP_8A:.4f}  "
        f"(max-Sharpe in central 60% of feasible LW frontier — identical to Step 8 Task 2)")
    log(f"  8B grid   : same full Step 7 vol-cap grid (no shrinkage to [0.10, 0.25])")

    # ── Data and split ────────────────────────────────────────────────────────
    (returns_df, tickers, mu_trailing_ref, esg_vec, sector_map,
     n_train, n_test, r_train, r_test,
     _active_mu_col) = load_and_align_data()

    # ── Covariance: train window only ─────────────────────────────────────────
    sigma_cache = estimate_covariances(r_train, len(tickers))

    # ── Mu: re-estimated on train window only (OOS-clean) ────────────────────
    # mu_trailing_ref (from metadata) overlaps with the test period — not used
    # for optimisation. mu_vec below is strictly train-window — no look-ahead.
    mu_vec = estimate_mu_train(r_train)

    # ── Step 8A ───────────────────────────────────────────────────────────────
    df8a = run_step8a(
        sigma_cache, mu_vec, esg_vec, sector_map, r_train, r_test
    )

    # ── Step 8B ───────────────────────────────────────────────────────────────
    df8b = run_step8b(
        sigma_cache["ledoit_wolf"], mu_vec, esg_vec, sector_map, r_train, r_test
    )

    # ── Save tables ───────────────────────────────────────────────────────────
    log_section("SAVING OUTPUT TABLES")
    save_summary_tables(df8a, df8b)

    # ── Figures ───────────────────────────────────────────────────────────────
    log_section("GENERATING FIGURES")
    plot_fixedvol_heatmaps(df8a)
    plot_fixedvol_bars(df8a)
    plot_main_oos_frontiers(df8b)

    # ── Final summary ─────────────────────────────────────────────────────────
    log_section("STEP 8 COMPLETE — SUMMARY")
    log(f"  Dataset          : Bloomberg EURO STOXX 600  ({EXPECTED_N} stocks)")
    log(f"  mu_trailing_winsor (Step 7 reference): mean = {mu_trailing_ref.mean():.4f}")
    log(f"  mu_train_winsor    (Step 8 OOS-clean):  mean = {mu_vec.mean():.4f}")
    log(f"  Train window     : {n_train} days")
    log(f"  Test  window     : {n_test} days")

    n8a_feasible = sum(pd.notna(r) for r in df8a["n_holdings"])
    log(f"  8A combinations  : {n8a_feasible} / {len(df8a)} feasible")
    log(f"  8B frontier pts  : {len(df8b)} total "
        f"(unconstrained + ESG≥{int(ESG_FLOOR_8B)})")

    df8a_esg = df8a[df8a["case"] == "esg"]
    if not df8a_esg.empty and df8a_esg["test_sharpe"].notna().any():
        best8a = df8a_esg.loc[df8a_esg["test_sharpe"].idxmax()]
        log(f"  Best OOS Sharpe 8A : {best8a['test_sharpe']:.4f}  "
            f"({best8a['cov_method']}, ESG≥{best8a['esg_floor']:.0f})")

    if len(df8b) > 0:
        best8b = df8b.loc[df8b["test_sharpe"].idxmax()]
        log(f"  Best OOS Sharpe 8B : {best8b['test_sharpe']:.4f}  "
            f"(case={best8b['case']}, vc={best8b['vol_cap']:.3f})")

    log(f"\n  Total runtime    : {time.time() - t_start:.1f}s")
    log(f"  Tables  → {RESULTS_DIR}/")
    log(f"  Figures → {FIGURES_DIR}/")
    log(f"\nReady for Step 9 — Final comparison plots (Simulated vs Bloomberg).")

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = f"{RESULTS_DIR}/bloomberg_oos_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log_lines))
    print(f"\n  Log saved: {log_path}")


if __name__ == "__main__":
    main()
