"""
02_preprocess_simulated.py: Analytical Preprocessing and Optimisation Inputs
=============================================================================
ESG-Constrained Portfolio Optimisation - Independent Research Project

PIPELINE POSITION:
  01_load_simulated.py (data/clean/universe_clean.csv, data/clean/prices_clean.csv)
  02_preprocess_simulated.py (this file) → data/clean/meta_preprocessed.csv
                                           data/clean/log_returns.csv
                                           data/clean/sigma_*.npy
                                           data/clean/stock_diagnostics.csv
                                           data/clean/risk_summary.csv
                                           data/clean/preprocessing_summary.csv
                                           data/clean/preprocessing_log.txt

WHAT THIS SCRIPT DOES (analytical tasks only):
  A. Load Step 01 cleaned outputs
  B. Sanity check on prices (zeros, negatives, non-finite)
  C. Verify columns are in temporal order (Day_1 ... Day_N)
  D. ESG transformation to 0-100 scale; keep esg_raw, esg_winsor, esg_final
  E. Forward-fill short residual price gaps (<=5 consecutive NaNs only)
  F. Compute log returns
  G. Mild threshold for missing returns (drop if > 1% missing)
  H. Diagnostics: return quantiles, spike counts at 4 thresholds,
                  annualised vol distribution, per-stock table
  I. Empirical mu from log returns + consistency check vs metadata mu
  J. Covariance estimation (sample, Ledoit-Wolf, OAS) — annualised
  K. Mu: simulated main = mu_native (FICO); robustness = mu_empirical_log;
         mu_winsor = winsorised mu_native
  L. Save all optimisation-ready inputs + summaries + preprocessing log

Run:
    python src/02_preprocess_simulated.py

Dependencies:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf, OAS

# ─────────────────────────────────────────────
# PARAMETERS  (change only here, never inline)
# ─────────────────────────────────────────────
INPUT_META    = "data/clean/universe_clean.csv"
INPUT_PRICES  = "data/clean/prices_clean.csv"
OUTPUT_DIR    = "data/clean"
FIGURES_DIR   = "figures"
LOG_PATH      = "data/clean/preprocessing_log.txt"

ANNUAL_DAYS   = 252          # trading days for annualisation

# ESG winsorisation thresholds (simulated data only)
ESG_WIN_LOW   = 1            # lower percentile
ESG_WIN_HIGH  = 99           # upper percentile

# Missing-return threshold: drop stocks exceeding this fraction
MAX_MISSING_RETURN_FRAC = 0.01    # 1%

# Forward-fill limit for short residual price gaps
FFILL_LIMIT   = 5

# Spike thresholds: report counts at all four levels (10%, 20%, 50%, 100%)
SPIKE_THRESHOLDS = [0.10, 0.20, 0.50, 1.00]

# Dataset type: "simulated" or "bloomberg"
# Change to "bloomberg" when running on Bloomberg data (step 6 onward)
DATASET_TYPE  = "simulated"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PREPROCESSING LOG — accumulates messages, saved at the end
# ─────────────────────────────────────────────────────────────
log_lines = []

def log(msg: str) -> None:
    """Print to console and append to the preprocessing log."""
    print(msg)
    log_lines.append(msg)

# ─────────────────────────────────────────────────────────────
# A. LOAD STEP 1 CLEANED OUTPUTS
# ─────────────────────────────────────────────────────────────
log("=" * 65)
log("STEP 2 — ANALYTICAL PREPROCESSING")
log("=" * 65)
log(f"\n[A] Loading Step 1 outputs  (dataset type: {DATASET_TYPE})")

meta   = pd.read_csv(INPUT_META, index_col=0)
meta.index   = meta.index.astype(str).str.strip()
meta.columns = meta.columns.str.strip()

prices = pd.read_csv(INPUT_PRICES, index_col=0)
prices.index   = prices.index.astype(str).str.strip()
prices.columns = prices.columns.str.strip()

# Convert price values to numeric
prices = prices.apply(pd.to_numeric, errors="coerce")

log(f"  meta shape:    {meta.shape}   columns: {meta.columns.tolist()}")
log(f"  prices shape:  {prices.shape}  (stocks x days)")

# Ensure index alignment
common = meta.index.intersection(prices.index)
meta   = meta.loc[common].copy()
prices = prices.loc[common].copy()
log(f"  Stocks in both files after alignment: {len(common)}")

# Preserve the native simulated (FICO) mu immediately after loading,
# before any overwrite.  Role per dataset:
#   Simulated:  mu_native = MAIN spec; mu_empirical_log = robustness spec
#   Bloomberg:  mu_empirical_log = MAIN spec; mu_native not applicable
# Three mu columns will exist in the final output:
#   mu         = main specification     (mu_native for simulated, empirical for Bloomberg)
#   mu_winsor  = winsorised main mu     (winsorisation of the respective main mu, set in [K])
#   mu_native  = original FICO metadata mu (main for simulated, kept for reference on Bloomberg)
meta["mu_native"] = meta["mu"].copy()
log(f"  mu_native (original simulated FICO mu) preserved — "
    f"mean: {meta['mu_native'].mean():.4f}  std: {meta['mu_native'].std():.4f}")

# ─────────────────────────────────────────────────────────────
# B. PRICE SANITY CHECK (zeros, negatives, non-finite)
#    Cheap and avoids silent errors in return computation.
# ─────────────────────────────────────────────────────────────
log("\n[B] Price sanity check (before returns)")

n_zero     = (prices == 0).sum().sum()
n_negative = (prices < 0).sum().sum()
n_inf      = np.isinf(prices.values).sum()
n_nan_pre  = prices.isnull().sum().sum()

log(f"  Zero prices:        {n_zero}")
log(f"  Negative prices:    {n_negative}")
log(f"  Inf prices:         {n_inf}")
log(f"  NaN prices:         {n_nan_pre}")

if n_negative > 0:
    log("  *** WARNING: negative prices detected. "
        "These will produce NaN log returns. Check Step 1 output. ***")
if n_inf > 0:
    log("  *** WARNING: infinite prices detected. "
        "These will propagate into returns. ***")

# Replace zeros and negatives with NaN so log-returns are NaN (not -inf)
prices = prices.where(prices > 0, other=np.nan)

# ─────────────────────────────────────────────────────────────
# C. VERIFY TEMPORAL COLUMN ORDER
#    Sort columns numerically so returns are computed on correctly
#    ordered days (Day_1 < Day_2 < ... < Day_5000).
#    Lexicographic sort would wrongly place Day_10 before Day_2.
# ─────────────────────────────────────────────────────────────
log("\n[C] Verifying temporal column order")

cols = prices.columns.tolist()

def _col_sort_key(c):
    """Return numeric suffix if present, else fallback to raw string."""
    digits = "".join(filter(str.isdigit, str(c)))
    return int(digits) if digits else c

cols_sorted = sorted(cols, key=_col_sort_key)

if cols_sorted != cols:
    prices = prices[cols_sorted]
    log(f"  Columns were NOT in temporal order — reordered to numeric sort.")
    log(f"  First 5 cols (after sort): {cols_sorted[:5]}")
    log(f"  Last  5 cols (after sort): {cols_sorted[-5:]}")
else:
    log(f"  Columns already in correct temporal order. OK")
    log(f"  First 5 cols: {cols[:5]}")
    log(f"  Last  5 cols: {cols[-5:]}")

# ─────────────────────────────────────────────────────────────
# D. ESG TRANSFORMATION TO COMMON 0-100 SCALE
#    Column convention (final):
#      esg_raw    = original value from Step 1 metadata (traceability)
#      esg_winsor = after winsorisation (simulated) or raw × 10 (Bloomberg)
#      esg_final  = 0-100 scale used in optimization
#      esg        = alias for esg_final — the variable used by the optimizer
#    The original "esg" column from Step 1 is replaced by esg_final.
#    esg_raw and esg_winsor are kept only for methodological traceability.
#
#    Option A — common ESG convention across both datasets:
#      Simulated: esg_final on 0-100 (winsorised + linearly rescaled here).
#      Bloomberg: esg_final mapped to 0-100 via (BESG-1)/9*100 in step6.
#      Both optimizers (step3, step7) read 'esg' on the same 0-100 scale.
#      ESG floor grid is common: [40, 45, 50, 55, 60, 65].
# ─────────────────────────────────────────────────────────────
log("\n[D] ESG transformation")

esg_raw = meta["esg"].copy()
meta["esg_raw"] = esg_raw          # save original before overwriting
log(f"  esg_raw — min: {esg_raw.min():.2f}  max: {esg_raw.max():.2f}  "
    f"mean: {esg_raw.mean():.2f}  NaN: {esg_raw.isna().sum()}")

if DATASET_TYPE == "bloomberg":
    # Bloomberg ESG is 0-10 → multiply by 10 to reach 0-100
    meta["esg_winsor"] = esg_raw * 10.0      # no tail winsorisation needed
    meta["esg_final"]  = esg_raw * 10.0
    esg_transform_note = "Bloomberg ESG rescaled: original x 10 → [0, 100]"
    log("  Bloomberg ESG: multiplied by 10  (0-10 → 0-100)")

else:
    # Simulated ESG: winsorise tails first, then linearly rescale to [0, 100]
    lo = np.nanpercentile(esg_raw, ESG_WIN_LOW)
    hi = np.nanpercentile(esg_raw, ESG_WIN_HIGH)
    esg_winsorised = esg_raw.clip(lower=lo, upper=hi)
    meta["esg_winsor"] = esg_winsorised

    esg_min = esg_winsorised.min()
    esg_max = esg_winsorised.max()
    esg_rescaled = (esg_winsorised - esg_min) / (esg_max - esg_min) * 100.0
    meta["esg_final"] = esg_rescaled

    esg_transform_note = (
        f"Simulated ESG winsorised [{ESG_WIN_LOW}th-{ESG_WIN_HIGH}th pct = "
        f"{lo:.2f}-{hi:.2f}] then rescaled to [0, 100]"
    )
    log(f"  Winsorised at [{ESG_WIN_LOW}th, {ESG_WIN_HIGH}th] pct → [{lo:.2f}, {hi:.2f}]")
    log(f"  Then linearly rescaled to [0, 100]")

esg_final = meta["esg_final"].values

# Overwrite the original "esg" column with esg_final so Step 3 can simply use "esg"
# without needing to know the transformation history.
# esg_raw and esg_winsor remain for traceability; the bare "esg" column now equals esg_final.
meta["esg"] = meta["esg_final"]
log(f"  esg_final — min: {esg_final.min():.2f}  max: {esg_final.max():.2f}  "
    f"mean: {esg_final.mean():.2f}")
log(f"  Column 'esg' overwritten with esg_final (0-100 scale). "
    f"esg_raw and esg_winsor kept for traceability.")

# ─────────────────────────────────────────────────────────────
# E. FORWARD-FILL SHORT RESIDUAL PRICE GAPS (<=FFILL_LIMIT)
# ─────────────────────────────────────────────────────────────
log(f"\n[E] Forward-fill short residual price gaps (limit = {FFILL_LIMIT} days)")

n_missing_before = prices.isnull().sum().sum()

# prices is (stocks x days); ffill along the time axis (axis=1)
# Transpose → time as rows → ffill → transpose back
prices_filled = prices.T.ffill(limit=FFILL_LIMIT).T

n_missing_after = prices_filled.isnull().sum().sum()
n_filled        = n_missing_before - n_missing_after

log(f"  Missing before ffill:  {n_missing_before}")
log(f"  Cells filled:          {n_filled}")
log(f"  Missing after ffill:   {n_missing_after}  (longer gaps remain as NaN)")

# ─────────────────────────────────────────────────────────────
# F. COMPUTE LOG RETURNS
#    prices shape: (stocks x days)
#    returns shape: (stocks x days-1)
#
#    Annualisation convention:
#      - returns are daily log returns (dimensionless)
#      - covariance annualised by x ANNUAL_DAYS (= 252)
#      - volatility annualised as sqrt(annualised variance)
#      - mu = main spec: mu_native (simulated) or empirical (Bloomberg); see [I]
#        mu_empirical_log = 252 x mean daily log return (robustness for simulated)
#        mu_native = original FICO simulated mu (main for simulated); see [A]
# ─────────────────────────────────────────────────────────────
log("\n[F] Computing log returns")

log_returns = np.log(prices_filled / prices_filled.shift(axis=1, periods=1))
log_returns = log_returns.iloc[:, 1:]            # drop first col (all NaN)
log_returns = log_returns.dropna(axis=1, how="all")   # drop fully-NaN date columns

log(f"  Log returns shape: {log_returns.shape}  (stocks x days)")

# ─────────────────────────────────────────────────────────────
# G. MILD THRESHOLD FOR MISSING RETURNS (drop if > 1% missing)
# ─────────────────────────────────────────────────────────────
log(f"\n[G] Dropping stocks with > {MAX_MISSING_RETURN_FRAC*100:.0f}% missing returns")

n_days_ret        = log_returns.shape[1]
missing_frac      = log_returns.isnull().sum(axis=1) / n_days_ret
excessive_missing = missing_frac > MAX_MISSING_RETURN_FRAC

n_dropped_returns = excessive_missing.sum()
log(f"  Stocks dropped (excessive missing returns): {n_dropped_returns}")
if n_dropped_returns > 0:
    log(f"  Examples: {log_returns.index[excessive_missing].tolist()[:10]}")

log_returns   = log_returns[~excessive_missing]
meta          = meta.loc[log_returns.index]
prices_filled = prices_filled.loc[log_returns.index]
esg_final     = meta["esg_final"].values

log(f"  Stocks remaining after return filter: {len(log_returns)}")

# ─────────────────────────────────────────────────────────────
# H. DIAGNOSTICS
#    H1 — Return quantile distribution (full return matrix)
#    H2 — Spike counts at 4 thresholds (10%, 20%, 50%, 100%)
#    H3 — Annualised volatility per stock (sample std x sqrt(252))
#    H4 — Per-stock diagnostic table (saved to CSV)
#    H5 — ESG distribution
#    H6 — Mean daily return sanity check
#    H7 — Zero-return concentration (price stickiness / discretization)
#
#    IMPORTANT: No stocks are dropped here.
#    Preprocessing diagnoses and prepares.
#    Exclusion decisions belong to a later explicit robustness step.
# ─────────────────────────────────────────────────────────────
log("\n[H] Diagnostics")

# ── H1: Return quantile distribution ──
log("\n  H1 — Return quantile distribution")
returns_flat = log_returns.values.flatten()
returns_flat = returns_flat[~np.isnan(returns_flat)]

quantile_levels = [0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
q_vals = np.quantile(returns_flat, quantile_levels)
log(f"  Daily log returns (all stocks, all days):")
for ql, qv in zip(quantile_levels, q_vals):
    log(f"    {ql*100:5.1f}th pct:  {qv:+.6f}")

abs_returns_flat = np.abs(returns_flat)
q_abs = np.quantile(abs_returns_flat, quantile_levels)
log(f"  Absolute daily log returns:")
for ql, qv in zip(quantile_levels, q_abs):
    log(f"    {ql*100:5.1f}th pct:  {qv:.6f}")

# ── H2: Spike counts at multiple thresholds ──
log("\n  H2 — Spike counts per threshold")
for thr in SPIKE_THRESHOLDS:
    n_stocks_above = (log_returns.abs() > thr).any(axis=1).sum()
    n_obs_above    = (log_returns.abs() > thr).sum().sum()
    log(f"    |r| > {thr*100:5.1f}%:  {n_stocks_above} stocks affected,  "
        f"{n_obs_above} observations")

# ── H3: Annualised volatility per stock ──
# Source: direct sample std of daily log returns × sqrt(252).
# This is NOT the same as sqrt of the Ledoit-Wolf diagonal,
# which is computed later in [J]. The two will be close but not identical.
# Summary statistics use this sample-based annualised vol
# unless explicitly stated otherwise.
log("\n  H3 — Annualised volatility per stock  "
    "(sample std of daily log returns x sqrt(252))")
daily_vol      = log_returns.std(axis=1)
annual_vol     = daily_vol * np.sqrt(ANNUAL_DAYS)
abs_max_return = log_returns.abs().max(axis=1)

log(f"    Mean:    {annual_vol.mean():.4f}")
log(f"    Median:  {annual_vol.median():.4f}")
log(f"    Min:     {annual_vol.min():.4f}")
log(f"    Max:     {annual_vol.max():.4f}")
log(f"    Stocks with annualised vol > 100%: {(annual_vol > 1.0).sum()}")
log(f"    Stocks with annualised vol > 200%: {(annual_vol > 2.0).sum()}")

# ── H4: Per-stock diagnostic table ──
log("\n  H4 — Building per-stock diagnostic table")
diag_rows = []
for ticker in log_returns.index:
    r = log_returns.loc[ticker].dropna()
    row = {
        "ticker":           ticker,
        "annualized_vol":   r.std() * np.sqrt(ANNUAL_DAYS),
        "max_abs_return":   r.abs().max(),
        "spike_count_10":   (r.abs() > 0.10).sum(),
        "spike_count_20":   (r.abs() > 0.20).sum(),
        "spike_count_50":   (r.abs() > 0.50).sum(),
        "spike_count_100":  (r.abs() > 1.00).sum(),
        "missing_days":     log_returns.loc[ticker].isna().sum(),
    }
    diag_rows.append(row)

stock_diagnostics = pd.DataFrame(diag_rows).set_index("ticker")

diag_path = f"{OUTPUT_DIR}/stock_diagnostics.csv"
stock_diagnostics.to_csv(diag_path)
log(f"    Saved: {diag_path}  shape: {stock_diagnostics.shape}")

# ── H5: ESG distribution ──
log(f"\n  H5 — ESG distribution (esg_final, post-transformation)")
log(f"    Mean: {esg_final.mean():.2f}  Std: {esg_final.std():.2f}")
log(f"    Min:  {esg_final.min():.2f}   Max: {esg_final.max():.2f}")
log(f"    Stocks with ESG >= 70: {(esg_final >= 70).sum()} "
    f"({(esg_final >= 70).mean()*100:.1f}%)")

# ── H6: Mean daily return sanity check ──
mean_daily_return = log_returns.mean().mean()
log(f"\n  H6 — Mean daily log return (should be ~0.0001 to 0.001): "
    f"{mean_daily_return:.6f}")
if abs(mean_daily_return) > 0.01:
    log("  *** WARNING: mean daily return seems large. "
        "Check that prices are in level form, not percentage form. ***")
else:
    log("  OK — returns appear to be in decimal form.")

# ── H7: Zero-return concentration ──
# A high fraction of exact zeros may indicate price discretization,
# rounding in the simulated data, or artificial price stickiness.
log(f"\n  H7 — Zero-return concentration")
n_total_obs    = log_returns.size
n_valid_obs    = log_returns.notna().sum().sum()
n_exact_zero   = (log_returns == 0.0).sum().sum()
frac_zero      = n_exact_zero / n_valid_obs if n_valid_obs > 0 else 0.0

# Count stocks where consecutive prices are identical (price stickiness)
# prices_filled: (stocks x days); count consecutive equal prices per stock
price_diff       = prices_filled.diff(axis=1)          # NaN at first col; 0 where price repeated
n_repeated_price = (price_diff == 0.0).sum().sum()
frac_repeated    = n_repeated_price / prices_filled.notna().sum().sum()

log(f"    Exact zero returns:        {n_exact_zero}  "
    f"({frac_zero*100:.2f}% of valid return observations)")
log(f"    Consecutive equal prices:  {n_repeated_price}  "
    f"({frac_repeated*100:.2f}% of valid price observations)")

if frac_zero > 0.05:
    log(f"    *** NOTE: > 5% of returns are exactly zero. "
        f"Likely reflects price discretization or rounding in simulated data.")
    log(f"        This creates artificial stickiness in the return process "
        f"and contributes to zero-valued quantiles at low percentiles. ***")
else:
    log(f"    Zero concentration is within normal range.")

# ─────────────────────────────────────────────────────────────
# I. EMPIRICAL MU — COMPUTATION AND SPECIFICATION ASSIGNMENT
#    mu_empirical_log = 252 x mean(daily log return)  per stock
#
#    Simulated main specification:  mu_native (original FICO metadata mu)
#      → meta["mu"] is NOT overwritten; it already equals mu_native from [A].
#      → mu_empirical_log is retained as the robustness / symmetry check.
#
#    Bloomberg main specification:  empirical mu
#      → meta["mu"] is overwritten with mu_empirical_log.
#      → mu_native is not meaningful for Bloomberg (no FICO metadata).
#
#    Rationale: using mu_native as the simulated main spec gives fidelity to
#    the original DGP design; empirical mu for Bloomberg gives economic sense.
#    A symmetry robustness run swaps the two within the simulated universe.
# ─────────────────────────────────────────────────────────────
log("\n[I] Empirical mu — computation and specification assignment")

mu_empirical_log = log_returns.mean(axis=1) * ANNUAL_DAYS
meta["mu_empirical_log"] = mu_empirical_log

if DATASET_TYPE == "bloomberg":
    # Bloomberg: empirical mu is the main specification
    meta["mu"] = mu_empirical_log
    log("  Bloomberg dataset: meta['mu'] set to empirical mu (main specification).")
    log("  mu_native is not applicable for Bloomberg data.")
else:
    # Simulated: mu_native (FICO) is the main specification — do NOT overwrite
    # meta["mu"] already holds mu_native, preserved in [A] before any transform.
    log("  Simulated dataset: meta['mu'] retains mu_native (FICO, main specification).")
    log("  meta['mu_empirical_log'] holds empirical mu (robustness specification).")

mu_native_vec = meta["mu_native"].values
mu_emp        = mu_empirical_log.values

corr_mu = np.corrcoef(mu_native_vec, mu_emp)[0, 1]
mae_mu  = np.mean(np.abs(mu_native_vec - mu_emp))
bias_mu = np.mean(mu_native_vec - mu_emp)

log(f"  mu_native (FICO)      — mean: {mu_native_vec.mean():.4f}  std: {mu_native_vec.std():.4f}")
log(f"  mu / mu_empirical_log — mean: {mu_emp.mean():.4f}   std: {mu_emp.std():.4f}")
log(f"  Correlation:          {corr_mu:.4f}")
log(f"  Mean absolute error:  {mae_mu:.4f}")
log(f"  Mean signed error:    {bias_mu:.4f}  (native minus empirical)")

discrepancy = pd.Series(np.abs(mu_native_vec - mu_emp), index=meta.index)
top10_disc  = discrepancy.nlargest(10)
log(f"\n  Top 10 largest |mu_native - mu_empirical| discrepancies:")
for ticker, val in top10_disc.items():
    log(f"    {ticker:15s}  {val:.4f}")

if abs(corr_mu) < 0.3:
    log("\n  *** NOTE: Low correlation between mu_native (FICO) and empirical mu.")
    log("      This reflects a dataset-level divergence, not a coding error.")
    log("")
    log("      Simulated main specification:  mu_native (FICO metadata mu)")
    log("      Simulated robustness check:    mu_empirical_log (empirical mu — symmetry swap)")
    log("      Bloomberg main specification:  mu_empirical_log (empirical mu)")
    log("")
    log("      Implication: using mu_native as the simulated baseline preserves")
    log("      fidelity to the original DGP design. The low corr with empirical mu")
    log("      motivates keeping empirical mu as an explicit robustness check. ***")

# ─────────────────────────────────────────────────────────────
# J. COVARIANCE ESTIMATION (3 methods, annualised)
#    Input for sklearn: shape (days x stocks)
#
#    Annualisation convention:
#      Var(annual return) ≈ 252 x Var(daily return)  [for i.i.d. returns]
#      → multiply daily covariance matrix by 252
#
#    Ledoit-Wolf rationale:
#      Analytical shrinkage coefficient (no cross-validation needed),
#      well-established (Ledoit & Wolf, 2004), computationally efficient,
#      well-suited when N is of similar order to T.
# ─────────────────────────────────────────────────────────────
log("\n[J] Estimating covariance matrices (annualised by x 252)")

# Fill any residual NaN with 0 for covariance estimation (very few at this point)
R_for_cov = log_returns.fillna(0).values.T    # shape: (days x stocks)

# Sample covariance (baseline, no regularisation)
Sigma_sample = np.cov(R_for_cov, rowvar=False) * ANNUAL_DAYS
log(f"  Sample covariance:  shape {Sigma_sample.shape}")

# Ledoit-Wolf analytical shrinkage
lw       = LedoitWolf().fit(R_for_cov)
Sigma_lw = lw.covariance_ * ANNUAL_DAYS
log(f"  Ledoit-Wolf:        shrinkage coefficient = {lw.shrinkage_:.4f}")

# OAS (Oracle Approximating Shrinkage) — alternative for robustness comparison
oas       = OAS().fit(R_for_cov)
Sigma_oas = oas.covariance_ * ANNUAL_DAYS
log(f"  OAS:                shrinkage coefficient = {oas.shrinkage_:.4f}")

# ─────────────────────────────────────────────────────────────
# K. MU: CLEAN NAMING CONVENTION
#    mu               = main expected return
#                         simulated → mu_native (FICO metadata mu)
#                         Bloomberg → mu_empirical_log (empirical mu)
#    mu_winsor        = winsorised main mu (winsorisation of meta["mu"])
#    mu_empirical_log = 252 x mean daily log return (robustness for simulated)
#    mu_native        = original FICO metadata mu   (main for simulated)
#
#    Simulated main:      mu_native  |  Simulated robustness: mu_empirical_log
#    Bloomberg main:      mu_empirical_log  (mu_native not applicable)
#
#    Remove any old/redundant columns that may have come from Step 1.
# ─────────────────────────────────────────────────────────────
log("\n[K] Expected return (mu) — clean naming convention")
if DATASET_TYPE == "bloomberg":
    log("  Main specification:       empirical mu  (mu = 252 x mean daily log return)")
    log("  Robustness specification: mu_native     (not applicable for Bloomberg)")
else:
    log("  Main specification:       mu_native     (original FICO/simulated metadata mu)")
    log("  Robustness specification: mu_empirical_log  (empirical mu — symmetry check)")

for old_col in ["mu_wins", "mu_winsorised"]:
    if old_col in meta.columns:
        meta.drop(columns=[old_col], inplace=True)
        log(f"  Dropped redundant column: {old_col}")

# mu = mu_native (simulated main) or mu_empirical_log (Bloomberg main); see [I]
mu_vec = meta["mu"].values
log(f"  mu (main) — min: {mu_vec.min():.4f}  max: {mu_vec.max():.4f}  "
    f"mean: {mu_vec.mean():.4f}")

# mu_winsor: winsorisation applied to the empirical mu (the main specification)
mu_lo    = np.percentile(mu_vec, 1)
mu_hi    = np.percentile(mu_vec, 99)
mu_winso = np.clip(mu_vec, mu_lo, mu_hi)
meta["mu_winsor"] = mu_winso

log(f"  mu_winsor (winsorised main mu) — min: {mu_winso.min():.4f}  "
    f"max: {mu_winso.max():.4f}  mean: {mu_winso.mean():.4f}")
log(f"  Values changed by winsorisation: "
    f"{((mu_vec < mu_lo) | (mu_vec > mu_hi)).sum()}")
log(f"  mu_native (FICO robustness) — min: {meta['mu_native'].min():.4f}  "
    f"max: {meta['mu_native'].max():.4f}  mean: {meta['mu_native'].mean():.4f}")
log(f"\n  Final metadata columns: {meta.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# L. SAVE ALL OPTIMIZATION-READY OUTPUTS
# ─────────────────────────────────────────────────────────────
log(f"\n[L] Saving outputs to {OUTPUT_DIR}/")

# 1 — Preprocessed metadata (all columns)
meta.to_csv(f"{OUTPUT_DIR}/meta_preprocessed.csv")
log(f"  meta_preprocessed.csv       — {meta.shape[0]} stocks, "
    f"columns: {meta.columns.tolist()}")

# 2 — Log returns
log_returns.to_csv(f"{OUTPUT_DIR}/log_returns.csv")
log(f"  log_returns.csv             — {log_returns.shape}")

# 3 — Covariance matrices (annualised)
np.save(f"{OUTPUT_DIR}/sigma_sample_annual.npy", Sigma_sample)
np.save(f"{OUTPUT_DIR}/sigma_lw_annual.npy",     Sigma_lw)
np.save(f"{OUTPUT_DIR}/sigma_oas_annual.npy",    Sigma_oas)
log(f"  sigma_sample_annual.npy     — {Sigma_sample.shape}")
log(f"  sigma_lw_annual.npy         — {Sigma_lw.shape}")
log(f"  sigma_oas_annual.npy        — {Sigma_oas.shape}")

# 4 — Risk summary: one row per stock
risk_summary = meta[["esg_raw", "esg_final", "mu", "mu_winsor",
                      "mu_empirical_log", "mu_native"]].copy()
risk_summary["annualized_vol"]   = annual_vol
risk_summary["max_abs_return"]   = abs_max_return
risk_summary["spike_count_10"]   = stock_diagnostics["spike_count_10"]
risk_summary["spike_count_20"]   = stock_diagnostics["spike_count_20"]
risk_summary["spike_count_50"]   = stock_diagnostics["spike_count_50"]
risk_summary["spike_count_100"]  = stock_diagnostics["spike_count_100"]
risk_summary.to_csv(f"{OUTPUT_DIR}/risk_summary.csv")
log(f"  risk_summary.csv            — {risk_summary.shape}")

# 5 — One-row preprocessing summary (for reproducibility)
N = len(meta)
preprocessing_summary = pd.DataFrame([{
    "dataset_type":              DATASET_TYPE,
    "n_stocks":                  N,
    "n_trading_days":            log_returns.shape[1],
    "esg_mean":                  esg_final.mean(),
    "esg_std":                   esg_final.std(),
    "mu_mean":                   mu_vec.mean(),
    "mu_std":                    mu_vec.std(),
    "avg_annualized_vol":        annual_vol.mean(),
    "n_stocks_vol_gt_100pct":    (annual_vol > 1.0).sum(),
    "n_stocks_spike_50pct":      (log_returns.abs() > 0.50).any(axis=1).sum(),
    "corr_mu_vs_empirical":      corr_mu,
    "mae_mu_vs_empirical":       mae_mu,
    "zero_return_frac":          frac_zero,
    "repeated_price_frac":       frac_repeated,
    "lw_shrinkage":              lw.shrinkage_,
    "oas_shrinkage":             oas.shrinkage_,
    "n_prices_filled":           n_filled,
    "n_stocks_dropped_returns":  n_dropped_returns,
}])
preprocessing_summary.to_csv(f"{OUTPUT_DIR}/preprocessing_summary.csv", index=False)
log(f"  preprocessing_summary.csv   — 1 row, {preprocessing_summary.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# DIAGNOSTIC PLOTS (saved to figures/)
# ─────────────────────────────────────────────────────────────
log(f"\n[Plots] Generating diagnostic figures...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Step 2 Diagnostics — "
    f"{'FICO Simulated' if DATASET_TYPE == 'simulated' else 'Bloomberg'} Universe",
    fontsize=14
)

# Plot 1: ESG score distribution (esg_final, post-transformation)
axes[0, 0].hist(esg_final, bins=40, color="#4C9BE8", edgecolor="white", linewidth=0.5)
axes[0, 0].axvline(55, color="red", linestyle="--", linewidth=1.5, label="ESG floor = 55 (baseline)")
axes[0, 0].set_title("ESG Score Distribution (esg_final, 0-100 scale)")
axes[0, 0].set_xlabel("ESG Score")
axes[0, 0].set_ylabel("Number of stocks")
axes[0, 0].legend()

# Plot 2: Annualised volatility distribution
axes[0, 1].hist(annual_vol, bins=40, color="#5DBE8A", edgecolor="white", linewidth=0.5)
axes[0, 1].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Vol = 100%")
axes[0, 1].set_title("Annualised Volatility per Stock")
axes[0, 1].set_xlabel("Annualised Volatility")
axes[0, 1].set_ylabel("Number of stocks")
axes[0, 1].legend()

# Plot 3: Max absolute daily return per stock (spike diagnostic, 4 thresholds)
axes[0, 2].hist(abs_max_return, bins=60, color="#E84C4C", edgecolor="white", linewidth=0.5)
thr_colors = ["orange", "red", "darkred", "black"]
for thr, col in zip(SPIKE_THRESHOLDS, thr_colors):
    axes[0, 2].axvline(thr, color=col, linestyle="--", linewidth=1.2,
                        label=f"|r| > {thr*100:.0f}%")
axes[0, 2].set_title("Max |Daily Return| per Stock")
axes[0, 2].set_xlabel("Max absolute daily log return")
axes[0, 2].set_ylabel("Number of stocks")
axes[0, 2].legend(fontsize=8)

# Plot 4: Sector breakdown
sector_counts = meta["sector"].value_counts()
axes[1, 0].barh(sector_counts.index, sector_counts.values, color="#4C9BE8")
axes[1, 0].set_title("Stocks per Sector")
axes[1, 0].set_xlabel("Number of stocks")

# Plot 5: Native (FICO) mu vs empirical mu scatter — robustness comparison
mu_native_plot = meta["mu_native"].values
axes[1, 1].scatter(mu_native_plot, mu_emp, alpha=0.4, s=15, color="#9B8FE8")
lims = [min(mu_native_plot.min(), mu_emp.min()) - 0.01,
        max(mu_native_plot.max(), mu_emp.max()) + 0.01]
axes[1, 1].plot(lims, lims, "r--", linewidth=1.2, label="45 line (perfect match)")
axes[1, 1].set_title(
    f"Native FICO mu vs Empirical mu\n(corr = {corr_mu:.3f}, MAE = {mae_mu:.4f})"
)
axes[1, 1].set_xlabel("mu_native  (FICO simulated, main for simulated)")
axes[1, 1].set_ylabel("mu_empirical_log  (empirical = 252 x mean daily return, robustness for simulated)")
axes[1, 1].legend(fontsize=8)

# Plot 6: Implied correlation heatmap (first 50 stocks, Ledoit-Wolf)
N50   = min(50, Sigma_lw.shape[0])
S50   = Sigma_lw[:N50, :N50]
D50   = np.sqrt(np.diag(S50))
with np.errstate(divide="ignore", invalid="ignore"):
    Corr50 = S50 / np.outer(D50, D50)
Corr50 = np.clip(Corr50, -1, 1)

sns.heatmap(Corr50, ax=axes[1, 2], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False,
            cbar_kws={"shrink": 0.8})
axes[1, 2].set_title("Implied Correlation Matrix — First 50 Stocks (Ledoit-Wolf)")

plt.tight_layout()
fig_path = f"{FIGURES_DIR}/step2_diagnostics.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
log(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────
# SAVE PREPROCESSING LOG
# ─────────────────────────────────────────────────────────────
log_lines_extra = [
    "",
    "-" * 65,
    "METHODOLOGICAL CHOICES RECORDED (for reproducibility)",
    "-" * 65,
    f"Dataset type:                  {DATASET_TYPE}",
    f"ESG transformation:            {esg_transform_note}",
    f"ESG columns saved:             esg_raw, esg_winsor, esg_final",
    f"Column 'esg' in output:        overwritten with esg_final (0-100 scale)",
    f"ESG final convention:          common 0-100 optimization scale (Option A)",
    f"  → simulated: winsorised + linearly rescaled to [0, 100]",
    f"  → Bloomberg (step6): (BESG-1)/9*100 maps 1-10 to 0-100",
    f"  → both optimizers use 'esg'; floor grid: [40, 45, 50, 55, 60, 65]",
    f"Mu naming convention:          mu (main: mu_native simulated / empirical Bloomberg) | mu_winsor (winsorised main) | mu_native (FICO metadata) | mu_empirical_log (empirical = 252 x mean daily return)",
    f"Simulated main specification:  mu_native = original FICO/simulated metadata mu",
    f"Simulated robustness spec:     mu_empirical_log = empirical mu (symmetry check)",
    f"Bloomberg main specification:  mu_empirical_log = empirical mu",
    f"Mu correlation (FICO vs emp):  {corr_mu:.4f}  (mu_native vs mu_empirical_log)",
    f"Missing prices handling:       forward-fill <= {FFILL_LIMIT} consecutive gaps",
    f"Return construction:           daily log returns, annualised x {ANNUAL_DAYS}",
    f"Covariance annualisation:      daily covariance x {ANNUAL_DAYS}",
    f"Volatility (H3 / summary):     sample std of daily returns x sqrt({ANNUAL_DAYS})  [NOT LW diagonal]",
    f"Volatility (LW diagonal):      sqrt(diag(Sigma_lw)) — available but not used in summary",
    f"Return missingness filter:     drop stocks with > {MAX_MISSING_RETURN_FRAC*100:.0f}% missing",
    f"Spike thresholds reported:     {[f'{t*100:.0f}%' for t in SPIKE_THRESHOLDS]}",
    f"Stocks with |r| > 50%:         {(log_returns.abs() > 0.50).any(axis=1).sum()}",
    f"Zero-return fraction:          {frac_zero*100:.2f}% of valid observations",
    f"Consecutive equal prices:      {frac_repeated*100:.2f}% of valid price observations",
    f"Spike exclusions at preproc:   NONE — flagged only; exclusions in robustness step",
    f"LW shrinkage coefficient:      {lw.shrinkage_:.4f}",
    f"OAS shrinkage coefficient:     {oas.shrinkage_:.4f}",
    f"Final universe:                {N} stocks",
    f"Trading days (returns):        {log_returns.shape[1]}",
]
log_lines.extend(log_lines_extra)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\n[Log] Preprocessing log saved to {LOG_PATH}")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Final universe:                    {N} stocks")
print(f"  Trading days (returns):            {log_returns.shape[1]}")
print(f"  Sectors:                           {meta['sector'].nunique()}")
print(f"  ESG >= 70:                         {(esg_final >= 70).sum()} "
      f"({(esg_final >= 70).mean()*100:.1f}%)")
print(f"  ESG < 70:                          {(esg_final < 70).sum()} "
      f"({(esg_final < 70).mean()*100:.1f}%)")
print(f"  Avg annualised vol (sample std x sqrt(252)):  {annual_vol.mean():.4f}")
print(f"  Avg mu (main):                     {mu_vec.mean():.4f}  "
      f"({'mu_native/FICO' if DATASET_TYPE == 'simulated' else 'empirical'})")
print(f"  Avg mu (winsor):                   {mu_winso.mean():.4f}")
print(f"  Avg mu_empirical_log:              {mu_emp.mean():.4f}  "
      f"({'robustness spec' if DATASET_TYPE == 'simulated' else 'same as main'})")
print(f"  Corr(mu_native, mu_empirical):     {corr_mu:.4f}")
print(f"  Stocks with spikes > 50%:          "
      f"{(log_returns.abs() > 0.50).any(axis=1).sum()}")
print(f"  LW shrinkage coefficient:          {lw.shrinkage_:.4f}")
print(f"  OAS shrinkage coefficient:         {oas.shrinkage_:.4f}")
print(f"\nOutputs saved to {OUTPUT_DIR}/")
print(f"  meta_preprocessed.csv, log_returns.csv")
print(f"  sigma_sample/lw/oas_annual.npy")
print(f"  stock_diagnostics.csv, risk_summary.csv, preprocessing_summary.csv")
print(f"\nReady for Step 3 — Portfolio Optimisation.")
print("=" * 65)