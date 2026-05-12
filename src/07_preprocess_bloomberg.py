"""
07_preprocess_bloomberg.py: Bloomberg Analytical Preprocessing
================================================================
Bloomberg EURO STOXX 600 - ESG-Constrained Portfolio Optimisation
Independent Research Project

PIPELINE POSITION:
  06_load_bloomberg.py → data/clean_bloomberg/bloomberg_meta_clean.csv
                          data/clean_bloomberg/bloomberg_prices_clean.csv
  07_preprocess_bloomberg.py (this file) → data/clean_bloomberg/bloomberg_meta_analytical.csv
                                            data/clean_bloomberg/bloomberg_Sigma_{sample,lw,oas}.csv
                                            data/clean_bloomberg/bloomberg_returns.csv
                                            data/clean_bloomberg/bloomberg_preproc_log.txt
                                            figures/bloomberg_*.png

PURPOSE:
  Analytical preprocessing of the clean Bloomberg universe.
  Step 06 answered: which stocks survive structural cleaning?
  Step 07 answers:  what statistical inputs do those stocks generate?

METHODOLOGICAL CHOICES (recorded here and in the log):
  1. Expected returns : empirical historical mean of daily log returns, annualised.
                        Bloomberg has no native "given mu" like the FICO dataset,
                        so the historical estimate is the primary return input.
                        Bloomberg main specification: mu_trailing_winsor
                        (3-year trailing winsorised mu, as used in 08_optimize_bloomberg.py)
  2. ESG              : Bloomberg BESG native scale 1-10 retained as esg_raw.
                        esg_final = Bloomberg BESG mapped to common 0-100 optimisation scale
                        via: esg_final = (esg_raw - 1) / 9 * 100
                        (1→0, 5.5→50, 10→100 — linear, range-preserving)
                        esg = esg_final. Common scale shared with simulated dataset.
                        Native BESG equivalent of a 0-100 floor f: BESG = f / 100 * 9 + 1
  3. Time window      : inherited from Step 6 (2016-01-01 to 2025-12-31).
  4. Missing data     : validation only; structural cleaning was completed in Step 6.
  5. Covariance       : annualised sample + Ledoit-Wolf + OAS shrinkage estimators.
  6. Winsorization    : mu_winsor clips at [1st, 99th] percentile for robustness check.

Run:
    python src/07_preprocess_bloomberg.py
"""

# =============================================================================
# 0. IMPORTS AND PATHS
# =============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.covariance import LedoitWolf, OAS

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("future.no_silent_downcasting", True)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "data", "clean_bloomberg")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TRADING_DAYS  = 252
SAMPLE_START  = "2016-01-01"
SAMPLE_END    = "2025-12-31"

# Logging accumulator -- written to file at the end
log_lines = []

def log(msg="") -> None:
    """Print to console and append to the preprocessing log buffer."""
    print(msg)
    log_lines.append(str(msg))

log("=" * 70)
log("  STEP 7 -- BLOOMBERG ANALYTICAL PREPROCESSING")
log("=" * 70)

# =============================================================================
# 1. LOAD STEP 6 CLEAN OUTPUTS
# =============================================================================
log("\n-- 1. LOAD STEP 6 CLEAN OUTPUTS -------------------------------------")

meta_path   = os.path.join(RESULTS_DIR, "bloomberg_meta_clean.csv")
prices_path = os.path.join(RESULTS_DIR, "bloomberg_prices_clean.csv")

for path in [meta_path, prices_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run 06_load_bloomberg.py first.")

meta   = pd.read_csv(meta_path)
prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

# Normalise ALL column names to lowercase immediately after loading.
# This handles whatever capitalisation Step 6 used when saving.
meta.columns   = meta.columns.str.strip().str.lower()
# The meta CSV stores tickers in the 'index' column (Step 6 saved the index as a column)
if "ticker" not in meta.columns and "index" in meta.columns:
    meta = meta.rename(columns={"index": "ticker"})
prices.columns = prices.columns.str.strip().str.upper()  # tickers stay UPPER
prices         = prices.sort_index()
meta["ticker"] = meta["ticker"].astype(str).str.strip().str.upper()

n_stocks = len(meta)
n_days   = len(prices)

log(f"  meta   : {meta.shape[0]} rows x {meta.shape[1]} columns")
log(f"  prices : {prices.shape[0]} days x {prices.shape[1]} stocks")

# Alignment check
meta_tickers  = set(meta["ticker"])
price_tickers = set(prices.columns)
if meta_tickers != price_tickers:
    only_meta  = meta_tickers - price_tickers
    only_price = price_tickers - meta_tickers
    raise ValueError(
        f"Ticker mismatch between meta and prices.\n"
        f"  Only in meta  : {only_meta}\n"
        f"  Only in prices: {only_price}\n"
        "Re-run 06_load_bloomberg.py.")
log(f"  Ticker alignment : OK ({n_stocks} stocks in both files)")

# Date range check
actual_start = prices.index[0].date().isoformat()
actual_end   = prices.index[-1].date().isoformat()
log(f"  Date range       : {actual_start} to {actual_end}")
if actual_start != SAMPLE_START or actual_end != SAMPLE_END:
    log(f"  WARNING: expected {SAMPLE_START} to {SAMPLE_END} -- check Step 6 output.")
else:
    log(f"  Date range       : matches expected window  OK")

# =============================================================================
# 2. STANDARDISE INTERNAL METADATA SCHEMA
# =============================================================================
log("\n-- 2. STANDARDISE METADATA SCHEMA -----------------------------------")

# Since we lowercased all columns on load, the rename map uses lowercase keys.
# This maps the original Bloomberg field names (now lowercased) to pipeline names.
rename_map = {
    "ticker"              : "ticker",       # already correct, included for clarity
    "name"                : "name",
    "sector"              : "sector",
    "country"             : "country",
    "esg_score"           : "esg_raw",      # Bloomberg BESG -> pipeline name
    "environmental_score" : "esg_env",
    "social_score"        : "esg_soc",
    "governance_score"    : "esg_gov",
    "cur_mkt_cap"         : "mkt_cap",
    "gics_industry_name"  : "industry",
    "price"               : "price_last",
    "isin"                : "isin",
}
meta = meta.rename(columns={k: v for k, v in rename_map.items()
                             if k in meta.columns})

# Enforce column order: identifier fields first, then analytical fields
id_cols  = ["ticker", "isin", "name", "sector", "industry", "country", "mkt_cap",
            "price_last"]
esg_cols = ["esg_raw", "esg_env", "esg_soc", "esg_gov"]
id_cols  = [c for c in id_cols  if c in meta.columns]
esg_cols = [c for c in esg_cols if c in meta.columns]
meta     = meta[id_cols + esg_cols]

# Align row order to price column order (canonical order = price columns)
meta = meta.set_index("ticker").reindex(prices.columns).reset_index()
meta = meta.rename(columns={"index": "ticker"})  # ensure column is always named 'ticker'

log(f"  Schema standardised. Columns: {meta.columns.tolist()}")
log(f"  Row order aligned to price column order.")

# =============================================================================
# 3. FINAL PRICE SANITY VALIDATION
# =============================================================================
log("\n-- 3. FINAL PRICE SANITY VALIDATION ---------------------------------")
log("  (validation only -- structural cleaning completed in Step 6)")

issues = {}
issues["negative"]  = int((prices.values < 0).sum())
issues["zero"]      = int((prices.values == 0).sum())
issues["infinite"]  = int(np.isinf(prices.values).sum())
issues["nan"]       = int(np.isnan(prices.values).sum())

for label, count in issues.items():
    status = "OK" if count == 0 else f"WARNING: {count} cells"
    log(f"  {label:<12}: {status}")

if any(v > 0 for v in issues.values()):
    log("  ACTION: non-zero issue count detected. Investigate before proceeding.")
else:
    log("  All checks passed -- price matrix is clean.")

# =============================================================================
# 4. CONFIRM TIME STRUCTURE
# =============================================================================
log("\n-- 4. CONFIRM TIME STRUCTURE ----------------------------------------")

sorted_asc = prices.index.is_monotonic_increasing
log(f"  Dates sorted ascending : {sorted_asc}")
log(f"  First date             : {prices.index[0].date()}")
log(f"  Last date              : {prices.index[-1].date()}")
log(f"  Total trading days     : {n_days}")

gaps = prices.index.to_series().diff().dt.days.dropna()
large_gaps = gaps[gaps > 5]
log(f"  Calendar gaps > 5 days : {len(large_gaps)}"
    + ("  (likely holiday clusters -- expected)" if len(large_gaps) > 0 else ""))

# =============================================================================
# 5. ESG ANALYTICAL TRANSFORMATION
# =============================================================================
log("\n-- 5. ESG ANALYTICAL TRANSFORMATION ---------------------------------")
log("  Bloomberg BESG native scale: 1-10 (higher = better ESG profile)")
log("  Common 0-100 ESG scale across both datasets (range-preserving linear map):")
log("    esg_raw   = original Bloomberg BESG score (1-10), retained for traceability")
log("    esg_final = Bloomberg BESG mapped to common 0-100 optimisation scale")
log("    esg       = alias of esg_final — the variable used by the optimiser")
log("  Mapping formula: esg_final = (esg_raw - 1) / 9 * 100")
log("    (1→0, 5.5→50, 10→100 — linear, range-preserving)")
log("  Inverse (native BESG equivalent of a 0-100 floor f): BESG = f / 100 * 9 + 1")
log("  This common scale matches the simulated dataset (02_preprocess_simulated.py).")
log("  ESG floor sweep in Step 8 uses the common scale: [40, 45, 50, 55, 60, 65]")

# Map Bloomberg BESG (1-10) → common 0-100 optimisation scale
meta["esg_final"] = (meta["esg_raw"] - 1.0) / 9.0 * 100.0
meta["esg"]       = meta["esg_final"]   # alias used in optimiser

esg = meta["esg_final"]
log(f"\n  esg_raw   : range [{meta['esg_raw'].min():.2f}, {meta['esg_raw'].max():.2f}] "
    f"  mean {meta['esg_raw'].mean():.3f}  median {meta['esg_raw'].median():.3f}  "
    f"(native Bloomberg BESG, 1-10)")
log(f"  esg_final : range [{esg.min():.2f}, {esg.max():.2f}] "
    f"  mean {esg.mean():.3f}  median {esg.median():.3f}  "
    f"(common optimisation scale, 0-100)")

# How many stocks pass each candidate floor (on the common 0-100 scale)
log("\n  Stocks above candidate ESG floors (common 0-100 scale):")
for floor in [40, 45, 50, 55, 60, 65]:
    n   = (esg >= floor).sum()
    pct = n / len(esg) * 100
    log(f"    >= {floor:2d} : {n:3d} stocks  ({pct:.1f}%)"
        f"  [native BESG equivalent: {floor / 100 * 9 + 1:.2f}]")

# =============================================================================
# 6. COMPUTE DAILY LOG RETURNS
# =============================================================================
log("\n-- 6. COMPUTE DAILY LOG RETURNS -------------------------------------")

returns = np.log(prices / prices.shift(1)).iloc[1:]   # drop first NaN row

n_ret_days   = len(returns)
n_ret_stocks = returns.shape[1]
nan_in_ret   = returns.isna().sum().sum()

log(f"  Return matrix shape : {n_ret_days} days x {n_ret_stocks} stocks")
log(f"  Residual NaN cells  : {nan_in_ret}"
    + ("  OK" if nan_in_ret == 0 else "  WARNING -- investigate"))

# =============================================================================
# 7. LIGHT RETURN-QUALITY CHECKS
# =============================================================================
log("\n-- 7. LIGHT RETURN-QUALITY CHECKS -----------------------------------")
log("  Diagnostic only -- no stocks removed here.")

# Missing returns by stock
missing_by_stock = returns.isna().sum()
stocks_with_missing = (missing_by_stock > 0).sum()
log(f"  Stocks with any missing returns : {stocks_with_missing}")

# Extreme spike detection (|return| > 50% in one day)
spike_mask  = returns.abs() > 0.50
spike_count = spike_mask.sum().sum()
spike_stocks = (spike_mask.sum(axis=0) > 0).sum()
log(f"  Single-day |return| > 50%       : {spike_count} observations "
    f"in {spike_stocks} stocks")

# Zero-return concentration (possible stale prices)
zero_mask   = (returns == 0)
zero_pct_by_stock = zero_mask.mean()
high_zero   = (zero_pct_by_stock > 0.10).sum()
log(f"  Stocks with >10% zero returns   : {high_zero}  "
    f"(potential stale-price / low-liquidity flag)")

if spike_count > 0 or high_zero > 0:
    log("  NOTE: flagged stocks retained -- review before use.")

# =============================================================================
# 8. STATISTICAL DIAGNOSTICS ON RETURNS
# =============================================================================
log("\n-- 8. RETURN DIAGNOSTICS --------------------------------------------")

ann_vol   = returns.std() * np.sqrt(TRADING_DAYS)
ann_ret   = returns.mean() * TRADING_DAYS
max_abs   = returns.abs().max()
skew_vals = returns.skew()
kurt_vals = returns.kurtosis()

log(f"\n  Annualised volatility (cross-stock):")
log(f"    Min    : {ann_vol.min():.4f}")
log(f"    Median : {ann_vol.median():.4f}")
log(f"    Mean   : {ann_vol.mean():.4f}")
log(f"    Max    : {ann_vol.max():.4f}")

log(f"\n  Annualised mean return (cross-stock):")
log(f"    Min    : {ann_ret.min():.4f}")
log(f"    Median : {ann_ret.median():.4f}")
log(f"    Mean   : {ann_ret.mean():.4f}")
log(f"    Max    : {ann_ret.max():.4f}")

log(f"\n  Return skewness (cross-stock):")
log(f"    Mean   : {skew_vals.mean():.3f}  "
    f"Std : {skew_vals.std():.3f}  "
    f"Range : [{skew_vals.min():.2f}, {skew_vals.max():.2f}]")

log(f"\n  Return excess kurtosis (cross-stock):")
log(f"    Mean   : {kurt_vals.mean():.3f}  "
    f"(heavy tails expected in daily equity returns)")

# Per-stock diagnostics table
diag = pd.DataFrame({
    "ticker"       : returns.columns,
    "ann_return"   : ann_ret.values,
    "ann_vol"      : ann_vol.values,
    "sharpe_hist"  : (ann_ret / ann_vol).values,
    "skewness"     : skew_vals.values,
    "kurtosis"     : kurt_vals.values,
    "max_abs_ret"  : max_abs.values,
    "pct_zero_ret" : zero_pct_by_stock.values,
    "n_spikes_50"  : spike_mask.sum(axis=0).values,
})

# =============================================================================
# 9. BUILD MAIN MU (historical empirical expected return)
# =============================================================================
log("\n-- 9. BUILD EXPECTED RETURN VECTOR (mu) -----------------------------")
log("  Bloomberg has no native 'given mu' (unlike the FICO simulated dataset).")
log("  Primary input: historical empirical mean of daily log returns, annualised.")
log(f"  Formula: mu_i = mean(r_i) * {TRADING_DAYS}")

mu_raw    = returns.mean() * TRADING_DAYS          # main expected return
mu_winsor = mu_raw.copy()                           # robustness version

# Winsorise at [1st, 99th] percentile to reduce sensitivity to outlier stocks
p1, p99 = mu_raw.quantile(0.01), mu_raw.quantile(0.99)
mu_winsor = mu_winsor.clip(lower=p1, upper=p99)

log(f"\n  mu_raw    : range [{mu_raw.min():.4f}, {mu_raw.max():.4f}]  "
    f"mean {mu_raw.mean():.4f}  median {mu_raw.median():.4f}")
log(f"  mu_winsor : range [{mu_winsor.min():.4f}, {mu_winsor.max():.4f}]  "
    f"(clipped at p1={p1:.4f}, p99={p99:.4f})")
log(f"  Stocks modified by winsorisation: "
    f"{((mu_winsor != mu_raw)).sum()}")

# Store in metadata
meta["mu"]        = mu_raw.values
meta["mu_winsor"] = mu_winsor.values

# ── 3-year trailing window mu ─────────────────────────────────────────
TRAILING_DAYS = 252 * 3  # approximately 3 years of trading days

# Use only the last TRAILING_DAYS rows of returns (days × stocks)
returns_trailing = returns.iloc[-TRAILING_DAYS:]

mu_trailing = returns_trailing.mean() * TRADING_DAYS  # annualised

# Winsorise trailing mu at p1/p99
p1_t, p99_t = mu_trailing.quantile(0.01), mu_trailing.quantile(0.99)
mu_trailing_winsor = mu_trailing.clip(lower=p1_t, upper=p99_t)

meta["mu_trailing"]        = mu_trailing.values
meta["mu_trailing_winsor"] = mu_trailing_winsor.values

# Diagnostic
log(f"\n  mu_trailing      : range [{mu_trailing.min():.4f}, {mu_trailing.max():.4f}]  "
    f"mean {mu_trailing.mean():.4f}  median {mu_trailing.median():.4f}  "
    f"(window: last {len(returns_trailing)} days of {len(returns)} total)")
log(f"  mu_trailing_winsor: range [{mu_trailing_winsor.min():.4f}, {mu_trailing_winsor.max():.4f}]  "
    f"(clipped at p1={p1_t:.4f}, p99={p99_t:.4f})")
log(f"  Stocks modified by trailing winsorisation: {(mu_trailing_winsor != mu_trailing).sum()}")

# =============================================================================
# 10. ESTIMATE COVARIANCE MATRICES
# =============================================================================
log("\n-- 10. ESTIMATE COVARIANCE MATRICES ---------------------------------")
log(f"  All matrices annualised by multiplying by {TRADING_DAYS}.")

R = returns.values  # (T, n) array

# Sample covariance
Sigma_sample = np.cov(R, rowvar=False) * TRADING_DAYS
log(f"\n  Sample covariance:")
log(f"    Shape           : {Sigma_sample.shape}")
log(f"    Min eigenvalue  : {np.linalg.eigvalsh(Sigma_sample).min():.6f}")
log(f"    Condition number: {np.linalg.cond(Sigma_sample):.2e}")

# Ledoit-Wolf shrinkage
lw_model    = LedoitWolf().fit(R)
Sigma_lw    = lw_model.covariance_ * TRADING_DAYS
lw_shrink   = lw_model.shrinkage_
log(f"\n  Ledoit-Wolf covariance:")
log(f"    Shrinkage coeff : {lw_shrink:.4f}")
log(f"    Min eigenvalue  : {np.linalg.eigvalsh(Sigma_lw).min():.6f}")
log(f"    Condition number: {np.linalg.cond(Sigma_lw):.2e}")

# OAS shrinkage
oas_model   = OAS().fit(R)
Sigma_oas   = oas_model.covariance_ * TRADING_DAYS
oas_shrink  = oas_model.shrinkage_
log(f"\n  OAS covariance:")
log(f"    Shrinkage coeff : {oas_shrink:.4f}")
log(f"    Min eigenvalue  : {np.linalg.eigvalsh(Sigma_oas).min():.6f}")
log(f"    Condition number: {np.linalg.cond(Sigma_oas):.2e}")

tickers = prices.columns.tolist()

# =============================================================================
# 11. ANALYSE CORRELATION STRUCTURE
# =============================================================================
log("\n-- 11. CORRELATION STRUCTURE ----------------------------------------")
log("  Real market data shows genuine cross-asset correlation structure.")
log("  For reference: compare with simulated dataset.")

def corr_stats(Sigma, label, tickers):
    vol  = np.sqrt(np.diag(Sigma))
    D    = np.diag(1.0 / vol)
    C    = D @ Sigma @ D
    n    = len(tickers)
    # Extract upper triangle (off-diagonal)
    idx  = np.triu_indices(n, k=1)
    off  = C[idx]
    log(f"\n  {label}:")
    log(f"    Avg off-diagonal correlation : {off.mean():.4f}")
    log(f"    Std of correlations          : {off.std():.4f}")
    log(f"    Min / Max correlation        : [{off.min():.4f}, {off.max():.4f}]")
    log(f"    Pct of pairs > 0.50          : {(off > 0.50).mean()*100:.1f}%")
    log(f"    Pct of pairs < 0             : {(off < 0).mean()*100:.1f}%")
    return {"estimator": label,
            "avg_corr": round(off.mean(), 4),
            "std_corr": round(off.std(), 4),
            "min_corr": round(off.min(), 4),
            "max_corr": round(off.max(), 4),
            "pct_gt_050": round((off > 0.50).mean()*100, 2),
            "pct_negative": round((off < 0).mean()*100, 2)}, off

corr_rows = []
for Sig, lbl in [(Sigma_sample, "Sample"),
                 (Sigma_lw,     "Ledoit-Wolf"),
                 (Sigma_oas,    "OAS")]:
    row, off_diag = corr_stats(Sig, lbl, tickers)
    corr_rows.append(row)

corr_summary = pd.DataFrame(corr_rows)

# =============================================================================
# 12. BUILD FINAL OPTIMIZATION-READY METADATA
# =============================================================================
log("\n-- 12. BUILD OPTIMIZATION-READY METADATA ----------------------------")

# Final analytical metadata: one row per stock, all optimizer inputs present
meta_analytical = meta.copy()

# Verify completeness
missing_mu  = meta_analytical["mu"].isna().sum()
missing_esg = meta_analytical["esg_final"].isna().sum()
log(f"  Rows            : {len(meta_analytical)}")
log(f"  Missing mu      : {missing_mu}")
log(f"  Missing esg     : {missing_esg}")
log(f"  Columns         : {meta_analytical.columns.tolist()}")

if missing_mu > 0 or missing_esg > 0:
    log("  WARNING: missing values in key optimizer inputs -- investigate.")
else:
    log("  All optimizer inputs complete -- ready for Step 7.")

# =============================================================================
# 13. SAVE ALL OUTPUTS
# =============================================================================
log("\n-- 13. SAVING OUTPUTS -----------------------------------------------")

# Helper
def save_csv(df_or_arr, fname, index=True, tickers_idx=None, tickers_col=None):
    path = os.path.join(RESULTS_DIR, fname)
    if isinstance(df_or_arr, np.ndarray):
        df = pd.DataFrame(df_or_arr,
                          index=tickers_idx if tickers_idx is not None else None,
                          columns=tickers_col if tickers_col is not None else None)
    else:
        df = df_or_arr
    df.to_csv(path, index=index)
    log(f"  Saved: {fname}  ({df.shape})")

save_csv(meta_analytical,  "bloomberg_meta_analytical.csv", index=False)
save_csv(returns,          "bloomberg_returns.csv")
save_csv(Sigma_sample,     "bloomberg_Sigma_sample.csv",
         tickers_idx=tickers, tickers_col=tickers)
save_csv(Sigma_lw,         "bloomberg_Sigma_lw.csv",
         tickers_idx=tickers, tickers_col=tickers)
save_csv(Sigma_oas,        "bloomberg_Sigma_oas.csv",
         tickers_idx=tickers, tickers_col=tickers)
save_csv(diag,             "bloomberg_stock_diagnostics.csv", index=False)
save_csv(corr_summary,     "bloomberg_corr_summary.csv",      index=False)

# Preprocessing summary (one-row overview for reproducibility)
preproc_summary = pd.DataFrame([{
    "n_stocks"         : n_stocks,
    "n_return_days"    : n_ret_days,
    "sample_start"     : SAMPLE_START,
    "sample_end"       : SAMPLE_END,
    "mu_mean"          : round(mu_raw.mean(), 4),
    "mu_median"        : round(mu_raw.median(), 4),
    "vol_mean"         : round(ann_vol.mean(), 4),
    "vol_median"       : round(ann_vol.median(), 4),
    "esg_mean"         : round(esg.mean(), 3),
    "esg_median"       : round(esg.median(), 3),
    "lw_shrinkage"     : round(lw_shrink, 4),
    "oas_shrinkage"    : round(oas_shrink, 4),
    "avg_corr_sample"  : corr_rows[0]["avg_corr"],
    "avg_corr_lw"      : corr_rows[1]["avg_corr"],
}])
save_csv(preproc_summary, "bloomberg_preproc_summary.csv", index=False)

# =============================================================================
# 14. FIGURES
# =============================================================================
log("\n-- 14. FIGURES ------------------------------------------------------")

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

def savefig(fname):
    path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {fname}")

# Fig 1: ESG distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(esg.dropna(), bins=30, color="#4472C4", edgecolor="white", linewidth=0.4)
ax.axvline(esg.median(), color="#C00000", linewidth=1.2, linestyle="--",
           label=f"Median = {esg.median():.2f}")
ax.set_xlabel("ESG Score (common optimization scale, 0-100)")
ax.set_ylabel("Number of stocks")
ax.set_title(f"ESG Score Distribution — Bloomberg Universe (n={n_stocks})")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("bloomberg_esg_distribution.png")

# Fig 2: Annualised volatility distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ann_vol, bins=35, color="#70AD47", edgecolor="white", linewidth=0.4)
ax.axvline(ann_vol.median(), color="#C00000", linewidth=1.2, linestyle="--",
           label=f"Median = {ann_vol.median():.3f}")
ax.set_xlabel("Annualised Volatility")
ax.set_ylabel("Number of stocks")
ax.set_title("Annualised Volatility Distribution — Bloomberg Universe")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("bloomberg_vol_distribution.png")

# Fig 3: Mu distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(mu_raw, bins=35, color="#ED7D31", edgecolor="white", linewidth=0.4,
        label="mu_raw")
ax.hist(mu_winsor, bins=35, color="#A9D18E", edgecolor="white", linewidth=0.4,
        alpha=0.6, label="mu_winsor")
ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
ax.set_xlabel("Annualised Expected Return")
ax.set_ylabel("Number of stocks")
ax.set_title("Expected Return Distribution (mu_raw vs mu_winsor)")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("bloomberg_mu_distribution.png")

# Fig 4: Max absolute daily return (spike indicator)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(max_abs, bins=40, color="#FF0000", edgecolor="white", linewidth=0.4,
        alpha=0.75)
ax.axvline(0.50, color="black", linewidth=1.0, linestyle="--",
           label="|return| = 50% spike threshold")
ax.set_xlabel("Maximum Absolute Daily Return")
ax.set_ylabel("Number of stocks")
ax.set_title("Maximum Absolute Daily Return — Spike Indicator")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("bloomberg_return_spikes.png")

# Fig 5: Off-diagonal correlation distribution (Sample estimator)
vol_s  = np.sqrt(np.diag(Sigma_sample))
D_s    = np.diag(1.0 / vol_s)
C_s    = D_s @ Sigma_sample @ D_s
n_     = len(tickers)
idx_   = np.triu_indices(n_, k=1)
off_s  = C_s[idx_]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(off_s, bins=60, color="#7030A0", edgecolor="white", linewidth=0.3,
        alpha=0.8)
ax.axvline(off_s.mean(), color="#C00000", linewidth=1.2, linestyle="--",
           label=f"Mean = {off_s.mean():.3f}")
ax.set_xlabel("Pairwise Correlation")
ax.set_ylabel("Number of stock pairs")
ax.set_title("Pairwise Correlation Distribution — Sample Estimator")
ax.legend(frameon=False)
ax.text(0.98, 0.92, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("bloomberg_corr_distribution.png")

# Fig 6: Sector breakdown
sector_counts = meta_analytical["sector"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
sector_counts.sort_values().plot(kind="barh", ax=ax, color="#4472C4",
                                  edgecolor="white")
ax.set_xlabel("Number of stocks")
ax.set_title(f"Sector Breakdown — Bloomberg Clean Universe (n={n_stocks})")
ax.text(0.98, 0.02, "Source: Author's elaboration.",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
savefig("bloomberg_sector_breakdown.png")

# =============================================================================
# 15. FINAL ANALYTICAL SUMMARY
# =============================================================================
log("\n-- 15. FINAL ANALYTICAL SUMMARY -------------------------------------")
log(f"""
  Dataset             : Bloomberg EURO STOXX 600
  Final universe      : {n_stocks} stocks
  Return observations : {n_ret_days} trading days per stock
  Sample window       : {SAMPLE_START} to {SAMPLE_END}

  Expected returns (mu):
    mean              : {mu_raw.mean():.4f}
    median            : {mu_raw.median():.4f}
    range             : [{mu_raw.min():.4f}, {mu_raw.max():.4f}]

  Annualised volatility:
    mean              : {ann_vol.mean():.4f}
    median            : {ann_vol.median():.4f}
    range             : [{ann_vol.min():.4f}, {ann_vol.max():.4f}]

  ESG (common optimisation scale, 0-100):
    mean              : {esg.mean():.3f}
    median            : {esg.median():.3f}
    range             : [{esg.min():.2f}, {esg.max():.2f}]
    mapping           : (Bloomberg BESG - 1) / 9 * 100
    (raw Bloomberg BESG 1-10 retained separately in esg_raw)

  Covariance estimators:
    Ledoit-Wolf shrinkage : {lw_shrink:.4f}
    OAS shrinkage         : {oas_shrink:.4f}

  Correlation structure (sample estimator):
    avg off-diagonal      : {corr_rows[0]['avg_corr']:.4f}
    std off-diagonal      : {corr_rows[0]['std_corr']:.4f}
""")

log("=" * 70)
log("  STEP 7 COMPLETE -- Run 08_optimize_bloomberg.py next.")
log("=" * 70)

# =============================================================================
# 16. SAVE PREPROCESSING LOG
# =============================================================================
log_path = os.path.join(RESULTS_DIR, "bloomberg_preproc_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"  Log saved: bloomberg_preproc_log.txt")