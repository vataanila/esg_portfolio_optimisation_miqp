"""
06_load_bloomberg.py: Bloomberg Data Loading and Structural Cleaning
=====================================================================
Bloomberg EURO STOXX 600 - ESG-Constrained Portfolio Optimisation
Independent Research Project

PIPELINE POSITION:
  data/raw/TICKERS.csv        (metadata: ticker, ESG, sector, country)
  data/raw/SXXP_FINAL.csv     (daily closing prices — Bloomberg format)
  06_load_bloomberg.py (this file) → data/clean_bloomberg/bloomberg_meta_clean.csv
                                      data/clean_bloomberg/bloomberg_prices_clean.csv

PURPOSE:
  Load, inspect, and clean Bloomberg EURO STOXX 600 data.
  Mirrors the structure of the simulated pipeline:
  load → inspect → diagnose duplicates → clean step by step
  → final clean objects + before/after summary.

BLOOMBERG-SPECIFIC FORMAT NOTES:
  - Separator: semicolon (;)
  - SXXP_FINAL.csv has 5 header rows before data begins (Bloomberg artefact)
  - Missing prices encoded as the string "#N/A N/A"
  - ESG scores use the Bloomberg BESG scale (1-10), not the 0-100 FICO scale

Run:
    python src/06_load_bloomberg.py
"""

# =============================================================================
# 0. IMPORTS AND PATHS
# =============================================================================
import os
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "clean_bloomberg")
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKERS_CSV = os.path.join(RAW_DIR, "TICKERS.csv")
PRICES_CSV  = os.path.join(RAW_DIR, "SXXP_FINAL.csv")

# ------------------------------------------------------------------
# Analysis window: fixed for the full dataset period.
# ------------------------------------------------------------------
SAMPLE_START = "2016-01-01"
SAMPLE_END   = "2025-12-31"

# Stock-level coverage thresholds (applied INSIDE the sample window)
# A stock must have its first valid price no later than EARLY_START_CUTOFF
# and must still have valid prices after LATE_END_CUTOFF.
# Together with the >20% missingness filter, these form two complementary
# screens: a global missingness filter and an edge-coverage filter.
EARLY_START_CUTOFF    = "2017-01-01"  # must have data by this date
LATE_END_CUTOFF       = "2024-12-31"  # must still have data after this date
MAX_MISSING_PRICE_PCT = 0.20          # drop if >20% of days missing overall
MAX_FFILL_DAYS        = 5             # forward-fill gaps up to 5 consecutive days

print("=" * 70)
print("  STEP 6 -- BLOOMBERG DATA: LOAD + CLEAN PIPELINE")
print("=" * 70)
print(f"\n  Analysis window : {SAMPLE_START} to {SAMPLE_END}")
print(f"  (Full Bloomberg export: 2016-01-01 to 2025-12-31)")

# =============================================================================
# 1. LOAD RAW METADATA (TICKERS.csv)
# =============================================================================
print("\n-- 1. LOAD RAW METADATA ---------------------------------------------")

meta_raw = pd.read_csv(TICKERS_CSV, sep=";")
meta_raw.columns = meta_raw.columns.str.strip()

if "Ticker" not in meta_raw.columns:
    raise ValueError("Column 'Ticker' not found in TICKERS.csv -- check column names.")

# Standardise: strip whitespace, consistent uppercase
meta_raw["Ticker"] = meta_raw["Ticker"].astype(str).str.strip().str.upper()

for col in ["ESG_SCORE", "ENVIRONMENTAL_SCORE", "SOCIAL_SCORE",
            "GOVERNANCE_SCORE", "CUR_MKT_CAP", "Price"]:
    if col in meta_raw.columns:
        meta_raw[col] = pd.to_numeric(meta_raw[col], errors="coerce")

n_meta_raw = len(meta_raw)
print(f"Loaded: {n_meta_raw} rows x {meta_raw.shape[1]} columns")
print(f"Columns: {meta_raw.columns.tolist()}")

# =============================================================================
# 2. LOAD RAW PRICES (SXXP_FINAL.csv -- Bloomberg multi-row header)
# =============================================================================
print("\n-- 2. LOAD RAW PRICES (Bloomberg format) ----------------------------")

prices_raw = pd.read_csv(PRICES_CSV, sep=";", header=None, low_memory=False)
print(f"Raw file: {prices_raw.shape[0]} rows x {prices_raw.shape[1]} columns")
print(f"Bloomberg reported range: {prices_raw.iloc[0, 1]} to {prices_raw.iloc[1, 1]}")

# Ticker names in row 3; data starts at row 6
tickers_from_prices = (prices_raw.iloc[3, 1:]
                       .astype(str).str.strip().str.upper().tolist())
tickers_from_prices = [t for t in tickers_from_prices if t not in ("NAN", "")]

prices_data = prices_raw.iloc[6:, :len(tickers_from_prices) + 1].copy()
prices_data.columns = ["Date"] + tickers_from_prices

prices_data["Date"] = pd.to_datetime(
    prices_data["Date"], dayfirst=True, errors="coerce")
prices_data = prices_data.dropna(subset=["Date"]).set_index("Date").sort_index()

prices_data = (prices_data
               .replace({"#N/A N/A": np.nan,
                         "#N/A Requesting Data...": np.nan})
               .apply(pd.to_numeric, errors="coerce"))
prices_data = prices_data.dropna(how="all")

n_prices_raw = prices_data.shape[1]
n_days_raw   = prices_data.shape[0]
print(f"Reconstructed: {n_days_raw} days x {n_prices_raw} stocks")
print(f"Actual range: {prices_data.index[0].date()} to {prices_data.index[-1].date()}")

# =============================================================================
# 3. APPLY SAMPLE WINDOW (first operation on prices)
# =============================================================================
# This is applied before any stock-level cleaning so that all downstream
# missingness calculations, coverage checks, and day counts refer to the
# same fixed period. The raw file may extend into 2025; we cut it here.
print("\n-- 3. APPLY SAMPLE WINDOW -------------------------------------------")

n_days_before_trim = len(prices_data)
prices_data = prices_data.loc[SAMPLE_START:SAMPLE_END]
n_days_after_trim  = len(prices_data)

print(f"  Days before trim : {n_days_before_trim}")
print(f"  Days after trim  : {n_days_after_trim}  ({SAMPLE_START} to {SAMPLE_END})")
print(f"  Days removed     : {n_days_before_trim - n_days_after_trim}  "
      f"(outside analysis window)")
print(f"  Stocks in file   : {n_prices_raw}  (unchanged -- trimming is time-only)")

# =============================================================================
# 4. RAW DIAGNOSTICS -- METADATA
# =============================================================================
print("\n-- 4. RAW METADATA DIAGNOSTICS --------------------------------------")

print("\nMissing values per column:")
print(meta_raw.isnull().sum().to_string())

esg = meta_raw["ESG_SCORE"].dropna()
print(f"\nESG (Bloomberg BESG, 1-10 scale):")
print(f"  Valid: {len(esg)} / {n_meta_raw} | Range: [{esg.min():.2f}, {esg.max():.2f}] | "
      f"Mean: {esg.mean():.2f} | Median: {esg.median():.2f}")

print("\nSector distribution:")
print(meta_raw["Sector"].value_counts().to_string())
print(f"  Missing sector: {meta_raw['Sector'].isna().sum()}")

print("\nCountry distribution (top 10):")
print(meta_raw["COUNTRY"].value_counts().head(10).to_string())
print(f"  Missing country: {meta_raw['COUNTRY'].isna().sum()}")

# =============================================================================
# 5. RAW DIAGNOSTICS -- PRICES (within sample window)
# =============================================================================
print("\n-- 5. RAW PRICE DIAGNOSTICS (within sample window) -----------------")

valid_vals = prices_data.values[~np.isnan(prices_data.values)]
print(f"  Negative prices: {(valid_vals < 0).sum()} | Zero: {(valid_vals == 0).sum()} | "
      f"Min: {valid_vals.min():.2f} | Max: {valid_vals.max():.2f}")

nan_pct = prices_data.isna().mean()
print(f"\n  Missing price structure:")
print(f"    0% missing    : {(nan_pct == 0).sum()} stocks")
print(f"    0-5% missing  : {((nan_pct > 0) & (nan_pct < 0.05)).sum()} stocks")
print(f"    5-20% missing : {((nan_pct >= 0.05) & (nan_pct <= 0.20)).sum()} stocks")
print(f"    >20% missing  : {(nan_pct > 0.20).sum()} stocks  (above removal threshold)")

# =============================================================================
# 6. DUPLICATE DIAGNOSTICS (raw, before any cleaning)
# =============================================================================
print("\n-- 6. DUPLICATE DIAGNOSTICS (raw) -----------------------------------")

dup_meta = meta_raw[meta_raw["Ticker"].duplicated(keep=False)]["Ticker"].unique()
print(f"  Duplicate tickers in metadata : {len(dup_meta)}"
      + (f"  -> {list(dup_meta)}" if len(dup_meta) > 0 else "  (none)"))

price_col_counts = pd.Series(tickers_from_prices).value_counts()
dup_price = price_col_counts[price_col_counts > 1].index.tolist()
print(f"  Duplicate tickers in prices   : {len(dup_price)}"
      + (f"  -> {dup_price}" if len(dup_price) > 0 else "  (none)"))

# =============================================================================
# 7. TICKER ALIGNMENT CHECK
# =============================================================================
print("\n-- 7. TICKER ALIGNMENT CHECK ----------------------------------------")

meta_tickers  = set(meta_raw["Ticker"])
price_tickers = set(tickers_from_prices)
in_both    = meta_tickers & price_tickers
only_meta  = meta_tickers - price_tickers
only_price = price_tickers - meta_tickers

print(f"  In metadata only : {len(only_meta)}"
      + (f"  -> {sorted(only_meta)}" if only_meta else ""))
print(f"  In prices only   : {len(only_price)}"
      + (f"  -> {sorted(only_price)}" if only_price else ""))
print(f"  In both          : {len(in_both)}"
      + ("  (perfect alignment)" if not only_meta and not only_price else ""))

# =============================================================================
# 8. CLEANING PIPELINE
# =============================================================================
print("\n-- 8. CLEANING PIPELINE ---------------------------------------------")
print(f"\n  Starting universe: {n_meta_raw} stocks\n")

# After each step, re-align meta rows and price columns so that one
# consistent universe count flows through all steps.
# Both the stock SET and the stock ORDER are enforced: meta rows are
# reindexed to follow price column order, so the two objects are always
# in sync for any downstream matrix operation.
# The warning (rather than a hard crash) is intentional during development:
# if metadata rows and price columns diverge unexpectedly, we want to see
# the discrepancy and investigate rather than have the script abort silently
# mid-run. Once the data format is confirmed stable this can be tightened.
def realign(meta_in, prices_in):
    # Build common universe in price-column order (defines canonical order)
    common = [t for t in prices_in.columns if t in set(meta_in["Ticker"])]
    p = prices_in[common].copy()
    # Reindex metadata to the same order -- enforces set AND order
    m = (meta_in.set_index("Ticker")
                .reindex(common)
                .reset_index())
    if len(m) != p.shape[1]:
        print(f"  WARNING: meta rows ({len(m)}) != price columns ({p.shape[1]}) "
              f"after realign -- investigate before continuing.")
    return m, p

removal_log = []  # (label, n_removed, n_remaining)

# -- A: Keep only tickers present in BOTH files ------------------------------
meta   = meta_raw[meta_raw["Ticker"].isin(in_both)].copy()
prices = prices_data[[t for t in prices_data.columns if t in in_both]].copy()
meta, prices = realign(meta, prices)
removed = n_meta_raw - len(meta)
removal_log.append(("A: not in both files", removed, len(meta)))
print(f"  [A] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- B: Remove duplicate tickers in metadata ---------------------------------
before = len(meta)
meta   = meta.drop_duplicates(subset="Ticker", keep="first")
meta, prices = realign(meta, prices)
removed = before - len(meta)
removal_log.append(("B: duplicate tickers (meta)", removed, len(meta)))
print(f"  [B] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- C: Remove duplicate columns in prices -----------------------------------
before = len(meta)
prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]
meta, prices = realign(meta, prices)
removed = before - len(meta)
removal_log.append(("C: duplicate tickers (prices)", removed, len(meta)))
print(f"  [C] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- D: Remove stocks with missing ESG score ---------------------------------
# ESG is a hard model input -- stocks without a score cannot enter the
# optimiser and must be excluded at this stage.
before = len(meta)
meta   = meta[meta["ESG_SCORE"].notna()].copy()
meta, prices = realign(meta, prices)
removed = before - len(meta)
removal_log.append(("D: missing ESG score", removed, len(meta)))
print(f"  [D] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- E: Handle missing Sector / Country --------------------------------------
# Sector and country are used only for descriptive analysis (breakdowns,
# tables) and are not imposed as hard constraints in the optimiser.
# Stocks with missing values are therefore retained; missing entries are
# filled with "Unknown" so that downstream groupby operations remain
# consistent. If sector constraints are added later, this decision
# must be revisited.
n_miss_sector  = meta["Sector"].isna().sum()
n_miss_country = meta["COUNTRY"].isna().sum()
meta["Sector"]  = meta["Sector"].fillna("Unknown")
meta["COUNTRY"] = meta["COUNTRY"].fillna("Unknown")
removal_log.append(("E: missing sector/country (filled Unknown)", 0, len(meta)))
print(f"  [E] removed   0  ->  {len(meta)} stocks remain  "
      f"(filled {n_miss_sector} sector, {n_miss_country} country with 'Unknown')")

# -- F: Remove stocks with >20% missing prices -------------------------------
# Global missingness filter: if more than 20% of trading days in the
# sample window have no price, the stock's history is too incomplete
# to produce reliable return and covariance estimates.
before      = len(meta)
nan_pct_now = prices.isna().mean()
keep        = nan_pct_now[nan_pct_now <= MAX_MISSING_PRICE_PCT].index
prices      = prices[keep]
meta, prices = realign(meta, prices)
removed     = before - len(meta)
removal_log.append((f"F: >{int(MAX_MISSING_PRICE_PCT*100)}% missing prices",
                    removed, len(meta)))
print(f"  [F] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- G: Remove stocks with late start or early end ---------------------------
# Edge-coverage filter: a stock may pass the 20% global threshold yet
# still be absent for a long stretch at the start or end of the window.
# We require a valid price by EARLY_START_CUTOFF (so early-year coverage
# is adequate) and at least one valid price after LATE_END_CUTOFF (so the
# stock has not been delisted well before the end of the analysis period).
# These cutoffs are set one year inside the sample boundaries, allowing
# for a small grace period while excluding stocks with structurally
# incomplete histories.
before      = len(meta)
first_valid = prices.apply(lambda col: col.first_valid_index())
last_valid  = prices.apply(lambda col: col.last_valid_index())
too_late    = first_valid[first_valid > pd.Timestamp(EARLY_START_CUTOFF)].index
too_early   = last_valid[last_valid   < pd.Timestamp(LATE_END_CUTOFF)].index
incomplete  = too_late.union(too_early)

if len(incomplete) > 0:
    print(f"         {len(too_late)} stocks have no data before {EARLY_START_CUTOFF}")
    print(f"         {len(too_early)} stocks have no data after  {LATE_END_CUTOFF}")
    prices = prices.drop(columns=incomplete)
    meta, prices = realign(meta, prices)

removed = before - len(meta)
removal_log.append(("G: late start / early end", removed, len(meta)))
print(f"  [G] removed {removed:3d}  ->  {len(meta)} stocks remain")

# -- H: Forward-fill short gaps only (no back-fill) --------------------------
# Carries the last known price forward over short gaps caused by
# non-synchronous national holidays across European markets.
# The 5-day limit prevents filling genuine suspensions or halts.
# Back-fill is excluded: it uses future prices to fill earlier missing
# observations, which is indefensible in a return-based analysis.
before_nan = prices.isna().sum().sum()
prices     = prices.ffill(limit=MAX_FFILL_DAYS)
after_nan  = prices.isna().sum().sum()
print(f"\n  [H: forward-fill only, max {MAX_FFILL_DAYS} days] "
      f"NaN cells: {before_nan} -> {after_nan}")

if after_nan > 0:
    still_nan = prices.columns[prices.isna().any()].tolist()
    prices    = prices.drop(columns=still_nan)
    meta, prices = realign(meta, prices)
    print(f"         Dropped {len(still_nan)} stock(s) with unfillable gaps.")
    removal_log.append(("H: unfillable gaps after forward-fill",
                        len(still_nan), len(meta)))
else:
    removal_log.append(("H: unfillable gaps after forward-fill", 0, len(meta)))

n_final      = len(meta)
n_days_final = len(prices)

# =============================================================================
# 9. FINAL CLEAN DATASET SUMMARY
# =============================================================================
print("\n-- 9. FINAL CLEAN DATASET SUMMARY -----------------------------------")

print(f"\n  {'Step':<48} {'Removed':>8} {'Remaining':>10}")
print("  " + "-" * 68)
print(f"  {'Raw metadata':<48} {'':>8} {n_meta_raw:>10}")
for label, removed, remaining in removal_log:
    print(f"  {label:<48} {removed:>8} {remaining:>10}")
print("  " + "-" * 68)
print(f"  {'FINAL CLEAN UNIVERSE':<48} {'':>8} {n_final:>10} stocks")
print(f"  {'FINAL TRADING DAYS':<48} {'':>8} {n_days_final:>10}")

pct_retained = n_final / n_meta_raw * 100
print(f"\n  Cross-section (stocks):")
print(f"    {n_meta_raw} raw  ->  {n_final} clean  "
      f"({n_meta_raw - n_final} removed by stock-level filters, {pct_retained:.1f}% retained)")
print(f"\n  Time dimension (trading days):")
print(f"    {n_days_raw} in raw file")
print(f"    {n_days_after_trim} after sample-window trim  "
      f"({SAMPLE_START} to {SAMPLE_END})")
print(f"    {n_days_final} final  (days with >=1 valid price after stock cleaning)")

esg_clean = meta["ESG_SCORE"]
print(f"\n  ESG (1-10 scale, clean universe):")
print(f"    Range: [{esg_clean.min():.2f}, {esg_clean.max():.2f}] | "
      f"Mean: {esg_clean.mean():.3f} | Median: {esg_clean.median():.3f} | "
      f"Std: {esg_clean.std():.3f}")

print("\n  Sector breakdown (clean universe):")
print(meta["Sector"].value_counts().to_string())

print("\n  Country breakdown (clean universe, top 10):")
print(meta["COUNTRY"].value_counts().head(10).to_string())

# =============================================================================
# 10. SAVE CLEAN OUTPUTS
# =============================================================================
print("\n-- 10. SAVING OUTPUTS -----------------------------------------------")

# Align meta row order to price column order before saving
meta = meta.set_index("Ticker").reindex(prices.columns).reset_index()

meta_out   = os.path.join(RESULTS_DIR, "bloomberg_meta_clean.csv")
prices_out = os.path.join(RESULTS_DIR, "bloomberg_prices_clean.csv")

meta.to_csv(meta_out, index=False)
prices.to_csv(prices_out)

print(f"  bloomberg_meta_clean.csv   -- {meta.shape[0]} stocks x {meta.shape[1]} cols")
print(f"  bloomberg_prices_clean.csv -- {prices.shape[0]} days x {prices.shape[1]} stocks")

# =============================================================================
# 11. FINAL CLEAN OBJECTS (passed to step6_preprocess_bloomberg.py)
# =============================================================================
meta_clean   = meta.copy()    # DataFrame: n_final rows, Ticker as first column
prices_clean = prices.copy()  # DataFrame: n_days_final rows (date index), n_final cols

print(f"""
-- FINAL CLEAN OBJECTS ---------------------------------------------------
  meta_clean   : {meta_clean.shape[0]} stocks x {meta_clean.shape[1]} columns
  prices_clean : {prices_clean.shape[0]} trading days x {prices_clean.shape[1]} stocks
  Date range   : {prices_clean.index[0].date()} to {prices_clean.index[-1].date()}
-------------------------------------------------------------------------
""")

print("=" * 70)
print("  STEP 6 COMPLETE -- Run 07_preprocess_bloomberg.py next.")
print("=" * 70)