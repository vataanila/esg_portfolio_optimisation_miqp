"""
01_load_simulated.py: Load and Inspect Raw Simulated Data
=========================================================
ESG-Constrained Portfolio Optimisation - Independent Research Project

Pipeline position:
  Step 01 (this file)  →  structural cleaning  →  data/clean/universe_clean.csv
                                                    data/clean/prices_clean.csv

Files needed in data/raw/:
  - stockprices2500.csv   (simulated daily prices, 2500 stocks)
  - shares2500_fixed.csv  (FICO-assigned returns, ESG scores, sector labels)

Run from the project root:
  python src/01_load_simulated.py

Dependencies:
  pip install pandas numpy
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
PATH_PRICES = "data/raw/stockprices2500.csv"
PATH_META   = "data/raw/shares2500_fixed.csv"

# ─────────────────────────────────────────────
# 1. LOAD METADATA
# ─────────────────────────────────────────────
print("=" * 60)
print("LOADING METADATA (shares2500_fixed.csv)")
print("=" * 60)

meta = pd.read_csv(PATH_META, index_col=0)
meta.index   = meta.index.str.strip()
meta.columns = meta.columns.str.strip()
meta = meta.rename(columns={
    "Return":    "mu",
    "Sector":    "sector",
    "ESG score": "esg"
})

print(f"Shape:         {meta.shape}")
print(f"Columns:       {meta.columns.tolist()}")
print(f"Index sample:  {meta.index[:5].tolist()}")
print(f"\nFirst 5 rows:")
print(meta.head())

print(f"\nMissing values:")
print(meta.isnull().sum())

print(f"\nSector breakdown:")
print(meta["sector"].value_counts())

print(f"\nESG score stats:")
print(meta["esg"].describe())

print(f"\nReturn (mu) stats:")
print(meta["mu"].describe())

# ─────────────────────────────────────────────
# 2. LOAD PRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOADING PRICES (stockprices2500.csv)")
print("=" * 60)

prices_raw = pd.read_csv(PATH_PRICES, index_col=0)
prices_raw.index   = prices_raw.index.str.strip()
prices_raw.columns = prices_raw.columns.str.strip()

print(f"Shape:         {prices_raw.shape}  (stocks × days)")
print(f"Index sample:  {prices_raw.index[:5].tolist()}")
print(f"Columns sample (first 5):  {prices_raw.columns[:5].tolist()}")
print(f"Columns sample (last 5):   {prices_raw.columns[-5:].tolist()}")

print(f"\nFirst 3 rows, first 6 columns:")
print(prices_raw.iloc[:3, :6])

print(f"\nMissing values per stock (first 10 stocks):")
print(prices_raw.isnull().sum(axis=1).head(10))

print(f"\nTotal missing values: {prices_raw.isnull().sum().sum()}")
print(f"Stocks with ANY missing:  {(prices_raw.isnull().sum(axis=1) > 0).sum()}")
print(f"Stocks with ALL present:  {(prices_raw.isnull().sum(axis=1) == 0).sum()}")

# ─────────────────────────────────────────────
# 3. CHECK ALIGNMENT BETWEEN THE TWO FILES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALIGNMENT CHECK")
print("=" * 60)

meta_stocks   = set(meta.index)
prices_stocks = set(prices_raw.index)

in_both        = meta_stocks & prices_stocks
only_in_meta   = meta_stocks - prices_stocks
only_in_prices = prices_stocks - meta_stocks

print(f"Stocks in metadata:       {len(meta_stocks)}")
print(f"Stocks in prices:         {len(prices_stocks)}")
print(f"Stocks in BOTH:           {len(in_both)}")
print(f"Only in metadata:         {len(only_in_meta)}")
print(f"Only in prices:           {len(only_in_prices)}")

if only_in_meta:
    print(f"  → Examples: {list(only_in_meta)[:5]}")
if only_in_prices:
    print(f"  → Examples: {list(only_in_prices)[:5]}")

# ─────────────────────────────────────────────
# 4. QUICK SANITY CHECKS ON PRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRICE SANITY CHECKS")
print("=" * 60)

n_zero     = (prices_raw <= 0).sum().sum()
n_negative = (prices_raw < 0).sum().sum()
print(f"Zero or negative prices:  {n_zero}")
print(f"Negative prices:          {n_negative}")

print(f"\nPrice range across all stocks:")
print(f"  Min: {prices_raw.min().min():.4f}")
print(f"  Max: {prices_raw.max().max():.4f}")
print(f"  Mean: {prices_raw.mean().mean():.4f}")

# ─────────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — WHAT TO EXPECT IN STEP 02")
print("=" * 60)
print(f"  Total stocks (in both files):  {len(in_both)}")
print(f"  Total trading days:            {prices_raw.shape[1]}")
print(f"  Missing ESG scores:            {meta['esg'].isnull().sum()}")
print(f"  Missing mu (return):           {meta['mu'].isnull().sum()}")
print(f"  Stocks with missing prices:    {(prices_raw.isnull().sum(axis=1) > 0).sum()}")
print("\nIf everything looks reasonable above, you are ready for Step 02.")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# DATA CLEANING SECTION
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DATA CLEANING — FULL PIPELINE")
print("=" * 60)

n_start_prices = len(prices_raw)
n_start_meta   = len(meta)

# ─────────────────────────────────────────────────────────────
# C1 — Remove invalid stock identifiers from the price file
# ─────────────────────────────────────────────────────────────

def _is_valid_ticker(s: str) -> bool:
    """True if the string contains at least one letter (real ticker)."""
    s = str(s).strip()
    try:
        float(s)
        return False
    except ValueError:
        pass
    return any(c.isalpha() for c in s)

valid_ticker_mask  = prices_raw.index.map(_is_valid_ticker)
invalid_ticker_ids = prices_raw.index[~valid_ticker_mask].tolist()
prices_c1          = prices_raw[valid_ticker_mask].copy()

n_removed_invalid = len(invalid_ticker_ids)
print(f"\n[C1] Invalid identifiers removed from price file: {n_removed_invalid}")
if invalid_ticker_ids:
    print(f"     Examples: {invalid_ticker_ids[:10]}")

# ─────────────────────────────────────────────────────────────
# C2 — Keep only stocks present in BOTH files
# ─────────────────────────────────────────────────────────────
common_stocks     = prices_c1.index.intersection(meta.index)
only_in_prices_c2 = prices_c1.index.difference(meta.index).tolist()
only_in_meta_c2   = meta.index.difference(prices_c1.index).tolist()

prices_c2 = prices_c1.loc[common_stocks].copy()
meta_c2   = meta.loc[common_stocks].copy()

n_removed_prices_no_meta = len(only_in_prices_c2)
n_removed_meta_no_prices = len(only_in_meta_c2)

print(f"\n[C2] Stocks kept after intersection with metadata: {len(common_stocks)}")
print(f"     Removed — in prices but not in metadata: {n_removed_prices_no_meta}")
print(f"     Removed — in metadata but not in prices: {n_removed_meta_no_prices}")

# ─────────────────────────────────────────────────────────────
# C3 — Drop stocks with missing 'mu' in metadata
# ─────────────────────────────────────────────────────────────
missing_mu_mask = meta_c2["mu"].isnull()
removed_mu_ids  = meta_c2.index[missing_mu_mask].tolist()
n_removed_mu    = len(removed_mu_ids)

prices_c3 = prices_c2[~missing_mu_mask].copy()
meta_c3   = meta_c2[~missing_mu_mask].copy()

print(f"\n[C3] Stocks removed due to missing 'mu':  {n_removed_mu}")
if removed_mu_ids:
    print(f"     Examples: {removed_mu_ids[:10]}")

# ─────────────────────────────────────────────────────────────
# C4 — Drop stocks with missing 'esg' in metadata
# ─────────────────────────────────────────────────────────────
missing_esg_mask = meta_c3["esg"].isnull()
removed_esg_ids  = meta_c3.index[missing_esg_mask].tolist()
n_removed_esg    = len(removed_esg_ids)

prices_c4 = prices_c3[~missing_esg_mask].copy()
meta_c4   = meta_c3[~missing_esg_mask].copy()

print(f"\n[C4] Stocks removed due to missing 'esg': {n_removed_esg}")
if removed_esg_ids:
    print(f"     Examples: {removed_esg_ids[:10]}")

# ─────────────────────────────────────────────────────────────
# C5 — Inspect missing values in the price matrix
# ─────────────────────────────────────────────────────────────
n_days         = prices_c4.shape[1]
missing_counts = prices_c4.isnull().sum(axis=1)
missing_pct    = (missing_counts / n_days * 100).round(2)

print(f"\n[C5] Missing-price distribution across {len(prices_c4)} stocks "
      f"({n_days} trading days):")
print(f"     Stocks with  0 % missing:          {(missing_pct == 0).sum()}")
print(f"     Stocks with  0–1 % missing:        {((missing_pct > 0) & (missing_pct <= 1)).sum()}")
print(f"     Stocks with  1–5 % missing:        {((missing_pct > 1) & (missing_pct <= 5)).sum()}")
print(f"     Stocks with  5–10 % missing:       {((missing_pct > 5) & (missing_pct <= 10)).sum()}")
print(f"     Stocks with >10 % missing:         {(missing_pct > 10).sum()}")

# ─────────────────────────────────────────────────────────────
# C6 — Remove stocks with > 5 % missing prices
# ─────────────────────────────────────────────────────────────
MISSING_THRESHOLD_PCT = 5.0

excessive_missing_mask = missing_pct > MISSING_THRESHOLD_PCT
removed_missing_ids    = prices_c4.index[excessive_missing_mask].tolist()
n_removed_missing      = len(removed_missing_ids)

prices_clean = prices_c4[~excessive_missing_mask].copy()
meta_clean   = meta_c4.loc[prices_clean.index].copy()

print(f"\n[C6] Threshold: > {MISSING_THRESHOLD_PCT:.0f}% missing prices")
print(f"     Stocks removed:  {n_removed_missing}")
if removed_missing_ids:
    print(f"     Examples: {removed_missing_ids[:10]}")

# ─────────────────────────────────────────────────────────────
# FINAL CLEANING SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL CLEANING SUMMARY")
print("=" * 60)
print(f"  Starting stocks in price file:              {n_start_prices}")
print(f"  − Removed — invalid identifiers [C1]:      {n_removed_invalid}")
print(f"  − Removed — not in metadata [C2]:          {n_removed_prices_no_meta}")
print(f"    (metadata rows not in prices, discarded) {n_removed_meta_no_prices}")
print(f"  − Removed — missing 'mu' [C3]:             {n_removed_mu}")
print(f"  − Removed — missing 'esg' [C4]:            {n_removed_esg}")
print(f"  − Removed — >5% missing prices [C6]:       {n_removed_missing}")
print(f"  {'─'*45}")
print(f"  Final stocks kept (prices & metadata):     {len(prices_clean)}")
print(f"  Trading days retained:                     {prices_clean.shape[1]}")
print("=" * 60)
print("\nClean datasets available as: prices_clean, meta_clean")
print("These will be passed to 02_preprocess_simulated.py automatically")
print("if you import or run this file as a module.\n")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS FOR STEP 02
# ─────────────────────────────────────────────────────────────
os.makedirs("data/clean", exist_ok=True)
meta_clean.to_csv("data/clean/universe_clean.csv")
prices_clean.to_csv("data/clean/prices_clean.csv")
print("Saved: data/clean/universe_clean.csv")
print("Saved: data/clean/prices_clean.csv\n")
