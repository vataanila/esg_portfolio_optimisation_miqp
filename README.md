# ESG-Constrained Portfolio Optimisation with MIQP

## Why I built this project

I developed this project to understand how ESG constraints affect portfolio construction when realistic investment rules are added to a classical mean-variance framework. My focus was not only on solving the optimisation problem, but also on building a reproducible pipeline from raw financial data to out-of-sample portfolio evaluation.


## Overview

This project applies constrained portfolio optimisation to two equity universes: a simulated dataset and a Bloomberg EURO STOXX 600 dataset. The goal is to analyse how ESG requirements, sector caps, cardinality constraints and covariance estimation choices affect the efficient frontier and out-of-sample portfolio performance.

## Methodology

At each point on the efficient frontier the following MIQP is solved:

```
maximise    w' mu
subject to:
  w' Sigma w  <= vol_cap^2          volatility cap (swept for frontier)
  sum(w_i)     = 1                  fully invested
  w' esg      >= esg_floor          ESG floor (omitted for unconstrained)
  w_i         <= W_MAX * z_i        max weight 20% per stock
  w_i         >= W_MIN * z_i        min weight 1% per stock
  sum(z_i)    <= K_MAX              at most 100 holdings
  sum(z_i)    >= K_MIN              at least 50 holdings
  sum_{j in s} w_j <= 0.25          sector cap 25%, main pipeline only
  w_i         >= 0                  long-only
  z_i in {0, 1}                     binary stock selection
```

where Sigma is one of three annualised covariance matrices (Sample, Ledoit-Wolf, or OAS)
and mu is the expected-return vector:

- Simulated dataset: expected returns provided in the metadata file.
- Bloomberg dataset: 3-year trailing winsorised empirical mean (mu_trailing_winsor), main specification.

The problem is solved by Gurobi 13 via cvxpy. The MIP optimality gap is set to 0.01% with
a 120-second time limit per solve.

Three covariance estimators are compared:

- Sample: classical sample covariance
- Ledoit-Wolf: analytical shrinkage towards scaled identity (Ledoit and Wolf, 2004)
- OAS: Oracle Approximating Shrinkage (Chen, Wiesel, Eldar and Hero, 2010)

The efficient frontier is traced by sweeping the volatility cap over 30 geometric points
(np.geomspace), giving denser coverage in the curved low-volatility region. The fixed vol cap
used for the ESG sweep and baseline portfolio is the max-Sharpe point in the central 60% of
the feasible unconstrained Ledoit-Wolf frontier, chosen to avoid boundary effects.

The ESG floor sweep covers floors [40, 45, 50, 55, 60, 65] on a common 0-100 scale for
both datasets.

## Data

The analysis uses two independent datasets.

**Simulated dataset**: a universe of approximately 2,300 synthetic stocks with
expected returns, ESG scores on a 0-100 scale, and synthetic price history (~5,000 daily
observations).

**Bloomberg dataset**: EURO STOXX 600 daily closing prices (2016-2025) and Bloomberg ESG
scores (BESG, native 1-10 scale). BESG scores are mapped to the common 0-100 optimisation
scale as follows:

```
esg = (BESG - 1) / 9 * 100
```

Raw Bloomberg data are not included due to licensing restrictions. The simulated raw files
are also not committed. The folder structure is kept with .gitkeep files. See
[docs/data_note.md](docs/data_note.md) for the full data note.

| Dataset | File | Description |
|---|---|---|
| Simulated prices | data/raw/stockprices2500.csv | 2,500 synthetic stocks, daily prices |
| Simulated metadata | data/raw/shares2500_fixed.csv | FICO returns, ESG (0-100), sector |
| Bloomberg prices | data/raw/SXXP_FINAL.csv | EURO STOXX 600 closes, Bloomberg CSV format |
| Bloomberg metadata | data/raw/TICKERS.csv | Ticker, BESG (1-10), sector, country |

## Pipeline

Run all scripts from the project root directory in the order shown.

**Simulated dataset:**

```bash
python src/01_load_simulated.py         # structural cleaning -> data/clean/
python src/02_preprocess_simulated.py   # returns, covariance, ESG transform
python src/03_optimize_simulated.py     # MIQP frontier (3 estimators)
python src/03b_esg_sweep_simulated.py   # ESG floor sweep and baseline portfolio
python src/04_oos_simulated.py          # out-of-sample analysis
python src/05_bootstrap_simulated.py    # bootstrap Sharpe confidence intervals
```

**Bloomberg dataset:**

```bash
python src/06_load_bloomberg.py         # structural cleaning -> data/clean_bloomberg/
python src/07_preprocess_bloomberg.py   # returns, covariance, ESG mapping
python src/08_optimize_bloomberg.py     # MIQP frontier
python src/09_oos_bloomberg.py          # out-of-sample analysis
python src/10_bootstrap_bloomberg.py    # bootstrap Sharpe confidence intervals
```

**Cross-dataset comparison:**

```bash
python src/11_covariance_comparison.py  # 3 estimators x 2 datasets
```

## Out-of-sample validation

The out-of-sample analysis uses a 70/30 chronological train/test split. Covariance matrices
are re-estimated on the training window only, with no look-ahead into the test period. For
the Bloomberg OOS scripts (09 and 10), mu is also re-estimated on the training window using
a trailing mean winsorised at p1/p99. Weights are frozen (optimised on the train-window
covariance) and applied to test-period returns to obtain realised performance metrics.

The IS-OOS Sharpe gap is used as a simple diagnostic for potential overfitting. It helps assess whether the portfolio performs materially worse out of sample than it does in sample.

## Bootstrap inference

Statistical uncertainty around OOS Sharpe ratios is quantified via stationary block bootstrap
(Politis and Romano, 1994) with B = 1,000 resamples and a fixed block length of 21 trading
days, approximately one trading month. The bootstrap resamples OOS daily portfolio returns
and recomputes the annualised Sharpe ratio in each resample. If the 95% confidence interval includes zero, the evidence for a positive OOS Sharpe ratio is weak at conventional significance levels.

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/vataanila/esg_portfolio_optimisation_miqp.git
cd esg_portfolio_optimisation_miqp

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Activate Gurobi licence
#    Academic licences: https://www.gurobi.com/academia/academic-program-and-licenses/
grbgetkey <YOUR-LICENCE-KEY>

# 5. Verify the solver
python check_gurobi.py

# 6. Run the pipeline in order (see Pipeline section above)
python src/01_load_simulated.py
# ...
```

Gurobi requires a valid licence to solve MIQP problems at this scale (~2,300 binary
variables). Free academic licences are available at gurobi.com.

## Limitations

- This is a research project. The results are not a live trading strategy and should not
  be read as investment advice.
- Expected return estimates are noisy. The trailing mean for Bloomberg is particularly
  sensitive to the estimation window.
- MIQP results depend on constraint settings. Small changes to K_MAX, W_MAX, or sector
  caps can shift the frontier materially.
- Bloomberg data cannot be redistributed, so the Bloomberg part of the pipeline cannot be
  reproduced without a Bloomberg terminal subscription.
- Bootstrap confidence intervals are wide relative to the point estimates, reflecting
  genuine statistical uncertainty in OOS Sharpe ratios.

## Project structure

```
.
|-- src/
|   |-- 01_load_simulated.py           structural cleaning, simulated
|   |-- 02_preprocess_simulated.py     returns, covariance, ESG, simulated
|   |-- 03_optimize_simulated.py       MIQP frontier, simulated
|   |-- 03b_esg_sweep_simulated.py     ESG floor sweep, simulated
|   |-- 04_oos_simulated.py            out-of-sample analysis, simulated
|   |-- 05_bootstrap_simulated.py      bootstrap Sharpe CIs, simulated
|   |-- 06_load_bloomberg.py           structural cleaning, Bloomberg
|   |-- 07_preprocess_bloomberg.py     returns, covariance, ESG, Bloomberg
|   |-- 08_optimize_bloomberg.py       MIQP frontier, Bloomberg
|   |-- 09_oos_bloomberg.py            out-of-sample analysis, Bloomberg
|   |-- 10_bootstrap_bloomberg.py      bootstrap Sharpe CIs, Bloomberg
|   `-- 11_covariance_comparison.py    cross-dataset estimator comparison
|
|-- docs/
|   |-- methodology.md                 full methodology write-up
|   |-- pipeline_overview.md           data flow and pipeline diagram
|   `-- data_note.md                   data sources, access, and licensing
|
|-- data/
|   |-- raw/                           input data, not committed
|   |-- clean/                         cleaned simulated data (generated)
|   |-- clean_bloomberg/               cleaned Bloomberg data (generated)
|   |-- results/                       simulated in-sample results (generated)
|   |-- results_simulated/             simulated OOS and bootstrap results (generated)
|   `-- results_bloomberg/             Bloomberg results (generated)
|
|-- figures/                           generated PNG figures (generated)
|-- check_gurobi.py                    Gurobi licence diagnostic
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Author

Anila Vata
MSc Finance, University of Pavia

## References

- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.
- Politis, D.N. and Romano, J.P. (1994). The stationary bootstrap. Journal of the American
  Statistical Association, 89(428), 1303-1313.
- Chen, Y., Wiesel, A., Eldar, Y.C. and Hero, A.O. (2010). Shrinkage algorithms for MMSE
  covariance estimation. IEEE Transactions on Signal Processing, 58(10), 5016-5029.
