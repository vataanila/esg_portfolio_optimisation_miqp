# Methodology

## Research Question

Does imposing a weighted-average ESG floor on an institutional equity portfolio optimised via
Mixed-Integer Quadratic Programming (MIQP) lead to a statistically significant loss of
risk-adjusted return, and does the answer depend on the choice of covariance estimator or
the dataset used?

---

## 1. Portfolio Optimisation Model (MIQP)

At each point on the efficient frontier the following MIQP is solved:

```
maximise    w' mu
subject to:
  w' Sigma w  <= sigma_cap^2         volatility cap
  sum(w_i)     = 1                   fully invested
  w' esg      >= esg_floor           ESG floor (omitted for unconstrained runs)
  w_i         <= W_MAX * z_i         W_MAX = 0.20
  w_i         >= W_MIN * z_i         W_MIN = 0.01
  sum(z_i)    <= K_MAX               K_MAX = 100
  sum(z_i)    >= K_MIN               K_MIN = 50
  sum_{j in s} w_j <= 0.25          sector cap, main pipeline
  w_i         >= 0
  z_i in {0, 1}
```

The problem is a Mixed-Integer Quadratic Programme because of the binary selection variables
z_i. It is solved by Gurobi 13 (academic licence) via cvxpy. The MIP optimality gap is set
to 0.01% (1e-4) with a 120-second time limit per solve.

### Cardinality constraints

Cardinality constraints (K_MIN, K_MAX) are linked to continuous weights via the big-M
formulation:
- w_i <= W_MAX * z_i forces z_i = 1 whenever stock i is held.
- w_i >= W_MIN * z_i prevents token positions below the minimum weight.

### ESG constraint

The ESG constraint w' esg >= esg_floor is a linear constraint on the weighted-average ESG
score of the portfolio. Both datasets use a common 0-100 scale:
- Simulated: ESG scores are provided directly on the 0-100 scale.
- Bloomberg: BESG scores (native scale 1-10) are mapped to 0-100 via
  (BESG - 1) / 9 * 100. The inverse formula is BESG = f / 100 * 9 + 1 for a 0-100 floor f.

---

## 2. Covariance Estimators

Three covariance matrices are compared at each stage of the pipeline:

| Estimator | Description | Implementation |
|---|---|---|
| Sample | Classical sample covariance | np.cov(R, rowvar=False) |
| Ledoit-Wolf | Analytical shrinkage towards scaled identity (Ledoit and Wolf, 2004) | sklearn.covariance.LedoitWolf |
| OAS | Oracle Approximating Shrinkage (Chen et al., 2010) | sklearn.covariance.OAS |

All covariance matrices are annualised by multiplying by 252 trading days, then symmetrised
and regularised with a small diagonal ridge (+ 1e-8 * I) for numerical stability.

---

## 3. Efficient Frontier Construction

The frontier is traced by sweeping the volatility cap sigma_cap over a geometric grid
(np.geomspace) of 30 points. Geometric spacing gives denser coverage in the curved
low-volatility region where the frontier bends most sharply.

Vol cap ranges:
- Simulated: [0.30, 3.00] (inflated synthetic volatility scale)
- Bloomberg: [0.10, 0.80] (real equity annual vol; universe mean ~30%, max ~72%)

Fixed vol cap selection (ESG sweep and baseline portfolio):
The vol cap used for the ESG floor sensitivity analysis is chosen data-driven rather than by
an arbitrary percentile rule. Specifically: the max-Sharpe point in the central 60% of the
feasible unconstrained Ledoit-Wolf frontier (trimming the bottom 20% and top 20% of the
feasible vol range). This selects a well-diversified portfolio in the curved interior of the
frontier, away from boundary effects.

---

## 4. Expected Return Specification

| Dataset | Main specification | Robustness |
|---|---|---|
| Simulated | mu_native - FICO metadata mu | mu_winsor - winsorised at p1/p99 |
| Bloomberg | mu_trailing_winsor - 3-year trailing winsorised empirical mean | mu_winsor - full-sample winsorised |

Bloomberg has no native "given mu" equivalent. The main specification uses a 3-year trailing
window to reduce look-ahead in the out-of-sample evaluation. Winsorisation at the
1st/99th percentile is applied to reduce sensitivity to outlier stocks.

---

## 5. ESG Floor Sensitivity Analysis

The ESG floor sweep is run at the fixed vol cap (see Section 3) for all three covariance
estimators and a range of floors:
- Simulated: floors [40, 45, 50, 55, 60, 65] on the 0-100 scale
- Bloomberg: floors [40, 45, 50, 55, 60, 65] on the 0-100 scale

The return cost of each floor is defined as the reduction in annualised expected return
relative to the unconstrained portfolio at the same vol cap. The baseline portfolio uses
Ledoit-Wolf covariance with ESG floor = 55.

---

## 6. Out-of-Sample Evaluation

The out-of-sample analysis uses a 70/30 chronological train/test split:

1. All covariance matrices are re-estimated on the train window only - no look-ahead into
   the test period.
2. For the Bloomberg OOS scripts (09 and 10), mu is also re-estimated on the training window
   using a trailing mean winsorised at p1/p99. This gives a theoretically clean OOS
   evaluation with no look-ahead in either the covariance or the return estimate.
3. For the simulated dataset, the FICO metadata mu is used as-is (it is not derived from the
   observed price history, so no look-ahead applies).
4. Weights are frozen (optimised on train-window Sigma) and applied to the test-period
   returns.
5. Realised performance (return, volatility, Sharpe) is computed from the frozen weights
   applied to test returns.

The IS-OOS Sharpe gap measures the degree to which in-sample optimisation overfits the
estimated covariance structure.

---

## 7. Bootstrap Inference

Statistical uncertainty around OOS Sharpe ratios is quantified via stationary block bootstrap
(Politis and Romano, 1994):

- B = 1,000 bootstrap resamples per portfolio
- Block length: optimal block length selected by the arch library's automatic method
- Confidence level: 95% (two-sided)
- The bootstrap resamples OOS daily returns (preserving autocorrelation structure) and
  recomputes the annualised Sharpe ratio in each resample.

A 95% confidence interval overlapping zero indicates that the Sharpe ratio is not
statistically distinguishable from zero at the 5% level.

---

## 8. Covariance Estimator Comparison (Script 11)

Script 11_covariance_comparison.py runs a standalone comparison with a reduced cardinality
cap (K_MAX = 10, K_MIN = 5) to amplify the sensitivity of portfolio construction to the
choice of estimator. Smaller portfolios show a larger spread in frontier metrics across
estimators, making model risk more visible. The sector cap is not applied in this step to
keep the model lean.

---

## References

- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411.
- Politis, D.N. and Romano, J.P. (1994). The stationary bootstrap. Journal of the American
  Statistical Association, 89(428), 1303-1313.
- Chen, Y., Wiesel, A., Eldar, Y.C. and Hero, A.O. (2010). Shrinkage algorithms for MMSE
  covariance estimation. IEEE Transactions on Signal Processing, 58(10), 5016-5029.
