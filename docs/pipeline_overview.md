# Pipeline Overview

## Data Flow

Input files in data/raw/ feed two independent pipelines:

- Simulated pipeline: stockprices2500.csv, shares2500_fixed.csv
- Bloomberg pipeline: SXXP_FINAL.csv, TICKERS.csv

Both pipelines write intermediate outputs to data/clean/ or data/clean_bloomberg/, then
results to data/results/, data/results_simulated/, or data/results_bloomberg/. All figures
are written to figures/.

## Simulated Dataset Pipeline

```
01_load_simulated.py
    Input : data/raw/stockprices2500.csv
            data/raw/shares2500_fixed.csv
    Output: data/clean/prices_clean.csv
            data/clean/meta_clean.csv

02_preprocess_simulated.py
    Input : data/clean/prices_clean.csv
            data/clean/meta_clean.csv
    Output: data/clean/log_returns.csv
            data/clean/meta_preprocessed.csv
            data/clean/sigma_sample_annual.npy
            data/clean/sigma_lw_annual.npy
            data/clean/sigma_oas_annual.npy

03_optimize_simulated.py
    Input : data/clean/meta_preprocessed.csv
            data/clean/sigma_*.npy
    Output: data/results/frontier_*.csv
            data/results/simulated_selected_vol_cap.txt
            data/results/cov_comparison.csv
            data/results/optimisation_log.txt
            figures/step3_frontier.png
            figures/step3_cov_comparison.png

03b_esg_sweep_simulated.py
    Input : data/clean/meta_preprocessed.csv
            data/clean/sigma_*.npy
            data/results/simulated_selected_vol_cap.txt
    Output: data/results/esg_sensitivity.csv
            data/results/baseline_weights.csv
            figures/step3_esg_sensitivity.png

04_oos_simulated.py
    Input : data/clean/meta_preprocessed.csv
            data/clean/log_returns.csv
            data/clean/sigma_*.npy
            data/results/simulated_selected_vol_cap.txt
    Output: data/results_simulated/simulated_oos_fixedvol_summary.csv
            data/results_simulated/simulated_oos_frontier_main.csv
            data/results_simulated/simulated_oos_log.txt
            figures/step4a_*.png
            figures/step4b_*.png

05_bootstrap_simulated.py
    Input : data/clean/meta_preprocessed.csv
            data/clean/log_returns.csv
            data/clean/sigma_*.npy
            data/results/simulated_selected_vol_cap.txt
    Output: data/results_simulated/simulated_bootstrap_summary.csv
            data/results_simulated/simulated_bootstrap_log.txt
            figures/step5_bootstrap_ci.png
```

## Bloomberg Dataset Pipeline

```
06_load_bloomberg.py
    Input : data/raw/SXXP_FINAL.csv
            data/raw/TICKERS.csv
    Output: data/clean_bloomberg/bloomberg_meta_clean.csv
            data/clean_bloomberg/bloomberg_prices_clean.csv

07_preprocess_bloomberg.py
    Input : data/clean_bloomberg/bloomberg_meta_clean.csv
            data/clean_bloomberg/bloomberg_prices_clean.csv
    Output: data/clean_bloomberg/bloomberg_meta_analytical.csv
            data/clean_bloomberg/bloomberg_returns.csv
            data/clean_bloomberg/bloomberg_Sigma_sample.csv
            data/clean_bloomberg/bloomberg_Sigma_lw.csv
            data/clean_bloomberg/bloomberg_Sigma_oas.csv
            data/clean_bloomberg/bloomberg_preproc_log.txt
            figures/bloomberg_*.png

08_optimize_bloomberg.py
    Input : data/clean_bloomberg/bloomberg_meta_analytical.csv
            data/clean_bloomberg/bloomberg_Sigma_*.csv
    Output: data/results_bloomberg/bloomberg_frontier_*.csv
            data/results_bloomberg/bloomberg_selected_vol_cap.txt
            data/results_bloomberg/bloomberg_esg_sensitivity.csv
            data/results_bloomberg/bloomberg_baseline_weights.csv
            data/results_bloomberg/bloomberg_cov_comparison.csv
            data/results_bloomberg/bloomberg_optimisation_log.txt
            figures/step7_*.png

09_oos_bloomberg.py
    Input : data/clean_bloomberg/bloomberg_meta_analytical.csv
            data/clean_bloomberg/bloomberg_returns.csv
            data/results_bloomberg/bloomberg_selected_vol_cap.txt
    Output: data/results_bloomberg/bloomberg_oos_fixedvol_summary.csv
            data/results_bloomberg/bloomberg_oos_frontier_main.csv
            data/results_bloomberg/bloomberg_oos_log.txt
            figures/step8a_*.png
            figures/step8b_*.png

10_bootstrap_bloomberg.py
    Input : data/clean_bloomberg/bloomberg_meta_analytical.csv
            data/clean_bloomberg/bloomberg_returns.csv
            data/results_bloomberg/bloomberg_selected_vol_cap.txt
    Output: data/results_bloomberg/bloomberg_bootstrap_summary.csv
            data/results_bloomberg/bloomberg_bootstrap_log.txt
            figures/step9_bootstrap_ci.png
```

## Cross-Dataset Comparison

```
11_covariance_comparison.py
    Input : data/clean/meta_preprocessed.csv
            data/clean/sigma_*.npy
            data/clean/log_returns.csv
            data/clean_bloomberg/bloomberg_meta_analytical.csv
            data/clean_bloomberg/bloomberg_Sigma_*.csv
            data/clean_bloomberg/bloomberg_returns.csv
    Output: data/results/covariance_comparison_results.csv
            data/results/covariance_comparison_table.csv
            figures/fig_cov_comparison_simulated.png
            figures/fig_cov_comparison_bloomberg.png
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frontier spacing | np.geomspace (geometric) | Denser coverage in the curved low-vol region |
| Fixed vol cap | Max-Sharpe in central 60% of feasible LW frontier | Data-driven; avoids arbitrary percentile; trims boundary instability |
| ESG scale | Common 0-100 for both datasets | Enables direct comparison; Bloomberg BESG mapped via (BESG-1)/9*100 |
| Train/test split | 70% train, 30% test (chronological) | Standard for time-series OOS evaluation |
| Covariance for OOS | Re-estimated on train window only | No look-ahead bias |
| Bloomberg OOS mu | Re-estimated on train window, winsorised p1/p99 | No look-ahead bias |
| Bootstrap block length | Automatic (arch library) | Adapts to the autocorrelation structure of returns |
