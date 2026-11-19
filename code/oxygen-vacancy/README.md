# Neutral oxygen-vacancy modelling and transfer to experimental activation energy

## Run the result check

Run the following command from the repository root.

```sh
python3 code/oxygen-vacancy/check_results.py --data oxygen-vacancy
```

This command uses Python 3.8 or later and the standard library. It reads the published CSV or CSV.GZ files and calculates the S2 mean absolute errors from out-of-fold predictions for all five models on the 824 DFT formulas. For experimental transfer it averages the two held-out absolute errors per record and then averages across the 191 matched records. It checks the baseline and real auxiliary comparison, along with the means of the 20 shuffled and 20 random controls, against the archived contrast tables.

The ExtraTrees and training-median DFT errors are 0.362 and 1.217 eV. The experimental baseline and real-auxiliary errors are 200.83697 and 200.46107 meV. Their difference is 0.37591 meV.

## Refit models from the released tables

Run these commands from the repository root in a Python environment with NumPy, pandas, scikit-learn and XGBoost. The archived model modules import XGBoost, including its OpenMP runtime, when fitting ExtraTrees.

```sh
python3 code/oxygen-vacancy/refit_dft.py --data oxygen-vacancy --fold 0
python3 code/oxygen-vacancy/refit_transfer.py --repo .
```

The DFT command uses the released 139 composition features, host targets and nested splits for population B. It repeats hyperparameter selection within the three inner folds, refits the ExtraTrees model and compares the held-out predictions with the released values. Fold 0 has 160 held-out formulas and an MAE of 0.357215088 eV. Use `--fold all` to fit the five outer folds and calculate the pooled MAE across 824 formulas.

The transfer command uses the `primary-benchmark/` and `oxygen-vacancy/` data folders. It refits the baseline and real minimum-vacancy-energy arms for repeat 0, fold 0 of P3 with exact cleaning. It uses the archived fold membership, selected downstream parameters and precomputed auxiliary values, then compares the 18 held-out predictions from each model with the released predictions. The original procedure for selecting these parameters and generating the auxiliary values is documented in the source below.

## Original modelling source

The original modelling files are preserved under `original/scripts/`. The `phase3a_1/` files cover vacancy-label preparation, nested split construction, DFT fitting and DFT accuracy estimates. The `phase3a_2/analyze.py` module implements the descriptor-based domain and training exclusions. The `phase3b/` files implement auxiliary fitting, the matched downstream model, controls, and paired score estimation. `run_transfer_r9.py` is the revision-specific archived entry point. It reuses earlier DFT model-selection records.

The DFT model grids, fixed settings and recorded package versions are in `config/dft_settings.json`, an excerpt from the original analysis plan. `config/transfer_settings.json` contains the archived transfer plan. These files document the scientific settings.

The archived training entry points expect the original acquisition, curation and results directory layout, the pre-fit manifests and the intermediate model-selection caches. The result check and the two refit commands above use the released tables. Confidence intervals are supplied with the archived results; the check recalculates the point estimates.
