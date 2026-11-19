# Experimental benchmark and retrospective external validation

Run the commands below from the repository root with the study's Python 3.12 environment and pinned dependencies. The input folders are `primary-benchmark/` and `external-validation/`. Use `--data-root` if they live elsewhere.

## Score the released predictions

```sh
python code/benchmark-external/score.py --data-root .
```

This command recomputes absolute and squared error from each reported prediction and observation. For the main experimental benchmark it averages repeated test losses within each record and applies the original source-group bootstrap. For family exclusions (P4) it reports both record-weighted and equal-family estimates. For external validation it gives the equal-group and record-weighted MAEs and checks both estimates and their bootstrap bounds against `external-validation/model_summary.csv`.

For ExtraTrees, the internal record-weighted MAEs are 106.811728 meV under P1 and 249.624045 meV under P3. For the 34 retrospective external records, the equal-group MAE across eight groups is 133.728899 meV and the record-weighted MAE is 76.150897 meV.

## Refit a published model

```sh
python code/benchmark-external/replay.py --data-root . --split R9_K_FE_P3_r0_f2 --model ExtraTreesRegressor
python code/benchmark-external/replay.py --data-root . --external --model ExtraTreesRegressor
```

The first command refits one internal outer split using its archived inner folds and model grid. The second tunes on the 312 internal records using the archived five source-group folds, refits on all 312 records, and predicts the external set. External predictions are made before the script reads the file containing the external observations and archived predictions.

Choose `DummyRegressor`, `Ridge`, `RBF-SVR`, `ExtraTreesRegressor` or `XGBoost` with `--model`. The script reports the selected parameters and the maximum absolute difference from the public prediction table for primary and external fits. For the secondary benchmark splits it reports the selected parameters and number of predictions. Add `--output predictions.csv` to save one run with row identifiers. Use a new output filename for each run.

## Source and scope

- `original/scripts/benchmark/` contains the archived regressor definitions, tuning grids, training-only preprocessing, feature definitions and benchmark scoring. `original/scripts/A3_2_external_validation/` contains the original external training and scoring stages. The accompanying `audit_core.py` and `A2_applicability_domain/domain.py` provide their original imports.
- `original/scripts/run_benchmark_r9.py` and `summarize_benchmark_r9.py` document the historical revision workflow, including its selective reuse of earlier checkpoints. These historical drivers expect the original project tree and checkpoint records. Use the public entry points above to recompute from the repository files.
- The 34 outer splits for the primary cohort are in `primary-benchmark/benchmark_splits.json`. The 76 archived outer splits for the bulk, total and quality-filtered analyses are in `memberships/secondary_benchmark_splits.json`. Each split contains the training, test and inner-fold identifiers. The recorded five inner folds for model selection before external prediction are in `memberships/external_training_folds.json`.
- `score.py` and `replay.py` adapt the archived functions to the released file layout. The inputs to model fitting are the released 144-column feature matrices. The commands cover the prediction and scoring workflows, starting after literature curation and composition featurisation.
