# Variance decomposition and applicability domain

Run from the repository root with Python and the project dependencies installed.

```sh
python code/variance-and-domain/check_variance_and_domain.py --repo-root .
```

Add `--output results/variance-and-domain` to write the recalculated tables. Inputs come from `primary-benchmark/`. Dependencies are NumPy, pandas, SciPy, scikit-learn, threadpoolctl and pymatgen.

## Variance decomposition

The 312-row analysis uses Gaussian restricted maximum likelihood to partition the variance of apparent activation energy by literature source (normalized DOI), material family and residual. The baseline includes an intercept and the total-versus-bulk measurement indicator. M1 adds a source random intercept, M2 adds a family random intercept, and M3 includes both crossed random intercepts. M4 also adjusts for principal components of the 139 composition descriptors. This descriptive adjustment uses cohort-wide median imputation, removal of near-constant descriptors, standardization and full-SVD PCA. It retains the components needed for 90% cumulative explained variance up to a maximum of 20. Each retained score is divided by the square root of its eigenvalue. The reported variance fractions divide each estimated variance by the sum of the source, family and residual variances.

## Applicability domain

For each published P3 and P4 split the code fits descriptor imputation, filtering, scaling and the PCA basis on the training rows. Distances are Euclidean and the PCA scores are unwhitened. Each test row receives its distance to the nearest training row. The row reference is the 95th percentile of the training rows’ leave-one-row-out nearest-neighbor distances. The group reference excludes all training neighbors in the focal row’s connected group before taking the same percentile across training rows. Both use a linear quantile. A test row is inside the primary domain if its distance is at or below the relevant threshold and all its elements occur in training.

The script calculates both references in PCA and standardized descriptor space for ten P3 folds and in PCA space for four P4 family exclusions. The P3 summary joins the archived held-out ExtraTrees errors by split and row. It calculates in-domain and out-of-domain MAE in meV, Spearman correlation with normalized distance, and MAE for the 80% of prediction instances having the smallest normalized distances.

## Verification and scope

The command checks 15 REML variance fractions, 24 pairs of reference thresholds and four P3 error summaries against the study outputs in `expected/`. On the project environment all 807 held-out PCA distances (624 P3 and 183 P4 prediction instances) and all 807 original row-reference domain classifications were independently compared with the archived row-level outputs. Spearman correlations allow an absolute numerical tolerance of 0.00002 for floating-point differences in ranks among near ties.

`reml.py` is the original REML implementation. `kernels.py` collects the original composition compression, model fitting, cluster sampling, distance, threshold, ranking and selection functions. The command uses the released tables to reproduce the point estimates described above. The historical confidence-interval resampling runs, sensitivity analyses outside these point summaries, and predictive model training are outside this command’s scope.
