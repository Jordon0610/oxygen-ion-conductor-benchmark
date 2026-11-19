# Retrospective external validation

This dataset accompanies *Activation energy prediction and conductivity selection in oxygen ion conductors*. It covers 34 literature measurements in eight source-connected groups. The DOI on each record identifies its experimental source.

- external_records_and_predictions.csv contains study record identifiers, source DOIs, compositions, conductivity components, temperature intervals as reported, the observed activation energies, and predictions and absolute errors for five models. The combined_connected_group_id column identifies the groups used in the reported analysis. Energy and absolute error are in meV.
- external_features.csv contains the 144 input features for each external record. Join using external_record_id. The feature definitions are in the primary-benchmark folder.
- group_level_results.csv gives errors for each model in each of the eight groups.
- model_summary.csv gives the archived model-level results and uncertainty intervals.

For ExtraTrees the mean of the eight group MAEs is 133.7 meV. Giving every record equal weight yields 76.2 meV. The corresponding group-weighted MAE for the training-median baseline is 230.1 meV. The small number of groups limits the precision of group-level estimates.

The files were selected from the archived five-model evaluation. The row and group errors were checked against the archived summaries. No models were retrained.
