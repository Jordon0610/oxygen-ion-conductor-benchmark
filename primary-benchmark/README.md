# Primary benchmark data for author review

Study: Activation energy prediction and conductivity selection in oxygen ion conductors

This extract covers the primary experimental activation-energy benchmark (K_FE).
It contains the main 312-record cohort and P1–P4 benchmark outputs. Other
manuscript analyses, including vacancy transfer, external evaluation and LSGM
conductivity selection, are outside this extract.

## Files

- parent_membership.csv lists identifiers, inclusion flags and assigned connected
  groups for all 483 parent records. Groups were defined using the parent population.
- experimental_cohort_312.csv gives the 312 included records (155 bulk, 157 total),
  covering 47 connected groups. Ea is the experimental apparent activation energy in
  meV. dataset_csv_line is the line in the original CSV, counting its header.
  The Formula and normalized_formula columns reproduce the curated table.
- benchmark_features_312.csv contains model inputs for those 312 records. Join on
  row_id. feature_dictionary.csv defines the 144 feature columns.
- benchmark_splits.json contains 34 outer splits, with their original identifiers
  and ordered train, test, purge and inner training/validation memberships.
  P1, P2 and P3 have ten outer splits each, and P4 has four.
- held_out_predictions.csv contains 10,275 held-out predictions across five
  estimators and the 34 outer splits. Energy values and absolute errors are in
  meV. Squared errors are in square meV. Predictions were taken from archived
  results without refitting.

## Source and changes

The upstream compilation is by Seong-Hoon Jang, Shin Kiyohara, Hitoshi Takamura
and Yu Kumagai (2026), Charting the Landscape of Oxygen Ion Conductors: A 60-Year
Dataset, version 1. https://doi.org/10.5281/zenodo.18947543
Upstream dataset licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The project added composition curation, component classification, cohort flags,
group membership, feature construction, split assignments and model outputs.
The archived analysis uses revised nominal composition and oxygen metadata for
R0023 and excludes R0053 from the primary Ea target. The parent row IDs preserve
links to the compilation. Values and assignments were copied from the project's
archived benchmark files. Only rows and fields relevant to this extract were
selected. No analyses were rerun.
