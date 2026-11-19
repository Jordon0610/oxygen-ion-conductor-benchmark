# Neutral oxygen-vacancy auxiliary analysis

Data accompanying *Activation energy prediction and conductivity selection in oxygen ion conductors*.

## Vacancy formation energy

The dft folder contains the site-level labels, formula-level targets, 139 composition features, feature definitions, validation splits, held-out predictions and model summaries. The published site table defines the 1,745 neutral vacancy labels used in population B (824 formulas and 679 chemical systems). Alternative quality filters are identified by the population column. Formation energies are in eV. The primary target is the minimum across retained sites for a mapped formula or host. Protocol S2 holds out non-oxygen chemical systems. In S2 the ExtraTrees MAE for the primary target is 0.362 eV and the training-median baseline MAE is 1.217 eV.

## Transfer to experimental activation energy

The transfer folder contains the DFT exclusion masks for each split, experimental eligibility and train/test memberships, real and control auxiliary values, held-out predictions, record-level losses averaged over repeats, and model and contrast summaries. Use row_id to join experimental rows to the primary-benchmark dataset. Activation energies and MAEs in the transfer comparison are in meV. Auxiliary energies are in eV as indicated by auxiliary_unit. Random controls use the unit shown in that field.

For the reported primary comparison, select context primary_exact_ET and contrast baseline_minus_real in paired_transfer_contrasts.csv. Equivalently, select protocol P3, cleaning exact_clean and downstream_model ExtraTreesRegressor in outer_fold_predictions.csv. Compare baseline with real_Efmin. Calculate absolute errors, average over the two held-out predictions for each record and take the mean across the same 191 records in 32 source-connected groups. The baseline MAE is 200.83697 meV and the auxiliary MAE is 200.46107 meV. The estimated reduction is 0.37591 meV (95% group-bootstrap interval -7.04 to 6.05 meV). Other arms and contexts retain their archive identifiers.

## Data sources

Vacancy data were derived from Yu Kumagai, Naoki Tsunoda, Akira Takahashi and Fumiyasu Oba, *Insights into oxygen vacancies from high-throughput first-principles calculations*, Physical Review Materials 5, 123803 (2021). https://doi.org/10.1103/PhysRevMaterials.5.123803
Source data version: https://github.com/kumagai-group/oxygen_vacancies_db/blob/e310353ecbf698499c85a5ec5416b76e82dcecfd/vacancy_formation_energy_ml/charge0.csv
Upstream data licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The experimental data originate from Seong-Hoon Jang, Shin Kiyohara, Hitoshi Takamura and Yu Kumagai, *Charting the Landscape of Oxygen Ion Conductors: A 60-Year Dataset*, v1 (2026), https://doi.org/10.5281/zenodo.18947543 (CC BY 4.0). Source filtering, target aggregation, group assignments, features, model outputs and comparisons are study-specific. This folder collects archived numerical results. No models were retrained during packaging.

## Reading compressed data files

Large CSV and JSON files use gzip compression. To read a CSV directly in Python, pass its .csv.gz path to pandas.read_csv. To read compressed JSON, open the .json.gz file with gzip.open in text mode and use json.load. Decompressing a file restores the original data bytes.
