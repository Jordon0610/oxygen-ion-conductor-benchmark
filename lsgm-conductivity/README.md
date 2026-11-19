# Conductivity selection in strontium- and magnesium-doped lanthanum gallate (LSGM)

Data accompanying *Activation energy prediction and conductivity selection in oxygen ion conductors*.

## Contents

- observations contains source-linked numerical observations, references to the original tables and figures, recipe and phase information, measurement components, and the fitted-parameter and pair registers. See sources.csv for the bibliographic source codes.
- composition_contrasts contains archived composition contrasts and predictions, including historical checks. The figure numbers embedded in these filenames refer to the archived analysis. The files on fixed compositions, processing and density document the corresponding physical comparisons in the paper.
- conductivity_selection contains saved source-excluded predictions, pairwise choices and regret at 800 °C, repeat and block summaries, and source-local comparisons. saved_source_excluded_predictions.csv gives two held-out repeats for each of 38 parents and three estimators. The principal rule uses ExtraTrees.
- substitution_paths contains the core pair and block records, the wider candidate set, path summaries, and candidate-omission and conductivity-perturbation results. In the robustness tables, a candidate is removed along with its incident pairs. The named primary and without_R0024 subsets can be identified in the core tables.
- arrhenius_parameters links activation energy and log-prefactor within each reported fit. The LT and HT branches retain their distinct identifiers.

## Main comparison

Huang, Tichy and Goodenough (1998) report the bulk conductivity at 800 °C and low-temperature activation energies for the source-defined primary comparison. The DOI is https://doi.org/10.1111/j.1151-2916.1998.tb02662.x. Conductivities come directly from the source table.

In substitution_paths/core/pair_repeat.csv, select subset primary and model ExtraTreesRegressor. Each of two saved repeats contains the same 14 pairs of 11 parents, in 10 substitution-path blocks. M selects the lower predicted activation energy, E the lower reported low-temperature activation energy, and U gives each candidate equal probability. The full-set loss for a tie is the expected loss under equal choice. Pair regret is the base-10 logarithm of the better conductivity divided by the selected conductivity, in decades. Within each repeat, average pair regrets inside each block, give blocks equal weight, and average the two repeats. The resulting regrets are 0.016465 (M), 0.035613 (E), and 0.026904 (U).

The expanded Huang comparison includes 38 parents and 88 pairs and overlaps the primary set. Shared candidates and pairs retain their identifiers. Path-level results distinguish comparisons with E from comparisons with U. The fixed-Sr comparison improved on E in both candidate sets, while the differences for fixed Mg and fixed total Sr+Mg changed sign. In the 14 core comparisons, the shifts in activation energy and log-prefactor share the same sign in 13 pairs in each of the two fitted branches.

## Sources and units

sources.csv identifies the original reports for the literature observations. Numerical source locators are retained alongside the derived tables. Huang & Petric (1996) and Stevenson et al. (1997) provide additional source-local comparisons; Huang, Feng & Goodenough (1996) provide processing context. The parent identifiers used with the saved models can be cross-referenced with the primary-benchmark folder. Activation-energy columns specify eV or meV and the directly tabulated conductivity is in S/cm. All values and model predictions come from the archived analyses.
