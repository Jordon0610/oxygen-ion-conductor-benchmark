# LSGM conductivity selection and Arrhenius parameter analysis

Use Python 3.11 or later. All six calculation scripts use the Python standard library. From the repository root, run the following command with an empty or new output directory.

```shell
python code/lsgm-conductivity/reproduce.py --data lsgm-conductivity --output reproduced/lsgm
```

The workflow starts from the source observations, comparison registries, matched parent predictions and saved source-excluded model predictions supplied in the repository. It computes the decision rules and conductivity regret, the source-linked Arrhenius parameter contrasts, the core and wider substitution-path results, and the leave-one-parent and conductivity perturbation analyses. The 14 accepted Huang comparisons are nested within the wider set of 88 pairs. The workflow retains exact prediction ties and the published comparison weights.

The output contains `conductivity_selection/`, `arrhenius_parameters/` and `substitution_paths/`. The path and robustness calculations use the decision tables produced earlier in the same run. The reproduced primary ExtraTrees mean regret is 0.0164653684 decades for predicted activation energy and 0.0356133232 decades for reported activation energy.

## Relationship to the study code

The decision and parameter analysis scripts are the original analysis files. The wrapper maps the repository filenames to their original input names in a temporary directory. The four path and robustness scripts retain the original scientific calculations. The distribution removes checks against private preliminary drafts, retains checks against the released source records, and computes the robustness input digest directly from the regenerated table. It includes no preliminary drafts or reference fixtures. Plotting exports and upstream model fitting are outside this entry point.

In a release check using Python 3.13, 42 regenerated CSV files matched the repository CSV files byte for byte, comprising 10 decision tables, 6 Arrhenius tables, and 26 path and robustness tables.
