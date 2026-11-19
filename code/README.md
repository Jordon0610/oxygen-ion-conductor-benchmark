# Analysis code

This directory accompanies *Activation energy prediction and conductivity selection in oxygen ion conductors*. Run the commands below from the repository root, with the four data directories alongside `code/`.

| Analysis | Inputs from the repository | What the entry point computes |
| --- | --- | --- |
| [Experimental benchmark and external validation](benchmark-external/README.md) | `primary-benchmark/`, `external-validation/` | Scores and bootstrap intervals from held-out predictions, or new model fits from the released feature matrices and archived splits. |
| [Variance decomposition and applicability domain](variance-and-domain/README.md) | `primary-benchmark/` | Source, family and residual variance fractions, training-reference distance thresholds and held-out error summaries. |
| [Neutral oxygen vacancies](oxygen-vacancy/README.md) | `oxygen-vacancy/`, `primary-benchmark/` | DFT and experimental transfer scores, nested ExtraTrees DFT refits, and a representative baseline and auxiliary transfer refit. |
| [LSGM conductivity selection](lsgm-conductivity/README.md) | `lsgm-conductivity/` | Conductivity selection, regret, Arrhenius contrasts and substitution-path sensitivity tables from observations and saved model predictions. |

## Start with the released predictions

The following two commands use only the Python standard library (Python 3.11 or later). Use a new or empty directory for LSGM output.

```sh
python code/oxygen-vacancy/check_results.py --data oxygen-vacancy
python code/lsgm-conductivity/reproduce.py --data lsgm-conductivity --output reproduced/lsgm
```

The oxygen-vacancy command reports a 0.376 meV reduction in mean absolute error for the real auxiliary relative to the matched baseline. For the 14 accepted LSGM pairs, the mean conductivity regret is 0.0165 decades using the corrected ExtraTrees predictions and 0.0356 decades using the reported low-temperature activation energy.

## Run the model and statistical analyses

The archived Python package versions are in [requirements-benchmark-lock.txt](requirements-benchmark-lock.txt). The model checks used Python 3.12.14 on macOS. For example, from a fresh Python 3.12 virtual environment:

```sh
python -m pip install -r code/requirements-benchmark-lock.txt
python code/benchmark-external/score.py --data-root .
python code/variance-and-domain/check_variance_and_domain.py --repo-root .
python code/benchmark-external/replay.py --data-root . --split R9_K_FE_P3_r0_f2 --model ExtraTreesRegressor
```

On macOS, importing XGBoost requires an accessible OpenMP runtime (`libomp`). The component guides give commands for external model refitting, the DFT and transfer refits, and details of the inputs and expected values. Commands for newly fitted models are marked separately from commands that score archived predictions. Historical training drivers under `original/` retain the project layout used during the study. The commands documented here use the repository layout.
