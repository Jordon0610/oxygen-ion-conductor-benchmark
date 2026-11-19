"""Refit the primary S2 vacancy-formation-energy model with the released tables."""
import argparse
import gzip
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGET = "Ef_min_sourcefiltered_q0"
MODEL = "ExtraTreesRegressor"
TOLERANCE_EV = 1e-6


def data_path(folder, name):
    for candidate in (folder / name, folder / (name + ".gz")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {name} (plain or gzip-compressed) in {folder}")


def read_splits(folder):
    path = data_path(folder, "splits.json")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)["B"]["S2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to the repository's oxygen-vacancy data folder")
    parser.add_argument("--fold", choices=["0", "1", "2", "3", "4", "all"], default="0",
                        help="Outer fold to refit; use all to compute the overall MAE")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    # The archived modules use sibling imports when run as scripts.
    sys.path.insert(0, str(HERE / "original" / "scripts" / "phase3a_1"))
    from engine import fit_predict_dft, select_hyperparameters

    dft = args.data / "dft"
    features = pd.read_csv(data_path(dft, "composition_features.csv"), index_col="formula_id",
                           float_precision="round_trip")
    hosts = pd.read_csv(data_path(dft, "source_clean_host_targets.csv"), float_precision="round_trip")
    target = hosts[hosts["population"] == "B"].set_index("formula_id")[TARGET]
    archived = pd.read_csv(data_path(dft, "held_out_predictions.csv"), float_precision="round_trip")
    archived = archived[(archived["population"] == "B") & (archived["protocol"] == "S2")
                        & (archived["target"] == TARGET) & (archived["model"] == MODEL)]
    settings = json.loads((HERE / "config" / "dft_settings.json").read_text(encoding="utf-8"))
    absolute_errors = []

    for fold in read_splits(dft):
        fold_number = fold["fold"]
        if args.fold != "all" and str(fold_number) != args.fold:
            continue
        train, test = fold["train"], fold["test"]
        inner = [(item["train"], item["test"]) for item in fold["inner"]]
        scope = f"B/{TARGET}/S2/fold{fold_number}"
        params, _ = select_hyperparameters(features.loc[train], target.loc[train], inner, MODEL,
                                            settings["models"][MODEL], settings["model_seed"], scope)
        predicted = fit_predict_dft(features.loc[train], target.loc[train], features.loc[test], MODEL,
                                    params, settings["model_seed"], scope + "/outer_refit", train)
        reference = archived[archived["fold"] == fold_number].set_index("formula_id")
        if not reference.index.is_unique or set(test) != set(reference.index):
            raise ValueError(f"Archived test identifiers differ for fold {fold_number}.")
        reference = reference.loc[test]
        if not np.allclose(target.loc[test], reference["observed_eV"], rtol=0, atol=1e-10):
            raise ValueError(f"Archived reference energies differ for fold {fold_number}.")
        difference = np.abs(predicted - reference["predicted_eV"])
        deviation = float(difference.max())
        losses = np.abs(predicted - target.loc[test].to_numpy())
        print(f"Fold {fold_number}: n={len(test)}, selected={params}, MAE={losses.mean():.9f} eV, "
              f"max archived prediction difference={deviation:.3g} eV", flush=True)
        if not np.all(difference <= TOLERANCE_EV):
            raise ValueError(f"Predictions differ from the archive by more than {TOLERANCE_EV:g} eV.")
        absolute_errors.extend(losses)

    if not absolute_errors:
        raise ValueError("No matching fold in the split file.")
    if args.fold == "all":
        if len(absolute_errors) != len(target):
            raise ValueError("Pooled fold count does not match the primary DFT population.")
        print(f"S2 ExtraTrees MAE over {len(absolute_errors)} formulas: {np.mean(absolute_errors):.9f} eV")


if __name__ == "__main__":
    main()
