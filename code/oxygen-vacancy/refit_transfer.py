"""Refit the matched experimental baseline and real-vacancy arm on one archived outer fold.

The auxiliary predictions and selected downstream hyperparameters are fixed. The
original implementation fits preprocessing and the ExtraTrees estimator on the ordered
training records for each arm. Archived held-out predictions are read only for scoring.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "original" / "scripts" / "phase3b"))
from common import fit_predict  # noqa: E402


def load_csv(folder, filename, **options):
    path = folder / filename
    if not path.is_file():
        path = folder / (filename + ".gz")
    return pd.read_csv(path, float_precision="round_trip", low_memory=False, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1],
                        help="repository root with primary-benchmark and oxygen-vacancy data")
    repo = parser.parse_args().repo
    config = pd.read_csv(HERE / "config" / "transfer_one_fold.csv").iloc[0]
    config_id = config["config_id"]
    params = json.loads(config["selected_parameters"])
    transfer = repo / "oxygen-vacancy" / "transfer"
    primary = repo / "primary-benchmark"
    features = load_csv(primary, "benchmark_features_312.csv", index_col="row_id")
    records = load_csv(primary, "experimental_cohort_312.csv", index_col="row_id")
    membership = load_csv(transfer, "frozen_eligible_membership.csv")
    fold = membership.loc[membership["config_id"] == config_id]
    train = fold.loc[fold["role"] == "train", "row_id"].tolist()
    test = fold.loc[fold["role"] == "test", "row_id"].tolist()
    if len(set(train)) != config["N_train_rows"] or len(set(test)) != config["N_test_rows"] or set(train) & set(test):
        raise ValueError("Unexpected training or test membership")
    y_train = records.loc[train, "Ea"].rename("Ea_meV")
    scalars = load_csv(transfer, "auxiliary_values_real.csv")
    selected = scalars.loc[(scalars["config_id"] == config_id) & (scalars["arm"] == "real_Efmin")]
    scalar = selected.set_index("row_id", verify_integrity=True)["auxiliary_prediction"]
    if set(scalar.index) != set(train + test):
        raise ValueError("Auxiliary predictions do not cover this fold")
    predicted = {}
    for arm in ["baseline", "real_Efmin"]:
        x_train = features.loc[train].copy()
        x_test = features.loc[test].copy()
        if arm == "real_Efmin":
            x_train["auxiliary_scalar"] = scalar.loc[train]
            x_test["auxiliary_scalar"] = scalar.loc[test]
        predicted[arm], _ = fit_predict(config["downstream_model"], params, x_train,
                                        y_train, x_test, arm, dft=False)
    archived = load_csv(transfer, "outer_fold_predictions.csv")
    for arm, estimates in predicted.items():
        match = ((archived["config_id"] == config_id) &
                 (archived["downstream_model"] == config["downstream_model"]) &
                 (archived["arm"] == arm))
        expected = archived.loc[match].set_index("row_id", verify_integrity=True)["predicted_Ea_meV"]
        if set(expected.index) != set(test):
            raise ValueError("Archived predictions do not match held-out records")
        errors = np.abs(estimates - expected.loc[test].to_numpy())
        np.testing.assert_allclose(estimates, expected.loc[test], rtol=0, atol=1e-6)
        print(f"{config_id} {arm}: {len(test)} predictions, maximum difference {errors.max():.3g} meV")


if __name__ == "__main__":
    main()
