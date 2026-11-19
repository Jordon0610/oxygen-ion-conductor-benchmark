"""Fit one model with the archived training implementation and exported study inputs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "original" / "scripts"))
from benchmark.design import MODELS
from benchmark.engine import tune_and_fit


def read_features(path, identifier):
    frame = pd.read_csv(path, index_col=identifier, float_precision="round_trip")
    if not frame.index.is_unique or frame.shape[1] != 144:
        raise ValueError(f"Unexpected feature table at {path}")
    return frame


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fit(args):
    primary = args.data_root / "primary-benchmark"
    external = args.data_root / "external-validation"
    features = read_features(primary / "benchmark_features_312.csv", "row_id")
    cohort = pd.read_csv(primary / "experimental_cohort_312.csv", index_col="row_id")
    if not cohort.index.is_unique or set(cohort.index) != set(features.index):
        raise ValueError("Cohort and feature identifiers differ")

    if args.external:
        split_id = "A3_2_internal_final"
        train_ids = cohort.index.tolist()
        inner = read_json(HERE / "memberships" / "external_training_folds.json")
        test_features = read_features(external / "external_features.csv", "external_record_id")
        if not test_features.columns.equals(features.columns):
            raise ValueError("External and training features differ in names or order")
        for fold in inner:
            training_groups = set(cohort.loc[fold["train_ids"], "connected_group_id"])
            validation_groups = set(cohort.loc[fold["validation_ids"], "connected_group_id"])
            if training_groups & validation_groups:
                raise ValueError("External model selection folds share source-connected groups")
    else:
        split_id = args.split
        primary_splits = read_json(primary / "benchmark_splits.json")
        secondary_splits = read_json(HERE / "memberships" / "secondary_benchmark_splits.json")
        by_id = {row["split_id"]: row for row in primary_splits + secondary_splits}
        if split_id not in by_id:
            raise SystemExit(f"Unknown split {split_id}. See the split identifiers in the membership files.")
        split = by_id[split_id]
        train_ids = split["train_ids"]
        if set(train_ids) & set(split["test_ids"]):
            raise ValueError("Outer training and test identifiers overlap")
        inner = split["inner_splits"]
        test_features = features.loc[split["test_ids"]]

    with threadpool_limits(limits=2):
        model, parameters, _ = tune_and_fit(
            args.model, features.loc[train_ids], cohort.loc[train_ids, "Ea"].astype(float),
            inner, split_id, lambda event: None
        )
        values = model.predict(test_features)
    predictions = pd.Series(values, index=test_features.index, name="prediction_meV")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite replay predictions")
    print(f"{split_id}: {args.model}, {len(train_ids)} training records, {len(values)} predictions")
    print(f"Selected parameters: {json.dumps(parameters, sort_keys=True)}")

    if args.external:
        reference = pd.read_csv(external / "external_records_and_predictions.csv",
                                index_col="external_record_id", float_precision="round_trip")
        reference = reference[f"{args.model}_prediction"]
    elif split["cohort"] == "K_FE":
        reference = pd.read_csv(primary / "held_out_predictions.csv", float_precision="round_trip")
        reference = reference[(reference["split_id"] == split_id) & (reference["model"] == args.model)]
        reference = reference.set_index("row_id")["y_pred"]
    else:
        reference = None

    if reference is not None:
        if not reference.index.is_unique or set(reference.index) != set(predictions.index):
            raise ValueError("Replay and archived prediction identifiers differ")
        max_difference = (predictions - reference.loc[predictions.index]).abs().max()
        print(f"Maximum absolute difference from archived predictions: {max_difference:.10g} meV")
    if args.output:
        predictions.to_csv(args.output, mode="x", float_format="%.17g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Directory containing the study data folders")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split", help="One archived outer split ID from the primary or secondary cohorts")
    mode.add_argument("--external", action="store_true", help="Fit on 312 internal rows and predict on 34 external rows")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, help="Write row IDs and predictions to a new CSV")
    fit(parser.parse_args())
