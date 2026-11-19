"""Recompute benchmark and external errors from the exported study predictions."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent / "original" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from benchmark.summarize import aggregate, per_row
from A3_2_external_validation.score_stage2 import GROUP, MODELS, draws, interval, metric_arrays


def benchmark_scores(folder):
    rows = pd.read_csv(folder / "held_out_predictions.csv", float_precision="round_trip")
    rows["ae"] = (rows["y_pred"] - rows["y_true"]).abs()
    rows["se"] = (rows["y_pred"] - rows["y_true"]) ** 2
    rows["dummy_ae"] = (rows["dummy_pred"] - rows["y_true"]).abs()
    collapsed = per_row(rows)
    results = []
    for (protocol, model), records in collapsed.groupby(["protocol", "model"], sort=False):
        for macro_family in ([False, True] if protocol == "P4" else [False]):
            metrics = aggregate(records, macro=macro_family, stratified=(protocol == "P4"))
            results.append({"protocol": protocol, "model": model, **metrics})
    return pd.DataFrame(results)


def external_scores(folder):
    rows = pd.read_csv(folder / "external_records_and_predictions.csv", float_precision="round_trip")
    reference = pd.read_csv(folder / "model_summary.csv", index_col="model", float_precision="round_trip")
    idx = draws(rows[GROUP].nunique())
    results = []
    for model in MODELS:
        error_col = f"{model}_absolute_error"
        rows[error_col] = (rows[f"{model}_prediction"] - rows["observed_Ea_meV"]).abs()
        macro_samples, row_samples = metric_arrays(rows, model, idx)
        macro_low, macro_high = interval(macro_samples)
        row_low, row_high = interval(row_samples)
        values = {
            "macro_group_MAE_meV": rows.groupby(GROUP)[error_col].mean().mean(),
            "row_weighted_MAE_meV": rows[error_col].mean(),
            "macro_group_MAE_CI_low": macro_low, "macro_group_MAE_CI_high": macro_high,
            "row_weighted_MAE_CI_low": row_low, "row_weighted_MAE_CI_high": row_high,
        }
        np.testing.assert_allclose(list(values.values()), reference.loc[model, list(values)].to_numpy(dtype=float),
                                   rtol=1e-10, atol=1e-8, err_msg=model)
        results.append({"model": model, **values})
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Directory containing the study data folders")
    args = parser.parse_args()
    primary = benchmark_scores(args.data_root / "primary-benchmark")
    print("Internal benchmark, all tested rows, with repeats averaged within record")
    print(primary[["protocol", "model", "aggregation", "N_rows", "MAE_meV", "MAE_CI_low", "MAE_CI_high"]]
          .to_string(index=False))
    external = external_scores(args.data_root / "external-validation")
    print("\nRetrospective external validation, verified against model_summary.csv")
    print(external.to_string(index=False))
