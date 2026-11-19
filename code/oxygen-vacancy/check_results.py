"""Check the published oxygen-vacancy results using the row-level predictions."""
import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_rows(data_dir, relative_path):
    path = data_dir / relative_path
    if path.is_file():
        source = path.open(encoding="utf-8-sig", newline="")
    else:
        source = gzip.open(str(path) + ".gz", "rt", encoding="utf-8-sig", newline="")
    with source:
        yield from csv.DictReader(source)


def check_close(calculated, stored, label):
    if not math.isclose(calculated, float(stored), rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{label}: calculated {calculated}, archived {stored}")


def check_dft(data_dir):
    settings = ("B", "Ef_min_sourcefiltered_q0", "S2")
    summaries = {}
    for row in read_rows(data_dir, "dft/source_clean_dft_model_results.csv"):
        if (row["population"], row["target"], row["protocol"]) == settings:
            summaries[row["model"]] = row

    errors = defaultdict(dict)
    systems = set()
    for row in read_rows(data_dir, "dft/held_out_predictions.csv"):
        if (row["population"], row["target"], row["protocol"]) != settings:
            continue
        formula = row["formula_id"]
        model = row["model"]
        if formula in errors[model]:
            raise ValueError(f"Duplicate DFT prediction for {model}, {formula}")
        error = abs(float(row["observed_eV"]) - float(row["predicted_eV"]))
        check_close(error, row["absolute_error_eV"], formula)
        errors[model][formula] = error
        systems.add(row["chemical_system"])

    models = {"ExtraTreesRegressor", "DummyRegressor", "XGBoost", "Ridge", "RBF-SVR"}
    if set(errors) != models or set(summaries) != models or len(systems) != 679:
        raise ValueError("Unexpected set of DFT models or chemical systems")
    formulas = set(errors["DummyRegressor"])
    if len(formulas) != 824 or any(set(errors[m]) != formulas for m in models):
        raise ValueError("The models do not cover the expected 824 formulas")

    for model, values in errors.items():
        check_close(mean(values.values()), summaries[model]["MAE_eV"], model)
    for model in ("ExtraTreesRegressor", "DummyRegressor"):
        print(f"DFT S2 {model} MAE: {mean(errors[model].values()):.6f} eV")


def check_transfer(data_dir):
    controls = [f"shuffle_{i}" for i in range(1000, 1020)]
    random_controls = [f"random_{i}" for i in range(2000, 2020)]
    arms = {"baseline", "real_Efmin", *controls, *random_controls}
    errors = defaultdict(lambda: defaultdict(dict))
    group_for_record = {}
    for row in read_rows(data_dir, "transfer/outer_fold_predictions.csv"):
        setting = (row["protocol"], row["cleaning"], row["downstream_model"])
        if setting != ("P3", "exact_clean", "ExtraTreesRegressor") or row["arm"] not in arms:
            continue
        record, group = row["row_id"], row["connected_group_id"]
        repeat = int(row["repeat"])
        repeats = errors[row["arm"]][record]
        if repeat in repeats or group_for_record.setdefault(record, group) != group:
            raise ValueError(f"Inconsistent group or repeated prediction for {record}")
        error = abs(float(row["observed_Ea_meV"]) - float(row["predicted_Ea_meV"]))
        check_close(error, row["absolute_error_meV"], record)
        repeats[repeat] = error

    records = set(group_for_record)
    if len(records) != 191 or len(set(group_for_record.values())) != 32 or set(errors) != arms:
        raise ValueError("Unexpected primary transfer membership or comparison arms")
    scores = {}
    for arm, losses in errors.items():
        if set(losses) != records or any(set(repeats) != {0, 1} for repeats in losses.values()):
            raise ValueError(f"Incomplete paired transfer predictions for {arm}")
        scores[arm] = mean(mean(repeats.values()) for repeats in losses.values())

    reference = {
        row["contrast"]: row
        for row in read_rows(data_dir, "transfer/paired_transfer_contrasts.csv")
        if row["context"] == "primary_exact_ET"
    }
    real = scores["real_Efmin"]
    comparisons = {
        "baseline_minus_real": scores["baseline"],
        "mean_shuffle_minus_real": mean(scores[arm] for arm in controls),
        "mean_random_minus_real": mean(scores[arm] for arm in random_controls),
    }
    print(f"Experimental MAE, baseline and real auxiliary: {scores['baseline']:.6f}, {real:.6f} meV")
    for name, comparator in comparisons.items():
        check_close(comparator, reference[name]["left_MAE_meV"], name)
        check_close(real, reference[name]["right_MAE_meV"], name)
        check_close(comparator - real, reference[name]["delta_meV"], name)
        print(f"{name}: {comparator - real:.6f} meV")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "oxygen-vacancy",
                        help="path to the oxygen-vacancy data folder at the repository root")
    data = parser.parse_args().data
    check_dft(data)
    check_transfer(data)


if __name__ == "__main__":
    main()
