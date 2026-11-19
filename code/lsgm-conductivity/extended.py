#!/usr/bin/env python3
"""Reconstruct all38 and source-local choices from accepted numeric input columns.

Run from the discovery package directory:
    python3 code/extended.py
All outputs are CSVs under results/extended. Python 3.11+ standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ExtraTreesRegressor", "XGBoost", "DummyRegressor")
PATH_NAMES = {
    "FIXED_X": "Fixed Sr, varying Mg",
    "FIXED_Y": "Fixed Mg, varying Sr",
    "FIXED_SUM": "Fixed total Sr+Mg substitution",
}
PATH_QUALIFICATION = {
    "FIXED_X": "Nominal oxygen deficiency varies with Mg at fixed Sr.",
    "FIXED_Y": "Nominal oxygen deficiency varies with Sr at fixed Mg.",
    "FIXED_SUM": "Nominal total substitution and nominal oxygen deficiency are fixed. "
    "Mobile vacancy concentrations were unmeasured in this comparison.",
}
TOLERANCE = Decimal("1e-12")
ZERO = Decimal(0)
ONE = Decimal(1)
HALF = Decimal("0.5")

ALIGNMENT_CODES = {
    "aligned": "CONCORDANT_LOWER_Ea_HIGHER_SIGMA",
    "misaligned": "DISCORDANT_LOWER_Ea_NOT_HIGHER_SIGMA",
    "Ea_tie_only": "Ea_TIE_SIGMA_ORDER_ONLY",
    "sigma_tie_only": "SIGMA_TIE_Ea_ORDER_ONLY",
}

SOURCE_SCOPE = {
    "HUANG1998": "Exploratory all38 native adjacent E2 pairs across accepted phase categories. "
    "Source-reported bulk conductivity. Candidate, phase and path-block composition vary jointly.",
    "HP1996": "Source-local reported Ea with unresolved fitted branch. "
    "DC from impedance with bulk/total separation unspecified.",
    "STEVENSON1997": "Source-local low/high fitted Ea branches analyzed separately. "
    "Effective total/macroscopic DC. Both branches reuse the same conductivity values.",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    columns = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def number(value):
    value = Decimal(str(value).strip().replace("−", "-").replace("×10^", "e"))
    if not value.is_finite():
        raise ValueError(f"Expected a finite printed value, got {value}")
    return value


def keyed(rows, columns):
    result = {}
    for row in rows:
        key = tuple(str(row[column]) for column in columns)
        if key in result:
            raise ValueError(f"Duplicate key {columns}: {key}")
        result[key] = row
    return result


def grouped(rows, columns):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[column] for column in columns)].append(row)
    return dict(sorted(result.items()))


def lower_choice(a, b):
    return ONE if a < b else ZERO if a > b else HALF


def regret(probability_a, probability_best_a, log_gap):
    # The gap is zero when the conductivities tie. The same gap appears for
    # every selector, so ties and helpful/harmful labels use exact equality.
    probability_worse = 1 - probability_a if probability_best_a == ONE else probability_a
    return log_gap * probability_worse


def choose_metadata(rule, probability_a, pair, gap, probability_best_a):
    resolved = probability_a != HALF
    choice = pair["sample_a"] if probability_a == ONE else pair["sample_b"]
    return {
        f"{rule}_probability_choose_a": probability_a,
        f"{rule}_unique": int(resolved),
        f"{rule}_selected_parent": choice if resolved else "",
        f"{rule}_regret_decades": regret(probability_a, probability_best_a, gap),
    }


def reconstruct(pair, global_a=None, global_b=None, local_a=None, local_b=None):
    a, b = map(number, (pair["Ea_a_eV"], pair["Ea_b_eV"]))
    sigma_a, sigma_b = map(number, (pair["sigma800_a_S_cm"], pair["sigma800_b_S_cm"]))
    x_a, y_a, x_b, y_b = (number(pair[field]) for field in ("x_a", "y_a", "x_b", "y_b"))
    path_check = {"FIXED_X": x_a == x_b and y_a != y_b,
                  "FIXED_Y": y_a == y_b and x_a != x_b,
                  "FIXED_SUM": x_a + y_a == x_b + y_b and x_a != x_b}
    assert path_check[pair["path_type"]] and pair["adjacent_in_source"] == "True", pair
    if min(sigma_a, sigma_b) <= 0:
        raise ValueError(f"Nonpositive conductivity in {pair['source_pair_id']}")
    gap = abs((sigma_b / sigma_a).log10())
    e = lower_choice(a, b)
    o = lower_choice(sigma_b, sigma_a)
    alignment = (
        "both_tie" if e == o == HALF else
        "Ea_tie_only" if e == HALF else
        "sigma_tie_only" if o == HALF else
        "aligned" if e == o else "misaligned"
    )
    if alignment in ALIGNMENT_CODES:
        assert pair["Ea_sigma800_order"] == ALIGNMENT_CODES[alignment], pair
    result = dict(pair)
    result.update(
        analysis_scope=SOURCE_SCOPE[pair["source"]],
        path_name=PATH_NAMES[pair["path_type"]],
        path_qualification=PATH_QUALIFICATION[pair["path_type"]],
        sigma800_a_decimal=sigma_a,
        sigma800_b_decimal=sigma_b,
        log10_conductivity_gap_decades=gap,
        alignment=alignment,
        Ea_direction_reconstructed="INCREASE" if a < b else "DECREASE" if a > b else "TIE",
    )
    for rule, probability_a in (("E", e), ("U", HALF), ("O", o)):
        result.update(choose_metadata(rule, probability_a, pair, gap, o))
    result["E_minus_U_decades"] = result["E_regret_decades"] - result["U_regret_decades"]
    if global_a is None:
        result.update(repeat="", repeat_scope="Single source branch",
                      predictor_scope="Source-local reported activation energy")
        return result

    assert global_a["split_id"] == global_b["split_id"]
    assert local_a["fit_id"] == local_b["fit_id"]
    assert number(global_a["y_true"]) == 1000 * a and number(global_b["y_true"]) == 1000 * b, pair
    m = lower_choice(number(global_a["y_pred"]), number(global_b["y_pred"]))
    local_m = lower_choice(number(local_a["prediction_meV"]), number(local_b["prediction_meV"]))
    assert m == local_m, (pair["source_pair_id"], global_a["model"], global_a["repeat"])
    result.update(
        model=global_a["model"],
        repeat=global_a["repeat"],
        repeat_scope="Dependent saved P3 repeat",
        global_split_id=global_a["split_id"],
        local_fit_id=local_a["fit_id"],
        global_prediction_a_meV=global_a["y_pred"],
        global_prediction_b_meV=global_b["y_pred"],
        bound_local_prediction_a_meV=local_a["prediction_meV"],
        bound_local_prediction_b_meV=local_b["prediction_meV"],
        local_global_selection_agree=True,
        R0024_distortion_warning_a=local_a["R0024_distortion_warning"],
        R0024_distortion_warning_b=local_b["R0024_distortion_warning"],
    )
    result.update(choose_metadata("M", m, pair, gap, o))
    difference = result["M_regret_decades"] - result["E_regret_decades"]
    effect = "helpful" if difference < 0 else "harmful" if difference > 0 else "unchanged"
    result.update(
        M_minus_E_decades=difference,
        M_minus_U_decades=result["M_regret_decades"] - result["U_regret_decades"],
        M_E_expected_consequence=effect,
        M_E_unique_disagreement="",
        identity_E_selects_lower_conductivity="",
        identity_d_q_times_1_minus_2m="",
        identity_residual_decades="",
    )
    if HALF not in (m, e):
        disagreement = int(m != e)
        e_is_worse = int(e != o and o != HALF)
        identity = gap * disagreement * (1 - 2 * e_is_worse)
        assert identity == difference, (pair, identity, difference)
        result.update(
            M_E_unique_disagreement=disagreement,
            identity_E_selects_lower_conductivity=e_is_worse,
            identity_d_q_times_1_minus_2m=identity,
            identity_residual_decades=difference - identity,
        )
    return result


def weights(rows):
    per_repeat = grouped(rows, ("repeat",))
    for subset in per_repeat.values():
        blocks = grouped(subset, ("path_id",))
        paths = grouped(subset, ("path_type",))
        for block in blocks.values():
            path_blocks = len({row["path_id"] for row in paths[(block[0]["path_type"],)]})
            for row in block:
                row.update(
                    pair_count_in_block=len(block),
                    block_count_in_path=path_blocks,
                    block_count_in_scope=len(blocks),
                    repeat_count_in_scope=len(per_repeat),
                    pair_weight_within_repeat=ONE / (len(blocks) * len(block)),
                    pair_weight_within_path_and_repeat=ONE / (path_blocks * len(block)),
                    pair_weight_in_scope=ONE / (len(per_repeat) * len(blocks) * len(block)),
                    pair_weight_in_path=ONE / (len(per_repeat) * path_blocks * len(block)),
                )


def mean(rows, field):
    # Equal pairs within blocks, equal blocks within each repeat, equal repeats.
    repeats = grouped(rows, ("repeat",))
    repeat_values = []
    for subset in repeats.values():
        blocks = grouped(subset, ("path_id",))
        block_values = [sum((row[field] for row in block), ZERO) / len(block) for block in blocks.values()]
        repeat_values.append(sum(block_values, ZERO) / len(block_values))
    return sum(repeat_values, ZERO) / len(repeat_values)


def summarize(rows, columns):
    result = []
    for key, group in grouped(rows, columns).items():
        first = group[0]
        row = dict(zip(columns, key))
        parent_ids = {pair[k] for pair in group for k in ("sample_a", "sample_b")}
        block_count = len({pair["path_id"] for pair in group})
        row.update(
            path_name=PATH_NAMES.get(row.get("path_type"), ""),
            n_parents=len(parent_ids),
            n_unique_source_pairs=len({pair["source_pair_id"] for pair in group}),
            n_blocks=block_count,
            n_pair_instances=len(group),
            n_saved_repeats=len({pair["repeat"] for pair in group} - {""}),
            n_source_origins=1,
            component=first["component"],
            analysis_scope=first["analysis_scope"],
            phase_labels=" | ".join(sorted({pair[k] for pair in group for k in ("phase_a", "phase_b")})),
        )
        for rule in ("E", "U", "O", "M"):
            if f"{rule}_unique" in first:
                count = sum(pair[f"{rule}_unique"] for pair in group)
                row.update({
                    f"{rule}_regret_decades": mean(group, f"{rule}_regret_decades"),
                    f"{rule}_unique_choices": count,
                    f"{rule}_unresolved_choices": len(group) - count,
                    f"{rule}_raw_coverage": Decimal(count) / len(group),
                    f"{rule}_weighted_coverage": mean(group, f"{rule}_unique"),
                })
        for label in ("aligned", "misaligned", "Ea_tie_only", "sigma_tie_only", "both_tie"):
            row[f"n_{label}"] = sum(pair["alignment"] == label for pair in group)
        for direction in ("INCREASE", "DECREASE", "TIE"):
            row[f"n_Ea_{direction.lower()}"] = sum(
                pair["Ea_direction_reconstructed"] == direction for pair in group)
        row["E_minus_U_decades"] = mean(group, "E_minus_U_decades")
        if "path_type" in row and "path_id" not in row:
            row["path_block_weight_in_overall"] = first["block_count_in_path"] / Decimal(first["block_count_in_scope"])
        if "M_regret_decades" in row:
            for comparator in ("E", "U"):
                row[f"M_minus_{comparator}_decades"] = mean(group, f"M_minus_{comparator}_decades")
            unique = [pair for pair in group if pair["M_E_unique_disagreement"] != ""]
            row["n_M_E_both_unique"] = len(unique)
            row["n_M_E_unique_disagreements"] = sum(pair["M_E_unique_disagreement"] for pair in unique)
            for label in ("helpful", "harmful", "unchanged"):
                row[f"n_M_E_{label}_both_unique"] = sum(
                    pair["M_E_expected_consequence"] == label for pair in unique)
                row[f"n_M_E_{label}_expected_all"] = sum(
                    pair["M_E_expected_consequence"] == label for pair in group)
            if "path_type" in row and "path_id" not in row:
                row["path_contribution_to_overall_M_minus_E"] = (
                    row["path_block_weight_in_overall"] * row["M_minus_E_decades"])
        result.append(row)
    return result


def compare_table(artifact, expected, actual, keys, text_fields, numeric_fields):
    old, new = keyed(expected, keys), keyed(actual, keys)
    assert old.keys() == new.keys(), (artifact, old.keys() - new.keys(), new.keys() - old.keys())
    records = []
    for column in (*text_fields, *numeric_fields):
        if column in numeric_fields:
            error = max(abs(number(old[key][column]) - number(new[key][column])) for key in old)
            assert error <= TOLERANCE, (artifact, column, error)
        else:
            assert all(str(old[key][column]) == str(new[key][column]) for key in old), (artifact, column)
            error = ""
        records.append(dict(artifact=artifact, checked_field=column, n_keyed_rows=len(old),
                            max_absolute_difference=error, result="PASS"))
    return records


def verify_source_strata(accepted, source_path):
    actual = [{
        **row, "n_pair_contexts": row["n_pair_instances"], "Ea_increase": row["n_Ea_increase"],
        "Ea_decrease": row["n_Ea_decrease"], "Ea_ties": row["n_Ea_tie"],
        "order_concordant": row["n_aligned"], "order_discordant": row["n_misaligned"],
        "order_Ea_tie": row["n_Ea_tie_only"], "order_sigma_tie": row["n_sigma_tie_only"],
        "order_both_tie": row["n_both_tie"], "not_independent_tests": "True",
    } for row in source_path]
    fields = ("n_pair_contexts", "n_unique_source_pairs", "n_source_origins", "Ea_increase",
              "Ea_decrease", "Ea_ties", "order_concordant", "order_discordant",
              "order_Ea_tie", "order_sigma_tie", "order_both_tie")
    return compare_table("accepted_path_strata.csv: HP and Stevenson",
                         [row for row in accepted if row["source"] in {"HP1996", "STEVENSON1997"}],
                         actual, ("source", "branch", "population", "path_type"),
                         ("not_independent_tests",), fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "kit" / "inputs")
    parser.add_argument("--global-predictions", type=Path,
                        default=ROOT / "bindings" / "R9_global_FE_P3_LSGM_filtered_saved_predictions.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "extended")
    args = parser.parse_args()
    output = args.output.resolve()
    for source in (args.inputs.resolve(), args.global_predictions.resolve().parent):
        if output == source or source in output.parents:
            raise ValueError("Choose a separate output directory")

    registry = read_csv(args.inputs / "R7_WITHIN_SOURCE_RESULTS.csv")
    source_rows = [row for row in registry if row["registry_type"] == "WITHIN_NATIVE_ADJACENT"]
    huang = [row for row in source_rows if (row["source"], row["branch"]) == ("HUANG1998", "E2_LT")]
    keyed(huang, ("pair_id",))
    assert len(huang) == 88 and len({row["path_id"] for row in huang}) == 22
    parents = {row[key] for row in huang for key in ("sample_a", "sample_b")}
    assert len(parents) == 38
    local_pairs = [row for row in source_rows if row["source"] in {"HP1996", "STEVENSON1997"}]
    keyed(local_pairs, ("source", "branch", "path_id", "pair_id"))
    assert Counter((row["source"], row["branch"]) for row in local_pairs) == {
        ("HP1996", "AUTHOR_REPORTED_REGIME_UNRESOLVED"): 31,
        ("STEVENSON1997", "LOW_T"): 4, ("STEVENSON1997", "HIGH_T"): 4,
    }

    keys = ("model", "repeat", "row_id")
    local = keyed(read_csv(args.inputs / "full_38_parent_prediction_context.csv"), keys)
    global_predictions = keyed(read_csv(args.global_predictions), keys)
    expected = {(model, repeat, parent) for model in MODELS for repeat in ("0", "1") for parent in parents}
    assert local.keys() == global_predictions.keys() == expected
    for key, original in local.items():
        canonical = global_predictions[key]
        assert original["arm"] == "FE_MATCHED" and original["author_parameter"] == "E2"
        assert canonical["result_version"] == "GLOBAL_FE_R9" and canonical["protocol"] == "P3"
        assert canonical["split_id"] == "R9_K_FE_" + original["split_id"].removeprefix("K_")
        assert number(canonical["y_true"]) == number(original["observed_E2_meV_original"])
        assert abs(number(canonical["y_pred"]) - number(original["prediction_meV"])) <= TOLERANCE

    with localcontext() as context:
        context.prec = 50
        broad = []
        for model in MODELS:
            for repeat in ("0", "1"):
                for pair in huang:
                    a = (model, repeat, pair["sample_a"])
                    b = (model, repeat, pair["sample_b"])
                    broad.append(reconstruct(pair, global_predictions[a], global_predictions[b], local[a], local[b]))
        source = [reconstruct(pair) for pair in local_pairs]
        broad_keys = ("source", "branch", "population", "model")
        source_keys = ("source", "branch", "population")
        for subset in grouped(broad, broad_keys).values():
            weights(subset)
        for subset in grouped(source, source_keys).values():
            weights(subset)

        broad_paths = summarize(broad, broad_keys + ("path_type",))
        broad_overall = summarize(broad, broad_keys)
        source_paths = summarize(source, source_keys + ("path_type",))
        source_overall = summarize(source, source_keys)
        verification = verify_source_strata(read_csv(args.inputs / "accepted_path_strata.csv"), source_paths)

        # All broad pairs match canonical/printed selection identities including ties.
        assert all(row["M_probability_choose_a"] == HALF for row in broad if row["model"] == "DummyRegressor")
        for row in broad_overall:
            path_contributions = sum(p["path_contribution_to_overall_M_minus_E"] for p in broad_paths
                                     if p["model"] == row["model"])
            assert abs(path_contributions - row["M_minus_E_decades"]) <= Decimal("1e-45")
        scope = []
        for row in broad_paths + source_paths:
            scope.append({key: row[key] for key in (
                "source", "branch", "population", "path_type", "path_name", "n_parents",
                "n_unique_source_pairs", "n_blocks", "n_saved_repeats", "component", "phase_labels",
                "analysis_scope", "path_block_weight_in_overall",
            ) if key in row} | {
                "model": row.get("model", ""),
                "path_qualification": PATH_QUALIFICATION[row["path_type"]],
                "weighting": "Equal pairs within block, equal blocks within repeat, equal saved repeats"
                             if "model" in row else "Equal pairs within block, equal blocks within source branch",
                "choice_and_ties": "Lower Ea, higher direct conductivity reference, uniform comparator. "
                                   "Equal-probability expected regret for exact central-value ties.",
                "dependence": "Shared composition parents and source series. Stevenson branches share conductivities. "
                              "Huang repeats share the same measured source pairs.",
            })

        output.mkdir(parents=True, exist_ok=True)
        outputs = {
            "huang_all38_pair_results.csv": broad,
            "huang_all38_block_results.csv": summarize(broad, broad_keys + ("repeat", "path_type", "path_id")),
            "huang_all38_repeat_path_results.csv": summarize(broad, broad_keys + ("repeat", "path_type")),
            "huang_all38_repeat_results.csv": summarize(broad, broad_keys + ("repeat",)),
            "huang_all38_path_results.csv": broad_paths,
            "huang_all38_overall_results.csv": broad_overall,
            "source_local_pair_results.csv": source,
            "source_local_block_results.csv": summarize(source, source_keys + ("path_type", "path_id")),
            "source_local_path_results.csv": source_paths,
            "source_local_overall_results.csv": source_overall,
            "extended_scope_weight_source_qualifications.csv": scope,
            "extended_independent_verification.csv": verification,
        }
        for filename, records in outputs.items():
            write_csv(output / filename, records)
        receipt = {
            "interpretation": "Retrospective descriptive source contexts",
            "all38_parents": len(parents), "all38_pairs_per_repeat": len(huang),
            "all38_blocks_per_repeat": 22, "all38_pair_repeat_model_rows": len(broad),
            "saved_local_global_IDs_checked": len(local), "all38_choice_invariance_checks": len(broad),
            "max_local_global_printed_prediction_difference_meV": str(max(
                abs(number(local[key]["prediction_meV"]) - number(global_predictions[key]["y_pred"]))
                for key in expected)),
            "exact_unique_choice_accounting_identity_checks": sum(
                row["identity_residual_decades"] != "" for row in broad),
            "HP_pair_count": 31, "Stevenson_pair_count_each_branch": 4,
            "numeric_comparison_tolerance": str(TOLERANCE),
            "tie_definition": "Exact equality of saved Decimal predictions or printed target values",
            "source_local_models": "Source-local E, O and uniform comparisons from printed values",
            "source_strata_fields_checked": len(verification),
            "file_rows": {name: len(rows) for name, rows in outputs.items()},
        }
        (output / "extended_status.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
