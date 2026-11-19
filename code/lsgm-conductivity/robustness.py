#!/usr/bin/env python3
"""Bounded Huang core influence and fixed-choice parent perturbations.

Usage: python code/robustness.py --kit kit --output results/robustness
Standard library only. Selections come from saved numeric columns.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from hashlib import sha256
import json
from math import lcm
from pathlib import Path

getcontext().prec = 60
D = Decimal
F = Fraction
MODELS = ("ExtraTreesRegressor", "XGBoost", "DummyRegressor")
SCOPES = ("all_paths", "FIXED_X", "FIXED_Y", "FIXED_SUM")
COMPARATORS = {"reported_Ea": "E", "uniform": "U"}
RULES = ("M", "E", "U", "O")


@dataclass(frozen=True)
class Pair:
    model: str
    repeat: str
    number: str
    pair_id: str
    contrast_id: str
    path_type: str
    block: str
    a: str
    b: str
    probability_a: dict[str, Fraction]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows, fields):
    indexed = {tuple(row[field] for field in fields): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate key in {fields}")
    return indexed


def choose_low(a, b):
    return F(1) if a < b else F(0) if a > b else F(1, 2)


def as_decimal(value):
    return D(value.numerator) / D(value.denominator)


def checked_number(value):
    number = D(value)
    if not number.is_finite():
        raise ValueError(f"Nonfinite value: {value}")
    return number


def load_core(kit):
    path = kit / "inputs/all_pair_repeat_decisions.csv"
    pairs, sigma, geometry = [], {}, {}
    source_rows = read_csv(path)
    unique_index(source_rows, ("model", "repeat", "pair_number", "rule"))
    # Use one copy of the numerical inputs for each model/repeat/pair key.
    for row in source_rows:
        if row["rule"] != "M_predicted_Ea":
            continue
        if (row["source"], row["population"], row["branch"], row["component"]) != (
            "HUANG1998", "CORE11_ACCEPTED14", "E2_LT", "Author-labelled bulk"
        ):
            raise ValueError("Unexpected primary source or population")
        a, b = row["sample_a"], row["sample_b"]
        for side, parent in (("a", a), ("b", b)):
            value = checked_number(row[f"sigma800_{side}_S_cm"])
            if value <= 0 or sigma.setdefault(parent, value) != value:
                raise ValueError(f"Inconsistent parent conductivity: {parent}")
        pair_geometry = (a, b, row["path_type"], row["path_id"], row["pair_id"],
                         row["accepted_contrast_id"], row["Ea_a_eV"], row["Ea_b_eV"])
        if geometry.setdefault(row["pair_number"], pair_geometry) != pair_geometry:
            raise ValueError("Core geometry changes with model or repeat")
        p_m = choose_low(checked_number(row["saved_prediction_a_meV"]),
                         checked_number(row["saved_prediction_b_meV"]))
        p_e = choose_low(checked_number(row["Ea_a_eV"]), checked_number(row["Ea_b_eV"]))
        p_o = choose_low(sigma[b], sigma[a])
        pairs.append(Pair(row["model"], row["repeat"], row["pair_number"], row["pair_id"],
                          row["accepted_contrast_id"], row["path_type"], row["path_id"], a, b,
                          {"M": p_m, "E": p_e, "U": F(1, 2), "O": p_o}))
    assert len(source_rows) == 252
    assert len(pairs) == 84 and len(sigma) == 11 and len(geometry) == 14
    assert {p.model for p in pairs} == set(MODELS)
    for model in MODELS:
        for repeat in ("0", "1"):
            rows = [p for p in pairs if p.model == model and p.repeat == repeat]
            assert len(rows) == 14 and {p.number for p in rows} == set(geometry)
            assert len({p.block for p in rows}) == 10
    assert all(p.probability_a["M"] == F(1, 2) for p in pairs if p.model == "DummyRegressor")
    return pairs, sigma, sha256(path.read_bytes()).hexdigest()


def weighted_pairs(pairs):
    groups = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        groups[pair.repeat][pair.block].append(pair)
    for blocks in groups.values():
        for block in blocks.values():
            weight = F(1, len(groups) * len(blocks) * len(block))
            yield from ((pair, weight) for pair in block)


def weighted_summary(pairs, logs):
    weights = list(weighted_pairs(pairs))
    assert sum(weight for _, weight in weights) == 1
    regrets, coverage = {}, {}
    for rule in RULES:
        regret, unique = D(0), F(0)
        for pair, weight in weights:
            a, b = logs[pair.a], logs[pair.b]
            prob_a = pair.probability_a[rule]
            # Compute the expected gap to the better observed choice.
            gap = abs(a - b)
            probability_lower = prob_a if a < b else 1 - prob_a
            regret += as_decimal(weight * probability_lower) * gap
            unique += weight * (prob_a != F(1, 2))
        regrets[rule], coverage[rule] = regret, unique
    return {
        "pairs": len({p.number for p in pairs}),
        "blocks": len({p.block for p in pairs}),
        "repeats": len({p.repeat for p in pairs}),
        "pair_repeat_instances": len(pairs),
        "parents_represented": len({parent for p in pairs for parent in (p.a, p.b)}),
        **{f"{rule}_regret_decades": regrets[rule] for rule in RULES},
        "M_minus_E": regrets["M"] - regrets["E"],
        "M_minus_U": regrets["M"] - regrets["U"],
        **{f"{rule}_unique_instances": sum(p.probability_a[rule] != F(1, 2) for p in pairs)
           for rule in ("M", "E", "O")},
        **{f"{rule}_weighted_coverage": as_decimal(coverage[rule]) for rule in ("M", "E", "O")},
    }


def keep_scope(pairs, scope):
    return [p for p in pairs if scope == "all_paths" or p.path_type == scope]


def influence_rows(pairs, sigma):
    logs = {parent: value.log10() for parent, value in sigma.items()}
    summary, block_rows, repeat_rows, weights = [], [], [], []
    for omitted in sorted(sigma):
        for model in MODELS:
            retained = [p for p in pairs if p.model == model and omitted not in (p.a, p.b)]
            removed = [p for p in pairs if p.model == model and omitted in (p.a, p.b)]
            assert {p.repeat for p in retained} == {"0", "1"}
            for scope in SCOPES:
                selected = keep_scope(retained, scope)
                base = {
                    "omitted_parent": omitted,
                    "model": model,
                    "scope": scope,
                    "removed_incident_pair_numbers": "|".join(
                        sorted({p.number for p in keep_scope(removed, scope)}, key=int)),
                }
                if not selected:
                    summary.append({**base, "status": "NO_RETAINED_PAIRS", "pairs": 0,
                                    "blocks": 0, "repeats": 0})
                    continue
                summary.append({**base, "status": "ESTIMABLE", **weighted_summary(selected, logs)})
                for pair, weight in weighted_pairs(selected):
                    weights.append({**base, "repeat": pair.repeat, "block": pair.block,
                                    "path_type": pair.path_type, "pair_number": pair.number,
                                    "pair_id": pair.pair_id, "accepted_contrast_id": pair.contrast_id,
                                    "sample_a": pair.a, "sample_b": pair.b,
                                    "overall_scope_weight_exact": str(weight)})
                for repeat in ("0", "1"):
                    repeat_pairs = [p for p in selected if p.repeat == repeat]
                    repeat_rows.append({**base, "repeat": repeat,
                                        **weighted_summary(repeat_pairs, logs)})
                    for block in sorted({p.block for p in repeat_pairs}):
                        block_pairs = [p for p in repeat_pairs if p.block == block]
                        block_rows.append({**base, "repeat": repeat, "block": block,
                                           "path_type": block_pairs[0].path_type,
                                           **weighted_summary(block_pairs, logs)})
    return summary, block_rows, repeat_rows, weights


def log_contrast(coefficients, sigma):
    # Rational product establishes the mathematical sign, including an exact zero.
    denominator = lcm(*(coefficient.denominator for coefficient in coefficients.values()))
    ratio = F(1)
    for parent, coefficient in coefficients.items():
        ratio *= F(sigma[parent]) ** int(coefficient * denominator)
    sign = (ratio > 1) - (ratio < 1)
    value = (D(ratio.numerator).log10() - D(ratio.denominator).log10()) / D(denominator)
    return value, sign


def perturbation_rows(pairs, sigma):
    logs = {parent: value.log10() for parent, value in sigma.items()}
    radii, parent_rows = [], []
    for subset in ("primary", "without_R0024"):
        for model in MODELS:
            retained = [p for p in pairs if p.model == model
                        and (subset == "primary" or "R0024" not in (p.a, p.b))]
            for scope in SCOPES:
                selected = keep_scope(retained, scope)
                weights = list(weighted_pairs(selected))
                summary = weighted_summary(selected, logs)
                for comparator, rule in COMPARATORS.items():
                    base = {"subset": subset, "model": model, "scope": scope,
                            "comparator": comparator}
                    parents = sorted({parent for p in selected for parent in (p.a, p.b)})
                    coefficients = dict.fromkeys(parents, F(0))
                    for pair, weight in weights:
                        signed_weight = weight * (pair.probability_a[rule] - pair.probability_a["M"])
                        coefficients[pair.a] += signed_weight
                        coefficients[pair.b] -= signed_weight
                    assert sum(coefficients.values()) == 0
                    gap, sign = log_contrast(coefficients, sigma)
                    norm = sum(abs(value) for value in coefficients.values())
                    assert abs(gap - summary[f"M_minus_{rule}"]) < D("1e-55")
                    if norm:
                        radius = abs(gap) / as_decimal(norm)
                        lower, upper = D(10) ** (-radius), D(10) ** radius
                        status = "FINITE_BREAK_EVEN_RADIUS" if sign else "ZERO_GAP_SIGN_UNSTABLE"
                    else:
                        assert sign == 0 and gap == 0
                        radius = lower = upper = ""
                        status = "IDENTICAL_UNDER_EVERY_PARENT_PERTURBATION"
                    perturbations = {}
                    for parent, coefficient in coefficients.items():
                        direction = (coefficient > 0) - (coefficient < 0)
                        shift = -sign * direction * radius if norm else ""
                        factor = D(10) ** shift if norm else ""
                        perturbations[parent] = shift
                        parent_rows.append({**base, "parent": parent,
                                            "sigma800_S_cm": sigma[parent],
                                            "coefficient_exact": str(coefficient),
                                            "coefficient": as_decimal(coefficient),
                                            "attaining_log10_shift": shift,
                                            "attaining_multiplier": factor})
                    residual = gap + sum(as_decimal(coefficients[parent]) * shift
                                         for parent, shift in perturbations.items()) if norm else D(0)
                    assert abs(residual) < D("1e-55")
                    if norm:
                        shifted_logs = {**logs, **{parent: logs[parent] + shift
                                                   for parent, shift in perturbations.items()}}
                        # The best conductivity can switch under perturbation.
                        # Both regrets use the same perturbed best, so it cancels.
                        direct_residual = weighted_summary(selected, shifted_logs)[f"M_minus_{rule}"]
                        assert abs(direct_residual) < D("1e-55")
                    else:
                        direct_residual = D(0)
                    radii.append({**base, "pairs": summary["pairs"], "blocks": summary["blocks"],
                                  "repeats": summary["repeats"],
                                  "parents_represented": len(parents),
                                  "nonzero_parent_coefficients": sum(bool(c) for c in coefficients.values()),
                                  "model_minus_comparator_regret": gap, "baseline_sign": sign,
                                  "coefficient_L1_exact": str(norm), "coefficient_L1": as_decimal(norm),
                                  "uniform_log10_break_even_radius": radius,
                                  "lower_multiplicative_factor": lower,
                                  "upper_multiplicative_factor": upper,
                                  "attaining_linear_residual_decades": residual,
                                  "attaining_direct_regret_residual_decades": direct_residual,
                                  "status": status})
    return radii, parent_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kit, output = args.kit.resolve(), args.output.resolve()
    if output == kit or kit in output.parents or output in kit.parents:
        parser.error("Use a separate output directory")
    pairs, sigma, input_hash = load_core(kit)
    influence, blocks, repeats, weights = influence_rows(pairs, sigma)
    radii, coefficients = perturbation_rows(pairs, sigma)
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("leave_one_parent_influence.csv", influence),
        ("leave_one_parent_block_regrets.csv", blocks),
        ("leave_one_parent_repeat_regrets.csv", repeats),
        ("leave_one_parent_pair_weights.csv", weights),
        ("conductivity_perturbation_radii.csv", radii),
        ("conductivity_parent_coefficients.csv", coefficients),
    ):
        write_csv(output / filename, rows)
    with (output / "robustness_scope.json").open("x") as stream:
        json.dump({
            "source": "Huang1998 author-labelled bulk, E2_LT and directly tabulated sigma at 800 C",
            "core_numerical_input_sha256": input_hash,
            "models": MODELS,
            "leave_one_parent_cases": 11,
            "omission_rule": "Delete incident pairs and retain surviving original blocks. Reweight surviving blocks equally within repeat. No bridging.",
            "tie_rule": "Exact numeric equality. M and E retain ties with equal-probability expected regret. U is a uniform comparator. O is the direct conductivity reference.",
            "coefficient_definition": "For each pair with hierarchical weight w, c_a += w*(p_comparator(a)-p_M(a)) and c_b -= that amount.",
            "perturbation": "One log10 conductivity shift per parent, shared across all occurrences. Sup norm bound epsilon. Fixed M, E and U choice probabilities.",
            "sharp_bound": "D(e) = D(0) + sum(c_j*e_j). Extremal magnitude is epsilon*sum(abs(c_j)). Break-even at abs(D(0))/sum(abs(c_j)) when the denominator is positive.",
            "attainment": "Each shift is -sign(D)*sign(c_j)*radius. At the boundary equality is attainable and a larger radius permits reversal. When c=0, D is invariant and the radius field is blank.",
            "inference": "Retrospective descriptive influence and deterministic sensitivity. Shared pairs and repeats are dependent. The radius describes a hypothetical perturbation set. Ea-order variation, refitting, and experimental uncertainty are outside that set.",
        }, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with localcontext() as context:
        context.prec = 60
        main()
