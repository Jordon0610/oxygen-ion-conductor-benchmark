"""Reconstruct selections and compare losses in the original Huang core scopes."""
import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 50
D = Decimal
ROOT = Path(__file__).resolve().parents[1]
RULES = {"M": "M_predicted_Ea", "E": "E_author_Ea", "O": "O_source_conductivity"}
CHOICES = ("M", "E", "U", "O")
TOLERANCE = D("1e-12")
SUBSETS = {"primary": "PRIMARY_14_PAIRS_10_BLOCKS", "without_R0024": "WITHOUT_R0024_12_PAIRS_8_BLOCKS"}
METRICS = [f"regret_{r}" for r in CHOICES] + ["M_minus_E", "M_minus_U"]


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write(path, records):
    assert records, path
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def group(records, *fields):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[f] for f in fields)].append(r)
    return dict(sorted(groups.items()))


def mean(values):
    values = list(values)
    return sum(values, D(0)) / len(values)


def choose(a, b):
    return D(1) if a < b else D(0) if b < a else D("0.5")


def chosen(p_a, a, b):
    return a if p_a == 1 else b if p_a == 0 else ""


def close(new, old):
    return abs(D(str(new)) - D(str(old))) <= TOLERANCE


def reconstruct(root):
    raw = read(root / "kit/inputs/all_pair_repeat_decisions.csv")
    pred = {(r["repeat"], r["model"], r["row_id"]): r for r in
            read(root / "bindings/R9_global_FE_P3_LSGM_filtered_saved_predictions.csv")}
    r7 = {r["accepted_contrast_id"]: r for r in read(root / "bindings/R7__03_core14_target_context.csv")
          if r["branch"] == "E2_LT"}
    assert len(raw) == 252 and len(pred) == 228 and len(r7) == 14
    fixed = ("source", "population", "path_id", "path_type", "pair_id", "sample_a", "sample_b",
             "x_a", "y_a", "x_b", "y_b", "orientation", "component", "branch", "Ea_a_eV", "Ea_b_eV",
             "sigma800_a_id", "sigma800_b_id", "sigma800_a_S_cm", "sigma800_b_S_cm",
             "phase_a", "phase_b", "source_locators_a", "source_locators_b", "carrier_limit_a", "carrier_limit_b")
    numeric = {"x_a", "y_a", "x_b", "y_b", "Ea_a_eV", "Ea_b_eV", "sigma800_a_S_cm", "sigma800_b_S_cm"}
    pairs, checks = [], []
    for (model, repeat, pair), rules in group(raw, "model", "repeat", "pair_number").items():
        archived = {r["rule"]: r for r in rules}
        assert len(rules) == 3 and set(archived) == set(RULES.values())
        row = archived[RULES["M"]]
        reference = r7[row["accepted_contrast_id"]]
        for key in fixed:
            assert all(r[key] == row[key] for r in rules)
            if key in numeric:
                assert D(row[key]) == D(reference[key]), (pair, key)
            else:
                assert row[key] == reference[key], (pair, key)
        a, b = row["sample_a"], row["sample_b"]
        p1, p2 = pred[repeat, model, a], pred[repeat, model, b]
        assert p1["split_id"] == p2["split_id"]
        ea, eb = D(row["Ea_a_eV"]), D(row["Ea_b_eV"])
        sa, sb = D(row["sigma800_a_S_cm"]), D(row["sigma800_b_S_cm"])
        assert sa.is_finite() and sb.is_finite() and min(sa, sb) > 0
        assert (D(p1["y_true"]), D(p2["y_true"])) == (1000 * ea, 1000 * eb)
        for r in rules:
            assert D(r["saved_prediction_a_meV"]) == D(p1["y_pred"])
            assert D(r["saved_prediction_b_meV"]) == D(p2["y_pred"])
        gap = (max(sa, sb) / min(sa, sb)).log10()
        probs = {
            "M": choose(D(p1["y_pred"]), D(p2["y_pred"])),
            "E": choose(ea, eb),
            "U": D("0.5"),
            "O": choose(-sa, -sb),
        }
        losses = {r: gap * (1 - p if sa >= sb else p) for r, p in probs.items()}
        for label, rule in RULES.items():
            old = archived[rule]
            assert chosen(probs[label], a, b) == old["selected_parent"]
            assert (probs[label] == D("0.5")) == (old["unresolved_tie"] == "True")
            diff = abs(losses[label] - D(old["expected_regret_decades_equal_tie_choice"]))
            assert diff <= TOLERANCE
            checks.append({"model": model, "repeat": repeat, "pair_number": pair, "rule": rule,
                           "choice_and_tie_match": True, "absolute_regret_difference_decades": diff})
        both_unique = all(probs[r] != D("0.5") for r in ("M", "E"))
        unique_q = int(probs["M"] != probs["E"]) if both_unique else ""
        mismatch = int(losses["E"] > 0) if probs["E"] != D("0.5") else ""
        delta = losses["M"] - losses["E"]
        accounting = gap * unique_q * (1 - 2 * mismatch) if both_unique else ""
        if both_unique:
            assert accounting == delta, (model, repeat, pair)
        alignment = "Ea_tie" if mismatch == "" else "sigma_tie" if sa == sb else "misaligned" if mismatch else "aligned"
        omitted = "R0024" in {a, b}
        assert (row["without_R0024"] == "False") == omitted
        pairs.append({
            **{k: row[k] for k in fixed},
            "accepted_contrast_id": row["accepted_contrast_id"], "pair_number": pair,
            "repeat": repeat, "split_id": p1["split_id"], "model": model,
            "saved_prediction_a_meV": p1["y_pred"], "saved_prediction_b_meV": p2["y_pred"],
            "formula_a": reference["formula_a"], "formula_b": reference["formula_b"],
            "printed_Ea_pm_a": reference["Ea_pm_a"], "printed_Ea_pm_b": reference["Ea_pm_b"],
            "ordering_uncertainty": row["ordering_uncertainty"],
            "Ea_to_800C_scope": row["Ea_to_800C_scope"], "without_R0024": not omitted,
            **{f"p_a_{r}": p for r, p in probs.items()},
            **{f"chosen_{r}": chosen(p, a, b) for r, p in probs.items()},
            **{f"regret_{r}": loss for r, loss in losses.items()},
            **{f"unique_{r}": int(probs[r] != D("0.5")) for r in CHOICES},
            "log_conductivity_gap_d": gap, "proxy_alignment": alignment,
            "both_M_E_unique": both_unique, "q_disagreement_if_unique": unique_q,
            "m_E_selects_lower_sigma_if_unique": mismatch,
            "consequence_M_vs_E": "helpful" if delta < 0 else "harmful" if delta > 0 else "unchanged",
            "M_minus_E": delta, "M_minus_U": losses["M"] - losses["U"],
            "identity_d_q_1_minus_2m": accounting,
        })
    assert len(pairs) == 84
    return pairs, checks


def stats(records):
    return {k: mean(r[k] for r in records) for k in METRICS + [f"unique_{r}" for r in CHOICES]}


def summarize(pairs):
    scoped, blocks, repeated, paths, overall, weights, joint = [], [], [], [], [], [], []
    for subset in SUBSETS:
        p = [dict(subset=subset, **r) for r in pairs if subset == "primary" or r["without_R0024"]]
        scoped.extend(p)
        # R0030 remains eligible after removing R0024, with no surviving incident pair.
        expected = (11, 14, 10) if subset == "primary" else (9, 12, 8)
        for (model, repeat), pr in group(p, "model", "repeat").items():
            by_block = group(pr, "path_type", "path_id")
            n_blocks = len(by_block)
            assert (len({r[k] for r in pr for k in ("sample_a", "sample_b")}), len(pr), n_blocks) == expected
            for (path_type, path_id), br in by_block.items():
                bs = {"subset": subset, "model": model, "repeat": repeat, "path_type": path_type,
                      "path_id": path_id, "pairs": len(br), "block_weight_within_repeat": Fraction(1, n_blocks),
                      "path_block_weight_within_repeat": Fraction(1, sum(k[0] == path_type for k in by_block)),
                      **stats(br)}
                blocks.append(bs)
                for r in br:
                    weights.append({"subset": subset, "model": model, "repeat": repeat, "path_type": path_type,
                                    "path_id": path_id, "pair_number": r["pair_number"],
                                    "overall_pair_repeat_weight": Fraction(1, 2 * n_blocks * len(br)),
                                    "path_pair_repeat_weight": bs["path_block_weight_within_repeat"] / (2 * len(br))})
            current = [r for r in blocks if (r["subset"], r["model"], r["repeat"]) == (subset, model, repeat)]
            for pt, pbs in [(key[0], value) for key, value in group(current, "path_type").items()] + [("all_paths", current)]:
                context = [r for r in pr if pt == "all_paths" or r["path_type"] == pt]
                repeated.append({"subset": subset, "model": model, "repeat": repeat, "scope": pt,
                                 "pairs": len(context), "blocks": len(pbs),
                                 "overall_block_weight": Fraction(len(pbs), n_blocks),
                                 **stats(pbs),
                                 **{f"n_unique_{r}": sum(q[f"unique_{r}"] for q in context) for r in CHOICES}})
        for (model, pt), repeats in group([r for r in repeated if r["subset"] == subset], "model", "scope").items():
            n_pairs, n_blocks = repeats[0]["pairs"], repeats[0]["blocks"]
            block_weight = repeats[0]["overall_block_weight"]
            assert len(repeats) == 2 and all((r["pairs"], r["blocks"], r["overall_block_weight"]) ==
                                             (n_pairs, n_blocks, block_weight) for r in repeats)
            summary = {
                "subset": subset, "model": model, "scope": pt, "pairs": n_pairs, "blocks": n_blocks,
                "pair_repeat_instances": 2 * n_pairs, "overall_block_weight": block_weight,
                **stats(repeats),
                **{f"n_unique_{r}": sum(x[f"n_unique_{r}"] for x in repeats) for r in CHOICES},
            }
            for r in CHOICES:
                summary[f"raw_coverage_{r}"] = D(summary[f"n_unique_{r}"]) / summary["pair_repeat_instances"]
            summary["contribution_to_overall_M_minus_E"] = (
                summary["M_minus_E"] * block_weight.numerator / block_weight.denominator
            )
            (overall if pt == "all_paths" else paths).append(summary)
            context = [r for r in p if r["model"] == model and (pt == "all_paths" or r["path_type"] == pt)]
            for (alignment, comparable, q, outcome), entries in group(
                context, "proxy_alignment", "both_M_E_unique", "q_disagreement_if_unique", "consequence_M_vs_E"
            ).items():
                joint.append({"subset": subset, "model": model, "scope": pt, "proxy_alignment": alignment,
                              "both_M_E_unique": comparable, "q_disagreement_if_unique": q,
                              "M_vs_E_conductivity_consequence": outcome,
                              "pair_repeat_instances": len(entries), "mean_log_sigma_gap": mean(r["log_conductivity_gap_d"] for r in entries)})
    for g in overall:
        related = [p for p in paths if p["subset"] == g["subset"] and p["model"] == g["model"]]
        assert sum(p["overall_block_weight"] for p in related) == 1
        assert close(sum(p["contribution_to_overall_M_minus_E"] for p in related), g["M_minus_E"])
    return scoped, blocks, repeated, paths, overall, weights, joint


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or args.root / "results/core"
    out.mkdir(parents=True, exist_ok=True)
    pairs, record_checks = reconstruct(args.root)
    scoped, blocks, repeated, paths, overall, weights, joint = summarize(pairs)
    for name, records in (
        ("pair_repeat", scoped), ("block_repeat", blocks), ("repeat_summary", repeated),
        ("path_summary", paths), ("overall_summary", overall), ("weight_ledger", weights),
        ("joint_alignment_disagreement", joint), ("archived_selection_checks", record_checks),
    ):
        write(out / f"{name}.csv", records)
