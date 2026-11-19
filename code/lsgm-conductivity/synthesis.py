"""Build scope comparison tables from the independently reconstructed results."""
import argparse
import json
from decimal import Decimal as D
from fractions import Fraction

from core import ROOT, read, write


def main(root):
    results = root / "results"
    rows = read(results / "core/path_summary.csv")
    extended = read(results / "extended/huang_all38_path_results.csv")
    paths = [
        "FIXED_X", "FIXED_Y", "FIXED_SUM",
    ]
    names = dict(zip(paths, ("Fixed Sr, varying Mg", "Fixed Mg, varying Sr", "Fixed total Sr+Mg")))
    weights = []
    comparison = []
    for row in rows:
        comparison.append({
            "candidate_view": row["subset"], "model": row["model"], "path_type": row["scope"],
            "path": names[row["scope"]], "pairs_per_repeat": row["pairs"], "blocks": row["blocks"],
            "path_weight_in_overall": row["overall_block_weight"],
            **{f"regret_{r}": row[f"regret_{r}"] for r in ("M", "E", "U", "O")},
            "M_minus_E": row["M_minus_E"], "M_minus_U": row["M_minus_U"],
            "M_unique_decisions": row["n_unique_M"], "pair_repeat_instances": row["pair_repeat_instances"],
            "M_weighted_coverage": row["unique_M"],
            "M_minus_E_contribution": row["contribution_to_overall_M_minus_E"],
        })
    for row in extended:
        comparison.append({
            "candidate_view": "all38_native", "model": row["model"], "path_type": row["path_type"],
            "path": names[row["path_type"]], "pairs_per_repeat": row["n_unique_source_pairs"], "blocks": row["n_blocks"],
            "path_weight_in_overall": str(Fraction(int(row["n_blocks"]), 22)),
            **{f"regret_{r}": row[f"{r}_regret_decades"] for r in ("M", "E", "U", "O")},
            "M_minus_E": row["M_minus_E_decades"], "M_minus_U": row["M_minus_U_decades"],
            "M_unique_decisions": row["M_unique_choices"], "pair_repeat_instances": row["n_pair_instances"],
            "M_weighted_coverage": row["M_weighted_coverage"],
            "M_minus_E_contribution": row["path_contribution_to_overall_M_minus_E"],
        })
    write(results / "path_scope_comparison.csv", comparison)

    phases = {
        "primary": "11 p-cubic parent records in the accepted phase-restricted comparison",
        "without_R0024": "10 eligible p-cubic records, of which 9 occur in surviving pairs",
        "all38_native": "11 p-cubic, 15 214+p-cubic, 6 237+p-cubic, 6 214+237+p-cubic",
    }
    details = {
        "FIXED_X": "Sr fixed, Mg and nominal oxygen deficiency vary.",
        "FIXED_Y": "Mg fixed, Sr and nominal oxygen deficiency vary.",
        "FIXED_SUM": "Nominal total Sr+Mg substitution and ideal nominal oxygen deficiency are fixed. Mobile vacancy concentration is unmeasured.",
    }
    for row in comparison:
        if row["model"] != "ExtraTreesRegressor":
            continue
        v = row["candidate_view"]
        weights.append({
            "source": "HUANG1998", "branch": "E2_LT", "component": "Author-labelled bulk",
            "scope": v, "overall_parents_eligible": {"primary": 11, "without_R0024": 10, "all38_native": 38}[v],
            "overall_parents_represented": {"primary": 11, "without_R0024": 9, "all38_native": 38}[v],
            "path_type": row["path_type"], "pairs_per_repeat": row["pairs_per_repeat"],
            "blocks_in_path": row["blocks"], "path_weight_in_overall": row["path_weight_in_overall"],
            "phase_and_population": phases[v], "path_qualification": details[row["path_type"]],
            "scope_qualification": {
                "primary": "Accepted original 14 pairs across ten blocks",
                "without_R0024": "Remove original pairs 6 and 12 and their empty blocks. R0030 has no remaining incident pair.",
                "all38_native": "Existing 88 native adjacent pairs across 22 blocks. Exploratory wider candidate, phase and block population.",
            }[v],
            "endpoint": "Direct source-table conductivity at 800 °C. E2 is the reported low-temperature proxy.",
            "weighting": "Equal pairs within original block, equal blocks within scope, then equal saved repeats",
            "n_saved_repeats": 2, "n_experimental_origins": 1,
        })
    write(results / "core_and_wider_scope_records.csv", weights)

    original = {r["pair_id"] for r in read(results / "core/pair_repeat.csv") if r["subset"] == "primary"}
    wider = {r["pair_id"] for r in read(results / "extended/huang_all38_pair_results.csv")}
    assert len(original) == 14 and len(wider) == 88 and original <= wider
    (results / "scope_overlap.json").write_text(json.dumps({
        "core_pair_id_count": len(original), "wider_pair_id_count": len(wider),
        "shared_pair_id_count": len(original & wider),
        "interpretation": "Pair identity is nested. Path-block identities and weighting are specific to each view.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT, type=type(ROOT))
    main(parser.parse_args().root)
