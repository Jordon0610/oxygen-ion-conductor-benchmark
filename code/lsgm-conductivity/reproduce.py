"""Recalculate LSGM decision, parameter and substitution-path tables from the study data."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent


def run(script, *arguments):
    subprocess.run([sys.executable, "-B", str(HERE / script), *map(str, arguments)], check=True)


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main(data, output):
    data, output = data.resolve(), output.resolve()
    if not (data / "observations/02_ANALYSIS_OBSERVATIONS.csv").is_file():
        raise ValueError(f"Expected the lsgm-conductivity data directory at {data}")
    if output == data or data in output.parents or output in data.parents:
        raise ValueError("Choose an output directory separate from the source data.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty or new output directory.")

    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        observation = data / "observations"
        decisions = work / "conductivity_selection"
        parameters = work / "arrhenius_parameters"
        inputs = work / "decision_inputs"
        names = {
            "R7__03_core14_target_context.csv": observation / "source_tables/03_core14_target_context.csv",
            "R9__Figure4_all_matched_parent_predictions.csv": data / "composition_contrasts/Figure4_all_matched_parent_predictions.csv",
            "R9_global_FE_P3_LSGM_filtered_saved_predictions.csv": data / "conductivity_selection/saved_source_excluded_predictions.csv",
            "R7__05_WITHIN_SOURCE_RESULTS.csv": observation / "05_WITHIN_SOURCE_RESULTS.csv",
        }
        for name, source in names.items():
            copy(source, inputs / name)
        run("calculate_decision_analysis_A.py", "--inputs", inputs, "--output", decisions)

        parameter_inputs = work / "parameter_inputs"
        for name in ("02_ANALYSIS_OBSERVATIONS.csv", "03_TYPED_PARAMETER_AND_ALIAS_LINKS.csv",
                     "04_COMPARISON_REGISTRY.csv", "05_WITHIN_SOURCE_RESULTS.csv"):
            copy(observation / name, parameter_inputs / name)
        run("compute_parameter_analysis.py", "--inputs", parameter_inputs, "--out", parameters)

        kit = work / "kit"
        kit_inputs = kit / "inputs"
        bindings = work / "bindings"
        for name in ("R7__03_core14_target_context.csv", "R9_global_FE_P3_LSGM_filtered_saved_predictions.csv"):
            copy(names[name], bindings / name)
        for name in ("all_pair_repeat_decisions.csv", "full_38_parent_prediction_context.csv"):
            copy(decisions / name, kit_inputs / name)
        copy(observation / "05_WITHIN_SOURCE_RESULTS.csv", kit_inputs / "R7_WITHIN_SOURCE_RESULTS.csv")
        copy(observation / "accepted_path_strata.csv", kit_inputs / "accepted_path_strata.csv")
        run("core.py", "--root", work)
        run("extended.py", "--inputs", kit_inputs, "--output", work / "results/extended",
            "--global-predictions", names["R9_global_FE_P3_LSGM_filtered_saved_predictions.csv"])
        run("robustness.py", "--kit", kit, "--output", work / "results/robustness")
        run("synthesis.py", "--root", work)

        output.mkdir(parents=True, exist_ok=True)
        for source, name in ((decisions, "conductivity_selection"), (parameters, "arrhenius_parameters"),
                             (work / "results", "substitution_paths")):
            shutil.copytree(source, output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="The repository's lsgm-conductivity data folder")
    parser.add_argument("--output", type=Path, required=True, help="An empty or new directory for results")
    args = parser.parse_args()
    main(args.data, args.output)
