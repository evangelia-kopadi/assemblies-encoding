"""Generate practical success-rate summary from encoding-ablation raw outputs.

Default behavior reads raw sweep files from the current run folder and writes a
single CSV artifact there. This keeps the paper's practical summary reproducible
from run-local artifacts.

Usage:
  python -m experiments.generate_practical_success_summary
  python -m experiments.generate_practical_success_summary --methods pc,ges --threshold 0.6
  python -m experiments.generate_practical_success_summary --input-root results --output-root results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.artifact_aliases import write_table_alias_copy
from experiments.experiment_defaults import get_output_filepath, get_run_output_dir


RAW_BY_METHOD = {
    "pc": "encoding_ablation_pc_raw.csv",
    "ges": "encoding_ablation_ges_raw.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate practical success-rate summary from ablation raw files"
    )
    parser.add_argument(
        "--methods",
        default="pc,ges",
        help="Comma-separated methods to include (allowed: pc,ges)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Assembly_F1 threshold for condition-level success (default: 0.6)",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Optional folder containing encoding_ablation_<method>_raw.csv files (default: current run folder)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output folder for summary CSV (default: current run folder)",
    )
    parser.add_argument(
        "--output-name",
        default="practical_success_rate_summary.csv",
        help="Output CSV filename",
    )
    return parser.parse_args()


def _parse_methods(raw: str) -> list[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    invalid = [m for m in methods if m not in RAW_BY_METHOD]
    if invalid:
        raise ValueError(f"Unsupported methods in --methods: {invalid}")
    seen: set[str] = set()
    ordered: list[str] = []
    for m in methods:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def _resolve_input_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).resolve()
    return Path(get_run_output_dir()).resolve()


def _resolve_output_path(path_arg: str | None, output_name: str) -> Path:
    if path_arg:
        out_root = Path(path_arg).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        return out_root / output_name
    return Path(get_output_filepath(output_name)).resolve()


def main() -> int:
    args = parse_args()
    methods = _parse_methods(args.methods)

    input_root = _resolve_input_root(args.input_root)
    output_path = _resolve_output_path(args.output_root, args.output_name)

    print(f"Input root: {input_root}")
    print(f"Output file: {output_path}")
    print(f"Threshold: Assembly_F1 > {args.threshold}")

    frames: list[pd.DataFrame] = []
    for method in methods:
        file_name = RAW_BY_METHOD[method]
        file_path = input_root / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing required raw file for method '{method}': {file_path}"
            )
        df = pd.read_csv(file_path)
        df["Method"] = method
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df["EncodingFamily"] = all_df["Config"].str.startswith("Deterministic").map(
        {True: "Det-k", False: "Bernoulli"}
    )
    all_df["Success"] = all_df["Assembly_F1"] > float(args.threshold)

    summary = (
        all_df.groupby(["Method", "EncodingFamily"], as_index=False)
        .agg(
            Runs=("Success", "size"),
            Mean_Assembly_F1=("Assembly_F1", "mean"),
            Mean_AN_Gap=("Gap_AssemblyMinusNeuron", "mean"),
            Success_Rate=("Success", "mean"),
        )
        .sort_values(["Method", "EncodingFamily"])
    )

    summary["Success_Rate"] = (summary["Success_Rate"] * 100.0).round(1)
    summary["Mean_Assembly_F1"] = summary["Mean_Assembly_F1"].round(3)
    summary["Mean_AN_Gap"] = summary["Mean_AN_Gap"].round(3)
    summary = summary.rename(
        columns={
            "Success_Rate": "Success_Rate_Percent",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    alias_path = write_table_alias_copy(output_path)

    print("Saved practical success summary:")
    print(output_path)
    if alias_path is not None:
        print(f"Saved table alias: {alias_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
