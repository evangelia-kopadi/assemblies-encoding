"""Generate flat single-run benchmark CSV artifacts.

This script writes the table_1_single_run_* artifact family from the same
runner used by the validation scripts. It evaluates the baseline Bernoulli
(0.30/0.10) setting and deterministic-k step 10 once per benchmark dataset,
for a given causal discovery method (PC or GES).

Usage:
  python -m experiments.generate_single_run_table_artifacts
  python -m experiments.generate_single_run_table_artifacts --method ges
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.experiment_defaults import DEFAULTS, get_output_filepath, runner_kwargs
from experiments.run_sensitivity_sweep import build_datasets
from src.runner import run_causal_dag_validation


DATASET_ORDER = ["Alzheimers", "Stroke", "Credit", "Student", "Vaccine"]
DATASET_DISPLAY = {
    "Alzheimers": "Alzheimer",
    "Stroke": "Stroke Risk",
    "Credit": "Credit Default",
    "Student": "Student Success",
    "Vaccine": "Vaccine Efficacy",
}
DATASET_DOMAIN = {
    "Alzheimers": "Medical",
    "Stroke": "Medical",
    "Credit": "Finance",
    "Student": "Education",
    "Vaccine": "Medical",
}
CONFIGS = [
    (
        "Deterministic-k (step 10)",
        {
            "deterministic_k_encoding": True,
            "deterministic_k_step": 10,
            "deterministic_k_readout_mode": "pool_mean",
            "stimulus_k": DEFAULTS.assembly_k,
        },
    ),
    (
        "Bernoulli (0.30/0.10)",
        {
            "deterministic_k_encoding": False,
            "positive_prob": 0.30,
            "negative_prob": 0.10,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-run table artifacts")
    parser.add_argument(
        "--method",
        choices=["pc", "ges"],
        default="pc",
        help="Causal discovery method (default: pc)",
    )
    return parser.parse_args()


def _metrics(result: dict, comparison_key: str) -> dict:
    metrics = result["comparison"][comparison_key]
    return {
        "f1": float(metrics["f1"]),
        "recall": float(metrics["recall"]),
        "precision": float(metrics["precision"]),
    }


def _round(value: float) -> float:
    return round(float(value), 3)


def _file_prefix(method: str) -> str:
    # PC keeps the original flat name for backward compatibility.
    return "table_1_single_run" if method == "pc" else f"table_1_{method}_single_run"


def main() -> None:
    args = parse_args()
    method = args.method
    prefix = _file_prefix(method)

    print(f"Running single-run benchmark with method={method.upper()}")

    datasets = build_datasets(DEFAULTS.n_samples)
    rows: list[dict] = []

    total = len(DATASET_ORDER) * len(CONFIGS)
    run_index = 0

    for config_name, overrides in CONFIGS:
        for dataset_name in DATASET_ORDER:
            run_index += 1
            dataset = datasets[dataset_name]
            print(f"[{run_index}/{total}] {config_name} | {dataset_name}")

            df, ground_truth_edges = dataset["gen"](DEFAULTS.seed)
            result = run_causal_dag_validation(
                method=method,
                df=df,
                var_names=dataset["var_names"],
                ground_truth_edges=ground_truth_edges,
                positive_values_map=dataset["positive_values_map"],
                strict_neuron_causal_discovery=False,
                strict_assembly_causal_discovery=False,
                **runner_kwargs(**overrides),
            )

            assembly = _metrics(result, "assembly_vs_gt")
            neuron = _metrics(result, "neuron_vs_gt")

            rows.append(
                {
                    "Method": method.upper(),
                    "Config": config_name,
                    "Dataset": DATASET_DISPLAY[dataset_name],
                    "Domain": DATASET_DOMAIN[dataset_name],
                    "GT Edges": len(ground_truth_edges),
                    "Assembly F1": _round(assembly["f1"]),
                    "Neuron F1": _round(neuron["f1"]),
                    "A-N Gap": _round(assembly["f1"] - neuron["f1"]),
                    "Recall (Assembly vs GT)": _round(assembly["recall"]),
                    "Precision (Assembly vs GT)": _round(assembly["precision"]),
                    "Recall (Neuron vs GT)": _round(neuron["recall"]),
                    "Precision (Neuron vs GT)": _round(neuron["precision"]),
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_path = Path(get_output_filepath(f"{prefix}_metrics.csv"))
    metrics_df.to_csv(metrics_path, index=False)

    means = (
        metrics_df.groupby("Config", as_index=False)[
            [
                "Assembly F1",
                "Neuron F1",
                "A-N Gap",
                "Recall (Assembly vs GT)",
                "Precision (Assembly vs GT)",
                "Recall (Neuron vs GT)",
                "Precision (Neuron vs GT)",
            ]
        ]
        .mean()
        .round(3)
    )
    means["Method"] = method.upper()
    means_path = Path(get_output_filepath(f"{prefix}_means.csv"))
    means.to_csv(means_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved means: {means_path}")
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
