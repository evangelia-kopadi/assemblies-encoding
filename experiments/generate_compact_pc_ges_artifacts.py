"""Generate compact PC/GES single-seed comparison artifacts.

The compact artifact family is derived from the full sensitivity sweep raw CSVs.
It keeps seed 42 and the baseline Bernoulli plus deterministic-k step 10 settings
for a concise PC/GES side-by-side check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_defaults import get_output_filepath, get_run_output_dir


INPUTS = {
    "pc": "table_2_3_pc_sensitivity_raw.csv",
    "ges": "table_4_ges_sensitivity_raw.csv",
}
CONFIG_LABELS = {
    "Bernoulli_pos0.30_neg0.10": "Bernoulli baseline",
    "Deterministic_kstep10": "Deterministic-k",
}


def main() -> None:
    run_dir = Path(get_run_output_dir())
    frames: list[pd.DataFrame] = []

    for method, filename in INPUTS.items():
        input_path = run_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing {input_path}. Run the {method.upper()} sensitivity sweep first."
            )

        df = pd.read_csv(input_path)
        selected = df[
            (df["Seed"] == 42) & df["Config"].isin(CONFIG_LABELS.keys())
        ].copy()
        if len(selected) != 10:
            raise ValueError(
                f"Expected 10 compact rows for {method}, found {len(selected)}"
            )

        selected["Method"] = method
        selected["Config"] = selected["Config"].map(CONFIG_LABELS)
        selected["Gap"] = selected["Assembly_F1"] - selected["Neuron_F1"]
        frames.append(
            selected[
                [
                    "Dataset",
                    "Seed",
                    "Method",
                    "Config",
                    "Neuron_F1",
                    "Assembly_F1",
                    "Gap",
                ]
            ]
        )

    compact = pd.concat(frames, ignore_index=True).sort_values(
        ["Method", "Dataset", "Config"]
    )
    raw_path = Path(get_output_filepath("table_2_3_pc_ges_compact_raw.csv"))
    compact.to_csv(raw_path, index=False)

    summary = (
        compact.groupby(["Method", "Config"], as_index=False)
        .agg(
            Neuron_F1_mean=("Neuron_F1", "mean"),
            Assembly_F1_mean=("Assembly_F1", "mean"),
            Gap_mean=("Gap", "mean"),
        )
        .sort_values(["Method", "Config"])
    )
    summary_path = Path(get_output_filepath("table_2_3_pc_ges_compact_summary.csv"))
    summary.to_csv(summary_path, index=False)

    print(f"Saved compact raw: {raw_path}")
    print(f"Saved compact summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
