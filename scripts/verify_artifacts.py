from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY = ROOT / "results" / "sensitivity"
TABLE_SUMMARIES = ROOT / "results" / "table_summaries"
STUDENT_TABLE4 = ROOT / "results" / "student_success_table4"


def family_name(config: str) -> str:
    if config.startswith("Bernoulli"):
        return "Bernoulli"
    if config.startswith("Deterministic"):
        return "Deterministic-k"
    return "Other"


def summarize_sweep(path: Path, label: str) -> None:
    df = pd.read_csv(path)
    df["Family"] = df["Config"].map(family_name)

    print(f"\n{label} sensitivity sweep")
    print("- rows:", len(df))
    print("- datasets:", ", ".join(sorted(df["Dataset"].unique())))
    print("- seeds:", ", ".join(str(x) for x in sorted(df["Seed"].unique())))
    print("- configs:", ", ".join(sorted(df["Config"].unique())))

    counts = df.groupby("Family").size().to_dict()
    means = df.groupby("Family")[["Neuron_F1", "Assembly_F1"]].mean().round(3)
    print("- family counts:", counts)
    print(means.to_string())

    expected = {"Bernoulli": 250, "Deterministic-k": 150}
    for key, value in expected.items():
        actual = int(counts.get(key, 0))
        if actual != value:
            raise AssertionError(f"{label}: expected {value} {key} rows, found {actual}")


def check_raw_stats() -> None:
    stats = pd.read_csv(TABLE_SUMMARIES / "raw_stats_for_tables.csv")
    families = stats[stats["Kind"].eq("Family")].copy()
    print("\nFamily rows from raw_stats_for_tables.csv")
    print(families.to_string(index=False))

    expected_rows = {
        "PC Bernoulli": (250, 0.888, 0.538),
        "PC Deterministic-k": (150, 0.905, 0.872),
        "GES Bernoulli": (250, 0.987, 0.544),
        "GES Deterministic-k": (150, 0.989, 0.962),
    }
    by_name = {row["Name"]: row for _, row in families.iterrows()}
    for name, (expected_n, expected_neuron, expected_assembly) in expected_rows.items():
        if name not in by_name:
            raise AssertionError(f"Missing family row: {name}")
        row = by_name[name]
        actual = (int(row["n"]), round(float(row["NeuronMean"]), 3), round(float(row["AssemblyMean"]), 3))
        expected = (expected_n, expected_neuron, expected_assembly)
        if actual != expected:
            raise AssertionError(f"Unexpected values for {name}: {actual} != {expected}")


def check_student_multiseed() -> None:
    df = pd.read_csv(STUDENT_TABLE4 / "summary.csv")
    same = (df["neuron_f1"].round(12) == df["assembly_f1"].round(12)).all()
    mean_neuron = df["neuron_f1"].mean()
    mean_assembly = df["assembly_f1"].mean()
    print("\nStudent Success multi-seed")
    print("- rows:", len(df))
    print("- seed range:", int(df["seed"].min()), "..", int(df["seed"].max()))
    print(f"- mean neuron F1: {mean_neuron:.3f}")
    print(f"- mean assembly F1: {mean_assembly:.3f}")
    print("- per-seed neuron/assembly F1 identical:", same)
    if len(df) != 50 or int(df["seed"].min()) != 42 or int(df["seed"].max()) != 91:
        raise AssertionError("Unexpected Student Success seed coverage")
    if not same:
        raise AssertionError("Student Success neuron/assembly F1 values are not identical per seed")


def main() -> int:
    summarize_sweep(SENSITIVITY / "pc" / "raw.csv", "PC")
    summarize_sweep(SENSITIVITY / "ges" / "raw.csv", "GES")
    check_raw_stats()
    check_student_multiseed()
    print("\nArtifact verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())