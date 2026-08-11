"""Promote one selected run into frozen results.

Usage:
  python -m experiments.freeze_results_from_run --run-id 20260811_101010
  python -m experiments.freeze_results_from_run --run-id 20260811_101010 --overwrite
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_COPY_FILES = [
    "table_2_single_run_pc_metrics.csv",
    "table_2_single_run_pc_means.csv",
    "table_2_single_run_ges_metrics.csv",
    "table_2_single_run_ges_means.csv",
    "table_3_pc_raw.csv",
    "table_3_pc_summary.csv",
    "table_3_pc_overall.csv",
    "table_3_ges_raw.csv",
    "table_3_ges_summary.csv",
    "table_3_ges_overall.csv",
    "table_4_practical_success_rate_summary.csv",
    "table_5_interventional_summary.csv",
    "table_5_interventional_dodeltas.csv",
    "table_5_interventional_report.md",
    "experiments_configuration.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy artifacts from runs/<run-id> into frozen results/"
    )
    parser.add_argument("--run-id", required=True, help="Run folder id under runs/")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root folder containing run directories (default: runs)",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Destination frozen results folder (default: results)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    return parser.parse_args()


def _copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists and --overwrite not set: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.runs_root).resolve() / args.run_id
    results_dir = Path(args.results_root).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    missing: list[str] = []

    for name in DEFAULT_COPY_FILES:
        src = run_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dst = results_dir / name
        _copy_file(src, dst, overwrite=args.overwrite)
        copied.append(dst)

    print(f"Run source: {run_dir}")
    print(f"Frozen destination: {results_dir}")
    print("Copied files:")
    for path in copied:
        print(f"  - {path.name}")

    if missing:
        print("Missing in run (skipped):")
        for name in missing:
            print(f"  - {name}")

    print("Freeze promotion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
