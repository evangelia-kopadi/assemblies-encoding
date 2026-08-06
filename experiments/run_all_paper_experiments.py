"""Run all experiment scripts needed to regenerate current paper data artifacts.

This orchestrator executes the full paper pipeline in a fixed order:
1) Single-run dataset validations and flat CSV artifacts
2) Sensitivity sweeps for both methods (PC and GES)
3) Compact PC/GES derived CSV artifacts when both methods are run
4) Student Success multiseed robustness evaluation (Table 5 family)

Usage:
  python experiments/run_all_paper_experiments.py
  python experiments/run_all_paper_experiments.py --dry-run
  python experiments/run_all_paper_experiments.py --skip-validate
  python experiments/run_all_paper_experiments.py --methods pc,ges
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    name: str
    args: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all scripts required to regenerate paper experiment data"
    )
    parser.add_argument(
        "--methods",
        default="pc,ges",
        help="Comma-separated methods for run_sensitivity_sweep.py (allowed: pc,ges)",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validate_*.py scripts and single-run CSV artifact generation",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip sensitivity sweeps and compact PC/GES artifact generation",
    )
    parser.add_argument(
        "--skip-multiseed",
        action="store_true",
        help="Skip Student Success multiseed evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them",
    )
    return parser.parse_args()


def _parse_methods(raw: str) -> list[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    invalid = [m for m in methods if m not in {"pc", "ges"}]
    if invalid:
        raise ValueError(f"Unsupported methods in --methods: {invalid}")
    # Keep order, remove duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for m in methods:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def build_steps(methods: list[str], args: argparse.Namespace) -> list[Step]:
    steps: list[Step] = []

    if not args.skip_validate:
        steps.extend(
            [
                Step(
                    "Validate Alzheimer dataset",
                    ["experiments/alzheimers/validate_alzheimers.py"],
                ),
                Step(
                    "Validate Stroke dataset",
                    ["experiments/stroke_risk/validate_stroke.py"],
                ),
                Step(
                    "Validate Credit dataset",
                    ["experiments/credit_default/validate_credit.py"],
                ),
                Step(
                    "Validate Student Success dataset",
                    ["experiments/student_success/validate_student_success.py"],
                ),
                Step(
                    "Validate Vaccine dataset",
                    ["experiments/vaccine_efficacy/validate_vaccine.py"],
                ),
                Step(
                    "Generate single-run benchmark CSV artifacts",
                    ["experiments/generate_single_run_table_artifacts.py"],
                ),
            ]
        )

    if not args.skip_sweep:
        for method in methods:
            steps.append(
                Step(
                    f"Run sensitivity sweep ({method.upper()})",
                    ["experiments/run_sensitivity_sweep.py", "--method", method],
                )
            )
        if set(methods) == {"pc", "ges"}:
            steps.append(
                Step(
                    "Generate compact PC/GES comparison CSV artifacts",
                    ["experiments/generate_compact_pc_ges_artifacts.py"],
                )
            )

    if not args.skip_multiseed:
        steps.append(
            Step(
                "Run Student Success multiseed evaluation",
                ["experiments/student_success/evaluate_student_success_multiseed.py"],
            )
        )

    return steps


def run_step(step: Step, dry_run: bool) -> int:
    cmd = [sys.executable, *step.args]
    print("\n" + "=" * 80)
    print(step.name)
    print("Command:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    methods = _parse_methods(args.methods)
    steps = build_steps(methods=methods, args=args)

    if not steps:
        print("No steps selected. Nothing to run.")
        return 0

    print("Planned pipeline steps:")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step.name}")

    for i, step in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] Starting: {step.name}")
        code = run_step(step, dry_run=args.dry_run)
        if code != 0:
            print(f"\nPipeline stopped: '{step.name}' failed with exit code {code}.")
            return code

    print("\nAll selected experiment steps completed successfully.")
    print("Artifacts are written under runs/YYYYMMDD via get_output_filepath().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
