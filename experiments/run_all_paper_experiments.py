"""Run all experiment scripts needed to regenerate current paper data artifacts.

This orchestrator executes the full paper pipeline in a fixed order:
1) Single-run dataset validations and flat CSV artifacts
2) Sensitivity sweeps for both methods (PC and GES)
3) Compact PC/GES derived CSV artifacts when both methods are run
4) Student Success multiseed robustness evaluation (Table 5 family)

Usage:
  python -m experiments.run_all_paper_experiments
  python -m experiments.run_all_paper_experiments --dry-run
  python -m experiments.run_all_paper_experiments --skip-validate
  python -m experiments.run_all_paper_experiments --methods pc,ges
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.experiment_defaults import RUN_ID_ENV_VAR, get_results_root, get_run_id


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
                    ["-m", "experiments.alzheimers.validate_alzheimers"],
                ),
                Step(
                    "Validate Stroke dataset",
                    ["-m", "experiments.stroke_risk.validate_stroke"],
                ),
                Step(
                    "Validate Credit dataset",
                    ["-m", "experiments.credit_default.validate_credit"],
                ),
                Step(
                    "Validate Student Success dataset",
                    ["-m", "experiments.student_success.validate_student_success"],
                ),
                Step(
                    "Validate Vaccine dataset",
                    ["-m", "experiments.vaccine_efficacy.validate_vaccine"],
                ),
                Step(
                    "Generate single-run benchmark CSV artifacts (PC)",
                    ["-m", "experiments.generate_single_run_table_artifacts", "--method", "pc"],
                ),
                Step(
                    "Generate single-run benchmark CSV artifacts (GES)",
                    ["-m", "experiments.generate_single_run_table_artifacts", "--method", "ges"],
                ),
            ]
        )

    if not args.skip_sweep:
        for method in methods:
            steps.append(
                Step(
                    f"Run sensitivity sweep ({method.upper()})",
                    ["-m", "experiments.run_sensitivity_sweep", "--method", method],
                )
            )
        if set(methods) == {"pc", "ges"}:
            steps.append(
                Step(
                    "Generate compact PC/GES comparison CSV artifacts",
                    ["-m", "experiments.generate_compact_pc_ges_artifacts"],
                )
            )

    if not args.skip_multiseed:
        steps.append(
            Step(
                "Run Student Success multiseed evaluation",
                ["-m", "experiments.student_success.evaluate_student_success_multiseed"],
            )
        )

    return steps


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def run_step(step: Step, dry_run: bool, env: dict[str, str]) -> tuple[int, float]:
    cmd = [sys.executable, *step.args]
    print("\n" + "=" * 80)
    print(step.name)
    print("Command:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return 0, 0.0

    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    elapsed = time.perf_counter() - started
    code = int(completed.returncode)
    status = "finished" if code == 0 else f"failed with exit code {code}"
    print(f"Step {status} after {_format_duration(elapsed)}.")
    return code, elapsed


def main() -> int:
    args = parse_args()
    methods = _parse_methods(args.methods)
    run_id = get_run_id()
    run_env = os.environ.copy()
    run_env[RUN_ID_ENV_VAR] = run_id
    steps = build_steps(methods=methods, args=args)

    if not steps:
        print("No steps selected. Nothing to run.")
        return 0

    print("Planned pipeline steps:")
    run_output_dir = Path(get_results_root()) / run_id
    print(f"Run output directory: {run_output_dir}")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step.name}")

    total_started = time.perf_counter()
    for i, step in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] Starting: {step.name}")
        code, _elapsed = run_step(step, dry_run=args.dry_run, env=run_env)
        if code != 0:
            print(f"\nPipeline stopped: '{step.name}' failed with exit code {code}.")
            print(
                "Elapsed time before failure: "
                f"{_format_duration(time.perf_counter() - total_started)}."
            )
            return code

    print("\nAll selected experiment steps completed successfully.")
    write_verb = "would be written" if args.dry_run else "are written"
    print(f"Artifacts {write_verb} under {run_output_dir} via get_output_filepath().")
    if not args.dry_run:
        print(f"Total elapsed time: {_format_duration(time.perf_counter() - total_started)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
