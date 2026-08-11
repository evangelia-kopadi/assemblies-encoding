from types import SimpleNamespace

import pytest

from experiments.generate_paper_data_only import _parse_methods, build_steps


def test_build_steps_default_calls_all_required_scripts_in_order():
    args = SimpleNamespace(skip_single_run=False, skip_sweep=False)

    steps = build_steps(methods=["pc", "ges"], args=args)

    assert [step.args for step in steps] == [
        ["-m", "experiments.generate_single_run_table_artifacts", "--method", "pc"],
        ["-m", "experiments.generate_single_run_table_artifacts", "--method", "ges"],
        ["-m", "experiments.generate_encoding_ablation", "--method", "pc"],
        ["-m", "experiments.generate_encoding_ablation", "--method", "ges"],
        [
            "-m",
            "experiments.generate_practical_success_summary",
            "--methods",
            "pc,ges",
        ],
    ]


def test_build_steps_respects_skip_single_run():
    args = SimpleNamespace(skip_single_run=True, skip_sweep=False)

    steps = build_steps(methods=["pc", "ges"], args=args)

    assert [step.args for step in steps] == [
        ["-m", "experiments.generate_encoding_ablation", "--method", "pc"],
        ["-m", "experiments.generate_encoding_ablation", "--method", "ges"],
        [
            "-m",
            "experiments.generate_practical_success_summary",
            "--methods",
            "pc,ges",
        ],
    ]


def test_build_steps_respects_skip_sweep():
    args = SimpleNamespace(skip_single_run=False, skip_sweep=True)

    steps = build_steps(methods=["pc", "ges"], args=args)

    assert [step.args for step in steps] == [
        ["-m", "experiments.generate_single_run_table_artifacts", "--method", "pc"],
        ["-m", "experiments.generate_single_run_table_artifacts", "--method", "ges"],
    ]


def test_parse_methods_deduplicates_preserves_order_and_normalizes_case():
    methods = _parse_methods(" GES,pc,ges ")

    assert methods == ["ges", "pc"]


def test_parse_methods_rejects_invalid_values():
    with pytest.raises(ValueError, match="Unsupported methods"):
        _parse_methods("pc,not-a-method")


def test_parse_methods_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        _parse_methods("   ")

