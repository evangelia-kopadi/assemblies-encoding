import pytest

from src.validation.observational.dag_comparison import compare_dags


def test_compare_dags_reports_directed_and_skeleton_metrics_separately():
    ground_truth = [("A", "B"), ("B", "C")]
    discovered = [("A", "B"), ("C", "B")]

    metrics = compare_dags(ground_truth, discovered, ["A", "B", "C"])

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)

    assert metrics["skel_tp"] == 2
    assert metrics["skel_fp"] == 0
    assert metrics["skel_fn"] == 0
    assert metrics["skel_precision"] == pytest.approx(1.0)
    assert metrics["skel_recall"] == pytest.approx(1.0)
    assert metrics["skel_f1"] == pytest.approx(1.0)


def test_compare_dags_handles_empty_discovery_without_dividing_by_zero():
    metrics = compare_dags([("A", "B")], [], ["A", "B", "C"])

    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["skel_f1"] == 0.0
