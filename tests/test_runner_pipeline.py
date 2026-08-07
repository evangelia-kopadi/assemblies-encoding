from types import SimpleNamespace

import pandas as pd
import pytest

from src.discovery.errors import CausalDiscoveryBackendError, CausalDiscoveryError
from src.runner import CausalDAGValidationResult, run_causal_dag_validation


def test_run_causal_dag_validation_returns_typed_mapping_result(monkeypatch):
    calls = []

    def fake_pc_algorithm(data_df, var_names, alpha, strict, raise_on_error):
        calls.append(
            {
                "columns": list(data_df.columns),
                "var_names": list(var_names),
                "alpha": alpha,
                "strict": strict,
                "raise_on_error": raise_on_error,
            }
        )
        return [("A", "B")], SimpleNamespace(name="fake_pc_graph")

    monkeypatch.setattr("src.runner.run_pc_algorithm", fake_pc_algorithm)

    df = pd.DataFrame(
        {
            "A": [0, 1, 0, 1, 0, 1],
            "B": [0, 0, 1, 1, 0, 1],
        }
    )

    result = run_causal_dag_validation(
        df=df,
        var_names=["A", "B"],
        ground_truth_edges=[("A", "B")],
        neurons_per_var=8,
        assembly_k=2,
        n_train=4,
        n_presentations=1,
        beta=0.05,
        seed=3,
        alpha_pc=0.07,
        method="pc",
        positive_values_map={"A": {1}, "B": {1}},
        positive_prob=1.0,
        negative_prob=0.0,
        strict_neuron_causal_discovery=False,
        strict_assembly_causal_discovery=False,
        assembly_stability_auto_on_degenerate=False,
    )

    assert isinstance(result, CausalDAGValidationResult)
    assert result["comparison"] is result.comparison
    assert result["compression"] is result.compression
    assert result.to_dict()["var_names"] == ["A", "B"]

    assert result.ground_truth_edges == [("A", "B")]
    assert result.neuron_edges == [("A", "B")]
    assert result.assembly_edges == [("A", "B")]
    assert result.neuron_df.shape == (6, 2)
    assert result.assembly_df.shape == (6, 2)
    assert result.compression["ratio"] > 0
    assert result.comparison["assembly_vs_gt"]["f1"] == 1.0

    assert calls == [
        {
            "columns": ["A", "B"],
            "var_names": ["A", "B"],
            "alpha": 0.07,
            "strict": False,
            "raise_on_error": True,
        },
        {
            "columns": ["A", "B"],
            "var_names": ["A", "B"],
            "alpha": 0.07,
            "strict": False,
            "raise_on_error": False,
        },
    ]


def test_run_causal_dag_validation_raises_on_discovery_failure(monkeypatch):
    def failing_pc_algorithm(data_df, var_names, alpha, strict, raise_on_error):
        assert raise_on_error is True
        raise CausalDiscoveryBackendError("PC algorithm failed for variables ['A', 'B']: test failure")

    monkeypatch.setattr("src.runner.run_pc_algorithm", failing_pc_algorithm)

    df = pd.DataFrame(
        {
            "A": [0, 1, 0, 1],
            "B": [0, 0, 1, 1],
        }
    )

    with pytest.raises(CausalDiscoveryError, match="test failure"):
        run_causal_dag_validation(
            df=df,
            var_names=["A", "B"],
            ground_truth_edges=[("A", "B")],
            neurons_per_var=8,
            assembly_k=2,
            n_train=2,
            n_presentations=1,
            beta=0.05,
            seed=3,
            method="pc",
            positive_values_map={"A": {1}, "B": {1}},
            positive_prob=1.0,
            negative_prob=0.0,
            assembly_stability_auto_on_degenerate=False,
        )
