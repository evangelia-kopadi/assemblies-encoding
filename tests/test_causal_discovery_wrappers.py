import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.discovery.ges import run_ges_algorithm
from src.discovery.pc import CPDAGEdge, parse_causallearn_cpdag, run_pc_algorithm


def _install_module(monkeypatch, dotted_name, **attributes):
    parts = dotted_name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name not in sys.modules:
            monkeypatch.setitem(sys.modules, package_name, types.ModuleType(package_name))

    module = types.ModuleType(dotted_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, dotted_name, module)
    return module


def _graph_result(matrix):
    return SimpleNamespace(G=SimpleNamespace(graph=np.asarray(matrix)))


def test_pc_wrapper_parses_directed_and_undirected_cpdag_edges(monkeypatch):
    def fake_pc(data_array, alpha, indep_test, stable, show_progress, node_names):
        assert data_array.shape == (3, 3)
        assert alpha == 0.05
        assert indep_test == "fisherz"
        assert stable is True
        assert show_progress is False
        assert node_names == ["A", "B", "C"]
        return _graph_result(
            [
                [0, -1, 0],
                [1, 0, -1],
                [0, -1, 0],
            ]
        )

    _install_module(monkeypatch, "causallearn.search.ConstraintBased.PC", pc=fake_pc)
    df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4], "C": [3, 4, 5]})

    edges, graph = run_pc_algorithm(df, ["A", "B", "C"], strict=False)
    strict_edges, _ = run_pc_algorithm(df, ["A", "B", "C"], strict=True)

    assert graph is not None
    assert edges == [("A", "B"), ("B", "C")]
    assert strict_edges == [("A", "B")]


def test_ges_wrapper_parses_directed_and_undirected_cpdag_edges(monkeypatch):
    def fake_ges(data_array, score_func, node_names):
        assert data_array.shape == (3, 3)
        assert score_func == "local_score_BIC"
        assert node_names == ["A", "B", "C"]
        return {
            "G": SimpleNamespace(
                graph=np.asarray(
                    [
                        [0, -1, 0],
                        [1, 0, -1],
                        [0, -1, 0],
                    ]
                )
            )
        }

    _install_module(monkeypatch, "causallearn.search.ScoreBased.GES", ges=fake_ges)
    df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4], "C": [3, 4, 5]})

    edges, record = run_ges_algorithm(df, ["A", "B", "C"], strict=False)
    strict_edges, _ = run_ges_algorithm(df, ["A", "B", "C"], strict=True)

    assert record is not None
    assert edges == [("A", "B"), ("B", "C")]
    assert strict_edges == [("A", "B")]


def test_parse_causallearn_cpdag_labels_endpoint_kinds():
    cg = _graph_result(
        [
            [0, -1, 1, 2],
            [1, 0, -1, 0],
            [-1, -1, 0, 0],
            [2, 0, 0, 0],
        ]
    )

    edges = parse_causallearn_cpdag(cg=cg, node_names=["A", "B", "C", "D"])

    assert edges == [
        CPDAGEdge(u="A", v="B", kind="directed"),
        CPDAGEdge(u="C", v="A", kind="directed"),
        CPDAGEdge(u="A", v="D", kind="other"),
        CPDAGEdge(u="B", v="C", kind="undirected"),
    ]
