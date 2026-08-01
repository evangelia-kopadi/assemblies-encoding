"""PC (Peter-Clark) causal discovery helpers.

Full method name: Peter-Clark constraint-based causal discovery.

How it works: run causal-learn's PC algorithm with Fisher-Z conditional
independence tests; receive a CPDAG/equivalence-class graph; parse endpoint
encodings into directed and undirected adjacencies for the downstream causal
validation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CPDAGEdge:
    u: str
    v: str
    kind: str  # "directed" | "undirected" | "other"
    # For kind == "directed", direction is u -> v.


def run_pc_algorithm(data_df, var_names, alpha=0.05, strict=False):
    """
    Run PC algorithm for causal discovery on a dataset.

    Args:
        data_df: DataFrame with variables as columns
        var_names: List of variable names
        alpha: Significance level for independence tests
        strict: If True, omit undirected CPDAG adjacencies from returned edges

    Returns:
        edges: List of (source, target) tuples
        graph: causal-learn graph object
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc

        data_array = data_df[var_names].values
        cg = pc(
            data_array,
            alpha=alpha,
            indep_test="fisherz",
            stable=True,
            show_progress=False,
            node_names=var_names,
        )

        edges = []
        graph_matrix = cg.G.graph

        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i != j and graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                    edges.append((var_names[i], var_names[j]))
                elif i < j and graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                    if not strict:
                        edges.append((var_names[i], var_names[j]))

        return edges, cg

    except Exception as exc:
        print(f"    Warning: PC algorithm failed: {exc}")
        return [], None


def parse_causallearn_cpdag(*, cg: object, node_names: Sequence[str]) -> list[CPDAGEdge]:
    """Parse a causal-learn PC result into a CPDAG edge list.

    causal-learn uses an adjacency matrix with endpoint encoding:
      - i -> j  iff  G[i,j] == -1 and G[j,i] == 1
      - i - j   iff  G[i,j] == -1 and G[j,i] == -1  (undirected)

    Any other non-zero adjacency is returned with kind="other".
    """

    if cg is None or getattr(cg, "G", None) is None:
        return []

    graph_matrix = cg.G.graph
    n = len(node_names)
    edges: list[CPDAGEdge] = []

    for i in range(n):
        for j in range(i + 1, n):
            a = int(graph_matrix[i, j])
            b = int(graph_matrix[j, i])
            if a == 0 and b == 0:
                continue

            u = node_names[i]
            v = node_names[j]

            if a == -1 and b == 1:
                edges.append(CPDAGEdge(u=u, v=v, kind="directed"))
            elif a == 1 and b == -1:
                edges.append(CPDAGEdge(u=v, v=u, kind="directed"))
            elif a == -1 and b == -1:
                edges.append(CPDAGEdge(u=u, v=v, kind="undirected"))
            else:
                edges.append(CPDAGEdge(u=u, v=v, kind="other"))

    edges.sort(key=lambda edge: (edge.kind, edge.u, edge.v))
    return edges