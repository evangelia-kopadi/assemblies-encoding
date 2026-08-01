"""GES causal discovery helpers.

Full method name: Greedy Equivalence Search score-based causal discovery.

How it works: call causal-learn's GES implementation with a local BIC score;
search over equivalence classes of DAGs; parse the returned graph into this
repo's edge-list format so GES results can be compared with PC, neuron-level,
and assembly-level DAGs.
"""

from __future__ import annotations


def run_ges_algorithm(data_df, var_names, strict=False):
    """
    Run GES (Greedy Equivalence Search) algorithm for causal discovery.

    GES uses a score-based BIC objective rather than conditional-independence
    testing. It can therefore behave differently from PC, especially with
    limited data.

    Args:
        data_df: DataFrame with variables as columns
        var_names: List of variable names
        strict: If True, omit undirected CPDAG adjacencies from returned edges

    Returns:
        edges: List of (source, target) tuples
        graph: causal-learn GES record
    """
    try:
        from causallearn.search.ScoreBased.GES import ges

        data_array = data_df[var_names].values
        record = ges(data_array, score_func="local_score_BIC", node_names=var_names)

        edges = []
        graph_matrix = record["G"].graph

        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i != j and graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                    edges.append((var_names[i], var_names[j]))
                elif i < j and graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                    if not strict:
                        edges.append((var_names[i], var_names[j]))

        return edges, record

    except Exception as exc:
        print(f"    Warning: GES algorithm failed: {exc}")
        return [], None