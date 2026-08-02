"""DAG visualization helpers.

How it works: render ground-truth, neuron-level, and assembly-level DAGs
side-by-side with a shared hierarchical layout and optional PNG export.
"""

import matplotlib.pyplot as plt
import networkx as nx


def visualize_three_dags(ground_truth_edges, neuron_edges, assembly_edges, var_names, save_path=None):
    """Visualize the three DAGs side-by-side with hierarchical tree layout."""

    def hierarchical_layout(edges, var_names):
        """Create hierarchical positions for nodes based on topological order."""
        graph = nx.DiGraph()
        graph.add_nodes_from(var_names)
        graph.add_edges_from(edges)

        in_degree = {node: 0 for node in var_names}
        for source, target in edges:
            in_degree[target] += 1

        levels = {}
        level_0 = [node for node in var_names if in_degree[node] == 0]
        for node in level_0:
            levels[node] = 0

        for node in var_names:
            if node not in levels:
                parents = [source for source, target in edges if target == node]
                if all(p in levels and levels[p] == 0 for p in parents):
                    levels[node] = 1

        level_2 = [node for node in var_names if node not in levels]
        for node in level_2:
            levels[node] = 2

        pos = {}
        max_level = max(levels.values()) if levels else 0

        nodes_per_level = {}
        for node, level in levels.items():
            nodes_per_level.setdefault(level, []).append(node)

        for level, nodes in nodes_per_level.items():
            y = 1.0 - (level / max(max_level, 1))
            num_nodes = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = (i + 1) / (num_nodes + 1)
                pos[node] = (x, y)

        return pos

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle(
        "Causal DAG Comparison (Ground Truth vs Neuron vs Assembly)",
        fontsize=16,
        fontweight="bold",
    )

    dags = [
        ("Ground Truth", ground_truth_edges, "lightgreen"),
        ("Neuron DAG", neuron_edges, "lightblue"),
        ("Assembly DAG", assembly_edges, "lightcoral"),
    ]

    pos_gt = hierarchical_layout(ground_truth_edges, var_names)

    for ax, (title, edges, color) in zip(axes, dags):
        graph = nx.DiGraph()
        graph.add_nodes_from(var_names)
        graph.add_edges_from(edges)

        pos = pos_gt

        gt_set = set(ground_truth_edges)
        correct_edges = [e for e in edges if e in gt_set] if title != "Ground Truth" else edges
        spurious_edges = [e for e in edges if e not in gt_set] if title != "Ground Truth" else []

        if spurious_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=spurious_edges,
                edge_color="red",
                arrows=True,
                arrowsize=30,
                width=4,
                ax=ax,
                connectionstyle="arc3,rad=0.2",
                arrowstyle="-|>",
                node_size=3000,
                style="dashed",
                alpha=0.8,
            )

        if correct_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=correct_edges,
                edge_color="black",
                arrows=True,
                arrowsize=25,
                width=2.5,
                ax=ax,
                connectionstyle="arc3,rad=0.15",
                arrowstyle="-|>",
                node_size=3000,
            )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=color,
            node_size=3000,
            alpha=0.9,
            ax=ax,
            node_shape="s",
        )

        nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold", ax=ax)

        ax.set_title(f"{title}\n({len(edges)} edges)", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.2)

        if spurious_edges:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="black", linewidth=2.5, label="Correct Edge"),
                Line2D([0], [0], color="red", linewidth=4, linestyle="--", label="Spurious Edge"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n  DAG comparison saved to: {save_path}")

    return fig
