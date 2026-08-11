"""DAG visualization.

How it works: render ground-truth, neuron-level, and assembly-level DAGs
side-by-side with a shared hierarchical layout and optional PNG export.
"""

import matplotlib.pyplot as plt
import networkx as nx

from ..logging_configuration import get_logger


LOGGER = get_logger(__name__)


def visualize_three_dags(
    ground_truth_edges,
    neuron_edges,
    assembly_edges,
    var_names,
    save_path=None,
    *,
    bw_safe: bool = False,
    compact_layout: bool = False,
):
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
            if compact_layout:
                y = 0.9 - 0.8 * (level / max(max_level, 1))
            else:
                y = 1.0 - (level / max(max_level, 1))
            num_nodes = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = (i + 1) / (num_nodes + 1)
                if compact_layout:
                    spread = 1.24 if level == 0 and num_nodes == 3 else 1.12
                    x = 0.5 + (x - 0.5) * spread
                pos[node] = (x, y)

        return pos

    if compact_layout:
        figsize = (11.2, 3.75)
        node_size = 3000
        label_font_size = 9.6
        title_font_size = 12.6
        legend_font_size = 9
        label_font_weight = "semibold"
        title_font_weight = "semibold"
    else:
        figsize = (22, 8)
        node_size = 3000
        label_font_size = 9
        title_font_size = 14
        legend_font_size = 10
        label_font_weight = "bold"
        title_font_weight = "bold"

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if not compact_layout:
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
    # bw_safe keeps color, while marker shapes and dashed edges avoid color-only cues.
    node_shapes = {"Ground Truth": "o", "Neuron DAG": "s", "Assembly DAG": "h"}

    pos_gt = hierarchical_layout(ground_truth_edges, var_names)
    if compact_layout:
        label_overrides = {
            "Hypertension": "Hyper-\ntension",
            "Atherosclerosis": "Athero-\nsclerosis",
            "BloodClotting": "Blood\nClotting",
        }
    else:
        label_overrides = {}
    node_labels = {node: label_overrides.get(node, node) for node in var_names}

    for ax, (title, edges, color) in zip(axes, dags):
        graph = nx.DiGraph()
        graph.add_nodes_from(var_names)
        graph.add_edges_from(edges)

        pos = pos_gt

        gt_set = set(ground_truth_edges)
        correct_edges = (
            [e for e in edges if e in gt_set] if title != "Ground Truth" else edges
        )
        spurious_edges = (
            [e for e in edges if e not in gt_set] if title != "Ground Truth" else []
        )

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
                node_size=node_size,
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
                node_size=node_size,
            )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=color,
            node_size=node_size,
            alpha=0.9,
            ax=ax,
            node_shape=node_shapes[title] if bw_safe else "s",
            edgecolors="black" if bw_safe else "none",
            linewidths=1.2 if bw_safe else 0,
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            labels=node_labels,
            font_size=label_font_size,
            font_weight=label_font_weight,
            ax=ax,
        )

        ax.set_title(f"{title}\n({len(edges)} edges)", fontsize=title_font_size, fontweight=title_font_weight, pad=7)
        ax.axis("off")
        if compact_layout:
            ax.set_xlim(-0.12, 1.12)
            ax.set_ylim(-0.08, 1.08)
        else:
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.2)

        if spurious_edges:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="black", linewidth=2.5, label="Correct Edge"),
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linewidth=4,
                    linestyle="--",
                    label="Spurious Edge",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=legend_font_size)

    if compact_layout:
        plt.tight_layout(pad=0.5, w_pad=1.1)
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        LOGGER.info(f"\n  DAG comparison saved to: {save_path}")

    return fig


def visualize_two_dags(
    ground_truth_edges,
    assembly_edges,
    var_names,
    encoding_label,
    save_path=None,
):
    """Two-panel (ground truth + assembly) paper-style figure: circles, minimal arc."""
    import matplotlib.pyplot as plt
    import networkx as nx

    def _layout(edges, var_names):
        g = nx.DiGraph()
        g.add_nodes_from(var_names)
        g.add_edges_from(edges)
        in_deg = {n: 0 for n in var_names}
        for s, t in edges:
            in_deg[t] += 1
        levels = {}
        for n in var_names:
            if in_deg[n] == 0:
                levels[n] = 0
        for n in var_names:
            if n not in levels:
                parents = [s for s, t in edges if t == n]
                if parents and all(p in levels for p in parents):
                    levels[n] = max(levels[p] for p in parents) + 1
        for n in var_names:
            if n not in levels:
                levels[n] = max(levels.values(), default=0) + 1
        by_level = {}
        for n, lv in levels.items():
            by_level.setdefault(lv, []).append(n)
        pos = {}
        max_lv = max(levels.values())
        for lv, nodes in by_level.items():
            y = 1.0 - lv / max(max_lv, 1)
            for i, n in enumerate(sorted(nodes)):
                pos[n] = ((i + 1) / (len(nodes) + 1), y)
        return pos

    colors = {"Ground Truth": "lightgreen", "Assembly": "lightcoral"}
    panels = [("Ground Truth", ground_truth_edges), (f"Assembly\n({encoding_label})", assembly_edges)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    pos = _layout(ground_truth_edges, var_names)
    gt_set = set(map(tuple, ground_truth_edges))

    for ax, (title, edges) in zip(axes, panels):
        g = nx.DiGraph()
        g.add_nodes_from(var_names)
        g.add_edges_from(edges)
        is_assembly = title.startswith("Assembly")
        color = colors["Assembly"] if is_assembly else colors["Ground Truth"]
        correct = [e for e in edges if tuple(e) in gt_set] if is_assembly else list(edges)
        spurious = [e for e in edges if tuple(e) not in gt_set] if is_assembly else []
        missing = [e for e in ground_truth_edges if tuple(e) not in set(map(tuple, edges))] if is_assembly else []

        if correct:
            # long edges (y-span > 0.45) get same arc as missing/spurious for visual consistency
            def _arc(e): return 0.25 if abs(pos[e[0]][1] - pos[e[1]][1]) > 0.45 else 0.05
            short_c = [e for e in correct if _arc(e) == 0.05]
            long_c  = [e for e in correct if _arc(e) == 0.25]
            if short_c:
                nx.draw_networkx_edges(g, pos, edgelist=short_c, edge_color="black",
                    arrows=True, arrowsize=18, width=1.5, ax=ax,
                    connectionstyle="arc3,rad=0.05", arrowstyle="-|>", node_size=2800)
            if long_c:
                nx.draw_networkx_edges(g, pos, edgelist=long_c, edge_color="black",
                    arrows=True, arrowsize=18, width=1.5, ax=ax,
                    connectionstyle="arc3,rad=0.25", arrowstyle="-|>", node_size=2800)
        if spurious:
            nx.draw_networkx_edges(g, pos, edgelist=spurious, edge_color="red",
                arrows=True, arrowsize=18, width=1.5, ax=ax,
                connectionstyle="arc3,rad=0.35", arrowstyle="-|>",
                node_size=2800, style="dashed", alpha=0.9)
        if missing:
            nx.draw_networkx_edges(g, pos, edgelist=missing, edge_color="#555555",
                arrows=True, arrowsize=18, width=1.5, ax=ax,
                connectionstyle="arc3,rad=0.25", arrowstyle="-|>",
                node_size=2800, style=(0,(5,4)), alpha=0.85)

        nx.draw_networkx_nodes(g, pos, node_color=color, node_size=2800,
            alpha=0.9, ax=ax, node_shape="o", edgecolors="none")
        nx.draw_networkx_labels(g, pos, font_size=9, font_weight="normal", ax=ax)

        edge_count = len(edges)
        ax.set_title(f"{title}\n({edge_count} edges)", fontsize=12, fontweight="normal", pad=8)
        ax.axis("off")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.15)

        if is_assembly and (spurious or missing):
            from matplotlib.lines import Line2D
            legend = []
            if spurious:
                legend.append(Line2D([0], [0], color="red", linewidth=2.5, linestyle="--", label="Spurious edge"))
            if missing:
                legend.append(Line2D([0], [0], color="#555555", linewidth=1.5, linestyle=(0,(5,4)), label="Missing edge"))
            ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout(pad=0.8, w_pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig

