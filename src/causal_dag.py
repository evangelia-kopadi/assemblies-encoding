"""Causal DAG comparison, reporting, and visualization helpers.

Full method name: Directed Acyclic Graph (DAG) structural comparison.

How it works: compare recovered neuron-level and assembly-level edge lists with
the SCM ground-truth edge list; compute precision, recall, F1, missing edges, and
spurious edges; optionally print tree-style reports and visualization plots.
"""

import networkx as nx
import matplotlib.pyplot as plt

def _edge_skeleton(edges):
    """Return undirected skeleton edges as a set of frozensets({u,v})."""
    return {frozenset((u, v)) for (u, v) in edges if u != v}


def print_dag_as_tree(edges, var_names, title="DAG", ground_truth_edges=None):
    """
    Print DAG in a hierarchical tree format with difference highlighting.
    
    Args:
        edges: List of (source, target) tuples
        var_names: List of all variable names
        title: Title for the tree
        ground_truth_edges: Optional ground truth edges for comparison
    """
    if not edges:
        print(f'\n{title}: No edges discovered')
        return
    
    edge_set = set(edges)
    gt_set = set(ground_truth_edges) if ground_truth_edges else set()
    
    print(f'\n{title}:')
    print('=' * 70)
    
    # Print all edges with status indicators
    print('\nEdges:')
    for source, target in sorted(edges):
        edge = (source, target)
        if ground_truth_edges:
            if edge in gt_set:
                status = 'OK CORRECT'
            else:
                status = 'X SPURIOUS (not in ground truth)'
            print(f'  {source:20} -> {target:20}  {status}')
        else:
            print(f'  {source:20} -> {target}')
    
    # Check for missing edges
    if ground_truth_edges:
        missing = gt_set - edge_set
        if missing:
            print('\nMissing Edges (in ground truth but not discovered):')
            for source, target in sorted(missing):
                print(f'  {source:20} -> {target:20}  X FALSE NEGATIVE')
    
    print(f'\nTotal Edges: {len(edges)}')
    
    if ground_truth_edges:
        tp = len(edge_set & gt_set)
        fp = len(edge_set - gt_set)
        fn = len(gt_set - edge_set)
        print(f'  OK Correct: {tp}  |  X Spurious: {fp}  |  X Missing: {fn}')
    
    print('=' * 70)


def compare_dags(ground_truth_edges, discovered_edges, var_names):
    """
    Compare discovered DAG against ground truth.
    
    Args:
        ground_truth_edges: List of (source, target) ground truth edges
        discovered_edges: List of (source, target) discovered edges
        var_names: List of variable names
        
    Returns:
        dict: Comparison metrics (precision, recall, F1, etc.)
    """
    gt_set = set(ground_truth_edges)
    disc_set = set(discovered_edges)
    
    # True positives: edges in both
    tp = len(gt_set & disc_set)
    
    # False positives: discovered but not in ground truth
    fp = len(disc_set - gt_set)
    
    # False negatives: in ground truth but not discovered
    fn = len(gt_set - disc_set)
    
    # True negatives: all possible edges minus union
    all_possible_edges = len(var_names) * (len(var_names) - 1)
    tn = all_possible_edges - tp - fp - fn
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Skeleton (direction-agnostic) metrics
    gt_skel = _edge_skeleton(ground_truth_edges)
    disc_skel = _edge_skeleton(discovered_edges)

    skel_tp = len(gt_skel & disc_skel)
    skel_fp = len(disc_skel - gt_skel)
    skel_fn = len(gt_skel - disc_skel)

    skel_precision = skel_tp / (skel_tp + skel_fp) if (skel_tp + skel_fp) > 0 else 0.0
    skel_recall = skel_tp / (skel_tp + skel_fn) if (skel_tp + skel_fn) > 0 else 0.0
    skel_f1 = 2 * skel_precision * skel_recall / (skel_precision + skel_recall) if (skel_precision + skel_recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_edges_gt': len(gt_set),
        'n_edges_discovered': len(disc_set),
        'skel_tp': skel_tp,
        'skel_fp': skel_fp,
        'skel_fn': skel_fn,
        'skel_precision': skel_precision,
        'skel_recall': skel_recall,
        'skel_f1': skel_f1,
        'n_skel_edges_gt': len(gt_skel),
        'n_skel_edges_discovered': len(disc_skel),
    }


def visualize_three_dags(ground_truth_edges, neuron_edges, assembly_edges, var_names, save_path=None):
    """
    Visualize the three DAGs side-by-side with hierarchical tree layout.
    
    Args:
        ground_truth_edges: List of (source, target) tuples
        neuron_edges: List of (source, target) tuples
        assembly_edges: List of (source, target) tuples
        var_names: List of variable names
        save_path: Path to save figure
    """
    def hierarchical_layout(edges, var_names):
        """Create hierarchical positions for nodes based on topological order."""
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_nodes_from(var_names)
        G.add_edges_from(edges)
        
        # Find nodes at each level
        in_degree = {node: 0 for node in var_names}
        for source, target in edges:
            in_degree[target] += 1
        
        # Level 0: Root nodes (no incoming edges)
        levels = {}
        level_0 = [node for node in var_names if in_degree[node] == 0]
        for node in level_0:
            levels[node] = 0
        
        # Level 1: Nodes that only depend on level 0
        level_1 = []
        for node in var_names:
            if node not in levels:
                parents = [source for source, target in edges if target == node]
                if all(p in levels and levels[p] == 0 for p in parents):
                    levels[node] = 1
                    level_1.append(node)
        
        # Level 2: Everything else
        level_2 = [node for node in var_names if node not in levels]
        for node in level_2:
            levels[node] = 2
        
        # Create positions
        pos = {}
        max_level = max(levels.values()) if levels else 0
        
        # Count nodes per level
        nodes_per_level = {}
        for node, level in levels.items():
            if level not in nodes_per_level:
                nodes_per_level[level] = []
            nodes_per_level[level].append(node)
        
        # Position nodes
        for level, nodes in nodes_per_level.items():
            y = 1.0 - (level / max(max_level, 1))  # Higher levels at top
            num_nodes = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                x = (i + 1) / (num_nodes + 1)  # Spread evenly
                pos[node] = (x, y)
        
        return pos
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle('Causal DAG Comparison (Ground Truth vs Neuron vs Assembly)', 
                 fontsize=16, fontweight='bold')
    
    dags = [
        ('Ground Truth', ground_truth_edges, 'lightgreen'),
        ('Neuron DAG', neuron_edges, 'lightblue'),
        ('Assembly DAG', assembly_edges, 'lightcoral')
    ]
    
    pos_gt = hierarchical_layout(ground_truth_edges, var_names)

    for ax, (title, edges, color) in zip(axes, dags):
        G = nx.DiGraph()
        G.add_nodes_from(var_names)
        G.add_edges_from(edges)
        
        # Use hierarchical layout        
        pos = pos_gt
        
        # Separate correct and spurious edges
        gt_set = set(ground_truth_edges)
        correct_edges = [e for e in edges if e in gt_set] if title != 'Ground Truth' else edges
        spurious_edges = [e for e in edges if e not in gt_set] if title != 'Ground Truth' else []
        
        # DRAW ORDER: Spurious first (underneath), then correct (on top)
        
        # 1. Draw spurious edges FIRST (so they're underneath)
        if spurious_edges:
            nx.draw_networkx_edges(G, pos, edgelist=spurious_edges, 
                                  edge_color='red', arrows=True,
                                  arrowsize=30, width=4, ax=ax,
                                  connectionstyle='arc3,rad=0.2',
                                  arrowstyle='-|>', node_size=3000,
                                  style='dashed', alpha=0.8)
        
        # 2. Draw correct edges SECOND (on top)
        if correct_edges:
            nx.draw_networkx_edges(G, pos, edgelist=correct_edges,
                                  edge_color='black', arrows=True, 
                                  arrowsize=25, width=2.5, ax=ax, 
                                  connectionstyle='arc3,rad=0.15',
                                  arrowstyle='-|>', node_size=3000)
        
        # 3. Draw nodes LAST (on top of everything)
        nx.draw_networkx_nodes(G, pos, node_color=color, node_size=3000, 
                              alpha=0.9, ax=ax, node_shape='s')
        
        # 4. Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
        
        ax.set_title(f'{title}\n({len(edges)} edges)', fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.2)
        
        # Add legend if there are spurious edges
        if spurious_edges:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2.5, label='Correct Edge'),
                Line2D([0], [0], color='red', linewidth=4, linestyle='--', label='Spurious Edge')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\n  DAG comparison saved to: {save_path}')
    
    return fig

def print_dag_comparison_report(ground_truth_edges, neuron_edges, assembly_edges, var_names):
    """
    Print detailed comparison report for the three DAGs with difference highlighting.
    """
    print('\n' + '='*70)
    print('CAUSAL DAG COMPARISON REPORT')
    print('='*70)
    
    # Print all three DAGs in tree format with comparison
    print_dag_as_tree(ground_truth_edges, var_names, 'GROUND TRUTH DAG', None)
    print_dag_as_tree(neuron_edges, var_names, 'NEURON DAG', ground_truth_edges)
    print_dag_as_tree(assembly_edges, var_names, 'ASSEMBLY DAG', ground_truth_edges)
    
    # Show edge-by-edge comparison
    print('\n' + '='*70)
    print('EDGE-BY-EDGE COMPARISON')
    print('='*70)
    
    gt_set = set(ground_truth_edges)
    neuron_set = set(neuron_edges)
    assembly_set = set(assembly_edges)
    
    # Spurious edges in neurons
    neuron_spurious = neuron_set - gt_set
    if neuron_spurious:
        print('\n[WARN] SPURIOUS EDGES IN NEURON DAG:')
        for source, target in sorted(neuron_spurious):
            print(f'  {source} -> {target}')
            print(f'    -> This edge is NOT in ground truth!')
            print(f'    -> Likely due to confounding/correlation vs causation')
    
    # Spurious edges in assemblies
    assembly_spurious = assembly_set - gt_set
    if assembly_spurious:
        print('\n[WARN] SPURIOUS EDGES IN ASSEMBLY DAG:')
        for source, target in sorted(assembly_spurious):
            print(f'  {source} -> {target}')
    else:
        print('\nOK ASSEMBLY DAG: No spurious edges!')
    
    # Missing edges
    neuron_missing = gt_set - neuron_set
    if neuron_missing:
        print('\nX MISSING EDGES IN NEURON DAG:')
        for source, target in sorted(neuron_missing):
            print(f'  {source} -> {target}')
    
    assembly_missing = gt_set - assembly_set
    if assembly_missing:
        print('\nX MISSING EDGES IN ASSEMBLY DAG:')
        for source, target in sorted(assembly_missing):
            print(f'  {source} -> {target}')
    else:
        print('\nOK ASSEMBLY DAG: No missing edges!')
    
    # Edges removed by assembly compression
    removed_by_assembly = neuron_set - assembly_set
    if removed_by_assembly:
        print('\n[TARGET] EDGES REMOVED BY ASSEMBLY COMPRESSION:')
        for source, target in sorted(removed_by_assembly):
            is_spurious = (source, target) in neuron_spurious
            status = '(GOOD - removed spurious edge!)' if is_spurious else '(BAD - removed true edge)'
            print(f'  {source} -> {target}  {status}')
    
    # Continue with quantitative metrics...
    neuron_comparison = compare_dags(ground_truth_edges, neuron_edges, var_names)
    assembly_comparison = compare_dags(ground_truth_edges, assembly_edges, var_names)
    neuron_to_assembly = compare_dags(neuron_edges, assembly_edges, var_names)
    
    print('\n' + '='*70)
    print('QUANTITATIVE METRICS')
    print('='*70)
    print('\n[1] NEURON DAG vs GROUND TRUTH:')
    print(f'  Precision: {neuron_comparison["precision"]:.3f}')
    print(f'  Recall: {neuron_comparison["recall"]:.3f}')
    print(f'  F1 Score: {neuron_comparison["f1"]:.3f}')
    print(f'  Skeleton F1: {neuron_comparison["skel_f1"]:.3f} (P={neuron_comparison["skel_precision"]:.3f}, R={neuron_comparison["skel_recall"]:.3f})')
    
    print('\n[2] ASSEMBLY DAG vs GROUND TRUTH:')
    print(f'  Precision: {assembly_comparison["precision"]:.3f}')
    print(f'  Recall: {assembly_comparison["recall"]:.3f}')
    print(f'  F1 Score: {assembly_comparison["f1"]:.3f}')
    print(f'  Skeleton F1: {assembly_comparison["skel_f1"]:.3f} (P={assembly_comparison["skel_precision"]:.3f}, R={assembly_comparison["skel_recall"]:.3f})')
    
    print('\n[3] ASSEMBLY DAG vs NEURON DAG:')
    print(f'  Agreement: {neuron_to_assembly["precision"]:.3f}')
    
    # Overall assessment
    print('\n' + '='*70)
    print('PRESERVATION ASSESSMENT:')
    print('='*70)
    
    preservation_score = (assembly_comparison['f1'] + neuron_to_assembly['precision']) / 2
    
    if preservation_score > 0.8:
        status = 'EXCELLENT'
        symbol = 'OKOK'
    elif preservation_score > 0.6:
        status = 'GOOD'
        symbol = 'OK'
    elif preservation_score > 0.4:
        status = 'MODERATE'
        symbol = '~'
    else:
        status = 'POOR'
        symbol = 'X'
    
    print(f'  {symbol} Overall Preservation: {preservation_score:.3f} - {status}')
    print(f'  Assembly F1 vs GT: {assembly_comparison["f1"]:.3f}')
    print(f'  Assembly-Neuron Agreement: {neuron_to_assembly["precision"]:.3f}')
    
    return {
        'neuron_vs_gt': neuron_comparison,
        'assembly_vs_gt': assembly_comparison,
        'assembly_vs_neuron': neuron_to_assembly,
        'preservation_score': preservation_score
    }

