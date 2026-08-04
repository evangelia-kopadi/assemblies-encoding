"""DAG structural evaluation helpers.

Directed Acyclic Graph (DAG) structural comparison.

How it works: compare recovered neuron-level and assembly-level edge lists with
SCM ground-truth edge lists; compute precision, recall, F1, skeleton metrics,
missing edges, and spurious edges; and produce text reports.
"""


def _edge_skeleton(edges):
    """Return undirected skeleton edges as a set of frozensets({u,v})."""
    return {frozenset((u, v)) for (u, v) in edges if u != v}


def print_dag_as_tree(edges, var_names, title="DAG", ground_truth_edges=None):
    """Print DAG in a hierarchical tree format with difference highlighting."""
    if not edges:
        print(f"\n{title}: No edges discovered")
        return

    edge_set = set(edges)
    gt_set = set(ground_truth_edges) if ground_truth_edges else set()

    print(f"\n{title}:")
    print("=" * 70)

    print("\nEdges:")
    for source, target in sorted(edges):
        edge = (source, target)
        if ground_truth_edges:
            if edge in gt_set:
                status = "OK CORRECT"
            else:
                status = "X SPURIOUS (not in ground truth)"
            print(f"  {source:20} -> {target:20}  {status}")
        else:
            print(f"  {source:20} -> {target}")

    if ground_truth_edges:
        missing = gt_set - edge_set
        if missing:
            print("\nMissing Edges (in ground truth but not discovered):")
            for source, target in sorted(missing):
                print(f"  {source:20} -> {target:20}  X FALSE NEGATIVE")

    print(f"\nTotal Edges: {len(edges)}")

    if ground_truth_edges:
        tp = len(edge_set & gt_set)
        fp = len(edge_set - gt_set)
        fn = len(gt_set - edge_set)
        print(f"  OK Correct: {tp}  |  X Spurious: {fp}  |  X Missing: {fn}")

    print("=" * 70)


def compare_dags(ground_truth_edges, discovered_edges, var_names):
    """Compare discovered DAG against ground truth and return metrics."""
    gt_set = set(ground_truth_edges)
    disc_set = set(discovered_edges)

    tp = len(gt_set & disc_set)
    fp = len(disc_set - gt_set)
    fn = len(gt_set - disc_set)

    all_possible_edges = len(var_names) * (len(var_names) - 1)
    tn = all_possible_edges - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    gt_skel = _edge_skeleton(ground_truth_edges)
    disc_skel = _edge_skeleton(discovered_edges)

    skel_tp = len(gt_skel & disc_skel)
    skel_fp = len(disc_skel - gt_skel)
    skel_fn = len(gt_skel - disc_skel)

    skel_precision = skel_tp / (skel_tp + skel_fp) if (skel_tp + skel_fp) > 0 else 0.0
    skel_recall = skel_tp / (skel_tp + skel_fn) if (skel_tp + skel_fn) > 0 else 0.0
    skel_f1 = (
        2 * skel_precision * skel_recall / (skel_precision + skel_recall)
        if (skel_precision + skel_recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_edges_gt": len(gt_set),
        "n_edges_discovered": len(disc_set),
        "skel_tp": skel_tp,
        "skel_fp": skel_fp,
        "skel_fn": skel_fn,
        "skel_precision": skel_precision,
        "skel_recall": skel_recall,
        "skel_f1": skel_f1,
        "n_skel_edges_gt": len(gt_skel),
        "n_skel_edges_discovered": len(disc_skel),
    }


def print_dag_comparison_report(
    ground_truth_edges, neuron_edges, assembly_edges, var_names
):
    """Print detailed comparison report for the three DAGs."""
    print("\n" + "=" * 70)
    print("CAUSAL DAG COMPARISON REPORT")
    print("=" * 70)

    print_dag_as_tree(ground_truth_edges, var_names, "GROUND TRUTH DAG", None)
    print_dag_as_tree(neuron_edges, var_names, "NEURON DAG", ground_truth_edges)
    print_dag_as_tree(assembly_edges, var_names, "ASSEMBLY DAG", ground_truth_edges)

    print("\n" + "=" * 70)
    print("EDGE-BY-EDGE COMPARISON")
    print("=" * 70)

    gt_set = set(ground_truth_edges)
    neuron_set = set(neuron_edges)
    assembly_set = set(assembly_edges)

    neuron_spurious = neuron_set - gt_set
    if neuron_spurious:
        print("\n[WARN] SPURIOUS EDGES IN NEURON DAG:")
        for source, target in sorted(neuron_spurious):
            print(f"  {source} -> {target}")
            print("    -> This edge is NOT in ground truth!")
            print("    -> Likely due to confounding/correlation vs causation")

    assembly_spurious = assembly_set - gt_set
    if assembly_spurious:
        print("\n[WARN] SPURIOUS EDGES IN ASSEMBLY DAG:")
        for source, target in sorted(assembly_spurious):
            print(f"  {source} -> {target}")
    else:
        print("\nOK ASSEMBLY DAG: No spurious edges!")

    neuron_missing = gt_set - neuron_set
    if neuron_missing:
        print("\nX MISSING EDGES IN NEURON DAG:")
        for source, target in sorted(neuron_missing):
            print(f"  {source} -> {target}")

    assembly_missing = gt_set - assembly_set
    if assembly_missing:
        print("\nX MISSING EDGES IN ASSEMBLY DAG:")
        for source, target in sorted(assembly_missing):
            print(f"  {source} -> {target}")
    else:
        print("\nOK ASSEMBLY DAG: No missing edges!")

    removed_by_assembly = neuron_set - assembly_set
    if removed_by_assembly:
        print("\n[TARGET] EDGES REMOVED BY ASSEMBLY COMPRESSION:")
        for source, target in sorted(removed_by_assembly):
            is_spurious = (source, target) in neuron_spurious
            status = (
                "(GOOD - removed spurious edge!)"
                if is_spurious
                else "(BAD - removed true edge)"
            )
            print(f"  {source} -> {target}  {status}")

    neuron_comparison = compare_dags(ground_truth_edges, neuron_edges, var_names)
    assembly_comparison = compare_dags(ground_truth_edges, assembly_edges, var_names)
    neuron_to_assembly = compare_dags(neuron_edges, assembly_edges, var_names)

    print("\n" + "=" * 70)
    print("QUANTITATIVE METRICS")
    print("=" * 70)
    print("\n[1] NEURON DAG vs GROUND TRUTH:")
    print(f"  Precision: {neuron_comparison['precision']:.3f}")
    print(f"  Recall: {neuron_comparison['recall']:.3f}")
    print(f"  F1 Score: {neuron_comparison['f1']:.3f}")
    print(
        f"  Skeleton F1: {neuron_comparison['skel_f1']:.3f} "
        f"(P={neuron_comparison['skel_precision']:.3f}, R={neuron_comparison['skel_recall']:.3f})"
    )

    print("\n[2] ASSEMBLY DAG vs GROUND TRUTH:")
    print(f"  Precision: {assembly_comparison['precision']:.3f}")
    print(f"  Recall: {assembly_comparison['recall']:.3f}")
    print(f"  F1 Score: {assembly_comparison['f1']:.3f}")
    print(
        f"  Skeleton F1: {assembly_comparison['skel_f1']:.3f} "
        f"(P={assembly_comparison['skel_precision']:.3f}, R={assembly_comparison['skel_recall']:.3f})"
    )

    print("\n[3] ASSEMBLY DAG vs NEURON DAG:")
    print(f"  Agreement: {neuron_to_assembly['precision']:.3f}")

    print("\n" + "=" * 70)
    print("PRESERVATION ASSESSMENT:")
    print("=" * 70)

    preservation_score = (
        assembly_comparison["f1"] + neuron_to_assembly["precision"]
    ) / 2

    if preservation_score > 0.8:
        status = "EXCELLENT"
        symbol = "OKOK"
    elif preservation_score > 0.6:
        status = "GOOD"
        symbol = "OK"
    elif preservation_score > 0.4:
        status = "MODERATE"
        symbol = "~"
    else:
        status = "POOR"
        symbol = "X"

    print(f"  {symbol} Overall Preservation: {preservation_score:.3f} - {status}")
    print(f"  Assembly F1 vs GT: {assembly_comparison['f1']:.3f}")
    print(f"  Assembly-Neuron Agreement: {neuron_to_assembly['precision']:.3f}")

    return {
        "neuron_vs_gt": neuron_comparison,
        "assembly_vs_gt": assembly_comparison,
        "assembly_vs_neuron": neuron_to_assembly,
        "preservation_score": preservation_score,
    }
