"""Causal DAG Validation Runner.

Full method name: causal structure preservation pipeline from the paper.

How it works: take one SCM dataset and ground-truth DAG; encode observations as
neural activity; compute the raw-neuron readout as the within-run baseline; run
a stand-alone assembly-formation stage; extract the assembly-level readout; run
PC or GES under matched settings on both readouts; compare both recovered DAGs
against ground truth; compute MI-based information preservation; optionally
print observational diagnostics. This file orchestrates methods, while individual
methods live in their own modules.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .representation.brain import Brain
from .encoding.bernoulli import encode_bernoulli_dataframe
from .encoding.deterministic_k import build_deterministic_k_map, encode_deterministic_k_dataframe
from .representation.assembly_feature_extraction import extract_assembly_features
from .representation.assembly_formation import form_assemblies
from .representation.information_preservation_mi import (
    compute_mi_matrix,
    validate_information_preservation,
)
from .discovery.ges import run_ges_algorithm
from .discovery.pc import run_pc_algorithm
from .validation.observational.dag_comparison import print_dag_comparison_report
from .visualization.dag_plotting import visualize_three_dags

Edge = Tuple[str, str]

def _make_shuffled_mapping(var_names: Sequence[str], *, seed: int) -> Dict[str, str]:
    """Return a deterministic non-identity permutation mapping of var->area."""
    names = list(var_names)
    if len(names) < 2:
        return {n: n for n in names}
    rng = np.random.default_rng(seed=seed)
    for _ in range(100):
        perm = list(rng.permutation(names))
        if perm != names:
            return {src: dst for src, dst in zip(names, perm)}
    # Fallback: rotate by 1 (guaranteed non-identity for len>=2)
    perm = names[1:] + names[:1]
    return {src: dst for src, dst in zip(names, perm)}

def _extract_variable_features(
    neural_data: np.ndarray,
    var_names: Sequence[str],
    *,
    neurons_per_var: int,
    stimulus_indices: dict[str, dict[object, np.ndarray]] | None = None,
    positive_values_map=None,
    deterministic_k_readout_mode: str = "positive_subset_mean",
    pca_per_var: bool = False,
) -> pd.DataFrame:
    """Extract one scalar feature per variable per sample.

    Default behavior uses pool-mean firing rate. When `pca_per_var` is True,
    each variable is reduced to the first principal component of its local
    neuron population (deterministic sign) for higher variance signal.

    For deterministic-k stimulus-set encoding, pool-mean is typically uninformative
    (it is constant when each value activates exactly k neurons), so we instead
    measure activation on the stimulus-neuron subset corresponding to the
    configured positive value(s).
    """

    features: Dict[str, np.ndarray] = {}

    for var_idx, var_name in enumerate(var_names):
        start_idx = var_idx * neurons_per_var
        end_idx = start_idx + neurons_per_var
        activations = neural_data[:, start_idx:end_idx]

        if pca_per_var:
            centered = activations - activations.mean(axis=0, keepdims=True)
            try:
                _u, s, vh = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                vh = None

            if vh is not None and s.size > 0 and np.isfinite(vh[0]).all():
                v0 = vh[0]
                if v0.sum() < 0:
                    v0 = -v0
                features[var_name] = centered @ v0
                continue

            features[var_name] = activations.mean(axis=1)
            continue

        if (
            deterministic_k_readout_mode == "positive_subset_mean"
            and stimulus_indices is not None
            and var_name in stimulus_indices
        ):
            pos_vals = None
            if positive_values_map and var_name in positive_values_map:
                pos_vals = list(positive_values_map[var_name])

            if pos_vals:
                local = np.concatenate(
                    [stimulus_indices[var_name][v] for v in pos_vals if v in stimulus_indices[var_name]],
                    axis=0,
                )
                if local.size > 0:
                    features[var_name] = activations[:, local].mean(axis=1)
                    continue

            # Fallback: choose a deterministic "positive" value if not specified
            vals = sorted(stimulus_indices[var_name].keys(), key=lambda x: str(x))
            if vals:
                local = stimulus_indices[var_name][vals[-1]]
                features[var_name] = activations[:, local].mean(axis=1)
                continue

        features[var_name] = activations.mean(axis=1)

    return pd.DataFrame(features)


def _variance_guard(
    df: pd.DataFrame,
    var_names: list[str],
    *,
    eps: float,
) -> tuple[list[str], list[str]]:
    "Return (kept_vars, dropped_vars) based on near-constant/non-finite columns."
    dropped: list[str] = []
    for v in var_names:
        x = df[v].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            dropped.append(v)
            continue
        if float(np.nanstd(x)) < eps:
            dropped.append(v)

    kept = [v for v in var_names if v not in dropped]
    return kept, dropped


def _bootstrap_stable_edges(
    *,
    data_df: pd.DataFrame,
    var_names: list[str],
    method: str,
    alpha: float,
    strict: bool,
    n_bootstrap: int,
    sample_frac: float,
    threshold: float,
    seed: int,
) -> list[Edge]:
    """Bootstrap stability selection for causal discovery.

    Counts edge *skeleton* frequency across bootstrap resamples, then orients
    stable edges using majority direction (ties broken deterministically).

    Returns a list of directed edges.
    """
    from collections import Counter

    method_norm = method.strip().lower()
    if method_norm not in {"pc", "ges"}:
        raise ValueError(f"Unknown method={method!r}; expected 'pc' or 'ges'")

    if n_bootstrap <= 0:
        return []

    n = len(data_df)
    if n == 0 or len(var_names) < 2:
        return []

    sample_size = max(2, int(round(n * float(sample_frac))))
    if sample_size > n:
        sample_size = n

    # Base run (used only for deterministic tie-breaking)
    if method_norm == "pc":
        base_edges, _ = run_pc_algorithm(data_df, var_names, alpha=alpha, strict=strict)
    else:
        base_edges, _ = run_ges_algorithm(data_df, var_names, strict=strict)

    base_set = set(base_edges)

    skel_counts: Counter[frozenset[str]] = Counter()
    dir_counts: Counter[tuple[str, str]] = Counter()

    rng = np.random.default_rng(seed=seed)

    for _b in range(n_bootstrap):
        idx = rng.integers(0, n, size=sample_size)
        df_b = data_df.iloc[idx]

        if method_norm == "pc":
            edges, _ = run_pc_algorithm(df_b, var_names, alpha=alpha, strict=strict)
        else:
            edges, _ = run_ges_algorithm(df_b, var_names, strict=strict)

        for u, v in edges:
            if u == v:
                continue
            skel_counts[frozenset((u, v))] += 1
            dir_counts[(u, v)] += 1

    stable_undirected = [
        und for und, c in skel_counts.items() if (c / n_bootstrap) >= threshold
    ]

    oriented: list[Edge] = []
    for und in stable_undirected:
        a, b = sorted(tuple(und), key=lambda x: str(x))
        ab = dir_counts.get((a, b), 0)
        ba = dir_counts.get((b, a), 0)

        if ab > ba:
            oriented.append((a, b))
        elif ba > ab:
            oriented.append((b, a))
        else:
            # Tie: prefer base orientation if present, else deterministic a->b
            if (a, b) in base_set:
                oriented.append((a, b))
            elif (b, a) in base_set:
                oriented.append((b, a))
            else:
                oriented.append((a, b))

    oriented.sort()
    return oriented


def run_causal_dag_validation(
    df: pd.DataFrame,
    var_names: Sequence[str],
    ground_truth_edges: Sequence[Edge],
    *,
    neurons_per_var: int = 1666,
    assembly_k: int = 100,
    n_train: int = 200,
    n_presentations: int = 3,
    beta: float = 0.05,
    seed: int = 42,
    alpha_pc: float = 0.05,
    method: str = "pc",
    assembly_method: str | None = None,
    positive_values_map=None,
    positive_prob: float = 0.30,
    negative_prob: float = 0.10,
    deterministic_k_encoding: bool = False,
    stimulus_k: int | None = None,
    deterministic_k_readout_mode: str = "positive_subset_mean",
    pca_per_var: bool = False,
    deterministic_k_step: int = 10,
    jitter_std: float = 0.0,
    strict_causal_discovery: bool = True,
    strict_neuron_causal_discovery: bool | None = None,
    strict_assembly_causal_discovery: bool | None = None,
    skip_causal_discovery: bool = False,
    skip_neuron_dag: bool = False,
    shuffled_mapping_control: bool = False,
    disable_plasticity_control: bool = False,
    visualize_save_path: str | None = None,
    # Assembly-only robustness knobs (used when assembly features are degenerate)
    assembly_variance_guard_eps: float = 1e-6,
    assembly_variance_guard_mode: str = "jitter",  # "jitter" or "drop"
    assembly_variance_guard_jitter_std: float = 1e-3,
    assembly_stability_auto_on_degenerate: bool = True,
    assembly_stability_selection: bool = False,
    assembly_stability_n_bootstrap: int = 20,
    assembly_stability_frac: float = 0.80,
    assembly_stability_threshold: float = 0.70,
    assembly_stability_seed_offset: int = 4242,
    brain_max_support_ratio: float | None = None,
):
    """Run the paper-aligned causal-structure preservation validation.

    strict_causal_discovery=True makes causal discovery failures raise
    instead of being silently converted into empty-edge graphs.
    """

    print()
    print("=" * 70)
    print("CAUSAL STRUCTURE PRESERVATION PIPELINE")
    print("=" * 70)
    print(f"  Dataset: {len(df)} samples, {len(var_names)} variables")
    print(f"  Ground Truth Edges: {len(ground_truth_edges)}")
    print("  Paper flow: Stage I SCM observations (input) -> Stage II neural encoding")
    print("              -> Stage III raw-neuron baseline readout")
    print("              -> Stage IV assembly formation")
    print("              -> Stage V assembly readout -> Stage VI PC/GES graph recovery")

    # =========================================================================
    # STAGE II: Neural Encoding
    # =========================================================================
    print()
    print("[STAGE II] Neural Encoding...")
    if deterministic_k_encoding:
        allowed = {"positive_subset_mean", "pool_mean"}
        if deterministic_k_readout_mode not in allowed:
            raise ValueError(
                f"Invalid deterministic_k_readout_mode={deterministic_k_readout_mode!r}; expected one of {sorted(allowed)}"
            )
    neural_data_for_neurons = None
    neural_data_for_brain = None
    stimulus_indices = None

    if deterministic_k_encoding:
        k_stim = stimulus_k if stimulus_k is not None else assembly_k
        stimulus_k_map = build_deterministic_k_map(
            df,
            list(var_names),
            base_k=int(k_stim),
            k_step=int(deterministic_k_step),
        )
        neural_data_for_neurons, stimulus_indices = encode_deterministic_k_dataframe(
            df,
            list(var_names),
            neurons_per_var=neurons_per_var,
            stimulus_k_map=stimulus_k_map,
            seed=seed,
        )
        neural_data_for_brain = neural_data_for_neurons
        print(
            f"  Encoded {len(df)} samples to {neural_data_for_neurons.shape[1]:,} neurons "
            f"(deterministic-k, base={k_stim}, step={deterministic_k_step})"
        )
    else:
        neural_data_for_neurons = encode_bernoulli_dataframe(
            df,
            list(var_names),
            neurons_per_var=neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            seed=seed,
        )
        neural_data_for_brain = neural_data_for_neurons
        print(f"  Encoded {len(df)} samples to {neural_data_for_neurons.shape[1]:,} neurons (Bernoulli)")

    # =====================================================================
    # STAGE III: Raw-neuron baseline readout
    # =====================================================================
    print()
    print("[STAGE III] Raw-neuron baseline readout...")
    neuron_df = _extract_variable_features(
        neural_data_for_neurons,
        var_names,
        neurons_per_var=neurons_per_var,
        stimulus_indices=stimulus_indices if deterministic_k_encoding else None,
        positive_values_map=positive_values_map,
        deterministic_k_readout_mode=deterministic_k_readout_mode,
        pca_per_var=pca_per_var,
    )
    print(f"  Raw-neuron readout features extracted: {neuron_df.shape}")

    # =========================================================================
    # STAGE IV: Assembly formation
    # =========================================================================
    print()
    print("[STAGE IV] Assembly Formation (Papadimitriou Brain)...")
    brain = Brain(p=beta, save_size=True, save_winners=True, seed=seed, max_support_ratio=brain_max_support_ratio if brain_max_support_ratio is not None else 1.0)
    if disable_plasticity_control:
        brain.disable_plasticity = True
        print("  Control: plasticity disabled (Hebbian updates off)")

    if shuffled_mapping_control:
        target_area_by_var_name = _make_shuffled_mapping(list(var_names), seed=seed)
        print("  Control: shuffled stimulus->area mapping enabled")
        for src, dst in target_area_by_var_name.items():
            print(f"    {src} -> {dst}")
    else:
        target_area_by_var_name = {v: v for v in var_names}
    for var_name in var_names:
        brain.add_area(var_name, n=neurons_per_var, k=assembly_k, beta=beta)

    brain = form_assemblies(
        brain,
        neural_data_for_brain,
        list(var_names),
        neurons_per_var,
        n_train=n_train,
        n_presentations=n_presentations,
        target_area_by_var_name=target_area_by_var_name,
        shuffle_seed=seed,
    )

    effective_n_train = min(n_train, neural_data_for_brain.shape[0])
    print(f"  Training: {effective_n_train} samples/var, {n_presentations} rounds -> {effective_n_train * n_presentations} projections per area")
    print(f"  Assembly params: k={assembly_k} winners/projection; w=support size (union of winners)")

    for var_name in var_names:
        area = brain.area_by_name[var_name]
        winners_now = len(area.winners) if area.winners else 0
        print(f"    {var_name}: k={area.k}, winners_now={winners_now}, support_w={area.w}/{area.n}")

    # =====================================================================
    # STAGE V: Assembly-level readout
    # =====================================================================
    print()
    print("[STAGE V] Assembly-level readout...")
    source_var_by_target_area_name = {dst: src for src, dst in target_area_by_var_name.items()}

    assembly_features = extract_assembly_features(
        neural_data_for_brain,
        brain,
        list(var_names),
        neurons_per_var,
        source_var_by_target_area_name=source_var_by_target_area_name,
    )

    assembly_features_base = {k: v for k, v in assembly_features.items() if "_x_" not in k}
    assembly_df = pd.DataFrame(assembly_features_base)
    print(f"  Assembly readout features extracted: {assembly_df.shape}")

    if jitter_std > 0:
        rng = np.random.default_rng(seed=seed + 999)
        neuron_df[:] = neuron_df.to_numpy() + rng.normal(0.0, jitter_std, neuron_df.shape)
        assembly_df[:] = assembly_df.to_numpy() + rng.normal(0.0, jitter_std, assembly_df.shape)
        print(f"  Applied deterministic jitter std={jitter_std} to features")

    obs_diagnostics = None

    if skip_causal_discovery:
        total_neurons = sum(brain.area_by_name[var].n for var in var_names)
        total_assembly_neurons = sum(brain.area_by_name[var].w for var in var_names)
        compression_ratio = total_neurons / total_assembly_neurons if total_assembly_neurons > 0 else float('inf')

        return {
            'neuron_df': neuron_df,
            'assembly_df': assembly_df,
            'brain': brain,
            'var_names': list(var_names),
            'compression': {
                'original_neurons': total_neurons,
                'assembly_neurons': total_assembly_neurons,
                'ratio': compression_ratio,
            },
        }

    # =========================================================================
    # STAGE VI: Matched causal discovery on raw-neuron and assembly readouts
    # =========================================================================
    print()
    print("[STAGE VI] Matched causal discovery...")

    strict_neuron = (strict_causal_discovery if strict_neuron_causal_discovery is None else bool(strict_neuron_causal_discovery))
    strict_assembly = (strict_causal_discovery if strict_assembly_causal_discovery is None else bool(strict_assembly_causal_discovery))

    method_norm = method.strip().lower()
    if method_norm not in {"pc", "ges"}:
        raise ValueError(f"Unknown method={method!r}; expected 'pc' or 'ges'")

    assembly_method_norm = (assembly_method.strip().lower() if assembly_method is not None else method_norm)
    if assembly_method_norm not in {"pc", "ges"}:
        raise ValueError(f"Unknown assembly_method={assembly_method!r}; expected 'pc' or 'ges'")

    print()
    print("  [DAG 1] Ground Truth:")
    for source, target in ground_truth_edges:
        print(f"    {source} -> {target}")

    if skip_neuron_dag:
        neuron_edges, neuron_graph = [], None
        print()
        print("  [DAG 2] Skipped neuron causal discovery (skip_neuron_dag=True)")
    else:
        print()
        print(f"  [DAG 2] Running {method_norm.upper()} on raw-neuron readout features...")
        if method_norm == "pc":
            neuron_edges, neuron_graph = run_pc_algorithm(
                neuron_df,
                list(var_names),
                alpha=alpha_pc,
                strict=strict_neuron,
            )
        else:
            neuron_edges, neuron_graph = run_ges_algorithm(
                neuron_df,
                list(var_names),
                strict=strict_neuron,
            )

        print(f"    Discovered {len(neuron_edges)} edges")
        for source, target in neuron_edges:
            print(f"    {source} -> {target}")

    print()
    print(f"  [DAG 3] Running {assembly_method_norm.upper()} on assembly readout features...")

    assembly_all_names = list(var_names)
    _kept_names, assembly_dropped_vars = _variance_guard(
        assembly_df,
        assembly_all_names,
        eps=assembly_variance_guard_eps,
    )

    guard_mode = str(assembly_variance_guard_mode).strip().lower()
    if guard_mode not in {"jitter", "drop"}:
        raise ValueError(f"Unknown assembly_variance_guard_mode={assembly_variance_guard_mode!r}; expected 'jitter' or 'drop'")

    if assembly_dropped_vars:
        if guard_mode == "drop":
            print(
                "    Variance-guard: dropping "
                f"{len(assembly_dropped_vars)} near-constant/non-finite assembly feature(s) "
                f"(eps={assembly_variance_guard_eps:g}): "
                + ", ".join(assembly_dropped_vars)
            )
        else:
            print(
                "    Variance-guard: detected "
                f"{len(assembly_dropped_vars)} near-constant/non-finite assembly feature(s) "
                f"(eps={assembly_variance_guard_eps:g}); applying deterministic jitter std={assembly_variance_guard_jitter_std:g}"
                + ": "
                + ", ".join(assembly_dropped_vars)
            )
    else:
        print(f"    Variance-guard: OK (eps={assembly_variance_guard_eps:g})")

    if guard_mode == "drop":
        assembly_guarded_names = _kept_names
        assembly_df_for_cd = assembly_df[assembly_guarded_names].copy() if assembly_guarded_names else assembly_df.iloc[:, :0].copy()
    else:
        # Keep all variables, but add small deterministic jitter to degenerate columns
        assembly_guarded_names = assembly_all_names
        assembly_df_for_cd = assembly_df[assembly_guarded_names].copy()
        if assembly_dropped_vars and assembly_variance_guard_jitter_std > 0:
            rng = np.random.default_rng(seed=seed + 20202)
            for col in assembly_dropped_vars:
                assembly_df_for_cd[col] = assembly_df_for_cd[col].to_numpy(dtype=float) + rng.normal(
                    0.0, float(assembly_variance_guard_jitter_std), size=len(assembly_df_for_cd)
                )

    def _run_one(df_for_cd: pd.DataFrame, names: list[str]) -> tuple[list[Edge], object | None]:
        if len(names) < 2:
            return [], None

        if assembly_method_norm == "pc":
            edges, graph = run_pc_algorithm(
                df_for_cd,
                names,
                alpha=alpha_pc,
                strict=strict_assembly,
            )

            # PC + Fisher-Z can fail on singular correlation matrices.
            # If it still fails even after variance-guard, retry once with deterministic jitter.
            if graph is None:
                retry_jitter = jitter_std if jitter_std > 0 else 0.05
                rng = np.random.default_rng(seed=seed + 12345)
                arr = df_for_cd.to_numpy() + rng.normal(0.0, retry_jitter, df_for_cd.shape)
                df_retry = pd.DataFrame(arr, columns=df_for_cd.columns)
                print(f"    Retrying PC on assembly features with deterministic jitter std={retry_jitter}")
                edges, graph = run_pc_algorithm(
                    df_retry,
                    names,
                    alpha=alpha_pc,
                    strict=strict_assembly,
                )

            return edges, graph

        edges, graph = run_ges_algorithm(df_for_cd, names, strict=strict_assembly)
        return edges, graph

    assembly_edges, assembly_graph = _run_one(assembly_df_for_cd, assembly_guarded_names)

    assembly_stability_used = False
    assembly_stability_details = None

    need_stability = assembly_stability_selection or (
        assembly_stability_auto_on_degenerate and bool(assembly_dropped_vars)
    )

    if need_stability:
        if len(assembly_guarded_names) < 2:
            print("    Stability selection: skipped (too few non-degenerate variables)")
        else:
            assembly_stability_used = True
            print(
                "    Stability selection: "
                f"bootstrapping n={assembly_stability_n_bootstrap}, "
                f"frac={assembly_stability_frac:.2f}, "
                f"threshold={assembly_stability_threshold:.2f}"
            )

            assembly_edges = _bootstrap_stable_edges(
                data_df=assembly_df_for_cd,
                var_names=assembly_guarded_names,
                method=assembly_method_norm,
                alpha=alpha_pc,
                strict=strict_assembly,
                n_bootstrap=assembly_stability_n_bootstrap,
                sample_frac=assembly_stability_frac,
                threshold=assembly_stability_threshold,
                seed=seed + int(assembly_stability_seed_offset),
            )

            assembly_stability_details = {
                "method": assembly_method_norm,
                "n_bootstrap": int(assembly_stability_n_bootstrap),
                "sample_frac": float(assembly_stability_frac),
                "threshold": float(assembly_stability_threshold),
                "seed": int(seed + int(assembly_stability_seed_offset)),
            }

    print(f"    Discovered {len(assembly_edges)} edges")
    for source, target in assembly_edges:
        print(f"    {source} -> {target}")

    # =========================================================================
    # EVALUATION: Compare recovered graphs and MI preservation
    # =========================================================================
    print()
    print("[EVALUATION] Comparing recovered graphs and MI preservation...")
    if skip_neuron_dag:
        gt_set = set(ground_truth_edges)
        asm_set = set(assembly_edges)
        correct = asm_set & gt_set
        spurious = asm_set - gt_set
        missing = gt_set - asm_set
        prec = len(correct) / len(asm_set) if asm_set else 0.0
        rec = len(correct) / len(gt_set) if gt_set else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        comparison_results = {
            "ground_truth_edges": list(gt_set),
            "assembly_vs_gt": {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "correct": sorted(correct),
                "spurious": sorted(spurious),
                "missing": sorted(missing),
            },
            "neuron_vs_gt": None,
            "assembly_vs_neuron": None,
            "preservation_score": None,
        }
        print("  (Neuron DAG skipped; reporting assembly vs ground truth only)")
        print(f"  Assembly precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
    else:
        comparison_results = print_dag_comparison_report(
            list(ground_truth_edges),
            neuron_edges,
            assembly_edges,
            list(var_names),
        )

    if visualize_save_path:
        visualize_three_dags(
            list(ground_truth_edges),
            neuron_edges,
            assembly_edges,
            list(var_names),
            save_path=visualize_save_path,
        )
    mi_neurons = compute_mi_matrix({c: neuron_df[c].to_numpy() for c in var_names}, list(var_names))
    mi_assemblies = compute_mi_matrix({c: assembly_df[c].to_numpy() for c in var_names}, list(var_names))
    validation = validate_information_preservation(mi_neurons, mi_assemblies, threshold=0.1)

    total_neurons = sum(brain.area_by_name[var].n for var in var_names)
    total_assembly_neurons = sum(brain.area_by_name[var].w for var in var_names)
    compression_ratio = total_neurons / total_assembly_neurons if total_assembly_neurons > 0 else float('inf')

    return {
        'ground_truth_edges': list(ground_truth_edges),
        'neuron_edges': neuron_edges,
        'assembly_edges': assembly_edges,
        'neuron_df': neuron_df,
        'assembly_df': assembly_df,
        'neuron_graph': neuron_graph,
        'assembly_graph': assembly_graph,
        'comparison': comparison_results,
        'mi_neurons': mi_neurons,
        'mi_assemblies': mi_assemblies,
        'validation': validation,
        'brain': brain,
        'var_names': list(var_names),
        'observational_diagnostics': obs_diagnostics,
        'assembly_causal_discovery': {
            'variance_guard_eps': float(assembly_variance_guard_eps),
            'variance_guard_mode': str(assembly_variance_guard_mode),
            'variance_guard_jitter_std': float(assembly_variance_guard_jitter_std),
            'dropped_vars': list(assembly_dropped_vars),
            'guarded_vars': list(assembly_guarded_names),
            'stability_used': bool(assembly_stability_used),
            'stability': assembly_stability_details,
        },
        'controls': {
            'shuffled_mapping': bool(shuffled_mapping_control),
            'disable_plasticity': bool(disable_plasticity_control),
            'target_area_by_var_name': dict(target_area_by_var_name),
            'deterministic_k_encoding': bool(deterministic_k_encoding),
            'stimulus_k': int(stimulus_k if stimulus_k is not None else assembly_k) if deterministic_k_encoding else None,
            'deterministic_k_step': int(deterministic_k_step) if deterministic_k_encoding else None,
        },
        'compression': {
            'original_neurons': total_neurons,
            'assembly_neurons': total_assembly_neurons,
            'ratio': compression_ratio,
        },
    }








