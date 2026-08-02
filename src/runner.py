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
from .evaluation.dag_evaluation import print_dag_comparison_report
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








