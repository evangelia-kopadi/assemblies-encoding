"""Causal DAG Validation Runner.

Causal structure preservation pipeline.

How it works: take one SCM dataset and ground-truth DAG; encode observations as
neural activity; compute the raw-neuron readout as the within-run baseline; run
a stand-alone assembly-formation stage; extract the assembly-level readout; run
PC or GES under matched settings on both readouts; compare both recovered DAGs
against ground truth; compute MI-based information preservation; optionally
log observational diagnostics. This file orchestrates methods, while individual
methods live in their own modules.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_configuration import configure_logging, get_logger

from .representation.brain import Brain
from .encoding.bernoulli import encode_bernoulli_dataframe
from .encoding.deterministic_k import (
    build_deterministic_k_map,
    encode_deterministic_k_dataframe,
)
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

LOGGER = get_logger(__name__)

Edge = Tuple[str, str]


@dataclass(frozen=True)
class EncodingStageResult:
    neural_data_for_neurons: np.ndarray
    neural_data_for_brain: np.ndarray
    stimulus_indices: dict[str, dict[object, np.ndarray]] | None
    stimulus_k_used: int | None


@dataclass(frozen=True)
class AssemblyStageResult:
    brain: Brain
    target_area_by_var_name: dict[str, str]


@dataclass(frozen=True)
class CausalDiscoveryOptions:
    method: str
    assembly_method: str
    strict_neuron: bool
    strict_assembly: bool
    raise_on_error: bool


@dataclass(frozen=True)
class AssemblyFeatureGuardResult:
    df_for_cd: pd.DataFrame
    guarded_names: list[str]
    dropped_vars: list[str]
    summary: dict[str, Any]


@dataclass(frozen=True)
class DiscoveryStageResult:
    neuron_edges: list[Edge]
    neuron_graph: object | None
    assembly_edges: list[Edge]
    assembly_graph: object | None
    assembly_causal_discovery: dict[str, Any]


@dataclass(frozen=True)
class CausalDAGValidationResult(Mapping[str, Any]):
    """Typed runner result with dict-style compatibility for existing scripts."""

    _data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dict copy for serialization or legacy callers."""
        return dict(self._data)

    @property
    def ground_truth_edges(self) -> list[Edge]:
        return self._data["ground_truth_edges"]

    @property
    def neuron_edges(self) -> list[Edge]:
        return self._data["neuron_edges"]

    @property
    def assembly_edges(self) -> list[Edge]:
        return self._data["assembly_edges"]

    @property
    def neuron_df(self) -> pd.DataFrame:
        return self._data["neuron_df"]

    @property
    def assembly_df(self) -> pd.DataFrame:
        return self._data["assembly_df"]

    @property
    def comparison(self) -> dict[str, Any]:
        return self._data["comparison"]

    @property
    def compression(self) -> dict[str, Any]:
        return self._data["compression"]

    @property
    def brain(self) -> Brain:
        return self._data["brain"]

    @property
    def var_names(self) -> list[str]:
        return self._data["var_names"]


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
                    [
                        stimulus_indices[var_name][v]
                        for v in pos_vals
                        if v in stimulus_indices[var_name]
                    ],
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
    raise_on_error: bool,
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
        base_edges, _ = run_pc_algorithm(
            data_df,
            var_names,
            alpha=alpha,
            strict=strict,
            raise_on_error=raise_on_error,
        )
    else:
        base_edges, _ = run_ges_algorithm(
            data_df,
            var_names,
            strict=strict,
            raise_on_error=raise_on_error,
        )

    base_set = set(base_edges)

    skel_counts: Counter[frozenset[str]] = Counter()
    dir_counts: Counter[tuple[str, str]] = Counter()

    rng = np.random.default_rng(seed=seed)

    for _b in range(n_bootstrap):
        idx = rng.integers(0, n, size=sample_size)
        df_b = data_df.iloc[idx]

        if method_norm == "pc":
            edges, _ = run_pc_algorithm(
                df_b,
                var_names,
                alpha=alpha,
                strict=strict,
                raise_on_error=raise_on_error,
            )
        else:
            edges, _ = run_ges_algorithm(
                df_b,
                var_names,
                strict=strict,
                raise_on_error=raise_on_error,
            )

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


def _print_pipeline_header(
    df: pd.DataFrame,
    var_names: Sequence[str],
    ground_truth_edges: Sequence[Edge],
) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("CAUSAL STRUCTURE PRESERVATION PIPELINE")
    LOGGER.info("=" * 70)
    LOGGER.info(f"  Dataset: {len(df)} samples, {len(var_names)} variables")
    LOGGER.info(f"  Ground Truth Edges: {len(ground_truth_edges)}")
    LOGGER.info("  Paper flow: Stage I SCM observations (input) -> Stage II neural encoding")
    LOGGER.info("              -> Stage III raw-neuron baseline readout")
    LOGGER.info("              -> Stage IV assembly formation")
    LOGGER.info("              -> Stage V assembly readout -> Stage VI PC/GES graph recovery")


def _encode_observations(
    *,
    df: pd.DataFrame,
    var_names: list[str],
    neurons_per_var: int,
    assembly_k: int,
    positive_values_map: Any,
    positive_prob: float,
    negative_prob: float,
    deterministic_k_encoding: bool,
    stimulus_k: int | None,
    deterministic_k_readout_mode: str,
    deterministic_k_step: int,
    seed: int,
) -> EncodingStageResult:
    LOGGER.info("")
    LOGGER.info("[STAGE II] Neural Encoding...")

    if deterministic_k_encoding:
        allowed = {"positive_subset_mean", "pool_mean"}
        if deterministic_k_readout_mode not in allowed:
            raise ValueError(
                f"Invalid deterministic_k_readout_mode={deterministic_k_readout_mode!r}; expected one of {sorted(allowed)}"
            )

        stimulus_k_used = stimulus_k if stimulus_k is not None else assembly_k
        stimulus_k_map = build_deterministic_k_map(
            df,
            var_names,
            base_k=int(stimulus_k_used),
            k_step=int(deterministic_k_step),
        )
        neural_data, stimulus_indices = encode_deterministic_k_dataframe(
            df,
            var_names,
            neurons_per_var=neurons_per_var,
            stimulus_k_map=stimulus_k_map,
            seed=seed,
        )
        LOGGER.info(
            f"  Encoded {len(df)} samples to {neural_data.shape[1]:,} neurons "
            f"(deterministic-k, base={stimulus_k_used}, step={deterministic_k_step})"
        )
        return EncodingStageResult(
            neural_data_for_neurons=neural_data,
            neural_data_for_brain=neural_data,
            stimulus_indices=stimulus_indices,
            stimulus_k_used=int(stimulus_k_used),
        )

    neural_data = encode_bernoulli_dataframe(
        df,
        var_names,
        neurons_per_var=neurons_per_var,
        positive_values_map=positive_values_map,
        positive_prob=positive_prob,
        negative_prob=negative_prob,
        seed=seed,
    )
    LOGGER.info(f"  Encoded {len(df)} samples to {neural_data.shape[1]:,} neurons (Bernoulli)")
    return EncodingStageResult(
        neural_data_for_neurons=neural_data,
        neural_data_for_brain=neural_data,
        stimulus_indices=None,
        stimulus_k_used=None,
    )


def _extract_neuron_readout(
    *,
    encoded: EncodingStageResult,
    var_names: list[str],
    neurons_per_var: int,
    positive_values_map: Any,
    deterministic_k_encoding: bool,
    deterministic_k_readout_mode: str,
    pca_per_var: bool,
) -> pd.DataFrame:
    LOGGER.info("")
    LOGGER.info("[STAGE III] Raw-neuron baseline readout...")
    neuron_df = _extract_variable_features(
        encoded.neural_data_for_neurons,
        var_names,
        neurons_per_var=neurons_per_var,
        stimulus_indices=encoded.stimulus_indices if deterministic_k_encoding else None,
        positive_values_map=positive_values_map,
        deterministic_k_readout_mode=deterministic_k_readout_mode,
        pca_per_var=pca_per_var,
    )
    LOGGER.info(f"  Raw-neuron readout features extracted: {neuron_df.shape}")
    return neuron_df


def _train_assembly_stage(
    *,
    encoded: EncodingStageResult,
    var_names: list[str],
    neurons_per_var: int,
    assembly_k: int,
    n_train: int,
    n_presentations: int,
    beta: float,
    seed: int,
    shuffled_mapping_control: bool,
    disable_plasticity_control: bool,
    brain_max_support_ratio: float | None,
) -> AssemblyStageResult:
    LOGGER.info("")
    LOGGER.info("[STAGE IV] Assembly Formation (Neural Assemblies Brain)...")

    brain = Brain(
        p=beta,
        save_size=True,
        save_winners=True,
        seed=seed,
        max_support_ratio=(
            brain_max_support_ratio if brain_max_support_ratio is not None else 1.0
        ),
    )
    if disable_plasticity_control:
        brain.disable_plasticity = True
        LOGGER.info("  Control: plasticity disabled (Hebbian updates off)")

    if shuffled_mapping_control:
        target_area_by_var_name = _make_shuffled_mapping(var_names, seed=seed)
        LOGGER.info("  Control: shuffled stimulus->area mapping enabled")
        for src, dst in target_area_by_var_name.items():
            LOGGER.info(f"    {src} -> {dst}")
    else:
        target_area_by_var_name = {v: v for v in var_names}

    for var_name in var_names:
        brain.add_area(var_name, n=neurons_per_var, k=assembly_k, beta=beta)

    brain = form_assemblies(
        brain,
        encoded.neural_data_for_brain,
        var_names,
        neurons_per_var,
        n_train=n_train,
        n_presentations=n_presentations,
        target_area_by_var_name=target_area_by_var_name,
        shuffle_seed=seed,
    )

    effective_n_train = min(n_train, encoded.neural_data_for_brain.shape[0])
    LOGGER.info(
        f"  Training: {effective_n_train} samples/var, {n_presentations} rounds -> {effective_n_train * n_presentations} projections per area"
    )
    LOGGER.info(
        f"  Assembly params: k={assembly_k} winners/projection; w=support size (union of winners)"
    )

    for var_name in var_names:
        area = brain.area_by_name[var_name]
        winners_now = len(area.winners) if area.winners else 0
        LOGGER.info(
            f"    {var_name}: k={area.k}, winners_now={winners_now}, support_w={area.w}/{area.n}"
        )

    return AssemblyStageResult(
        brain=brain,
        target_area_by_var_name=target_area_by_var_name,
    )


def _extract_assembly_readout(
    *,
    encoded: EncodingStageResult,
    assembly_stage: AssemblyStageResult,
    var_names: list[str],
    neurons_per_var: int,
) -> pd.DataFrame:
    LOGGER.info("")
    LOGGER.info("[STAGE V] Assembly-level readout...")

    source_var_by_target_area_name = {
        dst: src for src, dst in assembly_stage.target_area_by_var_name.items()
    }
    assembly_features = extract_assembly_features(
        encoded.neural_data_for_brain,
        assembly_stage.brain,
        var_names,
        neurons_per_var,
        source_var_by_target_area_name=source_var_by_target_area_name,
    )

    assembly_features_base = {
        k: v for k, v in assembly_features.items() if "_x_" not in k
    }
    assembly_df = pd.DataFrame(assembly_features_base)
    LOGGER.info(f"  Assembly readout features extracted: {assembly_df.shape}")
    return assembly_df


def _apply_feature_jitter(
    *,
    neuron_df: pd.DataFrame,
    assembly_df: pd.DataFrame,
    jitter_std: float,
    seed: int,
) -> None:
    if jitter_std <= 0:
        return

    rng = np.random.default_rng(seed=seed + 999)
    neuron_df[:] = neuron_df.to_numpy() + rng.normal(0.0, jitter_std, neuron_df.shape)
    assembly_df[:] = assembly_df.to_numpy() + rng.normal(
        0.0, jitter_std, assembly_df.shape
    )
    LOGGER.info(f"  Applied deterministic jitter std={jitter_std} to features")


def _compute_compression(
    *,
    brain: Brain,
    var_names: list[str],
) -> dict[str, float]:
    total_neurons = sum(brain.area_by_name[var].n for var in var_names)
    total_assembly_neurons = sum(brain.area_by_name[var].w for var in var_names)
    compression_ratio = (
        total_neurons / total_assembly_neurons
        if total_assembly_neurons > 0
        else float("inf")
    )
    return {
        "original_neurons": total_neurons,
        "assembly_neurons": total_assembly_neurons,
        "ratio": compression_ratio,
    }


def _normalize_causal_discovery_options(
    *,
    method: str,
    assembly_method: str | None,
    strict_causal_discovery: bool,
    strict_neuron_causal_discovery: bool | None,
    strict_assembly_causal_discovery: bool | None,
    raise_on_causal_discovery_error: bool,
) -> CausalDiscoveryOptions:
    strict_neuron = (
        strict_causal_discovery
        if strict_neuron_causal_discovery is None
        else bool(strict_neuron_causal_discovery)
    )
    strict_assembly = (
        strict_causal_discovery
        if strict_assembly_causal_discovery is None
        else bool(strict_assembly_causal_discovery)
    )

    method_norm = method.strip().lower()
    if method_norm not in {"pc", "ges"}:
        raise ValueError(f"Unknown method={method!r}; expected 'pc' or 'ges'")

    assembly_method_norm = (
        assembly_method.strip().lower() if assembly_method is not None else method_norm
    )
    if assembly_method_norm not in {"pc", "ges"}:
        raise ValueError(
            f"Unknown assembly_method={assembly_method!r}; expected 'pc' or 'ges'"
        )

    return CausalDiscoveryOptions(
        method=method_norm,
        assembly_method=assembly_method_norm,
        strict_neuron=bool(strict_neuron),
        strict_assembly=bool(strict_assembly),
        raise_on_error=bool(raise_on_causal_discovery_error),
    )


def _run_causal_discovery_on_features(
    *,
    method: str,
    data_df: pd.DataFrame,
    var_names: list[str],
    alpha_pc: float,
    strict: bool,
    raise_on_error: bool,
) -> tuple[list[Edge], object | None]:
    if len(var_names) < 2:
        return [], None

    if method == "pc":
        return run_pc_algorithm(
            data_df,
            var_names,
            alpha=alpha_pc,
            strict=strict,
            raise_on_error=raise_on_error,
        )
    return run_ges_algorithm(
        data_df,
        var_names,
        strict=strict,
        raise_on_error=raise_on_error,
    )


def _prepare_assembly_features_for_causal_discovery(
    *,
    assembly_df: pd.DataFrame,
    var_names: list[str],
    assembly_variance_guard_eps: float,
    assembly_variance_guard_mode: str,
    assembly_variance_guard_jitter_std: float,
    seed: int,
) -> AssemblyFeatureGuardResult:
    kept_names, dropped_vars = _variance_guard(
        assembly_df,
        var_names,
        eps=assembly_variance_guard_eps,
    )

    guard_mode = str(assembly_variance_guard_mode).strip().lower()
    if guard_mode not in {"jitter", "drop"}:
        raise ValueError(
            f"Unknown assembly_variance_guard_mode={assembly_variance_guard_mode!r}; expected 'jitter' or 'drop'"
        )

    if dropped_vars:
        if guard_mode == "drop":
            LOGGER.info(
                "    Variance-guard: dropping "
                f"{len(dropped_vars)} near-constant/non-finite assembly feature(s) "
                f"(eps={assembly_variance_guard_eps:g}): "
                + ", ".join(dropped_vars)
            )
        else:
            LOGGER.info(
                "    Variance-guard: detected "
                f"{len(dropped_vars)} near-constant/non-finite assembly feature(s) "
                f"(eps={assembly_variance_guard_eps:g}); applying deterministic jitter std={assembly_variance_guard_jitter_std:g}"
                + ": "
                + ", ".join(dropped_vars)
            )
    else:
        LOGGER.info(f"    Variance-guard: OK (eps={assembly_variance_guard_eps:g})")

    if guard_mode == "drop":
        guarded_names = kept_names
        df_for_cd = (
            assembly_df[guarded_names].copy()
            if guarded_names
            else assembly_df.iloc[:, :0].copy()
        )
    else:
        guarded_names = var_names
        df_for_cd = assembly_df[guarded_names].copy()
        if dropped_vars and assembly_variance_guard_jitter_std > 0:
            rng = np.random.default_rng(seed=seed + 20202)
            for col in dropped_vars:
                df_for_cd[col] = df_for_cd[col].to_numpy(dtype=float) + rng.normal(
                    0.0,
                    float(assembly_variance_guard_jitter_std),
                    size=len(df_for_cd),
                )

    return AssemblyFeatureGuardResult(
        df_for_cd=df_for_cd,
        guarded_names=guarded_names,
        dropped_vars=dropped_vars,
        summary={
            "variance_guard_eps": float(assembly_variance_guard_eps),
            "variance_guard_mode": guard_mode,
            "variance_guard_jitter_std": float(assembly_variance_guard_jitter_std),
            "dropped_vars": list(dropped_vars),
            "guarded_vars": list(guarded_names),
        },
    )


def _run_assembly_discovery_once(
    *,
    df_for_cd: pd.DataFrame,
    var_names: list[str],
    method: str,
    alpha_pc: float,
    strict: bool,
    raise_on_error: bool,
    jitter_std: float,
    seed: int,
) -> tuple[list[Edge], object | None]:
    first_attempt_raise_on_error = not (method == "pc" and len(var_names) >= 2)
    edges, graph = _run_causal_discovery_on_features(
        method=method,
        data_df=df_for_cd,
        var_names=var_names,
        alpha_pc=alpha_pc,
        strict=strict,
        raise_on_error=raise_on_error and first_attempt_raise_on_error,
    )

    if method == "pc" and len(var_names) >= 2 and graph is None:
        retry_jitter = jitter_std if jitter_std > 0 else 0.05
        rng = np.random.default_rng(seed=seed + 12345)
        arr = df_for_cd.to_numpy() + rng.normal(0.0, retry_jitter, df_for_cd.shape)
        df_retry = pd.DataFrame(arr, columns=df_for_cd.columns)
        LOGGER.info(
            f"    Retrying PC on assembly features with deterministic jitter std={retry_jitter}"
        )
        edges, graph = run_pc_algorithm(
            df_retry,
            var_names,
            alpha=alpha_pc,
            strict=strict,
            raise_on_error=raise_on_error,
        )

    return edges, graph


def _run_assembly_causal_discovery(
    *,
    assembly_df: pd.DataFrame,
    var_names: list[str],
    options: CausalDiscoveryOptions,
    alpha_pc: float,
    jitter_std: float,
    seed: int,
    assembly_variance_guard_eps: float,
    assembly_variance_guard_mode: str,
    assembly_variance_guard_jitter_std: float,
    assembly_stability_auto_on_degenerate: bool,
    assembly_stability_selection: bool,
    assembly_stability_n_bootstrap: int,
    assembly_stability_frac: float,
    assembly_stability_threshold: float,
    assembly_stability_seed_offset: int,
) -> tuple[list[Edge], object | None, dict[str, Any]]:
    LOGGER.info("")
    LOGGER.info(
        f"  [DAG 3] Running {options.assembly_method.upper()} on assembly readout features..."
    )

    guard = _prepare_assembly_features_for_causal_discovery(
        assembly_df=assembly_df,
        var_names=var_names,
        assembly_variance_guard_eps=assembly_variance_guard_eps,
        assembly_variance_guard_mode=assembly_variance_guard_mode,
        assembly_variance_guard_jitter_std=assembly_variance_guard_jitter_std,
        seed=seed,
    )

    assembly_edges, assembly_graph = _run_assembly_discovery_once(
        df_for_cd=guard.df_for_cd,
        var_names=guard.guarded_names,
        method=options.assembly_method,
        alpha_pc=alpha_pc,
        strict=options.strict_assembly,
        raise_on_error=options.raise_on_error,
        jitter_std=jitter_std,
        seed=seed,
    )

    assembly_stability_used = False
    assembly_stability_details = None

    need_stability = assembly_stability_selection or (
        assembly_stability_auto_on_degenerate and bool(guard.dropped_vars)
    )

    if need_stability:
        if len(guard.guarded_names) < 2:
            LOGGER.info("    Stability selection: skipped (too few non-degenerate variables)")
        else:
            assembly_stability_used = True
            LOGGER.info(
                "    Stability selection: "
                f"bootstrapping n={assembly_stability_n_bootstrap}, "
                f"frac={assembly_stability_frac:.2f}, "
                f"threshold={assembly_stability_threshold:.2f}"
            )

            assembly_edges = _bootstrap_stable_edges(
                data_df=guard.df_for_cd,
                var_names=guard.guarded_names,
                method=options.assembly_method,
                alpha=alpha_pc,
                strict=options.strict_assembly,
                raise_on_error=options.raise_on_error,
                n_bootstrap=assembly_stability_n_bootstrap,
                sample_frac=assembly_stability_frac,
                threshold=assembly_stability_threshold,
                seed=seed + int(assembly_stability_seed_offset),
            )

            assembly_stability_details = {
                "method": options.assembly_method,
                "n_bootstrap": int(assembly_stability_n_bootstrap),
                "sample_frac": float(assembly_stability_frac),
                "threshold": float(assembly_stability_threshold),
                "seed": int(seed + int(assembly_stability_seed_offset)),
            }

    LOGGER.info(f"    Discovered {len(assembly_edges)} edges")
    for source, target in assembly_edges:
        LOGGER.info(f"    {source} -> {target}")

    summary = dict(guard.summary)
    summary.update(
        {
            "raise_on_error": bool(options.raise_on_error),
            "stability_used": bool(assembly_stability_used),
            "stability": assembly_stability_details,
        }
    )
    return assembly_edges, assembly_graph, summary


def _run_matched_causal_discovery(
    *,
    ground_truth_edges: list[Edge],
    neuron_df: pd.DataFrame,
    assembly_df: pd.DataFrame,
    var_names: list[str],
    method: str,
    assembly_method: str | None,
    alpha_pc: float,
    jitter_std: float,
    seed: int,
    strict_causal_discovery: bool,
    strict_neuron_causal_discovery: bool | None,
    strict_assembly_causal_discovery: bool | None,
    raise_on_causal_discovery_error: bool,
    skip_neuron_dag: bool,
    assembly_variance_guard_eps: float,
    assembly_variance_guard_mode: str,
    assembly_variance_guard_jitter_std: float,
    assembly_stability_auto_on_degenerate: bool,
    assembly_stability_selection: bool,
    assembly_stability_n_bootstrap: int,
    assembly_stability_frac: float,
    assembly_stability_threshold: float,
    assembly_stability_seed_offset: int,
) -> DiscoveryStageResult:
    LOGGER.info("")
    LOGGER.info("[STAGE VI] Matched causal discovery...")

    options = _normalize_causal_discovery_options(
        method=method,
        assembly_method=assembly_method,
        strict_causal_discovery=strict_causal_discovery,
        strict_neuron_causal_discovery=strict_neuron_causal_discovery,
        strict_assembly_causal_discovery=strict_assembly_causal_discovery,
        raise_on_causal_discovery_error=raise_on_causal_discovery_error,
    )

    LOGGER.info("")
    LOGGER.info("  [DAG 1] Ground Truth:")
    for source, target in ground_truth_edges:
        LOGGER.info(f"    {source} -> {target}")

    if skip_neuron_dag:
        neuron_edges, neuron_graph = [], None
        LOGGER.info("")
        LOGGER.info("  [DAG 2] Skipped neuron causal discovery (skip_neuron_dag=True)")
    else:
        LOGGER.info("")
        LOGGER.info(f"  [DAG 2] Running {options.method.upper()} on raw-neuron readout features...")
        neuron_edges, neuron_graph = _run_causal_discovery_on_features(
            method=options.method,
            data_df=neuron_df,
            var_names=var_names,
            alpha_pc=alpha_pc,
            strict=options.strict_neuron,
            raise_on_error=options.raise_on_error,
        )

        LOGGER.info(f"    Discovered {len(neuron_edges)} edges")
        for source, target in neuron_edges:
            LOGGER.info(f"    {source} -> {target}")

    assembly_edges, assembly_graph, assembly_causal_discovery = (
        _run_assembly_causal_discovery(
            assembly_df=assembly_df,
            var_names=var_names,
            options=options,
            alpha_pc=alpha_pc,
            jitter_std=jitter_std,
            seed=seed,
            assembly_variance_guard_eps=assembly_variance_guard_eps,
            assembly_variance_guard_mode=assembly_variance_guard_mode,
            assembly_variance_guard_jitter_std=assembly_variance_guard_jitter_std,
            assembly_stability_auto_on_degenerate=assembly_stability_auto_on_degenerate,
            assembly_stability_selection=assembly_stability_selection,
            assembly_stability_n_bootstrap=assembly_stability_n_bootstrap,
            assembly_stability_frac=assembly_stability_frac,
            assembly_stability_threshold=assembly_stability_threshold,
            assembly_stability_seed_offset=assembly_stability_seed_offset,
        )
    )

    return DiscoveryStageResult(
        neuron_edges=neuron_edges,
        neuron_graph=neuron_graph,
        assembly_edges=assembly_edges,
        assembly_graph=assembly_graph,
        assembly_causal_discovery=assembly_causal_discovery,
    )


def _compare_recovered_graphs(
    *,
    ground_truth_edges: list[Edge],
    discovery: DiscoveryStageResult,
    var_names: list[str],
    skip_neuron_dag: bool,
) -> dict[str, Any]:
    LOGGER.info("")
    LOGGER.info("[EVALUATION] Comparing recovered graphs and MI preservation...")

    if skip_neuron_dag:
        gt_set = set(ground_truth_edges)
        asm_set = set(discovery.assembly_edges)
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
        LOGGER.info("  (Neuron DAG skipped; reporting assembly vs ground truth only)")
        LOGGER.info(f"  Assembly precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
        return comparison_results

    return print_dag_comparison_report(
        ground_truth_edges,
        discovery.neuron_edges,
        discovery.assembly_edges,
        var_names,
    )


def _compute_information_validation(
    *,
    neuron_df: pd.DataFrame,
    assembly_df: pd.DataFrame,
    var_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mi_neurons = compute_mi_matrix(
        {c: neuron_df[c].to_numpy() for c in var_names}, var_names
    )
    mi_assemblies = compute_mi_matrix(
        {c: assembly_df[c].to_numpy() for c in var_names}, var_names
    )
    validation = validate_information_preservation(
        mi_neurons, mi_assemblies, threshold=0.1
    )
    return mi_neurons, mi_assemblies, validation


def _build_controls_summary(
    *,
    assembly_stage: AssemblyStageResult,
    deterministic_k_encoding: bool,
    encoded: EncodingStageResult,
    deterministic_k_step: int,
    shuffled_mapping_control: bool,
    disable_plasticity_control: bool,
) -> dict[str, Any]:
    return {
        "shuffled_mapping": bool(shuffled_mapping_control),
        "disable_plasticity": bool(disable_plasticity_control),
        "target_area_by_var_name": dict(assembly_stage.target_area_by_var_name),
        "deterministic_k_encoding": bool(deterministic_k_encoding),
        "stimulus_k": (
            int(encoded.stimulus_k_used)
            if deterministic_k_encoding and encoded.stimulus_k_used is not None
            else None
        ),
        "deterministic_k_step": (
            int(deterministic_k_step) if deterministic_k_encoding else None
        ),
    }


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
    raise_on_causal_discovery_error: bool = True,
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
    quiet: bool = False,
    log_level: int | str | None = None,
    log_file: str | None = None,
) -> CausalDAGValidationResult:
    """Run the paper-aligned causal-structure preservation validation.

    strict_causal_discovery controls whether undirected CPDAG adjacencies
    are omitted unless per-DAG strict overrides are supplied.
    raise_on_causal_discovery_error=True makes backend failures raise instead
    of being silently converted into empty-edge graphs.
    quiet=True suppresses INFO-level progress logs. log_level and log_file can
    be used for debug/quiet modes or persistent logs in long sweeps.
    """

    if quiet:
        configure_logging(level="WARNING", log_file=log_file)
    elif log_level is not None or log_file is not None:
        configure_logging(level=log_level or "INFO", log_file=log_file)

    var_list = list(var_names)
    ground_truth_list = list(ground_truth_edges)

    _print_pipeline_header(df, var_list, ground_truth_list)

    encoded = _encode_observations(
        df=df,
        var_names=var_list,
        neurons_per_var=neurons_per_var,
        assembly_k=assembly_k,
        positive_values_map=positive_values_map,
        positive_prob=positive_prob,
        negative_prob=negative_prob,
        deterministic_k_encoding=deterministic_k_encoding,
        stimulus_k=stimulus_k,
        deterministic_k_readout_mode=deterministic_k_readout_mode,
        deterministic_k_step=deterministic_k_step,
        seed=seed,
    )

    neuron_df = _extract_neuron_readout(
        encoded=encoded,
        var_names=var_list,
        neurons_per_var=neurons_per_var,
        positive_values_map=positive_values_map,
        deterministic_k_encoding=deterministic_k_encoding,
        deterministic_k_readout_mode=deterministic_k_readout_mode,
        pca_per_var=pca_per_var,
    )

    assembly_stage = _train_assembly_stage(
        encoded=encoded,
        var_names=var_list,
        neurons_per_var=neurons_per_var,
        assembly_k=assembly_k,
        n_train=n_train,
        n_presentations=n_presentations,
        beta=beta,
        seed=seed,
        shuffled_mapping_control=shuffled_mapping_control,
        disable_plasticity_control=disable_plasticity_control,
        brain_max_support_ratio=brain_max_support_ratio,
    )

    assembly_df = _extract_assembly_readout(
        encoded=encoded,
        assembly_stage=assembly_stage,
        var_names=var_list,
        neurons_per_var=neurons_per_var,
    )

    _apply_feature_jitter(
        neuron_df=neuron_df,
        assembly_df=assembly_df,
        jitter_std=jitter_std,
        seed=seed,
    )

    obs_diagnostics = None
    compression = _compute_compression(brain=assembly_stage.brain, var_names=var_list)

    if skip_causal_discovery:
        return CausalDAGValidationResult(
            {
                "neuron_df": neuron_df,
                "assembly_df": assembly_df,
                "brain": assembly_stage.brain,
                "var_names": var_list,
                "compression": compression,
            }
        )

    discovery = _run_matched_causal_discovery(
        ground_truth_edges=ground_truth_list,
        neuron_df=neuron_df,
        assembly_df=assembly_df,
        var_names=var_list,
        method=method,
        assembly_method=assembly_method,
        alpha_pc=alpha_pc,
        jitter_std=jitter_std,
        seed=seed,
        strict_causal_discovery=strict_causal_discovery,
        strict_neuron_causal_discovery=strict_neuron_causal_discovery,
        strict_assembly_causal_discovery=strict_assembly_causal_discovery,
        raise_on_causal_discovery_error=raise_on_causal_discovery_error,
        skip_neuron_dag=skip_neuron_dag,
        assembly_variance_guard_eps=assembly_variance_guard_eps,
        assembly_variance_guard_mode=assembly_variance_guard_mode,
        assembly_variance_guard_jitter_std=assembly_variance_guard_jitter_std,
        assembly_stability_auto_on_degenerate=assembly_stability_auto_on_degenerate,
        assembly_stability_selection=assembly_stability_selection,
        assembly_stability_n_bootstrap=assembly_stability_n_bootstrap,
        assembly_stability_frac=assembly_stability_frac,
        assembly_stability_threshold=assembly_stability_threshold,
        assembly_stability_seed_offset=assembly_stability_seed_offset,
    )

    comparison_results = _compare_recovered_graphs(
        ground_truth_edges=ground_truth_list,
        discovery=discovery,
        var_names=var_list,
        skip_neuron_dag=skip_neuron_dag,
    )

    if visualize_save_path:
        visualize_three_dags(
            ground_truth_list,
            discovery.neuron_edges,
            discovery.assembly_edges,
            var_list,
            save_path=visualize_save_path,
        )

    mi_neurons, mi_assemblies, validation = _compute_information_validation(
        neuron_df=neuron_df,
        assembly_df=assembly_df,
        var_names=var_list,
    )

    return CausalDAGValidationResult(
        {
            "ground_truth_edges": ground_truth_list,
            "neuron_edges": discovery.neuron_edges,
            "assembly_edges": discovery.assembly_edges,
            "neuron_df": neuron_df,
            "assembly_df": assembly_df,
            "neuron_graph": discovery.neuron_graph,
            "assembly_graph": discovery.assembly_graph,
            "comparison": comparison_results,
            "mi_neurons": mi_neurons,
            "mi_assemblies": mi_assemblies,
            "validation": validation,
            "brain": assembly_stage.brain,
            "var_names": var_list,
            "observational_diagnostics": obs_diagnostics,
            "assembly_causal_discovery": discovery.assembly_causal_discovery,
            "controls": _build_controls_summary(
                assembly_stage=assembly_stage,
                deterministic_k_encoding=deterministic_k_encoding,
                encoded=encoded,
                deterministic_k_step=deterministic_k_step,
                shuffled_mapping_control=shuffled_mapping_control,
                disable_plasticity_control=disable_plasticity_control,
            ),
            "compression": compression,
        }
    )
