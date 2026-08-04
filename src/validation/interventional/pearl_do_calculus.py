"""Pearl-style do-intervention validation helpers.

Pearl-style intervention evaluation with SCM do(...) samples.

How it works: use synthetic SCM generators as intervention oracles; generate
baseline and do(...) samples; estimate changes in target probabilities or feature
means; compare raw, neuron, and assembly effect directions/magnitudes as an
interventional validation layer for the observational PC/GES results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence, Set

import numpy as np
import pandas as pd

from ...encoding.bernoulli import encode_bernoulli_dataframe
from ...encoding.deterministic_k import build_deterministic_k_map, encode_deterministic_k_dataframe
from ...representation.assembly_feature_extraction import extract_assembly_features


@dataclass(frozen=True)
class DoEffect:
    source: str
    target: str
    do_value: object
    delta: float


def sign(value: float, *, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def probability_of_values(df: pd.DataFrame, column: str, positive_values: Set[object]) -> float:
    return float(df[column].isin(positive_values).mean())


def estimate_do_delta(
    *,
    df_baseline: pd.DataFrame,
    df_do: pd.DataFrame,
    var: str,
    positive_values: set[object],
) -> float:
    """Compute change in P(var in positive_values) between do and baseline."""
    base = probability_of_values(df_baseline, var, positive_values)
    do_probability = probability_of_values(df_do, var, positive_values)
    return float(do_probability - base)


def infer_edge_direction_from_oracle(
    *,
    generator: Callable[..., tuple[pd.DataFrame, object]],
    df_baseline: pd.DataFrame,
    baseline_seed: int,
    n_eval: int,
    positive_values_map: Mapping[str, set[object]],
    a: str,
    b: str,
    eps: float,
) -> dict:
    """Use an SCM generator as an interventional oracle for an adjacency.

    We compare two intervention effects:
      delta on b under do(a)
      delta on a under do(b)

    Returns a verdict in {"a->b", "b->a", "both", "neither", "unknown"}.
    """

    if a not in positive_values_map or b not in positive_values_map:
        return {
            "a": a,
            "b": b,
            "delta_b_do_a": None,
            "delta_a_do_b": None,
            "verdict": "unknown",
            "reason": "missing positive_values_map entries",
        }

    a_val = sorted(positive_values_map[a], key=lambda value: str(value))[0]
    b_val = sorted(positive_values_map[b], key=lambda value: str(value))[0]

    df_do_a, _ = generator(n_patients=n_eval, seed=baseline_seed, do={a: a_val})
    df_do_b, _ = generator(n_patients=n_eval, seed=baseline_seed, do={b: b_val})

    delta_b_do_a = estimate_do_delta(
        df_baseline=df_baseline,
        df_do=df_do_a,
        var=b,
        positive_values=set(positive_values_map[b]),
    )
    delta_a_do_b = estimate_do_delta(
        df_baseline=df_baseline,
        df_do=df_do_b,
        var=a,
        positive_values=set(positive_values_map[a]),
    )

    a_to_b = abs(delta_b_do_a) > eps
    b_to_a = abs(delta_a_do_b) > eps

    if a_to_b and not b_to_a:
        verdict = "a->b"
    elif b_to_a and not a_to_b:
        verdict = "b->a"
    elif a_to_b and b_to_a:
        verdict = "both"
    else:
        verdict = "neither"

    return {
        "a": a,
        "b": b,
        "a_val": a_val,
        "b_val": b_val,
        "delta_b_do_a": float(delta_b_do_a),
        "delta_a_do_b": float(delta_a_do_b),
        "verdict": verdict,
        "eps": float(eps),
    }


def summarize_against_ground_truth(
    *,
    inferred_directed_edges: Iterable[tuple[str, str]],
    ground_truth_edges: Iterable[tuple[str, str]],
) -> dict:
    gt = set(ground_truth_edges)
    inferred = set(inferred_directed_edges)

    correct = sorted(inferred & gt)
    spurious = sorted(inferred - gt)
    missing = sorted(gt - inferred)

    return {
        "correct": correct,
        "spurious": spurious,
        "missing": missing,
        "precision": (len(correct) / len(inferred)) if inferred else 0.0,
        "recall": (len(correct) / len(gt)) if gt else 0.0,
    }


def compute_assembly_feature_flip_map(
    df: pd.DataFrame,
    *,
    assembly_features_base: Mapping[str, np.ndarray],
    positive_values_map: Mapping[str, Set[object]] | None,
) -> Dict[str, bool]:
    """Calibrate per-variable polarity for assembly features.

    Returns var->flip where flip=True means the assembly feature is higher for the
    negative class. In that case, downstream code uses (1 - feature) so higher
    means positive.
    """

    flips: Dict[str, bool] = {}
    if not positive_values_map:
        return flips

    for var, positive_values in positive_values_map.items():
        if var not in df.columns or var not in assembly_features_base:
            continue

        mask = df[var].isin(set(positive_values))
        if not bool(mask.any()) or not bool((~mask).any()):
            continue

        arr = np.asarray(assembly_features_base[var], dtype=float)
        mean_pos = float(np.nanmean(arr[mask.to_numpy()]))
        mean_neg = float(np.nanmean(arr[(~mask).to_numpy()]))
        flips[var] = mean_pos < mean_neg

    return flips


def compute_neuron_and_assembly_mean_features(
    df: pd.DataFrame,
    *,
    brain: object,
    var_names: Sequence[str],
    neurons_per_var: int,
    positive_values_map: Mapping[str, Set[object]] | None,
    positive_prob: float,
    negative_prob: float,
    seed: int,
    deterministic_k_encoding: bool = True,
    deterministic_k_step: int = 10,
    stimulus_k: int = 100,
    align_assembly_to_positive: bool = True,
    assembly_flip_map: Mapping[str, bool] | None = None,
    return_assembly_flip_map: bool = False,
) -> tuple[Dict[str, float], Dict[str, float]] | tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
    """Return per-variable mean feature values for neurons and assemblies."""

    stimulus_indices = None

    if deterministic_k_encoding:
        stimulus_k_map = build_deterministic_k_map(
            df,
            list(var_names),
            base_k=int(stimulus_k),
            k_step=int(deterministic_k_step),
        )
        neural, stimulus_indices = encode_deterministic_k_dataframe(
            df,
            list(var_names),
            neurons_per_var=neurons_per_var,
            stimulus_k_map=stimulus_k_map,
            seed=seed,
        )
    else:
        neural = encode_bernoulli_dataframe(
            df,
            list(var_names),
            neurons_per_var=neurons_per_var,
            positive_values_map=positive_values_map,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            seed=seed,
        )

    neuron_means: Dict[str, float] = {}
    for var_idx, var in enumerate(var_names):
        start = var_idx * neurons_per_var
        end = start + neurons_per_var
        activations = neural[:, start:end]

        if stimulus_indices is not None and var in stimulus_indices:
            positive_values = None
            if positive_values_map and var in positive_values_map:
                positive_values = list(positive_values_map[var])

            if positive_values:
                arrays = [stimulus_indices[var][value] for value in positive_values if value in stimulus_indices[var]]
                if arrays:
                    local = np.concatenate(arrays, axis=0)
                    if local.size > 0:
                        per_sample = activations[:, local].mean(axis=1)
                        neuron_means[var] = float(per_sample.mean())
                        continue

            values = sorted(stimulus_indices[var].keys(), key=lambda value: str(value))
            if values:
                local = stimulus_indices[var][values[-1]]
                per_sample = activations[:, local].mean(axis=1)
                neuron_means[var] = float(per_sample.mean())
                continue

        neuron_means[var] = float(activations.mean(axis=1).mean())

    source_var_by_target_area_name = {var: var for var in var_names}
    assembly_features = extract_assembly_features(
        neural,
        brain,
        list(var_names),
        neurons_per_var,
        source_var_by_target_area_name=source_var_by_target_area_name,
    )
    assembly_features_base = {key: value for key, value in assembly_features.items() if "_x_" not in key}

    flip_map: Dict[str, bool] = {}
    if align_assembly_to_positive:
        flip_map = dict(assembly_flip_map) if assembly_flip_map is not None else compute_assembly_feature_flip_map(
            df,
            assembly_features_base=assembly_features_base,
            positive_values_map=positive_values_map,
        )

    assembly_means: Dict[str, float] = {}
    for var in var_names:
        arr = np.asarray(assembly_features_base[var], dtype=float)
        if flip_map.get(var, False):
            arr = 1.0 - arr
        assembly_means[var] = float(np.nanmean(arr))

    if return_assembly_flip_map:
        return neuron_means, assembly_means, flip_map

    return neuron_means, assembly_means
