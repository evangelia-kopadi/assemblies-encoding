"""Neuron-level feature extraction from encoded spike patterns.

Deterministic-k neuron feature extraction.

How it works: encode a dataframe with deterministic-k stimulus sets, then reduce
each variable's neuron block to one scalar feature per sample. The default readout
uses the stimulus-neuron subset for configured positive values, with a deterministic
fallback value when no positive-value map is supplied.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Set

import numpy as np
import pandas as pd

from ..encoding.deterministic_k import build_deterministic_k_map, encode_deterministic_k_dataframe


def encode_for_deterministic_k_features(
    df: pd.DataFrame,
    var_names: Sequence[str],
    *,
    neurons_per_var: int,
    seed: int,
    stimulus_k: int = 100,
    deterministic_k_step: int = 10,
):
    """Return deterministic-k neural data and value-specific stimulus indices."""

    stimulus_k_map = build_deterministic_k_map(
        df,
        list(var_names),
        base_k=int(stimulus_k),
        k_step=int(deterministic_k_step),
    )
    return encode_deterministic_k_dataframe(
        df,
        list(var_names),
        neurons_per_var=neurons_per_var,
        stimulus_k_map=stimulus_k_map,
        seed=seed,
    )


def extract_neuron_features_from_encoded(
    neural_data: np.ndarray,
    stimulus_indices: Mapping[str, Mapping[object, np.ndarray]],
    var_names: Sequence[str],
    *,
    neurons_per_var: int,
    positive_values_map: Mapping[str, Set[object]] | None = None,
) -> pd.DataFrame:
    """Extract one neuron-level scalar feature per variable and sample."""

    neuron_features: dict[str, np.ndarray] = {}
    for var_idx, var_name in enumerate(var_names):
        start_idx = var_idx * neurons_per_var
        end_idx = start_idx + neurons_per_var
        activations = neural_data[:, start_idx:end_idx]

        if var_name in stimulus_indices:
            positive_values = None
            if positive_values_map and var_name in positive_values_map:
                positive_values = list(positive_values_map[var_name])

            if positive_values:
                arrays = [
                    stimulus_indices[var_name][value]
                    for value in positive_values
                    if value in stimulus_indices[var_name]
                ]
                if arrays:
                    local = np.concatenate(arrays, axis=0)
                    if local.size > 0:
                        neuron_features[var_name] = activations[:, local].mean(axis=1)
                        continue

            values = sorted(stimulus_indices[var_name].keys(), key=lambda value: str(value))
            if values:
                local = stimulus_indices[var_name][values[-1]]
                neuron_features[var_name] = activations[:, local].mean(axis=1)
                continue

        neuron_features[var_name] = activations.mean(axis=1)

    return pd.DataFrame(neuron_features)


def extract_neuron_features_for_new_data(
    df: pd.DataFrame,
    *,
    var_names: Sequence[str],
    neurons_per_var: int,
    seed: int,
    positive_values_map: Mapping[str, Set[object]] | None = None,
    stimulus_k: int = 100,
    deterministic_k_step: int = 10,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[object, np.ndarray]]]:
    """Encode new data and return neuron features plus encoded data."""

    neural_data, stimulus_indices = encode_for_deterministic_k_features(
        df,
        var_names,
        neurons_per_var=neurons_per_var,
        seed=seed,
        stimulus_k=stimulus_k,
        deterministic_k_step=deterministic_k_step,
    )
    neuron_df = extract_neuron_features_from_encoded(
        neural_data,
        stimulus_indices,
        var_names,
        neurons_per_var=neurons_per_var,
        positive_values_map=positive_values_map,
    )
    return neuron_df, neural_data, stimulus_indices