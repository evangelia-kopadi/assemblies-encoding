"""Deterministic-k neural encoding.

Full method name: deterministic-k stimulus-set encoding for symbolic SCM variables.

How it works: each distinct value of a variable is assigned a deterministic,
disjoint subset of neurons inside that variable's neuron block. The subset size
is k(value) = base_k + value_index * k_step, so values remain deterministic but
are still distinguishable by simple firing-rate readouts.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def ordered_observed_values(series: pd.Series) -> list[object]:
    """Return values in the deterministic order used for stimulus allocation."""

    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        return list(series.cat.categories)
    return sorted(series.dropna().unique(), key=lambda value: str(value))


def build_deterministic_k_map(
    df: pd.DataFrame,
    variables: Sequence[str],
    *,
    base_k: int,
    k_step: int,
) -> dict[str, dict[object, int]]:
    """Build var/value -> k maps for deterministic-k encoding."""

    stimulus_k_map: dict[str, dict[object, int]] = {}
    for var_name in variables:
        values = ordered_observed_values(df[var_name])
        stimulus_k_map[var_name] = {
            value: int(base_k + value_idx * int(k_step))
            for value_idx, value in enumerate(values)
        }
    return stimulus_k_map


def encode_deterministic_k_dataframe(
    df: pd.DataFrame,
    variables: Sequence[str],
    *,
    neurons_per_var: int = 1666,
    stimulus_k_map: Mapping[str, Mapping[object, int]],
    value_to_index_map: dict[str, dict[object, int]] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, dict[object, np.ndarray]]]:
    """Encode a dataframe using deterministic value-specific stimulus sets."""

    rng = np.random.default_rng(seed)

    n_samples = len(df)
    total_neurons = len(variables) * neurons_per_var
    neural_data = np.zeros((n_samples, total_neurons), dtype=int)

    if value_to_index_map is None:
        value_to_index_map = {}

    stimulus_indices: dict[str, dict[object, np.ndarray]] = {}

    for var_idx, var_name in enumerate(variables):
        if var_name not in value_to_index_map:
            values = ordered_observed_values(df[var_name])
            value_to_index_map[var_name] = {value: value_idx for value_idx, value in enumerate(values)}

        if var_name not in stimulus_k_map:
            raise ValueError(f"missing stimulus_k_map for variable {var_name!r}")

        available = np.arange(neurons_per_var)
        rng.shuffle(available)

        indices_for_values: dict[object, np.ndarray] = {}
        cursor = 0
        for value, _value_idx in sorted(value_to_index_map[var_name].items(), key=lambda item: item[1]):
            k = int(stimulus_k_map[var_name].get(value, 0))
            if k <= 0:
                raise ValueError(f"Invalid k={k} for {var_name} value {value!r}")
            if cursor + k > neurons_per_var:
                raise ValueError(
                    f"Variable {var_name} needs more than {neurons_per_var} neurons "
                    f"for deterministic-k encoding (cursor={cursor}, k={k})."
                )
            indices_for_values[value] = available[cursor:cursor + k]
            cursor += k

        stimulus_indices[var_name] = indices_for_values

        offset = var_idx * neurons_per_var
        for row_idx, value in enumerate(df[var_name].values):
            if pd.isna(value):
                continue
            local_idx = indices_for_values[value]
            neural_data[row_idx, offset + local_idx] = 1

    return neural_data, stimulus_indices