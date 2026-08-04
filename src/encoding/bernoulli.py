"""Bernoulli neural encoding.

Full method name: Bernoulli spike-pattern encoding for symbolic SCM variables.

How it works: each variable owns a fixed neuron block. For every observation,
neurons in that block fire independently with one probability for configured
positive values and another probability for all other values. Concatenating the
blocks gives the full neural representation used by the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_POSITIVE_VALUES = {
    "Carrier",
    "Sedentary",
    "Poor",
    "High",
    "Decline",
    "Elderly",
    "Smoker",
    "Present",
    "Stroke",
    "Hypercoagulable",
    1,
    True,
}


def encode_bernoulli_variable(
    values,
    n_neurons: int = 1666,
    positive_prob: float = 0.30,
    negative_prob: float = 0.10,
    positive_values=None,
    seed: int | None = None,
) -> np.ndarray:
    """Encode one symbolic variable as Bernoulli spike patterns."""

    rng = np.random.default_rng(seed)

    if positive_values is None:
        unique_values = sorted(set(values), key=lambda value: str(value))
        if len(unique_values) == 2:
            positive_values = {unique_values[1]}
        else:
            positive_values = DEFAULT_POSITIVE_VALUES

    n_samples = len(values)
    spikes = np.zeros((n_samples, n_neurons), dtype=int)

    for row_idx, value in enumerate(values):
        probability = positive_prob if value in positive_values else negative_prob
        spikes[row_idx] = (rng.random(n_neurons) < probability).astype(int)

    return spikes


def encode_bernoulli_dataframe(
    df: pd.DataFrame,
    variables,
    neurons_per_var: int = 1666,
    positive_prob: float = 0.30,
    negative_prob: float = 0.10,
    positive_values_map=None,
    seed: int | None = None,
) -> np.ndarray:
    """Encode a dataframe by concatenating one Bernoulli neuron block per variable."""

    n_samples = len(df)
    total_neurons = len(variables) * neurons_per_var
    neural_data = np.zeros((n_samples, total_neurons), dtype=int)

    for var_idx, var_name in enumerate(variables):
        start_idx = var_idx * neurons_per_var
        end_idx = start_idx + neurons_per_var
        positive_values = (
            positive_values_map.get(var_name)
            if positive_values_map and var_name in positive_values_map
            else None
        )
        var_seed = None if seed is None else seed + var_idx

        neural_data[:, start_idx:end_idx] = encode_bernoulli_variable(
            df[var_name].values,
            n_neurons=neurons_per_var,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            positive_values=positive_values,
            seed=var_seed,
        )

    return neural_data


def extract_variable_neurons(
    neural_data: np.ndarray,
    var_idx: int,
    neurons_per_var: int = 1666,
) -> np.ndarray:
    """Extract the neuron block assigned to one variable."""

    start_idx = var_idx * neurons_per_var
    end_idx = start_idx + neurons_per_var
    return neural_data[:, start_idx:end_idx]
