import numpy as np
import pandas as pd

from src.encoding.bernoulli import encode_bernoulli_dataframe, encode_bernoulli_variable
from src.encoding.deterministic_k import (
    build_deterministic_k_map,
    encode_deterministic_k_dataframe,
)


def test_bernoulli_variable_uses_configured_positive_values():
    values = ["No", "Yes", "No", "Yes"]

    spikes = encode_bernoulli_variable(
        values,
        n_neurons=4,
        positive_prob=1.0,
        negative_prob=0.0,
        positive_values={"Yes"},
        seed=123,
    )

    assert spikes.tolist() == [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ]


def test_bernoulli_dataframe_concatenates_variable_blocks():
    df = pd.DataFrame(
        {
            "Risk": ["Low", "High"],
            "Default": ["No", "Yes"],
        }
    )

    encoded = encode_bernoulli_dataframe(
        df,
        ["Risk", "Default"],
        neurons_per_var=3,
        positive_prob=1.0,
        negative_prob=0.0,
        positive_values_map={"Risk": {"High"}, "Default": {"Yes"}},
        seed=7,
    )

    assert encoded.shape == (2, 6)
    np.testing.assert_array_equal(encoded[0], [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(encoded[1], [1, 1, 1, 1, 1, 1])


def test_deterministic_k_map_and_encoding_have_value_specific_counts():
    df = pd.DataFrame(
        {
            "Exposure": ["Low", "Medium", "High", "Low"],
            "Outcome": [0, 1, 1, 0],
        }
    )

    stimulus_k_map = build_deterministic_k_map(
        df,
        ["Exposure", "Outcome"],
        base_k=2,
        k_step=1,
    )

    assert stimulus_k_map["Exposure"] == {"High": 2, "Low": 3, "Medium": 4}
    assert stimulus_k_map["Outcome"] == {0: 2, 1: 3}

    encoded, stimulus_indices = encode_deterministic_k_dataframe(
        df,
        ["Exposure", "Outcome"],
        neurons_per_var=10,
        stimulus_k_map=stimulus_k_map,
        seed=11,
    )

    assert encoded.shape == (4, 20)
    assert encoded[0, :10].sum() == 3
    assert encoded[1, :10].sum() == 4
    assert encoded[2, :10].sum() == 2
    assert encoded[0, 10:].sum() == 2
    assert encoded[1, 10:].sum() == 3

    exposure_sets = [set(indices.tolist()) for indices in stimulus_indices["Exposure"].values()]
    for left_idx, left in enumerate(exposure_sets):
        for right in exposure_sets[left_idx + 1 :]:
            assert left.isdisjoint(right)
