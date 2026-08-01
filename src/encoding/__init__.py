"""Neural encoding methods used by the paper."""

from .bernoulli import encode_bernoulli_dataframe, encode_bernoulli_variable, extract_variable_neurons
from .deterministic_k import build_deterministic_k_map, encode_deterministic_k_dataframe
