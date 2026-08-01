"""NACA: Neural Assembly Causal Analysis.

Full method name: Neural Assembly Causal Analysis package interface.

How it works: expose the package-level API for neural encoding, Brain assembly
simulation, assembly feature extraction, and MI-based information-preservation
checks. Method-specific modules such as PC, GES, and Pearl-style do-intervention
validation live in dedicated subpackages.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .brain import Brain
from .encoding.bernoulli import encode_bernoulli_dataframe, encode_bernoulli_variable, extract_variable_neurons
from .validation.assembly_feature_extraction import extract_assembly_features
from .validation.assembly_formation import form_assemblies
from .validation.information_preservation_mi import (
    mutual_information,
    compute_mi_matrix,
    validate_information_preservation,
)

__all__ = [
    "Brain",
    "encode_bernoulli_variable",
    "encode_bernoulli_dataframe",
    "extract_variable_neurons",
    "mutual_information",
    "compute_mi_matrix",
    "extract_assembly_features",
    "validate_information_preservation",
    "form_assemblies",
]