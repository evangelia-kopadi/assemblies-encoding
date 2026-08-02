"""Shared defaults for validation and ablation experiments.

Full method name: shared experiment-configuration defaults.

How it works: centralize common sample-size, encoding, Brain, assembly, and
causal-discovery parameters; expose runner_kwargs(...) so each dataset script can
override only the knobs it needs locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ExperimentDefaults:
    # Data generation
    n_samples: int = 5000
    seed: int = 42

    # Neural encoding / brain simulation
    neurons_per_var: int = 1666
    assembly_k: int = 100
    n_train: int = 200
    n_presentations: int = 3
    beta: float = 0.05

    # Causal discovery
    alpha_pc: float = 0.05  # Standard default; reduces spurious edges

    # Encoding probabilities
    positive_prob: float = 0.30
    negative_prob: float = 0.10


DEFAULTS = ExperimentDefaults()


def runner_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Default kwargs for src.runner.run_causal_dag_validation.

    Experiments should call: run_causal_dag_validation(..., **runner_kwargs(alpha_pc=0.1))
    """

    base: Dict[str, Any] = {
        "neurons_per_var": DEFAULTS.neurons_per_var,
        "assembly_k": DEFAULTS.assembly_k,
        "n_train": DEFAULTS.n_train,
        "n_presentations": DEFAULTS.n_presentations,
        "beta": DEFAULTS.beta,
        "seed": DEFAULTS.seed,
        "alpha_pc": DEFAULTS.alpha_pc,
        "deterministic_k_encoding": True,
        "deterministic_k_readout_mode": "pool_mean",
        "deterministic_k_step": 10,
        "positive_prob": DEFAULTS.positive_prob,
        "negative_prob": DEFAULTS.negative_prob,
    }

    base.update(overrides)
    return base
