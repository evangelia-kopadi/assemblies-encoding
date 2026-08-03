"""Shared defaults for validation and ablation experiments.

Full method name: shared experiment-configuration defaults.

How it works: loads parameters from experiments/config.json if it exists;
falls back to hardcoded defaults otherwise. Exposes runner_kwargs(...) so each
dataset script can override only the knobs it needs locally.

To change defaults across all scripts, edit experiments/config.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


def _load_config() -> dict:
    """Load experiments/config.json from the experiments/ folder."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _cfg_path = os.path.join(_here, "config.json")
    if os.path.exists(_cfg_path):
        with open(_cfg_path, encoding="utf-8-sig") as _f:
            return json.load(_f)
    return {}


_cfg = _load_config()
_dg = _cfg.get("data_generation", {})
_ne = _cfg.get("neural_encoding", {})
_cd = _cfg.get("causal_discovery", {})


@dataclass(frozen=True)
class ExperimentDefaults:
    # Data generation
    n_samples: int = _dg.get("n_samples", 5000)
    seed: int = _dg.get("seed", 42)

    # Neural encoding / brain simulation
    neurons_per_var: int = _ne.get("neurons_per_var", 1666)
    assembly_k: int = _ne.get("assembly_k", 100)
    n_train: int = _ne.get("n_train", 200)
    n_presentations: int = _ne.get("n_presentations", 3)
    beta: float = _ne.get("beta", 0.05)

    # Causal discovery
    alpha_pc: float = _cd.get("alpha_pc", 0.05)

    # Encoding probabilities
    positive_prob: float = _ne.get("positive_prob", 0.30)
    negative_prob: float = _ne.get("negative_prob", 0.10)


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
        "deterministic_k_encoding": _cd.get("deterministic_k_encoding", True),
        "deterministic_k_readout_mode": _cd.get("deterministic_k_readout_mode", "pool_mean"),
        "deterministic_k_step": _cd.get("deterministic_k_step", 10),
        "positive_prob": DEFAULTS.positive_prob,
        "negative_prob": DEFAULTS.negative_prob,
    }

    base.update(overrides)
    return base
