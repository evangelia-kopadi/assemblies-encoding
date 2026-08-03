"""Experiment parameter loader.

Single source of truth: experiments/config.json.
Edit config.json to change any default across all scripts.
CLI args in individual scripts still override these values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


def _load_config() -> dict:
    _here = os.path.dirname(os.path.abspath(__file__))
    _cfg_path = os.path.join(_here, "experiments_configuration.json")
    with open(_cfg_path, encoding="utf-8-sig") as _f:
        return json.load(_f)


_cfg = _load_config()
_dg = _cfg["data_generation"]
_ne = _cfg["neural_encoding"]
_cd = _cfg["causal_discovery"]


@dataclass(frozen=True)
class ExperimentDefaults:
    n_samples: int = _dg["n_samples"]
    seed: int = _dg["seed"]
    neurons_per_var: int = _ne["neurons_per_var"]
    assembly_k: int = _ne["assembly_k"]
    n_train: int = _ne["n_train"]
    n_presentations: int = _ne["n_presentations"]
    beta: float = _ne["beta"]
    alpha_pc: float = _cd["alpha_pc"]
    positive_prob: float = _ne["positive_prob"]
    negative_prob: float = _ne["negative_prob"]


DEFAULTS = ExperimentDefaults()


def runner_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Default kwargs for src.runner.run_causal_dag_validation."""
    base: Dict[str, Any] = {
        "neurons_per_var": DEFAULTS.neurons_per_var,
        "assembly_k": DEFAULTS.assembly_k,
        "n_train": DEFAULTS.n_train,
        "n_presentations": DEFAULTS.n_presentations,
        "beta": DEFAULTS.beta,
        "seed": DEFAULTS.seed,
        "alpha_pc": DEFAULTS.alpha_pc,
        "deterministic_k_encoding": _cd["deterministic_k_encoding"],
        "deterministic_k_readout_mode": _cd["deterministic_k_readout_mode"],
        "deterministic_k_step": _cd["deterministic_k_step"],
        "positive_prob": DEFAULTS.positive_prob,
        "negative_prob": DEFAULTS.negative_prob,
    }
    base.update(overrides)
    return base
