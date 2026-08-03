"""Experiment parameter loader.

Single source of truth: experiments/experiments_configuration.json.
Edit experiments_configuration.json to change any default across all scripts.
CLI args in individual scripts still override these values.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
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
_out = _cfg.get("output", {})


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


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_configuration.json")
_CONFIG_COPIED = False


def get_results_root() -> str:
    """Return absolute base output root for experiment artifacts.

    Config key: output.results_root in experiments_configuration.json.
    Relative paths are resolved from repository root.
    """
    configured = str(_out.get("results_root", "runs")).strip()
    if not configured:
        configured = "runs"
    if os.path.isabs(configured):
        return configured
    return os.path.normpath(os.path.join(_REPO_ROOT, configured))


def get_run_output_dir() -> str:
    """Return absolute daily run directory under configured results root.

    Layout: <results_root>/YYYYMMDD
    """
    results_root = get_results_root()
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    return os.path.join(results_root, date_str)


def _ensure_results_dir_and_config() -> str:
    """Ensure daily run directory exists and copy config there once.

    Returns the daily run directory path.
    """
    global _CONFIG_COPIED
    run_output_dir = get_run_output_dir()
    os.makedirs(run_output_dir, exist_ok=True)

    if not _CONFIG_COPIED:
        config_dest = os.path.join(run_output_dir, "experiments_configuration.json")
        shutil.copy2(_CFG_FILE, config_dest)
        _CONFIG_COPIED = True

    return run_output_dir


def get_output_filepath(filename: str) -> str:
    """Get absolute path for an output file in runs/YYYYMMDD."""
    run_output_dir = _ensure_results_dir_and_config()
    return os.path.join(run_output_dir, filename)


def make_run_output_dir(script_name: str) -> str:
    """Backward-compatible alias that returns runs/YYYYMMDD.

    script_name is ignored to enforce the single daily folder layout.
    """
    _ = script_name
    return _ensure_results_dir_and_config()
