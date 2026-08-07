"""Experiment parameter loader.

Single source of truth: experiments/experiments_configuration.json.
Edit experiments_configuration.json to change any default across all scripts.
CLI args in individual scripts still override these values.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
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
_CFG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiments_configuration.json"
)
_CONFIG_COPIED = False
_RUN_METADATA_RECORDED_DIRS: set[str] = set()


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


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _get_git_metadata() -> dict[str, Any]:
    status = _run_git_command(["status", "--short", "--untracked-files=no"])
    return {
        "commit_sha": _run_git_command(["rev-parse", "HEAD"]),
        "branch": _run_git_command(["branch", "--show-current"]),
        "is_dirty_tracked": bool(status) if status is not None else None,
    }


def _normalize_package_name(name: str) -> str:
    return name.replace("_", "-").lower()


def _lockfile_matches_environment(lockfile_path: str) -> bool:
    if not os.path.exists(lockfile_path):
        return False

    pinned: dict[str, str] = {}
    with open(lockfile_path, encoding="utf-8") as lockfile:
        for raw_line in lockfile:
            line = raw_line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, version = line.split("==", 1)
            pinned[_normalize_package_name(name)] = version.strip()

    if not pinned:
        return False

    for package_name, expected_version in pinned.items():
        try:
            installed_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return False
        if installed_version != expected_version:
            return False

    return True


def _detect_dependency_mode() -> dict[str, str]:
    env_value = os.environ.get("ASSEMBLIES_ENCODING_DEPENDENCY_MODE")
    if env_value:
        return {
            "mode": env_value,
            "source": "ASSEMBLIES_ENCODING_DEPENDENCY_MODE",
        }

    lockfile_path = os.path.join(_REPO_ROOT, "requirements-lock.txt")
    if _lockfile_matches_environment(lockfile_path):
        return {
            "mode": "requirements-lock.txt",
            "source": "installed packages match requirements-lock.txt",
        }

    return {
        "mode": "requirements.txt",
        "source": "default assumption; set ASSEMBLIES_ENCODING_DEPENDENCY_MODE to override",
    }


def _current_command_metadata() -> dict[str, Any]:
    argv = list(sys.argv)
    return {
        "timestamp_utc": _utc_now_iso(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "argv": argv,
        "command": subprocess.list2cmdline([sys.executable, *argv]),
    }


def _write_run_metadata(run_output_dir: str) -> None:
    metadata_path = os.path.join(run_output_dir, "run_metadata.json")
    now = _utc_now_iso()

    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    else:
        metadata = {
            "schema_version": 1,
            "created_at_utc": now,
            "run_directory": run_output_dir,
            "config_snapshot": os.path.basename(_CFG_FILE),
            "git": _get_git_metadata(),
            "python": {
                "version": sys.version,
                "version_info": list(sys.version_info[:3]),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "dependency_mode": _detect_dependency_mode(),
            "commands": [],
        }

    metadata["updated_at_utc"] = now
    metadata.setdefault("commands", []).append(_current_command_metadata())

    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    os.replace(tmp_path, metadata_path)


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

    if run_output_dir not in _RUN_METADATA_RECORDED_DIRS:
        _write_run_metadata(run_output_dir)
        _RUN_METADATA_RECORDED_DIRS.add(run_output_dir)

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
