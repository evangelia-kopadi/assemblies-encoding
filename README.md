# assemblies-encoding

Clean reproducibility package for the SUM 2026 paper experiments on causal structure preservation in neural assemblies under neural encoding uncertainty.

This repository is a minimal extraction from the larger development workspace. It keeps only the code, generated artifacts, and figures needed to reproduce or extend the paper experiments.

## Provenance and Attribution

Parts of the neural assembly simulation core (notably `src/representation/brain.py`) are borrowed/adapted from the original assemblies codebase by Daniel Mitropolsky and collaborators (Princeton University & MIT): https://github.com/dmitropolsky/assemblies (source file: https://github.com/dmitropolsky/assemblies/blob/a7ded3b23aa1cce10b8979801a27da8a97bc23d3/brain.py; pinned commit: https://github.com/dmitropolsky/assemblies/commit/a7ded3b23aa1cce10b8979801a27da8a97bc23d3), as acknowledged in-file.

## Associated paper

This repository accompanies the SUM 2026 manuscript **"Causal Structure Preservation in Neural Assemblies under Encoding Uncertainty"**. The editable paper source is available at `paper/causal_structure_preservation_in_neural_assemblies_under_encoding_uncertainty.md`, with its bibliography and build script in the same folder. References to Tables 1-5 in this README refer to that manuscript.

## Experiment roles

The repository contains both single-run and multi-run artifacts.

The single-run benchmark (`results/`) supports the concrete deterministic-k example summarized in the manuscript's single-run benchmark table. It is useful as a small, inspectable run, not as the main robustness evidence.

The main encoding conclusion is based on the five-dataset, ten-seed sensitivity sweeps (`results/`). In those sweeps, each SCM generator produces symbolic observations that pass through the same Bernoulli and deterministic-k encoding grid before PC/GES graph recovery. Each discovery method has 400 runs.

The DAG diagnostics artifacts are generated separately from the paper workflow so paper reruns stay lean.

## Result-artifact mapping

- Topology benchmark (paper Table 1) -> generator definitions in `experiments/<dataset>/validate_*.py` and `experiments/generate_table3_multiseed_run_artifacts.py`
- Single-run benchmark (paper Table 2) -> `results/table_2_*.csv`
- Encoding ablation PC (paper Table 3) -> `results/table_3_pc_*.csv`
- Encoding ablation GES (paper Table 3) -> `results/table_3_ges_*.csv`
- Interventional robustness (paper Table 5) -> `results/table_5_interventional_*.{csv,md}`
- Practical success-rate summary (derived from sweeps; paper Table 4) -> `results/table_4_practical_success_rate_summary.csv`
- DAG diagnostics artifacts (optional, manual) -> run-local `runs/YYYYMMDD_HHMMSS/*_causal_results.txt` and `*_3dag_comparison.png`

## Encoding grid used in the paper

The 400-run sensitivity study uses five datasets, ten seeds, and eight encoding configurations.

Seeds:

```text
42, 73, 101, 131, 151, 181, 211, 241, 271, 301
```

Bernoulli configurations:

```text
positive_prob / negative_prob
0.20 / 0.10
0.30 / 0.05
0.30 / 0.10
0.30 / 0.15
0.40 / 0.10
```

Deterministic-k configurations:

```text
k-step = 5
k-step = 10
k-step = 15
```

Counts:

```text
5 datasets x 10 seeds x 5 Bernoulli configs = 250 Bernoulli runs
5 datasets x 10 seeds x 3 deterministic-k configs = 150 deterministic-k runs
Total = 400 runs per discovery method
```

## Quick start

**Python 3.9 or later is required** for general use. The frozen paper artifacts were produced with **Python 3.13**.

The `.venv/` directory is intentionally not committed. Create a local virtual environment first, then install either the exact reproduction dependencies or the broader development dependencies.

Create and activate a local environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Git Bash, macOS, or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

For exact frozen-artifact reproduction of Tables 1-5, use Python 3.13 and the pinned lockfile:

```powershell
python --version  # expected: Python 3.13.x
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m experiments.generate_paper_data
```

`requirements-lock.txt` is a full `pip freeze` from the canonical latest-env run (Python 3.13.x; generated 2026-08-06). The `--no-deps` flag keeps pip from changing those pinned versions while installing this repository in editable mode, so `src.*` and `experiments.*` imports resolve.

For general development or extension work on Python 3.9+:

```powershell
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` contains broad compatible ranges. It should preserve the paper conclusions, but exact row-level identity with `results/` requires Python 3.13.x, `requirements-lock.txt`, the frozen configuration, and the same script version.

The full pipeline writes fresh artifacts under `runs/YYYYMMDD_HHMMSS/`, where the timestamp is fixed at script start; reruns on the same day therefore create separate folders. Compare or promote those files against `results/` when refreshing the frozen reference set. Each run directory also includes `experiments_configuration.json` and `run_metadata.json`, recording the git commit SHA, Python version, dependency mode, timestamps, working directory, and command arguments for scripts that wrote artifacts there. Set `ASSEMBLIES_ENCODING_DEPENDENCY_MODE=requirements-lock.txt` or `requirements.txt` to override dependency-mode inference.


### Runtime and hardware expectations

The full paper pipeline is CPU-bound and does not require a GPU. Runtime depends mainly on CPU speed, available cores, memory bandwidth, and the installed numerical stack. The PC/GES encoding ablation sweeps dominate the wall-clock time.

Practical baseline for a reasonable local run:

- Python 3.13.x with `requirements-lock.txt` for exact frozen-artifact reproduction.
- Modern 4-core CPU or better; 8 cores / 16 threads is a more comfortable target for full reruns.
- 16 GB RAM is recommended; 8 GB can work for smaller/partial runs but may be less comfortable while other applications are open.
- SSD-backed working directory, because each run writes CSV, text, metadata, and figure artifacts under `runs/YYYYMMDD_HHMMSS/`.

The pipeline runner prints elapsed time per step and total elapsed time. To measure wall-clock time externally:

```powershell
Measure-Command { python -m experiments.generate_paper_data }
```

```bash
time python -m experiments.generate_paper_data
```

Use `python -m experiments.generate_paper_data` for the paper artifact pipeline. Additional diagnostics artifacts are generated manually with `python -m experiments.generate_validation_dag_artifacts`.

Observed on this project with current defaults: `python -m experiments.generate_paper_data` completed in about 2h 39m 50s (final GES sweep step: about 1h 10m).

## What "reproducibility" means in this repository

The frozen artifacts under `results/` are tied to three things: the pinned latest-env environment (`requirements-lock.txt`), frozen run settings (`results/experiments_configuration.json` for the current canonical set), and the script versions that produced them.

Fresh reruns with broad dependencies from `requirements.txt` are expected to preserve the paper conclusions, but exact row-level identity requires Python 3.13.x, `requirements-lock.txt`, the frozen configuration, and the same script version.

## Local generated artifacts

Experiment scripts write local outputs through `experiments/experiment_defaults.py`. Each script process uses a run folder named `runs/YYYYMMDD_HHMMSS/`; the pipeline runner fixes this timestamp at startup and passes the same run id to every child script, so artifacts from one pipeline run land in the same folder.

These local run folders contain generated CSV, text, figure, configuration-snapshot, and `run_metadata.json` files. They are intentionally excluded from version control by `.gitignore` via `runs/`. Keep local reruns there, and only copy/promote selected files into `results/` when intentionally refreshing the frozen paper reference artifacts.

## Paper table runbook (scripts and commands)

Open a shell at the repository root first (the directory that contains `README.md`, `pyproject.toml`, `src/`, and `experiments/`):

```bash
cd path/to/assemblies-encoding
```

Run all `python -m experiments...` commands from this repository root, not from inside `experiments/`; otherwise Python cannot resolve the top-level `experiments` package. All defaults are read from `experiments/experiments_configuration.json`.

### Table 1 - Benchmark topologies

The topology table is fixed by the SCM generator definitions in `experiments/<dataset>/validate_*.py` (imported by `experiments/generate_table3_multiseed_run_artifacts.py`).

### Table 2 - Single-run benchmark (5 datasets)

This workflow uses `n_train=200` from config by default and writes the `table_2_single_run_*.csv` artifact family.

```bash
python -m experiments.generate_table2_single_run_artifacts
```

For detailed per-dataset logs and DAG PNGs, run the validation scripts:

```bash
python -m experiments.alzheimers.validate_alzheimers
python -m experiments.stroke_risk.validate_stroke
python -m experiments.credit_default.validate_credit
python -m experiments.vaccine_efficacy.validate_vaccine
python -m experiments.student_success.validate_student_success
```

### Table 3 - Encoding ablation sweep (400 runs per method)

```bash
python -m experiments.generate_table3_multiseed_run_artifacts --method pc
python -m experiments.generate_table3_multiseed_run_artifacts --method ges
```

### Table 4 - Practical success-rate summary (derived)

```bash
python -m experiments.generate_table4_practical_success_summary --methods pc,ges
```

### Table 5 - Interventional robustness (multiseed)

```bash
python -m experiments.generate_table5_intervention_multiseed
```

### DAG diagnostics artifacts (manual, optional)

This optional step writes per-dataset, per-method, per-encoding causal reports and 3-DAG comparison figures:

```bash
python -m experiments.generate_validation_dag_artifacts
```

Fresh outputs are written under `runs/YYYYMMDD_HHMMSS/`. Frozen reference artifacts used by the paper stay under `results/`.

## Orchestrator commands

Use these entry points depending on what you need:

- Paper artifacts only (recommended for reproducing manuscript tables):

```bash
python -m experiments.generate_paper_data
```

- Optional diagnostics artifacts (manual text/figures):

```bash
python -m experiments.generate_validation_dag_artifacts
```

Preview pipeline plan without running:

```bash
python -m experiments.generate_paper_data --dry-run
```

Select methods explicitly (default is both):

```bash
python -m experiments.generate_paper_data --methods pc,ges
```

## Parameter quick reference

Parameter source of truth:
- Core defaults are loaded from `experiments/experiments_configuration.json`.
- Shared loader and default wiring are implemented in `experiments/experiment_defaults.py`.
- Script CLI arguments override configuration defaults when provided.

Paper pipeline parameters (`experiments/generate_paper_data.py`):
- `--methods`: methods to run, comma-separated, allowed values `pc` and `ges` (default `pc,ges`).
- `--skip-single-run`: skip `generate_table2_single_run_artifacts.py`.
- `--skip-sweep`: skip `generate_table3_multiseed_run_artifacts.py` runs.
- `--dry-run`: print plan and commands only, no execution.

Encoding ablation parameters (`experiments/generate_table3_multiseed_run_artifacts.py`):
- `--datasets`: dataset list (default from `sensitivity_sweep.datasets`).
- `--seeds`: seed list (default from `sensitivity_sweep.seeds`).
- `--n-samples`: per-dataset sample count.
- `--n-train`: training examples for the assembly stage.
- `--n-presentations`: repeated presentation cycles.
- `--method`: causal discovery method, `pc` or `ges`.
- `--alpha-pc`: PC conditional-independence significance level.

Output location:
- New run artifacts are written under `runs/YYYYMMDD_HHMMSS` via `get_output_filepath`.
- Frozen paper reference artifacts remain under `results`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



































