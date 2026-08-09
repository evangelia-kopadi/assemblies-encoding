# assemblies-encoding

Clean reproducibility package for the SUM 2026 paper experiments on causal structure preservation in neural assemblies under neural encoding uncertainty.

This repository is a minimal extraction from the larger development workspace. It keeps only the code, generated artifacts, and figures needed to reproduce or extend the paper experiments.

## Provenance and Attribution

Parts of the neural assembly simulation core (notably `src/representation/brain.py`) are borrowed/adapted from the original assemblies codebase by Daniel Mitropolsky and collaborators (Princeton University & MIT): https://github.com/dmitropolsky/assemblies (source file: https://github.com/dmitropolsky/assemblies/blob/a7ded3b23aa1cce10b8979801a27da8a97bc23d3/brain.py; pinned commit: https://github.com/dmitropolsky/assemblies/commit/a7ded3b23aa1cce10b8979801a27da8a97bc23d3), as acknowledged in-file.

## Associated paper

This repository accompanies the SUM 2026 manuscript **"Causal Structure Preservation in Neural Assemblies under Encoding Uncertainty"**. The editable paper source is available at `paper/causal_structure_preservation_in_neural_assemblies_under_encoding_uncertainty.md`, with its bibliography and build script in the same folder. References to Tables 1-5 in this README refer to that manuscript.

## What is included

- `src/`: neural encoding, assembly formation, causal discovery, validation, and metrics code.
  - `src/encoding/bernoulli.py`: Bernoulli spike-pattern encoding.
  - `src/encoding/deterministic_k.py`: deterministic-k stimulus-set encoding.
  - `src/discovery/pc.py`: PC-specific observational discovery helpers.
  - `src/discovery/ges.py`: GES-specific observational discovery helpers.
  - `src/validation/interventional/pearl_do_calculus.py`: Pearl-style `do(...)` intervention validation helpers.
  - `src/representation/information_preservation_mi.py`: MI-based neuron/assembly information-preservation metrics.
  - `src/representation/assembly_formation.py`: Papadimitriou-style assembly formation.
  - `src/representation/neuron_feature_extraction.py`: neuron-level feature extraction from encoded spike patterns.
  - `src/representation/assembly_feature_extraction.py`: assembly-level feature extraction from learned Brain connectomes.
- `experiments/`: SCM dataset generators used for the paper's main encoding sensitivity sweep:
  - Alzheimer
  - Stroke Risk
  - Credit Default
  - Student Success
  - Vaccine Efficacy
- `notebooks/pipeline_visualization.ipynb`: executable visual walkthrough of the Stage I-VI pipeline for Bernoulli and deterministic-k encodings.
- `experiments/run_sensitivity_sweep.py`: PC/GES encoding sensitivity sweep runner.
- `experiments/student_success/evaluate_student_success_multiseed.py`: Student Success multi-seed graph and intervention robustness check used for the Student Success validation study.
- `results/`: frozen single-run benchmark artifacts under flat compatibility filenames for the manuscript's single-run benchmark table (for example, `table_1_single_run_means.csv` and `table_1_single_run_metrics.csv`).
- `results/`: frozen multi-run PC/GES sensitivity sweep artifacts with paper-mapped filenames: `table_3_pc_*.csv` and `table_4_ges_sensitivity_*.csv`.
- `results/`: sensitivity sweep raw, summary, and overall CSV files for PC (Table 3) and GES (Table 4).
- `results/`: frozen Student Success 50-seed robustness artifacts under flat names such as `table_4_student_summary.csv`, `table_4_student_dodeltas.csv`, and `table_4_student_report.md`.
- `figures/`: figures referenced by the paper source.
- `paper/`: editable paper source, bibliography, and build script.

## Experiment roles

The repository contains both single-run and multi-run artifacts.

The single-run benchmark (`results/`) supports the concrete deterministic-k example summarized in the manuscript's single-run benchmark table. It is useful as a small, inspectable run, not as the main robustness evidence.

The main encoding conclusion is based on the five-dataset, ten-seed sensitivity sweeps (`results/`). In those sweeps, each SCM generator produces symbolic observations that pass through the same Bernoulli and deterministic-k encoding grid before PC/GES graph recovery. Each discovery method has 400 runs.

The Student Success multi-seed experiment (`results/`) is a separate 50-seed robustness and intervention validation check.

## Result-artifact mapping

- Topology benchmark (paper Table 1) -> generator definitions in `experiments/<dataset>/validate_*.py` and `experiments/run_sensitivity_sweep.py`
- Single-run benchmark (paper Table 2; flat compatibility filenames) -> `results/table_1_single_run_*.csv`
- Encoding ablation PC (paper Table 3) -> `results/table_3_pc_*.csv`
- Encoding ablation GES (paper Table 3) -> `results/table_3_ges_*.csv`
- Student Success validation (paper Table 5) -> `results/table_4_student_*.csv`


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

**Python 3.9 or later is required** for general use. The frozen paper artifacts were produced with **Python 3.13.4**.

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

For exact frozen-artifact reproduction of Tables 1-5, use Python 3.13.4 and the pinned lockfile:

```powershell
python --version  # expected: Python 3.13.4
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m experiments.run_all_paper_experiments
```

`requirements-lock.txt` is a full `pip freeze` from the canonical latest-env run (Python 3.13.4; generated 2026-08-06). The `--no-deps` flag keeps pip from changing those pinned versions while installing this repository in editable mode, so `src.*` and `experiments.*` imports resolve.

For general development or extension work on Python 3.9+:

```powershell
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` contains broad compatible ranges. It should preserve the paper conclusions, but exact row-level identity with `results/` requires Python 3.13.4, `requirements-lock.txt`, the frozen configuration, and the same script version.

The full pipeline writes fresh artifacts under `runs/YYYYMMDD_HHMMSS/`, where the timestamp is fixed at script start; reruns on the same day therefore create separate folders. Compare or promote those files against `results/` when refreshing the frozen reference set. Each run directory also includes `experiments_configuration.json` and `run_metadata.json`, recording the git commit SHA, Python version, dependency mode, timestamps, working directory, and command arguments for scripts that wrote artifacts there. Set `ASSEMBLIES_ENCODING_DEPENDENCY_MODE=requirements-lock.txt` or `requirements.txt` to override dependency-mode inference.


### Runtime and hardware expectations

The full paper pipeline is CPU-bound and does not require a GPU. Runtime depends mainly on CPU speed, available cores, memory bandwidth, and the installed numerical stack. The PC/GES sensitivity sweeps dominate the wall-clock time because they run the five-dataset, ten-seed encoding grid for both discovery methods.

Practical baseline for a reasonable local run:

- Python 3.13.4 with `requirements-lock.txt` for exact frozen-artifact reproduction.
- Modern 4-core CPU or better; 8 cores / 16 threads is a more comfortable target for full reruns.
- 16 GB RAM is recommended; 8 GB can work for smaller/partial runs but may be less comfortable while other applications are open.
- SSD-backed working directory, because each run writes CSV, text, metadata, and figure artifacts under `runs/YYYYMMDD_HHMMSS/`.

The one-command orchestrator prints elapsed time for each step and the total runtime. To record an external wall-clock measurement on your own machine, wrap the full command in a shell timer:

```powershell
Measure-Command { python -m experiments.run_all_paper_experiments }
```

```bash
time python -m experiments.run_all_paper_experiments
```

For a quicker smoke test or for machines with limited CPU/RAM, run selected parts first with `--skip-sweep`, `--skip-multiseed`, or `--methods pc` / `--methods ges`.

## What "reproducibility" means in this repository

The frozen artifacts under `results/` are tied to three things: the pinned latest-env environment (`requirements-lock.txt`), frozen run settings (`results/experiments_configuration_frozen_sweeps.json`, synchronized with `experiments/experiments_configuration.json` for the current canonical set), and the script versions that produced them.

Fresh reruns with broad dependencies from `requirements.txt` are expected to preserve the paper conclusions, but exact row-level identity requires Python 3.13.4, `requirements-lock.txt`, the frozen configuration, and the same script version.

## Local generated artifacts

Experiment scripts write local outputs through `experiments/experiment_defaults.py`. Each script process uses a run folder named `runs/YYYYMMDD_HHMMSS/`; the one-command orchestrator fixes this timestamp at startup and passes the same run id to every child script, so all artifacts from one pipeline run land in the same folder.

These local run folders contain generated CSV, text, figure, configuration-snapshot, and `run_metadata.json` files. They are intentionally excluded from version control by `.gitignore` via `runs/`. Keep local reruns there, and only copy/promote selected files into `results/` when intentionally refreshing the frozen paper reference artifacts.

## Paper Table Runbook (Scripts and Commands)

Open a shell at the repository root first, meaning the directory that contains `README.md`, `pyproject.toml`, `src/`, and `experiments/`:

```bash
cd path/to/assemblies-encoding
```

Run the `python -m experiments...` commands from this repository root, not from inside `experiments/`; otherwise Python cannot resolve the top-level `experiments` package. All defaults are read from `experiments/experiments_configuration.json`.

### Table 1 - Benchmark topologies

The topology table is fixed by the SCM generator definitions in `experiments/<dataset>/validate_*.py` and `experiments/run_sensitivity_sweep.py`.

### Table 2 - Single-run benchmark (5 datasets)

This workflow uses `n_train=200` from config by default and writes the flat `table_1_single_run_*.csv` artifact family.

```bash
python -m experiments.generate_single_run_table_artifacts
```

For detailed per-dataset logs and DAG PNGs, run the validation scripts:

```bash
python -m experiments.alzheimers.validate_alzheimers
python -m experiments.stroke_risk.validate_stroke
python -m experiments.credit_default.validate_credit
python -m experiments.vaccine_efficacy.validate_vaccine
python -m experiments.student_success.validate_student_success
```

### Table 3 - PC sensitivity sweep (400 runs)

```bash
python -m experiments.run_sensitivity_sweep --method pc
```

### Table 4 - GES robustness sweep (400 runs)

```bash
python -m experiments.run_sensitivity_sweep --method ges
```

### Compact PC/GES derived artifacts

After both full sweeps have run, generate the compact single-seed PC/GES comparison files:

```bash
python -m experiments.generate_compact_pc_ges_artifacts
```

### Table 5 - Student 50-seed robustness

```bash
python -m experiments.student_success.evaluate_student_success_multiseed
```

Fresh outputs are written under `runs/YYYYMMDD_HHMMSS/`. Frozen reference artifacts used by the paper stay under `results/`.

## Notes for extension

- Add a new SCM generator under `experiments/<dataset>/`.
- Register it in `experiments/run_sensitivity_sweep.py` inside `build_datasets()`.
- Add new encoding configurations in `build_sweep_configs()`.
- Keep discovery settings fixed when comparing representation effects, unless the goal is explicitly algorithm tuning.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.




## One-command full pipeline (all paper tables)

To run all paper data-generation scripts in the recommended order (Table 1 topology sources, Table 2 single-run artifacts, Table 3 PC sweep, Table 4 GES sweep, compact PC/GES derived artifacts, Table 5 multiseed), use:

```bash
python -m experiments.run_all_paper_experiments
```

Preview the planned commands without running:

```bash
python -m experiments.run_all_paper_experiments --dry-run
```

Skip parts if needed:

```bash
python -m experiments.run_all_paper_experiments --skip-validate
python -m experiments.run_all_paper_experiments --skip-sweep
python -m experiments.run_all_paper_experiments --skip-multiseed
```

Select methods explicitly (default is both):

```bash
python -m experiments.run_all_paper_experiments --methods pc,ges
```

## Parameter quick reference

Parameter source of truth:
- Core defaults are loaded from experiments/experiments_configuration.json.
- Shared loader and default wiring are implemented in experiments/experiment_defaults.py.
- Script CLI arguments override configuration defaults when provided.

Orchestrator parameters (experiments/run_all_paper_experiments.py):
- --methods: sweep methods to run, comma-separated, allowed values pc and ges (default pc,ges).
- --skip-validate: skip all five validate_*.py scripts and the single-run CSV generator.
- --skip-sweep: skip run_sensitivity_sweep.py runs and compact PC/GES derived artifacts.
- --skip-multiseed: skip evaluate_student_success_multiseed.py.
- --dry-run: print plan and commands only, no execution.

Sensitivity sweep parameters (experiments/run_sensitivity_sweep.py):
- --datasets: dataset list (default from sensitivity_sweep.datasets).
- --seeds: seed list (default from sensitivity_sweep.seeds).
- --n-samples: per-dataset sample count.
- --n-train: training examples for the assembly stage.
- --n-presentations: repeated presentation cycles.
- --method: causal discovery method, pc or ges.
- --alpha-pc: PC conditional-independence significance level.

Student multiseed parameters (experiments/student_success/evaluate_student_success_multiseed.py):
- --n-seeds: number of sequential seeds to evaluate.
- --seed0: starting seed.
- --n-samples: training cohort size per seed.
- --n-eval: intervention evaluation cohort size.
- --alpha-primary: primary PC alpha.
- --alpha-fallback: fallback PC alpha if needed.
- --verbose: print per-seed logs.

Output location:
- New run artifacts are written under runs/YYYYMMDD_HHMMSS via get_output_filepath.
- Frozen paper reference artifacts remain under results.
