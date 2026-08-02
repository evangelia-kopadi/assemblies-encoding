# assemblies-encoding

Clean reproducibility package for the SUM 2026 paper experiments on causal structure preservation in neural assemblies under neural encoding uncertainty.

This repository is a minimal extraction from the larger development workspace. It keeps only the code, generated artifacts, and figures needed to reproduce or extend the paper experiments.

## Provenance and Attribution

Parts of the neural assembly simulation core (notably `src/representation/brain.py`) are borrowed/adapted from the original assemblies codebase by Daniel Mitropolsky and collaborators (Princeton University & MIT): https://github.com/dmitropolsky/assemblies (source file: https://github.com/dmitropolsky/assemblies/blob/a7ded3b23aa1cce10b8979801a27da8a97bc23d3/brain.py; pinned commit: https://github.com/dmitropolsky/assemblies/commit/a7ded3b23aa1cce10b8979801a27da8a97bc23d3), as acknowledged in-file.

## Associated paper

This repository accompanies the SUM 2026 manuscript **"Causal Structure Preservation in Neural Assemblies under Encoding Uncertainty"**. The editable paper source is available at `paper/causal_structure_preservation_in_neural_assemblies_under_encoding_uncertainty.md`, with its bibliography and build script in the same folder. References to Tables 1-4 in this README refer to that manuscript.

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
- `run_sensitivity_sweep.py`: PC/GES encoding sensitivity sweep runner.
- `experiments/student_success/evaluate_student_success_multiseed.py`: Student Success multi-seed graph and intervention robustness check used for the paper's Table 4.
- `results/single_run/`: frozen single-run benchmark artifacts for the deterministic-k seed-42 run used in Table 1.
- `results/sensitivity/`: frozen multi-run PC/GES sensitivity sweep artifacts used for the paper's Tables 2-3.
- `results/table_summaries/`: compact table-summary CSV/TXT files derived from the frozen runs.
- `results/student_success_table4/`: frozen Student Success 50-seed robustness artifacts used for the paper's Table 4.
- `figures/`: figures referenced by the paper source.
- `paper/`: editable paper source, bibliography, and build script.

## Experiment roles

The repository contains both single-run and multi-run artifacts.

The single-run benchmark (`results/single_run/`) supports the concrete deterministic-k example summarized in Table 1. It is useful as a small, inspectable run, not as the main robustness evidence.

The main encoding conclusion is based on the five-dataset, ten-seed sensitivity sweeps (`results/sensitivity/`). In those sweeps, each SCM generator produces symbolic observations that pass through the same Bernoulli and deterministic-k encoding grid before PC/GES graph recovery. Each discovery method has 400 runs.

The Student Success multi-seed experiment (`results/student_success_table4/`) is a separate 50-seed robustness and intervention check for the paper's Table 4.

## Table-to-artifact mapping

- Table 1 -> `results/single_run/`
- Tables 2-3 -> `results/sensitivity/` and `results/table_summaries/`
- Table 4 -> `results/student_success_table4/`


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

**Python 3.9 or later is required** (developed and tested on Python 3.13).

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Single-run Table 1 rerun

Table 1 is produced by running the per-dataset validate scripts at seed 42 with deterministic-k step 10. For example, to reproduce the Student Success row:

```bash
python experiments/student_success/validate_student_success.py
```

Run the corresponding `validate_*.py` under `experiments/<dataset>/` for the other four datasets. Each script writes its metrics to the terminal; the frozen values used in the paper are kept under `results/single_run/`.

## Smoke-test rerun

Use a small run first:

```bash
python run_sensitivity_sweep.py --datasets Student,Vaccine --seeds 42 --n-samples 200 --n-train 20 --method pc
```

This writes local `sensitivity_sweep_*.csv` files in the repository root. They are ignored by git so that the frozen paper artifacts under `results/sensitivity/` remain unchanged. To compare a fresh rerun against the frozen values, open both CSVs side by side; column names and row order match.

## Full sensitivity reruns

PC:

```bash
python run_sensitivity_sweep.py --datasets Alzheimers,Stroke,Credit,Student,Vaccine --seeds 42,73,101,131,151,181,211,241,271,301 --n-samples 800 --n-train 120 --method pc
```

GES:

```bash
python run_sensitivity_sweep.py --datasets Alzheimers,Stroke,Credit,Student,Vaccine --seeds 42,73,101,131,151,181,211,241,271,301 --n-samples 800 --n-train 120 --method ges
```

## Student Success paper Table 4 rerun

```bash
python experiments/student_success/evaluate_student_success_multiseed.py --n-seeds 50 --seed0 42 --n-samples 1000 --n-eval 1000
```

This writes fresh outputs under `experiments/student_success/` and `docs/results/`. The frozen values for the paper's Table 4 are kept under `results/student_success_table4/`.

## Notes for extension

- Add a new SCM generator under `experiments/<dataset>/`.
- Register it in `run_sensitivity_sweep.py` inside `build_datasets()`.
- Add new encoding configurations in `build_sweep_configs()`.
- Keep discovery settings fixed when comparing representation effects, unless the goal is explicitly algorithm tuning.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.







