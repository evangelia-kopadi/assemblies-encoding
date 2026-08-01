# Manifest

## Paper

- `paper/causal_structure_preservation_in_neural_assemblies_under_encoding_uncertainty.md`: editable SUM 2026 manuscript source.
- `paper/references.bib`: bibliography used by the manuscript.
- `paper/llncs-pandoc-template.tex`: LNCS Pandoc template.
- `paper/build_paper.ps1`: local paper build script.

## Code

- `src/`: core implementation copied from `naca-validation`.
  - `src/encoding/bernoulli.py`: Bernoulli spike-pattern encoding.
  - `src/encoding/deterministic_k.py`: deterministic-k stimulus-set encoding.
  - `src/discovery/pc.py`: PC-specific observational discovery helpers.
  - `src/discovery/ges.py`: GES-specific observational discovery helpers.
  - `src/validation/pearl_do_calculus.py`: Pearl-style `do(...)` intervention validation helpers.
  - `src/causal_dag.py`: DAG comparison, reporting, and visualization helpers.
  - `src/validation/information_preservation_mi.py`: MI-based neuron/assembly information-preservation metrics.
  - `src/validation/assembly_formation.py`: Papadimitriou-style assembly formation.
  - `src/validation/neuron_feature_extraction.py`: neuron-level feature extraction from encoded spike patterns.
  - `src/validation/assembly_feature_extraction.py`: assembly-level feature extraction from learned Brain connectomes.
- `run_sensitivity_sweep.py`: sensitivity sweep used for PC and GES experiments.
- `experiments/alzheimers/validate_alzheimers.py`
- `experiments/stroke_risk/validate_stroke.py`
- `experiments/credit_default/validate_credit.py`
- `experiments/student_success/validate_student_success.py`
- `experiments/student_success/evaluate_student_success_multiseed.py`
- `experiments/vaccine_efficacy/validate_vaccine.py`

## Notebooks

- `notebooks/pipeline_visualization.ipynb`: visual Stage I-VI walkthrough of the neuron and assembly pipeline under Bernoulli and deterministic-k encodings.

## Frozen results

### Single-run benchmark

- `results/single_run/pc_seed42_metrics.csv`
- `results/single_run/pc_seed42_means.csv`

### Multi-run sensitivity sweeps

- `results/sensitivity/pc/raw.csv`
- `results/sensitivity/pc/summary.csv`
- `results/sensitivity/pc/overall.csv`
- `results/sensitivity/ges/raw.csv`
- `results/sensitivity/ges/summary.csv`
- `results/sensitivity/ges/overall.csv`

### Table summaries

- `results/table_summaries/raw_stats_for_tables.csv`
- `results/table_summaries/raw_stats_for_tables.txt`
- `results/table_summaries/pc_ges_compact_summary.csv`
- `results/table_summaries/pc_ges_compact_raw.csv`

### Student Success Table 4 robustness check

- `results/student_success_table4/summary.csv`
- `results/student_success_table4/do_deltas.csv`
- `results/student_success_table4/report.md`

## Frozen figures

- `figures/causal_information_preservation_pipeline.png`
- `figures/local-to-global-index-mapping-exact-code.png`
- `figures/raw_vs_assembly_local_global_example.png`
- `figures/bernoulli_simple_3dag.png`
- `figures/deterministic_k_simple_3dag.png`
- `figures/sensitivity_full_overview_400runs.png`
- `figures/sensitivity_gap_representative_400runs.png`

