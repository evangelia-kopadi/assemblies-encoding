# assemblies-encoding

Reproducibility package for the paper **"Causal Structure Preservation in Neural Assemblies under Encoding Uncertainty"** (SUM 2026).

## External dependencies

- `src/representation/brain.py` is adapted from [dmitropolsky/assemblies](https://github.com/dmitropolsky/assemblies) (pinned commit `a7ded3b`), as acknowledged in-file.
- Causal discovery (PC and GES) uses the [causal-learn](https://github.com/py-why/causal-learn) library.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Use `requirements-lock.txt` with Python 3.13 for exact frozen-artifact reproduction. Use `requirements.txt` for general use on Python 3.9+.

## Reproduce paper tables

All commands run from the repository root.

**Tables 2–4** (single command):
```bash
python -m experiments.generate_paper_data
```

**Table 5** (interventional sign-match, run separately — heavy; paper used `--n-seeds 50`):
```bash
python -m experiments.generate_table5_multiseed_intervention          # default: 10 seeds
python -m experiments.generate_table5_multiseed_intervention --n-seeds 50  # paper replication
```

**Individual tables 2–4:**
```bash
python -m experiments.generate_table2_single_run_artifacts                      # Table 2
python -m experiments.generate_table3_multiseed_run_artifacts --method pc       # Table 3 (PC)
python -m experiments.generate_table3_multiseed_run_artifacts --method ges      # Table 3 (GES)
python -m experiments.generate_table4_practical_success_summary --methods pc,ges # Table 4
```

**DAG diagnostic figures (optional):**
```bash
python -m experiments.generate_validation_dag_artifacts
```

Fresh outputs are written under `runs/YYYYMMDD_HHMMSS/`. Frozen reference artifacts are under `results/`.

## Frozen results

`results/` contains the canonical paper artifacts. The SCM topologies are defined in `experiments/<dataset>/validate_*.py`.

## License

MIT — see [LICENSE](LICENSE).
