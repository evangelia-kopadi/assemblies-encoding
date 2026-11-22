# Readout verification: does the Bernoulli loss come from the assembly readout? Preregistration

**Paper:** E. Kopadi and D. Kalles, "Causal Structure Preservation in Neural Assemblies Under Encoding Uncertainty", SUM 2026 (thesis Chapter 5).
**Written:** 2026-11-22, before the confirmatory runs. The scripts in this folder were written and smoke-tested beforehand on one dataset and one seed that is not part of the confirmatory set (Stroke, seed 999, PC). Those smoke-test values were seen; they are kept in `results/readout_sweep_pc_smoke.csv` and are not part of the analysis.

## Background

In the encoding sensitivity sweep, assembly-level F1 was well below raw-neuron F1 under Bernoulli encoding and close to it under deterministic-k. The mean assembly-minus-neuron gap was −0.385 (PC) and −0.367 (GES) for Bernoulli, against −0.025 and −0.013 for deterministic-k. The thesis attributes the loss to the encoding (Section 5.6.1).

Assembly formation sees only the number of active neurons in a variable's block, and creates one stimulus per distinct count among the 120 training rows. The assembly readout looks up the stimulus of a row's exact count. When that count was not formed, the readout falls back to the raw rate c/N (`src/representation/assembly_feature_extraction.py`). The fallback is on a different scale: about 0.1–0.3, against top-k weight means of about 15–40 for formed stimuli.

Under deterministic-k every count is formed. Under Bernoulli, about 30% of rows have an unseen count.

A single-variable diagnostic (binary variable, Bernoulli 0.30/0.10, one seed) gave the following correlations between readout and value:

| Rows | Readout | Correlation |
|:--|:--|--:|
| Rows with a formed count | assembly readout | 0.99 |
| All rows | original readout | 0.37 |
| All rows | readout using the nearest formed count | 0.99 |

This study tests whether that result carries over to the full pipeline.

## Rules

1. The criteria below are fixed by this commit. Any later change is made in a new commit that states the reason. Results are then reported under both the original and the changed criteria.
2. The library in `src/` and the original experiment scripts are not modified. The variants are swapped in during each run by replacing the names that `src.runner` imported (`readout_sweep.py`).
3. Every result row records the seed, the configuration and the git commit.
4. Results are reported whatever they are. A failed criterion is reported as a failure.

## Setup (as in the original sweep, `experiments/generate_table3_multiseed_run_artifacts.py`)

- **Datasets:** the five SCMs (Alzheimers, Stroke, Credit, Student, Vaccine), 800 rows.
- **Seeds:** 42, 73, 101, 131, 151, 181, 211, 241, 271, 301.
- **Network and formation:** N = 1666 neurons per block, k = 100, p = β = 0.05, 120 training rows, 3 presentation rounds.
- **Discovery:** PC (α = 0.05) and GES, run separately, with the same settings as the original sweep.
- **Bernoulli settings (p+/p−):** 0.20/0.10, 0.30/0.10, 0.40/0.10, 0.30/0.05, 0.30/0.15.

**Arms.** All arms share the same seed per dataset and seed. A0–A2 share the encoding and the formed network, and differ only in the readout.

| Arm | Encoding | Readout of a row whose count was not formed |
|:--|:--|:--|
| A0 | Bernoulli | raw rate c/N (original) |
| A1 | Bernoulli | stimulus of the nearest formed count for that variable (ties: the smaller count) |
| A2 | Bernoulli | weights drawn as the Brain draws them for a new stimulus (Binomial(count, p) per support neuron), without changing the network; same top-k mean |
| FC | Fixed-count Bernoulli: every row activates a fresh random set of exactly round(N·p+) or round(N·p−) neurons | original (no count is unseen) |
| DK | Deterministic-k, step 10 (reference) | original |

**Unit of analysis.** The 50 dataset-by-seed cells. For each Bernoulli arm the cell value is the mean over the five settings. Confidence intervals are 95% percentile intervals from a cluster bootstrap over cells (10,000 resamples, generator seed 0).

## Criteria (each evaluated separately for PC and for GES)

- **R0, replication check** (not a hypothesis): A0 reproduces the stored sweep (`results/table_3_{pc,ges}_overall.csv`). The mean assembly F1 per Bernoulli setting is within 0.05 of the stored value. A larger difference is reported and flagged, and the analysis still proceeds.
- **H1, primary.** Under A1, the mean assembly-minus-neuron gap over the Bernoulli runs is non-inferior to zero with margin 0.05: the lower bound of its 95% interval is at least −0.05.
- **H2.** A1 improves on A0: the paired difference in mean assembly F1 (A1 − A0) has a 95% interval whose lower bound is above 0.
- **H3.** Under FC, the mean assembly-minus-neuron gap is non-inferior to zero with margin 0.05, as in H1.

**Reported alongside.**
- A2 under the H1 criterion.
- Per setting and per dataset means for all arms.
- The share of rows with an unseen count.
- The paired difference A1 − DK in assembly F1, with its interval.
- Spearman correlation between p+ − p− and the setting means under A1 and A0, to see whether separation still orders the settings.

## Interpretation fixed in advance

| Outcome (both PC and GES) | Reading |
|:--|:--|
| H1, H2 and H3 hold | The assembly-level loss under Bernoulli in the original sweep comes mainly from the readout's treatment of unseen counts. Chapter 5 (5.3, 5.4, 5.6.1) is revised accordingly, with a note on the published version. |
| H2 holds, H1 fails | The fallback explains part of the loss. The remaining gap under A1 is reported as not explained by this readout change. |
| H2 fails | The single-variable diagnostic does not carry over to the full pipeline. Chapter 5's reading stands, and this result is added to it. |
| H3 holds | A random choice of active positions does no harm once the count is reproducible. This is consistent with formation using only the count. |
| PC and GES disagree | Reported per method. No combined verdict is drawn. |

## Run

```bash
python experiments/readout-verification/readout_sweep.py --method pc
python experiments/readout-verification/readout_sweep.py --method ges
python experiments/readout-verification/analyse.py
```

1,050 runs per method (50 cells × 21 arm-setting combinations), about 4.5 minutes per cell on the development machine.
