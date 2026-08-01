# Student Success: multi-seed evaluation (n=50)

- Seeds: 42 .. 91 (total 50)
- Samples per seed: n_samples=1000, n_eval=1000

## Defaults

- neurons_per_var=1666, k=100, n_train=200, n_presentations=3, beta=0.05, positive_prob=0.3, negative_prob=0.1

## 3-DAG metrics (mean with 95% CI)

- Neuron DAG vs GT F1: 0.944 [0.919, 0.969]
- Assembly DAG vs GT F1: 0.944 [0.919, 0.969]
- Preservation score: 0.972 [0.959, 0.984]

## do() effect preservation

- Sign agreement (pooled across seeds x interventions):
  - Exam: neuron=1.000, assembly=1.000
  - Grade: neuron=1.000, assembly=1.000
- Spearman correlation of effect deltas (pooled):
  - Exam: raw vs neuron=1.000, raw vs assembly=1.000
  - Grade: raw vs neuron=1.000, raw vs assembly=1.000

## Artifacts

- Summary CSV: `experiments/student_success/student_success_multiseed_summary.csv`
- do() deltas CSV: `experiments/student_success/student_success_multiseed_dodeltas.csv`

## Magnitude / importance preservation (beyond direction)

These metrics ask whether *bigger raw do-effects correspond to bigger feature-space effects*, not just the same sign.

| Outcome | PearsonR_raw_vs_neuron | PearsonR_raw_vs_assembly | SpearmanRho_abs_raw_vs_abs_neuron | SpearmanRho_abs_raw_vs_abs_assembly | CalibSlope_neuron_to_raw | CalibR2_neuron_to_raw | CalibMAE_neuron_to_raw | CalibSlope_assembly_to_raw | CalibR2_assembly_to_raw | CalibMAE_assembly_to_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exam | 1.000 | 1.000 | 1.000 | 1.000 | 5.002 | 1.000 | 0.000 | 5.001 | 1.000 | 0.001 |
| Grade | 1.000 | 1.000 | 1.000 | 1.000 | 5.001 | 1.000 | 0.000 | 5.001 | 1.000 | 0.000 |

Notes: PearsonR uses signed deltas; SpearmanRho uses |delta| to test importance ordering; calibration fits raw ≈ a·feature.

