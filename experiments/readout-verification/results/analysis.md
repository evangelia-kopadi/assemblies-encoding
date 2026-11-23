# Readout verification: results

## PC  (1050 runs, commits ['07ac996'])

R0 replication (A0 vs stored mean assembly F1):
  Bernoulli_pos0.20_neg0.10: 0.337 vs 0.337 (diff +0.000)
  Bernoulli_pos0.30_neg0.05: 0.516 vs 0.516 (diff +0.000)
  Bernoulli_pos0.30_neg0.10: 0.455 vs 0.455 (diff +0.000)
  Bernoulli_pos0.30_neg0.15: 0.361 vs 0.361 (diff +0.000)
  Bernoulli_pos0.40_neg0.10: 0.549 vs 0.549 (diff +0.000)
  R0: within 0.05

H1 (A1 gap non-inferior, margin 0.05): mean gap -0.016 [-0.030, -0.002] -> MET
H2 (A1 - A0 assembly F1 > 0): +0.370 [+0.307, +0.431] -> MET
H3 (FC gap non-inferior, margin 0.05): mean gap +0.000 [+0.000, +0.000] -> MET

Secondary: A2 gap (H1 criterion): mean gap -0.010 [-0.026, +0.007] -> MET
Secondary: A1 - DK assembly F1: -0.002 [-0.024, +0.020]
Secondary: Spearman(p+ - p-, setting mean F1) under A0: +1.00
Secondary: Spearman(p+ - p-, setting mean F1) under A1: +0.90

Means per setting and arm (assembly F1 / gap / unseen share):
                                 Neuron_F1  Assembly_F1    Gap  Unseen_share
  Setting                   Arm                                             
  Bernoulli_pos0.20_neg0.10 A0       0.832        0.337 -0.495         0.273
                            A1       0.832        0.801 -0.031         0.273
                            A2       0.832        0.819 -0.013         0.273
                            FC       0.834        0.834  0.000         0.000
  Bernoulli_pos0.30_neg0.05 A0       0.829        0.516 -0.313         0.250
                            A1       0.829        0.819 -0.009         0.250
                            A2       0.829        0.829  0.001         0.250
                            FC       0.834        0.834  0.000         0.000
  Bernoulli_pos0.30_neg0.10 A0       0.832        0.455 -0.377         0.289
                            A1       0.832        0.816 -0.015         0.289
                            A2       0.832        0.823 -0.009         0.289
                            FC       0.834        0.834  0.000         0.000
  Bernoulli_pos0.30_neg0.15 A0       0.821        0.361 -0.460         0.314
                            A1       0.821        0.790 -0.031         0.314
                            A2       0.821        0.800 -0.021         0.314
                            FC       0.834        0.834  0.000         0.000
  Bernoulli_pos0.40_neg0.10 A0       0.830        0.549 -0.281         0.296
                            A1       0.830        0.837  0.008         0.296
                            A2       0.830        0.823 -0.007         0.296
                            FC       0.834        0.834  0.000         0.000
  Deterministic_kstep10     DK       0.840        0.815 -0.025         0.000

Means per dataset and arm (gap):
  Arm            A0     A1     A2     DK   FC
  Dataset                                    
  Alzheimers -0.368 -0.001 -0.001  0.000  0.0
  Credit     -0.468 -0.003  0.002  0.000  0.0
  Stroke     -0.589 -0.008 -0.025 -0.042  0.0
  Student    -0.417 -0.017 -0.022 -0.085  0.0
  Vaccine    -0.084 -0.048 -0.003  0.000  0.0

Verdict inputs: H1=met, H2=met, H3=met

## GES  (1050 runs, commits ['07ac996'])

R0 replication (A0 vs stored mean assembly F1):
  Bernoulli_pos0.20_neg0.10: 0.477 vs 0.477 (diff +0.000)
  Bernoulli_pos0.30_neg0.05: 0.666 vs 0.666 (diff +0.000)
  Bernoulli_pos0.30_neg0.10: 0.574 vs 0.574 (diff +0.000)
  Bernoulli_pos0.30_neg0.15: 0.453 vs 0.453 (diff +0.000)
  Bernoulli_pos0.40_neg0.10: 0.604 vs 0.604 (diff +0.000)
  R0: within 0.05

H1 (A1 gap non-inferior, margin 0.05): mean gap -0.022 [-0.039, -0.009] -> MET
H2 (A1 - A0 assembly F1 > 0): +0.345 [+0.285, +0.401] -> MET
H3 (FC gap non-inferior, margin 0.05): mean gap +0.000 [+0.000, +0.000] -> MET

Secondary: A2 gap (H1 criterion): mean gap -0.025 [-0.039, -0.012] -> MET
Secondary: A1 - DK assembly F1: -0.012 [-0.038, +0.013]
Secondary: Spearman(p+ - p-, setting mean F1) under A0: +0.80
Secondary: Spearman(p+ - p-, setting mean F1) under A1: +0.70

Means per setting and arm (assembly F1 / gap / unseen share):
                                 Neuron_F1  Assembly_F1    Gap  Unseen_share
  Setting                   Arm                                             
  Bernoulli_pos0.20_neg0.10 A0       0.912        0.477 -0.435         0.273
                            A1       0.912        0.866 -0.046         0.273
                            A2       0.912        0.884 -0.028         0.273
                            FC       0.933        0.933  0.000         0.000
  Bernoulli_pos0.30_neg0.05 A0       0.926        0.666 -0.259         0.250
                            A1       0.926        0.927  0.001         0.250
                            A2       0.926        0.917 -0.009         0.250
                            FC       0.933        0.933  0.000         0.000
  Bernoulli_pos0.30_neg0.10 A0       0.926        0.574 -0.352         0.289
                            A1       0.926        0.919 -0.007         0.289
                            A2       0.926        0.910 -0.017         0.289
                            FC       0.933        0.933  0.000         0.000
  Bernoulli_pos0.30_neg0.15 A0       0.914        0.453 -0.462         0.314
                            A1       0.914        0.878 -0.036         0.314
                            A2       0.914        0.867 -0.047         0.314
                            FC       0.933        0.933  0.000         0.000
  Bernoulli_pos0.40_neg0.10 A0       0.932        0.604 -0.328         0.296
                            A1       0.932        0.910 -0.022         0.296
                            A2       0.932        0.908 -0.024         0.296
                            FC       0.933        0.933  0.000         0.000
  Deterministic_kstep10     DK       0.926        0.912 -0.014         0.000

Means per dataset and arm (gap):
  Arm            A0     A1     A2     DK   FC
  Dataset                                    
  Alzheimers -0.569 -0.032 -0.035  0.000  0.0
  Credit     -0.343 -0.005 -0.016  0.000  0.0
  Stroke     -0.460  0.005 -0.002 -0.022  0.0
  Student    -0.281  0.003  0.000 -0.046  0.0
  Vaccine    -0.183 -0.081 -0.072  0.000  0.0

Verdict inputs: H1=met, H2=met, H3=met
