"""MI-based information-preservation metrics.

Full method name: Mutual Information (MI) information-preservation validation.

How it works: discretize each continuous neuron/assembly feature at its median;
compute pairwise MI matrices for neuron features and assembly features; compare
the absolute difference between the two MI matrices.
"""

from __future__ import annotations

import numpy as np


def mutual_information(x_values, y_values):
    """Compute mutual information MI(X;Y) in bits for discrete values."""
    joint = {}
    for x_value, y_value in zip(x_values, y_values):
        joint[(x_value, y_value)] = joint.get((x_value, y_value), 0) + 1

    n = len(x_values)
    p_x = {}
    p_y = {}

    for x_value in x_values:
        p_x[x_value] = p_x.get(x_value, 0) + 1
    for y_value in y_values:
        p_y[y_value] = p_y.get(y_value, 0) + 1

    mi = 0.0
    for (x_value, y_value), count in joint.items():
        p_xy = count / n
        if p_xy > 0 and p_x[x_value] > 0 and p_y[y_value] > 0:
            mi += p_xy * np.log2(
                p_xy / ((p_x[x_value] / n) * (p_y[y_value] / n))
            )

    return mi


def compute_mi_matrix(features, var_names):
    """Compute pairwise MI matrix for all variables.

    Continuous feature arrays are discretized at their median before MI is
    estimated, matching the original validation behavior.
    """
    n_vars = len(var_names)
    mi_matrix = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(var_names):
        for j, var2 in enumerate(var_names):
            if i == j:
                continue

            x_values = (features[var1] > np.median(features[var1])).astype(int)
            y_values = (features[var2] > np.median(features[var2])).astype(int)
            mi_matrix[i, j] = mutual_information(x_values, y_values)

    return mi_matrix


def validate_information_preservation(mi_neurons, mi_assemblies, threshold=0.1):
    """Compare neuron-level and assembly-level MI matrices."""
    delta_mi = np.abs(mi_neurons - mi_assemblies)
    delta_mi_mean = delta_mi[delta_mi > 0].mean() if (delta_mi > 0).any() else 0.0
    delta_mi_max = delta_mi.max()

    return {
        "delta_mi_mean": delta_mi_mean,
        "delta_mi_max": delta_mi_max,
        "threshold": threshold,
        "passed": delta_mi_max < threshold,
        "delta_matrix": delta_mi,
    }