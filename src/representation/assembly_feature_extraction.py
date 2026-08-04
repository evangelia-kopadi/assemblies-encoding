"""Assembly-level feature extraction from a fixed Brain.

Fixed-Brain assembly feature extraction from learned connectomes.

How it works: reuse a Brain whose assemblies have already been formed, project
encoded neural samples through the learned stimulus-area connectomes, and reduce
each target area to one assembly-level scalar feature per sample.
"""

from __future__ import annotations

import numpy as np


def extract_assembly_features(
    neural_data,
    brain,
    var_names,
    neurons_per_var=1666,
    *,
    source_var_by_target_area_name=None,
    top_k=None,
):
    """Extract one scalar assembly feature per variable per sample.

    In the default Brain simulation, Area.winners are indices into the assembly
    support, not indices into the original neuron pool. We therefore derive an
    assembly scalar from learned stimulus-area connectomes stored in
    brain.connectomes_by_stimulus.
    """

    n_samples = neural_data.shape[0]
    assembly_features = {}
    name_to_idx = {var_name: idx for idx, var_name in enumerate(var_names)}

    for target_area_name in var_names:
        source_var_name = (
            source_var_by_target_area_name.get(target_area_name, target_area_name)
            if source_var_by_target_area_name is not None
            else target_area_name
        )
        if source_var_name not in name_to_idx:
            raise ValueError(
                f"Unknown source var {source_var_name!r} for target area {target_area_name!r}"
            )

        source_var_idx = name_to_idx[source_var_name]
        start_idx = source_var_idx * neurons_per_var
        end_idx = start_idx + neurons_per_var
        area = brain.area_by_name[target_area_name]

        if top_k is not None:
            k = int(top_k)
        else:
            k = int(getattr(area, "k", 0) or 0)

        activations = np.zeros(n_samples, dtype=float)

        for sample_idx in range(n_samples):
            spikes = neural_data[sample_idx, start_idx:end_idx]
            num_firing = int(spikes.sum())

            if num_firing <= 0:
                activations[sample_idx] = 0.0
                continue

            stim_name = f"{source_var_name}_n{num_firing}"

            try:
                conn = brain.connectomes_by_stimulus[stim_name][target_area_name]
            except Exception:
                conn = None

            if conn is None or getattr(conn, "size", 0) == 0:
                activations[sample_idx] = float(num_firing) / float(neurons_per_var)
                continue

            weights = np.asarray(conn, dtype=float)
            if k > 0 and weights.size > k:
                idx = np.argpartition(weights, -k)[-k:]
                activations[sample_idx] = float(weights[idx].mean())
            else:
                activations[sample_idx] = float(weights.mean())

        assembly_features[target_area_name] = activations

    return assembly_features
