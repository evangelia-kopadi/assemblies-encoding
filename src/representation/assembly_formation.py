"""Neural assembly formation.

NEMO-style neural assembly formation with k-winner
take-all projection and Hebbian plasticity.

How it works: present encoded variable stimuli to Brain areas for repeated
projection rounds; the Brain updates winners/supports through assembly dynamics.
"""

from __future__ import annotations

import numpy as np


def form_assemblies(
    brain,
    neural_data,
    var_names,
    neurons_per_var,
    n_train=200,
    n_presentations=3,
    *,
    target_area_by_var_name=None,
    shuffle_seed=None,
):
    """Form assemblies for variables using the Neural Assemblies Brain method."""

    if target_area_by_var_name is None:
        target_area_by_var_name = {var_name: var_name for var_name in var_names}

    rng = (
        np.random.default_rng(shuffle_seed)
        if shuffle_seed is not None
        else np.random.default_rng()
    )

    stimuli = []
    for sample_idx in range(min(n_train, neural_data.shape[0])):
        for var_idx, var_name in enumerate(var_names):
            start_idx = var_idx * neurons_per_var
            end_idx = start_idx + neurons_per_var

            spikes = neural_data[sample_idx, start_idx:end_idx]
            num_firing = int(spikes.sum())

            if num_firing > 0:
                stim_name = f"{var_name}_n{num_firing}"
                if stim_name not in brain.stimulus_size_by_name:
                    brain.add_stimulus(stim_name, num_firing)
                stimuli.append((var_name, stim_name))

    for _round_idx in range(n_presentations):
        rng.shuffle(stimuli)

        for source_var_name, stim_name in stimuli:
            target_area_name = target_area_by_var_name.get(source_var_name, source_var_name)
            try:
                brain.project(
                    areas_by_stim={stim_name: [target_area_name]},
                    dst_areas_by_src_area={},
                    verbose=0,
                )
            except RuntimeError:
                stimuli = [
                    (var_name, stimulus_name)
                    for (var_name, stimulus_name) in stimuli
                    if target_area_by_var_name.get(var_name, var_name) != target_area_name
                ]
                break

    return brain
