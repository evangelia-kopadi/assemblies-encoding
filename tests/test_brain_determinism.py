import numpy as np

from src.representation.brain import Brain


def _projection_fingerprint():
    brain = Brain(p=0.05, save_size=True, save_winners=True, seed=123)
    brain.add_area("A", n=120, k=10, beta=0.05)
    brain.add_stimulus("stim_20", 20)

    for _ in range(4):
        brain.project(areas_by_stim={"stim_20": ["A"]}, dst_areas_by_src_area={})

    area = brain.area_by_name["A"]
    connectome = brain.connectomes_by_stimulus["stim_20"]["A"]
    return {
        "support": area.w,
        "winners": tuple(area.winners),
        "saved_w": tuple(area.saved_w),
        "connectome": connectome.copy(),
    }


def test_brain_projection_uses_seeded_rng_for_truncated_normal_sampling():
    first = _projection_fingerprint()
    second = _projection_fingerprint()

    assert first["support"] == second["support"]
    assert first["winners"] == second["winners"]
    assert first["saved_w"] == second["saved_w"]
    np.testing.assert_array_equal(first["connectome"], second["connectome"])
