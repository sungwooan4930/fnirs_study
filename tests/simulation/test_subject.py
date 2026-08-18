import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.subject import SubjectProfile, make_subjects


def test_makes_requested_number():
    rng = set_all_seeds(0)
    subs = make_subjects(7, 0.5, rng)
    assert len(subs) == 7
    assert all(isinstance(s, SubjectProfile) for s in subs)


def test_subject_ids_are_zero_padded_and_unique():
    rng = set_all_seeds(0)
    subs = make_subjects(3, 0.5, rng)
    assert [s.subject_id for s in subs] == ["sub-01", "sub-02", "sub-03"]


def test_zero_variance_gives_identical_thetas():
    rng = set_all_seeds(0)
    subs = make_subjects(5, 0.0, rng)
    assert all(s.theta == 0.0 for s in subs)


def test_larger_variance_spreads_thetas_more():
    thetas_small = np.array([s.theta for s in make_subjects(200, 0.25, set_all_seeds(1))])
    thetas_large = np.array([s.theta for s in make_subjects(200, 4.0, set_all_seeds(1))])
    assert thetas_large.std() > thetas_small.std()


def test_profile_is_immutable():
    import dataclasses
    import pytest

    sub = make_subjects(1, 0.5, set_all_seeds(0))[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        sub.theta = 99.0


def test_theta_sd_equals_sqrt_of_variance():
    """Verify that theta sample SD equals sqrt(subject_variance), not variance itself.

    Draws 2000 subjects at variance=4.0. Expected SD is 2.0.
    Tolerance 0.15 accepts the correct sqrt() conversion but rejects
    the un-sqrt'd value (4.0 would be 2.0 away, failing the assertion).
    """
    rng = set_all_seeds(42)
    subs = make_subjects(2000, 4.0, rng)
    thetas = np.array([s.theta for s in subs])
    observed_sd = thetas.std(ddof=0)  # population std
    expected_sd = np.sqrt(4.0)  # Should be 2.0
    assert abs(observed_sd - expected_sd) < 0.15
