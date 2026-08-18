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


def test_zero_variance_does_not_shift_the_random_stream():
    """τ²=0에서도 생성기 상태 소모량이 같아야 한다.

    np.zeros로 우회하면 rng.normal이 소모했을 상태가 남아, 이후에
    뽑히는 블록 순서·노이즈가 τ²>0 조건과 전혀 다른 난수열이 된다.
    그러면 T4의 개인차 스윕이 "개인차만 바꾼" 비교가 아니게 되고,
    τ²=0 팔의 수치는 다른 조건과 비교할 근거를 잃는다.
    """
    rng_zero = set_all_seeds(0)
    make_subjects(5, 0.0, rng_zero)
    after_zero = rng_zero.normal(size=4)

    rng_nonzero = set_all_seeds(0)
    make_subjects(5, 0.5, rng_nonzero)
    after_nonzero = rng_nonzero.normal(size=4)

    assert np.allclose(after_zero, after_nonzero)


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
