import numpy as np
import pytest

from src.simulation.session import plan_sessions
from src.simulation.subject import SubjectProfile

SUBJ = SubjectProfile(subject_id="sub-01", theta=0.0)


def make_cfg(**drift_over):
    drift = {
        "fnirs_gain_sigma": 0.20,
        "fnirs_offset_sigma": 0.10,
        "eeg_gain_sigma": 0.15,
        "eeg_noise_sigma": 0.15,
        "within_session_rate": 0.30,
        "within_session_fraction": 0.33,
        "between_session_scale": 4.0,
        "assignment": "sampled",
    }
    drift.update(drift_over)
    return {
        "n_sessions": 3,
        "practice": {"rate": 0.15},
        "drift": drift,
        "eeg": {"n_channels": 8, "sfreq_hz": 250},
        "fnirs": {"n_channels": 6, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
    }


def test_returns_one_plan_per_session_in_order():
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert [p.session_idx for p in plans] == [0, 1, 2]


def test_practice_gain_decays_geometrically_and_stays_positive():
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert plans[0].practice_gain == pytest.approx(1.0)
    assert plans[1].practice_gain == pytest.approx(0.85)
    assert plans[2].practice_gain == pytest.approx(0.85 ** 2)


def test_practice_gain_never_goes_negative_for_many_sessions():
    """선형 감소였다면 세션 7에서 음수가 되어 인지부하가 뒤집힌다."""
    cfg = make_cfg()
    cfg["n_sessions"] = 20
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(0))
    assert all(p.practice_gain > 0 for p in plans)


def test_session_zero_also_carries_drift():
    """'무드리프트 기준 세션'을 두면 정규화 테스트가 부당하게 쉬워진다."""
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert not np.allclose(plans[0].drift.fnirs_gain, 1.0)


def test_sessions_differ_from_each_other():
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert not np.allclose(plans[0].drift.fnirs_gain, plans[1].drift.fnirs_gain)


def test_zero_sigma_consumes_the_same_rng_state():
    """분기하면 '드리프트 없음' 조건이 다른 난수열 위에서 돈다.

    그러면 T2의 결함 주입('드리프트를 끄면 붕괴가 사라진다')이 단일 변수
    조작이 아니게 되고, T4의 b_ref도 b_hat의 정당한 기준이 되지 못한다.
    """
    off = dict(
        fnirs_gain_sigma=0.0, fnirs_offset_sigma=0.0,
        eeg_gain_sigma=0.0, eeg_noise_sigma=0.0,
        within_session_rate=0.0,
    )
    rng_a = np.random.default_rng(3)
    plan_sessions(SUBJ, make_cfg(), rng_a)
    rng_b = np.random.default_rng(3)
    plan_sessions(SUBJ, make_cfg(**off), rng_b)
    assert rng_a.random() == rng_b.random()


def test_zero_sigma_really_removes_drift():
    off = dict(
        fnirs_gain_sigma=0.0, fnirs_offset_sigma=0.0,
        eeg_gain_sigma=0.0, eeg_noise_sigma=0.0,
        within_session_rate=0.0,
    )
    plans = plan_sessions(SUBJ, make_cfg(**off), np.random.default_rng(3))
    for p in plans:
        assert np.allclose(p.drift.fnirs_gain, 1.0)
        assert np.allclose(p.drift.fnirs_offset, 0.0)
        assert np.allclose(p.drift.eeg_gain, 1.0)
        assert p.drift.within_rate == 0.0


def test_fixed_2x2_assignment_covers_all_four_cells():
    cfg = make_cfg(assignment="fixed_2x2")
    cfg["n_sessions"] = 4
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(0))
    cells = {(p.drift.between_big, p.drift.within_big) for p in plans}
    assert cells == {(False, False), (False, True), (True, False), (True, True)}


def test_fixed_2x2_session_two_is_the_critical_negative_cell():
    """①② 큼 + ③ 작음 — 여기가 플래그되면 drift_flag가 둘을 혼동한 것이다."""
    cfg = make_cfg(assignment="fixed_2x2")
    cfg["n_sessions"] = 4
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(0))
    assert plans[2].drift.between_big is True
    assert plans[2].drift.within_big is False
    assert plans[2].drift.within_rate == 0.0


def test_between_big_actually_widens_the_gain_spread():
    cfg = make_cfg(assignment="fixed_2x2")
    cfg["n_sessions"] = 4
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(5))
    small = np.abs(np.log(plans[0].drift.fnirs_gain)).mean()
    big = np.abs(np.log(plans[2].drift.fnirs_gain)).mean()
    assert big > small * 2


def test_unknown_assignment_is_rejected():
    with pytest.raises(ValueError, match="unknown drift assignment"):
        plan_sessions(SUBJ, make_cfg(assignment="whatever"), np.random.default_rng(0))
