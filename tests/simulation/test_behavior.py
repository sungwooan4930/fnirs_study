import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.components.behavior import generate_behavior
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _make(effect_size, lead_delta_s=1.2, seed=0):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    return tl, generate_behavior(tl, sub, rng, effect_size=effect_size, lead_delta_s=lead_delta_s)


def test_one_entry_per_stimulus():
    tl, log = _make(0.8)
    assert len(log.onsets) == len(tl.stim_onsets)
    assert len(log.correct) == len(log.rt) == len(log.onsets)


def test_correct_is_binary():
    _, log = _make(0.8)
    assert set(np.unique(log.correct)).issubset({0, 1})


def test_rt_is_positive():
    _, log = _make(0.8)
    assert (log.rt > 0).all()


def test_accuracy_drops_with_load_measured_at_lead_time():
    tl, log = _make(0.8, lead_delta_s=1.2)
    driving_load = tl.load_at(log.onsets - 1.2)
    acc_low = log.correct[driving_load == 0].mean()
    acc_high = log.correct[driving_load == 2].mean()
    assert acc_high < acc_low


def test_rt_rises_with_load_measured_at_lead_time():
    tl, log = _make(0.8, lead_delta_s=1.2)
    driving_load = tl.load_at(log.onsets - 1.2)
    assert log.rt[driving_load == 2].mean() > log.rt[driving_load == 0].mean()


def test_lead_delta_shifts_which_state_drives_behavior():
    # Δ가 크게 다르면 같은 시드에서도 행동 계열이 달라져야 한다
    _, a = _make(0.8, lead_delta_s=0.0)
    _, b = _make(0.8, lead_delta_s=10.0)
    assert not np.array_equal(a.rt, b.rt)


def test_zero_effect_size_removes_load_dependence():
    tl, log = _make(0.0)
    driving_load = tl.load_at(log.onsets - 1.2)
    lo = log.rt[driving_load == 0].mean()
    hi = log.rt[driving_load == 2].mean()
    assert hi == pytest.approx(lo, abs=0.05)


def test_rejects_negative_lead_delta():
    with pytest.raises(ValueError):
        _make(0.8, lead_delta_s=-1.0)
