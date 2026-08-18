import numpy as np

from src.common.seeding import set_all_seeds
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.simulation.recording import generate_recording
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 1,
    "subject_variance": 0.0,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}
RT_BINS = [0.5, 0.8]


def _setup(lead_delta_s=1.2):
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.0, rng)[0]
    rec = generate_recording(sub, SIM_CFG, rng)
    win = make_windows(rec.timeline, 5.0, 1.0)
    labels, keep = build_labels(rec, win, lead_delta_s=lead_delta_s, rt_bins=RT_BINS)
    return rec, win, labels, keep


def test_returns_expected_targets():
    _, _, labels, _ = _setup()
    assert set(labels) == {"cognitive_load", "accuracy", "response_latency"}


def test_all_arrays_same_length_as_windows():
    _, win, labels, keep = _setup()
    n = len(win.start_s)
    assert all(len(v) == n for v in labels.values())
    assert len(keep) == n


def test_cognitive_load_matches_window_label():
    _, win, labels, _ = _setup()
    assert np.array_equal(labels["cognitive_load"], win.load_level)


def test_cognitive_load_has_three_levels():
    _, _, labels, keep = _setup()
    assert set(np.unique(labels["cognitive_load"][keep])) == {0, 1, 2}


def test_accuracy_is_binary():
    _, _, labels, keep = _setup()
    assert set(np.unique(labels["accuracy"][keep])).issubset({0, 1})


def test_response_latency_uses_fixed_bins():
    _, _, labels, keep = _setup()
    assert set(np.unique(labels["response_latency"][keep])).issubset({0, 1, 2})


def test_keep_mask_drops_windows_without_a_lead_stimulus():
    # Δ가 녹화 길이만큼 크면 선행 자극이 하나도 없다
    _, _, _, keep = _setup(lead_delta_s=1000.0)
    assert not keep.any()


def test_most_windows_are_kept_with_normal_delta():
    _, _, _, keep = _setup(lead_delta_s=1.2)
    assert keep.mean() > 0.9


def test_lead_target_comes_after_window_end():
    rec, win, labels, keep = _setup(lead_delta_s=1.2)
    # 선행 자극 시각이 창 끝보다 뒤인지 직접 확인
    for i in np.flatnonzero(keep)[:50]:
        later = rec.behavior.onsets[rec.behavior.onsets >= win.end_s[i] + 1.2]
        assert len(later) > 0
        assert later[0] > win.end_s[i]
