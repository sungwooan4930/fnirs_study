import numpy as np

from src.common.seeding import set_all_seeds
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.simulation.recording import generate_recording
from src.simulation.session import SessionDriftParams, SessionPlan
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 1,
    "subject_variance": 0.0,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "baseline_duration_s": 20,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}
RT_BINS = [0.5, 0.8]


def _neutral_plan(subject, sim_cfg):
    """드리프트 없는 SessionPlan — 이 파일의 라벨 테스트는 신호 왜곡과 무관하다."""
    n_eeg = int(sim_cfg["eeg"]["n_channels"])
    n_fnirs = int(sim_cfg["fnirs"]["n_channels"])
    drift = SessionDriftParams(
        fnirs_gain=np.ones(n_fnirs),
        fnirs_offset=np.zeros(n_fnirs),
        eeg_gain=np.ones(n_eeg),
        eeg_noise_scale=np.zeros(n_eeg),
        within_rate=0.0,
        between_big=False,
        within_big=False,
    )
    return SessionPlan(subject=subject, session_idx=0, practice_gain=1.0, drift=drift)


def _setup(lead_delta_s=1.2):
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.0, rng)[0]
    plan = _neutral_plan(sub, SIM_CFG)
    rec = generate_recording(plan, SIM_CFG, rng)
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


def test_lead_target_selects_correct_stimulus():
    """선행 타깃이 정확히 올바른 자극을 선택하는지 검증."""
    rec, win, labels, keep = _setup(lead_delta_s=1.2)
    onsets = rec.behavior.onsets

    # 유효한 창 샘플에서 직접 선행 자극 인덱스 계산
    for i in np.flatnonzero(keep)[:50]:
        lead_time = win.end_s[i] + 1.2
        expected_idx = np.searchsorted(onsets, lead_time, side="left")

        # 정확도와 반응시간이 해당 자극의 값과 일치해야 함
        assert labels["accuracy"][i] == rec.behavior.correct[expected_idx]
        expected_latency = np.digitize(rec.behavior.rt[expected_idx], bins=RT_BINS)
        assert labels["response_latency"][i] == expected_latency


def test_keep_mask_exact_boundaries():
    """keep 마스크가 정확히 선행 자극이 존재하는 창만 True인지 검증."""
    rec, win, _, keep = _setup(lead_delta_s=1.2)
    onsets = rec.behavior.onsets

    # 독립적으로 계산한 유효 마스크
    expected_keep = np.searchsorted(onsets, win.end_s + 1.2, side="left") < len(onsets)

    assert np.array_equal(keep, expected_keep)


def test_negative_lead_delta_raises_error():
    """lead_delta_s < 0이면 ValueError를 발생시킨다."""
    rec, win, _, _ = _setup()

    try:
        build_labels(rec, win, lead_delta_s=-0.5, rt_bins=RT_BINS)
        assert False, "ValueError should have been raised"
    except ValueError as e:
        assert "lead_delta_s must be >= 0" in str(e)


def test_response_latency_bin_mapping():
    """rt_bins=[0.5, 0.8]에서 특정 RT가 올바른 bin으로 매핑되는지 확인."""
    # np.digitize with bins=[0.5, 0.8] gives:
    # x < 0.5 -> 0
    # 0.5 <= x < 0.8 -> 1
    # x >= 0.8 -> 2
    assert np.digitize(0.4, bins=RT_BINS) == 0
    assert np.digitize(0.65, bins=RT_BINS) == 1
    assert np.digitize(0.9, bins=RT_BINS) == 2
