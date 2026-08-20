import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.components.base import N_MODULATED_EEG
from src.simulation.components.eeg_erp import P300_LATENCY_S, generate_eeg_erp
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}
SFREQ = 250.0


def _make(effect_size, seed=0):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    sig = generate_eeg_erp(tl, sub, rng, sfreq=SFREQ, n_channels=30, effect_size=effect_size)
    return tl, sig


def _mean_erp_at_p300(tl, sig, level):
    onsets = tl.stim_onsets[tl.load_at(tl.stim_onsets) == level]
    idx = np.round((onsets + P300_LATENCY_S) * SFREQ).astype(int)
    idx = idx[idx < sig.shape[1]]
    return sig[0, idx].mean()


def test_output_shape():
    _, sig = _make(0.8)
    # 세션 총 길이 = 시작·종료 베이스라인(20초*2) + 과제 블록(3*2*30=180초)
    total_duration_s = 2 * TASK_CFG["baseline_duration_s"] + 180.0
    assert sig.shape == (30, int(round(total_duration_s * SFREQ)))


def test_no_nans():
    _, sig = _make(0.8)
    assert np.isfinite(sig).all()


def test_signal_is_near_zero_between_stimuli():
    tl, sig = _make(0.8)
    # 자극 사이 간격이 2초, ERP는 1초 내에 끝나므로 1.8초 지점은 거의 0
    idx = np.round((tl.stim_onsets[:-1] + 1.8) * SFREQ).astype(int)
    assert np.abs(sig[0, idx]).max() < 0.05


def test_p300_is_positive_deflection():
    tl, sig = _make(0.8)
    assert _mean_erp_at_p300(tl, sig, 0) > 0


def test_p300_amplitude_decreases_with_load():
    tl, sig = _make(0.8)
    assert _mean_erp_at_p300(tl, sig, 2) < _mean_erp_at_p300(tl, sig, 0)


def test_zero_effect_size_removes_load_modulation():
    tl, sig = _make(0.0)
    low = _mean_erp_at_p300(tl, sig, 0)
    high = _mean_erp_at_p300(tl, sig, 2)
    assert abs(high - low) < 1e-9


def test_unmodulated_channels_are_silent():
    _, sig = _make(0.8)
    assert np.abs(sig[N_MODULATED_EEG:]).max() == 0.0
