import numpy as np
import pytest
from scipy import signal as sp_signal

from src.common.seeding import set_all_seeds
from src.simulation.components.base import N_MODULATED_EEG, load_fraction, pink_noise
from src.simulation.components.eeg_oscillation import generate_eeg_oscillation
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


def _make(effect_size, seed=0, n_channels=30):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    sig = generate_eeg_oscillation(
        tl, sub, rng, sfreq=SFREQ, n_channels=n_channels, effect_size=effect_size
    )
    return tl, sig


def _band_power(x, sfreq, lo, hi):
    f, pxx = sp_signal.welch(x, fs=sfreq, nperseg=min(512, len(x)))
    return pxx[(f >= lo) & (f < hi)].sum()


def test_output_shape():
    _, sig = _make(0.8)
    # 세션 총 길이 = 시작·종료 베이스라인(20초*2) + 과제 블록(3*2*30=180초)
    total_duration_s = 2 * TASK_CFG["baseline_duration_s"] + 180.0
    expected_samples = int(round(total_duration_s * SFREQ))
    assert sig.shape == (30, expected_samples)


def test_no_nans():
    _, sig = _make(0.8)
    assert np.isfinite(sig).all()


def test_theta_power_increases_with_load():
    tl, sig = _make(0.8)
    ch = sig[0]
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    theta_low = _band_power(ch[low_mask], SFREQ, 4, 8)
    theta_high = _band_power(ch[high_mask], SFREQ, 4, 8)
    assert theta_high > theta_low


def test_alpha_power_decreases_with_load():
    tl, sig = _make(0.8)
    ch = sig[0]
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    alpha_low = _band_power(ch[low_mask], SFREQ, 8, 13)
    alpha_high = _band_power(ch[high_mask], SFREQ, 8, 13)
    assert alpha_high < alpha_low


def test_zero_effect_size_removes_load_modulation():
    tl, sig = _make(0.0)
    ch = sig[0]
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    theta_low = _band_power(ch[low_mask], SFREQ, 4, 8)
    theta_high = _band_power(ch[high_mask], SFREQ, 4, 8)
    assert theta_high == pytest.approx(theta_low, rel=0.35)


def test_unmodulated_channels_have_no_load_effect():
    tl, sig = _make(0.8)
    ch = sig[N_MODULATED_EEG]  # 변조 대상 밖 첫 채널
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    theta_low = _band_power(ch[low_mask], SFREQ, 4, 8)
    theta_high = _band_power(ch[high_mask], SFREQ, 4, 8)
    assert theta_high == pytest.approx(theta_low, rel=0.35)


def test_reproducible_for_same_seed():
    _, a = _make(0.8, seed=7)
    _, b = _make(0.8, seed=7)
    assert np.array_equal(a, b)


# --- base.py 직접 테스트 (controller ruling: 후속 태스크 전체가 의존하므로 직접 검증) ---


def test_load_fraction_maps_index_to_ratio():
    result = load_fraction(np.array([0, 1, 2]), n_levels=3)
    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_load_fraction_degenerate_single_level_no_div_by_zero():
    result = load_fraction(np.array([0, 0, 0]), n_levels=1)
    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_pink_noise_shape_and_finite():
    rng = set_all_seeds(0)
    noise = pink_noise(2000, 4, rng)
    assert noise.shape == (4, 2000)
    assert np.isfinite(noise).all()


def test_pink_noise_spectrum_falls_with_frequency():
    rng = set_all_seeds(0)
    sfreq = 250.0
    noise = pink_noise(5000, 1, rng)[0]
    f, pxx = sp_signal.welch(noise, fs=sfreq, nperseg=512)
    low_power = pxx[(f >= 1) & (f < 5)].sum()
    high_power = pxx[(f >= 40) & (f < 100)].sum()
    assert low_power > high_power
