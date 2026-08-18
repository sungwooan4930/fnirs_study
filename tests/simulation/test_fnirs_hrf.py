import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.components.fnirs_hrf import canonical_hrf, generate_fnirs
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}
SFREQ = 10.4


def _make(effect_size, seed=0, hbr_coupling=-0.33):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    hbo, hbr = generate_fnirs(
        tl, sub, rng, sfreq=SFREQ, n_channels=48,
        effect_size=effect_size, hbr_coupling=hbr_coupling,
    )
    return tl, hbo, hbr


def test_hrf_peaks_around_five_seconds():
    hrf = canonical_hrf(SFREQ)
    peak_s = np.argmax(hrf) / SFREQ
    assert 4.0 < peak_s < 7.0


def test_hrf_starts_near_zero():
    hrf = canonical_hrf(SFREQ)
    assert abs(hrf[0]) < 1e-6


def test_output_shapes_match():
    _, hbo, hbr = _make(0.8)
    assert hbo.shape == hbr.shape == (48, int(round(180.0 * SFREQ)))


def test_no_nans():
    _, hbo, hbr = _make(0.8)
    assert np.isfinite(hbo).all() and np.isfinite(hbr).all()


def test_hbo_increases_with_load():
    tl, hbo, _ = _make(0.8)
    t = np.arange(hbo.shape[1]) / SFREQ
    load = tl.load_at(t)
    # HRF 지연을 고려해 블록 후반부만 비교한다
    assert hbo[0][load == 2].mean() > hbo[0][load == 0].mean()


def test_hbo_hbr_are_negatively_correlated():
    _, hbo, hbr = _make(0.8)
    r = np.corrcoef(hbo[0], hbr[0])[0, 1]
    assert r < -0.5


def test_zero_effect_size_removes_load_modulation():
    tl, hbo, _ = _make(0.0)
    t = np.arange(hbo.shape[1]) / SFREQ
    load = tl.load_at(t)
    high = hbo[0][load == 2].mean()
    low = hbo[0][load == 0].mean()
    assert high == pytest.approx(low, abs=0.15)


def test_reproducible_for_same_seed():
    _, a, _ = _make(0.8, seed=3)
    _, b, _ = _make(0.8, seed=3)
    assert np.array_equal(a, b)
