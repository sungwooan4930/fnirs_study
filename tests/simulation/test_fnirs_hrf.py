import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.components.fnirs_hrf import canonical_hrf, generate_fnirs
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
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
    # 세션 총 길이 = 시작·종료 베이스라인(20초*2) + 과제 블록(3*2*30=180초)
    total_duration_s = 2 * TASK_CFG["baseline_duration_s"] + 180.0
    assert hbo.shape == hbr.shape == (48, int(round(total_duration_s * SFREQ)))


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


from src.simulation.components.fnirs_hrf import generate_fnirs
from src.simulation.state import TASK, build_timeline
from src.simulation.subject import SubjectProfile

_TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _fnirs_task_mean(practice_gain: float) -> float:
    """같은 시드로 fNIRS를 만들고 과제 구간 HbO 평균을 돌려준다."""
    tl = build_timeline(
        _TASK_CFG, np.random.default_rng(7), practice_gain=practice_gain
    )
    hbo, _ = generate_fnirs(
        tl,
        SubjectProfile(subject_id="sub-01", theta=0.0),
        np.random.default_rng(11),
        sfreq=10.4,
        n_channels=4,
        effect_size=1.0,
        hbr_coupling=-0.33,
    )
    t = np.arange(hbo.shape[1]) / 10.4
    task_mask = tl.kind_at(t) == TASK
    return float(hbo[:, task_mask].mean())


def test_practice_gain_reaches_the_fnirs_signal():
    """연습 효과가 신호에 도달하지 않으면 T4는 원인 불명으로 실패한다."""
    full = _fnirs_task_mean(1.0)
    halved = _fnirs_task_mean(0.5)
    assert full > 0.05, "기준 조건에서 과제 구간 HbO가 양수여야 비교가 성립한다"
    # 잡음이 동일 시드로 같으므로 응답 성분만 절반이 된다
    assert halved < full * 0.75
