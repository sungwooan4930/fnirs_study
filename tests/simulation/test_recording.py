import copy
import json

import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.recording import generate_dataset, generate_recording, write_bids_metadata
from src.simulation.session import SessionDriftParams, SessionPlan, plan_sessions
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 3,
    "n_sessions": 1,
    "subject_variance": 0.5,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "baseline_duration_s": 20,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "practice": {"rate": 0.15},
    "drift": {
        "fnirs_gain_sigma": 0.20,
        "fnirs_offset_sigma": 0.10,
        "eeg_gain_sigma": 0.15,
        "eeg_noise_sigma": 0.15,
        "within_session_rate": 0.30,
        "within_session_fraction": 0.33,
        "between_session_scale": 4.0,
        "assignment": "sampled",
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def _neutral_plan(subject, sim_cfg, *, session_idx=0, practice_gain=1.0):
    """드리프트·잡음이 전혀 없는 SessionPlan.

    신호 성분(오실레이션·ERP)만 정확히 재현해야 하는 테스트용 — plan_sessions가
    뽑는 실제 드리프트(특히 EEG_NOISE_BASE 기반 잡음 바닥)는 sigma=0이어도
    완전히 0이 되지 않으므로, 정확한 잔차 비교에는 수동으로 중립 plan을 만든다.
    """
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
    return SessionPlan(
        subject=subject,
        session_idx=session_idx,
        practice_gain=practice_gain,
        drift=drift,
    )


def test_recording_shapes():
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.5, rng)[0]
    plan = plan_sessions(sub, SIM_CFG, rng)[0]
    rec = generate_recording(plan, SIM_CFG, rng)
    assert rec.eeg.shape[0] == 30
    assert rec.hbo.shape[0] == 48
    assert rec.hbr.shape == rec.hbo.shape
    assert rec.eeg.shape[1] == int(round(rec.timeline.duration_s * 250))
    assert rec.hbo.shape[1] == int(round(rec.timeline.duration_s * 10.4))


def test_recording_has_no_nans():
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.5, rng)[0]
    plan = plan_sessions(sub, SIM_CFG, rng)[0]
    rec = generate_recording(plan, SIM_CFG, rng)
    assert np.isfinite(rec.eeg).all()
    assert np.isfinite(rec.hbo).all()
    assert np.isfinite(rec.hbr).all()


def test_dataset_has_requested_subject_count():
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    assert len(recs) == 3
    assert [r.subject_id for r in recs] == ["sub-01", "sub-02", "sub-03"]


def test_subjects_differ_from_each_other():
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    assert not np.array_equal(recs[0].eeg, recs[1].eeg)


def test_dataset_is_reproducible():
    a = generate_dataset(SIM_CFG, set_all_seeds(11))
    b = generate_dataset(SIM_CFG, set_all_seeds(11))
    assert np.array_equal(a[0].eeg, b[0].eeg)
    assert np.array_equal(a[2].hbo, b[2].hbo)


def test_bids_metadata_written(tmp_path):
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    write_bids_metadata(recs, tmp_path)

    desc = json.loads((tmp_path / "dataset_description.json").read_text(encoding="utf-8"))
    assert desc["BIDSVersion"]
    assert "synthetic" in desc["Name"].lower()

    lines = (tmp_path / "participants.tsv").read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].split("\t")[0] == "participant_id"
    assert len(lines) == 4  # 헤더 + 피험자 3명


def test_subjects_have_different_timelines():
    """Verify each subject gets its own counterbalanced block order."""
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    load_0 = recs[0].timeline.load_level
    load_1 = recs[1].timeline.load_level
    # The two subjects' load sequences should differ (different block order)
    assert not np.array_equal(load_0, load_1), (
        "Subjects must have different block orders (different timelines). "
        "If all subjects share the same timeline, counterbalancing is lost."
    )


def test_eeg_contains_erp_component():
    """Verify ERP component via exact residual extraction and unmodulated channel check.

    The ERP is recoverable by replaying build_timeline + generate_eeg_oscillation
    with an identically seeded generator, then subtracting from rec.eeg. This
    yields the pure ERP signal, uncontaminated by background oscillations.

    Assertions:
    1. Unmodulated channels (indices ≥10) have zero residual (ERP only on ch 0-9)
    2. Modulated channels have non-zero ERP (fails if ERP line is dropped)
    """
    from src.simulation.components.eeg_oscillation import generate_eeg_oscillation
    from src.simulation.state import build_timeline

    # Generate full recording (full RNG consumption path)
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.0, rng)[0]
    plan = _neutral_plan(sub, SIM_CFG)
    rec = generate_recording(plan, SIM_CFG, rng)

    # Replay only first two components (timeline + oscillation) with fresh seeded RNG
    rng2 = set_all_seeds(0)
    sub2 = make_subjects(1, 0.0, rng2)[0]
    tl2 = build_timeline(SIM_CFG["task"], rng2)
    osc = generate_eeg_oscillation(
        tl2, sub2, rng2,
        sfreq=float(SIM_CFG["eeg"]["sfreq_hz"]),
        n_channels=int(SIM_CFG["eeg"]["n_channels"]),
        effect_size=float(SIM_CFG["effect_size"]),
    )

    # ERP residual = full recording minus oscillation-only
    residual = rec.eeg - osc

    # Unmodulated channels (≥10) should have zero residual
    # (N_MODULATED_EEG=10, so only ch 0-9 receive ERP)
    unmod_residual_max = np.max(np.abs(residual[10:, :]))
    assert unmod_residual_max < 1e-8, (
        f"Unmodulated channel residual should be zero, got max={unmod_residual_max:.2e}. "
        f"Replay is out of step with generate_recording."
    )

    # Modulated channels (0-9) should have non-zero ERP
    # P300_AMPLITUDE=1.0, so expect peaks ~0.3-1.0 µV after modulation and load variation
    mod_residual_max = np.max(np.abs(residual[:10, :]))
    assert mod_residual_max > 0.1, (
        f"Modulated channels ERP residual too small ({mod_residual_max:.4f}µV). "
        f"If generate_eeg_erp term was dropped from the EEG sum, this fails."
    )


# 이름을 모듈 상단의 SIM_CFG와 분리한다 — 재사용하면 모듈 로드 시점에
# 상단 SIM_CFG를 덮어써서 앞선 테스트들(30ch/48ch 기대)이 깨진다.
SIM_CFG_MULTI_SESSION = {
    "n_subjects": 3,
    "n_sessions": 2,
    "subject_variance": 0.5,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "baseline_duration_s": 20,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "practice": {"rate": 0.15},
    "drift": {
        "fnirs_gain_sigma": 0.20,
        "fnirs_offset_sigma": 0.10,
        "eeg_gain_sigma": 0.15,
        "eeg_noise_sigma": 0.15,
        "within_session_rate": 0.30,
        "within_session_fraction": 0.33,
        "between_session_scale": 4.0,
        "assignment": "sampled",
    },
    "eeg": {"n_channels": 8, "sfreq_hz": 250},
    "fnirs": {"n_channels": 6, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def _no_drift_cfg():
    cfg = copy.deepcopy(SIM_CFG_MULTI_SESSION)
    cfg["drift"].update(
        fnirs_gain_sigma=0.0, fnirs_offset_sigma=0.0,
        eeg_gain_sigma=0.0, eeg_noise_sigma=0.0, within_session_rate=0.0,
    )
    return cfg


def test_dataset_has_one_recording_per_subject_session():
    recs = generate_dataset(SIM_CFG_MULTI_SESSION, np.random.default_rng(0))
    assert len(recs) == SIM_CFG_MULTI_SESSION["n_subjects"] * SIM_CFG_MULTI_SESSION["n_sessions"]
    pairs = [(r.subject_id, r.session_idx) for r in recs]
    assert len(set(pairs)) == len(pairs)
    assert set(r.session_idx for r in recs) == {0, 1}


def test_same_subject_different_sessions_have_different_channel_levels():
    """세션 간 베이스라인 이동이 실제로 생겨야 정규화를 시험할 수 있다."""
    recs = generate_dataset(SIM_CFG_MULTI_SESSION, np.random.default_rng(0))
    s0 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 0)
    s1 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 1)
    assert not np.allclose(s0.hbo.mean(axis=1), s1.hbo.mean(axis=1), atol=1e-6)


def test_without_drift_channel_levels_stay_comparable():
    recs = generate_dataset(_no_drift_cfg(), np.random.default_rng(0))
    s0 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 0)
    s1 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 1)
    spread_no_drift = float(np.abs(s0.hbo.mean(axis=1) - s1.hbo.mean(axis=1)).mean())

    recs_d = generate_dataset(SIM_CFG_MULTI_SESSION, np.random.default_rng(0))
    d0 = next(r for r in recs_d if r.subject_id == "sub-01" and r.session_idx == 0)
    d1 = next(r for r in recs_d if r.subject_id == "sub-01" and r.session_idx == 1)
    spread_with_drift = float(np.abs(d0.hbo.mean(axis=1) - d1.hbo.mean(axis=1)).mean())

    assert spread_with_drift > spread_no_drift * 3


def test_behavior_is_untouched_by_drift():
    """행동은 장비 재부착의 영향을 받지 않는다 (CLAUDE.md §3.8)."""
    a = generate_dataset(SIM_CFG_MULTI_SESSION, np.random.default_rng(0))[0]
    b = generate_dataset(_no_drift_cfg(), np.random.default_rng(0))[0]
    assert np.array_equal(a.behavior.correct, b.behavior.correct)
    assert np.allclose(a.behavior.rt, b.behavior.rt)


def test_recording_timeline_knows_its_session():
    recs = generate_dataset(SIM_CFG_MULTI_SESSION, np.random.default_rng(0))
    for r in recs:
        assert r.timeline.session_idx == r.session_idx
