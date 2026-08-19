"""합성 녹화 조립.

신호 성분들을 가산 결합해 한 피험자의 EEG·fNIRS·행동을 만들고,
피험자 목록 전체로 데이터셋을 구성한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.simulation.components.behavior import BehaviorLog, generate_behavior
from src.simulation.components.drift import apply_channel_drift, apply_noise_scaling
from src.simulation.components.eeg_erp import generate_eeg_erp
from src.simulation.components.eeg_oscillation import generate_eeg_oscillation
from src.simulation.components.fnirs_hrf import generate_fnirs
from src.simulation.session import SessionPlan, plan_sessions
from src.simulation.state import CognitiveStateTimeline, build_timeline
from src.simulation.subject import make_subjects

BIDS_VERSION = "1.9.0"


@dataclass(frozen=True)
class SyntheticRecording:
    subject_id: str
    session_idx: int
    eeg: np.ndarray          # (n_eeg_ch, n_eeg_samples)
    eeg_sfreq: float
    hbo: np.ndarray          # (n_fnirs_ch, n_fnirs_samples)
    hbr: np.ndarray
    fnirs_sfreq: float
    behavior: BehaviorLog
    timeline: CognitiveStateTimeline


def generate_recording(
    plan: SessionPlan,
    sim_cfg: dict,
    rng: np.random.Generator,
) -> SyntheticRecording:
    """한 피험자의 한 세션 녹화를 만든다.

    성분들이 깨끗한 신호를 만들고, 그 뒤에 세션 드리프트를 얹는다.
    행동에는 드리프트를 적용하지 않는다 — 정오답률·반응시간은 장비
    재부착의 영향을 받지 않으므로 절대값을 유지한다 (CLAUDE.md §3.8).
    """
    effect_size = float(sim_cfg["effect_size"])
    eeg_cfg = sim_cfg["eeg"]
    fnirs_cfg = sim_cfg["fnirs"]
    subject = plan.subject
    drift = plan.drift

    timeline = build_timeline(
        sim_cfg["task"], rng,
        session_idx=plan.session_idx,
        practice_gain=plan.practice_gain,
    )

    eeg = generate_eeg_oscillation(
        timeline, subject, rng,
        sfreq=float(eeg_cfg["sfreq_hz"]),
        n_channels=int(eeg_cfg["n_channels"]),
        effect_size=effect_size,
    )
    eeg = eeg + generate_eeg_erp(
        timeline, subject, rng,
        sfreq=float(eeg_cfg["sfreq_hz"]),
        n_channels=int(eeg_cfg["n_channels"]),
        effect_size=effect_size,
    )
    # ② 임피던스: 진폭이 아니라 잡음이 커진다. 이득보다 먼저 얹어야
    #    세션 내 드리프트 램프가 잡음에도 함께 걸린다.
    eeg = apply_noise_scaling(eeg, noise_scale=drift.eeg_noise_scale, rng=rng)
    eeg = apply_channel_drift(
        eeg,
        gain=drift.eeg_gain,
        offset=np.zeros(int(eeg_cfg["n_channels"])),
        within_rate=drift.within_rate,
    )

    hbo, hbr = generate_fnirs(
        timeline, subject, rng,
        sfreq=float(fnirs_cfg["sfreq_hz"]),
        n_channels=int(fnirs_cfg["n_channels"]),
        effect_size=effect_size,
        hbr_coupling=float(fnirs_cfg["hbr_coupling"]),
    )
    # ① 옵토드 재부착: HbO·HbR은 같은 광경로에서 나오므로 동일한 이득·
    #    오프셋을 받는다. 다른 값을 주면 실제와 다르고 정규화가 더 쉬워진다.
    hbo = apply_channel_drift(
        hbo, gain=drift.fnirs_gain, offset=drift.fnirs_offset,
        within_rate=drift.within_rate,
    )
    hbr = apply_channel_drift(
        hbr, gain=drift.fnirs_gain, offset=drift.fnirs_offset,
        within_rate=drift.within_rate,
    )

    behavior = generate_behavior(
        timeline, subject, rng,
        effect_size=effect_size,
        lead_delta_s=float(sim_cfg["lead_delta_s"]),
    )

    return SyntheticRecording(
        subject_id=subject.subject_id,
        session_idx=plan.session_idx,
        eeg=eeg,
        eeg_sfreq=float(eeg_cfg["sfreq_hz"]),
        hbo=hbo,
        hbr=hbr,
        fnirs_sfreq=float(fnirs_cfg["sfreq_hz"]),
        behavior=behavior,
        timeline=timeline,
    )


def generate_dataset(sim_cfg: dict, rng: np.random.Generator) -> list[SyntheticRecording]:
    """설정된 피험자 수 × 세션 수만큼 녹화를 만든다."""
    subjects = make_subjects(
        int(sim_cfg["n_subjects"]),
        float(sim_cfg["subject_variance"]),
        rng,
    )
    recordings: list[SyntheticRecording] = []
    for subject in subjects:
        for plan in plan_sessions(subject, sim_cfg, rng):
            recordings.append(generate_recording(plan, sim_cfg, rng))
    return recordings


def write_bids_metadata(recordings: list[SyntheticRecording], root: Path) -> None:
    """BIDS 메타데이터만 쓴다.

    실장비 모델과 fNIRS 몽타주가 미확정이므로(스펙 §11.3) SNIRF·EDF
    신호 파일은 쓰지 않는다. 장비 확정 후 확장한다.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    description = {
        "Name": "Synthetic multimodal cognitive state testbed",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "src.simulation.recording"}],
    }
    (root / "dataset_description.json").write_text(
        json.dumps(description, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["participant_id\tn_sessions\tn_eeg_channels\tn_fnirs_channels\tduration_s"]
    seen: dict[str, list] = {}
    for rec in recordings:
        seen.setdefault(rec.subject_id, []).append(rec)
    for subject_id, recs in seen.items():
        first = recs[0]
        lines.append(
            f"{subject_id}\t{len(recs)}\t{first.eeg.shape[0]}\t{first.hbo.shape[0]}"
            f"\t{first.timeline.duration_s:.1f}"
        )
    (root / "participants.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
