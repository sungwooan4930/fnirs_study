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
from src.simulation.components.eeg_erp import generate_eeg_erp
from src.simulation.components.eeg_oscillation import generate_eeg_oscillation
from src.simulation.components.fnirs_hrf import generate_fnirs
from src.simulation.state import CognitiveStateTimeline, build_timeline
from src.simulation.subject import SubjectProfile, make_subjects

BIDS_VERSION = "1.9.0"


@dataclass(frozen=True)
class SyntheticRecording:
    subject_id: str
    eeg: np.ndarray          # (n_eeg_ch, n_eeg_samples)
    eeg_sfreq: float
    hbo: np.ndarray          # (n_fnirs_ch, n_fnirs_samples)
    hbr: np.ndarray
    fnirs_sfreq: float
    behavior: BehaviorLog
    timeline: CognitiveStateTimeline


def generate_recording(
    subject: SubjectProfile,
    sim_cfg: dict,
    rng: np.random.Generator,
) -> SyntheticRecording:
    """한 피험자의 녹화를 만든다."""
    effect_size = float(sim_cfg["effect_size"])
    eeg_cfg = sim_cfg["eeg"]
    fnirs_cfg = sim_cfg["fnirs"]

    timeline = build_timeline(sim_cfg["task"], rng)

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

    hbo, hbr = generate_fnirs(
        timeline, subject, rng,
        sfreq=float(fnirs_cfg["sfreq_hz"]),
        n_channels=int(fnirs_cfg["n_channels"]),
        effect_size=effect_size,
        hbr_coupling=float(fnirs_cfg["hbr_coupling"]),
    )

    behavior = generate_behavior(
        timeline, subject, rng,
        effect_size=effect_size,
        lead_delta_s=float(sim_cfg["lead_delta_s"]),
    )

    return SyntheticRecording(
        subject_id=subject.subject_id,
        eeg=eeg,
        eeg_sfreq=float(eeg_cfg["sfreq_hz"]),
        hbo=hbo,
        hbr=hbr,
        fnirs_sfreq=float(fnirs_cfg["sfreq_hz"]),
        behavior=behavior,
        timeline=timeline,
    )


def generate_dataset(sim_cfg: dict, rng: np.random.Generator) -> list[SyntheticRecording]:
    """설정된 수만큼 피험자 녹화를 만든다."""
    subjects = make_subjects(
        int(sim_cfg["n_subjects"]),
        float(sim_cfg["subject_variance"]),
        rng,
    )
    return [generate_recording(s, sim_cfg, rng) for s in subjects]


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

    lines = ["participant_id\tn_eeg_channels\tn_fnirs_channels\tduration_s"]
    for rec in recordings:
        lines.append(
            f"{rec.subject_id}\t{rec.eeg.shape[0]}\t{rec.hbo.shape[0]}"
            f"\t{rec.timeline.duration_s:.1f}"
        )
    (root / "participants.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
