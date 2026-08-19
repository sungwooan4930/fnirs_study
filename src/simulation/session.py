"""세션 계획 — 세션별 드리프트 파라미터와 연습 효과.

스펙 §5.1의 층 분리를 구현한다.

- ①옵토드 재부착 ②캡·임피던스 ③세션 내 드리프트 → **신호 변환** 파라미터
- ④연습 효과 → **인지상태** 파라미터(`practice_gain`)

여기서는 뽑기만 하고 적용은 `components/drift.py`와 `state.py`가 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.subject import SubjectProfile

#: EEG 잡음의 기준 진폭. eeg_oscillation의 BASE_AMPLITUDE(1.0) 대비 비율이다.
#: 임피던스 변동은 이 값을 채널별로 곱셈 변조한다.
EEG_NOISE_BASE: float = 0.1

#: fixed_2x2 배정표 — (between_big, within_big). 세션 인덱스 % 4로 고른다.
#: 인덱스 2가 스펙 §8.3의 핵심 음성 칸(①② 큼 · ③ 작음)이다.
_FIXED_2X2: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


@dataclass(frozen=True)
class SessionDriftParams:
    fnirs_gain: np.ndarray       # (n_fnirs_ch,) 곱셈 이득
    fnirs_offset: np.ndarray     # (n_fnirs_ch,) 가산 오프셋
    eeg_gain: np.ndarray         # (n_eeg_ch,) 곱셈 이득
    eeg_noise_scale: np.ndarray  # (n_eeg_ch,) 잡음 표준편차
    within_rate: float           # 세션 내 이득 이동률
    between_big: bool
    within_big: bool


@dataclass(frozen=True)
class SessionPlan:
    subject: SubjectProfile
    session_idx: int
    practice_gain: float
    drift: SessionDriftParams


def plan_sessions(
    subject: SubjectProfile,
    sim_cfg: dict,
    rng: np.random.Generator,
) -> list[SessionPlan]:
    """한 피험자의 세션 계획 목록을 만든다.

    **난수 소모 순서는 조건과 무관하게 항상 같다** (스펙 §5.5). 크기가 0인
    조건에서도 같은 개수를 뽑고 뒤에서 스케일만 0으로 만든다. 분기를 두면
    '드리프트 없음' 조건이 다른 난수열 위에서 돌아, 블록 순서도 잡음도
    달라진다 — T2의 결함 주입과 T4의 기준 실행이 단일 변수 조작이 아니게 된다.
    """
    n_sessions = int(sim_cfg["n_sessions"])
    if n_sessions < 1:
        raise ValueError(f"n_sessions must be >= 1, got {n_sessions}")

    drift_cfg = sim_cfg["drift"]
    assignment = str(drift_cfg["assignment"])
    if assignment not in ("sampled", "fixed_2x2"):
        raise ValueError(
            f"unknown drift assignment '{assignment}'; "
            "expected 'sampled' or 'fixed_2x2'"
        )

    n_fnirs = int(sim_cfg["fnirs"]["n_channels"])
    n_eeg = int(sim_cfg["eeg"]["n_channels"])

    sigma_fg = float(drift_cfg["fnirs_gain_sigma"])
    sigma_fo = float(drift_cfg["fnirs_offset_sigma"])
    sigma_eg = float(drift_cfg["eeg_gain_sigma"])
    sigma_en = float(drift_cfg["eeg_noise_sigma"])
    within_rate_big = float(drift_cfg["within_session_rate"])
    within_fraction = float(drift_cfg["within_session_fraction"])
    between_scale_big = float(drift_cfg["between_session_scale"])

    practice_rate = float(sim_cfg["practice"]["rate"])
    if not 0.0 <= practice_rate < 1.0:
        raise ValueError(
            f"practice.rate must be in [0, 1), got {practice_rate}; "
            "기하 감쇠 (1-rate)**s 가 양수를 유지하려면 1 미만이어야 한다"
        )

    plans: list[SessionPlan] = []
    for session_idx in range(n_sessions):
        # --- 난수는 항상 이 순서로, 항상 이 개수만큼 뽑는다 ---
        u = float(rng.random())
        z_fg = rng.normal(size=n_fnirs)
        z_fo = rng.normal(size=n_fnirs)
        z_eg = rng.normal(size=n_eeg)
        z_en = rng.normal(size=n_eeg)

        if assignment == "sampled":
            between_big = False
            within_big = u < within_fraction
        else:
            between_big, within_big = _FIXED_2X2[session_idx % len(_FIXED_2X2)]

        between_scale = between_scale_big if between_big else 1.0
        within_rate = within_rate_big if within_big else 0.0

        drift = SessionDriftParams(
            fnirs_gain=np.exp(z_fg * sigma_fg * between_scale),
            fnirs_offset=z_fo * sigma_fo * between_scale,
            eeg_gain=np.exp(z_eg * sigma_eg * between_scale),
            eeg_noise_scale=EEG_NOISE_BASE * np.exp(z_en * sigma_en * between_scale),
            within_rate=within_rate,
            between_big=between_big,
            within_big=within_big,
        )

        plans.append(
            SessionPlan(
                subject=subject,
                session_idx=session_idx,
                practice_gain=(1.0 - practice_rate) ** session_idx,
                drift=drift,
            )
        )

    return plans
