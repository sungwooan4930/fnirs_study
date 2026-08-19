"""세션 베이스라인 정규화.

CLAUDE.md §3.8을 구현한다. 정규화는 **세션 단위로 독립 수행**한다 — 여러
세션을 모아 정규화하면 세션 간 차이가 지워지고, 그것이야말로 측정하려는
대상이다.

`fit`은 **한 세션의 시작 베이스라인만** 받는다. 다른 세션·다른 피험자를
넘길 인자가 존재하지 않는다. contract.py의 `TestView.fit()`이 무조건
예외를 던지는 것과 같은 발상이며, 규율이 아니라 서명으로 막는다.

드리프트 계산은 시작·종료 베이스라인을 모두 봐야 하므로 클래스 메서드가
아니라 별도 함수다. `fit`이 시작 베이스라인만 받는다는 요점을 지키기 위함이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 정규화 종류. CLAUDE.md §3.8의 세 규칙에 대응한다.
NORMALIZATION_KINDS: tuple[str, ...] = (
    "concentration_delta",   # fNIRS HbO/HbR — 시작 베이스라인 평균 기준
    "band_power_db",         # EEG 대역 파워 — 시작 베이스라인 대비 dB
    "absolute",              # 행동 — 변환하지 않는다
)

#: 모달리티 → 정규화 종류. `absolute`가 무변환이라고 해서 생략할 수 있는
#: 것은 아니다. 명시적 선택을 요구해 "행동에 어떤 정규화를 적용할지
#: 생각하지 않고 넘어가는" 경로를 막는다.
MODALITY_KIND: dict[str, str] = {
    "eeg": "band_power_db",
    "fnirs": "concentration_delta",
    "behavior": "absolute",
}

#: dB 변환에서 log의 정의역을 지키기 위한 하한.
_DB_FLOOR: float = 1e-12


@dataclass(frozen=True)
class SessionDrift:
    per_channel: np.ndarray   # (n_features,) 상대 변화율. 제외된 채널은 nan
    aggregate: float          # 유효 채널 평균. 전부 제외되면 nan
    n_excluded: int


@dataclass(frozen=True)
class SessionQuality:
    subject_id: str
    session_idx: int
    drift_by_modality: dict[str, float]
    n_excluded_by_modality: dict[str, int]
    drift_flag: bool


class SessionBaseline:
    """한 세션의 시작 베이스라인을 기준으로 삼는 정규화기."""

    def __init__(self, reference: np.ndarray, kind: str) -> None:
        self.reference = reference
        self.kind = kind

    @classmethod
    def fit(cls, start_baseline: np.ndarray, kind: str) -> SessionBaseline:
        """시작 베이스라인 창들의 평균을 기준으로 삼는다.

        `start_baseline`은 (n_baseline_windows, n_features)다.
        """
        if kind not in NORMALIZATION_KINDS:
            raise ValueError(
                f"unknown normalization kind '{kind}'; expected one of "
                f"{list(NORMALIZATION_KINDS)}"
            )
        arr = np.asarray(start_baseline, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"start_baseline must be 2-D (n_windows, n_features), got {arr.shape}"
            )
        if arr.shape[0] == 0:
            raise ValueError(
                "baseline block produced no windows; 정규화 기준 구간을 만들 수 "
                "없다. baseline_duration_s가 window_s보다 짧지 않은지 확인하라"
            )
        return cls(reference=arr.mean(axis=0), kind=kind)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """정규화를 적용한다. `kind`는 fit에서 이미 고정됐다."""
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != len(self.reference):
            raise ValueError(
                f"x has {arr.shape[-1]} features but the baseline reference has "
                f"{len(self.reference)}"
            )
        if self.kind == "concentration_delta":
            return arr - self.reference
        if self.kind == "band_power_db":
            num = np.maximum(arr, _DB_FLOOR)
            den = np.maximum(self.reference, _DB_FLOOR)
            return 10.0 * np.log10(num / den)
        # absolute — 장비 재부착의 영향을 받지 않으므로 값을 그대로 둔다.
        # 원본을 그대로 돌려주면 호출자가 제자리 수정할 때 데이터가 오염되므로
        # 복사본을 돌려준다.
        return arr.copy()


def compute_drift(
    start_baseline: np.ndarray,
    end_baseline: np.ndarray,
    *,
    zero_atol: float,
) -> SessionDrift:
    """시작↔종료 베이스라인의 **상대** 변화율.

    임계를 절대값이 아니라 비율로 두는 이유: 절대값은 모달리티·단위·진입
    수준마다 스케일이 달라 하나의 숫자로 표현할 수 없다.

    드리프트는 **정규화 전 원 단위**에서 계산해야 한다. dB 변환 후에는
    기준값이 정의상 0이 되어 상대 비율이 성립하지 않는다.

    베이스라인 크기가 수치적으로 0에 가까운 채널은 비율이 발산하므로
    집계에서 제외하고 개수를 함께 돌려준다 — 조용히 버리면 "드리프트 없음"
    으로 오독된다.
    """
    start = np.asarray(start_baseline, dtype=float)
    end = np.asarray(end_baseline, dtype=float)
    if start.ndim != 2 or end.ndim != 2:
        raise ValueError("both baselines must be 2-D (n_windows, n_features)")
    if start.shape[0] == 0 or end.shape[0] == 0:
        raise ValueError("baseline block produced no windows; drift cannot be assessed")
    if start.shape[1] != end.shape[1]:
        raise ValueError(
            f"feature count mismatch: start {start.shape[1]} vs end {end.shape[1]}"
        )

    s = start.mean(axis=0)
    e = end.mean(axis=0)

    denom = np.abs(s)
    valid = denom > zero_atol

    per_channel = np.full(len(s), np.nan)
    per_channel[valid] = np.abs(e[valid] - s[valid]) / denom[valid]

    aggregate = float(per_channel[valid].mean()) if valid.any() else float("nan")
    return SessionDrift(
        per_channel=per_channel,
        aggregate=aggregate,
        n_excluded=int((~valid).sum()),
    )


def flag_drift(aggregates: dict[str, float], threshold: float) -> bool:
    """모달리티별 드리프트 중 하나라도 임계를 넘으면 플래그.

    nan(평가 불가)도 플래그한다. 평가할 수 없다는 것은 "문제 없음"이
    아니며, 조용히 통과시키면 신뢰할 수 없는 세션이 신뢰할 수 있는 것처럼
    종단 그래프에 찍힌다.
    """
    for value in aggregates.values():
        if np.isnan(value) or value > threshold:
            return True
    return False
