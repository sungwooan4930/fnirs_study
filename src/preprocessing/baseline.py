"""세션 베이스라인 정규화.

CLAUDE.md §3.8을 구현한다. 정규화는 **세션 단위로 독립 수행**한다 — 여러
세션을 모아 정규화하면 세션 간 차이가 지워지고, 그것이야말로 측정하려는
대상이다.

`fit`은 **한 세션의 시작 베이스라인만** 받는다. 다른 세션·다른 피험자를
넘길 인자가 존재하지 않는다. contract.py의 `TestView.fit()`이 무조건
예외를 던지는 것과 같은 발상이며, 규율이 아니라 서명으로 막는다.

드리프트 계산은 시작·종료 베이스라인을 모두 봐야 하므로 클래스 메서드가
아니라 별도 함수다. `fit`이 시작 베이스라인만 받는다는 요점을 지키기 위함이다.

**2026-08-19 정정 (Task 14/T3):** `concentration_delta`가 뺄셈만 하던
초판은 틀렸다. 근거·수정 내용은 `SessionBaseline` 클래스 docstring과
`docs/specs/2026-08-19-session-baseline-design.md` §6.2를 참조.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 정규화 종류. CLAUDE.md §3.8의 세 규칙에 대응한다.
NORMALIZATION_KINDS: tuple[str, ...] = (
    "concentration_delta",   # fNIRS HbO/HbR — 시작 베이스라인 평균·산포 기준
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

#: `concentration_delta`가 베이스라인 산포로 나눌 때의 하한. 채널이 거의
#: 상수(σ≈0)면 나눗셈이 발산하거나 잡음을 극단적으로 증폭한다 —
#: `compute_drift`의 `zero_atol`과 같은 결의 방어다. `config`의
#: `zero_atol` 기본값(1e-8)과 자릿수를 맞췄다.
_SCALE_FLOOR: float = 1e-8


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
    """한 세션의 시작 베이스라인을 기준으로 삼는 정규화기.

    **`concentration_delta`가 왜 뺄셈만으로는 부족한가.** 옵토드 재부착으로
    생기는 세션 간 드리프트의 지배적 성분은 결합도·유효 광경로 변화이며,
    이는 **곱셈적**이다 — 세션의 신호를 `y = gain·x + offset` 형태로 모델링할
    수 있다(`gain`이 결합도, `offset`이 잔여 오프셋). 시작 베이스라인 평균을
    빼기만 하면

        y - mean(y_base) = gain·(x - mean(x_base))

    이 되어 **오프셋은 지워지지만 이득(gain)은 그대로 남는다.** 세션마다
    다른 `gain`이 잔류하면, 세션 간 분류기는 진짜 인지상태 신호가 아니라
    이 잔류 스케일 차이를 학습할 수 있다 — Task 14 T3(회복) 검증에서 (이
    정정 이전, 뺄셈만 하던 원안 기준) 실측 회복률이 0.268(요구 ≥0.5)에
    그쳤고, `fnirs_gain_sigma=0`으로 두면(다른 드리프트는 유지) 회복률이
    0.378로 오르는 것으로 원인이 확인됐다. 아래 수정을 적용한 뒤 회복률은
    0.335로 개선됐다(잔여 미달의 원인은 정규화가 아니라 `cross_session`·
    `within_subject` 두 분할 방식의 구조적 난이도 차이 — 상세는
    `docs/specs/2026-08-19-session-baseline-design.md` §8.4.1). (참고로
    EEG의 `band_power_db`는 애초에 **비율** `10·log10(y/mean(y_base))`이라
    `gain`이 분자·분모에서 상쇄되므로 이 문제가 없다 — 두 모달리티가
    비대칭이었다.)

    그래서 `concentration_delta`는 베이스라인 산포로도 나눈다:

        (y - mean(y_base)) / std(y_base)
            = (gain·x + offset - (gain·mean(x_base) + offset)) / (gain·std(x_base))
            = (x - mean(x_base)) / std(x_base)

    `gain`이 분자·분모 양쪽에 곱해져 있어 소거된다. 결과 단위는 **베이스라인
    산포 대비 배수**(무차원)로 바뀐다 — 더 이상 원래의 농도 단위가 아니다.

    **이것이 "세션 전체를 z-score"하는 것과 다른 이유.** 분모(`std`)는
    **베이스라인 블록**의 산포이지 세션 전체(베이스라인+과제)의 산포가
    아니다. 베이스라인은 안정 상태 측정이라 정의상 과제 반응을 담지 않는다
    — 따라서 분모는 순수하게 "그 세션·그 채널의 잡음/생리적 배경 변동
    스케일"만 반영하고, 분자의 과제 반응 크기(= 연습 효과가 만드는 신호
    변화)는 그대로 보존된다. 세션 전체를 z-score하면 분모에 과제 반응 자체의
    분산이 섞여 들어가 신호를 갉아먹는다 — Task 15(T4, 기울기 보존율)가
    정확히 이 성질을 검증하도록 설계돼 있다.

    `kind`는 `fit`에서 한 번만 받는다. `apply`가 다시 받으면 fit과 다른
    `kind`를 넘길 수 있게 되고, 그건 의미 있는 사용처가 없으면서 조용히
    틀릴 수 있는 경로다.
    """

    def __init__(
        self, reference: np.ndarray, kind: str, scale: np.ndarray | None = None
    ) -> None:
        self.reference = reference
        self.kind = kind
        #: `concentration_delta`에서만 쓰는 베이스라인 산포(ddof=1, 이미
        #: `_SCALE_FLOOR`로 하한 처리됨). 다른 kind는 `None`.
        #: 기본값을 두는 이유: 외부에서 `SessionBaseline(reference=..., kind=...)`
        #: 처럼 `fit`을 거치지 않고 직접 구성하는 테스트 훅(예: 결함 주입)이
        #: 이미 존재하고, 그 경로까지 이 정정으로 깨뜨릴 이유가 없다. 다만
        #: `scale=None`인 채로 `concentration_delta`를 `apply`하면 옛 동작
        #: (뺄셈만, gain 미보정)으로 되돌아간다는 점은 호출자가 알아야 한다.
        self.scale = scale

    @classmethod
    def fit(cls, start_baseline: np.ndarray, kind: str) -> SessionBaseline:
        """시작 베이스라인 창들의 평균(과 `concentration_delta`라면 산포)을
        기준으로 삼는다.

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
        scale = None
        if kind == "concentration_delta":
            if arr.shape[0] >= 2:
                scale = arr.std(axis=0, ddof=1)
            else:
                # 창이 하나뿐이면 표본 표준편차(ddof=1)가 정의되지 않는다
                # (0/0). 아래 _SCALE_FLOOR 하한이 곧바로 적용되도록 0으로
                # 시작한다 — NaN을 만들어 하한 클램프를 무력화하지 않는다.
                scale = np.zeros(arr.shape[1])
            scale = np.maximum(scale, _SCALE_FLOOR)
        return cls(reference=arr.mean(axis=0), kind=kind, scale=scale)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """정규화를 적용한다. `kind`는 fit에서 이미 고정됐다."""
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != len(self.reference):
            raise ValueError(
                f"x has {arr.shape[-1]} features but the baseline reference has "
                f"{len(self.reference)}"
            )
        if self.kind == "concentration_delta":
            centered = arr - self.reference
            if self.scale is None:
                return centered
            return centered / self.scale
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
