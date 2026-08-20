"""라벨 구성 — 동시점 타깃과 선행 타깃.

cognitive_load는 창이 속한 블록의 부하다(동시점).

accuracy·response_latency는 선행 타깃이다. 창이 end_s에서 끝날 때
(end_s + lead_delta_s) 이후 첫 자극의 행동을 예측 대상으로 삼는다.
창 안의 행동 특징과 시간이 겹치지 않으므로, 행동을 입력이자 출력으로
쓸 때 생기는 자명한 누수가 발생하지 않는다 (계획서 가설 2).

rt_bins는 config의 고정 임계값이다. 데이터에서 분위수를 구하면
그 자체가 전역 fit이 되어 누수가 된다.
"""

from __future__ import annotations

import numpy as np

from src.simulation.state import TASK


#: 선행(lead) 타깃으로 만들 수 있는 라벨. config의 dataset.lead_targets는
#: 이 집합의 부분집합이어야 한다.
LEAD_TARGETS: tuple[str, ...] = ("accuracy", "response_latency")


def build_labels(
    rec,
    windows,
    *,
    lead_delta_s: float,
    rt_bins: list[float],
    lead_targets: list[str] | tuple[str, ...] = LEAD_TARGETS,
    include_baseline: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """창별 라벨과 유효 마스크를 만든다.

    `lead_targets`는 config의 `dataset.lead_targets`를 그대로 받는다.
    요청한 선행 타깃만 만들고, 모르는 이름이 오면 실행을 거부한다 —
    검증만 되고 아무도 읽지 않는 config 키를 남겨두면 오타가 조용히
    무시되어 스키마를 엄격하게 만든 이유가 사라진다.

    선행 타깃을 하나도 요청하지 않으면 "녹화 끝을 넘어 예측할 자극이
    없다"는 제약 자체가 사라지므로 `keep`은 전부 True다.

    `include_baseline`이 True면 베이스라인 창이 남는다. 그 경우 정규화
    기준 구간이 분석 데이터에도 들어가므로, 호출자는 그 타깃에 쓸 정규화
    기준을 따로 정해야 한다 — 정하지 않으면 Task 12의 정규화 출처 가드가
    fold 실행 시점에 실패시킨다.
    """
    if lead_delta_s < 0:
        raise ValueError(f"lead_delta_s must be >= 0, got {lead_delta_s}")

    lead_targets = tuple(lead_targets)
    unknown = [t for t in lead_targets if t not in LEAD_TARGETS]
    if unknown:
        raise ValueError(
            f"unknown lead target(s) {unknown}; expected a subset of "
            f"{list(LEAD_TARGETS)}"
        )

    n_win = len(windows.start_s)
    onsets = rec.behavior.onsets

    accuracy = np.zeros(n_win, dtype=int)
    latency = np.zeros(n_win, dtype=int)
    keep = np.zeros(n_win, dtype=bool)

    lead_times = windows.end_s + lead_delta_s
    # 각 창에 대해 lead_time 이후 첫 자극의 인덱스
    idx = np.searchsorted(onsets, lead_times, side="left")

    valid = idx < len(onsets)
    keep[valid] = True

    stim_idx = idx[valid]
    accuracy[valid] = rec.behavior.correct[stim_idx]
    latency[valid] = np.digitize(rec.behavior.rt[stim_idx], bins=rt_bins)

    labels = {"cognitive_load": windows.load_level.astype(int)}
    if "accuracy" in lead_targets:
        labels["accuracy"] = accuracy
    if "response_latency" in lead_targets:
        labels["response_latency"] = latency

    if not lead_targets:
        keep = np.ones(n_win, dtype=bool)

    if not include_baseline:
        # 베이스라인 창을 배제하는 이유 세 가지 (스펙 §6.5):
        # 1. 자기 자신으로 정규화한 표본은 퇴화한다 — 정규화 후 구조적으로
        #    0 근처가 되어 완벽히 분리되는 클래스가 공짜로 생긴다.
        # 2. chance level이 3클래스(33.3%)에서 4클래스(25%)로 조용히 바뀐다.
        #    "3수준 분류"의 정의는 CLAUDE.md §9.2에서 아직 미확정이다.
        # 3. session_drift가 베이스라인에서 계산되므로 신뢰도 플래그와
        #    플래그 대상이 같은 숫자가 된다.
        # 삭제가 아니라 배제다. 창 자체는 데이터셋에 남고, 피로 같은 타깃은
        # include_baseline=True로 옵트인할 수 있다.
        keep = keep & (windows.block_kind == TASK)

    return labels, keep
