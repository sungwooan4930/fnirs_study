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


def build_labels(
    rec,
    windows,
    *,
    lead_delta_s: float,
    rt_bins: list[float],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """창별 라벨과 유효 마스크를 만든다."""
    if lead_delta_s < 0:
        raise ValueError(f"lead_delta_s must be >= 0, got {lead_delta_s}")

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

    labels = {
        "cognitive_load": windows.load_level.astype(int),
        "accuracy": accuracy,
        "response_latency": latency,
    }
    return labels, keep
