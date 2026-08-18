"""슬라이딩 윈도우 생성.

계획서가 명시한 5초 창·1초 스텝은 80% 오버랩이다. 오버랩 자체는
문제가 아니지만, 인접 창이 train/test로 갈리면 성능이 통째로 허구가 된다.
그 방지는 분할기(피험자 단위)와 가드가 담당한다.

여기서는 라벨 모호성만 없앤다: 블록 경계를 넘는 창은 인지부하가
두 값에 걸치므로 버린다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowIndex:
    start_s: np.ndarray    # (n_win,)
    end_s: np.ndarray      # (n_win,)
    trial_id: np.ndarray   # (n_win,) int
    load_level: np.ndarray  # (n_win,) int


def make_windows(timeline, window_s: float, step_s: float) -> WindowIndex:
    """블록 경계를 넘지 않는 창 목록을 만든다."""
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if step_s <= 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    if step_s > window_s:
        raise ValueError(
            f"step_s ({step_s}) > window_s ({window_s}) would skip samples; "
            "계획서는 5초 창·1초 스텝을 명시한다"
        )

    starts = np.arange(0.0, timeline.duration_s - window_s + 1e-9, step_s)
    ends = starts + window_s

    start_trial = timeline.trial_at(starts)
    end_trial = timeline.trial_at(ends - 1e-6)
    keep = start_trial == end_trial

    if not keep.any():
        raise ValueError(
            f"no windows fit inside a block: window_s={window_s} is likely "
            "longer than block_duration_s"
        )

    starts = starts[keep]
    ends = ends[keep]

    return WindowIndex(
        start_s=starts,
        end_s=ends,
        trial_id=timeline.trial_at(starts),
        load_level=timeline.load_at(starts),
    )
