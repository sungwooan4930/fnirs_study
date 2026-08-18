"""인지상태 궤적 생성.

과제 블록 구조(n-back 0/2/3)가 인지부하를 결정하고, 경과 시간이 피로를
결정한다. 이 궤적이 이후 모든 신호 성분의 입력이자 라벨의 출처다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_SFREQ: float = 10.0


@dataclass(frozen=True)
class CognitiveStateTimeline:
    t: np.ndarray            # (n,) 초
    load_level: np.ndarray   # (n,) int — nback_levels의 인덱스 (0/1/2)
    fatigue: np.ndarray      # (n,) float 0..1, 단조 증가
    trial_id: np.ndarray     # (n,) int — 블록 번호
    stim_onsets: np.ndarray  # (n_stim,) 초
    duration_s: float

    def load_at(self, t: np.ndarray) -> np.ndarray:
        """임의 시점의 인지부하를 계단 보간으로 조회한다.

        블록 설계이므로 선형 보간이 아니라 계단 보간이 옳다.
        범위를 벗어나면 양 끝값으로 고정한다.
        """
        idx = np.searchsorted(self.t, t, side="right") - 1
        return self.load_level[np.clip(idx, 0, len(self.t) - 1)]

    def trial_at(self, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.t, t, side="right") - 1
        return self.trial_id[np.clip(idx, 0, len(self.t) - 1)]


def build_timeline(task_cfg: dict, rng: np.random.Generator) -> CognitiveStateTimeline:
    """과제 설정으로부터 인지상태 궤적을 만든다."""
    levels = list(task_cfg["nback_levels"])
    block_s = float(task_cfg["block_duration_s"])
    n_per_level = int(task_cfg["n_blocks_per_level"])
    stim_interval = float(task_cfg["stim_interval_s"])

    # 카운터밸런스: 블록 순서를 무작위로 섞어 순서 효과를 통제한다
    block_levels = np.repeat(np.arange(len(levels)), n_per_level)
    rng.shuffle(block_levels)

    n_blocks = len(block_levels)
    duration_s = n_blocks * block_s
    n_samples = int(round(duration_s * STATE_SFREQ))
    samples_per_block = int(round(block_s * STATE_SFREQ))

    t = np.arange(n_samples) / STATE_SFREQ
    load_level = np.repeat(block_levels, samples_per_block).astype(int)
    trial_id = np.repeat(np.arange(n_blocks), samples_per_block).astype(int)

    # 피로: 세션 경과에 따라 0 → 1 직전까지 선형 증가
    fatigue = np.linspace(0.0, 1.0, n_samples, endpoint=False)

    stim_onsets = np.arange(0.0, duration_s, stim_interval)

    return CognitiveStateTimeline(
        t=t,
        load_level=load_level,
        fatigue=fatigue,
        trial_id=trial_id,
        stim_onsets=stim_onsets,
        duration_s=duration_s,
    )
