"""인지상태 궤적 생성.

과제 블록 구조(n-back 0/2/3)가 인지부하를 결정하고, 경과 시간이 피로를
결정한다. 이 궤적이 이후 모든 신호 성분의 입력이자 라벨의 출처다.

세션은 `베이스라인 → [과제 블록 × N] → 베이스라인` 구조다 (CLAUDE.md §2.4).
시작 베이스라인은 정규화의 기준 구간이고, 종료 베이스라인은 세션 내
드리프트 추정치다. 둘 다 생략할 수 없다.

연습 효과(스펙 §5.4)는 신호가 아니라 **상태**에 심는다. 같은 n-back 수준이라도
세션을 거듭하면 실제로 덜 부담스러워지므로, 정규화가 지워서는 안 되는 변화다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.components.base import load_fraction

STATE_SFREQ: float = 10.0

#: 블록 종류
BASELINE: int = 0
TASK: int = 1

#: 베이스라인 구간의 load_level. 과제 조건이 아니므로 실제 수준(0..n-1)과
#: 겹치지 않는 음수를 쓴다. 라벨 빌더가 이 값으로 베이스라인을 골라낸다.
BASELINE_LOAD_SENTINEL: int = -1


@dataclass(frozen=True)
class CognitiveStateTimeline:
    t: np.ndarray            # (n,) 초
    load_level: np.ndarray   # (n,) int — nback_levels의 인덱스, 베이스라인은 -1
    block_kind: np.ndarray   # (n,) int — BASELINE / TASK
    fatigue: np.ndarray      # (n,) float 0..1, 단조 증가
    trial_id: np.ndarray     # (n,) int — 블록 번호 (세션 안에서 0부터)
    stim_onsets: np.ndarray  # (n_stim,) 초 — 과제 블록 안에만 있다
    duration_s: float
    session_idx: int
    n_levels: int
    practice_gain: float

    def _index_at(self, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.t, t, side="right") - 1
        return np.clip(idx, 0, len(self.t) - 1)

    def load_at(self, t: np.ndarray) -> np.ndarray:
        """임의 시점의 인지부하 수준을 계단 보간으로 조회한다.

        블록 설계이므로 선형 보간이 아니라 계단 보간이 옳다.
        범위를 벗어나면 양 끝값으로 고정한다.
        """
        return self.load_level[self._index_at(t)]

    def trial_at(self, t: np.ndarray) -> np.ndarray:
        return self.trial_id[self._index_at(t)]

    def kind_at(self, t: np.ndarray) -> np.ndarray:
        return self.block_kind[self._index_at(t)]

    def effective_load(self, t: np.ndarray) -> np.ndarray:
        """신호 성분이 구동되는 실효 부하 (0..1).

        베이스라인 구간은 안정 상태이므로 0이다 — 센티넬 -1을 그대로
        비율로 바꾸면 음수 구동이 되어 신호가 뒤집힌다.

        practice_gain이 여기 곱해진다. 라벨(과제 조건)은 바뀌지 않고
        상태만 바뀐다는 것이 스펙 §5.1의 층 분리다.
        """
        levels = np.clip(self.load_at(t), 0, None)
        return load_fraction(levels, self.n_levels) * self.practice_gain


def build_timeline(
    task_cfg: dict,
    rng: np.random.Generator,
    *,
    session_idx: int = 0,
    practice_gain: float = 1.0,
) -> CognitiveStateTimeline:
    """한 세션의 인지상태 궤적을 만든다."""
    levels = list(task_cfg["nback_levels"])
    block_s = float(task_cfg["block_duration_s"])
    baseline_s = float(task_cfg["baseline_duration_s"])
    n_per_level = int(task_cfg["n_blocks_per_level"])
    stim_interval = float(task_cfg["stim_interval_s"])

    if baseline_s <= 0:
        raise ValueError(
            f"baseline_duration_s must be positive, got {baseline_s}; "
            "CLAUDE.md §2.4는 매 세션 시작·종료 베이스라인을 생략 불가로 규정한다"
        )

    # 카운터밸런스: 블록 순서를 무작위로 섞어 순서 효과를 통제한다
    block_levels = np.repeat(np.arange(len(levels)), n_per_level)
    rng.shuffle(block_levels)

    samples_per_block = int(round(block_s * STATE_SFREQ))
    samples_per_baseline = int(round(baseline_s * STATE_SFREQ))

    kinds: list[np.ndarray] = []
    loads: list[np.ndarray] = []
    trials: list[np.ndarray] = []
    trial_counter = 0

    def _append(n: int, kind: int, load: int) -> None:
        nonlocal trial_counter
        kinds.append(np.full(n, kind, dtype=int))
        loads.append(np.full(n, load, dtype=int))
        trials.append(np.full(n, trial_counter, dtype=int))
        trial_counter += 1

    _append(samples_per_baseline, BASELINE, BASELINE_LOAD_SENTINEL)
    for level in block_levels:
        _append(samples_per_block, TASK, int(level))
    _append(samples_per_baseline, BASELINE, BASELINE_LOAD_SENTINEL)

    block_kind = np.concatenate(kinds)
    load_level = np.concatenate(loads)
    trial_id = np.concatenate(trials)

    n_samples = len(block_kind)
    duration_s = n_samples / STATE_SFREQ
    t = np.arange(n_samples) / STATE_SFREQ

    # 피로: 세션 경과에 따라 0 → 1 직전까지 선형 증가
    fatigue = np.linspace(0.0, 1.0, n_samples, endpoint=False)

    # 자극은 과제 블록 안에만 제시한다. 베이스라인은 고정점 응시이므로
    # 자극이 없다 — 넣으면 "안정 상태"가 아니게 되어 기준 구간이 오염된다.
    all_onsets = np.arange(0.0, duration_s, stim_interval)
    onset_idx = np.clip(np.searchsorted(t, all_onsets, side="right") - 1, 0, n_samples - 1)
    stim_onsets = all_onsets[block_kind[onset_idx] == TASK]

    return CognitiveStateTimeline(
        t=t,
        load_level=load_level,
        block_kind=block_kind,
        fatigue=fatigue,
        trial_id=trial_id,
        stim_onsets=stim_onsets,
        duration_s=duration_s,
        session_idx=int(session_idx),
        n_levels=len(levels),
        practice_gain=float(practice_gain),
    )
