# 세션 베이스라인 인프라 (B1) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 매 세션 베이스라인 측정과 세션별 독립 정규화를 파이프라인에 넣고, 정규화가 측정 드리프트를 지우면서 진짜 인지 변화는 보존한다는 것을 합성 데이터의 ground truth로 증명한다.

**Architecture:** 생성기에 세션 축과 베이스라인 블록을 추가하고, 변동원을 두 층에 나눠 심는다 — ①옵토드 ②캡·임피던스 ③세션내 드리프트는 **신호 변환**에, ④연습 효과는 **인지상태 자체**에. `SessionBaseline`이 한 세션의 시작 베이스라인만 받아 정규화하고, 계약·분할기·가드가 세션 축을 인지하도록 확장된다. 승인기준 T1~T6이 각각 결함 주입과 함께 붙는다.

**Tech Stack:** Python 3.11 · numpy 1.26 · scipy 1.16 · scikit-learn 1.7 · PyYAML · pytest. **새 의존성 없음** (`pyprep`은 B3에서 추가).

**Spec:** `docs/specs/2026-08-19-session-baseline-design.md`

## Global Constraints

- **분할 단위는 항상 피험자다.** 윈도우·에포크 단위 랜덤 분할 금지 (`CLAUDE.md` §5.1).
- **`StandardScaler`·PCA·특징 선택은 train fold 안에서만 fit** 한다. 전체 데이터 fit 금지.
- **세션 베이스라인은 그 세션 자신의 것만 쓴다.** 다른 피험자·다른 세션 통계 참조 금지 (`CLAUDE.md` §3.8).
- **난수열 규율:** 파라미터 크기가 0이어도 분기하지 않고 항상 `rng`를 소모한다. `np.zeros` 우회 금지 (스펙 §5.5). 이유는 `src/simulation/subject.py`의 `make_subjects` 주석에 이미 적혀 있다.
- **정확도를 보고할 때는 항상 chance level과 CV 방식을 함께 적는다.** LOSO와 within-subject를 혼용 표기 금지 (`CLAUDE.md` §5.4).
- **하드코딩 금지:** `n_subjects`·`n_sessions`·드리프트 파라미터·임계치는 전부 `config/*.yaml`에서만 온다 (`CLAUDE.md` §5.2, §9.2).
- **config 스키마는 적힌 키를 전부 필수로 취급한다.** 키를 하나 추가하면 `config/experiments/` 세 파일을 모두 갱신해야 한다.
- **모든 테스트는 대상 결함을 주입했을 때 실제로 실패하는 것을 함께 보인다.** 실패할 수 없는 테스트는 테스트가 아니다 (스펙 §8).
- **pytest 출력은 반드시 리다이렉션으로 캡처한다:** `python -m pytest ... > task-N-testlog.txt 2>&1`. 트랜스크립트를 손으로 옮겨 적지 않는다. 파일을 남겨 컨트롤러가 직접 읽는다.
- **커밋 메시지는 한국어**, 본문에 "왜"를 적는다. 기존 커밋 로그의 형식을 따른다.

## 상수 · 이름 규약 (전 태스크 공통)

| 이름 | 값/타입 | 정의 위치 |
|---|---|---|
| `BASELINE` | `0` (int) | `src/simulation/state.py` |
| `TASK` | `1` (int) | `src/simulation/state.py` |
| `BASELINE_LOAD_SENTINEL` | `-1` (int) | `src/simulation/state.py` |
| `NORMALIZATION_KINDS` | `("concentration_delta", "band_power_db", "absolute")` | `src/preprocessing/baseline.py` |
| `MODALITY_KIND` | `{"eeg": "band_power_db", "fnirs": "concentration_delta", "behavior": "absolute"}` | `src/preprocessing/baseline.py` |

## 파일 구조

**신규**

| 파일 | 책임 |
|---|---|
| `src/simulation/session.py` | 세션별 드리프트 파라미터 추출 (`SessionDriftParams`, `SessionPlan`, `plan_sessions`) |
| `src/simulation/components/drift.py` | 뽑힌 파라미터를 깨끗한 신호에 적용 (`apply_channel_drift`, `apply_noise_scaling`) |
| `src/preprocessing/__init__.py` | 빈 패키지 마커 |
| `src/preprocessing/baseline.py` | `SessionBaseline` · `SessionDrift` · `SessionQuality` · `compute_drift` |
| `config/experiments/session_recovery.yaml` | T2·T3용 (드리프트 on) |
| `config/experiments/session_clean.yaml` | T4 기준선용 (드리프트 off·정규화 off) |
| `config/experiments/session_quality.yaml` | T6용 (`drift.assignment: fixed_2x2`) |
| `tests/simulation/test_session.py` | |
| `tests/simulation/test_drift.py` | |
| `tests/preprocessing/test_baseline.py` | |
| `tests/evaluation/test_validation_session.py` | 승인기준 T1~T6 |

**수정**

| 파일 | 무엇을 |
|---|---|
| `src/common/config.py` | 스키마에 세션·드리프트·정규화 키 추가, `validate_config` 공개 |
| `src/simulation/state.py` | `block_kind`·`session_idx`·`n_levels`·`practice_gain`·`effective_load()`, 베이스라인 블록 |
| `src/simulation/components/{eeg_oscillation,eeg_erp,fnirs_hrf,behavior}.py` | `load_fraction(timeline.load_at(t))` → `timeline.effective_load(t)` |
| `src/simulation/recording.py` | 피험자×세션 루프, `SyntheticRecording.session_idx` |
| `src/datasets/windowing.py` | `WindowIndex.block_kind` 전파 |
| `src/datasets/labels.py` | 베이스라인 창 배제(옵트인) |
| `src/datasets/contract.py` | `session_ids` · `SplitKeys` · `normalization_source_mask` · trial_id 고유성 검증 |
| `src/evaluation/splitters.py` | `SplitKeys` 이행 · `allows_same_session` · `CrossSessionSplitter` |
| `src/evaluation/guards.py` | 창 겹침을 `(피험자,세션)` 단위로 · 세션 중첩 가드 · 정규화 출처 가드 |
| `src/evaluation/harness.py` | 새 가드 호출 · `allows_same_session` 존중 |
| `src/evaluation/metrics.py` | `aggregate_runs` — `cv_method`가 다르면 병합 거부 |
| `src/evaluation/runner.py` | P1(오버라이드 재검증) · 세션 루프 · 정규화 적용 · 전역 고유 trial_id |
| `config/experiments/{smoke,null,pilot}.yaml` | 새 키 추가 |

---

## Task 1: config 스키마 확장 + P1(오버라이드 검증 우회) 수정

**Files:**
- Modify: `src/common/config.py` (SCHEMA, `validate_config` 신설)
- Modify: `src/evaluation/runner.py` (`run_experiment` 초입)
- Modify: `config/experiments/smoke.yaml`, `config/experiments/null.yaml`, `config/experiments/pilot.yaml`
- Test: `tests/common/test_config.py`, `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces:
  - `src.common.config.validate_config(cfg: dict) -> None` — 스키마 위반 시 `ConfigError`
  - config 경로 `simulation.n_sessions`(int), `simulation.task.baseline_duration_s`(float), `simulation.practice.rate`(float), `simulation.drift.{fnirs_gain_sigma,fnirs_offset_sigma,eeg_gain_sigma,eeg_noise_sigma,within_session_rate,within_session_fraction,between_session_scale,assignment}`, `preprocessing.baseline.{normalize,drift_threshold_relative,zero_atol}`, `dataset.include_baseline`(bool), `evaluation.guards.{check_session_overlap,check_normalization_source}`(bool)

**왜 먼저인가:** 뒤의 모든 태스크가 config 키를 읽는다. 그리고 P1은 "키를 추가할수록 위험해지는" 결함이라 키를 추가하기 **전에** 고쳐야 한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/common/test_config.py` 끝에 추가:

```python
from src.common.config import ConfigError, load_config, validate_config


def test_smoke_config_carries_session_keys():
    cfg = load_config("config/experiments/smoke.yaml")
    assert cfg["simulation"]["n_sessions"] >= 2
    assert cfg["simulation"]["task"]["baseline_duration_s"] > 0
    assert cfg["simulation"]["practice"]["rate"] >= 0.0
    assert cfg["simulation"]["drift"]["assignment"] in ("sampled", "fixed_2x2")
    assert cfg["preprocessing"]["baseline"]["normalize"] is True
    assert cfg["dataset"]["include_baseline"] is False
    assert cfg["evaluation"]["guards"]["check_session_overlap"] is True
    assert cfg["evaluation"]["guards"]["check_normalization_source"] is True


def test_validate_config_rejects_unknown_key_after_manual_edit():
    cfg = load_config("config/experiments/smoke.yaml")
    cfg["simulation"]["n_sesions"] = 3          # 오타
    with pytest.raises(ConfigError, match=r"unknown key 'simulation\.n_sesions'"):
        validate_config(cfg)


def test_validate_config_rejects_missing_key():
    cfg = load_config("config/experiments/smoke.yaml")
    del cfg["simulation"]["n_sessions"]
    with pytest.raises(ConfigError, match=r"missing required key 'simulation\.n_sessions'"):
        validate_config(cfg)
```

`tests/evaluation/test_runner.py` 끝에 추가:

```python
from src.common.config import ConfigError


def test_typo_in_overrides_is_rejected_not_silently_absorbed():
    """P1: 오버라이드가 스키마 검증을 우회하면 오타 키가 조용히 통과한다.

    조용히 통과하면 config 해시만 바뀌어 새 결과 디렉토리가 생기고,
    아무 손잡이도 돌리지 않은 실행이 별개 조건인 것처럼 기록된다.
    """
    with pytest.raises(ConfigError, match=r"unknown key 'simulation\.n_sesions'"):
        run_experiment(
            "config/experiments/smoke.yaml",
            overrides={"simulation": {"n_sesions": 3}},
        )
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/common/test_config.py tests/evaluation/test_runner.py -v > task-1-testlog-red.txt 2>&1
```

Expected: `test_smoke_config_carries_session_keys` FAIL (`KeyError: 'n_sessions'`), `test_validate_config_*` FAIL (`ImportError: cannot import name 'validate_config'`), `test_typo_in_overrides_*` FAIL (`Failed: DID NOT RAISE`).

- [ ] **Step 3: 스키마를 확장하고 `validate_config`를 공개한다**

`src/common/config.py`의 `SCHEMA`를 아래로 교체:

```python
SCHEMA: dict[str, Any] = {
    "run_name": None,
    "seed": None,
    "simulation": {
        "n_subjects": None,
        "n_sessions": None,
        "subject_variance": None,
        "effect_size": None,
        "lead_delta_s": None,
        "task": {
            "nback_levels": None,
            "block_duration_s": None,
            "baseline_duration_s": None,
            "n_blocks_per_level": None,
            "stim_interval_s": None,
        },
        "practice": {"rate": None},
        "drift": {
            "fnirs_gain_sigma": None,
            "fnirs_offset_sigma": None,
            "eeg_gain_sigma": None,
            "eeg_noise_sigma": None,
            "within_session_rate": None,
            "within_session_fraction": None,
            "between_session_scale": None,
            "assignment": None,
        },
        "eeg": {"n_channels": None, "sfreq_hz": None},
        "fnirs": {"n_channels": None, "sfreq_hz": None, "hbr_coupling": None},
    },
    "windowing": {"window_s": None, "step_s": None},
    "features": {"extractor": None},
    "preprocessing": {
        "baseline": {
            "normalize": None,
            "drift_threshold_relative": None,
            "zero_atol": None,
        },
    },
    "dataset": {
        "targets": None,
        "lead_targets": None,
        "modalities": None,
        "rt_bins": None,
        "include_baseline": None,
    },
    "evaluation": {
        "splitter": None,
        "model": None,
        "guards": {
            "check_subject_overlap": None,
            "check_window_overlap": None,
            "check_session_overlap": None,
            "check_normalization_source": None,
        },
    },
    "output": {"results_dir": None},
}
```

같은 파일의 `load_config` 위에 공개 함수를 추가:

```python
def validate_config(cfg: dict) -> None:
    """이미 dict인 config를 스키마에 대해 검증한다.

    `load_config`가 파일을 읽은 뒤 호출하는 것과 같은 검사다. 오버라이드를
    병합한 뒤에도 같은 검사를 돌릴 수 있도록 분리했다 — 병합 후 검증하지
    않으면 오타 난 키가 조용히 흡수되고, config 해시만 바뀌어 아무 손잡이도
    돌리지 않은 실행이 별개 조건으로 기록된다.
    """
    if not isinstance(cfg, dict):
        raise ConfigError(f"top level must be a mapping, got {type(cfg).__name__}")
    _check_node(cfg, SCHEMA, "")
    _check_required(cfg, SCHEMA, "")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ConfigError(f"seed must be int, got {type(cfg['seed']).__name__}")
```

`load_config`의 본문 마지막 검증 3줄을 `validate_config(cfg)` 한 줄로 교체:

```python
def load_config(path: str | Path) -> dict:
    """YAML config를 읽고 스키마를 검증해 반환한다."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    validate_config(cfg)
    return cfg
```

- [ ] **Step 4: 러너가 오버라이드 병합 후 재검증하게 한다**

`src/evaluation/runner.py`의 import에 추가:

```python
from src.common.config import load_config, validate_config
```

`run_experiment` 초입을 교체:

```python
    cfg = load_config(config_path)
    if overrides:
        cfg = _deep_update(cfg, overrides)
        # P1: 병합 후 반드시 재검증한다. 하지 않으면 오타 난 오버라이드 키가
        # 조용히 통과하고, _config_hash만 바뀌어 새 결과 디렉토리가 생긴다 —
        # 아무 손잡이도 돌리지 않은 실행이 별개 조건처럼 기록된다.
        validate_config(cfg)
```

- [ ] **Step 5: config 세 파일에 새 키를 넣는다**

`config/experiments/smoke.yaml`을 아래로 교체:

```yaml
run_name: smoke
seed: 42

simulation:
  n_subjects: 6
  n_sessions: 2
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 30
    baseline_duration_s: 20
    n_blocks_per_level: 2
    stim_interval_s: 2.0
  practice:
    rate: 0.15
  drift:
    fnirs_gain_sigma: 0.20
    fnirs_offset_sigma: 0.10
    eeg_gain_sigma: 0.15
    eeg_noise_sigma: 0.15
    within_session_rate: 0.30
    within_session_fraction: 0.33
    between_session_scale: 4.0
    assignment: sampled
  # 주의: EEG 샘플링 속도 250 Hz는 개발 속도 최적화. 정규 1000 Hz는 pilot.yaml.
  eeg:   {n_channels: 30, sfreq_hz: 250}
  fnirs: {n_channels: 48, sfreq_hz: 10.4, hbr_coupling: -0.33}

windowing:
  window_s: 5.0
  step_s: 1.0

features:
  extractor: minimal

preprocessing:
  baseline:
    normalize: true
    drift_threshold_relative: 0.20
    zero_atol: 1.0e-8

dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy, response_latency]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]
  include_baseline: false

evaluation:
  splitter: loso
  model: logistic_regression
  guards:
    check_subject_overlap: true
    check_window_overlap: true
    check_session_overlap: true
    check_normalization_source: true

output:
  results_dir: results
```

`config/experiments/null.yaml`과 `config/experiments/pilot.yaml`에도 **같은 블록**을 넣는다. 각 파일에서 기존 값은 그대로 두고 아래만 추가한다:

- `simulation:` 아래 `n_sessions: 3`
- `simulation.task:` 아래 `baseline_duration_s: 60`
- `simulation:` 아래 `practice:` / `drift:` 블록 (smoke와 동일한 값)
- 최상위 `preprocessing:` 블록 (smoke와 동일)
- `dataset:` 아래 `include_baseline: false`
- `evaluation.guards:` 아래 `check_session_overlap: true`, `check_normalization_source: true`

> `n_sessions`가 smoke는 2, null·pilot은 3인 이유: smoke는 개발 속도용이고, null·pilot은 스펙 §5.2대로 추세선 최소 요건 3회를 만족해야 한다.

- [ ] **Step 6: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/common/test_config.py tests/evaluation/test_runner.py -v > task-1-testlog-green.txt 2>&1
```

Expected: 새 테스트 4개 PASS.

> 이 시점에서 `n_sessions` 등을 읽는 코드는 아직 없다. 러너 전체 스위트는 Task 13까지 빨간불일 수 있다. 위 두 파일만 초록이면 진행한다.

- [ ] **Step 7: 커밋**

```bash
git add src/common/config.py src/evaluation/runner.py config/experiments tests/common/test_config.py tests/evaluation/test_runner.py
git commit -m "feat: config 스키마에 세션·드리프트·정규화 키 추가 + P1 수정

오버라이드가 load_config 뒤에 병합되어 스키마 검증을 우회했다. 오타 난 키가
조용히 통과하고 _config_hash만 바뀌어, 아무 손잡이도 돌리지 않은 실행이
별개 조건처럼 새 디렉토리에 기록됐다. B1은 config 키를 여럿 추가하므로
위험이 커진 상태라 키 추가 전에 먼저 고친다.

validate_config를 공개해 load_config와 오버라이드 병합 양쪽에서 같은 검사를 쓴다."
```

---

## Task 2: 타임라인에 베이스라인 블록 · 세션 번호

**Files:**
- Modify: `src/simulation/state.py`
- Test: `tests/simulation/test_state.py`

**Interfaces:**
- Consumes: Task 1의 `simulation.task.baseline_duration_s`
- Produces:
  - `src.simulation.state.BASELINE = 0`, `TASK = 1`, `BASELINE_LOAD_SENTINEL = -1`
  - `CognitiveStateTimeline` 필드 추가: `block_kind: np.ndarray`, `session_idx: int`, `n_levels: int`, `practice_gain: float`
  - `CognitiveStateTimeline.kind_at(t: np.ndarray) -> np.ndarray`
  - `build_timeline(task_cfg: dict, rng, *, session_idx: int = 0, practice_gain: float = 1.0) -> CognitiveStateTimeline`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/simulation/test_state.py` 끝에 추가:

```python
from src.simulation.state import (
    BASELINE,
    BASELINE_LOAD_SENTINEL,
    TASK,
    build_timeline,
)

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def test_session_starts_and_ends_with_a_baseline_block():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    assert tl.block_kind[0] == BASELINE
    assert tl.block_kind[-1] == BASELINE
    # 가운데 어딘가는 과제여야 한다
    assert (tl.block_kind == TASK).any()


def test_baseline_blocks_are_two_and_bookend_the_task_blocks():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    baseline_trials = np.unique(tl.trial_id[tl.block_kind == BASELINE])
    assert len(baseline_trials) == 2
    assert baseline_trials[0] == tl.trial_id.min()
    assert baseline_trials[1] == tl.trial_id.max()


def test_baseline_samples_carry_the_load_sentinel():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    assert (tl.load_level[tl.block_kind == BASELINE] == BASELINE_LOAD_SENTINEL).all()
    assert (tl.load_level[tl.block_kind == TASK] >= 0).all()


def test_no_stimulus_lands_inside_a_baseline_block():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    assert len(tl.stim_onsets) > 0
    assert (tl.kind_at(tl.stim_onsets) == TASK).all()


def test_baseline_duration_matches_config():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    n_baseline = int((tl.block_kind == BASELINE).sum())
    expected = 2 * int(round(TASK_CFG["baseline_duration_s"] * 10.0))
    assert n_baseline == expected


def test_effective_load_scales_by_practice_gain():
    tl_a = build_timeline(TASK_CFG, np.random.default_rng(0), practice_gain=1.0)
    tl_b = build_timeline(TASK_CFG, np.random.default_rng(0), practice_gain=0.5)
    t = tl_a.t
    assert np.allclose(tl_b.effective_load(t), 0.5 * tl_a.effective_load(t))


def test_effective_load_treats_baseline_as_rest_not_negative():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    base_t = tl.t[tl.block_kind == BASELINE]
    assert (tl.effective_load(base_t) == 0.0).all()


def test_session_idx_is_recorded():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0), session_idx=2)
    assert tl.session_idx == 2
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_state.py -v > task-2-testlog-red.txt 2>&1
```

Expected: `ImportError: cannot import name 'BASELINE'`.

- [ ] **Step 3: `state.py`를 구현한다**

`src/simulation/state.py` 전체를 아래로 교체:

```python
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
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_state.py -v > task-2-testlog-green.txt 2>&1
```

Expected: 새 테스트 8개 PASS. 기존 테스트 중 `build_timeline`의 반환 필드나 `duration_s`를 가정한 것이 있으면 **베이스라인이 추가된 새 사실에 맞춰 수정**한다 (기대값을 바꾸는 것이지 기능을 되돌리는 것이 아니다).

- [ ] **Step 5: 커밋**

```bash
git add src/simulation/state.py tests/simulation/test_state.py
git commit -m "feat: 타임라인에 베이스라인 블록·세션 번호·실효 부하 추가

세션 구조를 CLAUDE.md §2.4의 '베이스라인 → 과제 블록 × N → 베이스라인'으로
바꾼다. 시작 베이스라인은 정규화 기준 구간, 종료 베이스라인은 세션 내
드리프트 추정치다.

베이스라인 구간의 load_level은 센티넬 -1이다. 실제 수준과 겹치지 않아야
라벨 빌더가 골라낼 수 있고, effective_load가 0으로 클립해 안정 상태
구동을 만든다 — 그대로 비율로 바꾸면 음수 구동이 되어 신호가 뒤집힌다.

자극은 과제 블록 안에만 제시한다. 베이스라인에 자극이 들어가면 안정
상태가 아니게 되어 기준 구간이 오염된다."
```
---

## Task 3: 신호 성분이 실효 부하를 쓰도록 전환 (연습 효과 전달 경로)

**Files:**
- Modify: `src/simulation/components/eeg_oscillation.py:44`
- Modify: `src/simulation/components/eeg_erp.py:46`
- Modify: `src/simulation/components/fnirs_hrf.py:50`
- Modify: `src/simulation/components/behavior.py:47`
- Test: `tests/simulation/test_fnirs_hrf.py`, `tests/simulation/test_behavior.py`

**Interfaces:**
- Consumes: Task 2의 `CognitiveStateTimeline.effective_load(t)`
- Produces: 없음 (내부 배선 변경). 네 성분 모두 `load_fraction(timeline.load_at(...))` 호출을 버린다.

**왜 별도 태스크인가:** Task 2가 `practice_gain`을 타임라인에 실었지만, 성분들이 여전히 `load_fraction(load_at(t))`를 직접 부르면 **연습 효과가 신호에 도달하지 않는다.** 그러면 T4는 항상 보존율 0을 보게 되고, 원인이 정규화인지 배선인지 구분할 수 없다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/simulation/test_fnirs_hrf.py` 끝에 추가:

```python
from src.simulation.components.fnirs_hrf import generate_fnirs
from src.simulation.state import TASK, build_timeline
from src.simulation.subject import SubjectProfile

_TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _fnirs_task_mean(practice_gain: float) -> float:
    """같은 시드로 fNIRS를 만들고 과제 구간 HbO 평균을 돌려준다."""
    tl = build_timeline(
        _TASK_CFG, np.random.default_rng(7), practice_gain=practice_gain
    )
    hbo, _ = generate_fnirs(
        tl,
        SubjectProfile(subject_id="sub-01", theta=0.0),
        np.random.default_rng(11),
        sfreq=10.4,
        n_channels=4,
        effect_size=1.0,
        hbr_coupling=-0.33,
    )
    t = np.arange(hbo.shape[1]) / 10.4
    task_mask = tl.kind_at(t) == TASK
    return float(hbo[:, task_mask].mean())


def test_practice_gain_reaches_the_fnirs_signal():
    """연습 효과가 신호에 도달하지 않으면 T4는 원인 불명으로 실패한다."""
    full = _fnirs_task_mean(1.0)
    halved = _fnirs_task_mean(0.5)
    assert full > 0.05, "기준 조건에서 과제 구간 HbO가 양수여야 비교가 성립한다"
    # 잡음이 동일 시드로 같으므로 응답 성분만 절반이 된다
    assert halved < full * 0.75
```

`tests/simulation/test_behavior.py` 끝에 추가:

```python
from src.simulation.components.behavior import generate_behavior
from src.simulation.state import build_timeline
from src.simulation.subject import SubjectProfile

_TASK_CFG_B = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _mean_rt(practice_gain: float) -> float:
    tl = build_timeline(
        _TASK_CFG_B, np.random.default_rng(7), practice_gain=practice_gain
    )
    log = generate_behavior(
        tl,
        SubjectProfile(subject_id="sub-01", theta=0.0),
        np.random.default_rng(11),
        effect_size=1.0,
        lead_delta_s=1.2,
    )
    return float(log.rt.mean())


def test_practice_gain_reaches_behavior():
    """연습하면 같은 n-back에서도 반응시간이 줄어야 한다."""
    assert _mean_rt(0.5) < _mean_rt(1.0)
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_fnirs_hrf.py tests/simulation/test_behavior.py -v > task-3-testlog-red.txt 2>&1
```

Expected: `test_practice_gain_reaches_the_fnirs_signal` FAIL (halved == full — 성분이 `practice_gain`을 모른다), `test_practice_gain_reaches_behavior` FAIL 동일 이유.

- [ ] **Step 3: 네 성분의 호출을 바꾼다**

`src/simulation/components/eeg_oscillation.py` — import에서 `load_fraction`을 지우고 44행을 교체:

```python
    load = timeline.effective_load(t)
```

`src/simulation/components/eeg_erp.py` — import에서 `load_fraction`을 지우고 46행을 교체:

```python
    loads = timeline.effective_load(timeline.stim_onsets)
```

`src/simulation/components/fnirs_hrf.py` — import를 `from src.simulation.components.base import HBO_COUPLING`로 줄이고 50행을 교체:

```python
    neural = timeline.effective_load(t)
```

`src/simulation/components/behavior.py` — import를 `from src.simulation.components.base import BEHAV_COUPLING`로 줄이고 47행을 교체:

```python
    driving_load = timeline.effective_load(onsets - lead_delta_s)
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/simulation -v > task-3-testlog-green.txt 2>&1
```

Expected: 새 테스트 2개 PASS. 기존 성분 테스트가 `load_fraction`을 직접 import해 기대값을 만들었다면 `timeline.effective_load`로 바꾼다.

- [ ] **Step 5: 커밋**

```bash
git add src/simulation/components tests/simulation
git commit -m "feat: 신호 성분이 timeline.effective_load를 쓰도록 전환

Task 2가 practice_gain을 타임라인에 실었지만 성분들이 여전히
load_fraction(load_at(t))를 직접 불렀다. 그 상태로는 연습 효과가 신호에
도달하지 않고, T4가 항상 보존율 0을 보되 원인이 정규화인지 배선인지
구분되지 않는다.

effective_load는 베이스라인 센티넬을 0으로 클립하는 책임도 함께 진다."
```

---

## Task 4: 세션 계획 — 드리프트 파라미터 추출

**Files:**
- Create: `src/simulation/session.py`
- Test: `tests/simulation/test_session.py`

**Interfaces:**
- Consumes: `src.simulation.subject.SubjectProfile`, Task 1의 `simulation.drift.*` / `simulation.practice.rate` / `simulation.n_sessions`
- Produces:
  - `SessionDriftParams(fnirs_gain, fnirs_offset, eeg_gain, eeg_noise_scale, within_rate, between_big, within_big)` — 앞 넷은 `np.ndarray`, `within_rate`는 `float`, 뒤 둘은 `bool`
  - `SessionPlan(subject: SubjectProfile, session_idx: int, practice_gain: float, drift: SessionDriftParams)`
  - `plan_sessions(subject, sim_cfg: dict, rng) -> list[SessionPlan]`
  - `EEG_NOISE_BASE: float = 0.1`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/simulation/test_session.py` (신규):

```python
import numpy as np
import pytest

from src.simulation.session import plan_sessions
from src.simulation.subject import SubjectProfile

SUBJ = SubjectProfile(subject_id="sub-01", theta=0.0)


def make_cfg(**drift_over):
    drift = {
        "fnirs_gain_sigma": 0.20,
        "fnirs_offset_sigma": 0.10,
        "eeg_gain_sigma": 0.15,
        "eeg_noise_sigma": 0.15,
        "within_session_rate": 0.30,
        "within_session_fraction": 0.33,
        "between_session_scale": 4.0,
        "assignment": "sampled",
    }
    drift.update(drift_over)
    return {
        "n_sessions": 3,
        "practice": {"rate": 0.15},
        "drift": drift,
        "eeg": {"n_channels": 8, "sfreq_hz": 250},
        "fnirs": {"n_channels": 6, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
    }


def test_returns_one_plan_per_session_in_order():
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert [p.session_idx for p in plans] == [0, 1, 2]


def test_practice_gain_decays_geometrically_and_stays_positive():
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert plans[0].practice_gain == pytest.approx(1.0)
    assert plans[1].practice_gain == pytest.approx(0.85)
    assert plans[2].practice_gain == pytest.approx(0.85 ** 2)


def test_practice_gain_never_goes_negative_for_many_sessions():
    """선형 감소였다면 세션 7에서 음수가 되어 인지부하가 뒤집힌다."""
    cfg = make_cfg()
    cfg["n_sessions"] = 20
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(0))
    assert all(p.practice_gain > 0 for p in plans)


def test_session_zero_also_carries_drift():
    """'무드리프트 기준 세션'을 두면 정규화 테스트가 부당하게 쉬워진다."""
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert not np.allclose(plans[0].drift.fnirs_gain, 1.0)


def test_sessions_differ_from_each_other():
    plans = plan_sessions(SUBJ, make_cfg(), np.random.default_rng(0))
    assert not np.allclose(plans[0].drift.fnirs_gain, plans[1].drift.fnirs_gain)


def test_zero_sigma_consumes_the_same_rng_state():
    """분기하면 '드리프트 없음' 조건이 다른 난수열 위에서 돈다.

    그러면 T2의 결함 주입('드리프트를 끄면 붕괴가 사라진다')이 단일 변수
    조작이 아니게 되고, T4의 b_ref도 b_hat의 정당한 기준이 되지 못한다.
    """
    off = dict(
        fnirs_gain_sigma=0.0, fnirs_offset_sigma=0.0,
        eeg_gain_sigma=0.0, eeg_noise_sigma=0.0,
        within_session_rate=0.0,
    )
    rng_a = np.random.default_rng(3)
    plan_sessions(SUBJ, make_cfg(), rng_a)
    rng_b = np.random.default_rng(3)
    plan_sessions(SUBJ, make_cfg(**off), rng_b)
    assert rng_a.random() == rng_b.random()


def test_zero_sigma_really_removes_drift():
    off = dict(
        fnirs_gain_sigma=0.0, fnirs_offset_sigma=0.0,
        eeg_gain_sigma=0.0, eeg_noise_sigma=0.0,
        within_session_rate=0.0,
    )
    plans = plan_sessions(SUBJ, make_cfg(**off), np.random.default_rng(3))
    for p in plans:
        assert np.allclose(p.drift.fnirs_gain, 1.0)
        assert np.allclose(p.drift.fnirs_offset, 0.0)
        assert np.allclose(p.drift.eeg_gain, 1.0)
        assert p.drift.within_rate == 0.0


def test_fixed_2x2_assignment_covers_all_four_cells():
    cfg = make_cfg(assignment="fixed_2x2")
    cfg["n_sessions"] = 4
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(0))
    cells = {(p.drift.between_big, p.drift.within_big) for p in plans}
    assert cells == {(False, False), (False, True), (True, False), (True, True)}


def test_fixed_2x2_session_two_is_the_critical_negative_cell():
    """①② 큼 + ③ 작음 — 여기가 플래그되면 drift_flag가 둘을 혼동한 것이다."""
    cfg = make_cfg(assignment="fixed_2x2")
    cfg["n_sessions"] = 4
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(0))
    assert plans[2].drift.between_big is True
    assert plans[2].drift.within_big is False
    assert plans[2].drift.within_rate == 0.0


def test_between_big_actually_widens_the_gain_spread():
    cfg = make_cfg(assignment="fixed_2x2")
    cfg["n_sessions"] = 4
    plans = plan_sessions(SUBJ, cfg, np.random.default_rng(5))
    small = np.abs(np.log(plans[0].drift.fnirs_gain)).mean()
    big = np.abs(np.log(plans[2].drift.fnirs_gain)).mean()
    assert big > small * 2


def test_unknown_assignment_is_rejected():
    with pytest.raises(ValueError, match="unknown drift assignment"):
        plan_sessions(SUBJ, make_cfg(assignment="whatever"), np.random.default_rng(0))
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_session.py -v > task-4-testlog-red.txt 2>&1
```

Expected: `ModuleNotFoundError: No module named 'src.simulation.session'`.

- [ ] **Step 3: `session.py`를 구현한다**

`src/simulation/session.py` (신규):

```python
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
```

> `sampled` 모드에서 `between_big`이 항상 False인 이유: ①②의 크기는 `sigma`가 이미 세션마다 다른 값을 뽑아 만든다. `between_scale`은 T6이 2×2를 **결정적으로** 배치하기 위한 손잡이이므로 일반 실행에서는 쓰지 않는다.

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_session.py -v > task-4-testlog-green.txt 2>&1
```

Expected: 11개 PASS.

- [ ] **Step 5: 결함 주입으로 난수열 테스트가 실제로 잡는지 확인한다**

`session.py`에 일시적으로 분기를 넣는다:

```python
        z_fg = rng.normal(size=n_fnirs) if sigma_fg > 0 else np.zeros(n_fnirs)
```

```bash
python -m pytest tests/simulation/test_session.py::test_zero_sigma_consumes_the_same_rng_state -v > task-4-testlog-inject.txt 2>&1
```

Expected: **FAIL**. 확인 후 분기를 되돌린다. 통과해버리면 테스트가 무의미하므로 테스트를 고친다.

- [ ] **Step 6: 커밋**

```bash
git add src/simulation/session.py tests/simulation/test_session.py
git commit -m "feat: 세션별 드리프트 파라미터와 연습 효과 추출

세션 0도 자기 드리프트를 갖는다 — '무드리프트 기준 세션'을 두면 실제와
다르고 정규화 테스트가 부당하게 쉬워진다.

연습 효과는 기하 감쇠 (1-rate)**s 다. 선형 감소는 n_sessions가 커지면
음수가 되어 인지부하가 뒤집히는데, n_sessions가 config인 이상 그 조합이
실제로 들어올 수 있다.

난수 소모 순서는 조건과 무관하게 고정한다. 분기하면 '드리프트 없음'
조건이 다른 난수열 위에서 돌아 T2 결함 주입과 T4 기준 실행이 단일 변수
조작이 아니게 된다."
```

---

## Task 5: 드리프트 적용

**Files:**
- Create: `src/simulation/components/drift.py`
- Test: `tests/simulation/test_drift.py`

**Interfaces:**
- Consumes: Task 4의 `SessionDriftParams` 필드들
- Produces:
  - `apply_channel_drift(x: np.ndarray, *, gain: np.ndarray, offset: np.ndarray, within_rate: float) -> np.ndarray`
  - `apply_noise_scaling(x: np.ndarray, *, noise_scale: np.ndarray, rng) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/simulation/test_drift.py` (신규):

```python
import numpy as np
import pytest

from src.simulation.components.drift import apply_channel_drift, apply_noise_scaling


def test_multiplicative_gain_is_applied_per_channel():
    x = np.ones((3, 100))
    out = apply_channel_drift(
        x, gain=np.array([1.0, 2.0, 0.5]), offset=np.zeros(3), within_rate=0.0
    )
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 2.0)
    assert np.allclose(out[2], 0.5)


def test_additive_offset_is_applied_per_channel():
    x = np.zeros((2, 50))
    out = apply_channel_drift(
        x, gain=np.ones(2), offset=np.array([1.5, -0.5]), within_rate=0.0
    )
    assert np.allclose(out[0], 1.5)
    assert np.allclose(out[1], -0.5)


def test_within_session_drift_ramps_the_gain_over_time():
    x = np.ones((1, 101))
    out = apply_channel_drift(
        x, gain=np.ones(1), offset=np.zeros(1), within_rate=0.5
    )
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, -1] == pytest.approx(1.5)
    assert out[0, 50] == pytest.approx(1.25)


def test_zero_within_rate_leaves_the_signal_flat_in_time():
    x = np.ones((1, 101))
    out = apply_channel_drift(
        x, gain=np.full(1, 2.0), offset=np.zeros(1), within_rate=0.0
    )
    assert np.allclose(out, 2.0)


def test_offset_is_not_ramped():
    """오프셋은 DC이므로 세션 내 이동의 대상이 아니다 — 이득만 이동한다."""
    x = np.zeros((1, 101))
    out = apply_channel_drift(
        x, gain=np.ones(1), offset=np.full(1, 3.0), within_rate=0.5
    )
    assert np.allclose(out, 3.0)


def test_shape_is_preserved():
    x = np.random.default_rng(0).normal(size=(4, 77))
    out = apply_channel_drift(
        x, gain=np.ones(4), offset=np.zeros(4), within_rate=0.2
    )
    assert out.shape == x.shape


def test_gain_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="gain length"):
        apply_channel_drift(
            np.ones((3, 10)), gain=np.ones(2), offset=np.zeros(3), within_rate=0.0
        )


def test_noise_scaling_raises_per_channel_variance():
    x = np.zeros((2, 4000))
    out = apply_noise_scaling(
        x, noise_scale=np.array([0.1, 1.0]), rng=np.random.default_rng(0)
    )
    assert out[0].std() == pytest.approx(0.1, rel=0.1)
    assert out[1].std() == pytest.approx(1.0, rel=0.1)


def test_noise_scaling_does_not_shift_the_mean():
    x = np.zeros((1, 8000))
    out = apply_noise_scaling(
        x, noise_scale=np.array([1.0]), rng=np.random.default_rng(0)
    )
    assert abs(out.mean()) < 0.05
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_drift.py -v > task-5-testlog-red.txt 2>&1
```

Expected: `ModuleNotFoundError: No module named 'src.simulation.components.drift'`.

- [ ] **Step 3: `drift.py`를 구현한다**

`src/simulation/components/drift.py` (신규):

```python
"""드리프트 적용 — 깨끗한 신호에 측정 왜곡을 얹는다.

기존 성분 플러그인은 손대지 않는다. 그들은 계속 '깨끗한 신호'를 만들고,
드리프트는 그 뒤에 곱해진다. 이 분리가 스펙 §5.1의 층 구분을 코드로
지키는 방식이다 — 여기서 하는 일은 전부 신호 변환이며 인지상태는
건드리지 않는다.

곱셈 이득이 핵심이다. 광 결합도는 곱셈적이라, 가산 오프셋만 심으면
정규화 난이도가 비현실적으로 쉬워진다(평균 빼기로 끝난다).
"""

from __future__ import annotations

import numpy as np


def apply_channel_drift(
    x: np.ndarray,
    *,
    gain: np.ndarray,
    offset: np.ndarray,
    within_rate: float,
) -> np.ndarray:
    """`y = gain(t) * x + offset`, `gain(t) = gain * (1 + within_rate * t/T)`.

    오프셋은 램프의 대상이 아니다. 세션 내 드리프트는 옵토드가 서서히
    밀리며 결합도가 변하는 현상이므로 곱셈 항에만 걸린다.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_channels, n_samples), got shape {x.shape}")
    n_ch, n_samples = x.shape
    if len(gain) != n_ch:
        raise ValueError(f"gain length {len(gain)} != n_channels {n_ch}")
    if len(offset) != n_ch:
        raise ValueError(f"offset length {len(offset)} != n_channels {n_ch}")

    ramp = 1.0 + within_rate * (np.arange(n_samples) / max(n_samples - 1, 1))
    return x * np.asarray(gain)[:, None] * ramp[None, :] + np.asarray(offset)[:, None]


def apply_noise_scaling(
    x: np.ndarray,
    *,
    noise_scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """채널별 잡음을 더한다 — 임피던스 상승 모사.

    임피던스가 오르면 진폭이 아니라 **잡음**이 커진다. 그래서 이득과
    분리해 가산으로 얹는다.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_channels, n_samples), got shape {x.shape}")
    if len(noise_scale) != x.shape[0]:
        raise ValueError(
            f"noise_scale length {len(noise_scale)} != n_channels {x.shape[0]}"
        )
    noise = rng.normal(0.0, 1.0, size=x.shape) * np.asarray(noise_scale)[:, None]
    return x + noise
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_drift.py -v > task-5-testlog-green.txt 2>&1
```

Expected: 9개 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/simulation/components/drift.py tests/simulation/test_drift.py
git commit -m "feat: 채널별 드리프트·잡음 스케일 적용

기존 성분 플러그인은 손대지 않는다. 깨끗한 신호를 만드는 책임과 측정
왜곡을 얹는 책임을 분리해야 스펙 §5.1의 층 구분이 코드에서 유지된다.

곱셈 이득이 핵심이다. 광 결합도는 곱셈적이라 가산 오프셋만 심으면
정규화가 평균 빼기로 끝나는 비현실적으로 쉬운 문제가 된다.

오프셋은 램프하지 않는다. 세션 내 드리프트는 결합도 변화이므로 곱셈
항에만 걸린다."
```

---

## Task 6: 녹화를 피험자 × 세션으로

**Files:**
- Modify: `src/simulation/recording.py`
- Test: `tests/simulation/test_recording.py`

**Interfaces:**
- Consumes: Task 2 `build_timeline(..., session_idx=, practice_gain=)`, Task 4 `plan_sessions`, Task 5 `apply_channel_drift`/`apply_noise_scaling`
- Produces:
  - `SyntheticRecording` 필드 추가: `session_idx: int`
  - `generate_recording(plan: SessionPlan, sim_cfg: dict, rng) -> SyntheticRecording` (**시그니처 변경**: 첫 인자가 `SubjectProfile`에서 `SessionPlan`으로)
  - `generate_dataset(sim_cfg, rng) -> list[SyntheticRecording]` — 길이 `n_subjects * n_sessions`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/simulation/test_recording.py` 끝에 추가:

```python
from src.simulation.recording import generate_dataset

SIM_CFG = {
    "n_subjects": 3,
    "n_sessions": 2,
    "subject_variance": 0.5,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "baseline_duration_s": 20,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "practice": {"rate": 0.15},
    "drift": {
        "fnirs_gain_sigma": 0.20,
        "fnirs_offset_sigma": 0.10,
        "eeg_gain_sigma": 0.15,
        "eeg_noise_sigma": 0.15,
        "within_session_rate": 0.30,
        "within_session_fraction": 0.33,
        "between_session_scale": 4.0,
        "assignment": "sampled",
    },
    "eeg": {"n_channels": 8, "sfreq_hz": 250},
    "fnirs": {"n_channels": 6, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def _no_drift_cfg():
    cfg = copy.deepcopy(SIM_CFG)
    cfg["drift"].update(
        fnirs_gain_sigma=0.0, fnirs_offset_sigma=0.0,
        eeg_gain_sigma=0.0, eeg_noise_sigma=0.0, within_session_rate=0.0,
    )
    return cfg


def test_dataset_has_one_recording_per_subject_session():
    recs = generate_dataset(SIM_CFG, np.random.default_rng(0))
    assert len(recs) == SIM_CFG["n_subjects"] * SIM_CFG["n_sessions"]
    pairs = [(r.subject_id, r.session_idx) for r in recs]
    assert len(set(pairs)) == len(pairs)
    assert set(r.session_idx for r in recs) == {0, 1}


def test_same_subject_different_sessions_have_different_channel_levels():
    """세션 간 베이스라인 이동이 실제로 생겨야 정규화를 시험할 수 있다."""
    recs = generate_dataset(SIM_CFG, np.random.default_rng(0))
    s0 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 0)
    s1 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 1)
    assert not np.allclose(s0.hbo.mean(axis=1), s1.hbo.mean(axis=1), atol=1e-6)


def test_without_drift_channel_levels_stay_comparable():
    recs = generate_dataset(_no_drift_cfg(), np.random.default_rng(0))
    s0 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 0)
    s1 = next(r for r in recs if r.subject_id == "sub-01" and r.session_idx == 1)
    spread_with = float(np.abs(s0.hbo.mean(axis=1) - s1.hbo.mean(axis=1)).mean())

    recs_d = generate_dataset(SIM_CFG, np.random.default_rng(0))
    d0 = next(r for r in recs_d if r.subject_id == "sub-01" and r.session_idx == 0)
    d1 = next(r for r in recs_d if r.subject_id == "sub-01" and r.session_idx == 1)
    spread_drift = float(np.abs(d0.hbo.mean(axis=1) - d1.hbo.mean(axis=1)).mean())

    assert spread_drift > spread_with * 3


def test_behavior_is_untouched_by_drift():
    """행동은 장비 재부착의 영향을 받지 않는다 (CLAUDE.md §3.8)."""
    a = generate_dataset(SIM_CFG, np.random.default_rng(0))[0]
    b = generate_dataset(_no_drift_cfg(), np.random.default_rng(0))[0]
    assert np.array_equal(a.behavior.correct, b.behavior.correct)
    assert np.allclose(a.behavior.rt, b.behavior.rt)


def test_recording_timeline_knows_its_session():
    recs = generate_dataset(SIM_CFG, np.random.default_rng(0))
    for r in recs:
        assert r.timeline.session_idx == r.session_idx
```

파일 상단 import에 `import copy`를 추가한다.

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/simulation/test_recording.py -v > task-6-testlog-red.txt 2>&1
```

Expected: `AttributeError: 'SyntheticRecording' object has no attribute 'session_idx'` 또는 `KeyError: 'n_sessions'`.

- [ ] **Step 3: `recording.py`를 고친다**

import에 추가:

```python
from src.simulation.components.drift import apply_channel_drift, apply_noise_scaling
from src.simulation.session import SessionPlan, plan_sessions
```

`SyntheticRecording`에 필드 추가 (`subject_id` 바로 아래):

```python
    session_idx: int
```

`generate_recording`을 교체:

```python
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
```

`generate_dataset`을 교체:

```python
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
```

`write_bids_metadata`의 participants 줄에 세션 수를 반영:

```python
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
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/simulation -v > task-6-testlog-green.txt 2>&1
```

Expected: 새 테스트 5개 PASS. 기존 `generate_recording(subject, ...)` 호출을 쓰던 테스트는 `SessionPlan`을 넘기도록 고친다.

- [ ] **Step 5: 커밋**

```bash
git add src/simulation/recording.py tests/simulation/test_recording.py
git commit -m "feat: 녹화를 피험자 × 세션으로 확장하고 드리프트를 얹는다

HbO와 HbR은 같은 광경로에서 나오므로 동일한 이득·오프셋을 받는다.
다른 값을 주면 실제와 다르고 정규화가 더 쉬워진다.

EEG는 잡음 스케일을 이득보다 먼저 얹는다 — 그래야 세션 내 드리프트
램프가 잡음에도 함께 걸린다.

행동에는 드리프트를 적용하지 않는다. 정오답률·반응시간은 장비 재부착의
영향을 받지 않으므로 절대값을 유지한다 (CLAUDE.md §3.8)."
```
---

## Task 7: 윈도잉에 블록 종류 전파

**Files:**
- Modify: `src/datasets/windowing.py`
- Test: `tests/datasets/test_windowing.py`

**Interfaces:**
- Consumes: Task 2의 `timeline.kind_at(t)`, `timeline.block_kind`
- Produces: `WindowIndex` 필드 추가 `block_kind: np.ndarray` (창별 `BASELINE`/`TASK`)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/datasets/test_windowing.py` 끝에 추가:

```python
from src.datasets.windowing import make_windows
from src.simulation.state import BASELINE, TASK, build_timeline

_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def test_windows_carry_block_kind():
    tl = build_timeline(_CFG, np.random.default_rng(0))
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    assert len(w.block_kind) == len(w.start_s)
    assert set(np.unique(w.block_kind)) == {BASELINE, TASK}


def test_baseline_windows_lie_entirely_inside_baseline_blocks():
    tl = build_timeline(_CFG, np.random.default_rng(0))
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    base = w.block_kind == BASELINE
    assert base.any()
    assert (tl.kind_at(w.start_s[base]) == BASELINE).all()
    assert (tl.kind_at(w.end_s[base] - 1e-6) == BASELINE).all()


def test_both_baseline_blocks_produce_windows():
    """시작·종료 베이스라인 둘 다 창을 내야 드리프트를 잴 수 있다."""
    tl = build_timeline(_CFG, np.random.default_rng(0))
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    base_trials = np.unique(w.trial_id[w.block_kind == BASELINE])
    assert len(base_trials) == 2


def test_baseline_shorter_than_window_is_rejected():
    cfg = dict(_CFG, baseline_duration_s=3)
    tl = build_timeline(cfg, np.random.default_rng(0))
    with pytest.raises(ValueError, match="baseline block"):
        make_windows(tl, window_s=5.0, step_s=1.0)
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/datasets/test_windowing.py -v > task-7-testlog-red.txt 2>&1
```

Expected: `AttributeError: 'WindowIndex' object has no attribute 'block_kind'`.

- [ ] **Step 3: `windowing.py`를 고친다**

`WindowIndex`에 필드 추가:

```python
@dataclass(frozen=True)
class WindowIndex:
    start_s: np.ndarray     # (n_win,)
    end_s: np.ndarray       # (n_win,)
    trial_id: np.ndarray    # (n_win,) int
    load_level: np.ndarray  # (n_win,) int — 베이스라인 창은 센티넬 -1
    block_kind: np.ndarray  # (n_win,) int — BASELINE / TASK
```

`make_windows`의 기존 검증 뒤에 베이스라인 길이 검증을 추가:

```python
    from src.simulation.state import BASELINE  # 순환 import 회피를 위해 지역 import

    baseline_lengths = [
        int((timeline.trial_id == t).sum()) / 10.0
        for t in np.unique(timeline.trial_id[timeline.block_kind == BASELINE])
    ]
    if baseline_lengths and min(baseline_lengths) < window_s:
        raise ValueError(
            f"baseline block is {min(baseline_lengths)}s but window_s is {window_s}s; "
            "베이스라인에서 창이 하나도 나오지 않으면 정규화 기준 구간을 만들 수 없다"
        )
```

> `10.0`은 `state.STATE_SFREQ`다. 지역 import로 `from src.simulation.state import BASELINE, STATE_SFREQ`를 함께 가져와 상수 대신 쓴다.

반환문에 `block_kind`를 추가:

```python
    return WindowIndex(
        start_s=starts,
        end_s=ends,
        trial_id=timeline.trial_at(starts),
        load_level=timeline.load_at(starts),
        block_kind=timeline.kind_at(starts),
    )
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/datasets/test_windowing.py -v > task-7-testlog-green.txt 2>&1
```

Expected: 새 테스트 4개 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/datasets/windowing.py tests/datasets/test_windowing.py
git commit -m "feat: 윈도우에 블록 종류(BASELINE/TASK) 전파

베이스라인 창을 골라낼 수 없으면 정규화 기준 구간도, 드리프트 지표도
만들 수 없다.

베이스라인 블록이 창보다 짧으면 즉시 거부한다 — 조용히 창 0개를
내놓으면 정규화가 빈 배열로 fit되어 한참 뒤에 무의미한 값으로 죽는다."
```

---

## Task 8: 라벨 빌더의 베이스라인 배제 (타깃별 옵트인)

**Files:**
- Modify: `src/datasets/labels.py`
- Test: `tests/datasets/test_labels.py`

**Interfaces:**
- Consumes: Task 7의 `windows.block_kind`
- Produces: `build_labels(rec, windows, *, lead_delta_s, rt_bins, lead_targets=LEAD_TARGETS, include_baseline: bool = False)` — `include_baseline=False`면 `keep`에서 베이스라인 창을 뺀다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/datasets/test_labels.py` 끝에 추가:

```python
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.simulation.recording import generate_dataset
from src.simulation.state import BASELINE, BASELINE_LOAD_SENTINEL, TASK

_SIM = {
    "n_subjects": 1,
    "n_sessions": 1,
    "subject_variance": 0.0,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "baseline_duration_s": 20,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "practice": {"rate": 0.15},
    "drift": {
        "fnirs_gain_sigma": 0.2, "fnirs_offset_sigma": 0.1,
        "eeg_gain_sigma": 0.15, "eeg_noise_sigma": 0.15,
        "within_session_rate": 0.3, "within_session_fraction": 0.33,
        "between_session_scale": 4.0, "assignment": "sampled",
    },
    "eeg": {"n_channels": 4, "sfreq_hz": 250},
    "fnirs": {"n_channels": 4, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def _rec_and_windows():
    rec = generate_dataset(_SIM, np.random.default_rng(0))[0]
    win = make_windows(rec.timeline, window_s=5.0, step_s=1.0)
    return rec, win


def test_baseline_windows_are_excluded_by_default():
    rec, win = _rec_and_windows()
    _, keep = build_labels(rec, win, lead_delta_s=1.2, rt_bins=[0.5, 0.8])
    assert (win.block_kind[keep] == TASK).all()
    assert (win.block_kind == BASELINE).any(), "이 설정에서 베이스라인 창이 있어야 시험이 성립한다"


def test_sentinel_label_never_survives_the_default_path():
    """센티넬이 살아남으면 chance level이 3클래스가 아니게 된다."""
    rec, win = _rec_and_windows()
    labels, keep = build_labels(rec, win, lead_delta_s=1.2, rt_bins=[0.5, 0.8])
    assert (labels["cognitive_load"][keep] != BASELINE_LOAD_SENTINEL).all()


def test_include_baseline_opt_in_keeps_them():
    rec, win = _rec_and_windows()
    _, keep_off = build_labels(rec, win, lead_delta_s=1.2, rt_bins=[0.5, 0.8])
    _, keep_on = build_labels(
        rec, win, lead_delta_s=1.2, rt_bins=[0.5, 0.8], include_baseline=True
    )
    assert int(keep_on.sum()) > int(keep_off.sum())
    assert (win.block_kind[keep_on] == BASELINE).any()
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/datasets/test_labels.py -v > task-8-testlog-red.txt 2>&1
```

Expected: `test_baseline_windows_are_excluded_by_default` FAIL (베이스라인 창이 `keep`에 남아 있다).

- [ ] **Step 3: `labels.py`를 고친다**

`build_labels` 시그니처에 `include_baseline: bool = False`를 추가하고, 반환 직전에 배제를 넣는다:

```python
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
        from src.simulation.state import TASK

        keep = keep & (windows.block_kind == TASK)

    return labels, keep
```

`build_labels`의 docstring에 한 문단 추가:

```
    `include_baseline`이 True면 베이스라인 창이 남는다. 그 경우 정규화
    기준 구간이 분석 데이터에도 들어가므로, 호출자는 그 타깃에 쓸 정규화
    기준을 따로 정해야 한다 — 정하지 않으면 Task 12의 정규화 출처 가드가
    fold 실행 시점에 실패시킨다.
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/datasets/test_labels.py -v > task-8-testlog-green.txt 2>&1
```

Expected: 새 테스트 3개 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/datasets/labels.py tests/datasets/test_labels.py
git commit -m "feat: 부하 타깃에서 베이스라인 창 배제 (타깃별 옵트인)

삭제가 아니라 배제다. 베이스라인은 피로를 가장 깨끗하게 측정하는
구간이므로 버리지 않고 태깅해 남기고, 부하 타깃의 라벨 빌더만 뺀다.

배제 이유는 라벨 부재가 아니라 (1) 자기 자신으로 정규화한 표본의 퇴화
(2) chance level이 조용히 바뀜 (3) 드리프트 지표의 순환이다."
```

---

## Task 9: `SessionBaseline` — 세션 베이스라인 정규화

**Files:**
- Create: `src/preprocessing/__init__.py` (빈 파일)
- Create: `src/preprocessing/baseline.py`
- Test: `tests/preprocessing/test_baseline.py` (`tests/preprocessing/__init__.py`는 만들지 않는다 — 기존 테스트 디렉토리도 패키지가 아니다)

**Interfaces:**
- Consumes: 없음 (순수 numpy)
- Produces:
  - `NORMALIZATION_KINDS: tuple[str, ...]`, `MODALITY_KIND: dict[str, str]`
  - `SessionBaseline.fit(start_baseline: np.ndarray, kind: str) -> SessionBaseline`
  - `SessionBaseline.apply(x: np.ndarray) -> np.ndarray`
  - `SessionBaseline.kind: str`, `SessionBaseline.reference: np.ndarray`
  - `SessionDrift(per_channel: np.ndarray, aggregate: float, n_excluded: int)`
  - `compute_drift(start_baseline: np.ndarray, end_baseline: np.ndarray, *, zero_atol: float) -> SessionDrift`
  - `SessionQuality(subject_id: str, session_idx: int, drift_by_modality: dict[str, float], n_excluded_by_modality: dict[str, int], drift_flag: bool)`
  - `flag_drift(aggregates: dict[str, float], threshold: float) -> bool`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/preprocessing/test_baseline.py` (신규):

```python
import inspect

import numpy as np
import pytest

from src.preprocessing.baseline import (
    MODALITY_KIND,
    NORMALIZATION_KINDS,
    SessionBaseline,
    compute_drift,
    flag_drift,
)


def test_fit_signature_cannot_receive_another_session():
    """경계를 규율이 아니라 서명으로 막는다 (스펙 §6.1).

    다른 세션·다른 피험자를 넘길 인자가 존재하지 않아야 한다.
    """
    params = list(inspect.signature(SessionBaseline.fit).parameters)
    assert params == ["start_baseline", "kind"]


def test_concentration_delta_subtracts_the_reference():
    base = np.array([[1.0, 2.0], [3.0, 4.0]])          # 평균 (2.0, 3.0)
    sb = SessionBaseline.fit(base, "concentration_delta")
    out = sb.apply(np.array([[2.0, 3.0], [5.0, 3.0]]))
    assert np.allclose(out, [[0.0, 0.0], [3.0, 0.0]])


def test_band_power_db_is_zero_at_the_reference_and_3db_when_doubled():
    base = np.array([[4.0, 9.0]])
    sb = SessionBaseline.fit(base, "band_power_db")
    assert np.allclose(sb.apply(np.array([[4.0, 9.0]])), 0.0)
    assert np.allclose(sb.apply(np.array([[8.0, 18.0]])), 10 * np.log10(2.0))


def test_absolute_leaves_the_values_alone():
    base = np.array([[0.9, 0.5]])
    sb = SessionBaseline.fit(base, "absolute")
    x = np.array([[0.7, 0.4], [0.95, 0.6]])
    assert np.allclose(sb.apply(x), x)


def test_absolute_returns_a_copy_not_the_same_object():
    sb = SessionBaseline.fit(np.array([[1.0]]), "absolute")
    x = np.array([[2.0]])
    out = sb.apply(x)
    out[0, 0] = 99.0
    assert x[0, 0] == 2.0


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown normalization kind"):
        SessionBaseline.fit(np.array([[1.0]]), "zscore")


def test_empty_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline block produced no windows"):
        SessionBaseline.fit(np.empty((0, 3)), "concentration_delta")


def test_modality_kind_covers_the_three_modalities():
    assert set(MODALITY_KIND) == {"eeg", "fnirs", "behavior"}
    assert set(MODALITY_KIND.values()) <= set(NORMALIZATION_KINDS)


def test_drift_is_a_relative_ratio():
    start = np.array([[10.0, 100.0]])
    end = np.array([[11.0, 100.0]])
    d = compute_drift(start, end, zero_atol=1e-8)
    assert np.allclose(d.per_channel, [0.1, 0.0])
    assert d.aggregate == pytest.approx(0.05)
    assert d.n_excluded == 0


def test_near_zero_baseline_channels_are_excluded_and_counted():
    start = np.array([[10.0, 0.0]])
    end = np.array([[11.0, 5.0]])
    d = compute_drift(start, end, zero_atol=1e-8)
    assert d.n_excluded == 1
    assert np.isnan(d.per_channel[1])
    assert d.aggregate == pytest.approx(0.1)


def test_all_channels_excluded_yields_nan_not_a_quiet_zero():
    start = np.zeros((1, 2))
    end = np.ones((1, 2))
    d = compute_drift(start, end, zero_atol=1e-8)
    assert d.n_excluded == 2
    assert np.isnan(d.aggregate)


def test_nan_drift_counts_as_flagged():
    """평가 불가는 '문제 없음'이 아니다. 조용히 통과시키면 안 된다."""
    assert flag_drift({"fnirs": float("nan")}, threshold=0.2) is True


def test_flag_fires_only_above_threshold():
    assert flag_drift({"fnirs": 0.1, "eeg": 0.05}, threshold=0.2) is False
    assert flag_drift({"fnirs": 0.1, "eeg": 0.31}, threshold=0.2) is True
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/preprocessing/test_baseline.py -v > task-9-testlog-red.txt 2>&1
```

Expected: `ModuleNotFoundError: No module named 'src.preprocessing'`.

- [ ] **Step 3: `baseline.py`를 구현한다**

`src/preprocessing/__init__.py`는 빈 파일로 만든다.

`src/preprocessing/baseline.py` (신규):

```python
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
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/preprocessing/test_baseline.py -v > task-9-testlog-green.txt 2>&1
```

Expected: 13개 PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/preprocessing tests/preprocessing
git commit -m "feat: SessionBaseline — 세션 베이스라인 정규화

fit은 한 세션의 시작 베이스라인만 받는다. 다른 세션을 넘길 인자가 아예
없다 — TestView.fit()이 무조건 예외를 던지는 것과 같은 발상이고, 규율이
아니라 서명으로 막는다.

드리프트는 별도 함수다. 두 베이스라인을 다 봐야 하는데 fit이 시작
베이스라인만 받는다는 요점을 깨뜨릴 수 없기 때문이다.

드리프트는 상대 비율이다. 절대값은 모달리티·단위마다 스케일이 달라
하나의 임계로 표현할 수 없다. 베이스라인이 0에 가까운 채널은 비율이
발산하므로 제외하되 개수를 함께 보고한다 — 조용히 버리면 '드리프트
없음'으로 오독된다. nan 집계는 플래그로 취급한다."
```

---

## Task 10: 계약 확장 — `session_ids` · `SplitKeys` · 정규화 출처 표식

**Files:**
- Modify: `src/datasets/contract.py`
- Test: `tests/datasets/test_contract.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `SplitKeys(subject_ids, session_ids, trial_ids, labels)` — frozen dataclass, 전부 `np.ndarray` (`labels`는 `np.ndarray | None`)
  - `Splitter.split(keys: SplitKeys) -> Iterator[tuple[np.ndarray, np.ndarray]]` (**프로토콜 변경**)
  - `Splitter.allows_same_session: ClassVar[bool]`
  - `WindowedDataset(X, y, subject_ids, session_ids, window_times, trial_ids, normalization_source)` (**시그니처 변경**, 전부 필수)
  - `WindowedDataset.get_session_ids() -> np.ndarray`
  - `_View.session_ids() -> np.ndarray`, `_View.normalization_source() -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/datasets/test_contract.py` 끝에 추가:

```python
from src.datasets.contract import SplitKeys, WindowedDataset


def _make_dataset(
    subjects=("a", "a", "b", "b"),
    sessions=(0, 1, 0, 1),
    trials=(0, 1, 2, 3),
    norm_src=(False, False, False, False),
):
    n = len(subjects)
    return WindowedDataset(
        X={"m": np.arange(n * 2, dtype=float).reshape(n, 2)},
        y={"t": np.array([0, 1, 0, 1])},
        subject_ids=np.array(subjects),
        session_ids=np.array(sessions),
        window_times=np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) + 5]),
        trial_ids=np.array(trials),
        normalization_source=np.array(norm_src),
    )


def test_dataset_exposes_session_ids():
    ds = _make_dataset()
    assert ds.get_session_ids().tolist() == [0, 1, 0, 1]


def test_trial_id_shared_across_two_sessions_is_rejected():
    """trial_id가 녹화마다 0부터 세면 세션 간 충돌한다.

    충돌하면 WithinSubjectSplitter가 두 세션의 블록을 한 덩어리로 묶어
    시행이 더 이상 일관된 단위가 아니게 된다.
    """
    with pytest.raises(ValueError, match="trial_id .* appears under"):
        _make_dataset(trials=(0, 0, 1, 1))


def test_trial_id_shared_across_two_subjects_is_rejected():
    with pytest.raises(ValueError, match="trial_id .* appears under"):
        _make_dataset(subjects=("a", "a", "b", "b"), sessions=(0, 0, 0, 0), trials=(0, 1, 0, 1))


def test_views_expose_session_ids_and_normalization_source():
    ds = _make_dataset(norm_src=(True, False, False, False))

    class _AllSplit:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            yield np.array([0, 1]), np.array([2, 3])

    fold = next(ds.iter_folds(_AllSplit()))
    assert fold.train.session_ids().tolist() == [0, 1]
    assert fold.test.session_ids().tolist() == [0, 1]
    assert fold.train.normalization_source().tolist() == [True, False]
    assert fold.test.normalization_source().tolist() == [False, False]


def test_splitter_receives_a_splitkeys_object():
    ds = _make_dataset()
    seen = {}

    class _Spy:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            seen["keys"] = keys
            yield np.array([0, 1]), np.array([2, 3])

    next(ds.iter_folds(_Spy(), stratify_target="t"))
    keys = seen["keys"]
    assert isinstance(keys, SplitKeys)
    assert keys.subject_ids.tolist() == ["a", "a", "b", "b"]
    assert keys.session_ids.tolist() == [0, 1, 0, 1]
    assert keys.trial_ids.tolist() == [0, 1, 2, 3]
    assert keys.labels.tolist() == [0, 1, 0, 1]


def test_labels_are_none_without_stratify_target():
    ds = _make_dataset()
    seen = {}

    class _Spy:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            seen["keys"] = keys
            yield np.array([0]), np.array([1])

    next(ds.iter_folds(_Spy()))
    assert seen["keys"].labels is None
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/datasets/test_contract.py -v > task-10-testlog-red.txt 2>&1
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'session_ids'`.

- [ ] **Step 3: `contract.py`를 고친다**

`Splitter` 프로토콜을 교체:

```python
@dataclass(frozen=True)
class SplitKeys:
    """분할기가 보는 축들.

    위치 인자를 하나씩 늘리지 않는 이유는, 축이 또 생기면(run, 과제 유형)
    모든 분할기 시그니처가 다시 깨지기 때문이다.
    """

    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    labels: np.ndarray | None


class Splitter(Protocol):
    """분할기 규약. iter_folds가 기대하는 호출 형태다."""

    #: 같은 피험자가 train과 test에 동시에 나타나도 되는 CV 방식인지.
    #: 생략하면 False로 간주한다(피험자 중첩은 위반). within-subject 계열만 True.
    allows_same_subject: ClassVar[bool]

    #: 같은 (피험자, 세션)이 train과 test에 동시에 나타나도 되는지.
    #: 생략하면 False. within-subject는 세션 안에서 시행을 나누므로 True이고,
    #: cross-session은 세션을 가르는 것이 목적이므로 False다.
    allows_same_session: ClassVar[bool]

    def split(self, keys: SplitKeys) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(train_idx, test_idx) 인덱스 쌍을 fold마다 하나씩 내놓는다.

        `keys.labels`는 층화(stratification)가 필요한 분할기만 쓴다. 라벨을
        분할에 쓰는 것은 누수가 아니다 — 여기 들어오는 라벨은 과제 설계
        (n-back 수준)에서 나온 사전 확정 정보이고, sklearn의
        `StratifiedKFold`가 y를 쓰는 것과 같은 용법이다. 모델 학습은
        여전히 train fold 안에서만 일어난다.
        """
        ...
```

`_View`에 두 메서드를 추가 (`window_times` 아래):

```python
    def session_ids(self) -> np.ndarray:
        return self._ds._session_ids[self._idx]

    def normalization_source(self) -> np.ndarray:
        """이 창이 세션 베이스라인 fit에 쓰였는지."""
        return self._ds._normalization_source[self._idx]
```

`WindowedDataset.__init__`을 교체:

```python
    def __init__(
        self,
        X: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        subject_ids: np.ndarray,
        session_ids: np.ndarray,
        window_times: np.ndarray,
        trial_ids: np.ndarray,
        normalization_source: np.ndarray,
    ) -> None:
        n = len(subject_ids)
        if n == 0:
            raise ValueError("dataset is empty")

        for name, arr in X.items():
            if len(arr) != n:
                raise ValueError(f"modality '{name}' length {len(arr)} != {n}")
        for name, arr in y.items():
            if len(arr) != n:
                raise ValueError(f"target '{name}' length {len(arr)} != {n}")
        for label, arr in (
            ("session_ids", session_ids),
            ("window_times", window_times),
            ("trial_ids", trial_ids),
            ("normalization_source", normalization_source),
        ):
            if len(arr) != n:
                raise ValueError(f"{label} length {len(arr)} != {n}")

        self._X = dict(X)
        self._y = dict(y)
        self._subject_ids = np.asarray(subject_ids)
        self._session_ids = np.asarray(session_ids)
        self._window_times = np.asarray(window_times, dtype=float)
        self._trial_ids = np.asarray(trial_ids)
        self._normalization_source = np.asarray(normalization_source, dtype=bool)

        self._check_trial_ids_globally_unique()

    def _check_trial_ids_globally_unique(self) -> None:
        """한 trial_id가 두 개의 (피험자, 세션)에 걸쳐 있으면 거부한다.

        생성기의 trial_id는 세션마다 0부터 센다. 그대로 쌓으면 sub-01의
        세션0-블록3과 세션1-블록3이 같은 trial_id를 갖고, 시행 단위로
        나누는 분할기가 두 세션의 블록을 한 덩어리로 묶는다. 시행이 더
        이상 일관된 단위가 아니게 되므로, 조립하는 쪽에서 전역 고유하게
        만들 책임을 여기서 강제한다.
        """
        owner: dict = {}
        for trial, subject, session in zip(
            self._trial_ids, self._subject_ids, self._session_ids
        ):
            key = (str(subject), int(session))
            previous = owner.setdefault(trial, key)
            if previous != key:
                raise ValueError(
                    f"trial_id {trial!r} appears under both {previous} and {key}; "
                    "trial_ids must be globally unique across (subject, session). "
                    "조립하는 쪽에서 녹화마다 오프셋을 더해야 한다"
                )
```

`get_trial_ids` 옆에 추가:

```python
    def get_session_ids(self) -> np.ndarray:
        return self._session_ids.copy()
```

`iter_folds`의 분할기 호출부를 교체:

```python
        if stratify_target is None:
            labels = None
        else:
            if stratify_target not in self._y:
                raise KeyError(
                    f"unknown target '{stratify_target}'; have {sorted(self._y)}"
                )
            labels = self._y[stratify_target]

        keys = SplitKeys(
            subject_ids=self._subject_ids,
            session_ids=self._session_ids,
            trial_ids=self._trial_ids,
            labels=labels,
        )
        pairs = splitter.split(keys)
```

`iter_folds`의 docstring도 `SplitKeys`를 넘긴다는 사실에 맞게 고친다.

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/datasets/test_contract.py -v > task-10-testlog-green.txt 2>&1
```

Expected: 새 테스트 6개 PASS. 기존 `WindowedDataset(...)` 호출을 쓰는 테스트는 새 인자를 채우도록 고친다.

- [ ] **Step 5: 커밋**

```bash
git add src/datasets/contract.py tests/datasets/test_contract.py
git commit -m "feat: 계약에 세션 축·SplitKeys·정규화 출처 표식 추가

분할기 시그니처를 SplitKeys 하나로 바꾼다. 위치 인자를 늘리면 축이 또
생길 때(run, 과제 유형) 같은 파손을 반복한다.

trial_id 전역 고유성을 계약이 강제한다. 생성기는 세션마다 0부터 세므로
그대로 쌓으면 sub-01의 세션0-블록3과 세션1-블록3이 같은 trial_id를 갖고,
시행 단위 분할기가 두 세션 블록을 한 덩어리로 묶는다 — 시행이 일관된
단위가 아니게 된다.

allows_same_session은 생략 시 False다. 아무것도 선언하지 않은 새 분할기가
가장 엄격한 쪽으로 떨어지도록 한다."
```

---

## Task 11: 분할기 — `SplitKeys` 이행 + `CrossSessionSplitter`

**Files:**
- Modify: `src/evaluation/splitters.py`
- Test: `tests/evaluation/test_splitters.py`

**Interfaces:**
- Consumes: Task 10의 `SplitKeys`
- Produces:
  - 네 분할기 모두 `split(self, keys: SplitKeys)` 시그니처
  - `LosoSplitter`: `allows_same_subject=False`, `allows_same_session=False`
  - `WithinSubjectSplitter`: `allows_same_subject=True`, **`allows_same_session=True`**
  - `CrossSessionSplitter(n/a)`: `allows_same_subject=True`, `allows_same_session=False`
  - `WindowRandomSplitter`: 둘 다 `False`
  - `get_splitter(name, *, seed)` — `"cross_session"` 추가

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/evaluation/test_splitters.py` 끝에 추가:

```python
from src.datasets.contract import SplitKeys
from src.evaluation.splitters import (
    CrossSessionSplitter,
    LosoSplitter,
    WithinSubjectSplitter,
    get_splitter,
)


def _keys(subjects, sessions, trials, labels=None):
    return SplitKeys(
        subject_ids=np.array(subjects),
        session_ids=np.array(sessions),
        trial_ids=np.array(trials),
        labels=None if labels is None else np.array(labels),
    )


def test_cross_session_holds_out_one_session_index_per_fold():
    keys = _keys(
        subjects=["a"] * 4 + ["b"] * 4,
        sessions=[0, 0, 1, 1] * 2,
        trials=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    folds = list(CrossSessionSplitter().split(keys))
    assert len(folds) == 2
    for train_idx, test_idx in folds:
        test_sessions = set(keys.session_ids[test_idx].tolist())
        train_sessions = set(keys.session_ids[train_idx].tolist())
        assert len(test_sessions) == 1
        assert not (test_sessions & train_sessions)


def test_cross_session_keeps_the_same_subjects_on_both_sides():
    """세션 간 비교는 같은 사람의 다른 날을 보는 것이다."""
    keys = _keys(["a"] * 4, [0, 0, 1, 1], [0, 1, 2, 3])
    train_idx, test_idx = next(iter(CrossSessionSplitter().split(keys)))
    assert set(keys.subject_ids[train_idx]) == set(keys.subject_ids[test_idx])


def test_cross_session_declares_its_allowances():
    assert CrossSessionSplitter.allows_same_subject is True
    assert CrossSessionSplitter.allows_same_session is False


def test_cross_session_needs_at_least_two_sessions():
    keys = _keys(["a", "a"], [0, 0], [0, 1])
    with pytest.raises(ValueError, match="needs at least 2 sessions"):
        list(CrossSessionSplitter().split(keys))


def test_within_subject_allows_the_same_session_on_both_sides():
    """세션 안에서 시행을 나누므로 같은 세션이 양쪽에 나타난다."""
    assert WithinSubjectSplitter.allows_same_session is True


def test_within_subject_never_crosses_a_session_boundary():
    keys = _keys(
        subjects=["a"] * 8,
        sessions=[0, 0, 0, 0, 1, 1, 1, 1],
        trials=[0, 1, 2, 3, 4, 5, 6, 7],
        labels=[0, 1, 0, 1, 0, 1, 0, 1],
    )
    for train_idx, test_idx in WithinSubjectSplitter(n_splits=2).split(keys):
        train_pairs = {
            (s, int(k)) for s, k in zip(keys.subject_ids[train_idx], keys.session_ids[train_idx])
        }
        test_pairs = {
            (s, int(k)) for s, k in zip(keys.subject_ids[test_idx], keys.session_ids[test_idx])
        }
        # 같은 세션이 양쪽에 있는 것은 정상이다. 다른 세션이 섞이는 것이 문제다.
        assert len(test_pairs) == 1
        assert test_pairs <= train_pairs


def test_loso_keeps_subjects_apart_even_with_shared_session_indices():
    """session_idx는 피험자를 가로질러 값이 겹친다. LOSO가 걸리면 안 된다."""
    keys = _keys(["a", "a", "b", "b"], [0, 1, 0, 1], [0, 1, 2, 3])
    for train_idx, test_idx in LosoSplitter().split(keys):
        assert not (
            set(keys.subject_ids[train_idx]) & set(keys.subject_ids[test_idx])
        )


def test_get_splitter_knows_cross_session():
    assert isinstance(get_splitter("cross_session"), CrossSessionSplitter)
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_splitters.py -v > task-11-testlog-red.txt 2>&1
```

Expected: `ImportError: cannot import name 'CrossSessionSplitter'`.

- [ ] **Step 3: `splitters.py`를 고친다**

각 분할기의 `split`을 `SplitKeys` 시그니처로 바꾸고 `allows_same_session`을 선언한다.

`LosoSplitter`:

```python
    allows_same_subject: ClassVar[bool] = False
    allows_same_session: ClassVar[bool] = False

    def split(self, keys):
        # keys.labels는 계약이 전달하지만 LOSO는 쓰지 않는다 (층화 대상이 아님).
        yield from LeaveOneGroupOut().split(
            np.zeros(len(keys.subject_ids)), groups=keys.subject_ids
        )
```

`WithinSubjectSplitter` — 클래스 속성과 `split` 앞부분을 교체:

```python
    allows_same_subject: ClassVar[bool] = True
    #: 세션 안에서 시행을 나누므로 같은 (피험자, 세션)이 train과 test 양쪽에
    #: 나타난다. 세션을 가로질러 나누면 세션 간 성능을 세션 내 성능으로
    #: 잘못 보고하게 된다 (CLAUDE.md §5.4).
    allows_same_session: ClassVar[bool] = True

    def split(self, keys):
        if keys.labels is None:
            raise ValueError(
                "WithinSubjectSplitter needs the label array to stratify blocks; "
                "call WindowedDataset.iter_folds(splitter, stratify_target=<target>). "
                "층화 없이 나누면 특정 fold의 train에서 한 수준이 통째로 사라진다."
            )
        labels = np.asarray(keys.labels)
        # 그룹 키는 (피험자)가 아니라 (피험자, 세션)이다.
        unit = np.array(
            [f"{s}/ses-{int(k)}" for s, k in zip(keys.subject_ids, keys.session_ids)]
        )

        for group in np.unique(unit):
            rows = np.flatnonzero(unit == group)
            trials = keys.trial_ids[rows]
            y = labels[rows]
            unique_trials = np.unique(trials)
            if len(unique_trials) < 2:
                raise ValueError(
                    f"'{group}' has only {len(unique_trials)} unique trial(s); "
                    "within-subject cross-validation needs at least 2 trials "
                    "per session"
                )
            ...
```

이하 본문은 기존 로직 그대로 두되 오류 메시지의 `subject '{subject}'`를 `'{group}'`으로 바꾼다.

`WindowRandomSplitter`:

```python
    allows_same_subject: ClassVar[bool] = False
    allows_same_session: ClassVar[bool] = False

    def split(self, keys):
        # keys.labels는 계약이 전달하지만 누수 시연용 분할기는 쓰지 않는다.
        yield from KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        ).split(np.zeros(len(keys.subject_ids)))
```

`WindowRandomSplitter` 뒤에 새 분할기를 추가:

```python
class CrossSessionSplitter:
    """세션 번호 하나씩을 남기는 leave-one-session-out.

    CLAUDE.md §3.8의 종단 프로필이 성립하려면 "세션 1로 학습해 세션 2를
    맞힌다"가 가능해야 한다. 같은 피험자가 train과 test 양쪽에 있는 것이
    이 분할 방식의 목적 그 자체다 (`allows_same_subject = True`).

    대신 **세션은 절대 공유하지 않는다** (`allows_same_session = False`).
    공유되면 세션 간 일반화를 측정한다는 주장 자체가 무너진다.

    이 분할기의 결과는 LOSO와 같은 표에 섞지 않는다 (CLAUDE.md §5.4).
    러너가 `cv_method`를 기록하고 집계가 혼합을 거부한다.
    """

    allows_same_subject: ClassVar[bool] = True
    allows_same_session: ClassVar[bool] = False

    def split(self, keys):
        sessions = np.asarray(keys.session_ids)
        unique = np.unique(sessions)
        if len(unique) < 2:
            raise ValueError(
                f"cross-session CV needs at least 2 sessions, got {len(unique)}; "
                "config의 simulation.n_sessions를 확인하라"
            )
        all_idx = np.arange(len(sessions))
        for session in unique:
            is_test = sessions == session
            yield all_idx[~is_test], all_idx[is_test]
```

`get_splitter`에 추가:

```python
    if name == "cross_session":
        return CrossSessionSplitter()
```

그리고 오류 메시지의 목록에 `'cross_session'`을 넣는다.

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_splitters.py -v > task-11-testlog-green.txt 2>&1
```

Expected: 새 테스트 8개 PASS. 기존 `splitter.split(subject_ids, trial_ids, labels)` 호출을 쓰는 테스트는 `SplitKeys`로 고친다.

- [ ] **Step 5: 커밋**

```bash
git add src/evaluation/splitters.py tests/evaluation/test_splitters.py
git commit -m "feat: SplitKeys 이행 + CrossSessionSplitter 추가

WithinSubjectSplitter의 그룹 키를 (피험자)에서 (피험자, 세션)으로 바꾼다.
세션을 가로질러 나누면 세션 간 성능을 세션 내 성능으로 잘못 보고하게
된다. 세션 안에서 나누므로 allows_same_session은 True다.

CrossSessionSplitter는 세션 번호 하나씩을 남긴다. 같은 피험자가 양쪽에
있는 것이 목적이고, 세션 공유는 주장 자체를 무너뜨리므로 금지한다."
```
---

## Task 12: 가드 3종 + 하네스 배선

**Files:**
- Modify: `src/evaluation/guards.py`
- Modify: `src/evaluation/harness.py`
- Test: `tests/evaluation/test_guards.py`, `tests/evaluation/test_harness.py`

**Interfaces:**
- Consumes: Task 10의 `_View.session_ids()`·`_View.normalization_source()`, Task 11의 `allows_same_session`
- Produces:
  - `check_session_overlap(train_subjects, train_sessions, test_subjects, test_sessions) -> None`
  - `check_window_overlap(train_subjects, train_sessions, train_times, test_subjects, test_sessions, test_times) -> None` (**시그니처 변경**)
  - `check_normalization_source(train_mask, test_mask) -> None`
  - `run_folds(...)`가 세 가드를 config 토글에 따라 호출

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/evaluation/test_guards.py` 끝에 추가:

```python
from src.datasets.contract import LeakageError
from src.evaluation.guards import (
    check_normalization_source,
    check_session_overlap,
    check_window_overlap,
)


def test_session_overlap_fires_on_a_shared_subject_session_pair():
    with pytest.raises(LeakageError, match="sessions appear in both"):
        check_session_overlap(
            np.array(["a"]), np.array([0]),
            np.array(["a"]), np.array([0]),
        )


def test_session_overlap_ignores_the_same_index_under_different_subjects():
    """session_idx는 피험자를 가로질러 값이 겹친다. LOSO가 걸리면 안 된다."""
    check_session_overlap(
        np.array(["a", "a"]), np.array([0, 1]),
        np.array(["b", "b"]), np.array([0, 1]),
    )


def test_window_overlap_no_longer_false_positives_across_sessions():
    """window_times는 세션마다 0부터 다시 센다.

    피험자만으로 묶으면 물리적으로 다른 녹화의 같은 시각이 겹침으로
    판정되어 CrossSessionSplitter가 아예 돌지 못한다.
    """
    times = np.array([[10.0, 15.0]])
    check_window_overlap(
        np.array(["a"]), np.array([0]), times,
        np.array(["a"]), np.array([1]), times,
    )


def test_window_overlap_still_fires_inside_one_session():
    with pytest.raises(LeakageError, match="window overlap"):
        check_window_overlap(
            np.array(["a"]), np.array([0]), np.array([[10.0, 15.0]]),
            np.array(["a"]), np.array([0]), np.array([[12.0, 17.0]]),
        )


def test_normalization_source_guard_fires_when_a_masked_window_reaches_a_fold():
    with pytest.raises(LeakageError, match="normalization source"):
        check_normalization_source(
            np.array([True, False]), np.array([False, False])
        )


def test_normalization_source_guard_passes_when_no_masked_window_is_present():
    check_normalization_source(np.array([False, False]), np.array([False]))
```

`tests/evaluation/test_harness.py` 끝에 추가:

```python
def test_run_folds_runs_the_session_guard_for_a_splitter_that_forbids_sharing():
    """세션 공유를 금지한다고 선언한 분할기가 실제로 공유하면 실패시킨다."""
    ds = _dataset_with_two_sessions()   # 아래 헬퍼

    class _Bad:
        allows_same_subject = True
        allows_same_session = False

        def split(self, keys):
            yield np.array([0, 1, 2, 3]), np.array([0, 1])

    with pytest.raises(LeakageError, match="sessions appear in both"):
        run_folds(
            ds, _Bad(), target="t", modalities=["m"],
            guards={"check_subject_overlap": True, "check_window_overlap": False,
                    "check_session_overlap": True, "check_normalization_source": True},
            seed=0,
        )


def test_run_folds_skips_the_session_guard_when_the_splitter_allows_sharing():
    ds = _dataset_with_two_sessions()

    class _WithinLike:
        allows_same_subject = True
        allows_same_session = True

        def split(self, keys):
            yield np.array([0, 2]), np.array([1, 3])

    results = run_folds(
        ds, _WithinLike(), target="t", modalities=["m"],
        guards={"check_subject_overlap": True, "check_window_overlap": False,
                "check_session_overlap": True, "check_normalization_source": True},
        seed=0,
    )
    assert len(results) == 1
```

헬퍼는 파일 상단 근처에 둔다:

```python
def _dataset_with_two_sessions():
    n = 4
    return WindowedDataset(
        X={"m": np.array([[0.0], [1.0], [0.1], [1.1]])},
        y={"t": np.array([0, 1, 0, 1])},
        subject_ids=np.array(["a", "a", "a", "a"]),
        session_ids=np.array([0, 0, 1, 1]),
        window_times=np.column_stack([np.arange(n, dtype=float) * 10,
                                      np.arange(n, dtype=float) * 10 + 5]),
        trial_ids=np.array([0, 1, 2, 3]),
        normalization_source=np.zeros(n, dtype=bool),
    )
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_guards.py tests/evaluation/test_harness.py -v > task-12-testlog-red.txt 2>&1
```

Expected: `ImportError: cannot import name 'check_session_overlap'`.

- [ ] **Step 3: `guards.py`를 고친다**

`check_window_overlap`을 교체하고 두 가드를 추가:

```python
def _session_keys(subjects: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    """(피험자, 세션)을 하나의 문자열 키로 만든다.

    session_idx만으로 비교하면 피험자 A의 세션 0과 피험자 B의 세션 0이 같은
    세션으로 취급되어 LOSO가 자기 가드에 걸린다. 반대로 피험자만으로 묶으면
    세션마다 0부터 다시 세는 window_times가 겹침으로 오판된다.
    """
    return np.array([f"{s}/ses-{int(k)}" for s, k in zip(subjects, sessions)])


def check_session_overlap(
    train_subjects: np.ndarray,
    train_sessions: np.ndarray,
    test_subjects: np.ndarray,
    test_sessions: np.ndarray,
) -> None:
    """같은 (피험자, 세션)이 train과 test에 동시에 있으면 실패시킨다."""
    train_keys = set(_session_keys(train_subjects, train_sessions).tolist())
    test_keys = set(_session_keys(test_subjects, test_sessions).tolist())
    shared = sorted(train_keys & test_keys)
    if shared:
        raise LeakageError(
            f"sessions appear in both train and test: {shared}. "
            "세션 간 일반화를 측정한다고 주장하는 분할에서 세션이 공유되면 "
            "그 주장 자체가 무너진다 (CLAUDE.md §5.4)."
        )


def check_window_overlap(
    train_subjects: np.ndarray,
    train_sessions: np.ndarray,
    train_times: np.ndarray,
    test_subjects: np.ndarray,
    test_sessions: np.ndarray,
    test_times: np.ndarray,
) -> None:
    """같은 (피험자, 세션) 안에서 train 창과 test 창이 시간상 겹치면 실패시킨다.

    서로 다른 피험자, 또는 같은 피험자라도 서로 다른 세션의 창은 물리적으로
    다른 녹화이므로 시각이 같아도 검사 대상이 아니다. window_times가 세션마다
    0부터 다시 세기 때문에, 피험자만으로 묶으면 세션 간 분할이 전부 오탐이 된다.

    5초 창·1초 스텝은 80% 오버랩이므로 창 단위 무작위 분할은 여기서 걸린다.
    """
    train_keys = _session_keys(train_subjects, train_sessions)
    test_keys = _session_keys(test_subjects, test_sessions)

    for key in set(train_keys.tolist()) & set(test_keys.tolist()):
        tr = train_times[train_keys == key]
        te = test_times[test_keys == key]
        # 반열린 구간 [start, end) 기준 겹침
        overlaps = (tr[:, None, 0] < te[None, :, 1]) & (te[None, :, 0] < tr[:, None, 1])
        if overlaps.any():
            i, j = np.argwhere(overlaps)[0]
            raise LeakageError(
                f"window overlap for {key}: train {tr[i].tolist()} "
                f"overlaps test {te[j].tolist()}. "
                "인접 윈도우가 train/test에 동시 존재하면 그 결과는 폐기 대상이다."
            )


def check_normalization_source(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """세션 베이스라인 fit에 쓰인 창이 fold에 들어오면 실패시킨다.

    "베이스라인은 train도 test도 아니다"는 설계상의 사실이 아니라 여기서
    지키는 불변식이다. 미래에 누가 베이스라인을 타깃에 옵트인하면 이 가드가
    먼저 터지고, 그 사람은 "이 타깃에는 다른 정규화 기준을 쓰겠다"를 명시적으로
    결정하게 된다 — 조용히 통과하지 않는다.
    """
    n_train = int(np.asarray(train_mask).sum())
    n_test = int(np.asarray(test_mask).sum())
    if n_train or n_test:
        raise LeakageError(
            f"{n_train + n_test} window(s) used as the normalization source "
            f"reached a fold (train={n_train}, test={n_test}). "
            "정규화 기준 구간이 분석 데이터에 들어가면 그 표본은 정규화 정의상 "
            "0 근처가 되어 공짜 클래스가 생긴다 (스펙 §6.4·§6.5)."
        )
```

- [ ] **Step 4: `harness.py`를 고친다**

import에 새 가드를 추가하고, `run_folds`의 가드 블록을 교체:

```python
    allows_same_subject = getattr(splitter, "allows_same_subject", False)
    allows_same_session = getattr(splitter, "allows_same_session", False)

    for fold in dataset.iter_folds(splitter, stratify_target=target):
        train_subj = fold.train.subject_ids()
        test_subj = fold.test.subject_ids()
        train_sess = fold.train.session_ids()
        test_sess = fold.test.session_ids()

        if guards.get("check_subject_overlap", True) and not allows_same_subject:
            check_subject_overlap(train_subj, test_subj)
        if guards.get("check_session_overlap", True) and not allows_same_session:
            check_session_overlap(train_subj, train_sess, test_subj, test_sess)
        if guards.get("check_window_overlap", True):
            check_window_overlap(
                train_subj, train_sess, fold.train.window_times(),
                test_subj, test_sess, fold.test.window_times(),
            )
        if guards.get("check_normalization_source", True):
            check_normalization_source(
                fold.train.normalization_source(),
                fold.test.normalization_source(),
            )
```

모듈 docstring에 한 문단 추가:

```
세션 중첩 가드도 분할기의 선언(`allows_same_session`)을 존중한다.
within-subject CV는 세션 안에서 시행을 나누므로 같은 세션이 양쪽에 있는 것이
설계 그 자체다. 반면 정규화 출처 가드는 어떤 분할기에서도 예외가 없다 —
기준 구간이 분석 데이터에 섞이는 것은 분할 방식과 무관한 오류이기 때문이다.
```

- [ ] **Step 5: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_guards.py tests/evaluation/test_harness.py -v > task-12-testlog-green.txt 2>&1
```

Expected: 새 테스트 8개 PASS. 기존 `check_window_overlap(4개 인자)` 호출은 6개 인자로 고친다.

- [ ] **Step 6: 커밋**

```bash
git add src/evaluation/guards.py src/evaluation/harness.py tests/evaluation
git commit -m "feat: 세션 중첩·정규화 출처 가드 추가, 창 겹침을 (피험자,세션) 단위로

창 겹침 가드가 피험자만으로 묶고 있었다. window_times는 세션마다 0부터
다시 세므로 물리적으로 다른 녹화의 같은 시각이 겹침으로 오판되어
CrossSessionSplitter가 아예 돌지 못한다.

세션 중첩 검사는 복합 키다. session_idx만 보면 피험자 A의 세션 0과
피험자 B의 세션 0이 같은 세션이 되어 LOSO가 자기 가드에 걸린다.

정규화 출처 가드는 어떤 분할기에서도 예외가 없다. 기준 구간이 분석
데이터에 섞이는 것은 분할 방식과 무관한 오류다."
```

---

## Task 13: 러너 통합 — 세션 루프 · 정규화 · 품질 기록

**Files:**
- Modify: `src/evaluation/runner.py`
- Modify: `src/evaluation/metrics.py` (`aggregate_runs` 추가)
- Test: `tests/evaluation/test_runner.py`, `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: Task 7~12 전부
- Produces:
  - `build_dataset(cfg, rng) -> tuple[WindowedDataset, int, list[SessionQuality]]` (**반환 변경**)
  - `run_experiment` 결과 디렉토리에 `session_quality.csv` 추가
  - `metrics.aggregate_runs(runs: list[dict]) -> dict` — `cv_method`가 섞이면 `ValueError`
  - `metrics.json`에 `n_sessions_flagged`(int), `cv_method`(기존)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/evaluation/test_metrics.py` 끝에 추가:

```python
from src.evaluation.metrics import aggregate_runs


def test_aggregate_runs_refuses_to_mix_cv_methods():
    """LOSO와 세션 간 결과를 같은 표에 섞으면 CLAUDE.md §5.4 위반이다."""
    runs = [
        {"cv_method": "loso", "accuracy_mean": 0.65, "chance_level": 1 / 3},
        {"cv_method": "cross_session", "accuracy_mean": 0.71, "chance_level": 1 / 3},
    ]
    with pytest.raises(ValueError, match="cannot mix cv_method"):
        aggregate_runs(runs)


def test_aggregate_runs_combines_same_scheme():
    runs = [
        {"cv_method": "loso", "accuracy_mean": 0.60, "chance_level": 1 / 3},
        {"cv_method": "loso", "accuracy_mean": 0.70, "chance_level": 1 / 3},
    ]
    out = aggregate_runs(runs)
    assert out["cv_method"] == "loso"
    assert out["n_runs"] == 2
    assert out["accuracy_mean"] == pytest.approx(0.65)
    assert out["chance_level"] == pytest.approx(1 / 3)


def test_aggregate_runs_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no runs"):
        aggregate_runs([])
```

`tests/evaluation/test_runner.py` 끝에 추가:

```python
def test_build_dataset_returns_session_quality_per_recording():
    cfg = load_config("config/experiments/smoke.yaml")
    rng = set_all_seeds(cfg["seed"])
    ds, _, qualities = build_dataset(cfg, rng)
    expected = cfg["simulation"]["n_subjects"] * cfg["simulation"]["n_sessions"]
    assert len(qualities) == expected
    assert {q.session_idx for q in qualities} == set(range(cfg["simulation"]["n_sessions"]))


def test_dataset_carries_session_ids_and_globally_unique_trials():
    cfg = load_config("config/experiments/smoke.yaml")
    rng = set_all_seeds(cfg["seed"])
    ds, _, _ = build_dataset(cfg, rng)
    assert set(ds.get_session_ids().tolist()) == set(range(cfg["simulation"]["n_sessions"]))
    # 계약이 이미 검증하지만, 조립 쪽이 오프셋을 실제로 더했는지 확인한다
    pairs = {
        (s, int(k)) for s, k in zip(ds.get_subject_ids(), ds.get_session_ids())
    }
    assert len(pairs) == cfg["simulation"]["n_subjects"] * cfg["simulation"]["n_sessions"]


def test_include_baseline_with_load_target_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="include_baseline"):
        run_experiment(
            "config/experiments/smoke.yaml",
            overrides={"dataset": {"include_baseline": True}},
        )


def test_run_experiment_writes_session_quality_csv(tmp_path):
    out = run_experiment(
        "config/experiments/smoke.yaml",
        overrides={"output": {"results_dir": str(tmp_path)}},
    )
    text = (out / "session_quality.csv").read_text(encoding="utf-8")
    assert "subject_id,session_idx,drift_flag" in text.splitlines()[0]


def test_cross_session_run_records_its_scheme(tmp_path):
    out = run_experiment(
        "config/experiments/smoke.yaml",
        overrides={
            "evaluation": {"splitter": "cross_session"},
            "output": {"results_dir": str(tmp_path)},
        },
    )
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cv_method"] == "cross_session"
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_runner.py tests/evaluation/test_metrics.py -v > task-13-testlog-red.txt 2>&1
```

Expected: `ImportError: cannot import name 'aggregate_runs'`, `ValueError: too many values to unpack`.

- [ ] **Step 3: `metrics.aggregate_runs`를 추가한다**

`src/evaluation/metrics.py` 끝에:

```python
def aggregate_runs(runs: list[dict]) -> dict:
    """여러 실행의 지표를 합친다. CV 방식이 다르면 거부한다.

    LOSO(교차 피험자)와 within-subject·cross-session은 서로 다른 질문에
    답하는 수치다. 같은 표에 넣으면 평균 자체가 의미를 잃고, 읽는 사람은
    그 사실을 알 수 없다 (CLAUDE.md §5.4).
    """
    if not runs:
        raise ValueError("no runs to aggregate")

    methods = sorted({r["cv_method"] for r in runs})
    if len(methods) > 1:
        raise ValueError(
            f"cannot mix cv_method values {methods} in one aggregate; "
            "LOSO 결과와 within-subject·cross-session 결과를 혼용 표기하는 것은 "
            "CLAUDE.md §5.4가 금지한다. 스킴별로 따로 집계하라"
        )

    chances = sorted({round(float(r["chance_level"]), 6) for r in runs})
    if len(chances) > 1:
        raise ValueError(
            f"cannot mix chance levels {chances} in one aggregate; "
            "클래스 수가 다른 실행들이다"
        )

    accuracies = np.array([float(r["accuracy_mean"]) for r in runs])
    return {
        "cv_method": methods[0],
        "chance_level": chances[0],
        "n_runs": len(runs),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=1)) if len(runs) > 1 else 0.0,
        "accuracy_worst": float(accuracies.min()),
    }
```

- [ ] **Step 4: `runner.build_dataset`을 고친다**

import에 추가:

```python
from src.preprocessing.baseline import (
    MODALITY_KIND,
    SessionBaseline,
    SessionQuality,
    compute_drift,
    flag_drift,
)
from src.simulation.state import BASELINE
```

`build_dataset`을 교체:

```python
def build_dataset(
    cfg: dict, rng: np.random.Generator
) -> tuple[WindowedDataset, int, list[SessionQuality]]:
    """합성 녹화를 만들고 창·특징·정규화·라벨을 거쳐 계약 객체를 조립한다.

    정규화는 **녹화(= 한 피험자의 한 세션) 단위로 독립 수행**한다. 기준은
    그 세션의 시작 베이스라인이고, 종료 베이스라인은 드리프트 추정에만
    쓴다 (CLAUDE.md §3.8).
    """
    win_cfg = cfg["windowing"]
    ds_cfg = cfg["dataset"]
    base_cfg = cfg["preprocessing"]["baseline"]
    lead_delta_s = float(cfg["simulation"]["lead_delta_s"])

    normalize = bool(base_cfg["normalize"])
    zero_atol = float(base_cfg["zero_atol"])
    drift_threshold = float(base_cfg["drift_threshold_relative"])
    include_baseline = bool(ds_cfg["include_baseline"])

    extract_features = get_extractor(cfg["features"]["extractor"])
    lead_targets = list(ds_cfg["lead_targets"])

    recordings = generate_dataset(cfg["simulation"], rng)

    feature_blocks: dict[str, list[np.ndarray]] = {}
    label_blocks: dict[str, list[np.ndarray]] = {}
    subjects, sessions, times, trials, norm_src = [], [], [], [], []
    qualities: list[SessionQuality] = []
    n_dropped = 0
    trial_offset = 0

    for rec in recordings:
        windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
        feats = extract_features(rec, windows)

        is_base = windows.block_kind == BASELINE
        start_mask = is_base & (windows.trial_id == windows.trial_id.min())
        end_mask = is_base & (windows.trial_id == windows.trial_id.max())
        if not start_mask.any() or not end_mask.any():
            raise ValueError(
                f"{rec.subject_id} ses-{rec.session_idx}: 시작 또는 종료 "
                "베이스라인에서 창이 나오지 않았다. CLAUDE.md §2.4는 두 "
                "베이스라인을 생략 불가로 규정한다"
            )

        # 드리프트는 **정규화 전 원 단위**에서 잰다. dB 변환 후에는 기준값이
        # 정의상 0이 되어 상대 비율이 성립하지 않는다.
        drift_by_modality: dict[str, float] = {}
        n_excluded_by_modality: dict[str, int] = {}
        for name, arr in feats.items():
            if MODALITY_KIND[name] == "absolute":
                continue  # 행동은 장비 재부착의 영향을 받지 않으므로 드리프트 대상이 아니다
            drift = compute_drift(arr[start_mask], arr[end_mask], zero_atol=zero_atol)
            drift_by_modality[name] = drift.aggregate
            n_excluded_by_modality[name] = drift.n_excluded

        qualities.append(
            SessionQuality(
                subject_id=rec.subject_id,
                session_idx=rec.session_idx,
                drift_by_modality=drift_by_modality,
                n_excluded_by_modality=n_excluded_by_modality,
                drift_flag=flag_drift(drift_by_modality, drift_threshold),
            )
        )

        if normalize:
            for name in list(feats):
                baseline = SessionBaseline.fit(
                    feats[name][start_mask], MODALITY_KIND[name]
                )
                feats[name] = baseline.apply(feats[name])

        labels, keep = build_labels(
            rec, windows,
            lead_delta_s=lead_delta_s,
            rt_bins=list(ds_cfg["rt_bins"]),
            lead_targets=lead_targets,
            include_baseline=include_baseline,
        )
        n_dropped += int((~keep).sum())

        for name, arr in feats.items():
            feature_blocks.setdefault(name, []).append(arr[keep])
        for name, arr in labels.items():
            label_blocks.setdefault(name, []).append(arr[keep])

        n_kept = int(keep.sum())
        subjects.append(np.full(n_kept, rec.subject_id))
        sessions.append(np.full(n_kept, rec.session_idx))
        times.append(np.column_stack([windows.start_s, windows.end_s])[keep])
        # trial_id는 세션마다 0부터 세므로 전역 오프셋을 더한다. 계약이
        # 고유성을 검증하지만, 만드는 책임은 여기에 있다.
        trials.append(windows.trial_id[keep] + trial_offset)
        # 정규화에 실제로 쓰인 창만 표식한다. normalize가 꺼져 있으면
        # 아무것도 fit되지 않았으므로 표식도 없다.
        norm_src.append(
            start_mask[keep] if normalize else np.zeros(n_kept, dtype=bool)
        )
        trial_offset += int(windows.trial_id.max()) + 1

    dataset = WindowedDataset(
        X={k: np.vstack(v) for k, v in feature_blocks.items()},
        y={k: np.concatenate(v) for k, v in label_blocks.items()},
        subject_ids=np.concatenate(subjects),
        session_ids=np.concatenate(sessions),
        window_times=np.vstack(times),
        trial_ids=np.concatenate(trials),
        normalization_source=np.concatenate(norm_src),
    )
    return dataset, n_dropped, qualities
```

> 표식의 의미는 "이 창이 세션 베이스라인 fit에 실제로 쓰였다"이다. `normalize`가 꺼져 있으면 아무것도 fit되지 않았으므로 표식도 없어야 한다 — 그래야 Task 16의 T5가 "가드가 무엇 때문에 터졌는가"를 명확히 물을 수 있다.

- [ ] **Step 5: `run_experiment`을 고친다**

`targets` 검증 바로 뒤에 추가:

```python
    if bool(cfg["dataset"]["include_baseline"]) and "cognitive_load" in targets:
        raise ValueError(
            "dataset.include_baseline=true 인데 target이 'cognitive_load'다. "
            "베이스라인 창은 과제 조건이 없어 센티넬 라벨(-1)을 갖는다 — 부하 "
            "분류에 넣으면 클래스가 하나 늘고 chance level이 33.3%에서 25%로 "
            "조용히 바뀐다. CLAUDE.md §9.2 4번('3수준 분류'의 정의)이 미확정이므로 "
            "코드가 임의로 확정하지 않는다."
        )
```

`build_dataset` 호출을 3-튜플로 받는다:

```python
    dataset, n_dropped, qualities = build_dataset(cfg, rng)
```

`metrics` 조립에 추가:

```python
    metrics["n_sessions_flagged"] = sum(1 for q in qualities if q.drift_flag)
    metrics["n_sessions"] = len(qualities)
```

`per_fold.csv` 기록 뒤에 품질 CSV를 추가:

```python
    with (out_dir / "session_quality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        modalities = sorted(qualities[0].drift_by_modality) if qualities else []
        writer.writerow(
            ["subject_id", "session_idx", "drift_flag"]
            + [f"drift_{m}" for m in modalities]
            + [f"n_excluded_{m}" for m in modalities]
        )
        for q in qualities:
            writer.writerow(
                [q.subject_id, q.session_idx, q.drift_flag]
                + [f"{q.drift_by_modality[m]:.6f}" for m in modalities]
                + [q.n_excluded_by_modality[m] for m in modalities]
            )
```

`log.txt`에 한 줄 추가:

```python
        f"n_sessions_flagged={metrics['n_sessions_flagged']}/{metrics['n_sessions']}\n"
```

- [ ] **Step 6: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests -v > task-13-testlog-green.txt 2>&1
```

Expected: **전체 스위트 초록.** 여기가 통합 지점이므로 앞 태스크에서 남은 시그니처 불일치가 전부 드러난다. 남은 실패를 모두 고친 뒤 진행한다.

- [ ] **Step 7: 커밋**

```bash
git add src/evaluation/runner.py src/evaluation/metrics.py tests/evaluation
git commit -m "feat: 러너에 세션 루프·베이스라인 정규화·품질 기록 통합

정규화는 녹화(한 피험자의 한 세션) 단위로 독립 수행한다. 여러 세션을
모아 정규화하면 세션 간 차이가 지워지고, 그것이야말로 측정하려는 대상이다.

드리프트는 정규화 전 원 단위에서 잰다. dB 변환 후에는 기준값이 정의상
0이라 상대 비율이 성립하지 않는다.

include_baseline=true + cognitive_load 조합을 명시적으로 거부한다.
센티넬 라벨이 클래스로 들어가면 chance level이 33.3%에서 25%로 조용히
바뀌는데, '3수준 분류'의 정의는 아직 미확정이다.

aggregate_runs는 cv_method가 섞이면 거부한다."
```

---

## Task 14: 승인기준 T1(널) · T2(붕괴) · T3(회복)

**Files:**
- Create: `config/experiments/session_recovery.yaml`
- Create: `tests/evaluation/test_validation_session.py`
- Modify: `docs/specs/2026-08-19-session-baseline-design.md` (§8.4 임계 확정 기록)
- Test: 위 신규 파일

**Interfaces:**
- Consumes: Task 13의 `run_experiment`, `build_dataset`
- Produces: 없음 (검증 전용)

- [ ] **Step 1: 검증 config를 만든다**

`config/experiments/session_recovery.yaml`:

```yaml
# T2·T3 전용. 드리프트가 켜져 있고 세션이 3회다.
# 개발 속도를 위해 EEG는 250 Hz, 피험자 8명으로 줄였다 — 검증 대상은
# 절대 성능이 아니라 정규화 전후의 차이이므로 규모가 작아도 성립한다.
run_name: session_recovery
seed: 42

simulation:
  n_subjects: 8
  n_sessions: 3
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 30
    baseline_duration_s: 20
    n_blocks_per_level: 2
    stim_interval_s: 2.0
  practice:
    rate: 0.15
  drift:
    fnirs_gain_sigma: 0.20
    fnirs_offset_sigma: 0.10
    eeg_gain_sigma: 0.15
    eeg_noise_sigma: 0.15
    within_session_rate: 0.30
    within_session_fraction: 0.33
    between_session_scale: 4.0
    assignment: sampled
  eeg:   {n_channels: 30, sfreq_hz: 250}
  fnirs: {n_channels: 48, sfreq_hz: 10.4, hbr_coupling: -0.33}

windowing:
  window_s: 5.0
  step_s: 1.0

features:
  extractor: minimal

preprocessing:
  baseline:
    normalize: true
    drift_threshold_relative: 0.20
    zero_atol: 1.0e-8

dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy, response_latency]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]
  include_baseline: false

evaluation:
  splitter: cross_session
  model: logistic_regression
  guards:
    check_subject_overlap: true
    check_window_overlap: true
    check_session_overlap: true
    check_normalization_source: true

output:
  results_dir: results
```

- [ ] **Step 2: T1(널) 테스트를 쓴다 — 결함 주입 포함**

`tests/evaluation/test_validation_session.py` (신규):

```python
"""승인기준 T1~T6 (스펙 §8).

각 테스트는 대상 결함을 주입했을 때 실제로 실패하는 것을 함께 보인다.
실패할 수 없는 테스트는 테스트가 아니다.
"""

import dataclasses
import json

import numpy as np
import pytest

from src.common.config import load_config, validate_config
from src.common.seeding import set_all_seeds
from src.datasets.features_minimal import extract_features
from src.evaluation import runner as runner_mod
from src.evaluation.runner import _deep_update, run_experiment
from src.simulation.recording import generate_dataset

RECOVERY = "config/experiments/session_recovery.yaml"


def _run(overrides, tmp_path, config=RECOVERY):
    over = _deep_update({"output": {"results_dir": str(tmp_path)}}, overrides)
    out = run_experiment(config, overrides=over)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- T1 널

NULL_OVER = {"simulation": {"effect_size": 0.0}}


def test_t1_null_stays_at_chance_before_and_after_normalization(tmp_path):
    """효과가 없으면 드리프트가 있어도 chance여야 한다.

    드리프트는 세션별 채널 이득인데 부하 수준은 세션 *안에서* 무작위
    배치되므로, 드리프트가 부하를 예측할 원리적 경로가 없다. 이 테스트가
    실제로 잡는 것은 §8.1의 두 구현 버그다.
    """
    for normalize in (False, True):
        m = _run(
            _deep_update(NULL_OVER, {"preprocessing": {"baseline": {"normalize": normalize}}}),
            tmp_path / f"norm{normalize}",
        )
        assert m["chance_level"] == pytest.approx(1 / 3)
        assert m["pooled_ci_low"] <= m["chance_level"] <= m["pooled_ci_high"], (
            f"normalize={normalize}: chance {m['chance_level']:.4f} 가 "
            f"CI [{m['pooled_ci_low']:.4f}, {m['pooled_ci_high']:.4f}] 밖이다"
        )


def test_t1_injection_1_load_conditional_normalization_breaks_the_null(tmp_path, monkeypatch):
    """주입 1: (세션 × 부하수준) 단위 중심화.

    잔차에 부하가 인코딩되어 effect_size=0인데도 chance를 넘는다.
    'groupby 키를 하나 더 넣으면 더 깨끗하다'는 흔한 유혹이 만드는 버그다.
    """

    def _load_conditional(rec, windows):
        feats = extract_features(rec, windows)
        for arr in feats.values():
            for level in np.unique(windows.load_level):
                m = windows.load_level == level
                arr[m] -= arr[m].mean(axis=0)
        return feats

    monkeypatch.setattr(runner_mod, "get_extractor", lambda name: _load_conditional)
    m = _run(NULL_OVER, tmp_path)
    assert m["pooled_ci_low"] > m["chance_level"], (
        "부하 조건부 정규화를 주입했는데도 널이 유지됐다 — T1이 이 버그를 못 잡는다"
    )


def test_t1_injection_2_load_dependent_drift_breaks_the_null(tmp_path, monkeypatch):
    """주입 2: 드리프트 파라미터가 부하에 의존.

    세션 단위로 한 번 뽑아야 할 것을 블록 루프 안에서 뽑으면 이렇게 된다.
    드리프트 자체가 부하의 대리변수가 된다.
    """
    real = generate_dataset

    def _load_dependent(sim_cfg, rng):
        out = []
        for rec in real(sim_cfg, rng):
            t = np.arange(rec.hbo.shape[1]) / rec.fnirs_sfreq
            gain = 1.0 + 0.5 * rec.timeline.effective_load(t)
            out.append(dataclasses.replace(rec, hbo=rec.hbo * gain[None, :]))
        return out

    monkeypatch.setattr(runner_mod, "generate_dataset", _load_dependent)
    m = _run(NULL_OVER, tmp_path)
    assert m["pooled_ci_low"] > m["chance_level"], (
        "부하 의존 드리프트를 주입했는데도 널이 유지됐다 — T1이 이 버그를 못 잡는다"
    )
```

> **두 주입 모두 널을 깨지 못하면 T1을 삭제한다** (스펙 §8.1). 그 경우 이 파일에서 T1 세 테스트를 지우고, 삭제 근거를 `run_logging.md`에 적는다.

- [ ] **Step 3: T1을 실행하고 결과를 본다**

```bash
python -m pytest tests/evaluation/test_validation_session.py -k t1 -v > task-14-t1-log.txt 2>&1
```

Expected: 3개 모두 PASS. 주입 테스트가 실패하면 그 주입이 실제로는 버그가 아니라는 뜻이므로, 주입을 강화하거나 T1을 삭제한다.

- [ ] **Step 4: T2·T3 테스트를 쓴다 — 임계를 지어내지 않는 형태로**

A+D의 T3에서 근거 없이 "15pp 향상"을 적어놨다가 실측 7.55pp를 만나 재보정한 일이 있다. 이번에는 **절대 임계를 아예 쓰지 않는다.** 대신 그 실행 자신이 만들어내는 기준점에 대고 비교한다.

| | 판정 기준 | 왜 이것이 기준점인가 |
|---|---|---|
| **T2 붕괴** | 정규화 off의 pooled CI가 **chance를 포함**한다 | "chance 근처로 붕괴"의 통계적 정의 그대로. 자유 파라미터 없음 |
| **T3 회복** | 정규화 on이 chance와 **세션 내 성능**(`within_subject`) 사이 간격의 **절반 이상**을 회복한다 | 세션 내 성능이 이 데이터에서 도달 가능한 상한이다. 0.5는 "절반 이상"이라는 사전 선언이지 관측에서 역산한 값이 아니다 |

`tests/evaluation/test_validation_session.py`에 추가:

```python
# ---------------------------------------------------------------- T2·T3

#: T3의 회복 기준. chance와 세션 내 성능 사이 간격의 이 비율 이상을 회복해야
#: 한다. 관측에서 역산한 값이 아니라 사전에 선언한 요구 수준이다 (스펙 §8.4).
T3_GAP_RECOVERY_MIN = 0.5


def test_t2_without_normalization_cross_session_accuracy_collapses(tmp_path):
    """정규화가 없으면 세션 간 분류가 chance로 무너진다.

    무너지지 않으면 드리프트가 신호를 흔들지 않는다는 뜻이고, 그러면 T3의
    '회복'은 아무것도 증명하지 못한다. T3의 존재 이유가 T2다.
    """
    m = _run({"preprocessing": {"baseline": {"normalize": False}}}, tmp_path)
    assert m["cv_method"] == "cross_session"
    assert m["pooled_ci_low"] <= m["chance_level"] <= m["pooled_ci_high"], (
        f"정규화 없이도 세션 간 분류가 {m['pooled_accuracy']:.4f} "
        f"(CI [{m['pooled_ci_low']:.4f}, {m['pooled_ci_high']:.4f}])로 chance를 "
        "넘었다. 드리프트가 실제로 신호를 압도하지 않는다는 뜻이다."
    )


def test_t2_injection_turning_drift_off_removes_the_collapse(tmp_path):
    """결함 주입: 드리프트를 끄면 붕괴가 사라져야 한다.

    사라지지 않으면 붕괴의 원인이 드리프트가 아니라는 뜻이고, T2가 엉뚱한
    것을 재고 있다는 뜻이다.
    """
    m = _run(
        {
            "preprocessing": {"baseline": {"normalize": False}},
            "simulation": {"drift": {
                "fnirs_gain_sigma": 0.0, "fnirs_offset_sigma": 0.0,
                "eeg_gain_sigma": 0.0, "eeg_noise_sigma": 0.0,
                "within_session_rate": 0.0,
            }},
        },
        tmp_path,
    )
    assert m["pooled_ci_low"] > m["chance_level"], (
        "드리프트를 껐는데도 세션 간 분류가 chance에 머물렀다 — 붕괴의 원인이 "
        "드리프트가 아니다."
    )


def test_t3_normalization_recovers_most_of_the_gap(tmp_path):
    """정규화 후 세션 간 성능이 세션 내 성능 쪽으로 회복한다.

    상한은 같은 데이터의 within_subject 성능이다. 이 데이터에서 도달 가능한
    최대치이므로, 절대 수치를 지어내지 않고도 '얼마나 회복했는가'를 물을 수 있다.
    """
    off = _run({"preprocessing": {"baseline": {"normalize": False}}}, tmp_path / "off")
    on = _run({"preprocessing": {"baseline": {"normalize": True}}}, tmp_path / "on")
    within = _run(
        {"evaluation": {"splitter": "within_subject"}}, tmp_path / "within"
    )
    chance = on["chance_level"]

    ceiling = within["pooled_accuracy"]
    assert ceiling > chance, "세션 내 성능조차 chance라면 데이터에 신호가 없다"

    recovered = (on["pooled_accuracy"] - chance) / (ceiling - chance)
    assert recovered >= T3_GAP_RECOVERY_MIN, (
        f"회복률 {recovered:.3f} < {T3_GAP_RECOVERY_MIN}. "
        f"off={off['pooled_accuracy']:.4f} on={on['pooled_accuracy']:.4f} "
        f"within={ceiling:.4f} chance={chance:.4f}"
    )


def test_t3_injection_foreign_baseline_fails_to_recover(tmp_path, monkeypatch):
    """결함 주입: 다른 세션의 베이스라인으로 정규화하면 회복하지 못한다."""
    from src.preprocessing.baseline import SessionBaseline

    cache: dict = {}

    class _ForeignBaseline(SessionBaseline):
        @classmethod
        def fit(cls, start_baseline, kind):
            real = SessionBaseline.fit(start_baseline, kind)
            reference = cache.setdefault(kind, real.reference)
            return SessionBaseline(reference=reference, kind=kind)

    monkeypatch.setattr(runner_mod, "SessionBaseline", _ForeignBaseline)
    on = _run({"preprocessing": {"baseline": {"normalize": True}}}, tmp_path / "foreign")
    within = _run({"evaluation": {"splitter": "within_subject"}}, tmp_path / "within")
    chance = on["chance_level"]
    recovered = (on["pooled_accuracy"] - chance) / (within["pooled_accuracy"] - chance)
    assert recovered < T3_GAP_RECOVERY_MIN, (
        f"남의 세션 베이스라인으로 정규화했는데도 회복률 {recovered:.3f}를 "
        "달성했다 — 정규화가 세션 고유 정보를 쓰고 있지 않다는 뜻이다."
    )
```

- [ ] **Step 5: 실행한다**

```bash
python -m pytest tests/evaluation/test_validation_session.py -v > task-14-testlog-green.txt 2>&1
```

Expected: T1 3개 + T2 2개 + T3 2개 PASS.

**실패했을 때의 규칙:**

- **T2가 실패**(정규화 없이도 chance를 넘음)하면 드리프트가 약하다는 뜻이다. `sigma`를 키우는 것은 정당한 보정이지만 **키운 사실과 이유를 `run_logging.md`에 적는다.**
- **T3이 실패**(회복률 0.5 미만)하면 **임계를 낮추지 않는다.** 정규화 구현이나 특징이 세션 이동을 못 잡고 있다는 뜻이므로 §6.2 구현을 먼저 의심한다. 그래도 안 되면 실패로 보고하고 사용자 판단을 구한다 (CLAUDE.md §5.4 — 미달을 미달로 보고한다).

- [ ] **Step 6: 관측값을 스펙 §8.4.1에 기록한다**

`docs/specs/2026-08-19-session-baseline-design.md` §8.4 끝에 추가:

```markdown
### 8.4.1 파일럿 관측 (session_recovery.yaml, seed 42)

| 조건 | pooled accuracy | 95% CI | chance |
|---|---|---|---|
| cross_session · 정규화 off | (측정값) | (측정값) | 0.3333 |
| cross_session · 정규화 on | (측정값) | (측정값) | 0.3333 |
| within_subject (상한) | (측정값) | (측정값) | 0.3333 |
| T3 회복률 | (측정값) | — | 요구 ≥ 0.5 |

**임계를 절대 수치로 두지 않았다.** T2는 chance가 CI 안에 있는지로, T3는
`within_subject` 상한 대비 회복률로 판정한다. 둘 다 그 실행 자신이 만드는
기준점이므로 사전에 숫자를 지어낼 필요가 없다.
```

- [ ] **Step 7: 커밋**

```bash
git add config/experiments/session_recovery.yaml tests/evaluation/test_validation_session.py docs/specs/2026-08-19-session-baseline-design.md
git commit -m "test: 승인기준 T1(널)·T2(붕괴)·T3(회복)

T1은 그냥 두면 자동 통과하는 테스트다. 드리프트는 세션별 채널 이득인데
부하는 세션 안에서 무작위 배치되므로 예측 경로가 없다. 그래서 실제로
잡는 두 버그를 주입해 함께 검증한다 — 부하 조건부 정규화, 부하 의존
드리프트.

T2가 T3의 존재 이유다. T3만 있으면 '정규화하니 잘 되네'이고, T2가
있어야 정규화가 무엇을 고쳤는지가 된다.

임계는 파일럿 측정 후 확정했다. 관측값과 근거를 스펙 §8.4.1에 남긴다."
```

---

## Task 15: 승인기준 T4 (음성 대조군 — 연습 효과 생존)

**Files:**
- Create: `config/experiments/session_clean.yaml`
- Modify: `tests/evaluation/test_validation_session.py`
- Modify: `docs/specs/2026-08-19-session-baseline-design.md` (§8.4.1에 보존율 추가)

**Interfaces:**
- Consumes: Task 6 `generate_dataset`, Task 7 `make_windows`, Task 9 `SessionBaseline`
- Produces: 없음 (검증 전용)

**왜 분류기를 안 쓰는가:** 스펙 §8.2 — 분류기 출력은 단위가 다르고, **잘 작동하는 분류기일수록 약해진 신호를 올바른 클래스로 되돌려 추정 부하가 감소하지 않는다.** 정규화가 정상인데도 실패한다. 그래서 정규화된 특징을 직접 회귀한다.

- [ ] **Step 1: 기준 config를 만든다**

`config/experiments/session_clean.yaml` — `session_recovery.yaml`을 복사하고 아래만 바꾼다:

```yaml
run_name: session_clean
```

```yaml
  drift:
    fnirs_gain_sigma: 0.0
    fnirs_offset_sigma: 0.0
    eeg_gain_sigma: 0.0
    eeg_noise_sigma: 0.0
    within_session_rate: 0.0
    within_session_fraction: 0.33
    between_session_scale: 4.0
    assignment: sampled
```

```yaml
preprocessing:
  baseline:
    normalize: false
    drift_threshold_relative: 0.20
    zero_atol: 1.0e-8
```

`seed`는 42로 **그대로 둔다.** 난수열 규율(스펙 §5.5) 덕분에 드리프트를 꺼도 같은 난수열 위에서 돌므로, 이 실행이 `b_hat`의 정당한 기준이 된다.

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`tests/evaluation/test_validation_session.py`에 추가:

```python
# ---------------------------------------------------------------- T4

from collections import defaultdict

from src.datasets.windowing import make_windows
from src.preprocessing.baseline import SessionBaseline
from src.simulation.state import BASELINE, TASK

CLEAN = "config/experiments/session_clean.yaml"

#: 회귀 대상 부하 수준. nback_levels [0,2,3] 의 인덱스 2 = 3-back.
#: 가장 부하가 높은 조건이라 연습 효과가 가장 크게 나타난다.
T4_LEVEL = 2


def _practice_slope(config_path, *, normalize, normalizer=SessionBaseline):
    """세션 번호에 대한 정규화된 fNIRS HbO 평균의 기울기.

    스펙 §8.2가 정의한 m(s)를 계산하고 1차 다항식을 적합한다.
    """
    cfg = load_config(config_path)
    validate_config(cfg)
    rng = set_all_seeds(int(cfg["seed"]))
    n_ch = int(cfg["simulation"]["fnirs"]["n_channels"])
    win = cfg["windowing"]

    per_session = defaultdict(list)
    for rec in generate_dataset(cfg["simulation"], rng):
        windows = make_windows(rec.timeline, win["window_s"], win["step_s"])
        feats = extract_features(rec, windows)["fnirs"]

        if normalize:
            is_base = windows.block_kind == BASELINE
            start = is_base & (windows.trial_id == windows.trial_id.min())
            feats = normalizer.fit(feats[start], "concentration_delta").apply(feats)

        sel = (windows.block_kind == TASK) & (windows.load_level == T4_LEVEL)
        assert sel.any(), "3-back 창이 없으면 회귀할 대상이 없다"
        # fnirs 특징 배치: [HbO 평균 n_ch개 | HbO 기울기 n_ch개]
        per_session[rec.session_idx].append(float(feats[sel][:, :n_ch].mean()))

    xs = np.array(sorted(per_session), dtype=float)
    ys = np.array([np.mean(per_session[int(s)]) for s in xs])
    return float(np.polyfit(xs, ys, 1)[0])


def test_t4_practice_effect_survives_normalization(tmp_path):
    """정규화가 측정 드리프트를 지우면서 진짜 학습은 남겨야 한다."""
    b_ref = _practice_slope(CLEAN, normalize=False)
    b_hat = _practice_slope(RECOVERY, normalize=True)

    assert abs(b_ref) > 1e-9, (
        "기준 실행에서 연습 효과 기울기가 0이다 — practice.rate가 특징에 "
        "도달하지 않았다는 뜻이므로 보존율을 정의할 수 없다"
    )
    ratio = b_hat / b_ref
    assert T4_RATIO_MIN <= ratio <= T4_RATIO_MAX, (
        f"기울기 보존율 {ratio:.3f} 가 [{T4_RATIO_MIN}, {T4_RATIO_MAX}] 밖이다. "
        "낮으면 정규화가 진짜 인지 변화까지 지웠다는 뜻이고, 높으면 없던 "
        "변화를 만들어냈다는 뜻이다 — 둘 다 실패다."
    )


def test_t4_injection_over_normalization_kills_the_practice_effect():
    """결함 주입: 세션별 전체 z-score.

    평균만이 아니라 분산까지 세션마다 맞추면 세션 간 크기 차이가 통째로
    사라진다. 측정 드리프트와 함께 진짜 학습도 지워진다.
    """

    class _ZScoreBaseline(SessionBaseline):
        @classmethod
        def fit(cls, start_baseline, kind):
            return _ZScoreBaseline(reference=np.asarray(start_baseline).mean(axis=0), kind=kind)

        def apply(self, x):
            arr = np.asarray(x, dtype=float)
            sd = arr.std(axis=0)
            sd[sd == 0] = 1.0
            return (arr - arr.mean(axis=0)) / sd

    b_ref = _practice_slope(CLEAN, normalize=False)
    b_bad = _practice_slope(RECOVERY, normalize=True, normalizer=_ZScoreBaseline)
    assert abs(b_bad / b_ref) < T4_RATIO_MIN, (
        "과한 정규화를 주입했는데도 보존율이 유지됐다 — T4가 이 실패 양식을 못 잡는다"
    )
```

- [ ] **Step 3: 보존율 한계를 사전에 선언한다**

관측에서 역산하면 테스트가 자기 자신을 증명하는 동어반복이 된다. 두 경계는 **의미에서 나온다.**

```python
#: 기울기 보존율의 허용 범위. 관측에서 역산한 값이 아니라 의미에서 나온 선언이다.
#: 하한: 연습 효과의 절반 미만만 남았다면 정규화가 진짜 인지 변화를 지운 것이다.
#: 상한: 1.5를 넘으면 정규화가 없던 변화를 만들어낸 것이다 (스펙 §8.2).
T4_RATIO_MIN = 0.5
T4_RATIO_MAX = 1.5
```

이 블록을 `tests/evaluation/test_validation_session.py`의 T4 섹션 상단에 넣는다.

- [ ] **Step 4: 테스트가 실패하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_validation_session.py -k t4 -v > task-15-testlog-red.txt 2>&1
```

Expected: `session_clean.yaml`이 아직 없으면 `FileNotFoundError`, 있으면 `_practice_slope` 미정의로 FAIL. Step 1·2를 먼저 마쳤다면 실제 보존율에 따라 PASS 또는 FAIL이 나온다.

**FAIL일 때의 규칙:**

- `b_ref ≈ 0`이면 **임계를 만지지 않는다.** `practice.rate`가 특징에 도달하지 않는다는 뜻이므로 Task 3의 배선을 확인한다.
- 보존율이 하한 미만이면 정규화가 과하다는 뜻이다. §6.2의 `concentration_delta`가 평균만 빼는지(분산까지 건드리지 않는지) 확인한다.
- 그래도 미달이면 **미달로 보고한다** (CLAUDE.md §5.4). 임계를 낮춰 맞추지 않는다.

- [ ] **Step 5: 테스트가 통과하는 것을 확인한다**

```bash
python -m pytest tests/evaluation/test_validation_session.py -k t4 -v > task-15-testlog-green.txt 2>&1
```

Expected: 2개 PASS.

- [ ] **Step 6: 관측된 보존율을 스펙 §8.4.1 표에 추가하고 커밋**

```bash
git add config/experiments/session_clean.yaml tests/evaluation/test_validation_session.py docs/specs/2026-08-19-session-baseline-design.md
git commit -m "test: 승인기준 T4 — 연습 효과 생존 (음성 대조군)

이 서브프로젝트의 존재 이유다. T2·T3만이면 '정규화가 측정 드리프트를
지운다'의 절반이고, T4가 '그러면서 진짜 학습은 안 지운다'를 맡는다.

분류기 출력을 쓰지 않는다. 단위가 다르고, 잘 작동하는 분류기일수록
약해진 신호를 올바른 클래스로 되돌려 추정 부하가 감소하지 않는다 —
정규화가 정상인데 T4가 실패한다. 정규화된 fNIRS 특징을 직접 회귀하고,
기준 기울기는 드리프트 off·정규화 off 실행에서 얻는다.

두 실행은 같은 시드를 쓴다. 난수열 규율 덕분에 드리프트를 꺼도 같은
난수열 위에서 돌므로 기준이 정당하다."
```

---

## Task 16: 승인기준 T5(누수) · T6(신뢰도 플래그)

**Files:**
- Create: `config/experiments/session_quality.yaml`
- Modify: `tests/evaluation/test_validation_session.py`

**Interfaces:**
- Consumes: Task 12의 `check_normalization_source`, Task 13의 `build_dataset` 3-튜플
- Produces: 없음 (검증 전용)

- [ ] **Step 1: T6 전용 config를 만든다**

`config/experiments/session_quality.yaml` — `session_recovery.yaml`을 복사하고 아래만 바꾼다:

```yaml
run_name: session_quality
```

```yaml
simulation:
  n_subjects: 4
  n_sessions: 4        # 2×2 배치에 4개 세션이 필요하다
```

```yaml
  drift:
    fnirs_gain_sigma: 0.20
    fnirs_offset_sigma: 0.10
    eeg_gain_sigma: 0.15
    eeg_noise_sigma: 0.15
    within_session_rate: 0.30
    within_session_fraction: 0.33
    between_session_scale: 4.0
    assignment: fixed_2x2
```

- [ ] **Step 2: T5·T6 테스트를 쓴다**

`tests/evaluation/test_validation_session.py`에 추가:

```python
# ---------------------------------------------------------------- T5 누수

from src.datasets.contract import LeakageError
from src.evaluation.harness import run_folds
from src.evaluation.splitters import CrossSessionSplitter

ALL_GUARDS = {
    "check_subject_overlap": True,
    "check_window_overlap": True,
    "check_session_overlap": True,
    "check_normalization_source": True,
}


def _dataset_with_baseline_opted_in():
    """베이스라인 창을 분석에 넣은 데이터셋.

    러너는 이 조합을 거부하므로(부하 타깃의 chance level이 바뀐다) 여기서는
    build_dataset을 직접 부르고 타깃을 accuracy로 바꿔 계약 객체만 얻는다.
    """
    cfg = load_config(RECOVERY)
    cfg = _deep_update(cfg, {
        "dataset": {"include_baseline": True, "targets": ["accuracy"]},
        "preprocessing": {"baseline": {"normalize": True}},
        "simulation": {"n_subjects": 3, "n_sessions": 2},
    })
    validate_config(cfg)
    rng = set_all_seeds(int(cfg["seed"]))
    dataset, _, _ = runner_mod.build_dataset(cfg, rng)
    return dataset


def test_t5_normalization_source_window_in_a_fold_is_rejected():
    dataset = _dataset_with_baseline_opted_in()
    with pytest.raises(LeakageError, match="normalization source"):
        run_folds(
            dataset, CrossSessionSplitter(),
            target="accuracy", modalities=["eeg", "fnirs", "behavior"],
            guards=ALL_GUARDS, seed=42,
        )


def test_t5_injection_disabling_the_guard_lets_it_through():
    """결함 주입: 가드를 끄면 통과해버려야 한다.

    통과하지 않으면 다른 무언가가 막고 있다는 뜻이고, 그러면 이 가드가
    실제로 무엇을 지키는지 알 수 없다.
    """
    dataset = _dataset_with_baseline_opted_in()
    results = run_folds(
        dataset, CrossSessionSplitter(),
        target="accuracy", modalities=["eeg", "fnirs", "behavior"],
        guards={**ALL_GUARDS, "check_normalization_source": False}, seed=42,
    )
    assert len(results) >= 2


def test_t5_fit_cannot_be_handed_another_session():
    """서명 자체가 다른 세션을 받지 못한다 (스펙 §6.1)."""
    import inspect

    params = list(inspect.signature(SessionBaseline.fit).parameters)
    assert params == ["start_baseline", "kind"]


# ---------------------------------------------------------------- T6 신뢰도

QUALITY = "config/experiments/session_quality.yaml"


def _qualities_by_session():
    cfg = load_config(QUALITY)
    rng = set_all_seeds(int(cfg["seed"]))
    _, _, qualities = runner_mod.build_dataset(cfg, rng)
    by_session: dict[int, list] = defaultdict(list)
    for q in qualities:
        by_session[q.session_idx].append(q)
    return by_session


def test_t6_flags_only_the_sessions_with_within_session_drift():
    """2×2 배치 (스펙 §8.3).

    세션 0: ①② 작음 · ③ 작음  → 플래그 ✗
    세션 1: ①② 작음 · ③ 큼    → 플래그 ✓
    세션 2: ①② 큼   · ③ 작음  → 플래그 ✗  ← 핵심 음성 칸
    세션 3: ①② 큼   · ③ 큼    → 플래그 ✓
    """
    by_session = _qualities_by_session()
    flagged = {s: [q.drift_flag for q in qs] for s, qs in by_session.items()}

    assert not any(flagged[0]), "세션 0(드리프트 없음)이 플래그됐다 — 오탐"
    assert all(flagged[1]), "세션 1(세션 내 드리프트 큼)이 플래그되지 않았다 — 미탐"
    assert not any(flagged[2]), (
        "세션 2가 플래그됐다. ①② 세션 간 드리프트는 정규화가 이미 처리하므로 "
        "플래그 사유가 아니다 — drift_flag가 세션 간 드리프트를 세션 내 "
        "드리프트로 오독하고 있다."
    )
    assert all(flagged[3]), "세션 3(세션 내 드리프트 큼)이 플래그되지 않았다 — 미탐"


def test_t6_injection_zero_threshold_destroys_specificity():
    """결함 주입: 임계를 0으로 낮추면 전부 플래그되어 특이도가 무너져야 한다."""
    cfg = load_config(QUALITY)
    cfg = _deep_update(cfg, {"preprocessing": {"baseline": {"drift_threshold_relative": 0.0}}})
    validate_config(cfg)
    rng = set_all_seeds(int(cfg["seed"]))
    _, _, qualities = runner_mod.build_dataset(cfg, rng)
    assert all(q.drift_flag for q in qualities)
```

- [ ] **Step 3: 실행하고 임계를 보정한다**

```bash
python -m pytest tests/evaluation/test_validation_session.py -k "t5 or t6" -v > task-16-testlog.txt 2>&1
```

T6이 실패하면 **테스트가 아니라 `drift_threshold_relative`를 보정한다.** 측정값을 먼저 본다:

```bash
python - <<'PY' >> task-16-testlog.txt 2>&1
from tests.evaluation.test_validation_session import _qualities_by_session
for s, qs in sorted(_qualities_by_session().items()):
    vals = [f"{m}={q.drift_by_modality[m]:.4f}" for q in qs[:1] for m in sorted(qs[0].drift_by_modality)]
    print(f"session {s}: flag={[q.drift_flag for q in qs]} {vals}")
PY
```

세션 1·3의 드리프트가 세션 0·2보다 확실히 큰데 임계가 그 사이에 없으면 임계를 옮긴다. **세션 2가 1·3만큼 크게 나오면 임계 문제가 아니다** — `compute_drift`가 세션 간 드리프트를 세션 내 드리프트로 재고 있다는 뜻이므로 §6.3 구현을 고친다.

확정한 임계를 `config/experiments/session_quality.yaml`과 스펙 §8.4.1에 반영한다.

- [ ] **Step 4: 전체 스위트를 돌린다**

```bash
python -m pytest tests -v > task-16-full-suite.txt 2>&1
```

Expected: 전부 PASS. 실패가 남으면 여기서 멈추고 고친다.

- [ ] **Step 5: 커밋**

```bash
git add config/experiments/session_quality.yaml tests/evaluation/test_validation_session.py docs/specs/2026-08-19-session-baseline-design.md
git commit -m "test: 승인기준 T5(누수)·T6(신뢰도 플래그)

T6의 민감도만 보면 순환이다. ③이 곧 세션 내 이득 드리프트이고
drift_flag는 시작↔종료 베이스라인 차이로 계산하므로 심은 것을 그대로
재는 셈이다. 실제로 묻는 것은 특이도이므로 2×2로 배치하고, '①② 큼 +
③ 작음'을 핵심 음성 칸으로 둔다 — 여기가 플래그되면 세션 간 드리프트를
세션 내 드리프트로 오독하고 있다는 뜻이다.

T5는 가드를 껐을 때 통과해버리는 것까지 확인한다. 통과하지 않으면 다른
무언가가 막고 있다는 뜻이라 이 가드가 무엇을 지키는지 알 수 없다."
```

---

## Task 17: 결과 기록 · 한계 명시

**Files:**
- Modify: `run_logging.md`
- Modify: `docs/specs/2026-08-19-session-baseline-design.md` (§12 확인)

**Interfaces:**
- Consumes: Task 14~16의 관측값
- Produces: 없음 (문서)

- [ ] **Step 1: `run_logging.md`에 B1 항목을 추가한다**

`.claude/skills/research-log` 규약대로 아래를 **전부** 적는다. 하나라도 빠지면 몇 달 뒤 그 숫자가 어디서 나왔는지 알 수 없다.

- 실행 config 경로와 seed
- git commit hash
- **CV 방식** (`cross_session` / `loso` / `within_subject`) — 절대 혼용 표기하지 않는다
- **chance level** (0.3333)
- T1~T6 각각의 관측값과 판정
- T2·T3·T4·T6의 확정 임계와 **그 근거**
- 최악 피험자 성능
- 결함 주입 6종이 각각 실제로 빨간불을 냈다는 사실

- [ ] **Step 2: 스펙 §12에 이번 실행에서 드러난 한계를 보탠다**

§12는 이미 6개 항목을 담고 있다. 구현 중 새로 알게 된 한계가 있으면 항목을 추가한다. 특히 확인할 것:

- T2·T3의 회복 폭이 드리프트 크기(config의 `sigma`)에 얼마나 민감한가 — 민감하다면 "이 크기의 드리프트에 대해서만 증명했다"를 명시
- T6의 2×2가 `between_session_scale=4.0`에 의존한다면 그 사실

- [ ] **Step 3: 결과 요약을 사용자에게 보고한다**

보고에는 반드시 포함한다:

- **chance level과 CV 방식을 모든 정확도 옆에** (CLAUDE.md §5.4)
- 목표 미달 항목이 있으면 미달로 보고한다. 임계를 낮춰 맞춘 경우 그 사실을 명시
- §12의 한계 — 특히 "B1의 정확도는 임시 특징 추출기 + 더미 로지스틱 회귀에서 나오므로 계획서 목표치와 나란히 놓을 수 없다"

- [ ] **Step 4: 커밋**

```bash
git add run_logging.md docs/specs/2026-08-19-session-baseline-design.md
git commit -m "docs: B1 실행 기록 — 승인기준 T1~T6 결과와 확정 임계

CV 방식·chance level·최악 피험자 성능·결함 주입 결과를 함께 기록한다.
임계는 파일럿 측정 후 확정했으며 근거를 함께 남긴다."
```

---

## 자체 검토 결과

**스펙 커버리지**

| 스펙 절 | 구현 태스크 |
|---|---|
| §4 확정 결정 (의존성·진입 수준·변동원·베이스라인 처리) | Task 1·8·9 (B2·B3 항목은 범위 밖으로 명시됨) |
| §5.1 층 분리 | Task 3·4·5 |
| §5.2 세션 구조 | Task 2 |
| §5.3 드리프트 모델 ①②③ | Task 4·5·6 |
| §5.4 연습 효과 ④ | Task 2·3·4 |
| §5.5 난수열 규율 | Task 4 Step 5 (결함 주입으로 검증) |
| §6.1 `SessionBaseline` | Task 9 |
| §6.2 `kind`별 규칙 | Task 9 |
| §6.3 신뢰도 플래그 | Task 9·13 |
| §6.4 누수 논증·가드 | Task 12·16 |
| §6.5 베이스라인 옵트인 | Task 8 |
| §7.1 결함 2건 | Task 10(trial_id)·Task 12(창 겹침) |
| §7.2 `SplitKeys` | Task 10·11 |
| §7.3 분할기 표 | Task 11 |
| §7.4 `cv_scheme` 분리 | Task 13 (`aggregate_runs`) |
| §8 승인기준 T1~T6 | Task 14·15·16 |
| §8.1 T1 결함 주입 | Task 14 Step 2 |
| §8.2 T4 기울기 보존율 | Task 15 |
| §8.3 T6 2×2 | Task 16 |
| §8.4 임계 파일럿 보정 | Task 14 Step 4 · Task 15 Step 4 · Task 16 Step 3 |
| §9 모듈 배치 | 전 태스크 |
| §9.1 P1 | Task 1 |
| §10 config 키 | Task 1 |
| §12 한계 | Task 17 |

**미포함(의도적):** §4의 fNIRS 진입 수준 이중화·`coupling_quality`·`qc_method`는 B2, PREP·ASR·아티팩트는 B3. §11의 미확정 사항은 확정하지 않는다.

**타입 일관성 확인**

- `SessionBaseline.fit(start_baseline, kind)` — Task 9 정의, Task 13·15·16에서 동일 호출
- `compute_drift(start, end, *, zero_atol)` — Task 9 정의, Task 13에서 동일 호출
- `build_dataset(cfg, rng) -> (dataset, n_dropped, qualities)` — Task 13 정의, Task 16에서 3-튜플로 언패킹
- `split(keys: SplitKeys)` — Task 10 정의, Task 11의 네 분할기·Task 10 스파이 테스트가 동일
- `check_window_overlap(6개 인자)` — Task 12 정의, harness가 동일 순서로 호출
- `MODALITY_KIND` 키는 `{"eeg","fnirs","behavior"}` — `features_minimal.extract_features`의 반환 키와 일치

**남은 판단 지점 (구현 중 사용자 확인이 필요할 수 있음)**

1. Task 14 Step 4에서 `T2_COLLAPSE_MAX < T3_RECOVERY_MIN`이 성립하지 않으면 — 드리프트를 키워 맞추지 말고 보고한다.
2. Task 15에서 `b_ref ≈ 0`이면 — 임계가 아니라 Task 3의 배선을 의심한다.
3. Task 16에서 세션 2가 세션 1·3만큼 드리프트가 크게 나오면 — 임계가 아니라 §6.3 구현을 고친다.
