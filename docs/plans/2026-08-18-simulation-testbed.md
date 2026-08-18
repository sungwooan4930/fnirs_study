# 합성 테스트베드 + LOSO 평가 프레임워크 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 합성 EEG·fNIRS·행동 신호를 생성하고, 누수를 구조적으로 차단하는 데이터셋 계약 위에서 LOSO-CV를 실행하여, 누수 가드가 실제로 작동함을 T1~T4 테스트로 증명한다.

**Architecture:** 세 계층으로 나뉜다. `simulation/`은 피험자 랜덤효과 → 6차원 인지상태 궤적 → 신호 성분(가산 결합) 순으로 합성 데이터를 만든다. `datasets/`는 윈도잉·특징·라벨을 거쳐 `WindowedDataset` 계약을 구성하며, 이 계약은 `iter_folds()`를 통해서만 데이터를 노출해 전역 fit을 타입 수준에서 막는다. `evaluation/`은 fold를 실행하고 가드를 걸고 지표를 기록한다.

**Tech Stack:** Python 3.11 · numpy · scipy · scikit-learn · PyYAML · pytest

**Spec:** `docs/specs/2026-08-18-simulation-testbed-design.md`

## Global Constraints

프로젝트 규약은 `CLAUDE.md`. 아래는 모든 작업에 암묵적으로 적용된다.

- **Python 3.11.9**, venv는 `D:/Study_fNIRS/.venv`. 모든 명령은 `.venv/Scripts/python.exe` 또는 `.venv/Scripts/pytest.exe`로 실행한다.
- **MNE·PyTorch는 설치하지 않는다.** 이 서브프로젝트(A+D)는 numpy·scipy·scikit-learn·PyYAML만 쓴다. MNE는 B, PyTorch는 E에서 도입한다.
- **참가자 수를 하드코딩하지 않는다.** 계획서 내부 불일치(100명 vs 30명)가 미해결이므로 `n_subjects`는 config 값으로만 다룬다.
- **모든 난수는 명시적 `np.random.Generator`를 통해서만 생성한다.** 전역 `np.random.*` 호출 금지. 시드 미지정 시 실행을 거부한다.
- **조용한 실패 금지.** 경고 대신 예외를 던진다. 연구 코드에서 경고는 무시되고, 무시된 경고는 논문에 실린다.
- **정확도를 보고할 때는 항상 chance level과 CV 방식을 함께 기록한다.** LOSO 결과와 within-subject 결과를 혼용 표기하지 않는다.
- 커밋 메시지는 conventional commits 한국어 (`feat:`, `test:`, `fix:`, `chore:`).
- `legacy_4ch/`의 코드를 import하지 않는다. 참조만 허용.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/common/seeding.py` | 시드 고정, `Generator` 생성 |
| `src/common/config.py` | YAML 로드, 미정의 키 거부, 필수 키 검증 |
| `src/simulation/subject.py` | `SubjectProfile` — 피험자 랜덤효과 θ_s |
| `src/simulation/state.py` | `CognitiveStateTimeline` — 6차원 상태 궤적, 자극 온셋 |
| `src/simulation/components/base.py` | `SignalComponent` 프로토콜 |
| `src/simulation/components/eeg_oscillation.py` | 1/f 배경 + θ·α 진동 (부하에 변조) |
| `src/simulation/components/eeg_erp.py` | P300·N200 템플릿 (자극 정렬) |
| `src/simulation/components/fnirs_hrf.py` | 신경활성 → HRF 컨볼루션 → HbO/HbR |
| `src/simulation/components/behavior.py` | 정답확률·RT (상태를 Δ만큼 선행 관측) |
| `src/simulation/recording.py` | `SyntheticRecording` 조립 + BIDS 메타 라이터 |
| `src/datasets/contract.py` | `WindowedDataset`·`FoldView`·`TrainView`·`TestView`·`LeakageError` |
| `src/datasets/windowing.py` | 5초창·1초스텝, 블록 경계 넘는 창 제외 |
| `src/datasets/features_minimal.py` | 임시 특징 추출 (B 완성 후 교체) |
| `src/datasets/labels.py` | 라벨 구성 + 선행 타깃 (t+Δ) |
| `src/evaluation/splitters.py` | `loso` · `within_subject` · `window_random`(누수 시연용) |
| `src/evaluation/guards.py` | subject 중복·윈도우 시간 겹침 검사 |
| `src/evaluation/harness.py` | fold 실행, 가드 호출, 모델 학습·예측 |
| `src/evaluation/metrics.py` | 집계, chance, 이항검정, 최악 피험자 |
| `src/evaluation/runner.py` | config → 오케스트레이션 → 결과 디렉토리 |

---

### Task 1: 프로젝트 골격 · 의존성 · 시드 유틸

**Files:**
- Modify: `pytest.ini`
- Create: `src/__init__.py`, `src/common/__init__.py`, `src/common/seeding.py`
- Test: `tests/common/test_seeding.py`

**Interfaces:**
- Consumes: 없음 (첫 작업)
- Produces: `set_all_seeds(seed: int) -> np.random.Generator` — 시드를 고정하고 Generator를 반환. 이후 모든 작업이 이 함수로 난수원을 얻는다.

- [ ] **Step 1: 의존성 설치**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pip.exe install "scikit-learn>=1.5"
```

Expected: 성공. MNE·PyTorch는 설치하지 않는다 (B·E에서 도입).
pandas도 아직 설치하지 않는다 — 러너는 stdlib `csv`로 쓰고, 여러 실행을 가로질러
집계할 필요가 생길 때 도입한다.

- [ ] **Step 2: pytest.ini 수정**

기존 `testpaths = tests`가 가리키는 `tests/`는 `legacy_4ch/tests/`로 이동해 존재하지 않는다.

```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    slow: 오래 걸리는 통합 검증 테스트 (T1~T4)
```

- [ ] **Step 3: 패키지 골격 생성**

```bash
cd /d/Study_fNIRS
mkdir -p src/common src/simulation/components src/datasets src/evaluation
mkdir -p tests/common tests/simulation tests/datasets tests/evaluation
touch src/__init__.py src/common/__init__.py src/simulation/__init__.py
touch src/simulation/components/__init__.py src/datasets/__init__.py src/evaluation/__init__.py
```

- [ ] **Step 4: 실패하는 테스트 작성**

`tests/common/test_seeding.py`:

```python
import numpy as np
import pytest

from src.common.seeding import set_all_seeds


def test_same_seed_gives_same_numbers():
    g1 = set_all_seeds(42)
    g2 = set_all_seeds(42)
    assert np.array_equal(g1.normal(size=10), g2.normal(size=10))


def test_different_seed_gives_different_numbers():
    g1 = set_all_seeds(42)
    g2 = set_all_seeds(43)
    assert not np.array_equal(g1.normal(size=10), g2.normal(size=10))


def test_rejects_non_integer_seed():
    with pytest.raises(TypeError):
        set_all_seeds("42")


def test_rejects_negative_seed():
    with pytest.raises(ValueError):
        set_all_seeds(-1)
```

- [ ] **Step 5: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/common/test_seeding.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.common.seeding'`

- [ ] **Step 6: 최소 구현**

`src/common/seeding.py`:

```python
"""시드 고정. 모든 난수는 여기서 만든 Generator를 통해서만 생성한다."""

from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> np.random.Generator:
    """시드를 고정하고 numpy Generator를 반환한다.

    전역 np.random과 random도 함께 고정하지만, 프로젝트 코드는 반환된
    Generator만 사용해야 한다. 전역 고정은 서드파티가 내부적으로 전역
    상태를 쓰는 경우를 위한 방어일 뿐이다.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
```

- [ ] **Step 7: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/common/test_seeding.py -v
```

Expected: 4 passed

- [ ] **Step 8: 커밋**

```bash
cd /d/Study_fNIRS
git add pytest.ini src/ tests/
git commit -m "feat: 프로젝트 골격 + 시드 유틸

pytest.ini가 legacy로 이동한 tests/를 가리키던 문제 수정.
slow 마커 등록 (T1~T4 통합 검증용)."
```

---

### Task 2: config 로더 (엄격 키 검사)

**Files:**
- Create: `src/common/config.py`, `config/experiments/smoke.yaml`
- Test: `tests/common/test_config.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `class ConfigError(Exception)`
  - `load_config(path: str | Path) -> dict` — 스키마에 없는 키가 있으면 `ConfigError`, `seed` 누락 시 `ConfigError`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/common/test_config.py`:

```python
import textwrap

import pytest

from src.common.config import ConfigError, load_config

MINIMAL = """
run_name: t
seed: 1
simulation:
  n_subjects: 2
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 20
    n_blocks_per_level: 1
    stim_interval_s: 2.0
  eeg: {n_channels: 4, sfreq_hz: 100}
  fnirs: {n_channels: 4, sfreq_hz: 10.4, hbr_coupling: -0.33}
windowing: {window_s: 5.0, step_s: 1.0}
features: {extractor: minimal}
dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]
evaluation:
  splitter: loso
  model: logistic_regression
  guards: {check_subject_overlap: true, check_window_overlap: true}
output: {results_dir: results}
"""


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_loads_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg["seed"] == 1
    assert cfg["simulation"]["n_subjects"] == 2


def test_rejects_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, MINIMAL + "\nbogus: 1\n"))


def test_rejects_unknown_nested_key(tmp_path):
    bad = MINIMAL.replace("  n_subjects: 2", "  n_subjects: 2\n  typo_key: 9")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_seed(tmp_path):
    bad = MINIMAL.replace("seed: 1\n", "")
    with pytest.raises(ConfigError, match="seed"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_required_section(tmp_path):
    bad = MINIMAL.replace("windowing: {window_s: 5.0, step_s: 1.0}\n", "")
    with pytest.raises(ConfigError, match="windowing"):
        load_config(_write(tmp_path, bad))
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/common/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.common.config'`

- [ ] **Step 3: 최소 구현**

`src/common/config.py`:

```python
"""실험 config 로더.

스키마에 정의되지 않은 키를 만나면 실행을 거부한다. 오타가 조용히
기본값으로 흡수되어 재현 불가능한 결과로 이어지는 것을 막기 위함이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """config가 스키마를 위반했을 때 발생."""


# dict면 하위 스키마, None이면 리프(값 검사는 하지 않음)
SCHEMA: dict[str, Any] = {
    "run_name": None,
    "seed": None,
    "simulation": {
        "n_subjects": None,
        "subject_variance": None,
        "effect_size": None,
        "lead_delta_s": None,
        "task": {
            "nback_levels": None,
            "block_duration_s": None,
            "n_blocks_per_level": None,
            "stim_interval_s": None,
        },
        "eeg": {"n_channels": None, "sfreq_hz": None},
        "fnirs": {"n_channels": None, "sfreq_hz": None, "hbr_coupling": None},
    },
    "windowing": {"window_s": None, "step_s": None},
    "features": {"extractor": None},
    "dataset": {
        "targets": None,
        "lead_targets": None,
        "modalities": None,
        "rt_bins": None,
    },
    "evaluation": {
        "splitter": None,
        "model": None,
        "guards": {"check_subject_overlap": None, "check_window_overlap": None},
    },
    "output": {"results_dir": None},
}

REQUIRED_TOP = (
    "run_name", "seed", "simulation", "windowing",
    "features", "dataset", "evaluation", "output",
)


def _check_node(node: Any, schema: Any, path: str) -> None:
    if schema is None:
        return
    if not isinstance(node, dict):
        raise ConfigError(f"{path or 'root'}: expected a mapping, got {type(node).__name__}")
    for key, value in node.items():
        if key not in schema:
            raise ConfigError(f"unknown key '{path}{key}'")
        _check_node(value, schema[key], f"{path}{key}.")


def load_config(path: str | Path) -> dict:
    """YAML config를 읽고 스키마를 검증해 반환한다."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    _check_node(cfg, SCHEMA, "")

    for key in REQUIRED_TOP:
        if key not in cfg:
            raise ConfigError(f"missing required key '{key}'")

    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ConfigError(f"seed must be int, got {type(cfg['seed']).__name__}")

    return cfg
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/common/test_config.py -v
```

Expected: 5 passed

- [ ] **Step 5: smoke config 작성**

`config/experiments/smoke.yaml` — 개발·테스트용 축소판. 파일럿 config는 Task 19에서 만든다.

```yaml
run_name: smoke
seed: 42

simulation:
  n_subjects: 6
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 30
    n_blocks_per_level: 2
    stim_interval_s: 2.0
  eeg:   {n_channels: 30, sfreq_hz: 250}
  fnirs: {n_channels: 48, sfreq_hz: 10.4, hbr_coupling: -0.33}

windowing:
  window_s: 5.0
  step_s: 1.0

features:
  extractor: minimal

dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy, response_latency]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]

evaluation:
  splitter: loso
  model: logistic_regression
  guards:
    check_subject_overlap: true
    check_window_overlap: true

output:
  results_dir: results
```

> **주의:** `eeg.sfreq_hz`가 계획서의 1000 Hz가 아니라 250 Hz다. smoke는 개발 속도를 위한 축소판이며 Task 19의 파일럿 config가 1000 Hz를 쓴다.

- [ ] **Step 6: 커밋**

```bash
cd /d/Study_fNIRS
git add src/common/config.py tests/common/test_config.py config/
git commit -m "feat: 엄격 config 로더

스키마에 없는 키를 만나면 ConfigError로 실행 거부.
오타가 조용히 기본값으로 흡수되는 것을 방지."
```

---

### Task 3: SubjectProfile (피험자 랜덤효과)

**Files:**
- Create: `src/simulation/subject.py`
- Test: `tests/simulation/test_subject.py`

**Interfaces:**
- Consumes: `set_all_seeds` (Task 1)
- Produces:
  - `@dataclass(frozen=True) SubjectProfile` — 필드 `subject_id: str`, `theta: float`
  - `make_subjects(n_subjects: int, subject_variance: float, rng: np.random.Generator) -> list[SubjectProfile]`

`theta`는 계획서 가설 2의 신경효율성 개인차다. 같은 인지부하에서도 피험자마다 뇌 반응 크기가 다르게 만드는 값이며, 이 값이 0이면 LOSO와 within-subject 성능 차이가 사라진다(T4가 이를 검증한다).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_subject.py`:

```python
import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.subject import SubjectProfile, make_subjects


def test_makes_requested_number():
    rng = set_all_seeds(0)
    subs = make_subjects(7, 0.5, rng)
    assert len(subs) == 7
    assert all(isinstance(s, SubjectProfile) for s in subs)


def test_subject_ids_are_zero_padded_and_unique():
    rng = set_all_seeds(0)
    subs = make_subjects(3, 0.5, rng)
    assert [s.subject_id for s in subs] == ["sub-01", "sub-02", "sub-03"]


def test_zero_variance_gives_identical_thetas():
    rng = set_all_seeds(0)
    subs = make_subjects(5, 0.0, rng)
    assert all(s.theta == 0.0 for s in subs)


def test_larger_variance_spreads_thetas_more():
    thetas_small = np.array([s.theta for s in make_subjects(200, 0.25, set_all_seeds(1))])
    thetas_large = np.array([s.theta for s in make_subjects(200, 4.0, set_all_seeds(1))])
    assert thetas_large.std() > thetas_small.std()


def test_profile_is_immutable():
    import dataclasses
    import pytest

    sub = make_subjects(1, 0.5, set_all_seeds(0))[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        sub.theta = 99.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_subject.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.subject'`

- [ ] **Step 3: 최소 구현**

`src/simulation/subject.py`:

```python
"""피험자 랜덤효과.

theta는 계획서 가설 2의 "신경효율성 개인차"다. 동일한 인지부하에서도
피험자마다 전전두엽 활성 크기가 다른 현상을 하나의 스칼라로 요약한다.
theta가 0이면 모든 피험자가 동일하게 반응하므로 LOSO가 within-subject만큼
쉬워진다. T4가 이 성질을 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectProfile:
    subject_id: str
    theta: float


def make_subjects(
    n_subjects: int,
    subject_variance: float,
    rng: np.random.Generator,
) -> list[SubjectProfile]:
    """피험자 프로파일 목록을 만든다.

    theta ~ N(0, subject_variance). subject_variance는 분산이므로
    표준편차는 sqrt를 취한다.
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    if subject_variance < 0:
        raise ValueError(f"subject_variance must be >= 0, got {subject_variance}")

    sd = float(np.sqrt(subject_variance))
    thetas = rng.normal(0.0, sd, size=n_subjects) if sd > 0 else np.zeros(n_subjects)

    return [
        SubjectProfile(subject_id=f"sub-{i + 1:02d}", theta=float(t))
        for i, t in enumerate(thetas)
    ]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_subject.py -v
```

Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/subject.py tests/simulation/test_subject.py
git commit -m "feat: SubjectProfile — 피험자 랜덤효과(신경효율성 개인차)"
```

---

### Task 4: CognitiveStateTimeline (인지상태 궤적)

**Files:**
- Create: `src/simulation/state.py`
- Test: `tests/simulation/test_state.py`

**Interfaces:**
- Consumes: `set_all_seeds` (Task 1)
- Produces:
  - `STATE_SFREQ: float = 10.0` — 상태 궤적 샘플링 레이트
  - `@dataclass(frozen=True) CognitiveStateTimeline` — 필드 `t`, `load_level`, `fatigue`, `trial_id`, `stim_onsets`, `duration_s`; 메서드 `load_at(t)`, `trial_at(t)`
  - `build_timeline(task_cfg: dict, rng: np.random.Generator) -> CognitiveStateTimeline`

`load_level`은 `nback_levels` 리스트의 **인덱스**(0/1/2)이지 n-back 수준값(0/2/3)이 아니다. 분류 라벨로 그대로 쓰기 위함이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_state.py`:

```python
import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.state import STATE_SFREQ, build_timeline

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def test_duration_is_blocks_times_duration():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert tl.duration_s == pytest.approx(3 * 2 * 30)


def test_samples_at_state_sfreq():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert len(tl.t) == int(tl.duration_s * STATE_SFREQ)
    assert tl.t[1] - tl.t[0] == pytest.approx(1.0 / STATE_SFREQ)


def test_all_load_levels_appear_equally_often():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    counts = np.bincount(tl.load_level, minlength=3)
    assert counts[0] == counts[1] == counts[2]


def test_block_order_is_shuffled_not_sorted():
    # 카운터밸런스: 블록 순서가 0,0,1,1,2,2 처럼 정렬돼 있으면 안 된다
    orders = set()
    for seed in range(10):
        tl = build_timeline(TASK_CFG, set_all_seeds(seed))
        first_of_each_block = tl.load_level[:: int(30 * STATE_SFREQ)]
        orders.add(tuple(first_of_each_block))
    assert len(orders) > 1


def test_fatigue_increases_monotonically():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert np.all(np.diff(tl.fatigue) >= 0)
    assert tl.fatigue[0] == pytest.approx(0.0)
    assert tl.fatigue[-1] < 1.0 + 1e-9


def test_trial_id_is_one_per_block():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert len(np.unique(tl.trial_id)) == 6


def test_stim_onsets_spaced_by_interval():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    gaps = np.diff(tl.stim_onsets)
    assert np.allclose(gaps, 2.0)
    assert tl.stim_onsets[-1] < tl.duration_s


def test_load_at_uses_step_interpolation():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    # 블록 중앙에서 조회한 값은 그 블록의 라벨과 같아야 한다
    mid_of_first_block = 15.0
    assert tl.load_at(np.array([mid_of_first_block]))[0] == tl.load_level[0]


def test_load_at_clamps_beyond_end():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert tl.load_at(np.array([tl.duration_s + 100.0]))[0] == tl.load_level[-1]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_state.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.state'`

- [ ] **Step 3: 최소 구현**

`src/simulation/state.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_state.py -v
```

Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/state.py tests/simulation/test_state.py
git commit -m "feat: CognitiveStateTimeline — 인지상태 궤적 + 카운터밸런스 블록"
```

---

### Task 5: SignalComponent 프로토콜 + EEG 진동 성분

**Files:**
- Create: `src/simulation/components/base.py`, `src/simulation/components/eeg_oscillation.py`
- Test: `tests/simulation/test_eeg_oscillation.py`

**Interfaces:**
- Consumes: `CognitiveStateTimeline` (Task 4), `SubjectProfile` (Task 3)
- Produces:
  - `base.py`: `N_MODULATED_EEG: int = 10`, `THETA_COUPLING: float = 1.0`, `ALPHA_COUPLING: float = 0.8`, `HBO_COUPLING: float = 0.8`, `BEHAV_COUPLING: float = 0.6`, `class SignalComponent(Protocol)`
  - `eeg_oscillation.py`: `generate_eeg_oscillation(timeline, subject, rng, *, sfreq, n_channels, effect_size) -> np.ndarray` shape `(n_channels, n_samples)`

결합 상수는 스펙 §6.2의 "성분별 기여 비율은 코드에 고정하고 `effect_size`가 전체를 스케일한다"를 구현한다. 앞 `N_MODULATED_EEG`개 채널만 부하에 반응하고 나머지는 배경 잡음만 갖는다 — 모든 채널이 반응하면 실제보다 쉬운 문제가 된다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_eeg_oscillation.py`:

```python
import numpy as np
import pytest
from scipy import signal as sp_signal

from src.common.seeding import set_all_seeds
from src.simulation.components.base import N_MODULATED_EEG
from src.simulation.components.eeg_oscillation import generate_eeg_oscillation
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}
SFREQ = 250.0


def _make(effect_size, seed=0, n_channels=30):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    sig = generate_eeg_oscillation(
        tl, sub, rng, sfreq=SFREQ, n_channels=n_channels, effect_size=effect_size
    )
    return tl, sig


def _band_power(x, sfreq, lo, hi):
    f, pxx = sp_signal.welch(x, fs=sfreq, nperseg=min(512, len(x)))
    return pxx[(f >= lo) & (f < hi)].sum()


def test_output_shape():
    _, sig = _make(0.8)
    expected_samples = int(round(180.0 * SFREQ))
    assert sig.shape == (30, expected_samples)


def test_no_nans():
    _, sig = _make(0.8)
    assert np.isfinite(sig).all()


def test_theta_power_increases_with_load():
    tl, sig = _make(0.8)
    ch = sig[0]
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    theta_low = _band_power(ch[low_mask], SFREQ, 4, 8)
    theta_high = _band_power(ch[high_mask], SFREQ, 4, 8)
    assert theta_high > theta_low


def test_alpha_power_decreases_with_load():
    tl, sig = _make(0.8)
    ch = sig[0]
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    alpha_low = _band_power(ch[low_mask], SFREQ, 8, 13)
    alpha_high = _band_power(ch[high_mask], SFREQ, 8, 13)
    assert alpha_high < alpha_low


def test_zero_effect_size_removes_load_modulation():
    tl, sig = _make(0.0)
    ch = sig[0]
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    theta_low = _band_power(ch[low_mask], SFREQ, 4, 8)
    theta_high = _band_power(ch[high_mask], SFREQ, 4, 8)
    assert theta_high == pytest.approx(theta_low, rel=0.35)


def test_unmodulated_channels_have_no_load_effect():
    tl, sig = _make(0.8)
    ch = sig[N_MODULATED_EEG]  # 변조 대상 밖 첫 채널
    low_mask = np.repeat(tl.load_level == 0, int(SFREQ / 10))
    high_mask = np.repeat(tl.load_level == 2, int(SFREQ / 10))
    theta_low = _band_power(ch[low_mask], SFREQ, 4, 8)
    theta_high = _band_power(ch[high_mask], SFREQ, 4, 8)
    assert theta_high == pytest.approx(theta_low, rel=0.35)


def test_reproducible_for_same_seed():
    _, a = _make(0.8, seed=7)
    _, b = _make(0.8, seed=7)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_eeg_oscillation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.components.base'`

- [ ] **Step 3: base.py 구현**

`src/simulation/components/base.py`:

```python
"""신호 성분 공통 정의.

결합 상수는 스펙 §6.2를 구현한다. 성분별 기여 비율을 코드에 고정하고
config의 effect_size가 그 전체를 스케일한다. 이렇게 두면 "effect_size를
키우면 모든 모달이 함께 강해진다"는 단순한 해석이 가능해진다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# 부하에 반응하는 EEG 채널 수. 실제로도 전전두 일부 채널만 반응하므로
# 전 채널을 변조하면 실제보다 쉬운 문제가 된다.
N_MODULATED_EEG: int = 10

# 성분별 기여 비율 (스펙 §6.2)
THETA_COUPLING: float = 1.0
ALPHA_COUPLING: float = 0.8
HBO_COUPLING: float = 0.8
BEHAV_COUPLING: float = 0.6


class SignalComponent(Protocol):
    """신호 성분의 호출 규약.

    모든 성분은 인지상태 궤적과 피험자 프로파일을 받아
    (n_channels, n_samples) 배열을 만든다.
    """

    def __call__(
        self,
        timeline,
        subject,
        rng: np.random.Generator,
        *,
        sfreq: float,
        n_channels: int,
        effect_size: float,
    ) -> np.ndarray: ...


def load_fraction(load_level: np.ndarray, n_levels: int = 3) -> np.ndarray:
    """부하 인덱스(0..n-1)를 0..1 비율로 바꾼다."""
    return load_level.astype(float) / max(n_levels - 1, 1)


def pink_noise(n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """1/f 배경 활동. 주파수 영역에서 1/sqrt(f)로 스케일해 생성한다."""
    white = rng.normal(size=(n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    return np.fft.irfft(spectrum * scale, n=n_samples, axis=1)
```

- [ ] **Step 4: eeg_oscillation.py 구현**

`src/simulation/components/eeg_oscillation.py`:

```python
"""EEG 진동 성분: 1/f 배경 + θ·α 진동.

인지부하가 오르면 전두 θ가 증가하고 두정 α가 감소한다는 계획서 §2.3의
후보 지표를 구현한다. 피험자의 theta(신경효율성)가 반응 크기를 개인별로
바꾼다.
"""

from __future__ import annotations

import numpy as np

from src.simulation.components.base import (
    ALPHA_COUPLING,
    N_MODULATED_EEG,
    THETA_COUPLING,
    load_fraction,
    pink_noise,
)

THETA_HZ = 6.0
ALPHA_HZ = 10.0
BASE_AMPLITUDE = 1.0
BACKGROUND_SCALE = 0.5


def generate_eeg_oscillation(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    sfreq: float,
    n_channels: int,
    effect_size: float,
) -> np.ndarray:
    """(n_channels, n_samples) EEG 신호를 만든다."""
    n_samples = int(round(timeline.duration_s * sfreq))
    t = np.arange(n_samples) / sfreq

    signal = BACKGROUND_SCALE * pink_noise(n_samples, n_channels, rng)

    load = load_fraction(timeline.load_at(t))
    # 개인차: theta가 클수록 같은 부하에 더 크게 반응한다
    gain = 1.0 + subject.theta

    theta_amp = BASE_AMPLITUDE * (1.0 + effect_size * THETA_COUPLING * gain * load)
    alpha_amp = BASE_AMPLITUDE * (1.0 - effect_size * ALPHA_COUPLING * gain * load)
    alpha_amp = np.clip(alpha_amp, 0.05, None)

    n_mod = min(N_MODULATED_EEG, n_channels)
    for ch in range(n_mod):
        phase_t = rng.uniform(0, 2 * np.pi)
        phase_a = rng.uniform(0, 2 * np.pi)
        signal[ch] += theta_amp * np.sin(2 * np.pi * THETA_HZ * t + phase_t)
        signal[ch] += alpha_amp * np.sin(2 * np.pi * ALPHA_HZ * t + phase_a)

    return signal
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_eeg_oscillation.py -v
```

Expected: 7 passed

- [ ] **Step 6: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/components/ tests/simulation/test_eeg_oscillation.py
git commit -m "feat: EEG 진동 성분 — 1/f 배경 + 부하 변조 θ·α

앞 10채널만 부하에 반응. 전 채널 변조는 실제보다 쉬운 문제가 됨."
```

---

### Task 6: EEG ERP 성분 (P300 · N200)

**Files:**
- Create: `src/simulation/components/eeg_erp.py`
- Test: `tests/simulation/test_eeg_erp.py`

**Interfaces:**
- Consumes: `CognitiveStateTimeline.stim_onsets`, `load_at` (Task 4); `N_MODULATED_EEG`, `load_fraction` (Task 5)
- Produces: `generate_eeg_erp(timeline, subject, rng, *, sfreq, n_channels, effect_size) -> np.ndarray` shape `(n_channels, n_samples)`

계획서 §2.3의 ERP 후보 지표를 구현한다. 인지부하가 오르면 P300 진폭이 감소한다(자원 고갈).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_eeg_erp.py`:

```python
import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.components.base import N_MODULATED_EEG
from src.simulation.components.eeg_erp import P300_LATENCY_S, generate_eeg_erp
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}
SFREQ = 250.0


def _make(effect_size, seed=0):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    sig = generate_eeg_erp(tl, sub, rng, sfreq=SFREQ, n_channels=30, effect_size=effect_size)
    return tl, sig


def _mean_erp_at_p300(tl, sig, level):
    onsets = tl.stim_onsets[tl.load_at(tl.stim_onsets) == level]
    idx = np.round((onsets + P300_LATENCY_S) * SFREQ).astype(int)
    idx = idx[idx < sig.shape[1]]
    return sig[0, idx].mean()


def test_output_shape():
    _, sig = _make(0.8)
    assert sig.shape == (30, int(round(180.0 * SFREQ)))


def test_no_nans():
    _, sig = _make(0.8)
    assert np.isfinite(sig).all()


def test_signal_is_near_zero_between_stimuli():
    tl, sig = _make(0.8)
    # 자극 사이 간격이 2초, ERP는 1초 내에 끝나므로 1.8초 지점은 거의 0
    idx = np.round((tl.stim_onsets[:-1] + 1.8) * SFREQ).astype(int)
    assert np.abs(sig[0, idx]).max() < 0.05


def test_p300_is_positive_deflection():
    tl, sig = _make(0.8)
    assert _mean_erp_at_p300(tl, sig, 0) > 0


def test_p300_amplitude_decreases_with_load():
    tl, sig = _make(0.8)
    assert _mean_erp_at_p300(tl, sig, 2) < _mean_erp_at_p300(tl, sig, 0)


def test_zero_effect_size_removes_load_modulation():
    tl, sig = _make(0.0)
    low = _mean_erp_at_p300(tl, sig, 0)
    high = _mean_erp_at_p300(tl, sig, 2)
    assert abs(high - low) < 1e-9


def test_unmodulated_channels_are_silent():
    _, sig = _make(0.8)
    assert np.abs(sig[N_MODULATED_EEG:]).max() == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_eeg_erp.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.components.eeg_erp'`

- [ ] **Step 3: 구현**

`src/simulation/components/eeg_erp.py`:

```python
"""ERP 성분: P300 · N200.

계획서 §2.3의 ERP 후보 지표. 인지부하가 오르면 처리 자원이 줄어
P300 진폭이 감소한다. N200은 반대 부호의 초기 성분이다.
"""

from __future__ import annotations

import numpy as np

from src.simulation.components.base import N_MODULATED_EEG, load_fraction

N200_LATENCY_S = 0.20
P300_LATENCY_S = 0.30
N200_WIDTH_S = 0.04
P300_WIDTH_S = 0.06
N200_AMPLITUDE = -0.4
P300_AMPLITUDE = 1.0
LOAD_ATTENUATION = 0.6  # 최대 부하에서 P300이 이 비율만큼 줄어든다


def _gaussian(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def generate_eeg_erp(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    sfreq: float,
    n_channels: int,
    effect_size: float,
) -> np.ndarray:
    """자극 온셋에 정렬된 ERP를 (n_channels, n_samples)로 만든다."""
    n_samples = int(round(timeline.duration_s * sfreq))
    signal = np.zeros((n_channels, n_samples))

    # 자극당 1초짜리 ERP 템플릿을 겹쳐 놓는다
    template_len = int(round(1.0 * sfreq))
    t_local = np.arange(template_len) / sfreq

    n200 = N200_AMPLITUDE * _gaussian(t_local, N200_LATENCY_S, N200_WIDTH_S)
    p300_shape = _gaussian(t_local, P300_LATENCY_S, P300_WIDTH_S)

    loads = load_fraction(timeline.load_at(timeline.stim_onsets))
    gain = 1.0 + subject.theta
    n_mod = min(N_MODULATED_EEG, n_channels)

    for onset, load in zip(timeline.stim_onsets, loads):
        start = int(round(onset * sfreq))
        end = min(start + template_len, n_samples)
        if start >= n_samples:
            continue
        span = end - start

        attenuation = 1.0 - effect_size * LOAD_ATTENUATION * gain * load
        attenuation = float(np.clip(attenuation, 0.05, None))
        wave = n200[:span] + P300_AMPLITUDE * attenuation * p300_shape[:span]

        signal[:n_mod, start:end] += wave

    return signal
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_eeg_erp.py -v
```

Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/components/eeg_erp.py tests/simulation/test_eeg_erp.py
git commit -m "feat: ERP 성분 — 부하에 따라 감쇠하는 P300 + N200"
```

---

### Task 7: fNIRS HRF 성분

**Files:**
- Create: `src/simulation/components/fnirs_hrf.py`
- Test: `tests/simulation/test_fnirs_hrf.py`

**Interfaces:**
- Consumes: `CognitiveStateTimeline.load_at` (Task 4); `HBO_COUPLING`, `load_fraction` (Task 5)
- Produces: `generate_fnirs(timeline, subject, rng, *, sfreq, n_channels, effect_size, hbr_coupling) -> tuple[np.ndarray, np.ndarray]` — `(hbo, hbr)` 각각 `(n_channels, n_samples)`

**의도적 단순화 (스펙 §6.3):** `hbr = hbr_coupling * hbo + 잡음`. 실제 HbR은 진폭이 HbO의 약 1/3이면서 **시간 지연도 다르다**. 여기서는 지연 차이를 모델링하지 않는다. B에서 정교화한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_fnirs_hrf.py`:

```python
import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.components.fnirs_hrf import canonical_hrf, generate_fnirs
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}
SFREQ = 10.4


def _make(effect_size, seed=0, hbr_coupling=-0.33):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    hbo, hbr = generate_fnirs(
        tl, sub, rng, sfreq=SFREQ, n_channels=48,
        effect_size=effect_size, hbr_coupling=hbr_coupling,
    )
    return tl, hbo, hbr


def test_hrf_peaks_around_five_seconds():
    hrf = canonical_hrf(SFREQ)
    peak_s = np.argmax(hrf) / SFREQ
    assert 4.0 < peak_s < 7.0


def test_hrf_starts_near_zero():
    hrf = canonical_hrf(SFREQ)
    assert abs(hrf[0]) < 1e-6


def test_output_shapes_match():
    _, hbo, hbr = _make(0.8)
    assert hbo.shape == hbr.shape == (48, int(round(180.0 * SFREQ)))


def test_no_nans():
    _, hbo, hbr = _make(0.8)
    assert np.isfinite(hbo).all() and np.isfinite(hbr).all()


def test_hbo_increases_with_load():
    tl, hbo, _ = _make(0.8)
    t = np.arange(hbo.shape[1]) / SFREQ
    load = tl.load_at(t)
    # HRF 지연을 고려해 블록 후반부만 비교한다
    assert hbo[0][load == 2].mean() > hbo[0][load == 0].mean()


def test_hbo_hbr_are_negatively_correlated():
    _, hbo, hbr = _make(0.8)
    r = np.corrcoef(hbo[0], hbr[0])[0, 1]
    assert r < -0.5


def test_zero_effect_size_removes_load_modulation():
    tl, hbo, _ = _make(0.0)
    t = np.arange(hbo.shape[1]) / SFREQ
    load = tl.load_at(t)
    high = hbo[0][load == 2].mean()
    low = hbo[0][load == 0].mean()
    assert high == pytest.approx(low, abs=0.15)


def test_reproducible_for_same_seed():
    _, a, _ = _make(0.8, seed=3)
    _, b, _ = _make(0.8, seed=3)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_fnirs_hrf.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.components.fnirs_hrf'`

- [ ] **Step 3: 구현**

`src/simulation/components/fnirs_hrf.py`:

```python
"""fNIRS 성분: 신경활성 → HRF 컨볼루션 → HbO/HbR.

의도적 단순화 (스펙 §6.3):
- hbr = hbr_coupling * hbo + 독립잡음. 실제 HbR은 진폭이 HbO의 약 1/3이면서
  시간 지연도 다르지만, 여기서는 지연 차이를 모델링하지 않는다.
- 정준 HRF 하나만 쓴다. 실제 HRF는 개인·부위별로 다르다.
- 채널 간 공간 상관을 넣지 않는다.
이 단순화들은 테스트베드 목적(하네스 검증)에는 무해하다. 검증 대상은
신호의 현실성이 아니라 "심은 효과를 하네스가 정직하게 회수하는가"이기 때문이다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.simulation.components.base import HBO_COUPLING, load_fraction

HRF_DURATION_S = 32.0
NOISE_SD = 0.15


def canonical_hrf(sfreq: float) -> np.ndarray:
    """이중 감마 정준 혈류반응함수. 최대치가 1이 되도록 정규화한다."""
    t = np.arange(0.0, HRF_DURATION_S, 1.0 / sfreq)
    peak = stats.gamma.pdf(t, a=6.0)
    undershoot = stats.gamma.pdf(t, a=16.0)
    hrf = peak - undershoot / 6.0
    return hrf / np.max(hrf)


def generate_fnirs(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    sfreq: float,
    n_channels: int,
    effect_size: float,
    hbr_coupling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(hbo, hbr)을 각각 (n_channels, n_samples)로 만든다."""
    n_samples = int(round(timeline.duration_s * sfreq))
    t = np.arange(n_samples) / sfreq

    neural = load_fraction(timeline.load_at(t))
    hrf = canonical_hrf(sfreq)

    gain = 1.0 + subject.theta
    response = np.convolve(neural, hrf)[:n_samples]
    response = response / max(len(hrf) / 4.0, 1.0)  # 컨볼루션 누적을 정규화
    response = effect_size * HBO_COUPLING * gain * response

    hbo = np.tile(response, (n_channels, 1))
    hbo += rng.normal(0.0, NOISE_SD, size=hbo.shape)

    hbr = hbr_coupling * hbo + rng.normal(0.0, NOISE_SD * 0.5, size=hbo.shape)

    return hbo, hbr
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_fnirs_hrf.py -v
```

Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/components/fnirs_hrf.py tests/simulation/test_fnirs_hrf.py
git commit -m "feat: fNIRS HRF 성분 — 신경활성 컨볼루션 + HbO/HbR 음상관

의도적 단순화(HbR 지연 미모델링, 단일 HRF, 공간상관 없음)는 docstring에 명시."
```

---

### Task 8: 행동 성분 (선행 지연 Δ)

**Files:**
- Create: `src/simulation/components/behavior.py`
- Test: `tests/simulation/test_behavior.py`

**Interfaces:**
- Consumes: `CognitiveStateTimeline` (Task 4); `BEHAV_COUPLING`, `load_fraction` (Task 5)
- Produces:
  - `@dataclass(frozen=True) BehaviorLog` — 필드 `onsets: np.ndarray`, `correct: np.ndarray`, `rt: np.ndarray`
  - `generate_behavior(timeline, subject, rng, *, effect_size, lead_delta_s) -> BehaviorLog`

**핵심:** 온셋 `u`의 행동은 시각 `u - Δ`의 인지상태가 결정한다. 따라서 시각 `w`의 뇌신호는 `w + Δ`의 행동을 예측할 수 있다 — 이것이 계획서 가설 2(행동 오류 약 1.2초 전 선행 신호)의 구현이며, 동시점 예측이 일으키는 라벨 누수를 피하는 장치다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_behavior.py`:

```python
import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.components.behavior import generate_behavior
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _make(effect_size, lead_delta_s=1.2, seed=0):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    return tl, generate_behavior(tl, sub, rng, effect_size=effect_size, lead_delta_s=lead_delta_s)


def test_one_entry_per_stimulus():
    tl, log = _make(0.8)
    assert len(log.onsets) == len(tl.stim_onsets)
    assert len(log.correct) == len(log.rt) == len(log.onsets)


def test_correct_is_binary():
    _, log = _make(0.8)
    assert set(np.unique(log.correct)).issubset({0, 1})


def test_rt_is_positive():
    _, log = _make(0.8)
    assert (log.rt > 0).all()


def test_accuracy_drops_with_load_measured_at_lead_time():
    tl, log = _make(0.8, lead_delta_s=1.2)
    driving_load = tl.load_at(log.onsets - 1.2)
    acc_low = log.correct[driving_load == 0].mean()
    acc_high = log.correct[driving_load == 2].mean()
    assert acc_high < acc_low


def test_rt_rises_with_load_measured_at_lead_time():
    tl, log = _make(0.8, lead_delta_s=1.2)
    driving_load = tl.load_at(log.onsets - 1.2)
    assert log.rt[driving_load == 2].mean() > log.rt[driving_load == 0].mean()


def test_lead_delta_shifts_which_state_drives_behavior():
    # Δ가 크게 다르면 같은 시드에서도 행동 계열이 달라져야 한다
    _, a = _make(0.8, lead_delta_s=0.0)
    _, b = _make(0.8, lead_delta_s=10.0)
    assert not np.array_equal(a.rt, b.rt)


def test_zero_effect_size_removes_load_dependence():
    tl, log = _make(0.0)
    driving_load = tl.load_at(log.onsets - 1.2)
    lo = log.rt[driving_load == 0].mean()
    hi = log.rt[driving_load == 2].mean()
    assert hi == pytest.approx(lo, abs=0.05)


def test_rejects_negative_lead_delta():
    with pytest.raises(ValueError):
        _make(0.8, lead_delta_s=-1.0)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_behavior.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.components.behavior'`

- [ ] **Step 3: 구현**

`src/simulation/components/behavior.py`:

```python
"""행동 성분: 정답 여부와 반응시간.

시각 u의 행동은 시각 (u - lead_delta_s)의 인지상태가 결정한다.
따라서 시각 w의 뇌신호로 시각 (w + lead_delta_s)의 행동을 예측할 수 있다.
이것이 계획서 가설 2("행동 오류 발현 약 1.2초 전 선행 신호")의 구현이며,
행동을 입력 특징이자 동시점 예측 타깃으로 쓸 때 생기는 라벨 누수를 피한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.components.base import BEHAV_COUPLING, load_fraction

BASE_ACCURACY = 0.95
ACCURACY_DROP = 0.35   # 최대 부하에서 정답률이 이만큼 떨어진다
BASE_RT_S = 0.55
RT_RISE_S = 0.40       # 최대 부하에서 RT가 이만큼 늘어난다
RT_NOISE_SD = 0.08


@dataclass(frozen=True)
class BehaviorLog:
    onsets: np.ndarray   # (n_stim,) 자극 제시 시각(초)
    correct: np.ndarray  # (n_stim,) 0/1
    rt: np.ndarray       # (n_stim,) 초


def generate_behavior(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    effect_size: float,
    lead_delta_s: float,
) -> BehaviorLog:
    """자극별 정답 여부와 반응시간을 만든다."""
    if lead_delta_s < 0:
        raise ValueError(f"lead_delta_s must be >= 0, got {lead_delta_s}")

    onsets = timeline.stim_onsets
    # 행동을 구동하는 것은 Δ만큼 앞선 시점의 상태다
    driving_load = load_fraction(timeline.load_at(onsets - lead_delta_s))

    gain = 1.0 + subject.theta
    scale = effect_size * BEHAV_COUPLING * gain * driving_load

    p_correct = np.clip(BASE_ACCURACY - ACCURACY_DROP * scale, 0.05, 0.99)
    correct = (rng.random(len(onsets)) < p_correct).astype(int)

    rt = BASE_RT_S + RT_RISE_S * scale + rng.normal(0.0, RT_NOISE_SD, size=len(onsets))
    rt = np.clip(rt, 0.05, None)

    return BehaviorLog(onsets=onsets, correct=correct, rt=rt)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_behavior.py -v
```

Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/components/behavior.py tests/simulation/test_behavior.py
git commit -m "feat: 행동 성분 — 상태를 Δ만큼 선행 관측

시각 u의 행동은 u-Δ의 상태가 결정. 뇌신호 w → 행동 w+Δ 예측이 성립하며
동시점 예측이 일으키는 라벨 누수를 회피한다 (계획서 가설 2)."
```

---

### Task 9: SyntheticRecording 조립 + BIDS 메타 라이터

**Files:**
- Create: `src/simulation/recording.py`
- Test: `tests/simulation/test_recording.py`

**Interfaces:**
- Consumes: Task 3~8 전부
- Produces:
  - `@dataclass(frozen=True) SyntheticRecording` — 필드 `subject_id: str`, `eeg: np.ndarray`, `eeg_sfreq: float`, `hbo: np.ndarray`, `hbr: np.ndarray`, `fnirs_sfreq: float`, `behavior: BehaviorLog`, `timeline: CognitiveStateTimeline`
  - `generate_recording(subject, sim_cfg: dict, rng) -> SyntheticRecording`
  - `generate_dataset(sim_cfg: dict, rng) -> list[SyntheticRecording]`
  - `write_bids_metadata(recordings, root: Path) -> None`

BIDS 라이터는 **메타데이터만** 쓴다(`dataset_description.json`, `participants.tsv`). 실장비 모델·몽타주가 미확정이라 SNIRF·EDF 신호 파일 작성은 스펙 §11.3에 따라 보류한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/simulation/test_recording.py`:

```python
import json

import numpy as np

from src.common.seeding import set_all_seeds
from src.simulation.recording import generate_dataset, generate_recording, write_bids_metadata
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 3,
    "subject_variance": 0.5,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def test_recording_shapes():
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.5, rng)[0]
    rec = generate_recording(sub, SIM_CFG, rng)
    assert rec.eeg.shape[0] == 30
    assert rec.hbo.shape[0] == 48
    assert rec.hbr.shape == rec.hbo.shape
    assert rec.eeg.shape[1] == int(round(rec.timeline.duration_s * 250))
    assert rec.hbo.shape[1] == int(round(rec.timeline.duration_s * 10.4))


def test_recording_has_no_nans():
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.5, rng)[0]
    rec = generate_recording(sub, SIM_CFG, rng)
    assert np.isfinite(rec.eeg).all()
    assert np.isfinite(rec.hbo).all()
    assert np.isfinite(rec.hbr).all()


def test_dataset_has_requested_subject_count():
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    assert len(recs) == 3
    assert [r.subject_id for r in recs] == ["sub-01", "sub-02", "sub-03"]


def test_subjects_differ_from_each_other():
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    assert not np.array_equal(recs[0].eeg, recs[1].eeg)


def test_dataset_is_reproducible():
    a = generate_dataset(SIM_CFG, set_all_seeds(11))
    b = generate_dataset(SIM_CFG, set_all_seeds(11))
    assert np.array_equal(a[0].eeg, b[0].eeg)
    assert np.array_equal(a[2].hbo, b[2].hbo)


def test_bids_metadata_written(tmp_path):
    recs = generate_dataset(SIM_CFG, set_all_seeds(0))
    write_bids_metadata(recs, tmp_path)

    desc = json.loads((tmp_path / "dataset_description.json").read_text(encoding="utf-8"))
    assert desc["BIDSVersion"]
    assert "synthetic" in desc["Name"].lower()

    lines = (tmp_path / "participants.tsv").read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].split("\t")[0] == "participant_id"
    assert len(lines) == 4  # 헤더 + 피험자 3명
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_recording.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulation.recording'`

- [ ] **Step 3: 구현**

`src/simulation/recording.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/test_recording.py -v
```

Expected: 6 passed

- [ ] **Step 5: 전체 시뮬레이션 테스트 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/simulation/ -v
```

Expected: 전부 통과

- [ ] **Step 6: 커밋**

```bash
cd /d/Study_fNIRS
git add src/simulation/recording.py tests/simulation/test_recording.py
git commit -m "feat: SyntheticRecording 조립 + BIDS 메타 라이터

신호 파일(SNIRF/EDF)은 장비·몽타주 확정 전까지 보류 (스펙 11.3)."
```

---

### Task 10: 데이터셋 계약 ★핵심★

**Files:**
- Create: `src/datasets/contract.py`
- Test: `tests/datasets/test_contract.py`

**Interfaces:**
- Consumes: 없음 (순수 자료구조)
- Produces:
  - `class LeakageError(RuntimeError)`
  - `@dataclass(frozen=True) WindowedDataset` — 생성자 인자 `X: dict[str, np.ndarray]`, `y: dict[str, np.ndarray]`, `subject_ids: np.ndarray`, `window_times: np.ndarray`, `trial_ids: np.ndarray`. 공개 API: `n_windows`, `modalities`, `targets`, `get_subject_ids()`, `get_trial_ids()`, `iter_folds(splitter)`
  - `class TrainView` — `X(modalities)`, `y(target)`, `subject_ids()`, `window_times()`, `groups()`
  - `class TestView` — `X(modalities)`, `y(target)`, `subject_ids()`, `window_times()`, `transform(fitted, modalities)`; `fit`/`fit_transform`은 `LeakageError`
  - `@dataclass(frozen=True) FoldView` — `fold_id: int`, `train: TrainView`, `test: TestView`

**이 작업이 서브프로젝트의 존재 이유다.** `WindowedDataset`은 `X`·`y`를 직접 노출하는 프로퍼티를 갖지 않는다. `iter_folds()`를 거치지 않으면 특징·라벨에 닿을 수 없고, 따라서 "전체 데이터에 scaler를 fit"하는 코드를 작성하는 것 자체가 불가능해진다.

`get_subject_ids()`·`get_trial_ids()`는 노출한다 — 분할기가 그룹 정보를 필요로 하고, 피험자 ID 자체는 누수원이 아니기 때문이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/datasets/test_contract.py`:

```python
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.datasets.contract import LeakageError, TestView, TrainView, WindowedDataset


def make_dataset(n=12):
    rng = np.random.default_rng(0)
    return WindowedDataset(
        X={"eeg": rng.normal(size=(n, 4)), "fnirs": rng.normal(size=(n, 3))},
        y={"cognitive_load": np.arange(n) % 3},
        subject_ids=np.array([f"sub-{i // 4 + 1:02d}" for i in range(n)]),
        window_times=np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float) + 5.0]),
        trial_ids=np.arange(n) // 2,
    )


class DummySplitter:
    """앞 8개를 train, 뒤 4개를 test로 주는 고정 분할기."""

    def split(self, subject_ids, trial_ids):
        idx = np.arange(len(subject_ids))
        yield idx[:8], idx[8:]


def test_no_public_x_or_y_attribute():
    ds = make_dataset()
    assert not hasattr(ds, "X")
    assert not hasattr(ds, "y")


def test_basic_properties():
    ds = make_dataset()
    assert ds.n_windows == 12
    assert sorted(ds.modalities) == ["eeg", "fnirs"]
    assert ds.targets == ["cognitive_load"]


def test_subject_ids_are_exposed_but_copied():
    ds = make_dataset()
    a = ds.get_subject_ids()
    a[0] = "TAMPERED"
    assert ds.get_subject_ids()[0] == "sub-01"


def test_iter_folds_yields_views():
    ds = make_dataset()
    folds = list(ds.iter_folds(DummySplitter()))
    assert len(folds) == 1
    assert isinstance(folds[0].train, TrainView)
    assert isinstance(folds[0].test, TestView)
    assert folds[0].fold_id == 0


def test_view_x_concatenates_modalities_in_given_order():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    x = fold.train.X(["eeg", "fnirs"])
    assert x.shape == (8, 7)


def test_view_x_rejects_unknown_modality():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(KeyError, match="bogus"):
        fold.train.X(["bogus"])


def test_view_y_rejects_unknown_target():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(KeyError, match="bogus"):
        fold.train.y("bogus")


def test_train_and_test_have_disjoint_rows():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    assert fold.train.X(["eeg"]).shape[0] == 8
    assert fold.test.X(["eeg"]).shape[0] == 4


def test_testview_fit_raises_leakage_error():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(LeakageError, match="fit"):
        fold.test.fit(StandardScaler())


def test_testview_fit_transform_raises_leakage_error():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    with pytest.raises(LeakageError, match="fit"):
        fold.test.fit_transform(StandardScaler())


def test_testview_transform_with_fitted_transformer_is_allowed():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    scaler = StandardScaler().fit(fold.train.X(["eeg"]))
    out = fold.test.transform(scaler, ["eeg"])
    assert out.shape == (4, 4)


def test_trainview_groups_returns_subject_ids():
    ds = make_dataset()
    fold = next(iter(ds.iter_folds(DummySplitter())))
    assert np.array_equal(fold.train.groups(), fold.train.subject_ids())


def test_rejects_length_mismatch():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="length"):
        WindowedDataset(
            X={"eeg": rng.normal(size=(10, 4))},
            y={"cognitive_load": np.arange(9)},
            subject_ids=np.array(["sub-01"] * 10),
            window_times=np.zeros((10, 2)),
            trial_ids=np.zeros(10, dtype=int),
        )


def test_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        WindowedDataset(
            X={"eeg": np.zeros((0, 4))},
            y={"cognitive_load": np.zeros(0, dtype=int)},
            subject_ids=np.array([], dtype="<U6"),
            window_times=np.zeros((0, 2)),
            trial_ids=np.zeros(0, dtype=int),
        )
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_contract.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.contract'`

- [ ] **Step 3: 구현**

`src/datasets/contract.py`:

```python
"""데이터셋 계약 — 누수를 구조적으로 차단하는 계층.

WindowedDataset은 X·y를 직접 노출하는 프로퍼티를 두지 않는다.
iter_folds()를 거치지 않으면 특징과 라벨에 닿을 수 없으므로
"전체 데이터에 scaler를 fit"하는 코드를 작성하는 것 자체가 불가능해진다.

TestView는 fit 계열 호출에서 LeakageError를 던진다. 이것은 설정으로
끌 수 없는 구조적 장치다 (스펙 §8 "가드 비활성화에 대하여").

fold 내부에서는 sklearn Pipeline·GridSearchCV를 평소대로 쓴다.
이 계층은 sklearn을 대체하지 않고 감싼다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class LeakageError(RuntimeError):
    """train/test 경계를 넘는 정보 흐름이 감지됐을 때 발생."""


class _View:
    """fold 한쪽 절반에 대한 읽기 전용 뷰."""

    def __init__(self, dataset: WindowedDataset, indices: np.ndarray, kind: str) -> None:
        self._ds = dataset
        self._idx = indices
        self._kind = kind

    def __len__(self) -> int:
        return len(self._idx)

    def X(self, modalities: list[str]) -> np.ndarray:
        """지정한 모달리티를 주어진 순서대로 가로로 이어붙여 반환한다."""
        blocks = []
        for name in modalities:
            if name not in self._ds._X:
                raise KeyError(f"unknown modality '{name}'; have {sorted(self._ds._X)}")
            blocks.append(self._ds._X[name][self._idx])
        return np.hstack(blocks)

    def y(self, target: str) -> np.ndarray:
        if target not in self._ds._y:
            raise KeyError(f"unknown target '{target}'; have {sorted(self._ds._y)}")
        return self._ds._y[target][self._idx]

    def subject_ids(self) -> np.ndarray:
        return self._ds._subject_ids[self._idx]

    def window_times(self) -> np.ndarray:
        return self._ds._window_times[self._idx]


class TrainView(_View):
    """fold의 학습 절반. sklearn 관용구를 그대로 사용할 수 있다."""

    def groups(self) -> np.ndarray:
        """중첩 CV용 그룹 벡터. 항상 피험자 ID다."""
        return self.subject_ids()


class TestView(_View):
    """fold의 평가 절반. 어떤 fit도 허용하지 않는다."""

    def transform(self, fitted_transformer: Any, modalities: list[str]) -> np.ndarray:
        """train에서 이미 fit된 변환기를 적용한다."""
        return fitted_transformer.transform(self.X(modalities))

    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise LeakageError(
            "cannot fit on the test view — fit only on TrainView. "
            "test 데이터로 fit하면 그 fold의 성능 수치는 폐기 대상이다."
        )

    def fit_transform(self, *args: Any, **kwargs: Any) -> None:
        raise LeakageError(
            "cannot fit_transform on the test view — fit on TrainView, "
            "then call TestView.transform(fitted, modalities)."
        )


@dataclass(frozen=True)
class FoldView:
    fold_id: int
    train: TrainView
    test: TestView


class WindowedDataset:
    """윈도우 단위 특징·라벨 묶음.

    생성자는 X/y/... 를 인자로 받지만 인스턴스에는 밑줄 이름으로만 보관한다.
    외부에서 ds.X 로 접근할 수 없다는 것이 이 클래스의 요점이므로
    dataclass를 쓰지 않고 __init__을 직접 정의한다.
    """

    def __init__(
        self,
        X: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        subject_ids: np.ndarray,
        window_times: np.ndarray,
        trial_ids: np.ndarray,
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
        if len(window_times) != n:
            raise ValueError(f"window_times length {len(window_times)} != {n}")
        if len(trial_ids) != n:
            raise ValueError(f"trial_ids length {len(trial_ids)} != {n}")

        self._X = dict(X)
        self._y = dict(y)
        self._subject_ids = np.asarray(subject_ids)
        self._window_times = np.asarray(window_times, dtype=float)
        self._trial_ids = np.asarray(trial_ids)

    @property
    def n_windows(self) -> int:
        return len(self._subject_ids)

    @property
    def modalities(self) -> list[str]:
        return sorted(self._X)

    @property
    def targets(self) -> list[str]:
        return sorted(self._y)

    def get_subject_ids(self) -> np.ndarray:
        return self._subject_ids.copy()

    def get_trial_ids(self) -> np.ndarray:
        return self._trial_ids.copy()

    def iter_folds(self, splitter: Any) -> Iterator[FoldView]:
        """분할기가 내놓는 fold를 뷰로 감싸 하나씩 내보낸다.

        데이터에 접근하는 유일한 경로다.
        """
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(self._subject_ids, self._trial_ids)
        ):
            yield FoldView(
                fold_id=fold_id,
                train=TrainView(self, np.asarray(train_idx), "train"),
                test=TestView(self, np.asarray(test_idx), "test"),
            )
```

`from dataclasses import dataclass, field` 중 `field`는 이제 쓰지 않으므로 import에서 뺀다. `dataclass`는 `FoldView`가 계속 쓴다.

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_contract.py -v
```

Expected: 14 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/datasets/contract.py tests/datasets/test_contract.py
git commit -m "feat: 데이터셋 계약 — 누수를 구조적으로 차단

X/y를 직접 노출하지 않고 iter_folds()로만 접근 가능.
TestView.fit()은 설정으로 끌 수 없는 LeakageError.
fold 내부에서는 sklearn을 그대로 사용한다."
```

---

### Task 11: 윈도잉

**Files:**
- Create: `src/datasets/windowing.py`
- Test: `tests/datasets/test_windowing.py`

**Interfaces:**
- Consumes: `CognitiveStateTimeline` (Task 4)
- Produces:
  - `@dataclass(frozen=True) WindowIndex` — 필드 `start_s: np.ndarray`, `end_s: np.ndarray`, `trial_id: np.ndarray`, `load_level: np.ndarray`
  - `make_windows(timeline, window_s: float, step_s: float) -> WindowIndex`

**블록 경계를 넘는 창은 버린다.** 5초 창이 두 블록에 걸치면 인지부하 라벨이 모호해지기 때문이다. 남은 창은 모두 단일 블록 안에 있다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/datasets/test_windowing.py`:

```python
import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.datasets.windowing import make_windows
from src.simulation.state import build_timeline

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _timeline():
    return build_timeline(TASK_CFG, set_all_seeds(0))


def test_windows_have_requested_length():
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    assert np.allclose(w.end_s - w.start_s, 5.0)


def test_windows_advance_by_step():
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    within_first_block = w.start_s[w.trial_id == w.trial_id[0]]
    assert np.allclose(np.diff(within_first_block), 1.0)


def test_no_window_crosses_a_block_boundary():
    tl = _timeline()
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    # 창 시작과 끝 직전이 같은 블록이어야 한다
    start_trial = tl.trial_at(w.start_s)
    end_trial = tl.trial_at(w.end_s - 1e-6)
    assert np.array_equal(start_trial, end_trial)


def test_label_matches_block_load():
    tl = _timeline()
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    assert np.array_equal(w.load_level, tl.load_at(w.start_s))


def test_count_matches_expected():
    # 블록 30초, 창 5초, 스텝 1초 -> 블록당 26개, 블록 6개
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    assert len(w.start_s) == 6 * 26


def test_all_windows_inside_recording():
    tl = _timeline()
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    assert w.start_s.min() >= 0.0
    assert w.end_s.max() <= tl.duration_s


def test_rejects_step_larger_than_window():
    with pytest.raises(ValueError, match="step_s"):
        make_windows(_timeline(), window_s=5.0, step_s=6.0)


def test_rejects_window_longer_than_block():
    with pytest.raises(ValueError, match="no windows"):
        make_windows(_timeline(), window_s=100.0, step_s=1.0)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_windowing.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.windowing'`

- [ ] **Step 3: 구현**

`src/datasets/windowing.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_windowing.py -v
```

Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/datasets/windowing.py tests/datasets/test_windowing.py
git commit -m "feat: 5초창·1초스텝 윈도잉 — 블록 경계 넘는 창 제외

경계를 넘는 창은 인지부하 라벨이 두 값에 걸쳐 모호해지므로 버린다."
```

---

### Task 12: 최소 특징 추출기

**Files:**
- Create: `src/datasets/features_minimal.py`
- Test: `tests/datasets/test_features_minimal.py`

**Interfaces:**
- Consumes: `SyntheticRecording` (Task 9), `WindowIndex` (Task 11)
- Produces: `extract_features(rec, windows) -> dict[str, np.ndarray]` — 키 `"eeg"`, `"fnirs"`, `"behavior"`
  - `eeg`: `(n_win, n_eeg_ch * 2)` — 채널별 θ(4–8 Hz)·α(8–13 Hz) 대역 파워
  - `fnirs`: `(n_win, n_fnirs_ch * 2)` — 채널별 HbO 평균·기울기
  - `behavior`: `(n_win, 2)` — 창 안 자극의 정답률·평균 RT

**임시 구현이다.** B(전처리) 완성 후 동일 시그니처로 교체한다. 창 안에 자극이 없으면 행동 특징은 `0.0`으로 채운다(결측 표시가 아니라 상수).

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/datasets/test_features_minimal.py`:

```python
import numpy as np

from src.common.seeding import set_all_seeds
from src.datasets.features_minimal import extract_features
from src.datasets.windowing import make_windows
from src.simulation.recording import generate_recording
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 1,
    "subject_variance": 0.0,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}


def _setup(effect_size=0.8):
    cfg = {**SIM_CFG, "effect_size": effect_size}
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.0, rng)[0]
    rec = generate_recording(sub, cfg, rng)
    win = make_windows(rec.timeline, 5.0, 1.0)
    return rec, win, extract_features(rec, win)


def test_feature_shapes():
    rec, win, feats = _setup()
    n = len(win.start_s)
    assert feats["eeg"].shape == (n, 60)
    assert feats["fnirs"].shape == (n, 96)
    assert feats["behavior"].shape == (n, 2)


def test_no_nans():
    _, _, feats = _setup()
    for name, arr in feats.items():
        assert np.isfinite(arr).all(), f"{name} has non-finite values"


def test_theta_feature_separates_load_levels():
    _, win, feats = _setup(effect_size=0.8)
    theta_ch0 = feats["eeg"][:, 0]
    assert theta_ch0[win.load_level == 2].mean() > theta_ch0[win.load_level == 0].mean()


def test_hbo_feature_separates_load_levels():
    _, win, feats = _setup(effect_size=0.8)
    hbo_mean_ch0 = feats["fnirs"][:, 0]
    assert hbo_mean_ch0[win.load_level == 2].mean() > hbo_mean_ch0[win.load_level == 0].mean()


def test_zero_effect_size_collapses_separation():
    _, win, feats = _setup(effect_size=0.0)
    hbo_mean_ch0 = feats["fnirs"][:, 0]
    hi = hbo_mean_ch0[win.load_level == 2].mean()
    lo = hbo_mean_ch0[win.load_level == 0].mean()
    assert abs(hi - lo) < 0.1


def test_behavior_accuracy_is_a_rate():
    _, _, feats = _setup()
    acc = feats["behavior"][:, 0]
    assert acc.min() >= 0.0 and acc.max() <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_features_minimal.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.features_minimal'`

- [ ] **Step 3: 구현**

`src/datasets/features_minimal.py`:

```python
"""임시 특징 추출기.

B(전처리 파이프라인)가 없는 상태에서 raw → 특징 경로를 잇기 위한 최소
구현이다. 특징이 없으면 LOSO 하네스를 검증할 수 없기 때문에 필요하다.
B 완성 후 동일 시그니처로 교체한다.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)


def _band_powers(segment: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_ch, n_samp) 구간에서 채널별 θ·α 대역 파워를 구한다."""
    nperseg = min(256, segment.shape[1])
    freqs, pxx = sp_signal.welch(segment, fs=sfreq, nperseg=nperseg, axis=1)
    theta = pxx[:, (freqs >= THETA_BAND[0]) & (freqs < THETA_BAND[1])].sum(axis=1)
    alpha = pxx[:, (freqs >= ALPHA_BAND[0]) & (freqs < ALPHA_BAND[1])].sum(axis=1)
    return theta, alpha


def extract_features(rec, windows) -> dict[str, np.ndarray]:
    """창별 특징을 모달리티별 배열로 반환한다."""
    n_win = len(windows.start_s)

    eeg_feats = np.zeros((n_win, rec.eeg.shape[0] * 2))
    fnirs_feats = np.zeros((n_win, rec.hbo.shape[0] * 2))
    behav_feats = np.zeros((n_win, 2))

    for i, (start, end) in enumerate(zip(windows.start_s, windows.end_s)):
        e0, e1 = int(round(start * rec.eeg_sfreq)), int(round(end * rec.eeg_sfreq))
        theta, alpha = _band_powers(rec.eeg[:, e0:e1], rec.eeg_sfreq)
        eeg_feats[i] = np.concatenate([theta, alpha])

        f0, f1 = int(round(start * rec.fnirs_sfreq)), int(round(end * rec.fnirs_sfreq))
        seg = rec.hbo[:, f0:f1]
        hbo_mean = seg.mean(axis=1)
        x = np.arange(seg.shape[1], dtype=float)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        hbo_slope = ((seg - hbo_mean[:, None]) * x_centered).sum(axis=1) / denom
        fnirs_feats[i] = np.concatenate([hbo_mean, hbo_slope])

        in_win = (rec.behavior.onsets >= start) & (rec.behavior.onsets < end)
        if in_win.any():
            behav_feats[i] = [
                rec.behavior.correct[in_win].mean(),
                rec.behavior.rt[in_win].mean(),
            ]
        # 창 안에 자극이 없으면 0.0으로 남긴다

    return {"eeg": eeg_feats, "fnirs": fnirs_feats, "behavior": behav_feats}
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_features_minimal.py -v
```

Expected: 6 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/datasets/features_minimal.py tests/datasets/test_features_minimal.py
git commit -m "feat: 최소 특징 추출기 (임시)

EEG 대역파워, fNIRS HbO 평균·기울기, 행동 정답률·RT.
B 전처리 완성 후 동일 시그니처로 교체한다."
```

---

### Task 13: 라벨 + 선행 타깃

**Files:**
- Create: `src/datasets/labels.py`
- Test: `tests/datasets/test_labels.py`

**Interfaces:**
- Consumes: `SyntheticRecording` (Task 9), `WindowIndex` (Task 11)
- Produces: `build_labels(rec, windows, *, lead_delta_s: float, rt_bins: list[float]) -> tuple[dict[str, np.ndarray], np.ndarray]` — `(labels, keep_mask)`
  - `labels["cognitive_load"]`: 창의 부하 인덱스 (0/1/2)
  - `labels["accuracy"]`: 선행 자극의 정오답 (0/1)
  - `labels["response_latency"]`: 선행 자극 RT를 `rt_bins`로 나눈 3수준 (0/1/2)
  - `keep_mask`: 선행 자극이 존재하는 창만 `True`

**선행 타깃 정의:** 창이 `end_s`에서 끝날 때, `end_s + lead_delta_s` **이후 첫 자극**의 행동이 타깃이다. 창 안의 행동 특징과 시간이 겹치지 않으므로 자명한 누수가 생기지 않는다. 해당 자극이 없으면 그 창을 버리고 개수를 기록한다(스펙 §9).

`rt_bins`는 **config의 고정 임계값**이다. 데이터에서 분위수를 계산하면 그 자체가 전역 fit이 되어 누수가 된다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/datasets/test_labels.py`:

```python
import numpy as np

from src.common.seeding import set_all_seeds
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.simulation.recording import generate_recording
from src.simulation.subject import make_subjects

SIM_CFG = {
    "n_subjects": 1,
    "subject_variance": 0.0,
    "effect_size": 0.8,
    "lead_delta_s": 1.2,
    "task": {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 30,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    },
    "eeg": {"n_channels": 30, "sfreq_hz": 250},
    "fnirs": {"n_channels": 48, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
}
RT_BINS = [0.5, 0.8]


def _setup(lead_delta_s=1.2):
    rng = set_all_seeds(0)
    sub = make_subjects(1, 0.0, rng)[0]
    rec = generate_recording(sub, SIM_CFG, rng)
    win = make_windows(rec.timeline, 5.0, 1.0)
    labels, keep = build_labels(rec, win, lead_delta_s=lead_delta_s, rt_bins=RT_BINS)
    return rec, win, labels, keep


def test_returns_expected_targets():
    _, _, labels, _ = _setup()
    assert set(labels) == {"cognitive_load", "accuracy", "response_latency"}


def test_all_arrays_same_length_as_windows():
    _, win, labels, keep = _setup()
    n = len(win.start_s)
    assert all(len(v) == n for v in labels.values())
    assert len(keep) == n


def test_cognitive_load_matches_window_label():
    _, win, labels, _ = _setup()
    assert np.array_equal(labels["cognitive_load"], win.load_level)


def test_cognitive_load_has_three_levels():
    _, _, labels, keep = _setup()
    assert set(np.unique(labels["cognitive_load"][keep])) == {0, 1, 2}


def test_accuracy_is_binary():
    _, _, labels, keep = _setup()
    assert set(np.unique(labels["accuracy"][keep])).issubset({0, 1})


def test_response_latency_uses_fixed_bins():
    _, _, labels, keep = _setup()
    assert set(np.unique(labels["response_latency"][keep])).issubset({0, 1, 2})


def test_keep_mask_drops_windows_without_a_lead_stimulus():
    # Δ가 녹화 길이만큼 크면 선행 자극이 하나도 없다
    _, _, _, keep = _setup(lead_delta_s=1000.0)
    assert not keep.any()


def test_most_windows_are_kept_with_normal_delta():
    _, _, _, keep = _setup(lead_delta_s=1.2)
    assert keep.mean() > 0.9


def test_lead_target_comes_after_window_end():
    rec, win, labels, keep = _setup(lead_delta_s=1.2)
    # 선행 자극 시각이 창 끝보다 뒤인지 직접 확인
    for i in np.flatnonzero(keep)[:50]:
        later = rec.behavior.onsets[rec.behavior.onsets >= win.end_s[i] + 1.2]
        assert len(later) > 0
        assert later[0] > win.end_s[i]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_labels.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.labels'`

- [ ] **Step 3: 구현**

`src/datasets/labels.py`:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/datasets/test_labels.py -v
```

Expected: 9 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/datasets/labels.py tests/datasets/test_labels.py
git commit -m "feat: 라벨 구성 + 선행 타깃

행동 타깃은 창 끝 + Δ 이후 첫 자극. 창 안 행동 특징과 시간이 겹치지
않아 자명한 누수를 회피한다. rt_bins는 config 고정값(분위수 계산 금지)."
```

---

### Task 14: 분할기

**Files:**
- Create: `src/evaluation/splitters.py`
- Test: `tests/evaluation/test_splitters.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `class Splitter(Protocol)` — `split(subject_ids, trial_ids) -> Iterator[tuple[np.ndarray, np.ndarray]]`
  - `get_splitter(name: str, *, seed: int = 0) -> Splitter` — `"loso"`, `"within_subject"`, `"window_random"`

**`window_random`은 의도적으로 누수를 일으키는 분할기다.** T3(누수 검출)에서만 쓰며, 이름 자체로 위험을 드러낸다. `sklearn`의 `split(X, y, groups)` 시그니처를 그대로 쓰지 않는 이유는 `within_subject`가 피험자 ID와 블록 ID **둘 다** 필요하기 때문이다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/evaluation/test_splitters.py`:

```python
import numpy as np
import pytest

from src.evaluation.splitters import get_splitter

SUBJ = np.array(["sub-01"] * 12 + ["sub-02"] * 12 + ["sub-03"] * 12)
TRIAL = np.tile(np.repeat(np.arange(3), 4), 3)


def test_loso_yields_one_fold_per_subject():
    folds = list(get_splitter("loso").split(SUBJ, TRIAL))
    assert len(folds) == 3


def test_loso_test_fold_is_exactly_one_subject():
    for train_idx, test_idx in get_splitter("loso").split(SUBJ, TRIAL):
        assert len(np.unique(SUBJ[test_idx])) == 1


def test_loso_train_and_test_subjects_are_disjoint():
    for train_idx, test_idx in get_splitter("loso").split(SUBJ, TRIAL):
        assert not (set(SUBJ[train_idx]) & set(SUBJ[test_idx]))


def test_loso_covers_every_row_exactly_once_as_test():
    seen = np.zeros(len(SUBJ), dtype=int)
    for _, test_idx in get_splitter("loso").split(SUBJ, TRIAL):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_within_subject_keeps_each_fold_inside_one_subject():
    for train_idx, test_idx in get_splitter("within_subject").split(SUBJ, TRIAL):
        assert len(np.unique(SUBJ[np.concatenate([train_idx, test_idx])])) == 1


def test_within_subject_splits_by_trial_not_by_window():
    for train_idx, test_idx in get_splitter("within_subject").split(SUBJ, TRIAL):
        assert not (set(TRIAL[train_idx]) & set(TRIAL[test_idx]))


def test_window_random_mixes_subjects_across_train_and_test():
    leaked = False
    for train_idx, test_idx in get_splitter("window_random", seed=0).split(SUBJ, TRIAL):
        if set(SUBJ[train_idx]) & set(SUBJ[test_idx]):
            leaked = True
    assert leaked, "window_random은 T3 시연을 위해 반드시 피험자를 섞어야 한다"


def test_unknown_splitter_name_raises():
    with pytest.raises(ValueError, match="unknown splitter"):
        get_splitter("bogus")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_splitters.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.splitters'`

- [ ] **Step 3: 구현**

`src/evaluation/splitters.py`:

```python
"""분할기.

sklearn의 split(X, y, groups) 시그니처를 쓰지 않는 이유는
within_subject가 피험자 ID와 블록 ID를 둘 다 필요로 하기 때문이다.
내부적으로는 sklearn의 LeaveOneGroupOut·GroupKFold·KFold를 쓴다.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut


class Splitter(Protocol):
    def split(
        self, subject_ids: np.ndarray, trial_ids: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


class LosoSplitter:
    """Leave-One-Subject-Out. 교차 피험자 일반화를 측정한다."""

    def split(self, subject_ids, trial_ids):
        yield from LeaveOneGroupOut().split(
            np.zeros(len(subject_ids)), groups=subject_ids
        )


class WithinSubjectSplitter:
    """피험자별로 블록 단위 K-fold.

    창 단위가 아니라 블록 단위로 나눈다. 창 단위로 나누면 80% 오버랩
    때문에 같은 피험자 안에서도 누수가 생긴다.
    """

    def __init__(self, n_splits: int = 3) -> None:
        self.n_splits = n_splits

    def split(self, subject_ids, trial_ids):
        for subject in np.unique(subject_ids):
            rows = np.flatnonzero(subject_ids == subject)
            trials = trial_ids[rows]
            n_splits = min(self.n_splits, len(np.unique(trials)))
            if n_splits < 2:
                continue
            for tr, te in GroupKFold(n_splits=n_splits).split(
                np.zeros(len(rows)), groups=trials
            ):
                yield rows[tr], rows[te]


class WindowRandomSplitter:
    """⚠ 의도적으로 누수를 일으키는 분할기.

    창을 무작위로 섞어 나눈다. 5초 창·1초 스텝은 80% 오버랩이므로
    인접 창이 train과 test에 동시에 들어가고, 같은 피험자가 양쪽에
    존재하게 된다. T3(누수 검출) 시연 전용이며 실제 실험에 쓰면 안 된다.
    """

    def __init__(self, n_splits: int = 5, seed: int = 0) -> None:
        self.n_splits = n_splits
        self.seed = seed

    def split(self, subject_ids, trial_ids):
        yield from KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        ).split(np.zeros(len(subject_ids)))


def get_splitter(name: str, *, seed: int = 0) -> Splitter:
    """이름으로 분할기를 만든다."""
    if name == "loso":
        return LosoSplitter()
    if name == "within_subject":
        return WithinSubjectSplitter()
    if name == "window_random":
        return WindowRandomSplitter(seed=seed)
    raise ValueError(
        f"unknown splitter '{name}'; expected one of "
        "'loso', 'within_subject', 'window_random'"
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_splitters.py -v
```

Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/evaluation/splitters.py tests/evaluation/test_splitters.py
git commit -m "feat: 분할기 — loso / within_subject / window_random

within_subject는 블록 단위로 나눈다 (창 단위는 80% 오버랩 누수).
window_random은 T3 누수 시연 전용."
```

---

### Task 15: 누수 가드

**Files:**
- Create: `src/evaluation/guards.py`
- Test: `tests/evaluation/test_guards.py`

**Interfaces:**
- Consumes: `LeakageError` (Task 10)
- Produces:
  - `check_subject_overlap(train_subjects: np.ndarray, test_subjects: np.ndarray) -> None`
  - `check_window_overlap(train_subjects, train_times, test_subjects, test_times) -> None`

**윈도우 겹침 검사는 같은 피험자 안에서만 한다.** 서로 다른 피험자의 창은 둘 다 `[0, 5)`초일 수 있지만 다른 녹화이므로 누수가 아니다. 이 구분을 놓치면 LOSO가 항상 실패한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/evaluation/test_guards.py`:

```python
import numpy as np
import pytest

from src.datasets.contract import LeakageError
from src.evaluation.guards import check_subject_overlap, check_window_overlap


def test_subject_overlap_passes_when_disjoint():
    check_subject_overlap(np.array(["sub-01", "sub-02"]), np.array(["sub-03"]))


def test_subject_overlap_raises_when_shared():
    with pytest.raises(LeakageError, match="sub-02"):
        check_subject_overlap(np.array(["sub-01", "sub-02"]), np.array(["sub-02"]))


def test_window_overlap_passes_for_different_subjects_at_same_time():
    # 다른 피험자의 같은 시각 창은 누수가 아니다
    check_window_overlap(
        np.array(["sub-01"]), np.array([[0.0, 5.0]]),
        np.array(["sub-02"]), np.array([[0.0, 5.0]]),
    )


def test_window_overlap_passes_for_same_subject_disjoint_times():
    check_window_overlap(
        np.array(["sub-01"]), np.array([[0.0, 5.0]]),
        np.array(["sub-01"]), np.array([[10.0, 15.0]]),
    )


def test_window_overlap_raises_for_same_subject_overlapping_times():
    with pytest.raises(LeakageError, match="overlap"):
        check_window_overlap(
            np.array(["sub-01"]), np.array([[0.0, 5.0]]),
            np.array(["sub-01"]), np.array([[4.0, 9.0]]),
        )


def test_window_overlap_treats_touching_edges_as_disjoint():
    # [0,5)와 [5,10)은 겹치지 않는다
    check_window_overlap(
        np.array(["sub-01"]), np.array([[0.0, 5.0]]),
        np.array(["sub-01"]), np.array([[5.0, 10.0]]),
    )
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_guards.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.guards'`

- [ ] **Step 3: 구현**

`src/evaluation/guards.py`:

```python
"""누수 가드.

계약(contract.py)이 전역 fit을 막는다면, 여기서는 분할 자체가
train/test 경계를 침범하지 않았는지 매 fold마다 확인한다.
"""

from __future__ import annotations

import numpy as np

from src.datasets.contract import LeakageError


def check_subject_overlap(
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> None:
    """같은 피험자가 train과 test에 동시에 있으면 실패시킨다."""
    shared = sorted(set(train_subjects.tolist()) & set(test_subjects.tolist()))
    if shared:
        raise LeakageError(
            f"subjects appear in both train and test: {shared}. "
            "분할 단위는 항상 피험자여야 한다 (CLAUDE.md 5.1)."
        )


def check_window_overlap(
    train_subjects: np.ndarray,
    train_times: np.ndarray,
    test_subjects: np.ndarray,
    test_times: np.ndarray,
) -> None:
    """같은 피험자 안에서 train 창과 test 창이 시간상 겹치면 실패시킨다.

    서로 다른 피험자의 창은 시각이 같아도 다른 녹화이므로 검사 대상이 아니다.
    5초 창·1초 스텝은 80% 오버랩이므로 창 단위 무작위 분할은 여기서 걸린다.
    """
    for subject in set(train_subjects.tolist()) & set(test_subjects.tolist()):
        tr = train_times[train_subjects == subject]
        te = test_times[test_subjects == subject]
        # 반열린 구간 [start, end) 기준 겹침
        overlaps = (tr[:, None, 0] < te[None, :, 1]) & (te[None, :, 0] < tr[:, None, 1])
        if overlaps.any():
            i, j = np.argwhere(overlaps)[0]
            raise LeakageError(
                f"window overlap for {subject}: train {tr[i].tolist()} "
                f"overlaps test {te[j].tolist()}. "
                "인접 윈도우가 train/test에 동시 존재하면 그 결과는 폐기 대상이다."
            )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_guards.py -v
```

Expected: 6 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/evaluation/guards.py tests/evaluation/test_guards.py
git commit -m "feat: 누수 가드 — 피험자 중복 / 윈도우 시간 겹침

윈도우 겹침은 같은 피험자 안에서만 검사한다. 다른 피험자의 같은 시각
창은 다른 녹화이므로 누수가 아니다."
```

---

### Task 16: 평가 하네스

**Files:**
- Create: `src/evaluation/harness.py`
- Test: `tests/evaluation/test_harness.py`

**Interfaces:**
- Consumes: `WindowedDataset`·`FoldView` (Task 10), `Splitter` (Task 14), 가드 (Task 15)
- Produces:
  - `@dataclass(frozen=True) FoldResult` — `fold_id: int`, `test_subjects: list[str]`, `n_train: int`, `n_test: int`, `accuracy: float`, `y_true: np.ndarray`, `y_pred: np.ndarray`
  - `make_model(name: str, seed: int)` — sklearn `Pipeline` 반환
  - `run_folds(dataset, splitter, *, target, modalities, guards, seed) -> list[FoldResult]`

`guards`는 `{"check_subject_overlap": bool, "check_window_overlap": bool}`. 파이프라인은 `StandardScaler` + `LogisticRegression`이며, `Pipeline.fit`이 scaler를 train에만 fit하므로 계약과 어긋나지 않는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/evaluation/test_harness.py`:

```python
import numpy as np
import pytest

from src.datasets.contract import LeakageError, WindowedDataset
from src.evaluation.harness import make_model, run_folds
from src.evaluation.splitters import get_splitter

GUARDS_ON = {"check_subject_overlap": True, "check_window_overlap": True}
GUARDS_OFF = {"check_subject_overlap": False, "check_window_overlap": False}


def make_dataset(n_subjects=4, n_per=30, separable=True, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, subs, times, trials = [], [], [], [], []
    for s in range(n_subjects):
        for i in range(n_per):
            level = i % 3
            offset = level * 3.0 if separable else 0.0
            xs.append(rng.normal(offset, 1.0, size=4))
            ys.append(level)
            subs.append(f"sub-{s + 1:02d}")
            times.append([float(i), float(i) + 5.0])
            trials.append(i // 10)
    return WindowedDataset(
        X={"eeg": np.array(xs)},
        y={"cognitive_load": np.array(ys)},
        subject_ids=np.array(subs),
        window_times=np.array(times, dtype=float),
        trial_ids=np.array(trials),
    )


def test_returns_one_result_per_fold():
    ds = make_dataset()
    results = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert len(results) == 4


def test_result_records_test_subjects_and_sizes():
    ds = make_dataset()
    r = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )[0]
    assert len(r.test_subjects) == 1
    assert r.n_train + r.n_test == ds.n_windows
    assert len(r.y_true) == len(r.y_pred) == r.n_test


def test_separable_data_beats_chance():
    ds = make_dataset(separable=True)
    results = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert np.mean([r.accuracy for r in results]) > 0.8


def test_non_separable_data_is_near_chance():
    ds = make_dataset(separable=False)
    results = run_folds(
        ds, get_splitter("loso"), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_ON, seed=0,
    )
    assert np.mean([r.accuracy for r in results]) < 0.5


def test_guards_on_reject_window_random_splitter():
    ds = make_dataset()
    with pytest.raises(LeakageError):
        run_folds(
            ds, get_splitter("window_random", seed=0), target="cognitive_load",
            modalities=["eeg"], guards=GUARDS_ON, seed=0,
        )


def test_guards_off_allow_window_random_splitter():
    ds = make_dataset()
    results = run_folds(
        ds, get_splitter("window_random", seed=0), target="cognitive_load",
        modalities=["eeg"], guards=GUARDS_OFF, seed=0,
    )
    assert len(results) == 5


def test_make_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("bogus", seed=0)


def test_single_class_fold_is_rejected():
    ds = WindowedDataset(
        X={"eeg": np.random.default_rng(0).normal(size=(20, 3))},
        y={"cognitive_load": np.array([0] * 10 + [1] * 10)},
        subject_ids=np.array(["sub-01"] * 10 + ["sub-02"] * 10),
        window_times=np.column_stack(
            [np.arange(20, dtype=float), np.arange(20, dtype=float) + 5.0]
        ),
        trial_ids=np.arange(20) // 5,
    )
    with pytest.raises(ValueError, match="single class"):
        run_folds(
            ds, get_splitter("loso"), target="cognitive_load",
            modalities=["eeg"], guards=GUARDS_ON, seed=0,
        )
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_harness.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.harness'`

- [ ] **Step 3: 구현**

`src/evaluation/harness.py`:

```python
"""fold 실행 하네스.

계약이 넘겨주는 FoldView마다 가드를 걸고, train에서만 학습한 모델로
test를 예측한다. sklearn Pipeline을 쓰므로 scaler는 자동으로 train에만
fit된다 — 계약이 막는 것과 같은 규칙을 파이프라인이 지킨다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.guards import check_subject_overlap, check_window_overlap


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    test_subjects: list[str]
    n_train: int
    n_test: int
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray


def make_model(name: str, seed: int) -> Pipeline:
    """이름으로 sklearn 파이프라인을 만든다."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    raise ValueError(f"unknown model '{name}'; expected 'logistic_regression'")


def run_folds(
    dataset,
    splitter,
    *,
    target: str,
    modalities: list[str],
    guards: dict,
    seed: int,
    model_name: str = "logistic_regression",
) -> list[FoldResult]:
    """모든 fold를 실행하고 결과 목록을 돌려준다."""
    results: list[FoldResult] = []

    for fold in dataset.iter_folds(splitter):
        train_subj = fold.train.subject_ids()
        test_subj = fold.test.subject_ids()

        if guards.get("check_subject_overlap", True):
            check_subject_overlap(train_subj, test_subj)
        if guards.get("check_window_overlap", True):
            check_window_overlap(
                train_subj, fold.train.window_times(),
                test_subj, fold.test.window_times(),
            )

        y_train = fold.train.y(target)
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"fold {fold.fold_id} train has a single class for target "
                f"'{target}'; accuracy would be meaningless"
            )

        model = make_model(model_name, seed)
        model.fit(fold.train.X(modalities), y_train)

        y_true = fold.test.y(target)
        y_pred = model.predict(fold.test.X(modalities))

        results.append(
            FoldResult(
                fold_id=fold.fold_id,
                test_subjects=sorted(set(test_subj.tolist())),
                n_train=len(train_subj),
                n_test=len(test_subj),
                accuracy=float((y_true == y_pred).mean()),
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    return results
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_harness.py -v
```

Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/evaluation/harness.py tests/evaluation/test_harness.py
git commit -m "feat: 평가 하네스 — fold별 가드 + train 전용 학습

sklearn Pipeline이 scaler를 train에만 fit하므로 계약과 같은 규칙을 지킨다.
단일 클래스 fold는 정확도가 무의미하므로 실행을 거부한다."
```

---

### Task 17: 지표 집계

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: `FoldResult` (Task 16)
- Produces: `aggregate(fold_results: list[FoldResult], *, n_classes: int, cv_method: str) -> dict`

반환 dict의 키(고정): `cv_method`, `chance_level`, `n_folds`, `n_windows_evaluated`, `accuracy_mean`, `accuracy_std`, `accuracy_worst`, `worst_fold_subjects`, `pooled_accuracy`, `pooled_ci_low`, `pooled_ci_high`, `binomtest_p`, `confusion`.

`CLAUDE.md` §5.4가 요구하는 대로 **chance level과 CV 방식을 항상 함께** 담고, 평균만이 아니라 **최악 피험자 성능**도 담는다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/evaluation/test_metrics.py`:

```python
import numpy as np
import pytest

from src.evaluation.harness import FoldResult
from src.evaluation.metrics import aggregate


def _fold(fold_id, subject, y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return FoldResult(
        fold_id=fold_id,
        test_subjects=[subject],
        n_train=100,
        n_test=len(y_true),
        accuracy=float((y_true == y_pred).mean()),
        y_true=y_true,
        y_pred=y_pred,
    )


PERFECT = [_fold(0, "sub-01", [0, 1, 2, 0], [0, 1, 2, 0]),
           _fold(1, "sub-02", [0, 1, 2, 0], [0, 1, 2, 0])]

MIXED = [_fold(0, "sub-01", [0, 1, 2, 0], [0, 1, 2, 0]),
         _fold(1, "sub-02", [0, 1, 2, 0], [1, 1, 1, 1])]


def test_records_cv_method_and_chance():
    m = aggregate(PERFECT, n_classes=3, cv_method="loso")
    assert m["cv_method"] == "loso"
    assert m["chance_level"] == pytest.approx(1 / 3)


def test_counts_folds_and_windows():
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    assert m["n_folds"] == 2
    assert m["n_windows_evaluated"] == 8


def test_mean_and_worst_accuracy():
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    assert m["accuracy_mean"] == pytest.approx(0.625)
    assert m["accuracy_worst"] == pytest.approx(0.25)
    assert m["worst_fold_subjects"] == ["sub-02"]


def test_pooled_accuracy_and_ci():
    m = aggregate(PERFECT, n_classes=3, cv_method="loso")
    assert m["pooled_accuracy"] == pytest.approx(1.0)
    assert m["pooled_ci_low"] <= m["pooled_accuracy"] <= m["pooled_ci_high"]


def test_chance_data_has_high_p_value():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=600)
    y_pred = rng.integers(0, 3, size=600)
    m = aggregate([_fold(0, "sub-01", y_true, y_pred)], n_classes=3, cv_method="loso")
    assert m["binomtest_p"] > 0.05
    assert m["pooled_ci_low"] <= 1 / 3 <= m["pooled_ci_high"]


def test_perfect_data_has_low_p_value():
    m = aggregate(PERFECT, n_classes=3, cv_method="loso")
    assert m["binomtest_p"] < 0.05


def test_confusion_matrix_shape_and_total():
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    conf = np.array(m["confusion"])
    assert conf.shape == (3, 3)
    assert conf.sum() == 8


def test_rejects_empty_results():
    with pytest.raises(ValueError, match="no folds"):
        aggregate([], n_classes=3, cv_method="loso")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.metrics'`

- [ ] **Step 3: 구현**

`src/evaluation/metrics.py`:

```python
"""지표 집계.

CLAUDE.md 5.4가 요구하는 대로 chance level과 CV 방식을 항상 함께 담고,
평균만이 아니라 최악 피험자 성능도 담는다. 평균만 보고하면 특정 피험자에서
완전히 실패하는 모델이 좋아 보인다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix


def aggregate(fold_results, *, n_classes: int, cv_method: str) -> dict:
    """fold 결과를 하나의 지표 dict으로 모은다."""
    if not fold_results:
        raise ValueError("no folds to aggregate")

    accuracies = np.array([f.accuracy for f in fold_results])
    y_true = np.concatenate([f.y_true for f in fold_results])
    y_pred = np.concatenate([f.y_pred for f in fold_results])

    n_correct = int((y_true == y_pred).sum())
    n_total = int(len(y_true))
    chance = 1.0 / n_classes

    test = stats.binomtest(n_correct, n_total, chance, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95)

    worst = int(np.argmin(accuracies))

    return {
        "cv_method": cv_method,
        "chance_level": float(chance),
        "n_folds": len(fold_results),
        "n_windows_evaluated": n_total,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "accuracy_worst": float(accuracies[worst]),
        "worst_fold_subjects": list(fold_results[worst].test_subjects),
        "pooled_accuracy": n_correct / n_total,
        "pooled_ci_low": float(ci.low),
        "pooled_ci_high": float(ci.high),
        "binomtest_p": float(test.pvalue),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_metrics.py -v
```

Expected: 8 passed

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: 지표 집계 — chance level·CV 방식·최악 피험자 필수 기록

평균만 보고하면 특정 피험자에서 완전히 실패하는 모델이 좋아 보인다."
```

---

### Task 18: 실험 러너

**Files:**
- Create: `src/evaluation/runner.py`
- Test: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: Task 1~17 전부
- Produces:
  - `build_dataset(cfg: dict, rng) -> tuple[WindowedDataset, int]` — `(데이터셋, 제외된 창 수)`
  - `run_experiment(config_path, *, overrides: dict | None = None) -> Path` — 결과 디렉토리 경로 반환

`overrides`는 중첩 dict를 부분 병합한다. T4(개인차 스윕)가 `subject_variance`만 바꿔 3회 실행하는 데 쓴다.

**결과 디렉토리 이름** = `<run_name>_seed<N>_<git_short_hash>`, 작업 트리가 dirty면 `-dirty`, 가드가 하나라도 꺼져 있으면 `-UNSAFE`를 덧붙인다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/evaluation/test_runner.py`:

```python
import json

import pytest
import yaml

from src.common.seeding import set_all_seeds
from src.evaluation.runner import build_dataset, run_experiment

BASE_CFG = {
    "run_name": "unit",
    "seed": 7,
    "simulation": {
        "n_subjects": 4,
        "subject_variance": 0.5,
        "effect_size": 0.8,
        "lead_delta_s": 1.2,
        "task": {
            "nback_levels": [0, 2, 3],
            "block_duration_s": 20,
            "n_blocks_per_level": 1,
            "stim_interval_s": 2.0,
        },
        "eeg": {"n_channels": 8, "sfreq_hz": 100},
        "fnirs": {"n_channels": 8, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
    },
    "windowing": {"window_s": 5.0, "step_s": 1.0},
    "features": {"extractor": "minimal"},
    "dataset": {
        "targets": ["cognitive_load"],
        "lead_targets": ["accuracy", "response_latency"],
        "modalities": ["eeg", "fnirs", "behavior"],
        "rt_bins": [0.5, 0.8],
    },
    "evaluation": {
        "splitter": "loso",
        "model": "logistic_regression",
        "guards": {"check_subject_overlap": True, "check_window_overlap": True},
    },
    "output": {"results_dir": "results"},
}


def _cfg_file(tmp_path, results_dir, **sim_overrides):
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["simulation"].update(sim_overrides)
    cfg["output"]["results_dir"] = str(results_dir)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_build_dataset_shapes_are_consistent():
    ds, dropped = build_dataset(BASE_CFG, set_all_seeds(0))
    assert ds.n_windows > 0
    assert dropped >= 0
    assert sorted(ds.modalities) == ["behavior", "eeg", "fnirs"]
    assert len(set(ds.get_subject_ids().tolist())) == 4


def test_build_dataset_has_all_three_targets():
    ds, _ = build_dataset(BASE_CFG, set_all_seeds(0))
    assert ds.targets == ["accuracy", "cognitive_load", "response_latency"]


def test_run_experiment_writes_all_artifacts(tmp_path):
    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    for name in ("config.yaml", "env.json", "metrics.json", "per_fold.csv", "log.txt"):
        assert (out / name).exists(), f"missing {name}"


def test_metrics_record_chance_and_cv_method(tmp_path):
    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cv_method"] == "loso"
    assert metrics["chance_level"] == pytest.approx(1 / 3)
    assert "n_windows_dropped" in metrics
    assert metrics["guards_disabled"] is False


def test_env_records_git_and_seed(tmp_path):
    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["seed"] == 7
    assert "git_commit" in env
    assert "git_dirty" in env
    assert env["python"].startswith("3.")


def test_same_seed_gives_identical_metrics(tmp_path):
    a = run_experiment(_cfg_file(tmp_path / "a", tmp_path / "ra"))
    b = run_experiment(_cfg_file(tmp_path / "b", tmp_path / "rb"))
    ma = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "metrics.json").read_text(encoding="utf-8"))
    assert ma["pooled_accuracy"] == mb["pooled_accuracy"]


def test_overrides_are_applied(tmp_path):
    out = run_experiment(
        _cfg_file(tmp_path, tmp_path / "results"),
        overrides={"simulation": {"subject_variance": 3.0}},
    )
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["simulation"]["subject_variance"] == 3.0


def test_disabled_guards_mark_run_unsafe(tmp_path):
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["evaluation"]["guards"]["check_subject_overlap"] = False
    cfg["output"]["results_dir"] = str(tmp_path / "results")
    p = tmp_path / "unsafe.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run_experiment(p)
    assert out.name.endswith("-UNSAFE")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["guards_disabled"] is True
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluation.runner'`

- [ ] **Step 3: 구현**

`src/evaluation/runner.py`:

```python
"""실험 러너 — 오케스트레이션과 결과 기록.

CLAUDE.md 5.2가 요구하는 "실험 1회 = 결과 디렉토리 1개"를 구현한다.
config 스냅샷·시드·git commit·환경을 함께 저장하지 않으면 몇 달 뒤
그 숫자가 어디서 나왔는지 알 수 없게 된다.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.common.config import load_config
from src.common.seeding import set_all_seeds
from src.datasets.contract import WindowedDataset
from src.datasets.features_minimal import extract_features
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.evaluation.harness import run_folds
from src.evaluation.metrics import aggregate
from src.evaluation.splitters import get_splitter
from src.simulation.recording import generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def build_dataset(cfg: dict, rng: np.random.Generator) -> tuple[WindowedDataset, int]:
    """합성 녹화를 만들고 창·특징·라벨을 거쳐 계약 객체를 조립한다."""
    win_cfg = cfg["windowing"]
    ds_cfg = cfg["dataset"]
    lead_delta_s = float(cfg["simulation"]["lead_delta_s"])

    recordings = generate_dataset(cfg["simulation"], rng)

    feature_blocks: dict[str, list[np.ndarray]] = {}
    label_blocks: dict[str, list[np.ndarray]] = {}
    subjects, times, trials = [], [], []
    n_dropped = 0

    for rec in recordings:
        windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
        feats = extract_features(rec, windows)
        labels, keep = build_labels(
            rec, windows,
            lead_delta_s=lead_delta_s,
            rt_bins=list(ds_cfg["rt_bins"]),
        )
        n_dropped += int((~keep).sum())

        for name, arr in feats.items():
            feature_blocks.setdefault(name, []).append(arr[keep])
        for name, arr in labels.items():
            label_blocks.setdefault(name, []).append(arr[keep])

        subjects.append(np.full(int(keep.sum()), rec.subject_id))
        times.append(np.column_stack([windows.start_s, windows.end_s])[keep])
        trials.append(windows.trial_id[keep])

    dataset = WindowedDataset(
        X={k: np.vstack(v) for k, v in feature_blocks.items()},
        y={k: np.concatenate(v) for k, v in label_blocks.items()},
        subject_ids=np.concatenate(subjects),
        window_times=np.vstack(times),
        trial_ids=np.concatenate(trials),
    )
    return dataset, n_dropped


def run_experiment(config_path, *, overrides: dict | None = None) -> Path:
    """config 하나를 실행하고 결과 디렉토리 경로를 반환한다."""
    cfg = load_config(config_path)
    if overrides:
        cfg = _deep_update(cfg, overrides)

    seed = int(cfg["seed"])
    rng = set_all_seeds(seed)

    guards = cfg["evaluation"]["guards"]
    guards_disabled = not all(bool(v) for v in guards.values())

    commit = _git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain"))

    run_id = f"{cfg['run_name']}_seed{seed}_{commit}"
    if dirty:
        run_id += "-dirty"
    if guards_disabled:
        run_id += "-UNSAFE"

    out_dir = Path(cfg["output"]["results_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, n_dropped = build_dataset(cfg, rng)

    target = cfg["dataset"]["targets"][0]
    n_classes = int(len(cfg["simulation"]["task"]["nback_levels"]))

    splitter = get_splitter(cfg["evaluation"]["splitter"], seed=seed)
    fold_results = run_folds(
        dataset, splitter,
        target=target,
        modalities=list(cfg["dataset"]["modalities"]),
        guards=guards,
        seed=seed,
        model_name=cfg["evaluation"]["model"],
    )

    metrics = aggregate(
        fold_results, n_classes=n_classes, cv_method=cfg["evaluation"]["splitter"]
    )
    metrics["n_windows_dropped"] = n_dropped
    metrics["guards_disabled"] = guards_disabled
    metrics["target"] = target

    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "env.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "git_commit": commit,
                "git_dirty": dirty,
                "git_diff": _git("diff") if dirty else "",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (out_dir / "per_fold.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold_id", "test_subjects", "n_train", "n_test", "accuracy"])
        for r in fold_results:
            writer.writerow(
                [r.fold_id, ";".join(r.test_subjects), r.n_train, r.n_test, f"{r.accuracy:.6f}"]
            )

    (out_dir / "log.txt").write_text(
        f"run_id={run_id}\n"
        f"cv_method={metrics['cv_method']}\n"
        f"chance_level={metrics['chance_level']:.4f}\n"
        f"accuracy_mean={metrics['accuracy_mean']:.4f}\n"
        f"accuracy_worst={metrics['accuracy_worst']:.4f}\n"
        f"n_windows_dropped={n_dropped}\n"
        f"guards_disabled={guards_disabled}\n",
        encoding="utf-8",
    )

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="합성 테스트베드 실험 실행")
    parser.add_argument("config", help="config/experiments/*.yaml 경로")
    args = parser.parse_args()

    result_dir = run_experiment(args.config)
    print(result_dir)
    print((result_dir / "log.txt").read_text(encoding="utf-8"))
```

CLI로도 쓸 수 있다:

```bash
cd /d/Study_fNIRS && .venv/Scripts/python.exe -m src.evaluation.runner config/experiments/smoke.yaml
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_runner.py -v
```

Expected: 8 passed

- [ ] **Step 5: 전체 테스트 확인**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe -v
```

Expected: 전부 통과

- [ ] **Step 6: 커밋**

```bash
cd /d/Study_fNIRS
git add src/evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "feat: 실험 러너 — config 스냅샷·시드·git·환경 기록

dirty 트리는 run_id에 -dirty와 git diff 전문을 남긴다.
가드가 꺼진 실행은 -UNSAFE 표식이 붙어 인용 금지 대상이 된다."
```

---

### Task 19: T1 널 테스트 + T2 효과 회복 (승인 기준)

**Files:**
- Create: `config/experiments/pilot.yaml`, `config/experiments/null.yaml`
- Test: `tests/evaluation/test_validation_effect.py`

**Interfaces:**
- Consumes: `run_experiment` (Task 18)
- Produces: 없음 (검증 전용)

스펙 §8의 T1·T2다. **T1이 가장 강력한 단일 검사**다 — 파이프라인 어디에 누수가 있든 효과가 0인 데이터에서 chance를 넘기기 때문이다.

- [ ] **Step 1: 파일럿 config 작성**

`config/experiments/pilot.yaml` — 계획서 스펙(EEG 1000 Hz)을 따르는 실행용 설정:

```yaml
run_name: pilot
seed: 42

simulation:
  n_subjects: 12
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 60
    n_blocks_per_level: 3
    stim_interval_s: 2.0
  eeg:   {n_channels: 30, sfreq_hz: 1000}
  fnirs: {n_channels: 48, sfreq_hz: 10.4, hbr_coupling: -0.33}

windowing:
  window_s: 5.0
  step_s: 1.0

features:
  extractor: minimal

dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy, response_latency]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]

evaluation:
  splitter: loso
  model: logistic_regression
  guards:
    check_subject_overlap: true
    check_window_overlap: true

output:
  results_dir: results
```

`config/experiments/null.yaml` — `pilot.yaml`과 동일하되 두 줄만 다르다:

```yaml
run_name: null_effect
seed: 42

simulation:
  n_subjects: 12
  subject_variance: 0.5
  effect_size: 0.0        # ← 널 데이터
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 60
    n_blocks_per_level: 3
    stim_interval_s: 2.0
  eeg:   {n_channels: 30, sfreq_hz: 1000}
  fnirs: {n_channels: 48, sfreq_hz: 10.4, hbr_coupling: -0.33}

windowing:
  window_s: 5.0
  step_s: 1.0

features:
  extractor: minimal

dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy, response_latency]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]

evaluation:
  splitter: loso
  model: logistic_regression
  guards:
    check_subject_overlap: true
    check_window_overlap: true

output:
  results_dir: results
```

- [ ] **Step 2: 검증 테스트 작성**

`tests/evaluation/test_validation_effect.py`:

```python
"""T1(널 테스트) · T2(효과 회복) — 스펙 8절 승인 기준."""

import json

import pytest

from src.evaluation.runner import run_experiment

CHANCE = 1 / 3


def _metrics(config, tmp_path, **overrides):
    ov = {"output": {"results_dir": str(tmp_path)}}
    ov.update(overrides)
    out = run_experiment(config, overrides=ov)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


@pytest.mark.slow
def test_t1_null_effect_stays_at_chance(tmp_path):
    m = _metrics("config/experiments/null.yaml", tmp_path)
    assert m["cv_method"] == "loso"
    assert m["pooled_ci_low"] <= CHANCE <= m["pooled_ci_high"], (
        f"널 데이터인데 chance가 신뢰구간 밖이다 "
        f"({m['pooled_ci_low']:.3f}, {m['pooled_ci_high']:.3f}) — 파이프라인에 누수가 있다"
    )
    assert m["binomtest_p"] > 0.05


@pytest.mark.slow
def test_t1_null_holds_at_pilot_scale_too(tmp_path):
    """파일럿 규모(EEG 1000 Hz, 12명)에서도 널 데이터는 chance여야 한다.

    null.yaml과 같은 결론이지만 데이터 규모가 다르다. 창 수가 많아질수록
    누수가 있으면 신뢰구간이 좁아지며 chance를 벗어나므로 검출력이 높다.
    """
    m = _metrics(
        "config/experiments/pilot.yaml", tmp_path,
        simulation={"effect_size": 0.0},
    )
    assert m["pooled_ci_low"] <= CHANCE <= m["pooled_ci_high"]


@pytest.mark.slow
def test_t2_effect_is_recovered_above_chance(tmp_path):
    m = _metrics("config/experiments/pilot.yaml", tmp_path)
    assert m["pooled_accuracy"] > CHANCE
    assert m["binomtest_p"] < 0.01, "심어둔 효과를 하네스가 회수하지 못했다"


@pytest.mark.slow
def test_t2_accuracy_is_not_suspiciously_perfect(tmp_path):
    """LOSO에서 100%에 가까우면 누수를 의심해야 한다."""
    m = _metrics("config/experiments/pilot.yaml", tmp_path)
    assert m["pooled_accuracy"] < 0.98, (
        f"LOSO 정확도 {m['pooled_accuracy']:.3f}는 개인차가 있는 합성 데이터에서 "
        "나올 수 없는 값이다 — 누수를 점검하라"
    )


@pytest.mark.slow
def test_t2_reports_worst_subject_not_just_mean(tmp_path):
    m = _metrics("config/experiments/pilot.yaml", tmp_path)
    assert m["accuracy_worst"] <= m["accuracy_mean"]
    assert m["worst_fold_subjects"]
```

- [ ] **Step 3: T1·T2 실행**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_validation_effect.py -v -m slow
```

Expected: 5 passed

**T1이 실패하면 멈춰라.** 파이프라인 어딘가에 누수가 있다는 뜻이며, `.claude/skills/leakage-check` 스킬을 실행해 원인을 찾을 때까지 다음 작업으로 넘어가지 않는다.

- [ ] **Step 4: 파일럿 관측값을 베이스라인 파일로 고정**

스펙 §11.4가 남겨둔 항목이다. 관측값을 코드에 손으로 적어 넣는 대신 베이스라인 파일에 기록한다.

```bash
cd /d/Study_fNIRS
mkdir -p tests/baselines
.venv/Scripts/python.exe -c "
import json, pathlib
from src.evaluation.runner import run_experiment
out = run_experiment('config/experiments/pilot.yaml')
m = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
pathlib.Path('tests/baselines/t2_pilot.json').write_text(
    json.dumps({'pooled_accuracy': m['pooled_accuracy'],
                'accuracy_mean': m['accuracy_mean'],
                'accuracy_worst': m['accuracy_worst'],
                'chance_level': m['chance_level'],
                'cv_method': m['cv_method']}, indent=2),
    encoding='utf-8')
print(json.dumps(m, indent=2, ensure_ascii=False))
"
cat tests/baselines/t2_pilot.json
```

출력된 `pooled_accuracy`를 `docs/specs/2026-08-18-simulation-testbed-design.md` §11.4에도 적어 넣는다.

아래 회귀 테스트를 `tests/evaluation/test_validation_effect.py` 끝에 덧붙인다. 손으로 채울 자리가 없다.

```python
@pytest.mark.slow
def test_t2_matches_recorded_baseline(tmp_path):
    """파일럿 관측값에서 크게 벗어나면 무언가 바뀐 것이다."""
    import pathlib

    baseline = json.loads(
        pathlib.Path("tests/baselines/t2_pilot.json").read_text(encoding="utf-8")
    )
    m = _metrics("config/experiments/pilot.yaml", tmp_path)
    assert m["cv_method"] == baseline["cv_method"]
    assert abs(m["pooled_accuracy"] - baseline["pooled_accuracy"]) < 0.05, (
        f"파일럿 정확도가 베이스라인 {baseline['pooled_accuracy']:.3f}에서 "
        f"{m['pooled_accuracy']:.3f}로 이동했다"
    )
```

베이스라인이 의도적으로 바뀌어야 할 때(생성기·특징 변경 등)는 위 명령을 다시 돌려 파일을 갱신하고, **무엇을 왜 바꿨는지 `run_logging.md`에 남긴다.**

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add config/experiments/ tests/evaluation/test_validation_effect.py docs/specs/
git commit -m "test: T1 널 테스트 + T2 효과 회복 (승인 기준)

T1은 파이프라인 전역 누수를 잡는 가장 강력한 단일 검사다.
파일럿 관측값을 스펙 11.4에 고정하고 회귀 테스트로 묶었다."
```

---

### Task 20: T3 누수 검출 (이 서브프로젝트의 존재 이유)

**Files:**
- Test: `tests/evaluation/test_validation_leakage.py`

**Interfaces:**
- Consumes: `run_experiment` (Task 18), `window_random` 분할기 (Task 14)
- Produces: 없음 (검증 전용)

누수를 **일부러** 만들어 두 가지를 증명한다. (a) 가드가 켜져 있으면 차단된다. (b) 가드를 끄면 정확도가 LOSO 대비 15%p 이상 부풀려진다 — 즉 누수가 실제로 성능을 왜곡한다.

(b)가 없으면 가드가 지키는 것이 무엇인지 증명되지 않는다.

- [ ] **Step 1: 검증 테스트 작성**

`tests/evaluation/test_validation_leakage.py`:

```python
"""T3(누수 검출) — 스펙 8절. 이 서브프로젝트의 존재 이유."""

import json

import pytest

from src.datasets.contract import LeakageError
from src.evaluation.runner import run_experiment

PILOT = "config/experiments/pilot.yaml"
INFLATION_THRESHOLD = 0.15


def _run(tmp_path, name, **overrides):
    ov = {"output": {"results_dir": str(tmp_path / name)}}
    ov.update(overrides)
    return run_experiment(PILOT, overrides=ov)


def _acc(out_dir):
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["pooled_accuracy"]


@pytest.mark.slow
def test_t3a_guards_block_window_random_split(tmp_path):
    with pytest.raises(LeakageError, match="subjects appear in both"):
        _run(tmp_path, "blocked", evaluation={"splitter": "window_random"})


@pytest.mark.slow
def test_t3b_leakage_actually_inflates_accuracy(tmp_path):
    loso = _acc(_run(tmp_path, "loso"))
    leaky = _acc(
        _run(
            tmp_path, "leaky",
            evaluation={
                "splitter": "window_random",
                "guards": {"check_subject_overlap": False, "check_window_overlap": False},
            },
        )
    )
    assert leaky - loso >= INFLATION_THRESHOLD, (
        f"창 단위 무작위 분할이 LOSO({loso:.3f}) 대비 "
        f"{leaky - loso:.3f}만 부풀렸다. 15%p 미만이면 T3 설계를 재검토하라 "
        "— 누수가 성능을 왜곡하지 않는다면 가드가 지키는 것이 무엇인지 불분명하다"
    )


@pytest.mark.slow
def test_t3_unsafe_run_is_marked_in_directory_name(tmp_path):
    out = _run(
        tmp_path, "marked",
        evaluation={
            "splitter": "window_random",
            "guards": {"check_subject_overlap": False, "check_window_overlap": False},
        },
    )
    assert out.name.endswith("-UNSAFE")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["guards_disabled"] is True


@pytest.mark.slow
def test_t3_testview_fit_is_blocked_regardless_of_guard_config(tmp_path):
    """TestView.fit 차단은 설정으로 끌 수 없는 구조적 장치다."""
    from src.common.config import load_config
    from src.common.seeding import set_all_seeds
    from src.evaluation.runner import build_dataset
    from src.evaluation.splitters import get_splitter
    from sklearn.preprocessing import StandardScaler

    cfg = load_config(PILOT)
    cfg["evaluation"]["guards"] = {
        "check_subject_overlap": False,
        "check_window_overlap": False,
    }
    cfg["simulation"]["n_subjects"] = 3
    cfg["simulation"]["task"]["n_blocks_per_level"] = 1
    cfg["simulation"]["task"]["block_duration_s"] = 20
    cfg["simulation"]["eeg"]["sfreq_hz"] = 100

    dataset, _ = build_dataset(cfg, set_all_seeds(0))
    fold = next(iter(dataset.iter_folds(get_splitter("loso"))))

    with pytest.raises(LeakageError):
        fold.test.fit(StandardScaler())
```

- [ ] **Step 2: T3 실행**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_validation_leakage.py -v -m slow
```

Expected: 4 passed

- [ ] **Step 3: 커밋**

```bash
cd /d/Study_fNIRS
git add tests/evaluation/test_validation_leakage.py
git commit -m "test: T3 누수 검출 — 가드가 실제로 작동함을 증명

(a) 가드 ON이면 창 단위 무작위 분할이 차단된다.
(b) 가드 OFF면 정확도가 LOSO 대비 15%p 이상 부풀려진다.
(b)가 없으면 가드가 지키는 것이 무엇인지 증명되지 않는다."
```

---

### Task 21: T4 개인차 스윕

**Files:**
- Test: `tests/evaluation/test_validation_subject_variance.py`

**Interfaces:**
- Consumes: `run_experiment` (Task 18)
- Produces: 없음 (검증 전용)

계획서 가설 2("동일 수행 수준에서도 개인 간 전전두엽 활성도가 상이한 신경효율성 개인차")를 데이터가 실제로 담고 있는지 확인한다. 개인차가 없으면 LOSO와 within-subject의 차이가 사라져 **LOSO 검증 자체가 무의미**해진다.

- [ ] **Step 1: 검증 테스트 작성**

`tests/evaluation/test_validation_subject_variance.py`:

```python
"""T4(개인차 스윕) — 스펙 8절."""

import json

import pytest

from src.evaluation.runner import run_experiment

PILOT = "config/experiments/pilot.yaml"
VARIANCES = [0.0, 0.5, 2.0]


def _acc(tmp_path, splitter, variance):
    out = run_experiment(
        PILOT,
        overrides={
            "output": {"results_dir": str(tmp_path / f"{splitter}_{variance}")},
            "evaluation": {"splitter": splitter},
            "simulation": {"subject_variance": variance},
        },
    )
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))["pooled_accuracy"]


@pytest.mark.slow
def test_t4_loso_degrades_monotonically_with_subject_variance(tmp_path):
    accs = [_acc(tmp_path, "loso", v) for v in VARIANCES]
    assert accs[0] >= accs[1] >= accs[2], (
        f"개인차를 키워도 LOSO 성능이 단조 하락하지 않았다: {accs}. "
        "개인차가 신호에 반영되지 않았다면 LOSO 검증이 무의미해진다"
    )
    assert accs[0] - accs[2] > 0.02, (
        f"개인차 0 → 2.0에서 LOSO 하락폭이 {accs[0] - accs[2]:.3f}로 너무 작다"
    )


@pytest.mark.slow
def test_t4_within_subject_is_robust_to_subject_variance(tmp_path):
    accs = [_acc(tmp_path, "within_subject", v) for v in VARIANCES]
    assert accs[2] > accs[0] - 0.10, (
        f"within-subject가 개인차에 크게 흔들렸다: {accs}. "
        "피험자 내부에서는 개인차가 상수이므로 영향이 작아야 한다"
    )


@pytest.mark.slow
def test_t4_within_subject_beats_loso_when_variance_is_high(tmp_path):
    loso = _acc(tmp_path, "loso", 2.0)
    within = _acc(tmp_path, "within_subject", 2.0)
    assert within > loso, (
        f"개인차가 큰데도 within-subject({within:.3f})가 "
        f"LOSO({loso:.3f})를 넘지 못했다"
    )


@pytest.mark.slow
def test_t4_cv_methods_are_recorded_distinctly(tmp_path):
    """LOSO와 within-subject 결과를 혼용 표기하지 않는다 (CLAUDE.md 5.4)."""
    for splitter in ("loso", "within_subject"):
        out = run_experiment(
            PILOT,
            overrides={
                "output": {"results_dir": str(tmp_path / f"label_{splitter}")},
                "evaluation": {"splitter": splitter},
            },
        )
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["cv_method"] == splitter
```

- [ ] **Step 2: T4 실행**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe tests/evaluation/test_validation_subject_variance.py -v -m slow
```

Expected: 4 passed

임계값(`0.02`, `0.10`)이 실패하면 관측값을 확인하고 스펙 §11.4에 기록한 뒤 조정한다. **단조성 자체가 깨지면 임계값이 아니라 생성기를 의심하라** — `subject.theta`가 신호에 실제로 반영되고 있는지 Task 5·7의 gain 계산을 확인한다.

- [ ] **Step 3: 전체 테스트 실행**

```bash
cd /d/Study_fNIRS && .venv/Scripts/pytest.exe -v
```

Expected: 전부 통과 (slow 포함)

- [ ] **Step 4: 진행 로그 기록**

`.claude/skills/research-log` 스킬을 실행해 `run_logging.md`에 기록한다. 실험 기록 필수 항목(결과 디렉토리, git commit, config, 시드, 데이터 범위, **CV 방식**, **chance level**, 성능 평균·최악 피험자, 누수 점검 결과)을 빠짐없이 적는다.

- [ ] **Step 5: 커밋**

```bash
cd /d/Study_fNIRS
git add tests/evaluation/test_validation_subject_variance.py run_logging.md
git commit -m "test: T4 개인차 스윕 — LOSO 단조 하락 / within-subject 견고

개인차가 신호에 반영되지 않으면 LOSO 검증 자체가 무의미해진다.
계획서 가설 2(신경효율성 개인차)를 데이터가 담고 있는지 확인한다."
```

---

## 완료 기준

스펙 §12와 동일하다. 전부 체크되어야 이 서브프로젝트가 끝난다.

- [ ] T1~T4 전부 통과
- [ ] `.venv/Scripts/pytest.exe -v` 전체 통과
- [ ] 같은 시드로 두 번 실행 시 `metrics.json`이 동일 (Task 18 테스트가 강제)
- [ ] `results/<run_id>/`에 config·시드·git hash·환경이 빠짐없이 기록
- [ ] 누수를 일부러 만든 구성이 가드에 차단되는 것이 재현 가능한 테스트로 시연됨 (Task 20)
- [ ] `run_logging.md`에 파일럿 실행 결과 기록 (CV 방식·chance level 명시)
- [ ] 스펙 §11.4의 T2 정확도 범위가 관측값으로 확정됨

## 다음 서브프로젝트

**B(전처리 파이프라인)** — EEG PREP→필터→ASR→보간→에포크, fNIRS SCI→웨이블릿→대역통과→mBLL→에포크.
착수 시 이 계획의 다음 항목을 함께 처리한다.

- `src/simulation/components/artifacts/` 추가 (눈깜빡임·EMG·모션·광량드리프트) — B가 이를 제거하는지 검증하기 위해 필요
- `src/datasets/features_minimal.py`를 실제 특징 추출기로 교체 (시그니처 유지)
- fNIRS HbR 시간 지연 모델링, 개인·부위별 HRF, 채널 간 공간 상관 (스펙 §6.3의 의도적 단순화 해소)
- BIDS 라이터에 SNIRF·EDF 신호 파일 추가 (장비·몽타주 확정 후)
