# 합성 데이터 테스트베드 + LOSO 평가 프레임워크 설계

**날짜:** 2026-08-18
**대상:** 연구계획서 목표 3의 서브프로젝트 A+D
**상태:** 설계 승인 완료 → 구현 계획 작성 대기

---

## 1. 목적

계획서 목표 3(인지상태 추정 멀티모달 AI 모델)은 단일 스펙 범위를 넘는다(M5–8, 연구보조원 5명).
5개 서브시스템으로 분해했고, 이 문서는 그중 **A(합성 데이터 생성기) + D(LOSO 평가 프레임워크)**를 다룬다.

| | 서브시스템 | 의존 | 상태 |
|---|---|---|---|
| **A** | 합성 데이터 생성기 | — | **이 문서** |
| **B** | 전처리 파이프라인 (EEG PREP→ASR / fNIRS SCI→mBLL) | A | 미착수 |
| **C** | 특징 추출 + 데이터셋 | B | 계약·윈도잉·최소 특징(임시)만 이 문서. 실제 특징 추출은 B 이후 |
| **D** | LOSO-CV 평가 프레임워크 + 누수 가드 | C 계약 | **이 문서** |
| **E** | 융합 3전략 모델 | C, D | 미착수 |

### A와 D를 함께 묶는 이유

합성 데이터는 **효과크기를 우리가 심어 넣으므로 정답을 안다.** 따라서:

- 올바른 LOSO 하네스 → 심어둔 효과 근처를 재현
- 누수 있는 하네스 → 부풀려진 수치

즉 **누수 가드가 실제로 누수를 잡는다는 것을 증명할 수 있는 유일한 시점**이다.
실제 데이터에서는 정답을 모르므로 영원히 증명할 수 없다.

반대로 B(전처리)나 E(모델)를 먼저 만들면, 결과가 이상할 때 원인이 신호·전처리·모델·평가 중
어디인지 분리할 수 없다.

### 왜 이것이 최우선인가

`CLAUDE.md` §5.1 — 계획서의 **5초 창·1초 스텝은 80% 오버랩**이다. 윈도우 단위로 분할하면
정확도가 90%대로 치솟지만 전부 허구다. 누수는 **조용히 실패한다** — 성능이 *올라가서*
아무도 의심하지 않는다. 이 위에 논문 3편·특허 3건이 세워진다.

---

## 2. 범위

### 포함
- 합성 EEG(30ch/1kHz) · fNIRS(48ch/10.4Hz) · 행동 신호 생성
- 피험자별 랜덤효과(신경효율성 개인차) 모델링
- 최소 특징 추출기 (임시. B 완성 후 교체)
- 5초창·1초스텝 윈도잉
- 누수 방지 데이터셋 계약
- LOSO-CV 하네스 + 누수 가드
- 자체 경량 실험 러너 (YAML → 결과 디렉토리)
- 검증 테스트 4종

### 제외 (YAGNI — 명시적으로 자름)
| 제외 항목 | 이관 대상 |
|---|---|
| 아티팩트 모델 (눈깜빡임·EMG·모션·광량드리프트) | B |
| 실제 전처리 파이프라인 | B |
| 융합 3전략 모델 — 이번엔 더미 분류기(로지스틱 회귀)로 하네스만 검증 | E |
| BIDS 완전 준수 — 라이터 골격만 | 실장비 포맷 확정 후 |
| MLflow·실험 추적 UI | 필요 시 추후 |
| 실시간·LSL 스트리밍 | 목표 2 |

---

## 3. 확정된 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 하드웨어 전략 | 시뮬레이션 우선 | `CLAUDE.md` §9.1 |
| 신호 현실성 | 수준 2 **구조** + 단계적 구현 (성분 플러그인 아키텍처, 이번엔 통계적 성분 + HRF 컨볼루션) | 생성기를 두 번 만들지 않으면서 범위 억제 |
| 행동 2차원 처리 | **선행 예측으로 분리** — t 시점 뇌신호로 t+Δ 행동 예측 | 계획서 가설 2("행동 오류 발현 약 1.2초 전 선행 신호"). 동시점 예측은 §5.1 라벨 누수 |
| 누수 방지 | **계약 강제형** — `iter_folds()`로만 접근, `TestView.fit()` 차단 | 5명 × 1년 × 수백 실험을 사람의 규율에 맡길 수 없음 |
| 실험 추적 | 자체 경량 러너 (YAML + CLI + 결과 디렉토리 + pandas 집계) | 외부 전송 없음(IRB 안전), 의존성 최소 |

---

## 4. 아키텍처

### 4.1 데이터 흐름

```
SubjectProfile          피험자별 랜덤효과 θ_s ~ N(0, τ²)
        ↓
CognitiveStateTimeline  6차원 ground truth 궤적 (과제 블록 구조)
        ↓               신호 성분 플러그인 (가산 결합)
SyntheticRecording      EEG 30ch/1kHz · fNIRS 48ch/10.4Hz · 행동 이벤트
        ↓               [B 전처리 자리 — 이번엔 비워둠]
        ↓               최소 특징 추출기 (임시)
        ↓               5초창·1초스텝 윈도잉 (subject/trial 경계 보존)
WindowedDataset ★계약★   subject_ids 필수
        ↓               .iter_folds(splitter)
FoldView(TrainView, TestView)  →  모델 학습·평가
        ↓
ExperimentResult        results/<run_id>/  (config·시드·git hash·환경)
```

**최소 특징 추출기를 넣는 이유:** 전처리(B) 없이는 raw에서 특징으로 갈 길이 없고,
특징 없이는 LOSO 하네스를 검증할 수 없다. EEG 대역파워(Welch)·fNIRS HbO 평균/기울기·
행동 정답률/RT 수준의 임시 구현이며, B 완성 시 동일 인터페이스로 교체한다.

### 4.2 모듈 배치

```
src/
├── simulation/
│   ├── state.py            # CognitiveStateTimeline
│   ├── subject.py          # SubjectProfile (피험자 랜덤효과)
│   ├── components/
│   │   ├── base.py         # SignalComponent 프로토콜
│   │   ├── eeg_oscillation.py   # 1/f 배경 + θ·α 진동
│   │   ├── eeg_erp.py           # P300·N200 템플릿
│   │   ├── fnirs_hrf.py         # 신경활성 → HRF 컨볼루션 → HbO/HbR
│   │   └── behavior.py          # 정답확률·RT (t+Δ 관측)
│   └── recording.py        # SyntheticRecording 조립 + BIDS 라이터 골격
├── datasets/
│   ├── contract.py         # WindowedDataset · FoldView · TrainView · TestView
│   ├── windowing.py        # 5초창·1초스텝, 경계 보존
│   ├── features_minimal.py # 임시 특징 추출기 (B 완성 후 교체)
│   └── labels.py           # 4계층 정렬, 선행 타깃 생성
└── evaluation/
    ├── splitters.py        # LOSO · GroupKFold (subject 강제)
    ├── harness.py          # fold 실행, 중첩 CV
    ├── guards.py           # 누수 가드
    ├── metrics.py          # 정확도·chance·ΔAccuracy·준거상관·피험자별 분산
    └── runner.py           # YAML → 실행 → 결과 디렉토리
```

---

## 5. 데이터셋 계약 (핵심)

### 5.1 `WindowedDataset`

```python
@dataclass(frozen=True)
class WindowedDataset:
    _X: dict[str, np.ndarray]      # {"eeg","fnirs","behavior"} → (n_win, n_feat)
    _y: dict[str, np.ndarray]      # {"cognitive_load", ...} → (n_win,)
    _subject_ids: np.ndarray       # (n_win,) 필수
    _window_times: np.ndarray      # (n_win, 2) start/end 초. 경계 검사용
    _trial_ids: np.ndarray         # (n_win,)

    def iter_folds(self, splitter) -> Iterator[FoldView]: ...
```

**X·y를 직접 노출하는 프로퍼티는 두지 않는다.** fold를 거치지 않으면 데이터에 닿을 수 없다.
이것이 "전역 fit"을 타입 수준에서 불가능하게 만드는 장치다.

### 5.2 `TrainView` / `TestView`

```python
class TrainView:
    """fold의 학습 절반. sklearn 관용구를 그대로 사용 가능."""
    def X(self, modalities: list[str]) -> np.ndarray: ...
    def y(self, target: str) -> np.ndarray: ...
    def groups(self) -> np.ndarray: ...     # 중첩 CV용 subject id

class TestView:
    def X(self, modalities: list[str]) -> np.ndarray: ...
    def y(self, target: str) -> np.ndarray: ...
    def transform(self, transformer): ...              # 허용
    def fit(self, *a, **k):           raise LeakageError(...)
    def fit_transform(self, *a, **k): raise LeakageError(...)
```

`TrainView` 내부에서는 `Pipeline`·`GridSearchCV`를 평소대로 쓴다.
**sklearn을 대체하지 않고 감싼다.**

---

## 6. 합성 생성기

### 6.1 상태 → 신호 인과 구조

```
SubjectProfile          θ_s ~ N(0, τ²)          신경효율성 개인차 (계획서 가설 2)
        ↓
CognitiveStateTimeline  n-back 0/2/3 블록  →  인지부하 저/중/고
                        시간 경과          →  피로 단조 증가
                        블록 내 변동       →  주의·몰입
        ↓
EEG   = 1/f 배경 + θ진동·α진동(진폭이 상태·θ_s에 변조) + ERP 템플릿(자극 정렬)
fNIRS = 상태 → 신경활성 → HRF 컨볼루션 → HbO,   HbR = k·HbO + 독립잡음  (k < 0)
행동  = 상태(t) → 정답확률·RT 분포,   단 관측 시점은 t+Δ    ← 선행 예측의 근거
```

### 6.2 설정 가능한 손잡이 (검증 스윕용)

| 손잡이 | 의미 | 비고 |
|---|---|---|
| `effect_size` | 상태 → 신호 결합 강도의 **전역 배율** | `0`이면 널 데이터 (§8 T1) |
| `subject_variance` (τ²) | 개인차 크기 | 키우면 LOSO가 어려워짐 |
| `lead_delta_s` | 행동 선행 지연 | 계획서 근거 1.2초 |
| `n_subjects` | 피험자 수 | **하드코딩 금지** (`CLAUDE.md` §9.2 미확정) |

`effect_size`는 단일 스칼라가 여러 신호 성분에 동시 적용되는 **배율**이다.
성분별 기여 비율은 코드에 고정하고(예: θ파워 결합 1.0 · HbO 결합 0.8 · RT 결합 0.6),
`effect_size`가 이 전체를 스케일한다. 이렇게 두면 "d를 키우면 모든 모달이 함께 강해진다"는
단순한 해석이 가능해 스윕 결과를 읽기 쉽다. 실제 관측 d는 파일럿에서 측정해 §11.4에 고정한다.

### 6.3 의도적 단순화 (스펙에 명시하여 나중에 오해 방지)

| 단순화 | 실제 생리 | 이관 |
|---|---|---|
| `HbR = k·HbO + 잡음` (k ≈ −0.33) | HbR 진폭은 HbO의 약 1/3이고 **시간 지연도 다름** | B |
| 정준 HRF 단일 형태 | HRF는 개인·부위별로 다름 | B |
| 채널 간 공간 상관 없음 | 인접 채널은 상관됨 | B |
| 아티팩트 없음 | 눈깜빡임·EMG·모션·광량드리프트 | B |
| 상태 → 신호가 선형 | 비선형·포화 존재 | E 착수 시 재검토 |

이 단순화들은 **테스트베드 목적(하네스 검증)에는 무해**하다. 신호가 얼마나 현실적인지가 아니라
"심은 효과를 하네스가 정직하게 회수하는가"가 검증 대상이기 때문이다.

---

## 7. 평가 프레임워크

### 7.1 기록 지표

| 지표 | 비고 |
|---|---|
| 정확도 (평균 ± std) | |
| **chance level** | 3수준 → 33.3%. `CLAUDE.md` §5.4 필수 |
| **CV 방식 명시** | `loso` / `within_subject`. 혼용 표기 금지 |
| **피험자별 성능** | 평균만 보고 금지. **최악 피험자** 포함 |
| 혼동행렬 | |
| ΔAccuracy | 행동 단일 모달 대비 |
| 준거 상관 | 자기보고·성취와의 r (해당 시) |
| 제외된 윈도우 수 | 선행 타깃이 녹화 끝을 넘어 제외된 개수 |

### 7.2 결과 디렉토리

```
results/<run_id>/
├── config.yaml      # 실행 시점 config 스냅샷
├── env.json         # python·패키지 버전, git commit hash, platform
├── metrics.json     # 집계 지표
├── per_fold.csv     # fold(=피험자)별 성능
└── log.txt
```

`run_id` = `<run_name>_seed<N>_<git_short_hash>`

작업 트리가 dirty이면 `run_id`에 `-dirty` 접미사를 붙이고 `env.json`에 `git_dirty: true`와
`git diff` 전문을 함께 저장한다. 커밋되지 않은 코드로 낸 결과가 나중에 재현 불가능해지는 것을 막는다.

### 7.3 설정 스키마 (예시)

```yaml
run_name: testbed_smoke
seed: 42                        # 미지정 시 실행 거부

simulation:
  n_subjects: 12                # 미확정(§9.2) — 설정값으로만
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 60
    n_blocks_per_level: 3
  eeg:   { n_channels: 30, sfreq_hz: 1000 }
  fnirs: { n_channels: 48, sfreq_hz: 10.4, hbr_coupling: -0.33 }

windowing:
  window_s: 5.0
  step_s: 1.0

features:
  extractor: minimal            # B 완성 후 교체

dataset:
  targets: [cognitive_load]           # 3수준
  lead_targets: [accuracy, response_latency]
  modalities: [eeg, fnirs, behavior]

evaluation:
  splitter: loso
  model: logistic_regression          # 더미
  nested_cv: { enabled: false }
  guards:
    check_subject_overlap: true
    check_window_overlap: true

output:
  results_dir: results
```

**config에 없는 키가 오면 실행을 거부한다.** 오타로 인한 조용한 기본값 사용을 막기 위함이다.

---

## 8. 검증 전략 — 승인 기준

네 테스트가 통과해야 "테스트베드가 작동한다"고 말할 수 있다.

| # | 테스트 | 조건 | 합격 기준 | 실패가 뜻하는 것 |
|---|---|---|---|---|
| **T1** | **널 테스트** | `effect_size=0` 또는 라벨 셔플 | LOSO 정확도 95% CI가 chance(1/3)를 포함. 이항검정 p > 0.05 | **파이프라인 어딘가에 누수.** 가장 강력한 단일 검사 |
| **T2** | **효과 회복** | `effect_size=0.8` | chance보다 유의하게 높고 100%가 아님. 구체 범위는 파일럿 관측값을 회귀 기준으로 고정 | 생성기·특징·하네스 중 하나가 고장 |
| **T3** | **누수 검출** | (a) 가드 ON + 윈도우 랜덤 split 시도<br>(b) 가드 OFF + 동일 시도 | (a) `LeakageError` 발생<br>(b) 정확도가 LOSO 대비 **+15%p 이상** 부풀려짐 | 가드가 무력하거나, 누수가 실제로는 성능을 부풀리지 않음(=테스트 설계 오류) |
| **T4** | **개인차 스윕** | τ² ∈ {0, 0.5, 2.0} | LOSO 정확도가 **단조 하락**하고, within-subject 정확도는 τ² 증가에 따라 **유의하게 하락하지 않음** | 개인차가 신호에 반영 안 됨 → LOSO 검증이 무의미 |

**T3가 이 서브프로젝트의 존재 이유다.** 누수를 일부러 만들어 부풀려진 수치를 관측하고,
가드가 그것을 차단함을 증명한다.

### 가드 비활성화에 대하여

`evaluation.guards.*`를 `false`로 두는 것은 **T3(b) 시연 전용**이다. 실제 실험에서는 금지한다.
러너는 가드가 하나라도 꺼진 실행에 대해 `metrics.json`에 `guards_disabled: true`를 기록하고
결과 디렉토리 이름에 `-UNSAFE` 접미사를 붙인다. 이 표식이 붙은 결과는 논문·보고서에 인용할 수 없다.

`TestView.fit()` 차단은 설정으로 끌 수 없는 **구조적 장치**다. 가드 설정과 무관하게 항상 작동한다.

**T1이 가장 강력하다.** 파이프라인 어디에 누수가 있든 셔플 라벨에서 chance를 넘기 때문이다.

---

## 9. 오류 처리

원칙: **조용한 실패 금지.** 연구 코드에서 경고는 무시되고, 무시된 경고는 논문에 실린다.

| 상황 | 처리 |
|---|---|
| `TestView.fit()` 호출 | `LeakageError` — 즉시 중단. 경고 아님 |
| fold 간 subject 중복 | `LeakageError` — 하네스가 매 fold 검사 |
| train/test 윈도우 시간 겹침 | `LeakageError` — `_window_times`로 검사 |
| 선행 타깃이 녹화 끝을 넘음 | 해당 윈도우 제외 + **제외 수를 결과에 기록** |
| 시드 미지정 | 실행 거부 (`CLAUDE.md` §5.2) |
| config에 정의되지 않은 키 | 실행 거부 |
| 특정 fold에서 클래스가 1개뿐 | 실행 거부 — 정확도가 무의미해짐 |

---

## 10. 테스트 전략

| 계층 | 내용 |
|---|---|
| **단위** | 각 신호 성분이 지정한 통계적 성질을 갖는가 (θ파워가 상태에 따라 변하는가, HbO↔HbR 상관이 음인가, ERP가 자극에 정렬되는가) |
| **계약** | `TestView.fit` 차단, `iter_folds` 밖에서 데이터 접근 불가, subject 중복 탐지, 윈도우 겹침 탐지 |
| **통합** | §8의 T1~T4. 느리므로 `@pytest.mark.slow`로 분리 |
| **재현성** | 같은 시드 → bit-exact 동일 배열 |

TDD로 진행한다 (`CLAUDE.md` §6). 특히 계약 계층은 **차단되어야 할 동작을 먼저 테스트로 작성**한다.

---

## 11. 미확정 사항

`CLAUDE.md` §9.2와 동기화한다.

1. **참가자 수** — 계획서 내부 불일치(100명 vs 30명). 코드에 하드코딩하지 않고 `n_subjects` 설정값으로만 취급.
2. **"3수준 분류"의 정의** — 이 스펙은 **인지부하 3수준(n-back 0/2/3에서 유도)**으로 가정한다.
   6차원 각각을 3수준으로 하는 것인지 확인 필요.
3. **실장비 모델명·fNIRS 몽타주** — BIDS 라이터 완성에 필요. 이번엔 골격만.
4. **T2의 구체적 정확도 범위** — 파일럿 실행 관측값으로 확정 후 회귀 기준으로 고정.

---

## 12. 성공 기준

- [ ] T1~T4 전부 통과
- [ ] `pytest` 전체 통과, 계약 계층 커버리지 100%
- [ ] 같은 시드로 두 번 실행 시 `metrics.json`이 동일
- [ ] `results/<run_id>/`에 config·시드·git hash·환경이 빠짐없이 기록
- [ ] 누수를 일부러 만든 구성이 가드에 차단되는 것을 **재현 가능한 테스트로 시연**
- [ ] `run_logging.md`에 파일럿 실행 결과 기록 (CV 방식·chance level 명시)

---

## 13. 다음 단계

이 스펙 승인 후 `superpowers:writing-plans`로 구현 계획을 작성한다.
이후 서브프로젝트 순서: **B(전처리) → C(특징) → E(융합 3전략)**.
