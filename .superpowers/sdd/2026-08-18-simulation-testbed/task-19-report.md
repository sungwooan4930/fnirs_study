# Task 19 리포트 — T1 널 테스트 + T2 효과 회복 (승인 기준)

## 상태
통과. T1(널 테스트) 2건 + T2(효과 회복) 4건, 총 6개 `@pytest.mark.slow` 테스트 전부 green.

## 변경 파일
- `config/experiments/pilot.yaml` (신규) — 계획서 스펙(EEG 1000 Hz, 12명, effect_size=0.8, seed=42)
- `config/experiments/null.yaml` (신규) — pilot과 동일, `effect_size: 0.0`만 다름
- `tests/evaluation/test_validation_effect.py` (신규) — T1×2, T2×4(베이스라인 회귀 포함)
- `tests/baselines/t2_pilot.json` (신규) — 파일럿 실행으로 생성, 손으로 채운 값 없음
- `docs/specs/2026-08-18-simulation-testbed-design.md` §11.4 — 미확정 항목을 관측값으로 확정
- `run_logging.md` — Task 19 진행 로그 추가

## 사전 규모 점검 (Step 3 이전 단일 실행)
브리핑이 요구한 대로, 전체 슬로우 테스트 파일을 돌리기 전에 파일럿 1회를 먼저 실행해 시간·메모리를 측정했다.

- wall-clock: **56.3초**
- tracemalloc peak: **약 2.0 GB**
- 사전 추정("EEG 1000 Hz·12명·9블록×60초 ≈ 피험자당 1,600만 샘플, 전체 약 1.5 GB")과 합치했다.
- 실행 가능한 규모로 판단해 config를 축소하지 않고 그대로 진행했다.

## T1·T2 실행 결과 (`test_validation_effect.py -m slow`)

```
tests/evaluation/test_validation_effect.py::test_t1_null_effect_stays_at_chance PASSED
tests/evaluation/test_validation_effect.py::test_t1_null_holds_at_pilot_scale_too PASSED
tests/evaluation/test_validation_effect.py::test_t2_effect_is_recovered_above_chance PASSED
tests/evaluation/test_validation_effect.py::test_t2_accuracy_is_not_suspiciously_perfect PASSED
tests/evaluation/test_validation_effect.py::test_t2_reports_worst_subject_not_just_mean PASSED
tests/evaluation/test_validation_effect.py::test_t2_matches_recorded_baseline PASSED

6 passed in 210.19s (0:03:30)
```

전체 스위트: `168 passed in 217.52s (0:03:37)`, 경고 0건 (기존 162 + 신규 6).

## 관측값 (seed=42, EEG 1000 Hz, 12명, LOSO-CV, `config/experiments/pilot.yaml`)

| 지표 | 값 |
|---|---|
| chance level (3분류) | 0.3333 |
| pooled_accuracy (T2) | 0.6542 |
| accuracy_mean | 0.6542 |
| accuracy_worst | 0.2960 (sub-05) |
| pooled 95% CI | (0.6420, 0.6662) |
| binomtest_p (T2, effect_size=0.8) | 0.0 (< 0.01) |

`config/experiments/null.yaml` (effect_size=0.0, 그 외 동일): pooled CI가 chance(0.3333)를 포함하고 `binomtest_p > 0.05` — T1 통과, 파이프라인 전역 누수 없음을 뒷받침.

## 해석
- **T1**: 효과가 0인 데이터에서 LOSO 정확도가 chance를 벗어나지 않았다 — 윈도잉·특징·분할·적합 어디에도 전역 누수가 없다는 가장 강력한 단일 증거.
- **T2**: 심어둔 효과(0.8)가 유의하게(p≈0) 회수되었고, 동시에 0.6542로 98% 근처에 전혀 이르지 않았다 — 개인차가 있는 합성 데이터에서 LOSO가 근사 완벽 정확도를 내지 않는다는 것도 함께 확인되어, "너무 잘 맞아서 의심스러운" 패턴이 아니다.
- **베이스라인**: `tests/baselines/t2_pilot.json`은 위 명령을 코드로 실행해 생성했고 회귀 테스트가 이를 되읽는다. 값이 의도적으로 바뀔 때(생성기·특징 변경 등)는 같은 명령으로 다시 생성하고 `run_logging.md`에 이유를 남기는 절차를 스펙에 남겨두었다.

## 우려/후속
- T3(고의적 누수 → 가드 차단 시연)·T4(개인차 스윕)는 아직 미착수. 목표 3 서브프로젝트 A+D의 승인 기준 4개 중 2개만 이번 태스크에서 다뤘다.
- 파일럿 1회 56초·2GB 규모가 반복 실험(스윕 등)에서 누적되면 시간이 커질 수 있다 — T4 스윕 설계 시 참고 필요.

---

## Fix round 1/5 — 베이스라인 회귀 허용 오차 강화

**리뷰 지적:** `test_t2_matches_recorded_baseline`의 허용 오차 0.05는 존재하지 않는 노이즈에
맞춰 보정된 값이었다. Task 18에서 확인된 대로 `set_all_seeds`는 결정론적 `Generator`를
반환하고 같은 시드의 `run_experiment`는 `metrics.json` 전체가 비트 단위로 재현된다 —
즉 이 테스트가 흡수해야 할 실행 간 변동은 0이다. 반면 T1의 신뢰구간 반폭은 n=6000에서
약 1.19%p로, 그보다 3배 이상 넓은 0.05 허용 오차는 T1이라면 잡아낼 3~4%p 수준의 회귀를
조용히 통과시킬 수 있었다.

**조치:** 허용 오차를 `0.05` → `0.005`로 강화. 남은 여유는 플랫폼·numpy 빌드 간 부동소수점
마지막 자리 차이만 흡수하도록 의도된 것이며, 표본 노이즈를 위한 것이 아니라는 점을
docstring에 명시해 이후 다시 느슨하게 되돌리지 않도록 했다.

**재실행 결과 (`test_validation_effect.py -m slow`):** 6 passed in 256.92s (0:04:16).

**관측값 재확인:** 별도 실행에서 `pooled_accuracy = 0.6541666666666667`,
`tests/baselines/t2_pilot.json`의 값과 **완전히 일치**(diff = 0.0) — Task 18의
비트 재현성 결론이 그대로 유지됨을 확인했다. 표본 노이즈가 아니라는 리뷰의 전제가 맞았다.

**변경 파일:** `tests/evaluation/test_validation_effect.py` (허용 오차·docstring만 수정, 다른 5개
테스트·config·베이스라인 값은 변경 없음)
