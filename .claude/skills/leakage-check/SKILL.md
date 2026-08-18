---
name: leakage-check
description: 모델 학습·평가 코드의 데이터 누수(data leakage)를 점검한다. LOSO-CV, 슬라이딩 윈도우 분할, 전처리 fit 위치, 하이퍼파라미터 튜닝 절차를 다루는 코드를 작성·수정·리뷰할 때 반드시 실행. "성능이 예상보다 높다", "정확도 95%", "CV 점수" 같은 신호가 보일 때도 실행.
---

# 데이터 누수 점검

본 연구는 **5초 창 · 1초 스텝(80% 오버랩)** 슬라이딩 윈도우를 쓴다.
누수가 나면 정확도가 90%대로 치솟지만 **전부 허구**다. 발표·논문·특허가 무너진다.

점검 대상 코드를 읽고 아래 7개 항목을 **하나씩** 검증한다. 추측하지 말고 코드에서 근거를 찾는다.

## 1. 분할 단위가 피험자인가

```python
# ❌ 치명적 — 인접 윈도우가 train/test에 동시 존재
train_test_split(X_windows, y, test_size=0.2, shuffle=True)
KFold(n_splits=5, shuffle=True)

# ✅
LeaveOneGroupOut()          # groups=subject_ids
GroupKFold(n_splits=5)      # groups=subject_ids
```

**확인**: `groups` 인자에 피험자 ID가 실제로 전달되는가? 세션·시행 ID를 잘못 넘기지 않았는가?

## 2. 전처리 fit이 train fold 안에서만 일어나는가

```python
# ❌ 전체 데이터로 fit → test 분포 정보 유출
X = StandardScaler().fit_transform(X_all)
for tr, te in cv.split(X, y, groups): ...

# ✅ Pipeline으로 fold마다 재fit
pipe = Pipeline([('sc', StandardScaler()), ('clf', SVC())])
cross_val_score(pipe, X, y, groups=g, cv=LeaveOneGroupOut())
```

**대상**: `StandardScaler`, `MinMaxScaler`, `PCA`, `SelectKBest`, 결측 대치, 클래스 가중치,
채널 정규화, **베이스라인 보정 기준 구간**.

## 3. 특징 선택이 전체 데이터에서 이뤄지지 않았는가

상관·ANOVA·중요도 기반 특징 선택을 CV 바깥에서 한 번 수행하면 누수다.
반드시 fold 내부에서 수행한다.

## 4. 하이퍼파라미터 튜닝이 test fold를 보지 않는가

```python
# ❌ 같은 CV로 튜닝하고 그 점수를 최종 성능으로 보고
GridSearchCV(clf, params, cv=logo).fit(X, y, groups=g).best_score_

# ✅ 중첩(nested) CV — 바깥 fold는 튜닝에 관여하지 않음
inner = GroupKFold(4)
outer = LeaveOneGroupOut()
cross_val_score(GridSearchCV(clf, params, cv=inner), X, y, groups=g, cv=outer)
```

## 5. 시간적 누수 — 윈도우 경계

- 같은 시행(trial)의 윈도우가 서로 다른 fold에 갈라져 있지 않은가?
- 에포크 분할 전에 전체 시계열에 필터를 적용했다면, 필터의 시간 지원(support)이
  fold 경계를 넘어 정보를 옮기지 않는가? (특히 `filtfilt` 양방향 필터)
- 베이스라인 구간이 test 구간과 겹치지 않는가?

## 6. 라벨 누수

- 4계층 라벨 중 **행동(반응시간·정오답)** 을 특징으로도 쓰고 라벨로도 쓰지 않는가?
  → `accuracy`·`response_latency` 차원을 예측하면서 같은 값을 입력 특징에 넣으면 자명한 누수.
- 자기보고(NASA-TLX·KSS)는 과제 **후** 측정이다. 시행 중 윈도우 라벨로 쓸 때 시간 정합성을 확인한다.

## 7. 보고 정합성

- chance level이 명시되어 있는가? (3수준 분류 → 33.3%)
- LOSO 결과인지 within-subject 결과인지 **명시**되어 있는가?
- 피험자별 분산(std, 최악 피험자 성능)이 함께 보고되는가? 평균만 보고하지 않는다.

---

## 출력 형식

점검 후 다음 형식으로 보고한다:

```
## 누수 점검 결과

| # | 항목 | 판정 | 근거 |
|---|------|------|------|
| 1 | 분할 단위 | ✅/❌/N.A. | 파일:라인 + 인용 |
...

### 발견된 누수
(있으면) 심각도 · 위치 · 왜 누수인지 · 수정 방법

### 결론
이 코드의 성능 수치를 신뢰할 수 있는가: 예 / 아니오 / 조건부
```

누수가 하나라도 확인되면 **해당 실험 결과는 폐기 대상**임을 명시한다.
확신이 없으면 "확인 필요"로 표기하고 추측으로 통과시키지 않는다.
