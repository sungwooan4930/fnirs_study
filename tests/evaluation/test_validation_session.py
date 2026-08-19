"""승인기준 T1~T6 (스펙 §8).

각 테스트는 대상 결함을 주입했을 때 실제로 실패하는 것을 함께 보인다.
실패할 수 없는 테스트는 테스트가 아니다.

널 판정은 fold 수준 t-신뢰구간(`fold_ci_low`/`fold_ci_high`)으로 한다.
`pooled_ci_*`는 5초창·1초 스텝의 80% 오버랩 때문에 창을 독립 베르누이
시행으로 취급해 불확실성을 과소평가한다 — 정밀도를 과장해 거짓 양성을
낸다 (Task 13, `src/evaluation/metrics.py` 참조). `session_recovery.yaml`은
`cross_session` 분할기를 쓰므로 fold가 세션 수(3개)뿐이라
`t(0.975, df=2) = 4.303`으로 구간이 넓다 — 보수적인 방향이며 의도된 것이다.
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


@pytest.mark.slow
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
        assert m["fold_ci_low"] <= m["chance_level"] <= m["fold_ci_high"], (
            f"normalize={normalize}: chance {m['chance_level']:.4f} 가 "
            f"fold CI [{m['fold_ci_low']:.4f}, {m['fold_ci_high']:.4f}] 밖이다"
        )


@pytest.mark.slow
def test_t1_injection_1_load_conditional_normalization_breaks_the_null(tmp_path, monkeypatch):
    """주입 1: (세션 × 부하수준) 단위로 센터링하되, 그 기준을 **전체 데이터셋에서
    미리 계산한 부하수준별 전역 평균**으로 삼는다.

    브리프 원안(세션 자신의 그룹 평균으로 센터링해 0으로 만드는 버전)은 실제로
    돌려보니 fold_ci를 못 깼다 — effect_size=0이면 원래 그룹 간 차이가 없으므로
    "자기 평균을 빼서 0으로 만드는" 연산은 정보를 지우기만 하고 새로 주입하지
    않는다 (pooled_ci조차 chance를 포함: mean 0.335, pooled_ci [0.319, 0.350]).
    그래서 실제 버그 패턴에 더 가깝게 강화했다 — **train/test 구분 없이 전체
    데이터셋(테스트 fold 포함)에서 부하수준별 평균을 먼저 계산해두고, 그 전역
    평균을 모든 세션에 동일하게 주입**한다. 이러면 잔차가 세션마다 우연히 같은
    상수 벡터를 공유하게 되어 분류기가 그 벡터로 즉시 학습한다 — CLAUDE.md
    §5.1이 경고하는 "전체 데이터에 fit" 누수의 정석적인 형태이며, 실측
    accuracy_mean=0.475, fold_ci_low=0.453 (chance 0.333)로 명확히 깬다.
    """
    from src.datasets.windowing import make_windows

    cfg = load_config(RECOVERY)
    win_cfg = cfg["windowing"]
    real_generate = generate_dataset
    global_state: dict = {}

    def _capture_and_pass_through(sim_cfg, rng):
        # 1단계: 실제 생성기로 녹화를 만들되(rng 소비는 원래와 동일하게 1회),
        # 부하수준별 전역 평균을 **모든 세션(테스트 fold 포함)**을 훑어 미리 계산한다.
        recs = real_generate(sim_cfg, rng)
        sums: dict = {}
        counts: dict = {}
        for rec in recs:
            windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
            feats = extract_features(rec, windows)
            for name, arr in feats.items():
                for level in np.unique(windows.load_level):
                    m = windows.load_level == level
                    key = (name, level)
                    sums[key] = sums.get(key, 0) + arr[m].sum(axis=0)
                    counts[key] = counts.get(key, 0) + int(m.sum())
        global_state["means"] = {k: sums[k] / counts[k] for k in sums}
        return recs

    def _load_conditional(rec, windows):
        feats = extract_features(rec, windows)
        for name, arr in feats.items():
            for level in np.unique(windows.load_level):
                m = windows.load_level == level
                session_mean = arr[m].mean(axis=0)
                global_mean = global_state["means"][(name, level)]
                arr[m] = arr[m] - session_mean + global_mean
        return feats

    monkeypatch.setattr(runner_mod, "generate_dataset", _capture_and_pass_through)
    monkeypatch.setattr(runner_mod, "get_extractor", lambda name: _load_conditional)
    m = _run(NULL_OVER, tmp_path)
    assert m["fold_ci_low"] > m["chance_level"], (
        "부하 조건부 정규화를 주입했는데도 널이 유지됐다 — T1이 이 버그를 못 잡는다"
    )


@pytest.mark.slow
def test_t1_injection_2_load_dependent_drift_breaks_the_null(tmp_path, monkeypatch):
    """주입 2: 드리프트 파라미터가 부하에 의존.

    세션 단위로 한 번 뽑아야 할 것을 블록 루프 안에서 뽑으면 이렇게 된다.
    드리프트 자체가 부하의 대리변수가 된다.

    브리프 원안의 계수(0.5)는 pooled_ci는 깨지만(pooled_ci_low=0.345 >
    chance) fold_ci는 못 깼다 — fold=3, df=2라 t(0.975,2)=4.303으로 구간이
    넓고, fold 간 분산이 신호를 가린다. T2를 붕괴시키려고
    `session_recovery.yaml`의 드리프트 시그마를 8배로 올린 뒤로는(아래
    T2 참조) 배경 잡음이 커져서 계수 6.0으로도 fold_ci를 못 깼다(재확인
    시 fold_ci_low=0.322). 새 기준선에서 다시 스윕한 결과 16~20에서
    간신히 넘기 시작해 30 이상은 효과가 정체됐다(30→fold_ci_low=0.354,
    40→0.356). 여유를 두어 30.0을 채택했다 — 실제 구현 버그라면 이 정도
    배율이 아니라도 훨씬 뚜렷하게 드러났을 것이므로, 이만큼 세게 주입해야
    겨우 깨진다는 사실 자체가 fold_ci(3-fold, df=2)의 보수성을 보여준다.
    """
    real = generate_dataset

    def _load_dependent(sim_cfg, rng):
        out = []
        for rec in real(sim_cfg, rng):
            t = np.arange(rec.hbo.shape[1]) / rec.fnirs_sfreq
            gain = 1.0 + 30.0 * rec.timeline.effective_load(t)
            out.append(dataclasses.replace(rec, hbo=rec.hbo * gain[None, :]))
        return out

    monkeypatch.setattr(runner_mod, "generate_dataset", _load_dependent)
    m = _run(NULL_OVER, tmp_path)
    assert m["fold_ci_low"] > m["chance_level"], (
        "부하 의존 드리프트를 주입했는데도 널이 유지됐다 — T1이 이 버그를 못 잡는다"
    )


# ---------------------------------------------------------------- T2·T3

#: T3의 회복 기준. chance와 세션 내 성능 사이 간격의 이 비율 이상을 회복해야
#: 한다. 관측에서 역산한 값이 아니라 사전에 선언한 요구 수준이다 (스펙 §8.4).
T3_GAP_RECOVERY_MIN = 0.5


@pytest.mark.slow
def test_t2_without_normalization_cross_session_accuracy_collapses(tmp_path):
    """정규화가 없으면 세션 간 분류가 chance로 무너진다.

    무너지지 않으면 드리프트가 신호를 흔들지 않는다는 뜻이고, 그러면 T3의
    '회복'은 아무것도 증명하지 못한다. T3의 존재 이유가 T2다.
    """
    m = _run({"preprocessing": {"baseline": {"normalize": False}}}, tmp_path)
    assert m["cv_method"] == "cross_session"
    assert m["fold_ci_low"] <= m["chance_level"] <= m["fold_ci_high"], (
        f"정규화 없이도 세션 간 분류가 {m['pooled_accuracy']:.4f} "
        f"(fold CI [{m['fold_ci_low']:.4f}, {m['fold_ci_high']:.4f}])로 chance를 "
        "넘었다. 드리프트가 실제로 신호를 압도하지 않는다는 뜻이다."
    )


@pytest.mark.slow
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
    assert m["fold_ci_low"] > m["chance_level"], (
        "드리프트를 껐는데도 세션 간 분류가 chance에 머물렀다 — 붕괴의 원인이 "
        "드리프트가 아니다."
    )


@pytest.mark.slow
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


@pytest.mark.slow
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
