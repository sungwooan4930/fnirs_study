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

    **곱셈(이득) 형태에서 덧셈(오프셋) 형태로 바꿨다.** `src/preprocessing/
    baseline.py`가 `concentration_delta`에 베이스라인 산포로 나누는 보정을
    추가한 뒤(Task 14 컨트롤러 정정 — fNIRS 곱셈 이득 드리프트를 소거하기
    위함), 곱셈 이득 기반 주입(`hbo *= 1 + k*load`)이 **바로 그 보정이
    상쇄하는 대상**이 되어 계수를 30→150까지 올려도 fold_ci_low가 0.29대에
    고정되고 더 이상 못 깼다 — 정규화가 제 역할을 하고 있다는 방증이지만,
    이 테스트의 목적(드리프트가 부하의 대리변수가 되는 버그를 잡는 것)에는
    맞지 않는다. 그래서 **베이스라인 구간에는 나타나지 않고 과제 블록에서만
    부하에 비례해 나타나는 덧셈 오프셋**(`hbo += k*load`)으로 바꿨다 —
    세션 시작 베이스라인만 보는 정규화는 이런 블록별 오프셋을 원리적으로
    잡을 수 없다(§3.8: 세션 내 드리프트는 애초에 보정이 아니라 플래그
    대상). 계수 0.05만으로도 fold_ci_low=0.639로 chance(0.333)를 크게
    넘어(mean 0.83) 명확히 깬다.
    """
    real = generate_dataset

    def _load_dependent(sim_cfg, rng):
        out = []
        for rec in real(sim_cfg, rng):
            t = np.arange(rec.hbo.shape[1]) / rec.fnirs_sfreq
            offset = 0.05 * rec.timeline.effective_load(t)
            out.append(dataclasses.replace(rec, hbo=rec.hbo + offset[None, :]))
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
