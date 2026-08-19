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
from scipy import stats as scipy_stats

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

#: T3의 회복 기준. `off`(정규화 없음)와 `ceiling_nodrift`(드리프트 비활성 +
#: 정규화 없음, 같은 분할기·같은 시드) 사이 간격의 이 비율 이상을 `on`이
#: 회복해야 한다. 관측에서 역산한 값이 아니라 사전에 선언한 요구 수준이다
#: (스펙 §8.4). **`within_subject`("세션 내 성능")는 상한이 아니다** — 컨트롤러
#: 정정(2026-08-19)으로 폐기됐다. 같은 세션 안에서 블록만 나누는 분할이라
#: `cross_session`과 애초에 답하는 질문이 다르고, 드리프트가 전혀 없어도
#: `cross_session`이 도달하지 못한다 (스펙 §8.4.1 실측: ceiling_nodrift=0.5743
#: vs within_subject=0.8583). 이 상수의 이름·수치(0.5)는 그대로지만, 무엇을
#: 상한으로 삼는지는 아래 `test_t3_normalization_recovers_most_of_the_gap`의
#: `ceiling_nodrift` 계산을 봐야 정확하다.
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


#: T2 결함 주입·ceiling_nodrift 계산에 공용으로 쓰는 "드리프트 완전 비활성"
#: 오버라이드.
_NO_DRIFT = {
    "fnirs_gain_sigma": 0.0, "fnirs_offset_sigma": 0.0,
    "eeg_gain_sigma": 0.0, "eeg_noise_sigma": 0.0,
    "within_session_rate": 0.0,
}



@pytest.mark.slow
def test_t3_normalization_recovers_most_of_the_gap(tmp_path):
    """정규화 후 세션 간 성능이 '드리프트가 없었다면 도달했을' 성능 쪽으로 회복한다.

    **상한을 `within_subject`에서 `드리프트 비활성 cross_session`으로 바꿨다**
    (컨트롤러 정정 — 최초 브리프의 결함이지 구현 결함이 아니다). `within_subject`
    는 같은 세션 **안에서** 블록만 나누는 분할이라, 모델이 그 세션의 채널
    이득을 이미 본 채로 평가된다 — 세션을 가로지르는 문제(`cross_session`)와는
    애초에 답하는 질문이 다르다. 실측이 이를 증명한다: 드리프트를 전부 꺼도
    (`_NO_DRIFT`, off와 동일하게 정규화도 끔) cross_session은 ~0.51에 그쳐
    within_subject(~0.86)에 크게 못 미친다 — 즉 원래 기준은 (a)드리프트가
    낸 손상과 (b) "세션 간 vs 세션 내"라는 분할 방식 자체의 본질적 격차를
    뒤섞어 재고 있었다. (b)까지 정규화 탓으로 돌리면 정규화가 완벽해도
    영원히 실패하는 기준이 된다.

    새 상한(`ceiling_nodrift`)은 "같은 분할기·같은 시드, 드리프트 시그마만
    0" — 측정 드리프트가 없었다면 실제로 도달 가능했을 성능이다. seed가
    같으므로 §5.5의 난수열 규율(드리프트를 꺼도 다른 난수열로 넘어가지
    않음)이 여기서도 성립한다.

    `within_subject`는 **맥락으로만** 실행해 실패 메시지에 함께 남긴다 —
    이것과 `cross_session` 계열 수치를 같은 것으로 표기하지 않는다
    (CLAUDE.md §5.4: 서로 다른 CV 방식을 혼용 표기 금지, 나란히·구분해서
    보고).
    """
    off = _run({"preprocessing": {"baseline": {"normalize": False}}}, tmp_path / "off")
    on = _run({"preprocessing": {"baseline": {"normalize": True}}}, tmp_path / "on")
    ceiling_nodrift = _run(
        _deep_update(
            {"preprocessing": {"baseline": {"normalize": False}}},
            {"simulation": {"drift": _NO_DRIFT}},
        ),
        tmp_path / "ceiling_nodrift",
    )
    # within_subject는 맥락용이다 — 상한이 아니다. 위 docstring 참조.
    within = _run({"evaluation": {"splitter": "within_subject"}}, tmp_path / "within")

    assert ceiling_nodrift["cv_method"] == "cross_session"
    assert off["cv_method"] == "cross_session"

    # 드리프트가 실제로 손상을 냈는지 먼저 확인한다 — 손상이 없으면 회복도
    # 공허하다. off 자신의 fold 수준 표준오차(fold_ci 폭에서 역산)보다 큰
    # 차이를 요구한다. t 임계값은 df=2를 하드코딩하지 않고 off 자신의
    # `n_folds`에서 유도한다(session_recovery.yaml의 n_sessions을 바꿔도
    # 조용히 틀리지 않도록) — src/evaluation/metrics.py가 fold_ci를 낼 때
    # 쓰는 것과 같은 식(`stats.t.ppf(0.975, df=n_folds-1)`)이다.
    t_crit = scipy_stats.t.ppf(0.975, df=off["n_folds"] - 1)
    sem_off = (off["fold_ci_high"] - off["fold_ci_low"]) / (2 * t_crit)
    damage = ceiling_nodrift["pooled_accuracy"] - off["pooled_accuracy"]
    assert damage > sem_off, (
        f"드리프트가 낸 손상({damage:.4f})이 off 자신의 표준오차({sem_off:.4f})"
        f"보다 작다 — ceiling_nodrift={ceiling_nodrift['pooled_accuracy']:.4f}, "
        f"off={off['pooled_accuracy']:.4f}. 드리프트가 사실상 신호를 흔들지 "
        "않았다면 T3의 '회복'은 아무것도 증명하지 못한다."
    )

    recovered = (on["pooled_accuracy"] - off["pooled_accuracy"]) / (
        ceiling_nodrift["pooled_accuracy"] - off["pooled_accuracy"]
    )
    assert recovered >= T3_GAP_RECOVERY_MIN, (
        f"회복률 {recovered:.3f} < {T3_GAP_RECOVERY_MIN}. "
        f"off={off['pooled_accuracy']:.4f} on={on['pooled_accuracy']:.4f} "
        f"ceiling_nodrift={ceiling_nodrift['pooled_accuracy']:.4f} "
        f"chance={on['chance_level']:.4f} — 참고(상한 아님, cv_method 다름): "
        f"within_subject={within['pooled_accuracy']:.4f}"
    )


@pytest.mark.slow
def test_t3_injection_foreign_baseline_fails_to_recover(tmp_path, monkeypatch):
    """결함 주입: 다른 세션의 베이스라인으로 정규화하면 회복하지 못한다.

    상한은 위 정정과 같은 이유로 `ceiling_nodrift`(드리프트 비활성
    cross_session)를 쓴다. `off`·`ceiling_nodrift`는 `normalize: false`라
    이 몽키패치(`SessionBaseline.fit`)의 영향을 받지 않는다 —
    `build_dataset`이 `normalize`가 꺼져 있으면 `SessionBaseline`을 아예
    호출하지 않기 때문이다.

    **`scale`을 반드시 함께 넘긴다.** 넘기지 않으면 `SessionBaseline.__init__`
    의 기본값(`scale=None`)이 적용돼 `concentration_delta`가 옛 동작(뺄셈만,
    §6.2 이득 보정 없음)으로 조용히 되돌아간다 — "타 세션 베이스라인" 하나만
    주입하려던 의도와 달리 "이득 보정 비활성화"까지 함께 주입해버려, 이 결함
    주입이 실제로는 두 가지 결함을 섞어 재는 것이 된다. `real.scale`(같은
    세션 자신의 산포)을 그대로 넘겨 참조(`reference`)만 타 세션 것으로
    바뀌도록 한다.
    """
    from src.preprocessing.baseline import SessionBaseline

    cache: dict = {}

    class _ForeignBaseline(SessionBaseline):
        @classmethod
        def fit(cls, start_baseline, kind):
            real = SessionBaseline.fit(start_baseline, kind)
            reference = cache.setdefault(kind, real.reference)
            return SessionBaseline(reference=reference, kind=kind, scale=real.scale)

    monkeypatch.setattr(runner_mod, "SessionBaseline", _ForeignBaseline)
    off = _run({"preprocessing": {"baseline": {"normalize": False}}}, tmp_path / "off")
    on = _run({"preprocessing": {"baseline": {"normalize": True}}}, tmp_path / "foreign")
    ceiling_nodrift = _run(
        _deep_update(
            {"preprocessing": {"baseline": {"normalize": False}}},
            {"simulation": {"drift": _NO_DRIFT}},
        ),
        tmp_path / "ceiling_nodrift",
    )
    recovered = (on["pooled_accuracy"] - off["pooled_accuracy"]) / (
        ceiling_nodrift["pooled_accuracy"] - off["pooled_accuracy"]
    )
    assert recovered < T3_GAP_RECOVERY_MIN, (
        f"남의 세션 베이스라인으로 정규화했는데도 회복률 {recovered:.3f}를 "
        f"달성했다 (off={off['pooled_accuracy']:.4f}, "
        f"foreign-on={on['pooled_accuracy']:.4f}, "
        f"ceiling_nodrift={ceiling_nodrift['pooled_accuracy']:.4f}) — "
        "정규화가 세션 고유 정보를 쓰고 있지 않다는 뜻이다."
    )


# ---------------------------------------------------------------- T4

from collections import defaultdict

from src.datasets.windowing import make_windows
from src.preprocessing.baseline import SessionBaseline
from src.simulation.state import BASELINE, TASK

CLEAN = "config/experiments/session_clean.yaml"

#: 회귀 대상 부하 수준. nback_levels [0,2,3] 의 인덱스 2 = 3-back.
#: 가장 부하가 높은 조건이라 연습 효과가 가장 크게 나타난다.
T4_LEVEL = 2

#: 기울기 보존율의 허용 범위. 관측에서 역산한 값이 아니라 의미에서 나온 선언이다.
#: 하한: 연습 효과의 절반 미만만 남았다면 정규화가 진짜 인지 변화를 지운 것이다.
#: 상한: 1.5를 넘으면 정규화가 없던 변화를 만들어낸 것이다 (스펙 §8.2).
T4_RATIO_MIN = 0.5
T4_RATIO_MAX = 1.5

#: b_ref 자체가 살아 있는지 확인하는 절대 하한. 단위는 "베이스라인 산포 대비
#: 배수"(§6.2 정정 후 concentration_delta의 단위)다. 리뷰(2026-08-19)가 실증한
#: 구멍: b_ref·b_hat 둘 다 같은 정규화를 거치므로, 정규화가 **양쪽 모두에서**
#: 연습 효과를 죽이면 비율은 잡음/잡음이 되어 우연히 [0.5, 1.5] 안에 떨어질 수
#: 있다 — `abs(b_ref) > 1e-9`는 죽은 값(관측 0.0005~0.017)보다 6~7자릿수 낮아
#: 전혀 못 막는다. `T4_REF_SLOPE_MIN`은 b_ref 자신이 "기준 노릇을 할 만큼
#: 살아있는지"를 먼저 검사한다.
#: 관측(살아있는 b_ref 1.995~3.674 / 양쪽 다 죽인 b_ref 0.0005~0.017, 여러
#: 시드)은 이 선언이 타당한지 **확인**하는 용도이지 값의 출처가 아니다 — 하한은
#: 최소 관측 생존값(1.995)의 1/4이고, 관측된 죽은 값의 최댓값(0.017)의 30배
#: 이상이다.
T4_REF_SLOPE_MIN = 0.5


def _practice_slope(config_path, *, normalize, normalizer=SessionBaseline):
    """세션 번호에 대한 fNIRS HbO 평균 특징의 기울기.

    스펙 §8.2가 정의한 m(s)를 계산하고 1차 다항식을 적합한다.

    **normalize 인자는 "정규화 파이프라인을 태우는지" 여부이며, "드리프트가
    있는지"와 독립이다.** 호출자가 config로 드리프트 유무를 고르고, 이 인자로
    정규화 유무를 고른다 — 둘을 합쳐 어떤 조합을 쓸지는 호출부(T4 테스트)가
    결정한다. §6.2 정정 이후 이 선택이 값의 **단위**를 바꾸므로(아래 테스트
    docstring 참조) 더는 사소한 스위치가 아니다.
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


class _ZScoreBaseline(SessionBaseline):
    """결함 주입: 분모를 베이스라인 블록이 아니라 세션 전체(과제 포함)의 산포로
    바꾼다. `fit`은 베이스라인 창만 받아 평균만 저장하고(산포 저장 안 함),
    `apply`는 자신에게 넘어온 배열(`_practice_slope`가 세션 전체 `feats`를
    넘긴다) 자체의 표준편차로 나눈다 — 분모가 처음부터 "세션 전체 산포"다.

    두 T4 결함 주입 테스트가 공유하므로 모듈 스코프로 뺐다.
    """

    @classmethod
    def fit(cls, start_baseline, kind):
        return cls(reference=np.asarray(start_baseline).mean(axis=0), kind=kind)

    def apply(self, x):
        arr = np.asarray(x, dtype=float)
        sd = arr.std(axis=0)
        sd[sd == 0] = 1.0
        return (arr - arr.mean(axis=0)) / sd


def _assert_ratio_valid(b_ref, b_hat, *, context=""):
    """T4 판정 로직.

    **`abs(b_ref) > 1e-9`가 아니라 `T4_REF_SLOPE_MIN`으로 b_ref 생존을 검사한다**
    (2026-08-19 리뷰 Critical 수정). `1e-9`는 "0으로 나누기만 막는" 수치적
    하한이라 죽은 b_ref(관측 0.0005~0.017)를 전혀 못 거른다 — 그 상태에서
    비율만 보면 잡음/잡음이 우연히 [0.5, 1.5] 안에 떨어져 정규화가 연습 효과를
    통째로 지운 경우조차 통과할 수 있다(`test_t4_injection_symmetric_kill_
    is_caught_by_ref_floor` 참조). 반드시 b_ref 생존 검사가 비율 검사보다
    먼저다 — 비율은 분모가 의미 있을 때만 의미가 있다.
    """
    assert abs(b_ref) >= T4_REF_SLOPE_MIN, (
        f"{context}기준 실행 자체에서 연습 효과가 사라졌다 — 비율을 계산할 근거가 "
        f"없다. |b_ref|={abs(b_ref):.6f} < 하한 {T4_REF_SLOPE_MIN}"
    )
    ratio = b_hat / b_ref
    assert T4_RATIO_MIN <= ratio <= T4_RATIO_MAX, (
        f"{context}기울기 보존율 {ratio:.3f} 가 [{T4_RATIO_MIN}, {T4_RATIO_MAX}] "
        f"밖이다. b_ref={b_ref:.4f} b_hat={b_hat:.4f}."
    )
    return ratio


@pytest.mark.slow
def test_t4_practice_effect_survives_normalization(tmp_path):
    """정규화가 측정 드리프트를 지우면서 진짜 학습은 남겨야 한다.

    **브리프 원안과 다르게 `b_ref`도 정규화 on으로 계산한다.** Task 14 중
    §6.2가 바뀌어 `concentration_delta`가 베이스라인 산포로도 나누게
    됐다 — 결과 단위가 "원 농도 단위"에서 "베이스라인 산포 대비 배수"
    (무차원)로 바뀌었다. `normalize=False`(원 단위)와 `normalize=True`
    (배수 단위)는 더 이상 같은 자로 잰 값이 아니다: 실측으로 확인한 결과
    `_practice_slope(CLEAN, normalize=False)` = 약 -0.0358 (원 단위)인 반면
    `_practice_slope(RECOVERY, normalize=True)` = 약 -2.5812 (배수 단위)라
    단순 비율이 약 72로 튄다 — 세션의 베이스라인 산포(σ≈0.014)가 작아서
    나눗셈이 값을 그만큼 확대했을 뿐, 연습 효과가 커진 게 아니다.

    그래서 `b_ref`도 같은 정규화 파이프라인을 통과시키고 **드리프트만
    끈다** — T3가 `within_subject`(다른 분할기)에서 `ceiling_nodrift`
    (같은 분할기, 드리프트만 0)로 상한을 바꾼 것과 같은 원칙이다(스펙
    §8.2 "2026-08-19 참고" 문단이 이미 이 원칙—"같은 파이프라인, 드리프트만
    다르다"—을 예고했다). 이러면 `b_ref`·`b_hat` 둘 다 "베이스라인 산포
    대비 배수" 단위로 맞춰져 비율이 다시 의미를 가진다.

    **분해로 확인한 원인(리뷰 2026-08-19):** 1.294의 29%p 초과분은 σ̂ 추정
    잡음이 아니라 **세션 내 드리프트 잔여물**이다. `within_session_rate=0`으로
    override하면 보존율이 정확히 1.000000이 된다 — 시작 베이스라인 정규화는
    곱셈 이득·가산 오프셋(①②)은 완전히 소거하지만, 세션 *내* 드리프트(③)는
    구조적으로 못 지운다(CLAUDE.md §3.8: ③은 정규화가 아니라 플래그 대상).
    seed 42/0/1/7의 보존율은 1.294/0.843/0.932/1.274로 1 부근에 분산돼 있고
    전부 [0.5, 1.5] 안이다 — 지배 요인은 24개 피험자-세션에 대한 거친
    베르누이 배정(`within_big = u < within_session_fraction`)이지 매끄러운
    잡음이 아니다.
    """
    b_ref = _practice_slope(CLEAN, normalize=True)
    b_hat = _practice_slope(RECOVERY, normalize=True)
    _assert_ratio_valid(b_ref, b_hat)


@pytest.mark.slow
def test_t4_injection_over_normalization_kills_the_practice_effect():
    """결함 주입: 분모를 베이스라인 블록이 아니라 세션 전체(과제 포함)의 산포로 바꾼다.

    **브리프 원안(세션별 전체 z-score)을 재설계 없이 그대로 썼다** — 검토해보니
    이미 요점을 구현하고 있었다(`_ZScoreBaseline` docstring 참조). 바뀐 것은
    비교 기준(`b_ref`)뿐이다: 위 테스트와 같은 이유로 `normalize=True`(드리프트
    off, 정규화 on)를 쓴다 — `normalize=False`를 분모로 쓰면 다시 단위가
    어긋나 결함 주입 없이도 보존율이 낮아 보이는 거짓 양성이 나온다.

    평균만이 아니라 분산까지 세션마다 맞추면, 과제 반응 크기(=연습 효과가
    만드는 신호 변화) 자체가 분모에 흡수돼 세션 간 크기 차이가 통째로
    사라진다. 측정 드리프트와 함께 진짜 학습도 지워진다.

    **비대칭 주입이다** — `b_hat`(RECOVERY)에만 결함 정규화기를 쓰고 `b_ref`
    (CLEAN)는 정상 `SessionBaseline`을 쓴다. 이 비대칭이 왜 현실적 결함까지
    다 잡지 못하는지는 `test_t4_injection_symmetric_kill_is_caught_by_ref_floor`
    참조 — 정규화 버그는 보통 코드 경로 하나이므로 양쪽에 다 걸린다.
    """
    b_ref = _practice_slope(CLEAN, normalize=True)
    b_bad = _practice_slope(RECOVERY, normalize=True, normalizer=_ZScoreBaseline)
    assert abs(b_bad / b_ref) < T4_RATIO_MIN, (
        f"과한 정규화를 주입했는데도 보존율({b_bad / b_ref:.4f})이 유지됐다 — "
        f"b_ref={b_ref:.4f} b_bad={b_bad:.4f}. T4가 이 실패 양식을 못 잡는다"
    )


@pytest.mark.slow
def test_t4_injection_symmetric_kill_is_caught_by_ref_floor():
    """결함 주입 2 (리뷰 2026-08-19 지적): 정규화 버그가 `b_ref`·`b_hat` 양쪽에
    다 걸리면 비율 하나만으로는 못 잡는다.

    위 비대칭 주입 테스트는 `b_hat`에만 결함 정규화기를 쓴다. 그런데 실제
    정규화 버그는 코드 경로 하나이므로 `CLAUDE.md` §3.8이 규정한 "세션 단위
    독립 정규화"가 양쪽 실행 모두에 적용된다 — `b_ref`도 같은 버그를 겪는다.

    `b_ref`·`b_hat`을 **둘 다** `_ZScoreBaseline`으로 계산하면, 리뷰가 여러
    시드로 재현한 대로 `b_ref` 자신이 살아있는 값(1.995~3.674)에서 죽은
    값(0.0005~0.017, 200~7000배 작음)으로 무너진다. 이 상태에서 비율만
    보면 seed 0(1.317)·seed 1(0.842) 모두 [0.5, 1.5] 안에 우연히 떨어져
    **통과한다** — 잡음/잡음이 1 근처에 분포하기 때문이다. seed 42(-0.177)만
    부호가 갈려 우연히 걸린다.

    `T4_REF_SLOPE_MIN` 절대 하한이 이 경로를 실제로 막는지 확인한다:
    `_assert_ratio_valid`가 비율을 보기 전에 `b_ref` 자신의 생존을 먼저
    검사해야 한다.
    """
    b_ref_dead = _practice_slope(CLEAN, normalize=True, normalizer=_ZScoreBaseline)
    b_hat_dead = _practice_slope(RECOVERY, normalize=True, normalizer=_ZScoreBaseline)

    assert abs(b_ref_dead) < T4_REF_SLOPE_MIN, (
        "가정이 깨졌다 — 양쪽을 다 죽였는데 b_ref가 여전히 하한(T4_REF_SLOPE_MIN) "
        f"위에 있다. b_ref_dead={b_ref_dead:.6f}. 이 결함 주입은 T4_REF_SLOPE_MIN이 "
        "막아야 할 실패 양식을 만들지 못한다."
    )

    with pytest.raises(AssertionError, match="비율을 계산할 근거가 없다"):
        _assert_ratio_valid(b_ref_dead, b_hat_dead, context="[대칭 주입] ")


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


@pytest.mark.slow
def test_t5_normalization_source_window_in_a_fold_is_rejected():
    dataset = _dataset_with_baseline_opted_in()
    with pytest.raises(LeakageError, match="normalization source"):
        run_folds(
            dataset, CrossSessionSplitter(),
            target="accuracy", modalities=["eeg", "fnirs", "behavior"],
            guards=ALL_GUARDS, seed=42,
        )


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "2026-08-19 갱신(Task 16, compute_drift 분모를 |mean(start)|에서 "
        "std(start, ddof=1)로 고친 뒤 재측정). 원래 결함(근사-영 평균 분모로 "
        "인한 발산, fnirs 값이 최대 174까지 치솟던 것)은 해소됐다 — 이제 값은 "
        "0.4~5 범위로 물리적으로 말이 된다. eeg는 여전히 깨끗이 분리된다(작은 "
        "군 max=0.788948 < 큰 군 min=3.515019). 임계를 그 중간값 2.15로 "
        "재보정했다(session_quality.yaml 주석 참조 — 단위가 '평균 대비 비율'"
        "에서 '표준편차 배수'로 바뀌어 예전 0.20은 더 이상 맞는 자리가 아니다). "
        "하지만 fnirs는 여전히 완전히는 분리되지 않는다 — 이번엔 자릿수 차이가 "
        "아니라 근소한 차이다: 세션0 sub-04의 fnirs=4.246897(핵심 칸이 아닌 "
        "'작은' 군에 있어야 하는데도 세션1 sub-02의 eeg=4.222576보다 크다) 하나 "
        "때문에, '세션1 전원 플래그 AND 세션0·2 전원 비플래그'를 만족하는 "
        "임계 구간이 [4.222576, 4.246897)로 존재하지 않는다(하한이 상한보다 "
        "크다). 채널 분해로 확인: 이 값은 슬로프 특징 하나가 아니라 HbO 평균· "
        "슬로프 둘 다에서 이 피험자·세션 조합이 유난히 크게 나온 것(mean_agg="
        "5.51, slope_agg=2.99) — 특정 특징의 근사-영 분모 때문이 아니라 "
        "n_subjects=4의 작은 표본에서 나온 피험자 간 생리적/표집 변동으로 "
        "보인다(subject_variance=0.5). 원래 결함(구조적 발산, 세션·피험자 "
        "무관하게 항상 임계를 넘음)과는 종류가 다르다 — 여전히 xfail로 남기되 "
        "'compute_drift 수정 필요'가 아니라 '더 큰 표본 또는 더 견고한 fnirs "
        "집계가 필요할 수 있음'으로 원인을 바꿔 기록한다. task-16-report.md "
        "참조."
    ),
    strict=True,
)
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


@pytest.mark.slow
def test_t6_injection_zero_threshold_destroys_specificity():
    """결함 주입: 임계를 0으로 낮추면 전부 플래그되어 특이도가 무너져야 한다."""
    cfg = load_config(QUALITY)
    cfg = _deep_update(cfg, {"preprocessing": {"baseline": {"drift_threshold_relative": 0.0}}})
    validate_config(cfg)
    rng = set_all_seeds(int(cfg["seed"]))
    _, _, qualities = runner_mod.build_dataset(cfg, rng)
    assert all(q.drift_flag for q in qualities)
