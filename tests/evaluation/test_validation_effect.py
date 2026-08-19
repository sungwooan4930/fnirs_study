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
    # 널 판정은 fold_ci(LOSO이므로 fold=피험자 수준 t-구간)로 한다. pooled_ci는
    # 5초창·1초 스텝의 80% 오버랩 때문에 창을 독립 시행으로 취급해 불확실성을
    # 과소평가한다 — 세션이 늘어 창 수가 커질수록 더 위험해진다
    # (metrics.aggregate 문서 참조). 실패 메시지에 두 구간을 모두 적어
    # 다음 사람이 그 차이를 보게 한다.
    assert m["fold_ci_low"] <= CHANCE <= m["fold_ci_high"], (
        f"널 데이터인데 chance가 fold 수준 신뢰구간 밖이다 "
        f"(fold_ci=[{m['fold_ci_low']:.3f}, {m['fold_ci_high']:.3f}], "
        f"pooled_ci=[{m['pooled_ci_low']:.3f}, {m['pooled_ci_high']:.3f}]) — "
        "파이프라인에 누수가 있다"
    )
    # binomtest_p는 삭제했다: pooled_ci와 같은 독립성 결함을 공유한다
    # (18000개 오버랩 창을 독립 베르누이 시행으로 세어 계산된 p값). 세션
    # 수·창 수가 늘수록 거짓 양성(널인데도 유의하다고 나옴)을 낸다. 널
    # 판정은 위 fold_ci 하나로 충분하다.


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
    # fold_ci를 쓰는 이유는 위 test_t1_null_effect_stays_at_chance와 같다.
    assert m["fold_ci_low"] <= CHANCE <= m["fold_ci_high"], (
        f"널 데이터인데 chance가 fold 수준 신뢰구간 밖이다 "
        f"(fold_ci=[{m['fold_ci_low']:.3f}, {m['fold_ci_high']:.3f}], "
        f"pooled_ci=[{m['pooled_ci_low']:.3f}, {m['pooled_ci_high']:.3f}])"
    )


@pytest.mark.slow
def test_t2_effect_is_recovered_above_chance(tmp_path):
    m = _metrics("config/experiments/pilot.yaml", tmp_path)
    assert m["pooled_accuracy"] > CHANCE
    # binomtest_p < 0.01은 예전 단언이었으나 T1에서 삭제한 것과 같은
    # 독립성 결함을 공유한다 (18000개 오버랩 창을 독립 시행으로 세어 계산된
    # p값 — 우연한 잡음도 유의하다고 착시시키는 방향으로 작동). 효과 회수는
    # fold_ci(LOSO이므로 피험자 수준)로 판정한다 — chance가 구간 **밖**(위쪽)
    # 이어야 진짜 효과다. 파일럿 실측: fold_ci_low ≈ 0.513으로 chance(0.333)를
    # 여유 있게 넘는다.
    assert m["fold_ci_low"] > CHANCE, (
        f"fold_ci_low({m['fold_ci_low']:.3f})가 chance({CHANCE:.3f})를 넘지 "
        "못했다 — 심어둔 효과를 하네스가 회수하지 못했다"
    )


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


@pytest.mark.slow
def test_t2_matches_recorded_baseline(tmp_path):
    """파일럿 관측값이 베이스라인에서 벗어나면 무언가 바뀐 것이다.

    허용 오차가 0.005로 좁은 이유: 이 실행은 표본 노이즈를 흡수할 필요가
    없다. `set_all_seeds`가 만드는 결정론적 Generator 덕분에 같은 시드의
    `run_experiment`는 비트 단위로 재현된다(Task 18에서 확인) — 실행마다
    달라지는 값이 아니다. 그러므로 이 허용 오차는 "정상적인 변동 범위"가
    아니라 "무시해도 되는 부동소수점 마지막 자리 차이(플랫폼·numpy 빌드
    간)"만을 위한 것이다. 0.05처럼 넓혀두면 T1의 신뢰구간 반폭(약
    1.19%p, n=6000)보다 훨씬 둔감해져서, T1이라면 잡아낼 수준의 회귀(예:
    특징 추출·스케일링 버그로 인한 3~4%p 이동)를 조용히 통과시킨다.
    나중에 이 값을 다시 넓히고 싶다면, 그 근거가 "재현성이 깨졌다"여야지
    "표본 노이즈"여서는 안 된다 — 후자라면 애초에 Task 18의 결론이
    틀렸다는 뜻이므로 먼저 그것부터 재확인해야 한다.
    """
    import pathlib

    baseline = json.loads(
        pathlib.Path("tests/baselines/t2_pilot.json").read_text(encoding="utf-8")
    )
    m = _metrics("config/experiments/pilot.yaml", tmp_path)
    assert m["cv_method"] == baseline["cv_method"]
    assert abs(m["pooled_accuracy"] - baseline["pooled_accuracy"]) < 0.005, (
        f"파일럿 정확도가 베이스라인 {baseline['pooled_accuracy']:.3f}에서 "
        f"{m['pooled_accuracy']:.3f}로 이동했다"
    )
