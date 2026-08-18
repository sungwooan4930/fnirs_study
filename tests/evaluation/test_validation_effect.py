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
