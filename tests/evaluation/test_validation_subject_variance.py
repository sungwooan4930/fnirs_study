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
