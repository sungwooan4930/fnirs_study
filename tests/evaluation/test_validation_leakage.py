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
