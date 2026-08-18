import json

import numpy as np
import pytest
import yaml

from src.common.seeding import set_all_seeds
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.evaluation.runner import build_dataset, run_experiment
from src.simulation.recording import generate_dataset
import src.evaluation.runner as runner_module

BASE_CFG = {
    "run_name": "unit",
    "seed": 7,
    "simulation": {
        "n_subjects": 4,
        "subject_variance": 0.5,
        "effect_size": 0.8,
        "lead_delta_s": 1.2,
        "task": {
            "nback_levels": [0, 2, 3],
            "block_duration_s": 20,
            "n_blocks_per_level": 1,
            "stim_interval_s": 2.0,
        },
        "eeg": {"n_channels": 8, "sfreq_hz": 100},
        "fnirs": {"n_channels": 8, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
    },
    "windowing": {"window_s": 5.0, "step_s": 1.0},
    "features": {"extractor": "minimal"},
    "dataset": {
        "targets": ["cognitive_load"],
        "lead_targets": ["accuracy", "response_latency"],
        "modalities": ["eeg", "fnirs", "behavior"],
        "rt_bins": [0.5, 0.8],
    },
    "evaluation": {
        "splitter": "loso",
        "model": "logistic_regression",
        "guards": {"check_subject_overlap": True, "check_window_overlap": True},
    },
    "output": {"results_dir": "results"},
}


def _cfg_file(tmp_path, results_dir, **sim_overrides):
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["simulation"].update(sim_overrides)
    cfg["output"]["results_dir"] = str(results_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_build_dataset_shapes_are_consistent():
    ds, dropped = build_dataset(BASE_CFG, set_all_seeds(0))
    assert ds.n_windows > 0
    assert dropped >= 0
    assert sorted(ds.modalities) == ["behavior", "eeg", "fnirs"]
    assert len(set(ds.get_subject_ids().tolist())) == 4


def test_build_dataset_has_all_three_targets():
    ds, _ = build_dataset(BASE_CFG, set_all_seeds(0))
    assert ds.targets == ["accuracy", "cognitive_load", "response_latency"]


def test_run_experiment_writes_all_artifacts(tmp_path):
    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    for name in ("config.yaml", "env.json", "metrics.json", "per_fold.csv", "log.txt"):
        assert (out / name).exists(), f"missing {name}"


def test_metrics_record_chance_and_cv_method(tmp_path):
    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cv_method"] == "loso"
    assert metrics["chance_level"] == pytest.approx(1 / 3)
    assert "n_windows_dropped" in metrics
    assert metrics["guards_disabled"] is False


def test_env_records_git_and_seed(tmp_path):
    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["seed"] == 7
    assert "git_commit" in env
    assert "git_dirty" in env
    assert env["python"].startswith("3.")


def test_same_seed_gives_identical_metrics(tmp_path):
    """metrics.json 전체가 동일해야 한다.

    pooled_accuracy 하나만 비교하면 accuracy_worst·confusion·
    worst_fold_subjects·n_windows_dropped 같은 다른 키가 갈라져도
    통과한다. T4(개인차 스윕)는 config 차이에서 나온 metrics.json 간
    차이를 연구 결론으로 읽으므로, 같은 시드에서 나오는 비결정성은
    여기서 잡혀야지 결과로 둔갑해서는 안 된다.
    """
    a = run_experiment(_cfg_file(tmp_path / "a", tmp_path / "ra"))
    b = run_experiment(_cfg_file(tmp_path / "b", tmp_path / "rb"))
    ma = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "metrics.json").read_text(encoding="utf-8"))
    assert ma == mb


def test_overrides_are_applied(tmp_path):
    out = run_experiment(
        _cfg_file(tmp_path, tmp_path / "results"),
        overrides={"simulation": {"subject_variance": 3.0}},
    )
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["simulation"]["subject_variance"] == 3.0


def test_disabled_guards_mark_run_unsafe(tmp_path):
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["evaluation"]["guards"]["check_subject_overlap"] = False
    cfg["output"]["results_dir"] = str(tmp_path / "results")
    p = tmp_path / "unsafe.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run_experiment(p)
    assert out.name.endswith("-UNSAFE")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["guards_disabled"] is True


def test_git_unavailable_records_unknown_not_clean(tmp_path, monkeypatch):
    """git 조회가 실패하면 '알 수 없음'을 '깨끗함'으로 위장하지 않는다.

    _git이 예외를 삼키고 빈 문자열을 돌려주던 예전 동작이면 commit이
    "nogit", dirty가 False가 되어 "clean하고 커밋된 코드에서 실행됨"으로
    읽힌다 — 실제로는 아무것도 확인되지 않았는데도. None을 돌려주게
    고치면 git_available=False, git_dirty=None(3값 논리의 '모름')이
    기록되어야 한다.
    """
    monkeypatch.setattr(runner_module, "_git", lambda *args: None)

    out = run_experiment(_cfg_file(tmp_path, tmp_path / "results"))
    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    assert env["git_available"] is False
    assert env["git_dirty"] is None
    assert "nogit" in out.name
    # guards는 여전히 켜져 있으므로 UNSAFE는 아니다 — git 미상과는 별개 축
    assert not out.name.endswith("-UNSAFE")


def test_build_dataset_preserves_row_alignment():
    """subject_id·trial_id·특징 행이 keep 마스크를 거친 뒤에도 정렬되어 있는지 확인한다.

    features/labels에는 정확한 keep을 적용하고 subject_ids에는 어긋난
    keep을 적용하는 버그가 있어도 전체 길이는 똑같이 유지될 수 있다
    (WindowedDataset의 길이 검사는 이런 종류의 어긋남을 잡지 못한다).
    같은 시드로 녹화를 독립적으로 다시 만들어 각 피험자의 trial_id
    시퀀스를 직접 재계산하고, build_dataset이 내놓은 결과에서 같은
    피험자로 필터링한 행과 원소 단위로 비교한다. 정렬이 한 칸이라도
    어긋나면 길이나 값이 달라져 실패한다.
    """
    seed = 123
    ds, _ = build_dataset(BASE_CFG, set_all_seeds(seed))

    # build_dataset과 동일한 시드로 녹화열을 독립적으로 재현
    recordings = generate_dataset(BASE_CFG["simulation"], set_all_seeds(seed))

    win_cfg = BASE_CFG["windowing"]
    lead_delta_s = float(BASE_CFG["simulation"]["lead_delta_s"])
    rt_bins = list(BASE_CFG["dataset"]["rt_bins"])

    ds_subjects = ds.get_subject_ids()
    ds_trials = ds.get_trial_ids()

    assert len(recordings) > 0
    for rec in recordings:
        windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
        _, keep = build_labels(
            rec, windows, lead_delta_s=lead_delta_s, rt_bins=rt_bins
        )
        expected_trials = windows.trial_id[keep]

        actual_trials = ds_trials[ds_subjects == rec.subject_id]
        assert np.array_equal(actual_trials, expected_trials), (
            f"row misalignment detected for subject {rec.subject_id}"
        )
