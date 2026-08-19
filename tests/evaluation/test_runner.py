import json

import numpy as np
import pytest
import yaml

from src.common.seeding import set_all_seeds
from src.datasets.labels import build_labels
from src.datasets.windowing import make_windows
from src.common.config import ConfigError, load_config
from src.evaluation.runner import build_dataset, run_experiment
from src.simulation.recording import generate_dataset
import src.evaluation.runner as runner_module

BASE_CFG = {
    "run_name": "unit",
    "seed": 7,
    "simulation": {
        "n_subjects": 4,
        "n_sessions": 2,
        "subject_variance": 0.5,
        "effect_size": 0.8,
        "lead_delta_s": 1.2,
        "task": {
            "nback_levels": [0, 2, 3],
            "block_duration_s": 20,
            "baseline_duration_s": 20,
            "n_blocks_per_level": 1,
            "stim_interval_s": 2.0,
        },
        "practice": {"rate": 0.15},
        "drift": {
            "fnirs_gain_sigma": 0.20,
            "fnirs_offset_sigma": 0.10,
            "eeg_gain_sigma": 0.15,
            "eeg_noise_sigma": 0.15,
            "within_session_rate": 0.30,
            "within_session_fraction": 0.33,
            "between_session_scale": 4.0,
            "assignment": "sampled",
        },
        "eeg": {"n_channels": 8, "sfreq_hz": 100},
        "fnirs": {"n_channels": 8, "sfreq_hz": 10.4, "hbr_coupling": -0.33},
    },
    "windowing": {"window_s": 5.0, "step_s": 1.0},
    "features": {"extractor": "minimal"},
    "preprocessing": {
        "baseline": {
            "normalize": True,
            "drift_threshold_relative": 0.20,
            "zero_atol": 1.0e-8,
        },
    },
    "dataset": {
        "targets": ["cognitive_load"],
        "lead_targets": ["accuracy", "response_latency"],
        "modalities": ["eeg", "fnirs", "behavior"],
        "rt_bins": [0.5, 0.8],
        "include_baseline": False,
    },
    "evaluation": {
        "splitter": "loso",
        "model": "logistic_regression",
        "guards": {
            "check_subject_overlap": True,
            "check_window_overlap": True,
            "check_session_overlap": True,
            "check_normalization_source": True,
        },
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
    ds, dropped, _ = build_dataset(BASE_CFG, set_all_seeds(0))
    assert ds.n_windows > 0
    assert dropped >= 0
    assert sorted(ds.modalities) == ["behavior", "eeg", "fnirs"]
    assert len(set(ds.get_subject_ids().tolist())) == 4


def test_build_dataset_has_all_three_targets():
    ds, _, _ = build_dataset(BASE_CFG, set_all_seeds(0))
    assert ds.targets == ["accuracy", "cognitive_load", "response_latency"]


def test_lead_targets_selects_which_lead_labels_are_built():
    """dataset.lead_targets가 실제로 소비되어야 한다.

    스키마에 있고 모든 config가 적어놓았는데 아무도 읽지 않으면,
    오타나 잘못된 값이 조용히 무시된다 — 엄격 스키마를 둔 이유가 사라진다.
    """
    import copy

    cfg = copy.deepcopy(BASE_CFG)
    cfg["dataset"]["lead_targets"] = ["accuracy"]
    ds, _, _ = build_dataset(cfg, set_all_seeds(0))
    assert ds.targets == ["accuracy", "cognitive_load"]


def test_unknown_lead_target_is_rejected():
    import copy

    cfg = copy.deepcopy(BASE_CFG)
    cfg["dataset"]["lead_targets"] = ["accuracy", "bogus_target"]
    with pytest.raises(ValueError, match="unknown lead target"):
        build_dataset(cfg, set_all_seeds(0))


def test_unknown_extractor_is_rejected():
    """features.extractor가 실제로 추출기를 고른다.

    읽히지 않는 키였을 때는 `extractor: real`로 바꿔도 최소 추출기가
    돌면서 초록불이 떴다.
    """
    import copy

    cfg = copy.deepcopy(BASE_CFG)
    cfg["features"]["extractor"] = "real"
    with pytest.raises(ValueError, match="unknown feature extractor"):
        build_dataset(cfg, set_all_seeds(0))


def test_multi_target_config_is_rejected(tmp_path):
    """targets[1:]을 조용히 버리지 않는다."""
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["dataset"]["targets"] = ["cognitive_load", "accuracy"]
    cfg["output"]["results_dir"] = str(tmp_path / "results")
    p = tmp_path / "multi.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="multi-target evaluation is not supported"):
        run_experiment(p)


def test_chance_level_follows_the_evaluated_target(tmp_path):
    """chance_level은 과제 config가 아니라 평가하는 라벨에서 나와야 한다.

    n_classes = len(nback_levels)로 두면 이진 타깃(accuracy)을 평가해도
    chance_level이 0.3333으로 기록되고 binomtest가 틀린 귀무가설을 쓴다.
    오류도 나지 않으므로 config 한 줄이 CLAUDE.md 5.4 위반을 제조한다.
    """
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["dataset"]["targets"] = ["accuracy"]
    cfg["output"]["results_dir"] = str(tmp_path / "results")
    p = tmp_path / "binary.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run_experiment(p)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["target"] == "accuracy"
    assert metrics["chance_level"] == pytest.approx(0.5)
    assert np.array(metrics["confusion"]).shape == (2, 2)


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


def test_overrides_change_the_run_directory(tmp_path):
    """오버라이드가 다르면 결과 디렉토리도 달라야 한다.

    run_id가 run_name·seed·commit만 담으면 T4 스윕 6개 조건이 전부
    같은 디렉토리를 가리키고 mkdir(exist_ok=True)가 조용히 통과해
    metrics.json이 차례로 덮어써진다. 5개가 파괴되고 마지막 하나만
    남는데, 그 하나는 내부적으로 일관되어 아무 이상도 드러나지 않는다
    (스펙 7.2 · CLAUDE.md 5.2).
    """
    cfg = _cfg_file(tmp_path, tmp_path / "results")
    base = run_experiment(cfg)
    varied = run_experiment(cfg, overrides={"simulation": {"subject_variance": 3.0}})
    assert base.name != varied.name
    assert base.resolve() != varied.resolve()


def test_same_config_reuses_the_same_run_directory(tmp_path):
    """해시는 실효 config에서만 나온다 — 같은 실험은 같은 디렉토리다."""
    cfg = _cfg_file(tmp_path, tmp_path / "results")
    assert run_experiment(cfg).name == run_experiment(cfg).name


def test_results_dir_does_not_change_the_run_id(tmp_path):
    """어디에 저장하느냐는 실험의 정체성이 아니다."""
    a = run_experiment(_cfg_file(tmp_path / "a", tmp_path / "ra"))
    b = run_experiment(_cfg_file(tmp_path / "b", tmp_path / "rb"))
    assert a.name == b.name


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
    """subject_id·session_id·trial_id·특징 행이 keep 마스크를 거친 뒤에도 정렬되어 있는지 확인한다.

    features/labels에는 정확한 keep을 적용하고 subject_ids에는 어긋난
    keep을 적용하는 버그가 있어도 전체 길이는 똑같이 유지될 수 있다
    (WindowedDataset의 길이 검사는 이런 종류의 어긋남을 잡지 못한다).
    같은 시드로 녹화를 독립적으로 다시 만들어 각 (피험자, 세션)의 trial_id
    시퀀스를 직접 재계산하고, build_dataset이 내놓은 결과에서 같은
    (피험자, 세션)으로 필터링한 행과 원소 단위로 비교한다. 정렬이 한 칸이라도
    어긋나면 길이나 값이 달라져 실패한다.

    trial_id는 녹화(= 한 피험자의 한 세션)마다 0부터 세지만 계약은 전역
    고유성을 요구하므로, build_dataset은 녹화 순서대로 누적 오프셋을
    더한다 (Task 13). 이 테스트는 generate_dataset이 내놓는 순서가
    build_dataset과 동일하다는 전제로 같은 오프셋을 독립적으로 재계산한다.
    """
    seed = 123
    ds, _, _ = build_dataset(BASE_CFG, set_all_seeds(seed))

    # build_dataset과 동일한 시드로 녹화열을 독립적으로 재현
    recordings = generate_dataset(BASE_CFG["simulation"], set_all_seeds(seed))

    win_cfg = BASE_CFG["windowing"]
    lead_delta_s = float(BASE_CFG["simulation"]["lead_delta_s"])
    rt_bins = list(BASE_CFG["dataset"]["rt_bins"])

    ds_subjects = ds.get_subject_ids()
    ds_sessions = ds.get_session_ids()
    ds_trials = ds.get_trial_ids()

    assert len(recordings) > 0
    trial_offset = 0
    for rec in recordings:
        windows = make_windows(rec.timeline, win_cfg["window_s"], win_cfg["step_s"])
        _, keep = build_labels(
            rec, windows, lead_delta_s=lead_delta_s, rt_bins=rt_bins
        )
        expected_trials = windows.trial_id[keep] + trial_offset

        mask = (ds_subjects == rec.subject_id) & (ds_sessions == rec.session_idx)
        actual_trials = ds_trials[mask]
        assert np.array_equal(actual_trials, expected_trials), (
            f"row misalignment detected for subject {rec.subject_id} "
            f"ses-{rec.session_idx}"
        )
        trial_offset += int(windows.trial_id.max()) + 1


def test_typo_in_overrides_is_rejected_not_silently_absorbed():
    """P1: 오버라이드가 스키마 검증을 우회하면 오타 키가 조용히 통과한다.

    조용히 통과하면 config 해시만 바뀌어 새 결과 디렉토리가 생기고,
    아무 손잡이도 돌리지 않은 실행이 별개 조건인 것처럼 기록된다.
    """
    with pytest.raises(ConfigError, match=r"unknown key 'simulation\.n_sesions'"):
        run_experiment(
            "config/experiments/smoke.yaml",
            overrides={"simulation": {"n_sesions": 3}},
        )


def test_build_dataset_returns_session_quality_per_recording():
    cfg = load_config("config/experiments/smoke.yaml")
    rng = set_all_seeds(cfg["seed"])
    ds, _, qualities = build_dataset(cfg, rng)
    expected = cfg["simulation"]["n_subjects"] * cfg["simulation"]["n_sessions"]
    assert len(qualities) == expected
    assert {q.session_idx for q in qualities} == set(range(cfg["simulation"]["n_sessions"]))


def test_dataset_carries_session_ids_and_globally_unique_trials():
    cfg = load_config("config/experiments/smoke.yaml")
    rng = set_all_seeds(cfg["seed"])
    ds, _, _ = build_dataset(cfg, rng)
    assert set(ds.get_session_ids().tolist()) == set(range(cfg["simulation"]["n_sessions"]))
    # 계약이 이미 검증하지만, 조립하는 쪽이 오프셋을 실제로 더했는지 확인한다
    pairs = {
        (s, int(k)) for s, k in zip(ds.get_subject_ids(), ds.get_session_ids())
    }
    assert len(pairs) == cfg["simulation"]["n_subjects"] * cfg["simulation"]["n_sessions"]


def test_include_baseline_with_load_target_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="include_baseline"):
        run_experiment(
            "config/experiments/smoke.yaml",
            overrides={"dataset": {"include_baseline": True}},
        )


def test_run_experiment_writes_session_quality_csv(tmp_path):
    out = run_experiment(
        "config/experiments/smoke.yaml",
        overrides={"output": {"results_dir": str(tmp_path)}},
    )
    text = (out / "session_quality.csv").read_text(encoding="utf-8")
    assert "subject_id,session_idx,drift_flag" in text.splitlines()[0]


def test_cross_session_run_records_its_scheme(tmp_path):
    out = run_experiment(
        "config/experiments/smoke.yaml",
        overrides={
            "evaluation": {"splitter": "cross_session"},
            "output": {"results_dir": str(tmp_path)},
        },
    )
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cv_method"] == "cross_session"
