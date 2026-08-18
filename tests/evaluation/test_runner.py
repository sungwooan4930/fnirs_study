import json

import pytest
import yaml

from src.common.seeding import set_all_seeds
from src.evaluation.runner import build_dataset, run_experiment

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
    a = run_experiment(_cfg_file(tmp_path / "a", tmp_path / "ra"))
    b = run_experiment(_cfg_file(tmp_path / "b", tmp_path / "rb"))
    ma = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "metrics.json").read_text(encoding="utf-8"))
    assert ma["pooled_accuracy"] == mb["pooled_accuracy"]


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
