import textwrap

import pytest

from src.common.config import ConfigError, load_config, validate_config

MINIMAL = """
run_name: t
seed: 1
simulation:
  n_subjects: 2
  n_sessions: 2
  subject_variance: 0.5
  effect_size: 0.8
  lead_delta_s: 1.2
  task:
    nback_levels: [0, 2, 3]
    block_duration_s: 20
    baseline_duration_s: 20
    n_blocks_per_level: 1
    stim_interval_s: 2.0
  practice: {rate: 0.15}
  drift:
    fnirs_gain_sigma: 0.20
    fnirs_offset_sigma: 0.10
    eeg_gain_sigma: 0.15
    eeg_noise_sigma: 0.15
    within_session_rate: 0.30
    within_session_fraction: 0.33
    between_session_scale: 4.0
    assignment: sampled
  eeg: {n_channels: 4, sfreq_hz: 100}
  fnirs: {n_channels: 4, sfreq_hz: 10.4, hbr_coupling: -0.33}
windowing: {window_s: 5.0, step_s: 1.0}
features: {extractor: minimal}
preprocessing:
  baseline: {normalize: true, drift_threshold_relative: 0.20, zero_atol: 1.0e-8}
dataset:
  targets: [cognitive_load]
  lead_targets: [accuracy]
  modalities: [eeg, fnirs, behavior]
  rt_bins: [0.5, 0.8]
  include_baseline: false
evaluation:
  splitter: loso
  model: logistic_regression
  guards: {check_subject_overlap: true, check_window_overlap: true, check_session_overlap: true, check_normalization_source: true}
output: {results_dir: results}
"""


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_loads_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg["seed"] == 1
    assert cfg["simulation"]["n_subjects"] == 2


def test_rejects_unknown_top_level_key(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, MINIMAL + "\nbogus: 1\n"))


def test_rejects_unknown_nested_key(tmp_path):
    bad = MINIMAL.replace("  n_subjects: 2", "  n_subjects: 2\n  typo_key: 9")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_seed(tmp_path):
    bad = MINIMAL.replace("seed: 1\n", "")
    with pytest.raises(ConfigError, match="seed"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_required_section(tmp_path):
    bad = MINIMAL.replace("windowing: {window_s: 5.0, step_s: 1.0}\n", "")
    with pytest.raises(ConfigError, match="windowing"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_nested_section(tmp_path):
    """simulation은 있는데 simulation.eeg가 없는 config를 통과시키지 않는다.

    예전에는 최상위 키만 검사해서 이런 config가 로더를 통과한 뒤 한참
    뒤에 맨 KeyError로 죽었다 — 어느 config의 어느 경로가 문제인지
    알려주지 않는 실패다.
    """
    bad = MINIMAL.replace("  eeg: {n_channels: 4, sfreq_hz: 100}\n", "")
    with pytest.raises(ConfigError, match=r"missing required key 'simulation\.eeg'"):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_nested_leaf(tmp_path):
    bad = MINIMAL.replace(
        "  guards: {check_subject_overlap: true, check_window_overlap: true, "
        "check_session_overlap: true, check_normalization_source: true}",
        "  guards: {check_subject_overlap: true, check_session_overlap: true, "
        "check_normalization_source: true}",
    )
    with pytest.raises(
        ConfigError,
        match=r"missing required key 'evaluation\.guards\.check_window_overlap'",
    ):
        load_config(_write(tmp_path, bad))


def test_rejects_missing_deep_task_key(tmp_path):
    bad = MINIMAL.replace("    stim_interval_s: 2.0\n", "")
    with pytest.raises(
        ConfigError, match=r"missing required key 'simulation\.task\.stim_interval_s'"
    ):
        load_config(_write(tmp_path, bad))


def test_smoke_config_carries_session_keys():
    cfg = load_config("config/experiments/smoke.yaml")
    assert cfg["simulation"]["n_sessions"] >= 2
    assert cfg["simulation"]["task"]["baseline_duration_s"] > 0
    assert cfg["simulation"]["practice"]["rate"] >= 0.0
    assert cfg["simulation"]["drift"]["assignment"] in ("sampled", "fixed_2x2")
    assert cfg["preprocessing"]["baseline"]["normalize"] is True
    assert cfg["dataset"]["include_baseline"] is False
    assert cfg["evaluation"]["guards"]["check_session_overlap"] is True
    assert cfg["evaluation"]["guards"]["check_normalization_source"] is True


def test_validate_config_rejects_unknown_key_after_manual_edit():
    cfg = load_config("config/experiments/smoke.yaml")
    cfg["simulation"]["n_sesions"] = 3          # 오타
    with pytest.raises(ConfigError, match=r"unknown key 'simulation\.n_sesions'"):
        validate_config(cfg)


def test_validate_config_rejects_missing_key():
    cfg = load_config("config/experiments/smoke.yaml")
    del cfg["simulation"]["n_sessions"]
    with pytest.raises(ConfigError, match=r"missing required key 'simulation\.n_sessions'"):
        validate_config(cfg)
