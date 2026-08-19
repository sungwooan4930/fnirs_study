"""실험 config 로더.

스키마에 정의되지 않은 키를 만나면 실행을 거부한다. 오타가 조용히
기본값으로 흡수되어 재현 불가능한 결과로 이어지는 것을 막기 위함이다.

같은 이유로 스키마에 있는 키가 **빠진** 경우도 거부한다. 중첩 리프까지
전부 필수다. 예전에는 최상위 키만 검사해서 `simulation:`은 있는데
`simulation.eeg:`가 없는 config가 로더를 통과한 뒤 한참 뒤에 맨
`KeyError: 'eeg'`로 죽었다 — 어느 config의 어느 경로가 문제인지
알려주지 않는 실패다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """config가 스키마를 위반했을 때 발생."""


# dict면 하위 스키마, None이면 리프(값 검사는 하지 않음).
# 스키마에 적힌 키는 전부 필수다 — 선택적 키는 현재 없다.
SCHEMA: dict[str, Any] = {
    "run_name": None,
    "seed": None,
    "simulation": {
        "n_subjects": None,
        "n_sessions": None,
        "subject_variance": None,
        "effect_size": None,
        "lead_delta_s": None,
        "task": {
            "nback_levels": None,
            "block_duration_s": None,
            "baseline_duration_s": None,
            "n_blocks_per_level": None,
            "stim_interval_s": None,
        },
        "practice": {"rate": None},
        "drift": {
            "fnirs_gain_sigma": None,
            "fnirs_offset_sigma": None,
            "eeg_gain_sigma": None,
            "eeg_noise_sigma": None,
            "within_session_rate": None,
            "within_session_fraction": None,
            "between_session_scale": None,
            "assignment": None,
        },
        "eeg": {"n_channels": None, "sfreq_hz": None},
        "fnirs": {"n_channels": None, "sfreq_hz": None, "hbr_coupling": None},
    },
    "windowing": {"window_s": None, "step_s": None},
    "features": {"extractor": None},
    "preprocessing": {
        "baseline": {
            "normalize": None,
            "drift_threshold_relative": None,
            "zero_atol": None,
        },
    },
    "dataset": {
        "targets": None,
        "lead_targets": None,
        "modalities": None,
        "rt_bins": None,
        "include_baseline": None,
    },
    "evaluation": {
        "splitter": None,
        "model": None,
        "guards": {
            "check_subject_overlap": None,
            "check_window_overlap": None,
            "check_session_overlap": None,
            "check_normalization_source": None,
        },
    },
    "output": {"results_dir": None},
}


def _check_node(node: Any, schema: Any, path: str) -> None:
    """스키마에 없는 키를 거부한다."""
    if schema is None:
        return
    if not isinstance(node, dict):
        raise ConfigError(f"{path or 'root'}: expected a mapping, got {type(node).__name__}")
    for key, value in node.items():
        if key not in schema:
            raise ConfigError(f"unknown key '{path}{key}'")
        _check_node(value, schema[key], f"{path}{key}.")


def _check_required(node: Any, schema: Any, path: str) -> None:
    """스키마에 있는 키가 빠졌는지 경로까지 밝혀 거부한다."""
    if schema is None:
        return
    for key, sub in schema.items():
        full = f"{path}{key}"
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"missing required key '{full}'")
        _check_required(node[key], sub, f"{full}.")


def validate_config(cfg: dict) -> None:
    """이미 dict인 config를 스키마에 대해 검증한다.

    `load_config`가 파일을 읽은 뒤 호출하는 것과 같은 검사다. 오버라이드를
    병합한 뒤에도 같은 검사를 돌릴 수 있도록 분리했다 — 병합 후 검증하지
    않으면 오타 난 키가 조용히 흡수되고, config 해시만 바뀌어 아무 손잡이도
    돌리지 않은 실행이 별개 조건으로 기록된다.
    """
    if not isinstance(cfg, dict):
        raise ConfigError(f"top level must be a mapping, got {type(cfg).__name__}")
    _check_node(cfg, SCHEMA, "")
    _check_required(cfg, SCHEMA, "")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ConfigError(f"seed must be int, got {type(cfg['seed']).__name__}")


def load_config(path: str | Path) -> dict:
    """YAML config를 읽고 스키마를 검증해 반환한다."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    validate_config(cfg)
    return cfg
