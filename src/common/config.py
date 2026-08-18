"""실험 config 로더.

스키마에 정의되지 않은 키를 만나면 실행을 거부한다. 오타가 조용히
기본값으로 흡수되어 재현 불가능한 결과로 이어지는 것을 막기 위함이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """config가 스키마를 위반했을 때 발생."""


# dict면 하위 스키마, None이면 리프(값 검사는 하지 않음)
SCHEMA: dict[str, Any] = {
    "run_name": None,
    "seed": None,
    "simulation": {
        "n_subjects": None,
        "subject_variance": None,
        "effect_size": None,
        "lead_delta_s": None,
        "task": {
            "nback_levels": None,
            "block_duration_s": None,
            "n_blocks_per_level": None,
            "stim_interval_s": None,
        },
        "eeg": {"n_channels": None, "sfreq_hz": None},
        "fnirs": {"n_channels": None, "sfreq_hz": None, "hbr_coupling": None},
    },
    "windowing": {"window_s": None, "step_s": None},
    "features": {"extractor": None},
    "dataset": {
        "targets": None,
        "lead_targets": None,
        "modalities": None,
        "rt_bins": None,
    },
    "evaluation": {
        "splitter": None,
        "model": None,
        "guards": {"check_subject_overlap": None, "check_window_overlap": None},
    },
    "output": {"results_dir": None},
}

REQUIRED_TOP = (
    "run_name", "seed", "simulation", "windowing",
    "features", "dataset", "evaluation", "output",
)


def _check_node(node: Any, schema: Any, path: str) -> None:
    if schema is None:
        return
    if not isinstance(node, dict):
        raise ConfigError(f"{path or 'root'}: expected a mapping, got {type(node).__name__}")
    for key, value in node.items():
        if key not in schema:
            raise ConfigError(f"unknown key '{path}{key}'")
        _check_node(value, schema[key], f"{path}{key}.")


def load_config(path: str | Path) -> dict:
    """YAML config를 읽고 스키마를 검증해 반환한다."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    _check_node(cfg, SCHEMA, "")

    for key in REQUIRED_TOP:
        if key not in cfg:
            raise ConfigError(f"missing required key '{key}'")

    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ConfigError(f"seed must be int, got {type(cfg['seed']).__name__}")

    return cfg
