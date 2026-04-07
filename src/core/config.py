from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class DeviceConfig:
    wavelengths_nm: list[int]
    sampling_rate_hz: float
    n_channels: int
    sds_mm: float
    source_detector_pairs: list[list[int]]


@dataclass
class ProcessingConfig:
    bandpass_low_hz: float
    bandpass_high_hz: float
    baseline_window_sec: float
    extinction_coefficients: dict[str, list[float]]
    dpf: list[float]


@dataclass
class StorageConfig:
    data_dir: str
    session_filename_format: str


@dataclass
class SimulatorConfig:
    hbo_amplitude: float
    hbo_freq_hz: float
    noise_std: float


@dataclass
class AppConfig:
    device: DeviceConfig
    processing: ProcessingConfig
    storage: StorageConfig
    simulator: SimulatorConfig

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            device=DeviceConfig(**data["device"]),
            processing=ProcessingConfig(**data["processing"]),
            storage=StorageConfig(**data["storage"]),
            simulator=SimulatorConfig(**data["simulator"]),
        )
