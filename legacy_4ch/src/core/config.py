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

    def __post_init__(self) -> None:
        if len(self.source_detector_pairs) != self.n_channels:
            raise ValueError(
                f"source_detector_pairs length {len(self.source_detector_pairs)} "
                f"does not match n_channels {self.n_channels}"
            )
        if not all(len(p) == 2 for p in self.source_detector_pairs):
            raise ValueError("Each source_detector_pair must have exactly 2 elements [source, detector]")


@dataclass
class ProcessingConfig:
    bandpass_low_hz: float
    bandpass_high_hz: float
    baseline_window_sec: float
    extinction_coefficients: dict[str, list[float]]
    dpf: list[float]


@dataclass
class StorageConfig:
    data_dir: Path
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
        try:
            storage_data = data["storage"].copy()
            storage_data["data_dir"] = Path(storage_data["data_dir"])
            return cls(
                device=DeviceConfig(**data["device"]),
                processing=ProcessingConfig(**data["processing"]),
                storage=StorageConfig(**storage_data),
                simulator=SimulatorConfig(**data["simulator"]),
            )
        except KeyError as e:
            raise ValueError(f"Config file is missing required section or field: {e}") from e
        except TypeError as e:
            raise ValueError(f"Config file has unexpected or missing fields: {e}") from e
