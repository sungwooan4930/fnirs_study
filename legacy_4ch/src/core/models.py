from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RawPacket:
    """장치에서 수신한 원시 패킷.

    channel_intensities: [ch0_wl0, ch0_wl1, ch0_wl2, ch1_wl0, ch1_wl1, ch1_wl2, ...]
    n_wavelengths: 파장 수 (config.device.wavelengths_nm의 길이와 동일)
    """
    timestamp: float
    channel_intensities: list[float]
    n_wavelengths: int

    def __post_init__(self) -> None:
        if self.n_wavelengths <= 0:
            raise ValueError(f"n_wavelengths must be positive, got {self.n_wavelengths}")
        if len(self.channel_intensities) % self.n_wavelengths != 0:
            raise ValueError(
                f"channel_intensities length {len(self.channel_intensities)} "
                f"is not divisible by n_wavelengths {self.n_wavelengths}"
            )

    def intensity_by_channel(self, channel: int) -> list[float]:
        """채널 인덱스로 해당 채널의 모든 파장 강도값을 반환한다."""
        n_channels = len(self.channel_intensities) // self.n_wavelengths
        if channel < 0 or channel >= n_channels:
            raise IndexError(f"channel {channel} out of range [0, {n_channels})")
        start = channel * self.n_wavelengths
        return self.channel_intensities[start : start + self.n_wavelengths]


@dataclass
class ProcessedSample:
    """처리된 단일 시간점 데이터."""
    timestamp: float
    hbo: np.ndarray   # shape: (n_channels,), μmol/L
    hbr: np.ndarray   # shape: (n_channels,), μmol/L
    concentration_index: float  # [0.0, 1.0]

    def __post_init__(self) -> None:
        if self.hbo.shape != self.hbr.shape:
            raise ValueError(f"채널 수 불일치: hbo={self.hbo.shape}, hbr={self.hbr.shape}")
