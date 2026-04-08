from __future__ import annotations
import time
import math
import random
from src.core.hal import FNIRSDevice
from src.core.models import RawPacket
from src.core.config import AppConfig


class FNIRSSimulator(FNIRSDevice):
    """fNIRS 신호 시뮬레이터.

    sin파 + 노이즈로 현실적인 fNIRS 신호를 생성한다.
    HbO는 저주파 sin파 형태, 파장별로 서로 다른 감도로 반영된다.
    n_wavelengths는 config에서 동적으로 결정된다 (하드코딩 금지).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._n_wavelengths = len(config.device.wavelengths_nm)
        self._streaming = False
        self._start_time: float = 0.0
        self._sample_index: int = 0

    def connect(self) -> bool:
        return True

    def start_stream(self) -> None:
        self._streaming = True
        self._start_time = time.perf_counter()
        self._sample_index = 0

    def stop_stream(self) -> None:
        self._streaming = False

    def disconnect(self) -> None:
        pass

    def read_packet(self) -> RawPacket:
        """다음 샘플 타이밍까지 대기 후 패킷 반환."""
        cfg_d = self._config.device
        cfg_s = self._config.simulator
        sr = cfg_d.sampling_rate_hz
        n_ch = cfg_d.n_channels
        n_wl = self._n_wavelengths

        # 샘플 타이밍 대기
        target_time = self._start_time + self._sample_index / sr
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        t = self._sample_index / sr
        self._sample_index += 1
        timestamp = time.perf_counter()

        # 신호 생성: [ch0_wl0, ch0_wl1, ch0_wl2, ch1_wl0, ...]
        # 파장 인덱스에 따라 HbO/HbR 기여가 달라지는 현실적인 시뮬레이션
        intensities: list[float] = []
        for ch in range(n_ch):
            phase_offset = ch * (2 * math.pi / n_ch)
            hbo_signal = cfg_s.hbo_amplitude * math.sin(
                2 * math.pi * cfg_s.hbo_freq_hz * t + phase_offset
            )
            for wl_idx in range(n_wl):
                # 파장 인덱스에 따른 HbO 민감도 (장파장일수록 HbO에 민감)
                hbo_sensitivity = 0.5 + wl_idx * 0.3
                val = 1.0 + hbo_sensitivity * hbo_signal + random.gauss(0, cfg_s.noise_std)
                intensities.append(max(0.01, val))

        return RawPacket(
            timestamp=timestamp,
            channel_intensities=intensities,
            n_wavelengths=n_wl,
        )
