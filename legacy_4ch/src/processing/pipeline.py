from __future__ import annotations
import threading
from collections import deque
from typing import Callable
import numpy as np
from src.core.ring_buffer import RingBuffer
from src.core.models import RawPacket, ProcessedSample
from src.core.config import AppConfig
from src.processing.filters import bandpass_filter
from src.processing.mbll import modified_beer_lambert
from src.processing.concentration import ConcentrationIndex


class ProcessingPipeline(threading.Thread):
    """링 버퍼에서 패킷을 읽어 HbO/HbR 및 집중도 지수를 계산하는 스레드.

    on_sample 콜백으로 처리 결과를 GUI 스레드에 전달한다.
    """

    _BUFFER_SEC = 10.0  # 내부 슬라이딩 윈도우 길이 (초)

    def __init__(
        self,
        buffer: RingBuffer,
        config: AppConfig,
        concentration_index: ConcentrationIndex,
        on_sample: Callable[[ProcessedSample], None],
    ) -> None:
        super().__init__(daemon=True)
        self._buffer = buffer
        self._config = config
        self._concentration_index = concentration_index
        self._on_sample = on_sample
        self._stop_event = threading.Event()

        sr = config.device.sampling_rate_hz
        max_samples = int(self._BUFFER_SEC * sr)
        self._window: deque[RawPacket] = deque(maxlen=max_samples)

    def run(self) -> None:
        cfg = self._config
        sr = cfg.device.sampling_rate_hz
        # sosfiltfilt with order=6 bandpass requires padlen=3*(2*order+1)=39 samples minimum
        # Use max of baseline window requirement and filter requirement
        _filter_min = 40  # > padlen=39 for order=6 sosfiltfilt
        min_samples = max(int(cfg.processing.baseline_window_sec * sr) + 1, _filter_min)

        while not self._stop_event.is_set():
            packet = self._buffer.get(timeout=0.1)
            if packet is None:
                continue
            self._window.append(packet)

            if len(self._window) < min_samples:
                continue

            self._process_window()

    def _process_window(self) -> None:
        cfg = self._config
        packets = list(self._window)
        n_ch = cfg.device.n_channels
        n_wl = len(cfg.device.wavelengths_nm)  # CRITICAL: derive dynamically, never hardcode
        n_t = len(packets)

        # (n_channels, n_wavelengths, n_samples) array
        raw = np.zeros((n_ch, n_wl, n_t))
        for t, pkt in enumerate(packets):
            for ch in range(n_ch):
                raw[ch, :, t] = pkt.intensity_by_channel(ch)

        # Bandpass filter per wavelength slice.
        # bandpass_filter removes DC (mean) from the signal, which causes raw intensity
        # to oscillate around 0 and breaks MBLL (log of negative/near-zero values).
        # Solution: restore the per-channel-wavelength mean after filtering so that
        # intensities remain positive and MBLL can compute a meaningful baseline.
        for wl in range(n_wl):
            slice_2d = raw[:, wl, :]  # shape (n_ch, n_t)
            mean_offset = slice_2d.mean(axis=1, keepdims=True)  # (n_ch, 1)
            filtered = bandpass_filter(
                slice_2d,
                low_hz=cfg.processing.bandpass_low_hz,
                high_hz=cfg.processing.bandpass_high_hz,
                sampling_rate_hz=cfg.device.sampling_rate_hz,
            )
            raw[:, wl, :] = filtered + mean_offset  # restore mean so intensities stay positive

        # mBLL → HbO/HbR: (n_channels, n_samples)
        hbo_all, hbr_all = modified_beer_lambert(
            raw,
            cfg.device.wavelengths_nm,
            cfg.processing.extinction_coefficients["hbo"],
            cfg.processing.extinction_coefficients["hbr"],
            cfg.processing.dpf,
            cfg.device.sds_mm,
        )

        # Latest sample (last time point)
        hbo_now = hbo_all[:, -1]  # shape (n_channels,)
        hbr_now = hbr_all[:, -1]
        ci = self._concentration_index.compute(hbo_now, hbr_now)

        sample = ProcessedSample(
            timestamp=packets[-1].timestamp,
            hbo=hbo_now,
            hbr=hbr_now,
            concentration_index=ci,
        )
        self._on_sample(sample)

    def stop(self) -> None:
        self._stop_event.set()
