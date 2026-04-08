import numpy as np
import pytest
from src.core.models import RawPacket, ProcessedSample


def test_raw_packet_construction():
    packet = RawPacket(
        timestamp=1000.0,
        channel_intensities=[0.5] * 12,  # 4채널 × 3파장
        n_wavelengths=3,
    )
    assert packet.timestamp == 1000.0
    assert len(packet.channel_intensities) == 12
    assert packet.n_wavelengths == 3


def test_raw_packet_intensity_by_channel():
    # 채널 0: [wl0=0.1, wl1=0.2, wl2=0.3]
    # 채널 1: [wl0=0.4, wl1=0.5, wl2=0.6]
    intensities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    packet = RawPacket(timestamp=0.0, channel_intensities=intensities, n_wavelengths=3)
    ch0 = packet.intensity_by_channel(channel=0)
    assert ch0 == [0.1, 0.2, 0.3]
    ch1 = packet.intensity_by_channel(channel=1)
    assert ch1 == [0.4, 0.5, 0.6]
    ch3 = packet.intensity_by_channel(channel=3)
    assert ch3 == [1.0, 1.1, 1.2]


def test_processed_sample_construction():
    hbo = np.array([0.1, 0.2, 0.3, 0.4])
    hbr = np.array([-0.05, -0.03, -0.02, -0.04])
    sample = ProcessedSample(
        timestamp=1000.0,
        hbo=hbo,
        hbr=hbr,
        concentration_index=0.65,
    )
    assert sample.concentration_index == 0.65
    assert sample.hbo.shape == (4,)
    assert sample.hbr.shape == (4,)


def test_processed_sample_validates_channel_count():
    with pytest.raises(ValueError, match="채널 수"):
        ProcessedSample(
            timestamp=0.0,
            hbo=np.zeros(3),   # 잘못된 채널 수 (hbr는 4)
            hbr=np.zeros(4),
            concentration_index=0.0,
        )


def test_raw_packet_validates_intensity_length():
    with pytest.raises(ValueError, match="divisible"):
        RawPacket(
            timestamp=0.0,
            channel_intensities=[0.1, 0.2, 0.3, 0.4, 0.5],  # 5 is not divisible by 3
            n_wavelengths=3,
        )


def test_raw_packet_intensity_by_channel_out_of_range():
    packet = RawPacket(
        timestamp=0.0,
        channel_intensities=[0.1, 0.2, 0.3] * 4,  # 4 channels × 3 wavelengths
        n_wavelengths=3,
    )
    with pytest.raises(IndexError):
        packet.intensity_by_channel(channel=4)  # valid range is 0-3
