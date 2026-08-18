import time
import pytest
from src.core.ring_buffer import RingBuffer
from src.acquisition.simulator import FNIRSSimulator
from src.acquisition.acquisition_thread import AcquisitionThread


def test_acquisition_thread_fills_buffer(app_config):
    device = FNIRSSimulator(app_config)
    buffer = RingBuffer(capacity=50)
    thread = AcquisitionThread(device=device, buffer=buffer)

    thread.start()
    time.sleep(0.15)  # 100Hz × 0.15s ≈ 15 packets
    thread.stop()
    thread.join(timeout=2.0)

    collected = []
    while True:
        p = buffer.get(timeout=0.01)
        if p is None:
            break
        collected.append(p)

    assert len(collected) >= 5  # minimum 5 packets collected


def test_acquisition_thread_stops_cleanly(app_config):
    device = FNIRSSimulator(app_config)
    buffer = RingBuffer(capacity=50)
    thread = AcquisitionThread(device=device, buffer=buffer)

    thread.start()
    time.sleep(0.05)
    thread.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()


def test_acquisition_thread_is_daemon(app_config):
    """스레드는 daemon=True여야 메인 종료 시 자동 종료된다."""
    device = FNIRSSimulator(app_config)
    buffer = RingBuffer(capacity=10)
    thread = AcquisitionThread(device=device, buffer=buffer)
    assert thread.daemon is True


def test_acquisition_thread_packets_have_correct_structure(app_config):
    """수집된 패킷은 4채널 × 3파장 = 12개 강도값을 가져야 한다."""
    device = FNIRSSimulator(app_config)
    buffer = RingBuffer(capacity=50)
    thread = AcquisitionThread(device=device, buffer=buffer)

    thread.start()
    time.sleep(0.15)
    thread.stop()
    thread.join(timeout=2.0)

    packet = buffer.get(timeout=0.1)
    assert packet is not None
    assert len(packet.channel_intensities) == 12  # 4ch × 3wl
    assert packet.n_wavelengths == 3
