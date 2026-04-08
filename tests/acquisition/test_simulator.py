import time
import pytest
from src.core.config import AppConfig
from src.acquisition.simulator import FNIRSSimulator


@pytest.fixture
def config(tmp_path):
    yaml_content = """
device:
  wavelengths_nm: [780, 850, 950]
  sampling_rate_hz: 100
  n_channels: 4
  sds_mm: 30.0
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
    - [2, 1]
    - [2, 2]
processing:
  bandpass_low_hz: 0.01
  bandpass_high_hz: 0.5
  baseline_window_sec: 0.3
  extinction_coefficients:
    hbo: [0.975, 0.901, 1.046]
    hbr: [2.755, 0.781, 0.260]
  dpf: [6.51, 5.86, 5.12]
storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"
simulator:
  hbo_amplitude: 0.5
  hbo_freq_hz: 0.1
  noise_std: 0.05
"""
    f = tmp_path / "settings.yaml"
    f.write_text(yaml_content)
    return AppConfig.from_yaml(f)


def test_simulator_implements_hal(config):
    from src.core.hal import FNIRSDevice
    sim = FNIRSSimulator(config)
    assert isinstance(sim, FNIRSDevice)


def test_simulator_connect_returns_true(config):
    sim = FNIRSSimulator(config)
    assert sim.connect() is True


def test_simulator_read_packet_returns_raw_packet(config):
    from src.core.models import RawPacket
    sim = FNIRSSimulator(config)
    sim.connect()
    sim.start_stream()
    packet = sim.read_packet()
    assert isinstance(packet, RawPacket)
    # 4채널 × 3파장 = 12개의 강도값
    assert len(packet.channel_intensities) == 12
    assert packet.n_wavelengths == 3
    sim.stop_stream()
    sim.disconnect()


def test_simulator_packet_timestamp_increases(config):
    sim = FNIRSSimulator(config)
    sim.connect()
    sim.start_stream()
    p1 = sim.read_packet()
    p2 = sim.read_packet()
    assert p2.timestamp > p1.timestamp
    sim.stop_stream()
    sim.disconnect()


def test_simulator_respects_sampling_rate(config):
    """100Hz 설정에서 10개 패킷 수집 시 약 0.1초 걸려야 한다."""
    sim = FNIRSSimulator(config)
    sim.connect()
    sim.start_stream()
    start = time.perf_counter()
    for _ in range(10):
        sim.read_packet()
    elapsed = time.perf_counter() - start
    sim.stop_stream()
    sim.disconnect()
    assert 0.05 < elapsed < 0.5  # 넓은 허용 범위
