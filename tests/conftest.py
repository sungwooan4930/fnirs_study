import pytest
from pathlib import Path
from src.core.config import AppConfig


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    # 테스트용: sampling_rate_hz=100, baseline_window_sec=0.3
    # → min_samples = int(0.3*100)+1 = 31, 1.5초 대기 시 ~150패킷 수집 가능
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
