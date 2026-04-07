import pytest
from pathlib import Path
from src.core.config import AppConfig


def test_config_loads_from_yaml(tmp_path):
    yaml_content = """
device:
  wavelengths_nm: [735, 850]
  sampling_rate_hz: 10
  n_channels: 8
  sds_mm: 30.0
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
processing:
  bandpass_low_hz: 0.01
  bandpass_high_hz: 0.5
  baseline_window_sec: 5.0
  extinction_coefficients:
    hbo: [1.4067, 0.9012]
    hbr: [3.7216, 0.7234]
  dpf: [6.51, 5.86]
storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"
simulator:
  hbo_amplitude: 0.5
  hbo_freq_hz: 0.1
  noise_std: 0.05
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    config = AppConfig.from_yaml(config_file)

    assert config.device.wavelengths_nm == [735, 850]
    assert config.device.sampling_rate_hz == 10
    assert config.device.n_channels == 8
    assert config.device.sds_mm == 30.0
    assert len(config.device.source_detector_pairs) == 2
    assert config.processing.bandpass_low_hz == 0.01
    assert config.processing.extinction_coefficients["hbo"] == [1.4067, 0.9012]
    assert config.simulator.noise_std == 0.05


def test_config_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        AppConfig.from_yaml(Path("nonexistent.yaml"))
