import numpy as np
import pytest
from src.core.config import AppConfig
from src.core.models import ProcessedSample
from src.ui.main_window import MainWindow


@pytest.fixture
def config(tmp_path):
    yaml_content = """
device:
  wavelengths_nm: [780, 850, 950]
  sampling_rate_hz: 10
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


def test_main_window_opens(qtbot, config):
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()


def test_update_sample_updates_ci_bar(qtbot, config):
    window = MainWindow(config)
    qtbot.addWidget(window)
    sample = ProcessedSample(
        timestamp=1.0,
        hbo=np.array([1.0] * 4),
        hbr=np.array([-0.1] * 4),
        concentration_index=0.75,
    )
    window.update_sample(sample)
    assert window._ci_bar.value() == 75
    assert window._ci_value_label.text() == "75%"


def test_update_sample_appends_plot_data(qtbot, config):
    """update_sample 호출 시 각 채널의 plot data가 추가된다."""
    window = MainWindow(config)
    qtbot.addWidget(window)
    sample = ProcessedSample(
        timestamp=1.0,
        hbo=np.array([0.5, 0.3, 0.1, 0.4]),
        hbr=np.array([-0.1, -0.05, 0.0, -0.2]),
        concentration_index=0.5,
    )
    window.update_sample(sample)
    assert len(window._hbo_data[0]) == 1
    assert len(window._hbr_data[3]) == 1
    assert len(window._time_axis) == 1


def test_sliding_window_trims_to_max_points(qtbot, config):
    """max_points 초과 시 가장 오래된 샘플이 제거된다 (최근 5초만 유지)."""
    window = MainWindow(config)
    qtbot.addWidget(window)
    # sampling_rate=10, PLOT_WINDOW_SEC=5 → max_points=50
    max_pts = window._max_points
    for i in range(max_pts + 10):
        s = ProcessedSample(
            timestamp=float(i) / 10.0,
            hbo=np.array([0.1] * 4),
            hbr=np.array([-0.1] * 4),
            concentration_index=0.5,
        )
        window.update_sample(s)
    assert len(window._hbo_data[0]) == max_pts
    assert len(window._time_axis) == max_pts


def test_time_axis_latest_is_zero(qtbot, config):
    """x축에서 가장 최근 포인트는 t=0이어야 한다."""
    window = MainWindow(config)
    qtbot.addWidget(window)
    for i in range(5):
        s = ProcessedSample(
            timestamp=float(i),
            hbo=np.array([0.1] * 4),
            hbr=np.array([-0.1] * 4),
            concentration_index=0.5,
        )
        window.update_sample(s)
    t0 = window._time_axis[-1]
    x = [t - t0 for t in window._time_axis]
    assert x[-1] == 0.0
    assert x[0] < 0.0


def test_n_channels_from_config(qtbot, config):
    """채널 수는 config에서 읽어야 한다."""
    window = MainWindow(config)
    qtbot.addWidget(window)
    assert window._n_channels == 4
    assert len(window._hbo_curves) == 4
    assert len(window._hbr_curves) == 4
