from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QStatusBar, QTabWidget,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from src.core.models import ProcessedSample
from src.core.config import AppConfig
from src.ui.widgets.calibration_widget import CalibrationWidget
from src.ui.widgets.brain_map_widget import BrainMapWidget
from src.ui.widgets.time_series_widget import TimeSeriesWidget


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0d0d1a;
    color: #e0e0e0;
    font-family: Arial;
}
QTabWidget::pane {
    border: 1px solid #2a2a4a;
    background: #0d0d1a;
}
QTabBar::tab {
    background: #1a1a2e;
    color: #888888;
    padding: 8px 24px;
    border: 1px solid #2a2a4a;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #0d0d1a;
    color: #ffffff;
    border-bottom: 2px solid #4a9eff;
}
QTabBar::tab:disabled {
    color: #333344;
}
QPushButton {
    background: #1e2a4a;
    color: #ffffff;
    border: 1px solid #4a9eff;
    padding: 6px 20px;
    border-radius: 4px;
}
QPushButton:disabled {
    background: #111122;
    color: #333344;
    border-color: #222233;
}
QPushButton:hover:enabled {
    background: #2a3a6a;
}
QProgressBar {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 3px;
    text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4a9eff, stop:1 #44ddaa);
    border-radius: 2px;
}
QLabel { color: #e0e0e0; }
QStatusBar { color: #888888; background: #0a0a14; }
"""


class MainWindow(QMainWindow):
    """fNIRS 실시간 모니터링 메인 윈도우.

    탭 구성:
    - Calibration: 캘리브레이션 화면 (시작 시 표시)
    - 3D Brain: 채널별 HbO/HbR 뇌 지도
    - Time Series: 채널별 실시간 시계열 그래프
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._n_channels = config.device.n_channels

        self.setWindowTitle("fNIRS 집중도 모니터")
        self.resize(1280, 860)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 상단 바 ──────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(12, 8, 12, 8)

        title = QLabel("fNIRS 집중도 모니터")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #4a9eff;")
        top_bar.addWidget(title)
        top_bar.addStretch()

        # 집중도 미터
        ci_label = QLabel("집중도")
        ci_label.setStyleSheet("color: #888;")
        self._ci_bar = QProgressBar()
        self._ci_bar.setRange(0, 100)
        self._ci_bar.setValue(0)
        self._ci_bar.setFixedWidth(220)
        self._ci_bar.setFixedHeight(22)
        self._ci_value_label = QLabel("--")
        self._ci_value_label.setFixedWidth(40)
        self._ci_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # 시작/중지 버튼
        self._start_btn = QPushButton("▶  측정 시작")
        self._stop_btn = QPushButton("■  측정 중지")
        self._stop_btn.setEnabled(False)

        top_bar.addWidget(ci_label)
        top_bar.addSpacing(6)
        top_bar.addWidget(self._ci_bar)
        top_bar.addWidget(self._ci_value_label)
        top_bar.addSpacing(16)
        top_bar.addWidget(self._start_btn)
        top_bar.addWidget(self._stop_btn)
        layout.addLayout(top_bar)

        # ── 탭 위젯 ──────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab 0: Calibration
        self._calibration = CalibrationWidget(self._config)
        self._tabs.addTab(self._calibration, "Calibration")

        # Tab 1: 3D Brain
        self._brain_map = BrainMapWidget(self._n_channels)
        self._tabs.addTab(self._brain_map, "3D Brain")

        # Tab 2: Time Series
        self._time_series = TimeSeriesWidget(self._config)
        self._tabs.addTab(self._time_series, "Time Series")

        # 캘리브레이션 완료 전 모니터링 탭 비활성화
        self._tabs.setTabEnabled(1, False)
        self._tabs.setTabEnabled(2, False)
        self._tabs.setCurrentIndex(0)

        self._calibration.calibration_done.connect(self._on_calibration_done)

        layout.addWidget(self._tabs)

        # 상태바
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("준비 — 측정 시작 버튼을 누르세요.")

    def _on_calibration_done(self) -> None:
        """캘리브레이션 완료 → 모니터링 탭 활성화 후 Time Series로 이동."""
        self._tabs.setTabEnabled(1, True)
        self._tabs.setTabEnabled(2, True)
        self._tabs.setCurrentIndex(2)
        self.statusBar().showMessage("캘리브레이션 완료 — 측정 중...")

    @Slot(object)
    def update_sample(self, sample: ProcessedSample) -> None:
        """ProcessingPipeline on_sample 콜백 — 모든 탭 데이터 갱신."""
        # 집중도 바
        ci_pct = int(sample.concentration_index * 100)
        self._ci_bar.setValue(ci_pct)
        self._ci_value_label.setText(f"{ci_pct}%")

        # 뇌 지도
        self._brain_map.update_data(sample.hbo, sample.hbr)

        # 시계열
        self._time_series.update_sample(sample)

    def start_session(self) -> None:
        """측정 시작 — 캘리브레이션 시작."""
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._tabs.setCurrentIndex(0)
        self._calibration.start_calibration()
        self.statusBar().showMessage("캘리브레이션 중...")

    def stop_session(self) -> None:
        """측정 중지."""
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._tabs.setTabEnabled(1, False)
        self._tabs.setTabEnabled(2, False)
        self._tabs.setCurrentIndex(0)
        self.statusBar().showMessage("측정 중지됨")
