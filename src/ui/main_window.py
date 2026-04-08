from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QStatusBar,
)
from PySide6.QtCore import Slot
from PySide6.QtGui import QFont
from src.core.models import ProcessedSample
from src.core.config import AppConfig


class MainWindow(QMainWindow):
    """fNIRS 실시간 모니터링 메인 윈도우 (프로토타입).

    ProcessingPipeline의 on_sample 콜백으로부터 update_sample()을 통해
    실시간 HbO/HbR 그래프와 집중도 미터를 업데이트한다.
    """

    PLOT_WINDOW_SEC = 30.0

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._n_channels = config.device.n_channels  # NEVER hardcode
        self._sr = config.device.sampling_rate_hz
        self._max_points = int(self.PLOT_WINDOW_SEC * self._sr)

        self.setWindowTitle("fNIRS 집중도 모니터")
        self.resize(1200, 800)
        self._build_ui()
        self._init_plot_data()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 상단: 집중도 미터
        top_layout = QHBoxLayout()
        ci_label = QLabel("집중도")
        ci_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self._ci_bar = QProgressBar()
        self._ci_bar.setRange(0, 100)
        self._ci_bar.setValue(0)
        self._ci_bar.setFixedHeight(30)
        self._ci_value_label = QLabel("0%")
        self._ci_value_label.setFont(QFont("Arial", 14))
        top_layout.addWidget(ci_label)
        top_layout.addWidget(self._ci_bar, stretch=1)
        top_layout.addWidget(self._ci_value_label)
        main_layout.addLayout(top_layout)

        # 중앙: 실시간 HbO/HbR 그래프 (n_channels개)
        self._plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self._plot_widget, stretch=1)
        self._plots: list[pg.PlotItem] = []
        self._hbo_curves: list[pg.PlotDataItem] = []
        self._hbr_curves: list[pg.PlotDataItem] = []
        cols = min(self._n_channels, 4)  # max 4 columns
        for ch in range(self._n_channels):
            p = self._plot_widget.addPlot(row=ch // cols, col=ch % cols)
            p.setTitle(f"Ch {ch + 1}", size="10pt")
            p.setLabel("left", "μmol/L")
            p.showGrid(x=True, y=True, alpha=0.3)
            hbo_curve = p.plot(pen=pg.mkPen("r", width=1.5), name="HbO")
            hbr_curve = p.plot(pen=pg.mkPen("b", width=1.5), name="HbR")
            self._plots.append(p)
            self._hbo_curves.append(hbo_curve)
            self._hbr_curves.append(hbr_curve)

        # 하단: 컨트롤 버튼
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("측정 시작")
        self._stop_btn = QPushButton("측정 중지")
        self._stop_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self._start_btn)
        btn_layout.addWidget(self._stop_btn)
        main_layout.addLayout(btn_layout)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("준비")

    def _init_plot_data(self) -> None:
        self._hbo_data: list[list[float]] = [[] for _ in range(self._n_channels)]
        self._hbr_data: list[list[float]] = [[] for _ in range(self._n_channels)]

    @Slot(object)
    def update_sample(self, sample: ProcessedSample) -> None:
        """ProcessingPipeline의 on_sample 콜백에서 호출된다.

        Args:
            sample: 처리된 단일 시간점 데이터. concentration_index는 [0.0, 1.0]
                범위이며 0-100 정수로 스케일하여 CI 바에 표시한다.
                hbo/hbr 배열의 채널별 최신 값을 슬라이딩 윈도우 버퍼에 추가하고
                각 채널의 그래프를 갱신한다.
        """
        ci_pct = int(sample.concentration_index * 100)
        self._ci_bar.setValue(ci_pct)
        self._ci_value_label.setText(f"{ci_pct}%")

        for ch in range(self._n_channels):
            self._hbo_data[ch].append(float(sample.hbo[ch]))
            self._hbr_data[ch].append(float(sample.hbr[ch]))
            # HbO와 HbR을 항상 함께 트리밍하여 두 리스트 길이를 동기화한다
            if len(self._hbo_data[ch]) > self._max_points:
                self._hbo_data[ch] = self._hbo_data[ch][-self._max_points:]
            if len(self._hbr_data[ch]) > self._max_points:
                self._hbr_data[ch] = self._hbr_data[ch][-self._max_points:]
            self._hbo_curves[ch].setData(self._hbo_data[ch])
            self._hbr_curves[ch].setData(self._hbr_data[ch])

    def start_session(self) -> None:
        """측정 세션을 시작한다. start 버튼 비활성화, stop 버튼 활성화."""
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.statusBar().showMessage("측정 중...")

    def stop_session(self) -> None:
        """측정 세션을 중지한다. start 버튼 활성화, stop 버튼 비활성화."""
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage("측정 중지됨")
