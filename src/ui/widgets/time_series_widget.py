from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Slot
from src.core.models import ProcessedSample
from src.core.config import AppConfig


class TimeSeriesWidget(QWidget):
    """채널별 HbO/HbR 실시간 시계열 그래프."""

    PLOT_WINDOW_SEC = 5.0
    PLOT_Y_MIN = -5.0
    PLOT_Y_MAX = 5.0

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._n_channels = config.device.n_channels
        self._sr = config.device.sampling_rate_hz
        self._max_points = max(int(self.PLOT_WINDOW_SEC * self._sr), 2)
        self._build_ui()
        self._init_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground("#0d0d1a")
        layout.addWidget(self._plot_widget)

        self._hbo_curves: list[pg.PlotDataItem] = []
        self._hbr_curves: list[pg.PlotDataItem] = []
        self._plots: list[pg.PlotItem] = []

        cols = min(self._n_channels, 4)
        for ch in range(self._n_channels):
            p = self._plot_widget.addPlot(row=ch // cols, col=ch % cols)
            p.setTitle(f"Ch {ch + 1}", size="10pt", color="#cccccc")
            p.setLabel("left", "μmol/L", color="#888888")
            p.setLabel("bottom", "sec", color="#888888")
            p.setXRange(-self.PLOT_WINDOW_SEC, 0, padding=0)
            p.setYRange(self.PLOT_Y_MIN, self.PLOT_Y_MAX, padding=0)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.getAxis("left").setPen(pg.mkPen("#444"))
            p.getAxis("bottom").setPen(pg.mkPen("#444"))
            hbo = p.plot(pen=pg.mkPen("#ff4444", width=1.5), name="HbO")
            hbr = p.plot(pen=pg.mkPen("#4488ff", width=1.5), name="HbR")
            self._plots.append(p)
            self._hbo_curves.append(hbo)
            self._hbr_curves.append(hbr)

    def _init_data(self) -> None:
        self._hbo_data: list[list[float]] = [[] for _ in range(self._n_channels)]
        self._hbr_data: list[list[float]] = [[] for _ in range(self._n_channels)]
        self._time_axis: list[float] = []

    @Slot(object)
    def update_sample(self, sample: ProcessedSample) -> None:
        self._time_axis.append(sample.timestamp)
        if len(self._time_axis) > self._max_points:
            self._time_axis = self._time_axis[-self._max_points:]
        t0 = self._time_axis[-1]
        x = [t - t0 for t in self._time_axis]

        for ch in range(self._n_channels):
            self._hbo_data[ch].append(float(sample.hbo[ch]))
            self._hbr_data[ch].append(float(sample.hbr[ch]))
            if len(self._hbo_data[ch]) > self._max_points:
                self._hbo_data[ch] = self._hbo_data[ch][-self._max_points:]
            if len(self._hbr_data[ch]) > self._max_points:
                self._hbr_data[ch] = self._hbr_data[ch][-self._max_points:]
            self._hbo_curves[ch].setData(x, self._hbo_data[ch])
            self._hbr_curves[ch].setData(x, self._hbr_data[ch])
