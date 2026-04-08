from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from src.core.config import AppConfig


class SourceDetectorMapWidget(QWidget):
    """Source-Detector 배치도를 그리는 위젯."""

    def __init__(self, n_channels: int = 4, parent=None):
        super().__init__(parent)
        self._n_channels = n_channels
        # channel_status: None = uncalibrated, True = ok, False = bad
        self._channel_status: list[bool | None] = [None] * n_channels
        self.setMinimumSize(300, 200)

    def set_channel_status(self, statuses: list[bool | None]) -> None:
        self._channel_status = statuses
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0d0d1a"))

        # Title
        painter.setPen(QColor("#aaaaaa"))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(0, 0, w, 24, Qt.AlignmentFlag.AlignCenter, "Source - Detector Map")

        # Layout: 4 channels in 2×2 grid with source/detector pairs
        # Sources (filled blue circles), Detectors (open white circles)
        positions = [
            (0.25, 0.35), (0.75, 0.35),   # top row
            (0.25, 0.70), (0.75, 0.70),   # bottom row
        ]

        r = min(w, h) * 0.07
        for i, (fx, fy) in enumerate(positions[:self._n_channels]):
            x, y = int(fx * w), int(fy * h)
            status = self._channel_status[i] if i < len(self._channel_status) else None

            # Source (filled)
            sx = x - int(r * 1.2)
            if status is None:
                painter.setBrush(QBrush(QColor("#2a5a8a")))
            elif status:
                painter.setBrush(QBrush(QColor("#4a9eff")))
            else:
                painter.setBrush(QBrush(QColor("#8a2a2a")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(sx - r), int(y - r), int(r * 2), int(r * 2))
            painter.setPen(QColor("#aaaaaa"))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(int(sx - r), int(y - r), int(r * 2), int(r * 2),
                             Qt.AlignmentFlag.AlignCenter, str(i + 1))

            # Detector (open)
            dx = x + int(r * 1.2)
            if status is None:
                pen_color = QColor("#555566")
            elif status:
                pen_color = QColor("#44dd88")
            else:
                pen_color = QColor("#dd4444")
            painter.setPen(QPen(pen_color, 2))
            painter.setBrush(QBrush(QColor("#0d0d1a")))
            painter.drawEllipse(int(dx - r), int(y - r), int(r * 2), int(r * 2))
            painter.setPen(pen_color)
            painter.drawText(int(dx - r), int(y - r), int(r * 2), int(r * 2),
                             Qt.AlignmentFlag.AlignCenter, str(i + 1))

        # Legend
        lx, ly = 20, h - 30
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#4a9eff")))
        painter.drawEllipse(lx, ly, 10, 10)
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(lx + 14, ly - 2, 60, 16, Qt.AlignmentFlag.AlignVCenter, "Source")

        painter.setPen(QPen(QColor("#aaaaaa"), 2))
        painter.setBrush(QBrush(QColor("#0d0d1a")))
        painter.drawEllipse(lx + 80, ly, 10, 10)
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(lx + 94, ly - 2, 60, 16, Qt.AlignmentFlag.AlignVCenter, "Detector")

        painter.end()


class CalibrationWidget(QWidget):
    """캘리브레이션 화면. SNR 측정 후 진행 버튼 활성화."""

    calibration_done = Signal()  # 진행 버튼 클릭 시

    _CAL_STEPS = 30  # 타이머 틱 수
    _GOOD_SNR_THRESHOLD = 40  # dB

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._n_channels = config.device.n_channels
        self._wavelengths = config.device.wavelengths_nm
        self._n_wl = len(self._wavelengths)
        self._step = 0
        self._is_done = False

        # Simulated SNR per channel per wavelength (dB)
        rng = np.random.default_rng(42)
        self._snr_values: np.ndarray = rng.uniform(35, 55, size=(self._n_channels, self._n_wl))

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    def start_calibration(self) -> None:
        self._step = 0
        self._is_done = False
        self._proceed_btn.setEnabled(False)
        self._status_label.setText("Calibrating...")
        self._sd_map.set_channel_status([None] * self._n_channels)
        for bars in self._bar_items:
            bars.setOpts(height=[0] * self._n_channels)
        self._timer.start(100)  # 100ms × 30 steps = 3초

    def _on_tick(self) -> None:
        self._step += 1
        progress = self._step / self._CAL_STEPS

        # 점진적으로 SNR 바 채우기
        for wl_idx, bars in enumerate(self._bar_items):
            heights = [self._snr_values[ch, wl_idx] * progress
                       for ch in range(self._n_channels)]
            bars.setOpts(height=heights)

        if self._step >= self._CAL_STEPS:
            self._timer.stop()
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._is_done = True
        # 채널 상태 판정
        statuses = [
            bool(self._snr_values[ch].mean() >= self._GOOD_SNR_THRESHOLD)
            for ch in range(self._n_channels)
        ]
        self._sd_map.set_channel_status(statuses)
        ok = sum(statuses)
        self._status_label.setText(
            f"Calibration Complete — {ok}/{self._n_channels} channels OK"
        )
        self._proceed_btn.setEnabled(True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 상단: SD맵 + 뇌 지도
        top = QHBoxLayout()
        self._sd_map = SourceDetectorMapWidget(self._n_channels)
        top.addWidget(self._sd_map, stretch=2)

        # 간이 뇌 지도 (QLabel placeholder)
        brain_label = QLabel("Brain Map")
        brain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brain_label.setStyleSheet("color:#555; border:1px solid #333; border-radius:4px;")
        brain_label.setFixedSize(180, 160)
        top.addWidget(brain_label)
        layout.addLayout(top)

        # SNR 바 차트 (파장별): x=채널 인덱스, y=SNR(dB)
        snr_layout = QHBoxLayout()
        self._bar_items: list[pg.BarGraphItem] = []
        self._snr_plots: list[pg.PlotWidget] = []
        x = np.arange(self._n_channels)
        colors = ["#4a9eff", "#44dd88", "#ffaa44"]  # 파장별 색상

        for wl_idx, wl in enumerate(self._wavelengths):
            pw = pg.PlotWidget(title=f"{wl} nm SNR")
            pw.setBackground("#0d0d1a")
            pw.setLabel("left", "SNR (dB)")
            pw.setLabel("bottom", "Channel")
            pw.setYRange(0, 60, padding=0.1)
            pw.setXRange(-0.6, self._n_channels - 0.4, padding=0)
            pw.getAxis("bottom").setTicks(
                [[(i, f"Ch{i + 1}") for i in range(self._n_channels)]]
            )
            bars = pg.BarGraphItem(
                x=x, height=[0] * self._n_channels,
                width=0.6, brush=colors[wl_idx % len(colors)]
            )
            pw.addItem(bars)
            self._bar_items.append(bars)
            self._snr_plots.append(pw)
            snr_layout.addWidget(pw)
        layout.addLayout(snr_layout)

        # 하단: 상태 + 버튼
        bottom = QHBoxLayout()
        self._status_label = QLabel("장치를 연결하고 측정 시작 버튼을 누르세요.")
        self._status_label.setStyleSheet("color: #aaa;")
        bottom.addWidget(self._status_label, stretch=1)
        self._proceed_btn = QPushButton("진행  ▶")
        self._proceed_btn.setEnabled(False)
        self._proceed_btn.clicked.connect(self.calibration_done.emit)
        bottom.addWidget(self._proceed_btn)
        layout.addLayout(bottom)
