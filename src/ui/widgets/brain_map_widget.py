from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QRadialGradient, QFont


class BrainMapWidget(QWidget):
    """전전두엽 채널 위치에 HbO/HbR를 색상으로 표시하는 2D 뇌 지도."""

    # 전전두엽 4채널 위치 (정규화 좌표 0~1, 뇌 타원 기준)
    # 상단: 왼쪽 Ch1, 오른쪽 Ch2 / 하단: 왼쪽 Ch3, 오른쪽 Ch4
    _CHANNEL_POS = [
        (0.35, 0.52),  # Ch1 left-top PFC
        (0.65, 0.52),  # Ch2 right-top PFC
        (0.35, 0.68),  # Ch3 left-bottom PFC
        (0.65, 0.68),  # Ch4 right-bottom PFC
    ]
    _BLOB_RADIUS = 0.10  # 뇌 타원 크기 기준 반지름 비율

    def __init__(self, n_channels: int = 4, parent=None):
        super().__init__(parent)
        self._n_channels = n_channels
        self._hbo = np.zeros(n_channels)
        self._hbr = np.zeros(n_channels)
        self.setMinimumSize(400, 400)

    def update_data(self, hbo: np.ndarray, hbr: np.ndarray) -> None:
        self._hbo = hbo.copy()
        self._hbr = hbr.copy()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor("#000000"))

        # 뇌 외곽선 (타원)
        margin = 0.08
        brain_x = int(w * margin)
        brain_y = int(h * margin)
        brain_w = int(w * (1 - 2 * margin))
        brain_h = int(h * (1 - 2 * margin))

        painter.setPen(QPen(QColor("#555555"), 2))
        painter.setBrush(QBrush(QColor("#1a1a1a")))
        painter.drawEllipse(brain_x, brain_y, brain_w, brain_h)

        # 중앙 세로선 (반구 경계)
        cx = w // 2
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawLine(cx, brain_y + 10, cx, brain_y + brain_h - 10)

        # HbO/HbR 블롭
        blob_r = int(brain_w * self._BLOB_RADIUS)
        hbo_max = max(abs(self._hbo.max()), 0.1)
        hbr_max = max(abs(self._hbr.min()), 0.1)

        for ch in range(min(self._n_channels, len(self._CHANNEL_POS))):
            fx, fy = self._CHANNEL_POS[ch]
            cx_blob = int(brain_x + fx * brain_w)
            cy_blob = int(brain_y + fy * brain_h)

            # HbO: 따뜻한 색 (노랑→빨강)
            hbo_norm = max(self._hbo[ch] / hbo_max, 0.0)
            if hbo_norm > 0.01:
                grad = QRadialGradient(cx_blob, cy_blob, blob_r)
                alpha = int(min(hbo_norm * 200, 200))
                r = int(255 * min(hbo_norm * 1.5, 1.0))
                g = int(200 * max(1.0 - hbo_norm, 0.3))
                grad.setColorAt(0.0, QColor(r, g, 50, alpha))
                grad.setColorAt(1.0, QColor(r, g, 50, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(grad))
                painter.drawEllipse(cx_blob - blob_r, cy_blob - blob_r, blob_r * 2, blob_r * 2)

            # HbR: 차가운 색 (하늘→파랑)
            hbr_norm = max(-self._hbr[ch] / hbr_max, 0.0)
            if hbr_norm > 0.01:
                grad2 = QRadialGradient(cx_blob, cy_blob, blob_r)
                alpha2 = int(min(hbr_norm * 180, 180))
                b = int(255 * min(hbr_norm * 1.5, 1.0))
                g2 = int(180 * max(1.0 - hbr_norm * 0.5, 0.4))
                grad2.setColorAt(0.0, QColor(50, g2, b, alpha2))
                grad2.setColorAt(1.0, QColor(50, g2, b, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(grad2))
                painter.drawEllipse(cx_blob - blob_r, cy_blob - blob_r, blob_r * 2, blob_r * 2)

            # 채널 번호
            painter.setPen(QColor("#cccccc"))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(cx_blob - 15, cy_blob - blob_r - 14, 30, 14,
                             Qt.AlignmentFlag.AlignCenter, f"Ch{ch + 1}")

        # "Anterior" 레이블
        painter.setPen(QColor("#555555"))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(0, brain_y - 2, w, 20, Qt.AlignmentFlag.AlignCenter, "Anterior")

        painter.end()
