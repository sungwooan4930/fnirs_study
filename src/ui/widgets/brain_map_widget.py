from __future__ import annotations
import numpy as np
from scipy.interpolate import RBFInterpolator
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QImage, QPixmap, QLinearGradient, QRadialGradient,
)


class _BrainCanvas(QWidget):
    """뇌 배경 + RBF 보간 오버레이 렌더링 캔버스."""

    # 전전두엽 4채널 위치 (정규화 0~1, 뇌 타원 기준)
    _CH_POS = np.array([
        [0.37, 0.36],   # Ch1 left-anterior PFC
        [0.63, 0.36],   # Ch2 right-anterior PFC
        [0.37, 0.52],   # Ch3 left-posterior PFC
        [0.63, 0.52],   # Ch4 right-posterior PFC
    ])

    # 뇌 타원 파라미터 (정규화)
    _BX, _BY = 0.08, 0.06
    _BW, _BH = 0.84, 0.88
    _CX, _CY = 0.50, 0.50
    _RX, _RY = 0.42, 0.44
    _PFC_Y_MAX = 0.64       # PFC 마스크 하단 경계

    VMIN, VMAX = -5.0, 5.0
    _OVERLAY_RES = 200      # interpolation grid 해상도

    def __init__(self, n_channels: int = 4, parent=None):
        super().__init__(parent)
        self._n_channels = n_channels
        self._values = np.zeros(n_channels)
        self._label = "HbO"
        self._bg_pixmap: QPixmap | None = None
        self._overlay_img: QImage | None = None
        self.setMinimumSize(400, 400)

    def set_values(self, values: np.ndarray, label: str) -> None:
        """신호 값 갱신 후 재렌더링."""
        self._values = values.copy()
        self._label = label
        self._overlay_img = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._bg_pixmap = None
        self._overlay_img = None
        super().resizeEvent(event)

    # ── 뇌 배경 ─────────────────────────────────────────────────────────────

    def _build_background(self, w: int, h: int) -> QPixmap:
        pm = QPixmap(w, h)
        pm.fill(QColor("#000000"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bx = int(w * self._BX)
        by = int(h * self._BY)
        bw = int(w * self._BW)
        bh = int(h * self._BH)
        cxp = bx + bw // 2
        cyp = by + bh // 2

        # 뇌 외곽 (방사형 그라디언트 회색)
        rg = QRadialGradient(cxp, cyp, max(bw, bh) // 2)
        rg.setColorAt(0.0,  QColor("#484848"))
        rg.setColorAt(0.55, QColor("#383838"))
        rg.setColorAt(0.85, QColor("#252525"))
        rg.setColorAt(1.0,  QColor("#141414"))
        p.setPen(QPen(QColor("#606060"), 2))
        p.setBrush(QBrush(rg))
        p.drawEllipse(bx, by, bw, bh)

        # 전전두엽 영역 파란 틴트
        pfc_h = int(bh * self._PFC_Y_MAX)
        pg2 = QLinearGradient(bx, by, bx, by + pfc_h)
        pg2.setColorAt(0.0, QColor(40, 60, 120, 65))
        pg2.setColorAt(1.0, QColor(40, 60, 120, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(pg2))
        p.drawEllipse(bx, by, bw, bh)

        # 대뇌종렬 (중앙 세로선)
        p.setPen(QPen(QColor("#1e1e1e"), 3))
        p.drawLine(cxp, by + 12, cxp, by + bh - 12)

        # 고랑(sulci) — 단순 호선
        p.setPen(QPen(QColor("#303030"), 1))
        sulci = [
            (0.28, 0.34, 160),
            (0.40, 0.36, 150),
            (0.53, 0.30, 140),
        ]
        for yf, wf, deg in sulci:
            yp = by + int(bh * yf)
            ww = int(bw * wf)
            p.drawArc(cxp - int(bw * 0.18) - ww // 2, yp - 7, ww, 14, 0, deg * 16)
            p.drawArc(cxp + int(bw * 0.18) - ww // 2, yp - 7, ww, 14, 0, deg * 16)

        # 채널 위치 마커
        for ch in range(min(self._n_channels, len(self._CH_POS))):
            fx, fy = self._CH_POS[ch]
            cx_ch = bx + int(fx * bw)
            cy_ch = by + int(fy * bh)
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.setBrush(QBrush(QColor(255, 255, 255, 80)))
            p.drawEllipse(cx_ch - 5, cy_ch - 5, 10, 10)
            p.setPen(QColor("#dddddd"))
            p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            p.drawText(cx_ch - 20, cy_ch - 19, 40, 14,
                       Qt.AlignmentFlag.AlignCenter, f"Ch{ch + 1}")

        # 레이블
        p.setPen(QColor("#777777"))
        p.setFont(QFont("Arial", 9))
        p.drawText(0, by - 20, w, 18, Qt.AlignmentFlag.AlignCenter,
                   "\u25b2  Anterior  (Prefrontal Cortex)")
        p.drawText(0, by + bh + 4, w, 18, Qt.AlignmentFlag.AlignCenter,
                   "\u25bc  Posterior")
        p.end()
        return pm

    # ── RBF 보간 오버레이 ────────────────────────────────────────────────────

    @staticmethod
    def _bwr(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """t in [0,1] -> (R,G,B) uint8. Blue(0) -> White(0.5) -> Red(1)."""
        t = np.clip(t, 0.0, 1.0)
        lo = t < 0.5
        s_lo = t * 2
        s_hi = (t - 0.5) * 2
        r = np.where(lo, s_lo * 255, 255).clip(0, 255).astype(np.uint8)
        g = np.where(lo, s_lo * 255, (1 - s_hi) * 255).clip(0, 255).astype(np.uint8)
        b = np.where(lo, 255, (1 - s_hi) * 255).clip(0, 255).astype(np.uint8)
        return r, g, b

    def _build_overlay(self) -> QImage:
        res = self._OVERLAY_RES
        xs = np.linspace(0, 1, res)
        ys = np.linspace(0, 1, res)
        xx, yy = np.meshgrid(xs, ys)

        # 뇌 타원 + PFC 마스크
        brain_mask = (
            ((xx - self._CX) / self._RX) ** 2 +
            ((yy - self._CY) / self._RY) ** 2
        ) <= 1.0
        pfc_mask = brain_mask & (yy <= self._PFC_Y_MAX)

        rows, cols = np.where(pfc_mask)
        pts = np.column_stack([xx[rows, cols], yy[rows, cols]])

        n = min(self._n_channels, len(self._CH_POS))
        values = self._values[:n]
        ch_pos = self._CH_POS[:n]

        # RBF 보간
        try:
            rbf = RBFInterpolator(ch_pos, values, kernel='gaussian', epsilon=6)
            interp = rbf(pts)
        except Exception:
            interp = np.zeros(len(pts))

        # Blue-White-Red 컬러맵
        t = (interp - self.VMIN) / (self.VMAX - self.VMIN)
        r_ch, g_ch, b_ch = self._bwr(t)

        # 알파: 뇌 경계·PFC 하단에서 페이드아웃
        dist_sq = (
            ((pts[:, 0] - self._CX) / self._RX) ** 2 +
            ((pts[:, 1] - self._CY) / self._RY) ** 2
        )
        y_fade = np.clip((self._PFC_Y_MAX - pts[:, 1]) / 0.12, 0.0, 1.0)
        alpha = (200 * (1 - dist_sq ** 0.55) * y_fade).clip(30, 200).astype(np.uint8)

        # RGBA 배열 조합
        rgba = np.zeros((res, res, 4), dtype=np.uint8)
        rgba[rows, cols, 0] = r_ch
        rgba[rows, cols, 1] = g_ch
        rgba[rows, cols, 2] = b_ch
        rgba[rows, cols, 3] = alpha

        img = QImage(rgba.tobytes(), res, res, res * 4,
                     QImage.Format.Format_RGBA8888)
        return img.copy()

    # ── paintEvent ──────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        if self._bg_pixmap is None:
            self._bg_pixmap = self._build_background(w, h)
        p.drawPixmap(0, 0, self._bg_pixmap)

        if self._overlay_img is None:
            self._overlay_img = self._build_overlay()

        bx = int(w * self._BX)
        by = int(h * self._BY)
        bw = int(w * self._BW)
        bh = int(h * self._BH)
        scaled = self._overlay_img.scaled(
            bw, bh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        p.drawImage(bx, by, scaled)

        self._draw_colorbar(p, w, h)
        p.end()

    def _draw_colorbar(self, p: QPainter, w: int, h: int) -> None:
        """Blue-White-Red 컬러바를 하단에 그린다."""
        cb_w, cb_h = 200, 14
        cb_x = (w - cb_w) // 2
        cb_y = h - 34

        grad = QLinearGradient(cb_x, cb_y, cb_x + cb_w, cb_y)
        grad.setColorAt(0.0, QColor("#0055ff"))
        grad.setColorAt(0.5, QColor("#ffffff"))
        grad.setColorAt(1.0, QColor("#ff2200"))
        p.setPen(QPen(QColor("#555555"), 1))
        p.setBrush(QBrush(grad))
        p.drawRect(cb_x, cb_y, cb_w, cb_h)

        p.setPen(QColor("#aaaaaa"))
        p.setFont(QFont("Arial", 8))
        p.drawText(cb_x - 54, cb_y, 52, cb_h,
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{self.VMIN:.0f} \u03bcmol/L")
        p.drawText(cb_x + cb_w + 4, cb_y, 56, cb_h,
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"+{self.VMAX:.0f} \u03bcmol/L")
        p.drawText(cb_x, cb_y, cb_w, cb_h, Qt.AlignmentFlag.AlignCenter, "0")

        p.setPen(QColor("#cccccc"))
        p.setFont(QFont("Arial", 9))
        p.drawText(0, cb_y - 20, w, 16, Qt.AlignmentFlag.AlignCenter,
                   f"{self._label}  \u00b7  Blue(\u2212)  White(0)  Red(+)")


class BrainMapWidget(QWidget):
    """전전두엽 채널 신호를 뇌 지도에 시각화. HbO / HbR 토글 포함."""

    def __init__(self, n_channels: int = 4, parent=None):
        super().__init__(parent)
        self._n_channels = n_channels
        self._hbo = np.zeros(n_channels)
        self._hbr = np.zeros(n_channels)
        self._show_hbo = True
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # HbO / HbR 토글
        toggle = QHBoxLayout()
        toggle.addStretch()
        self._hbo_btn = QPushButton("HbO")
        self._hbr_btn = QPushButton("HbR")
        for btn in (self._hbo_btn, self._hbr_btn):
            btn.setCheckable(True)
            btn.setFixedWidth(80)
        self._hbo_btn.setChecked(True)
        self._hbo_btn.clicked.connect(self._on_hbo)
        self._hbr_btn.clicked.connect(self._on_hbr)
        toggle.addWidget(self._hbo_btn)
        toggle.addWidget(self._hbr_btn)
        toggle.addStretch()
        layout.addLayout(toggle)

        self._canvas = _BrainCanvas(self._n_channels)
        layout.addWidget(self._canvas, stretch=1)

    def _on_hbo(self) -> None:
        self._show_hbo = True
        self._hbo_btn.setChecked(True)
        self._hbr_btn.setChecked(False)
        self._canvas.set_values(self._hbo, "HbO")

    def _on_hbr(self) -> None:
        self._show_hbo = False
        self._hbo_btn.setChecked(False)
        self._hbr_btn.setChecked(True)
        self._canvas.set_values(self._hbr, "HbR")

    def update_data(self, hbo: np.ndarray, hbr: np.ndarray) -> None:
        """ProcessingPipeline on_sample에서 호출. 현재 선택 신호로 캔버스 갱신."""
        self._hbo = hbo.copy()
        self._hbr = hbr.copy()
        values = self._hbo if self._show_hbo else self._hbr
        label  = "HbO"   if self._show_hbo else "HbR"
        self._canvas.set_values(values, label)
