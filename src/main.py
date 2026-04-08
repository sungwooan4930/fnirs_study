"""fNIRS 집중도 분석 시스템 — 진입점."""
from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QMetaObject, Qt, Q_ARG
from src.core.config import AppConfig
from src.core.ring_buffer import RingBuffer
from src.acquisition.simulator import FNIRSSimulator
from src.acquisition.acquisition_thread import AcquisitionThread
from src.processing.concentration import SimpleHbOIndex
from src.processing.pipeline import ProcessingPipeline
from src.ui.main_window import MainWindow


def main() -> None:
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    config = AppConfig.from_yaml(config_path)

    app = QApplication(sys.argv)
    window = MainWindow(config)

    buffer = RingBuffer(capacity=1000)
    device = FNIRSSimulator(config)
    acq_thread = AcquisitionThread(device=device, buffer=buffer)
    pipeline = ProcessingPipeline(
        buffer=buffer,
        config=config,
        concentration_index=SimpleHbOIndex(),
        on_sample=lambda s: QMetaObject.invokeMethod(
            window, "update_sample", Qt.ConnectionType.QueuedConnection,
            Q_ARG(object, s),
        ),
    )

    def on_start() -> None:
        acq_thread.start()
        pipeline.start()
        window.start_session()

    def on_stop() -> None:
        acq_thread.stop()
        pipeline.stop()
        window.stop_session()

    window._start_btn.clicked.connect(on_start)
    window._stop_btn.clicked.connect(on_stop)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
