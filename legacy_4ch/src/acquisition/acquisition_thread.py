from __future__ import annotations
import threading
from src.core.hal import FNIRSDevice
from src.core.ring_buffer import RingBuffer


class AcquisitionThread(threading.Thread):
    """장치에서 패킷을 읽어 링 버퍼에 쓰는 스레드.

    stop()을 호출하면 현재 read_packet() 완료 후 종료된다.
    """

    def __init__(self, device: FNIRSDevice, buffer: RingBuffer) -> None:
        super().__init__(daemon=True)
        self._device = device
        self._buffer = buffer
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            self._device.connect()
            self._device.start_stream()
            while not self._stop_event.is_set():
                packet = self._device.read_packet()
                self._buffer.put(packet)
        finally:
            self._device.stop_stream()
            self._device.disconnect()

    def stop(self) -> None:
        self._stop_event.set()
