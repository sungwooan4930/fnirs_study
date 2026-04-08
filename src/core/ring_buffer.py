from __future__ import annotations
import queue
from typing import Optional
from src.core.models import RawPacket


class RingBuffer:
    """thread-safe FIFO 버퍼. capacity 초과 시 가장 오래된 항목을 버린다."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._queue: queue.Queue[RawPacket] = queue.Queue(maxsize=capacity)

    def put(self, packet: RawPacket) -> None:
        """패킷을 버퍼에 넣는다. 가득 차면 가장 오래된 항목을 제거하고 삽입."""
        while True:
            try:
                self._queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()  # 가장 오래된 항목 제거
                except queue.Empty:
                    pass

    def get(self, timeout: float = 1.0) -> Optional[RawPacket]:
        """버퍼에서 패킷을 꺼낸다. timeout 초 안에 없으면 None 반환."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
