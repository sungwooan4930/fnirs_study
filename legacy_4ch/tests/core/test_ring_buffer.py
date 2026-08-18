import threading
import pytest
from src.core.ring_buffer import RingBuffer
from src.core.models import RawPacket


def make_packet(ts: float) -> RawPacket:
    return RawPacket(timestamp=ts, channel_intensities=[0.0] * 12, n_wavelengths=3)


def test_put_and_get_single_item():
    buf = RingBuffer(capacity=10)
    packet = make_packet(1.0)
    buf.put(packet)
    result = buf.get(timeout=1.0)
    assert result.timestamp == 1.0


def test_get_blocks_until_item_available():
    buf = RingBuffer(capacity=10)

    def producer():
        import time
        time.sleep(0.05)
        buf.put(make_packet(42.0))

    t = threading.Thread(target=producer)
    t.start()
    result = buf.get(timeout=1.0)
    t.join()
    assert result.timestamp == 42.0


def test_get_returns_none_on_timeout():
    buf = RingBuffer(capacity=10)
    result = buf.get(timeout=0.05)
    assert result is None


def test_overflow_drops_oldest():
    buf = RingBuffer(capacity=3)
    for i in range(5):
        buf.put(make_packet(float(i)))
    # capacity=3이므로 가장 오래된 2개는 버려짐
    collected = []
    while True:
        item = buf.get(timeout=0.01)
        if item is None:
            break
        collected.append(item.timestamp)
    assert len(collected) == 3
    assert collected == [2.0, 3.0, 4.0]


def test_thread_safe_concurrent_writes():
    buf = RingBuffer(capacity=100)
    n_threads = 10
    n_packets = 10

    def writer():
        for i in range(n_packets):
            buf.put(make_packet(float(i)))

    threads = [threading.Thread(target=writer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    count = 0
    while buf.get(timeout=0.01) is not None:
        count += 1
    assert count == n_threads * n_packets
