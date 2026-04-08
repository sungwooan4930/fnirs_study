import numpy as np
import pytest
from pathlib import Path
from src.core.models import ProcessedSample, RawPacket
from src.storage.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "test_session.h5")


def make_sample(ts: float) -> ProcessedSample:
    """4채널 ProcessedSample 생성."""
    return ProcessedSample(
        timestamp=ts,
        hbo=np.random.uniform(0, 1, size=4),
        hbr=np.random.uniform(-0.5, 0, size=4),
        concentration_index=0.5,
    )


def make_packet(ts: float) -> RawPacket:
    """4채널 × 3파장 = 12 강도값 RawPacket 생성."""
    return RawPacket(
        timestamp=ts,
        channel_intensities=[0.5] * 12,
        n_wavelengths=3,
    )


def test_store_and_load_processed_samples(store):
    samples = [make_sample(float(i)) for i in range(10)]
    store.open()
    for s in samples:
        store.write_processed(s)
    store.close()

    loaded = store.load_processed()
    assert len(loaded) == 10
    assert loaded[0].timestamp == 0.0
    assert loaded[-1].timestamp == 9.0
    assert loaded[0].hbo.shape == (4,)


def test_store_and_load_raw_packets(store):
    packets = [make_packet(float(i)) for i in range(5)]
    store.open()
    for p in packets:
        store.write_raw(p)
    store.close()

    loaded = store.load_raw()
    assert len(loaded) == 5
    assert loaded[2].timestamp == 2.0
    assert len(loaded[0].channel_intensities) == 12
    assert loaded[0].n_wavelengths == 3


def test_concentration_index_preserved(store):
    sample = make_sample(1.0)
    sample.concentration_index = 0.73
    store.open()
    store.write_processed(sample)
    store.close()

    loaded = store.load_processed()
    assert abs(loaded[0].concentration_index - 0.73) < 1e-6


def test_write_without_open_raises(store):
    packet = make_packet(0.0)
    with pytest.raises(RuntimeError, match="open"):
        store.write_raw(packet)


def test_write_processed_without_open_raises(store):
    sample = make_sample(0.0)
    with pytest.raises(RuntimeError, match="open"):
        store.write_processed(sample)
