import time
import pytest
from src.core.ring_buffer import RingBuffer
from src.acquisition.simulator import FNIRSSimulator
from src.acquisition.acquisition_thread import AcquisitionThread
from src.processing.concentration import SimpleHbOIndex
from src.processing.pipeline import ProcessingPipeline
from src.core.models import ProcessedSample


def test_pipeline_produces_processed_samples(app_config):
    # conftest app_config: sampling_rate_hz=100, baseline_window_sec=0.3
    # min_samples=31, wait 1.5s → ~150 packets → processing results available
    buffer = RingBuffer(capacity=500)
    device = FNIRSSimulator(app_config)
    acq = AcquisitionThread(device=device, buffer=buffer)
    index = SimpleHbOIndex()
    results: list[ProcessedSample] = []

    pipeline = ProcessingPipeline(
        buffer=buffer,
        config=app_config,
        concentration_index=index,
        on_sample=results.append,
    )

    acq.start()
    pipeline.start()
    time.sleep(1.5)
    acq.stop()
    pipeline.stop()
    acq.join(timeout=2.0)
    pipeline.join(timeout=2.0)

    assert len(results) > 0
    sample = results[-1]
    assert isinstance(sample, ProcessedSample)
    assert sample.hbo.shape == (4,)  # 4 channels
    assert sample.hbr.shape == (4,)
    assert 0.0 <= sample.concentration_index <= 1.0


def test_pipeline_stops_cleanly(app_config):
    buffer = RingBuffer(capacity=50)
    device = FNIRSSimulator(app_config)
    acq = AcquisitionThread(device=device, buffer=buffer)
    pipeline = ProcessingPipeline(
        buffer=buffer,
        config=app_config,
        concentration_index=SimpleHbOIndex(),
        on_sample=lambda s: None,
    )
    acq.start()
    pipeline.start()
    time.sleep(0.1)
    acq.stop()
    pipeline.stop()
    acq.join(timeout=2.0)
    pipeline.join(timeout=2.0)
    assert not pipeline.is_alive()


def test_pipeline_is_daemon(app_config):
    buffer = RingBuffer(capacity=10)
    pipeline = ProcessingPipeline(
        buffer=buffer,
        config=app_config,
        concentration_index=SimpleHbOIndex(),
        on_sample=lambda s: None,
    )
    assert pipeline.daemon is True
