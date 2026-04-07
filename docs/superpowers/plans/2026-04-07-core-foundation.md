# fNIRS Core Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 하드웨어 없이 전체 fNIRS 신호 파이프라인(수집 → 처리 → 시각화)을 구동할 수 있는 Core Foundation을 구축한다.

**Architecture:** Simulation-First + HAL 패턴. FNIRSDevice ABC를 중심으로 Simulator와 Real Device Driver가 동일한 인터페이스를 구현한다. Acquisition Thread → Ring Buffer → Processing Thread → Qt Signal → GUI의 단방향 데이터 흐름을 따른다.

**Tech Stack:** Python 3.10+, PySide6, PyQtGraph, NumPy, SciPy, h5py, pytest, pytest-qt, ruff, black

**범위 외 (별도 계획):**
- Report Layer (HDF5 → PDF/HTML 레포트)
- Real Device Driver (하드웨어 스펙 확정 후)
- 모바일/웹 UI (플랫폼 확정 후)

---

## Task 0: 개발 환경 세팅

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`

- [ ] **Step 1: Python 버전 확인**

```bash
python --version
```

Expected: `Python 3.10.x` 이상. 아니라면 python.org에서 3.10+ 설치.

- [ ] **Step 2: 가상환경 생성 및 활성화**

```bash
cd D:/Study_fNIRS
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Expected: 프롬프트 앞에 `(.venv)` 표시.

- [ ] **Step 3: requirements.txt 작성**

```
numpy>=1.26.0
scipy>=1.12.0
PySide6>=6.6.0
pyqtgraph>=0.13.3
h5py>=3.10.0
PyYAML>=6.0.1
```

- [ ] **Step 4: requirements-dev.txt 작성**

```
pytest>=8.0.0
pytest-qt>=4.3.1
ruff>=0.3.0
black>=24.0.0
```

- [ ] **Step 5: 패키지 설치**

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Expected: 에러 없이 설치 완료.

- [ ] **Step 6: .gitignore 작성**

```
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
data/
reports/
.superpowers/
*.h5
*.egg-info/
dist/
build/
```

- [ ] **Step 7: git 초기화 및 첫 커밋**

```bash
git init
git add requirements.txt requirements-dev.txt .gitignore PROJECT_GUIDELINES.md run_logging.md docs/
git commit -m "[env] 개발 환경 초기 세팅"
```

---

## Task 1: 프로젝트 골격 & 설정 파일

**Files:**
- Create: `config/settings.yaml`
- Create: `src/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/core/config.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_config.py`

- [ ] **Step 1: 디렉토리 구조 생성**

```bash
mkdir -p src/core src/acquisition src/processing src/storage src/ui
mkdir -p tests/core tests/acquisition tests/processing tests/storage
mkdir -p config data reports
touch src/__init__.py src/core/__init__.py src/acquisition/__init__.py
touch src/processing/__init__.py src/storage/__init__.py src/ui/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/acquisition/__init__.py
touch tests/processing/__init__.py tests/storage/__init__.py
```

- [ ] **Step 2: settings.yaml 작성 (placeholder 값으로)**

`config/settings.yaml`:
```yaml
device:
  # 하드웨어 담당자와 확정 필요 — 현재 모두 placeholder
  wavelengths_nm: [735, 850]       # 광원 파장 2개 (nm)
  sampling_rate_hz: 10             # 샘플링 레이트
  n_channels: 8                    # 채널 수
  sds_mm: 30.0                     # 소스-디텍터 간격 (mm)
  # 채널별 [소스 인덱스, 디텍터 인덱스] — placeholder
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
    - [2, 1]
    - [2, 2]
    - [3, 1]
    - [3, 2]
    - [4, 1]
    - [4, 2]

processing:
  bandpass_low_hz: 0.01            # 밴드패스 하한
  bandpass_high_hz: 0.5            # 밴드패스 상한
  baseline_window_sec: 5.0         # 베이스라인 구간 (초)
  # 몰흡광계수 (L/(mmol·cm)) — 735nm, 850nm placeholder
  extinction_coefficients:
    hbo: [1.4067, 0.9012]          # 파장별 HbO 흡광계수
    hbr: [3.7216, 0.7234]          # 파장별 HbR 흡광계수
  dpf: [6.51, 5.86]               # 차분 경로 인수 (파장별)

storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"

simulator:
  hbo_amplitude: 0.5               # μmol/L
  hbo_freq_hz: 0.1                 # 집중도 변화 주파수
  noise_std: 0.05                  # 노이즈 표준편차
```

- [ ] **Step 3: 테스트 먼저 작성**

`tests/core/test_config.py`:
```python
import pytest
from pathlib import Path
from src.core.config import AppConfig


def test_config_loads_from_yaml(tmp_path):
    yaml_content = """
device:
  wavelengths_nm: [735, 850]
  sampling_rate_hz: 10
  n_channels: 8
  sds_mm: 30.0
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
processing:
  bandpass_low_hz: 0.01
  bandpass_high_hz: 0.5
  baseline_window_sec: 5.0
  extinction_coefficients:
    hbo: [1.4067, 0.9012]
    hbr: [3.7216, 0.7234]
  dpf: [6.51, 5.86]
storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"
simulator:
  hbo_amplitude: 0.5
  hbo_freq_hz: 0.1
  noise_std: 0.05
"""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml_content)

    config = AppConfig.from_yaml(config_file)

    assert config.device.wavelengths_nm == [735, 850]
    assert config.device.sampling_rate_hz == 10
    assert config.device.n_channels == 8
    assert config.device.sds_mm == 30.0
    assert len(config.device.source_detector_pairs) == 2
    assert config.processing.bandpass_low_hz == 0.01
    assert config.processing.extinction_coefficients["hbo"] == [1.4067, 0.9012]
    assert config.simulator.noise_std == 0.05


def test_config_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        AppConfig.from_yaml(Path("nonexistent.yaml"))
```

- [ ] **Step 4: 테스트 실행 → 실패 확인**

```bash
pytest tests/core/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.config'`

- [ ] **Step 5: config.py 구현**

`src/core/config.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DeviceConfig:
    wavelengths_nm: list[int]
    sampling_rate_hz: float
    n_channels: int
    sds_mm: float
    source_detector_pairs: list[list[int]]


@dataclass
class ProcessingConfig:
    bandpass_low_hz: float
    bandpass_high_hz: float
    baseline_window_sec: float
    extinction_coefficients: dict[str, list[float]]
    dpf: list[float]


@dataclass
class StorageConfig:
    data_dir: str
    session_filename_format: str


@dataclass
class SimulatorConfig:
    hbo_amplitude: float
    hbo_freq_hz: float
    noise_std: float


@dataclass
class AppConfig:
    device: DeviceConfig
    processing: ProcessingConfig
    storage: StorageConfig
    simulator: SimulatorConfig

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            device=DeviceConfig(**data["device"]),
            processing=ProcessingConfig(**data["processing"]),
            storage=StorageConfig(**data["storage"]),
            simulator=SimulatorConfig(**data["simulator"]),
        )
```

- [ ] **Step 6: 테스트 실행 → 통과 확인**

```bash
pytest tests/core/test_config.py -v
```

Expected: `2 passed`

- [ ] **Step 7: 커밋**

```bash
git add config/ src/core/config.py tests/core/test_config.py src/ tests/
git commit -m "[core] 프로젝트 골격 및 AppConfig 구현"
```

---

## Task 2: 핵심 데이터 모델

**Files:**
- Create: `src/core/models.py`
- Create: `tests/core/test_models.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/core/test_models.py`:
```python
import numpy as np
import pytest
from src.core.models import RawPacket, ProcessedSample


def test_raw_packet_construction():
    packet = RawPacket(
        timestamp=1000.0,
        channel_intensities=[0.5] * 16,  # 8채널 × 2파장
    )
    assert packet.timestamp == 1000.0
    assert len(packet.channel_intensities) == 16


def test_raw_packet_intensity_by_channel():
    # 채널 0: [wl0=0.1, wl1=0.2], 채널 1: [wl0=0.3, wl1=0.4], ...
    intensities = [0.1, 0.2, 0.3, 0.4] + [0.5] * 12
    packet = RawPacket(timestamp=0.0, channel_intensities=intensities)
    ch0 = packet.intensity_by_channel(channel=0)
    assert ch0 == [0.1, 0.2]
    ch1 = packet.intensity_by_channel(channel=1)
    assert ch1 == [0.3, 0.4]


def test_processed_sample_construction():
    hbo = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    hbr = np.array([-0.05] * 8)
    sample = ProcessedSample(
        timestamp=1000.0,
        hbo=hbo,
        hbr=hbr,
        concentration_index=0.65,
    )
    assert sample.concentration_index == 0.65
    assert sample.hbo.shape == (8,)


def test_processed_sample_validates_channel_count():
    with pytest.raises(ValueError, match="채널 수"):
        ProcessedSample(
            timestamp=0.0,
            hbo=np.zeros(5),   # 잘못된 채널 수
            hbr=np.zeros(8),
            concentration_index=0.0,
        )
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/core/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.models'`

- [ ] **Step 3: models.py 구현**

`src/core/models.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RawPacket:
    """장치에서 수신한 원시 패킷. n_channels × n_wavelengths 강도값을 담는다."""
    timestamp: float
    channel_intensities: list[float]  # [ch0_wl0, ch0_wl1, ch1_wl0, ch1_wl1, ...]

    def intensity_by_channel(self, channel: int) -> list[float]:
        """채널 인덱스로 해당 채널의 모든 파장 강도값을 반환한다."""
        n_wavelengths = 2  # 파장 수는 항상 2 (HW 스펙 확정 후 config 연동)
        start = channel * n_wavelengths
        return self.channel_intensities[start : start + n_wavelengths]


@dataclass
class ProcessedSample:
    """처리된 단일 시간점 데이터."""
    timestamp: float
    hbo: np.ndarray   # shape: (n_channels,), μmol/L
    hbr: np.ndarray   # shape: (n_channels,), μmol/L
    concentration_index: float  # [0.0, 1.0]

    def __post_init__(self) -> None:
        if self.hbo.shape != self.hbr.shape:
            raise ValueError(f"채널 수 불일치: hbo={self.hbo.shape}, hbr={self.hbr.shape}")
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/core/test_models.py -v
```

Expected: `4 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/core/models.py tests/core/test_models.py
git commit -m "[core] RawPacket, ProcessedSample 데이터 모델 구현"
```

---

## Task 3: HAL 인터페이스

**Files:**
- Create: `src/core/hal.py`
- Create: `tests/core/test_hal.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/core/test_hal.py`:
```python
import pytest
from src.core.hal import FNIRSDevice
from src.core.models import RawPacket


def test_hal_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        FNIRSDevice()


def test_concrete_device_must_implement_all_methods():
    """모든 메서드를 구현하지 않은 서브클래스는 인스턴스 생성 불가."""
    class IncompleteDevice(FNIRSDevice):
        def connect(self) -> bool:
            return True
        # start_stream, stop_stream, read_packet, disconnect 미구현

    with pytest.raises(TypeError):
        IncompleteDevice()


def test_concrete_device_can_be_instantiated_when_complete():
    class MinimalDevice(FNIRSDevice):
        def connect(self) -> bool:
            return True
        def start_stream(self) -> None:
            pass
        def stop_stream(self) -> None:
            pass
        def read_packet(self) -> RawPacket:
            return RawPacket(timestamp=0.0, channel_intensities=[0.0] * 16)
        def disconnect(self) -> None:
            pass

    device = MinimalDevice()
    assert device.connect() is True
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/core/test_hal.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.hal'`

- [ ] **Step 3: hal.py 구현**

`src/core/hal.py`:
```python
from abc import ABC, abstractmethod
from src.core.models import RawPacket


class FNIRSDevice(ABC):
    """fNIRS 장치 추상 인터페이스.

    Real device driver와 Simulator 모두 이 클래스를 상속해 구현한다.
    상위 레이어(Acquisition Layer)는 이 인터페이스만 알면 된다.
    """

    @abstractmethod
    def connect(self) -> bool:
        """장치에 연결한다. 성공 시 True 반환."""
        ...

    @abstractmethod
    def start_stream(self) -> None:
        """데이터 스트리밍을 시작한다."""
        ...

    @abstractmethod
    def stop_stream(self) -> None:
        """데이터 스트리밍을 중지한다."""
        ...

    @abstractmethod
    def read_packet(self) -> RawPacket:
        """다음 패킷을 반환한다. 블로킹 호출."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """장치 연결을 해제한다."""
        ...
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/core/test_hal.py -v
```

Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/core/hal.py tests/core/test_hal.py
git commit -m "[core] FNIRSDevice HAL 인터페이스 구현"
```

---

## Task 4: Thread-safe 링 버퍼

**Files:**
- Create: `src/core/ring_buffer.py`
- Create: `tests/core/test_ring_buffer.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/core/test_ring_buffer.py`:
```python
import threading
import pytest
from src.core.ring_buffer import RingBuffer
from src.core.models import RawPacket


def make_packet(ts: float) -> RawPacket:
    return RawPacket(timestamp=ts, channel_intensities=[0.0] * 16)


def test_put_and_get_single_item():
    buf = RingBuffer(capacity=10)
    packet = make_packet(1.0)
    buf.put(packet)
    result = buf.get(timeout=1.0)
    assert result.timestamp == 1.0


def test_get_blocks_until_item_available():
    buf = RingBuffer(capacity=10)
    results = []

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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/core/test_ring_buffer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.core.ring_buffer'`

- [ ] **Step 3: ring_buffer.py 구현**

`src/core/ring_buffer.py`:
```python
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/core/test_ring_buffer.py -v
```

Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/core/ring_buffer.py tests/core/test_ring_buffer.py
git commit -m "[core] thread-safe RingBuffer 구현"
```

---

## Task 5: fNIRS 시뮬레이터

**Files:**
- Create: `src/acquisition/__init__.py` (이미 있음)
- Create: `src/acquisition/simulator.py`
- Create: `tests/acquisition/__init__.py`
- Create: `tests/acquisition/test_simulator.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/acquisition/test_simulator.py`:
```python
import time
import pytest
from src.core.config import AppConfig
from src.acquisition.simulator import FNIRSSimulator


@pytest.fixture
def config(tmp_path):
    yaml_content = """
device:
  wavelengths_nm: [735, 850]
  sampling_rate_hz: 100
  n_channels: 8
  sds_mm: 30.0
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
    - [2, 1]
    - [2, 2]
    - [3, 1]
    - [3, 2]
    - [4, 1]
    - [4, 2]
processing:
  bandpass_low_hz: 0.01
  bandpass_high_hz: 0.5
  baseline_window_sec: 5.0
  extinction_coefficients:
    hbo: [1.4067, 0.9012]
    hbr: [3.7216, 0.7234]
  dpf: [6.51, 5.86]
storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"
simulator:
  hbo_amplitude: 0.5
  hbo_freq_hz: 0.1
  noise_std: 0.05
"""
    f = tmp_path / "settings.yaml"
    f.write_text(yaml_content)
    return AppConfig.from_yaml(f)


def test_simulator_implements_hal(config):
    from src.core.hal import FNIRSDevice
    sim = FNIRSSimulator(config)
    assert isinstance(sim, FNIRSDevice)


def test_simulator_connect_returns_true(config):
    sim = FNIRSSimulator(config)
    assert sim.connect() is True


def test_simulator_read_packet_returns_raw_packet(config):
    from src.core.models import RawPacket
    sim = FNIRSSimulator(config)
    sim.connect()
    sim.start_stream()
    packet = sim.read_packet()
    assert isinstance(packet, RawPacket)
    # 8채널 × 2파장 = 16개의 강도값
    assert len(packet.channel_intensities) == 16
    sim.stop_stream()
    sim.disconnect()


def test_simulator_packet_timestamp_increases(config):
    sim = FNIRSSimulator(config)
    sim.connect()
    sim.start_stream()
    p1 = sim.read_packet()
    p2 = sim.read_packet()
    assert p2.timestamp > p1.timestamp
    sim.stop_stream()
    sim.disconnect()


def test_simulator_respects_sampling_rate(config):
    """100Hz 설정에서 10개 패킷 수집 시 약 0.1초 걸려야 한다."""
    sim = FNIRSSimulator(config)
    sim.connect()
    sim.start_stream()
    start = time.perf_counter()
    for _ in range(10):
        sim.read_packet()
    elapsed = time.perf_counter() - start
    sim.stop_stream()
    sim.disconnect()
    assert 0.05 < elapsed < 0.5  # 10% 오차 허용
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/acquisition/test_simulator.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.acquisition.simulator'`

- [ ] **Step 3: simulator.py 구현**

`src/acquisition/simulator.py`:
```python
from __future__ import annotations
import time
import math
import random
from src.core.hal import FNIRSDevice
from src.core.models import RawPacket
from src.core.config import AppConfig


class FNIRSSimulator(FNIRSDevice):
    """fNIRS 신호 시뮬레이터.

    sin파 + 노이즈로 현실적인 fNIRS 신호를 생성한다.
    HbO는 저주파 sin파 형태, HbR는 역방향으로 약하게 변한다.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._streaming = False
        self._start_time: float = 0.0
        self._sample_index: int = 0

    def connect(self) -> bool:
        return True

    def start_stream(self) -> None:
        self._streaming = True
        self._start_time = time.perf_counter()
        self._sample_index = 0

    def stop_stream(self) -> None:
        self._streaming = False

    def disconnect(self) -> None:
        pass

    def read_packet(self) -> RawPacket:
        """다음 샘플 타이밍까지 대기 후 패킷 반환."""
        cfg_d = self._config.device
        cfg_s = self._config.simulator
        sr = cfg_d.sampling_rate_hz
        n_ch = cfg_d.n_channels

        # 샘플 타이밍 대기
        target_time = self._start_time + self._sample_index / sr
        now = time.perf_counter()
        if target_time > now:
            time.sleep(target_time - now)

        t = self._sample_index / sr
        self._sample_index += 1
        timestamp = time.perf_counter()

        # 신호 생성: [ch0_wl0, ch0_wl1, ch1_wl0, ch1_wl1, ...]
        intensities: list[float] = []
        for ch in range(n_ch):
            phase_offset = ch * 0.3  # 채널별 위상 차이
            # 파장 0 (예: 735nm): HbR에 더 민감
            hbr_component = 0.3 * math.sin(2 * math.pi * cfg_s.hbo_freq_hz * t + phase_offset)
            wl0 = 1.0 - hbr_component + random.gauss(0, cfg_s.noise_std)
            # 파장 1 (예: 850nm): HbO에 더 민감
            hbo_component = cfg_s.hbo_amplitude * math.sin(2 * math.pi * cfg_s.hbo_freq_hz * t + phase_offset)
            wl1 = 1.0 + hbo_component + random.gauss(0, cfg_s.noise_std)
            intensities.extend([max(0.0, wl0), max(0.0, wl1)])

        return RawPacket(timestamp=timestamp, channel_intensities=intensities)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/acquisition/test_simulator.py -v
```

Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/acquisition/simulator.py tests/acquisition/test_simulator.py
git commit -m "[acquisition] FNIRSSimulator 구현 (HAL 인터페이스 준수)"
```

---

## Task 6: 신호 필터

**Files:**
- Create: `src/processing/filters.py`
- Create: `tests/processing/test_filters.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/processing/test_filters.py`:
```python
import numpy as np
import pytest
from src.processing.filters import bandpass_filter, baseline_correct


def test_bandpass_removes_dc():
    """DC 성분(0Hz)을 포함한 신호에서 밴드패스 후 DC가 제거되어야 한다."""
    sr = 100.0
    t = np.arange(0, 10, 1 / sr)
    # DC + 0.1Hz 신호
    signal = 5.0 + np.sin(2 * np.pi * 0.1 * t)
    filtered = bandpass_filter(signal, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    # DC가 제거되면 평균이 0에 가까워짐
    assert abs(np.mean(filtered)) < 0.5


def test_bandpass_preserves_in_band_signal():
    """밴드 내 신호(0.1Hz)는 필터 후에도 보존되어야 한다."""
    sr = 100.0
    t = np.arange(0, 30, 1 / sr)
    signal = np.sin(2 * np.pi * 0.1 * t)
    filtered = bandpass_filter(signal, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    # 신호 에너지 보존 확인 (끝부분 사용, 필터 엣지 효과 제외)
    assert np.std(filtered[500:]) > 0.3


def test_bandpass_attenuates_high_freq():
    """밴드 외 고주파(10Hz)는 크게 감쇠되어야 한다."""
    sr = 100.0
    t = np.arange(0, 10, 1 / sr)
    signal = np.sin(2 * np.pi * 10.0 * t)
    filtered = bandpass_filter(signal, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    assert np.std(filtered) < 0.1


def test_baseline_correct_removes_mean():
    """베이스라인 보정 후 초반 구간 평균이 0에 가까워야 한다."""
    sr = 10.0
    baseline_sec = 5.0
    signal = np.ones(100) * 3.0  # 상수 신호 (베이스라인 = 3.0)
    corrected = baseline_correct(signal, baseline_sec=baseline_sec, sampling_rate_hz=sr)
    assert abs(np.mean(corrected)) < 0.01


def test_filters_work_on_2d_array():
    """여러 채널(2D 배열)에 대해 필터가 동작해야 한다."""
    sr = 100.0
    t = np.arange(0, 10, 1 / sr)
    signals = np.stack([np.sin(2 * np.pi * 0.1 * t) for _ in range(8)])  # (8, 1000)
    filtered = bandpass_filter(signals, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    assert filtered.shape == signals.shape
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/processing/test_filters.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.processing.filters'`

- [ ] **Step 3: filters.py 구현**

`src/processing/filters.py`:
```python
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth 밴드패스 필터. 1D 또는 2D (n_channels, n_samples) 배열 지원."""
    nyq = sampling_rate_hz / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    if data.ndim == 1:
        return sp_signal.sosfiltfilt(sos, data)
    # 2D: 각 채널(행)에 독립적으로 적용
    return np.apply_along_axis(
        lambda ch: sp_signal.sosfiltfilt(sos, ch), axis=1, arr=data
    )


def baseline_correct(
    data: np.ndarray,
    baseline_sec: float,
    sampling_rate_hz: float,
) -> np.ndarray:
    """초반 baseline_sec 구간의 평균을 빼서 베이스라인을 보정한다."""
    n_baseline = int(baseline_sec * sampling_rate_hz)
    if data.ndim == 1:
        baseline_mean = np.mean(data[:n_baseline])
        return data - baseline_mean
    baseline_mean = np.mean(data[:, :n_baseline], axis=1, keepdims=True)
    return data - baseline_mean
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/processing/test_filters.py -v
```

Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/processing/filters.py tests/processing/test_filters.py src/processing/__init__.py tests/processing/__init__.py
git commit -m "[processing] 밴드패스 필터 및 베이스라인 보정 구현"
```

---

## Task 7: 수정된 Beer-Lambert Law (mBLL)

**Files:**
- Create: `src/processing/mbll.py`
- Create: `tests/processing/test_mbll.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/processing/test_mbll.py`:
```python
import numpy as np
import pytest
from src.processing.mbll import modified_beer_lambert


def test_mbll_output_shape():
    """8채널, 100샘플 입력 → HbO/HbR 각각 (8, 100) 출력."""
    raw = np.random.uniform(0.8, 1.2, size=(8, 2, 100))  # (ch, wl, t)
    wavelengths = [735, 850]
    ext_hbo = [1.4067, 0.9012]
    ext_hbr = [3.7216, 0.7234]
    dpf = [6.51, 5.86]
    sds_mm = 30.0

    hbo, hbr = modified_beer_lambert(raw, wavelengths, ext_hbo, ext_hbr, dpf, sds_mm)

    assert hbo.shape == (8, 100)
    assert hbr.shape == (8, 100)


def test_mbll_returns_floats():
    raw = np.ones((8, 2, 50))
    hbo, hbr = modified_beer_lambert(
        raw, [735, 850], [1.4067, 0.9012], [3.7216, 0.7234], [6.51, 5.86], 30.0
    )
    assert hbo.dtype == np.float64
    assert hbr.dtype == np.float64


def test_mbll_zero_change_gives_near_zero_concentration():
    """강도 변화가 없으면 (모든 값 동일) HbO/HbR 변화량이 0에 가까워야 한다."""
    raw = np.ones((8, 2, 100))  # 변화 없는 신호
    hbo, hbr = modified_beer_lambert(
        raw, [735, 850], [1.4067, 0.9012], [3.7216, 0.7234], [6.51, 5.86], 30.0
    )
    assert np.allclose(hbo, 0.0, atol=1e-10)
    assert np.allclose(hbr, 0.0, atol=1e-10)
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/processing/test_mbll.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.processing.mbll'`

- [ ] **Step 3: mbll.py 구현**

`src/processing/mbll.py`:
```python
from __future__ import annotations
import numpy as np


def modified_beer_lambert(
    raw_intensity: np.ndarray,
    wavelengths_nm: list[int],
    ext_hbo: list[float],
    ext_hbr: list[float],
    dpf: list[float],
    sds_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """수정된 Beer-Lambert Law로 raw intensity → HbO/HbR 농도 변화 계산.

    Args:
        raw_intensity: shape (n_channels, n_wavelengths, n_samples), 원시 광강도
        wavelengths_nm: 파장 목록 (nm)
        ext_hbo: 파장별 HbO 몰흡광계수 (L/(mmol·cm))
        ext_hbr: 파장별 HbR 몰흡광계수 (L/(mmol·cm))
        dpf: 파장별 차분 경로 인수
        sds_mm: 소스-디텍터 간격 (mm)

    Returns:
        hbo: shape (n_channels, n_samples), μmol/L
        hbr: shape (n_channels, n_samples), μmol/L
    """
    n_channels, n_wavelengths, n_samples = raw_intensity.shape
    sds_cm = sds_mm / 10.0

    # 광학 밀도 변화: ΔOD = -log(I / I0), I0 = 첫 번째 샘플
    i0 = raw_intensity[:, :, :1]  # (ch, wl, 1)
    # 0으로 나누기 방지
    safe_raw = np.where(raw_intensity == 0, 1e-10, raw_intensity)
    safe_i0 = np.where(i0 == 0, 1e-10, i0)
    delta_od = -np.log(safe_raw / safe_i0)  # (ch, wl, t)

    # 역행렬 방법으로 HbO/HbR 계산
    # A * [ΔHbO, ΔHbR]^T = ΔOD / (DPF * SDS)
    # A = [[ε_HbO_λ1, ε_HbR_λ1], [ε_HbO_λ2, ε_HbR_λ2]]
    A = np.array([[ext_hbo[i], ext_hbr[i]] for i in range(n_wavelengths)])  # (wl, 2)
    A_inv = np.linalg.pinv(A)  # (2, wl)

    hbo_list, hbr_list = [], []
    for ch in range(n_channels):
        # (wl, t) / (wl, 1) → element-wise
        dpf_arr = np.array(dpf).reshape(-1, 1)
        od_normalized = delta_od[ch] / (dpf_arr * sds_cm)  # (wl, t)
        concentrations = A_inv @ od_normalized  # (2, t): [ΔHbO, ΔHbR]
        # mmol/L → μmol/L
        hbo_list.append(concentrations[0] * 1000)
        hbr_list.append(concentrations[1] * 1000)

    return np.array(hbo_list), np.array(hbr_list)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/processing/test_mbll.py -v
```

Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/processing/mbll.py tests/processing/test_mbll.py
git commit -m "[processing] 수정된 Beer-Lambert Law (mBLL) 구현"
```

---

## Task 8: 집중도 지수 플러그인

**Files:**
- Create: `src/processing/concentration.py`
- Create: `tests/processing/test_concentration.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/processing/test_concentration.py`:
```python
import numpy as np
import pytest
from src.processing.concentration import ConcentrationIndex, SimpleHbOIndex


def test_concentration_index_is_abstract():
    with pytest.raises(TypeError):
        ConcentrationIndex()


def test_simple_hbo_index_returns_float():
    idx = SimpleHbOIndex()
    hbo = np.array([0.5, 0.3, 0.1, 0.4, 0.6, 0.2, 0.3, 0.5])
    hbr = np.array([-0.1, -0.05, 0.0, -0.2, -0.1, -0.05, 0.0, -0.1])
    result = idx.compute(hbo, hbr)
    assert isinstance(result, float)


def test_simple_hbo_index_range():
    """결과값은 [0.0, 1.0] 범위 이내여야 한다."""
    idx = SimpleHbOIndex()
    hbo = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    hbr = np.array([-1.0] * 8)
    result = idx.compute(hbo, hbr)
    assert 0.0 <= result <= 1.0


def test_simple_hbo_high_hbo_gives_high_index():
    """HbO가 높을수록 집중도 지수가 높아야 한다."""
    idx = SimpleHbOIndex()
    hbo_high = np.array([2.0] * 8)
    hbo_low = np.array([0.1] * 8)
    hbr = np.zeros(8)
    assert idx.compute(hbo_high, hbr) > idx.compute(hbo_low, hbr)


def test_plugin_is_swappable():
    """ConcentrationIndex를 상속하면 compute 메서드를 통해 동일하게 사용 가능."""
    class CustomIndex(ConcentrationIndex):
        def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float:
            return 0.42

    idx: ConcentrationIndex = CustomIndex()
    result = idx.compute(np.zeros(8), np.zeros(8))
    assert result == 0.42
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/processing/test_concentration.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.processing.concentration'`

- [ ] **Step 3: concentration.py 구현**

`src/processing/concentration.py`:
```python
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ConcentrationIndex(ABC):
    """집중도 지수 계산 플러그인 인터페이스.

    논문 기반 알고리즘이 확정되면 이 클래스를 상속해 구현한다.
    """

    @abstractmethod
    def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float:
        """HbO/HbR 배열로부터 집중도 지수 [0.0, 1.0]을 계산한다.

        Args:
            hbo: shape (n_channels,), 현재 시점 HbO 값 (μmol/L)
            hbr: shape (n_channels,), 현재 시점 HbR 값 (μmol/L)

        Returns:
            집중도 지수 [0.0, 1.0]
        """
        ...


class SimpleHbOIndex(ConcentrationIndex):
    """단순 HbO 평균 기반 집중도 지수 (placeholder).

    전전두엽 HbO 증가 = 집중도 증가 가정.
    논문 기반 알고리즘 확정 후 교체 예정.
    """

    _MAX_HBO_UMOL = 10.0  # 정규화 기준 (μmol/L)

    def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float:
        mean_hbo = float(np.mean(hbo))
        normalized = mean_hbo / self._MAX_HBO_UMOL
        return float(np.clip(normalized, 0.0, 1.0))
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/processing/test_concentration.py -v
```

Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/processing/concentration.py tests/processing/test_concentration.py
git commit -m "[processing] ConcentrationIndex 플러그인 구조 + SimpleHbOIndex 구현"
```

---

## Task 9: Acquisition Thread

**Files:**
- Create: `src/acquisition/acquisition_thread.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: conftest.py 작성 (공통 fixture)**

`tests/conftest.py`:
```python
import pytest
from pathlib import Path
from src.core.config import AppConfig


@pytest.fixture
def app_config(tmp_path) -> AppConfig:
    # 테스트용: sampling_rate_hz=100, baseline_window_sec=0.3
    # → min_samples = int(0.3*100)+1 = 31, 1.5초 대기 시 ~150패킷 수집 가능
    yaml_content = """
device:
  wavelengths_nm: [735, 850]
  sampling_rate_hz: 100
  n_channels: 8
  sds_mm: 30.0
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
    - [2, 1]
    - [2, 2]
    - [3, 1]
    - [3, 2]
    - [4, 1]
    - [4, 2]
processing:
  bandpass_low_hz: 0.01
  bandpass_high_hz: 0.5
  baseline_window_sec: 0.3
  extinction_coefficients:
    hbo: [1.4067, 0.9012]
    hbr: [3.7216, 0.7234]
  dpf: [6.51, 5.86]
storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"
simulator:
  hbo_amplitude: 0.5
  hbo_freq_hz: 0.1
  noise_std: 0.05
"""
    f = tmp_path / "settings.yaml"
    f.write_text(yaml_content)
    return AppConfig.from_yaml(f)
```

- [ ] **Step 2: 테스트 먼저 작성**

`tests/acquisition/test_acquisition_thread.py`:
```python
import time
import pytest
from src.core.ring_buffer import RingBuffer
from src.acquisition.simulator import FNIRSSimulator
from src.acquisition.acquisition_thread import AcquisitionThread


def test_acquisition_thread_fills_buffer(app_config):
    device = FNIRSSimulator(app_config)
    buffer = RingBuffer(capacity=50)
    thread = AcquisitionThread(device=device, buffer=buffer)

    thread.start()
    time.sleep(0.15)  # 100Hz × 0.15s ≈ 15개 패킷
    thread.stop()
    thread.join(timeout=2.0)

    collected = []
    while True:
        p = buffer.get(timeout=0.01)
        if p is None:
            break
        collected.append(p)

    assert len(collected) >= 5  # 최소 5개 수집


def test_acquisition_thread_stops_cleanly(app_config):
    device = FNIRSSimulator(app_config)
    buffer = RingBuffer(capacity=50)
    thread = AcquisitionThread(device=device, buffer=buffer)

    thread.start()
    time.sleep(0.05)
    thread.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
```

- [ ] **Step 3: 테스트 실행 → 실패 확인**

```bash
pytest tests/acquisition/test_acquisition_thread.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.acquisition.acquisition_thread'`

- [ ] **Step 4: acquisition_thread.py 구현**

`src/acquisition/acquisition_thread.py`:
```python
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
        self._device.connect()
        self._device.start_stream()
        try:
            while not self._stop_event.is_set():
                packet = self._device.read_packet()
                self._buffer.put(packet)
        finally:
            self._device.stop_stream()
            self._device.disconnect()

    def stop(self) -> None:
        self._stop_event.set()
```

- [ ] **Step 5: 테스트 실행 → 통과 확인**

```bash
pytest tests/acquisition/test_acquisition_thread.py -v
```

Expected: `2 passed`

- [ ] **Step 6: 커밋**

```bash
git add src/acquisition/acquisition_thread.py tests/acquisition/test_acquisition_thread.py tests/conftest.py
git commit -m "[acquisition] AcquisitionThread 구현 (simulator → ring buffer)"
```

---

## Task 10: Processing Pipeline

**Files:**
- Create: `src/processing/pipeline.py`
- Create: `tests/processing/test_pipeline.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/processing/test_pipeline.py`:
```python
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
    # min_samples=31, 1.5초 대기 시 ~150패킷 수집 → 처리 결과 충분히 생성됨
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
    assert sample.hbo.shape == (8,)
    assert sample.hbr.shape == (8,)
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/processing/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.processing.pipeline'`

- [ ] **Step 3: pipeline.py 구현**

`src/processing/pipeline.py`:
```python
from __future__ import annotations
import threading
from collections import deque
from typing import Callable
import numpy as np
from src.core.ring_buffer import RingBuffer
from src.core.models import RawPacket, ProcessedSample
from src.core.config import AppConfig
from src.processing.filters import bandpass_filter, baseline_correct
from src.processing.mbll import modified_beer_lambert
from src.processing.concentration import ConcentrationIndex


class ProcessingPipeline(threading.Thread):
    """링 버퍼에서 패킷을 읽어 HbO/HbR 및 집중도 지수를 계산하는 스레드.

    on_sample 콜백으로 처리 결과를 GUI 스레드에 전달한다.
    """

    _BUFFER_SEC = 10.0  # 내부 슬라이딩 윈도우 길이 (초)

    def __init__(
        self,
        buffer: RingBuffer,
        config: AppConfig,
        concentration_index: ConcentrationIndex,
        on_sample: Callable[[ProcessedSample], None],
    ) -> None:
        super().__init__(daemon=True)
        self._buffer = buffer
        self._config = config
        self._concentration_index = concentration_index
        self._on_sample = on_sample
        self._stop_event = threading.Event()

        sr = config.device.sampling_rate_hz
        n_ch = config.device.n_channels
        max_samples = int(self._BUFFER_SEC * sr)
        # (n_channels, n_wavelengths, max_samples) 슬라이딩 윈도우
        self._window: deque[RawPacket] = deque(maxlen=max_samples)

    def run(self) -> None:
        cfg = self._config
        sr = cfg.device.sampling_rate_hz
        min_samples = int(cfg.processing.baseline_window_sec * sr) + 1

        while not self._stop_event.is_set():
            packet = self._buffer.get(timeout=0.1)
            if packet is None:
                continue
            self._window.append(packet)

            if len(self._window) < min_samples:
                continue

            self._process_window()

    def _process_window(self) -> None:
        cfg = self._config
        packets = list(self._window)
        n_ch = cfg.device.n_channels
        n_wl = 2
        n_t = len(packets)

        # (n_channels, n_wavelengths, n_samples) 배열 구성
        raw = np.zeros((n_ch, n_wl, n_t))
        for t, pkt in enumerate(packets):
            for ch in range(n_ch):
                raw[ch, :, t] = pkt.intensity_by_channel(ch)

        # 밴드패스 필터 (채널×파장 각각)
        for wl in range(n_wl):
            raw[:, wl, :] = bandpass_filter(
                raw[:, wl, :],
                low_hz=cfg.processing.bandpass_low_hz,
                high_hz=cfg.processing.bandpass_high_hz,
                sampling_rate_hz=cfg.device.sampling_rate_hz,
            )

        # mBLL → HbO/HbR: (n_channels, n_samples)
        hbo_all, hbr_all = modified_beer_lambert(
            raw,
            cfg.device.wavelengths_nm,
            cfg.processing.extinction_coefficients["hbo"],
            cfg.processing.extinction_coefficients["hbr"],
            cfg.processing.dpf,
            cfg.device.sds_mm,
        )

        # 현재 시점 (마지막 샘플)
        hbo_now = hbo_all[:, -1]
        hbr_now = hbr_all[:, -1]
        ci = self._concentration_index.compute(hbo_now, hbr_now)

        sample = ProcessedSample(
            timestamp=packets[-1].timestamp,
            hbo=hbo_now,
            hbr=hbr_now,
            concentration_index=ci,
        )
        self._on_sample(sample)

    def stop(self) -> None:
        self._stop_event.set()
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/processing/test_pipeline.py -v
```

Expected: `2 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/processing/pipeline.py tests/processing/test_pipeline.py
git commit -m "[processing] ProcessingPipeline 구현 (버퍼→필터→mBLL→집중도)"
```

---

## Task 11: HDF5 데이터 저장

**Files:**
- Create: `src/storage/session_store.py`
- Create: `tests/storage/test_session_store.py`

- [ ] **Step 1: 테스트 먼저 작성**

`tests/storage/test_session_store.py`:
```python
import numpy as np
import pytest
from pathlib import Path
from src.core.models import ProcessedSample, RawPacket
from src.storage.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "test_session.h5")


def make_sample(ts: float) -> ProcessedSample:
    return ProcessedSample(
        timestamp=ts,
        hbo=np.random.uniform(0, 1, size=8),
        hbr=np.random.uniform(-0.5, 0, size=8),
        concentration_index=0.5,
    )


def make_packet(ts: float) -> RawPacket:
    return RawPacket(timestamp=ts, channel_intensities=[0.5] * 16)


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
    assert loaded[0].hbo.shape == (8,)


def test_store_and_load_raw_packets(store):
    packets = [make_packet(float(i)) for i in range(5)]
    store.open()
    for p in packets:
        store.write_raw(p)
    store.close()

    loaded = store.load_raw()
    assert len(loaded) == 5
    assert loaded[2].timestamp == 2.0
    assert len(loaded[0].channel_intensities) == 16


def test_concentration_index_preserved(store):
    sample = make_sample(1.0)
    sample.concentration_index = 0.73
    store.open()
    store.write_processed(sample)
    store.close()

    loaded = store.load_processed()
    assert abs(loaded[0].concentration_index - 0.73) < 1e-6
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/storage/test_session_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.storage.session_store'`

- [ ] **Step 3: session_store.py 구현**

`src/storage/session_store.py`:
```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
from src.core.models import RawPacket, ProcessedSample


class SessionStore:
    """fNIRS 세션 데이터를 HDF5 파일로 저장/로드한다.

    구조:
        /raw/timestamps          (N,)
        /raw/intensities         (N, n_ch*n_wl)
        /processed/timestamps    (M,)
        /processed/hbo           (M, n_channels)
        /processed/hbr           (M, n_channels)
        /processed/ci            (M,)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: h5py.File | None = None

    def open(self) -> None:
        self._file = h5py.File(self._path, "w")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def write_raw(self, packet: RawPacket) -> None:
        assert self._file is not None
        grp = self._file.require_group("raw")
        intensities = np.array(packet.channel_intensities)
        self._append_dataset(grp, "timestamps", np.array([packet.timestamp]))
        self._append_dataset(grp, "intensities", intensities[np.newaxis, :])

    def write_processed(self, sample: ProcessedSample) -> None:
        assert self._file is not None
        grp = self._file.require_group("processed")
        self._append_dataset(grp, "timestamps", np.array([sample.timestamp]))
        self._append_dataset(grp, "hbo", sample.hbo[np.newaxis, :])
        self._append_dataset(grp, "hbr", sample.hbr[np.newaxis, :])
        self._append_dataset(grp, "ci", np.array([sample.concentration_index]))

    def load_raw(self) -> list[RawPacket]:
        with h5py.File(self._path, "r") as f:
            grp = f["raw"]
            timestamps = grp["timestamps"][:]
            intensities = grp["intensities"][:]
        return [
            RawPacket(timestamp=float(timestamps[i]), channel_intensities=list(intensities[i]))
            for i in range(len(timestamps))
        ]

    def load_processed(self) -> list[ProcessedSample]:
        with h5py.File(self._path, "r") as f:
            grp = f["processed"]
            timestamps = grp["timestamps"][:]
            hbo = grp["hbo"][:]
            hbr = grp["hbr"][:]
            ci = grp["ci"][:]
        return [
            ProcessedSample(
                timestamp=float(timestamps[i]),
                hbo=hbo[i],
                hbr=hbr[i],
                concentration_index=float(ci[i]),
            )
            for i in range(len(timestamps))
        ]

    @staticmethod
    def _append_dataset(grp: h5py.Group, name: str, data: np.ndarray) -> None:
        if name not in grp:
            maxshape = (None,) + data.shape[1:]
            grp.create_dataset(name, data=data, maxshape=maxshape, chunks=True)
        else:
            ds = grp[name]
            ds.resize(ds.shape[0] + data.shape[0], axis=0)
            ds[-data.shape[0]:] = data
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/storage/test_session_store.py -v
```

Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/storage/session_store.py tests/storage/test_session_store.py src/storage/__init__.py tests/storage/__init__.py
git commit -m "[storage] SessionStore HDF5 저장/로드 구현"
```

---

## Task 12: PySide6 프로토타입 UI

**Files:**
- Create: `src/ui/main_window.py`
- Create: `src/main.py`

> **참고:** pytest-qt를 이용한 GUI 테스트는 QApplication이 필요하며, CI 환경에서는 화면이 없을 수 있다.
> `pytest-qt`의 `qtbot` fixture로 창 생성 여부만 검증한다.

- [ ] **Step 1: main_window.py 구현**

`src/ui/main_window.py`:
```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QStatusBar,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from src.core.models import ProcessedSample
from src.core.config import AppConfig


class MainWindow(QMainWindow):
    """fNIRS 실시간 모니터링 메인 윈도우 (프로토타입)."""

    N_CHANNELS = 8
    PLOT_WINDOW_SEC = 30.0

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._sr = config.device.sampling_rate_hz
        self._max_points = int(self.PLOT_WINDOW_SEC * self._sr)

        self.setWindowTitle("fNIRS 집중도 모니터")
        self.resize(1200, 800)
        self._build_ui()
        self._init_plot_data()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 상단: 집중도 미터
        top_layout = QHBoxLayout()
        ci_label = QLabel("집중도")
        ci_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self._ci_bar = QProgressBar()
        self._ci_bar.setRange(0, 100)
        self._ci_bar.setValue(0)
        self._ci_bar.setFixedHeight(30)
        self._ci_value_label = QLabel("0%")
        self._ci_value_label.setFont(QFont("Arial", 14))
        top_layout.addWidget(ci_label)
        top_layout.addWidget(self._ci_bar, stretch=1)
        top_layout.addWidget(self._ci_value_label)
        main_layout.addLayout(top_layout)

        # 중앙: 실시간 HbO/HbR 그래프 (8채널)
        self._plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self._plot_widget, stretch=1)
        self._plots: list[pg.PlotItem] = []
        self._hbo_curves: list[pg.PlotDataItem] = []
        self._hbr_curves: list[pg.PlotDataItem] = []
        for ch in range(self.N_CHANNELS):
            p = self._plot_widget.addPlot(row=ch // 4, col=ch % 4)
            p.setTitle(f"Ch {ch + 1}", size="10pt")
            p.setLabel("left", "μmol/L")
            p.showGrid(x=True, y=True, alpha=0.3)
            hbo_curve = p.plot(pen=pg.mkPen("r", width=1.5), name="HbO")
            hbr_curve = p.plot(pen=pg.mkPen("b", width=1.5), name="HbR")
            self._plots.append(p)
            self._hbo_curves.append(hbo_curve)
            self._hbr_curves.append(hbr_curve)

        # 하단: 컨트롤 버튼
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("측정 시작")
        self._stop_btn = QPushButton("측정 중지")
        self._stop_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self._start_btn)
        btn_layout.addWidget(self._stop_btn)
        main_layout.addLayout(btn_layout)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("준비")

    def _init_plot_data(self) -> None:
        self._hbo_data: list[list[float]] = [[] for _ in range(self.N_CHANNELS)]
        self._hbr_data: list[list[float]] = [[] for _ in range(self.N_CHANNELS)]

    @Slot(object)
    def update_sample(self, sample: ProcessedSample) -> None:
        """ProcessingPipeline의 on_sample 콜백에서 호출된다."""
        # 집중도 미터 업데이트
        ci_pct = int(sample.concentration_index * 100)
        self._ci_bar.setValue(ci_pct)
        self._ci_value_label.setText(f"{ci_pct}%")

        # 채널별 HbO/HbR 그래프 업데이트
        for ch in range(self.N_CHANNELS):
            self._hbo_data[ch].append(float(sample.hbo[ch]))
            self._hbr_data[ch].append(float(sample.hbr[ch]))
            if len(self._hbo_data[ch]) > self._max_points:
                self._hbo_data[ch] = self._hbo_data[ch][-self._max_points:]
                self._hbr_data[ch] = self._hbr_data[ch][-self._max_points:]
            self._hbo_curves[ch].setData(self._hbo_data[ch])
            self._hbr_curves[ch].setData(self._hbr_data[ch])
```

- [ ] **Step 2: main.py 작성**

`src/main.py`:
```python
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

    def on_start():
        acq_thread.start()
        pipeline.start()
        window._start_btn.setEnabled(False)
        window._stop_btn.setEnabled(True)
        window.statusBar().showMessage("측정 중...")

    def on_stop():
        acq_thread.stop()
        pipeline.stop()
        window._start_btn.setEnabled(True)
        window._stop_btn.setEnabled(False)
        window.statusBar().showMessage("측정 중지됨")

    window._start_btn.clicked.connect(on_start)
    window._stop_btn.clicked.connect(on_stop)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: UI 테스트 작성**

`tests/ui/__init__.py` 파일 생성 후 `tests/ui/test_main_window.py`:
```python
import numpy as np
import pytest
from src.core.models import ProcessedSample
from src.ui.main_window import MainWindow


@pytest.fixture
def config(tmp_path):
    from pathlib import Path
    from src.core.config import AppConfig
    yaml_content = """
device:
  wavelengths_nm: [735, 850]
  sampling_rate_hz: 10
  n_channels: 8
  sds_mm: 30.0
  source_detector_pairs:
    - [1, 1]
    - [1, 2]
    - [2, 1]
    - [2, 2]
    - [3, 1]
    - [3, 2]
    - [4, 1]
    - [4, 2]
processing:
  bandpass_low_hz: 0.01
  bandpass_high_hz: 0.5
  baseline_window_sec: 5.0
  extinction_coefficients:
    hbo: [1.4067, 0.9012]
    hbr: [3.7216, 0.7234]
  dpf: [6.51, 5.86]
storage:
  data_dir: "data"
  session_filename_format: "%Y%m%d_%H%M%S_session.h5"
simulator:
  hbo_amplitude: 0.5
  hbo_freq_hz: 0.1
  noise_std: 0.05
"""
    f = tmp_path / "settings.yaml"
    f.write_text(yaml_content)
    return AppConfig.from_yaml(f)


def test_main_window_opens(qtbot, config):
    window = MainWindow(config)
    qtbot.addWidget(window)
    window.show()
    assert window.isVisible()


def test_update_sample_updates_ci_bar(qtbot, config):
    window = MainWindow(config)
    qtbot.addWidget(window)
    sample = ProcessedSample(
        timestamp=1.0,
        hbo=np.array([1.0] * 8),
        hbr=np.array([-0.1] * 8),
        concentration_index=0.75,
    )
    window.update_sample(sample)
    assert window._ci_bar.value() == 75
    assert window._ci_value_label.text() == "75%"
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
pytest tests/ui/test_main_window.py -v
```

Expected: `2 passed`

- [ ] **Step 5: 앱 직접 실행 확인**

```bash
python src/main.py
```

Expected: 윈도우 열림 → "측정 시작" 버튼 클릭 → 8개 채널 실시간 그래프 업데이트, 집중도 미터 변동 확인.

- [ ] **Step 6: 전체 테스트 통과 확인**

```bash
pytest tests/ -v
```

Expected: 모든 테스트 passed (0 failed)

- [ ] **Step 7: 최종 커밋**

```bash
git add src/ui/main_window.py src/main.py src/ui/__init__.py tests/ui/
git commit -m "[ui] PySide6 프로토타입 UI 구현 (실시간 HbO/HbR + 집중도 미터)"
```

---

## 전체 테스트 실행

모든 태스크 완료 후:

```bash
pytest tests/ -v --tb=short
```

Expected 결과:
```
tests/core/test_config.py          :: 2 passed
tests/core/test_hal.py             :: 3 passed
tests/core/test_models.py          :: 4 passed
tests/core/test_ring_buffer.py     :: 5 passed
tests/acquisition/test_simulator.py           :: 5 passed
tests/acquisition/test_acquisition_thread.py  :: 2 passed
tests/processing/test_filters.py   :: 5 passed
tests/processing/test_mbll.py      :: 3 passed
tests/processing/test_concentration.py :: 5 passed
tests/processing/test_pipeline.py  :: 2 passed
tests/storage/test_session_store.py:: 3 passed
tests/ui/test_main_window.py       :: 2 passed
```
