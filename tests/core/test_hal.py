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
            return RawPacket(timestamp=0.0, channel_intensities=[0.0] * 12, n_wavelengths=3)
        def disconnect(self) -> None:
            pass

    device = MinimalDevice()
    assert device.connect() is True
