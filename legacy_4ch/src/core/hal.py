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
