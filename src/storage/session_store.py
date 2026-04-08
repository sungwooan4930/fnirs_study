from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
from src.core.models import RawPacket, ProcessedSample


class SessionStore:
    """fNIRS 세션 데이터를 HDF5 파일로 저장/로드한다.

    HDF5 파일 구조:
        /raw/timestamps          (N,)
        /raw/intensities         (N, n_ch*n_wl)
        /raw attrs: n_wavelengths
        /processed/timestamps    (M,)
        /processed/hbo           (M, n_channels)
        /processed/hbr           (M, n_channels)
        /processed/ci            (M,)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: h5py.File | None = None

    def open(self) -> None:
        """HDF5 파일을 쓰기 모드로 연다."""
        self._file = h5py.File(self._path, "w")

    def close(self) -> None:
        """HDF5 파일을 닫는다."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_raw(self, packet: RawPacket) -> None:
        """RawPacket을 /raw 그룹에 추가한다."""
        if self._file is None:
            raise RuntimeError("Store is not open. Call open() first.")
        grp = self._file.require_group("raw")
        # n_wavelengths를 그룹 속성으로 저장 (첫 쓰기 시)
        if "n_wavelengths" not in grp.attrs:
            grp.attrs["n_wavelengths"] = packet.n_wavelengths
        intensities = np.array(packet.channel_intensities)
        self._append_dataset(grp, "timestamps", np.array([packet.timestamp]))
        self._append_dataset(grp, "intensities", intensities[np.newaxis, :])

    def write_processed(self, sample: ProcessedSample) -> None:
        """ProcessedSample을 /processed 그룹에 추가한다."""
        if self._file is None:
            raise RuntimeError("Store is not open. Call open() first.")
        grp = self._file.require_group("processed")
        self._append_dataset(grp, "timestamps", np.array([sample.timestamp]))
        self._append_dataset(grp, "hbo", sample.hbo[np.newaxis, :])
        self._append_dataset(grp, "hbr", sample.hbr[np.newaxis, :])
        self._append_dataset(grp, "ci", np.array([sample.concentration_index]))

    def load_raw(self) -> list[RawPacket]:
        """저장된 RawPacket 목록을 반환한다."""
        with h5py.File(self._path, "r") as f:
            grp = f["raw"]
            if "n_wavelengths" not in grp.attrs:
                raise ValueError(
                    "HDF5 raw group is missing 'n_wavelengths' attribute. "
                    "The file may be corrupted or written by an incompatible version."
                )
            n_wavelengths = int(grp.attrs["n_wavelengths"])
            timestamps = grp["timestamps"][:]
            intensities = grp["intensities"][:]
        return [
            RawPacket(
                timestamp=float(timestamps[i]),
                channel_intensities=[float(v) for v in intensities[i]],
                n_wavelengths=n_wavelengths,
            )
            for i in range(len(timestamps))
        ]

    def load_processed(self) -> list[ProcessedSample]:
        """저장된 ProcessedSample 목록을 반환한다."""
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
        """데이터셋에 행을 추가한다. 없으면 생성한다."""
        if name not in grp:
            maxshape = (None,) + data.shape[1:]
            grp.create_dataset(name, data=data, maxshape=maxshape, chunks=True)
        else:
            ds = grp[name]
            ds.resize(ds.shape[0] + data.shape[0], axis=0)
            ds[-data.shape[0]:] = data
