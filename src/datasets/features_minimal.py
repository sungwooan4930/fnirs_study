"""임시 특징 추출기.

B(전처리 파이프라인)가 없는 상태에서 raw → 특징 경로를 잇기 위한 최소
구현이다. 특징이 없으면 LOSO 하네스를 검증할 수 없기 때문에 필요하다.
B 완성 후 동일 시그니처로 교체한다.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)


def _band_powers(segment: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_ch, n_samp) 구간에서 채널별 θ·α 대역 파워를 구한다."""
    nperseg = min(256, segment.shape[1])
    freqs, pxx = sp_signal.welch(segment, fs=sfreq, nperseg=nperseg, axis=1)
    theta = pxx[:, (freqs >= THETA_BAND[0]) & (freqs < THETA_BAND[1])].sum(axis=1)
    alpha = pxx[:, (freqs >= ALPHA_BAND[0]) & (freqs < ALPHA_BAND[1])].sum(axis=1)
    return theta, alpha


def extract_features(rec, windows) -> dict[str, np.ndarray]:
    """창별 특징을 모달리티별 배열로 반환한다."""
    n_win = len(windows.start_s)

    eeg_feats = np.zeros((n_win, rec.eeg.shape[0] * 2))
    fnirs_feats = np.zeros((n_win, rec.hbo.shape[0] * 2))
    behav_feats = np.zeros((n_win, 2))

    for i, (start, end) in enumerate(zip(windows.start_s, windows.end_s)):
        e0, e1 = int(round(start * rec.eeg_sfreq)), int(round(end * rec.eeg_sfreq))
        theta, alpha = _band_powers(rec.eeg[:, e0:e1], rec.eeg_sfreq)
        eeg_feats[i] = np.concatenate([theta, alpha])

        f0, f1 = int(round(start * rec.fnirs_sfreq)), int(round(end * rec.fnirs_sfreq))
        seg = rec.hbo[:, f0:f1]
        hbo_mean = seg.mean(axis=1)
        x = np.arange(seg.shape[1], dtype=float)
        x_centered = x - x.mean()
        denom = (x_centered ** 2).sum()
        hbo_slope = ((seg - hbo_mean[:, None]) * x_centered).sum(axis=1) / denom
        fnirs_feats[i] = np.concatenate([hbo_mean, hbo_slope])

        in_win = (rec.behavior.onsets >= start) & (rec.behavior.onsets < end)
        if in_win.any():
            behav_feats[i] = [
                rec.behavior.correct[in_win].mean(),
                rec.behavior.rt[in_win].mean(),
            ]
        # 창 안에 자극이 없으면 0.0으로 남긴다

    return {"eeg": eeg_feats, "fnirs": fnirs_feats, "behavior": behav_feats}
