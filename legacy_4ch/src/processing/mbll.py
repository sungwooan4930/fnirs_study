"""Modified Beer-Lambert Law (mBLL) for fNIRS signal processing.

Converts raw intensity signals to HbO/HbR concentration changes.

Hardware spec: 4 channels, 3 wavelengths [780, 850, 950] nm.
The extinction matrix A is shape (n_wavelengths, 2) — overdetermined system
requiring np.linalg.pinv(A), not a regular inverse.
"""

from __future__ import annotations

import warnings

import numpy as np


def modified_beer_lambert(
    raw_intensity: np.ndarray,       # shape (n_channels, n_wavelengths, n_samples)
    wavelengths_nm: list[int],
    ext_hbo: list[float],            # HbO extinction coefficients per wavelength, L/(mmol·cm)
    ext_hbr: list[float],            # HbR extinction coefficients per wavelength, L/(mmol·cm)
    dpf: list[float],                # differential path length factor per wavelength
    sds_mm: float,                   # source-detector separation in mm
) -> tuple[np.ndarray, np.ndarray]:  # hbo shape (n_channels, n_samples), hbr same
    """Apply the modified Beer-Lambert Law to convert intensity to concentration.

    Parameters
    ----------
    raw_intensity:
        Raw light intensity array, shape (n_channels, n_wavelengths, n_samples).
    wavelengths_nm:
        List of wavelengths in nanometres; length must equal n_wavelengths.
    ext_hbo:
        HbO molar extinction coefficients in L/(mmol·cm), one per wavelength.
    ext_hbr:
        HbR molar extinction coefficients in L/(mmol·cm), one per wavelength.
    dpf:
        Differential path length factors, one per wavelength (dimensionless).
    sds_mm:
        Source-detector separation in millimetres. Must be positive.

    Returns
    -------
    hbo : np.ndarray, shape (n_channels, n_samples)
        Oxygenated haemoglobin concentration change in μmol/L.
    hbr : np.ndarray, shape (n_channels, n_samples)
        Deoxygenated haemoglobin concentration change in μmol/L.

    Raises
    ------
    ValueError
        If the lengths of `ext_hbo`, `ext_hbr`, or `dpf` do not match
        `wavelengths_nm`, or if the wavelength dimension of `raw_intensity`
        does not match `wavelengths_nm`, or if `sds_mm` is not positive.

    Notes
    -----
    Baseline intensity I0 is taken as the first sample along the time axis
    (index 0). Ensure that the first sample represents a resting-state baseline
    before calling this function.
    """
    n_wavelengths = len(wavelengths_nm)

    # --- Input validation ---
    if len(ext_hbo) != n_wavelengths:
        raise ValueError(
            f"len(ext_hbo)={len(ext_hbo)} does not match "
            f"len(wavelengths_nm)={n_wavelengths}"
        )
    if len(ext_hbr) != n_wavelengths:
        raise ValueError(
            f"len(ext_hbr)={len(ext_hbr)} does not match "
            f"len(wavelengths_nm)={n_wavelengths}"
        )
    if len(dpf) != n_wavelengths:
        raise ValueError(
            f"len(dpf)={len(dpf)} does not match "
            f"len(wavelengths_nm)={n_wavelengths}"
        )
    if raw_intensity.shape[1] != n_wavelengths:
        raise ValueError(
            f"raw_intensity wavelength dimension {raw_intensity.shape[1]} does not match "
            f"len(wavelengths_nm)={n_wavelengths}"
        )
    if sds_mm <= 0:
        raise ValueError(f"sds_mm must be positive, got {sds_mm}")

    # --- Step 1: unit conversion ---
    sds_cm = sds_mm / 10.0

    # --- Step 2: compute delta optical density ---
    # intensity_baseline = first sample, shape (n_channels, n_wavelengths, 1)
    intensity_baseline = raw_intensity[:, :, 0:1].copy().astype(np.float64)
    intensity_safe = raw_intensity.astype(np.float64)

    # Guard against division by zero
    if np.any(raw_intensity == 0):
        warnings.warn(
            "raw_intensity contains zeros; replaced with 1e-10. "
            "Zero values produce large spurious concentration spikes.",
            UserWarning,
            stacklevel=2,
        )
    intensity_baseline[intensity_baseline == 0.0] = 1e-10
    intensity_safe[intensity_safe == 0.0] = 1e-10

    # delta_OD = -log(I / I0), shape (n_channels, n_wavelengths, n_samples)
    delta_od = -np.log(intensity_safe / intensity_baseline)

    # --- Step 3: build extinction matrix A, shape (n_wavelengths, 2) ---
    A = np.array([[ext_hbo[i], ext_hbr[i]] for i in range(n_wavelengths)],
                 dtype=np.float64)  # (n_wavelengths, 2)

    # --- Step 4: pseudo-inverse (overdetermined system) ---
    A_pinv = np.linalg.pinv(A)  # shape (2, n_wavelengths)

    # --- Step 5: vectorized solve across all channels ---
    dpf_arr = np.array(dpf, dtype=np.float64).reshape(1, -1, 1)  # (1, n_wavelengths, 1)

    # mBLL: delta_c = delta_OD / (epsilon * DPF * L), where L = SDS in cm
    # od_normalized = delta_OD / (DPF * L), then A_pinv handles the epsilon division
    od_normalized = delta_od / (dpf_arr * sds_cm)  # (n_channels, n_wavelengths, n_samples)

    # A_pinv shape: (2, n_wavelengths), od_normalized shape: (n_channels, n_wavelengths, n_samples)
    # Result shape: (n_channels, 2, n_samples)
    concentrations = np.einsum("aw,bws->bas", A_pinv, od_normalized)

    # Convert mmol/L → μmol/L
    hbo = concentrations[:, 0, :] * 1000.0  # (n_channels, n_samples)
    hbr = concentrations[:, 1, :] * 1000.0

    return hbo, hbr
