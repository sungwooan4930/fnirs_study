"""Tests for modified Beer-Lambert Law (mBLL) implementation.

Spec: 4 channels, 3 wavelengths [780, 850, 950] nm.
"""

import numpy as np
import pytest

from src.processing.mbll import modified_beer_lambert

# Shared test parameters (4 channels, 3 wavelengths)
WAVELENGTHS = [780, 850, 950]
EXT_HBO = [0.975, 0.901, 1.046]   # HbO extinction coefficients, L/(mmol·cm)
EXT_HBR = [2.755, 0.781, 0.260]   # HbR extinction coefficients, L/(mmol·cm)
DPF = [6.51, 5.86, 5.12]          # differential path length factor per wavelength
SDS_MM = 30.0                      # source-detector separation in mm


class TestOutputShape:
    """Test 1: output shape matches (n_channels, n_samples)."""

    def test_shape_4ch_3wl_100samples(self):
        raw = np.random.uniform(0.8, 1.2, size=(4, 3, 100))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (4, 100), f"hbo shape {hbo.shape} != (4, 100)"
        assert hbr.shape == (4, 100), f"hbr shape {hbr.shape} != (4, 100)"

    def test_shape_single_channel(self):
        raw = np.random.uniform(0.8, 1.2, size=(1, 3, 50))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (1, 50)
        assert hbr.shape == (1, 50)

    def test_shape_single_sample(self):
        raw = np.random.uniform(0.8, 1.2, size=(4, 3, 1))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (4, 1)
        assert hbr.shape == (4, 1)


class TestOutputDtype:
    """Test 2: output dtype is float64."""

    def test_hbo_dtype_float64(self):
        raw = np.random.uniform(0.8, 1.2, size=(4, 3, 100))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.dtype == np.float64, f"hbo dtype {hbo.dtype} != float64"

    def test_hbr_dtype_float64(self):
        raw = np.random.uniform(0.8, 1.2, size=(4, 3, 100))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbr.dtype == np.float64, f"hbr dtype {hbr.dtype} != float64"


class TestConstantSignal:
    """Test 3: constant signal → near-zero concentration change."""

    def test_constant_ones_gives_near_zero_hbo(self):
        raw = np.ones((4, 3, 100))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        # delta_OD = 0 everywhere → HbO should be ~0
        np.testing.assert_allclose(hbo, 0.0, atol=1e-10,
                                   err_msg="Constant signal should yield ~0 HbO")

    def test_constant_ones_gives_near_zero_hbr(self):
        raw = np.ones((4, 3, 100))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        np.testing.assert_allclose(hbr, 0.0, atol=1e-10,
                                   err_msg="Constant signal should yield ~0 HbR")

    def test_constant_non_one_value(self):
        """Any constant signal (not just 1.0) → near-zero change."""
        raw = np.full((4, 3, 100), fill_value=0.5)
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        np.testing.assert_allclose(hbo, 0.0, atol=1e-10)
        np.testing.assert_allclose(hbr, 0.0, atol=1e-10)


class TestInputValidation:
    """Test 4: mismatched list lengths raise ValueError."""

    def test_ext_hbo_wrong_length_raises(self):
        raw = np.ones((4, 3, 100))
        bad_ext_hbo = [0.975, 0.901]  # len 2, but wavelengths has len 3
        with pytest.raises(ValueError):
            modified_beer_lambert(raw, WAVELENGTHS, bad_ext_hbo, EXT_HBR, DPF, SDS_MM)

    def test_ext_hbr_wrong_length_raises(self):
        raw = np.ones((4, 3, 100))
        bad_ext_hbr = [2.755]  # len 1
        with pytest.raises(ValueError):
            modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, bad_ext_hbr, DPF, SDS_MM)

    def test_dpf_wrong_length_raises(self):
        raw = np.ones((4, 3, 100))
        bad_dpf = [6.51, 5.86]  # len 2
        with pytest.raises(ValueError):
            modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, bad_dpf, SDS_MM)

    def test_raw_wavelength_dim_mismatch_raises(self):
        raw = np.ones((4, 2, 100))  # 2 wavelengths, but wavelengths list has 3
        with pytest.raises(ValueError):
            modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)


class TestZeroValueGuard:
    """Test 5: zero values in raw_intensity don't crash (division by zero guard)."""

    def test_zero_baseline_does_not_crash(self):
        raw = np.ones((4, 3, 100))
        raw[:, :, 0] = 0.0  # zero baseline (I0)
        # Should not raise ZeroDivisionError or produce NaN/Inf that crashes
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (4, 100)
        assert hbr.shape == (4, 100)

    def test_zero_signal_samples_does_not_crash(self):
        raw = np.ones((4, 3, 100))
        raw[:, :, 50] = 0.0  # zero at mid-signal
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (4, 100)
        assert hbr.shape == (4, 100)

    def test_all_zeros_does_not_crash(self):
        raw = np.zeros((4, 3, 100))
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (4, 100)
        assert hbr.shape == (4, 100)


class TestPseudoInverse:
    """Verify overdetermined system is solved via pseudo-inverse (3 wavelengths, 2 unknowns)."""

    def test_uses_overdetermined_system(self):
        """With 3 wavelengths and 2 unknowns, result must exist (pinv handles it)."""
        raw = np.random.uniform(0.9, 1.1, size=(4, 3, 200))
        # Should not raise LinAlgError (which np.linalg.inv would on non-square matrix)
        hbo, hbr = modified_beer_lambert(raw, WAVELENGTHS, EXT_HBO, EXT_HBR, DPF, SDS_MM)
        assert hbo.shape == (4, 200)
        assert hbr.shape == (4, 200)
        # Results should be finite
        assert np.all(np.isfinite(hbo)), "hbo contains non-finite values"
        assert np.all(np.isfinite(hbr)), "hbr contains non-finite values"
