"""
Tests for division-by-zero guards throughout the codebase.

Ensures graceful handling or meaningful errors when encountering zero values.
"""

import numpy as np
import pytest
import scipp as sc


def test_tof_to_energy_handles_zero_tof():
    """Test that TOF=0 is handled gracefully"""
    from neunorm.tof.binning import tof_to_energy

    tof_zero = sc.scalar(0.0, unit="s")
    flight_path = sc.scalar(25.0, unit="m")

    # Should raise meaningful error (not division by zero)
    with pytest.raises((ValueError, ZeroDivisionError), match="TOF must be positive"):
        tof_to_energy(tof_zero, flight_path)


def test_tof_to_energy_handles_very_small_tof():
    """Test numerical stability with very small TOF"""
    from neunorm.tof.binning import tof_to_energy

    # Very high energy (short TOF)
    tof_tiny = sc.scalar(1e-10, unit="s")  # 0.1 ns
    flight_path = sc.scalar(25.0, unit="m")

    energy = tof_to_energy(tof_tiny, flight_path)

    # Should produce finite result
    assert np.isfinite(energy.value)
    assert energy.value > 0


def test_energy_to_tof_handles_zero_energy():
    """Test that energy=0 conversion is handled"""
    from neunorm.tof.binning import energy_to_wavelength

    energy_zero = sc.scalar(0.0, unit="eV")

    # Should raise meaningful error
    with pytest.raises(ValueError, match="Energy must be positive"):
        energy_to_wavelength(energy_zero)


def test_get_relative_uncertainty_handles_zero_data():
    """Test relative uncertainty when data contains zeros"""
    from neunorm.processing.uncertainty_calculator import get_relative_uncertainty

    # Data with zeros
    data = sc.DataArray(data=sc.array(dims=["x"], values=[100.0, 0.0, 200.0], unit="counts", dtype="float64"))
    data.variances = np.array([100.0, 0.0, 200.0])

    # Should either:
    # 1. Raise error about division by zero
    # 2. Return inf/nan for zero bins (and document this)
    # 3. Mask zero bins
    rel_unc = get_relative_uncertainty(data)

    # Zero bin should be inf or masked
    assert np.isinf(rel_unc.values[1]) or np.isnan(rel_unc.values[1])
    # Non-zero bins should be finite
    assert np.isfinite(rel_unc.values[0])
    assert np.isfinite(rel_unc.values[2])


def test_normalize_transmission_handles_zero_ob():
    """
    Test transmission calculation when OB has zero counts.

    Critical for normalizer module implementation.
    """
    # This will test the normalizer once we implement it (Priority 4)
    # For now, document the requirement
    pass  # Placeholder for Priority 4


def test_create_tof_bins_energy_handles_near_zero_energy():
    """Test bin creation when energy range includes very low energies"""
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.binning import create_tof_bins

    # Very low energy (near thermal) - high TOF
    config = BinningConfig(
        bins=100,
        bin_space="energy",
        energy_range=(0.001, 1.0),  # milli-eV to eV
        use_log_bin=True,
    )
    flight_path = sc.scalar(25.0, unit="m")

    tof_bins = create_tof_bins(config, flight_path)

    # Should produce finite TOF bins
    assert np.all(np.isfinite(tof_bins.values))
    assert np.all(tof_bins.values > 0)
