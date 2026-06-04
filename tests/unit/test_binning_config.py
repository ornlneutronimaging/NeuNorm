"""
Unit tests for BinningConfig Pydantic model.

Tests TOF/Energy/Wavelength binning configuration validation.
"""

import pytest
from pydantic import ValidationError


def test_binning_config_import():
    """Test that BinningConfig can be imported"""


def test_binning_config_energy_mode_requires_energy_range():
    """Energy binning must provide energy_range"""
    from neunorm.data_models.tof import BinningConfig

    # Should raise error if energy_range not provided
    with pytest.raises(ValidationError, match="energy_range required"):
        BinningConfig(bins=5000, bin_space="energy")

    # Should work if energy_range provided
    config = BinningConfig(bins=5000, bin_space="energy", energy_range=(1.0, 100.0))
    assert config.bins == 5000
    assert config.bin_space == "energy"
    assert config.energy_range == (1.0, 100.0)


def test_binning_config_wavelength_mode_requires_wavelength_range():
    """Wavelength binning must provide wavelength_range"""
    from neunorm.data_models.tof import BinningConfig

    # Should raise error if wavelength_range not provided
    with pytest.raises(ValidationError, match="wavelength_range required"):
        BinningConfig(bins=3000, bin_space="wavelength")

    # Should work if wavelength_range provided
    config = BinningConfig(bins=3000, bin_space="wavelength", wavelength_range=(0.5, 3.0))
    assert config.bins == 3000
    assert config.bin_space == "wavelength"
    assert config.wavelength_range == (0.5, 3.0)


def test_binning_config_tof_mode_optional_range():
    """TOF binning can work without explicit range"""
    from neunorm.data_models.tof import BinningConfig

    # Should work without tof_range (uses full detector range)
    config = BinningConfig(bins=5000, bin_space="tof")
    assert config.bins == 5000
    assert config.bin_space == "tof"
    assert config.tof_range is None

    # Should also work WITH tof_range
    config2 = BinningConfig(
        bins=5000,
        bin_space="tof",
        tof_range=(1e5, 1e7),  # 100 μs to 10 ms
    )
    assert config2.tof_range == (1e5, 1e7)


def test_binning_config_energy_range_validation():
    """Energy range must be positive and min < max"""
    from neunorm.data_models.tof import BinningConfig

    # Negative energy should fail
    with pytest.raises(ValidationError, match="must be positive"):
        BinningConfig(bins=5000, bin_space="energy", energy_range=(-1.0, 100.0))

    # Zero energy should fail
    with pytest.raises(ValidationError, match="must be positive"):
        BinningConfig(bins=5000, bin_space="energy", energy_range=(0.0, 100.0))

    # Min >= max should fail
    with pytest.raises(ValidationError, match="E_min.*must be less than.*E_max"):
        BinningConfig(bins=5000, bin_space="energy", energy_range=(100.0, 10.0))

    # Equal values should fail
    with pytest.raises(ValidationError, match="E_min.*must be less than.*E_max"):
        BinningConfig(bins=5000, bin_space="energy", energy_range=(50.0, 50.0))


def test_binning_config_wavelength_range_validation():
    """Wavelength range must be positive and min < max"""
    from neunorm.data_models.tof import BinningConfig

    # Negative wavelength should fail
    with pytest.raises(ValidationError, match="must be positive"):
        BinningConfig(bins=3000, bin_space="wavelength", wavelength_range=(-0.5, 3.0))

    # Zero wavelength should fail
    with pytest.raises(ValidationError, match="must be positive"):
        BinningConfig(bins=3000, bin_space="wavelength", wavelength_range=(0.0, 3.0))

    # Min >= max should fail
    with pytest.raises(ValidationError, match="wl_min.*must be less than.*wl_max"):
        BinningConfig(bins=3000, bin_space="wavelength", wavelength_range=(3.0, 0.5))


def test_binning_config_bins_must_be_positive():
    """Number of bins must be positive integer"""
    from neunorm.data_models.tof import BinningConfig

    # Zero bins should fail
    with pytest.raises(ValidationError, match="greater than 0"):
        BinningConfig(bins=0, bin_space="energy", energy_range=(1.0, 100.0))

    # Negative bins should fail
    with pytest.raises(ValidationError, match="greater than 0"):
        BinningConfig(bins=-100, bin_space="energy", energy_range=(1.0, 100.0))


def test_binning_config_invalid_bin_space():
    """bin_space must be one of: tof, energy, wavelength"""
    from neunorm.data_models.tof import BinningConfig

    with pytest.raises(ValidationError, match="Input should be 'tof', 'energy' or 'wavelength'"):
        BinningConfig(bins=5000, bin_space="invalid")


def test_binning_config_defaults():
    """Test default values"""
    from neunorm.data_models.tof import BinningConfig

    # Minimal config for TOF mode (only mode that doesn't require range)
    config = BinningConfig(bin_space="tof")

    assert config.bins == 5000  # Default
    assert config.bin_space == "tof"
    assert config.use_log_bin is True  # Default
    assert config.tof_range is None
    assert config.energy_range is None
    assert config.wavelength_range is None


def test_binning_config_can_store_multiple_ranges():
    """Config can store ranges for multiple domains (useful for dual-mode)"""
    from neunorm.data_models.tof import BinningConfig

    config = BinningConfig(
        bins=5000,
        bin_space="energy",
        energy_range=(1.0, 100.0),
        wavelength_range=(0.5, 3.0),  # Also store for reference/conversion
        use_log_bin=True,
    )

    assert config.energy_range == (1.0, 100.0)
    assert config.wavelength_range == (0.5, 3.0)
    # Primary binning is in energy space
    assert config.bin_space == "energy"


def test_binning_config_log_vs_linear():
    """Test both logarithmic and linear binning modes"""
    from neunorm.data_models.tof import BinningConfig

    # Logarithmic (default)
    config_log = BinningConfig(bins=1000, bin_space="energy", energy_range=(0.1, 1000.0), use_log_bin=True)
    assert config_log.use_log_bin is True

    # Linear
    config_linear = BinningConfig(bins=1000, bin_space="wavelength", wavelength_range=(0.5, 3.0), use_log_bin=False)
    assert config_linear.use_log_bin is False
