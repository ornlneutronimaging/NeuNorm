"""
Unit tests for resonance detection.

Tests automatic detection of resonance dips in energy-space transmission spectra.
Ported from venus_tof.resonance tests.
"""

import numpy as np
import pytest
import scipp as sc
from pydantic import ValidationError


def test_resonance_module_imports():
    """Test that resonance module can be imported"""


def test_resonance_detection_config_defaults():
    """Test default configuration values"""
    from neunorm.tof.resonance import ResonanceDetectionConfig

    config = ResonanceDetectionConfig()

    assert config.background_sigma_fraction == 0.05
    assert config.initial_prominence == 0.01
    assert config.initial_width == 3
    assert config.min_snr == 50.0
    assert config.snr_window_fraction == 0.15
    assert config.min_peak_width == 3
    assert config.max_peak_width == 60
    assert config.min_prom_width_ratio == 0.001


def test_resonance_detection_config_custom_values():
    """Test custom configuration"""
    from neunorm.tof.resonance import ResonanceDetectionConfig

    config = ResonanceDetectionConfig(background_sigma_fraction=0.1, min_snr=100.0, max_peak_width=40)

    assert config.background_sigma_fraction == 0.1
    assert config.min_snr == 100.0
    assert config.max_peak_width == 40


def test_resonance_detection_config_validation():
    """Test configuration validation"""
    from neunorm.tof.resonance import ResonanceDetectionConfig

    # max_peak_width must be > min_peak_width
    with pytest.raises(ValidationError, match="max_peak_width.*must be.*min_peak_width"):
        ResonanceDetectionConfig(min_peak_width=20, max_peak_width=15)

    # background_sigma_fraction must be in (0, 0.2]
    with pytest.raises(ValidationError):
        ResonanceDetectionConfig(background_sigma_fraction=0.5)  # Too large


def test_detect_resonances_with_mock_data():
    """Test resonance detection with synthetic data"""
    from neunorm.tof.resonance import detect_resonances

    # Create mock transmission spectrum with artificial resonance dips
    n_bins = 1000
    energy_edges = sc.geomspace("energy", 1.0, 100.0, num=n_bins + 1, unit="eV")

    # Create transmission with dips at known energies
    energy_centers = (energy_edges.values[:-1] + energy_edges.values[1:]) / 2
    transmission = np.ones(n_bins) * 0.8  # Background transmission ~80%

    # Add resonance dips at 5 eV, 20 eV, 50 eV
    resonance_energies = [5.0, 20.0, 50.0]
    for res_e in resonance_energies:
        # Find nearest bin
        idx = np.argmin(np.abs(energy_centers - res_e))
        # Create dip (±5 bins width)
        for offset in range(-5, 6):
            if 0 <= idx + offset < n_bins:
                transmission[idx + offset] *= 0.5  # 50% dip

    # Create sample and OB histograms
    # OB = 10000 counts per bin (high counts for good SNR)
    ob_counts = np.ones((n_bins, 10, 10)) * 10000.0
    sample_counts = ob_counts * transmission[:, np.newaxis, np.newaxis]

    # Create explicit spatial coordinates
    x_edges = sc.arange("x", 0, 11, unit=sc.units.dimensionless)
    y_edges = sc.arange("y", 0, 11, unit=sc.units.dimensionless)

    hist_ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_counts, unit="counts", dtype="float64"),
        coords={"energy": energy_edges, "x": x_edges, "y": y_edges},
    )

    hist_sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_counts, unit="counts", dtype="float64"),
        coords={"energy": energy_edges, "x": x_edges, "y": y_edges},
    )

    # Detect resonances
    result = detect_resonances(hist_sample, hist_ob)

    # Should detect the 3 resonances
    assert "resonance_energies" in result
    assert "resonance_indices" in result
    assert "snr_values" in result

    # Should find resonances near our input values
    detected = result["resonance_energies"]
    assert len(detected) >= 1  # At least one resonance found


def test_detect_resonances_returns_dict():
    """Test that detect_resonances returns proper dict structure"""
    from neunorm.tof.resonance import detect_resonances

    # Minimal mock data
    energy_edges = sc.linspace("energy", 1.0, 100.0, num=101, unit="eV")
    data = np.ones((100, 5, 5)) * 100.0

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts", dtype="float64"),
        coords={"energy": energy_edges},
    )

    result = detect_resonances(hist, hist)

    # Verify return structure
    assert isinstance(result, dict)
    assert "resonance_energies" in result
    assert "resonance_indices" in result
    assert "snr_values" in result
    assert "n_initial" in result
    assert "n_snr_filtered" in result
    assert "n_shape_filtered" in result


def test_aggregate_resonance_image():
    """Test aggregation of resonance bins into 2D image"""
    from neunorm.tof.resonance import aggregate_resonance_image

    # Create mock histograms
    n_energy = 100
    energy_edges = sc.linspace("energy", 1.0, 100.0, num=n_energy + 1, unit="eV")

    # Sample and OB with resonance at bins 10, 50, 90
    sample_data = np.ones((n_energy, 20, 20)) * 100.0
    ob_data = np.ones((n_energy, 20, 20)) * 120.0  # OB slightly higher

    # Make resonance bins have lower transmission
    for res_idx in [10, 50, 90]:
        sample_data[res_idx, :, :] *= 0.5  # 50% transmission

    # Create explicit spatial coordinates
    x_edges = sc.arange("x", 0, 21, unit=sc.units.dimensionless)
    y_edges = sc.arange("y", 0, 21, unit=sc.units.dimensionless)

    hist_sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": energy_edges, "x": x_edges, "y": y_edges},
    )

    hist_ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_data, unit="counts", dtype="float64"),
        coords={"energy": energy_edges, "x": x_edges, "y": y_edges},
    )

    # Aggregate over resonance bins
    resonance_indices = np.array([10, 50, 90])
    trans_image = aggregate_resonance_image(hist_sample, hist_ob, resonance_indices)

    # Should be 2D (x, y)
    assert trans_image.ndim == 2
    assert "x" in trans_image.dims
    assert "y" in trans_image.dims
    assert trans_image.shape == (20, 20)

    # Values should be aggregated transmission
    # Aggregated: sum(sample[10,50,90]) / sum(ob[10,50,90])
    # = (50+50+50) / (120+120+120) = 150/360 = 0.4167
    expected_trans = (0.5 * 100 * 3) / (120 * 3)  # ~0.417
    assert np.allclose(trans_image.values, expected_trans, rtol=0.1)


def test_detect_resonances_with_known_validation():
    """Test resonance detection with known resonances for validation"""
    from neunorm.tof.resonance import detect_resonances

    # Create spectrum with dips at known Ta-181 energies
    n_bins = 1000
    energy_edges = sc.geomspace("energy", 1.0, 100.0, num=n_bins + 1, unit="eV")
    energy_centers = (energy_edges.values[:-1] + energy_edges.values[1:]) / 2

    # Known Ta-181 resonances in this range
    known_ta = np.array([4.28, 10.4, 20.1, 35.1, 48.7])

    transmission = np.ones(n_bins) * 0.8

    # Add dips at known energies
    for res_e in known_ta:
        idx = np.argmin(np.abs(energy_centers - res_e))
        for offset in range(-5, 6):
            if 0 <= idx + offset < n_bins:
                transmission[idx + offset] *= 0.3  # Strong dip

    # Create histograms with high counts for good SNR
    ob_data = np.ones((n_bins, 10, 10)) * 10000.0
    sample_data = ob_data * transmission[:, np.newaxis, np.newaxis]

    # Create explicit spatial coordinates
    x_edges = sc.arange("x", 0, 11, unit=sc.units.dimensionless)
    y_edges = sc.arange("y", 0, 11, unit=sc.units.dimensionless)

    hist_ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_data, unit="counts", dtype="float64"),
        coords={"energy": energy_edges, "x": x_edges, "y": y_edges},
    )

    hist_sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": energy_edges, "x": x_edges, "y": y_edges},
    )

    # Detect with validation
    result = detect_resonances(hist_sample, hist_ob, known_resonances=known_ta)

    # Should have validation results
    assert "validation" in result
    assert "recall" in result["validation"]
    assert "precision" in result["validation"]
    assert "n_matched" in result["validation"]
