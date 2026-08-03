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


class TestVarianceFreeInputDtypes:
    """``detect_resonances`` accepts variance-free histograms in the dtypes the loaders produce.

    The SNR step reads a propagated variance, so a variance-free input has Poisson variance synthesized
    for it (``Var = counts``). scipp requires the values and the variances to share a dtype, and
    ``float32`` is what ``event_converter`` and ``tiff_loader`` produce natively while a hand-built
    histogram carries integers — so getting this wrong turns any of those into a scipp dtype error that
    names nothing in NeuNorm.
    """

    @staticmethod
    def _histogram(dtype, *, variances=False):
        n_energy = 120
        edges = np.geomspace(1.0, 100.0, n_energy + 1)
        values = np.full((n_energy, 8, 8), 1000).astype(dtype)
        data = sc.array(dims=["energy", "x", "y"], values=values, unit="counts")
        if variances:
            data = sc.array(dims=["energy", "x", "y"], values=values, variances=values.copy(), unit="counts")
        return sc.DataArray(data, coords={"energy": sc.array(dims=["energy"], values=edges, unit="eV")})

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
    def test_a_variance_free_histogram_is_accepted_whatever_its_count_dtype(self, dtype):
        """No dtype error, and a well-formed result dict."""
        from neunorm.tof.resonance import detect_resonances

        result = detect_resonances(self._histogram(dtype), self._histogram(dtype))

        assert isinstance(result, dict)
        for key in ("resonance_energies", "resonance_indices", "snr_values", "n_initial"):
            assert key in result

    def test_float32_counts_stay_float32_through_the_region_collapse(self):
        """Precision is preserved rather than promoted, because promoting doubles the working set.

        The region defaults to the whole detector here, so a promotion to float64 would double peak
        memory on the largest array in the run.
        """
        from neunorm.tof.resonance import _region_spectra

        spectrum_ta, spectrum_ob = _region_spectra(self._histogram(np.float32), self._histogram(np.float32), None)

        assert spectrum_ta.dtype == "float32", f"promoted to {spectrum_ta.dtype}"
        assert spectrum_ob.dtype == "float32"

    def test_integer_counts_are_promoted_rather_than_given_integer_variances(self):
        """Integer counts cannot carry a meaningful variance dtype, so they become float64."""
        from neunorm.tof.resonance import _region_spectra

        spectrum_ta, _ = _region_spectra(self._histogram(np.int64), self._histogram(np.int64), None)

        assert spectrum_ta.dtype == "float64"
        assert spectrum_ta.variances is not None

    def test_an_input_that_already_carries_variances_is_left_exactly_as_it_is(self):
        """Synthesis only fills a gap; it must never overwrite a real variance.

        Pinned with a variance deliberately unequal to the counts — the case where overwriting would be
        both wrong and invisible, since a Poisson-shaped input would hide the substitution.
        """
        from neunorm.tof.resonance import _with_poisson_variances

        values = np.array([[100.0, 100.0], [100.0, 100.0]])
        data = sc.DataArray(sc.array(dims=["x", "y"], values=values, variances=np.full((2, 2), 7.0), unit="counts"))

        out = _with_poisson_variances(data)

        np.testing.assert_array_equal(out.variances, np.full((2, 2), 7.0))
        assert out is data, "an input that already has variances should be returned untouched"
