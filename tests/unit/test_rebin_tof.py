"""
Unit tests for the HistogramRebinner Module
"""

import numpy as np
import pytest
import scipp as sc


def test_rebin_tof_by_bins():
    """Test the rebin_tof function when rebinning by number of bins"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # rebin by a width of 1, which should return the original data
    result = rebin_tof(data, width=1)
    assert result is data  # Should return the original data without modification

    # rebin by a width of 2, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, width=2)
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly

    # rebin by a width of 3, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, width=3)
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly

    # rebin by a width of 5, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, width=5)
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [0, 12500, 25000, 30000])  # New edges
    np.testing.assert_allclose(
        result.data.values[0], 50.0
    )  # First rebinned bin should have sum of first 5 original bins
    np.testing.assert_allclose(
        result.data.values[1], 50.0
    )  # Second rebinned bin should have sum of next 5 original bins
    np.testing.assert_allclose(result.data.values[2], 20.0)  # Last rebinned bin should have sum of last 2 original bins
    np.testing.assert_allclose(result.variances[0], 50.0)  # Variance of first rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[1], 50.0)  # Variance of second rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[2], 20.0)  # Variance of last rebinned bin should sum correctly

    # rebin by a width of 100, which should combine all TOF bins into one
    result = rebin_tof(data, width=100)
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [0, 30000])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_by_time():
    """Test the rebin_tof function when rebinning by time"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # rebin by a width of 2500, which should be unchanged since each original bin is 2500 us wide
    result = rebin_tof(data, width=2500, unit="time")
    assert sc.identical(result, data)  # Should be unchanged

    # rebin by a width of 5000, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, width=5000, unit="time")
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly

    # rebin by a width of 7500, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, width=7500, unit="time")
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly

    # rebin by a width of 12500, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, width=12500, unit="time")
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [0, 12500, 25000, 37500])  # New edges
    np.testing.assert_allclose(
        result.data.values[0], 50.0
    )  # First rebinned bin should have sum of first 5 original bins
    np.testing.assert_allclose(
        result.data.values[1], 50.0
    )  # Second rebinned bin should have sum of next 5 original bins
    np.testing.assert_allclose(result.data.values[2], 20.0)  # Last rebinned bin should have sum of last 2 original bins
    np.testing.assert_allclose(result.variances[0], 50.0)  # Variance of first rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[1], 50.0)  # Variance of second rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[2], 20.0)  # Variance of last rebinned bin should sum correctly

    # rebin by a width of 250000, which should combine all TOF bins into one
    result = rebin_tof(data, width=250000, unit="time")
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [0, 250000])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_by_wavelength():
    """Test the rebin_tof function when rebinning by wavelength"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 0, 30000, num=13, unit="us"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    tof_bin_width = 2500  # Each original TOF bin is 2500 us wide
    wavelength_bin_width = (
        sc.scalar(tof_bin_width, unit="us") * sc.constants.h / (sc.constants.m_n * sc.scalar(25.0, unit="m"))
    )  # Convert to wavelength bin width using l_source_to_detector = 25 m
    one_wavelength_bin = sc.to_unit(wavelength_bin_width, "Angstrom").value

    # rebin based on original bin width in wavelength, which should be unchanged
    result = rebin_tof(data, width=one_wavelength_bin, unit="wavelength")
    # assert sc.identical(result, data)  # Should be unchanged

    # rebin by a width of 5000, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, width=one_wavelength_bin * 2, unit="wavelength")
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly

    # rebin by a width of 7500, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, width=one_wavelength_bin * 3, unit="wavelength")
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(0, 30000, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly

    # rebin by a width of 12500, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, width=one_wavelength_bin * 5, unit="wavelength")
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [0, 12500, 25000, 37500])  # New edges
    np.testing.assert_allclose(
        result.data.values[0], 50.0
    )  # First rebinned bin should have sum of first 5 original bins
    np.testing.assert_allclose(
        result.data.values[1], 50.0
    )  # Second rebinned bin should have sum of next 5 original bins
    np.testing.assert_allclose(result.data.values[2], 20.0)  # Last rebinned bin should have sum of last 2 original bins
    np.testing.assert_allclose(result.variances[0], 50.0)  # Variance of first rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[1], 50.0)  # Variance of second rebinned bin should sum correctly
    np.testing.assert_allclose(result.variances[2], 20.0)  # Variance of last rebinned bin should sum correctly

    # rebin by a width of 250000, which should combine all TOF bins into one
    result = rebin_tof(data, width=one_wavelength_bin * 625, unit="wavelength")
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [0, 1562500])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_invalid():
    """Test that rebin_tof raises an error for invalid rebinning widths"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 1e5, 1e7, num=13, unit="ns"),
        },
    )
    data.variances = values

    with pytest.raises(ValueError):
        rebin_tof(data, width=0)  # Invalid width (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, width=-1)  # Invalid width (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, width=2, tof_dim="invalid_dim")  # Invalid TOF dimension

    with pytest.raises(ValueError):
        rebin_tof(data, width=2, unit="invalid_unit")  # Invalid unit for width

    with pytest.raises(ValueError):
        rebin_tof(data, width=2, unit="bins", logarithmic=True)  # Logarithmic binning not supported for 'bins' unit
