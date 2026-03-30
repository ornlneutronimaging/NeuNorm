"""
Unit tests for the HistogramRebinner Module
"""

import numpy as np
import pytest
import scipp as sc


def test_rebin_tof():
    """Test the rebin_tof function"""
    from neunorm.tof.histogram_rebinner import rebin_tof

    values = np.full((12, 5, 5), 10.0)  # 12 TOF bins, 5x5 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 1e5, 1e7, num=13, unit="ns"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # rebin by a factor of 1, which should return the original data
    result = rebin_tof(data, factor=1)
    assert result is data  # Should return the original data without modification

    # rebin by a factor of 2, which should combine every 2 adjacent TOF bins into one
    result = rebin_tof(data, factor=2)
    assert result.shape == (6, 5, 5)  # Should have half the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(1e5, 1e7, num=7))  # New edges
    np.testing.assert_allclose(result.data.values, 20.0)  # Each rebinned bin should have sum of 2 original bins
    np.testing.assert_allclose(result.variances, 20.0)  # Variance should also sum correctly

    # rebin by a factor of 3, which should combine every 3 adjacent TOF bins into one
    result = rebin_tof(data, factor=3)
    assert result.shape == (4, 5, 5)  # Should have a third of the number of TOF bins
    np.testing.assert_allclose(result.coords["tof"].values, np.linspace(1e5, 1e7, num=5))  # New edges
    np.testing.assert_allclose(result.data.values, 30.0)  # Each rebinned bin should have sum of 3 original bins
    np.testing.assert_allclose(result.variances, 30.0)  # Variance should also sum correctly

    # rebin by a factor of 5, we will have 2 full bins and one partial bin (last 2 original bins combined)
    result = rebin_tof(data, factor=5)
    assert result.shape == (3, 5, 5)  # Should have 3 TOF bins (2 full + 1 partial)
    np.testing.assert_allclose(result.coords["tof"].values, [1e5, 4.225e6, 8.35e6, 1e7])  # New edges
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

    # rebin by a factor of 100, which should combine all TOF bins into one
    result = rebin_tof(data, factor=100)
    assert result.shape == (1, 5, 5)  # Should have only one TOF bin
    np.testing.assert_allclose(result.coords["tof"].values, [1e5, 1e7])  # New edges
    np.testing.assert_allclose(result.data.values, 120.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 120.0)  # Variance should also sum correctly


def test_rebin_tof_invalid():
    """Test that rebin_tof raises an error for invalid rebinning factors"""
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
        rebin_tof(data, factor=0)  # Invalid factor (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, factor=-1)  # Invalid factor (non-positive)

    with pytest.raises(ValueError):
        rebin_tof(data, factor=2, tof_dim="invalid_dim")  # Invalid TOF dimension
