"""
Unit tests for the Spatial Rebinner module.
"""

import numpy as np
import pytest
import scipp as sc


def test_rebin_spatial_2d():
    """Test the rebin_spatial function on a 2D array"""
    from neunorm.processing.spatial_rebinner import rebin_spatial

    values = np.full((16, 16), 10.0)  # 16x16 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=values, unit="counts", dtype="float64"),
    )
    data.variances = values

    # rebin by a factor of 1, which should return the original data
    result = rebin_spatial(data, factor=1)
    assert result is data  # Should return the original data without modification

    # rebin by a factor of 2, which should combine every 2x2 spatial pixels into one
    result = rebin_spatial(data, factor=2)
    assert result.shape == (8, 8)  # Should have half the number of spatial pixels
    np.testing.assert_allclose(result.data.values, 40.0)  # Each rebinned bin should have sum of 4 original bins
    np.testing.assert_allclose(result.variances, 40.0)  # Variance should also sum correctly
    assert result.dims == ("x", "y")  # Should preserve dimension names

    # rebin by a factor of 16, which should combine all spatial pixels into one
    result = rebin_spatial(data, factor=16)
    assert result.shape == (1, 1)  # Should have only one spatial bin
    np.testing.assert_allclose(result.data.values, 2560.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 2560.0)  # Variance should also sum correctly

    # rebin by a factor of (2, 4), which should combine every 2 pixels in x and 4 pixels in y
    result = rebin_spatial(data, factor=(2, 4))
    assert result.shape == (8, 4)  # Should have 8 bins in x and 4 bins in y
    np.testing.assert_allclose(result.data.values, 80.0)  # Each rebinned bin should have sum of 8 original bins
    np.testing.assert_allclose(result.variances, 80.0)  # Variance should also sum correctly


def test_rebin_spatial_3d():
    """Test the rebin_spatial function on a 3D array"""
    from neunorm.processing.spatial_rebinner import rebin_spatial

    values = np.full((3, 16, 16), 10.0)  # 3 TOF bins, 16x16 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
        },
    )
    data.variances = values

    # rebin by a factor of 1, which should return the original data
    result = rebin_spatial(data, factor=1)
    assert result is data  # Should return the original data without modification

    # rebin by a factor of 2, which should combine every 2x2 spatial pixels into one
    result = rebin_spatial(data, factor=2)
    assert result.shape == (3, 8, 8)  # Should have half the number of spatial pixels
    np.testing.assert_allclose(result.data.values, 40.0)  # Each rebinned bin should have sum of 4 original bins
    np.testing.assert_allclose(result.variances, 40.0)  # Variance should also sum correctly
    assert result.dims == ("tof", "x", "y")  # Should preserve dimension names
    assert sc.identical(
        result.coords["tof"], data.coords["tof"]
    )  # Should preserve coordinates for non-spatial dimensions

    # rebin by a factor of 16, which should combine all spatial pixels into one
    result = rebin_spatial(data, factor=16)
    assert result.shape == (3, 1, 1)  # Should have only one spatial bin
    np.testing.assert_allclose(result.data.values, 2560.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 2560.0)  # Variance should also sum correctly
    assert result.dims == ("tof", "x", "y")  # Should preserve dimension names
    assert sc.identical(
        result.coords["tof"], data.coords["tof"]
    )  # Should preserve coordinates for non-spatial dimensions

    # rebin by a factor of (2, 4), which should combine every 2 pixels in x and 4 pixels in y
    result = rebin_spatial(data, factor=(2, 4))
    assert result.shape == (3, 8, 4)  # Should have 8 bins in x and 4 bins in y
    np.testing.assert_allclose(result.data.values, 80.0)  # Each rebinned bin should have sum of 8 original bins
    np.testing.assert_allclose(result.variances, 80.0)  # Variance should also sum correctly


def test_rebin_spatial_4d():
    """Test the rebin_spatial function on a 4D array"""
    from neunorm.processing.spatial_rebinner import rebin_spatial

    values = np.full((3, 16, 16, 2), 10.0)  # 3 TOF bins, 16x16 spatial pixels, 2 additional dimension
    data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y", "z"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof": sc.linspace("tof", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "z": sc.array(dims=["z"], values=[0, 1]),  # Additional dimension to test preservation
        },
    )
    data.variances = values

    # rebin by a factor of 1, which should return the original data
    result = rebin_spatial(data, factor=1)
    assert result is data  # Should return the original data without modification

    # rebin by a factor of 2, which should combine every 2x2 spatial pixels into one
    result = rebin_spatial(data, factor=2)
    assert result.shape == (3, 8, 8, 2)  # Should have half the number of spatial pixels
    np.testing.assert_allclose(result.data.values, 40.0)  # Each rebinned bin should have sum of 4 original bins
    np.testing.assert_allclose(result.variances, 40.0)  # Variance should also sum correctly
    assert result.dims == ("tof", "x", "y", "z")  # Should preserve dimension names
    assert sc.identical(
        result.coords["tof"], data.coords["tof"]
    )  # Should preserve coordinates for non-spatial dimensions
    assert sc.identical(result.coords["z"], data.coords["z"])  # Should preserve coordinates for non-spatial dimensions

    # rebin by a factor of 16, which should combine all spatial pixels into one
    result = rebin_spatial(data, factor=16)
    assert result.shape == (3, 1, 1, 2)  # Should have only one spatial bin
    np.testing.assert_allclose(result.data.values, 2560.0)  # All original bins combined
    np.testing.assert_allclose(result.variances, 2560.0)  # Variance should also sum correctly
    assert result.dims == ("tof", "x", "y", "z")  # Should preserve dimension names
    assert sc.identical(
        result.coords["tof"], data.coords["tof"]
    )  # Should preserve coordinates for non-spatial dimensions
    assert sc.identical(result.coords["z"], data.coords["z"])  # Should preserve coordinates for non-spatial dimensions

    # rebin by a factor of (2, 4), which should combine every 2 pixels in x and 4 pixels in y
    result = rebin_spatial(data, factor=(2, 4))
    assert result.shape == (3, 8, 4, 2)  # Should have 8 bins in x and 4 bins in y
    np.testing.assert_allclose(result.data.values, 80.0)  # Each rebinned bin should have sum of 8 original bins
    np.testing.assert_allclose(result.variances, 80.0)  # Variance should also sum correctly


def test_rebin_spatial_invalid_factors():
    """Test that invalid rebinning factors raise errors"""
    from neunorm.processing.spatial_rebinner import rebin_spatial

    values = np.full((16, 16), 10.0)  # 16x16 spatial pixels
    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=values, unit="counts", dtype="float64"),
    )
    data.variances = values

    # Test non-integer factor
    with pytest.raises(ValueError, match="Factor must be an integer or a tuple of two integers."):
        rebin_spatial(data, factor=2.5)

    # Test negative factor
    with pytest.raises(ValueError, match="Rebinning factors must be positive integers."):
        rebin_spatial(data, factor=-2)

    # Test zero factor
    with pytest.raises(ValueError, match="Rebinning factors must be positive integers."):
        rebin_spatial(data, factor=0)

    # Test non-divisible factor
    with pytest.raises(ValueError, match="Rebinning factors must divide the number of pixels in each dimension."):
        rebin_spatial(data, factor=3)

    # Test too large factor
    with pytest.raises(
        ValueError, match="Rebinning factors must be less than or equal to the number of pixels in each dimension."
    ):
        rebin_spatial(data, factor=32)

    # Test missing spatial dimensions
    with pytest.raises(ValueError, match="Input data must have spatial dimensions 'x' and 'y'."):
        rebin_spatial(sc.DataArray(data=sc.array(dims=["tof"], values=[1, 2, 3])), factor=2)
