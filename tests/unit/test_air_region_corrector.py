"""
Unit tests for air region correction.
"""

import numpy as np
import pytest
import scipp as sc


def test_apply_air_region_correction():
    """Test air region correction on 3D data."""
    from neunorm.processing.air_region_corrector import apply_air_region_correction

    # Create simple sample data
    sample_data = np.full((10, 16, 16), 10.0)
    # set higher values in region
    sample_data[:, 6:11, 6:11] = 20.0

    transmission = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"tof": sc.arange("tof", 10, unit="ms")},
    )
    transmission.variances = transmission.values.copy()  # Poisson

    corrected = apply_air_region_correction(transmission, air_roi=(6, 6, 11, 11))

    # Check dims and shape are the same as input
    assert corrected.dims == ("tof", "x", "y")
    assert corrected.values.shape == (10, 16, 16)

    # Values in air region should be scaled to 1.0, so values outside should be 10/20 = 0.5
    # and values inside should be 20/20 = 1.0.
    expected_values = np.full((10, 16, 16), 0.5)
    expected_values[:, 6:11, 6:11] = 1.0
    np.testing.assert_allclose(corrected.values, expected_values)

    # Check variance. Air region mean is 20 and the variance of the mean is 0.8.
    # So the variance outside the air region should be (0.5^2) * (1/10 + 0.8/20^2) = 0.0255
    # and the variance inside the air region should be (1.0^2) * (1/20 + 0.8/20^2) = 0.052
    expected_variance = np.full((10, 16, 16), 0.0255)
    expected_variance[:, 6:11, 6:11] = 0.052
    np.testing.assert_allclose(corrected.variances, expected_variance)


def test_apply_air_region_correction_2d():
    """Test air region correction on 2D data."""
    from neunorm.processing.air_region_corrector import apply_air_region_correction

    # Create simple sample data
    sample_data = np.full((16, 16), 10.0)
    # set higher values in region
    sample_data[6:11, 6:11] = 20.0

    transmission = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    transmission.variances = transmission.values.copy()  # Poisson

    corrected = apply_air_region_correction(transmission, air_roi=(6, 6, 11, 11))

    # Check dims and shape are the same as input
    assert corrected.dims == ("x", "y")
    assert corrected.values.shape == (16, 16)

    # Values in air region should be scaled to 1.0, so values outside should be 10/20 = 0.5
    # and values inside should be 20/20 = 1.0.
    expected_values = np.full((16, 16), 0.5)
    expected_values[6:11, 6:11] = 1.0
    np.testing.assert_allclose(corrected.values, expected_values)

    # Check variance. Air region mean is 20 and the variance of the mean is 0.8.
    # So the variance outside the air region should be (0.5^2) * (1/10 + 0.8/20^2) = 0.0255
    # and the variance inside the air region should be (1.0^2) * (1/20 + 0.8/20^2) = 0.052
    expected_variance = np.full((16, 16), 0.0255)
    expected_variance[6:11, 6:11] = 0.052
    np.testing.assert_allclose(corrected.variances, expected_variance)


def test_apply_air_region_correction_invalid_roi():
    """
    Test that apply_air_region_correction raises an error for invalid ROI specifications.
    """
    from neunorm.processing.air_region_corrector import apply_air_region_correction

    values = np.random.random((5, 5))  # Shape (x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=values, unit="counts", dtype="float64"),
    )

    # Define invalid ROI: only x0 and y0 provided
    roi_invalid = (0, 0)
    with pytest.raises(ValueError, match="ROI must be a tuple of 4 integers"):
        apply_air_region_correction(data, roi_invalid)

    # Define invalid ROI: non-integer values
    roi_invalid = (0.5, 0.5, 4.5, 4.5)
    with pytest.raises(ValueError, match="ROI must be a tuple of 4 integers"):
        apply_air_region_correction(data, roi_invalid)

    # Define invalid ROI: (x0, y0, x1, y1) where x1 <= x0
    roi_invalid = (3, 0, 2, 4)
    with pytest.raises(ValueError, match="Invalid ROI"):
        apply_air_region_correction(data, roi_invalid)

    # Define invalid ROI: (x0, y0, x1, y1) where y1 <= y0
    roi_invalid = (0, 3, 4, 2)
    with pytest.raises(ValueError, match="Invalid ROI"):
        apply_air_region_correction(data, roi_invalid)

    # Define invalid ROI: negative coordinates
    roi_invalid = (-1, -1, 4, 4)
    with pytest.raises(ValueError, match="Invalid ROI"):
        apply_air_region_correction(data, roi_invalid)

    # Define invalid ROI: exceeds data size
    roi_invalid = (0, 0, 6, 6)
    with pytest.raises(ValueError, match="ROI .* exceeds data size"):
        apply_air_region_correction(data, roi_invalid)


def test_apply_air_region_correction_no_spatial_dims():
    """
    Test that apply_air_region_correction raises an error when the DataArray does not have 'x' and 'y' dimensions.
    """
    from neunorm.processing.air_region_corrector import apply_air_region_correction

    values = np.random.random((3, 4))  # Shape (N_image=3, time=4)
    data = sc.DataArray(
        data=sc.array(dims=["N_image", "time"], values=values, unit="counts", dtype="float64"),
        coords={
            "N_image": sc.arange("N_image", 3, unit=None),
            "time": sc.arange("time", 4, unit=None),
        },
    )

    roi = (0, 0, 2, 2)  # ROI is irrelevant since there are no spatial dimensions
    with pytest.raises(ValueError, match="DataArray must have 'x' and 'y' dimensions"):
        apply_air_region_correction(data, roi)
