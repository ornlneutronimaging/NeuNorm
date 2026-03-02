"""
Unit tests for the ROI clipper module.
"""

import numpy as np
import pytest
import scipp as sc


def test_apply_roi_2d():
    """
    Test the apply_roi function for cropping spatial dimensions to a region of interest (ROI).
    """
    from neunorm.processing.roi_clipper import apply_roi

    values = np.random.random((5, 5))  # Shape (x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = data.values.copy()

    # Define ROI: (x0, y0, x1, y1)
    roi = (0, 3, 4, 4)  # This should crop to a 4x1 region
    cropped = apply_roi(data, roi)
    # Check that unit is preserved
    assert cropped.unit == sc.units.counts
    # Check that values and variances are correctly cropped
    assert cropped.values.shape == (4, 1)
    expected_values = values[0:4, 3:4]
    np.testing.assert_equal(cropped.values, expected_values)
    np.testing.assert_equal(cropped.variances, expected_values)

    # Check dims
    assert cropped.dims == ("x", "y")
    # Check that coordinates are updated correctly
    np.testing.assert_equal(cropped.coords["x"].values, [0, 1, 2, 3])
    np.testing.assert_equal(cropped.coords["y"].values, [3])


def test_apply_roi_3d():
    """
    Test the apply_roi function for cropping spatial dimensions to a region of interest (ROI).
    """
    from neunorm.processing.roi_clipper import apply_roi

    values = np.random.random((3, 5, 5))  # Shape (N_image=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "N_image": sc.arange("N_image", 3, unit=None),
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = data.values.copy()

    # Define ROI: (x0, y0, x1, y1)
    roi = (0, 3, 4, 4)  # This should crop to a 4x1 region in the x and y dimensions, preserving N_image dimension
    cropped = apply_roi(data, roi)
    # Check that unit is preserved
    assert cropped.unit == sc.units.counts
    # Check that values and variances are correctly cropped
    assert cropped.values.shape == (3, 4, 1)
    expected_values = values[:, 0:4, 3:4]
    np.testing.assert_equal(cropped.values, expected_values)
    np.testing.assert_equal(cropped.variances, expected_values)

    # Check dims
    assert cropped.dims == ("N_image", "x", "y")
    # Check that coordinates are updated correctly
    np.testing.assert_equal(cropped.coords["x"].values, [0, 1, 2, 3])
    np.testing.assert_equal(cropped.coords["y"].values, [3])
    # N_image coordinate should be unchanged
    np.testing.assert_equal(cropped.coords["N_image"].values, [0, 1, 2])


def test_apply_roi_4d():
    """
    Test the apply_roi function for cropping spatial dimensions to a region of interest (ROI).
    """
    from neunorm.processing.roi_clipper import apply_roi

    values = np.random.random((3, 5, 5, 2))  # Shape (N_image=3, x=5, y=5, z=2)
    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y", "z"], values=values, unit="counts", dtype="float64"),
        coords={
            "z": sc.arange("z", 2, unit=None),
            "N_image": sc.arange("N_image", 3, unit=None),
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = data.values.copy()

    # Define ROI: (x0, y0, x1, y1)
    roi = (0, 3, 4, 4)  # This should crop to a 4x1 region in the x and y dimensions, preserving N_image dimension
    cropped = apply_roi(data, roi)
    # Check that unit is preserved
    assert cropped.unit == sc.units.counts
    # Check that values and variances are correctly cropped
    assert cropped.values.shape == (3, 4, 1, 2)
    expected_values = values[:, 0:4, 3:4, :]
    np.testing.assert_equal(cropped.values, expected_values)
    np.testing.assert_equal(cropped.variances, expected_values)

    # Check dims
    assert cropped.dims == ("N_image", "x", "y", "z")
    # Check that coordinates are updated correctly
    np.testing.assert_equal(cropped.coords["x"].values, [0, 1, 2, 3])
    np.testing.assert_equal(cropped.coords["y"].values, [3])
    # N_image and z coordinate should be unchanged
    np.testing.assert_equal(cropped.coords["N_image"].values, [0, 1, 2])
    np.testing.assert_equal(cropped.coords["z"].values, [0, 1])


def test_apply_roi_invalid_roi():
    """
    Test that apply_roi raises an error for invalid ROI specifications.
    """
    from neunorm.processing.roi_clipper import apply_roi

    values = np.random.random((5, 5))  # Shape (x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )

    # Define invalid ROI: (x0, y0, x1, y1) where x1 <= x0
    roi_invalid = (3, 0, 2, 4)
    with pytest.raises(ValueError, match="Invalid ROI"):
        apply_roi(data, roi_invalid)

    # Define invalid ROI: (x0, y0, x1, y1) where y1 <= y0
    roi_invalid = (0, 3, 4, 2)
    with pytest.raises(ValueError, match="Invalid ROI"):
        apply_roi(data, roi_invalid)

    # Define invalid ROI: negative coordinates
    roi_invalid = (-1, -1, 4, 4)
    with pytest.raises(ValueError, match="Invalid ROI"):
        apply_roi(data, roi_invalid)

    # Define invalid ROI: exceeds data size
    roi_invalid = (0, 0, 6, 6)
    with pytest.raises(ValueError, match="ROI .* exceeds data size"):
        apply_roi(data, roi_invalid)


def test_apply_roi_no_spatial_dims():
    """
    Test that apply_roi raises an error when the DataArray does not have 'x' and 'y' dimensions.
    """
    from neunorm.processing.roi_clipper import apply_roi

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
        apply_roi(data, roi)
