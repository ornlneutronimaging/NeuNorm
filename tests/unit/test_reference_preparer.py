"""
Unit tests for reference_preparer
"""

import numpy as np
import pytest
import scipp as sc


def test_prepare_reference_mean():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. Depending on which pixel, mean is 50 or 102, median is 40 or 102
    stack_data = np.tile(
        [[10, 40, 100], [101, 102, 103]], (4, 2, 1)
    ).T  # this creates an array with shape (N_image=3, y=4, x=4)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=stack_data, unit="counts", dtype="float64"),
    )
    stack.variances = stack.values.copy()  # Poisson

    prepared = prepare_reference(stack, method="mean", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 50 and 102
    expected_values = np.tile([50.0, 102.0], (4, 2)).T
    np.testing.assert_allclose(prepared.values, expected_values)

    # Variance should be propagated. (10+40+100)/3^2 = 150/9, or (101+102+103)/3^2 = 34.
    expected_variance = np.tile([150 / 9, 34], (4, 2)).T
    np.testing.assert_allclose(prepared.variances, expected_variance)


def test_prepare_reference_median():
    """Test basic reference preparation: median"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. Depending on which pixel, mean is 50 or 102, median is 40 or 102
    stack_data = np.tile(
        [[10, 40, 100], [101, 102, 103]], (4, 2, 1)
    ).T  # this creates an array with shape (N_image=3, y=4, x=4)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=stack_data, unit="counts", dtype="float64"),
    )
    stack.variances = stack.values.copy()  # Poisson

    prepared = prepare_reference(stack, method="median", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 40 and 102
    expected_values = np.tile([40.0, 102.0], (4, 2)).T
    np.testing.assert_allclose(prepared.values, expected_values)

    # Variance should be propagated. Variance should be approximately (π/2) * (the variance from the mean calculation).
    expected_variance = np.tile([np.pi / 2 * 150 / 9, np.pi / 2 * 34], (4, 2)).T
    np.testing.assert_allclose(prepared.variances, expected_variance)


def test_prepare_reference_median_no_variance():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. mean is 50, median is 40
    stack_data = np.tile([10, 40, 100], (5, 5, 1)).T  # this creates an array with shape (3, 5, 5)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=stack_data, unit="counts", dtype="float64"),
    )

    prepared = prepare_reference(stack, method="median", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 40
    np.testing.assert_allclose(prepared.values, 40.0)

    # Variance should be None since input had no variance
    assert prepared.variances is None


def test_prepare_reference_2d():
    """Test that 2D input is returned unchanged."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=np.full((5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    prepared = prepare_reference(data, method="mean", dim="N_image")

    assert data is prepared  # Should be the same object


def test_prepare_reference_3d_single_frame_mean():
    """Test that 3D input with N_image=1 is returned as 2D."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((1, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    assert data.dims == ("N_image", "x", "y")

    prepared = prepare_reference(data, method="mean", dim="N_image")

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values and variances should be 42
    np.testing.assert_allclose(prepared.values, 42.0)
    np.testing.assert_allclose(prepared.variances, 42.0)


def test_prepare_reference_3d_single_frame_median():
    """Test that 3D input with N_image=1 is returned as 2D."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((1, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    assert data.dims == ("N_image", "x", "y")

    prepared = prepare_reference(data, method="median", dim="N_image")

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 42. Variance should be approximately (π/2) * 42.
    np.testing.assert_allclose(prepared.values, 42.0)
    np.testing.assert_allclose(prepared.variances, 42.0 * np.pi / 2, atol=1)


def test_wrong_dim():
    """Test that wrong dimension raises error."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((3, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    with pytest.raises(ValueError, match="Dimension 'wrong_dim' not found in input data"):
        prepare_reference(data, method="mean", dim="wrong_dim")


def test_wrong_method():
    """Test that wrong method raises error."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((3, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    with pytest.raises(ValueError, match="Unsupported method 'wrong_method'"):
        prepare_reference(data, method="wrong_method", dim="N_image")


def test_prepare_reference_different_variances():
    """Test that returns different variances for different input variances. Same values."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((3, 2, 2), 42.0), unit="counts", dtype="float64"),
    )
    # Create different variances for (x, y)
    variances = np.tile([[10, 20], [30, 40]], (3, 1, 1))  # this creates an array with shape (N_image=3, y=2, x=2)
    data.variances = variances

    prepared_mean = prepare_reference(data, method="mean", dim="N_image")
    prepared_median = prepare_reference(data, method="median", dim="N_image")

    # values should be the same
    np.testing.assert_allclose(prepared_mean.values, 42.0)
    np.testing.assert_allclose(prepared_median.values, 42.0)

    # variances should be different
    expected_mean_variance = variances.sum(axis=0) / 9  # mean variance along N_image
    np.testing.assert_allclose(prepared_mean.variances, expected_mean_variance)
    np.testing.assert_allclose(prepared_median.variances, expected_mean_variance * np.pi / 2)
