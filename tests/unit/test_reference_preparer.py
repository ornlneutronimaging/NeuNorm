"""
Unit tests for reference_preparer
"""

import numpy as np
import scipp as sc


def test_prepare_reference_mean():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. mean is 50, median is 40
    stack_data = np.tile([10, 40, 100], (5, 5, 1)).T  # this creates an array with shape (3, 5, 5)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "y", "x"], values=stack_data, unit="counts", dtype="float64"),
    )
    stack.variances = stack.values.copy()  # Poisson

    prepared = prepare_reference(stack, method="mean", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # Values should be 50
    np.testing.assert_allclose(prepared.values, 50.0)

    # Variance should be propagated. (10+40+100)/3^2 = 150/9
    np.testing.assert_allclose(prepared.variances, 150.0 / 9.0)


def test_prepare_reference_median():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. mean is 50, median is 40
    stack_data = np.tile([10, 40, 100], (5, 5, 1)).T  # this creates an array with shape (3, 5, 5)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "y", "x"], values=stack_data, unit="counts", dtype="float64"),
    )
    stack.variances = stack.values.copy()  # Poisson

    prepared = prepare_reference(stack, method="median", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # Values should be 40
    np.testing.assert_allclose(prepared.values, 40.0)

    # Variance should be propagated. It's an approximation but should be around 40.0.
    np.testing.assert_allclose(prepared.variances, 40.0, atol=1)


def test_prepare_reference_median_no_variance():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. mean is 50, median is 40
    stack_data = np.tile([10, 40, 100], (5, 5, 1)).T  # this creates an array with shape (3, 5, 5)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "y", "x"], values=stack_data, unit="counts", dtype="float64"),
    )

    prepared = prepare_reference(stack, method="median", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # Values should be 40
    np.testing.assert_allclose(prepared.values, 40.0)

    # Variance should be None since input had no variance
    assert prepared.variances is None


def test_prepare_reference_2d():
    """Test that 2D input is returned unchanged."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["y", "x"], values=np.full((5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    prepared = prepare_reference(data, method="mean", dim="N_image")

    assert data is prepared  # Should be the same object
