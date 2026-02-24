"""
Unit tests for gamma filtering.
"""

import numpy as np
import scipp as sc


def test_apply_gamma_filter_3d():
    """Test gamma filtering on 3D data."""
    from neunorm.filters.gamma_filter import apply_gamma_filter

    # Create simple sample data
    sample_data = np.full((10, 5, 5), 10.0)
    sample_data[5, 2, 2] = 100.0  # Add a gamma spike, should be replaced by median of 10

    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    corrected = apply_gamma_filter(sample, preserve_variance=False)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 10
    np.testing.assert_allclose(corrected.values, 10.0)

    # Variance should be updated to local median variance.
    np.testing.assert_allclose(corrected.variances, 10.0)


def test_apply_gamma_filter_high_threshold():
    """Test gamma filtering on data with a high threshold."""
    from neunorm.filters.gamma_filter import apply_gamma_filter

    # Create simple sample data
    sample_data = np.full((10, 5, 5), 10.0)
    sample_data[5][1] = 11.0  # Add some small difference so that the std is non-zero
    sample_data[5, 2, 2] = 100.0  # Add a gamma spike, should be replaced by median of 10

    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    corrected = apply_gamma_filter(sample, threshold_sigma=1000.0)

    # Nothing should be replaced since threshold is very high
    np.testing.assert_equal(corrected.values, sample_data)
    np.testing.assert_equal(corrected.variances, sample_data)


def test_apply_gamma_filter_basic_3d_preserve_variance():
    """Test gamma filtering on 3D data with variance preservation (preserve_variance=True)."""
    from neunorm.filters.gamma_filter import apply_gamma_filter

    # Create simple sample data
    sample_data = np.full((10, 5, 5), 10.0)
    sample_data[5, 2, 2] = 100.0  # Add a gamma spike, should be replaced by median of 10

    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    corrected = apply_gamma_filter(sample, preserve_variance=True)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 10
    np.testing.assert_allclose(corrected.values, 10.0)

    # Variance should be preserved.
    expected_variance = np.full((10, 5, 5), 10.0)
    expected_variance[5, 2, 2] = 100.0  # Should still be the original variance, not the local median variance
    np.testing.assert_allclose(corrected.variances, expected_variance)


def test_apply_gamma_filter_basic_2d():
    """Test basic gamma filtering on 2D data."""
    from neunorm.filters.gamma_filter import apply_gamma_filter

    # Create simple sample data
    sample_data = np.full((5, 5), 10.0)
    sample_data[2, 2] = 100.0  # Add a gamma spike, should be replaced by median of 10

    sample = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    corrected = apply_gamma_filter(sample, preserve_variance=False)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 10
    np.testing.assert_allclose(corrected.values, 10.0)

    # Variance should be propagated.
    np.testing.assert_allclose(corrected.variances, 10.0)
