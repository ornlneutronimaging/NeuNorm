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
    sample_data[5][1] = 11.0  # Add some small difference so that the std is non-zero
    sample_data[5, 2, 2] = 100.0  # Add a gamma spike, should be replaced by median of 10

    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    corrected = apply_gamma_filter(sample, preserve_variance=False)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 10 except for one row of 11. But the 100 should be replaced by 10.
    expected_values = np.full((10, 5, 5), 10.0)
    expected_values[5][1] = 11.0
    np.testing.assert_allclose(corrected.values, expected_values)

    # Variance should be updated to local median variance.
    expected_variance = np.full((10, 5, 5), 10.0)
    expected_variance[5][1] = 11.0  # This pixel is not an outlier, so variance should be unchanged.
    # This pixel is an outlier. 8 neighbuts, 5 with variance 10, 3 with variance 11.
    # So mean variance is (10*5 + 11*3) / 8.
    # Variance of the median is approximately (π / (2n)) * mean_variance = (π / 16) * mean_variance.
    expected_variance[5, 2, 2] = np.pi * (10 * 5 + 11 * 3) / 128
    np.testing.assert_allclose(corrected.variances, expected_variance)


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
    expected_variance = np.full((5, 5), 10.0)
    # This pixel is an outlier. Variance will be updated to the local median variance.
    # 8 neighbors with variance 10, so median variance is 10.
    # Variance of the median is approximately (π / (2n)) * mean_variance = (π / 16) * 10.
    expected_variance[2, 2] = 10.0 * np.pi / 16
    np.testing.assert_allclose(corrected.variances, expected_variance)


def test_apply_gamma_filter_edges():
    """Test basic gamma filtering with outliers at the edges."""
    from neunorm.filters.gamma_filter import apply_gamma_filter

    # Create simple sample data
    sample_data = np.full((5, 5), 10.0)
    sample_data[0, 2] = 100.0  # Add a gamma spike at the edge, should be replaced by median of 10

    sample = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    corrected = apply_gamma_filter(sample, preserve_variance=False, threshold_sigma=1)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 10
    np.testing.assert_allclose(corrected.values, 10.0)

    # Variance should be propagated.
    expected_variance = np.full((5, 5), 10.0)
    # This pixel is an outlier. Variance will be updated to the local median variance.
    # 7 neighbors with variance 10, 1 with 100. So mean variance is (10*7 + 100) / 8.
    # Variance of the median is approximately (π / (2n)) * mean_variance = (π / 16) * mean_variance.
    expected_variance[0, 2] = (10 * 7 + 100) / 8 * np.pi / 16
    np.testing.assert_allclose(corrected.variances, expected_variance)
