"""
Unit tests for dark current correction.
"""

import numpy as np
import scipp as sc


def test_subtract_dark_basic_3d():
    """Test basic dark correction: data - dark"""
    from neunorm.processing.dark_corrector import subtract_dark

    # Create simple sample and dark histograms
    sample_data = np.ones((10, 5, 5)) * 100.0
    dark_data = np.ones((10, 5, 5)) * 10.0

    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    dark = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=dark_data, unit="counts", dtype="float64"),
    )

    dark.variances = dark.values.copy()

    corrected = subtract_dark(sample, dark)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 90
    np.testing.assert_allclose(corrected.values, 90.0)

    # Variance should be propagated. 100 (sample) + 10 (dark) = 110
    np.testing.assert_allclose(corrected.variances, 110.0)


def test_subtract_dark_basic_2d():
    """Test basic dark correction: data - dark"""
    from neunorm.processing.dark_corrector import subtract_dark

    # Create simple sample and dark histograms
    sample_data = np.ones((10, 5, 5)) * 100.0
    dark_data = np.ones((5, 5)) * 10.0

    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    dark = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=dark_data, unit="counts", dtype="float64"),
    )
    dark.variances = dark.values.copy()

    corrected = subtract_dark(sample, dark)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # Values should be 90
    np.testing.assert_allclose(corrected.values, 90.0)

    # Variance should be propagated. 100 (sample) + 10 (dark) = 110
    np.testing.assert_allclose(corrected.variances, 110.0)


def test_subtract_dark_basic_clip_negative():
    """Test basic dark correction: data - dark"""
    from neunorm.processing.dark_corrector import subtract_dark

    # Create simple sample and dark histograms
    sample_data = np.ones((3, 2, 2)) * 100.0
    dark_data = np.ones((3, 2, 2)) * 10.0
    dark_data[1, 1, 1] = 150.0  # This will produce a negative value after subtraction
    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    dark = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=dark_data, unit="counts", dtype="float64"),
    )

    dark.variances = dark.values.copy()

    corrected = subtract_dark(sample, dark, clip_negative=True)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # check masked value
    assert "negative" not in corrected.masks  # Should be clipped, not masked

    # Values should be 90 except for the clipped value which should be 0
    expected_values = np.ones((3, 2, 2)) * 90.0
    expected_values[1, 1, 1] = 0.0
    np.testing.assert_allclose(corrected.values, expected_values)

    # Variance should be propagated
    expected_variances = np.ones((3, 2, 2)) * 110.0  # 100 (sample) + 10 (dark)
    expected_variances[1, 1, 1] = 250.0  # 100 (sample) + 150 (dark)
    np.testing.assert_allclose(corrected.variances, expected_variances)


def test_subtract_dark_basic_masked():
    """Test basic dark correction: data - dark"""
    from neunorm.processing.dark_corrector import subtract_dark

    # Create simple sample and dark histograms
    sample_data = np.ones((3, 2, 2)) * 100.0
    dark_data = np.ones((3, 2, 2)) * 10.0
    dark_data[1, 1, 1] = 150.0  # This will produce a negative value after subtraction
    sample = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
    )
    sample.variances = sample.values.copy()  # Poisson

    dark = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=dark_data, unit="counts", dtype="float64"),
    )

    dark.variances = dark.values.copy()

    corrected = subtract_dark(sample, dark, clip_negative=False)

    # Should still be in counts
    assert corrected.unit == sc.units.counts

    # check masked value
    expected_mask = np.zeros((3, 2, 2), dtype=bool)
    expected_mask[1, 1, 1] = True
    assert "negative" in corrected.masks
    assert np.array_equal(corrected.masks["negative"].values, expected_mask)

    # Values should be 90 except for the masked value which should be -50
    expected_values = np.ones((3, 2, 2)) * 90.0
    expected_values[1, 1, 1] = -50.0  # 100 (sample) - 150 (dark)
    np.testing.assert_allclose(corrected.values, expected_values)

    # Variance should be propagated
    expected_variances = np.ones((3, 2, 2)) * 110.0  # 100 (sample) + 10 (dark)
    expected_variances[1, 1, 1] = 250.0  # 100 (sample) + 150 (dark)
    np.testing.assert_allclose(corrected.variances, expected_variances)
