import numpy as np
import pytest
import scipp as sc

from neunorm.tof.statistics_analyzer import analyze_statistics


def test_analyze_statistics_recommend_1():
    """Test the analyze_statistics function with synthetic data. This should not recommend rebinning."""
    sample_data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=np.ones((10, 5, 5)), unit="counts", dtype="float64"),
        coords={"tof": sc.linspace("tof", 1, 100, num=11, unit="us")},
    )

    # summing will give 25 counts per bin, so SNR = sqrt(25) = 5, which is above the threshold of 3.

    report = analyze_statistics(sample_data)
    np.testing.assert_almost_equal(report.counts_per_bin, 25)
    np.testing.assert_almost_equal(report.snr_per_bin, 5)
    assert len(report.low_statistics_bins) == 0
    assert report.recommended_rebinning == 1
    assert report.preserve_regions == []


def test_analyze_statistics_recommend_3():
    """Test the analyze_statistics function with synthetic data. This should recommend rebinning."""
    sample_data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=np.ones((12, 2, 2)), unit="counts", dtype="float64"),
        coords={"tof": sc.linspace("tof", 1, 100, num=13, unit="us")},
    )

    # summing will give 4 counts per bin, so SNR = sqrt(4) = 2, which is below the threshold of 3.

    report = analyze_statistics(sample_data)
    np.testing.assert_almost_equal(report.counts_per_bin, 4)
    np.testing.assert_almost_equal(report.snr_per_bin, 2)
    assert len(report.low_statistics_bins) == 12
    assert report.recommended_rebinning == 3  # Rebinning by 3 will give 12 counts per bin, SNR = sqrt(12) ~ 3.46 > 3
    assert report.preserve_regions == []


def test_analyze_statistics_recommend_10():
    """Test the analyze_statistics function with synthetic data. This should recommend rebinning.
    Have all zero count except one bin."""
    data = np.zeros((10, 5, 5))
    data[5, :, :] = 1  # Only the sixth TOF bin has counts, the rest are zero.
    sample_data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=data, unit="counts", dtype="float64"),
        coords={"tof": sc.linspace("tof", 1, 100, num=11, unit="us")},
    )

    # Should recommend rebinning in a single bin since all other bins are zero.

    report = analyze_statistics(sample_data)
    np.testing.assert_almost_equal(report.counts_per_bin, [0, 0, 0, 0, 0, 25, 0, 0, 0, 0])
    np.testing.assert_almost_equal(report.snr_per_bin, [0, 0, 0, 0, 0, 5, 0, 0, 0, 0])
    assert len(report.low_statistics_bins) == 9
    assert report.recommended_rebinning == 10
    assert report.preserve_regions == []


def test_analyze_statistics_raise_all_zero():
    """Test the analyze_statistics function with synthetic data. This should raise a ValueError."""
    sample_data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=np.zeros((10, 2, 2)), unit="counts", dtype="float64"),
        coords={"tof": sc.linspace("tof", 1, 100, num=11, unit="us")},
    )

    # Since all counts are zero, SNR is zero and rebinning will not help. The function should raise a ValueError.

    with pytest.raises(ValueError, match="Cannot achieve desired SNR with rebinning"):
        analyze_statistics(sample_data)


def test_analyze_statistics_mask():
    """Test mask is taken into account in analyze_statistics function.
    This should recommend rebinning only if mask is used."""
    sample_data = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=np.ones((10, 5, 5)), unit="counts", dtype="float64"),
        coords={"tof": sc.linspace("tof", 1, 100, num=11, unit="us")},
    )

    # mask all but one pixel, so each bin will have only 1 count, SNR = 1, which is below the threshold of 3.
    # The function should recommend rebinning.
    sample_data.masks["bad_pixels"] = sc.array(dims=["x", "y"], values=np.ones((5, 5), dtype=bool))
    sample_data.masks["bad_pixels"].values[0, 0] = False  # Only pixel (0, 0) is good

    report = analyze_statistics(sample_data)
    np.testing.assert_almost_equal(report.counts_per_bin, 1)
    np.testing.assert_almost_equal(report.snr_per_bin, 1)
    assert len(report.low_statistics_bins) == 10
    assert report.recommended_rebinning == 10
    assert report.preserve_regions == []
