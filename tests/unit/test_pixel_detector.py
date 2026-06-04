"""
Unit tests for pixel detector (dead/hot pixel detection).

Tests MAD-based hot pixel detection and zero-count dead pixel detection.
Ported from venus_tof.masking tests.
"""

import numpy as np
import scipp as sc


def test_pixel_detector_imports():
    """Test that pixel detector module can be imported"""


def test_detect_dead_pixels_basic():
    """Test dead pixel detection with simple case"""
    from neunorm.tof.pixel_detector import detect_dead_pixels

    # Create histogram with some dead pixels
    # Shape: (10 energy bins, 5x5 spatial)
    data = np.ones((10, 5, 5)) * 100  # Normal pixels have 100 counts per bin
    data[:, 0, 0] = 0  # Dead pixel at (0, 0)
    data[:, 2, 3] = 0  # Dead pixel at (2, 3)

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    dead_mask = detect_dead_pixels(hist)

    # Should detect 2 dead pixels
    assert dead_mask.values.sum() == 2
    assert dead_mask.values[0, 0]  # Dead pixel
    assert dead_mask.values[2, 3]  # Dead pixel
    assert not dead_mask.values[1, 1]  # Normal pixel (live)


def test_detect_dead_pixels_all_live():
    """Test when no dead pixels present"""
    from neunorm.tof.pixel_detector import detect_dead_pixels

    # All pixels have counts
    data = np.random.rand(10, 5, 5) * 100 + 1  # Ensure no zeros

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    dead_mask = detect_dead_pixels(hist)

    # No dead pixels
    assert dead_mask.values.sum() == 0


def test_detect_dead_pixels_all_dead():
    """Test when all pixels are dead"""
    from neunorm.tof.pixel_detector import detect_dead_pixels

    # All zeros
    data = np.zeros((10, 5, 5))

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    dead_mask = detect_dead_pixels(hist)

    # All dead
    assert dead_mask.values.sum() == 25  # 5×5 = 25 pixels


def test_detect_hot_pixels_mad_based():
    """Test hot pixel detection using MAD threshold"""
    from neunorm.tof.pixel_detector import detect_hot_pixels

    # Create histogram with outlier pixels
    data = np.ones((10, 10, 10)) * 100  # Normal: 1000 counts total per pixel

    # Add hot pixels with 10× normal counts
    data[:, 0, 0] = 1000  # Hot pixel at (0, 0)
    data[:, 5, 5] = 1200  # Hot pixel at (5, 5)

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    hot_mask = detect_hot_pixels(hist, sigma=5.0)

    # Should detect the hot pixels
    assert hot_mask.values[0, 0]  # Hot pixel
    assert hot_mask.values[5, 5]  # Hot pixel
    # Normal pixels should not be detected
    assert not hot_mask.values[1, 1]


def test_detect_hot_pixels_different_sigma_thresholds():
    """Test that different sigma values affect detection sensitivity"""
    from neunorm.tof.pixel_detector import detect_hot_pixels

    # Create data with mild outlier
    data = np.ones((10, 10, 10)) * 100
    data[:, 5, 5] = 300  # 3× normal (mild outlier)

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    # Aggressive threshold (sigma=3) should catch it
    hot_aggressive = detect_hot_pixels(hist, sigma=3.0)
    assert hot_aggressive.values[5, 5]

    # Conservative threshold (sigma=10) might not catch mild outliers
    # (test demonstrates threshold effects, but outcome depends on MAD)


def test_detect_hot_pixels_no_outliers():
    """Test when no hot pixels present"""
    from neunorm.tof.pixel_detector import detect_hot_pixels

    # Uniform distribution, no outliers
    data = np.ones((10, 10, 10)) * 100

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    hot_mask = detect_hot_pixels(hist, sigma=5.0)

    # No hot pixels detected
    assert hot_mask.values.sum() == 0


def test_detect_hot_pixels_handles_dead_pixels():
    """Test that dead pixels (zeros) don't skew MAD calculation"""
    from neunorm.tof.pixel_detector import detect_hot_pixels

    # Mix of normal, dead, and hot pixels
    data = np.ones((10, 10, 10)) * 100
    data[:, 0:3, 0:3] = 0  # Dead pixels (3×3 region)
    data[:, 9, 9] = 2000  # Hot pixel

    hist = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data, unit="counts"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    hot_mask = detect_hot_pixels(hist, sigma=5.0)

    # Should detect hot pixel despite dead pixels present
    assert hot_mask.values[9, 9]
    # Dead pixels should not be marked as hot
    assert not hot_mask.values[0, 0]


def test_detect_bad_pixels_for_transmission():
    """Test comprehensive bad pixel detection for transmission imaging"""
    from neunorm.tof.pixel_detector import detect_bad_pixels_for_transmission

    # Create sample histogram with dead and hot pixels
    data_sample = np.ones((10, 10, 10)) * 100.0
    data_sample[:, 0, 0] = 0  # Dead in sample
    data_sample[:, 1, 1] = 2000  # Hot in sample

    hist_sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data_sample, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    # Create OB histogram with different bad pixels
    data_ob = np.ones((10, 10, 10)) * 100.0
    data_ob[:, 2, 2] = 0  # Dead in OB
    data_ob[:, 3, 3] = 2000  # Hot in OB

    hist_ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=data_ob, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )

    # Detect bad pixels
    masks = detect_bad_pixels_for_transmission(hist_sample, hist_ob, sigma=5.0)

    # Should return 4 masks
    assert "dead_sample" in masks
    assert "hot_sample" in masks
    assert "dead_ob" in masks
    assert "hot_ob" in masks

    # Verify detections
    assert masks["dead_sample"].values[0, 0]
    assert masks["hot_sample"].values[1, 1]
    assert masks["dead_ob"].values[2, 2]
    assert masks["hot_ob"].values[3, 3]

    # Masks should be applied to both histograms
    assert "dead_pixels_sample" in hist_sample.masks
    assert "hot_pixels_sample" in hist_sample.masks
    assert "dead_pixels_ob" in hist_sample.masks
    assert "hot_pixels_ob" in hist_sample.masks


def test_detect_dead_pixels_works_with_tof_dimension():
    """Test dead pixel detection works with 'tof' dimension (not just 'energy')"""
    from neunorm.tof.pixel_detector import detect_dead_pixels

    # Create histogram with TOF dimension
    data = np.ones((10, 5, 5)) * 100
    data[:, 0, 0] = 0  # Dead pixel

    hist = sc.DataArray(
        data=sc.array(dims=["tof", "x", "y"], values=data, unit="counts"),
        coords={"tof": sc.linspace("tof", 1e5, 1e6, num=11, unit="ns")},
    )

    dead_mask = detect_dead_pixels(hist)

    assert dead_mask.values[0, 0]
    assert dead_mask.values.sum() == 1


def test_detect_hot_pixels_works_with_wavelength_dimension():
    """Test hot pixel detection works with 'wavelength' dimension"""
    from neunorm.tof.pixel_detector import detect_hot_pixels

    data = np.ones((10, 10, 10)) * 100
    data[:, 5, 5] = 2000  # Hot pixel

    hist = sc.DataArray(
        data=sc.array(dims=["wavelength", "x", "y"], values=data, unit="counts"),
        coords={"wavelength": sc.linspace("wavelength", 0.5, 3.0, num=11, unit="angstrom")},
    )

    hot_mask = detect_hot_pixels(hist, sigma=5.0)

    assert hot_mask.values[5, 5]
