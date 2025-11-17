"""
Unit tests for transmission normalization.

Tests the core neutron imaging equation: T = Sample / OpenBeam
with variance propagation and mask handling.
"""

import numpy as np
import scipp as sc


def test_normalizer_imports():
    """Test that normalizer module can be imported"""


def test_normalize_transmission_basic():
    """Test basic transmission normalization: T = Sample / OB"""
    from neunorm.processing.normalizer import normalize_transmission

    # Create simple sample and OB histograms
    sample_data = np.ones((10, 5, 5)) * 80.0  # 80% transmission
    ob_data = np.ones((10, 5, 5)) * 100.0

    sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    sample.variances = sample.values.copy()  # Poisson

    ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    ob.variances = ob.values.copy()

    transmission = normalize_transmission(sample, ob)

    # Should be dimensionless
    assert transmission.unit == sc.units.one

    # Values should be ~0.8
    np.testing.assert_allclose(transmission.values, 0.8, rtol=0.01)

    # Variance should be propagated
    assert transmission.variances is not None


def test_normalize_transmission_with_masks():
    """Test that masks are properly combined during normalization"""
    from neunorm.processing.normalizer import normalize_transmission

    sample_data = np.ones((10, 10, 10)) * 100.0
    ob_data = np.ones((10, 10, 10)) * 100.0

    sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    sample.variances = sample.values.copy()

    # Add dead pixel mask
    dead_mask = sc.array(dims=["x", "y"], values=np.zeros((10, 10), dtype=bool))
    dead_mask.values[0, 0] = True  # Mask pixel (0, 0)
    sample.masks["dead_pixels"] = dead_mask

    ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    ob.variances = ob.values.copy()
    ob.masks["dead_pixels"] = dead_mask

    transmission = normalize_transmission(sample, ob)

    # Mask should be preserved
    assert "dead_pixels" in transmission.masks
    assert transmission.masks["dead_pixels"].values[0, 0]


def test_normalize_transmission_handles_zero_ob():
    """Test that zero OB counts are handled gracefully"""
    from neunorm.processing.normalizer import normalize_transmission

    sample_data = np.ones((10, 5, 5)) * 100.0
    ob_data = np.ones((10, 5, 5)) * 100.0
    ob_data[:, 2, 2] = 0  # Zero OB at pixel (2, 2)

    sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    sample.variances = sample.values.copy()

    ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    ob.variances = ob.values.copy()

    transmission = normalize_transmission(sample, ob)

    # Division by zero should produce inf or masked value
    assert np.isinf(transmission.values[:, 2, 2]).all() or np.isnan(transmission.values[:, 2, 2]).all()

    # Other pixels should be normal
    assert np.isfinite(transmission.values[:, 0, 0]).all()


def test_normalize_transmission_with_proton_charge():
    """Test normalization with proton charge correction"""
    from neunorm.processing.normalizer import normalize_transmission

    sample_data = np.ones((10, 5, 5)) * 100.0
    ob_data = np.ones((10, 5, 5)) * 100.0

    sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    sample.variances = sample.values.copy()

    ob = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=ob_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    ob.variances = ob.values.copy()

    # Sample collected with 500 C, OB with 505 C
    transmission = normalize_transmission(
        sample,
        ob,
        proton_charge_sample=500.0,
        proton_charge_ob=505.0,
        pc_uncertainty=0.005,  # 0.5%
    )

    # Should account for different proton charges
    # T = (Sample/500) / (OB/505) = (100/500) / (100/505) = 0.2 / 0.198 ≈ 1.01
    expected = (100.0 / 500.0) / (100.0 / 505.0)
    np.testing.assert_allclose(transmission.values, expected, rtol=0.01)

    # Variance should include systematic from proton charge
    assert transmission.variances is not None
