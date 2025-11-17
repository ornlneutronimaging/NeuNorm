"""
Unit tests for uncertainty quantification utilities.

Tests Poisson variance attachment and propagation using scipp.
"""

import numpy as np
import pytest
import scipp as sc


def test_uncertainty_module_imports():
    """Test that uncertainty module can be imported"""


def test_attach_poisson_variance_to_counts():
    """Test attaching Poisson variance to count data"""
    from neunorm.processing.uncertainty_calculator import attach_poisson_variance

    # Create count data
    counts = sc.array(dims=["x"], values=[100, 200, 300, 0, 50], unit="counts")
    data = sc.DataArray(data=counts)

    # Attach Poisson variance
    data_with_var = attach_poisson_variance(data)

    # Variance should equal counts (Poisson: σ² = N)
    assert data_with_var.variances is not None
    np.testing.assert_array_equal(data_with_var.variances, data.values)

    # Original data should not be modified
    assert data.variances is None


def test_attach_poisson_variance_requires_counts_unit():
    """Poisson variance only valid for count data"""
    from neunorm.processing.uncertainty_calculator import attach_poisson_variance

    # Create data with wrong unit
    data = sc.DataArray(
        data=sc.array(dims=["x"], values=[1.0, 2.0, 3.0], unit="m")  # Not counts!
    )

    # Should raise error
    with pytest.raises(ValueError, match="Poisson variance only valid for counts"):
        attach_poisson_variance(data)


def test_attach_poisson_variance_warns_on_overwrite():
    """Warn if data already has variance"""
    from neunorm.processing.uncertainty_calculator import attach_poisson_variance

    # Use float data for variance support
    counts = sc.array(dims=["x"], values=[100.0, 200.0], unit="counts", dtype="float64")
    data = sc.DataArray(data=counts)
    data.variances = np.array([50.0, 100.0])  # Pre-existing variance

    # Should still work but may log warning
    data_new = attach_poisson_variance(data)

    # Should overwrite with Poisson
    np.testing.assert_array_equal(data_new.variances, [100.0, 200.0])


def test_add_systematic_variance_to_existing():
    """Test adding systematic uncertainty to existing Poisson variance"""
    from neunorm.processing.uncertainty_calculator import add_systematic_variance

    # Data with Poisson variance (use float)
    counts = sc.array(dims=["x"], values=[100.0, 200.0, 300.0], unit="counts", dtype="float64")
    data = sc.DataArray(data=counts)
    data.variances = counts.values.copy()  # Poisson

    # Add 0.5% systematic (e.g., proton charge uncertainty)
    data_sys = add_systematic_variance(data, relative_uncertainty=0.005)

    # Total variance = Poisson + systematic
    # systematic = (0.005 * value)²
    expected_var_0 = 100 + (0.005 * 100) ** 2  # 100 + 0.25 = 100.25
    expected_var_1 = 200 + (0.005 * 200) ** 2  # 200 + 1.0 = 201.0
    expected_var_2 = 300 + (0.005 * 300) ** 2  # 300 + 2.25 = 302.25

    np.testing.assert_allclose(data_sys.variances[0], expected_var_0)
    np.testing.assert_allclose(data_sys.variances[1], expected_var_1)
    np.testing.assert_allclose(data_sys.variances[2], expected_var_2)


def test_add_systematic_variance_creates_if_none():
    """Test adding systematic to data without existing variance"""
    from neunorm.processing.uncertainty_calculator import add_systematic_variance

    # Use float data
    data = sc.DataArray(data=sc.array(dims=["x"], values=[100.0, 200.0], unit="counts", dtype="float64"))
    # No variance initially
    assert data.variances is None

    # Add systematic
    data_sys = add_systematic_variance(data, relative_uncertainty=0.01)  # 1%

    # Should create variance array
    assert data_sys.variances is not None
    expected = (0.01 * data.values) ** 2  # (1% of value)²
    np.testing.assert_allclose(data_sys.variances, expected)


def test_get_uncertainty_from_variance():
    """Test extracting uncertainty (σ) from variance (σ²)"""
    from neunorm.processing.uncertainty_calculator import get_uncertainty

    # Use float data
    data = sc.DataArray(data=sc.array(dims=["x"], values=[100.0, 200.0, 300.0], unit="counts", dtype="float64"))
    data.variances = np.array([100.0, 200.0, 300.0])  # Poisson

    uncertainty = get_uncertainty(data)

    # σ = √(σ²)
    expected = np.sqrt([100, 200, 300])
    np.testing.assert_allclose(uncertainty.values, expected)


def test_get_uncertainty_raises_if_no_variance():
    """get_uncertainty should fail if data has no variance"""
    from neunorm.processing.uncertainty_calculator import get_uncertainty

    data = sc.DataArray(data=sc.array(dims=["x"], values=[100, 200], unit="counts"))
    # No variance
    assert data.variances is None

    with pytest.raises(ValueError, match="Data has no variance"):
        get_uncertainty(data)


def test_get_relative_uncertainty():
    """Test computing relative uncertainty (σ/value)"""
    from neunorm.processing.uncertainty_calculator import get_relative_uncertainty

    # Use float data
    data = sc.DataArray(data=sc.array(dims=["x"], values=[100.0, 200.0, 400.0], unit="counts", dtype="float64"))
    data.variances = np.array([100.0, 200.0, 400.0])  # Poisson

    rel_unc = get_relative_uncertainty(data)

    # For Poisson: σ/N = √N / N = 1/√N
    expected = 1.0 / np.sqrt([100, 200, 400])
    np.testing.assert_allclose(rel_unc.values, expected)


def test_scipp_automatic_variance_propagation_division():
    """Verify scipp automatically propagates variance through division"""
    # This is a verification test, not testing our code

    # Create sample and OB with Poisson variance (use float)
    sample = sc.DataArray(data=sc.array(dims=["x"], values=[1000.0, 2000.0], unit="counts", dtype="float64"))
    sample.variances = sample.values.copy()  # var = N

    ob = sc.DataArray(data=sc.array(dims=["x"], values=[900.0, 1800.0], unit="counts", dtype="float64"))
    ob.variances = ob.values.copy()

    # Scipp automatic propagation
    transmission = sample / ob

    # Verify scipp computed variance correctly
    # var(A/B) = (A/B)² * (var(A)/A² + var(B)/B²)
    # For pixel 0: T = 1000/900 = 1.111
    # var(T) = (1000/900)² * (1000/1000² + 900/900²)
    #        = 1.234² * (0.001 + 0.00111)
    #        = 1.524 * 0.00211 ≈ 0.00322

    t0 = 1000 / 900
    var_t0_expected = (t0**2) * (1000 / 1000**2 + 900 / 900**2)

    assert transmission.variances is not None
    np.testing.assert_allclose(transmission.variances[0], var_t0_expected, rtol=1e-6)


def test_variance_propagation_preserves_dimensions():
    """Test that variance propagation works with multi-dimensional data"""
    from neunorm.processing.uncertainty_calculator import attach_poisson_variance

    # 3D data (tof, x, y)
    counts_3d = np.random.randint(50, 500, size=(10, 20, 20))
    data_3d = sc.DataArray(data=sc.array(dims=["tof", "x", "y"], values=counts_3d, unit="counts"))

    data_with_var = attach_poisson_variance(data_3d)

    # Variance should have same shape
    assert data_with_var.variances.shape == counts_3d.shape
    np.testing.assert_array_equal(data_with_var.variances, counts_3d)
