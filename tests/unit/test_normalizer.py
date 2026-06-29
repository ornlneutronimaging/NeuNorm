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
    np.testing.assert_allclose(transmission.values, 0.8)

    # variance should be propagated
    # Var(T) = (T^2) * (Var(Sample)/Sample^2 + Var(OB)/OB^2)
    # Var = (80/100)^2 * (1/80 + 1/100) = 0.0144
    np.testing.assert_allclose(transmission.variances, 0.0144)


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


# --- issue #142: shared-dark variance must not be double-counted ---


def _ccd_frames(sample_counts, ob_counts, dark_counts):
    """Build (sample 3D, ob 2D, dark 2D) DataArrays with Poisson variance for one pixel value."""
    s, o, d = float(sample_counts), float(ob_counts), float(dark_counts)
    sample = sc.DataArray(sc.array(dims=["N_image", "y", "x"], values=[[[s]]], variances=[[[s]]], unit="counts"))
    ob = sc.DataArray(sc.array(dims=["y", "x"], values=[[o]], variances=[[o]], unit="counts"))
    dark = sc.DataArray(sc.array(dims=["y", "x"], values=[[d]], variances=[[d]], unit="counts"))
    return sample, ob, dark


def test_normalize_with_dark_values_match_old_path():
    """normalize_with_dark returns the SAME transmission values as subtract_dark+normalize (#142)."""
    from neunorm.processing.dark_corrector import subtract_dark
    from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark

    sample, ob, dark = _ccd_frames(800, 1500, 60)
    # MARS (no proton charge)
    old = normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark))
    new = normalize_with_dark(sample, ob, dark)
    np.testing.assert_allclose(new.values, old.values, rtol=1e-12)

    # VENUS (proton charge): values still identical
    pc_s = sc.array(dims=["N_image"], values=[0.1], unit="C")
    pc_o = sc.scalar(0.2, unit="C")
    old_v = normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark), pc_s, pc_o)
    new_v = normalize_with_dark(sample, ob, dark, pc_s, pc_o)
    np.testing.assert_allclose(new_v.values, old_v.values, rtol=1e-12)


def test_normalize_with_dark_variance_matches_monte_carlo():
    """The corrected Var(T) matches a shared-dark Monte Carlo and is smaller than the old
    double-counted value by exactly 2*N*Var(D)/M^3 (MARS) (issue #142)."""
    from neunorm.processing.dark_corrector import subtract_dark
    from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark

    sample_counts, ob_counts, dark_counts = 800.0, 1500.0, 60.0
    sample, ob, dark = _ccd_frames(sample_counts, ob_counts, dark_counts)

    old = normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark))
    new = normalize_with_dark(sample, ob, dark)
    var_old = float(old.variances.ravel()[0])
    var_new = float(new.variances.ravel()[0])

    # Monte Carlo ground truth: the SAME dark draw feeds numerator and denominator.
    # 500k trials keep this a fast unit test; the exact analytic check below is the precise oracle.
    rng = np.random.default_rng(42)
    n = 500_000
    s = rng.normal(sample_counts, np.sqrt(sample_counts), n)
    o = rng.normal(ob_counts, np.sqrt(ob_counts), n)
    d = rng.normal(dark_counts, np.sqrt(dark_counts), n)
    var_mc = ((s - d) / (o - d)).var()

    # Corrected variance matches MC far better than the double-counted one.
    assert abs(var_new - var_mc) / var_mc < 0.01
    assert abs(var_old - var_mc) / var_mc > 0.03
    # The exact over-count removed is 2 * numerator * Var(D) / denominator^3.
    numerator, denominator = sample_counts - dark_counts, ob_counts - dark_counts
    np.testing.assert_allclose(var_old - var_new, 2 * numerator * dark_counts / denominator**3, rtol=1e-6)


def test_normalize_with_dark_reduces_variance_with_proton_charge():
    """VENUS path: the correction also reduces the variance (no double-counted Var(dark))."""
    from neunorm.processing.dark_corrector import subtract_dark
    from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark

    sample, ob, dark = _ccd_frames(800, 1500, 60)
    pc_s = sc.array(dims=["N_image"], values=[0.1], unit="C")
    pc_o = sc.scalar(0.2, unit="C")
    old = normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark), pc_s, pc_o)
    new = normalize_with_dark(sample, ob, dark, pc_s, pc_o)
    assert float(new.variances.ravel()[0]) < float(old.variances.ravel()[0])
    assert float(new.variances.ravel()[0]) > 0.0


def test_normalize_with_dark_pc_overcount_per_image_exact():
    """VENUS: the removed over-count equals 2*k^2*(S-D)*Var(D)/(O-D)^3 with per-image k (#142).

    Pins the exact k^2 = (pc_ob/pc_sample)^2 magnitude and the per-image broadcast for a
    multi-image sample with VARYING per-image proton charge (not just directional reduction).
    """
    from neunorm.processing.dark_corrector import subtract_dark
    from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark

    sample_vals = np.array([800.0, 820.0, 840.0])
    ob_val, dark_val = 1500.0, 60.0
    sample = sc.DataArray(
        sc.array(
            dims=["N_image", "y", "x"],
            values=sample_vals.reshape(3, 1, 1),
            variances=sample_vals.reshape(3, 1, 1),
            unit="counts",
        )
    )
    ob = sc.DataArray(sc.array(dims=["y", "x"], values=[[ob_val]], variances=[[ob_val]], unit="counts"))
    dark = sc.DataArray(sc.array(dims=["y", "x"], values=[[dark_val]], variances=[[dark_val]], unit="counts"))
    pc_sample_vals = np.array([0.10, 0.15, 0.20])
    pc_ob_val = 0.20
    pc_s = sc.array(dims=["N_image"], values=pc_sample_vals, unit="C")
    pc_o = sc.scalar(pc_ob_val, unit="C")

    old = normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark), pc_s, pc_o)
    new = normalize_with_dark(sample, ob, dark, pc_s, pc_o)

    # Independent analytic oracle (per image): over_count = 2*k^2*(S-D)*Var(D)/(O-D)^3.
    k = pc_ob_val / pc_sample_vals
    s = sample_vals - dark_val
    o = ob_val - dark_val
    expected = 2 * k**2 * s * dark_val / o**3
    diff = old.variances.ravel() - new.variances.ravel()
    np.testing.assert_allclose(diff, expected, rtol=1e-6)


def test_normalize_transmission_2d_ob():
    """Test basic transmission normalization: T = Sample / OB with an averaged 2D OB"""
    from neunorm.processing.normalizer import normalize_transmission

    # Create simple sample and OB histograms
    sample_data = np.full((10, 5, 5), 80.0)  # 80% transmission
    ob_data = np.full((5, 5), 100.0)

    sample = sc.DataArray(
        data=sc.array(dims=["energy", "x", "y"], values=sample_data, unit="counts", dtype="float64"),
        coords={"energy": sc.linspace("energy", 1, 100, num=11, unit="eV")},
    )
    sample.variances = sample.values.copy()  # Poisson

    ob = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=ob_data, unit="counts", dtype="float64"),
    )
    ob.variances = ob.values.copy()

    transmission = normalize_transmission(sample, ob)

    # Should be dimensionless
    assert transmission.unit == sc.units.one

    # Values should be ~0.8
    np.testing.assert_allclose(transmission.values, 0.8)

    # variance should be propagated
    # Var(T) = (T^2) * (Var(Sample)/Sample^2 + Var(OB)/OB^2)
    # Var = (80/100)^2 * (1/80 + 1/100) = 0.0144
    np.testing.assert_allclose(transmission.variances, 0.0144)


def test_normalize_transmission_requires_both_proton_charges():
    """One-sided proton charge -> non-dimensionless T, so it must raise (issue #163)."""
    import pytest

    from neunorm.processing.normalizer import normalize_transmission

    sample = sc.DataArray(data=sc.array(dims=["x"], values=[80.0, 80.0], unit="counts", dtype="float64"))
    sample.variances = sample.values.copy()
    ob = sc.DataArray(data=sc.array(dims=["x"], values=[100.0, 100.0], unit="counts", dtype="float64"))
    ob.variances = ob.values.copy()

    with pytest.raises(ValueError, match="both be provided or both omitted"):
        normalize_transmission(sample, ob, proton_charge_sample=500.0)
    with pytest.raises(ValueError, match="both be provided or both omitted"):
        normalize_transmission(sample, ob, proton_charge_ob=505.0)

    # both provided is fine and stays dimensionless
    t = normalize_transmission(sample, ob, proton_charge_sample=500.0, proton_charge_ob=505.0)
    assert t.unit == sc.units.one


# ---------------------------------------------------------------------------
# background_roi flux normalization (proton-charge proxy, issue #159)
# ---------------------------------------------------------------------------


def _bg_da(values, dims=("x", "y")):
    """Helper: scipp DataArray in counts with Poisson variance = values."""
    values = np.asarray(values, dtype=float)
    da = sc.DataArray(data=sc.array(dims=list(dims), values=values, unit="counts"))
    da.variances = values.copy()
    return da


def test_background_roi_matches_ratio_of_means():
    """T == (sample/ob) * (mean(ob[B]) / mean(sample[B])) — the validated 1.x formula (issue #159)."""
    from neunorm.processing.normalizer import normalize_transmission

    s_vals = np.array([[10, 10, 30, 40], [10, 10, 50, 60], [12, 14, 16, 18], [20, 22, 24, 26]], dtype=float)
    o_vals = np.array([[20, 20, 25, 35], [20, 20, 45, 55], [22, 24, 26, 28], [30, 32, 34, 36]], dtype=float)
    roi = (0, 0, 2, 2)  # (x0, y0, x1, y1), exclusive stops -> the [0:2, 0:2] block

    t = normalize_transmission(_bg_da(s_vals), _bg_da(o_vals), background_roi=roi)

    cs = s_vals[0:2, 0:2].mean()
    co = o_vals[0:2, 0:2].mean()
    expected = (s_vals / o_vals) * (co / cs)
    np.testing.assert_allclose(t.values, expected, rtol=1e-6)
    assert t.unit == sc.units.one


def test_background_roi_cancels_beam_flux():
    """A sample-free background ROI makes the per-image flux factors cancel -> recovers true T (issue #159)."""
    from neunorm.processing.normalizer import normalize_transmission

    truth = np.full((4, 4), 0.8)
    truth[0:2, 0:2] = 1.0  # background quadrant: no sample, transmission = 1
    flux_s, flux_o = 3.0, 5.0  # different beam intensity for sample vs open-beam acquisition
    sample = _bg_da(flux_s * truth * 100.0)
    ob = _bg_da(flux_o * np.ones((4, 4)) * 100.0)

    t = normalize_transmission(sample, ob, background_roi=(0, 0, 2, 2))
    np.testing.assert_allclose(t.values, truth, rtol=1e-6)


def test_background_roi_mutually_exclusive_with_proton_charge():
    """background_roi and proton_charge_* are mutually exclusive (issue #159)."""
    import pytest

    from neunorm.processing.normalizer import normalize_transmission

    sample = _bg_da(np.full((4, 4), 100.0))
    ob = _bg_da(np.full((4, 4), 100.0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize_transmission(
            sample, ob, proton_charge_sample=500.0, proton_charge_ob=505.0, background_roi=(0, 0, 2, 2)
        )


def test_background_roi_validation():
    """Invalid background_roi tuples / missing x,y dims raise (issue #159)."""
    import pytest

    from neunorm.processing.normalizer import normalize_transmission

    sample = _bg_da(np.full((4, 4), 100.0))
    ob = _bg_da(np.full((4, 4), 100.0))
    with pytest.raises(ValueError):
        normalize_transmission(sample, ob, background_roi=(0, 0, 2))  # not 4 ints
    with pytest.raises(ValueError):
        normalize_transmission(sample, ob, background_roi=(0, 0, 100, 100))  # out of bounds
    with pytest.raises(ValueError):
        normalize_transmission(sample, ob, background_roi=(2, 0, 1, 2))  # x1 <= x0


def test_background_roi_propagates_finite_variance():
    """background_roi normalization yields finite, non-negative propagated variance (issue #159)."""
    from neunorm.processing.normalizer import normalize_transmission

    t = normalize_transmission(
        _bg_da(np.full((4, 4), 100.0)), _bg_da(np.full((4, 4), 200.0)), background_roi=(0, 0, 2, 2)
    )
    assert t.variances is not None
    assert np.all(np.isfinite(t.variances))
    assert np.all(t.variances >= 0)


def test_background_roi_3d_per_spectral_bin():
    """For 3D (tof, x, y) data the ROI mean is computed per spectral bin (issue #159)."""
    from neunorm.processing.normalizer import normalize_transmission

    s = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4) + 10.0
    o = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4) + 20.0
    t = normalize_transmission(
        _bg_da(s, dims=("tof", "x", "y")), _bg_da(o, dims=("tof", "x", "y")), background_roi=(0, 0, 2, 2)
    )
    assert t.shape == (3, 4, 4)
    cs = s[:, 0:2, 0:2].mean(axis=(1, 2))
    co = o[:, 0:2, 0:2].mean(axis=(1, 2))
    expected = (s / o) * (co / cs)[:, None, None]
    np.testing.assert_allclose(t.values, expected, rtol=1e-6)
