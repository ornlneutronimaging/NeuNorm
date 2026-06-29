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


def test_background_roi_variance_first_order_value():
    """Propagated variance equals the hand-computed first-order combination (issue #159 review)."""
    from neunorm.processing.normalizer import normalize_transmission

    s = np.full((4, 4), 100.0)
    s[3, 3] = 80.0
    o = np.full((4, 4), 200.0)
    o[3, 3] = 160.0
    t = normalize_transmission(_bg_da(s), _bg_da(o), background_roi=(0, 0, 2, 2))
    # ROI means: cs=100 (Var(mean)=400/16=25), co=200 (Var=800/16=50). Pixel (3,3): S=80, O=160 -> T=1.
    # Var(T) = T^2*(Var(S)/S^2 + Var(cs)/cs^2 + Var(O)/O^2 + Var(co)/co^2)
    #        = 1*(80/6400 + 25/10000 + 160/25600 + 50/40000) = 0.0225
    np.testing.assert_allclose(t.values[3, 3], 1.0, rtol=1e-6)
    np.testing.assert_allclose(t.variances[3, 3], 0.0225, rtol=1e-6)


def test_background_roi_excludes_masked_pixels():
    """Masked pixels inside the ROI are excluded from the ROI mean (issue #159 review, P0)."""
    from neunorm.processing.normalizer import normalize_transmission

    s = np.full((4, 4), 100.0)
    s[0, 0] = 1.0e6  # huge outlier inside the ROI...
    sample = _bg_da(s)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True  # ...but masked, so it must not enter cs
    sample.masks["bad"] = sc.array(dims=["x", "y"], values=mask)
    ob = _bg_da(np.full((4, 4), 200.0))

    t = normalize_transmission(sample, ob, background_roi=(0, 0, 2, 2))
    # mask-aware cs = mean of the 3 unmasked ROI pixels = 100 (1e6 excluded). An out-of-ROI pixel
    # (S=100, O=200) -> T = (100/100)/(200/200) = 1. (With the masked outlier counted, T would be tiny.)
    np.testing.assert_allclose(t.values[3, 3], 1.0, rtol=1e-6)


def test_background_roi_with_dark_variance_matches_full_first_order():
    """dark + background_roi Var(T) matches the FULL first-order shared-dark propagation at an
    outside-ROI pixel (issue #159 review).

    The oracle is derived by hand, independent of the implementation's formula, to avoid circular
    validation. Spatially uniform so cs/co are exact, but S != O so k = co/cs != 1 is exercised.
    """
    from neunorm.processing.dark_corrector import subtract_dark
    from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark

    cnt_s, cnt_o, cnt_d, n = 100.0, 200.0, 10.0, 4  # 2x2 ROI -> n = 4; k = co/cs = 190/90 != 1
    s, o, d = _bg_da(np.full((4, 4), cnt_s)), _bg_da(np.full((4, 4), cnt_o)), _bg_da(np.full((4, 4), cnt_d))
    roi = (0, 0, 2, 2)
    corrected = normalize_with_dark(s, o, d, background_roi=roi)
    naive = normalize_transmission(subtract_dark(s, d), subtract_dark(o, d), background_roi=roi)

    np.testing.assert_allclose(corrected.values, 1.0, rtol=1e-12)  # uniform -> T = 1
    np.testing.assert_allclose(corrected.values, naive.values, rtol=1e-12)  # values unchanged

    # Independent first-order oracle for an OUTSIDE-ROI pixel with shared dark d (Poisson Var=value):
    #   Var(T)/T^2 = [Var(s-d)/(s-d)^2 + Var(o-d)/(o-d)^2 - 2 Var(d)/((s-d)(o-d))]   (pixel, #142)
    #              + [Var(cs)/cs^2     + Var(co)/co^2     - 2 Cov(cs,co)/(cs co)]     (ROI-mean, #159)
    # with cs=s-d, co=o-d, Var(cs)=(s+d)/n, Var(co)=(o+d)/n, Cov(cs,co)=Var(mean d_roi)=d/n.
    s_dc, o_dc = cnt_s - cnt_d, cnt_o - cnt_d
    pixel = (cnt_s + cnt_d) / s_dc**2 + (cnt_o + cnt_d) / o_dc**2 - 2 * cnt_d / (s_dc * o_dc)
    roi_mean = ((cnt_s + cnt_d) / n) / s_dc**2 + ((cnt_o + cnt_d) / n) / o_dc**2 - 2 * (cnt_d / n) / (s_dc * o_dc)
    np.testing.assert_allclose(corrected.variances[3, 3], pixel + roi_mean, rtol=1e-9)
    # the #159 ROI-mean covariance correction strictly reduces the reported variance
    assert corrected.variances[3, 3] < naive.variances[3, 3]


def test_background_roi_with_dark_subtracts_both_shared_dark_terms():
    """dark + background_roi removes BOTH shared-dark variance terms — the #142 pixel-level
    over-count AND the ROI-mean covariance term — on a non-uniform fixture (k != 1, per-pixel)."""
    from neunorm.processing.dark_corrector import subtract_dark
    from neunorm.processing.normalizer import _background_roi_means, normalize_transmission, normalize_with_dark

    roi, n = (0, 0, 2, 2), 4
    s = _bg_da(300.0 + np.arange(16).reshape(4, 4))  # 300..315
    o = _bg_da(500.0 + 2.0 * np.arange(16).reshape(4, 4))  # 500..530
    d = _bg_da(10.0 + 0.5 * np.arange(16).reshape(4, 4))  # small darks 10..17.5
    naive = normalize_transmission(subtract_dark(s, d), subtract_dark(o, d), background_roi=roi)
    corrected = normalize_with_dark(s, o, d, background_roi=roi)

    np.testing.assert_allclose(corrected.values, naive.values, rtol=1e-6)  # values unchanged

    s_dc, o_dc = s.values - d.values, o.values - d.values
    cs, co = _background_roi_means(subtract_dark(s, d), subtract_dark(o, d), roi)
    k = co.value / cs.value
    assert abs(k - 1.0) > 0.3  # fixture exercises k != 1
    pixel_term = 2.0 * (k**2) * s_dc * d.values / (o_dc**3)  # #142 pixel-level over-count
    cov = d.values[0:2, 0:2].sum() / n**2  # Cov(cs,co) = Var(mean D_roi) = (1/n^2) sum Var(D), Var(D)=D
    roi_mean_term = 2.0 * corrected.values**2 * cov / (cs.value * co.value)  # #159 ROI-mean over-count
    np.testing.assert_allclose(naive.variances - corrected.variances, pixel_term + roi_mean_term, rtol=1e-6)
    assert np.all(pixel_term > 0) and cov > 0


def test_background_roi_zero_mean_raises():
    """A zero (or non-finite) ROI mean is rejected with a clear error, not silent inf/nan output."""
    import pytest

    from neunorm.processing.normalizer import normalize_transmission

    s = _bg_da(np.full((4, 4), 100.0))
    o = np.full((4, 4), 200.0)
    o[0:2, 0:2] = 0.0  # open-beam ROI has zero counts -> co == 0
    with pytest.raises(ValueError, match="strictly positive and finite"):
        normalize_transmission(s, _bg_da(o), background_roi=(0, 0, 2, 2))


def test_background_roi_mixed_variance_inputs():
    """One-sided-variance inputs must not raise; the extra term uses only the side with variance."""
    from neunorm.processing.normalizer import normalize_transmission

    roi = (0, 0, 2, 2)

    def _no_var(values):
        v = np.asarray(values, dtype=float)
        return sc.DataArray(data=sc.array(dims=["x", "y"], values=v, unit="counts"))

    s_vals = np.full((4, 4), 100.0)
    o_vals = np.full((4, 4), 200.0)

    # sample has variance, ob does not -> output carries variance (from the sample side only)
    t1 = normalize_transmission(_bg_da(s_vals), _no_var(o_vals), background_roi=roi)
    assert t1.variances is not None
    # ob has variance, sample does not -> this is the case that previously raised in `ob / co`
    t2 = normalize_transmission(_no_var(s_vals), _bg_da(o_vals), background_roi=roi)
    assert t2.variances is not None
    # neither side has variance -> no output variance, and no crash
    t3 = normalize_transmission(_no_var(s_vals), _no_var(o_vals), background_roi=roi)
    assert t3.variances is None
    np.testing.assert_allclose(t3.values, 1.0, rtol=1e-6)


def test_background_roi_broadcast_ob_only_variance():
    """3D no-variance sample + 2D variance OB (broadcast path): OB variance must still propagate.

    Regression for the broadcast-branch gate that dropped the OB-variance term (and the co
    ROI-mean term) when the sample carried no variance — under-stating the reported error.
    """
    from neunorm.processing.normalizer import normalize_transmission

    s3 = sc.DataArray(sc.array(dims=["t", "x", "y"], values=np.full((2, 4, 4), 100.0), unit="counts"))
    o2 = _bg_da(np.full((4, 4), 200.0))  # 2D, carries Poisson variance
    t = normalize_transmission(s3, o2, background_roi=(0, 0, 2, 2))
    assert t.variances is not None
    assert np.all(t.variances > 0)  # OB pixel variance + co ROI-mean term, not silently dropped
