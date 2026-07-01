"""Value-match tests for the pooled multi-ROI + inclusive + OB-less background_roi extension.

The oracle mirrors iBeatles' ``_pooled_roi_means`` (its ``core/processing/normalization.py``):
pooled counts over all ROIs / pooled pixel count, with INCLUSIVE extents. A NeuNorm-backed iBeatles
must value-match this so it can delete its local copy.
"""

import json
from collections import deque

import numpy as np
import scipp as sc

from neunorm.data_models.roi import ROI
from neunorm.processing.normalizer import apply_background_roi, as_roi_bounds_list, normalize_transmission


def _da(arr, with_var=False):
    arr = np.asarray(arr, dtype=float)
    dims = ["x", "y"] if arr.ndim == 2 else ["N_image", "x", "y"]
    d = sc.DataArray(sc.array(dims=dims, values=arr.copy(), unit="counts"))
    if with_var:
        d.variances = arr.copy()
    return d


def _pooled(arr, rois_incl):
    """iBeatles pooled ROI mean: sum(counts over all ROIs) / sum(pixels), INCLUSIVE (x0,y0,w,h). arr[x,y]."""
    total, npix = 0.0, 0
    for x0, y0, w, h in rois_incl:
        region = arr[x0 : x0 + w + 1, y0 : y0 + h + 1]
        total += region.sum()
        npix += region.size
    return total / npix


def _make_nonuniform(base):
    """8x8 [x,y] frame with two bright patches so pooling differs from any single ROI."""
    a = np.full((8, 8), base)
    a[0:2, 0:2] = base * 2.0
    a[4:6, 4:6] = base * 4.0
    return a


def _incl_rois():
    return [(0, 0, 1, 1), (4, 4, 1, 1)]  # inclusive w=h=1 -> 2x2 patches at (0,0) and (4,4)


def _incl_roi_objs():
    return [ROI(x0=x0, y0=y0, width=w, height=h, inclusive=True) for x0, y0, w, h in _incl_rois()]


def test_as_roi_bounds_list_coerces_numpy_ints_to_plain_int():
    """NumPy integer bounds become built-in int, so provenance JSON-encodes losslessly.

    Without coercion, ``json.dumps`` either raises on np.int64 or (with ``default=str``) silently
    stringifies the bounds, corrupting round-tripped provenance types.
    """
    rois = [(np.int64(0), np.int64(0), np.int64(8), np.int64(8)), (10, 10, 18, 18)]
    bounds = as_roi_bounds_list(rois)
    assert all(type(v) is int for b in bounds for v in b)
    assert json.loads(json.dumps([list(b) for b in bounds])) == [[0, 0, 8, 8], [10, 10, 18, 18]]
    # single bare-4 with numpy ints too
    (single,) = as_roi_bounds_list((np.int64(1), np.int64(2), np.int64(3), np.int64(4)))
    assert all(type(v) is int for v in single) and single == (1, 2, 3, 4)


def test_as_roi_bounds_list_accepts_any_sequence():
    """Any non-str Sequence works, matching the BackgroundROILike type hint (not just tuple/list)."""
    assert as_roi_bounds_list(deque([(0, 0, 8, 8), (10, 10, 18, 18)])) == [(0, 0, 8, 8), (10, 10, 18, 18)]
    assert as_roi_bounds_list(deque([0, 0, 8, 8])) == [(0, 0, 8, 8)]  # bare-4 Sequence is a single ROI
    # strings are Sequences but never ROIs — still rejected with a clear error
    try:
        as_roi_bounds_list("0,0,8,8")
        raise AssertionError("str must be rejected")
    except ValueError:
        pass


def test_pooled_multi_roi_matches_ibeatles_formula():
    """T with a pooled multi-ROI background_roi == (S/pool(S)) / (O/pool(O)) by the iBeatles formula."""
    s, o = _make_nonuniform(50.0), _make_nonuniform(60.0)
    t = normalize_transmission(_da(s), _da(o), background_roi=_incl_roi_objs())
    expected = (s / _pooled(s, _incl_rois())) / (o / _pooled(o, _incl_rois()))
    np.testing.assert_allclose(t.values, expected, rtol=1e-6)
    # pooling is real: the pooled coefficient differs from a single ROI's mean
    assert not np.isclose(_pooled(s, _incl_rois()), _pooled(s, [_incl_rois()[0]]))


def test_single_roi_backward_compatible():
    """A single ROI (bare tuple or 1-element list) still gives the plain ROI mean."""
    s, o = _make_nonuniform(50.0), _make_nonuniform(60.0)
    t_tuple = normalize_transmission(_da(s), _da(o), background_roi=(0, 0, 2, 2))
    t_list = normalize_transmission(_da(s), _da(o), background_roi=[(0, 0, 2, 2)])
    np.testing.assert_array_equal(t_tuple.values, t_list.values)
    expected = (s / s[0:2, 0:2].mean()) / (o / o[0:2, 0:2].mean())
    np.testing.assert_allclose(t_tuple.values, expected, rtol=1e-6)


def test_inclusive_roi_covers_same_pixels_as_exclusive_tuple():
    """ROI(width=1, inclusive=True) (2 px) selects the same region as the exclusive tuple (0,0,2,2)."""
    s, o = _make_nonuniform(50.0), _make_nonuniform(60.0)
    t_incl = normalize_transmission(_da(s), _da(o), background_roi=ROI(x0=0, y0=0, width=1, height=1, inclusive=True))
    t_excl = normalize_transmission(_da(s), _da(o), background_roi=(0, 0, 2, 2))
    np.testing.assert_array_equal(t_incl.values, t_excl.values)


def test_apply_background_roi_ob_less_matches_hand():
    """Sample-only flux flattening (no OB) == data / pooled_mean(data)."""
    s = _make_nonuniform(50.0)
    out = apply_background_roi(_da(s), _incl_roi_objs())
    np.testing.assert_allclose(out.values, s / _pooled(s, _incl_rois()), rtol=1e-6)


def test_pooled_variance_free_and_with_variance():
    """Pooling works variance-free (iBeatles) and with Poisson variance; values are identical."""
    s, o = _make_nonuniform(50.0), _make_nonuniform(60.0)
    rois = _incl_roi_objs()
    t_free = normalize_transmission(_da(s), _da(o), background_roi=rois)
    assert t_free.variances is None
    t_var = normalize_transmission(_da(s, with_var=True), _da(o, with_var=True), background_roi=rois)
    assert t_var.variances is not None and np.all(t_var.variances >= 0)
    np.testing.assert_allclose(t_var.values, t_free.values, rtol=1e-6)


def test_pooled_coefficient_per_image_mask_does_not_collapse_denominator():
    """A per-image (3D) mask must keep the pixel-count denominator per-frame (P0 regression guard).

    With the old scalar denominator, an unmasked frame's pooled mean was inflated by
    n_roi/(n_roi - n_masked); this asserts each frame is normalized by its own unmasked count.
    """
    v = 50.0
    arr = np.full((3, 8, 8), v)  # (N_image, x, y), uniform
    da = sc.DataArray(sc.array(dims=["N_image", "x", "y"], values=arr, unit="counts"))
    mask = np.zeros((3, 8, 8), dtype=bool)
    mask[0, 0, 0] = True  # mask ONE ROI pixel on frame 0 only -> a 3D (per-image) mask
    da.masks["bad"] = sc.array(dims=["N_image", "x", "y"], values=mask)

    out = apply_background_roi(da, ROI(x0=0, y0=0, width=1, height=1, inclusive=True))  # 2x2 ROI
    # every frame's pooled mean is v (all *unmasked* ROI pixels are v), so the flattened output is 1
    # on every frame — including the unmasked frames (which the collapsed-denominator bug inflated).
    np.testing.assert_allclose(out.values, 1.0, rtol=1e-6)


def test_apply_background_roi_propagates_first_order_variance():
    """apply_background_roi adds the ROI-mean first-order term: Var = Var(d)/c^2 + out^2 * Var(c)/c^2."""
    s = _make_nonuniform(50.0)  # [7,7]=50 (outside both ROI patches); ROI (0,0,2,2) patch = 100
    out = apply_background_roi(_da(s, with_var=True), (0, 0, 2, 2))
    assert out.variances is not None
    pool = s[0:2, 0:2].mean()  # 100
    var_pool = s[0:2, 0:2].sum() / 4**2  # Var(mean) = sum(unmasked Poisson var)/n^2 = 25
    val = s[7, 7] / pool  # 0.5
    np.testing.assert_allclose(out.values[7, 7], val, rtol=1e-6)
    expected_var = s[7, 7] / pool**2 + val**2 * var_pool / pool**2
    np.testing.assert_allclose(out.variances[7, 7], expected_var, rtol=1e-6)


def test_background_roi_nonstrict_zero_count_roi_propagates_inf():
    """strict=False skips the positivity guard: a zero-count ROI yields inf, not ValueError.

    The 1.x / iBeatles semantics (a zero-count background ROI propagates inf in the sample-only
    path) — the escape hatch that lets downstreams reproduce 1.x outputs bit for bit.
    """
    import pytest

    s = np.ones((2, 8, 8))
    s[:, 0:2, 0:2] = 0.0  # ROI (0,0,2,2) pooled mean -> 0
    da = sc.DataArray(sc.array(dims=["N_image", "x", "y"], values=s, unit="counts"))

    # default remains strict: raises
    with pytest.raises(ValueError, match="strictly positive"):
        apply_background_roi(da, (0, 0, 2, 2))

    # non-strict: inf propagates (1.x), values outside the ROI are 1/0 = inf
    out = apply_background_roi(da, (0, 0, 2, 2), strict=False)
    assert np.isinf(out.values).any()

    # matched-OB path: same opt-out on normalize_transmission
    ob = sc.DataArray(sc.array(dims=["N_image", "x", "y"], values=np.ones((2, 8, 8)), unit="counts"))
    with pytest.raises(ValueError, match="strictly positive"):
        normalize_transmission(da, ob, background_roi=(0, 0, 2, 2))
    t = normalize_transmission(da, ob, background_roi=(0, 0, 2, 2), background_roi_strict=False)
    assert not np.isfinite(t.values).all()


def test_background_roi_nonstrict_still_raises_structural_errors():
    """strict=False relaxes ONLY the positivity guard — bad ROI bounds still raise."""
    import pytest

    da = sc.DataArray(sc.array(dims=["N_image", "x", "y"], values=np.ones((2, 8, 8)), unit="counts"))
    with pytest.raises(ValueError, match="exceeds"):
        apply_background_roi(da, (0, 0, 99, 99), strict=False)
