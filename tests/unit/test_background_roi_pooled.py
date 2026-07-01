"""Value-match tests for the pooled multi-ROI + inclusive + OB-less background_roi extension.

The oracle mirrors iBeatles' ``_pooled_roi_means`` (its ``core/processing/normalization.py``):
pooled counts over all ROIs / pooled pixel count, with INCLUSIVE extents. A NeuNorm-backed iBeatles
must value-match this so it can delete its local copy.
"""

import numpy as np
import scipp as sc

from neunorm.data_models.roi import ROI
from neunorm.processing.normalizer import apply_background_roi, normalize_transmission


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
