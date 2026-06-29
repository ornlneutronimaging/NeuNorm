"""Unit tests for the ROI data model (issue #159, Jean's named-ROI request)."""

import pytest

from neunorm.data_models.roi import ROI, as_roi_bounds


def test_roi_exported_from_top_level():
    """ROI is importable from the package root (mirrors 1.x `from NeuNorm.roi import ROI`)."""
    import neunorm

    assert neunorm.ROI is ROI


def test_roi_stops_form():
    """ROI(x0, y0, x1, y1) yields the exclusive-stop bounds tuple."""
    assert ROI(x0=10, y0=20, x1=30, y1=40).as_bounds() == (10, 20, 30, 40)


def test_roi_size_form_equivalent_to_stops():
    """ROI(..., width, height) resolves to x1=x0+width, y1=y0+height — equal to the stop form."""
    assert ROI(x0=10, y0=20, width=20, height=20).as_bounds() == (10, 20, 30, 40)
    assert ROI(x0=10, y0=20, width=20, height=20).as_bounds() == ROI(x0=10, y0=20, x1=30, y1=40).as_bounds()


def test_roi_mixed_forms_per_axis():
    """A stop on one axis and a size on the other is allowed."""
    assert ROI(x0=0, y0=5, x1=8, height=10).as_bounds() == (0, 5, 8, 15)
    assert ROI(x0=0, y0=5, width=8, y1=15).as_bounds() == (0, 5, 8, 15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x0": 0, "y0": 0, "x1": 10, "width": 10, "y1": 10},  # both x1 and width
        {"x0": 0, "y0": 0, "y1": 10},  # neither x1 nor width
        {"x0": 0, "y0": 0, "x1": 10, "height": 10, "width": 10},  # both y1 and height (via height+width)
        {"x0": 0, "y0": 0, "x1": 10},  # neither y1 nor height
    ],
)
def test_roi_requires_exactly_one_per_axis(kwargs):
    """Providing both (or neither) of stop/size on an axis raises."""
    with pytest.raises(ValueError, match="exactly one"):
        ROI(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x0": -1, "y0": 0, "x1": 10, "y1": 10},  # negative origin
        {"x0": 10, "y0": 0, "x1": 10, "y1": 10},  # x1 == x0 (empty)
        {"x0": 0, "y0": 0, "x1": 10, "height": 0},  # zero height -> y1 == y0
        {"x0": 0, "y0": 0, "width": -5, "y1": 10},  # negative width -> x1 < x0
    ],
)
def test_roi_rejects_degenerate_rectangles(kwargs):
    """Non-positive extents / out-of-order or negative bounds raise."""
    with pytest.raises(ValueError, match="Invalid ROI"):
        ROI(**kwargs)


def test_as_roi_bounds_accepts_roi_and_tuple():
    """as_roi_bounds coerces both an ROI and a bare (x0, y0, x1, y1) tuple/list."""
    assert as_roi_bounds(ROI(x0=1, y0=2, x1=3, y1=4)) == (1, 2, 3, 4)
    assert as_roi_bounds((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert as_roi_bounds([1, 2, 3, 4]) == (1, 2, 3, 4)


def _da(values, unit="counts"):
    import numpy as np
    import scipp as sc

    v = np.asarray(values, dtype=float)
    da = sc.DataArray(sc.array(dims=["x", "y"], values=v, unit=unit))
    da.variances = v.copy()
    return da


def test_roi_matches_tuple_in_apply_roi():
    """apply_roi gives identical results for an ROI and the equivalent (x0,y0,x1,y1) tuple."""
    import numpy as np

    from neunorm.processing.roi_clipper import apply_roi

    data = _da(np.arange(100.0).reshape(10, 10))
    a = apply_roi(data, (2, 3, 7, 8))
    b = apply_roi(data, ROI(x0=2, y0=3, x1=7, y1=8))
    c = apply_roi(data, ROI(x0=2, y0=3, width=5, height=5))
    assert a.sizes == b.sizes == c.sizes
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.values, c.values)


def test_roi_matches_tuple_in_background_roi_normalization():
    """normalize_transmission(background_roi=ROI(...)) == background_roi=(x0,y0,x1,y1)."""
    import numpy as np

    from neunorm.processing.normalizer import normalize_transmission

    s, o = _da(np.full((6, 6), 80.0)), _da(np.full((6, 6), 100.0))
    t_tuple = normalize_transmission(s, o, background_roi=(0, 0, 3, 3))
    t_roi = normalize_transmission(s, o, background_roi=ROI(x0=0, y0=0, x1=3, y1=3))
    t_size = normalize_transmission(s, o, background_roi=ROI(x0=0, y0=0, width=3, height=3))
    np.testing.assert_array_equal(t_tuple.values, t_roi.values)
    np.testing.assert_array_equal(t_tuple.variances, t_roi.variances)
    np.testing.assert_array_equal(t_tuple.values, t_size.values)


def test_roi_matches_tuple_in_air_region_correction():
    """apply_air_region_correction gives identical results for an ROI and the equivalent tuple."""
    import numpy as np

    from neunorm.processing.air_region_corrector import apply_air_region_correction

    t = _da(np.linspace(0.8, 1.2, 36).reshape(6, 6), unit="dimensionless")
    a = apply_air_region_correction(t, (0, 0, 3, 3))
    b = apply_air_region_correction(t, ROI(x0=0, y0=0, width=3, height=3))
    np.testing.assert_array_equal(a.values, b.values)
