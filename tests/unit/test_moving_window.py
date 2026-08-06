"""Unit tests for the mask-aware moving window.

The expected values here are **pinned literals**, derived from a naive loop implementation with
hand-written reflect indexing rather than from a second call of the code under test — a test that
asks the implementation what it does can only ever agree with itself.

The port checks compare against ``scipy.ndimage.convolve`` with a normalized box kernel, which is
literally what iBeatles' ``_apply_box_filter`` computes and is therefore the specification.
"""

import numpy as np
import pytest
import scipp as sc
from scipy.ndimage import convolve, uniform_filter

from neunorm.processing.moving_window import (
    EDGE_MODES,
    moving_window,
    moving_window_step_count,
)
from neunorm.utils.progress import STAGE_MOVING_WINDOW

# A 5x5 ramp, 0..24. Small enough that every window can be checked by hand.
RAMP = np.arange(25, dtype=float).reshape(5, 5)

# 3x3 box average of RAMP under scipy's 'reflect' edge, derived by explicit loops:
#   corner (0,0): rows {0,0,1} x cols {0,0,1} -> (0+0+1 + 0+0+1 + 5+5+6)/9 = 2.0
#   centre (2,2): (6+7+8+11+12+13+16+17+18)/9 = 108/9 = 12.0
RAMP_AVG_3X3 = np.array(
    [
        [2.0, 8.0 / 3.0, 11.0 / 3.0, 14.0 / 3.0, 16.0 / 3.0],
        [16.0 / 3.0, 6.0, 7.0, 8.0, 26.0 / 3.0],
        [31.0 / 3.0, 11.0, 12.0, 13.0, 41.0 / 3.0],
        [46.0 / 3.0, 16.0, 17.0, 18.0, 56.0 / 3.0],
        [56.0 / 3.0, 58.0 / 3.0, 61.0 / 3.0, 64.0 / 3.0, 22.0],
    ]
)


def _ramp_array(dims=("y", "x"), values=RAMP, variances=None):
    data = sc.array(dims=list(dims), values=values.copy(), unit="counts")
    if variances is not None:
        data.variances = variances.copy()
    return sc.DataArray(data)


# --------------------------------------------------------------------------------------------
# What the window computes
# --------------------------------------------------------------------------------------------


def test_box_average_matches_hand_computed_values():
    """Each pixel is the mean of the box around it, edges mirrored."""
    result = moving_window(_ramp_array(), {"x": 3, "y": 3})
    np.testing.assert_allclose(result.values, RAMP_AVG_3X3)
    # the three spot values worked out by hand in the module docstring's terms
    np.testing.assert_allclose(result.values[2, 2], 12.0)
    np.testing.assert_allclose(result.values[1, 1], 6.0)
    np.testing.assert_allclose(result.values[0, 0], 2.0)


def test_moving_sum_is_the_same_window_without_the_divisor():
    """``kind='sum'`` is the average times the kernel size, pixel for pixel."""
    average = moving_window(_ramp_array(), {"x": 3, "y": 3})
    total = moving_window(_ramp_array(), {"x": 3, "y": 3}, kind="sum")
    np.testing.assert_allclose(total.values[2, 2], 108.0)
    np.testing.assert_allclose(total.values[0, 0], 18.0)
    np.testing.assert_allclose(total.values, 9.0 * average.values)


def test_a_one_dimensional_window_leaves_the_other_axis_alone():
    """A size given for one dim only averages along that dim."""
    result = moving_window(_ramp_array(), {"x": 3})
    # row 2 of RAMP is [10, 11, 12, 13, 14]; a 3-wide mirror-edged mean gives [10.33, 11, 12, 13, 13.67]
    np.testing.assert_allclose(result.values[2], [31.0 / 3.0, 11.0, 12.0, 13.0, 41.0 / 3.0])


def test_identity_window_returns_the_input_unchanged():
    data = _ramp_array()
    assert moving_window(data, {"x": 1, "y": 1}) is data


# --------------------------------------------------------------------------------------------
# Port fidelity: this is what iBeatles computes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1, 3, 3), (1, 5, 5), (3, 3, 3), (1, 7, 1), (1, 4, 4), (2, 4, 6), (1, 6, 3)])
def test_matches_scipy_box_convolution_as_ibeatles_computes_it(size):
    """The unmasked result is iBeatles' ``convolve(data, ones(k)/k.sum())``, including even sizes."""
    rng = np.random.default_rng(20260806)
    values = rng.poisson(200, size=(4, 17, 19)).astype(np.float64)
    data = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=values, unit="counts"))

    result = moving_window(data, {"tof": size[0], "y": size[1], "x": size[2]})

    kernel = np.ones(size)
    kernel /= kernel.sum()
    np.testing.assert_allclose(result.values, convolve(values, kernel, mode="reflect"), rtol=1e-12, atol=1e-12)


def test_matches_scipy_uniform_filter_for_odd_sizes():
    """For odd windows the separable filter and the direct convolution agree, which is the fast path."""
    rng = np.random.default_rng(7)
    values = rng.poisson(50, size=(9, 11)).astype(np.float64)
    data = sc.DataArray(sc.array(dims=["y", "x"], values=values, unit="counts"))
    result = moving_window(data, {"x": 5, "y": 3})
    np.testing.assert_allclose(result.values, uniform_filter(values, size=[3, 5], mode="reflect"))


def test_an_even_window_leans_the_way_ibeatles_leans():
    """An even window has no centre pixel; iBeatles' convolution puts the extra pixel BEFORE it.

    Pinned because the two obvious scipy entry points disagree: ``uniform_filter`` alone shifts the
    response the other way (+0.50 px), which would silently move every feature in the image.
    """
    step = np.zeros((1, 41))
    step[0, 20:] = 1.0
    data = sc.DataArray(sc.array(dims=["y", "x"], values=step, unit="counts"))
    for k, expected_shift in ((3, 0.0), (4, -0.5), (5, 0.0), (6, -0.5)):
        smoothed = moving_window(data, {"x": k}).values[0]
        first = int(np.flatnonzero(smoothed >= 0.5)[0])
        crossing = first - 1 + (0.5 - smoothed[first - 1]) / (smoothed[first] - smoothed[first - 1])
        np.testing.assert_allclose(crossing - 19.5, expected_shift, atol=1e-9)


@pytest.mark.parametrize("mode", EDGE_MODES)
def test_every_edge_mode_is_scipys(mode):
    """The edge policy is passed straight through, so all of scipy's modes behave as scipy's."""
    rng = np.random.default_rng(3)
    values = rng.poisson(30, size=(8, 8)).astype(np.float64)
    data = sc.DataArray(sc.array(dims=["y", "x"], values=values, unit="counts"))
    result = moving_window(data, {"x": 3, "y": 3}, mode=mode)
    np.testing.assert_allclose(result.values, uniform_filter(values, size=[3, 3], mode=mode))


def test_the_interior_is_identical_whatever_the_edge_mode():
    """Edge policy is confined to a ``k // 2`` border, which is why mirroring is a safe default."""
    rng = np.random.default_rng(11)
    values = rng.poisson(30, size=(12, 12)).astype(np.float64)
    data = sc.DataArray(sc.array(dims=["y", "x"], values=values, unit="counts"))
    reflected = moving_window(data, {"x": 5, "y": 5}, mode="reflect").values
    nearest = moving_window(data, {"x": 5, "y": 5}, mode="nearest").values
    interior = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(reflected[interior], nearest[interior])


# --------------------------------------------------------------------------------------------
# Deviation 1: mask awareness
# --------------------------------------------------------------------------------------------


def test_mask_aware_window_excludes_a_dead_pixel_entirely():
    """One dead pixel corrupts ``k**2`` pixels when the filter is mask-blind, and none when it is not."""
    field = np.full((9, 9), 100.0)
    field[4, 4] = 0.0

    blind = sc.DataArray(sc.array(dims=["y", "x"], values=field.copy(), unit="counts"))
    aware = sc.DataArray(sc.array(dims=["y", "x"], values=field.copy(), unit="counts"))
    dead = np.zeros((9, 9), dtype=bool)
    dead[4, 4] = True
    aware.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)

    blind_result = moving_window(blind, {"x": 3, "y": 3}).values
    aware_result = moving_window(aware, {"x": 3, "y": 3}).values

    # mask-blind: the true level is 100, and a 3x3 window over one zero reads 800/9
    np.testing.assert_allclose(blind_result[4, 4], 800.0 / 9.0)
    np.testing.assert_allclose(blind_result[3, 3], 800.0 / 9.0)
    assert int((np.abs(blind_result - 100.0) > 1e-9).sum()) == 9

    # mask-aware: the true level everywhere, including under the dead pixel itself
    np.testing.assert_allclose(aware_result, 100.0)
    assert int((np.abs(aware_result - 100.0) > 1e-9).sum()) == 0


def test_masks_are_carried_through_untouched():
    """A dead pixel is still a dead pixel after filtering, whatever value the window computed."""
    data = _ramp_array()
    dead = np.zeros((5, 5), dtype=bool)
    dead[1, 1] = True
    data.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)
    data.masks["hot_pixels"] = sc.array(dims=["y", "x"], values=np.zeros((5, 5), dtype=bool))

    result = moving_window(data, {"x": 3, "y": 3})
    assert set(result.masks) == {"dead_pixels", "hot_pixels"}
    np.testing.assert_array_equal(result.masks["dead_pixels"].values, dead)


def test_an_explicit_mask_argument_filters_an_array_that_carries_none():
    """The pipelines detect bad pixels from the open beam but attach them to the sample."""
    field = np.full((9, 9), 100.0)
    field[4, 4] = 0.0
    unmasked = sc.DataArray(sc.array(dims=["y", "x"], values=field, unit="counts"))
    dead = np.zeros((9, 9), dtype=bool)
    dead[4, 4] = True
    masks = {"dead_pixels": sc.array(dims=["y", "x"], values=dead)}

    np.testing.assert_allclose(moving_window(unmasked, {"x": 3, "y": 3}, masks=masks).values, 100.0)
    # and the array itself still carries no masks, so nothing was smuggled onto it
    assert not moving_window(unmasked, {"x": 3, "y": 3}, masks=masks).masks


def test_a_two_dimensional_mask_applies_across_a_three_dimensional_stack():
    """A spatial (x, y) dead-pixel mask must broadcast over every spectral frame."""
    values = np.full((3, 7, 7), 100.0)
    values[:, 3, 3] = 0.0
    data = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=values, unit="counts"))
    dead = np.zeros((7, 7), dtype=bool)
    dead[3, 3] = True
    data.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)

    np.testing.assert_allclose(moving_window(data, {"x": 3, "y": 3}).values, 100.0)


def test_a_mask_on_an_absent_dimension_is_skipped():
    """A mask carrying a dim the data does not have cannot select pixels here."""
    data = _ramp_array()
    data.masks["missing_bins"] = sc.array(dims=["tof"], values=np.array([True, False]))
    np.testing.assert_allclose(moving_window(data, {"x": 3, "y": 3}).values, RAMP_AVG_3X3)


def test_a_fully_masked_window_keeps_its_input_value():
    """Nothing usable in the window: leave the pixel as it came in rather than inventing a NaN."""
    values = np.full((5, 5), 7.0)
    data = sc.DataArray(sc.array(dims=["y", "x"], values=values, variances=np.full((5, 5), 2.0), unit="counts"))
    data.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=np.ones((5, 5), dtype=bool))

    result = moving_window(data, {"x": 3, "y": 3})
    assert np.isfinite(result.values).all()
    np.testing.assert_allclose(result.values, 7.0)
    np.testing.assert_allclose(result.variances, 2.0)


# --------------------------------------------------------------------------------------------
# Deviation 2: variance propagation
# --------------------------------------------------------------------------------------------


def test_variance_follows_the_weights():
    """``Var_out = sum(w**2 Var)``: ``sum(Var)/k**2`` for an average, ``sum(Var)`` for a sum."""
    data = _ramp_array(variances=np.full((5, 5), 4.0))
    average = moving_window(data, {"x": 3, "y": 3})
    total = moving_window(data, {"x": 3, "y": 3}, kind="sum")
    np.testing.assert_allclose(average.variances[2, 2], 4.0 / 9.0)
    np.testing.assert_allclose(total.variances[2, 2], 36.0)
    np.testing.assert_allclose(total.variances, 81.0 * average.variances)


def test_variance_of_a_masked_window_divides_by_the_pixels_that_survived():
    """Eight usable pixels, not nine: the divisor is what the mask left, per pixel."""
    data = _ramp_array(variances=np.full((5, 5), 4.0))
    dead = np.zeros((5, 5), dtype=bool)
    dead[2, 2] = True
    data.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=dead)
    # the window centred on (1,1) covers (2,2), so it collects 8 good pixels: 8*4/8**2 = 0.5
    np.testing.assert_allclose(moving_window(data, {"x": 3, "y": 3}).variances[1, 1], 0.5)


def test_variance_matches_a_monte_carlo_spread():
    """The reported variance is the real spread of the estimator, checked against sampling."""
    rng = np.random.default_rng(20260806)
    lam, k, trials = 200.0, 3, 4000

    centres = np.empty(trials)
    reported = np.empty(trials)
    for i in range(trials):
        counts = rng.poisson(lam, size=(7, 7)).astype(np.float64)
        array = sc.DataArray(sc.array(dims=["y", "x"], values=counts, variances=counts.copy(), unit="counts"))
        filtered = moving_window(array, {"x": k, "y": k})
        centres[i] = filtered.values[3, 3]
        reported[i] = filtered.variances[3, 3]

    empirical = centres.var(ddof=1)
    # Poisson counts averaged over k**2 independent pixels: Var = lam / k**2
    np.testing.assert_allclose(empirical, lam / k**2, rtol=0.06)
    np.testing.assert_allclose(reported.mean(), empirical, rtol=0.06)


def test_data_without_variances_produces_none():
    assert moving_window(_ramp_array(), {"x": 3, "y": 3}).variances is None


# --------------------------------------------------------------------------------------------
# Deviation 3: sizes addressed by dim name
# --------------------------------------------------------------------------------------------


def test_sizes_are_addressed_by_dim_name_not_by_position():
    """The event path is (tof, x, y) and the histogram path (tof, y, x); a tuple would transpose."""
    rng = np.random.default_rng(5)
    yx = rng.poisson(100, size=(2, 11, 13)).astype(np.float64)  # (tof, y, x)
    as_yx = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=yx, unit="counts"))
    as_xy = sc.DataArray(sc.array(dims=["tof", "x", "y"], values=yx.transpose(0, 2, 1).copy(), unit="counts"))

    sizes = {"x": 3, "y": 5}
    from_yx = moving_window(as_yx, sizes).values
    from_xy = moving_window(as_xy, sizes).values

    # same physical data, same named kernel -> same answer, whatever order the dims come in
    np.testing.assert_allclose(from_yx, from_xy.transpose(0, 2, 1))

    # and the test discriminates: reading the sizes positionally would swap them and differ
    swapped = moving_window(as_xy, {"x": 5, "y": 3}).values
    assert not np.allclose(from_yx, swapped.transpose(0, 2, 1))


# --------------------------------------------------------------------------------------------
# dtype, shape and the caller's array
# --------------------------------------------------------------------------------------------


def test_float32_input_stays_float32():
    """A float32 stack must not double in memory just by being filtered."""
    rng = np.random.default_rng(2)
    values = rng.poisson(200, size=(3, 8, 8)).astype(np.float32)
    data = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=values, variances=values.copy(), unit="counts"))
    result = moving_window(data, {"x": 3, "y": 3})
    assert result.values.dtype == np.float32
    assert result.variances.dtype == np.float32


def test_float32_agrees_with_float64_to_single_precision():
    """Filtering in float32 rather than promoting costs nothing that matters at these magnitudes."""
    rng = np.random.default_rng(4)
    values = rng.poisson(200, size=(64, 64)).astype(np.float32)
    as32 = sc.DataArray(sc.array(dims=["y", "x"], values=values, unit="counts"))
    as64 = sc.DataArray(sc.array(dims=["y", "x"], values=values.astype(np.float64), unit="counts"))
    for k in (3, 9, 21):
        np.testing.assert_allclose(
            moving_window(as32, {"x": k, "y": k}).values.astype(np.float64),
            moving_window(as64, {"x": k, "y": k}).values,
            rtol=1e-6,
        )


def test_integer_counts_are_promoted_to_float():
    """An average of integers is not an integer."""
    rng = np.random.default_rng(6)
    data = sc.DataArray(sc.array(dims=["y", "x"], values=rng.poisson(50, (6, 6)), unit="counts"))
    result = moving_window(data, {"x": 3, "y": 3})
    assert result.values.dtype.kind == "f"


def test_shape_unit_dims_and_coords_survive():
    data = _ramp_array(dims=("y", "x"))
    data.coords["x"] = sc.arange("x", 5.0, unit="mm")
    result = moving_window(data, {"x": 3, "y": 3})
    assert result.dims == ("y", "x")
    assert result.shape == (5, 5)
    assert result.unit == data.unit
    np.testing.assert_allclose(result.coords["x"].values, np.arange(5.0))


def test_the_callers_array_is_not_modified():
    data = _ramp_array()
    before = data.values.copy()
    moving_window(data, {"x": 3, "y": 3})
    np.testing.assert_array_equal(data.values, before)


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def test_a_size_for_a_dimension_the_data_does_not_have_is_named():
    with pytest.raises(ValueError, match=r"'lambda'.*does not have.*y, x"):
        moving_window(_ramp_array(), {"lambda": 3})


@pytest.mark.parametrize("bad", [0, -1, -3])
def test_a_non_positive_size_is_rejected_with_its_own_message(bad):
    """scipy raises on these too, but with a message that names neither the size nor the axis."""
    with pytest.raises(ValueError, match=r"must be >= 1"):
        moving_window(_ramp_array(), {"x": bad})


@pytest.mark.parametrize("bad", [3.0, "3", None, True])
def test_a_non_integer_size_is_rejected(bad):
    with pytest.raises(ValueError, match=r"must be an integer"):
        moving_window(_ramp_array(), {"x": bad})


def test_an_empty_size_mapping_is_rejected():
    with pytest.raises(ValueError, match=r"at least one dimension"):
        moving_window(_ramp_array(), {})


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match=r"kind must be 'average' or 'sum'"):
        moving_window(_ramp_array(), {"x": 3}, kind="median")


def test_an_unknown_edge_mode_is_rejected():
    with pytest.raises(ValueError, match=r"mode must be one of"):
        moving_window(_ramp_array(), {"x": 3}, mode="bounce")


def test_even_sizes_are_accepted_as_ibeatles_accepts_them():
    """Not rejected: a window with no centre pixel is a convention, not an error."""
    assert moving_window(_ramp_array(), {"x": 4, "y": 2}).shape == (5, 5)


# --------------------------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------------------------


def test_step_count_matches_the_events_actually_emitted():
    """The declared total and the emitted count must agree, or a bar never fills."""
    rng = np.random.default_rng(8)
    counts = rng.poisson(30, size=(6, 6)).astype(np.float64)

    plain = sc.DataArray(sc.array(dims=["y", "x"], values=counts, unit="counts"))
    with_var = sc.DataArray(sc.array(dims=["y", "x"], values=counts, variances=counts.copy(), unit="counts"))
    with_mask = with_var.copy()
    with_mask.masks["dead_pixels"] = sc.array(dims=["y", "x"], values=np.zeros((6, 6), dtype=bool))

    for data, expected in ((plain, 1), (with_var, 2), (with_mask, 3)):
        events = []
        assert moving_window_step_count(data) == expected
        moving_window(data, {"x": 3, "y": 3}, progress=events.append)
        assert len(events) == expected
        assert {event.stage for event in events} == {STAGE_MOVING_WINDOW}
        assert events[-1].completed == events[-1].total == expected


def test_step_count_honours_an_explicit_mask_argument():
    counts = np.ones((6, 6))
    data = sc.DataArray(sc.array(dims=["y", "x"], values=counts, unit="counts"))
    masks = {"dead_pixels": sc.array(dims=["y", "x"], values=np.zeros((6, 6), dtype=bool))}
    assert moving_window_step_count(data) == 1
    assert moving_window_step_count(data, masks) == 2

    events = []
    moving_window(data, {"x": 3, "y": 3}, masks=masks, progress=events.append)
    assert len(events) == 2
