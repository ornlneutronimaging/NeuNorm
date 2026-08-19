"""Unit tests for ``roi_mean_spectrum``, the mask-aware pooled region mean per spectral bin.

Every expected number here is either a literal or computed with plain numpy from the input array.
None of them comes from a second call into the function under test: the whole purpose of this file
is to pin the arithmetic of the region collapse, and an expectation borrowed from the implementation
pins nothing at all.

The quantity being pinned is ``sum(counts over the selected, unmasked pixels) / count(those
pixels)`` per spectral bin, with ``Var(mean) = sum(Var over those pixels) / n**2``. It is deliberately
*not* ``sc.mean(data, dim=["y", "x"])`` — see :class:`TestAntiSimplification`.
"""

import numpy as np
import pytest
import scipp as sc

from neunorm.data_models.roi import ROI, MaskROI
from neunorm.processing.spectrum_reducer import roi_mean_spectrum

_NT, _NY, _NX = 3, 5, 6

# The rectangle used by most tests: x in [1, 4) and y in [1, 3) -> 3 x 2 = 6 pixels. Stops are
# EXCLUSIVE, so the numpy expectation is counts[:, 1:3, 1:4] (y first, x second).
_RECT = (1, 1, 4, 3)


def _ramp() -> np.ndarray:
    """Index ramp shared by the counts and variance builders below."""
    return np.arange(_NT * _NY * _NX, dtype=float).reshape(_NT, _NY, _NX)


def _counts() -> np.ndarray:
    """Deterministic non-monotonic counts, so selecting the wrong pixel set changes the mean.

    A plain ramp would give the same mean for several different pixel sets (its mean is the midpoint
    of any symmetric block), which would let an off-by-one in the bounds pass unnoticed.
    """
    return (_ramp() % 17) * 3.0 + 5.0


def _variances() -> np.ndarray:
    """Variances on a different cycle from the counts, so ``Var(mean)`` cannot accidentally match."""
    return (_ramp() % 11) + 1.0


def _stack(masks: dict = None) -> sc.DataArray:
    """A ``(tof, y, x)`` counts stack with variances and optional scipp exclusion masks."""
    data = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=_counts(), variances=_variances(), unit="counts"))
    for name, mask in (masks or {}).items():
        data.masks[name] = mask
    return data


def _selection(*, y: slice, x: slice) -> np.ndarray:
    """A boolean ``(y, x)`` selection covering one rectangular block (True = pixel in the region)."""
    sel = np.zeros((_NY, _NX), dtype=bool)
    sel[y, x] = True
    return sel


def _hand_pooled(selection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hand-computed ``(mean, variance-of-mean)`` per tof bin over a boolean ``(y, x)`` selection.

    Plain numpy: fancy-index the pixels the region keeps, sum them, divide by how many there are.
    """
    n = int(selection.sum())
    counts = _counts()[:, selection]
    variances = _variances()[:, selection]
    return counts.sum(axis=1) / n, variances.sum(axis=1) / n**2


def _dead_mask(pixels: list) -> sc.Variable:
    """A spatial dead/hot-pixel exclusion mask flagging the given ``(y, x)`` pixels."""
    dead = np.zeros((_NY, _NX), dtype=bool)
    for y, x in pixels:
        dead[y, x] = True
    return sc.array(dims=["y", "x"], values=dead)


class TestRectangleMeanAndVariance:
    def test_mean_over_a_rectangle_matches_a_numpy_slice(self):
        """With no mask the pooled mean is the plain arithmetic mean of the region's pixels.

        Pinned against ``counts[:, y0:y1, x0:x1].mean(axis=(1, 2))`` so the reduction cannot quietly
        acquire a weighting, an extra normalization, or the wrong pixel set.
        """
        result = roi_mean_spectrum(_stack(), _RECT)

        expected = _counts()[:, 1:3, 1:4].mean(axis=(1, 2))
        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_allclose(result.values, [38.0, 26.0, 22.5])  # the same numbers, written out
        assert result.dims == ("tof",)
        assert result.unit == sc.Unit("counts")

    def test_variance_is_summed_variance_over_pixel_count_squared(self):
        """``Var(mean) = sum(Var) / n**2`` — the variance of a sum of n independent pixels, scaled.

        The wrong-by-n forms (``sum(Var) / n``, i.e. forgetting that the divisor is squared) are
        pinned as *not* the answer, because they differ only by a factor the eye does not catch.
        """
        result = roi_mean_spectrum(_stack(), _RECT)

        summed = _variances()[:, 1:3, 1:4].sum(axis=(1, 2))
        np.testing.assert_allclose(result.variances, summed / 6**2)
        assert np.all(np.abs(result.variances - summed / 6) > 1.0)

    def test_rectangle_stops_are_exclusive(self):
        """``(x0, y0, x1, y1)`` are Python-slice bounds: x1/y1 are past the last pixel.

        Reading them as inclusive would silently widen the region by a row and a column, so the
        inclusive answer is computed too and required to differ.
        """
        result = roi_mean_spectrum(_stack(), _RECT)

        exclusive = _counts()[:, 1:3, 1:4].mean(axis=(1, 2))
        inclusive = _counts()[:, 1:4, 1:5].mean(axis=(1, 2))
        np.testing.assert_allclose(result.values, exclusive)
        assert np.all(np.abs(exclusive - inclusive) > 0.5)

    def test_roi_model_and_bare_tuple_agree(self):
        """An ``ROI`` given by width/height resolves to the same rectangle as the bare tuple form."""
        from_tuple = roi_mean_spectrum(_stack(), _RECT)
        from_model = roi_mean_spectrum(_stack(), ROI(x0=1, y0=1, width=3, height=2))

        np.testing.assert_allclose(from_model.values, _counts()[:, 1:3, 1:4].mean(axis=(1, 2)))
        np.testing.assert_allclose(from_model.values, from_tuple.values)


class TestMaskAwareness:
    def test_dead_pixels_inside_the_region_leave_both_sum_and_count(self):
        """A masked pixel contributes to neither the summed counts nor the pixel count.

        The two halves of that statement are pinned separately: dropping the pixel from the sum but
        not from the denominator (a plain ``width * height`` count) biases the mean low, and dropping
        it from the denominator but not the sum biases it high. Both wrong answers are computed here
        and required to differ from the result, so neither substitution can pass.
        """
        dead = [(2, 2), (1, 3)]  # both inside x in [1, 4), y in [1, 3)
        result = roi_mean_spectrum(_stack({"dead": _dead_mask(dead)}), _RECT)

        keep = np.ones((_NY, _NX), dtype=bool)
        for y, x in dead:
            keep[y, x] = False
        surviving = _selection(y=slice(1, 3), x=slice(1, 4)) & keep
        expected, expected_var = _hand_pooled(surviving)
        assert int(surviving.sum()) == 4  # 6 pixels in the rectangle, 2 of them masked

        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_allclose(result.variances, expected_var)

        masked_sum = _counts()[:, surviving].sum(axis=1)
        plain_pixel_count = 6
        assert np.all(np.abs(result.values - masked_sum / plain_pixel_count) > 1.0)
        whole_region_sum = _counts()[:, 1:3, 1:4].sum(axis=(1, 2))
        assert np.all(np.abs(result.values - whole_region_sum / 4) > 1.0)

    def test_variance_of_a_partially_masked_region_uses_the_unmasked_count(self):
        """``Var(mean)`` over a masked region divides by the *unmasked* count squared.

        Counting the masked pixels in the denominator understates the variance — the region really
        was measured with fewer pixels — and an understated uncertainty is the failure mode a user
        cannot see in the output.
        """
        dead = [(1, 1), (2, 3)]
        result = roi_mean_spectrum(_stack({"dead": _dead_mask(dead)}), _RECT)

        keep = np.ones((_NY, _NX), dtype=bool)
        for y, x in dead:
            keep[y, x] = False
        surviving = _selection(y=slice(1, 3), x=slice(1, 4)) & keep
        summed_var = _variances()[:, surviving].sum(axis=1)

        np.testing.assert_allclose(result.variances, summed_var / 4**2)
        assert np.all(np.abs(result.variances - summed_var / 6**2) > 0.01)


class TestRegionForms:
    def test_maskroi_covering_the_rectangle_gives_the_same_answer(self):
        """A ``MaskROI`` selecting exactly the rectangle's pixels reduces to the same spectrum.

        The two take different code paths (a sliced view versus a bounding-box view carrying the
        inverse selection as a scipp mask), so their agreement is not free — and a user drawing a
        rectangular mask in ImageJ must get the rectangle's answer.
        """
        masks = {"dead": _dead_mask([(2, 2)])}
        selection = _selection(y=slice(1, 3), x=slice(1, 4))
        expected, expected_var = _hand_pooled(selection & ~masks["dead"].values)

        from_rect = roi_mean_spectrum(_stack(masks), _RECT)
        from_mask = roi_mean_spectrum(_stack(masks), MaskROI(selection=selection))

        np.testing.assert_allclose(from_rect.values, expected)
        np.testing.assert_allclose(from_mask.values, expected)
        np.testing.assert_allclose(from_mask.variances, expected_var)

    def test_both_region_forms_reduce_to_the_spectral_dim_alone(self):
        """The output dims depend only on the input's dims, never on which region form was used.

        Downstream code (the transmission division, the ASCII writer) indexes the spectral axis, so
        a region form that returned a scalar instead of one value per bin would break it.

        This holds structurally rather than by a correction: the coefficient is ``total / n_unmasked``,
        and although ``n_unmasked`` collapses to a bare scalar on the MaskROI path with purely spatial
        masks, ``total`` is a sum over ``x`` and ``y`` only and so retains the spectral dim, which the
        division then keeps. Pinned here because that is not obvious from reading the reduction, and a
        reader who assumed otherwise would reach for a broadcast the shapes never require.
        """
        from_rect = roi_mean_spectrum(_stack(), _RECT)
        from_mask = roi_mean_spectrum(_stack(), MaskROI(selection=_selection(y=slice(1, 3), x=slice(1, 4))))

        assert from_rect.dims == ("tof",)
        assert from_mask.dims == ("tof",)
        assert from_rect.shape == (_NT,)
        assert from_mask.shape == (_NT,)

    def test_two_dimensional_input_reduces_to_a_scalar(self):
        """An image with no spectral dim collapses to a 0-D value, not a length-1 spectrum."""
        image = sc.DataArray(sc.array(dims=["y", "x"], values=_counts()[0], variances=_variances()[0], unit="counts"))

        result = roi_mean_spectrum(image, _RECT)

        assert result.dims == ()
        np.testing.assert_allclose(result.value, _counts()[0][1:3, 1:4].mean())
        np.testing.assert_allclose(result.variance, _variances()[0][1:3, 1:4].sum() / 6**2)


class TestPooledRegions:
    def test_overlapping_pair_counts_each_shared_pixel_once(self):
        """Pooled regions are reduced over their UNION, not by adding the regions up.

        Accumulating region by region counts the shared pixels twice: it inflates the mean toward
        the overlap's values, and — because each shared pixel's variance is then added as if it came
        from a second independent sample — understates the variance. The double-counted answer is
        computed here and required to differ.
        """
        first, second = (0, 0, 3, 3), (1, 1, 5, 4)  # 9 and 12 pixels, sharing a 2x2 block
        union = _selection(y=slice(0, 3), x=slice(0, 3)) | _selection(y=slice(1, 4), x=slice(1, 5))
        assert int(union.sum()) == 17  # 9 + 12 - 4 shared

        result = roi_mean_spectrum(_stack(), [first, second])

        expected, expected_var = _hand_pooled(union)
        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_allclose(result.variances, expected_var)

        counts = _counts()
        double_counted = (counts[:, 0:3, 0:3].sum(axis=(1, 2)) + counts[:, 1:4, 1:5].sum(axis=(1, 2))) / 21
        assert np.all(np.abs(result.values - double_counted) > 0.9)

    def test_overlapping_maskroi_and_rectangle_pool_to_the_same_union(self):
        """The union rule holds across region forms: a mask overlapping a rectangle is still a union."""
        selection = _selection(y=slice(0, 3), x=slice(0, 3))
        union = selection | _selection(y=slice(1, 4), x=slice(1, 5))
        expected, expected_var = _hand_pooled(union)

        result = roi_mean_spectrum(_stack(), [MaskROI(selection=selection), (1, 1, 5, 4)])

        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_allclose(result.variances, expected_var)

    def test_disjoint_pair_pools_both_regions(self):
        """Two regions that do not touch pool into one mean over all their pixels together.

        Not the mean of the two region means. The two regions here hold DIFFERENT pixel counts — 4 and
        6 — and that is what makes the distinction observable: pooling weights every pixel equally,
        where averaging the two region means would weight each pixel of the 4-pixel region 1.5x as
        heavily. With equal-sized regions the two forms coincide exactly and this test would pin
        nothing, so the mean-of-means is computed here and required to differ.
        """
        first, second = (0, 0, 2, 2), (3, 3, 6, 5)
        sel_first = _selection(y=slice(0, 2), x=slice(0, 2))
        sel_second = _selection(y=slice(3, 5), x=slice(3, 6))
        union = sel_first | sel_second
        expected, expected_var = _hand_pooled(union)

        # the wrong form, for contrast: each region's own mean, then their unweighted average
        mean_of_means = np.mean([_hand_pooled(sel_first)[0], _hand_pooled(sel_second)[0]], axis=0)

        result = roi_mean_spectrum(_stack(), [first, second])

        assert int(sel_first.sum()) == 4
        assert int(sel_second.sum()) == 6
        assert int(union.sum()) == 10, "disjoint, so the union holds the sum of the two counts"
        assert not np.allclose(expected, mean_of_means), (
            "the regions must differ in size enough that pooling and averaging-of-means disagree, or "
            f"this test cannot fail: pooled={expected} mean_of_means={mean_of_means}"
        )
        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_allclose(result.variances, expected_var)
        assert not np.allclose(result.values, mean_of_means)


class TestAntiSimplification:
    def test_pooled_mean_disagrees_with_scipp_dimension_wise_mean(self):
        """``sc.mean(region, dim=["y", "x"])`` is NOT this pooled mean once a pixel is masked.

        scipp reduces one dimension at a time, so that call averages each column over its own
        surviving rows and then averages those column means — an unweighted average of unequally
        sampled columns. The pooled mean instead divides the total counts by the total number of
        surviving pixels, which is the region's actual mean. The nested spelling
        ``mean("x").mean("y")`` is a third, order-dependent number.

        On the 3x3 block below (counts 1..9, with x=2 masked in rows y=1 and y=2, leaving 7 pixels
        summing to 30):

        * pooled            = 30 / 7      = 4.2857...   <- what roi_mean_spectrum returns
        * sc.mean(["y","x"]) = (4+5+3) / 3 = 4.0        <- column means, then averaged
        * mean("x").mean("y") = (2+4.5+7.5) / 3 = 4.667 <- row means, then averaged

        Substituting either scipp form would change every transmission spectrum this library
        produces over a detector with dead pixels, silently.
        """
        counts = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
            ]
        )
        block = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=counts.copy(), unit="counts"))
        dead = np.zeros((3, 3), dtype=bool)
        dead[1, 2] = True
        dead[2, 2] = True
        block.masks["dead"] = sc.array(dims=["y", "x"], values=dead)

        pooled = roi_mean_spectrum(block, (0, 0, 3, 3))
        dimension_wise = sc.mean(block, dim=["y", "x"])
        nested = sc.mean(sc.mean(block, dim="x"), dim="y")

        np.testing.assert_allclose(pooled.values, [30.0 / 7.0, 300.0 / 7.0])
        np.testing.assert_allclose(dimension_wise.values, [4.0, 40.0])
        np.testing.assert_allclose(nested.values, [14.0 / 3.0, 140.0 / 3.0])
        assert np.all(np.abs(pooled.values - dimension_wise.values) > 0.25)
        assert np.all(np.abs(pooled.values - nested.values) > 0.25)


class TestSurvivingCoordsAndMasks:
    def _annotated(self) -> sc.DataArray:
        """A stack carrying every kind of coord and mask a pipeline attaches before this reduction."""
        per_bin = sc.array(dims=["tof"], values=[False, False, True])
        data = _stack({"dead": _dead_mask([(2, 2)]), "missing_bin": per_bin})
        data.coords["tof"] = sc.array(dims=["tof"], values=[0.0, 1.0, 2.0, 3.0], unit="us")  # N + 1 edges
        data.coords["y"] = sc.array(dims=["y"], values=np.arange(_NY, dtype=float), unit="mm")
        data.coords["x"] = sc.array(dims=["x"], values=np.arange(_NX, dtype=float), unit="mm")
        data.coords["proton_charge"] = sc.scalar(1.5, unit="C")
        data.coords["spectra_tof"] = sc.array(dims=["tof"], values=[0.4, 1.4, 2.4], unit="us")
        data.coords.set_aligned("spectra_tof", False)
        return data

    def test_spectral_coords_survive_and_spatial_ones_do_not(self):
        """The reduction keeps what still has meaning and drops what it consumed.

        The ``N + 1`` ``tof`` bin edges must survive — without them the spectrum cannot be rebinned
        again or converted to wavelength — as must scalar metadata such as ``proton_charge``. The
        ``(y, x)`` pixel-position coords describe an axis that no longer exists, so keeping them
        would leave a DataArray whose coords contradict its dims.
        """
        result = roi_mean_spectrum(self._annotated(), _RECT)

        assert set(result.coords) == {"tof", "spectra_tof", "proton_charge"}
        assert "y" not in result.coords
        assert "x" not in result.coords
        assert result.coords["tof"].sizes == {"tof": _NT + 1}
        assert result.coords.is_edges("tof")
        np.testing.assert_allclose(result.coords["tof"].values, [0.0, 1.0, 2.0, 3.0])
        assert result.coords["proton_charge"].dims == ()
        np.testing.assert_allclose(result.coords["proton_charge"].value, 1.5)

    def test_alignment_flags_are_preserved(self):
        """Whether a coord is aligned decides if scipp refuses a later mismatched binary op.

        Re-attaching a coord defaults it to aligned, so an unaligned ``spectra_tof`` silently
        becoming aligned would start rejecting divisions that used to work — and an aligned ``tof``
        silently becoming unaligned would start permitting divisions between different time axes,
        which is the worse direction.
        """
        result = roi_mean_spectrum(self._annotated(), _RECT)

        assert result.coords["tof"].aligned is True
        assert result.coords["spectra_tof"].aligned is False

    def test_per_bin_masks_survive_and_pixel_masks_do_not(self):
        """A per-bin mask still applies to the spectrum; the dead-pixel mask has been consumed.

        The dead/hot pixel masks were folded into the mean (that is what makes it mask-aware), so
        carrying them onto a result with no spatial dims is both meaningless and, for a downstream
        guard reading ``masks``, actively misleading. A ``(tof,)`` mask marking bins that hold no
        data is the opposite: it must reach the exporter so those rows can be omitted.
        """
        result = roi_mean_spectrum(self._annotated(), _RECT)

        assert set(result.masks) == {"missing_bin"}
        assert result.masks["missing_bin"].dims == ("tof",)
        assert list(result.masks["missing_bin"].values) == [False, False, True]


class TestStrictGuard:
    def _flat(self, value: float) -> sc.DataArray:
        values = np.full((_NT, _NY, _NX), value)
        counts = sc.array(dims=["tof", "y", "x"], values=values, variances=np.ones_like(values), unit="counts")
        return sc.DataArray(counts)

    @pytest.mark.parametrize("value", [0.0, -2.0])
    def test_strict_rejects_a_non_positive_region_mean(self, value):
        """As a denominator, a region mean of zero or less is a fault, and must not become inf/NaN.

        The guard exists so the failure is reported where it can be diagnosed (the region contains
        no beam) rather than surfacing later as a spectrum full of inf.
        """
        with pytest.raises(ValueError, match="pooled mean must be strictly positive and finite"):
            roi_mean_spectrum(self._flat(value), _RECT, name="ob")

    def test_strict_error_names_the_argument_and_the_array(self):
        """The message must say which argument and which input, or the user cannot act on it."""
        with pytest.raises(ValueError) as excinfo:
            roi_mean_spectrum(self._flat(0.0), _RECT, region_arg="spectrum_roi", name="open beam")

        message = str(excinfo.value)
        assert "spectrum_roi" in message
        assert "open beam" in message

    @pytest.mark.parametrize("value", [0.0, -2.0])
    def test_non_strict_lets_a_non_positive_mean_through(self, value):
        """A zero or negative mean is a real measurement in a numerator — full absorption, or
        dark-subtracted counts scattering below zero — so ``strict=False`` must return it unchanged
        rather than raise or clamp it.
        """
        result = roi_mean_spectrum(self._flat(value), _RECT, strict=False)

        np.testing.assert_allclose(result.values, np.full(_NT, value))


class TestStructuralErrors:
    """Malformed input always raises, whatever ``strict`` says: ``strict`` only governs the
    non-positive-mean guard, never whether the region actually fits the data."""

    @pytest.mark.parametrize("strict", [True, False])
    def test_rectangle_outside_the_detector_raises(self, strict):
        with pytest.raises(ValueError, match="exceeds"):
            roi_mean_spectrum(_stack(), (0, 0, _NX + 4, 3), strict=strict)

    @pytest.mark.parametrize("strict", [True, False])
    def test_inverted_rectangle_raises(self, strict):
        with pytest.raises(ValueError, match="need 0 <= x0 < x1"):
            roi_mean_spectrum(_stack(), (3, 1, 2, 3), strict=strict)

    @pytest.mark.parametrize("strict", [True, False])
    def test_maskroi_of_the_wrong_shape_raises(self, strict):
        """A selection sized for a different detector (or a pre-crop frame) must not be broadcast
        or silently padded — its pixels do not correspond to these pixels."""
        wrong = MaskROI(selection=np.ones((_NY - 1, _NX - 1), dtype=bool))

        with pytest.raises(ValueError, match="does not match data size"):
            roi_mean_spectrum(_stack(), wrong, strict=strict)

    @pytest.mark.parametrize("strict", [True, False])
    def test_input_without_x_and_y_dims_raises(self, strict):
        """Region indices are resolved against ``x``/``y`` by name, so differently named spatial
        dims are refused rather than guessed at."""
        renamed = sc.DataArray(sc.array(dims=["tof", "row", "col"], values=_counts(), unit="counts"))

        with pytest.raises(ValueError, match="must have 'x' and 'y' dimensions"):
            roi_mean_spectrum(renamed, _RECT, strict=strict)

    @pytest.mark.parametrize("strict", [True, False])
    def test_empty_region_list_raises(self, strict):
        with pytest.raises(ValueError, match="at least one ROI"):
            roi_mean_spectrum(_stack(), [], strict=strict)
