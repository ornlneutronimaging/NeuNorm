"""Frame-index binning of an ROI spectrum: where the two possible orders agree, and where they do not.

An ROI spectrum can be built two ways, and only one of them is what the pipelines do:

* **Order A (the settled order).** Bin both stacks with ``rebin_tof``, collapse each side's region to
  a pooled mean per bin, divide once. This mirrors the image mode exactly — the pipelines already
  call ``rebin_tof(sample); rebin_tof(ob); normalize_transmission(...)`` — so resonance mode
  introduces no second convention.
* **Order B.** Collapse each stack to a 1-D spectrum first, bin the two spectra, divide once.

The two are the same quantity only under conditions that are easy to lose sight of, so they are
measured here rather than assumed:

* with a purely spatial ``(y, x)`` mask they agree exactly for ``sum`` and for ``mean``, at every
  binning tried (a factor of 2, a factor of 3, and an explicit non-uniform contiguous bin list);
* with a per-frame ``(tof, y, x)`` mask they **diverge**, because binning consumes the mask and
  leaves every bin with the full region as its denominator while order B keeps a different unmasked
  count for each frame;
* with ``median`` they **diverge** whenever a bin holds three or more frames, because a median does
  not commute with a mean.

Both divergent cases are pinned to order A, with the other order's numbers written down as well, so
the choice is recorded as a decision and not left as an accident of call order.

Two mechanical facts about the ``tof`` axis are pinned alongside: a stack carrying no timing
coordinate cannot be binned at all until an ``N + 1`` index edge coordinate is attached (verified by
running it — the rebinner raises today), and a representative time per bin survives a sum-mode rebin
even though :func:`scipp.rebin` cannot produce one, because the reduction rebuilds it from the bin
edges and refuses to trust a coordinate that a sum has corrupted.

Every expected number below is a literal or is computed with plain numpy from the synthetic input.
Where the two orders are compared against each other the invariant under test *is* their agreement,
so that comparison is the point — but each such case additionally pins order A against the numpy
oracle, so the pair cannot both drift together.
"""

import numpy as np
import pytest
import scipp as sc
from loguru import logger

from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.spectrum_reducer import normalize_roi_spectrum, roi_mean_spectrum
from neunorm.tof.histogram_rebinner import SPECTRA_TOF_COORD, rebin_tof, reduce_tof_bins

_N_TOF, _NY, _NX = 6, 5, 6

#: ``(x0, y0, x1, y1)`` with EXCLUSIVE stops: x in [1, 5), y in [1, 4) -> 4 x 3 = 12 pixels, away
#: from the frame edges so an off-by-one in the bounds changes the answer.
_REGION = (1, 1, 5, 4)
_X0, _Y0, _X1, _Y1 = _REGION

#: N + 1 TOF bin edges, microseconds; 6 frames so both a factor of 2 and a factor of 3 divide it.
_EDGES = np.linspace(1000.0, 7000.0, _N_TOF + 1)

#: The three binnings exercised for commutation, as the ``rebin_tof`` ``width`` and the equivalent
#: half-open frame ranges the numpy oracle uses. The bin list is deliberately non-uniform, so it is
#: not a restatement of either factor.
_BINNINGS = [
    (2, [(0, 2), (2, 4), (4, 6)]),
    (3, [(0, 3), (3, 6)]),
    ([[0, 1], [1, 4], [4, 6]], [(0, 1), (1, 4), (4, 6)]),
]


# --------------------------------------------------------------------------------------------
# synthetic input
# --------------------------------------------------------------------------------------------


def _ramp() -> np.ndarray:
    """Index ramp the counts and variances are built from."""
    return np.arange(_N_TOF * _NY * _NX, dtype=float).reshape(_N_TOF, _NY, _NX)


def _sample_counts() -> np.ndarray:
    """Non-monotonic sample counts: a plain ramp averages to its midpoint over any symmetric block,
    which would let a wrong pixel or frame set produce the right mean."""
    return (_ramp() % 13) * 7.0 + 30.0


def _sample_variances() -> np.ndarray:
    """Sample variances on a different cycle from the counts, so a values/variances mix-up shows."""
    return (_ramp() % 11) + 1.0


def _ob_counts() -> np.ndarray:
    """Open-beam counts, everywhere positive so the strict denominator guard is satisfied."""
    return (_ramp() % 19) * 11.0 + 400.0


def _ob_variances() -> np.ndarray:
    """Open-beam variances, again on their own cycle."""
    return (_ramp() % 7) * 2.0 + 3.0


def _stack(counts: np.ndarray, variances: np.ndarray, *, mask=None, mask_dims=None) -> sc.DataArray:
    """A ``(tof, y, x)`` counts stack with variances and an ``N + 1`` bin-edge ``tof`` axis."""
    data = sc.DataArray(
        sc.array(dims=["tof", "y", "x"], values=counts.copy(), variances=variances.copy(), unit="counts"),
        coords={"tof": sc.array(dims=["tof"], values=_EDGES.copy(), unit="us")},
    )
    if mask is not None:
        data.masks["spike"] = sc.array(dims=mask_dims, values=mask.copy())
    return data


def _spatial_mask() -> np.ndarray:
    """A ``(y, x)`` exclusion mask with one pixel inside the region and one outside it."""
    mask = np.zeros((_NY, _NX), dtype=bool)
    mask[2, 3] = True  # inside the region
    mask[0, 0] = True  # outside it, so the region bounds still matter
    return mask


def _per_frame_mask() -> np.ndarray:
    """A ``(tof, y, x)`` exclusion mask that masks a region pixel in some frames and not others.

    Frame 0 and frame 5 each lose one region pixel; frames 1-4 lose none. Under the factor-2 binning
    that puts an unequal pair in the first and last bins and a clean pair in the middle one, which is
    what makes the divergence visible in two bins out of three and absent in the third.
    """
    mask = np.zeros((_N_TOF, _NY, _NX), dtype=bool)
    mask[0, 2, 3] = True
    mask[5, 1, 2] = True
    return mask


def _capture_warnings():
    """Return ``(messages, remove)`` capturing loguru WARNING+ lines into ``messages``.

    loguru does not route through stdlib logging, so pytest's ``caplog`` is always empty for it and
    any assertion made against ``caplog.text`` would pass without testing anything.
    """
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    return messages, lambda: logger.remove(handler_id)


# --------------------------------------------------------------------------------------------
# the two orders, and the numpy oracle
# --------------------------------------------------------------------------------------------


def _order_a(sample: sc.DataArray, ob: sc.DataArray, width, reduction) -> sc.DataArray:
    """Bin both stacks, then collapse each region, then divide once — the settled order."""
    return normalize_roi_spectrum(
        rebin_tof(sample, width, reduction=reduction),
        rebin_tof(ob, width, reduction=reduction),
        _REGION,
    )


def _order_b(sample: sc.DataArray, ob: sc.DataArray, width, reduction) -> sc.DataArray:
    """Collapse each region, then bin the two 1-D spectra, then divide once — the other order.

    The division is the same ``normalize_transmission`` call ``normalize_roi_spectrum`` makes
    internally, so the only difference between this and :func:`_order_a` is where the binning sits.
    """
    sample_spectrum = rebin_tof(roi_mean_spectrum(sample, _REGION, strict=False), width, reduction=reduction)
    ob_spectrum = rebin_tof(roi_mean_spectrum(ob, _REGION, strict=True), width, reduction=reduction)
    return normalize_transmission(sample=sample_spectrum, ob=ob_spectrum)


def _region(values: np.ndarray) -> np.ndarray:
    """The region's pixels, in numpy order (y before x, exclusive stops)."""
    return values[:, _Y0:_Y1, _X0:_X1]


def _reduce_frames(values: np.ndarray, variances: np.ndarray, ranges, reduction):
    """Combine the frames of each half-open range with plain numpy, per pixel."""
    if reduction == "sum":
        out = np.stack([values[a:b].sum(axis=0) for a, b in ranges])
        out_var = np.stack([variances[a:b].sum(axis=0) for a, b in ranges])
    elif reduction == "mean":
        out = np.stack([values[a:b].mean(axis=0) for a, b in ranges])
        out_var = np.stack([variances[a:b].sum(axis=0) / (b - a) ** 2 for a, b in ranges])
    else:  # pragma: no cover - the median oracle is written out explicitly where it is needed
        raise AssertionError(f"no numpy oracle for reduction {reduction!r}")
    return out, out_var


def _pooled(values: np.ndarray, variances: np.ndarray, selected: np.ndarray):
    """Pooled region mean and its variance over the selected pixels: sum / n, and sum(Var) / n**2."""
    n = int(selected.sum())
    return (values * selected).sum(axis=(-2, -1)) / n, (variances * selected).sum(axis=(-2, -1)) / n**2


def _transmission(sample_mean, sample_var, ob_mean, ob_var):
    """``T = S / O`` with the propagated variance scipp's division produces.

    ``Var(T) = Var(S)/O**2 + S**2 * Var(O)/O**4`` — written out here rather than taken from the
    library, so a change to the propagation is a test failure and not a silently moved oracle.
    """
    return sample_mean / ob_mean, sample_var / ob_mean**2 + sample_mean**2 * ob_var / ob_mean**4


def _oracle_order_a(ranges, reduction, selected):
    """Order A computed entirely in numpy: reduce frames, then pool the region, then divide."""
    sample_binned, sample_binned_var = _reduce_frames(
        _region(_sample_counts()), _region(_sample_variances()), ranges, reduction
    )
    ob_binned, ob_binned_var = _reduce_frames(_region(_ob_counts()), _region(_ob_variances()), ranges, reduction)
    sample_mean, sample_var = _pooled(sample_binned, sample_binned_var, selected)
    ob_mean, ob_var = _pooled(ob_binned, ob_binned_var, selected)
    return _transmission(sample_mean, sample_var, ob_mean, ob_var)


# --------------------------------------------------------------------------------------------
# 1. sum and mean commute with the region collapse under a purely spatial mask
# --------------------------------------------------------------------------------------------


class TestSumAndMeanCommuteUnderASpatialMask:
    """Both orders give the same spectrum for ``sum`` and ``mean``, so choosing order A costs nothing.

    That is the whole justification for matching the image mode: if the orders disagreed, the choice
    would be a numerical decision rather than a consistency one. A purely spatial mask is the
    condition under which it holds — every bin then divides by the same unmasked pixel count, so the
    count cancels out of the ratio no matter when the collapse happens.
    """

    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    @pytest.mark.parametrize("width,ranges", _BINNINGS, ids=["factor2", "factor3", "explicit_bin_list"])
    def test_both_orders_agree_and_match_the_numpy_oracle(self, reduction, width, ranges):
        """Values and variances agree between the orders, and order A matches hand-computed numpy.

        The order-A-versus-numpy leg is what stops the two orders being wrong together: they share
        every arithmetic primitive, so their agreement alone would survive a broken pooled mean.
        """
        mask = _spatial_mask()
        sample = _stack(_sample_counts(), _sample_variances(), mask=mask, mask_dims=["y", "x"])
        ob = _stack(_ob_counts(), _ob_variances(), mask=mask, mask_dims=["y", "x"])

        first = _order_a(sample, ob, width, reduction)
        second = _order_b(sample, ob, width, reduction)

        assert first.sizes == {"tof": len(ranges)}
        np.testing.assert_allclose(first.values, second.values, rtol=1e-12)
        np.testing.assert_allclose(first.variances, second.variances, rtol=1e-12)

        expected, expected_var = _oracle_order_a(ranges, reduction, ~mask[_Y0:_Y1, _X0:_X1])
        np.testing.assert_allclose(first.values, expected, rtol=1e-12)
        np.testing.assert_allclose(first.variances, expected_var, rtol=1e-12)

    def test_the_pinned_spectrum_is_not_the_unmasked_one(self):
        """The masked region pixel is genuinely excluded from the pooled sums.

        Without this the commutation tests above would pass on a mask-blind collapse: with a purely
        spatial mask the unmasked *count* cancels out of the ratio, so only the mask-awareness of the
        summed counts is observable in the transmission at all.
        """
        mask = _spatial_mask()
        sample = _stack(_sample_counts(), _sample_variances(), mask=mask, mask_dims=["y", "x"])
        ob = _stack(_ob_counts(), _ob_variances(), mask=mask, mask_dims=["y", "x"])
        ranges = [(0, 2), (2, 4), (4, 6)]

        result = _order_a(sample, ob, 2, "sum")

        selected = ~mask[_Y0:_Y1, _X0:_X1]
        masked_aware, _ = _oracle_order_a(ranges, "sum", selected)
        mask_blind, _ = _oracle_order_a(ranges, "sum", np.ones_like(selected))
        assert not np.allclose(masked_aware, mask_blind, rtol=1e-6), "the fixture's mask must move the answer"
        np.testing.assert_allclose(result.values, masked_aware, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# 2. a per-frame mask makes the two orders diverge
# --------------------------------------------------------------------------------------------


class TestPerFrameMaskMakesTheOrdersDiverge:
    """With a ``(tof, y, x)`` mask the orders are different quantities, and order A is what ships.

    Binning consumes a per-frame mask: ``rebin_tof`` leaves each masked ``(frame, pixel)`` entry out
    of that pixel's combined value and the output carries no mask at all, so the collapse that follows
    divides by the **full** region. Order B instead divides each frame by that frame's own unmasked
    count before combining. The two coincide only when every frame in a bin has the same count.

    The pipelines bin first, so order A's numbers are the ones a user gets. Both are written down.
    """

    def _inputs(self):
        mask = _per_frame_mask()
        return (
            _stack(_sample_counts(), _sample_variances(), mask=mask, mask_dims=["tof", "y", "x"]),
            _stack(_ob_counts(), _ob_variances(), mask=mask, mask_dims=["tof", "y", "x"]),
            mask,
        )

    @staticmethod
    def _oracle_bin_first(mask, ranges):
        """Frames combined mask-aware, then the whole region pooled with no mask left."""
        keep = ~mask[:, _Y0:_Y1, _X0:_X1]
        full = np.ones((_Y1 - _Y0, _X1 - _X0), dtype=bool)

        def side(counts, variances):
            binned = np.stack([(_region(counts) * keep)[a:b].sum(axis=0) for a, b in ranges])
            binned_var = np.stack([(_region(variances) * keep)[a:b].sum(axis=0) for a, b in ranges])
            return _pooled(binned, binned_var, full)

        sample_mean, sample_var = side(_sample_counts(), _sample_variances())
        ob_mean, ob_var = side(_ob_counts(), _ob_variances())
        return _transmission(sample_mean, sample_var, ob_mean, ob_var)

    @staticmethod
    def _oracle_collapse_first(mask, ranges):
        """Each frame pooled over its own unmasked count, then those per-frame means summed."""
        keep = ~mask[:, _Y0:_Y1, _X0:_X1]
        per_frame_n = keep.sum(axis=(1, 2))

        def side(counts, variances):
            mean = (_region(counts) * keep).sum(axis=(1, 2)) / per_frame_n
            var = (_region(variances) * keep).sum(axis=(1, 2)) / per_frame_n**2
            return (
                np.array([mean[a:b].sum() for a, b in ranges]),
                np.array([var[a:b].sum() for a, b in ranges]),
            )

        sample_mean, sample_var = side(_sample_counts(), _sample_variances())
        ob_mean, ob_var = side(_ob_counts(), _ob_variances())
        return _transmission(sample_mean, sample_var, ob_mean, ob_var)

    def test_the_two_orders_do_not_agree(self):
        """Both orders are pinned to numpy, and the pair is asserted to be measurably apart."""
        sample, ob, mask = self._inputs()
        ranges = [(0, 2), (2, 4), (4, 6)]

        first = _order_a(sample, ob, 2, "sum")
        second = _order_b(sample, ob, 2, "sum")

        bin_first, bin_first_var = self._oracle_bin_first(mask, ranges)
        collapse_first, collapse_first_var = self._oracle_collapse_first(mask, ranges)

        np.testing.assert_allclose(first.values, bin_first, rtol=1e-12)
        np.testing.assert_allclose(first.variances, bin_first_var, rtol=1e-12)
        np.testing.assert_allclose(second.values, collapse_first, rtol=1e-12)
        np.testing.assert_allclose(second.variances, collapse_first_var, rtol=1e-12)

        assert not np.allclose(first.values, second.values, rtol=1e-5), (
            f"a per-frame mask must make the orders diverge; got {first.values} and {second.values}"
        )

    def test_only_the_bins_holding_an_unequal_pair_diverge(self):
        """The divergence tracks the per-frame unmasked count, which is the stated mechanism.

        Bins 0 and 2 each pair a frame that lost a region pixel with one that did not; bin 1 pairs
        two intact frames and therefore agrees. Pinning that pattern is what distinguishes the real
        cause from an unrelated discrepancy that happens to be the same size.
        """
        sample, ob, _ = self._inputs()

        first = _order_a(sample, ob, 2, "sum")
        second = _order_b(sample, ob, 2, "sum")

        np.testing.assert_allclose(first.values[1], second.values[1], rtol=1e-12)
        relative = np.abs(first.values - second.values) / first.values
        assert relative[0] > 1e-4, f"bin 0 should diverge; relative difference {relative[0]:g}"
        assert relative[2] > 1e-3, f"bin 2 should diverge; relative difference {relative[2]:g}"


# --------------------------------------------------------------------------------------------
# 3. median does not commute, and is pinned to the settled order
# --------------------------------------------------------------------------------------------


class TestMedianIsPinnedToTheSettledOrder:
    """A median does not commute with a mean, so the order is observable — and it is order A.

    ``median(mean over pixels)`` and ``mean over pixels(median)`` are different statistics for any
    bin of three or more frames; for two frames the sample median *is* the arithmetic mean, so the
    divergence only appears once bins are wide enough. Order A — bin the stacks, then collapse — is
    what the pipelines do, so that is what is pinned, with the other order's numbers recorded to show
    the size of the choice.
    """

    #: Two bins of three frames each: wide enough that the median is not the mean.
    RANGES = [(0, 3), (3, 6)]
    BIN_LIST = [[0, 3], [3, 6]]

    def _inputs(self):
        mask = _spatial_mask()
        return (
            _stack(_sample_counts(), _sample_variances(), mask=mask, mask_dims=["y", "x"]),
            _stack(_ob_counts(), _ob_variances(), mask=mask, mask_dims=["y", "x"]),
            ~mask[_Y0:_Y1, _X0:_X1],
        )

    def _oracle_bin_first(self, selected):
        """Per-pixel median over each bin's frames, then the pooled region mean of those medians.

        The median variance is NeuNorm's standard approximation for bins of three or more frames,
        ``Var(median) = (pi / 2n) * mean(Var)``, written out here rather than read back from the
        library.
        """

        def side(counts, variances):
            region, region_var = _region(counts), _region(variances)
            median = np.stack([np.median(region[a:b], axis=0) for a, b in self.RANGES])
            median_var = np.stack([(np.pi / (2 * (b - a))) * region_var[a:b].mean(axis=0) for a, b in self.RANGES])
            return _pooled(median, median_var, selected)

        sample_mean, sample_var = side(_sample_counts(), _sample_variances())
        ob_mean, ob_var = side(_ob_counts(), _ob_variances())
        return _transmission(sample_mean, sample_var, ob_mean, ob_var)

    def _oracle_collapse_first(self, selected):
        """Pooled region mean per frame, then the median of those per-frame means within each bin."""

        def side(counts, variances):
            mean, var = _pooled(_region(counts), _region(variances), selected)
            return (
                np.array([np.median(mean[a:b]) for a, b in self.RANGES]),
                np.array([(np.pi / (2 * (b - a))) * var[a:b].mean() for a, b in self.RANGES]),
            )

        sample_mean, sample_var = side(_sample_counts(), _sample_variances())
        ob_mean, ob_var = side(_ob_counts(), _ob_variances())
        return _transmission(sample_mean, sample_var, ob_mean, ob_var)

    def test_median_spectrum_is_the_bin_first_statistic(self):
        """The shipped result is ``pool(median over frames)``, not ``median(pooled per frame)``."""
        sample, ob, selected = self._inputs()

        result = _order_a(sample, ob, self.BIN_LIST, "median")

        expected, expected_var = self._oracle_bin_first(selected)
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)
        np.testing.assert_allclose(result.variances, expected_var, rtol=1e-12)

    def test_the_other_order_gives_a_measurably_different_spectrum(self):
        """Both medians are pinned and shown apart, so the settled choice is recorded, not incidental."""
        sample, ob, selected = self._inputs()

        first = _order_a(sample, ob, self.BIN_LIST, "median")
        second = _order_b(sample, ob, self.BIN_LIST, "median")

        collapse_first, collapse_first_var = self._oracle_collapse_first(selected)
        np.testing.assert_allclose(second.values, collapse_first, rtol=1e-12)
        np.testing.assert_allclose(second.variances, collapse_first_var, rtol=1e-12)

        assert not np.allclose(first.values, second.values, rtol=1e-5), (
            f"median must not commute with the region mean; got {first.values} and {second.values}"
        )
        relative = np.abs(first.values - second.values) / first.values
        assert np.all(relative > 5e-3), f"the median divergence should be ~1%; got {relative}"

    def test_two_frame_bins_still_commute_because_the_median_is_the_mean_there(self):
        """The divergence is a property of the statistic, not of the median code path.

        For bins of two frames the sample median equals the arithmetic mean exactly, so the same
        median reduction commutes. Pinning this keeps the divergence above attributable to the
        statistic rather than to something incidental in the median branch.
        """
        sample, ob, _ = self._inputs()

        first = _order_a(sample, ob, [[0, 2], [2, 4], [4, 6]], "median")
        second = _order_b(sample, ob, [[0, 2], [2, 4], [4, 6]], "median")

        np.testing.assert_allclose(first.values, second.values, rtol=1e-12)
        np.testing.assert_allclose(first.variances, second.variances, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# 4. a stack with no timing coordinate
# --------------------------------------------------------------------------------------------


class TestBinningAStackWithNoTimingCoordinate:
    """Frame-index binning needs an ``N + 1`` edge coordinate; without one the rebinner refuses.

    Verified by running it, not assumed: ``reduce_tof_bins`` rebuilds its output axis from the input
    edges, so a stack loaded without any timing coordinate (an ``N_image`` axis, as the CCD loaders
    produce) raises ``ValueError`` naming the missing coordinate. The library does **not** invent an
    index axis for you. The working route is therefore to attach a synthetic integer edge coordinate
    of length ``N + 1`` and bin by pure file index, which then behaves exactly like a TOF axis.
    """

    RANGES = [(0, 2), (2, 4), (4, 6)]

    @staticmethod
    def _bare(counts, variances, dim):
        """A stack on ``dim`` carrying no coordinate on that dim at all."""
        return sc.DataArray(
            sc.array(dims=[dim, "y", "x"], values=counts.copy(), variances=variances.copy(), unit="counts")
        )

    @pytest.mark.parametrize("dim", ["N_image", "tof"])
    def test_reduce_tof_bins_refuses_a_stack_with_no_edge_coordinate(self, dim):
        """The refusal is a ``ValueError`` that names the coordinate, on both the low-level reducer
        and the ``rebin_tof`` bin-list entry point."""
        data = self._bare(_sample_counts(), _sample_variances(), dim)

        with pytest.raises(ValueError, match=f"must carry a '{dim}' coordinate"):
            reduce_tof_bins(data, self.RANGES, reduction="sum", tof_dim=dim)
        with pytest.raises(ValueError, match=f"must carry a '{dim}' coordinate"):
            rebin_tof(data, [[0, 2], [2, 4], [4, 6]], reduction="sum", tof_dim=dim)

    def test_a_synthetic_index_edge_coordinate_bins_by_file_index(self):
        """With ``0..N`` integer edges attached, the frames bin by file index and the spectrum reduces.

        The rebuilt axis holds the bin boundaries as file indices and ``spectra_tof`` holds each bin's
        mean member index — 0.5, 2.5, 4.5 for pairs of frames — which is what says on the data itself
        which files a point came from once the frames are combined.
        """
        index_edges = sc.arange("N_image", 0, _N_TOF + 1, unit=None)
        mask = _spatial_mask()
        sample = self._bare(_sample_counts(), _sample_variances(), "N_image")
        ob = self._bare(_ob_counts(), _ob_variances(), "N_image")
        for side in (sample, ob):
            side.coords["N_image"] = index_edges
            side.masks["spike"] = sc.array(dims=["y", "x"], values=mask.copy())

        binned_sample = reduce_tof_bins(sample, self.RANGES, reduction="sum", tof_dim="N_image")
        binned_ob = reduce_tof_bins(ob, self.RANGES, reduction="sum", tof_dim="N_image")
        result = normalize_roi_spectrum(binned_sample, binned_ob, _REGION, tof_dim="N_image")

        assert result.dims == ("N_image",)
        assert result.sizes == {"N_image": 3}
        np.testing.assert_equal(result.coords["N_image"].values, np.array([0, 2, 4, 6]))
        np.testing.assert_allclose(result.coords[SPECTRA_TOF_COORD].values, [0.5, 2.5, 4.5], rtol=1e-12)

        expected, expected_var = _oracle_order_a(self.RANGES, "sum", ~mask[_Y0:_Y1, _X0:_X1])
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)
        np.testing.assert_allclose(result.variances, expected_var, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# 5. a representative time per bin survives a sum-mode rebin
# --------------------------------------------------------------------------------------------


class TestRepresentativeTimeSurvivesASumModeRebin:
    """Every bin gets a time inside its own bin, even though a sum-mode rebin cannot supply one.

    :func:`scipp.rebin` sums. An aligned ``spectra_tof`` is dropped by it; an unaligned one is
    carried through and **summed**, which makes it roughly the rebinning factor too large and puts it
    outside the bin it labels. The reduction therefore checks containment before trusting the
    coordinate and falls back to the bin's left edge, so the representative time a downstream time
    axis is reconstructed from is never a summed one.
    """

    def _stacks(self, *, bin_times=None, aligned=True):
        sample = _stack(_sample_counts(), _sample_variances())
        ob = _stack(_ob_counts(), _ob_variances())
        if bin_times is not None:
            for side in (sample, ob):
                side.coords[SPECTRA_TOF_COORD] = sc.array(dims=["tof"], values=bin_times.copy(), unit="us")
                side.coords.set_aligned(SPECTRA_TOF_COORD, aligned)
        return sample, ob

    def test_every_time_lies_inside_its_own_bin_after_a_sum_rebin(self):
        """A factor-2 sum rebin of an input with no ``spectra_tof`` still yields one per bin.

        Its values are the rebinned bins' left edges, and each is inside ``[left, right)``. That
        containment is the whole test: a time that labels a bin it does not fall in is not a
        representative time, and it is what a downstream time axis would be rebuilt from. The
        three-column ASCII file does not carry the column itself — the coordinate on the returned array
        is where it lives.
        """
        sample, ob = self._stacks()

        result = normalize_roi_spectrum(
            rebin_tof(sample, 2, reduction="sum"), rebin_tof(ob, 2, reduction="sum"), _REGION
        )

        assert SPECTRA_TOF_COORD in result.coords
        times = result.coords[SPECTRA_TOF_COORD]
        assert times.unit == sc.Unit("us")
        edges = result.coords["tof"].values
        np.testing.assert_allclose(edges, _EDGES[::2], rtol=1e-12)
        np.testing.assert_allclose(times.values, [1000.0, 3000.0, 5000.0], rtol=1e-12)
        assert np.all(times.values >= edges[:-1]) and np.all(times.values < edges[1:]), (
            f"every representative time must lie in its own bin; got {times.values} for edges {edges}"
        )

    def test_a_summed_spectra_tof_is_detected_and_replaced_by_the_bin_left_edge(self):
        """An unaligned ``spectra_tof`` survives the sum rebin as a SUM of times, and is rejected.

        The rebinned coordinate is first pinned to the numpy sum of each bin's member times, which is
        the corruption itself: 3000 us for a bin spanning [1000, 3000) us. The reduction then detects
        that the value is outside its bin, warns, and substitutes the left edge — so the output is the
        left edges, not the summed times, and the discrepancy is reported rather than silent.
        """
        per_frame_times = _EDGES[:_N_TOF]  # each frame's left-edge time, the VENUS spectra convention
        sample, ob = self._stacks(bin_times=per_frame_times, aligned=False)

        binned_sample = rebin_tof(sample, 2, reduction="sum")
        binned_ob = rebin_tof(ob, 2, reduction="sum")

        summed = np.array([per_frame_times[a:b].sum() for a, b in [(0, 2), (2, 4), (4, 6)]])
        np.testing.assert_allclose(binned_sample.coords[SPECTRA_TOF_COORD].values, summed, rtol=1e-12)
        left, right = _EDGES[::2][:-1], _EDGES[::2][1:]
        assert np.any(summed >= right), f"the summed times must fall outside their bins; got {summed}"

        messages, remove = _capture_warnings()
        try:
            result = normalize_roi_spectrum(binned_sample, binned_ob, _REGION)
        finally:
            remove()

        np.testing.assert_allclose(result.coords[SPECTRA_TOF_COORD].values, left, rtol=1e-12)
        np.testing.assert_allclose(result.coords[SPECTRA_TOF_COORD].values, [1000.0, 3000.0, 5000.0], rtol=1e-12)
        warned = "\n".join(messages)
        assert SPECTRA_TOF_COORD in warned, f"the substitution must be logged; captured:\n{warned}"
        assert "does not lie within its own TOF bins" in warned, f"captured:\n{warned}"

    def test_an_aligned_spectra_tof_is_dropped_by_the_rebin_and_rebuilt_without_a_warning(self):
        """The other survival mode: dropped rather than corrupted, so no warning is warranted.

        ``scipp.rebin`` carries only unaligned coordinates, so an aligned ``spectra_tof`` simply does
        not reach the reduction. Rebuilding from the edges is then the ordinary path, and warning
        about it would train users to ignore the warning that matters.
        """
        sample, ob = self._stacks(bin_times=_EDGES[:_N_TOF], aligned=True)

        binned_sample = rebin_tof(sample, 2, reduction="sum")
        assert SPECTRA_TOF_COORD not in binned_sample.coords

        messages, remove = _capture_warnings()
        try:
            result = normalize_roi_spectrum(binned_sample, rebin_tof(ob, 2, reduction="sum"), _REGION)
        finally:
            remove()

        np.testing.assert_allclose(result.coords[SPECTRA_TOF_COORD].values, [1000.0, 3000.0, 5000.0], rtol=1e-12)
        assert not [m for m in messages if SPECTRA_TOF_COORD in m], f"unexpected warning:\n{messages}"
