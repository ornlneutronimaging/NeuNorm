"""Collapse-then-divide invariants of ``normalize_roi_spectrum``.

The reduction behind resonance mode returns one transmission point per spectral bin, and its
correctness is almost entirely a question of ORDER: each side's region must be collapsed to a
mask-aware pooled mean *before* the division, because ``(Σ sample) / (Σ ob)`` is not
``Σ (sample / ob)``. The central test therefore computes both forms with plain numpy and pins the
reduction to the first while proving it is measurably far from the second.

Every expected number in this file is a literal or is computed with plain numpy from the synthetic
input. None comes from a second call of the code under test, so a regression cannot move the oracle
along with the result.

Also pinned here: the ``N + 1`` bin edges that keep the spectrum rebinnable, the per-bin
representative time, the variance formula on the collapsed means, the proton-charge systematic, the
diagnosable error raised when the two acquisitions sit on different time axes, the deliberately
one-sided zero guard (a zero open beam is a fault, a zero sample is a measurement), the open-beam
denominator bias an asymmetric pixel mask introduces, and that progress reporting moves no numbers.
"""

import numpy as np
import pytest
import scipp as sc

from neunorm.processing.spectrum_reducer import (
    normalize_roi_spectrum,
    normalize_roi_spectrum_step_count,
)
from neunorm.tof.histogram_rebinner import SPECTRA_TOF_COORD, rebin_tof
from neunorm.utils.progress import STAGE_REDUCE_SPECTRUM

_N_TOF = 5
_DETECTOR = 8
#: ``(x0, y0, x1, y1)`` with exclusive stops: a 4 x 4 = 16-pixel region away from the frame edges.
_REGION = (2, 1, 6, 5)
_PIXELS = (_REGION[2] - _REGION[0]) * (_REGION[3] - _REGION[1])
#: N + 1 TOF bin edges, microseconds.
_EDGES = np.linspace(1000.0, 6000.0, _N_TOF + 1)


# --------------------------------------------------------------------------------------
# synthetic inputs and the numpy oracle
# --------------------------------------------------------------------------------------


def _stack(counts, *, mask=None, edges=_EDGES, bin_times=None):
    """A ``(tof, x, y)`` counts stack carrying Poisson variances and an ``N + 1`` bin-edge axis."""
    counts = np.asarray(counts, dtype=float)
    data = sc.DataArray(sc.array(dims=["tof", "x", "y"], values=counts.copy(), unit="counts", variances=counts.copy()))
    data.coords["tof"] = sc.array(dims=["tof"], values=np.asarray(edges, dtype=float), unit="us")
    if bin_times is not None:
        data.coords[SPECTRA_TOF_COORD] = sc.array(dims=["tof"], values=np.asarray(bin_times, dtype=float), unit="us")
    if mask is not None:
        data.masks["hot_pixels"] = sc.array(dims=["x", "y"], values=np.asarray(mask, dtype=bool))
    return data


def _poisson_counts(rate, seed):
    """Counting statistics with real spread, so the ratio of means and the mean of ratios differ."""
    return np.random.default_rng(seed).poisson(rate, size=(_N_TOF, _DETECTOR, _DETECTOR)).astype(float)


def _in_region(counts):
    """The region's pixels, flattened per bin — the oracle's view of the ROI."""
    x0, y0, x1, y1 = _REGION
    return np.asarray(counts, dtype=float)[:, x0:x1, y0:y1].reshape(_N_TOF, -1)


def _pooled_mean(counts):
    """``sum(region counts) / pixel count`` per bin, the collapse the reduction must perform."""
    return _in_region(counts).sum(axis=1) / _PIXELS


def _pooled_mean_variance(counts):
    """Variance of that pooled mean for Poisson counts (``Var = counts``): ``Σ Var / n²``."""
    return _in_region(counts).sum(axis=1) / _PIXELS**2


def _collect():
    """A progress sink and the list it appends every event to."""
    events = []
    return events, events.append


# --------------------------------------------------------------------------------------
# the order of operations — the whole point of the mode
# --------------------------------------------------------------------------------------


def test_spectrum_is_the_ratio_of_means_and_not_the_mean_of_ratios():
    """One point per bin equals ``mean(sample[roi]) / mean(ob[roi])``, never ``mean(sample/ob)``.

    The two are different quantities, and on these Poisson counts (16 pixels, rates 12 and 20) they
    disagree bin by bin by between 4.1% and 14.4% — the per-pixel form biased high, as Jensen's
    inequality requires. A reduction that divided the stacks first and collapsed second would land on
    the wrong one of those two numbers while still looking like a plausible transmission spectrum,
    which is why both are computed here and the result is held away from the wrong one.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    in_sample, in_ob = _in_region(sample_counts), _in_region(ob_counts)

    ratio_of_means = in_sample.mean(axis=1) / in_ob.mean(axis=1)
    mean_of_ratios = (in_sample / in_ob).mean(axis=1)
    gap = (mean_of_ratios - ratio_of_means) / ratio_of_means
    # no zero open-beam pixel, so the wrong form is finite and the comparison is meaningful
    assert in_ob.min() > 0.0
    assert np.all(gap > 0.0), f"the per-pixel form must sit above the ratio of means; got {gap}"
    assert gap.min() > 0.04, f"the two forms must differ measurably; got {gap}"
    assert gap.max() < 0.15, f"gap unexpectedly large, oracle drifted: {gap}"

    spectrum = normalize_roi_spectrum(_stack(sample_counts), _stack(ob_counts), _REGION)

    assert spectrum.dims == ("tof",)
    assert spectrum.unit == "dimensionless"
    np.testing.assert_allclose(spectrum.values, ratio_of_means, rtol=1e-12)
    assert not np.allclose(spectrum.values, mean_of_ratios, rtol=0.02)


# --------------------------------------------------------------------------------------
# what the output carries
# --------------------------------------------------------------------------------------


def test_output_keeps_the_n_plus_one_bin_edges_and_one_time_per_bin():
    """The spectrum carries bin EDGES, not just a point coord, so it can still be rebinned.

    A 1-D result with only ``N`` point times would be a dead end: ``rebin_tof`` needs the ``N + 1``
    edge coordinate on the dimension it bins, and so does any later conversion to wavelength or
    energy. The rebin is exercised rather than asserted about, since only that proves the surviving
    coordinate is usable and not merely present.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    ratio_of_means = _in_region(sample_counts).mean(axis=1) / _in_region(ob_counts).mean(axis=1)

    spectrum = normalize_roi_spectrum(_stack(sample_counts), _stack(ob_counts), _REGION)

    assert spectrum.sizes == {"tof": _N_TOF}
    assert spectrum.coords["tof"].sizes["tof"] == _N_TOF + 1
    np.testing.assert_allclose(spectrum.coords["tof"].values, _EDGES, rtol=1e-12)

    times = spectrum.coords[SPECTRA_TOF_COORD]
    assert times.sizes["tof"] == _N_TOF
    np.testing.assert_allclose(times.values, _EDGES[:-1], rtol=1e-12)

    # the edges are load-bearing: summing all five bins into one must still work on the output
    rebinned = rebin_tof(spectrum, width=_N_TOF)
    assert rebinned.sizes == {"tof": 1}
    np.testing.assert_allclose(rebinned.values, [ratio_of_means.sum()], rtol=1e-12)


def test_an_existing_per_bin_mean_time_is_kept_rather_than_replaced_by_the_left_edge():
    """A ``spectra_tof`` that lies inside its own bins is the better time and survives the reduction.

    The left edge is only the fallback. When the inputs carry the per-bin mean of their member
    frames' times — what a mean-mode frame rebin attaches — that is the column the ASCII consumers
    want, and overwriting it with the left edge would silently coarsen the time axis.
    """
    midpoints = 0.5 * (_EDGES[:-1] + _EDGES[1:])
    sample = _stack(_poisson_counts(12.0, seed=1), bin_times=midpoints)
    ob = _stack(_poisson_counts(20.0, seed=2), bin_times=midpoints)

    spectrum = normalize_roi_spectrum(sample, ob, _REGION)

    np.testing.assert_allclose(spectrum.coords[SPECTRA_TOF_COORD].values, midpoints, rtol=1e-12)


# --------------------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------------------


def test_variance_is_propagated_from_the_collapsed_means():
    """``Var(T) = Var(S)/O² + S²·Var(O)/O⁴`` evaluated on the POOLED MEANS, not on pixels.

    The uncertainty column of the ASCII file is the square root of this. Propagating from per-pixel
    variances and then averaging would overstate it by roughly the pixel count, so the variance has
    to travel through the same collapse-then-divide path the values do.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    s, o = _pooled_mean(sample_counts), _pooled_mean(ob_counts)
    var_s, var_o = _pooled_mean_variance(sample_counts), _pooled_mean_variance(ob_counts)
    expected = var_s / o**2 + s**2 * var_o / o**4

    spectrum = normalize_roi_spectrum(_stack(sample_counts), _stack(ob_counts), _REGION)

    assert spectrum.variances is not None
    np.testing.assert_allclose(spectrum.variances, expected, rtol=1e-12)


def test_proton_charge_rescales_the_spectrum_and_adds_its_systematic():
    """A spectrum carries the same flux correction and the same 0.5% systematic as an image.

    Values follow ``(S/pc_s) / (O/pc_o)``, so an unequal charge pair rescales the whole spectrum by
    ``pc_ob / pc_sample``; the relative charge uncertainty then enters each side's variance before
    the division and can only make the result less certain. Both are pinned against the hand
    computation, and the ``pc_uncertainty=0`` run is the control that shows the systematic is
    actually applied rather than merely accepted as an argument.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    pc_sample, pc_ob, pc_rel = 400.0, 500.0, 0.005

    s = _pooled_mean(sample_counts) / pc_sample
    o = _pooled_mean(ob_counts) / pc_ob
    var_s = _pooled_mean_variance(sample_counts) / pc_sample**2
    var_o = _pooled_mean_variance(ob_counts) / pc_ob**2
    expected_values = s / o
    expected_var_plain = var_s / o**2 + s**2 * var_o / o**4
    var_s_sys = var_s + (pc_rel * s) ** 2
    var_o_sys = var_o + (pc_rel * o) ** 2
    expected_var_sys = var_s_sys / o**2 + s**2 * var_o_sys / o**4

    corrected = normalize_roi_spectrum(
        _stack(sample_counts),
        _stack(ob_counts),
        _REGION,
        proton_charge_sample=pc_sample,
        proton_charge_ob=pc_ob,
    )
    without_systematic = normalize_roi_spectrum(
        _stack(sample_counts),
        _stack(ob_counts),
        _REGION,
        proton_charge_sample=pc_sample,
        proton_charge_ob=pc_ob,
        pc_uncertainty=0.0,
    )

    assert corrected.unit == "dimensionless"
    np.testing.assert_allclose(corrected.values, expected_values, rtol=1e-12)
    np.testing.assert_allclose(corrected.variances, expected_var_sys, rtol=1e-12)
    np.testing.assert_allclose(without_systematic.values, expected_values, rtol=1e-12)
    np.testing.assert_allclose(without_systematic.variances, expected_var_plain, rtol=1e-12)
    # the correction is a rescaling by the charge ratio, and the systematic strictly widens sigma
    uncorrected = _pooled_mean(sample_counts) / _pooled_mean(ob_counts)
    np.testing.assert_allclose(corrected.values, uncorrected * (pc_ob / pc_sample), rtol=1e-12)
    assert np.all(np.sqrt(corrected.variances) > np.sqrt(without_systematic.variances))


# --------------------------------------------------------------------------------------
# the two axes must be the same axis
# --------------------------------------------------------------------------------------


def test_disagreeing_time_axes_raise_a_diagnosable_error_rather_than_a_scipp_mismatch():
    """Two acquisitions on different time axes are refused with an error that names the cause.

    TPX1 reads a separate ``*_Spectra.txt`` beside each directory, so a stale sidecar is the ordinary
    way this happens — and scipp's own refusal (``Mismatch in coordinate 'tof' in operation
    'divide'``) is accurate but tells the user nothing about which two files to look at. What is
    pinned is that the raised error is a ``ValueError`` naming the coordinate, both labels and the
    size of the disagreement, and that scipp's wording is NOT what reaches the user.
    """
    sample = _stack(_poisson_counts(12.0, seed=1))
    ob = _stack(_poisson_counts(20.0, seed=2), edges=_EDGES + 3.0)

    # what the user would see without the guard: a scipp error, and not even a ValueError
    with pytest.raises(sc.DatasetError, match="Mismatch in coordinate"):
        _ = sample / ob
    assert not issubclass(sc.DatasetError, ValueError)

    with pytest.raises(ValueError) as raised:
        normalize_roi_spectrum(
            sample,
            ob,
            _REGION,
            sample_label="run_1/img_00000_Spectra.txt",
            ob_label="ob_run/img_00000_Spectra.txt",
        )

    message = str(raised.value)
    assert "'tof'" in message, message
    assert "run_1/img_00000_Spectra.txt" in message, message
    assert "ob_run/img_00000_Spectra.txt" in message, message
    assert "max deviation 3" in message, message
    assert "Mismatch in coordinate" not in message, message


def test_matching_edges_but_disagreeing_per_bin_times_are_still_refused():
    """The guard covers every aligned coordinate, not just the binning axis.

    ``spectra_tof`` is written aligned, so a sample and an open beam whose bin EDGES agree while
    their per-bin mean times do not are still two different acquisitions — the case a stale TPX1
    sidecar actually produces. scipp would refuse the division on that coordinate instead, with the
    same unusable wording, so the same diagnosable error has to cover it.
    """
    midpoints = 0.5 * (_EDGES[:-1] + _EDGES[1:])
    sample = _stack(_poisson_counts(12.0, seed=1), bin_times=midpoints)
    ob = _stack(_poisson_counts(20.0, seed=2), bin_times=midpoints + 7.0)

    with pytest.raises(ValueError) as raised:
        normalize_roi_spectrum(sample, ob, _REGION, sample_label="sample_dir", ob_label="ob_dir")

    message = str(raised.value)
    assert f"'{SPECTRA_TOF_COORD}'" in message, message
    assert "sample_dir" in message and "ob_dir" in message, message
    assert "max deviation 7" in message, message
    assert "Mismatch in coordinate" not in message, message


# --------------------------------------------------------------------------------------
# the zero guard, and why it is one-sided
# --------------------------------------------------------------------------------------


def test_zero_open_beam_bin_raises_under_strict_and_propagates_inf_without():
    """A bin with no open beam is a fault in the DENOMINATOR: strict refuses, legacy propagates.

    An open-beam region mean of zero cannot produce a transmission, only ``inf``/``nan``. The default
    stops there and names the argument; ``spectrum_roi_strict=False`` reproduces 1.x, which emitted
    the non-finite value, and is kept for downstreams reproducing legacy output. Either way the other
    bins are untouched.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    x0, y0, x1, y1 = _REGION
    dead_bin = 2
    ob_counts[dead_bin, x0:x1, y0:y1] = 0.0
    with np.errstate(divide="ignore"):  # the oracle's own dead bin is 1/0 by construction
        expected = _pooled_mean(sample_counts) / _pooled_mean(ob_counts)

    with pytest.raises(ValueError) as raised:
        normalize_roi_spectrum(_stack(sample_counts), _stack(ob_counts), _REGION)
    assert "spectrum_roi" in str(raised.value), str(raised.value)
    assert "strictly positive" in str(raised.value), str(raised.value)

    legacy = normalize_roi_spectrum(_stack(sample_counts), _stack(ob_counts), _REGION, spectrum_roi_strict=False)
    assert np.isinf(legacy.values[dead_bin])
    assert not np.isfinite(legacy.variances[dead_bin])
    surviving = [i for i in range(_N_TOF) if i != dead_bin]
    np.testing.assert_allclose(legacy.values[surviving], expected[surviving], rtol=1e-12)


def test_a_fully_absorbing_sample_bin_is_zero_transmission_even_under_strict():
    """Zero counts in the NUMERATOR are a measurement, not a fault — a black resonance reads 0.

    This asymmetry is deliberate: ``spectrum_roi_strict`` guards the open beam alone. Raising on a
    zero sample region would reject exactly the feature the mode exists to measure, so the strict
    default must still return 0.0 for that bin and finite values everywhere else.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    x0, y0, x1, y1 = _REGION
    black_bin = 3
    sample_counts[black_bin, x0:x1, y0:y1] = 0.0
    expected = _pooled_mean(sample_counts) / _pooled_mean(ob_counts)

    spectrum = normalize_roi_spectrum(_stack(sample_counts), _stack(ob_counts), _REGION)

    np.testing.assert_allclose(spectrum.values[black_bin], 0.0, atol=0.0)
    np.testing.assert_allclose(spectrum.values, expected, rtol=1e-12)
    assert np.all(np.isfinite(spectrum.values))


# --------------------------------------------------------------------------------------
# masks: a region mean divides by its own pixel count
# --------------------------------------------------------------------------------------


def test_mask_symmetrization_removes_the_open_beam_denominator_bias():
    """A pixel masked on the sample only must be excluded from the open beam's mean as well.

    The pipelines attach the dead/hot masks to the sample and leave the open beam unmasked. Under a
    per-pixel division that is harmless, but a region MEAN divides by its own count of unmasked
    pixels, so the two sides would average over different pixel sets and the open beam's mean would
    keep a pixel known to be untrustworthy.

    Constructed to be non-vacuous: the masked pixel is three times as bright as the rest of the
    region in the open beam, so excluding it moves the denominator from 22.5 to 20.0 — a 12.5% bias
    that a uniform flux would have hidden entirely, both branches then agreeing.
    """
    levels = np.array([10.0, 12.0, 8.0, 9.0, 11.0])
    sample_counts = np.broadcast_to(levels[:, None, None], (_N_TOF, _DETECTOR, _DETECTOR)).copy()
    ob_counts = np.full((_N_TOF, _DETECTOR, _DETECTOR), 20.0)
    hot = np.zeros((_DETECTOR, _DETECTOR), dtype=bool)
    hot[3, 2] = True  # inside _REGION
    sample_counts[:, 3, 2] = levels * 3.0
    ob_counts[:, 3, 2] = 60.0

    kept = ~_in_region(np.broadcast_to(hot, (_N_TOF, _DETECTOR, _DETECTOR)))[0].astype(bool)
    assert kept.sum() == _PIXELS - 1
    sample_mean = _in_region(sample_counts)[:, kept].mean(axis=1)
    ob_mean_kept = _in_region(ob_counts)[:, kept].mean(axis=1)
    ob_mean_all = _in_region(ob_counts).mean(axis=1)
    # not vacuous: the two denominators genuinely differ
    assert np.all(np.abs(ob_mean_all - ob_mean_kept) / ob_mean_kept > 0.1)
    unbiased = sample_mean / ob_mean_kept
    biased = sample_mean / ob_mean_all

    sample, ob = _stack(sample_counts, mask=hot), _stack(ob_counts)
    symmetrized = normalize_roi_spectrum(sample, ob, _REGION)
    asymmetric = normalize_roi_spectrum(
        _stack(sample_counts, mask=hot), _stack(ob_counts), _REGION, symmetrize_masks=False
    )

    np.testing.assert_allclose(symmetrized.values, unbiased, rtol=1e-12)
    np.testing.assert_allclose(asymmetric.values, biased, rtol=1e-12)
    assert np.all(asymmetric.values < symmetrized.values)
    # the union of the masks is applied to copies: the caller's open beam does not acquire a mask
    assert list(ob.masks) == []
    assert list(sample.masks) == ["hot_pixels"]


# --------------------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("proton_charge", "declared"),
    [(None, 3), (400.0, 5)],
    ids=["no_proton_charge", "with_proton_charge"],
)
def test_step_count_predicts_the_events_actually_emitted(proton_charge, declared):
    """The declared total matches what is emitted on BOTH argument branches.

    A caller sizing a bar with ``normalize_roi_spectrum_step_count`` must get a bar that finishes:
    the two region collapses plus whatever the division reports, which is one step without a proton
    charge and three with it (each side's charge correction counts). Every event carries that same
    total, and the count only moves forwards, so a borrowed reporter cannot restart it mid-reduction.
    """
    assert normalize_roi_spectrum_step_count(proton_charge) == declared

    events, sink = _collect()
    charges = {} if proton_charge is None else {"proton_charge_sample": proton_charge, "proton_charge_ob": 500.0}
    normalize_roi_spectrum(
        _stack(_poisson_counts(12.0, seed=1)),
        _stack(_poisson_counts(20.0, seed=2)),
        _REGION,
        progress=sink,
        **charges,
    )

    assert events, "a callable sink must receive events"
    completed = [event.completed for event in events]
    assert max(completed) == declared, completed
    assert completed == sorted(completed), completed
    assert {event.total for event in events} == {declared}
    assert {event.stage for event in events} == {STAGE_REDUCE_SPECTRUM}


def test_progress_reporting_does_not_change_the_numbers():
    """Reporting is observation only: the spectrum is the same with and without a sink.

    Worth pinning because the reduction hands its reporter down to the normalization rather than
    opening a second count, so the progress argument reaches the arithmetic path itself.
    """
    sample_counts = _poisson_counts(12.0, seed=1)
    ob_counts = _poisson_counts(20.0, seed=2)
    hot = np.zeros((_DETECTOR, _DETECTOR), dtype=bool)
    hot[3, 2] = True  # exercises the mask-symmetrization note as well

    silent = normalize_roi_spectrum(_stack(sample_counts, mask=hot), _stack(ob_counts), _REGION)
    events, sink = _collect()
    reported = normalize_roi_spectrum(_stack(sample_counts, mask=hot), _stack(ob_counts), _REGION, progress=sink)

    assert events
    assert reported.dims == silent.dims
    np.testing.assert_allclose(reported.values, silent.values, rtol=1e-12)
    np.testing.assert_allclose(reported.variances, silent.variances, rtol=1e-12)
    np.testing.assert_allclose(reported.coords["tof"].values, silent.coords["tof"].values, rtol=1e-12)
    np.testing.assert_allclose(
        reported.coords[SPECTRA_TOF_COORD].values, silent.coords[SPECTRA_TOF_COORD].values, rtol=1e-12
    )


# --------------------------------------------------------------------------------------
# mask symmetrization: the branches the pipelines do not reach, but a direct caller can
# --------------------------------------------------------------------------------------


def test_masks_already_identical_on_both_sides_are_left_alone():
    """Symmetrization is a no-op when both sides already carry the same mask under the same name.

    The union of a mask with itself is that mask, so the numbers must match a run where only one side
    carried it — pinned against the hand-computed mean over the surviving pixels either way, so this
    cannot pass by both paths being equally wrong.
    """
    sample_counts = _poisson_counts(12.0, seed=5)
    ob_counts = _poisson_counts(20.0, seed=6)
    dead = np.zeros((_DETECTOR, _DETECTOR), dtype=bool)
    dead[2, 3] = True

    both = normalize_roi_spectrum(_stack(sample_counts, mask=dead), _stack(ob_counts, mask=dead), _REGION)
    one_side = normalize_roi_spectrum(_stack(sample_counts, mask=dead), _stack(ob_counts), _REGION)

    # the stack is (tof, x, y) and the mask is (x, y), so the region slice is [:, x0:x1, y0:y1] —
    # matching _in_region, which is reused here rather than re-derived
    x0, y0, x1, y1 = _REGION
    keep = ~dead[x0:x1, y0:y1].reshape(-1)
    n_keep = int(keep.sum())
    assert n_keep == _PIXELS - 1, "the dead pixel must fall inside the region, or this pins nothing"
    expected = (_in_region(sample_counts)[:, keep].sum(axis=1) / n_keep) / (
        _in_region(ob_counts)[:, keep].sum(axis=1) / n_keep
    )

    np.testing.assert_allclose(both.values, expected, rtol=1e-12)
    np.testing.assert_allclose(one_side.values, expected, rtol=1e-12)


def test_differently_shaped_stacks_are_refused_rather_than_collapsed_into_a_plausible_spectrum():
    """A shape mismatch the image mode catches for free must not slip through the region collapse.

    Dividing two differently-sized stacks per pixel raises a scipp ``DimensionError``. A region collapse
    removes the spatial dims BEFORE the division, so as long as the region fits in both, two different
    detectors would divide happily into a spectrum that looks entirely reasonable — the worst kind of
    wrong. Coordinate equality does not cover it, because a hand-built stack carries no x/y coords.
    """
    counts = _poisson_counts(12.0, seed=7)
    taller = np.repeat(counts, 2, axis=2)[:, :, : _DETECTOR + 2]

    with pytest.raises(ValueError, match="different shapes"):
        normalize_roi_spectrum(_stack(counts), _stack(taller), _REGION)

    # and the message names the inputs when they were labelled
    with pytest.raises(ValueError, match="sampleA"):
        normalize_roi_spectrum(_stack(counts), _stack(taller), _REGION, sample_label="sampleA", ob_label="obB")


def test_a_mask_whose_dims_the_other_side_lacks_is_refused():
    """Symmetrizing needs both sides on the same dims, and says so rather than raising from scipp.

    Reached through :func:`~neunorm.tof.resonance.detect_resonances`, which symmetrizes without the
    shape check above — so this guard is what stands between a caller and a ``DimensionError`` three
    frames down. Exercised on the helper directly, because ``normalize_roi_spectrum`` now rejects the
    shape mismatch first and would never get here.
    """
    from neunorm.processing.spectrum_reducer import _symmetrize_masks

    counts = _poisson_counts(12.0, seed=7)
    sample = _stack(counts)
    sample.masks["per_frame"] = sc.array(
        dims=["tof", "x", "y"], values=np.zeros((_N_TOF, _DETECTOR, _DETECTOR), dtype=bool)
    )
    flat = sc.DataArray(sc.array(dims=["x", "y"], values=counts[0].copy(), variances=counts[0].copy(), unit="counts"))

    with pytest.raises(ValueError, match="cannot symmetrize mask 'per_frame'"):
        _symmetrize_masks(sample, flat)


def test_provenance_records_the_binning_that_produced_the_spectrum():
    """The record has to name the reduction and the bin spec, or a gapped spectrum cannot be traced.

    Values are pinned as literals; ``spectrum_reduction_provenance`` is JSON-writer-safe, so the bin
    spec is stringified rather than left as a nested list the TIFF metadata writer would reject.
    """
    from neunorm.data_models.roi import as_region_list
    from neunorm.processing.spectrum_reducer import spectrum_reduction_provenance

    plain = spectrum_reduction_provenance(as_region_list(_REGION))
    assert plain == {"spectrum_roi": list(_REGION)}
    assert "rebin_by_tof" not in plain, "no binning was requested, so none is recorded"

    binned = spectrum_reduction_provenance(as_region_list(_REGION), reduction="median", rebin_by_tof=[[0, 2], [4, 6]])
    assert binned["spectrum_roi"] == list(_REGION)
    assert binned["rebin_by_tof"] == "[[0, 2], [4, 6]]"
    assert binned["rebin_reduction"] == "median"

    # rebin_by_tof=False is "no binning", not a binning of False
    assert "rebin_by_tof" not in spectrum_reduction_provenance(as_region_list(_REGION), rebin_by_tof=False)
