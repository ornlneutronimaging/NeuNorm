"""
ROI-level spectrum reduction — the reduction behind NeuNorm's "resonance mode".

Where the image mode divides sample by open beam pixel by pixel and returns a stack of images, this
module collapses a region of interest to **one number per spectral bin first** and divides second,
returning a 1-D transmission spectrum. That order is the whole point: the ratio of means is not the
mean of ratios, so a region-level measurement must be built from region-level counts.

The spatial reduction is the same mask-aware pooled region mean the ``background_roi`` and
``air_roi`` corrections already use — ``sum(counts over the region) / count(unmasked pixels)`` per
bin — so a region reduces identically whether it is a rectangle, an arbitrary-shape
:class:`~neunorm.data_models.roi.MaskROI`, or a pooled list of either.
"""

from typing import Literal, Optional, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.roi import RegionsLike, as_region_list
from neunorm.processing.normalizer import (
    _pooled_roi_coefficient,
    normalize_step_count,
    normalize_transmission,
)
from neunorm.tof.histogram_rebinner import SPECTRA_TOF_COORD
from neunorm.utils.progress import STAGE_REDUCE_SPECTRUM, ProgressLike, resolve_progress

# The dims a region mean reduces over. Fixed rather than derived: the pooled-coefficient machinery
# these functions delegate to is itself written against ``x``/``y``.
_SPATIAL_DIMS = ("x", "y")


def normalize_roi_spectrum_step_count(proton_charge_sample=None) -> int:
    """How many progress steps :func:`normalize_roi_spectrum` reports for these arguments.

    Its two region collapses plus whatever :func:`~neunorm.processing.normalizer.normalize_transmission`
    reports, because that function is handed a borrowed reporter and a borrowed reporter keeps the
    OUTER total. Deriving both from one function is what stops the two drifting apart — the same
    arrangement as :func:`~neunorm.processing.normalizer.normalize_with_dark_step_count`.

    Parameters
    ----------
    proton_charge_sample : optional
        The ``proton_charge_sample`` that will be passed on, or ``None``.

    Returns
    -------
    int
        The number of counted steps. Mask symmetrization is announced with a note and does not
        advance the count.
    """
    return 2 + normalize_step_count(proton_charge_sample=proton_charge_sample)


def _surviving_dims(data: sc.DataArray) -> tuple[str, ...]:
    """The dims left after the spatial ones are reduced away."""
    return tuple(d for d in data.dims if d not in _SPATIAL_DIMS)


def _carry_over(data: sc.DataArray, result: sc.DataArray, keep_dims: tuple[str, ...]) -> None:
    """Copy the coords and masks a spatial reduction leaves meaningful onto ``result``, in place.

    A coord or mask survives when every dim it carries survived the reduction: scalar metadata
    (``proton_charge``, ``detector``) and anything on the spectral axis (the ``tof`` bin edges, a
    per-bin ``spectra_tof``, a missing-bin mask). Spatial ``(y, x)`` coords and the dead/hot pixel
    masks are consumed by the reduction and are deliberately not carried. Alignment flags are
    preserved, because they are what decides whether scipp enforces equality on a later binary op.
    """
    for name, coord in data.coords.items():
        if set(coord.dims) <= set(keep_dims):
            result.coords[name] = coord
            result.coords.set_aligned(name, coord.aligned)
    for name, mask in data.masks.items():
        if set(mask.dims) <= set(keep_dims):
            result.masks[name] = mask


def roi_mean_spectrum(
    data: sc.DataArray,
    regions: RegionsLike,
    *,
    strict: bool = True,
    region_arg: str = "spectrum_roi",
    name: str = "data",
) -> sc.DataArray:
    """Collapse one or more spatial regions to a mask-aware **pooled mean per spectral bin**.

    The public form of the pooled region mean that
    :func:`~neunorm.processing.normalizer.normalize_transmission` uses for ``background_roi`` and
    :func:`~neunorm.processing.air_region_corrector.apply_air_region_correction` uses for
    ``air_roi``. That private helper stays the single implementation — this function adds the
    argument normalization and rebuilds a :class:`scipp.DataArray` around the result so the spectral
    coordinates survive; it changes no arithmetic.

    ``mean = sum(counts over every region) / count(selected, unmasked pixels)``, evaluated per
    spectral bin. Reductions are mask-aware on **both** sides of that ratio: a dead or hot pixel is
    excluded from the summed counts and from the pixel count alike, so a partially masked region
    still yields its true mean. Regions that overlap within a pooled list are reduced over their
    **union**, each pixel counted once.

    Do not substitute ``sc.mean(data, dim=["y", "x"])``. scipp reduces one dimension at a time, so
    under a mask that form is not the pooled mean and the nested ``mean('x').mean('y')`` is
    order-dependent as well. ``tests/unit/test_roi_spectrum.py`` pins the disagreement so the
    substitution cannot be made by accident.

    Parameters
    ----------
    data : sc.DataArray
        Counts carrying ``x`` and ``y`` dims, and usually a spectral dim such as ``tof``. Masks are
        honoured; variances, when present, propagate into the mean.
    regions : ROI, MaskROI, tuple, or a sequence of them
        The region(s) to collapse — an :class:`~neunorm.data_models.roi.ROI` (or a bare
        ``(x0, y0, x1, y1)`` tuple, exclusive stops), an arbitrary-shape
        :class:`~neunorm.data_models.roi.MaskROI` (selection mask: 1 = pixel in the region), or a
        sequence mixing those, which are pooled.
    strict : bool, optional
        With the default ``True``, a pooled mean that is not strictly positive and finite in every
        bin raises ``ValueError``. Pass ``False`` where a zero or negative mean is a legitimate
        measurement rather than a fault — a fully absorbing sample bin, or dark-subtracted counts
        that scatter below zero. Structural errors (bad bounds, a selection whose shape does not
        match the data, missing dims) always raise.
    region_arg : str, optional
        The caller's argument name, used in error messages. Defaults to ``"spectrum_roi"``.
    name : str, optional
        How to describe ``data`` in error messages (e.g. ``"sample"``, ``"open beam"``).

    Returns
    -------
    sc.DataArray
        The pooled mean with the spatial dims reduced away — 1-D over the spectral axis, or a
        0-D scalar for a plain image. Carries the counts' unit, the propagated variance, and every
        coordinate and mask whose dims survived the reduction (the ``tof`` bin edges among them, so
        the result can be rebinned or exported).

    Examples
    --------
    >>> spectrum = roi_mean_spectrum(sample, (10, 10, 30, 30))  # doctest: +SKIP
    """
    region_list = as_region_list(regions, arg_name=region_arg)
    coeff = _pooled_roi_coefficient(data, region_list, name, strict=strict, region_arg=region_arg)

    keep_dims = _surviving_dims(data)
    # No broadcast is needed here, which is worth stating because it looks as though one might be. The
    # coefficient is `total / n_unmasked`, and while `n_unmasked` IS a bare scalar on the MaskROI path
    # with purely spatial masks, it is only the divisor: `total` is `sc.sum(region, dim=["x", "y"])`,
    # which retains every non-spatial dim, so the quotient already carries exactly `keep_dims` for
    # every input shape, dim order, mask kind and region form.
    result = sc.DataArray(coeff)
    _carry_over(data, result, keep_dims)
    return result


def _bin_times(data: sc.DataArray, tof_dim: str) -> Optional[sc.Variable]:
    """Each bin's representative time, or ``None`` when the axis cannot supply one.

    Prefers an existing ``spectra_tof`` — the per-bin mean of its member frames' left-edge times,
    which is exactly what the reduction path of :func:`~neunorm.tof.histogram_rebinner.rebin_tof`
    attaches and is more informative than anything recoverable from the output edges alone.

    That coordinate is only trusted after a containment check, because the **sum** path of
    ``rebin_tof`` carries unaligned coords through :func:`scipp.rebin`, which SUMS them: a
    ``spectra_tof`` that survived a sum-mode rebin holds the sum of its member times, not their
    mean, and is wrong by roughly a factor of the rebinning factor. A summed time falls outside its
    own bin, so requiring each value to lie within ``[left, right)`` separates a genuine mean time
    from a corrupted one. When the check fails, or when no ``spectra_tof`` is present at all, the
    bin's left edge is used and the substitution is logged.
    """
    if tof_dim not in data.coords:
        return None
    edges = data.coords[tof_dim]
    n = data.sizes[tof_dim]
    if edges.sizes.get(tof_dim) != n + 1:
        # A point coord rather than bin edges: it already is one time per bin.
        return edges if edges.sizes.get(tof_dim) == n else None
    left = edges[tof_dim, :n]
    right = edges[tof_dim, 1:]
    if SPECTRA_TOF_COORD in data.coords:
        existing = data.coords[SPECTRA_TOF_COORD]
        if existing.sizes.get(tof_dim) == n and existing.unit == edges.unit:
            inside = (existing.values >= left.values) & (existing.values < right.values)
            if bool(np.all(inside)):
                return existing
            logger.warning(
                "'{}' does not lie within its own TOF bins (first offending bin: {}); it was most "
                "likely summed by a sum-mode rebin rather than averaged. Using each bin's left edge "
                "as its representative time instead.",
                SPECTRA_TOF_COORD,
                int(np.argmin(inside)),
            )
    return left.copy()


def _require_matching_axes(
    sample: sc.DataArray,
    ob: sc.DataArray,
    *,
    sample_label: Optional[str],
    ob_label: Optional[str],
) -> None:
    """Raise a diagnosable error when the two spectra's aligned axes disagree.

    scipp enforces exact equality on an **aligned** coordinate in a binary op, so two spectra whose
    ``tof`` axes differ raise ``DatasetError: Mismatch in coordinate 'tof' in operation 'divide'`` —
    accurate but no help in locating the cause. The cause is usually mundane and specific: VENUS
    TPX1 reads a separate ``*_Spectra.txt`` for the sample and open-beam directories, so a stale or
    mismatched sidecar puts the two acquisitions on different time axes.

    Reconciling the axes silently is the one thing not done here. Two different time axes mean the
    numerator and the denominator describe different time windows, and dividing them produces a
    plausible-looking spectrum that is not a transmission.

    The SIZES are checked as well as the coords, and that check is what catches a mismatch the image
    mode would have caught for free. Two stacks of different detector extent raise a ``DimensionError``
    when divided per pixel, but a region collapse removes the spatial dims BEFORE the division, so as
    long as the region fits in both, two differently-sized detectors divide happily into a spectrum
    that looks entirely reasonable. Coordinate equality does not cover it, because a stack built by hand
    need carry no ``x``/``y`` coords at all.
    """
    shared = {d: (sample.sizes[d], ob.sizes[d]) for d in sample.dims if d in ob.dims}
    differing = {d: pair for d, pair in shared.items() if pair[0] != pair[1]}
    if differing or set(sample.dims) != set(ob.dims):
        where = ""
        if sample_label or ob_label:
            where = f" (sample: {sample_label or 'unlabelled'}; open beam: {ob_label or 'unlabelled'})"
        raise ValueError(
            f"sample and open beam have different shapes{where}: sample {dict(sample.sizes)}, open beam "
            f"{dict(ob.sizes)}. A region collapse would hide this — it removes the spatial dims before "
            "the division, so two differently-sized detectors would still produce a spectrum."
        )

    for name in sample.coords:
        if not sample.coords[name].aligned or name not in ob.coords:
            continue
        theirs = ob.coords[name]
        if sc.identical(sample.coords[name], theirs):
            continue
        mine = sample.coords[name]
        detail = ""
        if mine.sizes == theirs.sizes and mine.unit == theirs.unit and mine.dtype == theirs.dtype:
            worst = float(np.max(np.abs(mine.values.astype(float) - theirs.values.astype(float))))
            detail = f", max deviation {worst:g} {mine.unit}"
        where = ""
        if sample_label or ob_label:
            where = f" (sample: {sample_label or 'unlabelled'}; open beam: {ob_label or 'unlabelled'})"
        raise ValueError(
            f"sample and open beam disagree on the aligned '{name}' axis{detail}{where}: "
            f"sample {mine.sizes} in {mine.unit}, open beam {theirs.sizes} in {theirs.unit}. "
            "The two acquisitions are on different axes, so their ratio would not be a "
            "transmission — check that each directory's spectra sidecar belongs to it."
        )


def _symmetrize_masks(sample: sc.DataArray, ob: sc.DataArray) -> tuple[sc.DataArray, sc.DataArray]:
    """Give sample and open beam the same exclusions, so their region means share a denominator.

    The pipelines attach the dead/hot pixel masks to the sample and leave the open beam unmasked.
    That is harmless when the division is per pixel — a masked pixel is masked in the result either
    way — but a region **mean** divides by its own count of unmasked pixels, so an asymmetric mask
    makes the numerator and the denominator average over different pixel sets. A pixel known to be
    dead then still contributes to the open beam's mean, inflating it and biasing the transmission
    low, and the bias does not announce itself.

    Both sides therefore get the union of both sides' masks. Returned as shallow copies: the mask
    dicts are rewritten, the data and variance buffers are shared, and the caller's arrays are
    never mutated.
    """
    out_sample = sample.copy(deep=False)
    out_ob = ob.copy(deep=False)
    for name in set(sample.masks) | set(ob.masks):
        in_s = sample.masks.get(name)
        in_o = ob.masks.get(name)
        if in_s is not None and in_o is not None:
            if sc.identical(in_s, in_o):
                continue
            combined = in_s | in_o
        else:
            combined = in_s if in_s is not None else in_o
        for target, source in ((out_sample, sample), (out_ob, ob)):
            if not set(combined.dims) <= set(source.dims):
                raise ValueError(
                    f"cannot symmetrize mask '{name}' with dims {combined.dims} onto an array with "
                    f"dims {source.dims}: sample and open beam must have the same dims at this point"
                )
            target.masks[name] = combined
    return out_sample, out_ob


def normalize_roi_spectrum(  # noqa: C901
    sample: sc.DataArray,
    ob: sc.DataArray,
    spectrum_roi: RegionsLike,
    *,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
    spectrum_roi_strict: bool = True,
    symmetrize_masks: bool = True,
    tof_dim: str = "tof",
    sample_label: Optional[str] = None,
    ob_label: Optional[str] = None,
    progress: ProgressLike = False,
    stage: str = STAGE_REDUCE_SPECTRUM,
) -> sc.DataArray:
    """Reduce a sample and open-beam stack to a 1-D transmission spectrum over one region.

    For each spectral bin the sample's mask-aware pooled mean counts over ``spectrum_roi`` are
    divided by the open beam's pooled mean over the **same** region, giving one data point per bin.

    Order of operations
    -------------------
    The region is collapsed to a scalar per bin **before** the division, never after. The two are
    not the same quantity::

        (Σ sample) / (Σ ob)   ≠   Σ (sample / ob)

    Averaging per-pixel transmissions instead — the ratio of means replaced by the mean of ratios —
    biases the result: measured 1.2% apart on synthetic Poisson counts over a 64-pixel region, and
    the per-pixel form additionally produces NaN wherever an open-beam pixel recorded zero counts in
    a bin, which is common at fine TOF binning. Collapsing first avoids both.
    :func:`~neunorm.tof.resonance.aggregate_resonance_image` states the same identity for the
    spectral direction.

    Any frame-index rebinning belongs **before** this call, applied to both stacks, exactly as in the
    image mode (``rebin_tof(sample, spec); rebin_tof(ob, spec); normalize_roi_spectrum(...)``). Sum
    and mean rebinning commute with the region collapse, so that order costs nothing; median does
    not commute, and the documented order is the one that holds.

    Parameters
    ----------
    sample : sc.DataArray
        Sample counts with ``x``, ``y`` and (usually) a spectral dim, carrying variances.
    ob : sc.DataArray
        Open-beam counts on the same axes.
    spectrum_roi : ROI, MaskROI, tuple, or a sequence of them
        The region whose mean **is** the measurement — distinct from ``roi`` (which crops),
        ``background_roi`` (a flux proxy) and ``air_roi`` (a scale correction). Indices are resolved
        against the arrays as passed: after a pipeline's crop and spatial rebin, that is the
        post-crop, post-rebin frame.
    proton_charge_sample, proton_charge_ob : float or sc.Variable, optional
        Integrated proton charge for the SNS beam correction, forwarded to
        :func:`~neunorm.processing.normalizer.normalize_transmission` together with
        ``pc_uncertainty``, so a spectrum carries the same flux correction and the same 0.5%
        systematic as an image.
    pc_uncertainty : float, optional
        Relative proton-charge uncertainty (default 0.005).
    spectrum_roi_strict : bool, optional
        Guards the **open beam's** region mean, the denominator: with the default ``True``, a bin
        whose open-beam mean is not strictly positive and finite raises ``ValueError`` rather than
        emitting inf or NaN. ``False`` lets it propagate, which is the legacy 1.x behaviour.

        The sample's mean is never guarded this way. Zero counts in the numerator are a real
        measurement — a fully absorbing bin, a black resonance — and must give transmission 0, not
        an exception.
    symmetrize_masks : bool, optional
        With the default ``True``, both sides are given the union of both sides' masks before the
        collapse, so the two means average over the same pixels. See :func:`_symmetrize_masks` for
        why an asymmetric mask biases a region mean when it would not bias a per-pixel division.
    tof_dim : str, optional
        Name of the spectral dim. Default ``"tof"``.
    sample_label, ob_label : str, optional
        How to describe each input if their axes disagree — a directory or sidecar path makes that
        error actionable.
    progress : bool or callable, optional
        Progress reporting, off by default. Reports the two region collapses and then hands its
        reporter to ``normalize_transmission``, so one count spans the whole reduction. A caller
        passing a pre-bound reporter must size it with
        :func:`normalize_roi_spectrum_step_count`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_REDUCE_SPECTRUM``.

    Returns
    -------
    sc.DataArray
        The transmission spectrum: 1-D over ``tof_dim``, dimensionless, with propagated variances.
        Carries the ``N + 1`` ``tof`` **bin edges** — so the spectrum can be rebinned again or
        converted to wavelength or energy — and a ``spectra_tof`` point coordinate giving each bin's
        representative time.

    Raises
    ------
    ValueError
        If the region is malformed or does not fit the data, if the two inputs disagree on an
        aligned axis, or (under ``spectrum_roi_strict``) if an open-beam bin's region mean is not
        strictly positive and finite.

    Examples
    --------
    >>> spectrum = normalize_roi_spectrum(sample, ob, (10, 10, 30, 30))  # doctest: +SKIP
    """
    region_list = as_region_list(spectrum_roi, arg_name="spectrum_roi")
    logger.info("Reducing ROI spectrum over region(s) {}", region_list)

    _require_matching_axes(sample, ob, sample_label=sample_label, ob_label=ob_label)

    with resolve_progress(progress, stage, total=normalize_roi_spectrum_step_count(proton_charge_sample)) as report:
        if symmetrize_masks and (sample.masks or ob.masks):
            report.note("symmetrizing sample and open-beam masks")
            sample, ob = _symmetrize_masks(sample, ob)

        report.note("collapsing sample region")
        # strict=False on the sample: a zero or negative region mean is a legitimate measurement
        # here (full absorption, or dark-subtracted counts scattering below zero), where in the
        # denominator it is a fault. The guard belongs on the open beam alone.
        sample_spectrum = roi_mean_spectrum(sample, region_list, strict=False, region_arg="spectrum_roi", name="sample")
        report()

        report.note("collapsing open-beam region")
        ob_spectrum = roi_mean_spectrum(
            ob, region_list, strict=spectrum_roi_strict, region_arg="spectrum_roi", name="ob"
        )
        report()

        transmission = normalize_transmission(
            sample=sample_spectrum,
            ob=ob_spectrum,
            proton_charge_sample=proton_charge_sample,
            proton_charge_ob=proton_charge_ob,
            pc_uncertainty=pc_uncertainty,
            progress=report,
            stage=stage,
        )

    # A representative time per bin, recovered from the authoritative bin edges when the input's own
    # `spectra_tof` is absent or was corrupted by a sum-mode rebin. Left ALIGNED, which is how
    # `reduce_tof_bins` writes it (histogram_rebinner.py:343 assigns it without clearing the flag),
    # so a later binary op between two spectra whose time axes disagree is refused rather than
    # silently reconciled — the same protection `_require_matching_axes` gives this function's own
    # inputs.
    if tof_dim in transmission.dims:
        times = _bin_times(transmission, tof_dim)
        if times is not None:
            transmission.coords[SPECTRA_TOF_COORD] = times

    logger.success("✓ ROI spectrum reduced to {} bin(s)", transmission.sizes.get(tof_dim, 1))
    return transmission


def spectrum_reduction_provenance(
    regions: list,
    *,
    reduction: Optional[Literal["mean", "sum", "median"]] = None,
    rebin_by_tof: object = None,
) -> dict:
    """Provenance for one spectrum reduction: what region, and how the frames were binned.

    Kept next to the reduction so the record cannot drift from the arithmetic it describes.

    ``rebin_by_tof`` must be the spec that actually RAN, not the caller's argument. They differ for
    ``rebin_by_tof=True``, where the factor comes from the statistics analysis: recording ``"True"``
    leaves the reader unable to say how many frames went into a point. ``reduction`` is likewise
    recorded as the effective one, because the default flips with the argument type — a factor sums, a
    bin list takes the mean — so ``None`` would leave the file silent about which happened.
    """
    from neunorm.data_models.roi import as_region_provenance

    record: dict = {"spectrum_roi": as_region_provenance(regions)}
    if rebin_by_tof is not None and rebin_by_tof is not False:
        record["rebin_by_tof"] = str(rebin_by_tof)
        if reduction is None:
            # rebin_tof's own defaults, spelled out rather than left implicit
            reduction = "mean" if isinstance(rebin_by_tof, (list, tuple)) else "sum"
    if reduction is not None:
        record["rebin_reduction"] = reduction
    return record
