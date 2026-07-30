"""
Transmission normalization for neutron imaging.

Implements the core neutron imaging equation: T = Sample / OpenBeam
with proper uncertainty propagation and beam corrections.
"""

import collections.abc
from typing import Optional, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.roi import ROI, MaskROI, RegionsLike, as_region_list, as_roi_bounds
from neunorm.processing.dark_corrector import subtract_dark
from neunorm.utils.progress import STAGE_NORMALIZE, ProgressLike, resolve_progress

# One region (rectangle or MaskROI) or a sequence of regions (pooled). A bare 4-int tuple/list is a
# single rectangle. Kept as a named alias for backward compatibility with existing imports.
BackgroundROILike = RegionsLike


def normalize_step_count(background_roi=None, proton_charge_sample=None) -> int:
    """How many progress steps :func:`normalize_transmission` will report for these arguments.

    Shared with :func:`normalize_with_dark`, which reports its own dark subtractions and then hands its
    reporter down: a borrowed reporter keeps the OUTER total, so the caller must declare the combined
    count. Deriving both from one function is what stops the two drifting apart.

    Public for the same reason a pipeline needs :data:`~neunorm.filters.gamma_filter.GAMMA_FILTER_STEPS`:
    a caller that hands in a pre-bound ``ProgressReporter`` must declare that stage's total itself,
    because ``resolve_progress`` does not let a callee re-bind one.

    Parameters
    ----------
    background_roi : optional
        The ``background_roi`` that will be passed to :func:`normalize_transmission`, or ``None``.
    proton_charge_sample : optional
        The ``proton_charge_sample`` that will be passed, or ``None``. Ignored when
        ``background_roi`` is given, since the two corrections are mutually exclusive.

    Returns
    -------
    int
        The number of counted steps. Work announced with a note (the background-ROI variance term)
        is deliberately not included: it does not advance the count.
    """
    n_steps = 1  # the division itself always runs
    if background_roi is not None:
        n_steps += 1
    elif proton_charge_sample is not None:
        n_steps += 2  # sample and OB are separate full-array divisions
    return n_steps


def normalize_with_dark_step_count(background_roi=None, proton_charge_sample=None) -> int:
    """How many progress steps :func:`normalize_with_dark` will report for these arguments.

    Its two dark subtractions plus whatever the delegate reports, so a pipeline declaring this as one
    stage of a run gets one continuous count instead of a bar that stops short. See
    :func:`normalize_step_count`.

    Returns
    -------
    int
        The number of counted steps, dark subtractions included.
    """
    return 2 + normalize_step_count(background_roi, proton_charge_sample)


def _as_plain_int_bounds(bounds: tuple) -> tuple[int, int, int, int]:
    """Coerce NumPy integer bounds to built-in ``int`` (JSON provenance stays numeric)."""
    return tuple(int(v) if isinstance(v, np.integer) else v for v in bounds)


def as_roi_bounds_list(background_roi: BackgroundROILike) -> list[tuple[int, int, int, int]]:
    """Normalize a ``background_roi`` argument to a list of exclusive ``(x0, y0, x1, y1)`` bounds.

    Accepts a single ROI (an :class:`~neunorm.data_models.roi.ROI` or a bare 4-int ``(x0,y0,x1,y1)``
    sequence) — backward compatible — or a **sequence** of those (pooled). A bare 4-int sequence is
    treated as ONE ROI; a sequence whose elements are ROIs or sequences is a list of ROIs. NumPy
    integer bounds are coerced to built-in ``int`` so provenance JSON-encodes losslessly.
    """
    if isinstance(background_roi, ROI):
        return [background_roi.as_bounds()]
    if isinstance(background_roi, (str, bytes)) or not isinstance(background_roi, collections.abc.Sequence):
        raise ValueError(
            f"background_roi must be an ROI, an (x0, y0, x1, y1) tuple, or a sequence of those; got {background_roi!r}"
        )
    if len(background_roi) == 4 and all(isinstance(i, (int, np.integer)) for i in background_roi):
        return [_as_plain_int_bounds(as_roi_bounds(tuple(background_roi)))]
    if len(background_roi) == 0:
        raise ValueError("background_roi list must contain at least one ROI")
    # a bare sequence of ints is a SINGLE ROI (handled above when len == 4); an int element here
    # means a malformed single ROI (wrong length), not a sequence of ROIs.
    if any(isinstance(e, (int, np.integer)) for e in background_roi):
        raise ValueError(f"background_roi must be a tuple of 4 integers (x0, y0, x1, y1); got {background_roi!r}")
    return [_as_plain_int_bounds(as_roi_bounds(r)) for r in background_roi]


def _unmasked_count(region: sc.DataArray) -> sc.Variable:
    """Per-spectral count of unmasked pixels in a spatial ROI (reduces x, y only, mask-aware).

    Sums a dimensionless field of ones carrying the region's masks, so masked pixels are excluded and
    a per-image ``(spectral, x, y)`` mask yields a per-spectral count (a scalar for a 2D/absent mask).
    """
    counter = region.copy()
    counter.data = sc.ones(sizes=region.data.sizes, dtype="int64", unit="one")
    return sc.sum(counter, dim=["x", "y"]).data


def _masked_ones_count(region: sc.DataArray) -> sc.Variable:
    """Unmasked-pixel count like ``_unmasked_count``, but sharing buffers instead of deep-copying.

    Used on the MaskROI path, where ``region`` can be a large bounding-box view: builds the ones
    counter directly (no copy of values/variances). When every mask is spatial the counter is 2D —
    the count is then a scalar rather than ``_unmasked_count``'s constant per-spectral vector, which
    divides/broadcasts identically.
    """
    if all(set(m.dims) <= {"x", "y"} for m in region.masks.values()):
        sizes = {d: s for d, s in region.sizes.items() if d in ("x", "y")}
    else:
        sizes = dict(region.data.sizes)
    counter = sc.DataArray(
        sc.ones(sizes=sizes, dtype="int64", unit="one"),
        masks={k: m for k, m in region.masks.items()},
    )
    return sc.sum(counter, dim=["x", "y"]).data


def _unique_mask_name(existing, base: str = "_region_sel") -> str:
    """A mask name not colliding with ``existing`` (a same-named user mask must never be replaced)."""
    name = base
    i = 1
    while name in existing:
        name = f"{base}_{i}"
        i += 1
    return name


def _mask_region_view(
    data: sc.DataArray, region: MaskROI, name: str, region_arg: str = "background_roi"
) -> sc.DataArray:
    """Bounding-box view of ``data`` with the region's inverse selection attached as a scipp mask.

    The bbox slice keeps temporaries proportional to the region, not the frame; the shallow copy
    shares data/variance buffers and never mutates ``data``'s own masks. The attached exclusion
    (``~selection``) composes (OR) with any existing dead/hot/per-image masks, so the same
    mask-aware reductions used for rectangles yield sum/count over ``selected & unmasked`` pixels.
    """
    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError(f"{name} must have 'x' and 'y' dimensions for {region_arg} normalization")
    ny, nx = region.shape
    if ny != data.sizes["y"] or nx != data.sizes["x"]:
        raise ValueError(
            f"{name} MaskROI selection shape (ny={ny}, nx={nx}) does not match data size "
            f"(y={data.sizes['y']}, x={data.sizes['x']})"
        )
    x0, y0, x1, y1 = region.bounding_box()
    work = data["x", x0:x1]["y", y0:y1].copy(deep=False)
    work.masks[_unique_mask_name(work.masks.keys())] = sc.array(dims=["y", "x"], values=~region.selection[y0:y1, x0:x1])
    return work


def _pooled_union_selection(rois_bounds, ny: int, nx: int) -> np.ndarray:
    """Boolean ``(ny, nx)`` union of every pooled region (rectangles and masks) — each pixel once.

    Collapses an overlapping pooled list to one equivalent selection so shared pixels are counted
    once. Bounds/shapes are assumed already validated (by ``_pooled_roi_coefficient``).
    """
    union = np.zeros((ny, nx), dtype=bool)
    for region_spec in rois_bounds:
        if isinstance(region_spec, MaskROI):
            union |= region_spec.selection
        else:
            x0, y0, x1, y1 = region_spec
            union[y0:y1, x0:x1] = True
    return union


def _pooled_regions_overlap(rois_bounds, ny: int, nx: int) -> bool:
    """True if any pixel is selected by more than one region in a pooled list (validated bounds).

    Rectangle-only lists (the common pooled case) use a cheap pairwise interval-overlap test with no
    per-frame allocation; a per-pixel ``ny*nx`` coverage map is built only when a ``MaskROI`` is
    present, where arbitrary shapes make pairwise geometry insufficient.
    """
    if not any(isinstance(r, MaskROI) for r in rois_bounds):
        # exclusive-stop rectangles overlap iff they intersect on both axes
        for i, (ax0, ay0, ax1, ay1) in enumerate(rois_bounds):
            for bx0, by0, bx1, by1 in rois_bounds[i + 1 :]:
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    return True
        return False
    coverage = np.zeros((ny, nx), dtype=np.int32)
    for region_spec in rois_bounds:
        if isinstance(region_spec, MaskROI):
            coverage += region_spec.selection
        else:
            x0, y0, x1, y1 = region_spec
            coverage[y0:y1, x0:x1] += 1
    return bool((coverage > 1).any())


def _require_positive_finite_coefficient(coeff: sc.Variable, data: sc.DataArray, name: str, region_arg: str) -> None:
    """Raise unless every non-missing pooled-coefficient bin is strictly positive and finite.

    Bins flagged by a coeff-aligned mask on ``data`` (a masked spectral bin may hold ``NaN`` by
    construction) are legitimately non-finite and are excluded from the guard, so masked data can
    still be air/background corrected. ``sc.sum(...).data`` upstream dropped the mask, so it is read
    back from ``data.masks`` here.
    """
    values = np.atleast_1d(coeff.values).astype(float)
    gap = np.zeros(values.shape, dtype=bool)
    for mask in data.masks.values():
        # A missing-bin mask (e.g. a 1-D per-bin tof mask) may be lower-dimensional than the
        # coefficient; broadcast any mask whose dims are a subset of coeff's and drop those bins.
        if set(mask.dims) <= set(coeff.dims):
            gap |= np.atleast_1d(sc.broadcast(mask, sizes=coeff.sizes).values)
    checked = values[~gap]
    if checked.size == 0 or not np.all(np.isfinite(checked)) or float(np.min(checked)) <= 0:
        worst = float(np.min(checked)) if checked.size else float("nan")
        raise ValueError(
            f"{region_arg} {name} pooled mean must be strictly positive and finite "
            f"(min={worst}); the ROI(s) must contain positive counts in every image"
        )


def _pooled_roi_coefficient(
    data: sc.DataArray,
    rois_bounds: list,
    name: str,
    strict: bool = True,
    region_arg: str = "background_roi",
) -> sc.Variable:
    """Per-image **pooled** background coefficient over one or more regions (rectangles or masks).

    ``coefficient = sum(counts over all regions) / sum(unmasked pixels over all regions)`` per
    spectral bin — the pooled ratio-of-means (1.x / iBeatles form). For a single region this is the
    plain mask-aware region mean. Reductions are mask-aware: masked dead/hot pixels are excluded
    from both the summed counts and the pixel count (spatial ``(x, y)`` masks assumed). Rectangle
    entries are exclusive-stop ``(x0, y0, x1, y1)`` bounds and keep the sliced fast path
    bit-for-bit; ``MaskROI`` entries contribute their selected pixels through the same mask-aware
    reductions. Regions that **overlap** in a pooled list are reduced over their **union** (each
    selected, unmasked pixel counted once) so the pooled mean and its variance are correct; a
    non-overlapping (or single-region) list keeps the per-region accumulation bit-for-bit. Returns a
    variance-bearing scipp Variable (the variance of the pooled mean).

    Raises ``ValueError`` on an invalid/out-of-bounds rectangle, a selection shape not matching the
    data, or missing ``x``/``y`` dims. With ``strict`` (default) it also rejects a
    non-positive/non-finite pooled mean (which would silently yield inf/nan output);
    ``strict=False`` skips only that guard and lets zeros propagate through the division — the 1.x
    semantics, for downstreams reproducing legacy outputs bit for bit. (A selection whose pixels
    are all dead/hot-masked behaves like an all-masked rectangle: strict raises, non-strict
    propagates inf/nan.)
    """
    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError(f"{name} must have 'x' and 'y' dimensions for {region_arg} normalization")
    total = None
    n_unmasked = None
    for region_spec in rois_bounds:
        if isinstance(region_spec, MaskROI):
            region = _mask_region_view(data, region_spec, name, region_arg=region_arg)
            roi_sum = sc.sum(region, dim=["x", "y"]).data  # selected & unmasked (mask-aware)
            roi_n = _masked_ones_count(region)
            total = roi_sum if total is None else total + roi_sum
            n_unmasked = roi_n if n_unmasked is None else n_unmasked + roi_n
            continue
        x0, y0, x1, y1 = region_spec
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid {region_arg} ({x0}, {y0}, {x1}, {y1}): need 0 <= x0 < x1 and 0 <= y0 < y1")
        if x1 > data.sizes["x"] or y1 > data.sizes["y"]:
            raise ValueError(
                f"{region_arg} ({x0}, {y0}, {x1}, {y1}) exceeds {name} size (x={data.sizes['x']}, y={data.sizes['y']})"
            )
        region = data["x", x0:x1]["y", y0:y1]
        roi_sum = sc.sum(region, dim=["x", "y"]).data  # per-spectral (mask-aware; var = sum of unmasked var)
        total = roi_sum if total is None else total + roi_sum
        # Unmasked pixel count, reduced over x,y ONLY (mask-aware) so a per-image (spectral, x, y)
        # mask stays per-spectral — mirroring sc.mean. Sum a masked field of ones: the same masks
        # exclude the same pixels from the count as from the counts sum above. (Collapsing all mask
        # dims would under-count the denominator for a 3D mask and inflate the coefficient.)
        roi_n = _unmasked_count(region)
        n_unmasked = roi_n if n_unmasked is None else n_unmasked + roi_n

    # Overlapping pooled regions double-count shared pixels in the accumulation above — inflating
    # the pooled mean and, because each shared pixel's variance is then added more than once as if
    # from independent samples, understating its variance. When the regions actually overlap,
    # recompute over their UNION (each selected & unmasked pixel exactly once); a non-overlapping
    # list keeps the per-region accumulation above bit-for-bit. Bounds/shapes are already validated.
    if len(rois_bounds) > 1:
        ny, nx = data.sizes["y"], data.sizes["x"]
        if _pooled_regions_overlap(rois_bounds, ny, nx):
            union_region = MaskROI(selection=_pooled_union_selection(rois_bounds, ny, nx))
            region = _mask_region_view(data, union_region, name, region_arg=region_arg)
            total = sc.sum(region, dim=["x", "y"]).data
            n_unmasked = _masked_ones_count(region)

    coeff = total / n_unmasked
    if strict:
        _require_positive_finite_coefficient(coeff, data, name, region_arg)
    return coeff


def _background_roi_means(
    sample: sc.DataArray,
    ob: sc.DataArray,
    rois_bounds: list[Union[tuple[int, int, int, int], MaskROI]],
    strict: bool = True,
) -> tuple[sc.Variable, sc.Variable]:
    """Per-image pooled background means (cs, co) for sample and OB over the same ROI list."""
    cs = _pooled_roi_coefficient(sample, rois_bounds, "sample", strict=strict)
    co = _pooled_roi_coefficient(ob, rois_bounds, "ob", strict=strict)
    return cs, co


def _roi_dark_mean_covariance(
    sample_dc: sc.DataArray,
    ob_dc: sc.DataArray,
    dark: sc.DataArray,
    rois_bounds: list[Union[tuple[int, int, int, int], MaskROI]],
) -> sc.Variable:
    """Covariance of the two **pooled** background-ROI means induced by the shared dark frame.

    ``cs = mean(S - D)`` and ``co = mean(O - D)`` pooled over the ROI list share the ROI dark
    pixels, so ``Cov(cs, co) = (1 / (n_s * n_o)) * sum_{k in A∩B} Var(D_k)`` where A / B are the
    pooled ROI pixels left unmasked in ``sample_dc`` / ``ob_dc`` (total counts ``n_s`` / ``n_o``) and
    A∩B is their intersection. With no masks this reduces to ``Var(pooled mean(D_roi))``. Spatial
    ``(x, y)`` masks (dead/hot pixels, as the CCD pipelines produce) give a scalar; a per-image
    ``(spectral, x, y)`` mask gives a per-spectral covariance (A∩B differs per image).

    Returns an ``sc.Variable`` in units of ``dark**2`` (counts**2): scalar for spatial masks,
    per-spectral for per-image masks.
    """

    def _excluded(da: sc.DataArray):  # OR of all masks over the ROI, or None when unmasked
        m = None
        for mask in da.masks.values():
            m = mask if m is None else (m | mask)
        return m

    # Collapse an overlapping pooled list to its single union region so a dark pixel shared by
    # several ROIs is counted once — otherwise the shared-dark covariance (and the n_s / n_o
    # denominators) double-count it. A non-overlapping list is unchanged. Bounds are already
    # validated by the _background_roi_means call that precedes this one.
    if len(rois_bounds) > 1:
        ny, nx = sample_dc.sizes["y"], sample_dc.sizes["x"]
        if _pooled_regions_overlap(rois_bounds, ny, nx):
            rois_bounds = [MaskROI(selection=_pooled_union_selection(rois_bounds, ny, nx))]

    n_s = None
    n_o = None
    intersection_var_sum = None  # sum over all ROIs of sum_{A∩B} Var(D)
    for region_spec in rois_bounds:
        if isinstance(region_spec, MaskROI):
            # The bbox views carry ~selection as one more mask, so _excluded() and the counts below
            # automatically restrict A/B to selected & unmasked pixels — the same generalization as
            # the pooled coefficient (A/B = selection ∩ unmasked).
            x0, y0, x1, y1 = region_spec.bounding_box()
            d_roi = dark["x", x0:x1]["y", y0:y1].copy()
            s_roi = _mask_region_view(sample_dc, region_spec, "sample")
            o_roi = _mask_region_view(ob_dc, region_spec, "ob")
            ms, mo = _excluded(s_roi), _excluded(o_roi)
            n_s_roi, n_o_roi = _masked_ones_count(s_roi), _masked_ones_count(o_roi)
        else:
            x0, y0, x1, y1 = region_spec
            d_roi = dark["x", x0:x1]["y", y0:y1].copy()
            s_roi = sample_dc["x", x0:x1]["y", y0:y1]
            o_roi = ob_dc["x", x0:x1]["y", y0:y1]
            ms, mo = _excluded(s_roi), _excluded(o_roi)
            # per-spectral unmasked counts (reduce masks over x,y only, mirroring
            # _pooled_roi_coefficient so a per-image (spectral, x, y) mask does not collapse the
            # denominator).
            n_s_roi, n_o_roi = _unmasked_count(s_roi), _unmasked_count(o_roi)
        n_s = n_s_roi if n_s is None else n_s + n_s_roi
        n_o = n_o_roi if n_o is None else n_o + n_o_roi
        # sum Var(D) over A∩B (ROI pixels kept in BOTH sample and OB).
        excl = ms if mo is None else (mo if ms is None else (ms | mo))
        if excl is not None and (set(excl.dims) - set(d_roi.dims)):
            # Mask carries a dim the 2D dark ROI lacks (per-image (spectral, x, y), a purely
            # spectral per-frame mask, etc.): A∩B differs along that dim, so the covariance is
            # per-spectral. Broadcast the 2D dark variance and sum the kept Var(D) over the shared
            # x, y — never attach a mask with an extra dim to the 2D dark (raises DimensionError).
            var_d = sc.variances(d_roi.data)  # (x, y), units dark**2
            keep = sc.where(excl, sc.scalar(0.0), sc.scalar(1.0))  # 1 where kept, broadcasts over x,y
            roi_var_sum = sc.sum(var_d * keep, dim=["x", "y"])  # per extra-dim bin, units dark**2
        else:
            # Spatial (2D) or no mask: mask the 2D dark ROI directly; sc.sum is mask-aware and
            # propagates variance as the sum of the unmasked variances.
            if excl is not None:
                d_roi.masks["_bg_excl"] = excl
            roi_var_sum = sc.variances(sc.sum(d_roi, dim=["x", "y"]).data)  # counts**2 (scalar)
        intersection_var_sum = roi_var_sum if intersection_var_sum is None else intersection_var_sum + roi_var_sum

    # n_s, n_o > 0 is guaranteed upstream under strict (_pooled_roi_coefficient raises on an
    # all-masked ROI); with strict=False an all-masked ROI gives n=0 and the resulting
    # non-finite covariance is zeroed by the isfinite guard on the over-count below.
    return intersection_var_sum / (n_s * n_o)


def normalize_transmission(  # noqa: C901
    sample: sc.DataArray,
    ob: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
    background_roi: Optional[BackgroundROILike] = None,
    background_roi_strict: bool = True,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_NORMALIZE,
) -> sc.DataArray:
    """
    Normalize sample by open beam to compute transmission.

    Formula: T = (Sample / pc_sample) / (OB / pc_ob)

    Handles:
    - Variance propagation (automatic via scipp)
    - Proton charge corrections (for SNS pulsed beam)
    - Systematic uncertainties
    - Mask preservation

    Parameters
    ----------
    sample : sc.DataArray
        Sample histogram with variance
    ob : sc.DataArray
        Open beam histogram with variance
    proton_charge_sample : float or sc.Variable, optional
        Integrated proton charge during sample acquisition (Coulombs)
        If provided, normalizes by beam intensity
    proton_charge_ob : float or sc.Variable, optional
        Integrated proton charge during OB acquisition (Coulombs)
    pc_uncertainty : float, optional
        Relative proton charge uncertainty (default: 0.005 = 0.5%)
        From PLEIADES measurements
    background_roi : ROI/MaskROI/tuple or a sequence of them, optional
        Sample-free background region(s) — an :class:`~neunorm.data_models.roi.ROI` (or a bare
        ``(x0, y0, x1, y1)`` tuple, exclusive stops), an arbitrary-shape
        :class:`~neunorm.data_models.roi.MaskROI` (selection mask: 1 = pixel in the region), **or a
        sequence mixing those, which are pooled** (``sum(counts over all regions) / sum(pixels)``).
        When given, each image is normalized by its pooled background mean — a proton-charge proxy
        for when proton charge is unavailable (e.g. MARS): ``T = (S/mean(S[B])) / (O/mean(O[B]))``.
        For legacy inclusive extents (a width-``w`` ROI spanning ``w+1`` pixels), use
        ``ROI(..., inclusive=True)``; see ``apply_background_roi`` for the open-beam-less form.
        Mutually exclusive with ``proton_charge_sample`` / ``proton_charge_ob``. Uncertainty is
        propagated first-order (the in-ROI sample/ROI-mean correlation is not corrected). Unless
        ``background_roi_strict=False``, raises ``ValueError`` if the pooled mean is not strictly
        positive and finite in every image. Indices (and mask shapes) are resolved against the
        passed arrays; if a pipeline crops with ``roi`` first, give ``background_roi`` in the
        post-crop frame.
    background_roi_strict : bool, optional
        With the default ``True``, a non-positive/non-finite pooled background mean raises
        ``ValueError``. ``False`` skips only that guard and lets zeros propagate through the
        division (inf/nan output) — the legacy 1.x semantics, for downstreams reproducing 1.x
        outputs bit for bit. Structural errors (bad ROI bounds, missing dims) always raise.
    progress : bool or callable, optional
        Progress reporting, off by default. This function has no item axis, so it reports named
        whole-array steps — the flux correction (background-ROI or proton-charge) and the division —
        with the total computed from the correction actually requested. Work conditional on a value
        only known mid-run is announced without advancing the count. Accepts an existing
        ``ProgressReporter``, which is how ``normalize_with_dark`` makes its steps continue one count
        rather than opening a second bar. See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_NORMALIZE``.

    Returns
    -------
    sc.DataArray
        Transmission with dimensions matching input
        Unit: dimensionless
        Includes propagated variance and systematic uncertainties

    Examples
    --------
    >>> # Basic normalization
    >>> transmission = normalize_transmission(hist_sample, hist_ob)

    >>> # With proton charge correction (SNS)
    >>> transmission = normalize_transmission(
    ...     hist_sample, hist_ob,
    ...     proton_charge_sample=500.0,
    ...     proton_charge_ob=505.0
    ... )

    Notes
    -----
    - Scipp automatically propagates variance through division
    - Masks are preserved from both sample and OB (OR combination)
    - Zero OB counts produce inf/nan (handle with masks before calling)
    """
    logger.info("Normalizing transmission: T = Sample / OB")

    roi_list = as_region_list(background_roi, arg_name="background_roi") if background_roi is not None else None

    # No item axis: a sequence of separable whole-array operations, each allocating arrays the size
    # of the stack. The count is computed from the arguments because the work varies — the
    # background-ROI and proton-charge corrections are mutually exclusive, and either may be absent —
    # so a literal total would leave the bar short or overshooting.
    with resolve_progress(progress, stage, total=normalize_step_count(background_roi, proton_charge_sample)) as report:
        # Background-ROI flux normalization: when no proton charge is available
        # (e.g. MARS), scale each image by its pooled mean counts in one or more sample-free ROIs so
        # per-image beam-flux differences cancel: T = (S/mean(S[B])) / (O/mean(O[B])). First-order UQ.
        if background_roi is not None:
            if proton_charge_sample is not None or proton_charge_ob is not None:
                raise ValueError(
                    "background_roi and proton_charge_sample/proton_charge_ob are mutually exclusive: "
                    "background_roi is the flux-normalization proxy for when proton charge is unavailable."
                )
            logger.info("Applying background-ROI flux normalization with ROI(s) {}", roi_list)
            report.note("background-ROI flux normalization")
            cs, co = _background_roi_means(sample, ob, roi_list, strict=background_roi_strict)
            # scipp refuses to broadcast a variance-bearing scalar across the image (it would introduce
            # correlations), so divide by the variance-free means and re-add their variance contribution
            # below. Handle cs and co INDEPENDENTLY — the two inputs may carry variance on one side only
            # (a variance-bearing co would otherwise make `ob / co` raise).
            cs_var = sc.variances(cs) if cs.variances is not None else None
            co_var = sc.variances(co) if co.variances is not None else None
            cs.variances = None
            co.variances = None
            sample_corrected = sample / cs
            ob_corrected = ob / co
            report()
        else:
            # Proton-charge correction must be applied to both sample and OB, or to neither: a one-sided
            # correction leaves counts/charge uncancelled, so the transmission would not be dimensionless.
            if (proton_charge_sample is None) != (proton_charge_ob is None):
                raise ValueError(
                    "proton_charge_sample and proton_charge_ob must both be provided or both omitted; "
                    "a one-sided proton-charge correction yields a non-dimensionless transmission."
                )

            # Apply proton charge corrections if provided
            if proton_charge_sample is not None:
                report.note("proton-charge correction, sample")
                if isinstance(proton_charge_sample, sc.Variable):
                    logger.info(
                        f"Applying proton charge correction: Sample mean pc={proton_charge_sample.mean().value} "
                        f"{proton_charge_sample.unit}"
                    )
                    sample_corrected = sample / proton_charge_sample
                else:
                    logger.info(f"  Applying proton charge correction: Sample pc={proton_charge_sample} C")
                    sample_corrected = sample / sc.scalar(proton_charge_sample, unit="C")

                # Add proton charge systematic uncertainty
                if sample_corrected.variances is not None:
                    pc_contribution = (pc_uncertainty * sample_corrected.values) ** 2
                    sample_corrected.variances = sample_corrected.variances + pc_contribution
                report()
            else:
                sample_corrected = sample

            if proton_charge_ob is not None:
                report.note("proton-charge correction, open beam")
                if isinstance(proton_charge_ob, sc.Variable):
                    logger.info(
                        f"Applying proton charge correction: OB mean pc={proton_charge_ob.mean().value} "
                        f"{proton_charge_ob.unit}"
                    )
                    ob_corrected = ob / proton_charge_ob
                else:
                    logger.info(f"  Applying proton charge correction: OB pc={proton_charge_ob:.1f} C")
                    ob_corrected = ob / sc.scalar(proton_charge_ob, unit="C")

                # Add proton charge systematic uncertainty
                if ob_corrected.variances is not None:
                    pc_contribution = (pc_uncertainty * ob_corrected.values) ** 2
                    ob_corrected.variances = ob_corrected.variances + pc_contribution
                report()
            else:
                ob_corrected = ob

        report.note("dividing sample by open beam")
        # Normalize
        if sample_corrected.dims == ob_corrected.dims:
            transmission = sample_corrected / ob_corrected
        else:
            # Need to broadcast to match dimensions
            ob_corrected_broadcast = ob_corrected.copy()
            ob_var = ob_corrected_broadcast.variances.copy() if ob_corrected_broadcast.variances is not None else None
            ob_corrected_broadcast.variances = None
            transmission = sample_corrected / ob_corrected_broadcast
            # Recombine variances across the broadcast (scipp cannot propagate a variance-bearing
            # denominator here, so the OB term is added manually). Handle EITHER side carrying variance:
            # a no-variance sample must not drop the OB contribution. When only the sample carries
            # variance, the division above already propagated it (the OB term is zero).
            if ob_var is not None:
                # Var(T) = (Var(Sample) / OB^2) + (Sample^2 * Var(OB) / OB^4)
                ob_term = sample_corrected.values**2 * ob_var / ob_corrected_broadcast.values**4
                if sample_corrected.variances is not None:
                    transmission.variances = sample_corrected.variances / ob_corrected_broadcast.values**2 + ob_term
                else:
                    transmission.variances = ob_term

        report()

        # copy dropped unaligned coordinates from input
        for coord in sample.coords:
            if not sample.coords[coord].aligned:
                transmission.coords[coord] = sample.coords[coord]

        # First-order contribution of the background-ROI mean uncertainty, added here because scipp
        # could not propagate it through the shared-scalar division above. Treats sample/ob/cs/co as
        # independent: Var(T) += T^2 * (Var(cs)/cs^2 + Var(co)/co^2). Accumulate whichever side carries
        # variance (inputs may be variance-bearing on one side only).
        #
        # Announced, not counted: whether it runs depends on the variances present on the result, which
        # is not known when the total is computed from the arguments.
        if (
            background_roi is not None
            and transmission.variances is not None
            and (cs_var is not None or co_var is not None)
        ):
            report.note("background-ROI variance contribution")
            coeff_rel_var = None
            if cs_var is not None:
                coeff_rel_var = cs_var / (cs * cs)
            if co_var is not None:
                co_term = co_var / (co * co)
                coeff_rel_var = co_term if coeff_rel_var is None else coeff_rel_var + co_term
            extra = sc.array(dims=list(transmission.dims), values=transmission.values**2) * coeff_rel_var
            # Keep the variance dtype stable (float32 pipelines), matching normalize_with_dark.
            transmission.variances = transmission.variances + extra.values.astype(
                transmission.variances.dtype, copy=False
            )

        logger.success("✓ Transmission normalized")

        return transmission


def normalize_with_dark(
    sample: sc.DataArray,
    ob: sc.DataArray,
    dark: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
    background_roi: Optional[BackgroundROILike] = None,
    background_roi_strict: bool = True,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_NORMALIZE,
) -> sc.DataArray:
    """Dark-correct and normalize in one step, treating the shared dark frame correctly.

    Computes ``T = (sample - dark) / (ob - dark)`` (with the optional proton-charge
    correction) where the **same** averaged ``dark`` is subtracted from both sample and open
    beam. ``subtract_dark`` + ``normalize_transmission`` would treat the numerator and
    denominator as statistically independent and propagate ``Var(dark)`` twice; this function
    removes that spurious shared-dark covariance term.

    The transmission **values** are identical to
    ``normalize_transmission(subtract_dark(sample, dark), subtract_dark(ob, dark), ...)`` — only
    the propagated variance is corrected (reduced) by ``2 * k**2 * (sample-dark) * Var(dark) /
    (ob-dark)**3``, with ``k = pc_ob / pc_sample`` (1 when no proton charge). The proton-charge
    systematic and the sample/open-beam Poisson terms are unchanged.

    Parameters
    ----------
    sample, ob, dark : sc.DataArray
        Sample, open-beam and (averaged) dark-current frames, each carrying Poisson variance.
        The same ``dark`` is subtracted from both ``sample`` and ``ob``.
    proton_charge_sample, proton_charge_ob : float or sc.Variable, optional
        Integrated proton charge for the SNS beam correction (see ``normalize_transmission``).
    pc_uncertainty : float, optional
        Relative proton-charge uncertainty (default 0.005).
    background_roi : ROI/MaskROI/tuple or a sequence of them, optional
        Background-ROI flux normalization (see ``normalize_transmission``), used instead of proton
        charge. Accepts the same forms as ``normalize_transmission`` — a rectangle, an arbitrary-shape
        ``MaskROI``, or a pooled sequence mixing them. The shared-dark correction then uses
        ``k = co/cs`` (ratio of dark-corrected ROI means) in place of the proton-charge ratio, and
        additionally removes the ROI-mean shared-dark
        covariance term ``2*T^2*Cov(cs,co)/(cs*co)`` (``Cov(cs,co) = Var(mean(D_roi))``) — the
        ROI-mean analog of the pixel-level correction. (The in-ROI pixel/ROI-mean correlation
        remains uncorrected, as documented on ``normalize_transmission``.)
    background_roi_strict : bool, optional
        See ``normalize_transmission``: ``False`` skips the strictly-positive/finite pooled-mean
        guard and lets zeros propagate (legacy 1.x semantics).
    progress : bool or callable, optional
        Progress reporting, off by default. Reports the two dark subtractions, then hands the reporter
        to ``normalize_transmission`` so its steps continue the same count instead of restarting. The
        declared total therefore covers both. See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_NORMALIZE``.

    Returns
    -------
    sc.DataArray
        Transmission with correctly-propagated variance (no shared-dark double-counting).
    """
    roi_list = as_region_list(background_roi, arg_name="background_roi") if background_roi is not None else None

    # Two dark subtractions of our own, then whatever normalize_transmission will report. It receives
    # this reporter and borrows it, and a borrowed reporter keeps the OUTER total, so the combined
    # count has to be declared here or the bar would stop short.
    n_steps = normalize_with_dark_step_count(background_roi, proton_charge_sample)
    with resolve_progress(progress, stage, total=n_steps) as report:
        report.note("dark-correcting sample")
        sample_dc = subtract_dark(sample, dark)
        report()
        report.note("dark-correcting open beam")
        ob_dc = subtract_dark(ob, dark)
        report()
        transmission = normalize_transmission(
            sample_dc,
            ob_dc,
            proton_charge_sample,
            proton_charge_ob,
            pc_uncertainty,
            background_roi=background_roi,
            background_roi_strict=background_roi_strict,
            progress=report,
            stage=stage,
        )

        # Correct the shared-dark double-count. normalize_transmission propagated
        # Var(dark) through BOTH numerator and denominator as if they were independent; the true
        # propagation (dark appears once) is smaller by 2*k^2*s*Var(D)/o^3. Subtract that term.
        if transmission.variances is None or dark.variances is None:
            return transmission

        # The correction below is the majority of this function's cost — 58% of the wall clock at
        # 80 x 512² — and it used to run after the progress context had closed, so a caller watched the
        # bar reach its total and the bars vanish, then waited out more than half the call with nothing
        # on screen. Announced rather than counted, and announced only after the guard above, so the
        # label never names work that is being skipped: whether it runs is not knowable until the
        # delegate has returned.
        report.note("correcting shared-dark variance")

        # Use scipp's unit-carrying value/variance accessors so (a) the (3D sample) vs (2D ob/dark)
        # broadcast and the per-image proton-charge ratio align by dimension name and (b) scipp
        # validates units: counts * counts**2 / counts**3 = dimensionless, matching Var(T).
        s_v = sc.values(sample_dc)  # counts
        o_v = sc.values(ob_dc)  # counts
        var_d_v = sc.variances(dark)  # counts**2
        over_count = 2.0 * s_v * var_d_v / (o_v**3)

        # The over-count scales with the squared flux coefficient k applied to S/O: k = pc_ob/pc_sample
        # for proton charge, or k = co/cs (ratio of dark-corrected ROI means) for background_roi. Use
        # the coefficient values only (variance-free) — this is a variance correction, first-order in k.
        if background_roi is not None:
            cs, co = _background_roi_means(sample_dc, ob_dc, roi_list, strict=background_roi_strict)
            cs_v, co_v = sc.values(cs), sc.values(co)
            k_squared = (co_v / cs_v) ** 2
        else:
            k_squared = _proton_charge_ratio_squared(proton_charge_sample, proton_charge_ob)
        if k_squared is not None:
            over_count = k_squared * over_count

        # background_roi shares the dark across BOTH ROI means: cs = mean(S-D) and co = mean(O-D) use the
        # same ROI dark pixels, so Cov(cs, co) = Var(mean(D_roi)) > 0. normalize_transmission added the
        # ROI-mean term T^2 * (Var(cs)/cs^2 + Var(co)/co^2) treating cs and co as independent; subtract
        # the missing covariance term 2 * T^2 * Cov(cs,co) / (cs*co) too — the ROI-mean analog of the
        # pixel-level correction. (The in-ROI pixel<->mean correlation stays uncorrected, as
        # documented; for a clean background ROI the dark-mean covariance is the only remaining term.)
        if background_roi is not None:
            # Cov(cs,co) is mask-consistent with cs/co (it counts only the ROI dark pixels left unmasked
            # in BOTH sample and OB), so a dead/hot pixel masked from one side does not pollute it.
            cov_cs_co = _roi_dark_mean_covariance(sample_dc, ob_dc, dark, roi_list)
            t_v = sc.values(transmission)
            over_count = over_count + 2.0 * t_v * t_v * cov_cs_co / (cs_v * co_v)

        over_values = sc.to_unit(over_count, "dimensionless").transpose(transmission.dims).values
        # Match the variance dtype so the correction never promotes a float32 pipeline to float64.
        over_values = over_values.astype(transmission.variances.dtype, copy=False)
        # Zero the correction where ob-dark == 0 (those pixels are already inf/nan in T).
        over_values = np.where(np.isfinite(over_values), over_values, 0.0)
        # Clamp to >= 0 defensively; the corrected variance is a true (non-negative) variance.
        transmission.variances = np.clip(transmission.variances - over_values, 0.0, None)
        logger.success("✓ Shared-dark variance double-count corrected")

        return transmission


def _proton_charge_ratio_squared(
    proton_charge_sample: Optional[Union[float, sc.Variable]],
    proton_charge_ob: Optional[Union[float, sc.Variable]],
) -> Optional[sc.Variable]:
    """Return ``(pc_ob / pc_sample)**2`` as a dimensionless scipp Variable, or None if either
    proton charge is absent (k = 1, the MARS / no-beam-correction case)."""
    if proton_charge_sample is None or proton_charge_ob is None:
        return None
    pc_s = (
        proton_charge_sample
        if isinstance(proton_charge_sample, sc.Variable)
        else sc.scalar(float(proton_charge_sample), unit="C")
    )
    pc_o = (
        proton_charge_ob if isinstance(proton_charge_ob, sc.Variable) else sc.scalar(float(proton_charge_ob), unit="C")
    )
    k = sc.to_unit(pc_o / pc_s, "dimensionless")
    return k * k


def apply_background_roi(
    data: sc.DataArray,
    background_roi: BackgroundROILike,
    strict: bool = True,
) -> sc.DataArray:
    """Flux-flatten a stack by its pooled background-ROI mean (no open beam).

    Returns ``data / pooled_mean(data over background_roi)`` — the **sample-only** form of the
    background-ROI flux proxy, for when there is no open beam to normalize against. ``background_roi``
    is a single region or a sequence of them — rectangles and/or arbitrary-shape ``MaskROI``
    selections — pooled as ``sum(counts) / sum(pixels)`` (overlapping regions are de-duplicated to
    their union); see ``normalize_transmission(..., background_roi=)`` for the with-open-beam form.

    First-order uncertainty from the pooled ROI mean is propagated
    (``Var += corrected**2 * Var(coeff) / coeff**2``); the in-ROI pixel/ROI-mean correlation is not
    corrected. Reductions are mask-aware. Raises ``ValueError`` if the pooled mean is not strictly
    positive and finite in every image (unless ``strict=False``).

    Parameters
    ----------
    data : sc.DataArray
        Image stack with ``x``/``y`` dims (e.g. ``(spectral, x, y)``), optionally carrying variance.
    background_roi : ROI, MaskROI, tuple, or a sequence of them
        Sample-free background region(s), pooled — rectangles and/or arbitrary-shape ``MaskROI``
        selections, mixable in one list.
    strict : bool, optional
        With the default ``True``, a non-positive/non-finite pooled mean raises ``ValueError``.
        ``False`` skips only that guard and lets zeros propagate through the division (inf/nan
        output) — the legacy 1.x semantics, for downstreams reproducing 1.x outputs bit for bit.
        Structural errors (bad ROI bounds, missing dims) always raise.

    Returns
    -------
    sc.DataArray
        ``data`` scaled so its pooled background-ROI mean is 1 per image.
    """
    roi_list = as_region_list(background_roi, arg_name="background_roi")
    logger.info("Applying sample-only background-ROI flux flattening with ROI(s) {}", roi_list)
    coeff = _pooled_roi_coefficient(data, roi_list, "data", strict=strict)
    coeff_var = sc.variances(coeff) if coeff.variances is not None else None
    coeff = coeff.copy()
    coeff.variances = None
    corrected = data / coeff
    if coeff_var is not None and corrected.variances is not None:
        rel = coeff_var / (coeff * coeff)
        extra = sc.array(dims=list(corrected.dims), values=corrected.values**2) * rel
        corrected.variances = corrected.variances + extra.values.astype(corrected.variances.dtype, copy=False)
    return corrected
