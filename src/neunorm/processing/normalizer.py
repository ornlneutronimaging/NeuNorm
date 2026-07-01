"""
Transmission normalization for neutron imaging.

Implements the core neutron imaging equation: T = Sample / OpenBeam
with proper uncertainty propagation and beam corrections.
"""

from typing import Optional, Sequence, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.roi import ROI, ROILike, as_roi_bounds
from neunorm.processing.dark_corrector import subtract_dark

# One ROI or a sequence of ROIs (pooled). A bare 4-int tuple/list is a single ROI.
BackgroundROILike = Union[ROILike, Sequence[ROILike]]


def _as_roi_bounds_list(background_roi: BackgroundROILike) -> list[tuple[int, int, int, int]]:
    """Normalize a ``background_roi`` argument to a list of exclusive ``(x0, y0, x1, y1)`` bounds.

    Accepts a single ROI (an :class:`~neunorm.data_models.roi.ROI` or a bare 4-int ``(x0,y0,x1,y1)``
    tuple/list) — backward compatible — or a **sequence** of those (pooled). A bare 4-int sequence is
    treated as ONE ROI; a sequence whose elements are ROIs or sequences is a list of ROIs.
    """
    if isinstance(background_roi, ROI):
        return [background_roi.as_bounds()]
    if (
        isinstance(background_roi, (tuple, list))
        and len(background_roi) == 4
        and all(isinstance(i, int) for i in background_roi)
    ):
        return [as_roi_bounds(background_roi)]
    if isinstance(background_roi, (tuple, list)):
        if len(background_roi) == 0:
            raise ValueError("background_roi list must contain at least one ROI")
        # a bare sequence of ints is a SINGLE ROI (handled above when len == 4); an int element here
        # means a malformed single ROI (wrong length), not a sequence of ROIs.
        if any(isinstance(e, (int, np.integer)) for e in background_roi):
            raise ValueError(f"background_roi must be a tuple of 4 integers (x0, y0, x1, y1); got {background_roi!r}")
        return [as_roi_bounds(r) for r in background_roi]
    raise ValueError(
        f"background_roi must be an ROI, an (x0, y0, x1, y1) tuple, or a sequence of those; got {background_roi!r}"
    )


def _pooled_roi_coefficient(
    data: sc.DataArray,
    rois_bounds: list[tuple[int, int, int, int]],
    name: str,
) -> sc.Variable:
    """Per-image **pooled** background coefficient over one or more ROIs.

    ``coefficient = sum(counts over all ROIs) / sum(unmasked pixels over all ROIs)`` per spectral
    bin — the pooled ratio-of-means (1.x / iBeatles form). For a single ROI this is the plain
    mask-aware ROI mean. Reductions are mask-aware: masked dead/hot pixels are excluded from both
    the summed counts and the pixel count (spatial ``(x, y)`` masks assumed). ``x1``/``y1`` are
    exclusive stops. Returns a variance-bearing scipp Variable (the variance of the pooled mean).

    Raises ``ValueError`` on an invalid/out-of-bounds ROI, missing ``x``/``y`` dims, no unmasked
    pixels, or a non-positive/non-finite pooled mean (which would silently yield inf/nan output).
    """
    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError(f"{name} must have 'x' and 'y' dimensions for background_roi normalization")
    total = None
    n_pixels = 0
    for x0, y0, x1, y1 in rois_bounds:
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid background_roi ({x0}, {y0}, {x1}, {y1}): need 0 <= x0 < x1 and 0 <= y0 < y1")
        if x1 > data.sizes["x"] or y1 > data.sizes["y"]:
            raise ValueError(
                f"background_roi ({x0}, {y0}, {x1}, {y1}) exceeds {name} size "
                f"(x={data.sizes['x']}, y={data.sizes['y']})"
            )
        region = data["x", x0:x1]["y", y0:y1]
        roi_sum = sc.sum(region, dim=["x", "y"]).data  # mask-aware; variance = sum of unmasked variances
        total = roi_sum if total is None else total + roi_sum
        n_xy = region.sizes["x"] * region.sizes["y"]
        excl = None
        for mask in region.masks.values():
            excl = mask if excl is None else (excl | mask)
        n_pixels += n_xy - (int(sc.sum(excl).value) if excl is not None else 0)

    if n_pixels <= 0:
        raise ValueError(f"background_roi {rois_bounds} {name}: no unmasked pixels in the pooled ROI(s)")
    coeff = total / n_pixels
    if not bool(sc.all(sc.isfinite(coeff)).value) or sc.min(coeff).value <= 0:
        raise ValueError(
            f"background_roi {name} pooled mean must be strictly positive and finite "
            f"(min={sc.min(coeff).value}); the ROI(s) must contain positive counts in every image"
        )
    return coeff


def _background_roi_means(
    sample: sc.DataArray,
    ob: sc.DataArray,
    rois_bounds: list[tuple[int, int, int, int]],
) -> tuple[sc.Variable, sc.Variable]:
    """Per-image pooled background means (cs, co) for sample and OB over the same ROI list."""
    cs = _pooled_roi_coefficient(sample, rois_bounds, "sample")
    co = _pooled_roi_coefficient(ob, rois_bounds, "ob")
    return cs, co


def _roi_dark_mean_covariance(
    sample_dc: sc.DataArray,
    ob_dc: sc.DataArray,
    dark: sc.DataArray,
    rois_bounds: list[tuple[int, int, int, int]],
) -> sc.Variable:
    """Covariance of the two **pooled** background-ROI means induced by the shared dark frame.

    ``cs = mean(S - D)`` and ``co = mean(O - D)`` pooled over the ROI list share the ROI dark
    pixels, so ``Cov(cs, co) = (1 / (n_s * n_o)) * sum_{k in A∩B} Var(D_k)`` where A / B are the
    pooled ROI pixels left unmasked in ``sample_dc`` / ``ob_dc`` (total counts ``n_s`` / ``n_o``) and
    A∩B is their intersection. With no masks this reduces to ``Var(pooled mean(D_roi))``. Assumes
    spatial ``(x, y)`` masks (dead/hot pixels), which is what the CCD pipelines produce.

    Returns a scalar ``sc.Variable`` carrying units of ``dark**2`` (counts**2).
    """

    def _excluded(da: sc.DataArray):  # OR of all masks over the ROI, or None when unmasked
        m = None
        for mask in da.masks.values():
            m = mask if m is None else (m | mask)
        return m

    n_s = 0
    n_o = 0
    intersection_var_sum = None  # sum over all ROIs of sum_{A∩B} Var(D)
    for x0, y0, x1, y1 in rois_bounds:
        d_roi = dark["x", x0:x1]["y", y0:y1].copy()
        ms = _excluded(sample_dc["x", x0:x1]["y", y0:y1])
        mo = _excluded(ob_dc["x", x0:x1]["y", y0:y1])
        n_xy = d_roi.sizes["x"] * d_roi.sizes["y"]
        n_s += n_xy - (int(sc.sum(ms).value) if ms is not None else 0)
        n_o += n_xy - (int(sc.sum(mo).value) if mo is not None else 0)
        # sum Var(D) over A∩B: mask the dark with the union of both sides' masks; sc.sum is
        # mask-aware and propagates variance as the sum of unmasked variances.
        excl = ms if mo is None else (mo if ms is None else (ms | mo))
        if excl is not None:
            d_roi.masks["_bg_excl"] = excl
        roi_var_sum = sc.variances(sc.sum(d_roi, dim=["x", "y"]).data)  # scalar, counts**2
        intersection_var_sum = roi_var_sum if intersection_var_sum is None else intersection_var_sum + roi_var_sum

    denom = n_s * n_o
    return intersection_var_sum / denom if denom > 0 else 0.0 * intersection_var_sum


def normalize_transmission(  # noqa: C901
    sample: sc.DataArray,
    ob: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
    background_roi: Optional[BackgroundROILike] = None,
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
    background_roi : ROI or tuple[int, int, int, int], optional
        Sample-free background ROI — an :class:`~neunorm.data_models.roi.ROI` or a bare
        ``(x0, y0, x1, y1)`` tuple with exclusive stops. When given, each image is normalized by its
        mean counts in this ROI — a proton-charge proxy for when proton charge is unavailable (e.g.
        MARS): ``T = (S/mean(S[B])) / (O/mean(O[B]))``.
        Mutually exclusive with ``proton_charge_sample`` / ``proton_charge_ob``. Uncertainty is
        propagated first-order (the in-ROI sample/ROI-mean correlation is not corrected). Raises
        ``ValueError`` if the ROI mean is not strictly positive and finite in every image. Indices
        are resolved against the passed arrays; if a pipeline crops with ``roi`` first, give
        ``background_roi`` in the post-crop frame.

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

    roi_list = _as_roi_bounds_list(background_roi) if background_roi is not None else None

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
        cs, co = _background_roi_means(sample, ob, roi_list)
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
        else:
            sample_corrected = sample

        if proton_charge_ob is not None:
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
        else:
            ob_corrected = ob

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

    # copy dropped unaligned coordinates from input
    for coord in sample.coords:
        if not sample.coords[coord].aligned:
            transmission.coords[coord] = sample.coords[coord]

    # First-order contribution of the background-ROI mean uncertainty, added here because scipp
    # could not propagate it through the shared-scalar division above. Treats sample/ob/cs/co as
    # independent: Var(T) += T^2 * (Var(cs)/cs^2 + Var(co)/co^2). Accumulate whichever side carries
    # variance (inputs may be variance-bearing on one side only).
    if background_roi is not None and transmission.variances is not None and (cs_var is not None or co_var is not None):
        coeff_rel_var = None
        if cs_var is not None:
            coeff_rel_var = cs_var / (cs * cs)
        if co_var is not None:
            co_term = co_var / (co * co)
            coeff_rel_var = co_term if coeff_rel_var is None else coeff_rel_var + co_term
        extra = sc.array(dims=list(transmission.dims), values=transmission.values**2) * coeff_rel_var
        # Keep the variance dtype stable (float32 pipelines), matching normalize_with_dark.
        transmission.variances = transmission.variances + extra.values.astype(transmission.variances.dtype, copy=False)

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
    background_roi : tuple[int, int, int, int], optional
        Background-ROI flux normalization (see ``normalize_transmission``), used instead of proton
        charge. The shared-dark correction then uses ``k = co/cs`` (ratio of dark-corrected ROI
        means) in place of the proton-charge ratio, and additionally removes the ROI-mean shared-dark
        covariance term ``2*T^2*Cov(cs,co)/(cs*co)`` (``Cov(cs,co) = Var(mean(D_roi))``) — the
        ROI-mean analog of the pixel-level correction. (The in-ROI pixel/ROI-mean correlation
        remains uncorrected, as documented on ``normalize_transmission``.)

    Returns
    -------
    sc.DataArray
        Transmission with correctly-propagated variance (no shared-dark double-counting).
    """
    roi_list = _as_roi_bounds_list(background_roi) if background_roi is not None else None

    sample_dc = subtract_dark(sample, dark)
    ob_dc = subtract_dark(ob, dark)
    transmission = normalize_transmission(
        sample_dc, ob_dc, proton_charge_sample, proton_charge_ob, pc_uncertainty, background_roi=background_roi
    )

    # Correct the shared-dark double-count. normalize_transmission propagated
    # Var(dark) through BOTH numerator and denominator as if they were independent; the true
    # propagation (dark appears once) is smaller by 2*k^2*s*Var(D)/o^3. Subtract that term.
    if transmission.variances is None or dark.variances is None:
        return transmission

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
        cs, co = _background_roi_means(sample_dc, ob_dc, roi_list)
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
) -> sc.DataArray:
    """Flux-flatten a stack by its pooled background-ROI mean (no open beam).

    Returns ``data / pooled_mean(data over background_roi)`` — the **sample-only** form of the
    background-ROI flux proxy, for when there is no open beam to normalize against. ``background_roi``
    is a single ROI or a sequence of ROIs (pooled as ``sum(counts) / sum(pixels)``); see
    ``normalize_transmission(..., background_roi=)`` for the with-open-beam transmission form.

    First-order uncertainty from the pooled ROI mean is propagated
    (``Var += corrected**2 * Var(coeff) / coeff**2``); the in-ROI pixel/ROI-mean correlation is not
    corrected. Reductions are mask-aware. Raises ``ValueError`` if the pooled mean is not strictly
    positive and finite in every image.

    Parameters
    ----------
    data : sc.DataArray
        Image stack with ``x``/``y`` dims (e.g. ``(spectral, x, y)``), optionally carrying variance.
    background_roi : ROI/tuple or a sequence of them
        Sample-free background region(s), pooled.

    Returns
    -------
    sc.DataArray
        ``data`` scaled so its pooled background-ROI mean is 1 per image.
    """
    roi_list = _as_roi_bounds_list(background_roi)
    logger.info("Applying sample-only background-ROI flux flattening with ROI(s) {}", roi_list)
    coeff = _pooled_roi_coefficient(data, roi_list, "data")
    coeff_var = sc.variances(coeff) if coeff.variances is not None else None
    coeff = coeff.copy()
    coeff.variances = None
    corrected = data / coeff
    if coeff_var is not None and corrected.variances is not None:
        rel = coeff_var / (coeff * coeff)
        extra = sc.array(dims=list(corrected.dims), values=corrected.values**2) * rel
        corrected.variances = corrected.variances + extra.values.astype(corrected.variances.dtype, copy=False)
    return corrected
