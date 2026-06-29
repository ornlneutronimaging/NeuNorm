"""
Transmission normalization for neutron imaging.

Implements the core neutron imaging equation: T = Sample / OpenBeam
with proper uncertainty propagation and beam corrections.
"""

from typing import Optional, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.processing.dark_corrector import subtract_dark


def _background_roi_means(
    sample: sc.DataArray,
    ob: sc.DataArray,
    background_roi: tuple[int, int, int, int],
) -> tuple[sc.Variable, sc.Variable]:
    """Validate ``background_roi`` and return the per-image mean counts (cs, co) over it.

    ``background_roi`` is ``(x0, y0, x1, y1)`` with exclusive stop indices (Python slice
    semantics, matching ``apply_roi`` / ``apply_air_region_correction``). The means reduce the
    spatial ``x``/``y`` dimensions, so for 3D ``(spectral, x, y)`` data they are computed per
    spectral bin. The returned scipp scalars/arrays carry the variance of the mean.
    """
    if not (
        isinstance(background_roi, (tuple, list))
        and len(background_roi) == 4
        and all(isinstance(i, int) for i in background_roi)
    ):
        raise ValueError("background_roi must be a tuple of 4 integers (x0, y0, x1, y1)")
    x0, y0, x1, y1 = background_roi
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid background_roi {background_roi}: need 0 <= x0 < x1 and 0 <= y0 < y1")
    for name, da in (("sample", sample), ("ob", ob)):
        if "x" not in da.dims or "y" not in da.dims:
            raise ValueError(f"{name} must have 'x' and 'y' dimensions for background_roi normalization")
        if x1 > da.sizes["x"] or y1 > da.sizes["y"]:
            raise ValueError(
                f"background_roi {background_roi} exceeds {name} size (x={da.sizes['x']}, y={da.sizes['y']})"
            )
    # Reduce on the DataArray (mask-aware) so masked dead/hot pixels in the ROI are excluded,
    # then take .data to drop coords (avoids coord-mismatch in the subsequent division).
    cs = sc.mean(sample["x", x0:x1]["y", y0:y1], dim=["x", "y"]).data
    co = sc.mean(ob["x", x0:x1]["y", y0:y1], dim=["x", "y"]).data
    return cs, co


def normalize_transmission(  # noqa: C901
    sample: sc.DataArray,
    ob: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
    background_roi: Optional[tuple[int, int, int, int]] = None,
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
    background_roi : tuple[int, int, int, int], optional
        Sample-free background ROI ``(x0, y0, x1, y1)`` with exclusive stops. When given,
        each image is normalized by its mean counts in this ROI — a proton-charge proxy for
        when proton charge is unavailable (e.g. MARS): ``T = (S/mean(S[B])) / (O/mean(O[B]))``.
        Mutually exclusive with ``proton_charge_sample`` / ``proton_charge_ob``. Uncertainty is
        propagated first-order (the in-ROI sample/ROI-mean correlation is not corrected).

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

    # Background-ROI flux normalization (issue #159): when no proton charge is available
    # (e.g. MARS), scale each image by its mean counts in a sample-free ROI so per-image
    # beam-flux differences cancel: T = (S/mean(S[B])) / (O/mean(O[B])). First-order UQ.
    if background_roi is not None:
        if proton_charge_sample is not None or proton_charge_ob is not None:
            raise ValueError(
                "background_roi and proton_charge_sample/proton_charge_ob are mutually exclusive: "
                "background_roi is the flux-normalization proxy for when proton charge is unavailable."
            )
        logger.info("Applying background-ROI flux normalization with ROI {}", background_roi)
        cs, co = _background_roi_means(sample, ob, background_roi)
        # scipp refuses to broadcast a variance-bearing scalar across the image (it would introduce
        # correlations), so divide by the variance-free means and add their variance contribution below.
        # Only when the inputs carry variance: strip the ROI-mean variance (scipp cannot broadcast
        # a variance-bearing scalar) and re-add its contribution after the division (below).
        if cs.variances is not None:
            cs_var, co_var = sc.variances(cs), sc.variances(co)
            cs.variances = None
            co.variances = None
        else:
            cs_var = co_var = None
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
        if ob_var is not None and sample_corrected.variances is not None:
            # Var(T) = (T^2) * (Var(Sample)/Sample^2 + Var(OB)/OB^2)
            # which is equivalent to:
            # Var(T) = (Var(Sample) / OB^2) + (Sample^2 * Var(OB) / OB^4)
            transmission.variances = (sample_corrected.variances / ob_corrected_broadcast.values**2) + (
                sample_corrected.values**2 * ob_var / ob_corrected_broadcast.values**4
            )

    # copy dropped unaligned coordinates from input
    for coord in sample.coords:
        if not sample.coords[coord].aligned:
            transmission.coords[coord] = sample.coords[coord]

    # First-order contribution of the background-ROI mean uncertainty, added here because scipp
    # could not propagate it through the shared-scalar division above. Treats sample/ob/cs/co as
    # independent: Var(T) += T^2 * (Var(cs)/cs^2 + Var(co)/co^2).
    if background_roi is not None and transmission.variances is not None and cs_var is not None:
        coeff_rel_var = cs_var / (cs * cs) + co_var / (co * co)
        extra = sc.array(dims=list(transmission.dims), values=transmission.values**2) * coeff_rel_var
        transmission.variances = transmission.variances + extra.values

    logger.success("✓ Transmission normalized")

    return transmission


def normalize_with_dark(
    sample: sc.DataArray,
    ob: sc.DataArray,
    dark: sc.DataArray,
    proton_charge_sample: Optional[Union[float, sc.Variable]] = None,
    proton_charge_ob: Optional[Union[float, sc.Variable]] = None,
    pc_uncertainty: float = 0.005,
    background_roi: Optional[tuple[int, int, int, int]] = None,
) -> sc.DataArray:
    """Dark-correct and normalize in one step, treating the shared dark frame correctly.

    Computes ``T = (sample - dark) / (ob - dark)`` (with the optional proton-charge
    correction) where the **same** averaged ``dark`` is subtracted from both sample and open
    beam. ``subtract_dark`` + ``normalize_transmission`` would treat the numerator and
    denominator as statistically independent and propagate ``Var(dark)`` twice; this function
    removes that spurious shared-dark covariance term (issue #142).

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
        means) in place of the proton-charge ratio.

    Returns
    -------
    sc.DataArray
        Transmission with correctly-propagated variance (no shared-dark double-counting).
    """
    sample_dc = subtract_dark(sample, dark)
    ob_dc = subtract_dark(ob, dark)
    transmission = normalize_transmission(
        sample_dc, ob_dc, proton_charge_sample, proton_charge_ob, pc_uncertainty, background_roi=background_roi
    )

    # Correct the shared-dark double-count (issue #142). normalize_transmission propagated
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
        cs, co = _background_roi_means(sample_dc, ob_dc, background_roi)
        k_squared = (sc.values(co) / sc.values(cs)) ** 2
    else:
        k_squared = _proton_charge_ratio_squared(proton_charge_sample, proton_charge_ob)
    if k_squared is not None:
        over_count = k_squared * over_count

    over_values = sc.to_unit(over_count, "dimensionless").transpose(transmission.dims).values
    # Match the variance dtype so the correction never promotes a float32 pipeline to float64.
    over_values = over_values.astype(transmission.variances.dtype, copy=False)
    # Zero the correction where ob-dark == 0 (those pixels are already inf/nan in T).
    over_values = np.where(np.isfinite(over_values), over_values, 0.0)
    # Clamp to >= 0 defensively; the corrected variance is a true (non-negative) variance.
    transmission.variances = np.clip(transmission.variances - over_values, 0.0, None)
    logger.success("✓ Shared-dark variance double-count corrected (issue #142)")

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
