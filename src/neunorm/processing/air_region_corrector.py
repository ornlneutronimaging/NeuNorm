"""
Air region correction.
"""

import scipp as sc
from loguru import logger

from neunorm.data_models.roi import RegionsLike, as_region_list
from neunorm.processing.normalizer import _pooled_roi_coefficient


def apply_air_region_correction(
    transmission: sc.DataArray,
    air_roi: RegionsLike,
) -> sc.DataArray:
    """Scale transmission so the air region has mean = 1.0.

    Requirements
    ------------

    - Calculate mean transmission in the user-specified air region(s)
    - Scale entire image so the air-region mean = 1.0
    - Support per-image correction (radiography) and per-TOF-bin correction (hyperspectral)
    - Propagate uncertainty from the air-region mean

    Formula
    -------

    T_final = T / mean(T[air region])

    Uncertainty (first order):
    σ_T_final = T_final × √[(σ_T/T)² + (σ_air/<T_air>)²]

    The mean is the mask-aware **pooled** region mean (``sum(T) / count(unmasked pixels)``, per
    spectral bin), shared with the ``background_roi`` machinery. For a masked air region this is
    the true region mean — dead/hot pixels are excluded from numerator and denominator — and a
    region is valid even when a full row of it is masked. Raises ``ValueError`` if the air mean is
    not strictly positive and finite in every image (a non-positive mean transmission in the air
    region is pathological and would silently corrupt the scale).

    Parameters
    ----------
    transmission : sc.DataArray
        Normalized transmission (after OB correction)
    air_roi : ROI, MaskROI, tuple, or a sequence of them
        Air region as an :class:`~neunorm.data_models.roi.ROI` (or a bare ``(x0, y0, x1, y1)``
        tuple, exclusive stops), an arbitrary-shape :class:`~neunorm.data_models.roi.MaskROI`
        (selection mask: 1 = pixel in the region), or a sequence mixing those (pooled).

    """
    regions = as_region_list(air_roi, arg_name="air_roi")

    logger.info(f"Applying air region correction with region(s): {regions}")

    coeff = _pooled_roi_coefficient(transmission, regions, "transmission", strict=True, region_arg="air_roi")

    # scipp refuses to divide by a variance-bearing scalar (it would introduce correlations), so
    # divide by the variance-free mean and add its first-order contribution manually — the same
    # scheme as apply_background_roi. At T == 0 pixels this keeps the variance finite (Var(T)/m²),
    # where the previous corrected**2 * (Var(T)/T**2 + ...) form produced 0 * inf = NaN.
    coeff_var = sc.variances(coeff) if coeff.variances is not None else None
    coeff = coeff.copy()
    coeff.variances = None
    corrected = transmission / coeff

    if coeff_var is not None and corrected.variances is not None:
        rel = coeff_var / (coeff * coeff)
        extra = sc.array(dims=list(corrected.dims), values=corrected.values**2) * rel
        corrected.variances = corrected.variances + extra.values.astype(corrected.variances.dtype, copy=False)

    logger.success("✓ Air region correction applied")
    return corrected
