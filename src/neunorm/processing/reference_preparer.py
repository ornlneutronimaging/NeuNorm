"""
Reference image preparation.
"""

import numpy as np
import scipp as sc
from loguru import logger


def median_with_variance(data: sc.DataArray, dim: str) -> sc.DataArray:
    """Compute the median and an approximation of the propagation of variance.

    Use the approximation of

    Var(median) ≈ (π / (2n)) * mean_variance

    Parameters
    ----------
    data : sc.DataArray
        Input data with associated variances.
    dim : str
        Dimension along which to compute the median.

    Returns
    -------
    sc.DataArray
        DataArray containing the median values and their estimated variances.
    """
    axis = data.dims.index(dim)
    out_dims = tuple(d for d in data.dims if d != dim)

    # Calculate mean variance along the specified dimension
    mean_variance = data.variances.mean(axis=axis)
    median_variance = (np.pi / (2 * data.sizes[dim])) * mean_variance

    return sc.DataArray(
        data=sc.array(
            dims=out_dims,
            values=np.median(data.values, axis=axis),
            unit=data.unit,
            variances=median_variance,
        )
    )


def _reduce_coord(name: str, coord: sc.Variable, method: str, dim: str) -> sc.Variable:
    """Reduce the per-frame coordinate ``name`` along ``dim`` with ``method`` ("mean" or "median").

    A coordinate whose dtype cannot be averaged, such as text, keeps its first and last values
    along ``dim`` instead.
    """
    try:
        return coord.mean(dim=dim) if method == "mean" else coord.median(dim=dim)
    except (TypeError, np.exceptions.AxisError):
        logger.info(
            "Could not reduce coordinate '{}' along dimension '{}'. Keeping its first and last values.",
            name,
            dim,
        )
        return sc.concat([coord[dim, 0:1], coord[dim, -1:]], dim=dim)


def prepare_reference(
    stack: sc.DataArray,
    method: str = "mean",
    dim: str = "frame",
) -> sc.DataArray:
    """Reduce a 3D frame stack to a 2D reference image.

    Unaligned coordinates are carried over to the reference. Those along ``dim`` are reduced with
    the same ``method``, except one whose dtype cannot be averaged, such as a text tag that differs
    per frame (e.g. the TIFF ``DateTime``): it keeps its first and last values along ``dim``, the
    span of the frames that were reduced. Aligned coordinates along ``dim`` are dropped and the
    others kept. Masks along ``dim`` leave the masked values out of the reduction and are dropped;
    the others are kept. ``method="median"`` on data with variances ignores masks: every value enters
    the median and its variance, and the reference has no masks and no aligned coordinates.

    Parameters
    ----------
    stack : sc.DataArray
        3D input with dimensions (frame, y, x).
    method : str
        Reduction method: "mean" or "median".
    dim : str
        Dimension along which to reduce.

    Returns
    -------
    sc.DataArray
        2D reference image (y, x) with propagated variances.
    """
    logger.info("Preparing reference image using method '{}' along dimension '{}'", method, dim)

    if len(stack.dims) == 2:
        return stack

    if dim not in stack.dims:
        raise ValueError(f"Dimension '{dim}' not found in input data. Available dimensions: {stack.dims}")

    if method == "mean":
        result = stack.mean(dim=dim)
    elif method == "median":
        if stack.variances is None:
            result = stack.median(dim=dim)
        else:
            result = median_with_variance(stack, dim=dim)
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'mean' or 'median'.")

    # Reduce unaligned coords along `dim` with the same method; copy the ones without `dim`.
    for name, coord in stack.coords.items():
        if not coord.aligned:
            result.coords[name] = _reduce_coord(name, coord, method, dim) if dim in coord.dims else coord
            result.coords.set_aligned(name, False)

    return result
