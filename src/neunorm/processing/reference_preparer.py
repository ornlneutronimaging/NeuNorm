"""
Reference image preparation.
"""

import numpy as np
import scipp as sc


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


def prepare_reference(
    stack: sc.DataArray,
    method: str = "mean",
    dim: str = "frame",
) -> sc.DataArray:
    """Reduce a 3D frame stack to a 2D reference image.

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

    if len(stack.dims) == 2:
        return stack

    if dim not in stack.dims:
        raise ValueError(f"Dimension '{dim}' not found in input data. Available dimensions: {stack.dims}")

    if method == "mean":
        return stack.mean(dim=dim)

    if method == "median":
        if stack.variances is None:
            return stack.median(dim=dim)
        else:
            return median_with_variance(stack, dim=dim)

    raise ValueError(f"Unsupported method '{method}'. Use 'mean' or 'median'.")
