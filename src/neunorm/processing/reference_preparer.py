"""
Reference image preparation.
"""

import numpy as np
import scipp as sc


def median_with_variance(data: sc.DataArray, dim: str, n_samples: int = 100_000) -> sc.DataArray:
    """Compute the median and an estimate of the propagation of variance.

    This is an approximation that uses a Monte Carlo approach to estimate the variance of the median.
    For each pixel, we generate n_samples Poisson random samples based on the input data values,
    compute the median for each sample, and then calculate the variance of those medians to estimate
    the variance of the median.

    Parameters
    ----------
    data : sc.DataArray
        Input data with Poisson statistics.
    dim : str
        Dimension along which to compute the median.
    n_samples : int
        Number of Monte Carlo samples to generate for variance estimation.

    Returns
    -------
    sc.DataArray        DataArray containing the median values and their estimated variances.
    """
    values = data.values
    axis = data.dims.index(dim)
    out_dims = tuple(d for d in data.dims if d != dim)

    rng = np.random.default_rng()
    samples = rng.poisson(lam=values, size=(n_samples,) + values.shape)
    medians = np.median(samples, axis=axis + 1)  # +1 because of leading sample axis
    median_variance = np.var(medians, axis=0, ddof=1)

    return sc.DataArray(
        data=sc.array(
            dims=out_dims,
            values=np.median(values, axis=axis),
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
