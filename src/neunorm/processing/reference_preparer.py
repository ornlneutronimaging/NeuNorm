"""
Reference image preparation.
"""

import numpy as np
import scipp as sc


def median_with_variance(data: sc.DataArray, dim: str, n_samples: int = 100_000) -> sc.DataArray:
    """Compute the median and an estimate of the propagation of variance."""
    if data.dims.index(dim) != 0:
        raise ValueError(f"Assuming dimension '{dim}' is first dimension of the input data.")

    values = data.values
    variance = data.variances

    nz, ny, nx = values.shape
    median_variance = np.empty((ny, nx), dtype=np.float64)

    rng = np.random.default_rng()

    for iy in range(ny):
        for ix in range(nx):
            # only allocates (n_samples, frame) for one pixel
            sims = rng.poisson(
                lam=variance[:, iy, ix],
                size=(n_samples, nz),
            )
            medians = np.median(sims, axis=1)
            median_variance[iy, ix] = np.var(medians, ddof=1)

    return sc.DataArray(
        data=sc.array(dims=data.dims[1:], values=np.median(values, axis=0), unit=data.unit, variances=median_variance)
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
