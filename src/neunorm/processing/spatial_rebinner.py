"""
Module for rebinning spatial dimensions.
Provides functionality to combine adjacent spatial pixels.
"""

import scipp as sc


def rebin_spatial(data: sc.DataArray, factor: int | tuple[int, int]) -> sc.DataArray:
    """Bin NxN spatial pixels. Trade-off: lose spatial resolution.

    Bin NxN spatial pixels by summing counts
    Support 2D, 3D, and 4D arrays (preserve other dimensions)
    Propagate variance correctly
    Handle edge cases (non-divisible dimensions)

    Parameters
    ----------
    data : sc.DataArray
        Input data with spatial dimensions 'x' and 'y'.
    factor : int or tuple[int, int]
        Number of adjacent pixels to combine in x and y directions.
        If a single integer is provided, it is used for both dimensions.
    """

    # Validate data has spatial dimensions
    if "x" not in data.dims or "y" not in data.dims:
        raise ValueError("Input data must have spatial dimensions 'x' and 'y'.")

    # Validate factor and determine factor_x and factor_y
    if isinstance(factor, int):
        factor_x = factor_y = factor
    elif isinstance(factor, tuple) and len(factor) == 2 and all(isinstance(f, int) for f in factor):
        factor_x, factor_y = factor
    else:
        raise ValueError("Factor must be an integer or a tuple of two integers.")

    if factor_x <= 0 or factor_y <= 0:
        raise ValueError("Rebinning factors must be positive integers.")

    if factor_x == 1 and factor_y == 1:
        return data  # No rebinning needed

    # check that they are divisible, if not raise an error
    x_size = data.sizes["x"]
    y_size = data.sizes["y"]

    if x_size < factor_x or y_size < factor_y:
        raise ValueError("Rebinning factors must be less than or equal to the number of pixels in each dimension.")
    if x_size % factor_x != 0 or y_size % factor_y != 0:
        raise ValueError("Rebinning factors must divide the number of pixels in each dimension.")

    # Rebin spatial dimensions by summing over the specified factors.
    # Use fold and sum to rebin the spatial dimensions.
    return (
        data.fold(dim="x", dims=["x", "x_to_sum"], shape=(-1, factor_x))
        .sum("x_to_sum")
        .fold(dim="y", dims=["y", "y_to_sum"], shape=(-1, factor_y))
        .sum("y_to_sum")
    )
