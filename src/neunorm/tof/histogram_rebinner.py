"""
Module for rebinning TOF histograms. Provides functionality to combine adjacent TOF bins and update edges accordingly.
"""

import scipp as sc


def rebin_tof(data: sc.DataArray, factor: int, tof_dim: str = "tof") -> sc.DataArray:
    """Combine N adjacent TOF bins. Returns rebinned data.

    Requirements
    - Combine N adjacent TOF bins by summing counts
    - Update TOF bin edges accordingly
    - Propagate variance correctly through summation

    Parameters
    ----------
    data : sc.DataArray
        Input data with TOF dimension.
    factor : int
        Number of adjacent TOF bins to combine. Must be a positive integer.
    tof_dim : str
        Name of the TOF dimension in the DataArray. Default is "tof".
    """

    if tof_dim not in data.dims:
        raise ValueError(f"Specified TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")

    if factor <= 0:
        raise ValueError("Rebinning factor must be a positive integer.")

    if factor == 1:
        return data  # No rebinning needed

    # create new TOF edges by taking every Nth edge from the original TOF edges
    new_tof_edges = data.coords[tof_dim][::factor]
    # add last edge if not included
    if not sc.identical(new_tof_edges[-1], data.coords[tof_dim][-1]):
        new_tof_edges = sc.concat([new_tof_edges, data.coords[tof_dim][-1:]], dim=tof_dim)

    # rebin histogrammed data by summing over the specified factor
    rebinned_data = sc.rebin(data, {tof_dim: new_tof_edges})
    return rebinned_data
