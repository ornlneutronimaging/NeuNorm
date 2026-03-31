"""
Module for rebinning TOF histograms. Provides functionality to combine adjacent TOF bins and update edges accordingly.
"""

import scipp as sc


def rebin_tof(  # noqa: C901
    data: sc.DataArray,
    width: float,
    unit: str = "bins",
    logarithmic: bool = False,
    tof_dim: str = "tof",
    l_source_to_detector: float = 25.0,
    detector_time_offset: float = 5000.0,
) -> sc.DataArray:
    """Combine N adjacent TOF bins. Returns rebinned data.

    Requirements
    - Combine N adjacent TOF bins by summing counts
    - Update TOF bin edges accordingly
    - Propagate variance correctly through summation

    Parameters
    ----------
    data : sc.DataArray
        Input data with TOF dimension.
    width : float
        Width of the new TOF bins in terms of the specified unit. Must be positive.
    unit : str
        Unit by which the new bin width is specified. Must be one of `time`, `wavelength` or `bins`. Default is `time`.
        If `bins`, width is interpreted as the number of adjacent bins to combine.
        If `time`, width is interpreted as the desired width of the new TOF bins in the same unit as the coordinates.
        If `wavelength`, width is interpreted as the desired width of the new TOF bins in Angstrom units,
        and converted to time using the provided source-to-detector distance and detector time offset.
    logarithmic : bool
        Whether to use logarithmic binning. Default is False.
    tof_dim : str
        Name of the TOF dimension in the DataArray. Default is "tof".
    l_source_to_detector : float
        Distance from the source to the detector in meters. Required for wavelength binning. Default is 25.0.
    detector_time_offset : float
        Time offset of the detector in same unit as TOF. Required for wavelength binning. Default is 5000.0.
    """

    if tof_dim not in data.dims:
        raise ValueError(f"Specified TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")

    if width <= 0:
        raise ValueError("Rebinning width must be positive.")

    if unit == "bins":
        if logarithmic:
            raise ValueError("Logarithmic binning is not supported when unit is 'bins'.")
        factor = int(width)

        if factor == 1:
            return data  # No rebinning needed
        # create new TOF edges by taking every Nth edge from the original TOF edges
        new_tof_edges = data.coords[tof_dim][::factor]
        # add last edge if not included
        if not sc.identical(new_tof_edges[-1], data.coords[tof_dim][-1]):
            new_tof_edges = sc.concat([new_tof_edges, data.coords[tof_dim][-1:]], dim=tof_dim)
    elif unit == "time":
        tof_edges = data.coords[tof_dim]
        if logarithmic:
            new_tof_edges = sc.geomspace(
                tof_edges[0], tof_edges[-1], num=int((sc.log(tof_edges[-1]) - sc.log(tof_edges[0])) / width) + 1
            )
        else:
            new_tof_edges = sc.scalar(width, unit=tof_edges.unit)
    elif unit == "wavelength":
        tof_edges = data.coords[tof_dim]
        lsd = sc.scalar(l_source_to_detector, unit="m")
        offset = sc.scalar(detector_time_offset, unit=tof_edges.unit)
        if logarithmic:
            # convert to wavelength edges, create logarithmic wavelength edges, then convert back to TOF edges
            wavelength_edges = sc.to_unit((tof_edges + offset) * sc.constants.h / (sc.constants.m_n * lsd), "Angstrom")
            new_wavelength_edges = sc.geomspace(
                wavelength_edges[0],
                wavelength_edges[-1],
                num=int((sc.log(wavelength_edges[-1]) - sc.log(wavelength_edges[0])) / width) + 1,
            )
            new_tof_edges = (
                sc.scalar(new_wavelength_edges, unit="Angstrom") * sc.constants.m_n * lsd / sc.constants.h - offset
            )
        else:
            new_tof_edges = sc.scalar(width, unit="Angstrom") * sc.constants.m_n * lsd / sc.constants.h
    else:
        raise ValueError("Invalid unit for rebinning width. Must be one of 'time', 'wavelength', or 'bins'.")

    # rebin histogrammed data by summing over the specified factor
    rebinned_data = sc.rebin(data, {tof_dim: new_tof_edges})
    return rebinned_data
