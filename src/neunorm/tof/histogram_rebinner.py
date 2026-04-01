"""
Module for rebinning TOF histograms. Provides functionality to combine adjacent TOF bins and update edges accordingly.
"""

import numpy as np
import scipp as sc


def rebin_with_snapped_boundaries(old_edges: sc.Variable, requested_tof_edges: sc.Variable):
    """
    For requested TOF edges that don't align with original TOF edges, snap to the nearest original edge on the left.
    This ensures that we only combine adjacent bins and don't create arbitrary bin schemes,
    which is a requirement for histogram-mode data.
    """
    idx = np.searchsorted(old_edges.values, requested_tof_edges.values, side="right") - 1

    return old_edges[idx]


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

    Constraints

    For histogram-mode data (TPX1, TPX3 histogram mode):
    - Can ONLY combine adjacent bins
    - Cannot create arbitrary bin schemes (would require raw events)
    - Cannot split bins

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
            last_bin = np.ceil(np.log(tof_edges.values[-1] / tof_edges.values[0]) / np.log1p(width))
            requested_tof_edges = sc.array(
                dims=[tof_dim], values=tof_edges.values[0] * (1 + width) ** np.arange(last_bin + 1), unit=tof_edges.unit
            )
        else:
            requested_tof_edges = sc.arange(
                dim=tof_dim,
                start=tof_edges.values[0],
                stop=tof_edges.values[-1] + width,
                step=width,
                unit=tof_edges.unit,
            )

        new_tof_edges = rebin_with_snapped_boundaries(tof_edges, requested_tof_edges)

    elif unit == "wavelength":
        tof_edges = data.coords[tof_dim]
        lsd = sc.scalar(l_source_to_detector, unit="m")
        offset = sc.scalar(detector_time_offset, unit=tof_edges.unit)
        if logarithmic:
            # convert to wavelength edges, create logarithmic wavelength edges, then convert back to TOF edges
            wavelength_edges = sc.to_unit((tof_edges + offset) * sc.constants.h / (sc.constants.m_n * lsd), "Angstrom")
            last_bin = np.ceil(np.log(wavelength_edges.values[-1] / wavelength_edges.values[0]) / np.log1p(width))
            requested_wavelength_edges = sc.array(
                dims=[tof_dim],
                values=wavelength_edges.values[0] * (1 + width) ** np.arange(last_bin + 1),
                unit="Angstrom",
            )
            requested_tof_edges = sc.to_unit(
                sc.to_unit(requested_wavelength_edges * sc.constants.m_n * lsd / sc.constants.h, tof_edges.unit)
                - offset,
                tof_edges.unit,
            )
        else:
            requested_tof_width = sc.to_unit(
                sc.scalar(width, unit="Angstrom") * sc.constants.m_n * lsd / sc.constants.h, tof_edges.unit
            )
            requested_tof_edges = sc.arange(
                dim=tof_dim,
                start=tof_edges[0],
                stop=tof_edges[-1] + requested_tof_width,
                step=requested_tof_width,
                unit=tof_edges.unit,
            )

        new_tof_edges = rebin_with_snapped_boundaries(tof_edges, requested_tof_edges)
    else:
        raise ValueError("Invalid unit for rebinning width. Must be one of 'time', 'wavelength', or 'bins'.")

    # rebin histogrammed data by summing over the specified factor
    rebinned_data = sc.rebin(data, {tof_dim: new_tof_edges})
    return rebinned_data
