"""
Module for rebinning TOF histograms. Provides functionality to combine adjacent TOF bins and update edges accordingly.
"""

from typing import Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.tof.coordinate_converter import convert_tof_to_wavelength, convert_wavelength_to_tof


def rebin_with_snapped_boundaries(old_edges: sc.Variable, requested_tof_edges: sc.Variable):
    """
    For requested TOF edges that don't align with original TOF edges, snap to the nearest original edge on the left.
    This ensures that we only combine adjacent bins and don't create arbitrary bin schemes,
    which is a requirement for histogram-mode data.

    Parameters
    ----------
    old_edges : sc.Variable
        Original TOF bin edges.
    requested_tof_edges : sc.Variable
        Desired TOF bin edges that may not align with original edges.

    Returns
    -------
    sc.Variable
        New TOF edges snapped to the nearest original edges on the left.
    """
    # Map requested edges to indices of the nearest original edge on the left.
    old_vals = np.asarray(old_edges.values)
    req_vals = np.asarray(requested_tof_edges.values)
    idx = np.searchsorted(old_vals, req_vals, side="right") - 1
    # Prevent negative indices (which would wrap to the last element) or indices
    # beyond the last edge. This keeps snapping within the valid range of edges.
    idx = np.clip(idx, 0, len(old_vals) - 1)
    snapped_vals = old_vals[idx]
    # Validate that snapped edges form a strictly increasing sequence of original edges.
    if snapped_vals.size == 0 or not np.all(np.diff(snapped_vals) > 0):
        raise ValueError(
            "Requested TOF binning would require splitting existing bins or "
            "would produce non-increasing/zero-width bins. "
            "Adjust the requested TOF edges or bin width so that consecutive "
            "snapped edges correspond to strictly increasing original bin edges."
        )

    return sc.array(dims=old_edges.dims, values=snapped_vals, unit=old_edges.unit)


def rebin_tof(  # noqa: C901
    data: sc.DataArray,
    width: Union[int, float, sc.Variable],
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
    width : Union[int, float, sc.Variable]
        Width of the new TOF bins in terms of the specified unit. Must be positive.
        If a sc.Variable is provided, it is interpreted as the desired edges of the new TOF bins,
        can be in units of time, wavelength or dimensionless (interpreted as bin indices),
        and will be convertible to the unit of the TOF coordinates.
    unit : str
        Unit by which the new bin width is specified. Must be one of `time`, `wavelength`, `bins`, or `manual`.
        Default is `bins`.
        If `bins`, width is interpreted as the number of adjacent bins to combine.
        If `time`, width is interpreted as the desired width of the new TOF bins in the same unit as the coordinates.
        If `wavelength`, width is interpreted as the desired width of the new TOF bins in Angstrom units,
        and converted to time using the provided source-to-detector distance and detector time offset.
        If `manual`, width is required to be a 1-D sc.Variable representing the explicit TOF bin edges;
        its values are interpreted as TOF coordinate values (time-of-flight) and must be in, or convertible to,
        the unit of the TOF coordinates, and are treated as bin edges rather than bin indices or counts.
    logarithmic : bool
        Whether to use logarithmic binning. Default is False.
    tof_dim : str
        Name of the TOF dimension in the DataArray. Default is "tof".
    l_source_to_detector : float
        Distance from the source to the detector in meters. Required for wavelength binning. Default is 25.0.
    detector_time_offset : float
        Time offset of the detector in same unit as TOF. Required for wavelength binning. Default is 5000.0.

    Returns
    -------
    sc.DataArray
        Rebinned DataArray with updated TOF bins and propagated variance.
    """

    if tof_dim not in data.dims:
        raise ValueError(f"Specified TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")

    if isinstance(width, sc.Variable) and unit != "manual":
        raise ValueError(
            "When width is provided as a sc.Variable, unit must be set to 'manual' and "
            "the variable should represent the desired edges of the new TOF bins."
        )

    if unit == "manual":
        if not isinstance(width, sc.Variable):
            raise ValueError(
                "When unit is 'manual', width must be provided as a sc.Variable "
                "representing the desired edges of the new TOF bins."
            )
        if width.size < 2:
            raise ValueError("Manual TOF edges must have at least two values.")

        if width.unit == sc.units.dimensionless:
            # Interpret as bin indices and extract TOF edges
            if not np.issubdtype(width.values.dtype, np.integer):
                raise ValueError(
                    "When width is a dimensionless sc.Variable, it must have an integer dtype representing bin indices."
                )
            if np.any(width.values < 0) or np.any(width.values >= data.coords[tof_dim].size):
                raise ValueError("Bin indices in width are out of bounds for the TOF dimension.")
            new_tof_edges = data.coords[tof_dim][width.values]
        else:
            # Try to convert to the unit of the TOF coordinates
            try:
                converted_width = sc.to_unit(width, data.coords[tof_dim].unit)
            except sc.UnitError as e:
                # now try wavelength
                try:
                    lsd = sc.scalar(l_source_to_detector, unit="m")
                    offset = sc.scalar(detector_time_offset, unit=data.coords[tof_dim].unit)
                    converted_width = (
                        sc.to_unit(
                            sc.to_unit(width, unit="Angstrom") * sc.constants.m_n * lsd / sc.constants.h,
                            data.coords[tof_dim].unit,
                        )
                        - offset
                    )

                except sc.UnitError as e2:
                    raise ValueError(
                        f"Width provided as a sc.Variable could not be converted to the unit of the TOF coordinates. "
                        f"Conversion to time failed with error: {e}. "
                        f"Conversion to wavelength failed with error: {e2}."
                    )
            new_tof_edges = rebin_with_snapped_boundaries(data.coords[tof_dim], converted_width)
    elif unit == "bins":
        if width <= 0:
            raise ValueError("Rebinning width must be positive.")

        if logarithmic:
            raise ValueError("Logarithmic binning is not supported when unit is 'bins'.")

        # check if width is an integer and if not, raise an error
        if not isinstance(width, int):
            raise ValueError(
                "When unit is 'bins', width must be an integer representing the number of adjacent bins to combine."
            )

        if width == 1:
            return data  # No rebinning needed
        # create new TOF edges by taking every Nth edge from the original TOF edges
        new_tof_edges = data.coords[tof_dim][::width]
        # add last edge if not included
        if not sc.identical(new_tof_edges[-1], data.coords[tof_dim][-1]):
            new_tof_edges = sc.concat([new_tof_edges, data.coords[tof_dim][-1:]], dim=tof_dim)
    elif unit == "time":
        if width <= 0:
            raise ValueError("Rebinning width must be positive.")

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
        if width <= 0:
            raise ValueError("Rebinning width must be positive.")
        tof_edges = data.coords[tof_dim]
        lsd = sc.scalar(l_source_to_detector, unit="m")
        offset = sc.scalar(detector_time_offset, unit=tof_edges.unit)
        if logarithmic:
            # convert to wavelength edges, create logarithmic wavelength edges, then convert back to TOF edges
            wavelength_edges = convert_tof_to_wavelength(tof_edges, lsd, offset)
            last_bin = np.ceil(np.log(wavelength_edges.values[-1] / wavelength_edges.values[0]) / np.log1p(width))
            requested_wavelength_edges = sc.array(
                dims=[tof_dim],
                values=wavelength_edges.values[0] * (1 + width) ** np.arange(last_bin + 1),
                unit="Angstrom",
            )
            requested_tof_edges = convert_wavelength_to_tof(requested_wavelength_edges, lsd, offset)
        else:
            requested_tof_width = convert_wavelength_to_tof(
                sc.scalar(width, unit="Angstrom"), lsd, sc.scalar(0, unit=tof_edges.unit)
            )
            requested_tof_edges = sc.arange(
                dim=tof_dim,
                start=tof_edges.values[0],
                stop=tof_edges.values[-1] + requested_tof_width.values,
                step=requested_tof_width.values,
                unit=tof_edges.unit,
            )

        new_tof_edges = rebin_with_snapped_boundaries(tof_edges, requested_tof_edges)
    else:
        raise ValueError("Invalid unit for rebinning width. Must be one of 'manual', 'time', 'wavelength', or 'bins'.")

    # rebin histogrammed data by summing over the specified factor
    rebinned_data = sc.rebin(data, {tof_dim: new_tof_edges})

    # copy over unaligned coords; only DataArray/Dataset can be passed to sc.rebin
    # so for coord Variables we build rebinned edges and preserve the rest as-is.
    for coord in data.coords:
        if not data.coords[coord].aligned:
            try:
                if tof_dim in data.coords[coord].dims:
                    # turn into DataArray to use sc.rebin for edge rebinning, then convert back to Variable
                    rebinned_edges = sc.rebin(
                        sc.DataArray(data.coords[coord], coords={tof_dim: data.coords[tof_dim]}),
                        {tof_dim: new_tof_edges},
                    ).data
                    rebinned_data.coords[coord] = rebinned_edges
                else:
                    rebinned_data.coords[coord] = data.coords[coord]
                rebinned_data.coords.set_aligned(coord, False)
            except (sc.BinEdgeError, sc.DimensionError) as e:
                logger.warning(f"Failed to rebin coordinate '{coord}' along TOF dimension. Error: {e}")

    return rebinned_data
