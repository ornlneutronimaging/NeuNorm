"""
ASCII writer for 1-D transmission spectra, in the comma-delimited ``*_Spectra.txt`` shape.
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.utils.progress import STAGE_EXPORT, ProgressLike, resolve_progress

#: The one header line the file carries, naming its three columns. No ``#`` prefix: the ecosystem's
#: readers skip exactly one line and then parse the rest as data
#: (``np.loadtxt(path, skiprows=1, delimiter=",")`` in :mod:`neunorm.loaders.metadata_loader`), so a
#: commented header would be counted as the skipped line and the first data row lost.
ASCII_SPECTRUM_HEADER = "bin_index,transmission,uncertainty"

#: Per-column formats. ``bin_index`` is an integer frame index; the two value columns are fixed-point
#: with six decimals, the precision the agreed format shows (``0,0.448760,0.010663``). The default
#: ``%.18e`` would render ``bin_index`` as ``0.000000000000000000e+00``.
_ASCII_SPECTRUM_FMT = ("%d", "%.6f", "%.6f")


def ascii_spectrum_export_step_count(spectrum: sc.DataArray) -> int:  # noqa: ARG001 - fixed cost, by design
    """How many progress steps :func:`write_ascii_spectrum` reports.

    One: the file is a single small write, and splitting it would report progress that does not exist.
    Public and taking the spectrum anyway, so a pipeline sizes this stage the same way it sizes the
    HDF5 and TIFF exports — ``hdf5_export_step_count``'s counterpart — and a later change to the
    writer cannot leave a caller's total stale.
    """
    return 1


def _resolved_bin_indices(spectrum: sc.DataArray, bin_indices: Optional[Sequence[int]], n: int) -> np.ndarray:
    """The first column's values: the given input-frame indices, or the row index when none are given."""
    if bin_indices is None:
        return np.arange(n, dtype=np.int64)
    indices = np.asarray(list(bin_indices))
    if indices.ndim != 1 or indices.size != n:
        raise ValueError(
            f"bin_indices must give one index per spectrum bin; got {indices.size} for a spectrum of "
            f"{n} bin(s) (dims {spectrum.dims})"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"bin_indices must be integers; got dtype {indices.dtype}")
    return indices.astype(np.int64, copy=False)


def write_ascii_spectrum(
    output_path: Union[Path, str],
    spectrum: sc.DataArray,
    bin_indices: Optional[Sequence[int]] = None,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_EXPORT,
) -> Path:
    """Write a 1-D transmission spectrum as a three-column comma-delimited ASCII file.

    Output structure
    ----------------

    Exactly one plain header line, then one row per spectrum bin::

        bin_index,transmission,uncertainty
        0,0.448760,0.010663
        1,0.451203,0.010701

    This mirrors the ``*_Spectra.txt`` convention the ORNL imaging tools already exchange, so the
    file is readable by NeuNorm's own spectra reader with no special casing:
    ``np.loadtxt(path, skiprows=1, delimiter=",")``.

    ``uncertainty`` is **1 sigma** — ``sqrt(variance)`` — the same quantity the HDF5 writer stores as
    ``/uncertainty``. A spectrum carrying no variances is rejected rather than written with a
    fabricated third column.

    ``bin_index`` identifies each point by its **first input frame index**, which is what makes the
    column mean "file index" when no rebinning was requested, as the format's users expect. Under a
    rebinning it stays traceable: frames 0-1 and 4-5 binned as ``[[0, 2], [4, 6]]`` give rows indexed
    ``0`` and ``4``, and the dropped 2-3 span simply has no row. The column therefore has gaps rather
    than renumbered rows, and the full bin-to-frame mapping belongs in the run's provenance.

    Values are written from the DataArray at six decimal places. Masks are not representable in a
    three-column format and are not written here; the HDF5 file written alongside carries them, along
    with the run's provenance and the spectrum's time axis.

    Parameters
    ----------
    output_path : Path or str
        File to write. Parent directories are created; an existing file is overwritten.
    spectrum : sc.DataArray
        The 1-D transmission spectrum, carrying variances.
    bin_indices : sequence of int, optional
        One input-frame index per bin, in order — normally each bin's first frame. Defaults to the
        row index ``0..N-1``, which is the same thing when no rebinning was applied.
    progress : bool or callable, optional
        Progress reporting, off by default. Reports the single write step. Accepts an existing
        ``ProgressReporter``, which is how a pipeline keeps one export count across the HDF5 and
        ASCII writes; size it with :func:`ascii_spectrum_export_step_count`. See
        :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_EXPORT``.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ValueError
        If ``spectrum`` is not 1-D, carries no variances, or ``bin_indices`` does not match it.
    PermissionError
        If the parent directory is not writeable.
    """
    if len(spectrum.dims) != 1:
        raise ValueError(
            f"write_ascii_spectrum needs a 1-D spectrum; got dims {spectrum.dims} with sizes "
            f"{dict(spectrum.sizes)}. Reduce the spatial dimensions first (see "
            "neunorm.processing.spectrum_reducer.normalize_roi_spectrum)."
        )
    if spectrum.variances is None:
        raise ValueError(
            "write_ascii_spectrum needs a spectrum carrying variances: the file's third column is a "
            "1-sigma uncertainty, and there is nothing truthful to write in it otherwise."
        )

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if path is writeable
    if not os.access(output_path.parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {output_path.parent}")

    dim = spectrum.dims[0]
    n = spectrum.sizes[dim]
    indices = _resolved_bin_indices(spectrum, bin_indices, n)

    with resolve_progress(progress, stage, total=ascii_spectrum_export_step_count(spectrum)) as report:
        report.note("writing ASCII spectrum")
        table = np.column_stack(
            [
                indices.astype(float),
                np.asarray(spectrum.values, dtype=float),
                np.sqrt(np.asarray(spectrum.variances, dtype=float)),
            ]
        )
        np.savetxt(
            output_path,
            table,
            delimiter=",",
            header=ASCII_SPECTRUM_HEADER,
            comments="",
            fmt=_ASCII_SPECTRUM_FMT,
        )
        report()

    logger.info("ASCII spectrum ({} bins) written to {}", n, output_path)
    return output_path
