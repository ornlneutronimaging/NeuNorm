"""
Output-path rules shared by every pipeline entry point and export step.
"""

from pathlib import Path
from typing import Optional

#: Suffixes written as HDF5, matched case-insensitively.
HDF5_SUFFIXES = (".hdf5", ".h5")
#: Suffixes written as a TIFF image stack, matched case-insensitively.
TIFF_SUFFIXES = (".tiff", ".tif")
#: Suffix written as the three-column ASCII spectrum, with an HDF5 file alongside it.
SPECTRUM_TEXT_SUFFIX = ".txt"


def unsupported_suffix_error(output_path: Path) -> ValueError:
    """Return the error for an ``output_path`` whose suffix no writer handles."""
    return ValueError(f"Unsupported output file format: {output_path.suffix}")


def spectrum_tiff_error(output_path: Path) -> ValueError:
    """Return the error for a TIFF ``output_path`` on a run that writes a ``spectrum_roi`` spectrum."""
    return ValueError(
        "spectrum_roi produces a 1-D spectrum, which cannot be written as a TIFF image stack "
        f"(got {output_path.name}). Use '.txt' for the three-column ASCII spectrum (an HDF5 file "
        "is written alongside it) or '.hdf5' for HDF5 only."
    )


def resolve_output_path(output_path: Optional[str | Path], *, spectrum: bool = False) -> Path:
    """Return ``output_path`` as a ``Path``, refusing a suffix the export step cannot write.

    Entry points call this before reading any input, so a missing path or an unsupported suffix fails
    at once rather than after the whole run. The messages are the ones the export step raises for the same path.

    Parameters
    ----------
    output_path : str or Path, optional
        Where the pipeline writes. ``None`` is refused.
    spectrum : bool, optional
        Whether the run writes a ``spectrum_roi`` spectrum. Image output takes ``.hdf5``/``.h5`` or
        ``.tiff``/``.tif``; spectrum output takes ``.hdf5``/``.h5`` or ``.txt`` and refuses TIFF.
        Suffixes match case-insensitively.

    Returns
    -------
    Path
        ``output_path`` as a ``Path``.

    Raises
    ------
    ValueError
        If ``output_path`` is ``None`` or its suffix is not one the run can write.
    """
    if output_path is None:
        raise ValueError("output_path is required")
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if spectrum and suffix in TIFF_SUFFIXES:
        raise spectrum_tiff_error(output_path)
    supported = HDF5_SUFFIXES + ((SPECTRUM_TEXT_SUFFIX,) if spectrum else TIFF_SUFFIXES)
    if suffix not in supported:
        raise unsupported_suffix_error(output_path)
    return output_path
