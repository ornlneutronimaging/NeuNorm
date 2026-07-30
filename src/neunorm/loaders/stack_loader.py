"""
Utility module for loading stacks of images in various formats.
"""

from pathlib import Path
from typing import Sequence

import scipp as sc

from neunorm.loaders.fits_loader import load_fits_stack
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.utils.progress import Progress


def load_stack(paths: Sequence[str | Path], *, progress: Progress = False) -> sc.DataArray:
    """
    Load a stack of images from the given file paths, supporting both TIFF and FITS formats.

    Check the extension of the first file in the list and call the appropriate loader function
    load_tiff_stack or load_fits_stack.

    Verify all files have the same extension and raise an error if not.

    Parameters
    ----------
    paths : Sequence[str | Path]
        Image files to load. All must share one extension.
    progress : bool or callable, optional
        Passed straight through to the chosen leaf loader, which emits one event per file. This
        pass-through is what gives the CCD pipelines per-file progress: they call ``load_stack``
        rather than the leaf loaders directly. See :mod:`neunorm.utils.progress`.
    """

    if not paths:
        raise ValueError("No file paths provided")

    first_ext = Path(paths[0]).suffix.lower()
    if first_ext in (".tiff", ".tif"):
        for path in paths:
            if Path(path).suffix.lower() != first_ext:
                raise ValueError(f"All files must have the same extension. Found mixed extensions: {paths}")
        return load_tiff_stack(paths, progress=progress)
    elif first_ext in (".fits", ".fit", ".fts"):
        for path in paths:
            if Path(path).suffix.lower() != first_ext:
                raise ValueError(f"All files must have the same extension. Found mixed extensions: {paths}")
        return load_fits_stack(paths, progress=progress)
    else:
        raise ValueError(
            f"Unsupported file format: {first_ext}. Supported are TIFF (.tiff, .tif) and FITS (.fits, .fit, .fts)."
        )
