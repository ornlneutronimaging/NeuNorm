"""
Utility module for loading stacks of images in various formats.
"""

from pathlib import Path
from typing import Optional, Sequence

import scipp as sc

from neunorm.loaders.fits_loader import load_fits_stack
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.utils.progress import STAGE_LOAD_SAMPLE, ProgressLike


def load_stack(
    paths: Sequence[str | Path],
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
    max_workers: Optional[int] = None,
) -> sc.DataArray:
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
    stage : str, optional
        Stage label the events carry, also forwarded. Defaults to ``STAGE_LOAD_SAMPLE``; pass
        ``STAGE_LOAD_OB`` or ``STAGE_LOAD_DARK`` when loading those, so a caller's callback can
        tell the three loads of a run apart.
    max_workers : int, optional
        Threads the chosen leaf loader decodes with, also forwarded. Both default to 8 and both
        read serially at ``max_workers=1``. Forwarded rather than left to the leaf default so a
        caller on a shared analysis filesystem can turn the concurrency down without reaching past
        this dispatcher; the pipelines take the default and do not expose it.
    """

    # Materialise before indexing: this function subscripts `paths[0]` and iterates it twice, so a
    # generator would raise TypeError. The leaf loaders accept one, so this does too.
    if not hasattr(paths, "__len__"):
        paths = list(paths)

    if not paths:
        raise ValueError("No file paths provided")

    first_ext = Path(paths[0]).suffix.lower()
    if first_ext in (".tiff", ".tif"):
        for path in paths:
            if Path(path).suffix.lower() != first_ext:
                raise ValueError(f"All files must have the same extension. Found mixed extensions: {paths}")
        return load_tiff_stack(paths, progress=progress, stage=stage, max_workers=max_workers)
    elif first_ext in (".fits", ".fit", ".fts"):
        for path in paths:
            if Path(path).suffix.lower() != first_ext:
                raise ValueError(f"All files must have the same extension. Found mixed extensions: {paths}")
        return load_fits_stack(paths, progress=progress, stage=stage, max_workers=max_workers)
    else:
        raise ValueError(
            f"Unsupported file format: {first_ext}. Supported are TIFF (.tiff, .tif) and FITS (.fits, .fit, .fts)."
        )
