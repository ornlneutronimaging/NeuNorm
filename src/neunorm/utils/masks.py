"""
Combining scipp masks into one boolean array.

Three places need the same thing — every mask on an array, OR-ed together and broadcast **by
dimension name** so a 2-D ``(y, x)`` dead-pixel mask expands correctly over a ``(tof, x, y)`` stack
whatever order the dims come in. They had grown three copies of the loop.

The one real difference between them is what to do with a mask carrying a dim the target does not
have, so that is the only knob: a reduction whose spatial axes are already gone wants such a mask
skipped, while an image writer flattening masks for export wants to hear about it rather than
silently drop a mask from the file it is writing.
"""

from typing import Mapping, Optional

import numpy as np
import scipp as sc


def combined_mask(
    sizes: Mapping[str, int],
    masks: Mapping[str, sc.Variable],
    *,
    skip_mismatched: bool = False,
) -> Optional[np.ndarray]:
    """One boolean array, ``True`` wherever any mask flags the element.

    Parameters
    ----------
    sizes : Mapping[str, int]
        The target shape, as a scipp ``sizes`` mapping. Masks are broadcast into it by dim name.
    masks : Mapping[str, sc.Variable]
        The masks to combine, e.g. ``data.masks``.
    skip_mismatched : bool, optional
        With ``True``, a mask carrying a dim that ``sizes`` does not have is skipped: it is
        meaningful to its own array but cannot select anything here. With the default ``False`` such
        a mask reaches :func:`scipp.broadcast` and raises, which is what a caller wants when
        silently dropping a mask would lose information.

    Returns
    -------
    Optional[np.ndarray]
        The combined mask, or ``None`` when no mask contributes — which lets a caller skip work
        rather than carry an all-``False`` array around.

    Examples
    --------
    >>> import numpy as np, scipp as sc
    >>> data = sc.DataArray(sc.array(dims=["tof", "y", "x"], values=np.zeros((2, 3, 3))))
    >>> data.masks["dead"] = sc.array(dims=["y", "x"], values=np.eye(3, dtype=bool))
    >>> combined_mask(data.sizes, data.masks).shape
    (2, 3, 3)
    >>> combined_mask(data.sizes, {}) is None
    True
    """
    selected = [mask for mask in masks.values() if not skip_mismatched or set(mask.dims) <= set(sizes)]
    if not selected:
        return None
    combined = np.zeros(tuple(sizes.values()), dtype=bool)
    for mask in selected:
        combined |= sc.broadcast(mask, sizes=sizes).values
    return combined
