"""
Dark current correction.
"""

import numpy as np
import scipp as sc
from loguru import logger


def subtract_dark(data: sc.DataArray, dark: sc.DataArray, clip_negative: bool = True) -> sc.DataArray:
    """Subtract dark current with variance propagation.

    data_corr = data - dark

    Requirements:
    - Subtract dark current from sample and OB images
    - Propagate variance correctly through subtraction using scipp
    - Handle negative values (clip to zero or flag as invalid)
    - Support both 2D dark (averaged) and 3D dark (per-frame) inputs

    Parameters
    ----------
    data : sc.DataArray
        Sample or OB histogram with variance
    dark : sc.DataArray
        Dark current histogram with variance
    clip_negative : bool
        If True, clip negative values to zero after subtraction (default: True)
        If False, value will be masked

    Returns
    -------
    sc.DataArray
        Dark-corrected data with propagated variance
    """
    logger.info("Subtracting dark current")

    # Perform subtraction (scipp auto-propagates variance)
    if dark.dims == data.dims:
        # 3D dark (per-frame)
        corr = data - dark
    elif dark.dims == ("x", "y") and set(dark.dims).issubset(set(data.dims)):
        # 2D dark (averaged) - need to create a 3D version by repeating missing dimension.
        dim_name = set(data.dims) - set(dark.dims)
        if len(dim_name) != 1:
            raise ValueError("Data has multiple dimensions not in dark, cannot determine which to repeat along")
        dim_name = dim_name.pop()
        # Broadcast dark to match data dimensions.
        # Can't use sc.broadcast directly because it doesn't handle variances, so we need to do it manually.
        dark_copy = dark.copy()
        var = dark_copy.variances.copy() if dark_copy.variances is not None else None
        dark_copy.variances = None
        corr = data - dark_copy
        if var is not None:
            # Let numpy handle variance broadcasting
            corr.variances = data.variances + var
    else:
        raise ValueError("Dark current dimensions are incompatible with data dimensions")

    if clip_negative:
        corr.values = np.clip(corr.values, 0, None)
    else:
        negative_mask = corr.values < 0
        corr.masks["negative"] = sc.array(dims=corr.dims, values=negative_mask, dtype=bool)

    return corr
