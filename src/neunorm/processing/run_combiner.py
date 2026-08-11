"""
Module for combining multiple runs of neutron imaging data into a single dataset with aggregated metadata.
"""

from typing import Sequence

import numpy as np
import scipp as sc
from loguru import logger


def _metadata_matches(a: sc.Variable, b: sc.Variable, atol: float) -> bool:
    """True when two metadata coords are identical, or — for numeric values of
    the same unit and shape — differ element-wise by at most ``atol``."""
    if sc.identical(a, b):
        return True
    if atol <= 0 or a.unit != b.unit or a.shape != b.shape:
        return False
    try:
        av = np.asarray(a.value if a.ndim == 0 else a.values, dtype=float)
        bv = np.asarray(b.value if b.ndim == 0 else b.values, dtype=float)
    except (TypeError, ValueError):
        return False  # non-numeric (e.g. string) values must match exactly
    return bool(np.all(np.abs(av - bv) <= atol))


def combine_runs(  # noqa: C901
    runs: list[sc.DataArray],
    metadata_keys_to_sum: Sequence[str] = ("acquisition_time", "p_charge"),
    metadata_check_match: Sequence[str] = (),
    normalize_by_runs: bool = False,
    metadata_match_atol: float = 0.0,
) -> sc.DataArray:
    """Combine multiple runs by summing with metadata aggregation.

    - Combines a list of sc.DataArrays representing individual runs by summing their data values and variances.
    - Masks are combined using logical OR. If a pixel is masked in any run, it will be masked in the combined result.
      This ensures that all bad pixels are properly accounted for in the final dataset.
    - The function also aggregates metadata by summing specified keys across runs, while retaining other metadata from
      the first run. This allows for accurate representation of total acquisition time, proton charge, and other
      relevant parameters in the combined dataset.

    Parameters
    ----------
    runs : list[sc.DataArray]
        List of DataArrays representing individual runs.
    metadata_keys_to_sum : Sequence[str], optional
        Sequence of metadata keys to aggregate across runs, by default ("acquisition_time", "p_charge").
        These coordinates are summed across runs unless ``normalize_by_runs=True``, in which case they
        are averaged instead.
    metadata_check_match : Sequence[str], optional
        Sequence of metadata keys that must match across all runs for combination, by default ()
    normalize_by_runs : bool, optional
        Whether to normalize the combined data by the number of runs, by default False
    metadata_match_atol : float, optional
        Absolute tolerance for the ``metadata_check_match`` comparison of numeric
        values (same unit and shape required), by default 0.0 (exact match).
        Non-numeric values (e.g. detector names) always require an exact match.
        Useful for slit/aperture readback positions that jitter slightly
        between runs.

    Returns
    -------
    sc.DataArray
        Combined DataArray with summed data and aggregated metadata.
    """
    logger.info("Combining {} runs with metadata keys to sum: {}", len(runs), metadata_keys_to_sum)

    if not runs:
        raise ValueError("No runs provided for combination")

    if len(runs) == 1:
        # Return a copy so the combined result never aliases caller-owned data
        # (the multi-run path below also builds a fresh array via .copy()).
        return runs[0].copy()

    # Validate all runs have the same shape, dimensions and required metadata keys for matching
    base_shape = runs[0].shape
    base_dims = runs[0].dims
    base_metadata = runs[0].coords
    for key in metadata_check_match:
        if key not in base_metadata:
            logger.error("Metadata key '{}' not found in base run for matching", key)
            raise ValueError(f"Metadata key '{key}' not found in base run for matching")

    for i, run in enumerate(runs[1:], 1):
        if run.shape != base_shape or run.dims != base_dims:
            logger.error(
                "Run {} has shape {} and dims {}, expected shape {} and dims {}",
                i,
                run.shape,
                run.dims,
                base_shape,
                base_dims,
            )
            raise ValueError(
                f"Run {i} has shape {run.shape} and dims {run.dims}, expected shape {base_shape} and dims {base_dims}"
            )
        for key in metadata_check_match:
            if key not in run.coords:
                logger.error("Metadata key '{}' not found in run {} for matching", key, i)
                raise ValueError(f"Metadata key '{key}' not found in run {i} for matching")
            if not _metadata_matches(run.coords[key], base_metadata[key], metadata_match_atol):
                logger.error(
                    "Metadata key '{}' does not match between run {} and base run: {} vs {}",
                    key,
                    i,
                    run.coords[key].value if run.coords[key].ndim == 0 else run.coords[key].values,
                    base_metadata[key].value if base_metadata[key].ndim == 0 else base_metadata[key].values,
                )
                raise ValueError(
                    f"Metadata key '{key}' does not match between run {i} and base run:"
                    f" {run.coords[key].value if run.coords[key].ndim == 0 else run.coords[key].values} vs"
                    f" {base_metadata[key].value if base_metadata[key].ndim == 0 else base_metadata[key].values}"
                )

    # Combine data by summing across runs. Coords that matched only within
    # metadata_match_atol would still fail scipp's exact aligned-coord check
    # in `+=`, so they are aligned to the base run's value on a shallow copy
    # (the caller's array is left untouched; metadata_keys_to_sum below still
    # aggregates the original per-run values).
    combined = runs[0].copy()
    for run in runs[1:]:
        patched = None
        for key in metadata_check_match:
            if not sc.identical(run.coords[key], base_metadata[key]):
                if patched is None:
                    patched = run.copy(deep=False)
                patched.coords[key] = base_metadata[key]
        combined += patched if patched is not None else run

    if normalize_by_runs:
        combined /= len(runs)

    # Aggregate metadata. Use first run data unless specified keys are to be summed
    for key in metadata_keys_to_sum:
        try:
            if normalize_by_runs:
                combined.coords[key] = sc.mean(sc.concat([run.coords[key] for run in runs], dim="run"), dim="run")
            else:
                combined.coords[key] = sc.sum(sc.concat([run.coords[key] for run in runs], dim="run"), dim="run")
            combined.coords.set_aligned(key, False)

        except (KeyError, sc.DimensionError) as e:
            logger.error("Metadata key '{}' not found in all runs for summation: {}", key, e)
            raise ValueError(f"Metadata key '{key}' not found in all runs for summation")

    logger.success("Successfully combined {} runs", len(runs))
    return combined
