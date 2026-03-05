"""
Module for combining multiple runs of neutron imaging data into a single dataset with aggregated metadata.
"""

from typing import Sequence

import scipp as sc
from loguru import logger


def combine_runs(
    runs: list[sc.DataArray], metadata_keys_to_sum: Sequence[str] = ("acquisition_time", "p_charge")
) -> sc.DataArray:
    """Combine multiple runs by summing with metadata aggregation.

    Parameters
    ----------
    runs : list[sc.DataArray]
        List of DataArrays representing individual runs.
    metadata_keys_to_sum : Sequence[str], optional
        Sequence of metadata keys to sum across runs, by default ("acquisition_time", "p_charge")

    Returns
    -------
    sc.DataArray
        Combined DataArray with summed data and aggregated metadata.
    """
    logger.info("Combining {} runs with metadata keys to sum: {}", len(runs), metadata_keys_to_sum)

    if not runs:
        raise ValueError("No runs provided for combination")

    if len(runs) == 1:
        return runs[0]

    # Validate all runs have the same shape and dimensions
    base_shape = runs[0].shape
    base_dims = runs[0].dims
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

    # Combine data by summing across runs
    combined = runs[0].copy()
    for run in runs[1:]:
        combined += run

    # Aggregate metadata. Use first run data unless specified keys are to be summed
    for key in metadata_keys_to_sum:
        try:
            combined.coords[key] = sc.sum(sc.concat([run.coords[key] for run in runs], dim="run"), dim="run")
        except (KeyError, sc.DimensionError) as e:
            logger.error("Metadata key '{}' not found in all runs for summation: {}", key, e)
            raise ValueError(f"Metadata key '{key}' not found in all runs for summation")

    logger.success("Successfully combined {} runs", len(runs))
    return combined
