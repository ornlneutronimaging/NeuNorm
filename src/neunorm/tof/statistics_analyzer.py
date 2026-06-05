"""Per-TOF-bin counting-statistics analysis and rebinning recommendations."""

from dataclasses import dataclass

import numpy as np
import scipp as sc


@dataclass
class StatisticsReport:
    """Summary of per-TOF-bin counting statistics.

    Attributes
    ----------
    counts_per_bin : np.ndarray
        Total counts in each TOF bin.
    snr_per_bin : np.ndarray
        Signal-to-noise ratio per bin (``sqrt(N)`` for Poisson statistics).
    low_statistics_bins : np.ndarray
        Indices of bins whose SNR is below the requested minimum.
    recommended_rebinning : int
        Suggested rebinning factor to reach the target SNR.
    preserve_regions : list[tuple[int, int]]
        ``(start, end)`` index ranges flagged to preserve (e.g. Bragg edges).
    """

    counts_per_bin: np.ndarray
    snr_per_bin: np.ndarray
    low_statistics_bins: np.ndarray  # indices
    recommended_rebinning: int  # factor
    preserve_regions: list[tuple[int, int]]  # (start, end) indices


def analyze_statistics(data: sc.DataArray, min_snr: float = 3.0, tof_dim: str = "tof") -> StatisticsReport:
    """Analyze per-TOF-bin statistics and recommend rebinning.

    Requirements
    - Calculate total counts per TOF bin
    - Calculate SNR per TOF bin: SNR = √(N)
    - Identify bins with inadequate statistics (below threshold)
    - Generate rebinning recommendation
    - Flag features to preserve (Bragg edges, resonances) # TODO

    Parameters
    ----------
    data : sc.DataArray
        Input data with TOF dimension and counts as values. Should have Poisson statistics (variance = counts).
    min_snr : float
        Minimum acceptable signal-to-noise ratio (SNR) per bin. Default is 3.0.
    tof_dim : str
        Name of the TOF dimension in the DataArray. Default is "tof".
    """

    if tof_dim not in data.dims:
        raise ValueError(f"Specified TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")

    total_counts = data.sum(dim=tuple(d for d in data.dims if d != tof_dim))
    if np.any(total_counts.values < 0):
        raise ValueError("Negative counts detected in data. Can't analyze statistics for negative counts.")
    total_bins = total_counts.sizes[tof_dim]
    snr = np.sqrt(total_counts.values)
    low_stats_bins = np.where(snr < min_snr)[0]
    if len(low_stats_bins) == 0:
        recommended_rebinning = 1
    else:
        # Simple heuristic: increase rebinning factor until all bins have sufficient SNR
        recommended_rebinning = 2
        while True:
            # Pad array to make it divisible by recommended_rebinning
            remainder = len(total_counts.values) % recommended_rebinning
            if remainder != 0:
                padded_counts = np.pad(total_counts.values, (0, recommended_rebinning - remainder), mode="constant")
            else:
                padded_counts = total_counts.values
            rebinned_counts = padded_counts.reshape(-1, recommended_rebinning).sum(axis=1)
            rebinned_snr = np.sqrt(rebinned_counts)
            if np.all(rebinned_snr >= min_snr):
                break
            recommended_rebinning += 1
            if recommended_rebinning > total_bins:
                raise ValueError(
                    "Cannot achieve desired SNR with rebinning. Consider adjusting min_snr or checking data quality."
                )

    return StatisticsReport(
        counts_per_bin=total_counts.values,
        snr_per_bin=snr,
        low_statistics_bins=low_stats_bins,
        recommended_rebinning=recommended_rebinning,
        preserve_regions=[],
    )
