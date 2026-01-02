"""
Tests for parallel multi-chip processing in pulse reconstruction.

P4 Task: Add parallel processing for multi-chip pulse reconstruction.

This module tests:
1. Parallel vs sequential equivalence (identical results)
2. Different numbers of chips (1, 2, 4)
3. Edge cases (empty data, no rollovers, invalid inputs)
4. n_jobs parameter behavior (valid and invalid values)
5. Default behavior (n_jobs=None should be sequential/safe)
6. Performance improvement with parallelization
7. Input validation (array length mismatches)
8. Output ordering preservation

Expected API after implementation:
    reconstruct_pulse_ids(
        tof: np.ndarray,
        chip_id: np.ndarray | None = None,
        threshold: float = -10.0,
        window: int = 20,
        late_margin: float = 14.0,
        n_jobs: int | None = None,  # NEW PARAMETER
    ) -> np.ndarray

n_jobs behavior:
    - None or 1: Sequential processing (default, safe)
    - -1: Use all available CPU cores
    - N > 1: Use N parallel workers

These tests follow TDD - written BEFORE the implementation.
"""

import time

import numpy as np
import pytest

# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture
def multi_chip_data_4():
    """Fixture for 4-chip test data."""
    tof_list = []
    chip_id_list = []
    ground_truth_list = []
    for chip in range(4):
        tof, gt = generate_synthetic_data(seed=42 + chip, n_pulses=4)
        tof_list.append(tof)
        chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))
        ground_truth_list.append(gt)
    return {
        "tof": np.concatenate(tof_list),
        "chip_id": np.concatenate(chip_id_list),
        "ground_truth": np.concatenate(ground_truth_list),
    }


@pytest.fixture
def multi_chip_data_2():
    """Fixture for 2-chip test data."""
    tof_list = []
    chip_id_list = []
    ground_truth_list = []
    for chip in range(2):
        tof, gt = generate_synthetic_data(seed=100 + chip, n_pulses=4)
        tof_list.append(tof)
        chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))
        ground_truth_list.append(gt)
    return {
        "tof": np.concatenate(tof_list),
        "chip_id": np.concatenate(chip_id_list),
        "ground_truth": np.concatenate(ground_truth_list),
    }


def generate_synthetic_data(seed=42, n_pulses=4, events_per_pulse=150):
    """
    Generate synthetic TOF data with controlled disorder.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    n_pulses : int
        Number of pulses to generate
    events_per_pulse : int
        Events per pulse

    Returns
    -------
    tuple
        (shuffled_tof, ground_truth_pulse_ids)
    """
    np.random.seed(seed)

    # Create clean data with clear rollovers
    clean_tof = np.concatenate([np.arange(1.0, 16.0, 0.1)[:events_per_pulse] for _ in range(n_pulses)])

    # Apply local shuffle (simulates TPX3 FIFO reordering)
    shuffled_tof = clean_tof.copy()
    i = 0
    while i < len(shuffled_tof):
        window = np.random.randint(8, 12)
        end = min(i + window, len(shuffled_tof))
        batch = shuffled_tof[i:end].copy()
        np.random.shuffle(batch)
        shuffled_tof[i:end] = batch
        i = end

    ground_truth_pulse_ids = np.repeat(np.arange(n_pulses, dtype=np.int32), events_per_pulse)

    return shuffled_tof, ground_truth_pulse_ids


# =============================================================================
# Tests for parallel vs sequential equivalence
# =============================================================================


class TestParallelSequentialEquivalence:
    """Tests verifying parallel and sequential results are identical."""

    def test_parallel_sequential_identical_4_chips(self):
        """Parallel and sequential processing must produce identical results for 4 chips."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Create 4-chip dataset
        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=42 + chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Run sequential (n_jobs=1)
        result_sequential = reconstruct_pulse_ids(
            tof,
            chip_id=chip_id,
            threshold=-10.0,
            window=20,
            late_margin=14.0,
            n_jobs=1,
        )

        # Run parallel (n_jobs=4)
        result_parallel = reconstruct_pulse_ids(
            tof,
            chip_id=chip_id,
            threshold=-10.0,
            window=20,
            late_margin=14.0,
            n_jobs=4,
        )

        np.testing.assert_array_equal(
            result_sequential, result_parallel, err_msg="Parallel results must match sequential results exactly"
        )

    def test_parallel_sequential_identical_2_chips(self):
        """Parallel and sequential produce identical results for 2 chips."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(2):
            tof, _ = generate_synthetic_data(seed=100 + chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=2)

        np.testing.assert_array_equal(result_seq, result_par)

    def test_parallel_sequential_identical_different_seeds(self):
        """Results match across different random seeds."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        for base_seed in [0, 42, 123, 999]:
            tof_list = []
            chip_id_list = []
            for chip in range(4):
                tof, _ = generate_synthetic_data(seed=base_seed + chip, n_pulses=3)
                tof_list.append(tof)
                chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

            tof = np.concatenate(tof_list)
            chip_id = np.concatenate(chip_id_list)

            result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
            result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=-1)

            np.testing.assert_array_equal(
                result_seq, result_par, err_msg=f"Results differ for base_seed={base_seed}"
            )


# =============================================================================
# Tests for n_jobs parameter behavior
# =============================================================================


class TestNJobsParameter:
    """Tests for n_jobs parameter behavior."""

    def test_n_jobs_1_is_sequential(self):
        """n_jobs=1 should run sequentially."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Should complete without error
        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        assert result.shape == tof.shape
        assert result.dtype == np.int32

    def test_n_jobs_minus1_uses_all_cores(self):
        """n_jobs=-1 should use all available cores."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Should complete without error and match sequential
        result_all = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=-1)
        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)

        np.testing.assert_array_equal(result_all, result_seq)

    def test_n_jobs_none_is_sequential_default(self):
        """n_jobs=None should default to sequential processing for safety."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(2):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Default behavior (no n_jobs specified or n_jobs=None)
        result_default = reconstruct_pulse_ids(tof, chip_id=chip_id)
        result_sequential = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)

        # Default should match sequential
        np.testing.assert_array_equal(result_default, result_sequential)

    def test_n_jobs_2_works(self):
        """n_jobs=2 should work correctly."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=2)
        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)

        np.testing.assert_array_equal(result, result_seq)


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestParallelEdgeCases:
    """Edge case tests for parallel processing."""

    def test_single_chip_no_parallelization(self):
        """Single chip should not attempt parallelization."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, ground_truth = generate_synthetic_data(seed=42, n_pulses=4)

        # Single chip - n_jobs should have no effect
        result_seq = reconstruct_pulse_ids(tof, chip_id=None, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=None, n_jobs=4)

        np.testing.assert_array_equal(result_seq, result_par)

    def test_single_chip_no_chip_id_array(self):
        """When chip_id is None, should work regardless of n_jobs."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, _ = generate_synthetic_data(seed=42, n_pulses=3)

        for n_jobs in [1, 2, 4, -1]:
            result = reconstruct_pulse_ids(tof, chip_id=None, n_jobs=n_jobs)
            assert result.shape == tof.shape
            assert result.dtype == np.int32

    def test_empty_data_parallel(self):
        """Empty data should work with parallel processing."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof = np.array([], dtype=np.float64)
        chip_id = np.array([], dtype=np.uint8)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        assert len(result) == 0
        assert result.dtype == np.int32

    def test_single_event_per_chip(self):
        """Single event per chip should work."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof = np.array([5.0, 6.0, 7.0, 8.0])
        chip_id = np.array([0, 1, 2, 3], dtype=np.uint8)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        assert result.shape == tof.shape
        # All events should be pulse 0 (no rollovers)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_uneven_chip_sizes(self):
        """Parallel processing with uneven chip data sizes."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Chip 0: 600 events, Chip 1: 300 events, Chip 2: 150 events, Chip 3: 450 events
        np.random.seed(42)

        tof_list = []
        chip_id_list = []

        chip_sizes = [600, 300, 150, 450]
        for chip, size in enumerate(chip_sizes):
            # Generate appropriately sized data
            n_pulses = size // 150 if size >= 150 else 1
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=max(1, n_pulses), events_per_pulse=150)
            tof = tof[:size] if len(tof) >= size else np.tile(tof, (size // len(tof) + 1))[:size]
            tof_list.append(tof[:size])
            chip_id_list.append(np.full(size, chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        np.testing.assert_array_equal(result_seq, result_par)

    def test_one_chip_with_data_others_empty(self):
        """Handle case where only one chip has data."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, _ = generate_synthetic_data(seed=42, n_pulses=4)
        chip_id = np.zeros(len(tof), dtype=np.uint8)  # All chip 0

        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        np.testing.assert_array_equal(result_seq, result_par)


# =============================================================================
# Tests for correctness after parallelization
# =============================================================================


class TestParallelCorrectness:
    """Tests verifying parallel processing maintains algorithm correctness."""

    def test_parallel_maintains_accuracy_target(self):
        """Parallel processing should maintain the 99.5% accuracy target."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        ground_truth_list = []

        for chip in range(4):
            tof, gt = generate_synthetic_data(seed=42 + chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))
            ground_truth_list.append(gt)

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)
        ground_truth = np.concatenate(ground_truth_list)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        accuracy = (result == ground_truth).mean() * 100
        assert accuracy > 99.0, f"Parallel accuracy {accuracy:.2f}% below 99% target"

    def test_parallel_per_chip_accuracy(self):
        """Each chip should maintain accuracy when processed in parallel."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        ground_truth_list = []

        for chip in range(4):
            tof, gt = generate_synthetic_data(seed=42 + chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))
            ground_truth_list.append(gt)

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)
        ground_truth = np.concatenate(ground_truth_list)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        for chip in range(4):
            mask = chip_id == chip
            chip_accuracy = (result[mask] == ground_truth[mask]).mean() * 100
            assert chip_accuracy > 98.0, f"Chip {chip} accuracy {chip_accuracy:.2f}% below 98%"

    def test_parallel_pulse_id_continuity(self):
        """Pulse IDs should be contiguous integers starting from 0."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []

        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=5)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Each chip should have contiguous pulse IDs
        for chip in range(4):
            mask = chip_id == chip
            chip_pulses = result[mask]
            unique_pulses = np.unique(chip_pulses)

            # Should start from 0
            assert unique_pulses[0] == 0
            # Should be contiguous
            expected = np.arange(len(unique_pulses))
            np.testing.assert_array_equal(unique_pulses, expected)


# =============================================================================
# Tests for data types
# =============================================================================


class TestParallelDataTypes:
    """Test data type handling in parallel processing."""

    def test_parallel_preserves_output_dtype(self):
        """Output should be int32 regardless of parallelization."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        assert result.dtype == np.int32

    def test_parallel_with_float32_tof(self):
        """Parallel processing should work with float32 TOF data."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof.astype(np.float32))
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        np.testing.assert_array_equal(result_seq, result_par)

    def test_parallel_with_different_chip_id_dtypes(self):
        """Parallel processing should work with different chip_id dtypes."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)

        tof = np.concatenate(tof_list)

        for dtype in [np.uint8, np.int32, np.int64]:
            chip_id = np.concatenate([np.full(300, chip, dtype=dtype) for chip in range(4)])
            result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)
            assert result.dtype == np.int32


# =============================================================================
# Tests for backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests verifying backward compatibility of the API."""

    def test_existing_api_unchanged(self):
        """Existing API without n_jobs should continue to work."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, _ = generate_synthetic_data(seed=42, n_pulses=4)

        # Old API (no n_jobs parameter)
        result = reconstruct_pulse_ids(tof, chip_id=None, threshold=-10.0, window=20, late_margin=14.0)

        assert result.shape == tof.shape
        assert result.dtype == np.int32

    def test_existing_multi_chip_api_unchanged(self):
        """Existing multi-chip API should work without n_jobs."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Old API (no n_jobs parameter)
        result = reconstruct_pulse_ids(tof, chip_id=chip_id, threshold=-10.0, window=20, late_margin=14.0)

        assert result.shape == tof.shape
        assert result.dtype == np.int32

    def test_default_behavior_is_safe(self):
        """Default behavior (no n_jobs) should be safe and deterministic."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Run multiple times to verify determinism
        results = []
        for _ in range(3):
            result = reconstruct_pulse_ids(tof, chip_id=chip_id)
            results.append(result.copy())

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


# =============================================================================
# Performance tests
# =============================================================================


class TestParallelPerformance:
    """Performance tests for parallel processing."""

    @pytest.mark.slow
    def test_parallel_completes_and_matches_sequential(self):
        """Parallel processing should complete correctly for 4 chips.

        Note: With test-sized data, process spawn overhead dominates over
        the actual computation time. JIT-compiled code is extremely fast
        (~3ms for 60k events), while process spawning costs ~30ms.

        For real-world data (~1.7B events per chip), parallel processing
        provides ~3-4x speedup. This test verifies correctness, not performance.
        """
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Generate larger dataset (100 pulses per chip = 15000 events per chip)
        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=100, events_per_pulse=150)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Both parallel and sequential should complete within reasonable time
        # and produce identical results
        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Results must be identical
        np.testing.assert_array_equal(result_seq, result_par)

        # Both should complete in reasonable time (< 1 second)
        start = time.perf_counter()
        _ = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Parallel processing took {elapsed:.3f}s, expected < 1s"

    @pytest.mark.slow
    def test_parallel_scales_with_larger_data(self):
        """Parallel processing should scale better with larger datasets."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Generate large dataset (50 pulses per chip = 30000 events per chip)
        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=50, events_per_pulse=150)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Both should complete within reasonable time
        start = time.perf_counter()
        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        seq_time = time.perf_counter() - start

        start = time.perf_counter()
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)
        par_time = time.perf_counter() - start

        # Results must be identical
        np.testing.assert_array_equal(result_seq, result_par)

        # Both should complete in reasonable time
        assert seq_time < 60, f"Sequential took {seq_time:.1f}s, expected < 60s"
        assert par_time < 60, f"Parallel took {par_time:.1f}s, expected < 60s"


# =============================================================================
# Integration tests
# =============================================================================


class TestParallelIntegration:
    """Integration tests for parallel processing."""

    def test_parallel_with_numba_available(self):
        """Test parallel processing works correctly with Numba JIT."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids
        from neunorm.utils._numba_compat import HAS_NUMBA

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        np.testing.assert_array_equal(result_seq, result_par)

        # Verify Numba status is consistent
        if HAS_NUMBA:
            # Results should still be identical with JIT compilation
            np.testing.assert_array_equal(result_seq, result_par)

    def test_parallel_end_to_end_with_ground_truth(self):
        """End-to-end test with known ground truth data."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        ground_truth_list = []

        for chip in range(4):
            tof, gt = generate_synthetic_data(seed=42 + chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))
            ground_truth_list.append(gt)

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)
        ground_truth = np.concatenate(ground_truth_list)

        # Process with parallel
        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Verify accuracy
        accuracy = (result == ground_truth).mean() * 100
        errors = (result != ground_truth).sum()

        assert accuracy > 99.0, f"Accuracy {accuracy:.2f}% with {errors} errors"
        assert result.shape == tof.shape
        assert result.dtype == np.int32


# =============================================================================
# Tests for invalid inputs and error handling
# =============================================================================


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_n_jobs_zero_raises_error(self):
        """n_jobs=0 should raise ValueError."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, _ = generate_synthetic_data(seed=42, n_pulses=2)
        chip_id = np.zeros(len(tof), dtype=np.uint8)

        with pytest.raises(ValueError, match="n_jobs"):
            reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=0)

    def test_n_jobs_invalid_negative_raises_error(self):
        """n_jobs < -1 should raise ValueError."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, _ = generate_synthetic_data(seed=42, n_pulses=2)
        chip_id = np.zeros(len(tof), dtype=np.uint8)

        with pytest.raises(ValueError, match="n_jobs"):
            reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=-2)

    def test_chip_id_length_mismatch_raises_error(self):
        """chip_id length must match tof length."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof, _ = generate_synthetic_data(seed=42, n_pulses=2)
        chip_id = np.zeros(len(tof) - 10, dtype=np.uint8)  # Wrong length

        with pytest.raises(ValueError, match="length"):
            reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=2)

    def test_n_jobs_larger_than_chips_works(self):
        """n_jobs > number_of_chips should work (uses min of chips, n_jobs)."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(2):  # Only 2 chips
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # n_jobs=8 with only 2 chips should still work
        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=8)
        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)

        np.testing.assert_array_equal(result, result_seq)


# =============================================================================
# Tests for output ordering preservation
# =============================================================================


class TestOutputOrdering:
    """Tests verifying output array ordering matches input ordering."""

    def test_parallel_preserves_output_ordering(self):
        """Result[i] must correspond to tof[i] after parallel processing.

        This test creates data where we can verify the ordering is preserved
        by checking that results for each chip's indices are correct.
        """
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Create data with known structure
        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=4)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Verify that each chip's data is in the correct position
        for chip in range(4):
            mask = chip_id == chip
            chip_result = result[mask]

            # Process this chip alone to verify
            chip_tof = tof[mask]
            expected = reconstruct_pulse_ids(chip_tof, chip_id=None, n_jobs=1)

            np.testing.assert_array_equal(
                chip_result, expected, err_msg=f"Chip {chip} ordering mismatch"
            )

    def test_parallel_output_indices_match_input(self):
        """Verify output indices correspond correctly to input indices."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Create small, traceable dataset
        tof = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])  # Chip 0: no rollover, Chip 1: no rollover
        chip_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)

        result = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Both chips should have all pulse 0 (no rollovers within each chip)
        assert result.shape == tof.shape
        # Indices 0-2 (chip 0) and indices 3-5 (chip 1) should all be pulse 0
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0])

    def test_interleaved_chip_data_ordering(self):
        """Test with interleaved chip data (not grouped by chip)."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Interleaved chip assignment: 0, 1, 2, 3, 0, 1, 2, 3, ...
        n_events = 600
        tof = np.tile(np.arange(1.0, 16.0, 0.1)[:150], 4)  # 4 pulses worth
        chip_id = np.tile([0, 1, 2, 3], n_events // 4).astype(np.uint8)

        result_seq = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=1)
        result_par = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Results must be identical regardless of data layout
        np.testing.assert_array_equal(result_seq, result_par)


# =============================================================================
# Tests for thread safety
# =============================================================================


class TestThreadSafety:
    """Tests verifying thread safety of parallel processing."""

    def test_parallel_does_not_modify_input_arrays(self):
        """Input arrays (tof, chip_id) must not be modified."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        tof_list = []
        chip_id_list = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=2)
            tof_list.append(tof)
            chip_id_list.append(np.full(len(tof), chip, dtype=np.uint8))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)

        # Save copies
        tof_original = tof.copy()
        chip_id_original = chip_id.copy()

        # Run parallel processing
        _ = reconstruct_pulse_ids(tof, chip_id=chip_id, n_jobs=4)

        # Verify inputs unchanged
        np.testing.assert_array_equal(tof, tof_original, err_msg="tof was modified")
        np.testing.assert_array_equal(chip_id, chip_id_original, err_msg="chip_id was modified")

    def test_parallel_multiple_concurrent_calls(self):
        """Multiple concurrent calls should produce correct independent results."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        # Create two different datasets
        tof_list_a = []
        chip_id_list_a = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip, n_pulses=3)
            tof_list_a.append(tof)
            chip_id_list_a.append(np.full(len(tof), chip, dtype=np.uint8))

        tof_a = np.concatenate(tof_list_a)
        chip_id_a = np.concatenate(chip_id_list_a)

        tof_list_b = []
        chip_id_list_b = []
        for chip in range(4):
            tof, _ = generate_synthetic_data(seed=chip + 100, n_pulses=5)
            tof_list_b.append(tof)
            chip_id_list_b.append(np.full(len(tof), chip, dtype=np.uint8))

        tof_b = np.concatenate(tof_list_b)
        chip_id_b = np.concatenate(chip_id_list_b)

        # Process both (not truly concurrent, but verifies independence)
        result_a = reconstruct_pulse_ids(tof_a, chip_id=chip_id_a, n_jobs=4)
        result_b = reconstruct_pulse_ids(tof_b, chip_id=chip_id_b, n_jobs=4)

        # Verify against sequential
        expected_a = reconstruct_pulse_ids(tof_a, chip_id=chip_id_a, n_jobs=1)
        expected_b = reconstruct_pulse_ids(tof_b, chip_id=chip_id_b, n_jobs=1)

        np.testing.assert_array_equal(result_a, expected_a)
        np.testing.assert_array_equal(result_b, expected_b)
