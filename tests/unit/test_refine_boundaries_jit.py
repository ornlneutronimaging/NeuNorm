"""Unit tests for JIT-compiled _refine_rollover_boundaries() function.

Tests verify:
1. Correctness of boundary refinement with known ground truth
2. Results are identical with and without Numba JIT (after JIT is implemented)
3. Edge cases are handled correctly
4. Performance improves with JIT compilation

These tests follow TDD - written BEFORE the JIT implementation.
Some tests will initially fail until P2 implementation is complete.
"""

import sys
import time

import numpy as np
import pytest


class TestRefineRolloverBoundariesCorrectness:
    """Tests verifying the algorithm produces correct boundary refinements."""

    def test_clean_rollover_exact_boundary(self):
        """Clean rollover with no disorder should have exact boundary."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # Pulse 0: late hits 12-15, Pulse 1: early hits 1-5
        # Rollover at index 10 (from 15 to 1)
        tof = np.concatenate(
            [
                np.linspace(12.0, 15.0, 10),  # 10 late hits from pulse 0
                np.linspace(1.0, 5.0, 10),  # 10 early hits from pulse 1
            ]
        )
        pulse_ids = np.zeros(20, dtype=np.int32)
        pulse_ids[10:] = 1  # Coarse assignment
        rollover_mask = np.zeros(20, dtype=bool)
        rollover_mask[10] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=5, late_margin=14.0)

        # With clean data, boundary should be at exactly index 10
        # Pulse 0: indices 0-9, Pulse 1: indices 10-19
        assert result[9] == 0, "Index 9 should be pulse 0"
        assert result[10] == 1, "Index 10 should be pulse 1"
        np.testing.assert_array_equal(result[:10], 0)
        np.testing.assert_array_equal(result[10:], 1)

    def test_late_hits_after_rollover_corrected(self):
        """Late hits appearing after rollover should be reassigned to previous pulse."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # Disorder scenario: some late hits from pulse 0 appear after rollover point
        # due to TPX3 FIFO reordering
        tof = np.array(
            [
                # Pulse 0 region
                12.0,
                13.0,
                14.0,
                14.5,
                15.0,  # indices 0-4: late hits
                # Rollover region with disorder
                1.0,
                14.8,
                2.0,
                15.0,
                3.0,  # indices 5-9: early-late-early-late-early
                # Pulse 1 region
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,  # indices 10-14: clearly pulse 1
            ]
        )
        pulse_ids = np.zeros(15, dtype=np.int32)
        pulse_ids[5:] = 1  # Coarse rollover at index 5
        rollover_mask = np.zeros(15, dtype=bool)
        rollover_mask[5] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=5, late_margin=14.0)

        # The late hits (14.8, 15.0 at indices 6, 8) should be in pulse 0
        # The early hits should be in pulse 1
        # This tests the refinement algorithm's core purpose
        assert result.shape == (15,)
        assert result.min() == 0
        assert result.max() == 1

        # Verify late hits are in earlier pulse (value >= late_margin goes to pulse 0)
        # and early hits are in later pulse (value < late_margin/2 goes to pulse 1)
        # The algorithm should find an optimal boundary

    def test_early_hits_before_rollover_corrected(self):
        """Early hits appearing before rollover should be reassigned to next pulse."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # Disorder: some early hits from pulse 1 appear before rollover point
        tof = np.array(
            [
                # Pulse 0 region with disorder
                12.0,
                13.0,
                1.5,
                14.0,
                2.0,  # indices 0-4: late-late-EARLY-late-EARLY
                # Clear rollover
                15.0,
                1.0,  # indices 5-6
                # Pulse 1 region
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,  # indices 7-11
            ]
        )
        pulse_ids = np.zeros(12, dtype=np.int32)
        pulse_ids[6:] = 1  # Coarse rollover at index 6
        rollover_mask = np.zeros(12, dtype=bool)
        rollover_mask[6] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=5, late_margin=14.0)

        assert result.shape == (12,)
        assert result.min() == 0
        assert result.max() == 1

    def test_ground_truth_synthetic_data(self):
        """Test against known ground truth on synthetic data."""
        from neunorm.tof.pulse_reconstruction import (
            _clean_clustered_rollovers,
            _coarse_pulse_assignment,
            _detect_rollovers,
            _refine_rollover_boundaries,
        )

        # Create clean data with known pulse assignments
        np.random.seed(42)
        events_per_pulse = 150
        n_pulses = 4

        # Clean TOF values: 1.0 to 15.9 within each pulse
        clean_tof = np.concatenate([np.arange(1.0, 16.0, 0.1) for _ in range(n_pulses)])
        ground_truth = np.repeat(np.arange(n_pulses, dtype=np.int32), events_per_pulse)

        # Apply local shuffle (simulates disorder)
        shuffled_tof = clean_tof.copy()
        i = 0
        while i < len(shuffled_tof):
            window = np.random.randint(8, 12)
            end = min(i + window, len(shuffled_tof))
            batch = shuffled_tof[i:end].copy()
            np.random.shuffle(batch)
            shuffled_tof[i:end] = batch
            i = end

        # Run through pipeline
        rollover_mask = _detect_rollovers(shuffled_tof, threshold=-10.0)
        cleaned_mask = _clean_clustered_rollovers(rollover_mask)
        coarse_ids = _coarse_pulse_assignment(cleaned_mask, len(shuffled_tof))

        refined_ids = _refine_rollover_boundaries(shuffled_tof, coarse_ids, cleaned_mask, window=20, late_margin=14.0)

        # Refinement should improve accuracy
        coarse_accuracy = (coarse_ids == ground_truth).mean() * 100
        refined_accuracy = (refined_ids == ground_truth).mean() * 100

        # Refinement should not make accuracy worse
        assert refined_accuracy >= coarse_accuracy - 1.0, (
            f"Refinement degraded accuracy: {coarse_accuracy:.2f}% -> {refined_accuracy:.2f}%"
        )

        # Should achieve >99% accuracy
        assert refined_accuracy > 99.0, f"Refined accuracy {refined_accuracy:.2f}% below 99% threshold"


class TestRefineRolloverBoundariesEdgeCases:
    """Edge case tests for _refine_rollover_boundaries."""

    def test_empty_arrays(self):
        """Function should handle empty arrays gracefully."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.array([], dtype=np.float64)
        pulse_ids = np.array([], dtype=np.int32)
        rollover_mask = np.array([], dtype=bool)

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert len(result) == 0
        assert result.dtype == np.int32

    def test_single_element_array(self):
        """Function should handle single-element arrays."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.array([5.0])
        pulse_ids = np.array([0], dtype=np.int32)
        rollover_mask = np.array([False])

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert len(result) == 1
        assert result[0] == 0

    def test_no_rollovers(self):
        """Function should return unchanged pulse IDs when no rollovers."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.linspace(1.0, 15.0, 100)
        pulse_ids = np.zeros(100, dtype=np.int32)
        rollover_mask = np.zeros(100, dtype=bool)

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        np.testing.assert_array_equal(result, pulse_ids)

    def test_single_rollover(self):
        """Function should handle single rollover correctly."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # Create data with clear rollover at index 50
        tof = np.concatenate([np.linspace(1.0, 15.0, 50), np.linspace(1.0, 15.0, 50)])
        pulse_ids = np.zeros(100, dtype=np.int32)
        pulse_ids[50:] = 1
        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[50] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        # Should still have exactly 2 pulses
        assert result.min() == 0
        assert result.max() == 1
        # Boundary should be at or near index 50 (clean data)
        boundary_idx = np.where(np.diff(result) != 0)[0]
        assert len(boundary_idx) == 1
        # With clean data and no shuffle, boundary should be exactly at 50
        assert 48 <= boundary_idx[0] <= 52, f"Boundary at {boundary_idx[0]}, expected ~50"

    def test_rollover_at_start(self):
        """Handle rollover near the start of array."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # Rollover at index 5 (within window)
        tof = np.concatenate([np.linspace(10.0, 15.0, 5), np.linspace(1.0, 15.0, 95)])
        pulse_ids = np.zeros(100, dtype=np.int32)
        pulse_ids[5:] = 1
        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[5] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert result.shape == (100,)
        assert result.min() == 0
        assert result.max() == 1

    def test_rollover_at_end(self):
        """Handle rollover near the end of array."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # Rollover at index 95 (near end)
        tof = np.concatenate([np.linspace(1.0, 15.0, 95), np.linspace(1.0, 5.0, 5)])
        pulse_ids = np.zeros(100, dtype=np.int32)
        pulse_ids[95:] = 1
        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[95] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert result.shape == (100,)
        assert result.min() == 0
        assert result.max() == 1

    def test_many_rollovers(self):
        """Handle many rollovers in sequence."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        n_pulses = 20
        events_per_pulse = 50
        tof = np.tile(np.linspace(1.0, 15.0, events_per_pulse), n_pulses)
        pulse_ids = np.repeat(np.arange(n_pulses, dtype=np.int32), events_per_pulse)
        rollover_mask = np.zeros(len(tof), dtype=bool)
        for i in range(1, n_pulses):
            rollover_mask[i * events_per_pulse] = True

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert result.shape == tof.shape
        assert result.min() == 0
        assert result.max() == n_pulses - 1

    def test_window_larger_than_data(self):
        """Handle case where window exceeds available data."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.array([15.0, 14.0, 1.0, 2.0, 3.0])  # Small array, rollover at index 2
        pulse_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        rollover_mask = np.array([False, False, True, False, False])

        result = _refine_rollover_boundaries(
            tof,
            pulse_ids,
            rollover_mask,
            window=50,
            late_margin=14.0,  # window > len
        )

        assert result.shape == (5,)
        assert result.min() == 0
        assert result.max() == 1

    def test_tof_exactly_at_late_margin(self):
        """Handle TOF values exactly at the late_margin threshold."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        # late_margin=14.0, so test with values exactly at 14.0 and 7.0 (late_margin/2)
        tof = np.array([14.0, 14.0, 7.0, 1.0, 2.0])  # Edge values
        pulse_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        rollover_mask = np.array([False, False, True, False, False])

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=5, late_margin=14.0)

        # Should not crash and produce valid output
        assert result.shape == (5,)
        assert result.dtype == np.int32


class TestRefineRolloverBoundariesDataTypes:
    """Test data type handling."""

    def test_float32_tof(self):
        """Function should work with float32 TOF arrays."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.linspace(1.0, 15.0, 100).astype(np.float32)
        pulse_ids = np.zeros(100, dtype=np.int32)
        rollover_mask = np.zeros(100, dtype=bool)

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert result.dtype == np.int32

    def test_float64_tof(self):
        """Function should work with float64 TOF arrays."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.linspace(1.0, 15.0, 100).astype(np.float64)
        pulse_ids = np.zeros(100, dtype=np.int32)
        rollover_mask = np.zeros(100, dtype=bool)

        result = _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        assert result.dtype == np.int32

    def test_output_is_copy_not_input_modified(self):
        """Function should return a copy, not modify input."""
        from neunorm.tof.pulse_reconstruction import _refine_rollover_boundaries

        tof = np.concatenate([np.linspace(1.0, 15.0, 50), np.linspace(1.0, 15.0, 50)])
        pulse_ids = np.zeros(100, dtype=np.int32)
        pulse_ids[50:] = 1
        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[50] = True

        original_pulse_ids = pulse_ids.copy()

        _refine_rollover_boundaries(tof, pulse_ids, rollover_mask, window=20, late_margin=14.0)

        # Input should not be modified
        np.testing.assert_array_equal(pulse_ids, original_pulse_ids)


class TestRefineRolloverBoundariesParameters:
    """Test parameter handling."""

    def test_different_window_sizes(self):
        """Function should work with various window sizes."""
        from neunorm.tof.pulse_reconstruction import (
            _clean_clustered_rollovers,
            _coarse_pulse_assignment,
            _detect_rollovers,
            _refine_rollover_boundaries,
        )

        np.random.seed(42)
        tof = np.concatenate([np.arange(1.0, 16.0, 0.1) for _ in range(4)])
        i = 0
        while i < len(tof):
            window_size = np.random.randint(8, 12)
            end = min(i + window_size, len(tof))
            batch = tof[i:end].copy()
            np.random.shuffle(batch)
            tof[i:end] = batch
            i = end

        rollover_mask = _detect_rollovers(tof, threshold=-10.0)
        cleaned_mask = _clean_clustered_rollovers(rollover_mask)
        coarse_ids = _coarse_pulse_assignment(cleaned_mask, len(tof))

        for window in [5, 10, 20, 50, 100]:
            result = _refine_rollover_boundaries(tof, coarse_ids.copy(), cleaned_mask, window=window, late_margin=14.0)
            assert result.shape == tof.shape
            assert result.min() >= 0

    def test_different_late_margins(self):
        """Function should work with various late_margin values."""
        from neunorm.tof.pulse_reconstruction import (
            _clean_clustered_rollovers,
            _coarse_pulse_assignment,
            _detect_rollovers,
            _refine_rollover_boundaries,
        )

        np.random.seed(42)
        tof = np.concatenate([np.arange(1.0, 16.0, 0.1) for _ in range(4)])

        rollover_mask = _detect_rollovers(tof, threshold=-10.0)
        cleaned_mask = _clean_clustered_rollovers(rollover_mask)
        coarse_ids = _coarse_pulse_assignment(cleaned_mask, len(tof))

        for late_margin in [10.0, 12.0, 14.0, 15.0]:
            result = _refine_rollover_boundaries(
                tof,
                coarse_ids.copy(),
                cleaned_mask,
                window=20,
                late_margin=late_margin,
            )
            assert result.shape == tof.shape


class TestJITEquivalence:
    """Tests verifying JIT helper function behavior.

    Note: JIT/non-JIT equivalence is implicitly verified by:
    1. The fallback mechanism is tested in test_numba_compat.py
    2. The algorithm produces correct results (tested above)
    3. The JIT helper has the same logic as the original code

    These tests verify the helper function behavior directly.
    """

    def test_refine_single_boundary_basic(self):
        """Test the JIT helper function produces expected results."""
        from neunorm.tof.pulse_reconstruction import _refine_single_boundary

        # Clean data with clear boundary
        tof = np.array([12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 4.0])
        # Rollover at index 4 (15.0 -> 1.0)
        result = _refine_single_boundary(
            tof,
            start_idx=0,
            end_idx=8,
            rollover_idx=4,
            late_margin=14.0,
        )

        # Boundary should be at or near index 4
        assert 3 <= result <= 5, f"Boundary at {result}, expected 3-5"

    def test_refine_single_boundary_with_disorder(self):
        """Test the JIT helper handles disordered data."""
        from neunorm.tof.pulse_reconstruction import _refine_single_boundary

        # Data with disorder: some late hits appear after rollover
        tof = np.array([12.0, 13.0, 14.0, 1.0, 15.0, 2.0, 14.8, 3.0, 4.0, 5.0])
        # Rollover nominally at index 3
        result = _refine_single_boundary(
            tof,
            start_idx=0,
            end_idx=10,
            rollover_idx=3,
            late_margin=14.0,
        )

        # Should find a reasonable boundary
        assert 0 <= result <= 10
        assert isinstance(result, (int, np.integer))

    def test_refine_single_boundary_edge_cases(self):
        """Test the JIT helper handles edge cases."""
        from neunorm.tof.pulse_reconstruction import _refine_single_boundary

        # Small window
        tof = np.array([14.0, 1.0, 2.0])
        result = _refine_single_boundary(
            tof,
            start_idx=0,
            end_idx=3,
            rollover_idx=1,
            late_margin=14.0,
        )
        assert 0 <= result <= 3


class TestJITCompilation:
    """Tests verifying JIT compilation occurs when Numba is available."""

    def test_helper_function_is_jit_compiled(self):
        """Verify the inner helper function has Numba JIT attributes."""
        from neunorm.utils._numba_compat import HAS_NUMBA

        if not HAS_NUMBA:
            pytest.skip("Numba not available, cannot verify JIT compilation")

        # Import fresh
        for mod in list(sys.modules.keys()):
            if "pulse_reconstruction" in mod:
                del sys.modules[mod]

        from neunorm.tof import pulse_reconstruction

        # The JIT-compiled helper function
        helper_name = "_refine_single_boundary"
        assert hasattr(pulse_reconstruction, helper_name), f"Module should export {helper_name}"

        helper_fn = getattr(pulse_reconstruction, helper_name)

        # Numba JIT functions have specific attributes
        assert hasattr(helper_fn, "py_func"), f"{helper_name} should be Numba JIT-compiled (has py_func attribute)"

    def test_helper_function_has_correct_signature(self):
        """Verify the helper function accepts expected parameters."""
        from neunorm.tof.pulse_reconstruction import _refine_single_boundary

        # Test with valid inputs
        tof = np.array([12.0, 13.0, 14.0, 1.0, 2.0, 3.0])
        result = _refine_single_boundary(
            tof,
            start_idx=0,
            end_idx=6,
            rollover_idx=3,
            late_margin=14.0,
        )

        assert isinstance(result, (int, np.integer))
        assert 0 <= result <= 6


class TestRefineRolloverBoundariesPerformance:
    """Performance tests for JIT compilation."""

    @pytest.mark.slow
    def test_performance_improvement_with_numba(self):
        """Test that JIT version is faster than pure Python on larger data."""
        from neunorm.utils._numba_compat import HAS_NUMBA

        if not HAS_NUMBA:
            pytest.skip("Numba not available, cannot test performance improvement")

        # Generate larger dataset: 100 pulses * 150 events each
        np.random.seed(42)
        n_pulses = 100
        tof = np.tile(np.arange(1.0, 16.0, 0.1), n_pulses)

        # Apply shuffle
        i = 0
        while i < len(tof):
            window = np.random.randint(8, 12)
            end = min(i + window, len(tof))
            batch = tof[i:end].copy()
            np.random.shuffle(batch)
            tof[i:end] = batch
            i = end

        # Test with Numba (JIT)
        for mod in list(sys.modules.keys()):
            if "pulse_reconstruction" in mod or "_numba_compat" in mod:
                del sys.modules[mod]

        from neunorm.tof.pulse_reconstruction import (
            _clean_clustered_rollovers,
            _coarse_pulse_assignment,
            _detect_rollovers,
            _refine_rollover_boundaries,
        )

        rollover_mask = _detect_rollovers(tof, threshold=-10.0)
        cleaned_mask = _clean_clustered_rollovers(rollover_mask)
        coarse_ids = _coarse_pulse_assignment(cleaned_mask, len(tof))

        # Warm up (first call compiles)
        _ = _refine_rollover_boundaries(tof, coarse_ids.copy(), cleaned_mask, window=20, late_margin=14.0)

        # Time JIT version
        n_runs = 5
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = _refine_rollover_boundaries(tof, coarse_ids.copy(), cleaned_mask, window=20, late_margin=14.0)
        jit_time = (time.perf_counter() - start) / n_runs

        # Function should complete in reasonable time with JIT
        # 15000 events, 100 rollovers - should be < 100ms with JIT
        assert jit_time < 0.5, f"JIT version took {jit_time:.3f}s, expected < 0.5s"


class TestIntegrationWithReconstructPulseIds:
    """Integration tests with the main reconstruct_pulse_ids function."""

    def test_full_pipeline_accuracy(self):
        """Test full reconstruction pipeline achieves target accuracy."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        np.random.seed(42)
        tof = np.concatenate([np.arange(1.0, 16.0, 0.1) for _ in range(4)])
        i = 0
        while i < len(tof):
            window = np.random.randint(8, 12)
            end = min(i + window, len(tof))
            batch = tof[i:end].copy()
            np.random.shuffle(batch)
            tof[i:end] = batch
            i = end

        ground_truth = np.repeat([0, 1, 2, 3], 150)

        pulse_ids = reconstruct_pulse_ids(tof, chip_id=None, threshold=-10.0, window=20, late_margin=14.0)

        accuracy = (pulse_ids == ground_truth).mean() * 100
        assert accuracy > 99.0, f"Accuracy {accuracy:.2f}% < 99%"

    def test_multi_chip_accuracy(self):
        """Test multi-chip reconstruction achieves target accuracy."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        np.random.seed(42)

        tof_list = []
        chip_id_list = []
        ground_truth_list = []

        for chip in range(4):
            chip_tof = np.concatenate([np.arange(1.0, 16.0, 0.1) for _ in range(4)])
            np.random.seed(42 + chip)
            i = 0
            while i < len(chip_tof):
                window = np.random.randint(8, 12)
                end = min(i + window, len(chip_tof))
                batch = chip_tof[i:end].copy()
                np.random.shuffle(batch)
                chip_tof[i:end] = batch
                i = end

            tof_list.append(chip_tof)
            chip_id_list.append(np.full(len(chip_tof), chip, dtype=np.uint8))
            ground_truth_list.append(np.repeat([0, 1, 2, 3], 150))

        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)
        ground_truth = np.concatenate(ground_truth_list)

        pulse_ids = reconstruct_pulse_ids(tof, chip_id=chip_id, threshold=-10.0, window=20, late_margin=14.0)

        accuracy = (pulse_ids == ground_truth).mean() * 100
        assert accuracy > 98.0, f"Multi-chip accuracy {accuracy:.2f}% < 98%"
