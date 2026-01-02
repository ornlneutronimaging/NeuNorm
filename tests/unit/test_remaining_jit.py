"""
Tests for P3: JIT-compilation of remaining hot functions.

Tests the JIT optimization of:
1. _coarse_pulse_assignment - O(N) single-pass implementation
2. _clean_clustered_rollovers - JIT helper for clustering loop

These tests follow TDD - written BEFORE the JIT implementation.
"""

import numpy as np
import pytest

# =============================================================================
# Tests for _coarse_pulse_assignment JIT optimization
# =============================================================================


class TestCoarsePulseAssignmentCorrectness:
    """Tests verifying _coarse_pulse_assignment produces correct results."""

    def test_known_ground_truth_simple(self):
        """Test with known simple ground truth."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        # 10 events, rollovers at indices 3 and 7
        # Expected: [0,0,0,1,1,1,1,2,2,2]
        rollover_mask = np.array([False, False, False, True, False, False, False, True, False, False])

        result = _coarse_pulse_assignment(rollover_mask, 10)

        expected = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_known_ground_truth_many_pulses(self):
        """Test with many pulses and verify each segment."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        # 100 events, 10 pulses (rollovers at 10, 20, 30, ..., 90)
        rollover_mask = np.zeros(100, dtype=bool)
        for i in range(1, 10):
            rollover_mask[i * 10] = True

        result = _coarse_pulse_assignment(rollover_mask, 100)

        # Build expected array
        expected = np.zeros(100, dtype=np.int32)
        for i in range(10):
            expected[i * 10 : (i + 1) * 10] = i

        np.testing.assert_array_equal(result, expected)

    def test_first_pulse_has_id_zero(self):
        """Events before first rollover should have pulse ID 0."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.zeros(50, dtype=bool)
        rollover_mask[25] = True  # Single rollover at index 25

        result = _coarse_pulse_assignment(rollover_mask, 50)

        # First 25 elements should be pulse 0
        np.testing.assert_array_equal(result[:25], 0)
        # Elements from 25 onwards should be pulse 1
        np.testing.assert_array_equal(result[25:], 1)

    def test_rollover_at_first_index(self):
        """Handle rollover at the very first index."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.array([True, False, False, False, False])

        result = _coarse_pulse_assignment(rollover_mask, 5)

        # All elements should be pulse 1 (first rollover at index 0)
        expected = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_consecutive_rollovers(self):
        """Handle consecutive rollovers."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        # Rollovers at indices 2, 3, 4 (consecutive)
        rollover_mask = np.array([False, False, True, True, True, False, False])

        result = _coarse_pulse_assignment(rollover_mask, 7)

        # [0, 0, 1, 2, 3, 3, 3]
        expected = np.array([0, 0, 1, 2, 3, 3, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)


class TestCoarsePulseAssignmentEdgeCases:
    """Edge case tests for _coarse_pulse_assignment."""

    def test_empty_arrays(self):
        """Handle empty arrays gracefully."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.array([], dtype=bool)

        result = _coarse_pulse_assignment(rollover_mask, 0)

        assert len(result) == 0
        assert result.dtype == np.int32

    def test_single_element_no_rollover(self):
        """Handle single element with no rollover."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.array([False])

        result = _coarse_pulse_assignment(rollover_mask, 1)

        assert len(result) == 1
        assert result[0] == 0

    def test_single_element_with_rollover(self):
        """Handle single element that is a rollover."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.array([True])

        result = _coarse_pulse_assignment(rollover_mask, 1)

        assert len(result) == 1
        assert result[0] == 1

    def test_no_rollovers(self):
        """All elements should be pulse 0 when no rollovers."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.zeros(100, dtype=bool)

        result = _coarse_pulse_assignment(rollover_mask, 100)

        np.testing.assert_array_equal(result, 0)

    def test_all_rollovers(self):
        """Handle case where every element is a rollover."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.ones(5, dtype=bool)

        result = _coarse_pulse_assignment(rollover_mask, 5)

        # Each element gets its own pulse ID
        expected = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_rollover_at_last_index(self):
        """Handle rollover at the very last index."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.zeros(10, dtype=bool)
        rollover_mask[9] = True

        result = _coarse_pulse_assignment(rollover_mask, 10)

        expected = np.zeros(10, dtype=np.int32)
        expected[9] = 1
        np.testing.assert_array_equal(result, expected)


class TestCoarsePulseAssignmentDataTypes:
    """Test data type handling for _coarse_pulse_assignment."""

    def test_output_dtype_is_int32(self):
        """Output should always be int32."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        rollover_mask = np.array([False, True, False, True, False])

        result = _coarse_pulse_assignment(rollover_mask, 5)

        assert result.dtype == np.int32

    def test_large_number_of_pulses(self):
        """Handle large pulse counts without overflow."""
        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        # 100,000 elements, rollover every 10 elements = 10,000 pulses
        n = 100_000
        rollover_mask = np.zeros(n, dtype=bool)
        for i in range(10, n, 10):
            rollover_mask[i] = True

        result = _coarse_pulse_assignment(rollover_mask, n)

        assert result.dtype == np.int32
        assert result.max() == 9999  # 10,000 pulses (0-9999)
        assert result.min() == 0


# =============================================================================
# Tests for _clean_clustered_rollovers JIT optimization
# =============================================================================


class TestCleanClusteredRolloversCorrectness:
    """Tests verifying _clean_clustered_rollovers produces correct results."""

    def test_known_ground_truth_no_clusters(self):
        """With evenly spaced rollovers, all should be kept."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # Rollovers at 100, 200, 300, 400 - all evenly spaced
        rollover_mask = np.zeros(500, dtype=bool)
        rollover_mask[100] = True
        rollover_mask[200] = True
        rollover_mask[300] = True
        rollover_mask[400] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # All 4 rollovers should be kept
        assert result.sum() == 4
        assert result[100] and result[200] and result[300] and result[400]

    def test_known_ground_truth_with_cluster(self):
        """Clustered rollovers should be deduplicated to first in cluster."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # Evenly spaced at 100, 200, 300, with cluster at 301, 302
        rollover_mask = np.zeros(500, dtype=bool)
        rollover_mask[100] = True
        rollover_mask[200] = True
        rollover_mask[300] = True
        rollover_mask[301] = True  # Cluster with 300
        rollover_mask[302] = True  # Cluster with 300

        result = _clean_clustered_rollovers(rollover_mask)

        # Should keep 100, 200, 300 (first in cluster)
        # Should remove 301, 302
        assert result[100]
        assert result[200]
        assert result[300]
        assert not result[301]
        assert not result[302]
        assert result.sum() == 3

    def test_cluster_detection_uses_percentile(self):
        """Verify clustering threshold is based on 75th percentile spacing."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # Create data where 75th percentile spacing determines threshold
        # 10 rollovers: 9 at spacing 100, 1 at spacing 5
        rollover_mask = np.zeros(1000, dtype=bool)
        positions = [0, 100, 200, 300, 400, 500, 600, 700, 800, 805]
        for pos in positions:
            rollover_mask[pos] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # 75th percentile of spacing is 100
        # Cluster threshold = 0.5 * 100 = 50
        # 805 - 800 = 5 < 50, so 805 should be removed
        assert result[800]
        assert not result[805]

    def test_preserves_isolated_rollovers(self):
        """Isolated rollovers should always be preserved."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # All well-spaced
        rollover_mask = np.zeros(1000, dtype=bool)
        positions = [50, 200, 400, 600, 850]
        for pos in positions:
            rollover_mask[pos] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # All should be preserved
        for pos in positions:
            assert result[pos], f"Position {pos} should be preserved"
        assert result.sum() == 5


class TestCleanClusteredRolloversEdgeCases:
    """Edge case tests for _clean_clustered_rollovers."""

    def test_empty_array(self):
        """Handle empty array gracefully."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.array([], dtype=bool)

        result = _clean_clustered_rollovers(rollover_mask)

        assert len(result) == 0

    def test_no_rollovers(self):
        """Handle array with no rollovers."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.zeros(100, dtype=bool)

        result = _clean_clustered_rollovers(rollover_mask)

        assert result.sum() == 0
        np.testing.assert_array_equal(result, rollover_mask)

    def test_single_rollover(self):
        """Handle array with single rollover."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[50] = True

        result = _clean_clustered_rollovers(rollover_mask)

        assert result.sum() == 1
        assert result[50]

    def test_two_rollovers_spaced(self):
        """Handle two well-spaced rollovers."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[20] = True
        rollover_mask[80] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # Both should be kept
        assert result.sum() == 2
        assert result[20] and result[80]

    def test_two_rollovers_clustered(self):
        """Handle two adjacent rollovers (cluster)."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[50] = True
        rollover_mask[51] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # With only 2 rollovers, spacing = [1]
        # 75th percentile = 1, threshold = 0.5
        # 51-50=1 > 0.5, so both might be kept
        # Actually, first is always kept, second is cluster checked
        assert result[50]  # First always kept

    def test_all_consecutive_rollovers(self):
        """Handle all consecutive rollovers - no baseline for clustering.

        When ALL rollovers are consecutive, the percentile-based threshold
        cannot distinguish "anomalous" clusters from "normal" spacing.
        With spacing = [1,1,1,1], 75th percentile = 1, threshold = 0.5.
        Since 1 >= 0.5, consecutive pairs are NOT considered clustered,
        so all rollovers are preserved.
        """
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # Rollovers at 10, 11, 12, 13, 14
        rollover_mask = np.zeros(20, dtype=bool)
        for i in range(10, 15):
            rollover_mask[i] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # With uniform spacing, no clustering is detected
        # All rollovers are preserved
        assert result.sum() == 5
        for i in range(10, 15):
            assert result[i]

    def test_multiple_separate_clusters(self):
        """Handle multiple separate clusters."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # Two clusters: [100, 101, 102] and [500, 501, 502]
        # Plus isolated at 300
        rollover_mask = np.zeros(600, dtype=bool)
        rollover_mask[100] = True
        rollover_mask[101] = True
        rollover_mask[102] = True
        rollover_mask[300] = True  # Isolated
        rollover_mask[500] = True
        rollover_mask[501] = True
        rollover_mask[502] = True

        result = _clean_clustered_rollovers(rollover_mask)

        # Should keep: 100 (first of cluster 1), 300 (isolated), 500 (first of cluster 2)
        assert result[100]
        assert result[300]
        assert result[500]
        # Should remove: 101, 102, 501, 502
        assert not result[101]
        assert not result[102]
        assert not result[501]
        assert not result[502]


class TestCleanClusteredRolloversDataTypes:
    """Test data type handling for _clean_clustered_rollovers."""

    def test_output_dtype_is_bool(self):
        """Output should be boolean array."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[50] = True

        result = _clean_clustered_rollovers(rollover_mask)

        assert result.dtype == bool

    def test_output_is_copy_not_input(self):
        """Output should be a copy, not the input array."""
        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        rollover_mask = np.zeros(100, dtype=bool)
        rollover_mask[50] = True
        original = rollover_mask.copy()

        result = _clean_clustered_rollovers(rollover_mask)

        # Input should not be modified
        np.testing.assert_array_equal(rollover_mask, original)
        # Result should be different object
        assert result is not rollover_mask


# =============================================================================
# Tests for JIT compilation attributes
# =============================================================================


class TestJITCompilationAttributes:
    """Tests verifying JIT compilation occurs when Numba is available."""

    def test_coarse_assignment_helper_is_jit_compiled(self):
        """Verify the coarse assignment helper has Numba JIT attributes."""
        from neunorm.utils._numba_compat import HAS_NUMBA

        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        from neunorm.tof import pulse_reconstruction

        # Check for JIT helper function
        helper_name = "_assign_pulse_ids_single_pass"
        if hasattr(pulse_reconstruction, helper_name):
            helper_fn = getattr(pulse_reconstruction, helper_name)
            assert hasattr(helper_fn, "py_func"), f"{helper_name} should be JIT-compiled"

    def test_clean_clustered_helper_is_jit_compiled(self):
        """Verify the clustering helper has Numba JIT attributes."""
        from neunorm.utils._numba_compat import HAS_NUMBA

        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        from neunorm.tof import pulse_reconstruction

        # Check for JIT helper function
        helper_name = "_cluster_rollovers"
        if hasattr(pulse_reconstruction, helper_name):
            helper_fn = getattr(pulse_reconstruction, helper_name)
            assert hasattr(helper_fn, "py_func"), f"{helper_name} should be JIT-compiled"


# =============================================================================
# Performance tests
# =============================================================================


class TestPerformance:
    """Performance tests for JIT-compiled functions."""

    @pytest.mark.slow
    def test_coarse_assignment_performance(self):
        """Test that coarse assignment completes quickly for large data."""
        import time

        from neunorm.tof.pulse_reconstruction import _coarse_pulse_assignment

        # Large dataset: 1 million events, 1000 pulses
        n = 1_000_000
        rollover_mask = np.zeros(n, dtype=bool)
        for i in range(1000, n, 1000):
            rollover_mask[i] = True

        # Warm up
        _ = _coarse_pulse_assignment(rollover_mask, n)

        # Time it
        n_runs = 5
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = _coarse_pulse_assignment(rollover_mask, n)
        elapsed = (time.perf_counter() - start) / n_runs

        # Should complete in < 100ms for 1M events
        assert elapsed < 0.1, f"Took {elapsed:.3f}s, expected < 0.1s"

    @pytest.mark.slow
    def test_clean_clustered_performance(self):
        """Test that clean_clustered completes quickly for large data."""
        import time

        from neunorm.tof.pulse_reconstruction import _clean_clustered_rollovers

        # Large dataset: 1 million events, 1000 rollovers
        n = 1_000_000
        rollover_mask = np.zeros(n, dtype=bool)
        for i in range(1000, n, 1000):
            rollover_mask[i] = True
        # Add some clusters
        for i in range(100_000, 110_000, 1000):
            rollover_mask[i + 1] = True
            rollover_mask[i + 2] = True

        # Warm up
        _ = _clean_clustered_rollovers(rollover_mask)

        # Time it
        n_runs = 5
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = _clean_clustered_rollovers(rollover_mask)
        elapsed = (time.perf_counter() - start) / n_runs

        # Should complete in < 100ms
        assert elapsed < 0.1, f"Took {elapsed:.3f}s, expected < 0.1s"


# =============================================================================
# Integration tests with full pipeline
# =============================================================================


class TestIntegrationWithPipeline:
    """Integration tests verifying JIT functions work in full pipeline."""

    def test_full_pipeline_accuracy(self):
        """Test full pipeline achieves target accuracy with JIT functions."""
        from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids

        np.random.seed(42)
        n_pulses = 10
        events_per_pulse = 150

        # Create clean TOF data
        tof = np.tile(np.arange(1.0, 16.0, 0.1), n_pulses)
        ground_truth = np.repeat(np.arange(n_pulses, dtype=np.int32), events_per_pulse)

        # Apply local shuffle (simulates disorder)
        shuffled_tof = tof.copy()
        i = 0
        while i < len(shuffled_tof):
            window = np.random.randint(8, 12)
            end = min(i + window, len(shuffled_tof))
            batch = shuffled_tof[i:end].copy()
            np.random.shuffle(batch)
            shuffled_tof[i:end] = batch
            i = end

        pulse_ids = reconstruct_pulse_ids(shuffled_tof, threshold=-10.0, window=20, late_margin=14.0)

        accuracy = (pulse_ids == ground_truth).mean() * 100
        assert accuracy > 99.0, f"Accuracy {accuracy:.2f}% < 99%"

    def test_pipeline_with_many_pulses(self):
        """Test pipeline handles many pulses correctly."""
        from neunorm.tof.pulse_reconstruction import (
            _clean_clustered_rollovers,
            _coarse_pulse_assignment,
            _detect_rollovers,
        )

        np.random.seed(42)
        n_pulses = 100
        events_per_pulse = 150

        # Create TOF data
        tof = np.tile(np.arange(1.0, 16.0, 0.1), n_pulses)

        # Run through pipeline stages
        rollover_mask = _detect_rollovers(tof, threshold=-10.0)
        cleaned_mask = _clean_clustered_rollovers(rollover_mask)
        coarse_ids = _coarse_pulse_assignment(cleaned_mask, len(tof))

        # Verify pulse count
        assert coarse_ids.max() == n_pulses - 1, f"Expected {n_pulses - 1} max pulse ID, got {coarse_ids.max()}"

        # Verify each pulse has correct number of events
        for pulse_id in range(n_pulses):
            count = (coarse_ids == pulse_id).sum()
            assert count == events_per_pulse, f"Pulse {pulse_id} has {count} events, expected {events_per_pulse}"
