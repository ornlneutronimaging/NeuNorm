"""Unit tests for pulse ID reconstruction algorithm."""

import numpy as np

from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids


def generate_synthetic_data(seed=42):
    """
    Generate synthetic TOF data with controlled disorder.

    Creates data matching TPX3 characteristics:
    - Pulse period = 16.0 ms
    - Values per pulse: 1.0 to 15.9 in 0.1 steps (150 values)
    - 4 pulses total (600 events)
    - Local shuffle with window size 8-11 events (simulates readout disorder)

    Parameters
    ----------
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (shuffled_tof, ground_truth_pulse_ids)
    """
    np.random.seed(seed)

    # Create clean data with clear rollovers
    clean_tof = np.concatenate([np.arange(1.0, 16.0, 0.1) for _ in range(4)])

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

    # Ground truth: pulse boundaries at positions 150, 300, 450
    ground_truth_pulse_ids = np.repeat([0, 1, 2, 3], 150)

    return shuffled_tof, ground_truth_pulse_ids


class TestPulseReconstruction:
    """Test suite for pulse ID reconstruction."""

    def test_single_chip_basic(self):
        """Test basic single-chip pulse reconstruction."""
        shuffled_tof, ground_truth = generate_synthetic_data(seed=42)

        pulse_ids = reconstruct_pulse_ids(shuffled_tof, chip_id=None, threshold=-10.0, window=20, late_margin=14.0)

        accuracy = (pulse_ids == ground_truth).mean() * 100
        errors = (pulse_ids != ground_truth).sum()

        assert pulse_ids.shape == shuffled_tof.shape
        assert pulse_ids.dtype == np.int32
        assert accuracy > 99.5, f"Expected accuracy >99.5%, got {accuracy:.2f}%"
        assert errors <= 3, f"Expected ≤3 errors, got {errors}"

    def test_multi_chip_synchronized(self):
        """Test multi-chip reconstruction with synchronized pulses."""
        # Create data for 4 chips, all measuring same pulses
        np.random.seed(42)

        tof_list = []
        chip_id_list = []
        ground_truth_list = []

        for chip in range(4):
            # Each chip has slightly different local disorder
            shuffled_tof, ground_truth = generate_synthetic_data(seed=42 + chip)
            tof_list.append(shuffled_tof)
            chip_id_list.append(np.full(len(shuffled_tof), chip, dtype=np.uint8))
            ground_truth_list.append(ground_truth)

        # Combine all chips
        tof = np.concatenate(tof_list)
        chip_id = np.concatenate(chip_id_list)
        ground_truth = np.concatenate(ground_truth_list)

        # Reconstruct pulse IDs
        pulse_ids = reconstruct_pulse_ids(tof, chip_id=chip_id, threshold=-10.0, window=20, late_margin=14.0)

        # Check each chip separately
        # Note: Different random seeds produce different disorder patterns
        # Some seeds are harder than others (98-99% range is acceptable per chip)
        for chip in range(4):
            chip_mask = chip_id == chip
            chip_accuracy = (pulse_ids[chip_mask] == ground_truth[chip_mask]).mean() * 100
            chip_errors = (pulse_ids[chip_mask] != ground_truth[chip_mask]).sum()

            # Each chip should achieve >98% (within 12 errors per 600 events)
            assert chip_accuracy > 98.0, (
                f"Chip {chip} accuracy {chip_accuracy:.2f}% < 98%, {chip_errors} errors in 600 events"
            )

        # Overall accuracy across all chips should be >99%
        overall_accuracy = (pulse_ids == ground_truth).mean() * 100
        total_errors = (pulse_ids != ground_truth).sum()
        assert overall_accuracy > 99.0, (
            f"Overall accuracy {overall_accuracy:.2f}% < 99%, {total_errors} errors in {len(tof)} events"
        )

    def test_no_rollovers(self):
        """Test with data containing no rollovers (single pulse)."""
        tof = np.linspace(1.0, 15.0, 100)
        pulse_ids = reconstruct_pulse_ids(tof)

        # All events should be in pulse 0
        assert np.all(pulse_ids == 0)
        assert pulse_ids.shape == tof.shape

    def test_single_rollover(self):
        """Test with data containing single rollover (2 pulses)."""
        tof = np.concatenate([np.linspace(1.0, 15.0, 50), np.linspace(1.0, 15.0, 50)])

        pulse_ids = reconstruct_pulse_ids(tof, threshold=-10.0)

        # Should detect the rollover and assign 2 pulses
        assert pulse_ids.min() == 0
        assert pulse_ids.max() == 1
        # First half should be pulse 0, second half pulse 1 (approximately)
        assert np.sum(pulse_ids == 0) > 40
        assert np.sum(pulse_ids == 1) > 40

    def test_return_types(self):
        """Test return types and shapes."""
        tof = np.random.uniform(0, 16, 1000)
        pulse_ids = reconstruct_pulse_ids(tof)

        assert isinstance(pulse_ids, np.ndarray)
        assert pulse_ids.dtype == np.int32
        assert pulse_ids.shape == tof.shape
        assert pulse_ids.min() >= 0

    def test_multi_chip_independence(self):
        """Test that chips are processed independently."""
        # Create 2 chips with different pulse structures
        tof_chip0 = np.concatenate([np.linspace(1, 15, 100) for _ in range(3)])  # 3 pulses
        tof_chip1 = np.concatenate([np.linspace(1, 15, 100) for _ in range(5)])  # 5 pulses

        tof = np.concatenate([tof_chip0, tof_chip1])
        chip_id = np.concatenate([np.zeros(300, dtype=np.uint8), np.ones(500, dtype=np.uint8)])

        pulse_ids = reconstruct_pulse_ids(tof, chip_id=chip_id, threshold=-10.0)

        # Check chip 0: should have 3 pulses (0, 1, 2)
        chip0_pulses = pulse_ids[chip_id == 0]
        assert chip0_pulses.max() <= 2

        # Check chip 1: should have 5 pulses (0, 1, 2, 3, 4)
        chip1_pulses = pulse_ids[chip_id == 1]
        assert chip1_pulses.max() <= 4

    def test_accuracy_target(self):
        """Test that algorithm meets accuracy target on synthetic data."""
        shuffled_tof, ground_truth = generate_synthetic_data(seed=42)

        pulse_ids = reconstruct_pulse_ids(shuffled_tof, threshold=-10.0, window=20, late_margin=14.0)

        accuracy = (pulse_ids == ground_truth).mean() * 100

        # Target from implementation_log.md: 99.67% achieved
        assert accuracy > 99.5, (
            f"Algorithm accuracy {accuracy:.2f}% below target 99.5%. "
            f"This may indicate regression in algorithm implementation."
        )
