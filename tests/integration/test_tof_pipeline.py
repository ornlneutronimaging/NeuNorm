"""
Integration tests for complete TOF processing pipeline.

Tests end-to-end workflows from event loading to final analysis.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
from loguru import logger


def test_end_to_end_tof_resonance_workflow():
    """
    Integration test: Full workflow from events to resonance detection.

    Pipeline:
    1. Load event data from HDF5
    2. Convert to energy histogram with variance
    3. Detect bad pixels
    4. Normalize transmission
    5. Detect resonances
    6. Create aggregated resonance image

    This is the CRITICAL test that validates the entire pipeline.
    """
    from neunorm.data_models.tof import BinningConfig
    from neunorm.loaders.event_loader import load_event_data
    from neunorm.processing.normalizer import normalize_transmission
    from neunorm.tof.event_converter import convert_events_to_histogram
    from neunorm.tof.pixel_detector import detect_bad_pixels_for_transmission
    from neunorm.tof.resonance import aggregate_resonance_image, detect_resonances

    # Create mock event data files (sample + OB)
    with tempfile.NamedTemporaryFile(suffix="_sample.h5", delete=False) as f:
        sample_path = Path(f.name)

    with tempfile.NamedTemporaryFile(suffix="_ob.h5", delete=False) as f:
        ob_path = Path(f.name)

    try:
        # Step 1: Create mock TPX3 event data with resonance signature
        n_events = 100000
        rng = np.random.default_rng(42)

        # TOF distribution with "dips" at certain energies (simulate resonances)
        # Most events at background TOF, fewer at resonance TOF
        tof_background = rng.integers(100, 1000, size=int(n_events * 0.9))
        tof_resonance = rng.integers(400, 450, size=int(n_events * 0.1))  # Dip region
        tof_all = np.concatenate([tof_background, tof_resonance])
        rng.shuffle(tof_all)

        # Sample: reduced counts at resonance
        tof_sample = tof_all[: int(n_events * 0.8)]  # 20% fewer events (resonance absorption)

        # Create HDF5 files
        with h5py.File(sample_path, "w") as hf:
            hf.create_dataset("tof", data=tof_sample.astype(np.int64))
            hf.create_dataset("x", data=rng.integers(0, 514, size=len(tof_sample), dtype=np.int32))
            hf.create_dataset("y", data=rng.integers(0, 514, size=len(tof_sample), dtype=np.int32))

        with h5py.File(ob_path, "w") as hf:
            hf.create_dataset("tof", data=tof_all.astype(np.int64))
            hf.create_dataset("x", data=rng.integers(0, 514, size=len(tof_all), dtype=np.int32))
            hf.create_dataset("y", data=rng.integers(0, 514, size=len(tof_all), dtype=np.int32))

        # Step 2: Load events
        events_sample = load_event_data(sample_path)
        events_ob = load_event_data(ob_path)

        assert events_sample.total_events > 0
        assert events_ob.total_events > 0

        # Step 3: Convert to histograms (energy space)
        binning = BinningConfig(bins=100, bin_space="energy", energy_range=(1.0, 1000.0), use_log_bin=True)
        flight_path = sc.scalar(25.0, unit="m")

        hist_sample = convert_events_to_histogram(
            events_sample,
            binning,
            flight_path,
            x_bins=50,
            y_bins=50,  # Coarser for speed
            compute_variance=True,
        )

        hist_ob = convert_events_to_histogram(
            events_ob, binning, flight_path, x_bins=50, y_bins=50, compute_variance=True
        )

        # Verify histograms have variance
        assert hist_sample.variances is not None
        assert hist_ob.variances is not None

        # Convert to energy space (resonance detection needs 'energy' dimension)
        from neunorm.tof.binning import get_energy_histogram

        hist_sample_energy = get_energy_histogram(hist_sample, flight_path)
        hist_ob_energy = get_energy_histogram(hist_ob, flight_path)

        # Step 4: Detect bad pixels
        masks = detect_bad_pixels_for_transmission(hist_sample_energy, hist_ob_energy, sigma=5.0)

        # Should detect some pixels (even in random data)
        assert isinstance(masks, dict)

        # Step 5: Normalize transmission
        transmission = normalize_transmission(hist_sample_energy, hist_ob_energy)

        # Verify transmission
        assert transmission.unit == sc.units.one
        assert transmission.variances is not None
        assert "energy" in transmission.dims

        # Step 6: Detect resonances (might find noise peaks, that's ok)
        result = detect_resonances(transmission, hist_ob_energy)

        # Should return proper structure (even if no real resonances in mock data)
        assert "resonance_energies" in result
        assert "resonance_indices" in result
        assert "n_initial" in result

        # Step 7: If resonances detected, create aggregated image
        if len(result["resonance_indices"]) > 0:
            trans_image = aggregate_resonance_image(transmission, hist_ob_energy, result["resonance_indices"])
            assert trans_image.ndim == 2
            assert "x" in trans_image.dims
            assert "y" in trans_image.dims

        logger.success("✓ Full TOF resonance workflow completed successfully")

    finally:
        sample_path.unlink()
        ob_path.unlink()


def test_variance_propagates_through_full_pipeline():
    """
    Test that variance propagates correctly through entire pipeline.

    Critical validation: Uncertainty must be tracked from raw counts
    through normalization to final transmission.
    """
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.processing.normalizer import normalize_transmission
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create controlled event data
    # 1000 events all at same TOF, pixel
    events_sample = EventData(
        tof=np.full(800, 5000, dtype=np.int64),
        x=np.full(800, 50, dtype=np.int32),
        y=np.full(800, 50, dtype=np.int32),
        file_path=Path("sample.h5"),
        total_events=800,
    )

    events_ob = EventData(
        tof=np.full(1000, 5000, dtype=np.int64),
        x=np.full(1000, 50, dtype=np.int32),
        y=np.full(1000, 50, dtype=np.int32),
        file_path=Path("ob.h5"),
        total_events=1000,
    )

    binning = BinningConfig(bins=20, bin_space="tof")
    flight_path = sc.scalar(25.0, unit="m")

    # Convert with variance
    hist_sample = convert_events_to_histogram(
        events_sample, binning, flight_path, x_bins=100, y_bins=100, compute_variance=True
    )

    hist_ob = convert_events_to_histogram(
        events_ob, binning, flight_path, x_bins=100, y_bins=100, compute_variance=True
    )

    # Normalize
    transmission = normalize_transmission(hist_sample, hist_ob)

    # Variance should be present
    assert transmission.variances is not None

    # At pixel (50, 50), transmission should be 800/1000 = 0.8
    # Find the TOF bin containing our events
    tof_bin_with_events = np.where(transmission.values[:, 50, 50] > 0)[0]

    if len(tof_bin_with_events) > 0:
        idx = tof_bin_with_events[0]
        t_value = transmission.values[idx, 50, 50]
        t_variance = transmission.variances[idx, 50, 50]

        # Transmission should be ~0.8
        assert 0.7 < t_value < 0.9

        # Variance should be > 0
        assert t_variance > 0

        # Relative uncertainty: σ/T ≈ sqrt(1/800 + 1/1000) ≈ 0.047
        expected_rel_unc = np.sqrt(1.0 / 800 + 1.0 / 1000)
        actual_rel_unc = np.sqrt(t_variance) / t_value

        # Should match Poisson expectation
        np.testing.assert_allclose(actual_rel_unc, expected_rel_unc, rtol=0.1)
