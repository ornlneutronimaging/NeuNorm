"""
Unit tests for the run combiner module.
"""

import numpy as np
import pytest
import scipp as sc


def test_combine_runs():
    """
    Test combining two runs with the same shape and dimensions but different data, masks and metadata.
    """
    from neunorm.processing.run_combiner import combine_runs

    # Create two runs with the same shape and dimensions but different data and metadata
    run_data = np.arange(10 * 5 * 5).reshape((10, 5, 5))  # Shape (tof_edges=10, x=5, y=5)
    run = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=run_data, unit="counts", dtype="float64"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},
    )
    run.variances = run.values.copy()

    dead_pixel_mask = np.zeros((5, 5), dtype=bool)
    dead_pixel_mask[2, 2] = True  # Mark one pixel as dead
    run.masks["dead_pixel_mask"] = sc.array(dims=["x", "y"], values=dead_pixel_mask, dtype=bool)

    run.coords["p_charge"] = sc.scalar(value=10, unit="C")
    run.coords.set_aligned("p_charge", False)
    run.coords["acquisition_time"] = sc.array(dims=["tof_edges"], values=np.linspace(2, 11, num=10), unit="s")
    run.coords.set_aligned("acquisition_time", False)
    run.coords["other_metadata"] = sc.scalar(value=42)
    run.coords.set_aligned("other_metadata", False)

    run2_data = np.arange(10 * 5 * 5).reshape((10, 5, 5)) * 4
    run2 = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=run2_data, unit="counts", dtype="float64"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},
    )
    run2.variances = run2.values.copy()

    dead_pixel_mask2 = np.zeros((5, 5), dtype=bool)
    dead_pixel_mask2[3, 3] = True  # Mark one pixel as dead
    run2.masks["dead_pixel_mask"] = sc.array(dims=["x", "y"], values=dead_pixel_mask2, dtype=bool)

    run2.coords["p_charge"] = sc.scalar(value=40, unit="C")
    run2.coords.set_aligned("p_charge", False)
    run2.coords["acquisition_time"] = sc.array(dims=["tof_edges"], values=np.linspace(1, 10, num=10), unit="s")
    run2.coords.set_aligned("acquisition_time", False)
    run2.coords["other_metadata"] = sc.scalar(value=13)
    run2.coords.set_aligned("other_metadata", False)

    # Combine the two samples
    combined = combine_runs([run, run2])

    # Should still be in counts
    assert combined.unit == sc.units.counts

    # Check coordinates, should be the same as the input samples
    assert combined.dims == ("tof_edges", "x", "y")
    assert "tof_edges" in combined.coords
    assert combined.coords["tof_edges"].values.shape == (11,)
    assert combined.coords["tof_edges"].unit == "us"
    np.testing.assert_equal(combined.coords["tof_edges"].values, np.linspace(1, 100, num=11))

    # Check shape
    assert combined.data.shape == (10, 5, 5)
    assert combined.variances.shape == (10, 5, 5)

    # Values should be sum of the two samples
    expected_values = np.arange(10 * 5 * 5).reshape((10, 5, 5)) * 5
    np.testing.assert_allclose(combined.values, expected_values)
    # Variance should be sum of the two samples
    np.testing.assert_allclose(combined.variances, expected_values)

    # Check mask. Both dead pixels should be masked in the combined result.
    assert "dead_pixel_mask" in combined.masks
    expected_mask = np.zeros((5, 5), dtype=bool)
    expected_mask[2, 2] = True
    expected_mask[3, 3] = True
    np.testing.assert_equal(combined.masks["dead_pixel_mask"].values, expected_mask)

    # p_charge should be sum of the two samples
    np.testing.assert_allclose(combined.coords["p_charge"].value, 10 + 40)
    assert combined.coords["p_charge"].unit == "C"
    # Acquisition time should be sum of the two samples
    np.testing.assert_allclose(
        combined.coords["acquisition_time"].values, np.linspace(2, 11, num=10) + np.linspace(1, 10, num=10)
    )
    assert combined.coords["acquisition_time"].unit == "s"
    # other_metadata should be from the first sample since it's not in the metadata_keys_to_sum list
    np.testing.assert_allclose(combined.coords["other_metadata"].value, 42)


def test_combine_runs_different_shapes():
    """Test that combining runs with different shapes raises an error."""
    from neunorm.processing.run_combiner import combine_runs

    # Create two runs with different shapes
    run1 = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=np.zeros((10, 5, 5)), unit="counts"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},
    )
    run1.variances = run1.values.copy()

    run2 = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=np.zeros((8, 5, 5)), unit="counts"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 80, num=9, unit="us")},
    )
    run2.variances = run2.values.copy()

    with pytest.raises(ValueError):
        combine_runs([run1, run2])


def test_combine_runs_different_dims():
    """Test that combining runs with different dimensions raises an error."""
    from neunorm.processing.run_combiner import combine_runs

    # Create two runs with different dimensions
    run1 = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=np.zeros((10, 5, 5)), unit="counts"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},
    )
    run1.variances = run1.values.copy()

    run2 = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.zeros((10, 5, 5)), unit="counts"),
        coords={"N_image": sc.arange("N_image", 10), "x": sc.arange("x", 5), "y": sc.arange("y", 5)},
    )
    run2.variances = run2.values.copy()

    with pytest.raises(ValueError):
        combine_runs([run1, run2])


def test_combine_runs_missing():
    """Test that combining an empty list of runs raises an error."""
    from neunorm.processing.run_combiner import combine_runs

    with pytest.raises(ValueError):
        combine_runs([])


def test_combine_runs_single_run():
    """Test that combining a single run returns the same run."""
    from neunorm.processing.run_combiner import combine_runs

    run = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=np.zeros((10, 5, 5)), unit="counts"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},
    )
    run.variances = run.values.copy()

    combined = combine_runs([run])
    assert combined is run


def test_combine_runs_missing_metadata_key():
    """Test that combining runs with a missing metadata key raises an error."""
    from neunorm.processing.run_combiner import combine_runs

    run1 = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=np.zeros((10, 5, 5)), unit="counts"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},
    )
    run1.coords["p_charge"] = sc.scalar(value=10, unit="C")
    run1.coords.set_aligned("p_charge", False)

    run2 = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=np.zeros((10, 5, 5)), unit="counts"),
        coords={"tof_edges": sc.linspace("tof_edges", 1, 100, num=11, unit="us")},  # Missing p_charge
    )

    with pytest.raises(ValueError, match="Metadata key 'p_charge' not found in all runs for summation"):
        combine_runs([run1, run2], metadata_keys_to_sum=["p_charge"])
