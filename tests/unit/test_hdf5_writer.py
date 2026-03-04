"""
Unit tests for the HDF5 writer module.
"""

import tempfile

import h5py
import numpy as np
import pytest
import scipp as sc


def test_write_hdf5():
    """Test the write_hdf5 function"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))  # Shape (tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values * 2  # Variance same shape as values
    data.masks["dead"] = sc.array(dims=["y", "x"], values=np.zeros((5, 5), dtype=bool))
    data.masks["hot"] = sc.array(dims=["y", "x"], values=np.ones((5, 5), dtype=bool))

    metadata = {
        "input_files": ["file1.fits", "file2.fits"],
        "processing_timestamp": "2024-06-01T12:00:00Z",
        "roi_applied": (0, 0, 5, 5),
        "num_runs_combined": 2,
        "software_version": "1.0.0",
    }

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)

        # Read back the file and check contents
        with h5py.File(f.name, "r") as f:
            # Check transmission data
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            assert f["transmission"].attrs["units"] == "counts"
            assert f["transmission"].dtype == np.float32
            # Check uncertainty data
            assert "uncertainty" in f
            np.testing.assert_allclose(f["uncertainty"][:], np.sqrt(values * 2))
            assert f["uncertainty"].dtype == np.float32
            # Check coordinates
            assert "tof_edges" in f
            np.testing.assert_equal(f["tof_edges"][:], np.linspace(1e5, 1e7, num=4))
            assert f["tof_edges"].attrs["units"] == "ns"
            assert f["tof_edges"].dtype == np.float64
            assert "x" in f
            np.testing.assert_equal(f["x"], np.arange(5))
            assert "y" in f
            np.testing.assert_equal(f["y"], np.arange(5))
            # Check masks
            assert "masks/dead" in f
            np.testing.assert_equal(f["masks/dead"], np.zeros((5, 5), dtype=bool))
            assert "masks/hot" in f
            np.testing.assert_equal(f["masks/hot"], np.ones((5, 5), dtype=bool))
            # Check metadata
            assert "metadata/input_files" in f
            np.testing.assert_equal(f["metadata/input_files"].asstr()[:], ["file1.fits", "file2.fits"])
            assert "metadata/processing_timestamp" in f
            np.testing.assert_equal(f["metadata/processing_timestamp"].asstr()[()], "2024-06-01T12:00:00Z")
            assert "metadata/roi_applied" in f
            np.testing.assert_equal(f["metadata/roi_applied"][:], [0, 0, 5, 5])
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))
            assert "metadata/software_version" in f
            np.testing.assert_equal(f["metadata/software_version"].asstr()[()], "1.0.0")


def test_write_hdf5_4d():
    """Test writing HDF5 file with 4D data (e.g. with TOF dimension)"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(2 * 3 * 5 * 5).reshape((2, 3, 5, 5))  # Shape (theta=2, tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["Z", "tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "Z": sc.linspace("Z", 0, 180, num=2),
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values * 2
    data.masks["dead"] = sc.array(dims=["y", "x"], values=np.zeros((5, 5), dtype=bool))

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data)

        # Read back the file and check contents
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            assert f["transmission"].attrs["units"] == "counts"
            assert f["transmission"].dtype == np.float32


def test_write_hdf5_no_variance():
    """Test writing HDF5 file when no variance is provided"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))  # Shape (tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data)

        # Read back the file and check that uncertainty dataset is not present
        with h5py.File(f.name, "r") as f:
            assert "uncertainty" not in f


def test_write_hdf5_no_masks():
    """Test writing HDF5 file when no masks are provided"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))  # Shape (tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values * 2  # Variance same shape as values

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data)

        # Read back the file and check that mask datasets are not present
        with h5py.File(f.name, "r") as f:
            assert "masks/dead" not in f
            assert "masks/hot" not in f


def test_write_hdf5_no_metadata():
    """Test writing HDF5 file when no metadata is provided"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))  # Shape (tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values * 2  # Variance same shape as values
    data.masks["dead"] = sc.array(dims=["y", "x"], values=np.zeros((5, 5), dtype=bool))

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data)

        # Read back the file and check that metadata group is not present
        with h5py.File(f.name, "r") as f:
            assert "metadata" not in f


def test_write_hdf5_invalid_path():
    """Test writing HDF5 file to an invalid path"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))  # Shape (tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )

    with pytest.raises(PermissionError):
        write_hdf5("/nonexistent/deep/path/file.h5", data)

    with pytest.raises(PermissionError):
        write_hdf5("/file.h5", data)


def test_write_hdf5_overwrite_existing_file():
    """Test that writing HDF5 file overwrites existing file"""
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))  # Shape (tof_edges=3, x=5, y=5)
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),  # N+1 edges for N bins
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values
    data.masks["dead"] = sc.array(dims=["y", "x"], values=np.zeros((5, 5), dtype=bool))

    metadata = {
        "input_files": ["file1.fits", "file2.fits"],
        "processing_timestamp": "2024-06-01T12:00:00Z",
        "roi_applied": (0, 0, 5, 5),
        "num_runs_combined": 2,
        "software_version": "1.0.0",
    }
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        temp_path = f.name

        # Write initial file
        write_hdf5(temp_path, data, metadata=metadata)

        # Read back and verify initial content
        with h5py.File(temp_path, "r") as f:
            assert f["transmission"].shape == (3, 5, 5)
            assert "metadata/software_version" in f

        # Overwrite with new data
        new_values = np.ones((2, 3, 3)) * 42
        new_data = sc.DataArray(
            data=sc.array(dims=["tof_edges", "x", "y"], values=new_values, unit="counts", dtype="float64"),
            coords={
                "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=3, unit="ns"),
                "y": sc.arange("y", 3, unit=None),
                "x": sc.arange("x", 3, unit=None),
            },
        )

        new_data.variances = new_values

        write_hdf5(temp_path, new_data)

        # Verify file was overwritten
        with h5py.File(temp_path, "r") as f:
            assert f["transmission"].shape == (2, 3, 3)
            np.testing.assert_allclose(f["transmission"][:], new_values)
            assert "metadata" not in f
