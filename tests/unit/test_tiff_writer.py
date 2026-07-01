"""Unit tests for TIFF writer DataGroup construction and SciTiff integration call."""

import json
import tempfile

import numpy as np
import pytest
import scipp as sc
from scitiff.io import load_scitiff


def test_write_tiff_stack_2d():
    """Test writing a 2D transmission DataArray with metadata and mask using scitiff."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(25, dtype=np.float64).reshape((5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="counts", dtype="float64"))
    transmission.variances = values

    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 3] = True
    transmission.masks["dead"] = sc.array(dims=["y", "x"], values=mask, dtype=bool)

    metadata = {
        "input_files": ["file1.fits", "file2.fits"],
        "processing_timestamp": "2024-06-01T12:00:00Z",
        "roi_applied": (0, 0, 5, 5),
        "num_runs_combined": 2,
        "software_version": "1.0.0",
        "boolean_flag": True,
    }

    daqmetadata = {
        "facility": "HFIR",
        "instrument": "MARS",
        "detector_type": "MARANA-4BV11",
        "source_type": "neutron",
    }

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
        # Save
        write_tiff_stack(f.name, transmission, metadata=metadata, daqmetadata=daqmetadata)
        # Load back the file to verify contents
        dg = load_scitiff(f.name)

    assert isinstance(dg, sc.DataGroup)
    assert "image" in dg
    assert "daq" in dg
    assert "extra" in dg

    # Check image data and metadata
    image = dg["image"]
    assert image.dtype == sc.DType.float32
    assert image.dims == ("y", "x")
    assert image.values.shape == (5, 5)
    np.testing.assert_allclose(image.values, values, rtol=1e-6)
    np.testing.assert_allclose(image.variances, values, rtol=1e-6)
    assert "scitiff-mask" in image.masks
    assert image.masks["scitiff-mask"].shape == (5, 5)
    np.testing.assert_array_equal(image.masks["scitiff-mask"].values, mask)

    # Check DAQ metadata
    daq = dg["daq"]  # this is type scitiff.DAQMetadata
    assert daq.facility == "HFIR"
    assert daq.instrument == "MARS"
    assert daq.detector_type == "MARANA-4BV11"
    assert daq.source_type == "neutron"

    # Check extra metadata
    extra = dg["extra"]
    assert json.loads(extra["input_files"]) == ["file1.fits", "file2.fits"]
    assert extra["processing_timestamp"] == "2024-06-01T12:00:00Z"
    np.testing.assert_equal(json.loads(extra["roi_applied"]), (0, 0, 5, 5))
    assert extra["num_runs_combined"] == 2
    assert extra["software_version"] == "1.0.0"
    assert extra["boolean_flag"] is True


def test_write_tiff_stack_drops_object_dtype_coords_and_masks():
    """Object-dtype (PyObject) coords/masks are dropped for scitiff >= 26.6; typed coords survive.

    scitiff 26.6 rejects object-dtype variables. ``write_tiff_stack`` must drop them (e.g.
    tuple-valued TIFF header tags carried over from the input files) while preserving typed
    coordinates and the image data.
    """
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(50, dtype=np.float64).reshape((2, 5, 5))
    da = sc.DataArray(data=sc.array(dims=["t", "y", "x"], values=values, unit="counts", dtype="float64"))
    da.coords["t"] = sc.arange("t", 2, unit="s", dtype="int64")  # typed coord: must survive
    # tuple-valued TIFF header tag stored as a PyObject scalar coord: must be dropped
    da.coords["BitsPerSample"] = sc.scalar((32,))
    assert da.coords["BitsPerSample"].dtype == sc.DType.PyObject
    # a PyObject mask: must be dropped (the write must not raise)
    da.masks["obj_mask"] = sc.scalar([1, 2, 3])
    assert da.masks["obj_mask"].dtype == sc.DType.PyObject

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
        write_tiff_stack(f.name, da)  # would raise if a PyObject variable reached scitiff
        dg = load_scitiff(f.name)

    image = dg["image"]
    np.testing.assert_allclose(image.values, values.astype("float32"), rtol=1e-6)
    assert "t" in image.coords  # typed coord preserved
    np.testing.assert_array_equal(image.coords["t"].values, [0, 1])
    assert "BitsPerSample" not in image.coords  # object-dtype coord dropped
    assert "obj_mask" not in image.masks  # object-dtype mask dropped


def test_write_tiff_stack_preserves_nested_path_provenance():
    """Nested per-run path groups round-trip unflattened, matching the HDF5 writer's provenance."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(25, dtype=np.float64).reshape((5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="counts", dtype="float64"))
    nested = [["r1a.tif", "r1b.tif"], ["r2a.tif", "r2b.tif", "r2c.tif"]]  # 2 runs, ragged

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
        write_tiff_stack(f.name, transmission, metadata={"sample_paths": nested})
        dg = load_scitiff(f.name)

    # decoded provenance keeps the exact nested structure (not flattened to one list)
    assert json.loads(dg["extra"]["sample_paths"]) == nested


def test_write_tiff_stack_3d():
    """Test writing a 3D transmission DataArray using scitiff."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(500, dtype=np.float64).reshape((20, 5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["t", "y", "x"], values=values, unit="counts", dtype="float64"))
    transmission.coords["t"] = sc.linspace("t", 1000, 10000, 21, unit="s")
    transmission.variances = values

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
        # Save
        write_tiff_stack(f.name, transmission)
        # Load back the file to verify contents
        dg = load_scitiff(f.name)

    assert isinstance(dg, sc.DataGroup)
    assert "image" in dg

    image = dg["image"]
    assert image.dtype == sc.DType.float32
    assert image.dims == ("t", "y", "x")
    assert image.values.shape == (20, 5, 5)
    np.testing.assert_allclose(image.values, values, rtol=1e-6)
    np.testing.assert_allclose(image.variances, values, rtol=1e-6)
    np.testing.assert_allclose(image.coords["t"].values, np.linspace(1000, 10000, 21))

    # Check that no masks are present
    assert len(image.masks) == 0

    # Check that extra metadata is None
    assert dg["extra"] is None


def test_write_tiff_no_variances():
    """Test writing a DataArray without variances."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(25, dtype=np.float64).reshape((5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="counts", dtype="float64"))

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
        write_tiff_stack(f.name, transmission)
        dg = load_scitiff(f.name)

    image = dg["image"]
    assert image.variances is None


def test_write_tiff_stack_unwriteable_path():
    """Test that writing to an unwriteable path raises a PermissionError."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(25, dtype=np.float64).reshape((5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="counts", dtype="float64"))

    with pytest.raises((PermissionError, OSError)):
        write_tiff_stack("/nonexistent/deep/path/file.tiff", transmission)

    with pytest.raises(PermissionError):
        write_tiff_stack("/file.tiff", transmission)


def test_write_tiff_stack_unsupported_metadata_type():
    """Test that unsupported metadata types raise a ValueError."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(25, dtype=np.float64).reshape((5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="counts", dtype="float64"))

    metadata = {
        "valid_string": "test",
        "valid_number": 42,
        "valid_list": [1, 2, 3],
        "invalid_dict": {"key": "value"},  # dicts are not supported
    }

    with pytest.raises(ValueError):
        write_tiff_stack("test.tiff", transmission, metadata=metadata)
