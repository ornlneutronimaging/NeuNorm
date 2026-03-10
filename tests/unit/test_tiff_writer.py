"""Unit tests for TIFF writer DataGroup construction and SciTiff integration call."""

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
    assert extra["input_files"].value == ["file1.fits", "file2.fits"]
    assert extra["processing_timestamp"] == "2024-06-01T12:00:00Z"
    np.testing.assert_equal(extra["roi_applied"].value, (0, 0, 5, 5))
    assert extra["num_runs_combined"] == 2
    assert extra["software_version"] == "1.0.0"
    assert extra["boolean_flag"] is True


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
