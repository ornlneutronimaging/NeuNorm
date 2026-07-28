"""Unit tests for TIFF writer DataGroup construction and SciTiff integration call."""

import json
import tempfile
from pathlib import Path

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


def test_write_tiff_stack_numpy_int_metadata_stays_numeric():
    """NumPy integer metadata (e.g. an ROI tuple of np.int64) round-trips as JSON numbers, not strings."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(25, dtype=np.float64).reshape((5, 5))
    transmission = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="counts", dtype="float64"))
    metadata = {"roi_applied": tuple(np.int64(v) for v in (5, 5, 25, 25))}

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
        write_tiff_stack(f.name, transmission, metadata=metadata)
        dg = load_scitiff(f.name)

    decoded = json.loads(dg["extra"]["roi_applied"])
    assert decoded == [5, 5, 25, 25]
    assert all(isinstance(v, int) for v in decoded)  # numeric, not "5" strings


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


def _tof_stack(n=3, ny=4, nx=4):
    """A (t, y, x) transmission stack with a bin-edge t axis and a per-bin spectra_tof point coord."""
    values = np.arange(n * ny * nx, dtype=np.float64).reshape((n, ny, nx))
    return sc.DataArray(
        data=sc.array(dims=["t", "y", "x"], values=values, unit="dimensionless", variances=np.ones((n, ny, nx))),
        coords={
            "t": sc.array(dims=["t"], values=np.arange(n + 1, dtype=float), unit="s"),  # bin edges (N+1)
            "spectra_tof": sc.array(dims=["t"], values=np.arange(n, dtype=float) + 0.5, unit="s"),  # points (N)
            "y": sc.arange("y", ny),
            "x": sc.arange("x", nx),
        },
    )


def test_write_tiff_stack_one_file_per_image():
    """one_file_per_image=True writes one scitiff per spectral image (one normalization per file),
    zero-padded so the files sort in spectral order, each carrying its own slice's data."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    stack = _tof_stack(n=3)
    with tempfile.TemporaryDirectory() as d:
        output_path = Path(d) / "norm.tiff"
        written = write_tiff_stack(output_path, stack, one_file_per_image=True)

        assert [p.name for p in written] == ["norm_00000.tiff", "norm_00001.tiff", "norm_00002.tiff"]
        assert sorted(p.name for p in Path(d).iterdir()) == [p.name for p in written]  # sort == spectral order
        assert not output_path.exists()  # the template itself is not written

        for index, path in enumerate(written):
            image = load_scitiff(path)["image"]
            # exactly ONE image per file: the only non-spatial dim is scitiff's stdev/mask channel
            # (concat_stdevs_and_mask=True), never a second spectral image.
            assert image.sizes["y"] == stack.sizes["y"]
            assert image.sizes["x"] == stack.sizes["x"]
            assert [d for d in image.dims if d not in ("y", "x", "c")] == []
            # channel 0 is the normalization itself; compare the FULL array, no squeeze/slicing
            values = sc.values(image).values
            frame = values[0] if values.ndim == 3 else values
            np.testing.assert_allclose(frame, stack.values[index], rtol=1e-6)


def test_write_tiff_stack_one_file_per_image_keeps_per_bin_coords():
    """Each per-image file records which bin it came from: that bin's t bounds and spectra_tof."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    stack = _tof_stack(n=3)
    with tempfile.TemporaryDirectory() as d:
        written = write_tiff_stack(Path(d) / "norm.tiff", stack, one_file_per_image=True)
        for index, path in enumerate(written):
            image = load_scitiff(path)["image"]
            np.testing.assert_allclose(image.coords["spectra_tof"].values, index + 0.5)
            np.testing.assert_allclose(image.coords["t"].values, [index, index + 1])  # this bin's bounds


def test_write_tiff_stack_default_is_single_stack():
    """Default (one_file_per_image=False) is unchanged: a single multi-page file, no per-image files."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    stack = _tof_stack(n=3)
    with tempfile.TemporaryDirectory() as d:
        output_path = Path(d) / "norm.tiff"
        written = write_tiff_stack(output_path, stack)
        assert written == [output_path]
        assert [p.name for p in Path(d).iterdir()] == ["norm.tiff"]
        assert load_scitiff(output_path)["image"].sizes["t"] == 3  # whole stack in one file


def test_write_tiff_stack_one_file_per_image_on_2d_writes_single_file():
    """A plain (y, x) radiograph has no spectral dim — it is already one image, so it is written
    as a single file rather than being split."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    values = np.arange(16, dtype=np.float64).reshape((4, 4))
    radiograph = sc.DataArray(data=sc.array(dims=["y", "x"], values=values, unit="dimensionless"))
    with tempfile.TemporaryDirectory() as d:
        output_path = Path(d) / "radio.tiff"
        written = write_tiff_stack(output_path, radiograph, one_file_per_image=True)
        assert written == [output_path]
        assert output_path.exists()


def test_write_tiff_stack_one_file_per_image_preserves_variances_mask_and_metadata():
    """Each per-image file must carry the SAME payload the stack does: that slice's stdevs, its mask,
    and the metadata/DAQ tags — verified on disk, not just assumed (review finding)."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    stack = _tof_stack(n=3)
    stack.variances = np.arange(3 * 4 * 4, dtype=np.float64).reshape((3, 4, 4)) + 1.0  # distinct per frame
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    stack.masks["dead"] = sc.array(dims=["y", "x"], values=mask)

    with tempfile.TemporaryDirectory() as d:
        written = write_tiff_stack(
            Path(d) / "norm.tiff",
            stack,
            metadata={"roi_applied": "false", "sample_paths": ["a.tiff", "b.tiff"]},
            daqmetadata={"facility": "ORNL", "instrument": "VENUS"},
            one_file_per_image=True,
        )
        assert len(written) == 3
        for index, path in enumerate(written):
            dg = load_scitiff(path)
            image = dg["image"]
            # stdevs for THIS slice survive the round trip
            stdevs = sc.stddevs(image).values
            frame_stdevs = stdevs[0] if stdevs.ndim == 3 else stdevs
            np.testing.assert_allclose(frame_stdevs, np.sqrt(stack.variances[index]), rtol=1e-6)
            # the mask travels with every file (scitiff round-trips it as "scitiff-mask")
            assert "scitiff-mask" in image.masks
            np.testing.assert_array_equal(image.masks["scitiff-mask"].values, mask)
            # metadata/DAQ tags travel with every file
            assert json.loads(dg["extra"]["sample_paths"]) == ["a.tiff", "b.tiff"]
            assert dg["extra"]["roi_applied"] == "false"


def test_write_tiff_stack_one_file_per_image_index_width_grows_past_five_digits():
    """The zero-pad width must grow with the image count so a lexicographic listing always equals
    spectral order (a fixed %05d would sort norm_100000 before norm_99999)."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    # 5 digits is the floor for small stacks
    small = _tof_stack(n=2)
    with tempfile.TemporaryDirectory() as d:
        names = [p.name for p in write_tiff_stack(Path(d) / "n.tiff", small, one_file_per_image=True)]
        assert names == ["n_00000.tiff", "n_00001.tiff"]

    # the width formula itself, checked at the 100000-image boundary without writing that many files
    assert max(5, len(str(100000 - 1))) == 5  # 0..99999 still fits in 5
    assert max(5, len(str(100001 - 1))) == 6  # 0..100000 needs 6, so ordering is preserved
    padded = [f"{i:0{max(5, len(str(100000)))}d}" for i in (99999, 100000)]
    assert padded == ["099999", "100000"]
    assert sorted(padded) == padded  # lexicographic == numeric


def test_write_tiff_stack_one_file_per_image_rejects_ambiguous_and_empty_spectral_dims():
    """Two non-spatial dims is an ambiguous split, and an empty spectral dim would write no file at
    all — both must raise instead of silently doing the wrong thing (review findings)."""
    from neunorm.exporters.tiff_writer import write_tiff_stack

    two_spectral = sc.DataArray(
        data=sc.array(dims=["angle", "t", "y", "x"], values=np.zeros((2, 3, 4, 4)), unit="dimensionless")
    )
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="exactly one non-spatial dimension"):
            write_tiff_stack(Path(d) / "n.tiff", two_spectral, one_file_per_image=True)

    empty = sc.DataArray(data=sc.array(dims=["t", "y", "x"], values=np.zeros((0, 4, 4)), unit="dimensionless"))
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="empty"):
            write_tiff_stack(Path(d) / "n.tiff", empty, one_file_per_image=True)
        assert list(Path(d).iterdir()) == []  # nothing written
