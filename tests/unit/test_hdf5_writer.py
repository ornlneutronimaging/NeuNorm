"""
Unit tests for the HDF5 writer module.
"""

import json
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


def test_write_hdf5_roundtrips_nested_metadata():
    """Nested list-of-lists metadata (per-run file paths) round-trips losslessly as JSON.

    Regression for https://github.com/ornlneutronimaging/NeuNorm/issues/140: h5py cannot
    store a ragged nested list natively, which previously aborted write_hdf5 *after* the
    bulk arrays were written (a corrupt partial file). Nested provenance is now serialized
    to a JSON string tagged with the dataset attribute ``encoding="json"`` and read back
    with ``json.loads`` — handling ragged and rectangular shapes uniformly without loss.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values * 2.0

    ragged = [["s_0001.tiff", "s_0002.tiff"], ["s_0003.tiff"]]  # unequal run sizes (production shape)
    rectangular = [["ob_0001.tiff", "ob_0002.tiff"], ["ob_0003.tiff", "ob_0004.tiff"]]
    metadata = {
        "sample_paths": ragged,
        "ob_paths": rectangular,
        "num_runs_combined": 2,  # a normal scalar that must still be written
    }

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        # Must NOT raise (previously TypeError on the ragged list).
        write_hdf5(f.name, data, metadata=metadata)

        with h5py.File(f.name, "r") as f:
            # The bulk array was written (file is not a corrupt partial).
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            # Nested provenance is preserved (not dropped) and round-trips exactly,
            # for both ragged and rectangular shapes.
            for key, expected in (("sample_paths", ragged), ("ob_paths", rectangular)):
                assert f[f"metadata/{key}"].attrs["encoding"] == "json"
                assert json.loads(f[f"metadata/{key}"].asstr()[()]) == expected
            # Scalar metadata still wrote.
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def _data_3d():
    values = np.arange(3 * 5 * 5).reshape((3, 5, 5))
    data = sc.DataArray(
        data=sc.array(dims=["tof_edges", "x", "y"], values=values, unit="counts", dtype="float64"),
        coords={
            "tof_edges": sc.linspace("tof_edges", 1e5, 1e7, num=4, unit="ns"),
            "y": sc.arange("y", 5, unit=None),
            "x": sc.arange("x", 5, unit=None),
        },
    )
    data.variances = values * 2.0
    return data, values


def test_write_hdf5_fallback_json_for_non_native_metadata():
    """A non-nested value h5py cannot store natively (e.g. a dict) falls back to JSON.

    Exercises the defensive fallback path: the native create_dataset raises,
    the partial dataset is removed, and the value is JSON-encoded instead of dropped.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    params = {"threshold": 3.0, "mode": "auto"}  # dict: not natively storable, JSON-serializable
    metadata = {"params": params, "num_runs_combined": 2}

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            assert f["metadata/params"].attrs["encoding"] == "json"
            assert json.loads(f["metadata/params"].asstr()[()]) == params
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def test_write_hdf5_skips_unserializable_nested_metadata_without_corrupting():
    """A nested value json.dumps cannot encode is skipped — never aborting the write.

    json.dumps can still raise (e.g. on a circular reference) even with default=str. A
    self-referential list is nested, so this exercises the *nested-branch* guard: the key is
    skipped so the bulk arrays and other metadata write intact and the file is not corrupt.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    circular = []  # nested (list-containing-list) AND not JSON-serializable
    circular.append(circular)
    metadata = {"weird": circular, "num_runs_combined": 2}

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            # Bulk arrays written; file is not a corrupt partial.
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            # The un-encodable key was skipped, not partially written.
            assert "metadata/weird" not in f
            # Metadata after the skipped key still wrote.
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def test_write_hdf5_skips_unstorable_fallback_metadata_without_corrupting():
    """The fallback branch skips a value h5py AND json both reject — without corrupting.

    A dict with a tuple key is not natively storable by h5py (TypeError) and not
    JSON-serializable (json requires str/number keys), so it exercises the native-write
    fallback's skip-with-warning path, distinct from the nested-branch guard.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    metadata = {"weird": {(1, 2): "v"}, "num_runs_combined": 2}  # not nested; h5py- and json-unstorable

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            assert "metadata/weird" not in f
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def test_write_hdf5_skips_malformed_key_metadata_without_corrupting():
    """A malformed metadata key (empty / trailing-slash) is skipped, not fatal.

    h5py rejects empty or slash-bearing dataset names with ValueError. That raise happens
    inside the per-key write *after* the bulk arrays, so the loop's backstop must skip the bad
    key (cleaning up any partial) and keep writing the rest rather than corrupt the file. This
    exercises the catch-all that guards the otherwise-unguarded create_dataset name.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    metadata = {
        "valid_before": "ok",  # valid key BEFORE the bad ones must survive (cleanup must not over-delete)
        "": [["a.tiff"], ["b.tiff"]],  # empty key -> name "metadata/" -> h5py ValueError (nested branch)
        "runs/": {"threshold": 3.0},  # trailing-slash key -> h5py ValueError (fallback branch)
        "num_runs_combined": 2,  # valid key after the bad ones must still write
    }

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            # Bulk arrays intact; loop continued past the malformed keys.
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            # Valid metadata both before AND after the bad keys is preserved.
            assert f["metadata/valid_before"].asstr()[()] == "ok"
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def test_write_hdf5_skips_unformattable_key_metadata_without_corrupting():
    """A non-string key whose formatting raises is skipped, not fatal.

    Name construction (f"metadata/{key}") runs inside the per-key backstop, so even a key
    object whose __format__/__str__/__repr__ raises is skipped with a warning rather than
    aborting the write. Realistic provenance dicts never contain such keys; this guards the
    public write_hdf5 API against the one remaining raise-after-bulk-arrays path.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    class BadKey:
        def __str__(self):
            raise RuntimeError("boom")

        def __repr__(self):
            raise RuntimeError("boom")

        def __format__(self, spec):
            raise RuntimeError("boom")

        def __hash__(self):
            return 1

        def __eq__(self, other):
            return self is other

    data, values = _data_3d()
    metadata = {BadKey(): "x", "num_runs_combined": 2}

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            # The valid key after the unformattable one still wrote (loop did not abort).
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def test_write_hdf5_colliding_keys_keep_first_without_dropping_earlier():
    """When two keys collide under str() (e.g. 1 and "1"), the first is kept.

    Both int 1 and str "1" map to dataset path "metadata/1". The earlier value must survive and
    the later colliding key is skipped — cleanup must never delete the earlier valid dataset.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    metadata = {1: "first", "1": "second", "after": "ok"}  # 1 and "1" collide -> "metadata/1"

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            # The first value at the colliding path survives (not deleted by the later key).
            assert f["metadata/1"].asstr()[()] == "first"
            # Metadata after the collision still wrote.
            assert f["metadata/after"].asstr()[()] == "ok"


def test_write_hdf5_skips_surrogate_key_metadata_without_corrupting():
    """A string key h5py cannot encode (lone surrogate) is skipped, not fatal.

    For a key like "\\udcff", create_dataset raises UnicodeEncodeError, and the `name in f`
    cleanup lookup would re-raise the same error — so the except-body cleanup must be guarded.
    The write must complete and later metadata must still be written.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    metadata = {"\udcff": "boom", "num_runs_combined": 2}  # lone surrogate key -> h5py UnicodeEncodeError

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


def test_write_hdf5_skips_unstorable_string_metadata_without_corrupting():
    """A string h5py cannot store (embedded NUL) is skipped — never aborting the write.

    h5py VLEN strings reject embedded NULs (ValueError) and lone surrogates
    (UnicodeEncodeError, a ValueError subclass); both raise *after* the bulk arrays are
    written, so the str branch must skip rather than leave a corrupt partial file.
    """
    from neunorm.exporters.hdf5_writer import write_hdf5

    data, values = _data_3d()
    metadata = {"note": "bad\x00null", "num_runs_combined": 2}  # embedded NUL -> h5py ValueError

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        write_hdf5(f.name, data, metadata=metadata)  # must not raise
        with h5py.File(f.name, "r") as f:
            assert "transmission" in f
            np.testing.assert_allclose(f["transmission"][:], values)
            assert "metadata/note" not in f
            # Metadata after the skipped key still wrote (no mid-loop abort).
            assert "metadata/num_runs_combined" in f
            np.testing.assert_equal(f["metadata/num_runs_combined"], np.array(2))


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

    with pytest.raises((PermissionError, OSError)):
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
