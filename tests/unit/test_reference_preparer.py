"""
Unit tests for reference_preparer
"""

from pathlib import Path

import numpy as np
import pytest
import scipp as sc


def test_prepare_reference_mean():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. Depending on which pixel, mean is 50 or 102, median is 40 or 102
    stack_data = np.tile(
        [[10, 40, 100], [101, 102, 103]], (4, 2, 1)
    ).T  # this creates an array with shape (N_image=3, y=4, x=4)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=stack_data, unit="counts", dtype="float64"),
    )
    stack.variances = stack.values.copy()  # Poisson

    prepared = prepare_reference(stack, method="mean", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 50 and 102
    expected_values = np.tile([50.0, 102.0], (4, 2)).T
    np.testing.assert_allclose(prepared.values, expected_values)

    # Variance should be propagated. (10+40+100)/3^2 = 150/9, or (101+102+103)/3^2 = 34.
    expected_variance = np.tile([150 / 9, 34], (4, 2)).T
    np.testing.assert_allclose(prepared.variances, expected_variance)


def test_prepare_reference_median():
    """Test basic reference preparation: median"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. Depending on which pixel, mean is 50 or 102, median is 40 or 102
    stack_data = np.tile(
        [[10, 40, 100], [101, 102, 103]], (4, 2, 1)
    ).T  # this creates an array with shape (N_image=3, y=4, x=4)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=stack_data, unit="counts", dtype="float64"),
    )
    stack.variances = stack.values.copy()  # Poisson

    prepared = prepare_reference(stack, method="median", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 40 and 102
    expected_values = np.tile([40.0, 102.0], (4, 2)).T
    np.testing.assert_allclose(prepared.values, expected_values)

    # Variance should be propagated. Variance should be approximately (π/2) * (the variance from the mean calculation).
    expected_variance = np.tile([np.pi / 2 * 150 / 9, np.pi / 2 * 34], (4, 2)).T
    np.testing.assert_allclose(prepared.variances, expected_variance)


def test_prepare_reference_median_no_variance():
    """Test basic reference preparation: mean"""
    from neunorm.processing.reference_preparer import prepare_reference

    # Create simple sample stack. mean is 50, median is 40
    stack_data = np.tile([10, 40, 100], (5, 5, 1)).T  # this creates an array with shape (3, 5, 5)

    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=stack_data, unit="counts", dtype="float64"),
    )

    prepared = prepare_reference(stack, method="median", dim="N_image")

    # Should still be in counts
    assert prepared.unit == sc.units.counts

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 40
    np.testing.assert_allclose(prepared.values, 40.0)

    # Variance should be None since input had no variance
    assert prepared.variances is None


def test_prepare_reference_2d():
    """Test that 2D input is returned unchanged."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["x", "y"], values=np.full((5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    prepared = prepare_reference(data, method="mean", dim="N_image")

    assert data is prepared  # Should be the same object


def test_prepare_reference_3d_single_frame_mean():
    """Test that 3D input with N_image=1 is returned as 2D."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((1, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    assert data.dims == ("N_image", "x", "y")

    prepared = prepare_reference(data, method="mean", dim="N_image")

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values and variances should be 42
    np.testing.assert_allclose(prepared.values, 42.0)
    np.testing.assert_allclose(prepared.variances, 42.0)


def test_prepare_reference_3d_single_frame_median():
    """Test that 3D input with N_image=1 is returned as 2D."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((1, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    assert data.dims == ("N_image", "x", "y")

    prepared = prepare_reference(data, method="median", dim="N_image")

    # dims should be (x, y)
    assert prepared.dims == ("x", "y")

    # Values should be 42. Variance should be approximately (π/2) * 42.
    np.testing.assert_allclose(prepared.values, 42.0)
    np.testing.assert_allclose(prepared.variances, 42.0 * np.pi / 2, atol=1)


def test_wrong_dim():
    """Test that wrong dimension raises error."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((3, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    with pytest.raises(ValueError, match="Dimension 'wrong_dim' not found in input data"):
        prepare_reference(data, method="mean", dim="wrong_dim")


def test_wrong_method():
    """Test that wrong method raises error."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((3, 5, 5), 42.0), unit="counts", dtype="float64"),
    )
    data.variances = data.values.copy()  # Poisson

    with pytest.raises(ValueError, match="Unsupported method 'wrong_method'"):
        prepare_reference(data, method="wrong_method", dim="N_image")


def test_prepare_reference_different_variances():
    """Test that returns different variances for different input variances. Same values."""
    from neunorm.processing.reference_preparer import prepare_reference

    data = sc.DataArray(
        data=sc.array(dims=["N_image", "x", "y"], values=np.full((3, 2, 2), 42.0), unit="counts", dtype="float64"),
    )
    # Create different variances for (x, y)
    variances = np.tile([[10, 20], [30, 40]], (3, 1, 1))  # this creates an array with shape (N_image=3, y=2, x=2)
    data.variances = variances

    prepared_mean = prepare_reference(data, method="mean", dim="N_image")
    prepared_median = prepare_reference(data, method="median", dim="N_image")

    # values should be the same
    np.testing.assert_allclose(prepared_mean.values, 42.0)
    np.testing.assert_allclose(prepared_median.values, 42.0)

    # variances should be different
    expected_mean_variance = variances.sum(axis=0) / 9  # mean variance along N_image
    np.testing.assert_allclose(prepared_mean.variances, expected_mean_variance)
    np.testing.assert_allclose(prepared_median.variances, expected_mean_variance * np.pi / 2)


def _stack_with_text_coord(n_frames: int) -> sc.DataArray:
    """(N_image, y, x) stack with Poisson variances and a per-frame text coordinate."""
    values = np.arange(n_frames * 4 * 5, dtype="float64").reshape(n_frames, 4, 5) + 10.0
    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "y", "x"], values=values, variances=values.copy(), unit="counts"),
        coords={"y": sc.arange("y", 4), "x": sc.arange("x", 5)},
    )
    stack.coords["ExposureTime"] = sc.array(dims=["N_image"], values=np.full(n_frames, 30.0))
    stack.coords["DateTime"] = sc.array(dims=["N_image"], values=[f"2026:05:27 10:{i:02d}:00" for i in range(n_frames)])
    stack.coords.set_aligned("ExposureTime", False)
    stack.coords.set_aligned("DateTime", False)
    return stack


@pytest.mark.parametrize(
    ("n_frames", "method"), [(2, "mean"), (2, "median"), (3, "mean"), (3, "median"), (5, "mean"), (5, "median")]
)
def test_text_coord_along_dim_keeps_first_and_last_values(n_frames, method):
    """A per-frame text coordinate keeps its first and last values; the image and other coords are reduced."""
    from neunorm.processing.reference_preparer import prepare_reference

    stack = _stack_with_text_coord(n_frames)

    prepared = prepare_reference(stack, method=method, dim="N_image")

    expected_datetime = sc.array(
        dims=["N_image"], values=["2026:05:27 10:00:00", f"2026:05:27 10:{n_frames - 1:02d}:00"]
    )
    assert sc.identical(prepared.coords["DateTime"], expected_datetime)
    assert not prepared.coords["DateTime"].aligned
    # Frame i holds 10 + p + 20 i at pixel p, so both the mean and the median over n frames are
    # 10 + p + 10 (n - 1); the variances equal the values, so their mean is that same number.
    expected = np.arange(20.0).reshape(4, 5) + 10.0 + 10.0 * (n_frames - 1)
    variance_factor = 1.0 / n_frames if method == "mean" else np.pi / (2 * n_frames)
    np.testing.assert_allclose(prepared.values, expected)
    np.testing.assert_allclose(prepared.variances, expected * variance_factor)
    np.testing.assert_allclose(prepared.coords["ExposureTime"].value, 30.0)
    assert not prepared.coords["ExposureTime"].aligned
    kept_aligned = {"x", "y"} if method == "mean" else set()
    assert set(prepared.coords) == kept_aligned | {"ExposureTime", "DateTime"}


@pytest.mark.parametrize("method", ["mean", "median"])
@pytest.mark.parametrize("n_frames", [2, 3])
def test_two_dimensional_text_coord_keeps_its_dimension_order(n_frames, method):
    """A (y, N_image) text coordinate keeps its first and last frames in (y, N_image) order."""
    from neunorm.processing.reference_preparer import prepare_reference

    def labels(frames):
        rows = [sc.array(dims=["N_image"], values=[f"r{j}f{i}" for i in frames]) for j in range(4)]
        return sc.concat(rows, dim="y")

    stack = _stack_with_text_coord(n_frames)
    stack.coords["label"] = labels(range(n_frames))
    stack.coords.set_aligned("label", False)

    prepared = prepare_reference(stack, method=method, dim="N_image")

    expected = labels([0, n_frames - 1])
    assert expected.dims == ("y", "N_image")
    assert sc.identical(prepared.coords["label"], expected)
    assert not prepared.coords["label"].aligned


def _stack_with_coord_kinds() -> sc.DataArray:
    """(N_image=3, y=2, x=2) stack with Poisson variances and aligned and unaligned coordinates.

    Aligned: ``x``, ``y`` and the per-frame ``frame_index``. Unaligned: the per-frame number
    ``ExposureTime`` [1, 2, 4] s, the scalar text ``run`` and the per-row ``row_offset`` over ``y``.
    """
    values = np.arange(12, dtype="float64").reshape(3, 2, 2) + 10.0
    stack = sc.DataArray(
        data=sc.array(dims=["N_image", "y", "x"], values=values, variances=values.copy(), unit="counts"),
        coords={
            "y": sc.arange("y", 2.0),
            "x": sc.arange("x", 2.0),
            "frame_index": sc.array(dims=["N_image"], values=[0.0, 1.0, 2.0]),
        },
    )
    stack.coords["ExposureTime"] = sc.array(dims=["N_image"], values=[1.0, 2.0, 4.0], unit="s")
    stack.coords["run"] = sc.scalar("run_42")
    stack.coords["row_offset"] = sc.array(dims=["y"], values=[0.5, 1.5], unit="mm")
    for name in ("ExposureTime", "run", "row_offset"):
        stack.coords.set_aligned(name, False)
    return stack


@pytest.mark.parametrize(("method", "expected"), [("mean", 7.0 / 3.0), ("median", 2.0)])
def test_numeric_unaligned_coord_along_dim_is_reduced_with_method(method, expected):
    """A per-frame number [1, 2, 4] s reduces to its mean 7/3 s or its median 2 s, still unaligned."""
    from neunorm.processing.reference_preparer import prepare_reference

    prepared = prepare_reference(_stack_with_coord_kinds(), method=method, dim="N_image")

    exposure = prepared.coords["ExposureTime"]
    assert exposure.dims == ()
    assert exposure.unit == sc.Unit("s")
    np.testing.assert_allclose(exposure.value, expected)
    assert not exposure.aligned


@pytest.mark.parametrize("method", ["mean", "median"])
def test_unaligned_coords_without_dim_are_copied_unchanged(method):
    """A scalar text coordinate and a coordinate over y are carried over as they are, unaligned."""
    from neunorm.processing.reference_preparer import prepare_reference

    prepared = prepare_reference(_stack_with_coord_kinds(), method=method, dim="N_image")

    assert sc.identical(prepared.coords["run"], sc.scalar("run_42"))
    assert sc.identical(prepared.coords["row_offset"], sc.array(dims=["y"], values=[0.5, 1.5], unit="mm"))
    assert not prepared.coords["run"].aligned
    assert not prepared.coords["row_offset"].aligned


@pytest.mark.parametrize(
    ("method", "expected_coords"),
    [("mean", {"x", "y", "ExposureTime", "run", "row_offset"}), ("median", {"ExposureTime", "run", "row_offset"})],
)
def test_aligned_coord_along_dim_is_dropped(method, expected_coords):
    """An aligned per-frame coordinate is dropped; median with variances drops the aligned x and y too."""
    from neunorm.processing.reference_preparer import prepare_reference

    prepared = prepare_reference(_stack_with_coord_kinds(), method=method, dim="N_image")

    assert "frame_index" not in prepared.coords
    assert set(prepared.coords) == expected_coords


def _stack_with_datetime() -> sc.DataArray:
    """``_stack_with_coord_kinds`` plus an unaligned per-frame ``DateTime`` with three distinct values."""
    stack = _stack_with_coord_kinds()
    stack.coords["DateTime"] = sc.array(
        dims=["N_image"], values=["2026:05:27 10:00:00", "2026:05:27 10:07:00", "2026:05:27 10:09:00"]
    )
    stack.coords.set_aligned("DateTime", False)
    return stack


@pytest.mark.parametrize("method", ["mean", "median"])
def test_text_coord_along_dim_keeps_its_first_and_last_value(method):
    """Three distinct DateTime values reduce to the first and the last, unaligned along N_image."""
    from neunorm.processing.reference_preparer import prepare_reference

    prepared = prepare_reference(_stack_with_datetime(), method=method, dim="N_image")

    expected = sc.array(dims=["N_image"], values=["2026:05:27 10:00:00", "2026:05:27 10:09:00"])
    assert sc.identical(prepared.coords["DateTime"], expected)
    assert not prepared.coords["DateTime"].aligned


def test_text_coord_fallback_is_logged_at_info():
    """Keeping a text coordinate's first and last values is reported at INFO, not WARNING."""
    from loguru import logger

    from neunorm.processing.reference_preparer import prepare_reference

    records = []
    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        prepare_reference(_stack_with_datetime(), method="mean", dim="N_image")
    finally:
        logger.remove(sink_id)

    levels = [r["level"].name for r in records if "Could not reduce coordinate 'DateTime'" in r["message"]]
    assert levels == ["INFO"]


def test_loaded_tiff_stack_with_per_frame_datetime_is_averaged(tmp_path):
    """Three TIFF frames whose DateTime tag differs reduce to their mean with the first and last DateTime."""
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack
    from neunorm.processing.reference_preparer import prepare_reference

    paths = []
    for i, value in enumerate((99, 100, 104)):
        path = tmp_path / f"ob_{i}.tif"
        tifffile.imwrite(path, np.full((8, 10), value, np.uint16), datetime=f"2026:05:27 10:0{i}:00")
        paths.append(path)

    prepared = prepare_reference(load_tiff_stack(paths), dim="N_image")

    assert prepared.dims == ("y", "x")
    np.testing.assert_allclose(prepared.values, 101.0)
    np.testing.assert_allclose(prepared.variances, 303.0 / 9)
    assert list(prepared.coords["DateTime"].values) == ["2026:05:27 10:00:00", "2026:05:27 10:02:00"]


@pytest.mark.parametrize("pipeline", ["mars", "venus"])
def test_ccd_pipelines_average_three_frame_references_with_per_frame_datetime(tmp_path, pipeline):
    """Open-beam and dark runs of three frames, each frame with its own DateTime, are averaged."""
    from PIL import Image

    from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
    from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline

    tags = (
        {65027: "ExposureTime:30.000000", 65025: "ManufacturerStr:DW936_BV"}
        if pipeline == "mars"
        else {65027: "IntegratedPCharge:0.2", 65025: "ManufacturerStr:ANDOR"}
    )

    def write(prefix, values, hour):
        paths = []
        for i, value in enumerate(values):
            img = Image.fromarray(np.full((16, 16), value, dtype=np.float32))
            exif = img.getexif()
            exif.update(tags)
            exif[306] = f"2026:05:27 {hour:02d}:{i:02d}:00"  # DateTime
            path = tmp_path / f"{prefix}_{i:03}.tiff"
            img.save(path, exif=exif)
            paths.append(path)
        return paths

    run = run_mars_ccd_pipeline if pipeline == "mars" else run_venus_ccd_pipeline
    transmission = run(
        sample_paths=[write("sample", (81, 82, 83, 84), hour=12)],
        ob_paths=[write("ob", (99, 100, 101), hour=10)],
        dark_paths=[write("dark", (4, 6, 8), hour=9)],
        output_path=tmp_path / "out.h5",
    )

    expected = (np.array([81.0, 82.0, 83.0, 84.0]) - 6.0) / (100.0 - 6.0)
    np.testing.assert_allclose(transmission.values, np.broadcast_to(expected[:, None, None], (4, 16, 16)), rtol=1e-6)
    assert list(transmission.coords["DateTime"].values) == [f"2026:05:27 12:{i:02d}:00" for i in range(4)]


def test_mars_ccd_pipeline_with_repository_tiff_stack_as_open_beam(tmp_path):
    """The bundled TIFF stack, whose InteropIndex tag differs per frame, serves as sample and open beam."""
    from neunorm.loaders.tiff_loader import load_tiff_stack
    from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

    paths = sorted((Path(__file__).parent.parent / "data" / "tif" / "sample").glob("*.tif"))
    stack = load_tiff_stack(paths)
    assert stack.coords["InteropIndex"].dims == ("N_image",)
    assert len(set(stack.coords["InteropIndex"].values)) == len(paths) == 3

    transmission = run_mars_ccd_pipeline(sample_paths=[paths], ob_paths=[paths], output_path=tmp_path / "out.h5")

    assert transmission.sizes == stack.sizes
    np.testing.assert_allclose(transmission.values, stack.values / stack.values.mean(axis=0))
    assert (tmp_path / "out.h5").is_file()
