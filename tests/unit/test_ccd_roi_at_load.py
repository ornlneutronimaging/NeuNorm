"""
Tests for cropping to the ROI as frames load in the MARS and VENUS CCD pipelines.

The pipelines crop every sample, open-beam and dark frame to ``roi`` while it is read. Their output
must equal the result of loading and combining whole frames and cropping afterwards, which these
tests rebuild from the public functions: ``load_stack`` -> ``combine_runs`` -> ``apply_roi`` ->
``prepare_reference`` -> ``detect_dead_pixels`` -> ``apply_gamma_filter`` -> normalization ->
``astype("float32")``. Failing inputs must raise the messages ``combine_runs`` and ``apply_roi``
raise for them, and families of different frame size a message naming both.

The frames are non-square and, apart from the pixels below, every pixel holds a different value.
Every sample frame has a dead pixel inside both ROIs, and the first sample frame of each run has
gamma spikes on the ROI corners and just outside them.
"""

import gc
import weakref
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import tifffile
from astropy.io import fits
from loguru import logger

from neunorm.data_models.roi import MaskROI, as_region_list, as_roi_bounds
from neunorm.filters.gamma_filter import apply_gamma_filter
from neunorm.loaders import fits_loader, tiff_loader
from neunorm.loaders.stack_loader import load_stack
from neunorm.pipelines import mars_ccd as mars_ccd_module
from neunorm.pipelines import venus_ccd as venus_ccd_module
from neunorm.pipelines._ccd_common import combine_owned_runs
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.normalizer import normalize_transmission, normalize_with_dark
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.tof.pixel_detector import detect_dead_pixels
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_LOAD_DARK,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
)

# Full frames are 24 rows by 32 columns; "small" runs come from a detector of another size.
NY, NX = 24, 32
SMALL_NY, SMALL_NX = 16, 20
DIMS = ("N_image", "y", "x")

FAMILIES = ("sample", "ob", "dark")
N_FRAMES = {"sample": 3, "ob": 2, "dark": 2}
BASE_COUNTS = {"sample": 5000, "ob": 20000, "dark": 10}
PROTON_CHARGE = {"sample": 0.1, "ob": 0.2, "dark": 0.1}
# How the pipelines name each family in their combine progress notes.
COMBINE_LABEL = {"sample": "sample", "ob": "open-beam", "dark": "dark"}

ROIS = {
    "interior": (5, 7, 27, 19),  # 22 x 12
    "square_offset": (3, 9, 15, 21),  # 12 x 12 with x0 != y0: a crop with x and y swapped keeps its shape
}
ROI_FITS_BOTH_SIZES = (2, 3, 10, 9)  # inside the full frame and the small one

# Regions in the cropped frame; both fit either ROI.
AIR_ROI = (1, 2, 5, 6)
BACKGROUND_ROI = (6, 1, 10, 4)

# (y, x) in full-frame indices.
DEAD = (10, 12)  # zero in every sample frame, inside both ROIs
SPIKES = (
    (7, 5),  # interior ROI, top-left corner
    (6, 4),  # one pixel diagonally outside it
    (12, 4),  # one column left of the interior ROI
    (18, 26),  # interior ROI, bottom-right corner
    (19, 27),  # one pixel diagonally outside it
    (9, 3),  # square-offset ROI, top-left corner
    (8, 2),  # one pixel diagonally outside it
    (20, 14),  # square-offset ROI, bottom-right corner
    (21, 15),  # one pixel diagonally outside it
)
SPIKE_COUNTS = 60000

MARS_MATCH = ["ExposureTime", "ManufacturerStr", "MotSlitVB.RBV", "MotSlitVT.RBV", "MotSlitHR.RBV", "MotSlitHL.RBV"]
METADATA_TAGS = {
    "RunNo": 65022,
    "ManufacturerStr": 65025,
    "ExposureTime": 65027,
    "IntegratedPCharge": 65029,
    "MotSlitVB.RBV": 65052,
    "MotSlitVT.RBV": 65054,
    "MotSlitHR.RBV": 65056,
    "MotSlitHL.RBV": 65058,
}

PIPELINES = {"mars": run_mars_ccd_pipeline, "venus": run_venus_ccd_pipeline}
PIPELINE_MODULES = {"mars": mars_ccd_module, "venus": venus_ccd_module}
LOAD_STAGES = {"sample": STAGE_LOAD_SAMPLE, "ob": STAGE_LOAD_OB, "dark": STAGE_LOAD_DARK}


def _frame(family: str, shape: tuple[int, int], run: int, index: int) -> np.ndarray:
    """A uint16 frame whose pixels differ from each other and from every other frame of the family.

    Full-size sample frames also get the dead pixel and, in frame 0, the gamma spikes.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = (BASE_COUNTS[family] + nx * yy + xx + 997 * index + 3001 * run).astype(np.uint16)
    if family == "sample" and shape == (NY, NX):
        data[DEAD] = 0
        if index == 0:
            for y, x in SPIKES:
                data[y, x] = SPIKE_COUNTS
    return data


def _write_frame(path: Path, data: np.ndarray, fmt: str, family: str, index: int) -> None:
    """Write one frame with the metadata both pipelines check when combining runs."""
    metadata = {
        "RunNo": 1000 + index,
        "ManufacturerStr": "DW936_BV",
        "ExposureTime": 30.0,
        "IntegratedPCharge": PROTON_CHARGE[family],
        "MotSlitVB.RBV": 42.3,
        "MotSlitVT.RBV": 42.8,
        "MotSlitHR.RBV": 41.4,
        "MotSlitHL.RBV": 42.4,
    }
    if fmt == "tiff":
        tags = [(METADATA_TAGS[key], "s", 0, f"{key}:{value}", True) for key, value in metadata.items()]
        tifffile.imwrite(path, data, extratags=tags, metadata=None)
    else:
        hdu = fits.PrimaryHDU(data)  # uint16 is stored as int16 with BZERO
        for key, value in metadata.items():
            hdu.header[f"HIERARCH {key}"] = value
        hdu.writeto(path)


def _write_run(directory: Path, fmt: str, family: str, run: int, n_frames: int, shape: tuple[int, int]) -> list[Path]:
    ext = ".tiff" if fmt == "tiff" else ".fits"
    paths = []
    for index in range(n_frames):
        path = directory / f"{family}_r{run}_{index:03}{ext}"
        _write_frame(path, _frame(family, shape, run, index), fmt, family, index)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Per format and family: two full-size runs, a run of another frame size, a run with one frame fewer
    and a run whose frames are full-size with rows and columns swapped."""
    root = tmp_path_factory.mktemp("ccd_roi_at_load")
    inputs = {}
    for fmt in ("tiff", "fits"):
        directory = root / fmt
        directory.mkdir()
        inputs[fmt] = {}
        for family in FAMILIES:
            n = N_FRAMES[family]
            inputs[fmt][family] = {
                "run0": _write_run(directory, fmt, family, 0, n, (NY, NX)),
                "run1": _write_run(directory, fmt, family, 1, n, (NY, NX)),
                "small": _write_run(directory, fmt, family, 2, n, (SMALL_NY, SMALL_NX)),
                "short": _write_run(directory, fmt, family, 3, n - 1, (NY, NX)),
                "transposed": _write_run(directory, fmt, family, 4, n, (NX, NY)),
            }
    return inputs


class _Timeline:
    """Frame reads and log records, in the order they happened."""

    def __init__(self):
        self.events: list[tuple] = []

    def reads(self, family: str) -> list[str]:
        """File names read for ``family``, sorted (frames of one stack are read concurrently)."""
        return sorted(event[1] for event in self.events if event[0] == "read" and event[1].startswith(f"{family}_"))

    def all_reads(self) -> list[str]:
        return [event[1] for event in self.events if event[0] == "read"]

    def messages(self, level: str) -> list[str]:
        return [event[2] for event in self.events if event[0] == "log" and event[1] == level]

    def crop_lines(self) -> list[tuple[str, str]]:
        return [
            (event[1], event[2])
            for event in self.events
            if event[0] == "log" and event[2].startswith("Cropping frames to ROI")
        ]

    def clear(self) -> None:
        self.events.clear()


@pytest.fixture
def timeline(monkeypatch):
    """Record every TIFF and FITS frame read and every log record at INFO or above."""
    record = _Timeline()
    for module, name in ((tiff_loader, "_read_tiff_frame"), (fits_loader, "_read_fits_frame")):
        read = getattr(module, name)

        def recording(path, _read=read):
            record.events.append(("read", Path(path).name))
            return _read(path)

        monkeypatch.setattr(module, name, recording)
    sink_id = logger.add(
        lambda message: record.events.append(("log", message.record["level"].name, message.record["message"])),
        level="INFO",
    )
    yield record
    logger.remove(sink_id)


def _names(*runs) -> list[str]:
    return sorted(path.name for run in runs for path in run)


def _combine_kwargs(pipeline: str, family: str, background_roi=None, metadata_match_atol: float = 0.0) -> dict:
    """The keyword arguments each pipeline passes to ``combine_runs`` for one family."""
    if pipeline == "mars":
        match = MARS_MATCH if family != "dark" else ["ExposureTime", "ManufacturerStr"]
        return {
            "metadata_keys_to_sum": ("ExposureTime",),
            "metadata_check_match": match,
            "normalize_by_runs": True,
            "metadata_match_atol": metadata_match_atol,
        }
    return {
        "metadata_keys_to_sum": () if background_roi is not None else ("IntegratedPCharge",),
        "metadata_check_match": ["ManufacturerStr"],
        "normalize_by_runs": True,
    }


def _load_and_combine(pipeline, groups, family, background_roi) -> sc.DataArray:
    return combine_runs([load_stack(paths) for paths in groups], **_combine_kwargs(pipeline, family, background_roi))


def _normalize(pipeline, sample, ob, dark, background_roi) -> sc.DataArray:
    """The pipeline's normalization: by ``background_roi`` when given, else by proton charge for VENUS."""
    correction = {}
    if background_roi is not None:
        correction["background_roi"] = background_roi
    elif pipeline == "venus":
        correction["proton_charge_sample"] = sample.coords["IntegratedPCharge"].astype("float32")
        correction["proton_charge_ob"] = ob.coords["IntegratedPCharge"].astype("float32")
    if dark is not None:
        return normalize_with_dark(sample, ob, dark, **correction)
    return normalize_transmission(sample, ob, **correction)


def _crop_after_combine(
    pipeline, sample_paths, ob_paths, dark_paths=None, *, roi=None, gamma_filter=True, air_roi=None, background_roi=None
) -> sc.DataArray:
    """The pipeline's processing, with whole frames loaded and combined and the ROI applied afterwards."""
    if background_roi is not None:
        background_roi = as_region_list(background_roi, arg_name="background_roi")

    sample = _load_and_combine(pipeline, sample_paths, "sample", background_roi)
    ob = _load_and_combine(pipeline, ob_paths, "ob", background_roi)
    dark = _load_and_combine(pipeline, dark_paths, "dark", background_roi) if dark_paths else None

    if roi is not None:
        roi = as_roi_bounds(roi)
        sample = apply_roi(sample, roi)
        ob = apply_roi(ob, roi)
        if dark is not None:
            dark = apply_roi(dark, roi)

    if dark is not None:
        dark = prepare_reference(dark, dim="N_image")
    ob = prepare_reference(ob, dim="N_image")
    sample.masks["dead_pixels"] = detect_dead_pixels(sample)
    if gamma_filter:
        sample = apply_gamma_filter(sample)

    transmission = _normalize(pipeline, sample, ob, dark, background_roi)
    if pipeline == "venus":
        if background_roi is not None and "IntegratedPCharge" in transmission.coords:
            del transmission.coords["IntegratedPCharge"]
        if air_roi is not None:
            transmission = apply_air_region_correction(transmission, as_roi_bounds(air_roi))
    return transmission.astype("float32")


def _run(pipeline, output_path, sample_paths, ob_paths, dark_paths=None, **kwargs) -> sc.DataArray:
    return PIPELINES[pipeline](
        sample_paths=sample_paths, ob_paths=ob_paths, dark_paths=dark_paths, output_path=output_path, **kwargs
    )


def _assert_written(output_path: Path, result: sc.DataArray, roi: tuple[int, int, int, int]) -> None:
    """The HDF5 file holds the returned data and detector-index coordinates offset by the ROI origin."""
    x0, y0, x1, y1 = roi
    with h5py.File(output_path, "r") as hf:
        np.testing.assert_array_equal(hf["x"][()], np.arange(x0, x1))
        np.testing.assert_array_equal(hf["y"][()], np.arange(y0, y1))
        np.testing.assert_array_equal(hf["transmission"][()], result.values)
        np.testing.assert_array_equal(hf["uncertainty"][()], np.sqrt(result.variances).astype("float32"))


def _assert_dead_pixel_masked(result: sc.DataArray, roi: tuple[int, int, int, int]) -> None:
    x0, y0, _, _ = roi
    expected = np.zeros((result.sizes["y"], result.sizes["x"]), dtype=bool)
    expected[DEAD[0] - y0, DEAD[1] - x0] = True
    np.testing.assert_array_equal(result.masks["dead_pixels"].transpose(["y", "x"]).values, expected)


# --- Output equals cropping after the combine --------------------------------------------------------------


@pytest.mark.parametrize("n_sample_runs", [1, 2], ids=["1_sample_run", "2_sample_runs"])
@pytest.mark.parametrize("gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize("with_dark", [False, True], ids=["no_dark", "dark"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
@pytest.mark.parametrize("roi_name", list(ROIS))
def test_mars_crop_at_load_matches_crop_after_combine(frames, tmp_path, roi_name, fmt, with_dark, gamma, n_sample_runs):
    inputs = frames[fmt]
    roi = ROIS[roi_name]
    sample_paths = [inputs["sample"][f"run{r}"] for r in range(n_sample_runs)]
    ob_paths = [inputs["ob"]["run0"], inputs["ob"]["run1"]]
    dark_paths = [inputs["dark"]["run0"]] if with_dark else None
    output_path = tmp_path / "out.h5"

    result = run_mars_ccd_pipeline(sample_paths, ob_paths, dark_paths, output_path, roi=roi, gamma_filter=gamma)
    expected = _crop_after_combine("mars", sample_paths, ob_paths, dark_paths, roi=roi, gamma_filter=gamma)

    x0, y0, x1, y1 = roi
    assert result.sizes == {"N_image": N_FRAMES["sample"], "y": y1 - y0, "x": x1 - x0}
    assert sc.identical(result, expected, equal_nan=True)
    _assert_dead_pixel_masked(result, roi)
    _assert_written(output_path, result, roi)


@pytest.mark.parametrize("n_sample_runs", [1, 2], ids=["1_sample_run", "2_sample_runs"])
@pytest.mark.parametrize("background_roi", [None, BACKGROUND_ROI], ids=["no_background_roi", "background_roi"])
@pytest.mark.parametrize("air_roi", [None, AIR_ROI], ids=["no_air_roi", "air_roi"])
@pytest.mark.parametrize("with_dark", [False, True], ids=["no_dark", "dark"])
@pytest.mark.parametrize("roi_name", list(ROIS))
def test_venus_crop_at_load_matches_crop_after_combine(
    frames, tmp_path, roi_name, with_dark, air_roi, background_roi, n_sample_runs
):
    inputs = frames["tiff"]
    roi = ROIS[roi_name]
    sample_paths = [inputs["sample"][f"run{r}"] for r in range(n_sample_runs)]
    ob_paths = [inputs["ob"]["run0"], inputs["ob"]["run1"]]
    dark_paths = [inputs["dark"]["run0"]] if with_dark else None
    output_path = tmp_path / "out.h5"

    result = run_venus_ccd_pipeline(
        sample_paths, ob_paths, dark_paths, output_path, roi=roi, air_roi=air_roi, background_roi=background_roi
    )
    expected = _crop_after_combine(
        "venus", sample_paths, ob_paths, dark_paths, roi=roi, air_roi=air_roi, background_roi=background_roi
    )

    x0, y0, x1, y1 = roi
    assert result.sizes == {"N_image": N_FRAMES["sample"], "y": y1 - y0, "x": x1 - x0}
    assert sc.identical(result, expected, equal_nan=True)
    _assert_dead_pixel_masked(result, roi)
    _assert_written(output_path, result, roi)


@pytest.mark.parametrize("gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize("with_dark", [False, True], ids=["no_dark", "dark"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_single_runs_without_roi_match_reference(frames, tmp_path, pipeline, with_dark, gamma):
    """With one run per family the loaded runs are processed without a copy; the output is unchanged."""
    inputs = frames["tiff"]
    sample_paths = [inputs["sample"]["run0"]]
    ob_paths = [inputs["ob"]["run0"]]
    dark_paths = [inputs["dark"]["run0"]] if with_dark else None

    first = _run(pipeline, tmp_path / "first.h5", sample_paths, ob_paths, dark_paths, gamma_filter=gamma)
    second = _run(pipeline, tmp_path / "second.h5", sample_paths, ob_paths, dark_paths, gamma_filter=gamma)
    expected = _crop_after_combine(pipeline, sample_paths, ob_paths, dark_paths, gamma_filter=gamma)

    assert first.sizes == {"N_image": N_FRAMES["sample"], "y": NY, "x": NX}
    assert sc.identical(first, expected, equal_nan=True)
    assert sc.identical(second, expected, equal_nan=True)
    _assert_dead_pixel_masked(first, (0, 0, NX, NY))


# --- Runs of one family with different uncropped shapes ----------------------------------------------------

GUARD_CASES = {
    # name: (runs of the family, in order; roi)
    "sizes_roi_fits_run0_only": (("run0", "small"), ROIS["interior"]),
    "sizes_roi_fits_both": (("run0", "small"), ROI_FITS_BOTH_SIZES),
    "frame_counts": (("run0", "short"), ROIS["interior"]),
    "sizes_third_run": (("run0", "run1", "small"), ROIS["interior"]),
}


def _shape_message(index: int, shape: tuple, base_shape: tuple) -> str:
    return f"Run {index} has shape {shape} and dims {DIMS}, expected shape {base_shape} and dims {DIMS}"


@pytest.mark.parametrize("case", list(GUARD_CASES))
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_runs_of_different_shape_raise_combine_message_while_the_family_loads(
    frames, tmp_path, timeline, pipeline, fmt, family, case
):
    """The error names the uncropped shapes, exactly as ``combine_runs`` does, before the next family is read."""
    inputs = frames[fmt]
    run_names, roi = GUARD_CASES[case]
    groups = {f: [inputs[f]["run0"]] for f in FAMILIES}
    groups[family] = [inputs[family][name] for name in run_names]

    whole = [load_stack(paths) for paths in groups[family]]
    bad = len(whole) - 1
    with pytest.raises(ValueError) as combined:
        combine_runs(whole, **_combine_kwargs(pipeline, family))
    message = _shape_message(bad, whole[bad].shape, whole[0].shape)
    assert str(combined.value) == message
    assert whole[0].shape == (N_FRAMES[family], NY, NX)

    timeline.clear()
    events = []
    with pytest.raises(ValueError) as raised:
        _run(
            pipeline,
            tmp_path / "out.h5",
            groups["sample"],
            groups["ob"],
            groups["dark"],
            roi=roi,
            progress=events.append,
        )

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == [message]

    # The bad run's frame 0 is enough to reject an ROI that does not fit it; otherwise the whole run is read.
    x0, y0, x1, y1 = roi
    _, bad_ny, bad_nx = whole[bad].shape
    bad_run = groups[family][bad]
    read_of_bad_run = bad_run if (x1 <= bad_nx and y1 <= bad_ny) else bad_run[:1]
    position = FAMILIES.index(family)
    for earlier in FAMILIES[:position]:
        assert timeline.reads(earlier) == _names(*groups[earlier])
    assert timeline.reads(family) == _names(*groups[family][:bad], read_of_bad_run)
    for later in FAMILIES[position + 1 :]:
        assert timeline.reads(later) == []
    combine_notes = [event.detail for event in events if event.stage == STAGE_COMBINE_RUNS and event.detail]
    assert f"combining {len(run_names)} {COMBINE_LABEL[family]} run(s)" not in combine_notes


def _frame_size_message(family: str, size: tuple[int, int], sample_size: tuple[int, int]) -> str:
    name = {"ob": "Open-beam", "dark": "Dark"}[family]
    return (
        f"{name} frames have size (y={size[0]}, x={size[1]}), but sample frames have size"
        f" (y={sample_size[0]}, x={sample_size[1]}); sample, open-beam and dark frames must be the same size"
    )


def _small_run0_error(inputs, family: str, roi: tuple[int, int, int, int]) -> tuple[str, list[str]]:
    """The error when ``roi`` fits full-size frames but not ``family``'s smaller run 0, and the ERROR log.

    The sample loads first, so its smaller frames fail the ROI with the message ``apply_roi`` raises,
    unlogged. A later family's smaller frames are reported as differing in size from the sample's, logged.
    """
    if family != "sample":
        message = _frame_size_message(family, (SMALL_NY, SMALL_NX), (NY, NX))
        return message, [message]
    with pytest.raises(ValueError) as cropped:
        apply_roi(load_stack(inputs[family]["small"]), roi)
    message = f"ROI (x1=27, y1=19) exceeds data size (x={SMALL_NX}, y={SMALL_NY})"
    assert str(cropped.value) == message
    return message, []


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_roi_fitting_a_later_run_but_not_run0_raises_on_run0(frames, tmp_path, timeline, pipeline, family):
    inputs = frames["tiff"]
    roi = ROIS["interior"]
    groups = {f: [inputs[f]["run0"]] for f in FAMILIES}
    groups[family] = [inputs[family]["small"], inputs[family]["run0"]]
    message, errors = _small_run0_error(inputs, family, roi)

    timeline.clear()
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], groups["dark"], roi=roi)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == errors
    position = FAMILIES.index(family)
    for earlier in FAMILIES[:position]:
        assert timeline.reads(earlier) == _names(*groups[earlier])
    assert timeline.reads(family) == _names(inputs[family]["small"][:1])
    for later in FAMILIES[position + 1 :]:
        assert timeline.reads(later) == []


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_roi_not_fitting_one_family_raises_on_its_first_frame(frames, tmp_path, timeline, pipeline, family):
    """The ROI fits the full-size frames of the other families but not this family's smaller frames."""
    inputs = frames["tiff"]
    roi = ROIS["interior"]
    groups = {f: [inputs[f]["run0"]] for f in FAMILIES}
    groups[family] = [inputs[family]["small"]]
    message, errors = _small_run0_error(inputs, family, roi)

    timeline.clear()
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], groups["dark"], roi=roi)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == errors
    position = FAMILIES.index(family)
    for earlier in FAMILIES[:position]:
        assert timeline.reads(earlier) == _names(*groups[earlier])
    assert timeline.reads(family) == _names(inputs[family]["small"][:1])
    for later in FAMILIES[position + 1 :]:
        assert timeline.reads(later) == []


@pytest.mark.parametrize("mismatch", ["small", "short"])
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_without_roi_mismatched_runs_raise_while_the_family_loads(
    frames, tmp_path, timeline, pipeline, family, mismatch
):
    """Without an ROI the runs are checked as they load, with the message ``combine_runs`` raises."""
    inputs = frames["tiff"]
    groups = {f: [inputs[f]["run0"]] for f in FAMILIES}
    groups[family] = [inputs[family]["run0"], inputs[family][mismatch]]

    whole = [load_stack(paths) for paths in groups[family]]
    with pytest.raises(ValueError) as combined:
        combine_runs(whole, **_combine_kwargs(pipeline, family))
    message = _shape_message(1, whole[1].shape, whole[0].shape)
    assert str(combined.value) == message

    timeline.clear()
    events = []
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], groups["dark"], progress=events.append)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == [message]
    assert timeline.crop_lines() == []
    position = FAMILIES.index(family)
    for loaded in FAMILIES[: position + 1]:
        assert timeline.reads(loaded) == _names(*groups[loaded])
    for later in FAMILIES[position + 1 :]:
        assert timeline.reads(later) == []
    combine_notes = [event.detail for event in events if event.stage == STAGE_COMBINE_RUNS and event.detail]
    assert f"combining 2 {COMBINE_LABEL[family]} run(s)" not in combine_notes


@pytest.mark.parametrize("fmt", ["tiff", "fits"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_without_roi_a_smaller_sample_run0_is_reported_as_a_sample_run_mismatch(
    frames, tmp_path, timeline, pipeline, fmt
):
    """Sample run 0 is smaller than sample run 1 and the open beam; the error names the sample runs."""
    inputs = frames[fmt]
    groups = {f: [inputs[f]["run0"]] for f in FAMILIES}
    groups["sample"] = [inputs["sample"]["small"], inputs["sample"]["run0"]]
    message = (
        "Run 1 has shape (3, 24, 32) and dims ('N_image', 'y', 'x'),"
        " expected shape (3, 16, 20) and dims ('N_image', 'y', 'x')"
    )

    timeline.clear()
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], groups["dark"])

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == [message]
    assert timeline.reads("sample") == _names(*groups["sample"])
    assert timeline.reads("ob") == []
    assert timeline.reads("dark") == []


# --- Families of different frame size ----------------------------------------------------------------------

FRAME_SIZE_CASES = {
    # name: (run of the odd family, roi)
    "smaller_no_roi": ("small", None),
    "smaller_roi_fits_both": ("small", ROI_FITS_BOTH_SIZES),
    "transposed_no_roi": ("transposed", None),
    "transposed_roi_fits_both": ("transposed", ROI_FITS_BOTH_SIZES),
}


@pytest.mark.parametrize("case", list(FRAME_SIZE_CASES))
@pytest.mark.parametrize("odd", FAMILIES)
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_families_of_different_frame_size_raise_while_the_family_loads(
    frames, tmp_path, timeline, pipeline, fmt, odd, case
):
    """The first family whose frames differ in size from the sample's raises, logged, before anything is normalized.

    The number of frames differs between the families in every case (3 sample, 2 open-beam, 2 dark).
    """
    inputs = frames[fmt]
    run_name, roi = FRAME_SIZE_CASES[case]
    groups = {f: [inputs[f]["run0"]] for f in FAMILIES}
    groups[odd] = [inputs[odd][run_name]]
    sizes = {f: load_stack(groups[f][0]).shape[1:] for f in FAMILIES}
    failing = "ob" if odd == "sample" else odd
    message = _frame_size_message(failing, sizes[failing], sizes["sample"])
    assert sizes[failing] != sizes["sample"]
    output_path = tmp_path / "out.h5"

    timeline.clear()
    events = []
    with pytest.raises(ValueError) as raised:
        _run(pipeline, output_path, groups["sample"], groups["ob"], groups["dark"], roi=roi, progress=events.append)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == [message]
    position = FAMILIES.index(failing)
    for loaded in FAMILIES[: position + 1]:
        assert timeline.reads(loaded) == _names(*groups[loaded])
    for later in FAMILIES[position + 1 :]:
        assert timeline.reads(later) == []
    assert STAGE_NORMALIZE not in {event.stage for event in events}
    combine_notes = [event.detail for event in events if event.stage == STAGE_COMBINE_RUNS and event.detail]
    assert f"combining 1 {COMBINE_LABEL[failing]} run(s)" not in combine_notes
    assert not output_path.exists()


# --- ROIs rejected without reading a file ------------------------------------------------------------------

INVALID_ROIS = {
    "empty": (0, 0, 0, 0),
    "numpy_ints": tuple(np.int64(v) for v in ROIS["interior"]),
    "floats": tuple(float(v) for v in ROIS["interior"]),
    "misordered": (27, 7, 5, 19),
    "negative": (-1, 7, 27, 19),
    "mask": MaskROI(selection=np.ones((NY, NX), dtype=bool)),
}


@pytest.mark.parametrize("roi_name", list(INVALID_ROIS))
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_invalid_roi_raises_the_same_error_before_any_read(frames, tmp_path, timeline, pipeline, roi_name):
    inputs = frames["tiff"]
    roi = INVALID_ROIS[roi_name]
    if isinstance(roi, MaskROI):
        # Rejected by the pipeline's argument check, which applies before loading.
        with pytest.raises(Exception) as reference:
            as_roi_bounds(roi)
    else:
        with pytest.raises(Exception) as reference:
            apply_roi(load_stack(inputs["sample"]["run0"]), roi)

    timeline.clear()
    with pytest.raises(Exception) as raised:
        _run(
            pipeline,
            tmp_path / "out.h5",
            [inputs["sample"]["run0"]],
            [inputs["ob"]["run0"]],
            [inputs["dark"]["run0"]],
            roi=roi,
        )

    assert type(raised.value) is type(reference.value)
    assert str(raised.value) == str(reference.value)
    assert timeline.all_reads() == []


# --- Log lines ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [1, 2], ids=["1_run", "2_runs"])
@pytest.mark.parametrize("with_dark", [False, True], ids=["no_dark", "dark"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_one_crop_log_line_per_family_before_it_loads(frames, tmp_path, timeline, pipeline, with_dark, n_runs):
    inputs = frames["tiff"]
    roi = ROIS["interior"]
    runs = [f"run{r}" for r in range(n_runs)]
    families = ["sample", "ob"] + (["dark"] if with_dark else [])
    groups = {f: [inputs[f][name] for name in runs] for f in families}

    _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], groups.get("dark"), roi=roi)

    assert timeline.crop_lines() == [("INFO", f"Cropping frames to ROI {roi} as they are loaded")] * len(families)
    # Collapsed to runs of the same kind, the record reads: crop line, that family's files, next crop line, ...
    sequence = []
    for event in timeline.events:
        if event[0] == "read":
            kind = event[1].split("_")[0]
        elif event[2].startswith("Cropping frames to ROI"):
            kind = "crop"
        else:
            continue
        if not sequence or sequence[-1] != kind:
            sequence.append(kind)
    assert sequence == [kind for family in families for kind in ("crop", family)]
    assert not any(message.startswith("Applying ROI") for message in timeline.messages("INFO"))


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_no_crop_log_line_without_roi(frames, tmp_path, timeline, pipeline):
    inputs = frames["tiff"]
    _run(pipeline, tmp_path / "out.h5", [inputs["sample"]["run0"]], [inputs["ob"]["run0"]], [inputs["dark"]["run0"]])
    assert timeline.crop_lines() == []


# --- Progress ----------------------------------------------------------------------------------------------


def _is_variances_note(event) -> bool:
    return event.detail.startswith("attaching variances")


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_progress_with_roi_matches_progress_without_roi(frames, tmp_path, pipeline):
    """Every stage reports the same counts and totals with an ROI; only the variances notes name the smaller frames."""
    inputs = frames["tiff"]
    groups = {f: [inputs[f]["run0"], inputs[f]["run1"]] for f in FAMILIES}
    roi = ROIS["interior"]
    events = {"whole": [], "roi": []}
    for name, region in (("whole", None), ("roi", roi)):
        _run(
            pipeline,
            tmp_path / f"{name}.h5",
            groups["sample"],
            groups["ob"],
            groups["dark"],
            roi=region,
            progress=events[name].append,
        )

    for family, stage in LOAD_STAGES.items():
        per_file = [e for e in events["roi"] if e.stage == stage and not _is_variances_note(e)]
        total = sum(len(run) for run in groups[family])
        assert [(e.completed, e.total) for e in per_file] == [(i, total) for i in range(1, total + 1)]
        assert sorted(e.detail for e in per_file) == _names(*groups[family])
    assert [(e.stage, e.completed, e.total) for e in events["roi"]] == [
        (e.stage, e.completed, e.total) for e in events["whole"]
    ]
    # Per-file detail is in completion order, so it is compared as a multiset.
    assert sorted((e.stage, e.detail) for e in events["roi"] if not _is_variances_note(e)) == sorted(
        (e.stage, e.detail) for e in events["whole"] if not _is_variances_note(e)
    )
    x0, y0, x1, y1 = roi
    whole_notes = [e.detail for e in events["whole"] if _is_variances_note(e)]
    roi_notes = [e.detail for e in events["roi"] if _is_variances_note(e)]
    assert len(roi_notes) == len(whole_notes) == 2 * len(FAMILIES)
    assert all(f" of {NX} x {NY} px " in note for note in whole_notes)
    assert all(f" of {x1 - x0} x {y1 - y0} px " in note for note in roi_notes)


# --- Memory ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("roi", [None, ROIS["interior"]], ids=["no_roi", "roi"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_per_run_stacks_are_released_once_combined(frames, tmp_path, monkeypatch, pipeline, roi):
    """With two runs per family, no per-run stack is still alive when dead pixels are detected."""
    module = PIPELINE_MODULES[pipeline]
    loaded = []
    alive = []
    load_runs = module.load_runs
    detect = module.detect_dead_pixels

    def tracking_load_runs(groups, **kwargs):
        runs = load_runs(groups, **kwargs)
        loaded.extend(weakref.ref(run) for run in runs)
        return runs

    def counting_live_runs(sample, *args, **kwargs):
        gc.collect()
        alive.append(sum(ref() is not None for ref in loaded))
        return detect(sample, *args, **kwargs)

    monkeypatch.setattr(module, "load_runs", tracking_load_runs)
    monkeypatch.setattr(module, "detect_dead_pixels", counting_live_runs)
    inputs = frames["tiff"]
    groups = {f: [inputs[f]["run0"], inputs[f]["run1"]] for f in FAMILIES}

    _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], groups["dark"], roi=roi)

    assert len(loaded) == 2 * len(FAMILIES)
    assert alive == [0]


# --- combine_owned_runs ------------------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_combine_owned_runs_returns_a_single_run_as_is(frames, pipeline):
    run = load_stack(frames["tiff"]["sample"]["run0"])
    assert combine_owned_runs([run], **_combine_kwargs(pipeline, "sample")) is run


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_combine_owned_runs_combines_several_runs_like_combine_runs(frames, pipeline, family):
    inputs = frames["tiff"][family]
    runs = [load_stack(inputs["run0"]), load_stack(inputs["run1"])]
    kwargs = _combine_kwargs(pipeline, family)
    assert sc.identical(combine_owned_runs(runs, **kwargs), combine_runs(runs, **kwargs), equal_nan=True)


def _runs_with_exposure_offset(frames, offset: float) -> list[sc.DataArray]:
    inputs = frames["tiff"]["sample"]
    runs = [load_stack(inputs["run0"]), load_stack(inputs["run1"])]
    shifted = runs[1].coords["ExposureTime"].copy()
    shifted.values = shifted.values + offset
    runs[1].coords["ExposureTime"] = shifted
    runs[1].coords.set_aligned("ExposureTime", False)
    return runs


def test_combine_owned_runs_raises_the_metadata_mismatch_error_of_combine_runs(frames):
    runs = _runs_with_exposure_offset(frames, 0.05)
    kwargs = _combine_kwargs("mars", "sample")
    with pytest.raises(ValueError) as reference:
        combine_runs(runs, **kwargs)
    assert str(reference.value).startswith("Metadata key 'ExposureTime' does not match between run 1 and base run")

    with pytest.raises(ValueError) as raised:
        combine_owned_runs(runs, **kwargs)
    assert str(raised.value) == str(reference.value)


def test_combine_owned_runs_forwards_the_metadata_tolerance(frames):
    runs = _runs_with_exposure_offset(frames, 0.05)
    kwargs = _combine_kwargs("mars", "sample", metadata_match_atol=0.1)
    assert sc.identical(combine_owned_runs(runs, **kwargs), combine_runs(runs, **kwargs), equal_nan=True)


def test_combine_owned_runs_rejects_no_runs():
    with pytest.raises(ValueError, match="No runs provided for combination"):
        combine_owned_runs([])
