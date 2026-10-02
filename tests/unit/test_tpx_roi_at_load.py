"""
Tests for cropping to the ROI as frames load in the VENUS TPX1 and TPX3 histogram pipelines.

The pipelines crop every sample and open-beam frame to ``roi`` while it is read. Their output must
equal loading and combining whole frames and cropping afterwards, which these tests rebuild from public
functions: ``load_metadata`` + ``load_tiff_stack`` -> ``combine_runs`` -> ``apply_roi`` -> dead (and,
for TPX3 histogram, hot) pixel detection -> spatial and TOF rebinning -> moving window ->
normalization or the region spectrum -> air-region correction -> wavelength and energy. Runs of one
family with different shapes must raise the message ``combine_runs`` raises, while that family loads;
sample and open-beam frames of different size a message naming both.

Frames are non-square and, apart from the dead and hot pixels below, every pixel holds a different
value, so a shifted or transposed crop changes values as well as shape.
"""

import gc
import json
import weakref
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import tifffile
from loguru import logger
from scitiff.io import load_scitiff

import neunorm.loaders._frame_stack as frame_stack
from neunorm.data_models.moving_window import MovingWindow
from neunorm.data_models.roi import as_region_list, as_roi_bounds
from neunorm.loaders import tiff_loader
from neunorm.loaders.metadata_loader import load_metadata
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.pipelines import _tof_spine
from neunorm.pipelines import venus_tpx1 as tpx1_module
from neunorm.pipelines import venus_tpx3_histogram as tpx3h_module
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline
from neunorm.processing.air_region_corrector import apply_air_region_correction
from neunorm.processing.moving_window import moving_window
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.roi_clipper import apply_roi
from neunorm.processing.run_combiner import combine_runs
from neunorm.processing.spatial_rebinner import rebin_spatial
from neunorm.processing.spectrum_reducer import normalize_roi_spectrum
from neunorm.tof.coordinate_converter import convert_tof_to_energy, convert_tof_to_wavelength
from neunorm.tof.histogram_rebinner import rebin_tof
from neunorm.tof.pixel_detector import detect_dead_pixels, detect_hot_pixels
from neunorm.tof.statistics_analyzer import analyze_statistics
from neunorm.utils.constants import VENUS_FLIGHT_PATH_M
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
    resolve_progress,
)

# Full frames are 24 rows by 30 columns; "small" runs come from a detector of another size.
NY, NX = 24, 30
SMALL_NY, SMALL_NX = 16, 20
N_FRAMES = 6
FAMILIES = ("sample", "ob")
BASE_COUNTS = {"sample": 5000, "ob": 20000}
PROTON_CHARGE = {"sample": 1000.0, "ob": 2000.0}
# How the pipelines name each family in their combine progress notes.
COMBINE_LABEL = {"sample": "sample", "ob": "open-beam"}

ROIS = {
    "interior": (4, 6, 28, 18),  # 24 x 12
    "top_left": (0, 0, 12, 6),
    "bottom_right": (18, 12, NX, NY),  # 12 x 12
    "full_frame": (0, 0, NX, NY),
}
ROI_FITS_BOTH_SIZES = (2, 3, 10, 9)  # inside the full frame and the small one

# (y, x) in full-frame indices: one of each inside every ROI, zero or very bright in every frame.
DEAD = ((8, 10), (2, 3), (20, 25))
HOT = ((10, 15), (4, 8), (15, 20))
HOT_COUNTS = 60000

PIPELINES = {"tpx1": run_venus_tpx1_pipeline, "tpx3h": run_venus_tpx3_histogram_pipeline}
PIPELINE_MODULES = {"tpx1": tpx1_module, "tpx3h": tpx3h_module}
FLIGHT_PATH = sc.scalar(VENUS_FLIGHT_PATH_M, unit="m")

# Every option is in post-crop pixels and fits every ROI above.
VARIANTS = {
    "image": {},
    "tiff": {"_suffix": ".tiff"},
    "tiff_one_file_per_image": {"_suffix": ".tiff", "tiff_one_file_per_image": True},
    "rebin_by_tof_factor": {"rebin_by_tof": 2},
    "rebin_by_tof_recommended": {"rebin_by_tof": True},
    "rebin_by_tof_list_median": {"rebin_by_tof": [[0, 2], [3, 6]], "rebin_reduction": "median"},
    "rebin_by_spatial": {"rebin_by_spatial": 2},
    "rebin_by_spatial_xy": {"rebin_by_spatial": (2, 3)},
    "air_roi": {"air_roi": (0, 0, 3, 3)},
    "moving_window": {"moving_window": MovingWindow(x=3, y=3)},
    "spectrum": {"_suffix": ".txt", "spectrum_roi": [(1, 1, 4, 4), (5, 2, 7, 5)]},
}


def _frame(family: str, shape: tuple[int, int], run: int, index: int) -> np.ndarray:
    """A uint16 frame whose pixels differ from each other and from every other frame of the family."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = (BASE_COUNTS[family] + nx * yy + xx + 997 * index + 3001 * run).astype(np.uint16)
    if shape == (NY, NX):
        for y, x in DEAD:
            data[y, x] = 0
        for y, x in HOT:
            data[y, x] = HOT_COUNTS
    return data


def _write_nexus(path: Path, family: str, run: int, n_frames: int, detector: str) -> None:
    """A NeXus file with what both pipelines read; the TPX3 flavor adds the TOF binning."""
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_dataset("proton_charge", data=[PROTON_CHARGE[family] * (run + 1)])
        entry.create_dataset("duration", data=[60.0])
        logs = entry.create_group("DASlogs")
        logs.create_group("BL10:Exp:IM:ImageFilePath").create_dataset("value", data=[[b"raw/unused"]])
        logs.create_group("BL10:Det:TH:DSPT1:TIDelay").create_dataset("average_value", data=[5000])
        logs.create_group("BL10:Exp:Det").create_dataset("value_strings", data=[[detector.encode()]])
        if detector == "MCP TPX3":
            logs.create_group("BL10:Det:T1:TSStart_RBV").create_dataset("value", data=[100])
            logs.create_group("BL10:Det:T1:TSBinSize_RBV").create_dataset("value", data=[5])
            logs.create_group("BL10:Det:T1:TSSize_RBV").create_dataset("value", data=[n_frames])


def _write_run(root: Path, family: str, run: int, n_frames: int, shape: tuple[int, int]) -> dict:
    """One run: its TIFFs and spectra sidecar in their own directory, and one NeXus file per pipeline."""
    directory = root / f"{family}_r{run}"
    directory.mkdir()
    tiffs = []
    for index in range(n_frames):
        path = directory / f"{family}_r{run}_{index:03}.tiff"
        tifffile.imwrite(path, _frame(family, shape, run, index))
        tiffs.append(path)
    with open(directory / f"{family}_r{run}_Spectra.txt", "w") as f:
        for index in range(n_frames):
            f.write(f"{1e-4 * (index + 1):.6e} 0\n")
    nexus = {}
    for pipeline, detector in (("tpx1", "MCP TPX1"), ("tpx3h", "MCP TPX3")):
        nexus[pipeline] = root / f"{family}_r{run}_{pipeline}.nxs.h5"
        _write_nexus(nexus[pipeline], family, run, n_frames, detector)
    return {"tiffs": tiffs, "nexus": nexus}


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Per family: two full-size runs, a run of another frame size, a run with one frame fewer and a
    run whose frames are full-size with rows and columns swapped."""
    root = tmp_path_factory.mktemp("tpx_roi_at_load")
    shapes = {"run0": (NY, NX), "run1": (NY, NX), "small": (SMALL_NY, SMALL_NX), "short": (NY, NX)}
    shapes["transposed"] = (NX, NY)
    inputs = {}
    for family in FAMILIES:
        inputs[family] = {}
        for run, (name, shape) in enumerate(shapes.items()):
            n = N_FRAMES - 1 if name == "short" else N_FRAMES
            inputs[family][name] = _write_run(root, family, run, n, shape)
    return inputs


def _arguments(pipeline: str, sample_runs: list[dict], ob_runs: list[dict]) -> dict:
    return {
        "sample_hdf5_paths": [run["nexus"][pipeline] for run in sample_runs],
        "ob_hdf5_paths": [run["nexus"][pipeline] for run in ob_runs],
        "sample_tiff_paths": [run["tiffs"] for run in sample_runs],
        "ob_tiff_paths": [run["tiffs"] for run in ob_runs],
    }


def _run(pipeline: str, output_path: Path, sample_runs, ob_runs, **kwargs) -> sc.DataArray:
    return PIPELINES[pipeline](output_path=output_path, **_arguments(pipeline, sample_runs, ob_runs), **kwargs)


# --- The crop-after-combine reference, from public functions ----------------------------------------------


def _load_whole_run(pipeline: str, run: dict) -> sc.DataArray:
    """One run's whole frames with the coordinates the pipeline attaches."""
    tiffs = run["tiffs"]
    data = load_tiff_stack(tiffs).rename_dims({"N_image": "tof"})
    if pipeline == "tpx1":
        metadata = load_metadata(run["nexus"]["tpx1"], read_spectra_tof=True, image_dir=Path(tiffs[0]).parent)
        left = metadata.pop("spectra_tof")
        edges = np.append(left.values, 2 * left.values[-1] - left.values[-2])
        data.coords["tof"] = sc.array(dims=["tof"], values=edges, unit=left.unit)
    else:
        metadata = load_metadata(run["nexus"]["tpx3h"])
        data.coords["tof"] = (
            sc.arange("tof", metadata["tof_num_bins"] + 1) * metadata["tof_bin_size"] + metadata["tof_start"]
        )
    for key, value in metadata.items():
        data.coords[key] = value
        data.coords.set_aligned(key, False)
    return data


def _combine(runs: list[sc.DataArray]) -> sc.DataArray:
    return combine_runs(
        runs,
        metadata_keys_to_sum=["proton_charge", "duration"],
        metadata_check_match=["detector_time_offset", "detector"],
        normalize_by_runs=True,
    )


def _attach_masks(pipeline: str, target: sc.DataArray, source: sc.DataArray) -> None:
    target.masks["dead_pixels"] = detect_dead_pixels(source)
    if pipeline == "tpx3h":
        target.masks["hot_pixels"] = detect_hot_pixels(source)


def _crop_after_combine(pipeline: str, sample_runs, ob_runs, *, roi=None, **options) -> sc.DataArray:
    """The pipeline's result with whole frames loaded and combined, and the ROI applied afterwards."""
    sample = _combine([_load_whole_run(pipeline, run) for run in sample_runs])
    ob = _combine([_load_whole_run(pipeline, run) for run in ob_runs])
    if roi is not None:
        roi = as_roi_bounds(roi)
        sample = apply_roi(sample, roi)
        ob = apply_roi(ob, roi)

    _attach_masks(pipeline, sample, ob)
    factor = options.get("rebin_by_spatial")
    if factor is not None:
        sample = rebin_spatial(sample, factor)
        ob = rebin_spatial(ob, factor)
        # TPX1 re-detects the masks from the open beam after a spatial rebin, TPX3 histogram from the sample.
        _attach_masks(pipeline, sample, ob if pipeline == "tpx1" else sample)
    spec = options.get("rebin_by_tof", False)
    if spec is not False:
        if spec is True:
            spec = analyze_statistics(ob).recommended_rebinning
        sample = rebin_tof(sample, spec, reduction=options.get("rebin_reduction"))
        ob = rebin_tof(ob, spec, reduction=options.get("rebin_reduction"))

    charges = {"proton_charge_sample": sample.coords["proton_charge"], "proton_charge_ob": ob.coords["proton_charge"]}
    if options.get("spectrum_roi") is not None:
        regions = as_region_list(options["spectrum_roi"], arg_name="spectrum_roi")
        result = normalize_roi_spectrum(sample, ob, regions, spectrum_roi_strict=True, **charges)
    else:
        window = options.get("moving_window")
        if window is not None:
            filtered = moving_window(sample, window.sizes(), kind=window.kind, mode=window.mode)
            ob = moving_window(ob, window.sizes(), kind=window.kind, mode=window.mode, masks=sample.masks)
            sample = filtered
        result = normalize_transmission(sample=sample, ob=ob, **charges)
        if options.get("air_roi") is not None:
            result = apply_air_region_correction(result, options["air_roi"])
    offset = sample.coords["detector_time_offset"]
    result.coords["wavelength"] = convert_tof_to_wavelength(result.coords["tof"], FLIGHT_PATH, offset)
    result.coords["energy"] = convert_tof_to_energy(result.coords["tof"], FLIGHT_PATH, offset)
    return result


def _as_written_to_tiff(expected: sc.DataArray) -> sc.DataArray:
    """``expected`` as a TIFF run returns it: ``tof`` renamed ``t`` and every mask merged into one."""
    expected = expected.rename_dims({"tof": "t"})
    merged = np.zeros(tuple(expected.sizes.values()), dtype=bool)
    for mask in expected.masks.values():
        merged |= sc.broadcast(mask, sizes=expected.sizes).transpose(expected.dims).values
    expected.masks.clear()
    expected.masks["scitiff-mask"] = sc.array(dims=expected.dims, values=merged)
    return expected


def _assert_written(output_path: Path, suffix: str, options: dict, result: sc.DataArray, roi) -> None:
    """The written file holds the returned data, detector-index coordinates and the ROI provenance.

    HDF5 and TIFF store the values as float32.
    """
    values = result.values.astype(np.float32)
    if suffix == ".txt":
        table = np.loadtxt(output_path, delimiter=",", skiprows=1)
        np.testing.assert_allclose(table[:, 1], result.values, rtol=0, atol=1e-6)
        output_path = output_path.with_suffix(".hdf5")
    if suffix == ".tiff":
        if options.get("tiff_one_file_per_image"):
            written = sorted(output_path.parent.glob(f"{output_path.stem}_*.tiff"))
            assert len(written) == result.sizes["t"]
            images = [load_scitiff(path)["image"].values.reshape(values.shape[1:]) for path in written]
            np.testing.assert_array_equal(np.stack(images), values)
            extra = load_scitiff(written[0])["extra"]
        else:
            loaded = load_scitiff(output_path)
            np.testing.assert_array_equal(loaded["image"].values, values)
            extra = loaded["extra"]
        if roi is None:
            assert "roi_applied" not in extra
        else:
            assert json.loads(extra["roi_applied"]) == list(roi)
        return
    with h5py.File(output_path, "r") as hf:
        np.testing.assert_array_equal(hf["transmission"][()], values)
        # A spatial rebin drops the pixel-index coordinates; a spectrum has none.
        for dim in ("x", "y"):
            if dim in result.coords:
                np.testing.assert_array_equal(hf[dim][()], result.coords[dim].values)
        if roi is None:
            assert "metadata/roi_applied" not in hf
        else:
            np.testing.assert_array_equal(hf["metadata/roi_applied"][()], roi)


def _spy_on_combined_runs(monkeypatch, pipeline: str) -> list[list[tuple]]:
    """Record the shape of every run each family hands to the run combine."""
    module = PIPELINE_MODULES[pipeline]
    seen = []
    combine = module.combine_owned_runs

    def recording(runs, **kwargs):
        seen.append([run.shape for run in runs])
        return combine(runs, **kwargs)

    monkeypatch.setattr(module, "combine_owned_runs", recording)
    return seen


# --- Output equals cropping after the combine --------------------------------------------------------------


@pytest.mark.parametrize("n_runs", [1, 2], ids=["1_run", "2_runs"])
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_crop_at_load_matches_crop_after_combine(frames, tmp_path, monkeypatch, pipeline, variant, n_runs):
    """For every ROI, and without one, the result and the file equal the crop-after-combine reference, and
    the runs reaching the run combine are already cropped."""
    sample_runs = [frames["sample"][f"run{r}"] for r in range(n_runs)]
    ob_runs = [frames["ob"][f"run{r}"] for r in range(n_runs)]
    options = dict(VARIANTS[variant])
    suffix = options.pop("_suffix", ".hdf5")
    combined = _spy_on_combined_runs(monkeypatch, pipeline)

    for name, roi in [("no_roi", None), *ROIS.items()]:
        combined.clear()
        output_path = tmp_path / name / f"out{suffix}"
        output_path.parent.mkdir()

        result = _run(pipeline, output_path, sample_runs, ob_runs, roi=roi, **options)
        expected = _crop_after_combine(pipeline, sample_runs, ob_runs, roi=roi, **options)

        x0, y0, x1, y1 = roi if roi is not None else (0, 0, NX, NY)
        assert combined == [[(N_FRAMES, y1 - y0, x1 - x0)] * n_runs] * 2, name
        if suffix == ".tiff":
            expected = _as_written_to_tiff(expected)
        assert sc.identical(result, expected, equal_nan=True), name
        _assert_written(output_path, suffix, options, result, roi)


# --- Runs of one family with different uncropped shapes ----------------------------------------------------

GUARD_CASES = {
    # name: (runs of the family, in order; roi)
    "sizes_roi_fits_run0_only": (("run0", "small"), ROIS["interior"]),
    "sizes_roi_fits_both": (("run0", "small"), ROI_FITS_BOTH_SIZES),
    "sizes_no_roi": (("run0", "small"), None),
    "frame_counts": (("run0", "short"), ROIS["interior"]),
    "frame_counts_no_roi": (("run0", "short"), None),
    "sizes_third_run": (("run0", "run1", "small"), ROIS["interior"]),
}


class _Timeline:
    """Frame reads, metadata reads and log records, in the order they happened."""

    def __init__(self):
        self.events: list[tuple] = []

    def reads(self, family: str) -> list[str]:
        """TIFF names read for ``family``, sorted (frames of one stack are read concurrently)."""
        return sorted(event[1] for event in self.events if event[0] == "read" and event[1].startswith(f"{family}_"))

    def all_reads(self) -> list[tuple]:
        return [event for event in self.events if event[0] in ("read", "metadata")]

    def messages(self, level: str) -> list[str]:
        return [event[2] for event in self.events if event[0] == "log" and event[1] == level]

    def clear(self) -> None:
        self.events.clear()


@pytest.fixture
def timeline(monkeypatch):
    """Record every TIFF frame read, every metadata read by either pipeline, and every log record at INFO+."""
    record = _Timeline()
    read = tiff_loader._read_tiff_frame

    def recording(path):
        record.events.append(("read", Path(path).name))
        return read(path)

    monkeypatch.setattr(tiff_loader, "_read_tiff_frame", recording)
    for module in PIPELINE_MODULES.values():
        load = module.load_metadata

        def recording_metadata(path, *args, _load=load, **kwargs):
            record.events.append(("metadata", Path(path).name))
            return _load(path, *args, **kwargs)

        monkeypatch.setattr(module, "load_metadata", recording_metadata)
    sink_id = logger.add(
        lambda message: record.events.append(("log", message.record["level"].name, message.record["message"])),
        level="INFO",
    )
    yield record
    logger.remove(sink_id)


def _names(*runs: dict) -> list[str]:
    return sorted(path.name for run in runs for path in run["tiffs"])


def _combine_notes(events) -> list[str]:
    return [event.detail for event in events if event.stage == STAGE_COMBINE_RUNS and event.detail]


@pytest.mark.parametrize("case", list(GUARD_CASES))
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_runs_of_different_shape_raise_combine_message_while_the_family_loads(
    frames, tmp_path, timeline, pipeline, family, case
):
    """The error names the uncropped shapes, exactly as ``combine_runs`` does, before anything is combined."""
    run_names, roi = GUARD_CASES[case]
    groups = {f: [frames[f]["run0"]] for f in FAMILIES}
    groups[family] = [frames[family][name] for name in run_names]

    whole = [_load_whole_run(pipeline, run) for run in groups[family]]
    bad = len(whole) - 1
    with pytest.raises(ValueError) as combined:
        _combine(whole)
    message = (
        f"Run {bad} has shape {whole[bad].shape} and dims ('tof', 'y', 'x'),"
        f" expected shape {whole[0].shape} and dims ('tof', 'y', 'x')"
    )
    assert str(combined.value) == message

    timeline.clear()
    events = []
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", groups["sample"], groups["ob"], roi=roi, progress=events.append)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == [message]
    # The bad run's frame 0 is enough to reject an ROI that does not fit it; otherwise the whole run is read.
    _, bad_ny, bad_nx = whole[bad].shape
    bad_run = groups[family][bad]
    fits = roi is None or (roi[2] <= bad_nx and roi[3] <= bad_ny)
    read_of_bad_run = bad_run["tiffs"] if fits else bad_run["tiffs"][:1]
    if family == "ob":
        assert timeline.reads("sample") == _names(*groups["sample"])
    else:
        assert timeline.reads("ob") == []
    assert timeline.reads(family) == sorted([*_names(*groups[family][:bad]), *(p.name for p in read_of_bad_run)])
    assert _combine_notes(events) == []


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_roi_fitting_a_later_run_but_not_run0_raises_on_run0(frames, tmp_path, timeline, pipeline):
    """The sample's smaller run 0 fails the ROI with the message ``apply_roi`` raises, on its first frame."""
    roi = ROIS["interior"]
    with pytest.raises(ValueError) as cropped:
        apply_roi(load_tiff_stack(frames["sample"]["small"]["tiffs"]), roi)
    message = f"ROI (x1=28, y1=18) exceeds data size (x={SMALL_NX}, y={SMALL_NY})"
    assert str(cropped.value) == message
    sample = [frames["sample"]["small"], frames["sample"]["run0"]]

    timeline.clear()
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", sample, [frames["ob"]["run0"]], roi=roi)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == []
    assert timeline.reads("sample") == [frames["sample"]["small"]["tiffs"][0].name]
    assert timeline.reads("ob") == []


# --- Families of different frame size ----------------------------------------------------------------------

FRAME_SIZE_CASES = {
    # name: (family whose frames differ, its run, roi)
    "small_ob_no_roi": ("ob", "small", None),
    "small_ob_roi_fits_both": ("ob", "small", ROI_FITS_BOTH_SIZES),
    "small_ob_roi_fits_sample_only": ("ob", "small", ROIS["interior"]),
    "transposed_ob_no_roi": ("ob", "transposed", None),
    "transposed_ob_roi_fits_both": ("ob", "transposed", ROI_FITS_BOTH_SIZES),
    "small_sample_no_roi": ("sample", "small", None),
    "small_sample_roi_fits_both": ("sample", "small", ROI_FITS_BOTH_SIZES),
    "transposed_sample_no_roi": ("sample", "transposed", None),
    "transposed_sample_roi_fits_both": ("sample", "transposed", ROI_FITS_BOTH_SIZES),
}


@pytest.mark.parametrize("case", list(FRAME_SIZE_CASES))
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_sample_and_open_beam_of_different_frame_size_raise_while_the_open_beam_loads(
    frames, tmp_path, timeline, pipeline, case
):
    """The open beam's uncropped frame size must match the sample's; the error names both, logged."""
    odd, run_name, roi = FRAME_SIZE_CASES[case]
    groups = {f: [frames[f]["run0"]] for f in FAMILIES}
    groups[odd] = [frames[odd][run_name]]
    sizes = {f: load_tiff_stack(groups[f][0]["tiffs"]).shape[1:] for f in FAMILIES}
    assert sizes["ob"] != sizes["sample"]
    message = (
        f"Open-beam frames have size (y={sizes['ob'][0]}, x={sizes['ob'][1]}), but sample frames have size"
        f" (y={sizes['sample'][0]}, x={sizes['sample'][1]}); sample and open-beam frames must be the same size"
    )
    output_path = tmp_path / "out.h5"

    timeline.clear()
    events = []
    with pytest.raises(ValueError) as raised:
        _run(pipeline, output_path, groups["sample"], groups["ob"], roi=roi, progress=events.append)

    assert str(raised.value) == message
    assert timeline.messages("ERROR") == [message]
    assert timeline.reads("sample") == _names(*groups["sample"])
    assert STAGE_NORMALIZE not in {event.stage for event in events}
    assert _combine_notes(events) == []
    assert not output_path.exists()


# --- ROIs rejected without reading a file ------------------------------------------------------------------

INVALID_ROIS = {
    "empty": (0, 0, 0, 0),
    "numpy_ints": tuple(np.int64(v) for v in ROIS["interior"]),
    "floats": tuple(float(v) for v in ROIS["interior"]),
    "misordered": (28, 6, 4, 18),
    "negative": (-1, 6, 28, 18),
}


@pytest.mark.parametrize("roi_name", list(INVALID_ROIS))
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_invalid_roi_raises_the_same_error_before_any_read(frames, tmp_path, timeline, pipeline, roi_name):
    roi = INVALID_ROIS[roi_name]
    with pytest.raises(ValueError) as reference:
        apply_roi(load_tiff_stack(frames["sample"]["run0"]["tiffs"]), roi)

    timeline.clear()
    with pytest.raises(ValueError) as raised:
        _run(pipeline, tmp_path / "out.h5", [frames["sample"]["run0"]], [frames["ob"]["run0"]], roi=roi)

    assert type(raised.value) is type(reference.value)
    assert str(raised.value) == str(reference.value)
    assert timeline.all_reads() == []


# --- Memory, log lines and progress ------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_only_roi_sized_stacks_are_allocated(frames, tmp_path, monkeypatch, pipeline):
    """Every stack the loader allocates, and both stacks the shared reduction receives, are ROI-sized."""
    allocations = []
    allocate = frame_stack.allocate_stack

    def spy(dims, shape):
        allocations.append((list(dims), list(shape)))
        return allocate(dims, shape)

    monkeypatch.setattr(frame_stack, "allocate_stack", spy)
    module = PIPELINE_MODULES[pipeline]
    reduce = module.reduce_tof_stacks
    reduced = []

    def recording_reduce(sample, ob, **kwargs):
        reduced.append((sample.shape, ob.shape))
        return reduce(sample, ob, **kwargs)

    monkeypatch.setattr(module, "reduce_tof_stacks", recording_reduce)
    runs = {f: [frames[f]["run0"], frames[f]["run1"]] for f in FAMILIES}
    x0, y0, x1, y1 = ROIS["interior"]

    _run(pipeline, tmp_path / "out.h5", runs["sample"], runs["ob"], roi=ROIS["interior"])

    assert allocations == [(["N_image", "y", "x"], [N_FRAMES, y1 - y0, x1 - x0])] * 4
    assert reduced == [((N_FRAMES, y1 - y0, x1 - x0),) * 2]


@pytest.mark.parametrize("roi", [None, ROIS["interior"]], ids=["no_roi", "roi"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_per_run_stacks_are_released_once_combined(frames, tmp_path, monkeypatch, pipeline, roi):
    """With two runs per family, no per-run stack is still alive when dead pixels are detected."""
    module = PIPELINE_MODULES[pipeline]
    loaded = []
    alive = []
    load_runs = module.load_runs
    detect = _tof_spine.detect_dead_pixels

    def tracking_load_runs(groups, **kwargs):
        runs = load_runs(groups, **kwargs)
        loaded.extend(weakref.ref(run) for run in runs)
        return runs

    def counting_live_runs(source):
        gc.collect()
        alive.append(sum(ref() is not None for ref in loaded))
        return detect(source)

    monkeypatch.setattr(module, "load_runs", tracking_load_runs)
    monkeypatch.setattr(_tof_spine, "detect_dead_pixels", counting_live_runs)
    runs = {f: [frames[f]["run0"], frames[f]["run1"]] for f in FAMILIES}

    _run(pipeline, tmp_path / "out.h5", runs["sample"], runs["ob"], roi=roi)

    assert len(loaded) == 4
    assert alive == [0]


@pytest.mark.parametrize("roi", [None, ROIS["interior"]], ids=["no_roi", "roi"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_a_single_run_reaches_the_reduction_uncopied(frames, tmp_path, monkeypatch, pipeline, roi):
    """With one run per family, the shared reduction receives the loaded sample and open-beam runs themselves."""
    module = PIPELINE_MODULES[pipeline]
    loaded = {}
    received = []
    load_runs = module.load_runs
    reduce = module.reduce_tof_stacks

    def recording_load_runs(groups, *, family, **kwargs):
        loaded[family] = load_runs(groups, family=family, **kwargs)
        return loaded[family]

    def recording_reduce(sample, ob, **kwargs):
        received.append((sample, ob))
        return reduce(sample, ob, **kwargs)

    monkeypatch.setattr(module, "load_runs", recording_load_runs)
    monkeypatch.setattr(module, "reduce_tof_stacks", recording_reduce)

    _run(pipeline, tmp_path / "out.h5", [frames["sample"]["run0"]], [frames["ob"]["run0"]], roi=roi)

    assert {family: len(runs) for family, runs in loaded.items()} == {"sample": 1, "open-beam": 1}
    assert len(received) == 1
    sample, ob = received[0]
    assert sample is loaded["sample"][0]
    assert ob is loaded["open-beam"][0]


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_one_crop_log_line_per_family_before_it_loads(frames, tmp_path, timeline, pipeline):
    roi = ROIS["interior"]
    runs = {f: [frames[f]["run0"], frames[f]["run1"]] for f in FAMILIES}

    _run(pipeline, tmp_path / "out.h5", runs["sample"], runs["ob"], roi=roi)

    crop_line = ("log", "INFO", f"Cropping frames to ROI {roi} as they are loaded")
    # Collapsed to runs of the same kind, the record reads: crop line, that family's files, next crop line.
    sequence = []
    for event in timeline.events:
        if event == crop_line:
            kind = "crop"
        elif event[0] == "read":
            kind = event[1].split("_")[0]
        else:
            continue
        if not sequence or sequence[-1] != kind:
            sequence.append(kind)
    assert sequence == ["crop", "sample", "crop", "ob"]
    assert not any(message.startswith("Applying ROI") for message in timeline.messages("INFO"))


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_progress_with_roi_matches_progress_without_roi(frames, tmp_path, pipeline):
    """Every stage reports the same counts and totals with an ROI; only the variances notes name the smaller frames."""
    runs = {f: [frames[f]["run0"], frames[f]["run1"]] for f in FAMILIES}
    roi = ROIS["interior"]
    events = {"whole": [], "roi": []}
    for name, region in (("whole", None), ("roi", roi)):
        _run(pipeline, tmp_path / f"{name}.h5", runs["sample"], runs["ob"], roi=region, progress=events[name].append)

    def is_variances_note(event) -> bool:
        return event.detail.startswith("attaching variances")

    for family, stage in (("sample", STAGE_LOAD_SAMPLE), ("ob", STAGE_LOAD_OB)):
        per_file = [e for e in events["roi"] if e.stage == stage and not is_variances_note(e)]
        total = 2 * N_FRAMES
        assert [(e.completed, e.total) for e in per_file] == [(i, total) for i in range(1, total + 1)]
        assert sorted(e.detail for e in per_file) == _names(*runs[family])
    assert [(e.stage, e.completed, e.total) for e in events["roi"]] == [
        (e.stage, e.completed, e.total) for e in events["whole"]
    ]
    x0, y0, x1, y1 = roi
    whole_notes = [e.detail for e in events["whole"] if is_variances_note(e)]
    roi_notes = [e.detail for e in events["roi"] if is_variances_note(e)]
    assert len(roi_notes) == len(whole_notes) == 4
    assert all(f" of {NX} x {NY} px " in note for note in whole_notes)
    assert all(f" of {x1 - x0} x {y1 - y0} px " in note for note in roi_notes)


# --- The shared reduction ----------------------------------------------------------------------------------


def test_the_shared_reduction_does_not_crop(frames, tmp_path):
    """``reduce_tof_stacks`` takes stacks already cropped to ``roi`` and only records it.

    The bounds passed here extend past the cropped stacks, so a second crop would raise.
    """
    roi = ROIS["interior"]
    sample = apply_roi(_combine([_load_whole_run("tpx1", frames["sample"]["run0"])]), roi)
    ob = apply_roi(_combine([_load_whole_run("tpx1", frames["ob"]["run0"])]), roi)
    output_path = tmp_path / "out.h5"
    with resolve_progress(False) as run_progress:
        result = _tof_spine.reduce_tof_stacks(
            sample,
            ob,
            output_path=output_path,
            profile=tpx1_module._TPX1_PROFILE,
            metadata={},
            roi=roi,
            flight_path=FLIGHT_PATH,
            run_progress=run_progress,
        )

    x0, y0, x1, y1 = roi
    assert result.sizes == {"tof": N_FRAMES, "y": y1 - y0, "x": x1 - x0}
    np.testing.assert_array_equal(result.coords["x"].values, np.arange(x0, x1))
    with h5py.File(output_path, "r") as hf:
        np.testing.assert_array_equal(hf["metadata/roi_applied"][()], roi)
