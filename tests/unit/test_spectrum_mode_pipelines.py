"""End-to-end tests for resonance mode on the three VENUS TOF pipelines.

The unit tests elsewhere pin the reduction and the writer in isolation. This file runs the whole
pipeline and then **reads the file off disk**, because that is the deliverable a user actually gets and
because a spectrum can be wrong in ways a DataArray assertion does not see — a plausible-looking number
in the wrong pixel frame, a bin_index column that has been renumbered, or a TIFF full of 1x1-pixel
images written where an error was due.

Expected transmissions are literal arithmetic over the synthetic frame values, never a second call into
the pipeline. The frames are uniform per image, so ``(sample_value / pc_sample) / (ob_value / pc_ob)``
is exact.
"""

import re

import h5py
import numpy as np
import pytest
from loguru import logger
from test_progress_pipelines import (
    _ccd_tiffs,
    _spectra_file,
    _tpx3_event_file,
    _venus_metadata_nexus,
)

from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline
from neunorm.utils.progress import STAGE_EXPORT, STAGE_REDUCE_SPECTRUM

_DETECTOR = 32
_FRAMES = 6
_SAMPLE_BASE = 81.0
_OB_BASE = 99.0
_PC_SAMPLE = 12345.0
_PC_OB = 24690.0
_REGION = (10, 10, 26, 26)

#: Frame i holds a uniform ``base + i``. ``combine_runs`` normalizes each run by its proton charge, so
#: the transmission of frame i is ``((81+i)/12345) / ((99+i)/24690)`` = ``2*(81+i)/(99+i)``.
_EXPECTED = 2.0 * (_SAMPLE_BASE + np.arange(_FRAMES)) / (_OB_BASE + np.arange(_FRAMES))

#: The ASCII columns are written at six decimal places, so a value read back from the file matches the
#: array to half of the last digit. This is the settled format's precision, not slack in the writer.
_ASCII_ATOL = 5e-7


def _collect_logs():
    """Capture loguru records, returning ``(messages, remove)``."""
    messages: list[str] = []
    sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
    return messages, lambda: logger.remove(sink_id)


def _events():
    """Progress events plus the sink to hand a pipeline."""
    captured = []
    return captured, captured.append


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------


@pytest.fixture
def tpx1_inputs(tmp_path):
    """TIFF frames plus the co-located ``*_Spectra.txt`` sidecars and NeXus metadata TPX1 reads."""
    sample_dir, ob_dir = tmp_path / "sample", tmp_path / "ob"
    sample_dir.mkdir()
    ob_dir.mkdir()
    left_edges = [round(0.1 * (i + 1), 1) for i in range(_FRAMES)]
    _spectra_file(sample_dir / "sample_Spectra.txt", left_edges)
    _spectra_file(ob_dir / "ob_Spectra.txt", left_edges)
    return {
        "sample_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "s.h5", _PC_SAMPLE, das_image_path=b"auto")],
        "ob_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "o.h5", _PC_OB, das_image_path=b"auto")],
        "sample_tiff_paths": [_ccd_tiffs(sample_dir, "s", _FRAMES, _SAMPLE_BASE, proton_charge=_PC_SAMPLE)],
        "ob_tiff_paths": [_ccd_tiffs(ob_dir, "o", _FRAMES, _OB_BASE, proton_charge=_PC_OB)],
    }


@pytest.fixture
def histogram_inputs(tmp_path):
    """TIFF frames plus the NeXus TOF binning the histogram pipeline builds its axis from."""
    return {
        "sample_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "hs.h5", _PC_SAMPLE, tof_bins=_FRAMES)],
        "ob_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "ho.h5", _PC_OB, tof_bins=_FRAMES)],
        "sample_tiff_paths": [_ccd_tiffs(tmp_path, "hs", _FRAMES, _SAMPLE_BASE, proton_charge=_PC_SAMPLE)],
        "ob_tiff_paths": [_ccd_tiffs(tmp_path, "ho", _FRAMES, _OB_BASE, proton_charge=_PC_OB)],
    }


@pytest.fixture
def event_inputs(tmp_path):
    """Flood-illuminated event files.

    ``tof_range`` is in NANOSECONDS while ``event_time_offset`` is read as microseconds and scaled up
    by the loader, so events at 100..125 us need a range of (100000, 125000). A range in microseconds
    gives an EMPTY histogram and every assertion below would pass vacuously — the counts are checked
    in :func:`test_event_histogram_is_not_empty` so that cannot happen silently.
    """
    return {
        "binning": BinningConfig(bins=5, bin_space="tof", tof_range=(100000, 125000), use_log_bin=False),
        "sample_paths": [
            _tpx3_event_file(
                tmp_path / "es.h5", 3, bank="bank100_events", offset=1_000_000, proton_charge=_PC_SAMPLE, n_tof=5
            )
        ],
        "ob_paths": [
            _tpx3_event_file(
                tmp_path / "eo.h5", 6, bank="bank100_events", offset=1_000_000, proton_charge=_PC_OB, n_tof=5
            )
        ],
        "detector_shape": (_DETECTOR, _DETECTOR),
    }


def _run_tpx1(inputs, output_path, **kwargs):
    return run_venus_tpx1_pipeline(output_path=output_path, **inputs, **kwargs)


def _run_histogram(inputs, output_path, **kwargs):
    return run_venus_tpx3_histogram_pipeline(output_path=output_path, **inputs, **kwargs)


def _run_event(inputs, output_path, **kwargs):
    return run_venus_tpx3_event_pipeline(output_path=output_path, **inputs, **kwargs)


_RUNNERS = {"venus_tpx1": _run_tpx1, "venus_tpx3_histogram": _run_histogram, "venus_tpx3_event": _run_event}


@pytest.fixture
def pipeline(request, tpx1_inputs, histogram_inputs, event_inputs):
    """A ``(name, callable(output_path, **kwargs))`` pair for the pipeline named by the parametrization."""
    name = request.param
    inputs = {"venus_tpx1": tpx1_inputs, "venus_tpx3_histogram": histogram_inputs, "venus_tpx3_event": event_inputs}[
        name
    ]
    runner = _RUNNERS[name]
    return name, lambda output_path, **kwargs: runner(inputs, output_path, **kwargs)


_ALL = pytest.mark.parametrize("pipeline", list(_RUNNERS), indirect=True)


# --------------------------------------------------------------------------------------------
# the setup is not vacuous
# --------------------------------------------------------------------------------------------


def test_event_histogram_is_not_empty(event_inputs, tmp_path):
    """Guards every event assertion below.

    A ``tof_range`` in the wrong unit yields a histogram with no counts, which makes a transmission of
    NaN and an assertion on "the file exists" pass while proving nothing. Checked here once, loudly.
    """
    spectrum = _run_event(event_inputs, tmp_path / "probe.hdf5", spectrum_roi=_REGION)

    assert np.all(np.isfinite(spectrum.values)), f"empty event histogram: {spectrum.values}"
    assert np.all(spectrum.values > 0.0)


# --------------------------------------------------------------------------------------------
# 1-2. end to end, on every pipeline
# --------------------------------------------------------------------------------------------


@_ALL
def test_spectrum_mode_writes_the_three_column_file_and_an_hdf5_sibling(pipeline, tmp_path):
    """Each pipeline produces the ASCII spectrum plus the HDF5 that carries what three columns cannot.

    Read as text, not as a DataArray: the header line and the row count are the format contract, and
    the HDF5 sibling is where the time axis and the provenance live.
    """
    name, run = pipeline
    output_path = tmp_path / f"{name}.txt"

    spectrum = run(output_path, spectrum_roi=_REGION)

    assert spectrum.dims == ("tof",), "a spectrum, not an image stack"
    assert spectrum.unit == "dimensionless"
    assert spectrum.variances is not None

    lines = output_path.read_text().splitlines()
    assert lines[0] == "bin_index,transmission,uncertainty"
    assert len(lines) - 1 == spectrum.sizes["tof"], "one row per bin, plus the single header line"

    # The settled format writes six decimal places, so the file agrees with the array to half of the
    # last written digit and no closer. A tighter tolerance here would be testing the format's
    # precision rather than the writer's correctness.
    table = np.loadtxt(output_path, skiprows=1, delimiter=",")
    np.testing.assert_allclose(table[:, 1], spectrum.values, atol=_ASCII_ATOL, rtol=0)
    np.testing.assert_allclose(table[:, 2], np.sqrt(spectrum.variances), atol=_ASCII_ATOL, rtol=0)

    sibling = output_path.with_suffix(".hdf5")
    assert sibling.exists(), "the ASCII run must write an HDF5 alongside it"
    with h5py.File(sibling) as f:
        for dataset in ("transmission", "uncertainty", "tof", "spectra_tof"):
            assert dataset in f, f"/{dataset} missing"
        assert f["tof"].shape[0] == spectrum.sizes["tof"] + 1, "N+1 bin edges, so it can be rebinned again"
        np.testing.assert_array_equal(f["metadata/spectrum_roi"][()], np.array(_REGION))


def test_the_transmission_is_the_hand_computed_ratio_of_region_means(tpx1_inputs, tmp_path):
    """The number, not merely the file.

    Uniform frames make the expectation exact arithmetic: frame i is a flat ``81 + i`` against a flat
    ``99 + i``, and each run is divided by its own proton charge, so every bin is
    ``((81+i)/12345) / ((99+i)/24690)``. Any region of a uniform frame has that same mean, so this also
    confirms the region collapse is a mean and not a sum.
    """
    output_path = tmp_path / "pinned.txt"

    spectrum = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION)

    np.testing.assert_allclose(spectrum.values, _EXPECTED, rtol=1e-9)
    table = np.loadtxt(output_path, skiprows=1, delimiter=",")
    np.testing.assert_allclose(table[:, 1], _EXPECTED, rtol=1e-5)


# --------------------------------------------------------------------------------------------
# 3-4. output format selection
# --------------------------------------------------------------------------------------------


@_ALL
def test_an_hdf5_output_path_writes_only_hdf5(pipeline, tmp_path):
    """``.hdf5`` asks for HDF5 only; no stray ASCII file appears beside it."""
    name, run = pipeline
    output_path = tmp_path / f"{name}.hdf5"

    run(output_path, spectrum_roi=_REGION)

    assert output_path.exists()
    assert not output_path.with_suffix(".txt").exists()
    with h5py.File(output_path) as f:
        assert "transmission" in f


@_ALL
def test_a_tiff_output_path_is_refused_and_writes_nothing(pipeline, tmp_path):
    """A spectrum must not be written as a TIFF, and the pipeline must say so rather than let it happen.

    This is not a theoretical guard. Handed a 1-D spectrum after the pipelines' usual ``tof`` -> ``t``
    rename, ``write_tiff_stack`` does NOT fail — it writes a multi-page TIFF of 1x1-pixel images that
    even reads back cleanly, so nothing downstream would flag it. Hence the check that the directory
    is still empty.
    """
    name, run = pipeline
    # An OWN directory: the synthetic input TIFFs live in tmp_path, so globbing there would find those
    # and the check would pass for the wrong reason.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output_path = out_dir / f"{name}.tiff"

    with pytest.raises(ValueError, match="cannot be written as a TIFF"):
        run(output_path, spectrum_roi=_REGION)

    assert not output_path.exists()
    assert list(out_dir.iterdir()) == [], "no file may be written, not even a partial one"


def test_an_unsupported_suffix_is_refused(tpx1_inputs, tmp_path):
    """An unknown extension is an error, as it is in image mode."""
    with pytest.raises(ValueError, match="Unsupported output file format"):
        _run_tpx1(tpx1_inputs, tmp_path / "spectrum.fits", spectrum_roi=_REGION)


# --------------------------------------------------------------------------------------------
# 5. arguments that make no sense for a spectrum
# --------------------------------------------------------------------------------------------


def test_air_roi_cannot_be_combined_with_spectrum_roi(tpx1_inputs, tmp_path):
    """The air correction rescales an image so its air region reads 1.0, which is meaningless for one
    value per bin. Refused explicitly rather than applied to a scalar."""
    with pytest.raises(ValueError, match="air_roi and spectrum_roi cannot be combined"):
        _run_tpx1(tpx1_inputs, tmp_path / "s.txt", spectrum_roi=_REGION, air_roi=(0, 0, 5, 5))


def test_tiff_one_file_per_image_cannot_be_combined_with_spectrum_roi(tpx1_inputs, tmp_path):
    """A per-image TIFF option cannot apply to output that has no images."""
    with pytest.raises(ValueError, match="tiff_one_file_per_image"):
        _run_tpx1(tpx1_inputs, tmp_path / "s.txt", spectrum_roi=_REGION, tiff_one_file_per_image=True)


# --------------------------------------------------------------------------------------------
# 6. frame-index binning through the pipeline
# --------------------------------------------------------------------------------------------


def test_an_integer_rebin_factor_halves_the_rows_and_labels_them_by_first_frame(tpx1_inputs, tmp_path):
    """``rebin_by_tof=2`` over six frames gives three rows indexed 0, 2, 4 — each bin's FIRST frame.

    Not the row index. The column is documented as "the same as the file index if no binning", and the
    only reading of that which survives binning is the first input frame of each bin.
    """
    output_path = tmp_path / "binned.txt"

    spectrum = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION, rebin_by_tof=2)

    assert spectrum.sizes["tof"] == 3
    table = np.loadtxt(output_path, skiprows=1, delimiter=",")
    np.testing.assert_array_equal(table[:, 0].astype(int), np.array([0, 2, 4]))

    # a sum-mode rebin adds the frames, so each bin's transmission is the ratio of the summed pairs
    sample_pairs = (_SAMPLE_BASE + np.arange(_FRAMES)).reshape(3, 2).sum(axis=1)
    ob_pairs = (_OB_BASE + np.arange(_FRAMES)).reshape(3, 2).sum(axis=1)
    np.testing.assert_allclose(table[:, 1], 2.0 * sample_pairs / ob_pairs, rtol=1e-5)


def test_no_binning_makes_the_index_column_the_file_index(tpx1_inputs, tmp_path):
    """The issue's own words: bin_index is the file index when nothing is binned."""
    output_path = tmp_path / "unbinned.txt"

    _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION)

    table = np.loadtxt(output_path, skiprows=1, delimiter=",")
    np.testing.assert_array_equal(table[:, 0].astype(int), np.arange(_FRAMES))


def test_a_gapped_bin_list_leaves_the_index_column_gapped_and_omits_the_dropped_span(tpx1_inputs, tmp_path):
    """``[[0, 2], [4, 6]]`` gives rows indexed 0 and 4, and frames 2-3 get no row at all.

    The file therefore has gaps rather than renumbered rows, which is what keeps a point traceable back
    to the files it came from. The full mapping is recorded in the HDF5 provenance.
    """
    output_path = tmp_path / "gapped.txt"

    spectrum = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION, rebin_by_tof=[[0, 2], [4, 6]])

    assert spectrum.sizes["tof"] == 2, "one output bin per requested range, none for the dropped span"
    table = np.loadtxt(output_path, skiprows=1, delimiter=",")
    indices = table[:, 0].astype(int)
    np.testing.assert_array_equal(indices, np.array([0, 4]))
    assert 2 not in indices, "the dropped 2-3 span must not be renumbered into a row"

    with h5py.File(output_path.with_suffix(".hdf5")) as f:
        np.testing.assert_array_equal(f["metadata/spectrum_bin_first_frame"][()], np.array([0, 4]))


# --------------------------------------------------------------------------------------------
# 7. the coordinate frame the region is resolved in
# --------------------------------------------------------------------------------------------


def test_a_crop_and_a_spatial_rebin_place_the_region_in_the_post_transform_frame(tmp_path):
    """A crop and a spatial rebin both run before the collapse, so the region selects the pixels the
    transformed array holds — and the run warns, because the numbers look plausible either way.

    Pinned with a NON-UNIFORM sample: only a known block carries a raised value, so a region resolved
    in the wrong frame would select different pixels and give a different, still-plausible number. The
    expectation is computed with plain numpy over the detector pixels the transformed region covers.
    """
    sample_dir, ob_dir = tmp_path / "s", tmp_path / "o"
    sample_dir.mkdir()
    ob_dir.mkdir()
    left_edges = [round(0.1 * (i + 1), 1) for i in range(_FRAMES)]
    _spectra_file(sample_dir / "s_Spectra.txt", left_edges)
    _spectra_file(ob_dir / "o_Spectra.txt", left_edges)

    # A raised 8x8 block at detector rows/cols 8..15. Everything else is the flat base.
    raised = np.full((_DETECTOR, _DETECTOR), _SAMPLE_BASE)
    raised[8:16, 8:16] = _SAMPLE_BASE * 2.0
    sample_tiffs = _ccd_tiffs_with_pattern(sample_dir, "s", _FRAMES, raised, _PC_SAMPLE)
    ob_tiffs = _ccd_tiffs(ob_dir, "o", _FRAMES, _OB_BASE, proton_charge=_PC_OB)

    inputs = {
        "sample_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "s.h5", _PC_SAMPLE, das_image_path=b"a")],
        "ob_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "o.h5", _PC_OB, das_image_path=b"a")],
        "sample_tiff_paths": [sample_tiffs],
        "ob_tiff_paths": [ob_tiffs],
    }

    # crop to detector x,y in [4, 28); then rebin 2x2. The raised block at detector 8..15 lands at
    # cropped 4..11, and at post-rebin 2..5. A spectrum_roi of (2, 2, 6, 6) therefore covers exactly
    # the raised block and nothing else.
    crop = (4, 4, 28, 28)
    spectrum_roi = (2, 2, 6, 6)

    messages, remove = _collect_logs()
    try:
        spectrum = _run_tpx1(inputs, tmp_path / "framed.txt", roi=crop, rebin_by_spatial=2, spectrum_roi=spectrum_roi)
    finally:
        remove()

    # Every detector pixel the region covers lies inside the raised block, so the sample region mean is
    # 2*base, and frame i adds i to every pixel (raised block included). The open beam is the flat base.
    expected = 2.0 * (2.0 * _SAMPLE_BASE + np.arange(_FRAMES)) / (_OB_BASE + np.arange(_FRAMES))
    np.testing.assert_allclose(spectrum.values, expected, rtol=1e-9)

    warnings = " ".join(str(m) for m in messages)
    assert "resolved AFTER the roi=" in warnings, "the crop's effect on the frame must be said out loud"
    assert "resolved AFTER rebin_by_spatial" in warnings, "the rebin's effect must be said out loud"


def _ccd_tiffs_with_pattern(directory, prefix, count, pattern, proton_charge):
    """TIFF frames carrying an arbitrary per-pixel pattern, plus ``i`` added to frame ``i``."""
    from PIL import Image

    paths = []
    for index in range(count):
        image = Image.fromarray((pattern + index).astype(np.float32))
        exif = image.getexif()
        exif[65027] = "ExposureTime:30.000000"
        exif[65022] = f"RunNo:{1000 + index}"
        exif[65025] = "ManufacturerStr:DW936_BV"
        exif[65024] = f"IntegratedPCharge:{proton_charge}"
        path = directory / f"{prefix}_{index:05}.tiff"
        image.save(path, exif=exif)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------------------------
# 8. progress
# --------------------------------------------------------------------------------------------


@_ALL
def test_spectrum_mode_reports_the_reduce_and_export_stages_to_completion(pipeline, tmp_path):
    """The new stage appears, and every stage reaches the total the pipeline declared for it."""
    name, run = pipeline
    events, sink = _events()

    run(tmp_path / f"{name}.txt", spectrum_roi=_REGION, progress=sink)

    by_stage: dict[str, list] = {}
    for event in events:
        by_stage.setdefault(event.stage, []).append(event)

    assert STAGE_REDUCE_SPECTRUM in by_stage, "the spectrum reduction must be a named stage"
    assert STAGE_EXPORT in by_stage

    for stage, stage_events in by_stage.items():
        totals = {e.total for e in stage_events}
        assert len(totals) == 1, f"{stage} declared more than one total: {totals}"
        total = totals.pop()
        if total is None:
            continue
        completed = [e.completed for e in stage_events]
        assert completed == sorted(completed), f"{stage} counted backwards: {completed}"
        assert max(completed) == total, f"{stage} stopped at {max(completed)} of {total}"


def test_the_new_stage_does_not_fire_in_image_mode(tpx1_inputs, tmp_path):
    """Image mode's reported stages must be exactly what they were, or every existing progress
    expectation for these pipelines silently changes."""
    events, sink = _events()

    _run_tpx1(tpx1_inputs, tmp_path / "image.hdf5", progress=sink)

    assert STAGE_REDUCE_SPECTRUM not in {e.stage for e in events}


@_ALL
def test_progress_reporting_does_not_change_the_spectrum(pipeline, tmp_path):
    """Reporting is observation, so the numbers must be identical with and without it."""
    name, run = pipeline
    _events_list, sink = _events()

    without = run(tmp_path / "a.txt", spectrum_roi=_REGION)
    with_reporting = run(tmp_path / "b.txt", spectrum_roi=_REGION, progress=sink)

    np.testing.assert_array_equal(without.values, with_reporting.values)
    np.testing.assert_array_equal(without.variances, with_reporting.variances)


# --------------------------------------------------------------------------------------------
# 9. image mode is untouched
# --------------------------------------------------------------------------------------------


@_ALL
def test_without_spectrum_roi_the_pipeline_still_returns_an_image_stack(pipeline, tmp_path):
    """The mode is opt-in. Omitting ``spectrum_roi`` must give the image stack it always gave."""
    name, run = pipeline
    output_path = tmp_path / f"{name}_image.hdf5"

    transmission = run(output_path, roi=None)

    assert "x" in transmission.dims and "y" in transmission.dims
    assert len(transmission.dims) == 3
    with h5py.File(output_path) as f:
        assert f["transmission"].ndim == 3


@_ALL
def test_the_strict_flag_is_accepted_in_spectrum_mode(pipeline, tmp_path):
    """``spectrum_roi_strict`` reaches the reduction on every pipeline. The synthetic open beam is
    positive everywhere, so both settings give the same spectrum here — this pins the wiring, and
    ``test_spectrum_uncertainty.py`` pins the policy itself."""
    name, run = pipeline

    strict = run(tmp_path / f"{name}_strict.txt", spectrum_roi=_REGION, spectrum_roi_strict=True)
    legacy = run(tmp_path / f"{name}_legacy.txt", spectrum_roi=_REGION, spectrum_roi_strict=False)

    np.testing.assert_allclose(strict.values, legacy.values, rtol=1e-12)


def test_a_maskroi_selection_works_end_to_end(tpx1_inputs, tmp_path):
    """An arbitrary-shape region reduces like a rectangle, all the way to the file."""
    from neunorm.data_models.roi import MaskROI

    selection = np.zeros((_DETECTOR, _DETECTOR), dtype=bool)
    selection[10:26, 10:26] = True
    output_path = tmp_path / "mask.txt"

    spectrum = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=MaskROI(selection=selection))

    np.testing.assert_allclose(spectrum.values, _EXPECTED, rtol=1e-9)
    with h5py.File(output_path.with_suffix(".hdf5")) as f:
        recorded = f["metadata/spectrum_roi"][()]
    assert b"mask" in recorded if isinstance(recorded, bytes) else "mask" in str(recorded)


def test_the_completion_log_names_both_written_files(tpx1_inputs, tmp_path):
    """A run that writes two files must report both, or a user looking for the HDF5 will not find it."""
    messages: list[str] = []
    sink_id = logger.add(lambda record: messages.append(record.record["message"]), level="SUCCESS")
    try:
        _run_tpx1(tpx1_inputs, tmp_path / "both.txt", spectrum_roi=_REGION)
    finally:
        logger.remove(sink_id)

    completion = [m for m in messages if "completed successfully" in m]
    assert completion, f"no completion line among {messages}"
    assert "both.txt" in completion[-1]
    assert re.search(r"both\.hdf5", completion[-1]), f"the HDF5 sibling is unreported: {completion[-1]}"


# --------------------------------------------------------------------------------------------
# 10. the HDF5 sibling must never destroy an input
# --------------------------------------------------------------------------------------------


def test_the_hdf5_sibling_refuses_to_overwrite_one_of_the_runs_own_inputs(tmp_path):
    """A ``.txt`` output also writes ``<stem>.hdf5``, and that must not land on an input file.

    This is the only place in the package that writes a path the user did not name, and ``write_hdf5``
    opens with mode ``"w"``, which truncates. A VENUS metadata file called ``run_1234.hdf5`` sitting
    where a user would plausibly ask for ``run_1234.txt`` would therefore be destroyed, so the derived
    path is checked against the run's own inputs before anything is written.
    """
    sample_dir, ob_dir = tmp_path / "s", tmp_path / "o"
    sample_dir.mkdir()
    ob_dir.mkdir()
    left_edges = [round(0.1 * (i + 1), 1) for i in range(_FRAMES)]
    _spectra_file(sample_dir / "s_Spectra.txt", left_edges)
    _spectra_file(ob_dir / "o_Spectra.txt", left_edges)
    # the metadata file is named run_1234.hdf5, in the directory the user will write into
    meta = _venus_metadata_nexus(tmp_path / "run_1234.hdf5", _PC_SAMPLE, tof_bins=_FRAMES, das_image_path=b"a")
    ob_meta = _venus_metadata_nexus(tmp_path / "ob_1234.hdf5", _PC_OB, tof_bins=_FRAMES, das_image_path=b"a")
    inputs = {
        "sample_hdf5_paths": [meta],
        "ob_hdf5_paths": [ob_meta],
        "sample_tiff_paths": [_ccd_tiffs(sample_dir, "s", _FRAMES, _SAMPLE_BASE, proton_charge=_PC_SAMPLE)],
        "ob_tiff_paths": [_ccd_tiffs(ob_dir, "o", _FRAMES, _OB_BASE, proton_charge=_PC_OB)],
    }
    with h5py.File(meta) as f:
        keys_before = sorted(f.keys())

    with pytest.raises(ValueError, match="would overwrite one of this run's own input files"):
        _run_tpx1(inputs, tmp_path / "run_1234.txt", spectrum_roi=_REGION)

    with h5py.File(meta) as f:
        assert sorted(f.keys()) == keys_before, "the input was modified despite the refusal"
    assert "entry" in keys_before


def test_overwriting_a_previous_output_of_our_own_is_still_allowed(tpx1_inputs, tmp_path):
    """The guard is about inputs, not about re-runs.

    Replacing your own previous output is normal and is what every other writer in the package does, so
    a second run over the same output path must succeed rather than be caught by the guard above.
    """
    output_path = tmp_path / "again.txt"

    first = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION)
    second = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION)

    np.testing.assert_allclose(second.values, first.values, rtol=1e-12)
    assert output_path.with_suffix(".hdf5").exists()


def test_provenance_records_the_factor_that_ran_not_the_request(tpx1_inputs, tmp_path):
    """``rebin_by_tof=True`` asks the statistics analysis for a factor; the file must record the ANSWER.

    Recording the literal ``True`` would leave a reader unable to say how many frames went into a
    point. The effective reduction is recorded too, because the default flips with the argument type —
    a factor sums, a bin list averages — so the file would otherwise be silent about which happened.
    """
    output_path = tmp_path / "recommended.txt"

    spectrum = _run_tpx1(tpx1_inputs, output_path, spectrum_roi=_REGION, rebin_by_tof=True)

    with h5py.File(output_path.with_suffix(".hdf5")) as f:
        recorded = f["metadata/rebin_by_tof"][()]
        reduction = f["metadata/rebin_reduction"][()]
        first_frames = f["metadata/spectrum_bin_first_frame"][()]
    recorded = recorded.decode() if isinstance(recorded, bytes) else str(recorded)
    reduction = reduction.decode() if isinstance(reduction, bytes) else str(reduction)

    assert recorded != "True", "the request was recorded instead of the factor that ran"
    assert recorded.isdigit(), f"expected the resolved integer factor, got {recorded!r}"
    assert reduction == "sum", "an integer factor sums, and the file must say so"
    # the recorded first-frame indices must be consistent with the recorded factor and the bin count
    factor = int(recorded)
    assert len(first_frames) == spectrum.sizes["tof"]
    np.testing.assert_array_equal(first_frames, np.arange(0, _FRAMES, factor)[: spectrum.sizes["tof"]])


def test_a_gapped_bin_list_warns_that_the_axis_is_no_longer_a_spectrum(tpx1_inputs, tmp_path):
    """Dropping frames leaves an axis the fitting tools must not be pointed at, and that is said aloud.

    The workflow guides and the rebinner already warn that a gapped axis invalidates resonance and
    Bragg-edge fitting. It matters more in spectrum mode, because a spectrum is exactly what those tools
    fit — and the file looks entirely well-formed either way. Warned, not refused: dropping frames
    between bins is deliberate and a user may want it for something other than fitting.
    """
    messages, remove = _collect_logs()
    try:
        _run_tpx1(tpx1_inputs, tmp_path / "gapped.txt", spectrum_roi=_REGION, rebin_by_tof=[[0, 2], [4, 6]])
    finally:
        remove()

    warned = " ".join(str(m) for m in messages)
    assert "NOT a continuous spectrum" in warned, f"no gap warning among {warned[:400]}"
    assert "2-3" in warned, "the warning must name the frames that were dropped"


def test_a_contiguous_bin_list_does_not_warn(tpx1_inputs, tmp_path):
    """The warning must not cry wolf: a list that covers every frame is a continuous spectrum.

    Without this, the gap warning above would pass just as well if the code warned unconditionally.
    """
    messages, remove = _collect_logs()
    try:
        _run_tpx1(tpx1_inputs, tmp_path / "contiguous.txt", spectrum_roi=_REGION, rebin_by_tof=[[0, 3], [3, 6]])
    finally:
        remove()

    warned = " ".join(str(m) for m in messages)
    assert "NOT a continuous spectrum" not in warned, f"warned about a contiguous list: {warned[:300]}"
