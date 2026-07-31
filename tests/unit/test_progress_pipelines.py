"""Progress reporting through the six pipeline entry points.

This is the task that makes ``progress=True`` reachable from the public API: before it, every
instrumented function existed but no pipeline accepted the argument, so a user of
``run_mars_ccd_pipeline`` could not see anything.

What is checked per pipeline: every stage reaches its declared total, the count never goes backwards,
the sample / open-beam / dark loads carry their own stage labels, the load count is **flat across input
runs** rather than restarting per run, ``progress=False`` leaves the output identical, and a callback
that raises aborts the run. Plus what the bars actually RENDER — three defects in this branch were
invisible to every event-level assertion.
"""

import contextlib
import io
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from PIL import Image

from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
from neunorm.pipelines.mars_tpx3 import run_mars_tpx3_pipeline
from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline
from neunorm.utils.progress import (
    STAGE_COMBINE_RUNS,
    STAGE_EXPORT,
    STAGE_GAMMA_FILTER,
    STAGE_HISTOGRAM,
    STAGE_LOAD_DARK,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
)

# --------------------------------------------------------------------------------------
# fixtures: the smallest synthetic inputs each pipeline accepts
# --------------------------------------------------------------------------------------

_DETECTOR = 32


def _ccd_tiffs(directory, prefix, count, value, *, motslit=False, proton_charge=None):
    """CCD-style TIFFs with the EXIF metadata the CCD pipelines match runs on."""
    paths = []
    for index in range(count):
        image = Image.fromarray(np.full((_DETECTOR, _DETECTOR), float(value + index), dtype=np.float32))
        exif = image.getexif()
        exif[65027] = "ExposureTime:30.000000"
        exif[65022] = f"RunNo:{1000 + index}"
        exif[65025] = "ManufacturerStr:DW936_BV"
        if motslit:  # MARS checks the slit positions match across runs
            exif[65052] = "MotSlitVB.RBV:42.3"
            exif[65054] = "MotSlitVT.RBV:42.8"
            exif[65056] = "MotSlitHR.RBV:41.4"
            exif[65058] = "MotSlitHL.RBV:42.4"
        if proton_charge is not None:  # VENUS normalizes by it
            exif[65024] = f"IntegratedPCharge:{proton_charge}"
        path = directory / f"{prefix}_{index:03}.tiff"
        image.save(path, exif=exif)
        paths.append(path)
    return paths


def _tpx3_event_file(path, repeats, *, bank="bank1_events", offset=0, proton_charge=None, n_tof=1):
    """A minimal TPX3-style NeXus event file: one flood-illuminated detector, `repeats` per pixel."""
    x = np.tile(np.arange(_DETECTOR), (_DETECTOR, 1)).flatten()
    y = np.tile(np.arange(_DETECTOR), (_DETECTOR, 1)).T.flatten()
    event_ids, tofs = [], []
    for frame in range(n_tof):
        event_ids.extend(np.tile(x + y * _DETECTOR + offset, repeats))
        tofs.extend(np.tile(np.full(_DETECTOR * _DETECTOR, 100 + frame * 5, dtype=np.int64), repeats))
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        if proton_charge is not None:
            entry.create_dataset("proton_charge", data=[proton_charge])
            entry.create_dataset("duration", data=[60.0])
        bank_group = entry.create_group(bank)
        bank_group.create_dataset("event_time_offset", data=tofs)
        bank_group.create_dataset("event_id", data=event_ids, dtype=np.int32)
        daslogs = entry.create_group("DASlogs")
        daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay").create_dataset("average_value", data=[5000])
        daslogs.create_group("BL10:Exp:Det").create_dataset("value_strings", data=[[b"MCP TPX3"]])
    return path


def _venus_metadata_nexus(path, proton_charge, *, tof_bins=5, das_image_path=None):
    """The NeXus metadata file the two histogram pipelines read TOF binning and proton charge from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_dataset("proton_charge", data=[proton_charge])
        entry.create_dataset("duration", data=[60.0])
        daslogs = entry.create_group("DASlogs")
        daslogs.create_group("BL10:Det:T1:TSStart_RBV").create_dataset("value", data=[100])
        daslogs.create_group("BL10:Det:T1:TSBinSize_RBV").create_dataset("value", data=[5])
        daslogs.create_group("BL10:Det:T1:TSSize_RBV").create_dataset("value", data=[tof_bins])
        daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay").create_dataset("average_value", data=[5000])
        daslogs.create_group("BL10:Exp:Det").create_dataset("value_strings", data=[[b"MCP TPX3"]])
        if das_image_path is not None:
            daslogs.create_group("BL10:Exp:IM:ImageFilePath").create_dataset("value", data=[[das_image_path]])
    return path


def _spectra_file(path, left_edges):
    """The co-located ``*_Spectra.txt`` sidecar TPX1 reads its TOF axis from."""
    with open(path, "w") as handle:
        handle.write("shutter_time,counts\n")
        for edge in left_edges:
            handle.write(f"{edge},1000\n")
    return path


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _collect():
    events = []
    return events, events.append


def _by_stage(events):
    stages = {}
    for event in events:
        stages.setdefault(event.stage, []).append(event)
    return stages


def _assert_every_stage_completes(events, expected_totals, label=""):
    """Every stage reaches exactly its declared total, monotonically, and no stage is a surprise."""
    stages = _by_stage(events)
    assert set(stages) == set(expected_totals), f"{label}: stages {sorted(stages)} != {sorted(expected_totals)}"
    for stage, stage_events in stages.items():
        counts = [e.completed for e in stage_events]
        totals = {e.total for e in stage_events}
        assert counts == sorted(counts), f"{label}/{stage}: count went backwards: {counts}"
        assert totals == {expected_totals[stage]}, f"{label}/{stage}: totals {totals} != {expected_totals[stage]}"
        if expected_totals[stage] is not None:
            assert max(counts) == expected_totals[stage], (
                f"{label}/{stage}: reached {max(counts)} of {expected_totals[stage]}"
            )


def _render(call):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        call()
    return buffer.getvalue()


def _rendered_bars(output):
    """Per-stage rendered percentages, in the order tqdm drew them.

    Only stages with a known total appear here — those are the ones that can reach 100%. A stage with
    ``total=None`` renders as a plain step counter with no percentage and no bar, so it is invisible to
    this and belongs to :func:`_rendered_counters` instead. That split is deliberate: a helper that
    silently skipped the counters would make "every bar completed" pass while a stage drew nothing.
    """
    bars = {}
    for line in output.replace("\r", "\n").split("\n"):
        match = re.match(r"^([a-z_]+):\s+(\d+)%\|", line.strip())
        if match:
            bars.setdefault(match.group(1), []).append(int(match.group(2)))
    return bars


def _rendered_counters(output):
    """Per-stage rendered step counts for the stages whose total is not knowable in advance."""
    counters = {}
    for line in output.replace("\r", "\n").split("\n"):
        match = re.match(r"^([a-z_]+):\s+(\d+)step", line.strip())
        if match:
            counters.setdefault(match.group(1), []).append(int(match.group(2)))
    return counters


# --------------------------------------------------------------------------------------
# MARS CCD — the carried end-to-end item, and the only pipeline with all three load families
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mars_ccd_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        yield {
            # two sample runs and two open-beam runs: the flat cross-run count is the point
            "sample_paths": [_ccd_tiffs(directory, f"sample{r}", 3, 81, motslit=True) for r in range(2)],
            "ob_paths": [_ccd_tiffs(directory, f"ob{r}", 2, 99, motslit=True) for r in range(2)],
            "dark_paths": [_ccd_tiffs(directory, "dark0", 2, 5, motslit=True)],
            "directory": directory,
        }


def test_mars_ccd_pipeline_reports_every_stage_end_to_end(mars_ccd_inputs, tmp_path):
    """The whole run, from the first TIFF to the written HDF5.

    Six sample TIFFs across two runs must count 1..6 under one total of 6 — not 1..3 twice — because a
    tqdm adapter computes `event.completed - bar.n` and a restart makes that delta NEGATIVE.
    """
    events, sink = _collect()

    run_mars_ccd_pipeline(
        sample_paths=mars_ccd_inputs["sample_paths"],
        ob_paths=mars_ccd_inputs["ob_paths"],
        dark_paths=mars_ccd_inputs["dark_paths"],
        output_path=tmp_path / "out.h5",
        progress=sink,
    )

    _assert_every_stage_completes(
        events,
        {
            STAGE_LOAD_SAMPLE: 6,  # 2 runs x 3 files
            STAGE_LOAD_OB: 4,  # 2 runs x 2 files
            STAGE_LOAD_DARK: 2,  # 1 run x 2 files
            STAGE_COMBINE_RUNS: 3,  # sample, open beam, dark
            STAGE_GAMMA_FILTER: 4,
            STAGE_NORMALIZE: 3,  # two dark subtractions + the division
            STAGE_EXPORT: 4,  # transmission, uncertainty, coords/masks, metadata
        },
        "mars_ccd",
    )
    # the load stage names each file, and the names are the ones handed in
    loaded = [e.detail for e in events if e.stage == STAGE_LOAD_SAMPLE and e.detail.endswith(".tiff")]
    assert loaded == [p.name for group in mars_ccd_inputs["sample_paths"] for p in group]


def test_mars_ccd_load_count_is_flat_across_runs(mars_ccd_inputs, tmp_path):
    """Pinned separately from the totals: this is the one property the whole offset design exists for."""
    events, sink = _collect()

    run_mars_ccd_pipeline(
        sample_paths=mars_ccd_inputs["sample_paths"],
        ob_paths=mars_ccd_inputs["ob_paths"],
        output_path=tmp_path / "flat.h5",
        progress=sink,
    )

    per_file = [e.completed for e in events if e.stage == STAGE_LOAD_SAMPLE and e.detail.endswith(".tiff")]
    assert per_file == [1, 2, 3, 4, 5, 6], per_file


def test_mars_ccd_skips_the_dark_stage_when_there_is_no_dark(mars_ccd_inputs, tmp_path):
    """No dark input means no dark bar at all — an empty stage would render as a bar stuck at 0%."""
    events, sink = _collect()

    run_mars_ccd_pipeline(
        sample_paths=mars_ccd_inputs["sample_paths"],
        ob_paths=mars_ccd_inputs["ob_paths"],
        output_path=tmp_path / "nodark.h5",
        progress=sink,
    )

    stages = _by_stage(events)
    assert STAGE_LOAD_DARK not in stages
    assert stages[STAGE_COMBINE_RUNS][0].total == 2, "the combine total must drop to two without a dark"
    assert max(e.completed for e in stages[STAGE_NORMALIZE]) == 1, "no dark: one division, no dark subtractions"


@pytest.mark.parametrize(
    ("label", "with_dark", "background_roi", "expected_normalize_total"),
    [
        ("dark only", True, None, 3),  # two dark subtractions + the division
        ("neither", False, None, 1),  # the division alone
        ("background ROI + dark", True, (0, 0, 8, 8), 4),  # + the ROI flux coefficient
        ("background ROI only", False, (0, 0, 8, 8), 2),
    ],
)
def test_mars_ccd_normalize_total_matches_whichever_branch_runs(
    mars_ccd_inputs, tmp_path, label, with_dark, background_roi, expected_normalize_total
):
    """Four mutually exclusive normalize branches, four different step counts.

    Each branch declares its own total from the helper the normalizer itself uses, so the bar has to
    finish whichever one runs — a single literal would be right for one branch and wrong for three.
    """
    events, sink = _collect()

    run_mars_ccd_pipeline(
        sample_paths=mars_ccd_inputs["sample_paths"],
        ob_paths=mars_ccd_inputs["ob_paths"],
        dark_paths=mars_ccd_inputs["dark_paths"] if with_dark else None,
        output_path=tmp_path / f"branch_{label.replace(' ', '_')}.h5",
        background_roi=background_roi,
        progress=sink,
    )

    normalize = _by_stage(events)[STAGE_NORMALIZE]
    assert {e.total for e in normalize} == {expected_normalize_total}, label
    assert max(e.completed for e in normalize) == expected_normalize_total, (
        f"{label}: reached {max(e.completed for e in normalize)} of {expected_normalize_total}"
    )


def test_mars_ccd_progress_does_not_change_the_output(mars_ccd_inputs, tmp_path):
    """Reporting is observation only."""
    without = run_mars_ccd_pipeline(
        sample_paths=mars_ccd_inputs["sample_paths"],
        ob_paths=mars_ccd_inputs["ob_paths"],
        dark_paths=mars_ccd_inputs["dark_paths"],
        output_path=tmp_path / "a.h5",
    )
    with_progress = run_mars_ccd_pipeline(
        sample_paths=mars_ccd_inputs["sample_paths"],
        ob_paths=mars_ccd_inputs["ob_paths"],
        dark_paths=mars_ccd_inputs["dark_paths"],
        output_path=tmp_path / "b.h5",
        progress=lambda _event: None,
    )

    assert sc.identical(without, with_progress)
    with h5py.File(tmp_path / "a.h5") as a, h5py.File(tmp_path / "b.h5") as b:
        np.testing.assert_array_equal(a["transmission"][()], b["transmission"][()])
        np.testing.assert_array_equal(a["uncertainty"][()], b["uncertainty"][()])


@pytest.mark.parametrize("cancel_stage", [STAGE_LOAD_SAMPLE, STAGE_GAMMA_FILTER, STAGE_EXPORT])
def test_mars_ccd_callback_raising_cancels_the_run(mars_ccd_inputs, tmp_path, cancel_stage):
    """Cancellation works from any stage — including export, where a best-effort handler lives."""

    class _CancelledError(RuntimeError):
        pass

    def cancel(event):
        if event.stage == cancel_stage:
            raise _CancelledError(f"cancelled during {event.stage}")

    with pytest.raises(_CancelledError, match=cancel_stage):
        run_mars_ccd_pipeline(
            sample_paths=mars_ccd_inputs["sample_paths"],
            ob_paths=mars_ccd_inputs["ob_paths"],
            output_path=tmp_path / f"cancel_{cancel_stage}.h5",
            progress=cancel,
        )


def test_mars_ccd_rendered_bars_all_complete(mars_ccd_inputs, tmp_path):
    """What the user actually sees: one bar per stage, each reaching 100%, none going backwards."""
    output = _render(
        lambda: run_mars_ccd_pipeline(
            sample_paths=mars_ccd_inputs["sample_paths"],
            ob_paths=mars_ccd_inputs["ob_paths"],
            dark_paths=mars_ccd_inputs["dark_paths"],
            output_path=tmp_path / "rendered.h5",
            progress=True,
        )
    )

    bars = _rendered_bars(output)
    expected = {
        STAGE_LOAD_SAMPLE,
        STAGE_LOAD_OB,
        STAGE_LOAD_DARK,
        STAGE_COMBINE_RUNS,
        STAGE_GAMMA_FILTER,
        STAGE_NORMALIZE,
        STAGE_EXPORT,
    }
    assert expected <= set(bars), f"missing bars: {sorted(expected - set(bars))}"
    for stage, percents in bars.items():
        assert percents == sorted(percents), f"{stage} rendered backwards: {percents}"
        assert percents[-1] == 100, f"{stage} never rendered 100%: {percents}"


# --------------------------------------------------------------------------------------
# VENUS CCD — the proton-charge normalization branch
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def venus_ccd_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        yield {
            "sample_paths": [_ccd_tiffs(directory, "vsample", 3, 81, proton_charge=1000.0)],
            "ob_paths": [_ccd_tiffs(directory, "vob", 2, 99, proton_charge=1010.0)],
            "dark_paths": [_ccd_tiffs(directory, "vdark", 2, 5, proton_charge=1.0)],
        }


def test_venus_ccd_pipeline_reports_every_stage_end_to_end(venus_ccd_inputs, tmp_path):
    """The proton-charge route costs two extra normalization steps, and the total must say so."""
    events, sink = _collect()

    run_venus_ccd_pipeline(
        sample_paths=venus_ccd_inputs["sample_paths"],
        ob_paths=venus_ccd_inputs["ob_paths"],
        dark_paths=venus_ccd_inputs["dark_paths"],
        output_path=tmp_path / "venus.h5",
        progress=sink,
    )

    _assert_every_stage_completes(
        events,
        {
            STAGE_LOAD_SAMPLE: 3,
            STAGE_LOAD_OB: 2,
            STAGE_LOAD_DARK: 2,
            STAGE_COMBINE_RUNS: 3,
            STAGE_GAMMA_FILTER: 4,
            STAGE_NORMALIZE: 5,  # 2 dark subtractions + 2 proton-charge divisions + the division
            STAGE_EXPORT: 4,
        },
        "venus_ccd",
    )


@pytest.mark.parametrize(
    ("label", "with_dark", "expected_normalize_total"),
    [("background ROI + dark", True, 4), ("background ROI only", False, 2)],
)
def test_venus_ccd_background_roi_replaces_the_proton_charge_steps(
    venus_ccd_inputs, tmp_path, label, with_dark, expected_normalize_total
):
    """A background ROI is the alternative to proton charge, not an addition, so it declares FEWER
    normalization steps than the proton-charge route — one flux coefficient instead of two divisions."""
    events, sink = _collect()

    run_venus_ccd_pipeline(
        sample_paths=venus_ccd_inputs["sample_paths"],
        ob_paths=venus_ccd_inputs["ob_paths"],
        dark_paths=venus_ccd_inputs["dark_paths"] if with_dark else None,
        output_path=tmp_path / f"vbranch_{label.replace(' ', '_')}.h5",
        background_roi=(0, 0, 8, 8),
        progress=sink,
    )

    normalize = _by_stage(events)[STAGE_NORMALIZE]
    assert {e.total for e in normalize} == {expected_normalize_total}, label
    assert max(e.completed for e in normalize) == expected_normalize_total, label


def test_venus_ccd_progress_does_not_change_the_output(venus_ccd_inputs, tmp_path):
    without = run_venus_ccd_pipeline(
        sample_paths=venus_ccd_inputs["sample_paths"],
        ob_paths=venus_ccd_inputs["ob_paths"],
        output_path=tmp_path / "v_a.h5",
    )
    with_progress = run_venus_ccd_pipeline(
        sample_paths=venus_ccd_inputs["sample_paths"],
        ob_paths=venus_ccd_inputs["ob_paths"],
        output_path=tmp_path / "v_b.h5",
        progress=lambda _event: None,
    )
    assert sc.identical(without, with_progress)


# --------------------------------------------------------------------------------------
# MARS TPX3 — the event path, where a file is not one cheap item
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mars_tpx3_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        yield {
            # two files in one run, so the per-file naming and the 4-allocations-per-file total both bite
            "sample_paths": [[_tpx3_event_file(directory / f"s{i}.hdf5", 3 + i) for i in range(2)]],
            "ob_paths": [[_tpx3_event_file(directory / f"o{i}.hdf5", 6 + i) for i in range(2)]],
        }


def test_mars_tpx3_pipeline_counts_allocations_per_file(mars_tpx3_inputs, tmp_path):
    """The event loader reports four full-event-length allocations per file, so the load total is
    four times the file count — and each file is named as it is opened."""
    events, sink = _collect()

    run_mars_tpx3_pipeline(
        sample_paths=mars_tpx3_inputs["sample_paths"],
        ob_paths=mars_tpx3_inputs["ob_paths"],
        output_path=tmp_path / "tpx3.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
        progress=sink,
    )

    stages = _by_stage(events)
    assert {e.total for e in stages[STAGE_LOAD_SAMPLE]} == {8}, "2 files x 4 allocations"
    assert max(e.completed for e in stages[STAGE_LOAD_SAMPLE]) == 8
    named = [e.detail for e in stages[STAGE_LOAD_SAMPLE] if e.detail.endswith(".hdf5")]
    assert named == ["s0.hdf5", "s1.hdf5"], named
    # histogramming has no total: the chunk count follows from each file's event count
    assert {e.total for e in stages[STAGE_HISTOGRAM]} == {None}
    histogram_counts = [e.completed for e in stages[STAGE_HISTOGRAM]]
    assert histogram_counts == sorted(histogram_counts), histogram_counts
    assert max(histogram_counts) >= 4, "one chunk per file across both families"
    # Independent expected numbers, not `stages[stage][0].total` — comparing a count against the total
    # the same code declared would pass however wrong both were.
    for stage, expected in ((STAGE_NORMALIZE, 1), (STAGE_EXPORT, 4), (STAGE_COMBINE_RUNS, 2), (STAGE_GAMMA_FILTER, 4)):
        counts = [e.completed for e in stages[stage]]
        assert {e.total for e in stages[stage]} == {expected}, f"{stage} declared {stages[stage][0].total}"
        assert max(counts) == expected, f"{stage} stopped at {max(counts)} of {expected}"


def test_mars_tpx3_rendered_output_includes_the_unknown_total_stage(mars_tpx3_inputs, tmp_path):
    """The event path renders two shapes at once, and the one without a total is easy to lose.

    Histogramming has no knowable total, so tqdm draws it as a plain step counter — no percentage, no
    bar glyph. I missed it in a hand-written render check whose filter only matched `NN%|`, which is
    exactly how a stage that draws nothing would go unnoticed. Both shapes are asserted here.
    """
    output = _render(
        lambda: run_mars_tpx3_pipeline(
            sample_paths=mars_tpx3_inputs["sample_paths"],
            ob_paths=mars_tpx3_inputs["ob_paths"],
            output_path=tmp_path / "tpx3_rendered.h5",
            detector_shape=(_DETECTOR, _DETECTOR),
            progress=True,
        )
    )

    bars = _rendered_bars(output)
    counters = _rendered_counters(output)

    assert {STAGE_LOAD_SAMPLE, STAGE_LOAD_OB, STAGE_COMBINE_RUNS, STAGE_GAMMA_FILTER, STAGE_NORMALIZE} <= set(bars)
    for stage, percents in bars.items():
        assert percents == sorted(percents), f"{stage} rendered backwards: {percents}"
        assert percents[-1] == 100, f"{stage} never rendered 100%: {percents}"

    assert STAGE_HISTOGRAM in counters, f"the histogram stage drew nothing: {sorted(counters)}"
    steps = counters[STAGE_HISTOGRAM]
    assert steps == sorted(steps), f"the histogram counter went backwards: {steps}"
    assert max(steps) >= 4, f"one chunk per file across both families, got {steps}"
    assert STAGE_HISTOGRAM not in bars, "a stage with no total must not render a percentage"


@pytest.mark.parametrize(
    ("label", "background_roi", "expected_normalize_total"),
    [("no correction", None, 1), ("background ROI", (0, 0, 8, 8), 2)],
)
def test_mars_tpx3_normalize_total_follows_the_correction_requested(
    mars_tpx3_inputs, tmp_path, label, background_roi, expected_normalize_total
):
    """MARS TPX3 has no dark and no proton charge, so its only variable is the background ROI."""
    events, sink = _collect()

    run_mars_tpx3_pipeline(
        sample_paths=mars_tpx3_inputs["sample_paths"],
        ob_paths=mars_tpx3_inputs["ob_paths"],
        output_path=tmp_path / f"tbranch_{label.replace(' ', '_')}.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
        background_roi=background_roi,
        progress=sink,
    )

    normalize = _by_stage(events)[STAGE_NORMALIZE]
    assert {e.total for e in normalize} == {expected_normalize_total}, label
    assert max(e.completed for e in normalize) == expected_normalize_total, label


def test_mars_tpx3_progress_does_not_change_the_output(mars_tpx3_inputs, tmp_path):
    without = run_mars_tpx3_pipeline(
        sample_paths=mars_tpx3_inputs["sample_paths"],
        ob_paths=mars_tpx3_inputs["ob_paths"],
        output_path=tmp_path / "t_a.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
    )
    with_progress = run_mars_tpx3_pipeline(
        sample_paths=mars_tpx3_inputs["sample_paths"],
        ob_paths=mars_tpx3_inputs["ob_paths"],
        output_path=tmp_path / "t_b.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
        progress=lambda _event: None,
    )
    assert sc.identical(without, with_progress)


# --------------------------------------------------------------------------------------
# VENUS TPX3 event — event load + histogram + normalize + export
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def venus_event_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        yield {
            "binning": BinningConfig(bins=5, bin_space="tof", tof_range=(100000, 125000), use_log_bin=False),
            "sample_paths": [
                _tpx3_event_file(
                    directory / "vsample.hdf5",
                    3,
                    bank="bank100_events",
                    offset=1_000_000,
                    proton_charge=12345,
                    n_tof=5,
                )
            ],
            "ob_paths": [
                _tpx3_event_file(
                    directory / "vob.hdf5",
                    6,
                    bank="bank100_events",
                    offset=1_000_000,
                    proton_charge=24690,
                    n_tof=5,
                )
            ],
        }


def test_venus_tpx3_event_pipeline_reports_every_stage(venus_event_inputs, tmp_path):
    events, sink = _collect()

    run_venus_tpx3_event_pipeline(
        sample_paths=venus_event_inputs["sample_paths"],
        ob_paths=venus_event_inputs["ob_paths"],
        binning=venus_event_inputs["binning"],
        output_path=tmp_path / "vevent.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
        progress=sink,
    )

    stages = _by_stage(events)
    assert {e.total for e in stages[STAGE_LOAD_SAMPLE]} == {4}, "one file x 4 allocations"
    assert max(e.completed for e in stages[STAGE_LOAD_SAMPLE]) == 4
    assert max(e.completed for e in stages[STAGE_LOAD_OB]) == 4
    assert {e.total for e in stages[STAGE_HISTOGRAM]} == {None}
    assert max(e.completed for e in stages[STAGE_NORMALIZE]) == 3, "division + two proton-charge steps"
    assert max(e.completed for e in stages[STAGE_EXPORT]) == 4
    assert STAGE_GAMMA_FILTER not in stages, "this pipeline has no gamma filter"


def test_venus_tpx3_event_per_image_tiff_export_counts_files(venus_event_inputs, tmp_path):
    """`tiff_one_file_per_image` is the one export path with a real item count, and the pipeline
    must declare it from the post-rebin image count rather than assume one write."""
    events, sink = _collect()

    run_venus_tpx3_event_pipeline(
        sample_paths=venus_event_inputs["sample_paths"],
        ob_paths=venus_event_inputs["ob_paths"],
        binning=venus_event_inputs["binning"],
        output_path=tmp_path / "frames.tiff",
        detector_shape=(_DETECTOR, _DETECTOR),
        tiff_one_file_per_image=True,
        progress=sink,
    )

    export = _by_stage(events)[STAGE_EXPORT]
    assert {e.total for e in export} == {5}, "five TOF bins, five files"
    assert [e.completed for e in export] == [1, 2, 3, 4, 5]
    assert [e.detail for e in export] == [f"frames_{i:05d}.tiff" for i in range(5)]


def test_venus_tpx3_event_accepts_an_unsized_path_sequence(venus_event_inputs, tmp_path):
    """A generator of paths must not abort the run.

    Regression: this pipeline computed its load total with a bare `len()`, so a generator or
    `Path.glob(...)` raised TypeError before a single file was opened — including with the default
    `progress=False` — where the pre-progress version simply ran. Its five siblings never did, because
    they route through `total_across_groups`.
    """
    events, sink = _collect()

    run_venus_tpx3_event_pipeline(
        sample_paths=(path for path in venus_event_inputs["sample_paths"]),
        ob_paths=(path for path in venus_event_inputs["ob_paths"]),
        binning=venus_event_inputs["binning"],
        output_path=tmp_path / "unsized.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
        progress=sink,
    )

    load = _by_stage(events)[STAGE_LOAD_SAMPLE]
    assert {e.total for e in load} == {None}, "an unsized input must report an unknown total"
    assert max(e.completed for e in load) == 4, "the file was still read, four allocations reported"


def test_venus_tpx3_event_progress_does_not_change_the_output(venus_event_inputs, tmp_path):
    common = {
        "sample_paths": venus_event_inputs["sample_paths"],
        "ob_paths": venus_event_inputs["ob_paths"],
        "binning": venus_event_inputs["binning"],
        "detector_shape": (_DETECTOR, _DETECTOR),
    }
    without = run_venus_tpx3_event_pipeline(output_path=tmp_path / "ve_a.h5", **common)
    with_progress = run_venus_tpx3_event_pipeline(
        output_path=tmp_path / "ve_b.h5", progress=lambda _event: None, **common
    )
    assert sc.identical(without, with_progress)


# --------------------------------------------------------------------------------------
# VENUS TPX3 histogram and TPX1 — pre-binned TIFF stacks with NeXus metadata
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def venus_histogram_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        yield {
            "sample_tiff_paths": [_ccd_tiffs(directory, "hsample", 5, 81)],
            "ob_tiff_paths": [_ccd_tiffs(directory, "hob", 5, 99)],
            "sample_hdf5_paths": [_venus_metadata_nexus(directory / "nexus" / "hs.nxs.h5", 12345)],
            "ob_hdf5_paths": [_venus_metadata_nexus(directory / "nexus" / "ho.nxs.h5", 24690)],
        }


def test_venus_tpx3_histogram_pipeline_reports_every_stage(venus_histogram_inputs, tmp_path):
    events, sink = _collect()

    run_venus_tpx3_histogram_pipeline(
        sample_hdf5_paths=venus_histogram_inputs["sample_hdf5_paths"],
        ob_hdf5_paths=venus_histogram_inputs["ob_hdf5_paths"],
        sample_tiff_paths=venus_histogram_inputs["sample_tiff_paths"],
        ob_tiff_paths=venus_histogram_inputs["ob_tiff_paths"],
        output_path=tmp_path / "hist.h5",
        progress=sink,
    )

    _assert_every_stage_completes(
        events,
        {
            STAGE_LOAD_SAMPLE: 5,
            STAGE_LOAD_OB: 5,
            STAGE_COMBINE_RUNS: 2,
            STAGE_NORMALIZE: 3,
            STAGE_EXPORT: 4,
        },
        "venus_tpx3_histogram",
    )


def test_venus_tpx3_histogram_reports_the_tof_rebin(venus_histogram_inputs, tmp_path):
    """`rebin_tof` takes no progress argument of its own, so the pipeline names its two calls."""
    from neunorm.utils.progress import STAGE_REBIN_TOF

    events, sink = _collect()

    run_venus_tpx3_histogram_pipeline(
        sample_hdf5_paths=venus_histogram_inputs["sample_hdf5_paths"],
        ob_hdf5_paths=venus_histogram_inputs["ob_hdf5_paths"],
        sample_tiff_paths=venus_histogram_inputs["sample_tiff_paths"],
        ob_tiff_paths=venus_histogram_inputs["ob_tiff_paths"],
        output_path=tmp_path / "hist_rebin.h5",
        rebin_by_tof=2,
        progress=sink,
    )

    rebin = _by_stage(events)[STAGE_REBIN_TOF]
    assert [e.detail for e in rebin if e.detail] == ["rebinning sample TOF", "rebinning open beam TOF"]
    assert max(e.completed for e in rebin) == 2


def test_venus_tpx3_histogram_per_image_export_counts_files(venus_histogram_inputs, tmp_path):
    """The same computed export count on a second pipeline, so the helper is not exercised through one
    call site only. Five TOF frames, no rebin, five files."""
    events, sink = _collect()

    run_venus_tpx3_histogram_pipeline(
        sample_hdf5_paths=venus_histogram_inputs["sample_hdf5_paths"],
        ob_hdf5_paths=venus_histogram_inputs["ob_hdf5_paths"],
        sample_tiff_paths=venus_histogram_inputs["sample_tiff_paths"],
        ob_tiff_paths=venus_histogram_inputs["ob_tiff_paths"],
        output_path=tmp_path / "hist_frames.tiff",
        tiff_one_file_per_image=True,
        progress=sink,
    )

    export = _by_stage(events)[STAGE_EXPORT]
    assert {e.total for e in export} == {5}
    assert [e.detail for e in export] == [f"hist_frames_{i:05d}.tiff" for i in range(5)]


def test_venus_tpx3_histogram_progress_does_not_change_the_output(venus_histogram_inputs, tmp_path):
    common = {
        "sample_hdf5_paths": venus_histogram_inputs["sample_hdf5_paths"],
        "ob_hdf5_paths": venus_histogram_inputs["ob_hdf5_paths"],
        "sample_tiff_paths": venus_histogram_inputs["sample_tiff_paths"],
        "ob_tiff_paths": venus_histogram_inputs["ob_tiff_paths"],
    }
    without = run_venus_tpx3_histogram_pipeline(output_path=tmp_path / "h_a.h5", **common)
    with_progress = run_venus_tpx3_histogram_pipeline(
        output_path=tmp_path / "h_b.h5", progress=lambda _event: None, **common
    )
    assert sc.identical(without, with_progress)


@pytest.fixture(scope="module")
def venus_tpx1_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        image_dir = directory / "autoreduce" / "sample"
        ob_dir = directory / "autoreduce" / "ob"
        image_dir.mkdir(parents=True)
        ob_dir.mkdir(parents=True)
        sample_tiffs = _ccd_tiffs(image_dir, "sample", 5, 81)
        ob_tiffs = _ccd_tiffs(ob_dir, "ob", 5, 99)
        _spectra_file(image_dir / "sample_Spectra.txt", [round(0.1 * (i + 1), 1) for i in range(5)])
        _spectra_file(ob_dir / "ob_Spectra.txt", [round(0.1 * (i + 1), 1) for i in range(5)])
        yield {
            "sample_tiff_paths": [sample_tiffs],
            "ob_tiff_paths": [ob_tiffs],
            "sample_hdf5_paths": [
                _venus_metadata_nexus(directory / "nexus" / "s.nxs.h5", 12345, das_image_path=b"autoreduce/sample")
            ],
            "ob_hdf5_paths": [
                _venus_metadata_nexus(directory / "nexus" / "o.nxs.h5", 24690, das_image_path=b"autoreduce/ob")
            ],
        }


def test_venus_tpx1_pipeline_reports_every_stage(venus_tpx1_inputs, tmp_path):
    events, sink = _collect()

    run_venus_tpx1_pipeline(
        sample_hdf5_paths=venus_tpx1_inputs["sample_hdf5_paths"],
        ob_hdf5_paths=venus_tpx1_inputs["ob_hdf5_paths"],
        sample_tiff_paths=venus_tpx1_inputs["sample_tiff_paths"],
        ob_tiff_paths=venus_tpx1_inputs["ob_tiff_paths"],
        output_path=tmp_path / "tpx1.h5",
        progress=sink,
    )

    _assert_every_stage_completes(
        events,
        {
            STAGE_LOAD_SAMPLE: 5,
            STAGE_LOAD_OB: 5,
            STAGE_COMBINE_RUNS: 2,
            STAGE_NORMALIZE: 3,
            STAGE_EXPORT: 4,
        },
        "venus_tpx1",
    )


def test_venus_tpx1_rendered_bars_all_complete(venus_tpx1_inputs, tmp_path):
    """A TOF pipeline rendered end to end, the counterpart to the CCD render above."""
    output = _render(
        lambda: run_venus_tpx1_pipeline(
            sample_hdf5_paths=venus_tpx1_inputs["sample_hdf5_paths"],
            ob_hdf5_paths=venus_tpx1_inputs["ob_hdf5_paths"],
            sample_tiff_paths=venus_tpx1_inputs["sample_tiff_paths"],
            ob_tiff_paths=venus_tpx1_inputs["ob_tiff_paths"],
            output_path=tmp_path / "tpx1_rendered.h5",
            progress=True,
        )
    )

    bars = _rendered_bars(output)
    expected = {STAGE_LOAD_SAMPLE, STAGE_LOAD_OB, STAGE_COMBINE_RUNS, STAGE_NORMALIZE, STAGE_EXPORT}
    assert expected <= set(bars), f"missing bars: {sorted(expected - set(bars))}"
    for stage, percents in bars.items():
        assert percents == sorted(percents), f"{stage} rendered backwards: {percents}"
        assert percents[-1] == 100, f"{stage} never rendered 100%: {percents}"


def test_venus_tpx1_progress_does_not_change_the_output(venus_tpx1_inputs, tmp_path):
    common = {
        "sample_hdf5_paths": venus_tpx1_inputs["sample_hdf5_paths"],
        "ob_hdf5_paths": venus_tpx1_inputs["ob_hdf5_paths"],
        "sample_tiff_paths": venus_tpx1_inputs["sample_tiff_paths"],
        "ob_tiff_paths": venus_tpx1_inputs["ob_tiff_paths"],
    }
    without = run_venus_tpx1_pipeline(output_path=tmp_path / "p_a.h5", **common)
    with_progress = run_venus_tpx1_pipeline(output_path=tmp_path / "p_b.h5", progress=lambda _e: None, **common)
    assert sc.identical(without, with_progress)


# --------------------------------------------------------------------------------------
# what every pipeline must have in common
# --------------------------------------------------------------------------------------

_PIPELINES = {
    "run_mars_ccd_pipeline": run_mars_ccd_pipeline,
    "run_venus_ccd_pipeline": run_venus_ccd_pipeline,
    "run_mars_tpx3_pipeline": run_mars_tpx3_pipeline,
    "run_venus_tpx1_pipeline": run_venus_tpx1_pipeline,
    "run_venus_tpx3_histogram_pipeline": run_venus_tpx3_histogram_pipeline,
    "run_venus_tpx3_event_pipeline": run_venus_tpx3_event_pipeline,
}


def test_a_generator_of_input_runs_is_not_consumed_by_counting_it(mars_ccd_inputs, tmp_path):
    """The load total must not be computed by exhausting the caller's input.

    `total_across_groups` iterates the outer container to sum the group lengths. Handed a *generator*
    of groups it would consume it, and the pipeline would then load nothing while reporting a total —
    so it refuses to iterate a container with no length and reports an unknown total instead.
    """
    groups = (group for group in mars_ccd_inputs["sample_paths"])
    events, sink = _collect()

    run_mars_ccd_pipeline(
        sample_paths=groups,
        ob_paths=mars_ccd_inputs["ob_paths"],
        output_path=tmp_path / "generator.h5",
        progress=sink,
    )

    loaded = [e.detail for e in events if e.stage == STAGE_LOAD_SAMPLE and e.detail.endswith(".tiff")]
    assert loaded == [p.name for group in mars_ccd_inputs["sample_paths"] for p in group], (
        "the generator was consumed before the load"
    )
    assert {e.total for e in events if e.stage == STAGE_LOAD_SAMPLE} == {None}, (
        "an unsized input must report an unknown total, not a wrong one"
    )


def test_every_pipeline_can_be_cancelled_from_its_first_event(
    mars_ccd_inputs,
    venus_ccd_inputs,
    mars_tpx3_inputs,
    venus_event_inputs,
    venus_histogram_inputs,
    venus_tpx1_inputs,
    tmp_path,
):
    """Cancellation is a property of all six, not just the one with the most tests.

    Raising on the very first event is the strictest form: whatever stage that is, the run must abort
    there rather than complete, and nothing may be written.
    """

    class _CancelledError(RuntimeError):
        pass

    def cancel(_event):
        raise _CancelledError("stop")

    calls = {
        "run_mars_ccd_pipeline": lambda out, progress: run_mars_ccd_pipeline(
            sample_paths=mars_ccd_inputs["sample_paths"],
            ob_paths=mars_ccd_inputs["ob_paths"],
            output_path=out,
            progress=progress,
        ),
        "run_venus_ccd_pipeline": lambda out, progress: run_venus_ccd_pipeline(
            sample_paths=venus_ccd_inputs["sample_paths"],
            ob_paths=venus_ccd_inputs["ob_paths"],
            output_path=out,
            progress=progress,
        ),
        "run_mars_tpx3_pipeline": lambda out, progress: run_mars_tpx3_pipeline(
            sample_paths=mars_tpx3_inputs["sample_paths"],
            ob_paths=mars_tpx3_inputs["ob_paths"],
            output_path=out,
            detector_shape=(_DETECTOR, _DETECTOR),
            progress=progress,
        ),
        "run_venus_tpx3_event_pipeline": lambda out, progress: run_venus_tpx3_event_pipeline(
            sample_paths=venus_event_inputs["sample_paths"],
            ob_paths=venus_event_inputs["ob_paths"],
            binning=venus_event_inputs["binning"],
            output_path=out,
            detector_shape=(_DETECTOR, _DETECTOR),
            progress=progress,
        ),
        "run_venus_tpx3_histogram_pipeline": lambda out, progress: run_venus_tpx3_histogram_pipeline(
            sample_hdf5_paths=venus_histogram_inputs["sample_hdf5_paths"],
            ob_hdf5_paths=venus_histogram_inputs["ob_hdf5_paths"],
            sample_tiff_paths=venus_histogram_inputs["sample_tiff_paths"],
            ob_tiff_paths=venus_histogram_inputs["ob_tiff_paths"],
            output_path=out,
            progress=progress,
        ),
        "run_venus_tpx1_pipeline": lambda out, progress: run_venus_tpx1_pipeline(
            sample_hdf5_paths=venus_tpx1_inputs["sample_hdf5_paths"],
            ob_hdf5_paths=venus_tpx1_inputs["ob_hdf5_paths"],
            sample_tiff_paths=venus_tpx1_inputs["sample_tiff_paths"],
            ob_tiff_paths=venus_tpx1_inputs["ob_tiff_paths"],
            output_path=out,
            progress=progress,
        ),
    }
    assert set(calls) == set(_PIPELINES), "a pipeline is missing from the cancellation coverage"

    for name, call in calls.items():
        output = tmp_path / f"cancel_{name}.h5"
        with pytest.raises(_CancelledError):
            call(output, cancel)
        assert not output.exists(), f"{name}: cancelled but still wrote {output.name}"
        # and the same call without cancelling does produce the file, so the raise is what stopped it
        completed = tmp_path / f"ok_{name}.h5"
        call(completed, False)
        assert completed.exists(), f"{name}: the uncancelled run did not write its output"


@pytest.mark.parametrize("name", sorted(_PIPELINES))
def test_every_pipeline_takes_progress_keyword_only_and_last(name):
    """`progress` must be keyword-only, so it can never shift a released positional argument, and last,
    so the convention is uniform across the six and matches the loaders."""
    import inspect

    parameters = inspect.signature(_PIPELINES[name]).parameters
    assert "progress" in parameters, f"{name} does not accept progress"
    assert parameters["progress"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{name}: progress is {parameters['progress'].kind.description}, not keyword-only"
    )
    assert list(parameters)[-1] == "progress", f"{name}: progress is not the last parameter: {list(parameters)}"
    assert parameters["progress"].default is False, f"{name}: progress must default to False"


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [("normal", {}), ("zero events", {}), ("max_events truncation", {"max_events": 10})],
)
def test_the_event_load_total_rests_on_an_exact_step_count(tmp_path, label, kwargs):
    """The event pipelines declare ``LOAD_EVENT_NEXUS_STEPS * file_count`` as their load total, which is
    only exact if the loader emits that many ticks on EVERY path. Pinned here, next to the pipelines
    that depend on it: a zero-event bank and a truncated read must still count four."""
    from neunorm.loaders.event_loader import LOAD_EVENT_NEXUS_STEPS, load_event_nexus

    n_events = 0 if label == "zero events" else 100
    path = tmp_path / f"{label.replace(' ', '_')}.h5"
    with h5py.File(path, "w") as f:
        bank = f.create_group("entry").create_group("bank1_events")
        bank.create_dataset("event_id", data=np.zeros(n_events, dtype=np.int32))
        bank.create_dataset("event_time_offset", data=np.zeros(n_events, dtype=np.float64))

    ticks = []
    load_event_nexus(
        path,
        detector_shape=(_DETECTOR, _DETECTOR),
        progress=lambda event: ticks.append(event.completed) if not event.detail else None,
        **kwargs,
    )

    assert ticks == list(range(1, LOAD_EVENT_NEXUS_STEPS + 1)), f"{label}: {ticks}"


def test_progress_false_never_imports_tqdm(tmp_path):
    """Every pipeline's `progress` docstring says the default is free. `tqdm` is the expensive part —
    importing `tqdm.auto` probes for ipywidgets — so a full run with reporting off must never load it.

    Run in a fresh interpreter: within the test session other tests have already imported tqdm, so
    checking `sys.modules` here would pass no matter what the pipeline does.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys
        import numpy as np
        from pathlib import Path
        from PIL import Image
        from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

        directory = Path({str(tmp_path)!r})

        def tiffs(prefix, count, value):
            paths = []
            for index in range(count):
                image = Image.fromarray(np.full((16, 16), float(value + index), dtype=np.float32))
                exif = image.getexif()
                exif[65027] = "ExposureTime:30.000000"
                exif[65022] = f"RunNo:{{1000 + index}}"
                exif[65025] = "ManufacturerStr:DW936_BV"
                for tag in (65052, 65054, 65056, 65058):
                    exif[tag] = "MotSlitVB.RBV:42.3"
                path = directory / f"{{prefix}}_{{index}}.tiff"
                image.save(path, exif=exif)
                paths.append(path)
            return paths

        run_mars_ccd_pipeline(
            sample_paths=[tiffs("s", 2, 81)],
            ob_paths=[tiffs("o", 2, 99)],
            output_path=directory / "off.h5",
        )
        print("TQDM_LOADED" if "tqdm" in sys.modules else "TQDM_ABSENT")
        """
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "TQDM_ABSENT" in result.stdout, f"progress=False imported tqdm: {result.stdout}"


def test_a_progress_value_a_pipeline_cannot_use_fails_before_the_run_starts(mars_ccd_inputs, tmp_path):
    """A typo'd progress argument must fail loudly, and must fail BEFORE the first file is read.

    The alternative — discovering it after a long load — is what makes a silently-ignored progress
    argument expensive. `resolve_progress` runs first, so the run never begins and no output appears.
    """
    output = tmp_path / "never_written.h5"

    with pytest.raises(TypeError, match="progress must be"):
        run_mars_ccd_pipeline(
            sample_paths=mars_ccd_inputs["sample_paths"],
            ob_paths=mars_ccd_inputs["ob_paths"],
            output_path=output,
            progress="yes",
        )

    assert not output.exists(), "the run started despite an unusable progress argument"


def test_mars_tpx3_does_not_hold_the_previous_file_s_events_while_reading_the_next(
    mars_tpx3_inputs, tmp_path, monkeypatch
):
    """The event load must release each file's raw events before the next file's allocations.

    Task 7 rewrote this pipeline's nested comprehension into loops so each file could be named. Lifting
    the loader's result into a local `events` was the natural way to write that — and it kept the
    previous file's full-event-length arrays resident through the next file's four allocations, measured
    at +26 MiB of peak RSS on 3.6M-event files (`.harness/event_peak_rss.py`). On the one path whose cost
    IS peak memory, that is a regression, so the call is nested again.

    Pinned by object lifetime rather than by an RSS number, which would be machine-dependent: at the
    moment file N's READ begins, file N-1's EventData must already be collectable.
    """
    import gc
    import weakref

    from neunorm.pipelines import mars_tpx3 as pipeline

    # A list of weakrefs, not a WeakSet: EventData is a pydantic model and unhashable, so a set cannot
    # hold it.
    refs = []
    alive_before_this_read = []

    real_loader = pipeline.load_event_nexus
    real_converter = pipeline.convert_events_to_2d_histogram

    def spy_loader(*args, **kwargs):
        # Counted HERE, before the read begins — not at conversion time. That distinction is the whole
        # test: with a named local the previous file's object is released only when the name is REBOUND,
        # which happens after this load completes, so by conversion time it is already gone and a check
        # there passes either way. I made exactly that mistake first; the mutation survived it.
        gc.collect()
        alive_before_this_read.append(sum(1 for ref in refs if ref() is not None))
        events = real_loader(*args, **kwargs)
        refs.append(weakref.ref(events))
        return events

    def spy_converter(events, *args, **kwargs):
        return real_converter(events, *args, **kwargs)

    monkeypatch.setattr(pipeline, "load_event_nexus", spy_loader)
    monkeypatch.setattr(pipeline, "convert_events_to_2d_histogram", spy_converter)

    run_mars_tpx3_pipeline(
        sample_paths=mars_tpx3_inputs["sample_paths"],  # two files in one run
        ob_paths=mars_tpx3_inputs["ob_paths"],
        output_path=tmp_path / "lifetime.h5",
        detector_shape=(_DETECTOR, _DETECTOR),
    )

    assert len(alive_before_this_read) >= 4, f"expected one read per file, saw {alive_before_this_read}"
    assert alive_before_this_read == [0] * len(alive_before_this_read), (
        "a previous file's raw events were still resident while the next file was being read: "
        f"{alive_before_this_read} (one entry per file; every entry must be 0)"
    )
