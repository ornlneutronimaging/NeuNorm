"""Progress reporting through the six pipeline entry points (#195, Task 7).

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
    """Per-stage rendered percentages, in the order tqdm drew them."""
    bars = {}
    for line in output.replace("\r", "\n").split("\n"):
        match = re.match(r"^([a-z_]+):\s+(\d+)%\|", line.strip())
        if match:
            bars.setdefault(match.group(1), []).append(int(match.group(2)))
    return bars


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
    for stage in (STAGE_NORMALIZE, STAGE_EXPORT, STAGE_COMBINE_RUNS, STAGE_GAMMA_FILTER):
        counts = [e.completed for e in stages[stage]]
        assert max(counts) == stages[stage][0].total, f"{stage} stopped at {max(counts)}"


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
