"""Progress reporting through the event-mode load path (#195, Task 4).

The event path has no per-file loop the way a TIFF stack does: `load_event_nexus` is one h5py slab
read followed by four numpy passes, each allocating a full event-length array, and the converters
chunk at 500M events so a typical run is a single chunk. So the useful signal here is *which* of
those allocations is running, not a count — reported as notes. That is also where the event path
peaks in memory, which is the wait a user actually notices.
"""

import contextlib
import io
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc

from neunorm.data_models.core import EventData
from neunorm.data_models.tof import BinningConfig
from neunorm.loaders.event_loader import load_event_nexus
from neunorm.tof.event_converter import (
    convert_events_to_2d_histogram,
    convert_events_to_histogram,
)
from neunorm.utils.progress import (
    STAGE_LOAD_OB,
    STAGE_NORMALIZE,
    ProgressReporter,
    _TqdmSink,
    resolve_progress,
)

DETECTOR = (64, 64)


class _RecordingTqdmSink(_TqdmSink):
    """Records every sink instance, so one a callee built internally is observable."""

    instances: list["_RecordingTqdmSink"] = []

    def __init__(self):
        super().__init__()
        _RecordingTqdmSink.instances.append(self)


@pytest.fixture
def recorded_sinks(monkeypatch):
    _RecordingTqdmSink.instances = []
    monkeypatch.setattr("neunorm.utils.progress._TqdmSink", _RecordingTqdmSink)
    return _RecordingTqdmSink.instances


def _collect():
    events = []
    return events, events.append


@pytest.fixture
def nexus_file(tmp_path):
    """A minimal SNS NeXus event bank that `load_event_nexus` accepts."""
    path = tmp_path / "VENUS_test.nxs.h5"
    n = 400
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as hf:
        bank = hf.create_group("entry/bank1_events")
        bank.create_dataset("event_id", data=rng.integers(0, DETECTOR[0] * DETECTOR[1], n, dtype=np.int64))
        bank.create_dataset("event_time_offset", data=rng.uniform(0, 16000, n).astype(np.float64))
    return path


def _events(n=300):
    rng = np.random.default_rng(1)
    return EventData(
        tof=rng.integers(0, 16000, n).astype(np.int64),
        x=rng.integers(0, DETECTOR[0], n).astype(np.int32),
        y=rng.integers(0, DETECTOR[1], n).astype(np.int32),
        file_path=Path("test.h5"),
        total_events=n,
    )


# --------------------------------------------------------------------------------------
# load_event_nexus — notes around the full-event-length allocations
# --------------------------------------------------------------------------------------


def test_event_loader_announces_each_allocation(nexus_file):
    """The four allocations are where the event path peaks; each must be named before it runs."""
    events, sink = _collect()

    load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=sink)

    details = [e.detail for e in events]
    assert any("events" in d for d in details), details
    assert any("time offsets" in d for d in details), details
    assert any("unrolling" in d for d in details), details
    assert any("converting tof" in d for d in details), details


def test_event_loader_counts_its_four_allocation_steps(nexus_file):
    """The loader has a known step sequence, so it counts 1..4 rather than reporting a single step.

    A `total=1` bar never renders a completion: tqdm draws it at 0% and, with `leave=False`, clears
    the line at close before the lone update can be redrawn. Counting the steps makes the bar move
    through 25/50/75/100% and names which allocation is running at each point.
    """
    events, sink = _collect()

    load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=sink)

    assert [e.completed for e in events] == [1, 2, 3, 4]
    assert {e.total for e in events} == {4}


def test_event_loader_stage_is_selectable(nexus_file):
    """An open-beam event load must not be reported as `load_sample`."""
    events, sink = _collect()

    load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=sink, stage=STAGE_LOAD_OB)

    assert {e.stage for e in events} == {STAGE_LOAD_OB}


def test_event_loader_progress_false_loads_identical_events(nexus_file):
    """Reporting must not change what is loaded."""
    without = load_event_nexus(nexus_file, detector_shape=DETECTOR)
    with_progress = load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=lambda _e: None)

    np.testing.assert_array_equal(without.tof, with_progress.tof)
    np.testing.assert_array_equal(without.x, with_progress.x)
    np.testing.assert_array_equal(without.y, with_progress.y)
    assert without.total_events == with_progress.total_events


def test_event_loader_releases_its_own_bar_when_the_read_fails(tmp_path, recorded_sinks):
    """A failure AFTER the first event is emitted must still release the bar.

    Getting this test to mean anything took three attempts, recorded so it is not weakened later:
      - a MISSING file raises before `resolve_progress` runs, so no reporter exists;
      - a file with no event bank raises before the first `report(...)`, and a bar is only built when
        the sink is first CALLED, so still nothing exists to leak.
    Both passed with the release deleted. Only a failure past the first emit exercises the context
    manager, so here `event_time_offset` holds strings: the first step reports, then `.astype(float)`
    raises with a live bar.
    """
    path = tmp_path / "bad_offsets.nxs.h5"
    with h5py.File(path, "w") as hf:
        bank = hf.create_group("entry/bank1_events")
        bank.create_dataset("event_id", data=np.arange(8, dtype=np.int64))
        bank.create_dataset("event_time_offset", data=np.array([b"x"] * 8, dtype="S1"))

    with pytest.raises(ValueError):
        load_event_nexus(path, detector_shape=DETECTOR, progress=True)

    assert recorded_sinks, "no internal sink was created"
    assert recorded_sinks[0]._bars == {}, "a bar opened before the failure was not released"


# --------------------------------------------------------------------------------------
# both converters — mars_tpx3 uses the 2-D one, so covering only 3-D leaves it silent
# --------------------------------------------------------------------------------------


def _convert_3d(events, **kw):
    return convert_events_to_histogram(
        events,
        BinningConfig(bins=8, bin_space="tof", tof_range=(0, 16000)),
        flight_path=sc.scalar(25.0, unit="m"),
        x_bins=DETECTOR[0],
        y_bins=DETECTOR[1],
        **kw,
    )


def _convert_2d(events, **kw):
    return convert_events_to_2d_histogram(events, DETECTOR, **kw)


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_converter_emits_per_chunk_and_notes_the_variance_attach(convert):
    """One event per chunk (a single chunk at the default 500M size), then a variance note."""
    events, sink = _collect()

    convert(_events(), progress=sink)

    per_chunk = [e for e in events if e.detail.startswith("chunk")]
    assert [e.completed for e in per_chunk] == [1]
    assert {e.total for e in per_chunk} == {1}
    assert any("Poisson variance" in e.detail for e in events)


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_converter_emits_one_event_per_chunk_when_chunked(convert):
    """With a chunk size that forces several passes the count really progresses."""
    events, sink = _collect()

    convert(_events(300), chunk_size=100, progress=sink)

    per_chunk = [e for e in events if e.detail.startswith("chunk")]
    assert [e.completed for e in per_chunk] == [1, 2, 3]
    assert {e.total for e in per_chunk} == {3}


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_converter_progress_false_gives_identical_histogram(convert):
    """Reporting must not change the histogram, its variances or its coords."""
    data = _events()
    assert sc.identical(convert(data), convert(data, progress=lambda _e: None))


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_converter_cancellation_propagates(convert):
    """Raising from the callback aborts the conversion."""

    class _CancelledError(RuntimeError):
        pass

    def cancel(event):
        if event.completed >= 2:
            raise _CancelledError("stop")

    with pytest.raises(_CancelledError):
        convert(_events(300), chunk_size=100, progress=cancel)


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_converter_closes_a_sink_it_created_itself(convert, recorded_sinks):
    """With `progress=True` the converter builds the sink, so it must retire it."""
    convert(_events(300), chunk_size=100, progress=True)

    assert len(recorded_sinks) == 1
    assert recorded_sinks[0]._bars == {}


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_converter_does_not_close_a_caller_supplied_sink(convert):
    """A converter handed a pre-bound reporter must leave the caller's bars open.

    This is the pipeline case: venus_tpx3_event will call load_event_nexus AND a converter under one
    reporter, so either closing the shared sink would restart the other's bar from zero.
    """
    sink = _TqdmSink()
    caller = ProgressReporter(sink, STAGE_NORMALIZE, total=3, owns_sink=True)

    convert(_events(300), chunk_size=100, progress=caller)

    assert sink._bars, "the caller's bar was closed by the converter"
    caller.close()
    assert sink._bars == {}


def test_the_ad_hoc_percent_printer_is_gone():
    """The old `logger.info` chunk-percent print is replaced by the callback, not duplicated.

    It fired only every tenth chunk with n_chunks > 1, i.e. never below 5 billion events, so it was
    dead code for every real run — and it wrote to a channel a caller cannot redirect or disable.
    """
    source = Path("src/neunorm/tof/event_converter.py").read_text()
    assert "Progress:" not in source
    assert "progress = (i + 1) / n_chunks" not in source


# --------------------------------------------------------------------------------------
# what the bar RENDERS — the class of check that caught the Task 3 defect
# --------------------------------------------------------------------------------------


def _render(fn):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        fn()
    return buffer.getvalue()


def test_rendered_event_loader_bar_is_drawn_with_the_right_shape(nexus_file):
    """The bar must be drawn, with this stage's name and its four-step denominator.

    Deliberately NOT asserting that it reaches 100% here: the four steps over a 400-event fixture
    finish inside tqdm's 0.1 s `mininterval`, so tqdm draws only the opening frame and `leave=False`
    clears the line at close. That is tqdm behaving as designed, not a defect, and asserting 100%
    would make this test a hostage to fixture size.

    Completion IS covered where the work is slow enough to redraw — see
    test_rendered_bar_reaches_100_percent_and_never_goes_backwards in test_progress_load_path.py,
    over 1000 files. Verified by hand on a 40,000,000-event, 610 MiB NeXus file, where this bar
    rendered:
        load_sample:   0%|          | 0/4 [00:00<?, ?item/s]
        load_sample:  50%|#####     | 2/4 [00:00<00:00, 17.78item/s, reading 40,000,000 events]
        load_sample: 100%|##########| 4/4 [00:00<00:00,  7.64item/s, unrolling event ids to x, y]
    """
    out = _render(lambda: load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=True))

    assert "load_sample" in out, f"no bar was drawn at all:\n{out}"
    assert "0/4" in out, f"bar has the wrong denominator (expected 4 steps):\n{out}"


@pytest.mark.parametrize("convert", [_convert_3d, _convert_2d], ids=["3d", "2d"])
def test_rendered_converter_bar_completes(convert):
    """Both converters must render a completing bar under progress=True."""
    out = _render(lambda: convert(_events(300), chunk_size=100, progress=True))

    assert "100%" in out, f"bar never completed:\n{out}"
    percents = [int(m) for m in re.findall(r"(\d+)%\|", out)]
    assert percents == sorted(percents), f"rendered progress went backwards: {percents}"


def test_resolve_progress_is_usable_as_a_context_manager():
    """`with resolve_progress(...) as report:` is the shape instrumented functions use, so that
    forgetting a `finally` cannot leak a bar again."""
    sink = _TqdmSink()
    reporter = ProgressReporter(sink, STAGE_NORMALIZE, total=2, owns_sink=True)

    with pytest.raises(RuntimeError, match="boom"):
        with reporter as report:
            report()
            assert sink._bars, "a bar should be open inside the block"
            raise RuntimeError("boom")

    assert sink._bars == {}, "__exit__ must release the bars even when the body raises"
    assert resolve_progress(False).__enter__() is not None
