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
from loguru import logger

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


def test_event_loader_names_each_step_then_counts_it(nexus_file):
    """Each allocation is NAMED before it runs and COUNTED after it returns.

    Naming first is what makes a stalled bar diagnostic — it says which allocation is thrashing.
    Counting only on success is what keeps the count honest: advancing first would report work that a
    failed read never did. See test_event_loader_count_does_not_overstate_a_failed_read.
    """
    events, sink = _collect()

    load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=sink)

    # note(name) at the current count, then an advancing tick with no detail
    assert [(e.completed, bool(e.detail)) for e in events] == [
        (0, True),
        (1, False),
        (1, True),
        (2, False),
        (2, True),
        (3, False),
        (3, True),
        (4, False),
    ]
    assert {e.total for e in events} == {4}
    named = [e.detail for e in events if e.detail]
    assert named == [
        f"reading {400:,} events",
        f"reading {400:,} time offsets",
        "unrolling event ids to x, y",
        "converting tof to ns",
    ]


def test_event_loader_count_does_not_overstate_a_failed_read(tmp_path):
    """A step that raises must not have been counted.

    Regression: the four steps used to advance BEFORE their work, so a failed slab read left the bar
    claiming a completed step that never happened.
    """
    path = tmp_path / "bad_offsets.nxs.h5"
    with h5py.File(path, "w") as hf:
        bank = hf.create_group("entry/bank1_events")
        bank.create_dataset("event_id", data=np.arange(8, dtype=np.int64))
        bank.create_dataset("event_time_offset", data=np.array([b"x"] * 8, dtype="S1"))

    events, sink = _collect()
    with pytest.raises(ValueError):
        load_event_nexus(path, detector_shape=DETECTOR, progress=sink)

    # step 1 (event_id) succeeded; step 2 (time offsets) raised, so the count must stop at 1
    assert max(e.completed for e in events) == 1


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
@pytest.mark.parametrize("n_events", [300, 250, 100, 1], ids=["exact", "remainder", "one-chunk", "one-event"])
def test_converter_emits_one_event_per_chunk_when_chunked(convert, n_events):
    """One event per chunk, and `total` must equal the real number of iterations.

    `n_chunks = ceil(n_events / chunk_size)` has to match `range(0, n_events, chunk_size)` exactly or
    the last chunk reads "chunk N of N-1". Parametrised over an exact split, a split with a remainder
    (250/100 -> 3), a single full chunk and a single event, since only the exact case was covered.
    """
    expected = -(-n_events // 100)
    events, sink = _collect()

    convert(_events(n_events), chunk_size=100, progress=sink)

    per_chunk = [e for e in events if e.detail.startswith("chunk")]
    assert [e.completed for e in per_chunk] == list(range(1, expected + 1))
    assert {e.total for e in per_chunk} == {expected}


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


def test_chunk_progress_goes_to_the_callback_not_the_log():
    """A multi-chunk conversion reports through the callback and logs no per-chunk progress line.

    Replaces a source grep, which tested text rather than behaviour: it passed if an equivalent
    percent print were reintroduced with different wording, and it read the source by a CWD-relative
    path so it failed outright when pytest ran from anywhere but the repo root.

    The old print fired only every tenth chunk when there was more than one — never below five
    billion events — and wrote to a channel a caller cannot redirect or disable.
    """
    captured = io.StringIO()
    sink_id = logger.add(captured, level="INFO", format="{message}")
    try:
        events, sink = _collect()
        _convert_3d(_events(300), chunk_size=100, progress=sink)
        logged = captured.getvalue()
    finally:
        logger.remove(sink_id)

    # positive control: this conversion DOES log, so an empty capture would not prove anything
    assert "Converting" in logged, f"the loguru sink captured nothing at all:\n{logged}"
    # ...but not a per-chunk progress line
    assert "Progress:" not in logged
    assert "chunks (" not in logged
    # the per-chunk information went to the callback instead
    assert [e.completed for e in events if e.detail.startswith("chunk")] == [1, 2, 3]


# --------------------------------------------------------------------------------------
# what the bar RENDERS — the class of check that caught the Task 3 defect
# --------------------------------------------------------------------------------------


def _render(fn):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        fn()
    return buffer.getvalue()


def test_rendered_event_loader_bar_reaches_100_percent(nexus_file):
    """The completed state must be drawn, on any fixture size.

    This became assertable only once `close()` refreshes each bar before retiring it. Before that,
    tqdm suppressed the final tick inside its 0.1 s `mininterval` and a fast four-step load rendered
    `0% -> 25% -> 50% -> 75%` and then vanished — indistinguishable from a stall. Found by rendering a
    20,000,000-event file; no event-level assertion could see it, because the events were correct.
    """
    out = _render(lambda: load_event_nexus(nexus_file, detector_shape=DETECTOR, progress=True))

    assert "load_sample" in out, f"no bar was drawn at all:\n{out}"
    assert "0/4" in out, f"bar has the wrong denominator (expected 4 steps):\n{out}"
    assert "100%" in out, f"bar never rendered its completion:\n{out}"
    percents = [int(m) for m in re.findall(r"(\d+)%\|", out)]
    assert percents == sorted(percents), f"rendered progress went backwards: {percents}"


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
