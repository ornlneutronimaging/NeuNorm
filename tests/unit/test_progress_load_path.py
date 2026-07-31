"""Progress reporting through the TIFF/FITS load path.

The load loop is the unit a user counts — "1000 files" — and it is the only place where a slow or
contended filesystem becomes visible per item. The two ticks after the loop matter for a different
reason: the loop only appends to a list, so the memory peak (a measured ~5x multiple of the stack,
from `np.stack` plus the variances copy) lands *after* the last file. Without them a bar reaches
100% and then goes silent through the part that can exhaust RAM.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest
import scipp as sc
from loguru import logger

from neunorm.loaders.fits_loader import load_fits_stack
from neunorm.loaders.stack_loader import load_stack
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.utils.progress import (
    STAGE_LOAD_DARK,
    STAGE_LOAD_OB,
    STAGE_LOAD_SAMPLE,
    ProgressReporter,
    _TqdmSink,
    resolve_progress,
)

DATA = Path(__file__).parent.parent / "data"


def _tiffs():
    return sorted((DATA / "tif" / "sample").glob("*.tif"))


def _fits():
    return sorted((DATA / "fits" / "sample").glob("image00*.fits"))


def _collect():
    events = []
    return events, events.append


# --------------------------------------------------------------------------------------
# per-file reporting
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_loader_emits_one_event_per_file(loader, paths_fn):
    """One advancing event per file, counting up to the number of files, each naming its file."""
    paths = paths_fn()
    assert len(paths) == 3, "fixture changed; the assertions below assume 3 files"
    events, sink = _collect()

    loader(paths, progress=sink)

    names = {p.name for p in paths}
    per_file = [e for e in events if e.detail in names]
    assert [e.completed for e in per_file] == [1, 2, 3]
    assert {e.total for e in per_file} == {3}
    assert {e.stage for e in per_file} == {STAGE_LOAD_SAMPLE}
    assert [e.detail for e in per_file] == [p.name for p in paths]


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_loader_announces_the_post_loop_allocations_without_advancing(loader, paths_fn):
    """The stack build and the variances copy are announced after the last file, as notes.

    They are announcements, not completions: each fires *before* its allocation so a bar that stops
    there tells the user exactly where the run is stuck. They therefore must not advance the count —
    `completed` is documented as absolute and monotonic, and a fresh per-call stage reporter would
    restart at 1 on every load and leave a bar frozen at its first tick.
    """
    events, sink = _collect()

    loader(paths_fn(), progress=sink)

    notes = [e for e in events if e.detail.startswith(("stacking", "attaching variances"))]
    assert len(notes) == 2
    assert notes[0].detail.startswith("stacking")
    assert notes[1].detail.startswith("attaching variances")
    # after the last file, and not advancing past it
    assert [e.completed for e in notes] == [3, 3]
    assert events[-2:] == notes
    assert {e.stage for e in notes} == {STAGE_LOAD_SAMPLE}


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_count_stays_monotonic_across_two_loads_on_one_reporter(loader, paths_fn):
    """Regression: the allocation announcements must not restart the count on the second load.

    They were emitted through `for_stage(..., total=None)`, which builds a reporter with offset 0,
    so every load re-emitted completed=1 for those stages. `_TqdmSink` only advances when
    `completed > bar.n`, so run 2 onwards was silently dropped and the documented adapter
    `bar.update(event.completed - bar.n)` did nothing.
    """
    paths = paths_fn()
    events, sink = _collect()
    base = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=2 * len(paths))

    loader(paths, progress=base.with_offset(0))
    loader(paths, progress=base.with_offset(len(paths)))

    counts = [e.completed for e in events]
    assert counts == sorted(counts), f"count went backwards across loads: {counts}"
    assert max(counts) == 2 * len(paths)
    assert {e.total for e in events} == {2 * len(paths)}


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_loaders_accept_a_non_sized_iterable(paths_fn):
    """`Path.glob()` and generators loaded fine before this instrumentation and must still.

    Regression: taking `len(paths)` for the progress total made both loaders raise TypeError before
    reading anything — even with the default `progress=False`.
    """
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack

    assert loader(iter(paths)).sizes == loader(paths).sizes

    events, sink = _collect()
    loader((p for p in paths), progress=sink)
    assert {e.total for e in events} == {3}, "a materialised iterable must still yield a real total"


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_stage_label_is_caller_selectable(paths_fn):
    """A direct open-beam or dark load must not be reported as `load_sample`."""
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack
    events, sink = _collect()

    loader(paths, progress=sink, stage=STAGE_LOAD_OB)

    assert {e.stage for e in events} == {STAGE_LOAD_OB}


# --------------------------------------------------------------------------------------
# load_stack pass-through — the CCD pipelines' only route to per-file progress
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_load_stack_passes_progress_to_whichever_leaf_it_picks(paths_fn):
    """`load_stack` dispatches on extension; both branches must forward `progress` and `stage`.

    Without this the CCD pipelines report nothing per file, because they call `load_stack` rather
    than a leaf loader.
    """
    paths = paths_fn()
    names = {p.name for p in paths}
    events, sink = _collect()

    load_stack(paths, progress=sink, stage=STAGE_LOAD_DARK)

    assert [e.completed for e in events if e.detail in names] == [1, 2, 3]
    assert {e.stage for e in events} == {STAGE_LOAD_DARK}


# --------------------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------------------


class _CancelledError(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_callback_raising_mid_load_propagates(loader, paths_fn):
    """Raising from the callback is how a caller aborts a long load."""

    def cancel(event):
        if event.completed == 2:
            raise _CancelledError("stop")

    with pytest.raises(_CancelledError):
        loader(paths_fn(), progress=cancel)


@pytest.mark.parametrize(
    ("loader", "paths_fn", "message"),
    [
        (load_tiff_stack, _tiffs, "Error loading TIFF stack"),
        (load_fits_stack, _fits, "Failed to load FITS files"),
    ],
    ids=["tiff", "fits"],
)
def test_cancelling_is_not_reported_as_a_read_failure(loader, paths_fn, message):
    """A cancelling callback must not be logged as an I/O error.

    Both loaders wrap their read in `except Exception: logger.error(...); raise`. If the tick were
    emitted inside that try, cancelling would tell the user their files failed to load. This pins
    the tick's placement, not merely its existence.

    Captured through a loguru sink, NOT pytest's `caplog`: loguru does not route to stdlib logging,
    so `caplog.text` is always empty here and any `not in caplog.text` assertion would pass no
    matter where the tick sat. The first half of this test is a positive control proving the sink
    really does capture the message, so the negative half cannot pass by the sink being broken.
    """
    captured = io.StringIO()
    sink_id = logger.add(captured, level="ERROR", format="{message}")
    try:
        # Positive control: a real read failure MUST reach the sink.
        with pytest.raises(Exception, match=".*"):
            loader([Path("no-such-directory") / "missing.tif"], progress=False)
        assert message in captured.getvalue(), "the loguru sink is not capturing the loader's error"

        captured.seek(0)
        captured.truncate()

        # The actual assertion: cancelling must not produce that same error line.
        def cancel(event):
            if event.completed == 1:
                raise _CancelledError("stop")

        with pytest.raises(_CancelledError):
            loader(paths_fn(), progress=cancel)

        assert message not in captured.getvalue()
    finally:
        logger.remove(sink_id)


# --------------------------------------------------------------------------------------
# the default path is untouched
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_progress_false_loads_identical_data(loader, paths_fn):
    """Reporting must not change what is loaded, including variances and coordinates."""
    paths = paths_fn()
    without = loader(paths)
    with_progress = loader(paths, progress=lambda _event: None)

    # sc.identical covers values, variances, units, dims, coord VALUES and alignment in one
    # assertion; comparing coordinate names only would pass on a change to any of the rest.
    assert sc.identical(without, with_progress)


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_bad_progress_value_is_rejected_at_the_loader(loader, paths_fn):
    """An invalid `progress` must fail loudly rather than silently disable reporting."""
    with pytest.raises(TypeError, match="progress must be False"):
        loader(paths_fn(), progress=1)


# --------------------------------------------------------------------------------------
# bar release — the loader owns the sink, so only its `finally` can close it
# --------------------------------------------------------------------------------------


class _RecordingTqdmSink(_TqdmSink):
    """A `_TqdmSink` that records every instance, so a sink a loader built internally is observable.

    Needed because `progress=True` builds the sink inside the callee. Checking `tqdm._instances`
    instead does not work: CPython drops the callee's reporter at return and `tqdm.__del__` closes
    the bars, so that assertion passes even with the release deleted — verified.
    """

    instances: list["_RecordingTqdmSink"] = []

    def __init__(self):
        super().__init__()
        _RecordingTqdmSink.instances.append(self)


@pytest.fixture
def recorded_sinks(monkeypatch):
    _RecordingTqdmSink.instances = []
    monkeypatch.setattr("neunorm.utils.progress._TqdmSink", _RecordingTqdmSink)
    return _RecordingTqdmSink.instances


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_leaf_does_not_close_a_caller_supplied_sink(paths_fn):
    """Regression: a loader handed a pre-bound reporter must leave the caller's bars alone.

    A bar is no longer auto-closed at completion, so if a leaf closed the caller's sink the caller's
    next event would rebuild the bar from zero. Rendered, a pipeline's bar flickered back to 0% on
    every loader call. `resolve_progress` now hands a callee a borrowed, non-owning view.
    """
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack
    sink = _TqdmSink()
    caller = ProgressReporter(sink, STAGE_LOAD_SAMPLE, total=2 * len(paths), owns_sink=True)

    loader(paths, progress=caller.with_offset(0))
    assert sink._bars, "the caller's bar was closed by the callee"
    first = sink._bars[STAGE_LOAD_SAMPLE]
    assert first.n == len(paths)

    loader(paths, progress=caller.with_offset(len(paths)))
    assert sink._bars[STAGE_LOAD_SAMPLE] is first, "the bar was rebuilt instead of continuing"
    assert first.n == 2 * len(paths), "the count restarted instead of advancing"

    caller.close()
    assert sink._bars == {}


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_leaf_closes_a_sink_it_created_itself(paths_fn, recorded_sinks):
    """With `progress=True` the loader builds the sink, so the loader must retire it."""
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack

    loader(paths, progress=True)

    assert len(recorded_sinks) == 1, f"expected exactly one internal sink, got {len(recorded_sinks)}"
    assert recorded_sinks[0]._bars == {}, "loader left its own bar open after a clean load"


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_leaf_closes_its_own_sink_on_a_read_failure(paths_fn, recorded_sinks):
    """The error path is where a bar leaked: the stage is abandoned, so only the context manager
    can release it."""
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack
    missing = Path("no-such-directory") / f"missing{paths[0].suffix}"

    with pytest.raises(Exception, match=".*"):
        loader([paths[0], missing], progress=True)

    assert recorded_sinks, "no internal sink was created"
    assert recorded_sinks[0]._bars == {}, "loader left its own bar open after a failed read"


# --------------------------------------------------------------------------------------
# what the bar actually RENDERS — no event-level test can see this
# --------------------------------------------------------------------------------------


def _render(loader, paths):
    """Return the text tqdm draws for a real `progress=True` load.

    `contextlib.redirect_stderr` works here because tqdm resolves `sys.stderr` when it builds the
    bar. (It does NOT work for loguru, which holds its own stream reference — see the cancellation
    test, which uses a loguru sink for that reason.)
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        loader(paths, progress=True)
    return buffer.getvalue()


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_rendered_bar_reaches_100_percent_and_never_goes_backwards(paths_fn):
    """Regression, found by watching a real 1000-file load rather than by any event assertion.

    The sink used to close and forget a bar at completion, so the two allocation notes that follow
    the read loop each built a NEW bar at 0% and closed it again. The rendered count ran
    0% -> 32% -> 64% -> 0% -> 0%, i.e. it appeared to restart twice at the very end of the load.
    Every event-level test passed throughout, because the events themselves were correct.
    """
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack

    out = _render(loader, paths)

    assert "100%" in out, f"bar never reached 100%:\n{out}"
    percents = [int(m) for m in re.findall(r"(\d+)%\|", out)]
    assert percents == sorted(percents), f"rendered progress went backwards: {percents}"


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_rendered_bar_shows_the_allocation_notes(paths_fn):
    """The notes exist to name the phase a stalled bar is stuck in, so they must be VISIBLE.

    They were previously erased: each landed on a freshly-built bar that was closed immediately,
    and `leave=False` clears a closed bar's line.
    """
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack

    out = _render(loader, paths)

    assert "stacking" in out, f"the stack-build note never rendered:\n{out}"
    assert "attaching variances" in out, f"the variances note never rendered:\n{out}"


# Not asserted here: that the per-file detail is VISIBLE mid-load. Whether tqdm redraws between two
# items is its own `mininterval` policy, not NeuNorm's contract, and on this 3-file fixture the load
# finishes inside one interval so only the first and last frames are ever drawn. Delivery of the
# detail is covered by test_loader_emits_one_event_per_file; its rendering was confirmed by hand on
# a 1000-file load, where the bar read
#   load_sample:  32%|###2  | 322/1000 [00:00<00:00, 3217.59item/s, frame_00320.tif]
