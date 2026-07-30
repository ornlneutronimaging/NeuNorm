"""Progress reporting through the TIFF/FITS load path (#195, Task 3).

The load loop is the unit a user counts — "1000 files" — and it is the only place where a slow or
contended filesystem becomes visible per item. The two ticks after the loop matter for a different
reason: the loop only appends to a list, so the memory peak (a measured ~5x multiple of the stack,
from `np.stack` plus the variances copy) lands *after* the last file. Without them a bar reaches
100% and then goes silent through the part that can exhaust RAM.
"""

import io
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


def _owned_sink_reporter(total):
    """A reporter over a tqdm sink WE keep a reference to, marked as NeuNorm-owned.

    Holding the sink is what makes the assertion meaningful. An earlier version of these tests
    checked `tqdm._instances` after calling the loader with `progress=True`; that passed even with
    the loader's `finally: report.close()` deleted, because CPython drops the loader's reporter at
    return and `tqdm.__del__` closes the bars anyway. Keeping the sink alive here means an
    unclosed bar stays observable.
    """
    sink = _TqdmSink()
    return sink, ProgressReporter(sink, STAGE_LOAD_SAMPLE, total=total, owns_sink=True)


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_loader_releases_owned_bars_on_success(paths_fn):
    """A clean load must leave no bar open in a sink the loader owns.

    Note this one does NOT guard the loader's `finally`: on success the per-file bar reaches its
    total and `_TqdmSink` closes it there, so it passes with `close()` deleted. The error-path test
    below is the actual regression guard.
    """
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack
    sink, reporter = _owned_sink_reporter(len(paths))

    loader(paths, progress=reporter)

    assert sink._bars == {}, "loader left a bar open after a clean load"


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_loader_releases_owned_bars_on_a_read_failure(paths_fn):
    """The error path is where the bar actually leaked: the stage is abandoned partway, so nothing
    reaches a completion point and only the loader's `finally` can close it."""
    paths = paths_fn()
    loader = load_tiff_stack if paths[0].suffix == ".tif" else load_fits_stack
    missing = Path("no-such-directory") / f"missing{paths[0].suffix}"
    sink, reporter = _owned_sink_reporter(2)

    with pytest.raises(Exception, match=".*"):
        loader([paths[0], missing], progress=reporter)

    assert sink._bars == {}, "loader left a bar open after a failed read"
