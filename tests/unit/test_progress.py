"""Unit tests for neunorm.utils.progress — the progress-reporting contract (#195).

Covers the False/True/callable dispatch, event immutability, the offset arithmetic that lets a
per-run leaf function contribute to one count spanning the whole run, and the loguru/tqdm
stream handoff.
"""

import io
import subprocess
import sys

import pytest
from loguru import logger

from neunorm.utils.progress import (
    NULL_REPORTER,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
    ProgressEvent,
    ProgressReporter,
    resolve_progress,
    tqdm_safe_logging,
)


def _collect():
    """A callback plus the list it appends to."""
    events = []
    return events, events.append


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------


def test_progress_false_is_the_shared_noop_reporter():
    """`progress=False` must cost nothing: the same singleton every time, and deriving from it
    returns itself rather than allocating, so threading it through a pipeline is free."""
    reporter = resolve_progress(False, STAGE_LOAD_SAMPLE, total=10)
    assert reporter is NULL_REPORTER
    assert reporter.for_stage(STAGE_NORMALIZE) is reporter
    assert reporter.with_offset(500) is reporter
    reporter(advance=3, detail="ignored")  # must not raise and must emit nothing


def test_progress_callable_receives_events():
    """A user callback gets one event per call, carrying the bound stage and total."""
    events, sink = _collect()
    reporter = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=3)
    reporter(detail="frame_00000.tif")
    reporter(detail="frame_00001.tif")

    assert [(e.stage, e.completed, e.total, e.detail) for e in events] == [
        (STAGE_LOAD_SAMPLE, 1, 3, "frame_00000.tif"),
        (STAGE_LOAD_SAMPLE, 2, 3, "frame_00001.tif"),
    ]


def test_progress_true_builds_a_tqdm_backed_reporter():
    """`progress=True` yields a working reporter that does not need a user callback."""
    reporter = resolve_progress(True, STAGE_LOAD_SAMPLE, total=2)
    assert isinstance(reporter, ProgressReporter)
    assert reporter is not NULL_REPORTER
    assert (reporter.stage, reporter.total) == (STAGE_LOAD_SAMPLE, 2)
    reporter(detail="a")  # drives tqdm for real; must not raise
    reporter(detail="b")
    assert reporter.completed == 2


@pytest.mark.parametrize("bad", [0, 1, "yes", 2.5, [], {}, None])
def test_progress_rejects_anything_else(bad):
    """Anything that is not False/True/callable is a TypeError, not a silent no-op. `0`/`1` are
    included deliberately: they are not `False`/`True` identities and must not sneak through."""
    with pytest.raises(TypeError, match="progress must be False"):
        resolve_progress(bad)


def test_resolve_progress_is_idempotent():
    """A reporter passed back in is reused, so a function can accept either a user's `progress`
    value or a reporter its caller already built."""
    events, sink = _collect()
    original = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=5)

    assert resolve_progress(original) is original

    restaged = resolve_progress(original, STAGE_NORMALIZE)
    assert restaged is not original
    assert (restaged.stage, restaged.total) == (STAGE_NORMALIZE, None)


# --------------------------------------------------------------------------------------
# the event
# --------------------------------------------------------------------------------------


def test_progress_event_is_immutable():
    """Events are handed to user code; they must not be mutable shared state."""
    event = ProgressEvent(STAGE_LOAD_SAMPLE, 1, 10, "x")
    with pytest.raises(Exception, match="cannot assign to field"):
        event.completed = 2


def test_progress_event_detail_defaults_to_empty():
    """`detail` is optional so emit sites that have nothing to say can omit it."""
    assert ProgressEvent(STAGE_NORMALIZE, 1, None).detail == ""


# --------------------------------------------------------------------------------------
# offset arithmetic — the reason reporters are pre-bound
# --------------------------------------------------------------------------------------


def test_with_offset_produces_one_flat_count_across_runs():
    """A leaf loader is called once per input run and counts from zero each time. Binding an
    offset per run is what turns those restarts into a single count over all files, which is what
    a determinate progress bar needs."""
    events, sink = _collect()
    base = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=5)

    # run 1 has 2 files, run 2 has 3; each "leaf call" gets its own offset reporter
    offset = 0
    for n_files in (2, 3):
        leaf = base.with_offset(offset)
        for _ in range(n_files):
            leaf()
        offset += n_files

    assert [e.completed for e in events] == [1, 2, 3, 4, 5]
    assert {e.total for e in events} == {5}


def test_with_offset_preserves_stage_and_total():
    """Offsetting must not lose the stage label or the grand total."""
    _, sink = _collect()
    base = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=7)
    derived = base.with_offset(4)
    assert (derived.stage, derived.total) == (STAGE_LOAD_SAMPLE, 7)


def test_for_stage_starts_a_fresh_count():
    """A new stage restarts at 1 and may carry a different total (or none)."""
    events, sink = _collect()
    base = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=2)
    base()
    step = base.for_stage(STAGE_NORMALIZE)
    step(detail="proton-charge correction")
    step(detail="division")

    assert [(e.stage, e.completed, e.total) for e in events] == [
        (STAGE_LOAD_SAMPLE, 1, 2),
        (STAGE_NORMALIZE, 1, None),
        (STAGE_NORMALIZE, 2, None),
    ]


def test_advance_greater_than_one():
    """Emit sites that finish several items at once can advance by more than one."""
    events, sink = _collect()
    reporter = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=10)
    reporter(advance=4)
    assert events[-1].completed == 4


# --------------------------------------------------------------------------------------
# the contract that makes progress libraries work
# --------------------------------------------------------------------------------------


def test_absolute_counts_drive_a_tqdm_bar_correctly():
    """`completed` is absolute, and tqdm's update() takes a DELTA, so the documented adapter is
    `bar.update(event.completed - bar.n)`. Pin it: a naive `bar.update(event.completed)` would
    overshoot, which is the trap this contract exists to avoid."""
    from tqdm.auto import tqdm

    bar = tqdm(total=5, file=io.StringIO(), leave=False)
    reporter = resolve_progress(lambda e: bar.update(e.completed - bar.n), STAGE_LOAD_SAMPLE, total=5)
    for _ in range(5):
        reporter()

    assert bar.n == 5
    bar.close()


def test_tqdm_sink_adopts_a_total_that_becomes_known_late():
    """Event-mode reads cannot know their item count up front, so a stage may start indeterminate
    and learn its total afterwards. The bar must adopt it rather than stay open-ended."""
    from neunorm.utils.progress import _TqdmSink

    sink = _TqdmSink()
    sink(ProgressEvent(STAGE_NORMALIZE, 1, None, "reading"))
    bar = sink._bars[STAGE_NORMALIZE]
    assert bar.total is None

    sink(ProgressEvent(STAGE_NORMALIZE, 2, 4, "counted"))
    assert bar.total == 4
    assert bar.n == 2
    bar.close()


def test_tqdm_sink_closes_and_forgets_a_finished_bar():
    """A completed stage closes its bar and drops it, so a later stage of the same name starts a
    fresh one instead of writing to a closed bar."""
    from neunorm.utils.progress import _TqdmSink

    sink = _TqdmSink()
    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 1, 2))
    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 2, 2))
    assert STAGE_LOAD_SAMPLE not in sink._bars

    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 1, 2))  # must not raise on a reused stage label
    assert STAGE_LOAD_SAMPLE in sink._bars
    sink._bars[STAGE_LOAD_SAMPLE].close()


def test_callback_exception_propagates_for_cancellation():
    """Raising from the callback is how a caller aborts a long run, so the reporter must not
    swallow it."""

    class RunCancelledError(RuntimeError):
        pass

    def cancel(event):
        if event.completed == 2:
            raise RunCancelledError("stop")

    reporter = resolve_progress(cancel, STAGE_LOAD_SAMPLE, total=10)
    reporter()
    with pytest.raises(RunCancelledError):
        reporter()


def test_tqdm_is_not_imported_when_progress_is_unused():
    """Importing NeuNorm must not drag in tqdm.auto, which probes for ipywidgets. Checked in a
    fresh interpreter because other tests in this session may already have imported it."""
    code = (
        "import sys; import neunorm.utils.progress as p; "
        "assert p.resolve_progress(False) is p.NULL_REPORTER; "
        "print('tqdm.auto' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", f"tqdm.auto imported eagerly: {out.stdout!r}"


# --------------------------------------------------------------------------------------
# loguru / bar stream handoff
# --------------------------------------------------------------------------------------


def test_tqdm_safe_logging_swaps_stderr_handlers_and_restores():
    """Inside the context, loguru's stderr handler is replaced (so records go through tqdm.write
    instead of landing on the bar's line); afterwards a stderr handler exists again."""

    def stderr_ids():
        # `logger._core.handlers` must be re-read every time: loguru's remove()/add() REPLACE the
        # dict rather than mutating it, so a captured reference goes stale and would still show the
        # displaced handler. (No public API for this; see the function's docstring.)
        return {
            handler_id
            for handler_id, handler in logger._core.handlers.items()
            if getattr(getattr(handler, "_sink", None), "_stream", None) is sys.stderr
        }

    before = stderr_ids()
    assert before, "expected loguru's default stderr handler to be present"

    with tqdm_safe_logging():
        assert not stderr_ids(), "the stderr handler should be displaced inside the context"

    after = stderr_ids()
    assert after, "a stderr handler must be restored on exit"
    assert len(after) == len(before)


def test_tqdm_safe_logging_leaves_a_reconfigured_logger_alone():
    """If a host application has taken loguru over and nothing writes to stderr, NeuNorm must not
    fight for the stream — and must not crash trying to remove handler 0, which no longer exists."""
    handlers = logger._core.handlers
    saved = dict(handlers)
    buffer = io.StringIO()
    for handler_id in list(saved):
        logger.remove(handler_id)
    app_sink = logger.add(buffer, level="INFO")
    try:
        with tqdm_safe_logging():
            logger.info("host application still owns the stream")
        assert "host application still owns the stream" in buffer.getvalue()
        assert app_sink in logger._core.handlers
    finally:
        logger.remove(app_sink)
        logger.add(sys.stderr, level="DEBUG")
