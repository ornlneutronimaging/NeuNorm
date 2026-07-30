"""Unit tests for neunorm.utils.progress — the progress-reporting contract (#195).

Covers the False/True/callable dispatch, event immutability, the offset arithmetic that lets a
per-run leaf function contribute to one count spanning the whole run, sink ownership, and the tqdm
bar lifecycle.
"""

import io
import subprocess
import sys

import pytest

from neunorm.utils.progress import (
    NULL_REPORTER,
    STAGE_LOAD_SAMPLE,
    STAGE_NORMALIZE,
    ProgressEvent,
    ProgressReporter,
    resolve_progress,
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


def test_resolve_progress_borrows_an_existing_reporter_without_rebinding_it():
    """An existing reporter keeps its stage, total and position — but comes back NOT owning the sink.

    Two separate rules meet here.

    Not re-binding: a stage or total supplied by the callee must be ignored, or a leaf would reset the
    offset to zero and replace the run-wide total with its own local count.

    Not owning: only whoever resolved a sink into being may retire it. A callee using
    `with resolve_progress(...)` would otherwise close the caller's bars on exit, and because a bar is
    no longer auto-closed at completion the caller's next event would rebuild it from zero — making a
    pipeline's bar flicker back to 0% on every instrumented call it makes.
    """
    _, sink = _collect()
    original = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=5)
    original(advance=2)

    borrowed = resolve_progress(original, STAGE_NORMALIZE, total=99)

    assert (borrowed.stage, borrowed.total) == (STAGE_LOAD_SAMPLE, 5), "stage/total must not be rebound"
    assert borrowed.completed == original.completed, "position must carry over"
    # A callable-backed reporter never owned the sink to begin with — the user's callback IS the
    # sink. Ownership only arises for progress=True; the next test covers that case.
    assert borrowed._owns_sink is False


def test_a_borrowed_reporter_shares_the_caller_s_count():
    """A callee's advances must move the SAME count the caller reads.

    Regression: `_borrowed()` used to snapshot the caller's position into a fresh counter, so after a
    callee advanced, the caller's next advance re-emitted a number the callee had already reported —
    the sequence went [1, 2, 3, 4, 3]. `ProgressEvent.completed` is documented absolute, and the
    documented adapter `bar.update(event.completed - bar.n)` would compute a NEGATIVE delta there.
    """
    events, sink = _collect()
    caller = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=6)
    caller()
    caller()

    callee = resolve_progress(caller)  # borrowed view handed to an inner function
    callee()
    callee()

    caller()  # the caller carries on afterwards

    counts = [e.completed for e in events]
    assert counts == [1, 2, 3, 4, 5], counts
    assert counts == sorted(counts), "the absolute count went backwards"
    assert caller.completed == 5, "the caller must see work the callee did"


def test_a_borrowed_reporter_close_does_not_touch_the_owner_s_sink():
    """Closing a borrowed reporter is a no-op; closing the owning one releases the bars."""
    from neunorm.utils.progress import _TqdmSink

    tqdm_sink = _TqdmSink()
    owner = ProgressReporter(tqdm_sink, STAGE_LOAD_SAMPLE, total=2, owns_sink=True)
    owner()
    assert tqdm_sink._bars, "a bar should be open"

    resolve_progress(owner).close()
    assert tqdm_sink._bars, "a borrowed reporter must not close the owner's bars"

    owner.close()
    assert tqdm_sink._bars == {}


def test_leaf_resolving_a_reporter_keeps_the_flat_count():
    """Regression: a leaf that resolves the reporter its caller passed down must not restart the
    count or shrink the denominator.

    This is the exact pattern the module docstring prescribes — a pipeline binds the grand total and
    a per-run offset, and each leaf call resolves what it was handed. It previously emitted
    completed=[1, 2, 1, 2] with total=2 throughout, instead of [1, 2, 3, 4] with total=4.
    """
    events, sink = _collect()
    base = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=4)

    offset = 0
    for run_files in (["a", "b"], ["c", "d"]):
        leaf = resolve_progress(base.with_offset(offset), STAGE_LOAD_SAMPLE, total=len(run_files))
        for name in run_files:
            leaf(detail=name)
        offset += len(run_files)

    assert [e.completed for e in events] == [1, 2, 3, 4]
    assert {e.total for e in events} == {4}


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


@pytest.mark.parametrize("bad", [0, -1, 0.9, 2.5, "3", None, True])
def test_advance_must_be_a_positive_int(bad):
    """A float would be truncated (advance=0.9 -> an event that advanced nothing) and a negative
    would drive the absolute count backwards, so a bad emit site must fail at the call."""
    _, sink = _collect()
    reporter = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=10)
    with pytest.raises((TypeError, ValueError)):
        reporter(advance=bad)


@pytest.mark.parametrize("bad", [0, -1, 0.9, "3"])
def test_noop_reporter_validates_advance_too(bad):
    """The progress=False path must reject a bad advance as well. Nearly the whole suite runs with
    progress disabled, so validating only when reporting is enabled would let a broken emit site
    ship and first surface in a user's session with progress=True."""
    with pytest.raises((TypeError, ValueError)):
        NULL_REPORTER(advance=bad)


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


def test_tqdm_sink_keeps_a_completed_bar_so_later_events_land_on_it():
    """A bar that reaches its total must SURVIVE, so a later event for that stage updates it.

    It used to be closed and forgotten at completion, which looked tidy and rendered badly: a note
    arriving after the last item found no bar, built a fresh one at 0%, and closed that too. A real
    1000-file load showed the count run 0% -> 32% -> 64% and then snap back to 0% twice, with the
    notes erased before they could be read. `close()` is now the only thing that retires a bar.
    """
    from neunorm.utils.progress import _TqdmSink

    sink = _TqdmSink()
    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 1, 2))
    bar = sink._bars[STAGE_LOAD_SAMPLE]
    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 2, 2))

    assert bar.disable is False, "a completed bar must stay open until close()"
    assert sink._bars[STAGE_LOAD_SAMPLE] is bar

    # a note after completion updates the SAME bar rather than spawning another
    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 2, 2, "stacking 2 frames"))
    assert sink._bars[STAGE_LOAD_SAMPLE] is bar
    assert bar.n == 2

    sink.close()
    assert bar.disable is True
    assert sink._bars == {}


def test_tqdm_sink_close_finalizes_indeterminate_and_abandoned_bars():
    """A total=None step stage never reaches a completion point and an abandoned stage never
    finishes, so close() is the only thing that can release them."""
    from neunorm.utils.progress import _TqdmSink

    sink = _TqdmSink()
    sink(ProgressEvent(STAGE_NORMALIZE, 1, None, "step one"))  # indeterminate: never self-closes
    sink(ProgressEvent(STAGE_LOAD_SAMPLE, 1, 10))  # abandoned partway
    step_bar = sink._bars[STAGE_NORMALIZE]
    partial_bar = sink._bars[STAGE_LOAD_SAMPLE]
    assert (step_bar.disable, partial_bar.disable) == (False, False)

    sink.close()

    assert step_bar.disable is True
    assert partial_bar.disable is True
    assert sink._bars == {}


def test_reporter_close_is_safe_for_callbacks_and_noop():
    """close() is exposed on the reporter so a pipeline can call it in a finally regardless of which
    progress form the caller chose; it must be harmless when there is no sink to close."""
    _, sink = _collect()
    resolve_progress(sink, STAGE_LOAD_SAMPLE, total=2).close()
    NULL_REPORTER.close()
    resolve_progress(True, STAGE_LOAD_SAMPLE, total=2).close()


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


def test_reporter_never_closes_a_caller_supplied_callback():
    """close() must not release something the caller owns. A callback can be a reusable object that
    happens to have a close() method; only the tqdm sink NeuNorm built is NeuNorm's to close."""

    class ReusableSink:
        def __init__(self):
            self.closed = False
            self.events = []

        def __call__(self, event):
            self.events.append(event)

        def close(self):
            self.closed = True

    sink = ReusableSink()
    reporter = resolve_progress(sink, STAGE_LOAD_SAMPLE, total=1)
    reporter()
    reporter.close()

    assert sink.events, "the callback should still have received its event"
    assert sink.closed is False, "NeuNorm closed a sink it does not own"
