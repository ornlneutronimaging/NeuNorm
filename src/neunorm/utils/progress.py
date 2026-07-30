"""
Progress reporting for long NeuNorm runs.

Normalizing a large stack can take a long time, and without feedback a user cannot tell a slow
run from a hung one. This module defines the contract NeuNorm uses to report where a run is.

Callers choose one of three forms for the ``progress`` argument of an instrumented function:

- ``False`` (the default) — nothing is reported and nothing is allocated.
- ``True`` — NeuNorm drives a :mod:`tqdm` bar itself, one per stage.
- a callable — it receives a :class:`ProgressEvent` for every item or step.

Usage
-----
Let NeuNorm own the bar::

    >>> transmission = run_venus_tpx1_pipeline(..., progress=True)  # doctest: +SKIP

Or drive your own, which is what makes any progress library work::

    >>> from tqdm.auto import tqdm  # doctest: +SKIP
    >>> bar = tqdm(total=1000)  # doctest: +SKIP
    >>> def report(event):  # doctest: +SKIP
    ...     bar.update(event.completed - bar.n)  # tqdm.update() takes a DELTA
    >>> transmission = run_venus_tpx1_pipeline(..., progress=report)  # doctest: +SKIP

Notes
-----
- Events are emitted **synchronously, from the calling thread, in order**. The callback does not
  need to be thread-safe.
- A callback that raises is not caught: raising is how a caller cancels a long run.
- ``total`` is ``None`` when the item count is not knowable in advance (event-mode reads, and
  stages that have no item axis at all and instead report named steps).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np


def _check_advance(advance: int) -> None:
    """Reject an advance that is not a positive integer.

    Checked on the ``progress=False`` path too, deliberately. Almost every test in the suite runs
    with progress disabled, so validating only when reporting is enabled would let a bad emit site
    ship and surface for the first time in a user's session with ``progress=True``.
    """
    if isinstance(advance, bool) or not isinstance(advance, (int, np.integer)):
        raise TypeError(f"advance must be an int, got {type(advance).__name__}")
    if advance < 1:
        raise ValueError(f"advance must be >= 1, got {advance}")


# Stage labels. These are the shared vocabulary between the functions that emit progress and the
# tests that assert on the emitted sequence; keep them stable, they reach users through events.
STAGE_LOAD_SAMPLE = "load_sample"
STAGE_LOAD_OB = "load_ob"
STAGE_LOAD_DARK = "load_dark"
STAGE_STACK_FRAMES = "stack_frames"
STAGE_ATTACH_VARIANCES = "attach_variances"
STAGE_COMBINE_RUNS = "combine_runs"
STAGE_GAMMA_FILTER = "gamma_filter"
STAGE_REBIN_TOF = "rebin_tof"
STAGE_NORMALIZE = "normalize"
STAGE_EXPORT = "export"

__all__ = [
    "STAGE_ATTACH_VARIANCES",
    "STAGE_COMBINE_RUNS",
    "STAGE_EXPORT",
    "STAGE_GAMMA_FILTER",
    "STAGE_LOAD_DARK",
    "STAGE_LOAD_OB",
    "STAGE_LOAD_SAMPLE",
    "STAGE_NORMALIZE",
    "STAGE_REBIN_TOF",
    "STAGE_STACK_FRAMES",
    "NULL_REPORTER",
    "Progress",
    "ProgressCallback",
    "ProgressLike",
    "ProgressEvent",
    "ProgressReporter",
    "resolve_progress",
]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One progress notification.

    Attributes
    ----------
    stage : str
        Which part of the run this came from, e.g. ``"load_sample"``. One of the module-level
        ``STAGE_*`` constants.
    completed : int
        Items finished in this stage so far, counted across the whole run rather than restarting
        per input file or run. This is an **absolute** count, so a tqdm adapter is
        ``bar.update(event.completed - bar.n)``.
    total : Optional[int]
        Items this stage will process, or ``None`` when that is not knowable in advance.
    detail : str
        Optional human-readable context, e.g. the file being read. May be empty.
    """

    stage: str
    completed: int
    total: Optional[int]
    detail: str = ""


ProgressCallback = Callable[[ProgressEvent], None]

#: What a CALLER passes for ``progress``: ``False``, ``True``, or a callable.
Progress = Union[bool, ProgressCallback]


class ProgressReporter:
    """A progress emitter with its stage, offset and total already bound.

    Internal plumbing, not something callers construct. A caller passes ``progress`` as a bool or
    a callable; NeuNorm turns that into reporters with :func:`resolve_progress` and hands those to
    the functions doing the work. Binding the offset here is what lets a leaf function — called
    once per input run — contribute to a single count that spans the whole run without needing to
    know anything about its callers.

    Resolve **once** per run and derive every other reporter from that one with :meth:`for_stage`
    and :meth:`with_offset`: derived reporters share the original's sink, whereas calling
    :func:`resolve_progress` again with ``progress=True`` would build a second, independent set of
    progress bars.
    """

    __slots__ = ("_completed", "_offset", "_owns_sink", "_sink", "_stage", "_total")

    def __init__(
        self,
        sink: ProgressCallback,
        stage: str = "",
        offset: int = 0,
        total: Optional[int] = None,
        owns_sink: bool = False,
    ) -> None:
        self._sink = sink
        self._stage = stage
        self._offset = int(offset)
        self._total = None if total is None else int(total)
        self._completed = 0
        # Only a sink NeuNorm built may be closed by close(). A caller's callback might be a
        # reusable object that happens to have a close() method, and closing it would release
        # something NeuNorm does not own.
        self._owns_sink = bool(owns_sink)

    @property
    def stage(self) -> str:
        """The stage label this reporter emits under."""
        return self._stage

    @property
    def total(self) -> Optional[int]:
        """The item total this reporter reports, or ``None`` if indeterminate."""
        return self._total

    @property
    def completed(self) -> int:
        """The absolute count last emitted (offset plus items advanced here)."""
        return self._offset + self._completed

    def __call__(self, advance: int = 1, detail: str = "") -> None:
        """Advance by ``advance`` items and emit one event.

        Stages with no item axis report named steps instead: build the reporter with
        ``total=None`` and call this once per step with a ``detail``, so something still moves.

        Raises
        ------
        TypeError
            If ``advance`` is not an integer. A float would be truncated, so ``advance=0.9`` would
            silently emit an event that advanced nothing.
        ValueError
            If ``advance`` is not positive. A count that can move backwards is not a progress count.
        """
        _check_advance(advance)
        self._completed += int(advance)
        self._sink(
            ProgressEvent(
                stage=self._stage,
                completed=self._offset + self._completed,
                total=self._total,
                detail=detail,
            )
        )

    def note(self, detail: str) -> None:
        """Emit an event at the current count, advancing nothing.

        For announcing work that has no item of its own — a large allocation about to happen, say —
        where the point is to name what is starting rather than to count something finished. The
        count stays absolute and monotonic, so a note cannot corrupt a bar or double-count; it
        changes the label and leaves the position alone.

        Prefer this over ``for_stage(..., total=None)`` for a step that recurs on every call of an
        inner function: a fresh stage reporter restarts at 1 each time, which contradicts
        :attr:`ProgressEvent.completed` being run-wide and leaves a bar stuck at its first tick.
        """
        self._sink(
            ProgressEvent(
                stage=self._stage,
                completed=self._offset + self._completed,
                total=self._total,
                detail=detail,
            )
        )

    def for_stage(self, stage: str, total: Optional[int] = None) -> "ProgressReporter":
        """A fresh reporter on the same sink for a different stage.

        The new reporter starts its count at zero, so this is for moving between the phases of one
        run — not for a step inside a function called repeatedly, which would restart the count on
        every call. Use :meth:`note` for that.
        """
        return ProgressReporter(self._sink, stage, offset=0, total=total, owns_sink=self._owns_sink)

    def with_offset(self, offset: int) -> "ProgressReporter":
        """A reporter for the same stage and total that starts counting from ``offset``."""
        return ProgressReporter(self._sink, self._stage, offset=offset, total=self._total, owns_sink=self._owns_sink)

    def close(self) -> None:
        """Release the progress bars NeuNorm opened, if any.

        Only does anything for ``progress=True``, where NeuNorm owns tqdm bars: a stage with an
        indeterminate total never reaches a completion point, and a stage abandoned by an early
        error never finishes, so neither closes on its own. Call this from a ``finally`` around the
        run.

        A caller-supplied callback is **never** closed, even if it happens to have a ``close``
        method — it may be a reusable object whose lifetime belongs to the caller.
        """
        if self._owns_sink:
            self._sink.close()


#: What an instrumented function ACCEPTS for ``progress``: anything a caller may pass, plus a
#: pre-bound :class:`ProgressReporter`, which is what an outer function threads to an inner one.
ProgressLike = Union[Progress, ProgressReporter]


def _null_sink(event: ProgressEvent) -> None:  # noqa: ARG001 - signature must match the sink type
    """Discard an event."""


class _NullReporter(ProgressReporter):
    """The ``progress=False`` reporter: every operation is a no-op that allocates nothing.

    ``for_stage`` and ``with_offset`` return ``self`` rather than a new object, so threading
    progress through a pipeline costs nothing at all when the caller did not ask for it.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(_null_sink)

    def __call__(self, advance: int = 1, detail: str = "") -> None:  # noqa: ARG002 - detail unused by design
        """Validate the advance, then do nothing.

        The validation is not skipped: it is what stops a bad emit site from passing unnoticed
        through a suite that runs almost entirely with progress disabled. No event is built and no
        counter is touched, so the per-item cost stays two comparisons.
        """
        _check_advance(advance)

    def note(self, detail: str) -> None:  # noqa: ARG002 - no-op by design
        """Do nothing."""

    def for_stage(self, stage: str, total: Optional[int] = None) -> "ProgressReporter":  # noqa: ARG002 - no-op
        """Return this same no-op reporter."""
        return self

    def with_offset(self, offset: int) -> "ProgressReporter":  # noqa: ARG002 - no-op by design
        """Return this same no-op reporter."""
        return self

    def close(self) -> None:
        """Nothing is held open."""


#: Shared no-op reporter used whenever ``progress`` is ``False``.
NULL_REPORTER = _NullReporter()


class _TqdmSink:
    """Sink for ``progress=True``: NeuNorm owns one :mod:`tqdm` bar per stage.

    ``tqdm.auto`` is imported on first use, not at module import, so the ordinary
    ``progress=False`` path never pays for it (importing ``tqdm.auto`` probes for ipywidgets).
    """

    def __init__(self) -> None:
        self._bars: dict = {}

    def __call__(self, event: ProgressEvent) -> None:
        from tqdm.auto import tqdm

        bar = self._bars.get(event.stage)
        if bar is None:
            bar = self._bars[event.stage] = tqdm(
                total=event.total,
                desc=event.stage,
                unit="item" if event.total is not None else "step",
                leave=False,
            )
        if bar.total is None and event.total is not None:
            bar.total = event.total
        advanced = event.completed > bar.n
        if advanced:
            bar.update(event.completed - bar.n)
        if event.detail:
            # A note advances nothing, so nothing would redraw it; refresh explicitly in that case.
            bar.set_postfix_str(event.detail, refresh=not advanced)

        # A bar is deliberately NOT closed when it reaches its total: :meth:`close` is the only thing
        # that retires one. Closing here looked tidier but rendered badly — a stage that reports
        # anything after its last item (a note about the allocations that follow a read loop, say)
        # found its bar gone, built a fresh one at 0%, and closed that too. Watching a real 1000-file
        # load showed the count run 0% -> 31% -> 70% and then jump back to 0% twice, with the notes
        # erased before they could be read, because `leave=False` clears each closed line.

    def close(self) -> None:
        """Close and forget every bar.

        The only thing that retires a bar, including one that has reached its total, so a caller
        must invoke it in a ``finally``. A stage with an indeterminate total never reaches a
        completion point and an abandoned stage never finishes, so without this they would stay open
        until garbage collection.
        """
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()


def resolve_progress(
    progress: ProgressLike,
    stage: str = "",
    total: Optional[int] = None,
) -> ProgressReporter:
    """Turn a caller's ``progress`` argument into a :class:`ProgressReporter`.

    Idempotent: an existing reporter is returned **unchanged**, so a function can accept either a
    user's ``progress`` value or a reporter its caller already built and treat both the same way.

    ``stage`` and ``total`` bind only when building a reporter from a bool or a callable. They are
    deliberately ignored for an existing reporter, because that reporter already carries the offset
    and the grand total its caller bound — re-binding them here would reset the offset to zero and
    replace the run-wide total with a local one, so a count spanning several input runs would
    restart per run and report the wrong denominator. A caller that genuinely wants a different
    stage must ask for it explicitly with :meth:`ProgressReporter.for_stage`.

    Parameters
    ----------
    progress : bool, callable, or ProgressReporter
        ``False`` for no reporting, ``True`` to let NeuNorm drive tqdm, a callable to receive
        :class:`ProgressEvent` objects, or an existing reporter, which is returned as-is.
    stage : str
        Stage label to bind. Ignored when ``progress`` is ``False`` or already a reporter.
    total : Optional[int]
        Item total to bind, or ``None`` when indeterminate. Ignored when ``progress`` is ``False``
        or already a reporter.

    Returns
    -------
    ProgressReporter
        A reporter ready to call.

    Raises
    ------
    TypeError
        If ``progress`` is neither a bool, nor callable, nor a reporter.
    """
    if isinstance(progress, ProgressReporter):
        return progress
    if progress is False:
        return NULL_REPORTER
    if progress is True:
        return ProgressReporter(_TqdmSink(), stage, total=total, owns_sink=True)
    if callable(progress):
        return ProgressReporter(progress, stage, total=total)
    raise TypeError(
        "progress must be False (no reporting), True (NeuNorm drives a tqdm bar), or a callable "
        f"accepting a ProgressEvent; got {type(progress).__name__}"
    )
