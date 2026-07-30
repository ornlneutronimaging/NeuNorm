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

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

import numpy as np
from loguru import logger

#: Handler attributes that together describe everything a caller can configure through
#: ``logger.add()`` and that we would fail to reproduce. Compared as a whole against a handler
#: loguru builds with no arguments, so this cannot drift from the installed loguru version.
_HANDLER_SHAPE_ATTRS = (
    "_levelno",
    "_filter",
    "_serialize",
    "_enqueue",
    "_is_formatter_dynamic",
    "_decolorized_format",
    "_colorize",
)


def _handler_shape(handler: object) -> tuple:
    """The configuration fingerprint of a loguru handler."""
    return tuple(getattr(handler, name, NotImplemented) for name in _HANDLER_SHAPE_ATTRS)


def _default_stderr_shape() -> tuple:
    """Fingerprint and colour setting of the handler loguru builds for ``sys.stderr`` by default.

    Measured by adding one and reading it back, so "default" always means what this loguru version
    actually does. The probe is removed immediately and writes nothing.
    """
    probe_id = logger.add(sys.stderr)
    try:
        handler = logger._core.handlers[probe_id]
        return _handler_shape(handler), getattr(handler, "_colorize", False)
    finally:
        logger.remove(probe_id)


def _is_default_stderr_handler(handler: object, default_shape: tuple) -> bool:
    """True when ``handler`` writes to ``sys.stderr`` and is configured exactly as loguru's default.

    The stream must be a real stream: with ``sys.stderr`` set to ``None`` an absent ``_stream``
    would compare equal and match handlers that are not streams at all.
    """
    stream = getattr(getattr(handler, "_sink", None), "_stream", None)
    return stream is not None and stream is sys.stderr and _handler_shape(handler) == default_shape


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
    "ProgressEvent",
    "ProgressReporter",
    "resolve_progress",
    "tqdm_safe_logging",
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

#: What an instrumented function accepts for ``progress``: ``False``, ``True``, or a callable.
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

    def for_stage(self, stage: str, total: Optional[int] = None) -> "ProgressReporter":
        """A fresh reporter on the same sink for a different stage."""
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
        if event.completed > bar.n:
            bar.update(event.completed - bar.n)
        if event.detail:
            bar.set_postfix_str(event.detail, refresh=False)
        if event.total is not None and event.completed >= event.total:
            bar.close()
            del self._bars[event.stage]

    def close(self) -> None:
        """Close and forget every bar still open.

        A stage with an indeterminate total never reaches a completion point, and a stage abandoned
        by an early error never finishes, so without this those bars stay open until garbage
        collection.
        """
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()


def resolve_progress(
    progress: Union[Progress, ProgressReporter],
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


@contextmanager
def tqdm_safe_logging() -> Iterator[None]:
    """Route loguru's stderr output through ``tqdm.write`` so it does not shred a progress bar.

    NeuNorm's log records and a tqdm bar both go to stderr, so log lines land on the bar's
    un-terminated line and corrupt it. Inside this context, handlers writing to ``sys.stderr`` are
    replaced by one that hands each record to ``tqdm.write``, which clears the bar, writes the
    line, and redraws.

    Only used for ``progress=True``, where NeuNorm owns the bar. When a caller supplies their own
    callback NeuNorm leaves logging alone.

    **Only a handler indistinguishable from loguru's own default is displaced.** Anything a host
    application configured is left in place, even though it will corrupt the bar, because loguru's
    public ``add()`` cannot faithfully recreate it: a handler's format lives in an opaque
    ``ColoredFormat``, so re-adding it would substitute loguru's defaults and permanently discard
    the host's format, filter, serialization and colour policy. A shredded bar is cosmetic;
    silently rewriting an application's logging is not.

    What counts as "loguru's default" is measured from loguru itself — a throwaway handler is added
    with no arguments and its shape read back — rather than from a hand-written list of attributes.
    An earlier version of this compared level, filter, serialization, queueing and
    formatter-dynamism but not the format *content*, so a handler added as
    ``logger.add(sys.stderr, format="...")`` matched on every checked attribute and had its format
    silently replaced.

    Notes
    -----
    - Does nothing when there is no default-shaped stderr handler — including when a host
      application has taken loguru over, which is exactly when it must not interfere.
    - Does nothing when ``sys.stderr`` is ``None`` (pythonw, or fd 2 closed). Without that guard the
      stream comparison degenerates to ``None is None`` and matches handlers that are not streams
      at all, such as a host's file sink.
    - Detection uses ``logger._core.handlers`` and private handler attributes; loguru exposes no
      public API for either. Handler ids change across the context, so nothing may depend on them.
    - Every mutation is inside the ``try``, and restoration runs before the temporary sink is
      removed and tolerates that sink having already vanished. A failure anywhere therefore cannot
      leave the session without a stderr handler, silently swallowing later records.
    - Restoration adds back only the shortfall. Code inside the context that reconfigures loguru
      with the usual ``logger.remove()`` then ``logger.add(sys.stderr)`` would otherwise end up
      with two default stderr handlers, emitting every subsequent record twice.
    - ``contextlib.redirect_stderr`` does **not** capture loguru output, so it is not an
      alternative to this.
    """
    from tqdm.auto import tqdm

    if sys.stderr is None:
        yield
        return

    core = getattr(logger, "_core", None)
    if core is None:
        yield
        return

    default_shape, default_colorize = _default_stderr_shape()
    displaced = [
        handler_id
        for handler_id, handler in dict(core.handlers).items()
        if _is_default_stderr_handler(handler, default_shape)
    ]

    if not displaced:
        yield
        return

    def _default_stderr_count() -> int:
        return sum(1 for handler in dict(core.handlers).values() if _is_default_stderr_handler(handler, default_shape))

    sink_id = None
    try:
        for handler_id in displaced:
            logger.remove(handler_id)
        sink_id = logger.add(
            lambda message: tqdm.write(message, end="", file=sys.stderr),
            colorize=default_colorize,
        )
        yield
    finally:
        # Restore before removing the temporary sink, and only the shortfall, so neither a failure
        # here nor a reconfiguration inside the context can lose or duplicate a stderr handler.
        for _ in range(max(0, len(displaced) - _default_stderr_count())):
            logger.add(sys.stderr)
        if sink_id is not None:
            try:
                logger.remove(sink_id)
            except ValueError:
                # Something inside the context reconfigured loguru and took the sink with it.
                pass
