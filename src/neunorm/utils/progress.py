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

from loguru import logger

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

    __slots__ = ("_completed", "_offset", "_sink", "_stage", "_total")

    def __init__(
        self,
        sink: ProgressCallback,
        stage: str = "",
        offset: int = 0,
        total: Optional[int] = None,
    ) -> None:
        self._sink = sink
        self._stage = stage
        self._offset = int(offset)
        self._total = None if total is None else int(total)
        self._completed = 0

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
        """
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
        return ProgressReporter(self._sink, stage, offset=0, total=total)

    def with_offset(self, offset: int) -> "ProgressReporter":
        """A reporter for the same stage and total that starts counting from ``offset``."""
        return ProgressReporter(self._sink, self._stage, offset=offset, total=self._total)


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

    def __call__(self, advance: int = 1, detail: str = "") -> None:  # noqa: ARG002 - no-op by design
        """Do nothing."""

    def for_stage(self, stage: str, total: Optional[int] = None) -> "ProgressReporter":  # noqa: ARG002 - no-op
        """Return this same no-op reporter."""
        return self

    def with_offset(self, offset: int) -> "ProgressReporter":  # noqa: ARG002 - no-op by design
        """Return this same no-op reporter."""
        return self


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


def resolve_progress(
    progress: Union[Progress, ProgressReporter],
    stage: str = "",
    total: Optional[int] = None,
) -> ProgressReporter:
    """Turn a caller's ``progress`` argument into a :class:`ProgressReporter`.

    Idempotent: passing a reporter back in returns it (re-staged if ``stage`` is given), so a
    function can accept either a user's ``progress`` value or a reporter its caller already built
    and treat both the same way.

    Parameters
    ----------
    progress : bool, callable, or ProgressReporter
        ``False`` for no reporting, ``True`` to let NeuNorm drive tqdm, a callable to receive
        :class:`ProgressEvent` objects, or an existing reporter to reuse.
    stage : str
        Stage label to bind. Ignored when ``progress`` is ``False``.
    total : Optional[int]
        Item total to bind, or ``None`` when indeterminate. Applied together with ``stage``, so an
        already-built reporter passed in without a ``stage`` keeps its own total.

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
        return progress.for_stage(stage, total) if stage else progress
    if progress is False:
        return NULL_REPORTER
    if progress is True:
        return ProgressReporter(_TqdmSink(), stage, total=total)
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

    Notes
    -----
    - If no handler is writing to ``sys.stderr`` — a host application has taken over loguru
      configuration — this does nothing rather than fight for the stream.
    - loguru exposes no public API for enumerating handlers or reading a handler's format, so
      discovery uses ``logger._core.handlers`` and restoration re-adds ``sys.stderr`` with the
      original level only. Handler ids change across the context; nothing may depend on them.
    - ``contextlib.redirect_stderr`` does **not** capture loguru output, so it is not an
      alternative to this.
    """
    from tqdm.auto import tqdm

    core = getattr(logger, "_core", None)
    handlers = dict(getattr(core, "handlers", {})) if core is not None else {}
    stderr_levels = {
        handler_id: getattr(handler, "_levelno", 0)
        for handler_id, handler in handlers.items()
        if getattr(getattr(handler, "_sink", None), "_stream", None) is sys.stderr
    }

    if not stderr_levels:
        yield
        return

    for handler_id in stderr_levels:
        logger.remove(handler_id)
    # Keep the most permissive of the levels we displaced so nothing is dropped meanwhile.
    sink_id = logger.add(
        lambda message: tqdm.write(message, end="", file=sys.stderr),
        level=min(stderr_levels.values()),
        colorize=True,
    )
    try:
        yield
    finally:
        logger.remove(sink_id)
        for level in stderr_levels.values():
            logger.add(sys.stderr, level=level)
