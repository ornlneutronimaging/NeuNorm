"""Progress reporting through the TIFF/FITS load path (#195, Task 3).

The load loop is the unit a user counts — "1000 files" — and it is the only place where a slow or
contended filesystem becomes visible per item. The two ticks after the loop matter for a different
reason: the loop only appends to a list, so the memory peak (a measured ~5x multiple of the stack,
from `np.stack` plus the variances copy) lands *after* the last file. Without them a bar reaches
100% and then goes silent through the part that can exhaust RAM.
"""

from pathlib import Path

import numpy as np
import pytest

from neunorm.loaders.fits_loader import load_fits_stack
from neunorm.loaders.stack_loader import load_stack
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.utils.progress import (
    STAGE_ATTACH_VARIANCES,
    STAGE_LOAD_SAMPLE,
    STAGE_STACK_FRAMES,
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
    """One event per file, counting up to the number of files, each naming the file it finished."""
    paths = paths_fn()
    assert len(paths) == 3, "fixture changed; the assertions below assume 3 files"
    events, sink = _collect()

    loader(paths, progress=sink)

    per_file = [e for e in events if e.stage == STAGE_LOAD_SAMPLE]
    assert [e.completed for e in per_file] == [1, 2, 3]
    assert {e.total for e in per_file} == {3}
    assert [e.detail for e in per_file] == [p.name for p in paths]


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_loader_reports_the_post_loop_allocations(loader, paths_fn):
    """The stack build and the variances copy each get a labelled tick, after the last file.

    These are where the memory actually peaks, so they must come after the per-file events rather
    than be folded into them.
    """
    events, sink = _collect()

    loader(paths_fn(), progress=sink)

    stages = [e.stage for e in events]
    assert stages.count(STAGE_STACK_FRAMES) == 1
    assert stages.count(STAGE_ATTACH_VARIANCES) == 1
    assert stages.index(STAGE_STACK_FRAMES) > max(i for i, s in enumerate(stages) if s == STAGE_LOAD_SAMPLE), (
        "the stack tick must follow every per-file event"
    )
    assert stages.index(STAGE_ATTACH_VARIANCES) > stages.index(STAGE_STACK_FRAMES)
    # Indeterminate: these are single steps, not counted items.
    assert all(e.total is None for e in events if e.stage in (STAGE_STACK_FRAMES, STAGE_ATTACH_VARIANCES))


# --------------------------------------------------------------------------------------
# load_stack pass-through — the CCD pipelines' only route to per-file progress
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("paths_fn", [_tiffs, _fits], ids=["tiff", "fits"])
def test_load_stack_passes_progress_to_whichever_leaf_it_picks(paths_fn):
    """`load_stack` dispatches on extension; both branches must forward `progress`.

    Without this the CCD pipelines report nothing per file, because they call `load_stack` rather
    than a leaf loader.
    """
    events, sink = _collect()

    load_stack(paths_fn(), progress=sink)

    assert [e.completed for e in events if e.stage == STAGE_LOAD_SAMPLE] == [1, 2, 3]


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
def test_cancelling_is_not_reported_as_a_read_failure(loader, paths_fn, message, caplog):
    """A cancelling callback must not be logged as an I/O error.

    Both loaders wrap their read in `except Exception: logger.error(...); raise`. If the tick were
    emitted inside that try, cancelling would tell the user their files failed to load. Pinning the
    tick's placement, not just its existence.
    """

    def cancel(event):
        if event.completed == 1:
            raise _CancelledError("stop")

    with caplog.at_level("ERROR"):
        with pytest.raises(_CancelledError):
            loader(paths_fn(), progress=cancel)

    assert message not in caplog.text


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

    np.testing.assert_array_equal(without.values, with_progress.values)
    np.testing.assert_array_equal(without.variances, with_progress.variances)
    assert without.dims == with_progress.dims
    assert set(without.coords) == set(with_progress.coords)


@pytest.mark.parametrize(
    ("loader", "paths_fn"),
    [(load_tiff_stack, _tiffs), (load_fits_stack, _fits)],
    ids=["tiff", "fits"],
)
def test_bad_progress_value_is_rejected_at_the_loader(loader, paths_fn):
    """An invalid `progress` must fail loudly rather than silently disable reporting."""
    with pytest.raises(TypeError, match="progress must be False"):
        loader(paths_fn(), progress=1)
