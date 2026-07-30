"""Progress reporting through export, and through the dark-corrected normalizer (#195, Task 6).

HDF5 is the primary output format and `write_hdf5` has no item axis at all — the bulk data leaves in
one or two whole-array `create_dataset` calls — so it reports named steps and can never be per-image.
`write_tiff_stack`'s `one_file_per_image` mode is the one export path with a determinate item count.

`normalize_with_dark` is here rather than with Task 5 because it was the gap that task's review found:
it is `normalize_transmission`'s sibling on the CCD path and was reporting nothing. It is also the
first place two instrumented functions compose, so it is where the borrowed-reporter and shared-counter
behaviour is exercised end to end.
"""

import contextlib
import filecmp
import io
import re

import numpy as np
import pytest
import scipp as sc

from neunorm.data_models.roi import ROI
from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.exporters.tiff_writer import write_tiff_stack
from neunorm.processing.normalizer import normalize_with_dark
from neunorm.utils.progress import STAGE_EXPORT, STAGE_NORMALIZE, ProgressReporter, _TqdmSink


class _RecordingTqdmSink(_TqdmSink):
    """Records every sink instance, so one a callee built internally is observable.

    Also records which stages ever opened a bar. Without that, "no bar is left open" passes
    trivially on a run that never opened one — the vacuous shape this branch has already produced
    seven times.
    """

    instances: list["_RecordingTqdmSink"] = []

    def __init__(self):
        super().__init__()
        self.ever_opened: list[str] = []
        _RecordingTqdmSink.instances.append(self)

    def __call__(self, event):
        super().__call__(event)
        if event.stage not in self.ever_opened:
            self.ever_opened.append(event.stage)


@pytest.fixture
def recorded_sinks(monkeypatch):
    _RecordingTqdmSink.instances = []
    monkeypatch.setattr("neunorm.utils.progress._TqdmSink", _RecordingTqdmSink)
    return _RecordingTqdmSink.instances


def _collect():
    events = []
    return events, events.append


def _transmission(shape=(3, 8, 8), variances=True):
    values = np.full(shape, 0.5)
    return sc.DataArray(
        sc.array(dims=["t", "y", "x"], values=values, variances=values.copy() if variances else None, unit=""),
        coords={
            "y": sc.arange("y", shape[1]),
            "x": sc.arange("x", shape[2]),
            "t": sc.arange("t", shape[0], unit="us"),
        },
    )


def _render(call):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        call()
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# write_hdf5 — named steps, computed total
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "variances", "metadata", "expected_total", "expected_names"),
    [
        (
            "variances + metadata",
            True,
            {"a": 1},
            4,
            ["writing transmission", "writing uncertainty", "writing metadata"],
        ),
        ("variances only", True, None, 3, ["writing transmission", "writing uncertainty"]),
        ("metadata only", False, {"a": 1}, 3, ["writing transmission", "writing metadata"]),
        ("neither", False, None, 2, ["writing transmission", "writing coordinates and masks"]),
    ],
)
def test_hdf5_total_matches_the_work_it_actually_does(
    tmp_path, label, variances, metadata, expected_total, expected_names
):
    """The uncertainty write and the metadata section are conditional, so the total is computed."""
    events, sink = _collect()

    write_hdf5(tmp_path / "out.h5", _transmission(variances=variances), metadata=metadata, progress=sink)

    assert {e.total for e in events} == {expected_total}, label
    assert max(e.completed for e in events) == expected_total, f"{label}: bar did not reach its total"
    named = [e.detail for e in events if e.detail]
    for expected in expected_names:
        assert expected in named, f"{label}: missing {expected!r} in {named}"
    assert {e.stage for e in events} == {STAGE_EXPORT}


def test_hdf5_names_each_step_before_it_runs(tmp_path):
    """Name before, count after — so the label on screen is the write currently running, not the
    one that just finished. Pinned as an exact sequence: the two bulk dataset writes are named
    separately, each name arrives at the count the step starts from, and the tick follows it."""
    events, sink = _collect()

    write_hdf5(tmp_path / "out.h5", _transmission(), progress=sink)

    assert [(e.detail, e.completed) for e in events] == [
        ("writing transmission", 0),
        ("", 1),
        ("writing uncertainty", 1),
        ("", 2),
        ("writing coordinates and masks", 2),
        ("", 3),
    ]


def test_hdf5_output_is_byte_identical_with_and_without_reporting(tmp_path):
    """Reporting must not change a single byte of the primary output format."""
    data = _transmission()
    without = tmp_path / "without.h5"
    with_progress = tmp_path / "with.h5"

    write_hdf5(without, data, metadata={"run": 1234})
    write_hdf5(with_progress, data, metadata={"run": 1234}, progress=lambda _e: None)

    assert filecmp.cmp(without, with_progress, shallow=False), "HDF5 bytes differ when reporting is on"


def test_hdf5_cancellation_is_not_swallowed_by_the_best_effort_handlers(tmp_path):
    """A cancelling callback must propagate.

    `write_hdf5` has five `except Exception` handlers with no re-raise — deliberate, so one bad
    metadata key cannot abort the bulk write. A tick placed inside any of them would turn a user's
    abort into a silently skipped metadata key, so every emit sits outside them.
    """

    class _CancelledError(RuntimeError):
        pass

    def cancel(event):
        if "metadata" in event.detail:
            raise _CancelledError("stop")

    with pytest.raises(_CancelledError):
        write_hdf5(tmp_path / "cancelled.h5", _transmission(), metadata={"a": 1}, progress=cancel)


# --------------------------------------------------------------------------------------
# write_tiff_stack — both modes
# --------------------------------------------------------------------------------------


def test_tiff_per_image_mode_emits_one_event_per_file(tmp_path):
    """The one export path with a determinate item count."""
    events, sink = _collect()

    written = write_tiff_stack(tmp_path / "norm.tiff", _transmission(), one_file_per_image=True, progress=sink)

    assert [e.completed for e in events] == [1, 2, 3]
    assert {e.total for e in events} == {3}
    assert [e.detail for e in events] == [p.name for p in written]


def test_tiff_stack_mode_is_not_silent(tmp_path):
    """The default single-file path reports its one write, so a large multi-page export is not silent."""
    events, sink = _collect()

    write_tiff_stack(tmp_path / "norm.tiff", _transmission(), progress=sink)

    assert [e.detail for e in events if e.detail] == ["writing norm.tiff"]
    assert max(e.completed for e in events) == 1
    assert {e.total for e in events} == {1}


@pytest.mark.parametrize("per_image", [False, True], ids=["stack", "per-image"])
def test_tiff_output_is_byte_identical_with_and_without_reporting(tmp_path, per_image):
    """Reporting must not change the written TIFFs."""
    data = _transmission()
    a = tmp_path / "a" / "norm.tiff"
    b = tmp_path / "b" / "norm.tiff"
    a.parent.mkdir()
    b.parent.mkdir()

    written_a = write_tiff_stack(a, data, one_file_per_image=per_image)
    written_b = write_tiff_stack(b, data, one_file_per_image=per_image, progress=lambda _e: None)

    assert [p.name for p in written_a] == [p.name for p in written_b]
    for pa, pb in zip(written_a, written_b):
        assert filecmp.cmp(pa, pb, shallow=False), f"TIFF bytes differ for {pa.name}"


def test_tiff_cancellation_propagates(tmp_path):
    """Raising from the callback aborts the export."""

    class _CancelledError(RuntimeError):
        pass

    def cancel(event):
        if event.completed >= 2:
            raise _CancelledError("stop")

    with pytest.raises(_CancelledError):
        write_tiff_stack(tmp_path / "norm.tiff", _transmission(), one_file_per_image=True, progress=cancel)


def test_tiff_releases_its_own_bar_when_a_write_fails_midway(tmp_path, recorded_sinks, monkeypatch):
    """A bar opened on image 1 must be released when image 2 fails, not left drawn on the terminal."""
    calls = {"n": 0}

    def failing_save(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("no space left on device")

    monkeypatch.setattr("neunorm.exporters.tiff_writer.save_scitiff", failing_save)

    with pytest.raises(OSError, match="no space left"):
        write_tiff_stack(tmp_path / "norm.tiff", _transmission(), one_file_per_image=True, progress=True)

    assert recorded_sinks, "no internal sink was created"
    assert recorded_sinks[0].ever_opened == [STAGE_EXPORT], "a bar was never opened, so releasing it proves nothing"
    assert recorded_sinks[0]._bars == {}, "the bar survived the failure"


# --------------------------------------------------------------------------------------
# normalize_with_dark — two instrumented functions composing
# --------------------------------------------------------------------------------------


def _stack(value, shape=(3, 8, 8)):
    values = np.full(shape, float(value))
    return sc.DataArray(sc.array(dims=["t", "y", "x"], values=values, variances=values.copy(), unit="counts"))


@pytest.mark.parametrize(
    ("label", "kwargs", "expected_total"),
    [
        ("no correction", {}, 3),
        (
            "proton charge",
            {
                "proton_charge_sample": sc.scalar(500.0, unit="C"),
                "proton_charge_ob": sc.scalar(505.0, unit="C"),
            },
            5,
        ),
        ("background roi", {"background_roi": ROI(x0=0, y0=0, x1=8, y1=8)}, 4),
    ],
)
def test_dark_normalizer_total_covers_its_own_steps_and_the_delegate_s(label, kwargs, expected_total):
    """Two dark subtractions plus whatever `normalize_transmission` reports, as ONE continuous count.

    The delegate borrows this reporter, and a borrowed reporter keeps the OUTER total, so the combined
    count must be declared here. Both are derived from one shared helper so they cannot drift apart.
    """
    events, sink = _collect()

    normalize_with_dark(_stack(60), _stack(110), _stack(10), progress=sink, **kwargs)

    counts = [e.completed for e in events]
    assert {e.total for e in events} == {expected_total}, label
    assert max(counts) == expected_total, f"{label}: reached {max(counts)} of {expected_total}"
    assert counts == sorted(counts), f"{label}: the count went backwards: {counts}"
    named = [e.detail for e in events if e.detail]
    assert named[:2] == ["dark-correcting sample", "dark-correcting open beam"]
    assert "dividing sample by open beam" in named, named
    assert {e.stage for e in events} == {STAGE_NORMALIZE}


def test_dark_normalizer_announces_the_variance_correction_that_dominates_its_cost():
    """The shared-dark variance correction is 58% of this function's wall clock at 80 x 512², and it
    once ran after the progress context closed — bar at 100%, bars gone, then more than half the call
    with nothing on screen. It is announced, not counted, because it is skipped without variances."""
    events, sink = _collect()

    normalize_with_dark(_stack(60), _stack(110), _stack(10), progress=sink)

    details = [e.detail for e in events if e.detail]
    assert details[-1] == "correcting shared-dark variance", details
    # announced at the total, without pushing past it
    last = events[-1]
    assert (last.completed, last.total) == (3, 3)


def test_dark_normalizer_does_not_announce_a_correction_it_skips():
    """Variance-free input takes the early return, so the label must not name work that never runs."""
    events, sink = _collect()

    plain = sc.DataArray(sc.array(dims=["t", "y", "x"], values=np.full((3, 8, 8), 60.0), unit="counts"))
    ob = sc.DataArray(sc.array(dims=["t", "y", "x"], values=np.full((3, 8, 8), 110.0), unit="counts"))
    dark = sc.DataArray(sc.array(dims=["t", "y", "x"], values=np.full((3, 8, 8), 10.0), unit="counts"))

    normalize_with_dark(plain, ob, dark, progress=sink)

    assert "correcting shared-dark variance" not in [e.detail for e in events]
    assert max(e.completed for e in events) == 3, "the bar must still reach its total on this path"


def test_dark_normalizer_does_not_close_a_caller_supplied_sink():
    """It is one stage of a pipeline run, so it must leave the caller's bars alone — and so must the
    delegate it hands its reporter to."""
    tqdm_sink = _TqdmSink()
    caller = ProgressReporter(tqdm_sink, STAGE_NORMALIZE, total=3, owns_sink=True)

    normalize_with_dark(_stack(60), _stack(110), _stack(10), progress=caller)

    assert tqdm_sink._bars, "the caller's bar was closed by a callee"
    caller.close()
    assert tqdm_sink._bars == {}


def test_dark_normalizer_output_is_pinned_to_values():
    """Pinned to computed values, not to a second call of the same function.

    T = (60 - 10) / (110 - 10) = 0.5. Instrumenting must not touch that.
    """
    out = normalize_with_dark(_stack(60), _stack(110), _stack(10))

    assert np.allclose(out.values, 0.5)
    assert sc.identical(out, normalize_with_dark(_stack(60), _stack(110), _stack(10), progress=lambda _e: None))


# --------------------------------------------------------------------------------------
# what the bars RENDER
# --------------------------------------------------------------------------------------


def test_rendered_export_bars_complete(tmp_path):
    """Each export path must draw a bar that reaches its completion and never regresses."""
    for label, call in (
        ("hdf5", lambda: write_hdf5(tmp_path / "r.h5", _transmission(), metadata={"a": 1}, progress=True)),
        (
            "tiff per-image",
            lambda: write_tiff_stack(tmp_path / "r.tiff", _transmission(), one_file_per_image=True, progress=True),
        ),
        ("tiff stack", lambda: write_tiff_stack(tmp_path / "s.tiff", _transmission(), progress=True)),
        ("dark normalize", lambda: normalize_with_dark(_stack(60), _stack(110), _stack(10), progress=True)),
    ):
        out = _render(call)
        assert "100%" in out, f"{label}: bar never rendered its completion:\n{out}"
        percents = [int(m) for m in re.findall(r"(\d+)%\|", out)]
        assert percents == sorted(percents), f"{label}: rendered progress went backwards: {percents}"


def test_rendered_hdf5_bar_names_its_steps(tmp_path):
    """The step names are the diagnostic value: HDF5 export cannot be per-item, so the label is all
    the user has to tell which write is running."""
    out = _render(lambda: write_hdf5(tmp_path / "r.h5", _transmission(shape=(8, 64, 64)), progress=True))

    assert "writing transmission" in out, out
