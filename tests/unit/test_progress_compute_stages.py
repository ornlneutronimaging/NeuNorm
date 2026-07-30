"""Progress reporting through the two dominant compute stages (#195, Task 5).

Neither `apply_gamma_filter` nor `normalize_transmission` has an item axis — both are sequences of
whole-array operations — so both report named steps rather than an item count. They are the two
places a run goes quiet for the longest: the gamma filter is enabled by default on the CCD and MARS
TPX3 pipelines and is the slowest per frame there (its internal `median_filter` is most of it), and
the normalizer dominates the TOF paths.

The step total is computed, not literal. Both functions have optional work, so a fixed total would
leave the bar short or overshooting depending on the arguments.
"""

import contextlib
import io
import re

import numpy as np
import pytest
import scipp as sc

from neunorm.data_models.roi import ROI
from neunorm.filters.gamma_filter import apply_gamma_filter
from neunorm.processing.normalizer import normalize_transmission
from neunorm.utils.progress import (
    STAGE_GAMMA_FILTER,
    STAGE_NORMALIZE,
    ProgressReporter,
    _TqdmSink,
)


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


def _stack(value, shape=(4, 32, 32), spike=False):
    values = np.full(shape, float(value))
    if spike:
        values[0, 5, 5] = 1e6
    return sc.DataArray(sc.array(dims=["t", "y", "x"], values=values, variances=values.copy(), unit="counts"))


def _render(call):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        call()
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# gamma filter
# --------------------------------------------------------------------------------------


def test_gamma_filter_names_each_step_then_counts_it():
    """Four named steps, each counted only after its work returns."""
    events, sink = _collect()

    apply_gamma_filter(_stack(20, spike=True), progress=sink)

    assert [(e.completed, bool(e.detail)) for e in events][:8] == [
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
    assert {e.stage for e in events} == {STAGE_GAMMA_FILTER}


def test_gamma_filter_names_the_median_filter_separately():
    """`median_filter` is most of the stage's cost, so a bar parked there must say so — otherwise the
    user cannot tell a slow filter from a hung one."""
    events, sink = _collect()

    apply_gamma_filter(_stack(20, spike=True), progress=sink)

    assert "local median" in [e.detail for e in events]


def test_gamma_filter_announces_the_optional_variance_recompute():
    """The per-outlier variance recomputation is announced, not counted: whether it runs depends on
    the outlier count, which is unknown when the total is fixed."""
    events, sink = _collect()

    apply_gamma_filter(_stack(20, spike=True), preserve_variance=False, progress=sink)

    notes = [e.detail for e in events if e.detail]
    assert any("recomputing variance" in d for d in notes), notes
    # announced without advancing past the declared total
    assert max(e.completed for e in events) == 4


def test_gamma_filter_output_is_pinned_to_values_not_to_itself():
    """The filtered result is pinned to computed values, not to a second call of the same function.

    The earlier version of this test compared `apply_gamma_filter(data)` against
    `apply_gamma_filter(data, progress=...)`. Both calls take one code path — `progress` only selects
    the sink — so it could not detect the re-indentation defect it read as guarding, and any error
    shared by both invocations was invisible. That is the seventh self-referential test in this
    branch; the review closed the gap independently with an AST-equivalence check and a 30-config
    differential run, and this pins the behaviour so the suite does too.
    """
    data = _stack(20, spike=True)
    out = apply_gamma_filter(data)

    # the spike is replaced by the local median of an otherwise-uniform field, i.e. the field value
    assert out.values[0, 5, 5] == pytest.approx(20.0)
    # every other pixel is untouched
    untouched = out.values.copy()
    untouched[0, 5, 5] = 20.0
    assert np.allclose(untouched, 20.0)
    # reporting changes nothing about that
    assert sc.identical(out, apply_gamma_filter(data, progress=lambda _e: None))


def test_gamma_filter_cancellation_propagates():
    """Raising from the callback aborts the filter."""

    class _CancelledError(RuntimeError):
        pass

    def cancel(event):
        if event.completed >= 2:
            raise _CancelledError("stop")

    with pytest.raises(_CancelledError):
        apply_gamma_filter(_stack(20, spike=True), progress=cancel)


def test_gamma_filter_releases_its_own_bar_when_cancelled(recorded_sinks):
    """`progress=True` makes the filter own the sink, so it must release it even when abandoned."""

    class _CancelledError(RuntimeError):
        pass

    original = _RecordingTqdmSink.__call__

    def exploding(self, event):
        original(self, event)
        if event.completed >= 2:
            raise _CancelledError("stop")

    _RecordingTqdmSink.__call__ = exploding
    try:
        with pytest.raises(_CancelledError):
            apply_gamma_filter(_stack(20, spike=True), progress=True)
    finally:
        _RecordingTqdmSink.__call__ = original

    assert recorded_sinks, "no internal sink was created"
    assert recorded_sinks[0]._bars == {}, "the filter left its own bar open after being cancelled"


# --------------------------------------------------------------------------------------
# normalizer — the step total varies with the arguments
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "kwargs", "expected_total", "expected_names"),
    [
        ("no correction", {}, 1, ["dividing sample by open beam"]),
        (
            "proton charge",
            {
                "proton_charge_sample": sc.scalar(500.0, unit="C"),
                "proton_charge_ob": sc.scalar(505.0, unit="C"),
            },
            3,
            [
                "proton-charge correction, sample",
                "proton-charge correction, open beam",
                "dividing sample by open beam",
            ],
        ),
        (
            "background roi",
            {"background_roi": ROI(x0=0, y0=0, x1=8, y1=8)},
            2,
            ["background-ROI flux normalization", "dividing sample by open beam"],
        ),
    ],
)
def test_normalizer_total_matches_the_work_it_actually_does(label, kwargs, expected_total, expected_names):
    """The declared total must equal the steps that run, on every branch.

    The background-ROI and proton-charge corrections are mutually exclusive and either may be absent,
    so a literal total would leave the bar short (no correction) or overshooting (both). The count is
    derived from the arguments instead.
    """
    events, sink = _collect()

    normalize_transmission(_stack(50), _stack(100), progress=sink, **kwargs)

    assert {e.total for e in events} == {expected_total}, label
    assert max(e.completed for e in events) == expected_total, f"{label}: bar did not reach its total"
    named = [e.detail for e in events if e.detail]
    for expected in expected_names:
        assert expected in named, f"{label}: missing step {expected!r} in {named}"
    assert {e.stage for e in events} == {STAGE_NORMALIZE}


def test_normalizer_announces_the_background_roi_variance_term():
    """That term runs only when the result carries variances, which is not knowable when the total is
    computed, so it is announced rather than counted."""
    events, sink = _collect()

    normalize_transmission(_stack(50), _stack(100), background_roi=ROI(x0=0, y0=0, x1=8, y1=8), progress=sink)

    assert "background-ROI variance contribution" in [e.detail for e in events]
    assert max(e.completed for e in events) == 2, "the announcement must not advance past the total"


def test_normalizer_output_is_pinned_to_values_not_to_itself():
    """The transmission is pinned to computed values, not to a second call of the same function.

    Same reasoning as the gamma-filter test above: comparing the instrumented function against itself
    cannot catch a control-flow change from moving ~110 lines into the `with` block, because both
    calls run the same code.
    """
    sample, ob = _stack(50), _stack(100)
    out = normalize_transmission(sample, ob)

    # T = 50 / 100
    assert np.allclose(out.values, 0.5)
    # Var(T) = T^2 * (Var(S)/S^2 + Var(O)/O^2) = 0.25 * (50/2500 + 100/10000)
    assert np.allclose(out.variances, 0.25 * (50 / 2500 + 100 / 10000))
    assert out.unit == "dimensionless"
    # reporting changes nothing about that
    assert sc.identical(out, normalize_transmission(sample, ob, progress=lambda _e: None))


def test_normalizer_cancellation_propagates():
    """Raising from the callback aborts normalization."""

    class _CancelledError(RuntimeError):
        pass

    with pytest.raises(_CancelledError):
        normalize_transmission(
            _stack(50),
            _stack(100),
            progress=lambda _e: (_ for _ in ()).throw(_CancelledError("stop")),
        )


def test_normalizer_does_not_close_a_caller_supplied_sink():
    """The pipeline case: the normalizer is one stage of a run, so it must not retire the caller's
    bars on the way out."""
    tqdm_sink = _TqdmSink()
    caller = ProgressReporter(tqdm_sink, STAGE_NORMALIZE, total=3, owns_sink=True)

    normalize_transmission(_stack(50), _stack(100), progress=caller)

    assert tqdm_sink._bars, "the caller's bar was closed by the normalizer"
    caller.close()
    assert tqdm_sink._bars == {}


# --------------------------------------------------------------------------------------
# what the bars RENDER — the check that caught three defects in Tasks 3 and 4
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "call"),
    [
        ("gamma", lambda: apply_gamma_filter(_stack(20, spike=True), progress=True)),
        ("normalize", lambda: normalize_transmission(_stack(50), _stack(100), progress=True)),
    ],
)
def test_rendered_bar_completes_and_never_goes_backwards(label, call):
    """Both stages must draw a bar that reaches its completion and never regresses."""
    out = _render(call)

    assert "100%" in out, f"{label}: bar never rendered its completion:\n{out}"
    percents = [int(m) for m in re.findall(r"(\d+)%\|", out)]
    assert percents == sorted(percents), f"{label}: rendered progress went backwards: {percents}"


def test_rendered_gamma_bar_shows_its_step_names():
    """The step names are the diagnostic value here — a bar with no label cannot tell a user which
    whole-array operation is slow."""
    out = _render(lambda: apply_gamma_filter(_stack(20, shape=(8, 128, 128), spike=True), progress=True))

    assert "local median" in out, f"the dominant step never rendered:\n{out}"
