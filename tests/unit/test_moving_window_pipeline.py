"""End-to-end tests for the moving window on the three VENUS TOF pipelines.

The primitive is pinned in ``test_moving_window.py``. What is checked here is that the window is
reached by every pipeline that should have it, at the point the issue asks for — after the runs are
combined and immediately before normalization — and that the numbers coming out the far end are the
ones that arithmetic says they should be.

Expected transmissions are computed with ``scipy.ndimage.convolve`` over the synthetic frame values,
which is iBeatles' own box filter and is not the code under test. The unfiltered case is asserted
first with the same arithmetic minus the filter, so the model of the pipeline being used for the
filtered assertion is itself checked rather than assumed.
"""

import h5py
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import convolve
from test_progress_pipelines import (
    _spectra_file,
    _tpx3_event_file,
    _venus_metadata_nexus,
)

from neunorm.data_models.moving_window import MovingWindow
from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline
from neunorm.utils.progress import STAGE_MOVING_WINDOW, STAGE_NORMALIZE

_DETECTOR = 32
_FRAMES = 4
_PC_SAMPLE = 12345.0
_PC_OB = 24690.0


_YY, _XX = np.mgrid[0:_DETECTOR, 0:_DETECTOR]


def _sample_frame(index):
    """Sample counts: a base level plus a short-period ripple for the box filter to work on."""
    return 81.0 + index + 20.0 * np.sin(_XX / 1.5) + 14.0 * np.cos(_YY / 2.0)


def _ob_frame(index):
    """Open-beam counts: a base level plus a longer-period ripple, different from the sample's."""
    return 99.0 + index + 12.0 * np.sin(_YY / 3.0) + 8.0 * np.cos(_XX / 5.0)


# Both frames vary SMOOTHLY, and that is load-bearing rather than cosmetic. These tests compare the
# pipeline against a mask-blind ``scipy.ndimage.convolve`` reference, so the synthetic data must trip
# no mask. ``detect_hot_pixels`` thresholds at ``median + 5 x MAD x 1.4826``, so a sharp feature on
# an otherwise uniform frame leaves the MAD at zero, puts the threshold on the median, and flags the
# feature as hot — after which the window would correctly exclude those pixels and disagree with the
# reference. It has to hold for the SAMPLE too, not just the open beam: ``venus_tpx3_histogram``
# re-detects its masks from the sample after a spatial rebin, where the other two use the open beam.
# Verified over every combination these tests exercise — raw, cropped, 2x-rebinned and both — that
# ``detect_hot_pixels`` and ``detect_dead_pixels`` each flag zero pixels.


def _as_stored(stack):
    """What a float32 TIFF actually holds, so the reference is not compared against more digits
    than the pipeline ever received."""
    return stack.astype(np.float32).astype(np.float64)


_SAMPLE_STACK = _as_stored(np.stack([_sample_frame(i) for i in range(_FRAMES)]))
_OB_STACK = _as_stored(np.stack([_ob_frame(i) for i in range(_FRAMES)]))


def _structured_tiffs(directory, prefix, frames, proton_charge):
    """TIFFs carrying the per-frame arrays above, with the EXIF the VENUS pipelines read."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame.astype(np.float32))
        exif = image.getexif()
        exif[65027] = "ExposureTime:30.000000"
        exif[65022] = f"RunNo:{1000 + index}"
        exif[65025] = "ManufacturerStr:DW936_BV"
        exif[65024] = f"IntegratedPCharge:{proton_charge}"
        path = directory / f"{prefix}_{index:03}.tiff"
        image.save(path, exif=exif)
        paths.append(path)
    return paths


def _box(stack, size):
    """iBeatles' normalized box filter over the two spatial axes of a (frame, y, x) stack."""
    kernel = np.ones((1, size, size))
    kernel /= kernel.sum()
    return convolve(stack, kernel, mode="reflect")


def _expected_transmission(sample_stack, ob_stack):
    """``(S / pc_sample) / (O / pc_ob)``, the formula the pipelines implement."""
    return (sample_stack / _PC_SAMPLE) / (ob_stack / _PC_OB)


def _read(path, dataset="transmission"):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset])


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------


@pytest.fixture
def tpx1_inputs(tmp_path):
    sample_dir, ob_dir = tmp_path / "sample", tmp_path / "ob"
    sample_dir.mkdir()
    ob_dir.mkdir()
    left_edges = [round(0.1 * (i + 1), 1) for i in range(_FRAMES)]
    _spectra_file(sample_dir / "sample_Spectra.txt", left_edges)
    _spectra_file(ob_dir / "ob_Spectra.txt", left_edges)
    return {
        "sample_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "s.h5", _PC_SAMPLE, das_image_path=b"auto")],
        "ob_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "o.h5", _PC_OB, das_image_path=b"auto")],
        "sample_tiff_paths": [_structured_tiffs(sample_dir, "s", _SAMPLE_STACK, _PC_SAMPLE)],
        "ob_tiff_paths": [_structured_tiffs(ob_dir, "o", _OB_STACK, _PC_OB)],
    }


@pytest.fixture
def histogram_inputs(tmp_path):
    return {
        "sample_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "hs.h5", _PC_SAMPLE, tof_bins=_FRAMES)],
        "ob_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "ho.h5", _PC_OB, tof_bins=_FRAMES)],
        "sample_tiff_paths": [_structured_tiffs(tmp_path / "hs", "hs", _SAMPLE_STACK, _PC_SAMPLE)],
        "ob_tiff_paths": [_structured_tiffs(tmp_path / "ho", "ho", _OB_STACK, _PC_OB)],
    }


@pytest.fixture
def event_inputs(tmp_path):
    """Flood-illuminated event files: uniform, so only the reachability checks use these."""
    return {
        "binning": BinningConfig(bins=5, bin_space="tof", tof_range=(100000, 125000), use_log_bin=False),
        "sample_paths": [
            _tpx3_event_file(
                tmp_path / "es.h5", 3, bank="bank100_events", offset=1_000_000, proton_charge=_PC_SAMPLE, n_tof=5
            )
        ],
        "ob_paths": [
            _tpx3_event_file(
                tmp_path / "eo.h5", 6, bank="bank100_events", offset=1_000_000, proton_charge=_PC_OB, n_tof=5
            )
        ],
        "detector_shape": (_DETECTOR, _DETECTOR),
    }


_RUNNERS = {
    "venus_tpx1": ("tpx1_inputs", run_venus_tpx1_pipeline),
    "venus_tpx3_histogram": ("histogram_inputs", run_venus_tpx3_histogram_pipeline),
    "venus_tpx3_event": ("event_inputs", run_venus_tpx3_event_pipeline),
}


@pytest.fixture
def pipeline(request, tpx1_inputs, histogram_inputs, event_inputs):
    """A ``callable(output_path, **kwargs)`` for the pipeline named by the parametrization."""
    available = {"tpx1_inputs": tpx1_inputs, "histogram_inputs": histogram_inputs, "event_inputs": event_inputs}
    fixture_name, runner = _RUNNERS[request.param]
    inputs = available[fixture_name]
    return lambda output_path, **kwargs: runner(output_path=output_path, **inputs, **kwargs)


_ALL = pytest.mark.parametrize("pipeline", list(_RUNNERS), indirect=True)
#: The two pipelines whose pixel values these tests control frame by frame.
_TIFF_BASED = pytest.mark.parametrize("pipeline", ["venus_tpx1", "venus_tpx3_histogram"], indirect=True)


# --------------------------------------------------------------------------------------------
# The window runs, in every pipeline that should have it
# --------------------------------------------------------------------------------------------


@_ALL
def test_the_window_runs_before_normalization(pipeline, tmp_path):
    """The stage fires, and it fires BEFORE the normalization stage — the point the issue asks for."""
    events = []
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3), progress=events.append)

    stages = [event.stage for event in events]
    assert STAGE_MOVING_WINDOW in stages
    assert stages.index(STAGE_MOVING_WINDOW) < stages.index(STAGE_NORMALIZE)


@_ALL
def test_no_window_stage_when_none_is_requested(pipeline, tmp_path):
    events = []
    pipeline(tmp_path / "out.hdf5", progress=events.append)
    assert STAGE_MOVING_WINDOW not in {event.stage for event in events}


@_ALL
def test_the_declared_total_matches_the_events_emitted(pipeline, tmp_path):
    """Both stacks are filtered under one stage, so the count must span both."""
    events = []
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3), progress=events.append)
    window_events = [event for event in events if event.stage == STAGE_MOVING_WINDOW]
    assert window_events
    assert window_events[-1].completed == window_events[-1].total
    # sample: values + variances + weights; open beam: the same, using the sample's masks
    assert window_events[-1].total == 6


@_ALL
def test_the_output_keeps_its_shape(pipeline, tmp_path):
    """A window is not a rebin: the array is the same size afterwards."""
    plain = pipeline(tmp_path / "plain.hdf5")
    filtered = pipeline(tmp_path / "filtered.hdf5", moving_window=MovingWindow(x=3, y=3))
    assert filtered.shape == plain.shape
    assert filtered.dims == plain.dims


# --------------------------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------------------------


@_TIFF_BASED
def test_an_unfiltered_run_is_the_plain_ratio(pipeline, tmp_path):
    """Checks the model the filtered assertion below depends on, rather than assuming it."""
    pipeline(tmp_path / "plain.hdf5")
    np.testing.assert_allclose(
        _read(tmp_path / "plain.hdf5"), _expected_transmission(_SAMPLE_STACK, _OB_STACK), rtol=1e-6
    )


@_TIFF_BASED
@pytest.mark.parametrize("size", [3, 5])
def test_a_filtered_run_is_the_ratio_of_the_filtered_stacks(pipeline, tmp_path, size):
    """Both stacks are filtered, then divided — as iBeatles does it."""
    pipeline(tmp_path / "filtered.hdf5", moving_window=MovingWindow(x=size, y=size))
    expected = _expected_transmission(_box(_SAMPLE_STACK, size), _box(_OB_STACK, size))
    np.testing.assert_allclose(_read(tmp_path / "filtered.hdf5"), expected, rtol=1e-6)


@_TIFF_BASED
def test_the_filtered_result_actually_differs_from_the_unfiltered_one(pipeline, tmp_path):
    """Guards against a window that is wired in but never applied, which every check above would pass."""
    plain = pipeline(tmp_path / "plain.hdf5")
    filtered = pipeline(tmp_path / "filtered.hdf5", moving_window=MovingWindow(x=3, y=3))
    assert not np.allclose(plain.values, filtered.values)


@_TIFF_BASED
def test_the_window_is_applied_after_the_spatial_rebin(pipeline, tmp_path):
    """The kernel frame is post-crop and post-rebin, so 3 on a 2x-rebinned stack spans 6 pixels."""
    pipeline(tmp_path / "out.hdf5", rebin_by_spatial=2, moving_window=MovingWindow(x=3, y=3))

    def rebinned(stack):
        frames, ny, nx = stack.shape
        return stack.reshape(frames, ny // 2, 2, nx // 2, 2).sum(axis=(2, 4))

    expected = _expected_transmission(_box(rebinned(_SAMPLE_STACK), 3), _box(rebinned(_OB_STACK), 3))
    np.testing.assert_allclose(_read(tmp_path / "out.hdf5"), expected, rtol=1e-6)


@_TIFF_BASED
def test_the_window_is_applied_after_the_roi_crop(pipeline, tmp_path):
    """Kernel indices are offsets into the cropped image, not detector pixels."""
    roi = (8, 6, 28, 30)  # x0, y0, x1, y1, exclusive stops
    pipeline(tmp_path / "out.hdf5", roi=roi, moving_window=MovingWindow(x=3, y=3))
    cropped = (slice(None), slice(roi[1], roi[3]), slice(roi[0], roi[2]))
    expected = _expected_transmission(_box(_SAMPLE_STACK[cropped], 3), _box(_OB_STACK[cropped], 3))
    np.testing.assert_allclose(_read(tmp_path / "out.hdf5"), expected, rtol=1e-6)


# --------------------------------------------------------------------------------------------
# moving_sum
# --------------------------------------------------------------------------------------------


@_ALL
def test_moving_sum_and_moving_average_are_indistinguishable_before_normalization(pipeline, tmp_path):
    """The kernel count cancels in the ratio, so the transmission is the same either way.

    Documented behaviour, not a latent surprise: a moving sum applied to both stacks before the
    division is ``k`` times the average on each side, and ``k`` cancels. It holds only here — after
    normalization a sum would scale transmission by ``k``, which is not a transmission.
    """
    average = pipeline(tmp_path / "avg.hdf5", moving_window=MovingWindow(x=3, y=3, kind="average"))
    total = pipeline(tmp_path / "sum.hdf5", moving_window=MovingWindow(x=3, y=3, kind="sum"))

    # The stacks reach this point in float32, so the agreement is to float32 round-off, not to
    # float64's. Measured across the three pipelines: at most 1.5e-8 relative, an eighth of one
    # float32 epsilon. The bound below is deliberately looser so it cannot fail on the other
    # architecture in CI, and is still six orders of magnitude tighter than any real difference.
    np.testing.assert_allclose(total.values, average.values, rtol=1e-6)

    # ... and so is the RELATIVE uncertainty, which is what a user reads off the error bars
    relative_average = np.sqrt(average.variances) / np.abs(average.values)
    relative_total = np.sqrt(total.variances) / np.abs(total.values)
    np.testing.assert_allclose(relative_total, relative_average, rtol=1e-6)


@_TIFF_BASED
def test_a_sum_run_and_an_average_run_agree_far_more_closely_than_either_differs_from_no_window(pipeline, tmp_path):
    """The equivalence is only worth pinning if a window changes anything at all here."""
    plain = pipeline(tmp_path / "plain.hdf5")
    average = pipeline(tmp_path / "avg.hdf5", moving_window=MovingWindow(x=3, y=3))
    total = pipeline(tmp_path / "sum.hdf5", moving_window=MovingWindow(x=3, y=3, kind="sum"))

    sum_vs_average = np.abs(total.values - average.values).max()
    window_vs_none = np.abs(average.values - plain.values).max()
    # Measured: 2.0e-7 against 0.112, a factor of 5.6e5. The bound asserts four orders of magnitude,
    # which leaves room for the other CI architecture without weakening the claim to nothing.
    assert sum_vs_average * 1e4 < window_vs_none


# --------------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------------


@_TIFF_BASED
def test_a_dead_pixel_is_excluded_from_the_window_rather_than_averaged_in(pipeline, tmp_path, monkeypatch):
    """A mask-blind window would spread one dead pixel across ``k**2`` transmission pixels."""
    import neunorm.pipelines._tof_spine as spine

    dead_y, dead_x = 20, 21
    real_detect = spine.detect_dead_pixels

    def fake_detect(hist):
        mask = real_detect(hist)
        mask.values[dead_y, dead_x] = True
        return mask

    monkeypatch.setattr(spine, "detect_dead_pixels", fake_detect)
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3))

    good = np.ones((_DETECTOR, _DETECTOR))
    good[dead_y, dead_x] = 0.0
    kernel = np.ones((1, 3, 3))
    expected = _expected_transmission(
        convolve(_SAMPLE_STACK * good, kernel, mode="reflect") / convolve(good[None], kernel, mode="reflect"),
        convolve(_OB_STACK * good, kernel, mode="reflect") / convolve(good[None], kernel, mode="reflect"),
    )
    got = _read(tmp_path / "out.hdf5")
    neighbourhood = (slice(None), slice(dead_y - 1, dead_y + 2), slice(dead_x - 1, dead_x + 2))
    np.testing.assert_allclose(got[neighbourhood], expected[neighbourhood], rtol=1e-6)


# Provenance in the written file is covered by ``test_moving_window_guards.py``, alongside the
# refusals and warnings it belongs with.
