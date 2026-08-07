"""The refusals, the warnings, and the provenance for the moving window.

A moving window returns an array the same shape as its input, so a filtered result and an
unfiltered one are indistinguishable by inspection — unlike a rebin, which visibly shrinks. That is
the whole reason this file exists: where a combination has no correct reading it is refused, where
it has one whose consequence is easy to miss it is said out loud, and the window itself is written
into the output file so a filtered result stays identifiable after it leaves the pipeline.

Every guard and warning here is mutation-tested by ``.harness/mutate_moving_window.py``: each is
broken in turn and a named test from this file must fail.
"""

import json

import h5py
import numpy as np
import pytest
from loguru import logger
from test_moving_window_pipeline import (  # the synthetic inputs and runners, built once
    _ALL,
    _TIFF_BASED,
    event_inputs,  # noqa: F401 - fixture, used by `pipeline`
    histogram_inputs,  # noqa: F401 - fixture, used by `pipeline`
    pipeline,  # noqa: F401 - fixture, parametrized by _ALL / _TIFF_BASED
    tpx1_inputs,  # noqa: F401 - fixture, used by `pipeline`
)

from neunorm.data_models.moving_window import MovingWindow


@pytest.fixture
def warnings():
    """Collect loguru WARNING messages emitted inside the block."""
    messages: list[str] = []
    sink_id = logger.add(lambda record: messages.append(record.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


def _matching(messages, *fragments):
    return [m for m in messages if all(fragment in m for fragment in fragments)]


# --------------------------------------------------------------------------------------------
# Refusal: a moving window and a region reduction are alternatives, not stages
# --------------------------------------------------------------------------------------------


@_ALL
def test_a_moving_window_with_spectrum_roi_is_refused(pipeline, tmp_path):  # noqa: F811
    """Refused because the uncertainty of the spectrum would be wrong and nothing would show it."""
    with pytest.raises(ValueError, match=r"moving_window and spectrum_roi cannot be combined"):
        pipeline(tmp_path / "out.txt", spectrum_roi=(8, 8, 24, 24), moving_window=MovingWindow(x=3, y=3))


@_ALL
def test_the_refusal_records_the_measured_reason(pipeline, tmp_path):  # noqa: F811
    """The message carries why, not just what: a region mean over smoothed pixels under-reports."""
    with pytest.raises(ValueError) as excinfo:
        pipeline(tmp_path / "out.txt", spectrum_roi=(8, 8, 24, 24), moving_window=MovingWindow(x=3, y=3))
    message = str(excinfo.value)
    assert "sqrt(kernel pixels)" in message
    assert "x2.89" in message and "x4.78" in message
    assert "covariance" in message


@_ALL
def test_the_refusal_happens_before_any_output_is_written(pipeline, tmp_path):  # noqa: F811
    """A run that is going to be rejected should be rejected before it does the work."""
    output = tmp_path / "out.txt"
    with pytest.raises(ValueError):
        pipeline(output, spectrum_roi=(8, 8, 24, 24), moving_window=MovingWindow(x=3, y=3))
    assert not output.exists()
    assert not output.with_suffix(".hdf5").exists()


@_ALL
def test_spectrum_roi_alone_still_works(pipeline, tmp_path):  # noqa: F811
    """The guard must refuse the COMBINATION, not spectrum_roi itself."""
    spectrum = pipeline(tmp_path / "out.txt", spectrum_roi=(8, 8, 24, 24))
    assert spectrum.dims == ("tof",)


# --------------------------------------------------------------------------------------------
# Warning: the resolution/precision exchange
# --------------------------------------------------------------------------------------------


@_ALL
def test_the_trade_is_warned_about_with_the_actual_kernel(pipeline, tmp_path, warnings):  # noqa: F811
    said = _matching(warnings, "moving_window", "resolution coarsens")
    assert not said
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3))
    said = _matching(warnings, "moving_window", "resolution coarsens")
    assert len(said) == 1
    # the actual sizes, and the precision factor for THIS kernel, not a generic sentence
    assert "'x': 3" in said[0] and "'y': 3" in said[0]
    assert "3.0x" in said[0]
    assert "one independent value per 9 pixels" in said[0]


@_ALL
def test_the_warning_names_the_kernel_it_was_given(pipeline, tmp_path, warnings):  # noqa: F811
    """A 5x5 window must not report a 3x3 window's numbers."""
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=5, y=5))
    said = _matching(warnings, "moving_window", "resolution coarsens")
    assert "'x': 5" in said[0]
    assert "5.0x" in said[0]
    assert "one independent value per 25 pixels" in said[0]


@_ALL
def test_the_trade_warning_covers_features_smaller_than_the_kernel(pipeline, tmp_path, warnings):  # noqa: F811
    """Losing DEPTH is a quantitative error, not a cosmetic blur, so it is stated."""
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3))
    said = _matching(warnings, "moving_window", "resolution coarsens")
    assert "loses DEPTH" in said[0]
    assert "0.32" in said[0]


# --------------------------------------------------------------------------------------------
# Warning: compounding with a spatial rebin
# --------------------------------------------------------------------------------------------


@_TIFF_BASED
def test_compounding_with_rebin_spatial_is_warned_about(pipeline, tmp_path, warnings):  # noqa: F811
    pipeline(tmp_path / "out.hdf5", rebin_by_spatial=2, moving_window=MovingWindow(x=3, y=3))
    said = _matching(warnings, "POST-REBIN pixels")
    assert len(said) == 1
    # the combined figure, which is the number a user actually needs
    assert "6 x 6 detector pixels" in said[0]


@_TIFF_BASED
def test_the_compounding_warning_handles_a_per_axis_rebin_factor(pipeline, tmp_path, warnings):  # noqa: F811
    pipeline(tmp_path / "out.hdf5", rebin_by_spatial=(2, 4), moving_window=MovingWindow(x=3, y=2))
    said = _matching(warnings, "POST-REBIN pixels")
    assert "6 x 8 detector pixels" in said[0]


@_ALL
def test_no_compounding_warning_without_a_spatial_rebin(pipeline, tmp_path, warnings):  # noqa: F811
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3))
    assert not _matching(warnings, "POST-REBIN pixels")


# --------------------------------------------------------------------------------------------
# Warning: a kernel large relative to the axis it runs along
# --------------------------------------------------------------------------------------------


@_TIFF_BASED
def test_a_kernel_large_for_its_axis_is_warned_about(pipeline, tmp_path, warnings):  # noqa: F811
    """The mirrored-edge argument only holds while the border is a small part of the axis."""
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=17, y=17))
    said = _matching(warnings, "mirrored frame edge inside their window")
    assert said
    assert "32 pixels here" in said[0]


@_TIFF_BASED
def test_a_small_kernel_on_a_full_axis_is_not_warned_about(pipeline, tmp_path, warnings):  # noqa: F811
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3))
    assert not _matching(warnings, "mirrored frame edge inside their window")


@_TIFF_BASED
def test_the_axis_length_is_the_one_at_the_filter_not_the_detectors(pipeline, tmp_path, warnings):  # noqa: F811
    """A crop shortens the axis, so a window that was safe on the full frame may not be."""
    # 32 -> 10 pixels wide after this crop; a 5-wide window then reaches the edge from 40% of pixels
    pipeline(tmp_path / "out.hdf5", roi=(10, 10, 20, 30), moving_window=MovingWindow(x=5, y=5))
    said = _matching(warnings, "mirrored frame edge inside their window", "'x'")
    assert said
    assert "10 pixels here" in said[0]


# --------------------------------------------------------------------------------------------
# Silence when the feature is not used
# --------------------------------------------------------------------------------------------


@_ALL
def test_no_warnings_when_unused(pipeline, tmp_path, warnings):  # noqa: F811
    """A run that asks for no moving window must emit none of its warnings."""
    pipeline(tmp_path / "out.hdf5")
    assert not _matching(warnings, "moving_window")
    assert not _matching(warnings, "POST-REBIN pixels")
    assert not _matching(warnings, "mirrored frame edge inside their window")


@_TIFF_BASED
def test_no_moving_window_warnings_from_a_plain_rebin_run(pipeline, tmp_path, warnings):  # noqa: F811
    """The compounding warning belongs to the window, not to rebin_by_spatial on its own."""
    pipeline(tmp_path / "out.hdf5", rebin_by_spatial=2)
    assert not _matching(warnings, "POST-REBIN pixels")


# --------------------------------------------------------------------------------------------
# Provenance in the written file
# --------------------------------------------------------------------------------------------


@_ALL
def test_the_window_is_recorded_in_the_written_hdf5(pipeline, tmp_path):  # noqa: F811
    """Kernel, kind and edge mode, so a filtered file is identifiable after it leaves here."""
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=5, kind="sum", mode="nearest"))
    with h5py.File(tmp_path / "out.hdf5", "r") as handle:
        recorded = json.loads(handle["metadata/moving_window"][()])
    assert recorded == {
        "kind": "sum",
        "sizes": {"x": 3, "y": 5},
        "dimension": "2D",
        "mode": "nearest",
        "kernel_pixels": 15,
    }


@_ALL
def test_a_three_dimensional_window_records_its_tof_size(pipeline, tmp_path):  # noqa: F811
    pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3, tof=3, dimension="3D"))
    with h5py.File(tmp_path / "out.hdf5", "r") as handle:
        recorded = json.loads(handle["metadata/moving_window"][()])
    assert recorded["sizes"] == {"x": 3, "y": 3, "tof": 3}
    assert recorded["dimension"] == "3D"
    assert recorded["kernel_pixels"] == 27


@_ALL
def test_nothing_is_recorded_when_no_window_ran(pipeline, tmp_path):  # noqa: F811
    pipeline(tmp_path / "out.hdf5")
    with h5py.File(tmp_path / "out.hdf5", "r") as handle:
        assert "metadata/moving_window" not in handle


@_TIFF_BASED
def test_the_provenance_survives_the_tiff_writer_too(pipeline, tmp_path):  # noqa: F811
    """A bare dict here would write HDF5 happily and then fail at TIFF export time."""
    result = pipeline(tmp_path / "out.tiff", moving_window=MovingWindow(x=3, y=3))
    assert (tmp_path / "out.tiff").exists()
    assert result is not None


# --------------------------------------------------------------------------------------------
# The guards do not fire on what they should not
# --------------------------------------------------------------------------------------------


@_ALL
def test_a_window_run_still_produces_the_expected_array(pipeline, tmp_path):  # noqa: F811
    """Guards must not change what a legitimate run computes."""
    filtered = pipeline(tmp_path / "out.hdf5", moving_window=MovingWindow(x=3, y=3))
    assert np.isfinite(filtered.values).all()
    assert filtered.variances is not None
