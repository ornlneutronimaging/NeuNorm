"""Unit tests for the :class:`MovingWindow` configuration model.

The model exists to carry the kernel size the issue asks for ("the option of giving the kernel size
to use in this average") without adding five loose parameters to three pipelines, and to refuse the
two configurations that would quietly not do what was asked.
"""

import pytest
from pydantic import ValidationError

from neunorm.data_models import MovingWindow as MovingWindowFromPackage
from neunorm.data_models.moving_window import MovingWindow
from neunorm.processing.moving_window import EDGE_MODES


def test_exported_from_the_data_models_package():
    assert MovingWindowFromPackage is MovingWindow


def test_defaults_are_a_two_dimensional_average():
    config = MovingWindow(x=3, y=3)
    assert config.kind == "average"
    assert config.dimension == "2D"
    assert config.mode == "reflect"
    assert config.sizes() == {"x": 3, "y": 3}
    assert config.kernel_pixels == 9


def test_sizes_are_named_not_positional():
    """A non-square kernel must survive the (tof, x, y) / (tof, y, x) difference between detectors."""
    assert MovingWindow(x=3, y=7).sizes() == {"x": 3, "y": 7}


def test_a_window_on_one_axis_only_leaves_the_other_at_one():
    assert MovingWindow(x=5).sizes() == {"x": 5, "y": 1}
    assert MovingWindow(x=5).kernel_pixels == 5


def test_a_three_dimensional_window_reaches_the_tof_axis():
    config = MovingWindow(x=3, y=3, tof=5, dimension="3D")
    assert config.sizes() == {"x": 3, "y": 3, "tof": 5}
    assert config.kernel_pixels == 45


def test_a_two_dimensional_window_cannot_reach_the_tof_axis():
    """Even if a tof size were set and the dimension later changed, ``sizes()`` must not leak it."""
    config = MovingWindow(x=3, y=3, tof=4, dimension="3D")
    assert "tof" in config.sizes()
    assert "tof" not in config.model_copy(update={"dimension": "2D"}).sizes()


def test_a_tof_size_without_the_three_dimensional_flag_is_refused():
    """Averaging along TOF blurs resonance dips, so it is asked for rather than inferred."""
    with pytest.raises(ValidationError, match=r"needs dimension='3D'"):
        MovingWindow(x=3, y=3, tof=5)


def test_an_identity_window_is_refused():
    """All sizes 1 would filter nothing; silently doing nothing is the worst of the options."""
    with pytest.raises(ValidationError, match=r"identity and would filter nothing"):
        MovingWindow()
    with pytest.raises(ValidationError, match=r"identity and would filter nothing"):
        MovingWindow(x=1, y=1, kind="sum")


@pytest.mark.parametrize("axis", ["x", "y", "tof"])
@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_size_is_refused(axis, bad):
    with pytest.raises(ValidationError, match=r"greater than or equal to 1"):
        MovingWindow(**{axis: bad})


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValidationError, match=r"'average' or 'sum'"):
        MovingWindow(x=3, y=3, kind="median")


def test_an_unknown_edge_mode_is_refused():
    with pytest.raises(ValidationError):
        MovingWindow(x=3, y=3, mode="bounce")


@pytest.mark.parametrize("mode", EDGE_MODES)
def test_every_mode_the_filter_accepts_the_config_also_accepts(mode):
    """The config's accepted set is the filter's, so one cannot drift from the other."""
    assert MovingWindow(x=3, y=3, mode=mode).mode == mode


def test_even_sizes_are_accepted_as_ibeatles_accepts_them():
    assert MovingWindow(x=4, y=2).sizes() == {"x": 4, "y": 2}


def test_provenance_records_what_a_reader_of_the_file_would_need():
    config = MovingWindow(x=3, y=5, kind="sum", mode="nearest")
    assert config.provenance() == {
        "kind": "sum",
        "sizes": {"x": 3, "y": 5},
        "dimension": "2D",
        "mode": "nearest",
        "kernel_pixels": 15,
    }
