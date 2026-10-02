"""``output_path`` handling at the six pipeline entry points.

A ``str`` path must write the same file as the equivalent ``Path``, suffixes match case-insensitively in
image and spectrum mode alike, and an ``output_path`` whose suffix the export step cannot write must be
refused before any input file is read: no reader is called and no progress event is emitted. The refusal
messages are the export step's own.
"""

import importlib
import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from scitiff.io import load_scitiff
from test_progress_pipelines import (
    _DETECTOR,
    _ccd_tiffs,
    _spectra_file,
    _tpx3_event_file,
    _venus_metadata_nexus,
)

from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
from neunorm.pipelines.mars_tpx3 import run_mars_tpx3_pipeline
from neunorm.pipelines.venus_ccd import run_venus_ccd_pipeline
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline
from neunorm.pipelines.venus_tpx3_histogram import run_venus_tpx3_histogram_pipeline

_FRAMES = 5
_REGION = (10, 10, 26, 26)

#: Metadata that differs between any two runs of the same inputs.
_RUN_STAMPED = ("processing_timestamp", "version")

#: The functions each pipeline module reads its input files through.
_READERS = {
    "mars_ccd": ["load_runs"],
    "venus_ccd": ["load_runs"],
    "mars_tpx3": ["load_event_nexus"],
    "venus_tpx1": ["load_metadata", "load_tiff_stack"],
    "venus_tpx3_histogram": ["load_metadata", "load_tiff_stack"],
    "venus_tpx3_event": ["load_metadata", "load_event_nexus"],
}

_ALL = sorted(_READERS)
_TOF = ["venus_tpx1", "venus_tpx3_event", "venus_tpx3_histogram"]


@pytest.fixture(scope="module")
def inputs():
    """The smallest synthetic inputs each of the six pipelines accepts."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        tpx1_sample, tpx1_ob = directory / "autoreduce" / "sample", directory / "autoreduce" / "ob"
        tpx1_sample.mkdir(parents=True)
        tpx1_ob.mkdir(parents=True)
        left_edges = [round(0.1 * (i + 1), 1) for i in range(_FRAMES)]
        _spectra_file(tpx1_sample / "sample_Spectra.txt", left_edges)
        _spectra_file(tpx1_ob / "ob_Spectra.txt", left_edges)
        yield {
            "mars_ccd": {
                "sample_paths": [_ccd_tiffs(directory, "mars_s", 3, 81, motslit=True)],
                "ob_paths": [_ccd_tiffs(directory, "mars_o", 2, 99, motslit=True)],
                "dark_paths": [_ccd_tiffs(directory, "mars_d", 2, 5, motslit=True)],
            },
            "venus_ccd": {
                "sample_paths": [_ccd_tiffs(directory, "vccd_s", 3, 81, proton_charge=1000.0)],
                "ob_paths": [_ccd_tiffs(directory, "vccd_o", 2, 99, proton_charge=1010.0)],
                "dark_paths": [_ccd_tiffs(directory, "vccd_d", 2, 5, proton_charge=1.0)],
            },
            "mars_tpx3": {
                "sample_paths": [[_tpx3_event_file(directory / "mars_s.hdf5", 3)]],
                "ob_paths": [[_tpx3_event_file(directory / "mars_o.hdf5", 6)]],
                "detector_shape": (_DETECTOR, _DETECTOR),
            },
            "venus_tpx1": {
                "sample_tiff_paths": [_ccd_tiffs(tpx1_sample, "sample", _FRAMES, 81)],
                "ob_tiff_paths": [_ccd_tiffs(tpx1_ob, "ob", _FRAMES, 99)],
                "sample_hdf5_paths": [
                    _venus_metadata_nexus(directory / "nx" / "t1s.h5", 12345, das_image_path=b"autoreduce/sample")
                ],
                "ob_hdf5_paths": [
                    _venus_metadata_nexus(directory / "nx" / "t1o.h5", 24690, das_image_path=b"autoreduce/ob")
                ],
            },
            "venus_tpx3_histogram": {
                "sample_tiff_paths": [_ccd_tiffs(directory, "hist_s", _FRAMES, 81)],
                "ob_tiff_paths": [_ccd_tiffs(directory, "hist_o", _FRAMES, 99)],
                "sample_hdf5_paths": [_venus_metadata_nexus(directory / "nx" / "hs.h5", 12345, tof_bins=_FRAMES)],
                "ob_hdf5_paths": [_venus_metadata_nexus(directory / "nx" / "ho.h5", 24690, tof_bins=_FRAMES)],
            },
            "venus_tpx3_event": {
                "sample_paths": [
                    _tpx3_event_file(
                        directory / "ev_s.hdf5",
                        3,
                        bank="bank100_events",
                        offset=1_000_000,
                        proton_charge=12345,
                        n_tof=5,
                    )
                ],
                "ob_paths": [
                    _tpx3_event_file(
                        directory / "ev_o.hdf5",
                        6,
                        bank="bank100_events",
                        offset=1_000_000,
                        proton_charge=24690,
                        n_tof=5,
                    )
                ],
                "binning": BinningConfig(bins=5, bin_space="tof", tof_range=(100000, 125000), use_log_bin=False),
                "detector_shape": (_DETECTOR, _DETECTOR),
            },
        }


_PIPELINES = {
    "mars_ccd": run_mars_ccd_pipeline,
    "venus_ccd": run_venus_ccd_pipeline,
    "mars_tpx3": run_mars_tpx3_pipeline,
    "venus_tpx1": run_venus_tpx1_pipeline,
    "venus_tpx3_histogram": run_venus_tpx3_histogram_pipeline,
    "venus_tpx3_event": run_venus_tpx3_event_pipeline,
}


def _run(name, inputs, output_path, **kwargs):
    return _PIPELINES[name](output_path=output_path, **inputs[name], **kwargs)


@pytest.fixture
def reader_calls(monkeypatch):
    """Count every call into the functions the pipeline modules read input files through."""
    calls = []
    for name, readers in _READERS.items():
        module = importlib.import_module(f"neunorm.pipelines.{name}")
        for reader in readers:
            original = getattr(module, reader)

            def counted(*args, _original=original, _label=f"{name}.{reader}", **kwargs):
                calls.append(_label)
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, reader, counted)
    return calls


def _hdf5_contents(path):
    """Every dataset in an HDF5 file with its attributes, less the run-stamped metadata."""
    stamped = {f"metadata/{key}" for key in _RUN_STAMPED}
    contents = {}

    def visit(name, item):
        if isinstance(item, h5py.Dataset) and name not in stamped:
            contents[name] = (item[()], dict(item.attrs))

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return contents


def _assert_same_hdf5(path_a, path_b):
    a, b = _hdf5_contents(path_a), _hdf5_contents(path_b)
    assert sorted(a) == sorted(b)
    assert "transmission" in a
    for name, (values, attrs) in a.items():
        np.testing.assert_array_equal(values, b[name][0], err_msg=name)
        assert attrs == b[name][1], name


def _assert_same_tiff(path_a, path_b):
    a, b = load_scitiff(path_a), load_scitiff(path_b)
    assert sorted(a.keys()) == sorted(b.keys())
    assert sc.identical(a["image"], b["image"])
    for key in a.keys():
        if key == "image":
            continue
        if key == "extra":
            assert {k: v for k, v in a[key].items() if k not in _RUN_STAMPED} == {
                k: v for k, v in b[key].items() if k not in _RUN_STAMPED
            }
        else:
            assert a[key] == b[key], key


# --------------------------------------------------------------------------------------
# a str output_path is a Path
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
@pytest.mark.parametrize("suffix", [".h5", ".tiff"])
def test_a_str_output_path_writes_the_same_file_as_a_path(name, suffix, inputs, tmp_path):
    from_path = tmp_path / f"path{suffix}"
    from_str = tmp_path / f"str{suffix}"

    _run(name, inputs, from_path)
    _run(name, inputs, str(from_str))

    if suffix == ".h5":
        _assert_same_hdf5(from_path, from_str)
    else:
        _assert_same_tiff(from_path, from_str)


@pytest.mark.parametrize("name", _TOF)
def test_a_str_spectrum_output_path_writes_the_same_files_as_a_path(name, inputs, tmp_path):
    """``.txt`` also writes ``<stem>.hdf5`` beside it, which is derived from the path."""
    from_path = tmp_path / "path" / "spectrum.txt"
    from_str = tmp_path / "str" / "spectrum.txt"

    _run(name, inputs, from_path, spectrum_roi=_REGION)
    _run(name, inputs, str(from_str), spectrum_roi=_REGION)

    assert from_str.read_bytes() == from_path.read_bytes()
    _assert_same_hdf5(from_path.with_suffix(".hdf5"), from_str.with_suffix(".hdf5"))


# --------------------------------------------------------------------------------------
# suffixes match case-insensitively
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
def test_the_suffix_of_a_str_output_path_matches_case_insensitively(name, inputs, tmp_path):
    output_path = tmp_path / "upper.H5"

    _run(name, inputs, str(output_path))

    with h5py.File(output_path, "r") as f:
        assert "transmission" in f


#: The spectrum each TOF pipeline writes for ``_REGION`` of the ``inputs`` fixture, by hand. The two
#: TIFF pipelines read uniform frames of ``81 + i`` against ``99 + i`` with proton charges 12345 and
#: 24690, so bin i is ``2 * (81 + i) / (99 + i)``. The event pipeline counts 3 events per pixel per bin
#: against 6 with the same proton charges, so every bin is 1. The TIFF pipelines carry float32, so the
#: HDF5 values match to float32 precision (``_SPECTRUM_RTOL``).
_SPECTRUM = {
    "venus_tpx1": [162 / 99, 164 / 100, 166 / 101, 168 / 102, 170 / 103],
    "venus_tpx3_histogram": [162 / 99, 164 / 100, 166 / 101, 168 / 102, 170 / 103],
    "venus_tpx3_event": [1.0, 1.0, 1.0, 1.0, 1.0],
}
_SPECTRUM_RTOL = 1e-6


@pytest.mark.parametrize("name", _TOF)
def test_an_upper_case_txt_spectrum_path_writes_the_ascii_spectrum_and_its_hdf5(name, inputs, tmp_path):
    """``.TXT`` is ``.txt``: the three-column file at the given path plus ``<stem>.hdf5`` beside it."""
    output_path = tmp_path / "spectrum.TXT"

    _run(name, inputs, output_path, spectrum_roi=_REGION)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["spectrum.TXT", "spectrum.hdf5"]
    lines = output_path.read_text().splitlines()
    assert lines[0] == "bin_index,transmission,uncertainty"
    table = np.loadtxt(output_path, skiprows=1, delimiter=",", ndmin=2)
    np.testing.assert_array_equal(table[:, 0], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(table[:, 1], _SPECTRUM[name], atol=5e-7, rtol=0)
    with h5py.File(tmp_path / "spectrum.hdf5", "r") as f:
        np.testing.assert_allclose(f["transmission"][()], _SPECTRUM[name], rtol=_SPECTRUM_RTOL)


@pytest.mark.parametrize("name", _TOF)
def test_an_upper_case_h5_spectrum_path_writes_hdf5_only(name, inputs, tmp_path):
    output_path = tmp_path / "spectrum.H5"

    _run(name, inputs, output_path, spectrum_roi=_REGION)

    assert [p.name for p in tmp_path.iterdir()] == ["spectrum.H5"]
    with h5py.File(output_path, "r") as f:
        np.testing.assert_allclose(f["transmission"][()], _SPECTRUM[name], rtol=_SPECTRUM_RTOL)


# --------------------------------------------------------------------------------------
# a missing output_path or an unwritable suffix is refused before any input is read
# --------------------------------------------------------------------------------------


def _exactly(message):
    return f"^{re.escape(message)}$"


def _unsupported(suffix):
    return _exactly(f"Unsupported output file format: {suffix}")


def _spectrum_tiff(name):
    return _exactly(
        f"spectrum_roi produces a 1-D spectrum, which cannot be written as a TIFF image stack (got {name}). "
        "Use '.txt' for the three-column ASCII spectrum (an HDF5 file is written alongside it) or '.hdf5' "
        "for HDF5 only."
    )


_IMAGE_REFUSALS = [
    ("out.bmp", _unsupported(".bmp")),
    ("out.txt", _unsupported(".txt")),
    ("out", _unsupported("")),
]
_SPECTRUM_REFUSALS = [
    ("out.tiff", _spectrum_tiff("out.tiff")),
    ("out.TIF", _spectrum_tiff("out.TIF")),
    ("out.bmp", _unsupported(".bmp")),
]


def _assert_refused_before_reading(call, pattern, reader_calls, out_dir):
    events = []
    with pytest.raises(ValueError, match=pattern):
        call(events.append)
    assert reader_calls == [], "an input file was read before the output path was refused"
    assert events == [], "the run started before the output path was refused"
    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize("name", _ALL)
@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
@pytest.mark.parametrize(("filename", "pattern"), _IMAGE_REFUSALS, ids=[f for f, _ in _IMAGE_REFUSALS])
def test_an_unsupported_image_suffix_is_refused_before_any_input_is_read(
    name, as_str, filename, pattern, inputs, reader_calls, tmp_path
):
    output_path = tmp_path / filename

    _assert_refused_before_reading(
        lambda sink: _run(name, inputs, str(output_path) if as_str else output_path, progress=sink),
        pattern,
        reader_calls,
        tmp_path,
    )


@pytest.mark.parametrize("name", _TOF)
@pytest.mark.parametrize("as_str", [False, True], ids=["Path", "str"])
@pytest.mark.parametrize(("filename", "pattern"), _SPECTRUM_REFUSALS, ids=[f for f, _ in _SPECTRUM_REFUSALS])
def test_an_unsupported_spectrum_suffix_is_refused_before_any_input_is_read(
    name, as_str, filename, pattern, inputs, reader_calls, tmp_path
):
    output_path = tmp_path / filename

    _assert_refused_before_reading(
        lambda sink: _run(
            name, inputs, str(output_path) if as_str else output_path, spectrum_roi=_REGION, progress=sink
        ),
        pattern,
        reader_calls,
        tmp_path,
    )


@pytest.mark.parametrize("name", ["mars_tpx3", *_TOF])
def test_a_missing_output_path_is_refused_before_any_input_is_read(name, inputs, reader_calls, tmp_path):
    """The four pipelines whose ``output_path`` has no default; the CCD pair's ``None`` refusal is pinned
    in their own test files."""
    _assert_refused_before_reading(
        lambda sink: _run(name, inputs, None, progress=sink),
        _exactly("output_path is required"),
        reader_calls,
        tmp_path,
    )


# the export step refuses an unsupported suffix on its own


@pytest.fixture
def entry_check_bypassed(monkeypatch):
    """Let any output_path through the entry check, so a run reaches the export step's own refusal."""
    for name in _ALL:
        module = importlib.import_module(f"neunorm.pipelines.{name}")
        monkeypatch.setattr(module, "resolve_output_path", lambda output_path, **_: Path(output_path))


@pytest.mark.usefixtures("entry_check_bypassed")
@pytest.mark.parametrize("name", _ALL)
def test_the_export_step_refuses_an_unsupported_image_suffix(name, inputs, tmp_path):
    with pytest.raises(ValueError, match=_unsupported(".bmp")):
        _run(name, inputs, tmp_path / "out.bmp")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("entry_check_bypassed")
@pytest.mark.parametrize("name", _TOF)
@pytest.mark.parametrize(("filename", "pattern"), _SPECTRUM_REFUSALS, ids=[f for f, _ in _SPECTRUM_REFUSALS])
def test_the_export_step_refuses_an_unsupported_spectrum_suffix(name, filename, pattern, inputs, tmp_path):
    with pytest.raises(ValueError, match=pattern):
        _run(name, inputs, tmp_path / filename, spectrum_roi=_REGION)
    assert list(tmp_path.iterdir()) == []
