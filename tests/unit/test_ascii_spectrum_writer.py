"""The ASCII spectrum file is an interoperability contract, so its bytes are pinned here.

NeuNorm's own spectra reader is ``np.loadtxt(path, skiprows=1, delimiter=",")`` — it skips exactly
one line and parses everything after it as data. That fixes the whole format: comma-delimited, one
plain header line with no ``#`` prefix (a commented header would be the skipped line and the first
data row would be lost), one row per bin. These tests therefore read the file back as **text** and
through that exact reader call, rather than through any NeuNorm helper, so a change to the
delimiter, the header, the column formats or the bin-index semantics fails here instead of in a
downstream tool months later.

The three columns are ``bin_index,transmission,uncertainty``; ``uncertainty`` is 1 sigma, so a
spectrum with no variances has nothing truthful to put there and is rejected rather than written
with a fabricated column. ``bin_index`` is each bin's **first input frame**, which is what lets a
rebinned spectrum be traced back to its files: a gapped bin list leaves gaps in the column instead
of renumbering the rows.
"""

import filecmp
import io
import os
from contextlib import redirect_stderr

import numpy as np
import pytest
import scipp as sc

from neunorm.exporters.ascii_writer import (
    ASCII_SPECTRUM_HEADER,
    ascii_spectrum_export_step_count,
    write_ascii_spectrum,
)
from neunorm.pipelines._tof_spine import spectrum_bin_indices
from neunorm.utils.progress import STAGE_EXPORT, STAGE_REDUCE_SPECTRUM

# The reference file the format was settled on, byte for byte.
_REFERENCE_HEADER = "bin_index,transmission,uncertainty"
_REFERENCE_ROW = "0,0.448760,0.010663"


def _spectrum(values, stdevs, dim="tof"):
    """A 1-D transmission spectrum from literal values and 1-sigma uncertainties."""
    values = np.asarray(values, dtype=float)
    variances = np.asarray(stdevs, dtype=float) ** 2
    return sc.DataArray(sc.array(dims=[dim], values=values, variances=variances, unit=""))


def _lines(path):
    return path.read_text().splitlines()


def _column(path, index):
    """The raw text tokens of one column, header row excluded."""
    return [line.split(",")[index] for line in _lines(path)[1:]]


def _collect():
    events = []
    return events, events.append


# --------------------------------------------------------------------------------------
# The literal file: one plain header line, one row per bin
# --------------------------------------------------------------------------------------


def test_exactly_one_plain_header_line_and_one_row_per_bin(tmp_path):
    """The reader skips one line and parses the rest, so a ``#`` prefix or a second header line
    would silently eat the first bin. Asserted on the text, not on a parsed array, because a
    parsed array cannot tell a skipped header from a lost data row."""
    path = tmp_path / "spectrum.txt"
    write_ascii_spectrum(path, _spectrum([0.5, 0.25, 0.125, 1.0], [0.01, 0.02, 0.03, 0.04]))

    text = path.read_text()
    lines = text.splitlines()

    assert lines[0] == ASCII_SPECTRUM_HEADER
    assert lines[0] == _REFERENCE_HEADER, "the header line is the settled format, not a free choice"
    assert not lines[0].startswith("#")
    assert len(lines) == 5, f"expected 1 header + 4 data rows, got {lines}"
    assert "#" not in text, "no comment prefix anywhere: the reader has no comment convention"
    assert "\t" not in text, "comma-delimited, not tab-delimited"
    assert text.endswith("\n")
    for line in lines[1:]:
        assert len(line.split(",")) == 3, f"three columns per row; got {line!r}"
        assert " " not in line, f"no padding around the delimiter; got {line!r}"


def test_reference_row_is_rendered_exactly(tmp_path):
    """The agreed example, pinned character for character: a six-decimal fixed-point value and
    uncertainty, and an integer first column (the default ``%.18e`` would render both value columns
    in exponent form and the index as ``0.000000000000000000e+00``)."""
    path = tmp_path / "reference.txt"
    write_ascii_spectrum(path, _spectrum([0.44876], [0.010663]))

    assert path.read_text() == f"{_REFERENCE_HEADER}\n{_REFERENCE_ROW}\n"
    assert _lines(path)[1] == "0,0.448760,0.010663"


def test_parses_under_the_readers_exact_loadtxt_call(tmp_path):
    """The interoperability contract itself: the call NeuNorm's spectra reader makes returns the
    spectrum value by value, with the third column equal to ``sqrt(variance)``."""
    values = [0.448760, 0.500000, 0.125000, 1.000000]
    stdevs = [0.010663, 0.002500, 0.123456, 0.050000]
    path = tmp_path / "spectrum.txt"
    write_ascii_spectrum(path, _spectrum(values, stdevs))

    table = np.loadtxt(path, skiprows=1, delimiter=",")

    assert table.shape == (4, 3)
    np.testing.assert_allclose(table[:, 0], np.array([0.0, 1.0, 2.0, 3.0]), atol=0)
    np.testing.assert_allclose(table[:, 1], np.asarray(values), atol=1e-9)
    # sqrt of the variances the spectrum carries, computed here with plain numpy from the literals.
    np.testing.assert_allclose(table[:, 2], np.sqrt(np.asarray(stdevs) ** 2), atol=1e-9)


# --------------------------------------------------------------------------------------
# The bin_index column
# --------------------------------------------------------------------------------------


def test_bin_index_defaults_to_the_row_index(tmp_path):
    """With no rebinning the column means "file index", and the row index is that same number."""
    path = tmp_path / "default.txt"
    write_ascii_spectrum(path, _spectrum([0.5, 0.4, 0.3], [0.01, 0.01, 0.01]))

    assert _column(path, 0) == ["0", "1", "2"]


def test_bin_indices_override_the_first_column(tmp_path):
    """Given indices are written verbatim as integers — not renumbered, not reformatted."""
    path = tmp_path / "given.txt"
    write_ascii_spectrum(path, _spectrum([0.5, 0.4, 0.3], [0.01, 0.01, 0.01]), bin_indices=[7, 11, 13])

    assert _column(path, 0) == ["7", "11", "13"]


def test_spectrum_bin_indices_are_each_bins_first_input_frame():
    """Pinned for the four argument shapes a run can take. The first-frame rule is what makes the
    column traceable back to the input files; the row index would lose that under any rebin."""
    assert spectrum_bin_indices(6, None) is None, "no rebin: the writer's own row index is the frame index"
    assert spectrum_bin_indices(6, 2) == [0, 2, 4]
    assert spectrum_bin_indices(6, [[0, 2], [2, 4], [4, 6]]) == [0, 2, 4]
    assert spectrum_bin_indices(6, [[0, 2], [4, 6]]) == [0, 4]


def test_gapped_bin_list_leaves_the_index_column_gapped(tmp_path):
    """Frames 2-3 are dropped by the bin list, so they have no row at all and the surviving rows
    keep their input-frame indices 0 and 4. Renumbering them 0 and 1 would silently claim the
    second point came from frame 1."""
    indices = spectrum_bin_indices(6, [[0, 2], [4, 6]])
    assert indices == [0, 4]

    path = tmp_path / "gapped.txt"
    write_ascii_spectrum(path, _spectrum([0.448760, 0.500000], [0.010663, 0.002500]), bin_indices=indices)

    first_column = _column(path, 0)
    assert first_column == ["0", "4"]
    assert "2" not in first_column, "the dropped 2-3 span must not appear as a row"
    assert "3" not in first_column
    assert len(_lines(path)) == 3, "1 header + 2 data rows: no blanked row for the dropped span"


# --------------------------------------------------------------------------------------
# Non-finite values
# --------------------------------------------------------------------------------------


def test_non_finite_values_round_trip_as_nan_and_inf(tmp_path):
    """A NaN transmission is a real result (an open-beam bin with zero counts), so it must survive
    the round trip as NaN rather than becoming 0.0 or an unparsable token."""
    path = tmp_path / "nonfinite.txt"
    write_ascii_spectrum(
        path,
        _spectrum([np.nan, np.inf, -np.inf, 0.500000], [0.500000, 0.250000, 1.000000, np.nan]),
    )

    assert _column(path, 1) == ["nan", "inf", "-inf", "0.500000"]
    assert _column(path, 2) == ["0.500000", "0.250000", "1.000000", "nan"]

    table = np.loadtxt(path, skiprows=1, delimiter=",")
    assert np.isnan(table[0, 1])
    assert table[1, 1] == np.inf
    assert table[2, 1] == -np.inf
    np.testing.assert_allclose(table[3, 1], 0.5, atol=1e-9)
    np.testing.assert_allclose(table[:3, 2], np.array([0.5, 0.25, 1.0]), atol=1e-9)
    assert np.isnan(table[3, 2])


# --------------------------------------------------------------------------------------
# Guards — each rejects before any file appears
# --------------------------------------------------------------------------------------


def test_rejects_a_spectrum_that_is_not_one_dimensional(tmp_path):
    """A three-column format cannot represent an image stack. The message names the dims and sizes,
    because the caller's mistake is having skipped the spatial reduction."""
    path = tmp_path / "stack.txt"
    stack = sc.DataArray(
        sc.array(dims=["tof", "y", "x"], values=np.full((2, 3, 4), 0.5), variances=np.full((2, 3, 4), 0.01), unit="")
    )

    with pytest.raises(ValueError) as excinfo:
        write_ascii_spectrum(path, stack)

    message = str(excinfo.value)
    assert "dims ('tof', 'y', 'x')" in message
    assert "'tof': 2" in message
    assert not path.exists(), "rejected before anything was written"

    scalar = sc.DataArray(sc.scalar(0.5, variance=0.01, unit=""))
    with pytest.raises(ValueError, match="1-D spectrum"):
        write_ascii_spectrum(tmp_path / "scalar.txt", scalar)


def test_rejects_a_spectrum_carrying_no_variances(tmp_path):
    """The third column is a 1-sigma uncertainty. With no variances there is nothing to put in it,
    and a zero or a copy of the value would be a fabricated uncertainty leaving in a data file.

    Both bin counts are checked, and the message is matched rather than just the exception type:
    without this guard a multi-bin spectrum fails downstream in ``np.column_stack`` with an
    unrelated shape ValueError — which a bare ``pytest.raises(ValueError)`` would accept — while a
    single-bin spectrum does not fail at all and writes ``nan`` into the uncertainty column."""
    for n, values in ((3, [0.5, 0.4, 0.3]), (1, [0.5])):
        path = tmp_path / f"novar{n}.txt"
        bare = sc.DataArray(sc.array(dims=["tof"], values=np.asarray(values, dtype=float), unit=""))
        assert bare.variances is None

        with pytest.raises(ValueError, match="carrying variances"):
            write_ascii_spectrum(path, bare)

        assert not path.exists()


def test_rejects_bin_indices_of_the_wrong_length(tmp_path):
    """One index per bin or none at all: a short or long list would misalign every row's provenance
    against its value."""
    spectrum = _spectrum([0.5, 0.4, 0.3], [0.01, 0.01, 0.01])

    for indices in ([0, 1], [0, 1, 2, 3], []):
        path = tmp_path / f"len{len(indices)}.txt"
        with pytest.raises(ValueError, match="one index per spectrum bin"):
            write_ascii_spectrum(path, spectrum, bin_indices=indices)
        assert not path.exists()


def test_rejects_non_integer_bin_indices(tmp_path):
    """The column is a frame index; a float would render through ``%d`` as a silently truncated
    integer."""
    spectrum = _spectrum([0.5, 0.4, 0.3], [0.01, 0.01, 0.01])
    path = tmp_path / "float_indices.txt"

    with pytest.raises(ValueError, match="must be integers"):
        write_ascii_spectrum(path, spectrum, bin_indices=[0.0, 1.5, 2.0])

    assert not path.exists()


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def test_creates_parent_directories_and_overwrites_an_existing_file(tmp_path):
    """A pipeline names an output path under a directory tree that may not exist yet, and a re-run
    must leave the file holding only the new run's rows — a truncating write, not an append."""
    path = tmp_path / "deep" / "nested" / "spectrum.txt"
    assert not path.parent.exists()

    write_ascii_spectrum(path, _spectrum([0.5, 0.4, 0.3], [0.01, 0.01, 0.01]))
    assert len(_lines(path)) == 4

    write_ascii_spectrum(path, _spectrum([0.44876], [0.010663]))
    assert path.read_text() == f"{_REFERENCE_HEADER}\n{_REFERENCE_ROW}\n"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_directory_raises_permission_error(tmp_path):
    """Refused up front with the directory named, rather than part-way through np.savetxt with a
    half-written file left behind."""
    spectrum = _spectrum([0.5], [0.01])

    with pytest.raises(PermissionError):
        write_ascii_spectrum("/spectrum_from_neunorm_test.txt", spectrum)

    with pytest.raises((PermissionError, OSError)):
        write_ascii_spectrum("/nonexistent/deep/path/spectrum.txt", spectrum)

    assert not (tmp_path / "spectrum.txt").exists()


# --------------------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------------------


def test_step_count_predicts_the_events_the_writer_emits(tmp_path):
    """The declared total is what a pipeline adds into its run-wide export count, so an emit site
    that reports more or fewer steps than the count promises leaves a bar that never fills or one
    that overshoots. The step is named before it is counted, so the label on screen is the write
    currently running."""
    spectrum = _spectrum([0.5, 0.4, 0.3], [0.01, 0.01, 0.01])
    assert ascii_spectrum_export_step_count(spectrum) == 1

    events, sink = _collect()
    write_ascii_spectrum(tmp_path / "progress.txt", spectrum, progress=sink)

    assert {e.total for e in events} == {1}
    assert max(e.completed for e in events) == 1, "the bar never reached its declared total"
    assert [(e.completed, e.detail) for e in events] == [(0, "writing ASCII spectrum"), (1, "")]
    assert {e.stage for e in events} == {STAGE_EXPORT}

    other, other_sink = _collect()
    write_ascii_spectrum(tmp_path / "staged.txt", spectrum, progress=other_sink, stage=STAGE_REDUCE_SPECTRUM)
    assert {e.stage for e in other} == {STAGE_REDUCE_SPECTRUM}


def test_file_is_byte_identical_whatever_progress_is(tmp_path):
    """Progress reporting is observation, not participation: the three accepted forms must produce
    the same bytes, or a user watching a bar would get a different file from one who was not."""
    spectrum = _spectrum([0.448760, 0.500000, 0.125000], [0.010663, 0.002500, 0.123456])
    _, sink = _collect()

    quiet = write_ascii_spectrum(tmp_path / "quiet.txt", spectrum, progress=False)
    watched = write_ascii_spectrum(tmp_path / "watched.txt", spectrum, progress=sink)
    with redirect_stderr(io.StringIO()):
        barred = write_ascii_spectrum(tmp_path / "barred.txt", spectrum, progress=True)

    assert filecmp.cmp(quiet, watched, shallow=False)
    assert filecmp.cmp(quiet, barred, shallow=False)
