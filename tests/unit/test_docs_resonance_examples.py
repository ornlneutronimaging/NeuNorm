"""Execute every Python example in ``docs/resonance_mode.md``.

Documentation that does not run is worse than no documentation: a user pastes it, it fails, and they
cannot tell whether the library or the page is wrong. So the examples are extracted from the page and
executed here rather than read — the same arrangement ``test_docs_progress_examples.py`` uses for the
progress page.

Each block runs in a fresh namespace seeded only with the names the page tells the reader to supply —
the four VENUS TPX1 path arguments, and the two histograms the detector example takes — against
synthetic TPX1 inputs, with the working directory in a temporary folder so ``output_path="spectrum.txt"``
lands there. The block text itself is exactly what the page shows.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import scipp as sc
from PIL import Image

DOC = Path(__file__).resolve().parents[2] / "docs" / "resonance_mode.md"
_DETECTOR = 32
_FRAMES = 6


def _tpx1_tiffs(directory, prefix, count, value):
    """TIFF frames carrying the EXIF metadata run-combining and the VENUS proton charge need."""
    paths = []
    for index in range(count):
        image = Image.fromarray(np.full((_DETECTOR, _DETECTOR), float(value + index), dtype=np.float32))
        exif = image.getexif()
        exif[65027] = "ExposureTime:30.000000"
        exif[65022] = f"RunNo:{1000 + index}"
        exif[65025] = "ManufacturerStr:DW936_BV"
        exif[65024] = "IntegratedPCharge:1000.0"
        path = directory / f"{prefix}_{index:05}.tiff"
        image.save(path, exif=exif)
        paths.append(path)
    return paths


def _nexus(path, proton_charge):
    """The NeXus metadata file TPX1 reads its proton charge and image directory from."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.create_dataset("proton_charge", data=[proton_charge])
        entry.create_dataset("duration", data=[60.0])
        logs = entry.create_group("DASlogs")
        logs.create_group("BL10:Det:T1:TSStart_RBV").create_dataset("value", data=[100])
        logs.create_group("BL10:Det:T1:TSBinSize_RBV").create_dataset("value", data=[5])
        logs.create_group("BL10:Det:T1:TSSize_RBV").create_dataset("value", data=[_FRAMES])
        logs.create_group("BL10:Det:TH:DSPT1:TIDelay").create_dataset("average_value", data=[5000])
        logs.create_group("BL10:Exp:Det").create_dataset("value_strings", data=[[b"MCP TPX1"]])
        logs.create_group("BL10:Exp:IM:ImageFilePath").create_dataset("value", data=[[b"autoreduce"]])
    return path


def _spectra_file(path, left_edges):
    """The co-located ``*_Spectra.txt`` sidecar TPX1 reads its TOF axis from."""
    with open(path, "w") as handle:
        handle.write("shutter_time,counts\n")
        for edge in left_edges:
            handle.write(f"{edge},1000\n")
    return path


def _energy_histogram(scale, *, dips=()):
    """A small (energy, x, y) histogram for the ``detect_resonances`` example."""
    n_energy = 200
    edges = np.geomspace(1.0, 100.0, n_energy + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    values = np.full((n_energy, _DETECTOR, _DETECTOR), float(scale))
    for energy in dips:
        values *= (1.0 - 0.4 * np.exp(-0.5 * ((centers - energy) / (0.02 * energy)) ** 2))[:, None, None]
    return sc.DataArray(
        sc.array(dims=["energy", "x", "y"], values=values, variances=values.copy(), unit="counts"),
        coords={"energy": sc.array(dims=["energy"], values=edges, unit="eV")},
    )


def _python_blocks(text):
    """Every ```python fenced block, in page order, with the line it starts on."""
    blocks = []
    for match in re.finditer(r"^```python\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        line = text[: match.start()].count("\n") + 1
        blocks.append((line, match.group(1)))
    return blocks


BLOCKS = _python_blocks(DOC.read_text())


def test_the_page_has_the_examples_it_should():
    """Guards the extractor itself: a regex that silently matched nothing would make every example test
    below pass by vacuum, and would keep passing if the page were emptied."""
    assert len(BLOCKS) >= 5, f"only found {len(BLOCKS)} python blocks in {DOC.name}"
    joined = "\n".join(body for _line, body in BLOCKS)
    assert "spectrum_roi=" in joined, "the basic spectrum-mode example is missing"
    assert 'np.loadtxt("spectrum.txt", skiprows=1, delimiter=",")' in joined, (
        "the reader example is missing, and it is the interoperability contract the format exists for"
    )
    assert "rebin_by_tof" in joined, "the frame-binning example is missing"
    assert "normalize_roi_spectrum" in joined, "the without-a-pipeline example is missing"
    assert "detect_resonances" in joined, "the resonance-detector example is missing"


def test_the_page_documents_the_ascii_format_exactly():
    """The three-column format is a cross-tool contract, so the page must show it verbatim — including
    the header line and the six-decimal rendering a downstream parser will meet."""
    text = DOC.read_text()
    assert "bin_index,transmission,uncertainty" in text
    assert "0,0.448760,0.010663" in text
    from neunorm.exporters.ascii_writer import ASCII_SPECTRUM_HEADER

    assert ASCII_SPECTRUM_HEADER in text, "the page's header line has drifted from the writer's"


def test_the_page_says_what_the_mode_does_not_do():
    """The progress page sets the precedent: stating the boundaries is what stops a user assuming a
    capability that is not there. Named explicitly so the section cannot be quietly dropped."""
    text = DOC.read_text()
    assert "## What resonance mode does not do" in text
    for expected in ("CCD pipelines", "shared-dark", "fit"):
        assert expected in text, f"the boundaries section no longer mentions {expected!r}"


@pytest.fixture
def documented_namespace(tmp_path, monkeypatch):
    """The names the page tells the reader to supply, and nothing else.

    Deliberately not seeding ``Path``, ``np`` or ``sc``: a block that forgot its own import would pass
    here while failing for a reader who pastes it, which is the failure this file exists to catch.
    """
    sample_dir = tmp_path / "sample"
    ob_dir = tmp_path / "ob"
    sample_dir.mkdir()
    ob_dir.mkdir()
    sample_tiffs = _tpx1_tiffs(sample_dir, "sample", _FRAMES, 81)
    ob_tiffs = _tpx1_tiffs(ob_dir, "ob", _FRAMES, 99)
    left_edges = [round(0.1 * (i + 1), 1) for i in range(_FRAMES)]
    _spectra_file(sample_dir / "sample_Spectra.txt", left_edges)
    _spectra_file(ob_dir / "ob_Spectra.txt", left_edges)

    monkeypatch.chdir(tmp_path)

    # The reader example reads spectrum.txt, which the FIRST example writes. Blocks each get a fresh
    # namespace and a fresh directory, so seed a real one — written by the real writer, not hand-typed
    # — and let the pipeline example overwrite it.
    from neunorm.exporters.ascii_writer import write_ascii_spectrum

    seeded = sc.DataArray(
        sc.array(
            dims=["tof"],
            values=np.array([0.44876, 0.451203, 0.449881]),
            variances=np.array([0.010663, 0.010701, 0.010684]) ** 2,
        )
    )
    write_ascii_spectrum(tmp_path / "spectrum.txt", seeded)

    return {
        "__name__": "docs_example",
        "sample_hdf5_paths": [_nexus(tmp_path / "nexus" / "s.nxs.h5", 12345)],
        "ob_hdf5_paths": [_nexus(tmp_path / "nexus" / "o.nxs.h5", 24690)],
        "sample_tiff_paths": [sample_tiffs],
        "ob_tiff_paths": [ob_tiffs],
        "hist_sample": _energy_histogram(1000.0, dips=(5.0, 20.0, 50.0)),
        "hist_ob": _energy_histogram(1000.0),
    }


@pytest.mark.parametrize("line,source", BLOCKS, ids=[f"line{line}" for line, _ in BLOCKS])
def test_every_documented_example_runs(documented_namespace, line, source):
    """Run the block verbatim. A NameError means the page shows an incomplete snippet; any other
    exception means the page shows something that does not work."""
    try:
        exec(compile(source, f"{DOC.name}:{line}", "exec"), documented_namespace)  # noqa: S102
    except NameError as exc:
        pytest.fail(f"{DOC.name}:{line} references a name the page never defines: {exc}")


def test_the_documented_pipeline_example_writes_the_file_the_page_describes(documented_namespace):
    """The first example claims a three-column ASCII file plus an HDF5 sibling. Run that block and check
    both, so the page's central promise is verified rather than asserted."""
    line, source = next((ln, src) for ln, src in BLOCKS if "spectrum.txt" in src and "run_venus_tpx1" in src)
    exec(compile(source, f"{DOC.name}:{line}", "exec"), documented_namespace)  # noqa: S102

    written = Path("spectrum.txt")
    assert written.exists()
    lines = written.read_text().splitlines()
    assert lines[0] == "bin_index,transmission,uncertainty"
    assert len(lines) == _FRAMES + 1, "one header line plus one row per frame"

    table = np.loadtxt(written, skiprows=1, delimiter=",")
    np.testing.assert_array_equal(table[:, 0].astype(int), np.arange(_FRAMES))

    # Uniform synthetic frames make the expected transmission exact arithmetic, so this pins the
    # NUMBER rather than merely the file's existence. Frame i holds 81+i counts against 99+i, and the
    # proton charges are 12345 and 24690 -- combine_runs normalizes per run, so the ratio is
    # ((81+i)/12345) / ((99+i)/24690) = 2*(81+i)/(99+i).
    expected = 2.0 * (81.0 + np.arange(_FRAMES)) / (99.0 + np.arange(_FRAMES))
    np.testing.assert_allclose(table[:, 1], expected, rtol=1e-5)

    assert Path("spectrum.hdf5").exists(), "the page promises an HDF5 file alongside the ASCII one"
    import h5py

    with h5py.File("spectrum.hdf5") as f:
        for expected_dataset in ("transmission", "uncertainty", "tof", "spectra_tof"):
            assert expected_dataset in f, f"/{expected_dataset} missing from the HDF5 sibling"
        assert "metadata/spectrum_roi" in f


def test_the_documented_binning_example_gives_the_bin_index_column_the_page_shows(documented_namespace):
    """The page states rebin_by_tof=2 over six frames yields rows indexed 0, 2, 4. Run it and read the
    column back out of the file, because that claim is what a user will check first."""
    line, source = next((ln, src) for ln, src in BLOCKS if "rebin_by_tof=2" in src)
    exec(compile(source, f"{DOC.name}:{line}", "exec"), documented_namespace)  # noqa: S102

    table = np.loadtxt(Path("binned.txt"), skiprows=1, delimiter=",")
    np.testing.assert_array_equal(table[:, 0].astype(int), np.array([0, 2, 4]))
