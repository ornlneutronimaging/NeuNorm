"""Keep ``docs/moving_window.md`` honest: run its examples, and re-measure its three tables.

The page states what a moving window costs, in numbers. Numbers in prose rot — the code changes and
the page keeps claiming what used to be true — so none of them are written by hand. Each table is
recomputed here from a fixed seed **using the real** :func:`~neunorm.processing.moving_window.moving_window`,
parsed back out of the page, and compared. Change the filter and this test fails until the page is
brought back in line; it cannot silently drift.

Each measurement is averaged over several noise realizations from the one fixed seed rather than
taken from a single draw, so the figures estimate the quantity itself rather than one sample of it.
The tolerance is a rounding tolerance, not a fudge factor: the page shows two decimals and the check
allows half of the last digit.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from neunorm.processing.moving_window import moving_window

DOC = Path(__file__).resolve().parents[2] / "docs" / "moving_window.md"

SEED = 20260805
OB_COUNTS = 200.0
KERNELS = (1, 3, 5, 7)
#: Half of the last digit the page prints.
_ROUNDING = 0.005


def _filtered_transmission(sample, ob, k):
    """What the pipeline does: filter both stacks with the same window, then divide."""
    if k == 1:
        return sample / ob
    sample_da = sc.DataArray(sc.array(dims=["y", "x"], values=sample, unit="counts"))
    ob_da = sc.DataArray(sc.array(dims=["y", "x"], values=ob, unit="counts"))
    return moving_window(sample_da, {"x": k, "y": k}).values / moving_window(ob_da, {"x": k, "y": k}).values


def _edge_width(profile, high, low):
    """The 90-to-10 transition width of a falling profile, linearly interpolated."""

    def crossing(level):
        first = int(np.flatnonzero(profile <= level)[0])
        return first - 1 + (profile[first - 1] - level) / (profile[first - 1] - profile[first])

    return crossing(low + 0.1 * (high - low)) - crossing(low + 0.9 * (high - low))


def measure_exchange(realizations=16, n=128):
    """Table 1: per-pixel noise and edge sharpness, for a step edge at 60% transmission."""
    rng = np.random.default_rng(SEED)
    truth = np.broadcast_to(np.where(np.arange(n)[None, :] < n // 2, 1.0, 0.6), (n, n))

    sigmas = {k: [] for k in KERNELS}
    widths = {k: [] for k in KERNELS}
    for _ in range(realizations):
        ob = rng.poisson(OB_COUNTS, size=(n, n)).astype(np.float64)
        sample = rng.poisson(OB_COUNTS * truth).astype(np.float64)
        for k in KERNELS:
            transmission = _filtered_transmission(sample, ob, k)
            # away from both the edge and the frame boundary, so neither is measured by accident
            sigmas[k].append(float(transmission[:, 10 : n // 2 - 10].std(ddof=1)))
            widths[k].append(_edge_width(transmission.mean(axis=0), 1.0, 0.6))

    sigma = {k: float(np.mean(v)) for k, v in sigmas.items()}
    width = {k: float(np.mean(v)) for k, v in widths.items()}
    return {
        k: {
            "sigma": sigma[k],
            "precision": sigma[1] / sigma[k],
            "width": width[k],
            "resolution": width[k] / width[1],
        }
        for k in KERNELS
    }


def measure_contrast(realizations=32, n=96, depth=0.5):
    """Table 2: the depth a fit over the feature would recover, as a fraction of the true depth.

    The mean over the feature's own footprint, which is what a fit sees — not the centre pixel, which
    for a feature exactly the size of the kernel is undiluted by construction and reads 1.00 however
    much the surrounding image has been smoothed.
    """
    rng = np.random.default_rng(SEED)
    recovered = {(k, f): [] for k in KERNELS for f in (3, 9)}
    for _ in range(realizations):
        for feature in (3, 9):
            truth = np.ones((n, n))
            low = n // 2 - feature // 2
            inside = (slice(low, low + feature), slice(low, low + feature))
            truth[inside] = 1.0 - depth
            ob = rng.poisson(OB_COUNTS, size=(n, n)).astype(np.float64)
            sample = rng.poisson(OB_COUNTS * truth).astype(np.float64)
            for k in KERNELS:
                transmission = _filtered_transmission(sample, ob, k)
                background = float(np.median(transmission[5:25, 5:25]))
                recovered[(k, feature)].append((background - transmission[inside].mean()) / depth)
    return {key: float(np.mean(v)) for key, v in recovered.items()}


def measure_correlation(n=256):
    """Table 3: the reported per-pixel sigma, and how much a pixel now shares with its neighbour.

    Measured over the INTERIOR, outside the ``k // 2`` border. The border is a boundary effect of its
    own — a mirrored edge makes the window read some pixels more than once, which correctly raises
    the variance there — and averaging it in would blend two separate stories into one column. What
    this table is for is the interior trade, where sigma falls as ``1 / k``.
    """
    rng = np.random.default_rng(SEED)
    counts = rng.poisson(OB_COUNTS, size=(n, n)).astype(np.float64)
    out = {}
    for k in (1, 3, 5):
        data = sc.DataArray(sc.array(dims=["y", "x"], values=counts.copy(), variances=counts.copy(), unit="counts"))
        filtered = moving_window(data, {"x": k, "y": k}) if k > 1 else data
        edge = k // 2
        inner = slice(edge, n - edge) if edge else slice(None)
        values = filtered.values[inner, inner]
        out[k] = {
            "sigma": float(np.sqrt(filtered.variances[inner, inner]).mean()),
            "correlation": float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1]),
        }
    return out


# --------------------------------------------------------------------------------------------
# The measurements say what the page says
# --------------------------------------------------------------------------------------------


def _table_rows(heading):
    """The data rows of the markdown table under ``heading``, as lists of cell strings."""
    text = DOC.read_text()
    start = text.index(heading)
    rows = []
    for line in text[start:].splitlines():
        if line.startswith("|") and "---" not in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            rows.append(cells)
        elif rows and not line.startswith("|"):
            break
    return rows[1:]  # drop the header row


def _number(cell):
    """The number in a table cell, whatever unit or sign decoration the page prints around it."""
    # "px" before "x", or the unit's own x is eaten and "0.8 px" becomes "0.8 p"
    return float(cell.replace("px", "").replace("x", "").replace("+", "").replace("−", "-").strip())


def test_the_exchange_table_matches_the_measurement():
    measured = measure_exchange()
    rows = _table_rows("## What it costs")
    assert [row[0] for row in rows] == [f"{k}x{k}" for k in KERNELS]
    for row, k in zip(rows, KERNELS):
        assert abs(_number(row[1]) - measured[k]["sigma"]) < _ROUNDING / 10, f"{k}: noise sigma"
        assert abs(_number(row[2]) - measured[k]["precision"]) < _ROUNDING, f"{k}: precision gain"
        assert abs(_number(row[3]) - measured[k]["width"]) < 0.05, f"{k}: edge width"
        assert abs(_number(row[4]) - measured[k]["resolution"]) < _ROUNDING, f"{k}: resolution cost"


def test_the_contrast_table_matches_the_measurement():
    measured = measure_contrast()
    rows = _table_rows("### A feature smaller than the window")
    assert [row[0] for row in rows] == [f"{k}x{k}" for k in KERNELS]
    for row, k in zip(rows, KERNELS):
        assert abs(_number(row[1]) - measured[(k, 3)]) < _ROUNDING, f"{k}: 3 px feature"
        assert abs(_number(row[2]) - measured[(k, 9)]) < _ROUNDING, f"{k}: 9 px feature"


def test_the_correlation_table_matches_the_measurement():
    measured = measure_correlation()
    rows = _table_rows("### Neighbouring pixels are no longer independent")
    assert [row[0] for row in rows] == ["1x1", "3x3", "5x5"]
    for row, k in zip(rows, (1, 3, 5)):
        assert abs(_number(row[1]) - measured[k]["sigma"]) < _ROUNDING, f"{k}: per-pixel sigma"
        assert abs(_number(row[2]) - measured[k]["correlation"]) < _ROUNDING, f"{k}: neighbour correlation"


# --------------------------------------------------------------------------------------------
# The measurements are measuring the right thing
# --------------------------------------------------------------------------------------------


def test_the_warning_quotes_the_figure_the_table_measures():
    """The run-time warning and the page must not be able to drift apart.

    The warning carries a measured contrast figure as a string literal. Asserting that literal in the
    guards test only checks a literal against itself; this reads the number out of the page's table —
    which is itself regenerated from the code above — and requires the warning to quote the same one.
    Re-measure the table and this fails until the warning is brought along.
    """
    import inspect

    from neunorm.pipelines import _tof_spine

    rows = _table_rows("### A feature smaller than the window")
    five_by_five = next(row for row in rows if row[0] == "5x5")
    warning_source = inspect.getsource(_tof_spine._warn_on_the_moving_window_trade)
    assert f"retains {five_by_five[1]} of its true contrast under a 5x5 window" in warning_source


def test_the_neighbour_correlation_matches_its_closed_form():
    """A ``k x k`` box shares ``k-1`` of its ``k`` columns with the window one pixel over."""
    measured = measure_correlation()
    for k in (3, 5):
        np.testing.assert_allclose(measured[k]["correlation"], (k - 1) / k, atol=0.02)


def test_the_precision_gain_is_the_kernel_width():
    """Averaging ``k**2`` independent pixels divides the standard deviation by ``k``."""
    measured = measure_exchange()
    for k in KERNELS:
        np.testing.assert_allclose(measured[k]["precision"], k, rtol=0.05)


def test_a_feature_far_larger_than_the_kernel_keeps_its_depth():
    """The contrast measurement must not report loss where there is none."""
    measured = measure_contrast()
    np.testing.assert_allclose(measured[(1, 3)], 1.0, atol=0.05)
    np.testing.assert_allclose(measured[(1, 9)], 1.0, atol=0.05)


# --------------------------------------------------------------------------------------------
# The page's examples run
# --------------------------------------------------------------------------------------------


def _python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_the_page_has_examples_to_run():
    assert _python_blocks(DOC.read_text()), "no python blocks found — the extraction regex is stale"


@pytest.mark.parametrize("index", range(len(_python_blocks(DOC.read_text()))))
def test_every_python_example_runs(index, tmp_path, monkeypatch):
    """A page whose examples do not run is worse than no page: the reader cannot tell what is wrong.

    Each block runs in a fresh namespace seeded only with the names the page tells the reader to
    supply — the four VENUS TPX1 path arguments — against real synthetic inputs, with the working
    directory in a temporary folder so ``output_path="transmission.hdf5"`` lands there. The block text
    is exactly what the page shows.
    """
    monkeypatch.chdir(tmp_path)
    block = _python_blocks(DOC.read_text())[index]
    compile(block, f"docs/moving_window.md[block {index}]", "exec")
    exec(block, dict(_tpx1_example_inputs(tmp_path)))  # noqa: S102 - executing the documentation is the point


def _tpx1_example_inputs(tmp_path):
    """The four path arguments the page's pipeline example asks the reader to provide."""
    from test_progress_pipelines import _ccd_tiffs, _spectra_file, _venus_metadata_nexus

    frames, sample_dir, ob_dir = 4, tmp_path / "sample", tmp_path / "ob"
    sample_dir.mkdir(exist_ok=True)
    ob_dir.mkdir(exist_ok=True)
    edges = [round(0.1 * (i + 1), 1) for i in range(frames)]
    _spectra_file(sample_dir / "sample_Spectra.txt", edges)
    _spectra_file(ob_dir / "ob_Spectra.txt", edges)
    return {
        "sample_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "s.h5", 12345.0, das_image_path=b"auto")],
        "ob_hdf5_paths": [_venus_metadata_nexus(tmp_path / "nx" / "o.h5", 24690.0, das_image_path=b"auto")],
        "sample_tiff_paths": [_ccd_tiffs(sample_dir, "s", frames, 81.0, proton_charge=12345.0)],
        "ob_tiff_paths": [_ccd_tiffs(ob_dir, "o", frames, 99.0, proton_charge=24690.0)],
    }
