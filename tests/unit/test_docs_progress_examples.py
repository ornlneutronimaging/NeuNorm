"""Execute every Python example in ``docs/progress.md`` (#195, Task 8).

Documentation that does not run is worse than no documentation: a user pastes it, it fails, and they
cannot tell whether the library or the page is wrong. So the examples are extracted from the page and
executed here rather than read.

Each block runs in a fresh namespace seeded with the names the page tells the reader to supply —
``sample_paths``, ``ob_paths``, ``dark_paths`` and the ``user_pressed_stop`` hook — against synthetic
MARS CCD TIFFs, with the working directory in a temporary folder so ``output_path="normalized.h5"``
lands there. The block text itself is exactly what the page shows.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

DOC = Path(__file__).resolve().parents[2] / "docs" / "progress.md"
_DETECTOR = 32


def _tiffs(directory, prefix, count, value):
    """CCD-style TIFFs carrying the EXIF metadata run-combining matches on."""
    paths = []
    for index in range(count):
        image = Image.fromarray(np.full((_DETECTOR, _DETECTOR), float(value + index), dtype=np.float32))
        exif = image.getexif()
        exif[65027] = "ExposureTime:30.000000"
        exif[65022] = f"RunNo:{1000 + index}"
        exif[65025] = "ManufacturerStr:DW936_BV"
        exif[65052] = "MotSlitVB.RBV:42.3"
        exif[65054] = "MotSlitVT.RBV:42.8"
        exif[65056] = "MotSlitHR.RBV:41.4"
        exif[65058] = "MotSlitHL.RBV:42.4"
        path = directory / f"{prefix}_{index:05}.tiff"
        image.save(path, exif=exif)
        paths.append(path)
    return paths


def _python_blocks(text):
    """Every ```python fenced block, in page order, with the line it starts on."""
    blocks = []
    for match in re.finditer(r"^```python\n(.*?)^```", text, re.MULTILINE | re.DOTALL):
        line = text[: match.start()].count("\n") + 1
        blocks.append((line, match.group(1)))
    return blocks


BLOCKS = _python_blocks(DOC.read_text())


def test_the_page_has_the_examples_the_task_asked_for():
    """Guards the extractor itself: a regex that silently matched nothing would make every example
    test below pass by vacuum."""
    assert len(BLOCKS) >= 8, f"only found {len(BLOCKS)} python blocks in {DOC.name}"
    joined = "\n".join(body for _line, body in BLOCKS)
    assert "progress=True" in joined, "the built-in-bar example is missing"
    assert "event.completed - bar.n" in joined, "the adapter example is missing"
    assert "raise RunCancelled" in joined, "the cancellation example is missing"
    assert "logger.remove()" in joined, "the log-collision remedy is missing"


@pytest.mark.parametrize("line,source", BLOCKS, ids=[f"line{line}" for line, _ in BLOCKS])
def test_every_documented_example_runs(tmp_path, monkeypatch, line, source):
    """Run the block verbatim. A NameError here means the page shows an incomplete snippet; any other
    exception means the page shows something that does not work."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.chdir(tmp_path)
    # Deliberately NOT setting TQDM_DISABLE: tqdm captures that variable when `tqdm.std` is first
    # imported, so setting it here disabled the bars for every rendered-output test that ran later in
    # the same process — 21 of them, all passing individually and failing together.

    namespace = {
        "__name__": "docs_example",
        # the three names every pipeline example tells the reader to supply
        "sample_paths": [_tiffs(inputs, "sample", 3, 81)],
        "ob_paths": [_tiffs(inputs, "ob", 2, 99)],
        "dark_paths": [_tiffs(inputs, "dark", 2, 5)],
        # the cancellation example's own hook; False so the block runs to completion here
        "user_pressed_stop": lambda: False,
        "Path": Path,
        "os": os,
    }

    try:
        exec(compile(source, f"{DOC.name}:{line}", "exec"), namespace)  # noqa: S102 - executing the docs is the point
    except NameError as exc:
        pytest.fail(f"{DOC.name}:{line} references a name the page never defines: {exc}")


def test_the_documented_pattern_for_your_own_function_behaves_as_described():
    """The page shows a `my_loader` shape for instrumenting your own code. Executing that block only
    proves it parses — nothing calls it. So call it, and check the part that IS observable here: one
    event per item, naming the item, under one run-wide total.

    Deliberately does NOT claim to verify the "count after the work" advice in the same example: the
    example's work is an ellipsis, so counting before or after produces identical events. I checked —
    swapping the two lines in the page leaves this test passing. Emit ordering is pinned where it is
    real, against the instrumented functions themselves
    (`test_progress_export.py::test_hdf5_names_each_step_before_it_runs`).
    """
    block = next(body for _line, body in BLOCKS if "def my_loader" in body)
    namespace = {}
    exec(compile(block, "my_loader", "exec"), namespace)  # noqa: S102 - executing the docs is the point

    events = []
    paths = [Path(f"frame_{index}.tif") for index in range(4)]
    namespace["my_loader"](paths, progress=events.append)

    assert [e.completed for e in events] == [1, 2, 3, 4], "not one event per item, counted after the work"
    assert [e.detail for e in events] == [p.name for p in paths], "the item is not named"
    assert {e.total for e in events} == {4}
    assert {e.stage for e in events} == {"load_sample"}


def _collision_probe(tmp_path, body):
    """Run a pipeline in a FRESH interpreter and count stderr lines holding both a bar and a log record.

    A subprocess is required, not a nicety: `contextlib.redirect_stderr` rebinds `sys.stderr`, which
    tqdm resolves at construction but loguru's default handler does not, so in-process capture shows the
    bars and hides the very records that collide with them. That is why this went unnoticed until it was
    measured this way.
    """
    import subprocess
    import sys
    import textwrap

    inputs = tmp_path / "probe"
    inputs.mkdir(exist_ok=True)
    sample = _tiffs(inputs, "s", 3, 81)
    ob = _tiffs(inputs, "o", 2, 99)
    preamble = f"""
        from pathlib import Path
        from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
        sample_paths = [[Path(p) for p in {[str(p) for p in sample]!r}]]
        ob_paths = [[Path(p) for p in {[str(p) for p in ob]!r}]]
        output = Path({str(tmp_path / "probe.h5")!r})
    """
    script = textwrap.dedent(preamble) + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "TQDM_DISABLE"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    def scan(text):
        lines = text.replace("\r", "\n").split("\n")
        records = [ln for ln in lines if " | INFO " in ln or " | SUCCESS " in ln]
        collisions = [ln for ln in lines if ("INFO" in ln or "SUCCESS" in ln) and ("%|" in ln or "item/s" in ln)]
        return len(records), len(collisions)

    return {"stderr": scan(proc.stderr), "stdout": scan(proc.stdout)}


def test_the_documented_log_remedy_removes_the_collisions_and_keeps_the_records(tmp_path):
    """The page's remedy has to do both things: stop garbling the bars AND keep the log records visible,
    on stderr where they were. Pinned because a remedy that silently relocates the log is worse than the
    collision it fixes — which is exactly what happens if `file=sys.stderr` is omitted."""
    baseline = _collision_probe(
        tmp_path,
        """
        run_mars_ccd_pipeline(sample_paths=sample_paths, ob_paths=ob_paths, output_path=output, progress=True)
        """,
    )
    # The remedy under test is the page's own block, extracted verbatim — not a copy kept in this file,
    # which could agree with itself while the documentation drifted.
    remedy = next(body for _line, body in BLOCKS if "tqdm.write" in body)
    remedied = _collision_probe(tmp_path, remedy)

    baseline_records, baseline_collisions = baseline["stderr"]
    remedied_records, remedied_collisions = remedied["stderr"]

    assert baseline_collisions > 0, "the collision the page describes did not happen, so nothing was fixed"
    assert remedied_collisions == 0, f"the documented remedy still leaves {remedied_collisions} collisions"
    assert remedied_records >= baseline_records, (
        f"the remedy lost log records: {remedied_records} on stderr vs {baseline_records} without it"
    )
    assert remedied["stdout"] == (0, 0), "the remedy moved log records to stdout — `file=sys.stderr` missing?"


def test_the_cancellation_example_actually_cancels(tmp_path, monkeypatch):
    """The cancellation block runs to completion above, because the stop hook returns False. Run it
    again with the hook returning True, so the example is shown to do what the page says it does."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.chdir(tmp_path)

    source = next(body for _line, body in BLOCKS if "raise RunCancelled" in body)
    printed = []
    namespace = {
        "__name__": "docs_example",
        "sample_paths": [_tiffs(inputs, "sample", 3, 81)],
        "ob_paths": [_tiffs(inputs, "ob", 2, 99)],
        "user_pressed_stop": lambda: True,
        "print": printed.append,
    }

    exec(compile(source, "cancellation", "exec"), namespace)  # noqa: S102 - executing the docs is the point

    assert printed, "the example neither raised nor printed"
    assert "cancelled during" in str(printed[0]), printed
    assert not (tmp_path / "normalized.h5").exists(), "a cancelled run left an output file"
