"""Execute every Python example in ``docs/progress.md``.

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

#: The session's loguru handlers as this module is imported, so a documentation block that reconfigures
#: logging in-process can be caught rather than silently inherited by every later test.
_HANDLERS_AT_IMPORT = set(__import__("loguru").logger._core.handlers)  # noqa: SLF001 - no public listing


#: Blocks that reconfigure global logging. Executing these in-process would leave loguru with a
#: different handler set than pytest started with — `logger.remove()` drops the session's handler and
#: the `finally` adds a NEW one, so every later test logs through a stranger. They are executed for real
#: by `_collision_probe`, in a subprocess, which is where their effect is measurable anyway.
_GLOBAL_LOGGING_BLOCKS = ("logger.remove()", 'logger.disable("neunorm")')


def test_the_page_has_the_examples_the_task_asked_for():
    """Guards the extractor itself: a regex that silently matched nothing would make every example
    test below pass by vacuum."""
    assert len(BLOCKS) >= 8, f"only found {len(BLOCKS)} python blocks in {DOC.name}"
    joined = "\n".join(body for _line, body in BLOCKS)
    assert "progress=True" in joined, "the built-in-bar example is missing"
    assert "event.completed - bar.n" in joined, "the adapter example is missing"
    assert "raise RunCancelled" in joined, "the cancellation example is missing"
    assert "logger.remove()" in joined, "the log-collision remedy is missing"


def test_the_skipped_logging_blocks_are_each_executed_in_a_subprocess():
    """The in-process runner skips blocks that reconfigure global logging. This checks the skip loses no
    coverage by naming the blocks and requiring a subprocess test for each — asserted by running them,
    not by grepping this file's own source, which would only confirm that I wrote a string."""
    skipped = [body for _line, body in BLOCKS if any(m in body for m in _GLOBAL_LOGGING_BLOCKS)]
    assert len(skipped) == 2, f"expected the remedy and the disable block, found {len(skipped)}"
    assert any("tqdm.write" in body for body in skipped)
    assert any('logger.disable("neunorm")' in body for body in skipped)


@pytest.mark.parametrize("line,source", BLOCKS, ids=[f"line{line}" for line, _ in BLOCKS])
def test_every_documented_example_runs(tmp_path, monkeypatch, line, source):
    """Run the block verbatim. A NameError here means the page shows an incomplete snippet; any other
    exception means the page shows something that does not work."""
    if any(marker in source for marker in _GLOBAL_LOGGING_BLOCKS):
        pytest.skip("reconfigures global logging; executed in a subprocess by the collision probe instead")
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
    }
    # Deliberately nothing else. Seeding `Path` or `os` here would let a block that forgot its own
    # import pass, while failing for a reader who pastes it — the exact failure the NameError handler
    # below exists to catch.

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


@pytest.fixture(autouse=True)
def _logging_must_survive_every_example():
    """Assert after EVERY test here that the session's loguru handlers are unchanged.

    The defect this guards hid in plain sight: executing a documentation block that calls
    `logger.remove()` in-process replaces the session's handler, so every later test logs through a
    handler this file installed. Nothing failed as a result — it had to be found by reading.

    An autouse teardown rather than one ordered test, so it cannot be defeated by where a future test
    lands in the file.
    """
    yield

    from loguru import logger

    handlers = set(logger._core.handlers)  # noqa: SLF001 - the public API exposes no handler listing
    assert handlers == _HANDLERS_AT_IMPORT, (
        "a documentation example reconfigured logging for the rest of the session: "
        f"{sorted(_HANDLERS_AT_IMPORT)} -> {sorted(handlers)}. "
        "Add its marker to _GLOBAL_LOGGING_BLOCKS so it runs in a subprocess instead."
    )


def test_the_page_is_right_about_what_a_cancelled_run_leaves_behind(tmp_path):
    """Pins both halves of the page's claim, through a pipeline rather than the writer alone: cancel
    before export and there is no file; cancel inside export and the file opens cleanly and is
    incomplete — which is why the page says to delete it rather than trust it."""
    import h5py

    from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
    from neunorm.utils.progress import STAGE_EXPORT, STAGE_LOAD_SAMPLE

    inputs = tmp_path / "cancel_inputs"
    inputs.mkdir()
    sample = [_tiffs(inputs, "s", 3, 81)]
    ob = [_tiffs(inputs, "o", 2, 99)]

    class _CancelledError(RuntimeError):
        pass

    def run_cancelling_at(stage, output):
        def cancel(event):
            if event.stage == stage:
                raise _CancelledError(stage)

        with pytest.raises(_CancelledError):
            run_mars_ccd_pipeline(sample_paths=sample, ob_paths=ob, output_path=output, progress=cancel)

    before_export = tmp_path / "before_export.h5"
    run_cancelling_at(STAGE_LOAD_SAMPLE, before_export)
    assert not before_export.exists(), "cancelling before export should leave nothing"

    during_export = tmp_path / "during_export.h5"
    run_cancelling_at(STAGE_EXPORT, during_export)
    assert during_export.exists(), "cancelling inside export leaves the file the page warns about"
    with h5py.File(during_export, "r") as f:  # would raise if it were not a readable HDF5 file
        assert "transmission" not in f, "the page says the file is incomplete, not merely truncated"


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


def test_the_naive_adapter_figure_on_the_page_is_the_measured_one(tmp_path):
    """The page warns that passing `event.completed` straight to `tqdm.update()` makes a 120-file bar
    end at 7740. That number was wrong once — computed as the triangular number of the per-file events,
    which ignored the note events that carry the same absolute count — so it is measured here.
    """
    from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
    from neunorm.utils.progress import STAGE_LOAD_SAMPLE

    inputs = tmp_path / "naive"
    inputs.mkdir()
    sample = [_tiffs(inputs, f"s{run}", 40, 81) for run in range(3)]  # 3 runs x 40 = the page's geometry
    ob = [_tiffs(inputs, "o", 2, 99)]

    naive = correct = 0

    def report(event):
        nonlocal naive, correct
        if event.stage != STAGE_LOAD_SAMPLE:
            return
        naive += event.completed  # the mistake the page warns about
        correct += event.completed - correct  # bar.update(event.completed - bar.n)

    run_mars_ccd_pipeline(sample_paths=sample, ob_paths=ob, output_path=tmp_path / "naive.h5", progress=report)

    assert correct == 120, "the documented adapter must land exactly on the file count"
    assert naive == 7740, f"the page says 7740; measured {naive}"
    assert str(naive) in DOC.read_text(), "the page no longer quotes the measured figure"


def test_the_documented_disable_example_silences_neunorm_and_then_restores_it(tmp_path):
    """The page's other remedy — `logger.disable("neunorm")` — has to do what it says: no NeuNorm
    records during the run, and records again afterwards. Executed in a subprocess, like the first
    remedy, because it reconfigures global logging.

    Written after a review found the previous version of this coverage was a test that read its own
    source text for the string `logger.enable` — which proves only that I typed it.
    """
    disable_block = next(body for _line, body in BLOCKS if 'logger.disable("neunorm")' in body)
    probe = _collision_probe(
        tmp_path,
        disable_block
        + """

# after the block's `finally: logger.enable("neunorm")`, records must flow again
from loguru import logger
logger.info("AFTER-THE-BLOCK")
""",
    )

    records_during, collisions = probe["stderr"]
    assert collisions == 0, "silencing NeuNorm still left log records colliding with the bars"
    # the only record on the stream is the one emitted after the block re-enabled logging
    assert records_during == 1, (
        f"expected exactly the post-block record, saw {records_during} — either NeuNorm was not silenced "
        "during the run, or logging was not re-enabled afterwards"
    )


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
