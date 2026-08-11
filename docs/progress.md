# Progress reporting

Normalizing a large stack takes minutes to hours, and without feedback there is no way to tell a slow
run from a hung one. All six pipelines take a `progress` argument, as do the functions that do the
expensive work inside them — the image and event loaders, the event converters, the gamma filter, both
normalizers, and both writers:

```python
from pathlib import Path

from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

transmission = run_mars_ccd_pipeline(
    sample_paths=sample_paths,
    ob_paths=ob_paths,
    dark_paths=dark_paths,
    output_path=Path("normalized.h5"),
    progress=True,
)
```

`progress` accepts three things:

| Value | Behaviour |
|---|---|
| `False` (default) | nothing is reported; nothing is allocated and `tqdm` is never imported |
| `True` | NeuNorm draws one [`tqdm`](https://tqdm.github.io/) bar per stage |
| a callable | it receives a {py:class}`~neunorm.utils.progress.ProgressEvent` for every item or step |

```{note}
`output_path` must be a `pathlib.Path`, not a string — the pipelines choose HDF5 or TIFF from
`output_path.suffix`. Every example below wraps it for that reason.
```

The callable is the actual contract — `progress=True` is a convenience built on it. A library that owns
the bar breaks headless runs, log files and any caller who wants their own display, so NeuNorm reports
events and lets you decide what to draw.

## What it tells you, and what it does not

You get **movement and a per-stage rate**. You do **not** get a whole-run ETA, and NeuNorm does not
pretend to: the stages differ in cost by 60–120×, and which stages run at all depends on the arguments
(an optional dark, a gamma filter, a TOF rebin, an air-region correction, HDF5 versus per-image TIFF).
"Stage 4 of 9" would be both nonlinear and unknowable at entry, so each stage reports its own progress
and the bars appear as their work starts.

Within a per-file stage the rate does mean something — `835 item/s` while reading frames of the same
size is a real throughput, and a drop in it is worth noticing. For the stages that report *named steps*
rather than items (the gamma filter, the normalizers, HDF5 export) the rate is far less useful, because
the steps differ in cost; there the **label** is the information — it tells you which operation you are
waiting on.

```{note}
A progress bar makes a long run legible; it does not make it shorter. For large stacks the wall clock
is dominated by peak memory rather than I/O — see the note at the end of this page.
```

## Example 1 — let NeuNorm draw the bars

```python
from pathlib import Path

from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

transmission = run_mars_ccd_pipeline(
    sample_paths=sample_paths,   # [[run1_file1, run1_file2, ...], [run2_file1, ...], ...]
    ob_paths=ob_paths,
    dark_paths=dark_paths,
    output_path=Path("normalized.h5"),
    progress=True,
)
```

Three sample runs of 40 frames at 512×512, two open-beam runs of 20, one dark run of 10:

```text
load_sample:  100%|██████████| 120/120 [00:01<00:00, 835.97item/s, attaching variances (40.0 MiB)]
load_ob:      100%|██████████| 40/40   [00:01<00:00,  29.68item/s, attaching variances (20.0 MiB)]
combine_runs: 100%|██████████| 3/3     [00:01<00:00,   2.31item/s, combining 1 dark run(s)]
load_dark:    100%|██████████| 10/10   [00:01<00:00,   7.76item/s, attaching variances (10.0 MiB)]
gamma_filter: 100%|██████████| 4/4     [00:01<00:00,   2.84item/s, detecting and replacing outliers]
normalize:    100%|██████████| 3/3     [00:00<00:00,  20.47item/s, correcting shared-dark variance]
export:       100%|██████████| 4/4     [00:00<00:00,  97.85item/s, writing metadata]
```

Note `load_sample` reaching **120/120**, not 40/40 three times: a load stage counts across the whole
run, not per input run.

That transcript is what the terminal shows *during* the run. NeuNorm's bars are built with
`leave=False`, so they are cleared when the run finishes rather than left behind — the run ends with a
clean terminal, not seven finished bars.

## Example 2 — drive your own bar

The callback receives one event per item or step. This is how any progress library is driven — a
notebook widget, a Qt progress dialog, a log line, a web socket:

```python
from pathlib import Path

from tqdm.auto import tqdm

from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

bars = {}


def report(event):
    bar = bars.get(event.stage)
    if bar is None:
        bar = bars[event.stage] = tqdm(total=event.total, desc=event.stage)
    bar.update(event.completed - bar.n)   # completed is ABSOLUTE; update() takes a DELTA
    if event.detail:
        bar.set_postfix_str(event.detail)


try:
    transmission = run_mars_ccd_pipeline(
        sample_paths=sample_paths,
        ob_paths=ob_paths,
        output_path=Path("normalized.h5"),
        progress=report,
    )
finally:
    for bar in bars.values():
        bar.close()
```

Run against 40 sample frames and 20 open-beam frames, that draws:

```text
load_sample:  100%|██████████| 40/40 [00:00<00:00, 209.93it/s, attaching variances (10.0 MiB)]
load_ob:      100%|██████████| 20/20 [00:00<00:00, 123.05it/s, attaching variances (5.0 MiB)]
combine_runs: 100%|██████████| 2/2   [00:00<00:00,  13.29it/s, combining 1 open-beam run(s)]
gamma_filter: 100%|██████████| 4/4   [00:00<00:00,  29.10it/s, detecting and replacing outliers]
normalize:    100%|██████████| 1/1   [00:00<00:00,  44.00it/s, dividing sample by open beam]
export:       100%|██████████| 4/4   [00:00<00:00, 242.44it/s, writing metadata]
```

Two differences from `progress=True`, both because the bar is now yours: the unit reads `it/s` instead of
`item/s` (NeuNorm passes `unit="item"`), and **the bars stay on screen when the run ends**. NeuNorm builds
its own with `leave=False`, so each line is cleared as the run finishes and you are left with a clean
terminal; a bar you create keeps tqdm's default and persists. Pass `leave=False` to `tqdm(...)` if you
prefer NeuNorm's behaviour.

**`event.completed` is an absolute, cumulative count, not an increment.** `tqdm.update()` takes a
delta, so the adapter is `bar.update(event.completed - bar.n)`. Passing `event.completed` straight to
`update()` makes the bar race past its total: measured on the 120-file load above, a bar that should end
at 120 ends at **7740**.

Each event carries four fields:

```python
def report(event):
    print(event.stage, event.completed, event.total, event.detail)
    # load_sample 37 120 frame_00036.tiff
```

- `stage` — which part of the run, one of the `STAGE_*` constants in
  {py:mod}`neunorm.utils.progress`.
- `completed` — items finished in that stage so far, counted across the whole run.
- `total` — items the stage will process, or `None` when that is not knowable in advance (event-mode
  histogramming, where the chunk count follows from a file's event count).
- `detail` — optional context: the file being read, or the named step running.

Events are emitted synchronously from the calling thread, in order, so the callback does not need to be
thread-safe.

## Example 3 — cancel a run

A callback that raises is not caught. That is the supported way to stop a long run:

```python
from pathlib import Path

from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline


class RunCancelled(RuntimeError):
    pass


def cancel_if_asked(event):
    if user_pressed_stop():          # your own check
        raise RunCancelled(f"cancelled during {event.stage} at {event.completed}")


try:
    run_mars_ccd_pipeline(
        sample_paths=sample_paths,
        ob_paths=ob_paths,
        output_path=Path("normalized.h5"),
        progress=cancel_if_asked,
    )
except RunCancelled as exc:
    print(exc)
```

```text
cancelled during load_sample at 1
```

Cancelling leaves no output file *unless* the export stage had already begun. The HDF5 writer writes in
place, so aborting inside export leaves a file at `output_path` that **opens cleanly and is incomplete**:
cancel on export's first event and you get a valid, empty HDF5 file; cancel later and you get the
datasets written so far. Neither is distinguishable from a finished file by opening it, so delete it
before retrying. That is not specific to cancellation — any error during export does the same.

## What each stage reports

The table below is about the **count** — what advances the number a bar shows. A callback receives more
events than that: NeuNorm also emits *notes*, which carry a label without advancing the count (naming a
large allocation, or a step whose cost is only known mid-run). So a callback counting its own invocations
will exceed the totals here; use `event.completed` rather than counting calls.

| Stage | What advances the count | Notes |
|---|---|---|
| `load_sample`, `load_ob`, `load_dark` | one event per file | across every input run, so the count does not restart per run. On the event path, four events per file instead — the loader's full-event-length allocations, so one huge NeXus file still shows movement |
| `histogram` | one event per event-chunk | `total` is `None`: the chunk count follows from each file's event count |
| `combine_runs` | one event per combined family | sample, open beam, dark |
| `gamma_filter` | four named steps | the third is the median filter, most of its cost |
| `rebin_tof` | two events | sample and open beam |
| `normalize` | named steps | the flux correction (background-ROI or proton-charge) and the division; the count depends on which correction was requested |
| `export` | named steps for HDF5 | one event per file with `tiff_one_file_per_image=True`, which is the only export path with a determinate item count |

Some work is deliberately **not** reported: the metadata reads, the ROI crop, the open-beam and dark
averaging, dead and hot pixel detection, the statistics analysis behind `rebin_by_tof=True`, the spatial
rebin and the air-region correction. Each is a single pass that runs between named stages, and inventing
a step for each would inflate the counts without adding information. Each pipeline's `progress` docstring
lists what its own run leaves out, because the lists differ — VENUS TPX1 detects dead pixels but not hot
ones, the MARS pipelines have no air-region correction, and only the TOF pipelines rebin.

One of those is worth knowing about specifically. On the three TOF pipelines each input run begins with
a **metadata read**: `load_metadata` opens a NeXus file (and, for TPX1, parses a `*_Spectra.txt` sidecar
that can run to thousands of rows). For the first run that happens before any bar exists, so a TOF run
is briefly silent at the very start; for later runs it happens with the load bar already live, so the
pause lands mid-stage. Either way that gap is the metadata, not a hang.

So a bar that pauses briefly between stages is expected. A bar that pauses *within* a stage is telling
you where the time goes — and if it pauses on `attaching variances (40.0 MiB)`, that is the allocation,
not a hang.

## Log records and bars share stderr

NeuNorm logs through [loguru](https://loguru.readthedocs.io/), whose default handler writes to stderr —
the same stream `tqdm` draws on. A log record written while a bar is live overwrites part of it. In a
measured MARS CCD run, **five records landed mid-bar**.

NeuNorm does not fix this for you, and the reason matters: the only way to route records around a bar is
to remove the handler that is writing them, and that handler belongs to your application. Removing and
re-adding it inside a library would discard a configuration NeuNorm cannot faithfully restore — your
format, level, filters, sinks and rotation. So the remedy lives in the caller, where the logging
configuration lives.

Route records through `tqdm.write`, which knows how to print above a live bar:

```python
import sys
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

logger.remove()                                   # drop YOUR handlers; re-add them afterwards
handler = logger.add(lambda message: tqdm.write(message, end="", file=sys.stderr))
try:
    run_mars_ccd_pipeline(
        sample_paths=sample_paths,
        ob_paths=ob_paths,
        output_path=Path("normalized.h5"),
        progress=True,
    )
finally:
    logger.remove(handler)
    logger.add(sys.stderr)                        # or restore your own configuration
```

Three details, all measured or easy to miss:

- **The last line restores a *default* handler, not yours.** `logger.add(sys.stderr)` gives you loguru's
  out-of-the-box format at `DEBUG` — not the format, level, filters or rotation you had before
  `logger.remove()`. If your application configures logging, re-apply your own configuration there
  instead. This is exactly why NeuNorm does not do any of this for you: it cannot know what to put back.
- **`logger.remove()` first.** Adding the `tqdm.write` sink *alongside* the default handler makes things
  worse, not better: every record is then emitted twice — one clean copy printed above the bar by
  `tqdm.write`, and one still written into the bar by your original handler. Measured on a small run,
  nine distinct records became twenty lines, ten of them still colliding.
- **`file=sys.stderr`.** `tqdm.write` defaults to **stdout**, so leaving it off silently moves every log
  record to a different stream. The bar looks perfect and the records are somewhere else.

If you would rather not see NeuNorm's records at all during a run:

```python
from pathlib import Path

from loguru import logger

from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

logger.disable("neunorm")
try:
    run_mars_ccd_pipeline(
        sample_paths=sample_paths,
        ob_paths=ob_paths,
        output_path=Path("normalized.h5"),
        progress=True,
    )
finally:
    logger.enable("neunorm")
```

## Notebooks

`progress=True` uses `tqdm.auto`, so there is no `notebook=` flag to set — that was the 1.x spelling,
and it is the one thing this feature replaces outright.

`tqdm.auto` picks the widget bar only inside a **live IPython kernel**, and only when `ipywidgets` is
installed. Having `ipywidgets` in the environment is not enough on its own: in a terminal session with
it installed, `tqdm.auto` still resolves to the text bar. Either works; you do not choose.

`TQDM_DISABLE=1` in the environment suppresses the bars without changing any code, which is what you
want in CI and in logs. It has to be set **before `tqdm` is first imported** — which in practice means
before you import a NeuNorm pipeline, and most simply in the environment your process starts with.
Setting it later has no effect: `tqdm` captures it at import, so a bar created afterwards is still
enabled. A callback is unaffected either way; your callback is yours.

## Reporting from your own code

The same contract is available for functions you write around NeuNorm:

```python
from neunorm.utils.progress import STAGE_LOAD_SAMPLE, resolve_progress


def my_loader(paths, *, progress=False):
    with resolve_progress(progress, STAGE_LOAD_SAMPLE, total=len(paths)) as report:
        for path in paths:
            ...                          # do the work
            report(detail=path.name)     # then count it
```

`resolve_progress` turns `False` / `True` / a callable — or a reporter handed down by a caller — into
one object with the same interface, and the `with` block releases any bars NeuNorm opened. Count *after*
the work, so a failure never reports a step that did not happen, and use `report.note("...")` to name
work that has no item of its own.

## A note on where the time actually goes

For large stacks the wall clock is dominated by **peak memory, not I/O**: `load_tiff_stack` holds a
measured 5.18× multiple of the stack while building it, and above roughly 1200×1200 frames the process
starts swapping, at which point per-file cost grows with the file count. Progress reporting makes that
wait legible and shows you which allocation you are waiting on; reducing it is separate work.

## See also

- {py:mod}`neunorm.utils.progress` — the full API: `ProgressEvent`, `ProgressReporter`,
  `resolve_progress`, and the `STAGE_*` constants.
- {doc}`migration` — the 1.x `notebook=True` flag and what replaced it.
