# Changelog

All notable changes to NeuNorm are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.4.0] - 2026-08-26

### Added

- **Spatial moving average and moving sum before normalization**
  ([#203](https://github.com/ornlneutronimaging/NeuNorm/issues/203)). A new keyword-only
  `moving_window` on `run_venus_tpx1_pipeline`, `run_venus_tpx3_histogram_pipeline` and
  `run_venus_tpx3_event_pipeline` replaces each pixel by the average — or, with `kind="sum"`, the
  total — of a box of pixels around it, applied to the sample and the open beam once they have been
  combined and immediately before they are divided. Ported from iBeatles' `moving_average`, whose box
  filter is `scipy.ndimage.convolve` with a normalized `ones()` kernel and a mirrored frame edge.

  Configured with `MovingWindow(x=3, y=3)`, whose sizes are given **by dimension name**: the event
  path produces `(tof, x, y)` and the histogram path `(tof, y, x)`, so a positional tuple would
  transpose the two spatial axes. `dimension="3D"` additionally averages along TOF. Sizes are in
  post-crop, post-`rebin_by_spatial` pixels, so a window of 3 on a 2x-rebinned stack spans 6 detector
  pixels.

  Two departures from the reference, both required by NeuNorm's own contracts: dead and hot pixels are
  excluded from the window rather than averaged into it (a mask-blind filter lets one dead pixel drag
  down `k**2` neighbours), and variances are propagated as `sum(w**2 Var)`.

  The window trades spatial resolution for per-pixel precision at a fixed rate — a `k x k` window
  improves per-pixel precision by `k` and coarsens resolution by `k` — while the array keeps its
  shape, so the result presents as full resolution while carrying roughly one independent value per
  `k**2` pixels. NeuNorm says so at run time, records the window under `/metadata/moving_window`, and
  refuses to combine it with `spectrum_roi` or `air_roi`, where a region reduction over correlated
  pixels under-reports its uncertainty by roughly `sqrt(kernel pixels)`. Applied before normalization
  `kind="sum"` and `kind="average"` give indistinguishable transmission, because the kernel count
  cancels in the ratio. See `docs/moving_window.md`, whose three measurement tables are regenerated
  and checked by the test suite.

- **Resonance mode: ROI-level normalization producing a transmission spectrum**
  ([#197](https://github.com/ornlneutronimaging/NeuNorm/issues/197)). A new keyword-only
  `spectrum_roi` on `run_venus_tpx1_pipeline`, `run_venus_tpx3_histogram_pipeline` and
  `run_venus_tpx3_event_pipeline` switches the output from a stack of normalized images to a **1-D
  transmission spectrum**: for each spectral bin the sample's mean counts over the region are divided
  by the open beam's mean counts over the same region, giving one point per bin. Written as a
  three-column ASCII file — `bin_index,transmission,uncertainty`, comma-delimited with exactly one
  plain header line — which is the `*_Spectra.txt` shape the ORNL imaging tools exchange and which
  NeuNorm's own reader parses with `np.loadtxt(path, skiprows=1, delimiter=",")`. An HDF5 file is
  written alongside it carrying the time axis, the masks and the provenance that three columns cannot
  hold. See `docs/resonance_mode.md`.

  The requirement is an order of operations: the region is collapsed to one number per bin **before**
  the division, because the ratio of means is not the mean of ratios — about 1.2% apart on synthetic
  Poisson counts over a 64-pixel region, and the per-pixel form additionally yields NaN wherever an
  open-beam pixel recorded zero counts in a bin. The region mean is the same mask-aware pooled mean
  `background_roi` and `air_roi` already use, so rectangles, arbitrary-shape `MaskROI` selections and
  pooled lists all behave identically, and a dead pixel inside the region is excluded from the summed
  counts and the pixel count alike.

  Both sides are given the **union of both sides' masks** before the collapse. A region mean divides by
  its own count of unmasked pixels, so a pixel masked on the sample alone leaves numerator and
  denominator averaging over different pixels while the dead pixel goes on inflating the open beam:
  measured on four pixels with one dead under non-uniform flux, 0.400 from a ratio of sums and 0.533
  from a ratio of means against a true 0.800, which only mask symmetry recovers. Switching from sums to
  means is not by itself the fix.

  Frame-index binning works as in image mode — `rebin_by_tof` takes the same integer factor, `True`, or
  explicit `[[start, stop], ...]` list, and the stacks are binned before the region is collapsed, so a
  spectrum run and an image run of the same data see the same counts. `bin_index` is each point's first
  input frame, which is the file index when no rebinning ran: a gapped list `[[0, 2], [4, 6]]` gives
  rows indexed 0 and 4, and the dropped span has no row. `spectrum_roi_strict` (default `True`) guards
  a non-positive or non-finite **open-beam** region mean; a zero *sample* mean is a real measurement — a
  fully absorbing bin — and gives transmission 0.0. TIFF output is refused for a spectrum, because the
  TIFF writer would otherwise quietly produce a multi-page file of 1x1-pixel images.

  `spectrum_roi` indices are resolved against the arrays as normalization sees them, so after a `roi`
  crop or a `rebin_by_spatial` they are **not** detector pixels — the same caveat `background_roi`
  carries. A run that combines them warns, because the numbers look plausible either way.

  Scope is the three VENUS TOF pipelines. The CCD pipelines and MARS TPX3 collapse or lack a spectral
  axis at normalization time, so they have no per-bin open beam to divide by.

- **`detect_resonances` takes a `spectrum_roi`**, so peaks are looked for where the sample is rather
  than across the whole detector. Its integrated spectrum is now the mask-aware pooled region mean with
  the masks symmetrized, instead of two independent bare sums — dividing two sums with no pixel count
  is biased whenever the two sides carry different masks (measured 0.400 against a true 0.800). The SNR
  is computed from the propagated transmission variance instead of a Poisson formula hard-coded over
  raw counts, which is what allows the spectrum to be a mean at all: feeding means to the old formula
  would have shrunk every SNR by `1/sqrt(N_pixels)`. For counting data the two agree to floating-point
  round-off — measured maximum relative SNR difference 1.9e-16 on synthetic Poisson data, with
  identical peaks — so detection on existing data is unchanged.

- **`neunorm.processing.spectrum_reducer`** — `roi_mean_spectrum` promotes the pooled region mean to a
  documented public function (no numerical change; the existing private helper stays the single
  implementation), and `normalize_roi_spectrum` reduces a sample and open-beam pair to a transmission
  spectrum carrying the `N+1` `tof` bin edges, so the result can be rebinned again or converted to
  wavelength or energy. It reuses `normalize_transmission`, so the proton-charge correction and its
  0.5% systematic come with it. Sample and open-beam inputs whose aligned time axes disagree raise a
  message naming the axis, the deviation and both inputs, rather than a raw scipp `DatasetError`.

- **`neunorm.exporters.ascii_writer`** — the package's first text exporter. `write_ascii_spectrum`
  writes the three-column spectrum; a spectrum that is not 1-D, or that carries no variances, is
  rejected rather than written with a fabricated uncertainty column.

### Changed

- **The three VENUS TOF pipelines share one implementation.** `venus_tpx1`, `venus_tpx3_histogram` and
  `venus_tpx3_event` each carried the same ~150 lines — crop, dead/hot detection, spatial rebin, TOF
  rebin, normalize, air correction, coordinate labelling, export — so that middle now lives once in
  `neunorm.pipelines._tof_spine.reduce_tof_stacks` and each entry point keeps only its own loading and
  metadata. 409/405/418 lines become 271/263/287, and all three `# noqa: C901` complexity suppressions
  are gone. Public signatures are unchanged apart from the two new keyword-only parameters.

  The differences that genuinely exist between the three are now named in a `TofPipelineProfile` rather
  than implied by which copy of the code you are reading, **including two that look like oversights and
  are preserved exactly**: `venus_tpx3_histogram` re-detects its dead/hot masks from the *sample* after
  a spatial rebin where its own pre-rebin detection and both other pipelines read the open beam, and
  `venus_tpx1` passes no `hot_pixel_mask` to the HDF5 writer. Changing either would change published
  output, so neither was changed here.

  Behaviour-preserving, and verified rather than asserted: every pipeline's written `transmission`,
  `uncertainty`, `tof`/`spectra_tof`/`wavelength`/`energy` coordinates and every mask are bit-identical
  before and after, across 31 runs covering all three pipelines against ten argument combinations plus
  three untouched pipelines as controls.

## [2.3.0] - 2026-08-13

### Added

- **Progress-reporting contract** — new `neunorm.utils.progress` module
  ([#195](https://github.com/ornlneutronimaging/NeuNorm/issues/195)), the foundation for reporting
  where a long normalization run has got to. Defines the immutable `ProgressEvent(stage, completed,
  total, detail)` handed to user callbacks, the `Progress` / `ProgressCallback` types, the shared
  `STAGE_*` labels, and `resolve_progress()`, which normalizes a caller's `progress` argument —
  `False` (no reporting), `True` (NeuNorm drives a `tqdm` bar), or a callable — into a pre-bound
  reporter. `completed` is an **absolute** count, so a `tqdm` adapter is
  `bar.update(event.completed - bar.n)`; `tqdm.update()` takes a delta, and passing the absolute
  count to it would overshoot. `total` is `None` where an item count is not knowable in advance.
  Events are emitted synchronously from the calling thread, and a callback that raises is not
  caught — that is how a caller cancels a run. `tqdm` is imported lazily, so the default
  `progress=False` path never pays for it. `close()` releases the bars NeuNorm opened — a stage with
  an indeterminate total has no completion point, and an abandoned stage never finishes — and never
  touches a caller's callback, which may be a reusable object NeuNorm does not own.
  Note that NeuNorm's log records and a `tqdm` bar both go to stderr, so log lines corrupt a bar.
  NeuNorm does not manage that for you: routing loguru through `tqdm.write` means displacing handlers
  that `add()` cannot faithfully restore, and silently rewriting an application's log format is worse
  than a corrupted bar. Routing NeuNorm's log records around a bar is the caller's to do, in their own
  application — `docs/progress.md` gives the remedy, and names the two ways to get it wrong.
  This entry also corrects the long-stale `neunorm.utils` docstring, which advertised "progress
  reporting" that did not exist and "validation helpers" that never have.

- **Progress reporting through the image load path** — `load_tiff_stack`, `load_fits_stack` and
  `load_stack` accept a keyword-only `progress` (and a `stage` label, so a direct open-beam or dark
  load is not reported as `load_sample`). Each emits one event per file read, naming the file, then
  announces the two whole-stack allocations that follow the read loop — the stack build and the
  variances copy. Those two are where the memory actually peaks: the read loop only appends to a
  list, so without them a bar reaches 100% at the last file and then goes silent through the part
  that can exhaust RAM and start swapping. They are emitted as *notes*, which carry a label without
  advancing the count, so a bar shows what is running without the count restarting on each call.
  `load_stack` forwards both arguments to whichever leaf loader it dispatches to, which is what
  gives the CCD pipelines per-file progress at all — they call `load_stack`, not the leaves.
  A progress callback that raises propagates, and is not reported as a failed file read. All three
  loaders again accept a non-sized iterable such as `Path.glob(...)`, as they did before reporting
  was added.

- **Progress reporting through the event-mode path** — `load_event_nexus` and both event converters
  (`convert_events_to_histogram` and `convert_events_to_2d_histogram`) accept a keyword-only
  `progress` and `stage`. The event path has no per-file loop, so what is reported is different in
  kind: `load_event_nexus` counts its four full-event-length allocations (two h5py slab reads, the
  event-id unroll, the TOF conversion), naming each before it runs, because that sequence is where
  the event path peaks in memory. The converters emit one event per chunk — at the default 500M
  events per chunk a typical run is a single tick, becoming a real progression only for
  billion-event datasets. Both converters are instrumented deliberately: `mars_tpx3` uses the 2-D
  one and `venus_tpx3_event` the 3-D one, so covering either alone would leave a pipeline silent.
  This also **removes an ad-hoc `logger.info` chunk-percent print** that fired on every tenth chunk
  *or the last one* whenever there was more than one chunk — so from about 500 million events
  upward, and then only once for most runs — and wrote to a channel a caller could not redirect or
  disable.

- **Progress reporting through the two dominant compute stages** — `apply_gamma_filter` and
  `normalize_transmission` accept a keyword-only `progress` and `stage`. Neither has an item axis, so
  both report named steps: the gamma filter reports four, the third being the
  `scipy.ndimage.median_filter` that is most of its cost; the normalizer reports its separable
  whole-array operations — the flux correction (background-ROI or proton-charge) and the division.
  Each step is named before it runs and counted after it returns, so a failure never reports work
  that did not happen. These are the two places a run goes quiet for longest: the gamma filter is
  **on by default** on both CCD pipelines and MARS TPX3 and is the slowest stage per frame there,
  and the normalizer dominates the TOF paths — at 300 x 512² its proton-charge steps take seconds
  each. Work that is conditional on a value only known mid-run (the per-outlier variance
  recomputation, the background-ROI variance term) is announced without advancing the count, so the
  declared total always matches the steps that actually run whichever arguments are given.

- **`ProgressReporter` is a context manager.** `with resolve_progress(...) as report:` releases the
  progress bars on the way out whether the body returned or raised. Every instrumented function now
  uses that shape, so a forgotten `finally` cannot leak an abandoned bar again — which it did once
  already. Sink ownership travels with it: `resolve_progress` hands a callee a **borrowed** view of a
  reporter it was given, so only whoever resolved a sink into being may retire it. Without that a
  callee's `with` block would close the caller's bars, and since a bar is no longer auto-closed at
  completion the caller's next event would rebuild it from zero — a pipeline's bar flickering back to
  0% on every instrumented call it makes.

- **Progress reporting through export, and through the dark-corrected normalizer** — `write_hdf5`,
  `write_tiff_stack` and `normalize_with_dark` accept a keyword-only `progress` and `stage`. HDF5 is
  the primary output format and `write_hdf5` has no item axis — the bulk data leaves in one or two
  whole-array writes — so it reports named steps and can never be per-image: the transmission
  dataset, the uncertainty dataset, the coordinate/mask section, and the metadata section. Its total
  is computed, because the uncertainty write happens only for variance-bearing data and the metadata
  section only when metadata is supplied. Every emit sits **outside** the writer's five non-re-raising
  handlers (three `except Exception`, one `except (TypeError, ValueError)`, one `except TypeError`):
  those exist so one un-writable metadata key or unserializable coordinate cannot abort the bulk-data
  write, and a tick placed inside one would swallow a cancelling callback's exception, turning a
  user's abort into a silently skipped metadata key. Cancelling — or any mid-write error, which was
  already possible — leaves a partially-written file at the output path; that is now documented on the
  parameter, since supporting cancellation makes it reachable on purpose. `write_tiff_stack` in `one_file_per_image` mode
  is the one export path with a determinate item count and emits one event per file, naming it; stack
  mode reports its single multi-page write as one step, so the default export path is not silent
  either. `normalize_with_dark` — `normalize_transmission`'s sibling on the CCD path, which was
  reporting nothing — reports its two dark subtractions and then hands its reporter to the normalizer,
  so a dark-corrected run counts through both functions as **one continuous bar** rather than
  restarting. Both totals come from one shared helper so they cannot drift apart. Its shared-dark
  variance correction is announced too: that correction is **58% of the function's wall clock** at
  80 x 512², and it ran outside the reporting context at first, so a caller saw the bar reach its total
  and the bars vanish and then waited out more than half the call with nothing on screen. Reporting
  does not change a single byte of any written file, nor any computed value: the dark normalizer's
  output is bit-identical across seven correction branches, variances included.

- **Progress documentation** — a new {doc}`progress <progress>` page (`docs/progress.md`, in the
  toctree under "Using") covering what progress reporting tells you and what it deliberately does not:
  you get movement and a per-stage rate, **not a whole-run ETA**, because stage costs differ by 60-120x
  and which stages run depends on the arguments. Three worked examples — the built-in bar, a callback
  driving your own bar, and cancelling a run — each executed and shown with its real output, plus a
  per-stage table of what is counted and a plain list of the operations that are not.

  It also documents the one interaction a user will otherwise hit blind: NeuNorm logs through loguru,
  whose default handler writes to the same stderr `tqdm` draws on, so records garble the bars — five of
  them in a measured MARS CCD run. The remedy given (remove your handler, add a `tqdm.write` sink with
  `file=sys.stderr`, restore afterwards) was tested against the alternatives, and the two easy mistakes
  are called out because both were measured: adding the sink *alongside* the default handler doubles the
  collisions rather than removing them, and omitting `file=sys.stderr` silently moves every log record
  to stdout. `docs/migration.md` gains the row for the lost 1.x flag — `Normalization(..., notebook=True)`
  → `progress=` — which is why the regression went unnoticed for a whole major version.

  Every Python block on the page is extracted and **executed** by `tests/unit/test_docs_progress_examples.py`,
  including the log remedy, which is run in a subprocess and checked to leave zero collisions with the
  records still on stderr. Six mutations of the documentation are each caught by those tests.

- **`progress` on all six pipelines** — `run_mars_ccd_pipeline`, `run_venus_ccd_pipeline`,
  `run_mars_tpx3_pipeline`, `run_venus_tpx1_pipeline`, `run_venus_tpx3_histogram_pipeline` and
  `run_venus_tpx3_event_pipeline` take a keyword-only `progress`, which is what makes any of this
  visible from the public API: `progress=True` draws one `tqdm` bar per stage, a callable receives every
  event, and raising from it cancels the run. A CCD run draws seven bars — sample, open-beam and dark
  loads, the run combine, the gamma filter, the normalization and the export; a TOF run adds the TOF
  rebin and, for event input, histogramming.

  Each **load stage counts across the whole run, not per input run**: three sample runs of 40 frames
  count 1…120 under one total of 120. That matters beyond neatness — the documented `tqdm` adapter is
  `bar.update(event.completed - bar.n)`, so a count that restarts per run computes a *negative* delta.
  Stage totals are declared by the pipeline, because a handed-down reporter deliberately keeps the total
  its caller bound; to stop those totals drifting from the ticks that actually arrive, each is derived
  from a helper exported by the code that emits them — `GAMMA_FILTER_STEPS`,
  `LOAD_EVENT_NEXUS_STEPS`, `normalize_step_count`, `normalize_with_dark_step_count`,
  `hdf5_export_step_count`, `tiff_export_step_count`.

  The event path reports differently, because reading one NeXus file is not one cheap item: each file is
  named as it is opened and then counted in the four full-event-length allocations it performs, so a
  single 40-million-event file still shows movement. Histogramming carries **no total** — the chunk
  count follows from a file's event count, which is not known until the file is read — rather than an
  assumed one-chunk-per-file that a large run would overshoot. Two operations no instrumented leaf
  covers are now reported by the pipelines themselves: `combine_runs`, the largest silent block in a
  multi-run job, and `rebin_tof`, whose median reduction is one of the slowest stages in a TOF run.
  What is still not reported, deliberately: the ROI crop, the open-beam/dark averaging, dead/hot pixel
  detection, the spatial rebin and the air-region correction — single whole-array passes between named
  stages. Each pipeline's `progress` docstring lists what its own run leaves unreported, so it is not
  left to be inferred from a bar that seems to pause — the lists differ, because the pipelines do:
  VENUS TPX1 detects dead pixels but not hot ones, MARS CCD and MARS TPX3 have no air-region correction,
  and only the TOF pipelines rebin.

- **Per-image TIFF export** — `write_tiff_stack(..., one_file_per_image=True)`, exposed on the
  VENUS TOF pipelines as `tiff_one_file_per_image=True`, writes **one scitiff file per spectral
  image** (one normalization per file) named `<stem>_00000.tiff`, `<stem>_00001.tiff`, … in spectral
  order, instead of a single multi-page stack. Requested by the instrument scientist for workflows
  built around opening individual images in ImageJ. Each per-image file is a **single-page** image
  holding only the normalization: packing the scitiff stdev and mask planes as extra channels would
  make a viewer report three times as many images, the extra two being an uncertainty plane and an
  all-zero mask. Pass `concat_stdevs_and_mask=True` to keep those channels; note that scitiff stores
  variances only in that channel, so a single-page file carries no uncertainty — read it from the HDF5
  output. Each file keeps its own slice's coordinates (that bin's `tof` bounds and `spectra_tof` time)
  plus the same metadata and masks as the stack. Stack output is unchanged, so existing workflows are
  unaffected.
- **Flexible TOF rebinning** — variable-width, list-defined bins with a choice of reduction
  ([#192](https://github.com/ornlneutronimaging/NeuNorm/issues/192)). The VENUS TOF pipelines
  (`venus_tpx1`, `venus_tpx3_histogram`, `venus_tpx3_event`) now accept an explicit
  `rebin_by_tof=[[start, stop], ...]` list of **half-open frame-index ranges** (Python convention)
  in addition to the existing `bool`/`int` factor, plus a new `rebin_reduction` parameter selecting
  how frames combine per bin — `"mean"` (default for a bin list), `"sum"`, or `"median"` — with
  scipp variance propagation: mean `ΣVar/N²`, sum `ΣVar`; the median value is exact and, for a bin of
  three or more frames, its uncertainty uses NeuNorm's standard median-variance approximation
  `Var(median) ≈ (π / (2n)) · mean(Var)` (with a warning that it is an estimate) — the same rule as
  `processing.reference_preparer.median_with_variance` and the gamma filter, so median uncertainties
  are consistent across the package. A one/two-frame bin uses the exact `Var(mean)`, since the median
  equals the mean there. An exact small-sample median variance would require per-pixel resampling
  (bootstrap): sound in principle, but impractical at detector scale, so it is deliberately not used. Existing `rebin_by_tof=int`/`True` behavior is
  unchanged (still sums; `rebin_reduction` defaults to that). The output has **exactly one image per
  requested range**: frames covered by no range are **dropped silently** — a deliberate choice made
  with the instrument scientist, wherever the omission falls (between ranges, before the first, or
  after the last). No extra bin is inserted, nothing is masked, and nothing is logged; the per-bin
  `spectra_tof` axis is the record of which frames each image covers. **Caveat, documented in the API
  and workflow guides:** dropping frames leaves the output images covering disjoint time bands, which
  a scipp `N+1` bin-edge axis cannot express exactly — leading edges stay exact, but the omitted span
  is absorbed into the closing edge of the preceding bin, widening that bin's implied `tof` (and
  derived `wavelength`/`energy`) span. Such output is not a continuous spectrum and should not be used
  for Bragg-edge or resonance analysis; prefer contiguous ranges there. Pixel values, variances and
  `spectra_tof` are exact regardless. Each rebinned bin carries a
  `spectra_tof` point coordinate (the mean of its member frames' times), persisted to the HDF5/TIFF
  output. The API lives in `neunorm.tof.histogram_rebinner`: the existing `rebin_tof` gains a
  `reduction` argument and accepts an explicit `[[start, stop], ...]` bin list as its `width` (its
  adjacent-bin, sum-only integer/time/wavelength/edge-snapping behavior is unchanged when summing),
  backed by the public helpers `reduce_tof_bins` and the `linear_bin_list` / `log_bin_list` bin-list
  generators (uniform and geometric frame-index bins, the latter porting iBeatles' logarithmic mode
  without its zero-start infinite loop).
- **Arbitrary-shape (mask-based) ROI regions** — `MaskROI`
  ([#180](https://github.com/ornlneutronimaging/NeuNorm/issues/180)). A pixel **selection** mask
  (same `(y, x)` size as the image; 1/nonzero = pixel *in* the region — the opposite polarity of
  scipp's exclusion masks) accepted by the **region-statistics** operations, anywhere they take a
  rectangular ROI: `normalize_transmission(background_roi=)` / `normalize_with_dark` /
  `apply_background_roi` (poolable with rectangles in one list, including the shared-dark ROI-mean
  covariance correction), `apply_air_region_correction`, and the pipeline `air_roi=`/`background_roi=`
  parameters. Construct from a numpy/scipp array (`MaskROI(selection=...)`), an image file drawn in
  e.g. ImageJ (`MaskROI.from_file("mask.tif")`, nonzero selects), or a mask already on a DataArray
  (`MaskROI.from_dataarray_mask(da, name, invert=...)`). Region statistics are mask-aware (dead/hot
  pixels excluded exactly as for rectangles). Cropping (`apply_roi` / the pipeline `roi=`) stays
  rectangle-only — an arbitrary shape has no rectangular crop, so a `MaskROI` is rejected there with
  a pointer to the region-statistics parameters. Output-file provenance records a JSON summary
  (shape, selected-pixel count, source, sha256), including `air_roi`. Accepting a `MaskROI` does not
  itself change what a rectangle does — but two other fixes in this release do move some
  rectangle-only results, so a rectangle is **not** bit-identical to 2.2.x across the board: see the
  air-region and overlapping-pooled-ROI entries under **Fixed**. Concretely, a rectangular `air_roi`
  now differs at floating-point round-off (~1e-16 relative), and an **overlapping** pooled
  `background_roi` / `air_roi` list of rectangles changes materially (the union fix — a constructed
  two-rectangle case moves the pooled mean by ~7% and its variance by ~14%). A single rectangle, or a
  non-overlapping list of them, is unchanged bit-for-bit apart from that air-region round-off.
- **A tolerance for the run-combine metadata match check** — `metadata_match_atol` on `combine_runs`
  and `run_mars_ccd_pipeline` ([#209](https://github.com/ornlneutronimaging/NeuNorm/issues/209)).
  `combine_runs` compared every coordinate named in `metadata_check_match` with `sc.identical`, which
  is exact, so a slit aperture readback (`MotSlit*.RBV`) differing in its last decimal place between
  two runs of the same measurement aborted the combine — and with it the whole MARS CCD pipeline,
  which checks four of those readbacks. The new parameter defaults to `0.0`, which is exactly the
  previous exact-match behavior. Given a positive tolerance, numeric coordinates **of the same unit
  and shape** match when they differ element-wise by at most `atol`; a unit or shape mismatch remains
  a mismatch whatever the tolerance, and non-numeric values (the detector and manufacturer name
  coordinates) still require exact equality, so a tolerance cannot mask a genuinely different
  detector. Coordinates that match only within the tolerance would still fail scipp's exact
  aligned-coordinate check in `+=`, so they are aligned to the base run's value on a **shallow copy**
  taken per run before summing: the caller's arrays are never modified, and `metadata_keys_to_sum`
  still aggregates each run's original values rather than the aligned ones. `run_mars_ccd_pipeline`
  forwards the tolerance to all three of its combine steps — sample, open beam and dark.

### Changed

- **Parameters added since v2.2.3 are now keyword-only, and the released positional order is guarded
  by tests.** `rebin_reduction` and `tiff_one_file_per_image` on the three VENUS TOF pipelines,
  `one_file_per_image` and `concat_stdevs_and_mask` on `write_tiff_stack`, `image_dir` on
  `load_metadata`, and `metadata_match_atol` on `combine_runs`, are keyword-only. All of them
  had been added positionally on `next`, and `rebin_reduction` in particular was inserted immediately
  after `rebin_by_tof`, shifting every later positional parameter — five of them in
  `venus_tpx3_event`. None of it had been released, so the shift is undone rather than frozen: the
  positional parameter names and order of all six pipelines, of `write_tiff_stack` and of
  `combine_runs` are again
  exactly those of v2.2.3 (some type annotations have widened since, which does not affect binding), and `tests/unit/test_public_signatures.py` pins them against the tag (not against the
  current code) plus asserts that anything added since is keyword-only and that no released parameter
  becomes positional-only. **No action is needed for any released caller** — this restores v2.2.3's
  order rather than departing from it. Callers written against `next` since 3bd2b07 that passed those
  four arguments positionally, or `metadata_match_atol` positionally since 4a976e2, must switch to
  keywords; every in-repo and documented call site already used keywords. `combine_runs` was itself
  absent from the guard until now, which is why its addition was the one that slipped through
  positionally — it is covered by both assertions from this release on.
- **`write_tiff_stack` now returns the files it wrote** — `list[Path]`, in write order, where v2.2.3
  returned `None`. The per-image mode makes the written set no longer inferable from the arguments
  alone, so it is reported rather than reconstructed. Nothing breaks: code that ignored the return
  value is unaffected.
- **The conda package now declares `scitiff`.** `exporters/tiff_writer.py` imports it at module
  level and every pipeline imports `write_tiff_stack`, but the conda recipe's run-dependencies
  omitted it, so `import neunorm.pipelines.<any>` failed on a conda install for the whole 2.x series
  to date. It went unnoticed because `neunorm/__init__.py` imports nothing but `_version`, so the
  release pipeline's import check passed vacuously; that check now imports all six pipelines instead.
  The PyPI wheel was never affected — it declares `scitiff>=26.1` through `[project] dependencies`.

### Fixed

- **Documentation build restored under Sphinx 9.** The API reference was silently dropping seven
  modules — all six `neunorm.pipelines.*` modules and `neunorm.exporters.tiff_writer` — and the
  Read the Docs build was failing outright. The cause is an upstream Sphinx 9 defect
  ([sphinx-doc/sphinx#14337](https://github.com/sphinx-doc/sphinx/issues/14337)): its rewritten
  autodoc walks a documented class's whole MRO and, for each class's module, rebuilds annotations
  from that module's source and writes them back onto the live classes, including third-party ones
  and not only the base classes themselves. Reaching pydantic's `BaseModel`
  restores the `__pydantic_extra__` annotation that pydantic clears at class-creation time, after
  which the next model declared `extra="allow"` (scitiff's) fails schema generation and every
  module importing it drops out of the docs. Setting `autodoc_use_type_comments = False` in
  `docs/conf.py` disables the offending pass; NeuNorm uses no `# type:` comments, so nothing is
  lost, and this becomes the Sphinx default in version 10. No runtime behaviour was affected — the
  package imported and the test suite passed throughout. A `docs` job was also added to CI, which
  runs the same warnings-as-errors build as Read the Docs, so a documentation-breaking dependency
  update now fails a pull-request check instead of merging unnoticed.
- **Air-region correction now uses the true (pooled) region mean.** The previous implementation
  averaged with `sc.mean` over a dim list, which reduces sequentially (a mean of row-means): with
  dead/hot-masked pixels in the air region it weighted rows unequally (a slightly wrong estimator
  and uncertainty), and a fully-masked row made the whole correction NaN. The air mean is now the
  mask-aware pooled mean shared with `background_roi` (`sum(T)/count(unmasked)`); values change
  only for masked air regions (unmasked regions agree to floating-point round-off). The propagated
  variance at `T == 0` pixels is now finite (the old formula produced `0 * inf = NaN`), a
  non-positive/non-finite air mean now raises `ValueError` instead of silently corrupting the
  scale, and errors name `air_roi`. Multiple pooled air regions are supported.
- **Overlapping pooled ROIs no longer understate their uncertainty.** When a pooled
  `background_roi` / `air_roi` list contained regions that overlap, the shared pixels were counted
  more than once — this not only inflated the pooled mean but understated its variance (and the
  shared-dark covariance) by adding the repeated pixels' variances as if from independent samples.
  Pooled statistics now reduce over the **union** of the regions (each selected, unmasked pixel
  counted once); non-overlapping lists are unchanged, bit-for-bit.
- **VENUS TPX1 read its TOF and shutter-count sidecars from the wrong directory, and its `tof` axis
  was not a bin-edge axis** ([#187](https://github.com/ornlneutronimaging/NeuNorm/issues/187)).
  `load_metadata` located the `*_Spectra.txt` and `*_ShutterCount.txt` sidecar files through the
  raw-acquisition directory recorded in the NeXus DAS log (`BL10:Exp:IM:ImageFilePath`). VENUS TPX1
  images come from the auto-reduction tree, which holds a different frame count than the raw tree, so
  the spectra read from there did not correspond to the image stack that had just been loaded — a
  silent mismatch between the TOF axis and the frames it labelled. `load_metadata` now takes a
  keyword-only `image_dir`: when given, both sidecars are read from that directory, and the returned
  `image_file_path` records it, so the stored provenance names the directory the data actually came
  from instead of the known-mismatched DAS-log path. `image_dir` is resolved to an absolute path, so a
  relative value does not depend on the working directory. Note the two sidecar readers still differ
  on a directory that does not exist: `load_spectra_tof` raises `FileNotFoundError`, while
  `load_shutter_counts` logs a warning and returns an empty array. Omitting `image_dir` leaves the
  DAS-log behavior unchanged. `run_venus_tpx1_pipeline` passes the parent directory of the TIFFs it
  loaded, for both the sample and open-beam legs.

  Separately, the pipeline built the `tof` coordinate straight from the sidecar's `shutter_time`
  column, which holds the **left (opening) edge** of each frame's bin — N values for N frames. scipp
  needs N+1 values for a bin-edge axis, so `tof` was a point coordinate and `rebin_by_tof` could not
  operate on a TPX1 result at all. The pipeline now appends the closing edge, extrapolated from the
  last observed step (exact for VENUS's fixed-width TOF grid). **The `tof` coordinate of a TPX1
  result therefore carries one more element than it did in v2.2.3**, and so do the `wavelength` and
  `energy` coordinates the pipeline derives from it — all three are persisted to the HDF5/TIFF
  output, so a reader that assumes a coordinate as long as the frame axis needs updating. That extra
  element is what makes them bin-edge axes; the frame data itself is unchanged. An empty TIFF path
  group now raises `ValueError` naming the problem rather than failing obscurely further in.
- **`get_energy_histogram` mislabelled its energy bins whenever a TOF-dependent coordinate or mask
  was present.** Converting TOF to energy reverses the data, because energy runs opposite to
  time-of-flight, but only the values and variances were reversed — any other TOF-dependent point
  coordinate (such as the `spectra_tof` this release adds) and any TOF-dependent mask kept their
  original order and so labelled the wrong bins. They are now reversed together with the data.
- **`write_hdf5` no longer silently drops masks other than the designated dead/hot pair.** The
  dead and hot masks keep their canonical `/masks/dead` and `/masks/hot` datasets; every other mask
  on the array — for instance a 1-D per-TOF-bin mask — is now written under `/masks/<name>` instead
  of being discarded. This **changes the contents of the primary output format**: an existing reader
  that enumerates `/masks` will find datasets that were not there in v2.2.3. Two new hard failures
  come with it, both raising `ValueError` rather than corrupting the file: a mask name containing
  `/` (which would create unintended nested groups), and a mask whose name collides with a canonical
  path already written.

## [2.2.3] - 2026-07-14

Maintenance release — CI, documentation-build, and dependency updates only. There are
**no `neunorm` library or API changes** since 2.2.2; the PyPI and conda artifacts are
functionally identical to 2.2.2.

### Fixed

- **Read the Docs builds reliably again.** The documentation now builds through pixi
  (`pixi install -e docs` + `pixi run -e docs sphinx-build`, the Read the Docs-official
  route) so it resolves the same locked, tested dependencies as CI instead of a bare
  `pip install .[docs]`, which floated `scitiff`/`pydantic` to an untested combination and
  broke autodoc at import.
  ([#184](https://github.com/ornlneutronimaging/NeuNorm/pull/184))

### Changed

- CI and security hardening: GitHub Actions pinned to full commit SHAs with version
  comments, `setup-pixi` bumped, Grype switched to a non-blocking SARIF upload, lockfile
  update PRs authored via a GitHub App token, and accepted upstream CVEs recorded in
  `.grype.yaml`.
  ([#178](https://github.com/ornlneutronimaging/NeuNorm/pull/178),
  [#179](https://github.com/ornlneutronimaging/NeuNorm/pull/179),
  [#181](https://github.com/ornlneutronimaging/NeuNorm/pull/181),
  [#183](https://github.com/ornlneutronimaging/NeuNorm/pull/183))

## [2.2.2] - 2026-07-02

### Fixed

- **`background_roi` can now reproduce legacy 1.x zero-count-ROI semantics** (completes the
  downstream-superset goal of
  [#172](https://github.com/ornlneutronimaging/NeuNorm/issues/172) /
  [#159](https://github.com/ornlneutronimaging/NeuNorm/issues/159)): the pooled-mean
  strictly-positive/finite guard can be opted out with
  `apply_background_roi(..., strict=False)` or
  `normalize_transmission(...)` / `normalize_with_dark(..., background_roi_strict=False)`,
  letting a zero-count background ROI propagate `inf`/`nan` through the division exactly as
  NeuNorm 1.x did (iBeatles pins this behavior). The default stays strict — a
  non-positive/non-finite pooled mean raises `ValueError` — and structural errors (bad ROI
  bounds, missing dims) always raise.

## [2.2.1] - 2026-07-01

### Added

- **`background_roi` now supports pooled multiple ROIs, inclusive extents, and an open-beam-less
  mode.** `normalize_transmission(..., background_roi=)` — and the MARS CCD/TPX3 and VENUS CCD
  pipelines — accept a **sequence** of ROIs, pooled as `sum(counts) / sum(pixels)` per image;
  `ROI(..., inclusive=True)` opts into legacy inclusive extents (a width-`w` ROI spans `w+1`
  pixels); and the new `neunorm.processing.normalizer.apply_background_roi(data, background_roi)`
  applies the flux proxy to a sample stack with no open beam. A single ROI / bare tuple is
  unchanged. This makes `background_roi` a superset of the downstream (iBeatles) pooled-inclusive
  re-implementation so it can be removed.
  ([#172](https://github.com/ornlneutronimaging/NeuNorm/issues/172),
  [#159](https://github.com/ornlneutronimaging/NeuNorm/issues/159))
- **Python 3.14 support.** NeuNorm builds, installs, and passes its full test suite on
  CPython 3.14; the development/CI pixi environments and `pixi.lock` move to 3.14
  (`requires-python` stays `>=3.11`).

### Fixed

- **TIFF export is compatible with scitiff ≥ 26.6.** scitiff 26.6 tightened its metadata schema
  to reject object-dtype (`PyObject`) scipp variables. `write_tiff_stack` now JSON-encodes
  sequence metadata (mirroring the HDF5 writer's provenance convention — read it back with
  `json.loads`) and drops object-dtype coords/masks (e.g. tuple-valued TIFF header tags carried
  over from the input files) that scitiff cannot serialize. HDF5 output and the written image
  data are unchanged.

## [2.2.0] - 2026-06-30

### Added

- **Background-ROI flux normalization (proton-charge proxy).** `normalize_transmission` gains a
  `background_roi=(x0, y0, x1, y1)` parameter that normalizes each image by its mean counts in a
  sample-free ROI — an approximation to proton-charge normalization for when proton charge is
  unavailable (e.g. MARS): `T = (S/mean(S[B])) / (O/mean(O[B]))`. Mutually exclusive with
  `proton_charge_sample`/`_ob`; uncertainty is propagated first-order (with the shared-dark
  covariance corrected on the `normalize_with_dark` path). Exposed on `run_mars_ccd_pipeline`,
  `run_mars_tpx3_pipeline`, and `run_venus_ccd_pipeline` via `background_roi=`.
  ([#159](https://github.com/ornlneutronimaging/NeuNorm/issues/159))
- **Named `ROI` type for region arguments.** A new `ROI` pydantic model (`neunorm.data_models.roi`) lets you specify any
  region by name — `ROI(x0=10, y0=20, x1=30, y1=40)` or `ROI(x0=10, y0=20, width=20, height=20)` —
  instead of remembering the order of a bare `(x0, y0, x1, y1)` tuple (requested by Jean Bilheux on
  [#159](https://github.com/ornlneutronimaging/NeuNorm/issues/159)). Accepted everywhere an ROI tuple
  is: `apply_roi` (`roi=`), `apply_air_region_correction` (`air_roi=`), `normalize_transmission`
  (`background_roi=`), and every pipeline's `roi` / `air_roi` / `background_roi`. Stops are exclusive
  (`width`/`height` resolve to `x1=x0+width`, `y1=y0+height`); bare tuples keep working unchanged.

- **`EventData` is now indexable, plus an `assign_chip_ids` helper.** `events[mask]` (boolean
  mask, index array, or slice) returns a new `EventData` with every per-event array filtered.
  `neunorm.tof.pulse_reconstruction.assign_chip_ids(x, y, detector_shape)` derives a chip id (0–3)
  from the pixel quadrant for a 2×2 quad Timepix3 detector, giving multi-chip
  `reconstruct_pulse_ids` a data source (the loaders do not record the originating chip).
  ([#163](https://github.com/ornlneutronimaging/NeuNorm/issues/163))

### Fixed

- **`load_event_data` no longer truncates a fractional TOF clock.** The raw-tick → nanosecond
  conversion now multiplies by the full `tof_clock` and rounds to `int64` ns; previously
  `int(tof_clock)` truncated a fractional clock (e.g. the ~1.5625 ns TPX3 fine clock to 1 ns).
  ([#163](https://github.com/ornlneutronimaging/NeuNorm/issues/163))
- **`normalize_transmission` rejects a one-sided proton-charge correction.** Supplying only one of
  `proton_charge_sample` / `proton_charge_ob` produced a non-dimensionless transmission; it now
  raises `ValueError` (both or neither). The pipelines always pass both, so behavior is unchanged.
  ([#163](https://github.com/ornlneutronimaging/NeuNorm/issues/163))
- **`combine_runs` no longer aliases caller data for a single run.** It returns a copy (matching
  the multi-run path), so mutating the combined result cannot leak back into the caller's input.
  ([#163](https://github.com/ornlneutronimaging/NeuNorm/issues/163))

## [2.1.0] - 2026-06-18

### Changed

- **CCD pipelines now compute in float32 end-to-end instead of float64.** The
  TIFF/FITS loaders load image data as `float32`, so `run_mars_ccd_pipeline` and
  `run_venus_ccd_pipeline` propagate, normalize, and return `float32` transmission
  (values and variances), roughly halving the in-memory footprint of large image
  stacks. float32 is sufficient for neutron imaging (16-bit detectors) and matches
  the already-float32 event/TOF path. On-disk HDF5/TIFF output was already written
  as float32, so **file dtypes are unchanged**; written values may differ from
  before by up to ~1e-7 (now computed in float32 rather than rounded from float64).
  ([#147](https://github.com/ornlneutronimaging/NeuNorm/issues/147))
- **Dark current is now optional for the CCD pipelines.** `run_mars_ccd_pipeline`
  and `run_venus_ccd_pipeline` accept `dark_paths=None` (the new default) or an
  empty list to skip dark-current correction; the dark-frame variance then does
  not contribute to the propagated uncertainty. Passing dark frames is unchanged
  and fully backward compatible. Output provenance gains a `dark_correction_applied`
  flag, and `dark_paths` is recorded only when dark frames were supplied.
  ([#146](https://github.com/ornlneutronimaging/NeuNorm/issues/146))

### Fixed

- **HDF5 writer no longer loses ragged provenance (and no longer crashes on it).**
  Nested per-run path metadata (`sample_paths`/`ob_paths`/`dark_paths`) is now stored
  as a round-trippable JSON string tagged with an `encoding="json"` dataset attribute
  — read it back with `json.loads(dataset.asstr()[()])`. Previously, runs with unequal
  file counts produced a ragged nested list that aborted `write_hdf5` *after* the bulk
  arrays were written (corrupt partial file); the interim guard avoided the crash by
  silently **dropping** that provenance. Flat lists, scalars, and strings are
  unchanged. ([#140](https://github.com/ornlneutronimaging/NeuNorm/issues/140))
- **Event-pipeline energy/wavelength binning now applies the detector time offset and a
  configurable flight path.** When `run_venus_tpx3_event_pipeline` histograms directly in
  `bin_space='energy'`/`'wavelength'`, the energy/wavelength bin edges are now built in raw
  detector-TOF space (applying `detector_time_offset`, the exact inverse of the coordinate
  labeling), so events land in the correct bins instead of being shifted by the offset. The
  flight path is now a single configurable `flight_path` parameter (default
  `VENUS_FLIGHT_PATH_M`) used for both binning and labeling, replacing the hardcoded 25 m
  literals. The public `get_energy_histogram` / `get_wavelength_histogram` helpers gained an
  `offset` argument so they label consistently with offset-aware bins, and the
  `run_venus_tpx1_pipeline` / `run_venus_tpx3_histogram_pipeline` pipelines also take a
  configurable `flight_path`. The default bin-in-TOF path is unaffected.
  ([#141](https://github.com/ornlneutronimaging/NeuNorm/issues/141))
- **Shared dark-frame variance is no longer double-counted in CCD transmission
  uncertainty.** With the same averaged dark subtracted from both sample and open beam,
  `T = (S−D)/(O−D)` was propagated as if numerator and denominator were independent, so
  `Var(D)` entered twice. A new `normalize_with_dark` computes the dark correction and
  normalization together and removes the spurious shared-dark covariance term, so the
  reported uncertainty is slightly smaller (the **transmission values are unchanged**). The
  CCD pipelines use it on the with-dark path; the no-dark path is unchanged.
  ([#142](https://github.com/ornlneutronimaging/NeuNorm/issues/142))

## [2.0.0] - 2026-06-09

NeuNorm 2.0 is a complete, [scipp](https://scipp.github.io/)-based rewrite of the
library. It is a **breaking change**: code written against the 1.x
`NeuNorm.normalization.Normalization` API will not run unchanged. See the
[1.x → 2.0 migration guide](docs/migration.md), and pin `NeuNorm<2` to stay on the
legacy API.

### Added

- **Scipp-native processing.** All data are `scipp.DataArray` objects that carry
  variances, so uncertainty is propagated automatically through every step.
- **Time-of-flight (TOF) support** for the VENUS pulsed source — wavelength-resolved
  (hyperspectral) transmission `T(λ, x, y)`, TOF binning, and histogram rebinning.
- **Event-mode processing** for Timepix3 detectors: NeXus/HDF5 event loading and
  pulse reconstruction (`neunorm.loaders.event_loader`, `neunorm.tof`).
- **End-to-end detector pipelines** in `neunorm.pipelines`, one per
  detector/facility combination: `run_mars_ccd_pipeline`, `run_mars_tpx3_pipeline`,
  `run_venus_ccd_pipeline`, `run_venus_tpx1_pipeline`,
  `run_venus_tpx3_histogram_pipeline`, and `run_venus_tpx3_event_pipeline`.
- **Composable processing functions** for building custom workflows: dark
  subtraction, reference (open-beam/dark) preparation, transmission normalization
  with proton-charge correction, ROI clipping, run combination, air-region
  correction, spatial rebinning, and Poisson/systematic uncertainty helpers
  (`neunorm.processing`).
- **HDF5 as the primary output format** (`neunorm.exporters.hdf5_writer.write_hdf5`),
  with detector masks and provenance metadata; TIFF export retained as secondary
  (`neunorm.exporters.tiff_writer.write_tiff_stack`, via scitiff).
- **Loaders** for TIFF, FITS, NeXus event, and NeXus metadata
  (`neunorm.loaders`), including shutter-count and TOF-spectra readers.
- **Resonance / Bragg-edge analysis and TOF statistics** (`neunorm.tof.resonance`,
  `neunorm.tof.statistics_analyzer`).
- **Pydantic v2 configuration models** (e.g. `BinningConfig`, `EventData`) for
  explicit, validated configuration.
- **Sphinx + autodoc documentation** published at
  [neunorm.readthedocs.io](https://neunorm.readthedocs.io), with per-workflow guides
  and a full API reference.

### Changed

- **Import name is now `neunorm` (lowercase)**, not `NeuNorm`. The PyPI/conda
  distribution name remains `NeuNorm`, so `pip install NeuNorm` is unchanged, but
  `import NeuNorm` becomes `import neunorm`.
- **Minimum Python is now 3.11.**
- **Development uses [pixi](https://pixi.sh)** (`pyproject.toml` `[tool.pixi.*]` +
  `pixi.lock`); the 1.x conda `environment.yml` / `conda.recipe` are retired and
  archived under `archive/neunorm-1.x/`.
- **Optional features are exposed as extras**: `viz` (plopp/matplotlib) and
  `performance` (Numba acceleration); Numba is optional and degrades to a no-op
  when absent.

### Removed

- **The entire 1.x stateful API**, including `NeuNorm.normalization.Normalization`
  and `NeuNorm.roi.ROI`. There is no drop-in compatibility shim — the flat-field
  normalization physics is preserved, but the API is new. The 1.x source is kept
  for reference under `archive/neunorm-1.x/`.

### Migration

See the [1.x → 2.0 migration guide](docs/migration.md) for a step-by-step mapping
from the legacy `Normalization` workflow to the 2.0 pipelines and composable
functions.

---

NeuNorm 1.x release history predates this changelog. The 1.x source, tests, and
documentation are archived under
[`archive/neunorm-1.x/`](archive/neunorm-1.x/); released 1.x versions remain
available on PyPI and the `conda-forge` channel (`pip install "NeuNorm<2"`).

[2.4.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.4.0
[2.3.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.3.0
[2.2.3]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.3
[2.2.2]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.2
[2.2.1]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.1
[2.2.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.0
[2.1.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.1.0
[2.0.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.0.0
