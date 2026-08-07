# Changelog

All notable changes to NeuNorm are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
  (shape, selected-pixel count, source, sha256), including `air_roi`. Rectangle-only inputs are
  bit-identical to 2.2.x.

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

[2.2.2]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.2
[2.2.1]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.1
[2.2.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.2.0
[2.1.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.1.0
[2.0.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.0.0
