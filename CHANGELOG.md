# Changelog

All notable changes to NeuNorm are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[2.1.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.1.0
[2.0.0]: https://github.com/ornlneutronimaging/NeuNorm/releases/tag/v2.0.0
