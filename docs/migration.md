# Migrating from NeuNorm 1.x to 2.0

NeuNorm 2.0 is a complete, [scipp](https://scipp.github.io/)-based rewrite. It is a
**breaking change**: code written against the 1.x
`NeuNorm.normalization.Normalization` API will not run unchanged, and there is **no
drop-in compatibility shim**. The flat-field normalization *physics* is preserved,
but the *API* is new.

```{note}
If you are not ready to migrate, pin the legacy line: `pip install "NeuNorm<2"`.
The 1.x source is archived under
[`archive/neunorm-1.x/`](https://github.com/ornlneutronimaging/NeuNorm/tree/main/archive/neunorm-1.x).
```

## What changed at a glance

| | 1.x | 2.0 |
|---|---|---|
| Import | `import NeuNorm` | `import neunorm` (lowercase) |
| Distribution name | `NeuNorm` | `NeuNorm` (unchanged — `pip install NeuNorm`) |
| Minimum Python | 2.7 / 3.x | **3.11+** |
| Core data type | NumPy arrays | `scipp.DataArray` (carries variances) |
| Uncertainty | not propagated | **propagated automatically** |
| API style | one stateful `Normalization` object | **functions + per-detector pipelines** |
| Primary output | TIFF | **HDF5** (TIFF secondary) |
| TOF / event mode | none | **first-class** (VENUS, Timepix) |
| Dev environment | conda `environment.yml` | **pixi** (`pixi.lock`) |

## The fastest path: use a pipeline

In 1.x you drove a stateful object through `load` → `normalization` → `export`.
In 2.0 the equivalent is a single **pipeline** call for your detector/facility,
which loads, corrects, normalizes, and writes the result (with uncertainty, masks,
and provenance) to HDF5.

**1.x**

```python
from NeuNorm.normalization import Normalization

o_norm = Normalization()
o_norm.load(file="sample.tif", data_type="sample")
o_norm.load(folder="/data/ob/", data_type="ob")
o_norm.load(folder="/data/df/", data_type="df")   # dark / "df" frames
o_norm.normalization()
o_norm.export(folder="/data/normalized/", data_type="normalized")
normalized = o_norm.get_normalized_data()           # NumPy array
```

**2.0** (MARS CCD/CMOS — continuous source, the closest analogue to the 1.x CCD case)

```python
from pathlib import Path
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

# Each inner list is one acquisition "run" to combine before processing.
transmission = run_mars_ccd_pipeline(
    sample_paths=[["sample_0001.tif", "sample_0002.tif"]],
    ob_paths=[["ob_0001.tif", "ob_0002.tif"]],
    dark_paths=[["dark_0001.tif"]],
    output_path=Path("/data/normalized.hdf5"),
    roi=(x0, y0, x1, y1),   # optional
    gamma_filter=True,      # on by default
)

# `transmission` is a scipp.DataArray; the NumPy values are transmission.values
```

Pick the pipeline that matches your detector and facility:

| Pipeline | Detector | Facility | TOF |
|---|---|---|---|
| `run_mars_ccd_pipeline` | CCD/CMOS | MARS (HFIR) | no |
| `run_mars_tpx3_pipeline` | Timepix3 | MARS (HFIR) | no |
| `run_venus_ccd_pipeline` | CCD/CMOS | VENUS (SNS) | no |
| `run_venus_tpx1_pipeline` | Timepix1 | VENUS (SNS) | yes |
| `run_venus_tpx3_histogram_pipeline` | Timepix3 | VENUS (SNS) | yes |
| `run_venus_tpx3_event_pipeline` | Timepix3 (event) | VENUS (SNS) | yes |

These share the same *flow* but not the same *signature* — TPX detectors omit
`dark_paths`, the TOF pipelines add `rebin_by_tof`/`rebin_by_spatial`, and
`run_venus_tpx3_event_pipeline` takes a `BinningConfig` and flat (per-run) path
lists. Check each function's signature in the {doc}`api` reference.

## The composable path: build your own workflow

If you need finer control than a pipeline offers, 2.0 exposes the individual steps
as functions (the pipelines themselves are just compositions of these — read a
pipeline's source for the exact ordering). The core operation is
`normalize_transmission`:

```python
from neunorm.loaders.stack_loader import load_stack
from neunorm.processing.normalizer import normalize_transmission

sample = load_stack(["sample_0001.tif", "sample_0002.tif"])  # scipp.DataArray
ob = load_stack(["ob_0001.tif", "ob_0002.tif"])

transmission = normalize_transmission(sample, ob)  # T = sample / ob, variances tracked
```

## Method / concept mapping

| 1.x | 2.0 |
|---|---|
| `Normalization()` (stateful object) | a `run_*_pipeline(...)` call, or composable functions |
| `.load(..., data_type="sample"/"ob")` | pipeline `sample_paths` / `ob_paths`, or `neunorm.loaders.stack_loader.load_stack` / `tiff_loader.load_tiff_stack` / `fits_loader.load_fits_stack` |
| `.load(..., data_type="df")` (dark/"df") | pipeline `dark_paths`, or `neunorm.processing.dark_corrector.subtract_dark(data, dark)` |
| `.normalization(roi=...)` | `run_*_pipeline(..., roi=...)`, or `neunorm.processing.normalizer.normalize_transmission(sample, ob, ...)` |
| `.df_correction()` | `neunorm.processing.dark_corrector.subtract_dark(data, dark)` |
| `.crop(roi=ROI(...))` | `neunorm.processing.roi_clipper.apply_roi(data, (x0, y0, x1, y1))` |
| auto/manual gamma filtering on `load()` | `neunorm.filters.gamma_filter.apply_gamma_filter(...)`, or pipeline `gamma_filter=True` |
| `.export(folder=..., file_type="tif")` | `neunorm.exporters.hdf5_writer.write_hdf5(...)` (primary) or `tiff_writer.write_tiff_stack(...)`; pipelines write automatically via `output_path` |
| `.get_normalized_data()` | the pipeline **returns** a `scipp.DataArray`; use `.values` for the NumPy array |
| `from NeuNorm.roi import ROI; ROI(x0, y0, x1, y1)` | a plain tuple `(x0, y0, x1, y1)` |
| `NeuNorm.normalization.DataType` | not needed — inputs are explicit function arguments |

## ROI

1.x used an `ROI` object; 2.0 uses a 4-integer tuple in the same order:

```python
# 1.x
from NeuNorm.roi import ROI
roi = ROI(x0=10, y0=10, x1=110, y1=110)

# 2.0
roi = (10, 10, 110, 110)   # (x0, y0, x1, y1), pixel bounds
```

## Working with the result

A 2.0 pipeline returns a `scipp.DataArray` instead of a NumPy array:

```python
transmission.values        # NumPy array of transmission values
transmission.variances     # propagated variances (None in 1.x — not tracked)
transmission.coords        # axis coordinates (e.g. wavelength for TOF data)
transmission.masks         # detector masks (e.g. dead pixels)
```

The normalized result is also written to `output_path` as HDF5 (and optionally
TIFF), so downstream tools can read it directly without re-running the pipeline.

## Next steps

- {doc}`api` — full API reference for every loader, processing function, and pipeline.
- The per-workflow guides under {doc}`workflows/README` walk through each
  detector/facility combination end to end.
