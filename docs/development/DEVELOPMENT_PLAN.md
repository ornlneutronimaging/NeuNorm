# NeuNorm v2.0 Development Plan

## Overview

Phased development plan to complete NeuNorm v2.0, building on existing codebase to support 5 detector workflows across MARS (HFIR) and VENUS (SNS) beamlines.

## Current State Assessment

### Already Implemented (`/NeuNorm/src/NeuNorm/`)
| Module | Status | Description |
|--------|--------|-------------|
| `data_models/core.py` | Complete | EventData (Pydantic v2) |
| `data_models/tof.py` | Complete | BinningConfig |
| `loaders/event_loader.py` | Complete | HDF5 TPX3/TPX4 events |
| `tof/pulse_reconstruction.py` | Complete | Pulse ID with Numba JIT |
| `tof/event_converter.py` | Complete | Events→3D histogram |
| `tof/pixel_detector.py` | Complete | Dead/hot pixel detection |
| `processing/normalizer.py` | Complete | T=Sample/OB with p_charge |
| `processing/uncertainty_calculator.py` | Complete | Poisson + systematic |

### NOT Implemented (Required)
- TIFF/FITS loaders
- Dark current correction
- Gamma filtering
- Air region correction
- Run combiner
- ROI clipper
- Statistics analyzer / Histogram rebinner
- Output writers

---

## Development Phases

### Phase 0: Core Infrastructure (2-3 weeks)

**Goal**: Complete foundational modules required by ALL workflows

| Module | File to Create | Description |
|--------|---------------|-------------|
| TIFF Loader | `loaders/tiff_loader.py` | Load TIFF stacks → sc.DataArray |
| FITS Loader | `loaders/fits_loader.py` | Load FITS stacks → sc.DataArray |
| Dark Corrector | `processing/dark_corrector.py` | Subtract dark with variance propagation |
| Gamma Filter | `filters/gamma_filter.py` | Remove gamma contamination (critical for MARS) |
| HDF5 Writer | `exporters/hdf5_writer.py` | Write transmission + uncertainty |
| TIFF Writer | `exporters/tiff_writer.py` | Write TIFF stacks |

**API Patterns**:
```python
# Loaders return scipp DataArrays with metadata
def load_tiff_stack(paths: list[Path], tof_edges: Optional[np.ndarray] = None) -> sc.DataArray

# Processing modules operate on scipp DataArrays
def subtract_dark(data: sc.DataArray, dark: sc.DataArray, clip_negative: bool = True) -> sc.DataArray
```

---

### Phase 1: MARS CCD/CMOS Pipeline (2 weeks)

**Goal**: First complete end-to-end pipeline (simplest workflow)

**New Modules**:
| Module | File | Description |
|--------|------|-------------|
| Run Combiner | `processing/run_combiner.py` | Sum histograms, aggregate metadata |
| ROI Clipper | `processing/roi_clipper.py` | Crop spatial dimensions |
| Pipeline | `pipelines/mars_ccd.py` | 10-step orchestration |

**Pipeline Steps** (from `docs/workflows/mars_ccd_cmos.md`):
1. Load TIFF/FITS (sample, OB, dark)
2. Run combine (optional)
3. ROI clip (optional)
4. Average dark/OB
5. Dead pixel detection (existing)
6. Gamma filtering (Phase 0)
7. Dark correction (Phase 0)
8. Normalization (existing)
9. Error propagation (existing)
10. Output (Phase 0)

---

### Phase 2: MARS TPX3 Pipeline (1-2 weeks)

**Goal**: Add event-mode support for MARS (no TOF, 2D histogram)

**Updates**:
- Add `convert_events_to_2d_histogram()` to `tof/event_converter.py`
- Create `pipelines/mars_tpx3.py`

**Key Differences from CCD**:
- Event loading instead of TIFF
- Hot pixel detection (existing)
- No dark correction (counting detector)

---

### Phase 3: VENUS CCD/CMOS Pipeline (1-2 weeks)

**Goal**: Add beam correction for pulsed source

**New Modules**:
| Module | File | Description |
|--------|------|-------------|
| Air Region Corrector | `processing/air_region_corrector.py` | Scale so air region = 1.0 |
| Metadata Loader | `loaders/metadata_loader.py` | Extract p_charge from DAQ |
| Pipeline | `pipelines/venus_ccd.py` | 12-step orchestration |

**Key Additions vs MARS**:
- p_charge beam correction (existing in normalizer)
- Air region correction (optional post-normalization)

---

### Phase 4: VENUS TPX1 Pipeline (2 weeks)

**Goal**: Add histogram-mode TOF support with rebinning

**New Modules**:
| Module | File | Description |
|--------|------|-------------|
| Statistics Analyzer | `tof/statistics_analyzer.py` | SNR per TOF bin, rebinning recommendation |
| Histogram Rebinner | `tof/histogram_rebinner.py` | Combine N adjacent bins |
| Spatial Rebinner | `processing/spatial_rebinner.py` | NxN pixel binning |
| Pipeline | `pipelines/venus_tpx1.py` | 11-step orchestration |

**Key Features**:
- Load pre-binned TIFF stacks with TOF dimension
- Analyze statistics, recommend rebinning
- Constrained rebinning (combine adjacent bins only)

---

### Phase 5: VENUS TPX3 Pipeline (2-3 weeks)

**Goal**: Complete most complex pipeline (event mode + histogram mode)

**Integration** - Most components exist, need orchestration:
- Event mode: 15-step pipeline using all modules
- Histogram mode: 12-step pipeline (simpler, no pulse reconstruction)

**Files to Create**:
- `pipelines/venus_tpx3_event.py`
- `pipelines/venus_tpx3_histogram.py`

---

## Module Dependencies

```
data_models (complete)
    │
    ├── loaders
    │   ├── event_loader (complete)
    │   ├── tiff_loader (Phase 0)
    │   ├── fits_loader (Phase 0)
    │   └── metadata_loader (Phase 3)
    │
    ├── processing
    │   ├── normalizer (complete)
    │   ├── uncertainty_calculator (complete)
    │   ├── dark_corrector (Phase 0)
    │   ├── run_combiner (Phase 1)
    │   ├── roi_clipper (Phase 1)
    │   ├── air_region_corrector (Phase 3)
    │   └── spatial_rebinner (Phase 4)
    │
    ├── filters
    │   └── gamma_filter (Phase 0)
    │
    ├── tof
    │   ├── pulse_reconstruction (complete)
    │   ├── event_converter (complete)
    │   ├── pixel_detector (complete)
    │   ├── statistics_analyzer (Phase 4)
    │   └── histogram_rebinner (Phase 4)
    │
    ├── exporters
    │   ├── hdf5_writer (Phase 0)
    │   └── tiff_writer (Phase 0)
    │
    └── pipelines (Phase 1-5)
```

---

## Testing Strategy

### Unit Tests (per module)
```
tests/unit/
  test_tiff_loader.py
  test_dark_corrector.py
  test_gamma_filter.py
  test_run_combiner.py
  test_statistics_analyzer.py
  test_histogram_rebinner.py
```

### Integration Tests (per pipeline)
```
tests/integration/
  test_mars_ccd_pipeline.py
  test_mars_tpx3_pipeline.py
  test_venus_ccd_pipeline.py
  test_venus_tpx1_pipeline.py
  test_venus_tpx3_pipeline.py
```

### Validation Criteria (from workflow docs)
- Transmission values in range [0, 1] (may exceed due to scattering)
- No NaN except where masks indicate
- Uncertainty > 0 for all valid pixels
- Total counts preserved through rebinning
- Beam correction factor ~1.0

---

## Documentation Integration

Copy workflow docs from `neunorm_refacotr/docs/workflows/` into NeuNorm repo:
```
NeuNorm/docs/
  workflows/
    mars_ccd_cmos.md
    mars_tpx3.md
    venus_ccd_cmos.md
    venus_tpx1.md
    venus_tpx3.md
```

---

## Timeline Summary

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Phase 0 | 2-3 weeks | Core infrastructure complete |
| Phase 1 | 2 weeks | MARS CCD/CMOS pipeline working |
| Phase 2 | 1-2 weeks | MARS TPX3 pipeline working |
| Phase 3 | 1-2 weeks | VENUS CCD/CMOS pipeline working |
| Phase 4 | 2 weeks | VENUS TPX1 pipeline working |
| Phase 5 | 2-3 weeks | VENUS TPX3 pipelines working |

**Total**: 10-14 weeks

---

## Critical Files Reference

**Patterns to follow**:
- `NeuNorm/src/NeuNorm/loaders/event_loader.py` - Loader pattern
- `NeuNorm/src/NeuNorm/processing/normalizer.py` - Processing module pattern
- `NeuNorm/src/NeuNorm/tof/pixel_detector.py` - Detection algorithm pattern

**Legacy code for reference**:
- `NeuNorm/archive/neunorm-1.x/NeuNorm/normalization.py` - TIFF loading, gamma filtering patterns

**Workflow specifications**:
- `docs/workflows/mars_ccd_cmos.md` - Phase 1 target
- `docs/workflows/venus_tpx3.md` - Phase 5 target (most complex)

---

## Verification

After each phase, verify:
1. Unit tests pass for new modules
2. Integration test passes for pipeline
3. Output matches validation criteria from workflow docs
4. Documentation updated with examples
