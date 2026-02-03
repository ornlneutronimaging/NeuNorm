# NeuNorm 2.0 Data Reduction Workflows

This directory contains detector-centric data reduction workflows for scoping NeuNorm 2.0 development.

---

## Workflow Index

### MARS (HFIR) - Priority 1

| Workflow | Detector | TOF | Complexity |
|----------|----------|-----|------------|
| [MARS CCD/CMOS](mars_ccd_cmos.md) | CCD/CMOS | No | Low |
| [MARS TPX3](mars_tpx3.md) | Timepix3 | No | Medium |

### VENUS (SNS) - Priority 2

| Workflow | Detector | TOF | Complexity |
|----------|----------|-----|------------|
| [VENUS CCD/CMOS](venus_ccd_cmos.md) | CCD/CMOS | No | Medium |
| [VENUS TPX1](venus_tpx1.md) | Timepix1 | Yes | Medium |
| [VENUS TPX3](venus_tpx3.md) | Timepix3 | Yes | High |
| [VENUS TPX4](venus_tpx4.md) | Timepix4 | Yes | High (planned) |

---

## Quick Reference: Key Differences

### By Input Data Type

| Data Type | Workflows |
|-----------|-----------|
| TIFF/FITS histogram | MARS CCD/CMOS, VENUS CCD/CMOS, VENUS TPX1 |
| Event files (HDF5) | MARS TPX3, VENUS TPX3, VENUS TPX4 |

### By Processing Requirements

| Requirement | Workflows |
|-------------|-----------|
| Dark current correction | MARS CCD/CMOS, VENUS CCD/CMOS |
| Gamma filtering (critical) | MARS CCD/CMOS, MARS TPX3 |
| Hot pixel detection | MARS TPX3, VENUS TPX3, VENUS TPX4 |
| Beam stability correction | VENUS CCD/CMOS, VENUS TPX1, VENUS TPX3, VENUS TPX4 |
| TOF binning | VENUS TPX1, VENUS TPX3, VENUS TPX4 |
| Pulse reconstruction | VENUS TPX3, VENUS TPX4 |
| Event-to-histogram | MARS TPX3, VENUS TPX3, VENUS TPX4 |

---

## Module Requirements Summary

### Shared Modules (All Workflows)

| Module | Description |
|--------|-------------|
| `processing.normalizer` | Compute transmission T = Sample/OB |
| `processing.uncertainty_calculator` | Error propagation |
| `processing.dead_pixel_detector` | Identify zero-count pixels |
| `processing.roi_clipper` | Apply region of interest |
| `processing.run_combiner` | Aggregate multiple acquisitions |
| `exporters.output_writer` | Write results |

### Histogram Loaders

| Module | Used By |
|--------|---------|
| `loaders.tiff_loader` | CCD/CMOS, TPX1 |
| `loaders.fits_loader` | CCD/CMOS, TPX1 |

### Event Processing

| Module | Used By |
|--------|---------|
| `loaders.event_loader` | TPX3, TPX4 |
| `tof.event_converter` | TPX3, TPX4 |
| `tof.pulse_reconstruction` | VENUS TPX3, TPX4 |

### Corrections

| Module | Used By |
|--------|---------|
| `processing.dark_corrector` | CCD/CMOS only |
| `processing.beam_corrector` | VENUS all |
| `filters.gamma_filter` | MARS all |
| `processing.hot_pixel_detector` | TPX3, TPX4 |

### TOF Processing

| Module | Used By |
|--------|---------|
| `tof.statistics_analyzer` | TPX1, TPX3, TPX4 |
| `tof.binning_recommender` | TPX1, TPX3, TPX4 |
| `tof.rebinner` | TPX1, TPX3, TPX4 |
| `tof.coordinate_converter` | TPX1, TPX3, TPX4 |

---

## Implementation Order Recommendation

Based on complexity and shared dependencies:

### Phase 1: Core Infrastructure
1. Data models (Pydantic)
2. TIFF/FITS loaders
3. Normalizer
4. Uncertainty calculator
5. Dead pixel detector
6. Output writer

### Phase 2: MARS CCD/CMOS (Simplest Complete Pipeline)
1. Dark corrector
2. Gamma filter
3. Integration: MARS CCD/CMOS end-to-end

### Phase 3: MARS TPX3
1. Event loader
2. Event-to-histogram converter
3. Hot pixel detector
4. Integration: MARS TPX3 end-to-end

### Phase 4: VENUS CCD/CMOS
1. Beam corrector (p_charge, ROI-based)
2. Metadata loader (DAQ)
3. Integration: VENUS CCD/CMOS end-to-end

### Phase 5: VENUS TPX1
1. TOF statistics analyzer
2. Binning recommender
3. Rebinner (uniform + heterogeneous)
4. Integration: VENUS TPX1 end-to-end

### Phase 6: VENUS TPX3
1. Pulse loader
2. Pulse reconstruction (JIT)
3. Integration: VENUS TPX3 end-to-end

### Phase 7: VENUS TPX4
1. Validate with TPX4 data
2. Adjust loaders if needed

---

## Output Summary

| Workflow | Transmission | Error | TOF Edges | Dead Mask | Hot Mask |
|----------|--------------|-------|-----------|-----------|----------|
| MARS CCD/CMOS | 3D | 3D | - | 2D | - |
| MARS TPX3 | 3D | 3D | - | 2D | 2D |
| VENUS CCD/CMOS | 3D | 3D | - | 2D | - |
| VENUS TPX1 | 4D | 4D | 1D | 2D | - |
| VENUS TPX3 | 4D | 4D | 1D | 2D | 2D |
| VENUS TPX4 | 4D | 4D | 1D | 2D | 2D |

---

## Document History

- **Created**: 2026-02-03
- **Purpose**: Development scoping for NeuNorm 2.0 refactoring
- **Organization**: Detector-centric per Jean Bilheux feedback
