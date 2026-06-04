# MARS CCD/CMOS Data Reduction Workflow

**Beamline**: MARS (HFIR)
**Detector**: CCD/CMOS camera
**Beam Type**: Continuous (no TOF)
**Applications**: nR (radiography), nCT (computed tomography), nGI (grating interferometry)

---

## Pipeline Flowchart

```mermaid
flowchart TD
    subgraph Input["1. Data Loading"]
        A1[TIFF/FITS] --> A[Load Sample]
        A2[TIFF/FITS] --> B[Load Open Beam]
        A3[TIFF/FITS] --> C[Load Dark Current]
    end

    subgraph RunCombine["2. Run Combining"]
        RC1{Multiple Runs?}
        RC2[Aggregate Data]
        RC3[Single Run]
    end

    subgraph ROI["3. ROI Clipping"]
        D{ROI Specified?}
        E[Apply ROI]
        F[Full Frame]
    end

    subgraph Prepare["4. Reference Preparation"]
        G[Average Dark → 2D]
        H[Average OB → 2D]
    end

    subgraph PixelDetect["5. Dead Pixel Detection"]
        PD[Identify Zero-Count Pixels]
        PM[Dead Pixel Mask]
    end

    subgraph Gamma["6. Gamma Filtering"]
        GF[Detect Gamma Spikes]
        GR[Replace with Median]
    end

    subgraph DarkCorr["7. Dark Correction"]
        DC1["Sample_corr = Sample - Dark"]
        DC2["OB_corr = OB - Dark"]
    end

    subgraph Norm["8. Normalization"]
        N["T = Sample_corr / OB_corr"]
    end

    subgraph UQ["9. Experiment Error"]
        UQ1[Poisson Statistics]
        UQ2[Error Propagation]
    end

    subgraph Output["10. Output"]
        O1[Transmission 3D]
        O2[Uncertainty 3D]
        O3[Dead Pixel Mask]
        O4[Metadata]
    end

    Input --> RC1
    RC1 -->|Yes| RC2
    RC1 -->|No| RC3
    RC2 --> D
    RC3 --> D
    D -->|Yes| E
    D -->|No| F
    E --> Prepare
    F --> Prepare
    G --> PD
    H --> PD
    PD --> PM
    PM --> Gamma
    GF --> GR
    Gamma --> DarkCorr
    DC1 --> N
    DC2 --> N
    N --> UQ1
    UQ1 --> UQ2
    UQ2 --> O1
    UQ2 --> O2
    PM --> O3
    O1 --> O4
    O2 --> O4
    O3 --> O4

    style Input fill:#e1f5ff
    style RunCombine fill:#f5e1ff
    style ROI fill:#fff4e1
    style Prepare fill:#e1ffe8
    style PixelDetect fill:#ffe1e1
    style Gamma fill:#ffe1cc
    style DarkCorr fill:#e1ffe1
    style Norm fill:#e1ffe1
    style UQ fill:#ffe1cc
    style Output fill:#f5e1ff
```

---

## 1. Inputs

| Input | Format | Required | Description |
|-------|--------|----------|-------------|
| Sample images | TIFF/FITS stack | Yes | Raw neutron transmission images |
| Open Beam (OB) | TIFF/FITS stack | Yes | Reference without sample |
| Dark Current | TIFF/FITS stack | Yes | Electronic noise baseline (beam off) |
| ROI | (x0, y0, x1, y1) | No | Region of interest to crop |

**Metadata** (from files or user):
- Acquisition time per image
- Detector gain settings

---

## 2. Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Data                                              │
│  ────────────────                                               │
│  • Load Sample stack → 3D array (N_images, y, x)                │
│  • Load OB stack → 3D array (N_ob, y, x)                        │
│  • Load Dark Current stack → 3D array (N_dark, y, x)            │
│  • Validate dimensions match (y, x must be same)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Run Combining (Optional)                               │
│  ────────────────────────────────                               │
│  IF multiple runs provided:                                     │
│    • Aggregate sample images across runs                        │
│    • Aggregate OB images across runs                            │
│    • Aggregate dark images across runs                          │
│    • Sum metadata (total acquisition time)                      │
│    • Track partial dead pixels per run                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: ROI Clipping (Optional)                                │
│  ───────────────────────────────                                │
│  IF ROI specified:                                              │
│    • Crop all arrays to ROI: arr[:, y0:y1, x0:x1]               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Prepare Reference Images                               │
│  ────────────────────────────────                               │
│  • Average dark images: Dark_avg = mean(Dark, axis=0) → 2D      │
│  • Average OB images: OB_avg = mean(OB, axis=0) → 2D            │
│  • (Or use median for robustness against outliers)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Dead Pixel Detection                                   │
│  ────────────────────────────                                   │
│  • Identify pixels with persistent zeros in OB_avg              │
│  • dead_mask = (OB_avg == 0) | (OB_avg - Dark_avg <= 0)         │
│  • Output: 2D boolean mask                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Gamma Filtering                                        │
│  ───────────────────────                                        │
│  CRITICAL for MARS (SANS beamline contamination)                │
│                                                                 │
│  FOR each image in Sample stack:                                │
│    • Detect gamma spikes (outliers > threshold)                 │
│    • Replace with local median (3x3 neighborhood)               │
│                                                                 │
│  Apply same filtering to OB_avg                                 │
│                                                                 │
│  Methods:                                                       │
│    a) Automatic: threshold = data_max * factor                  │
│    b) Manual: user-specified threshold                          │
│    c) Statistical: z-score based outlier detection              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Dark Current Correction                                │
│  ───────────────────────────────                                │
│  FOR each image i in Sample stack:                              │
│    Sample_corr[i] = Sample[i] - Dark_avg                        │
│                                                                 │
│  OB_corr = OB_avg - Dark_avg                                    │
│                                                                 │
│  Handle negative values:                                        │
│    • Clip to zero OR                                            │
│    • Flag as invalid                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Normalization                                          │
│  ─────────────────────                                          │
│  FOR each image i:                                              │
│                                                                 │
│    T[i] = Sample_corr[i] / OB_corr                              │
│                                                                 │
│  Handle division:                                               │
│    • Where dead_mask=True: T = NaN                              │
│    • Where OB_corr <= 0: T = NaN                                │
│                                                                 │
│  Formula:                                                       │
│    T = (I_sample - I_dark) / (I_OB - I_dark)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9: Experiment Error Propagation                           │
│  ────────────────────────────────────                           │
│  Poisson statistics for CCD counts:                             │
│    σ_sample = √(Sample)                                         │
│    σ_OB = √(OB_avg)                                             │
│    σ_dark = √(Dark_avg)                                         │
│                                                                 │
│  Error propagation through subtraction and division:            │
│                                                                 │
│    σ_T = T × √[ (σ_S/S_corr)² + (σ_OB/OB_corr)² +               │
│                 (σ_D)²×(1/S_corr² + 1/OB_corr²) ]               │
│                                                                 │
│  Where:                                                         │
│    S_corr = Sample - Dark                                       │
│    OB_corr = OB - Dark                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 10: Output                                                │
│  ────────────                                                   │
│  • Transmission: 3D array (N_images, y, x) or (θ, y, x) for CT  │
│  • Experiment Error: 3D array (same shape as Transmission)      │
│  • Dead Pixel Mask: 2D boolean array (y, x)                     │
│  • Metadata: processing parameters, provenance                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Output Specification

| Output | Dimensions | dtype | Description |
|--------|------------|-------|-------------|
| Transmission | (θ, y, x) | float32 | Normalized transmission values |
| Experiment Error | (θ, y, x) | float32 | Propagated uncertainty (1σ) |
| Dead Pixel Mask | (y, x) | bool | True = dead pixel |
| Metadata | dict | - | Processing provenance |

**Metadata contents**:
- Input file paths
- Processing timestamp
- Gamma filter parameters used
- ROI applied (if any)
- Number of runs combined (if any)
- Software version

---

## 4. Decision Points

| Step | Decision | Options |
|------|----------|---------|
| 2 | Multiple runs? | Combine or single run |
| 3 | ROI needed? | Apply crop or full frame |
| 4 | OB averaging | Mean vs Median |
| 6 | Gamma filter method | Automatic / Manual / Statistical |
| 7 | Negative value handling | Clip to zero / Flag invalid |

---

## 5. Development Components

### Required Modules

| Component | Purpose | Priority |
|-----------|---------|----------|
| `loaders.tiff_loader` | Load TIFF stacks | P0 |
| `loaders.fits_loader` | Load FITS stacks | P0 |
| `processing.run_combiner` | Aggregate multiple runs | P1 |
| `processing.roi_clipper` | Apply ROI to arrays | P1 |
| `processing.dead_pixel_detector` | Identify dead pixels | P0 |
| `filters.gamma_filter` | Remove gamma contamination | P0 |
| `processing.dark_corrector` | Subtract dark current | P0 |
| `processing.normalizer` | Compute transmission | P0 |
| `processing.uncertainty_calculator` | Error propagation | P0 |
| `exporters.output_writer` | Write results | P0 |

### Data Models

```
InputData:
  - sample: NDArray[float32]  # (N, y, x)
  - open_beam: NDArray[float32]  # (N_ob, y, x)
  - dark_current: NDArray[float32]  # (N_dark, y, x)
  - roi: Optional[Tuple[int, int, int, int]]
  - metadata: Dict

ProcessedData:
  - transmission: NDArray[float32]  # (N, y, x)
  - uncertainty: NDArray[float32]  # (N, y, x)
  - dead_pixel_mask: NDArray[bool]  # (y, x)
  - metadata: Dict
```

---

## 6. Validation Criteria

- [ ] Transmission values in expected range (typically 0-1, may exceed 1 due to scattering)
- [ ] No NaN values except where dead_mask=True
- [ ] Uncertainty > 0 for all valid pixels
- [ ] Dead pixel mask correctly identifies zero-count pixels
- [ ] Gamma filtering removes spikes without affecting valid data
