# VENUS TPX1 Data Reduction Workflow

**Beamline**: VENUS (SNS)
**Detector**: Timepix1 (TPX1)
**Beam Type**: Pulsed with TOF
**Data Mode**: Histogram (frame-mode, TOF bins fixed at acquisition)
**Input Format**: TIFF stacks (efficiency-corrected)
**Applications**: Bragg edge imaging, resonance imaging, hyperspectral nCT

---

## Data Flow Overview

```
EXTERNAL (Auto-Reduction):
  TPX1 Detector → FITS (raw histogram) → Efficiency correction → TIFF stacks

NeuNorm Pipeline:
  Input: TIFF stacks (TOF, y, x) - efficiency-corrected, TOF bins fixed
```

---

## Pipeline Flowchart

```mermaid
flowchart TD
    subgraph Input["1. Data Loading (TIFF)"]
        A1[TIFF Stack] --> A[Load Sample TOF,y,x]
        A2[TIFF Stack] --> B[Load OB TOF,y,x]
        A3[Metadata] --> C[Load TOF Bin Edges]
        A4[DAQ] --> M[Load p_charge, shutter_counts]
    end

    subgraph RunCombine["2. Run Combining"]
        RC1{Multiple Runs?}
        RC2[Sum Histograms + Metadata]
        RC3[Single Run]
    end

    subgraph ROI["3. ROI Clipping"]
        D{ROI Specified?}
        E[Apply Spatial ROI]
        F[Full Frame]
    end

    subgraph PixelDetect["4. Dead Pixel Detection"]
        PD[Sum OB over TOF → Identify Zeros]
        PM[Dead Pixel Mask 2D]
    end

    subgraph Stats["5. Statistics Analysis"]
        ST1[Count per TOF Bin]
        ST2[SNR Analysis]
        ST3[Rebinning Recommendation]
    end

    subgraph Rebin["6. Rebinning (Optional)"]
        RB1{Rebin?}
        RB2[Combine N Adjacent TOF Bins]
        RB3[Spatial Binning NxN]
        RB4[Keep Original]
    end

    subgraph BeamCorr["7. Beam Correction"]
        BC1["f = p_charge_OB / p_charge_Sample"]
        BC2["Shutter counts correlation check"]
    end

    subgraph Norm["8. Normalization"]
        N["T(TOF) = Sample(TOF) / OB(TOF) × f"]
    end

    subgraph AirCorr["9. Air Region Correction (Optional)"]
        AC1{Air ROI?}
        AC2["T_final = T / mean(T_air)"]
        AC3[Skip]
    end

    subgraph UQ["10. Experiment Error"]
        UQ1[Poisson per TOF bin]
        UQ2[p_charge σ]
        UQ3[Error Propagation]
    end

    subgraph Output["11. Output"]
        O1[Transmission 4D θ,TOF,y,x]
        O2[Uncertainty 4D]
        O3[TOF Bin Edges]
        O4[Dead Pixel Mask]
        O5[Metadata]
    end

    Input --> RC1
    RC1 -->|Yes| RC2
    RC1 -->|No| RC3
    RC2 --> D
    RC3 --> D
    D -->|Yes| E
    D -->|No| F
    E --> PD
    F --> PD
    PD --> PM
    PM --> ST1
    ST1 --> ST2
    ST2 --> ST3
    ST3 --> RB1
    RB1 -->|TOF| RB2
    RB1 -->|Spatial| RB3
    RB1 -->|No| RB4
    RB2 --> BC1
    RB3 --> BC1
    RB4 --> BC1
    BC1 --> BC2
    BC2 --> N
    N --> AC1
    AC1 -->|Yes| AC2
    AC1 -->|No| AC3
    AC2 --> UQ1
    AC3 --> UQ1
    UQ1 --> UQ2
    UQ2 --> UQ3
    UQ3 --> O1
    UQ3 --> O2
    UQ3 --> O3
    PM --> O4
    O1 --> O5
    O2 --> O5
    O3 --> O5
    O4 --> O5

    style Input fill:#e1f5ff
    style RunCombine fill:#f5e1ff
    style ROI fill:#fff4e1
    style PixelDetect fill:#ffe1e1
    style Stats fill:#e1f5ff
    style Rebin fill:#ffe1f5
    style BeamCorr fill:#e1f5ff
    style Norm fill:#e1ffe1
    style AirCorr fill:#fff4e1
    style UQ fill:#ffe1cc
    style Output fill:#f5e1ff
```

---

## 1. Inputs

| Input | Format | Required | Description |
|-------|--------|----------|-------------|
| Sample data | TIFF stack | Yes | 3D histogram (TOF, y, x), efficiency-corrected |
| Open Beam (OB) | TIFF stack | Yes | 3D reference (TOF, y, x), efficiency-corrected |
| TOF bin edges | Metadata/file | Yes | Time-of-flight bin boundaries (fixed at acquisition) |
| ROI | (x0, y0, x1, y1) | No | Spatial region of interest |

**Metadata** (from files or DAQ):

- Acquisition time per frame
- p_charge (proton charge - beam intensity proxy)
- shutter_counts (number of neutron pulses captured per frame)
- Source-to-detector distance (L)

**Key Characteristics**:

- **Frame mode**: Detector operates in frame readout mode with histogram accumulation
- **No dark current correction**: Counting detector (not integrating)
- **Efficiency pre-corrected**: Input TIFFs already have detector efficiency correction applied (external auto-reduction)
- **TOF bins fixed at acquisition**: Rebinning limited to combining adjacent bins
- **Shutter counts**: Tracks pulses captured per frame (compensates for frame readout gaps)

---

## 2. Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Data (TIFF Stacks)                                │
│  ───────────────────────────────                                │
│  • Load Sample TIFF stack → 4D array (N_rotations, TOF, y, x)   │
│    or 3D if single radiograph (TOF, y, x)                       │
│  • Load OB TIFF stack → 3D array (TOF, y, x)                    │
│  • Load TOF bin edges → 1D array (N_bins + 1)                   │
│  • Load metadata: p_charge, shutter_counts                      │
│  • Validate dimensions match                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Run Combining (Critical for VENUS)                     │
│  ──────────────────────────────────────────                     │
│  IF multiple runs provided:                                     │
│    • Sum histogram data across runs (sample, OB separately)     │
│    • Sum p_charge values                                        │
│    • Sum shutter_counts values                                  │
│    • Sum acquisition time                                       │
│    • Track partial dead pixels per run                          │
│                                                                 │
│  Note: All runs must have same TOF bin edges                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: ROI Clipping (Optional)                                │
│  ───────────────────────────────                                │
│  IF ROI specified:                                              │
│    • Crop spatial dimensions: arr[..., y0:y1, x0:x1]            │
│    • TOF dimension unchanged                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Dead Pixel Detection                                   │
│  ────────────────────────────                                   │
│  • Sum OB across TOF dimension: OB_summed = sum(OB, axis=TOF)   │
│  • Identify pixels with zero total counts                       │
│  • dead_mask = (OB_summed == 0)                                 │
│  • Output: 2D boolean mask (y, x)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Statistics Analysis & Rebinning Recommendation         │
│  ──────────────────────────────────────────────────────         │
│  Analyze count statistics per TOF bin:                          │
│                                                                 │
│  FOR each TOF bin t:                                            │
│    • N_counts[t] = sum(OB[t, :, :])  (excluding dead pixels)    │
│    • SNR[t] = √(N_counts[t])                                    │
│                                                                 │
│  Generate recommendation:                                       │
│    • Identify bins with inadequate statistics (SNR < threshold) │
│    • Recommend rebinning factor N (combine N adjacent bins)     │
│    • Or recommend spatial binning (NxN pixels)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Rebinning (Optional)                                   │
│  ────────────────────────────                                   │
│  IF rebinning requested:                                        │
│                                                                 │
│  Option A: TOF rebinning (combine N adjacent bins)              │
│    • Sum counts from N adjacent TOF bins                        │
│    • Update TOF bin edges: edges[::N]                           │
│    • Reduces TOF dimension by factor N                          │
│                                                                 │
│  Option B: Spatial rebinning (NxN pixel binning)                │
│    • Sum counts from NxN pixel blocks                           │
│    • Reduces (y, x) dimensions by factor N                      │
│    • Preserves TOF resolution                                   │
│                                                                 │
│  Note: Both options can be combined                             │
│  Note: Rebinning sums counts, preserving Poisson statistics     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Beam Correction                                        │
│  ───────────────────────                                        │
│  PRIMARY correction for VENUS pulsed source                     │
│                                                                 │
│  p_charge correction:                                           │
│    f_beam = p_charge_OB / p_charge_sample                       │
│                                                                 │
│  Shutter counts correlation check:                              │
│    • p_charge and shutter_counts should correlate               │
│    • Significant deviation indicates acquisition issues         │
│    • Shutter counts track pulses captured per frame             │
│      (compensates for frame readout gaps)                       │
│                                                                 │
│  Note: Correction factor applies uniformly across all TOF bins  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Normalization                                          │
│  ─────────────────────                                          │
│  FOR each rotation θ:                                           │
│    FOR each TOF bin t:                                          │
│                                                                 │
│      T[θ, t, y, x] = (Sample[θ, t, y, x] / OB[t, y, x]) × f_beam│
│                                                                 │
│  Handle division:                                               │
│    • Where dead_mask=True: T = NaN                              │
│    • Where OB[t, y, x] == 0: T = NaN                            │
│                                                                 │
│  Formula:                                                       │
│    T(TOF) = [I_sample(TOF) / I_OB(TOF)] × f_beam                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9: Air Region Correction (Optional)                       │
│  ─────────────────────────────────────────                      │
│  Post-normalization refinement if p_charge wasn't sufficient    │
│                                                                 │
│  IF Air ROI specified:                                          │
│    FOR each TOF bin t:                                          │
│      1. Calculate mean transmission in air region:              │
│         <T_air(t)> = mean(T[air_ROI, t])                        │
│                                                                 │
│      2. Scale to ensure air = 1.0:                              │
│         T_final(t) = T(t) / <T_air(t)>                          │
│                                                                 │
│  Note: Can apply per-TOF or globally (user choice)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 10: Experiment Error Propagation                          │
│  ─────────────────────────────────────                          │
│  Sources of uncertainty:                                        │
│    • Poisson: σ_N = √(N) for counts per TOF bin                 │
│    • p_charge: σ_p (Gaussian, from DAQ measurement)             │
│    • Air region: σ_air (if air correction applied)              │
│                                                                 │
│  Error propagation per TOF bin:                                 │
│                                                                 │
│    σ_T(TOF) = T(TOF) × √[ 1/N_sample(TOF) + 1/N_OB(TOF) +       │
│                           (σ_p_sample/p_sample)² +              │
│                           (σ_p_OB/p_OB)² ]                      │
│                                                                 │
│  Note: For rebinned data, counts are summed so σ = √(sum)       │
│  If air correction: add (σ_air/<T_air>)² term                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 11: Output                                                │
│  ────────────                                                   │
│  • Transmission: 4D array (θ, TOF, y, x)                        │
│  • Experiment Error: 4D array (same shape)                      │
│  • TOF Bin Edges: 1D array (N_bins + 1) - may differ if rebinned│
│  • Dead Pixel Mask: 2D boolean array (y, x)                     │
│  • Metadata: processing parameters, provenance                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Output Specification

| Output | Dimensions | dtype | Description |
|--------|------------|-------|-------------|
| Transmission | (θ, TOF, y, x) | float32 | TOF-resolved transmission |
| Experiment Error | (θ, TOF, y, x) | float32 | Propagated uncertainty (1σ) |
| TOF Bin Edges | (N_bins+1,) | float64 | Time-of-flight boundaries |
| Dead Pixel Mask | (y, x) | bool | True = dead pixel |
| Metadata | dict | - | Processing provenance |

**Metadata contents**:

- Input file paths
- Processing timestamp
- Original and final TOF bin edges
- Rebinning applied (TOF factor, spatial factor, or none)
- Total p_charge and shutter_counts
- Beam correction factor
- Air region correction applied (if any)
- ROI applied (if any)
- Number of runs combined (if any)
- Software version

---

## 4. Coordinate Conversions

TOF can be converted to energy or wavelength using flight path length:

```
TOF → Wavelength:
  λ = (h × TOF) / (m_n × L)

TOF → Energy:
  E = (1/2) × m_n × (L / TOF)²

where:
  h = Planck's constant
  m_n = neutron mass
  L = source-to-detector distance
```

---

## 5. Decision Points

| Step | Decision | Options |
|------|----------|---------|
| 2 | Multiple runs? | Combine or single run |
| 3 | ROI needed? | Apply crop or full frame |
| 5 | Statistics adequate? | Yes → proceed / No → recommend rebinning |
| 6 | Rebinning type | TOF (combine N bins) / Spatial (NxN) / None |
| 9 | Air region correction? | Apply if p_charge insufficient / Skip |

---

## 6. Rebinning Constraints

TPX1 histogram data has fixed TOF bins determined at acquisition. Rebinning options:

**TOF Rebinning**:
- Combine N adjacent bins into 1
- New bin edges = original_edges[::N]
- Must use integer factor N
- Cannot create arbitrary bin edges

**Spatial Rebinning**:
- Combine NxN pixel blocks
- Reduces spatial resolution
- Preserves TOF resolution

**Cannot do**:
- Arbitrary TOF bin edges (requires raw events)
- Heterogeneous TOF binning (variable width)
- These require event-mode data (TPX3)

---

## 7. Development Components

### Required Modules

| Component | Purpose | Priority |
|-----------|---------|----------|
| `loaders.tiff_loader` | Load TIFF histogram stacks | P0 |
| `loaders.metadata_loader` | Extract p_charge, shutter_counts, TOF edges | P0 |
| `processing.run_combiner` | Aggregate multiple runs | P0 |
| `processing.roi_clipper` | Apply ROI to arrays | P1 |
| `processing.dead_pixel_detector` | Identify dead pixels | P0 |
| `tof.statistics_analyzer` | Analyze bin occupancy, compute SNR | P0 |
| `tof.histogram_rebinner` | Combine adjacent TOF bins | P0 |
| `processing.spatial_rebinner` | Combine NxN pixel blocks | P1 |
| `processing.beam_corrector` | Apply p_charge correction | P0 |
| `processing.normalizer` | Compute transmission | P0 |
| `processing.air_region_corrector` | Optional post-normalization correction | P1 |
| `processing.uncertainty_calculator` | Error propagation | P0 |
| `tof.coordinate_converter` | TOF ↔ λ ↔ E | P1 |
| `exporters.output_writer` | Write results | P0 |

### Data Models

```
InputData:
  - sample: NDArray[float32]         # (N, TOF, y, x) or (TOF, y, x)
  - open_beam: NDArray[float32]      # (TOF, y, x)
  - tof_edges: NDArray[float64]      # (N_bins + 1,)
  - p_charge_sample: float32
  - p_charge_OB: float32
  - shutter_counts_sample: int
  - shutter_counts_OB: int
  - flight_path_length: float32      # meters
  - roi: Optional[Tuple[int, int, int, int]]
  - metadata: Dict

RebinConfig:
  - tof_factor: Optional[int]        # combine N adjacent TOF bins
  - spatial_factor: Optional[int]    # combine NxN pixels

ProcessedData:
  - transmission: NDArray[float32]   # (N, TOF, y, x)
  - uncertainty: NDArray[float32]    # (N, TOF, y, x)
  - tof_edges: NDArray[float64]      # (N_bins + 1,) - updated if rebinned
  - dead_pixel_mask: NDArray[bool]   # (y, x)
  - metadata: Dict
```

---

## 8. Validation Criteria

- [ ] Transmission values in expected range per TOF bin
- [ ] No NaN values except where dead_mask=True
- [ ] Uncertainty > 0 for all valid pixels and TOF bins
- [ ] Dead pixel mask correctly identifies zero-count pixels
- [ ] TOF bin edges monotonically increasing
- [ ] Rebinning preserves total counts
- [ ] Beam correction factor close to 1.0
- [ ] p_charge and shutter_counts correlate (if both available)
