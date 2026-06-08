# NeuNorm 2.0 - Modern Neutron Imaging Normalization

[![PyPI version](https://badge.fury.io/py/NeuNorm.svg)](https://badge.fury.io/py/NeuNorm)
[![codecov](https://codecov.io/gh/ornlneutronimaging/NeuNorm/branch/next/graph/badge.svg)](https://codecov.io/gh/ornlneutronimaging/NeuNorm)
[![Documentation Status](https://readthedocs.org/projects/neunorm/badge/?version=latest)](http://neunorm.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/97755175.svg)](https://zenodo.org/badge/latestdoi/97755175)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00815/status.svg)](https://doi.org/10.21105/joss.00815)

NeuNorm normalizes neutron imaging data and processes time-of-flight (TOF) data
for ORNL imaging facilities — MARS at HFIR and VENUS at SNS.

> **NeuNorm 2.0 is a complete, scipp-based rewrite and a breaking change from the
> 1.x series.** Code written against the 1.x `NeuNorm.normalization.Normalization`
> API will not run unchanged on 2.0. See the
> [1.x → 2.0 migration guide](docs/migration.md) to port your code, or pin
> `NeuNorm<2` to keep using the legacy API — see
> [NeuNorm 1.x (Legacy)](#neunorm-1x-legacy).

---

## Overview

NeuNorm 2.0 is a ground-up rewrite for modern neutron imaging workflows at ORNL facilities:

| Facility | Beamline | Beam Type | Detectors |
|----------|----------|-----------|-----------|
| HFIR | MARS | Continuous | CCD/CMOS, TPX3 |
| SNS | VENUS | Pulsed (TOF) | CCD/CMOS, TPX1, TPX3 |

### Key Features

- **Time-of-Flight (TOF) Support**: Full hyperspectral data processing
- **Event-Mode Processing**: Direct TPX3 event handling with pulse reconstruction
- **Automatic Uncertainty Propagation**: Via scipp's variance tracking
- **Dual-Facility Support**: Unified API for MARS and VENUS workflows
- **Modern Architecture**: Pydantic v2 data models, scipp-based processing

---

## Installation

From PyPI:

```bash
pip install NeuNorm
# optional extras: visualization (plopp/matplotlib) and Numba acceleration
pip install "NeuNorm[viz,performance]"
```

From conda (the `neutrons` channel):

```bash
conda install -c neutrons neunorm
```

From source, for development (uses [pixi](https://pixi.sh)):

```bash
git clone https://github.com/ornlneutronimaging/NeuNorm.git
cd NeuNorm
pixi install
pixi run test
```

---

## Quick Start

Each detector/facility combination has a ready-made pipeline in `neunorm.pipelines`
that loads the data, applies the appropriate corrections, and writes the normalized
transmission — with uncertainty, detector masks, and provenance metadata — to HDF5
(or TIFF):

```python
from pathlib import Path
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

# Each inner list is one acquisition "run" to combine before processing.
transmission = run_mars_ccd_pipeline(
    sample_paths=[["sample_0001.tiff", "sample_0002.tiff"]],
    ob_paths=[["ob_0001.tiff", "ob_0002.tiff"]],
    dark_paths=[["dark_0001.tiff"]],
    output_path=Path("normalized.hdf5"),
)
```

Each detector/facility has its own pipeline — `run_mars_tpx3_pipeline`,
`run_venus_ccd_pipeline`, `run_venus_tpx1_pipeline`,
`run_venus_tpx3_histogram_pipeline`, and `run_venus_tpx3_event_pipeline`. They
share the same load → correct → normalize → write-to-HDF5/TIFF flow, but each
takes detector-appropriate inputs — TPX detectors skip `dark_paths`, the TOF
pipelines add `rebin_by_tof`/`rebin_by_spatial`, and
`run_venus_tpx3_event_pipeline` takes a `BinningConfig` and flat (per-run) path
lists. Check each function's signature in the
[API reference](https://neunorm.readthedocs.io) or the per-workflow guides under
[Supported Workflows](#supported-workflows). Verify your install with:

```bash
python -c "import neunorm; print(neunorm.__version__)"
```

---

## Core Physics

### Normalization Principle

Neutron imaging normalization removes detector noise, beam fluctuations, and contamination to extract the true sample transmission:

$$
T(x, y) = f_{beam} \times \frac{I_{sample} - I_{dark}}{I_{OB} - I_{dark}}
$$

Where:
- **T**: Transmission (0-1, may exceed 1 due to scattering)
- **I_sample**: Raw sample measurement (counts)
- **I_OB**: Open beam reference (no sample)
- **I_dark**: Dark current (detector noise baseline)
- **f_beam**: Beam intensity correction factor

### Detector-Specific Corrections

| Detector Type | Dark Correction | Beam Correction | Hot Pixels |
|---------------|-----------------|-----------------|------------|
| CCD/CMOS | Required | Time or p_charge | Not needed |
| TPX1 (histogram) | Not needed | p_charge or shutter | Not needed |
| TPX3 (event/histogram) | Not needed | p_charge | Required |

### Uncertainty Propagation

For counting detectors, uncertainty follows Poisson statistics:

$$
\frac{\sigma_T}{T} = \sqrt{\frac{1}{N_{sample}} + \frac{1}{N_{OB}} + \left(\frac{\sigma_{p,sample}}{p_{sample}}\right)^2 + \left(\frac{\sigma_{p,OB}}{p_{OB}}\right)^2}
$$

For CCD/CMOS with dark correction, additional terms account for dark current uncertainty.

### Time-of-Flight (TOF) Processing

At VENUS (pulsed source), neutron wavelength is determined from flight time:

$$
\lambda = \frac{h \times t}{m_n \times L}
$$

Where:
- λ = neutron wavelength
- t = time-of-flight
- L = source-to-detector distance
- m_n = neutron mass
- h = Planck's constant

This enables **hyperspectral imaging** with wavelength-resolved transmission T(λ, x, y).

---

## Supported Workflows

| Workflow | Detector | Facility | TOF | Documentation |
|----------|----------|----------|-----|---------------|
| MARS CCD/CMOS | CCD/CMOS | HFIR | No | [mars_ccd_cmos.md](docs/workflows/mars_ccd_cmos.md) |
| MARS TPX3 | Timepix3 | HFIR | No | [mars_tpx3.md](docs/workflows/mars_tpx3.md) |
| VENUS CCD/CMOS | CCD/CMOS | SNS | No | [venus_ccd_cmos.md](docs/workflows/venus_ccd_cmos.md) |
| VENUS TPX1 | Timepix1 | SNS | Yes | [venus_tpx1.md](docs/workflows/venus_tpx1.md) |
| VENUS TPX3 | Timepix3 | SNS | Yes | [venus_tpx3.md](docs/workflows/venus_tpx3.md) |

---

## Architecture

### Technology Stack

- **Data Models**: Pydantic v2 for validation
- **Array Processing**: scipp with automatic variance propagation
- **TIFF I/O**: scitiff (scipp ecosystem)
- **Performance**: Numba JIT for hot paths (optional, via the `performance` extra)
- **Testing**: pytest with coverage

### Key Design Principles

1. **Scipp-native**: All processing uses `sc.DataArray` with automatic uncertainty tracking
2. **Modular pipelines**: Each processing step is an independent, testable module
3. **Workflow-driven**: Pipelines are composed based on detector/facility combination
4. **Explicit over implicit**: Configuration via Pydantic models, no hidden defaults

---

## Documentation

Full documentation — user guides plus an autodoc API reference — is hosted at
[neunorm.readthedocs.io](https://neunorm.readthedocs.io). The per-workflow guides
live under [`docs/workflows/`](docs/workflows/). Release history is in
[CHANGELOG.md](CHANGELOG.md), and the
[1.x → 2.0 migration guide](docs/migration.md) covers porting legacy code.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Development uses **pixi**; please run
`pixi run test` and `pixi run pre-commit run --all-files` before opening a pull
request. Branches follow the promotion path **`next` → `qa` → `main`** (`next` is
the default development branch).

---

## NeuNorm 1.x (Legacy)

NeuNorm 1.x — the `from NeuNorm.normalization import Normalization` API — is the
previous stable line. To keep using it, pin `NeuNorm<2` in your environment.
Archived 1.x documentation: [archive/neunorm-1.x/README.md](archive/neunorm-1.x/README.md).
To port existing 1.x code to 2.0, see the
[migration guide](docs/migration.md).

---

## References

- [scipp documentation](https://scipp.github.io/)
- [scitiff documentation](https://scipp.github.io/scitiff/)
- [ORNL Neutron Imaging](https://neutronimaging.ornl.gov/)

---

## Acknowledgements

This work is sponsored by the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory, managed by UT-Battelle LLC, under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use NeuNorm in your research, please cite:

```bibtex
@article{bilheux2018neunorm,
  title={NeuNorm: Open-source normalization of neutron imaging data in Python},
  author={Bilheux, Jean-Christophe},
  journal={Journal of Open Source Software},
  volume={3},
  number={28},
  pages={815},
  year={2018},
  doi={10.21105/joss.00815}
}
```
