# NeuNorm 2.0 - Modern Neutron Imaging Normalization

> **⚠️ DEVELOPMENT BRANCH**: This is the NeuNorm 2.0 development branch featuring a complete architectural rewrite. For the stable NeuNorm 1.x release, see the [`main` branch](https://github.com/ornlneutronimaging/NeuNorm/tree/main).

[![PyPI version](https://badge.fury.io/py/NeuNorm.svg)](https://badge.fury.io/py/NeuNorm)
[![codecov](https://codecov.io/gh/neutrons/NeuNorm/branch/next/graph/badge.svg)](https://codecov.io/gh/neutrons/NeuNorm)
[![Documentation Status](https://readthedocs.org/projects/neunorm/badge/?version=latest)](http://neunorm.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/97755175.svg)](https://zenodo.org/badge/latestdoi/97755175)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00815/status.svg)](https://doi.org/10.21105/joss.00815)

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

## Development Plan

### Roadmap

Development officially starts **February 18, 2026** with target completion in **May 2026**.

| Phase | Milestone | Due Date | Description |
|-------|-----------|----------|-------------|
| 0 | Core Infrastructure | Mar 6, 2026 | TIFF/FITS loaders, dark corrector, gamma filter, exporters |
| 1 | MARS CCD/CMOS | Mar 20, 2026 | First complete end-to-end pipeline |
| 2 | MARS TPX3 | Apr 3, 2026 | Event-mode support (no TOF) |
| 3 | VENUS CCD/CMOS | Apr 17, 2026 | p_charge beam correction |
| 4 | VENUS TPX1 | May 1, 2026 | Histogram-mode TOF with rebinning |
| 5 | VENUS TPX3 | May 15, 2026 | Event-mode TOF (most complex) |

### Project Board

Track progress: [NeuNorm v2.0 Development](https://github.com/orgs/ornlneutronimaging/projects/7)

### Module Status

#### Implemented
- `data_models/core.py` - EventData (Pydantic v2)
- `data_models/tof.py` - BinningConfig
- `loaders/event_loader.py` - HDF5 TPX3/TPX4 events
- `tof/pulse_reconstruction.py` - Pulse ID with Numba JIT
- `tof/event_converter.py` - Events → 3D histogram
- `tof/pixel_detector.py` - Dead/hot pixel detection
- `processing/normalizer.py` - T = Sample/OB with p_charge
- `processing/uncertainty_calculator.py` - Poisson + systematic

#### To Be Implemented
- TIFF/FITS loaders (using [scitiff](https://github.com/scipp/scitiff))
- Dark current correction
- Gamma filtering
- Air region correction
- Run combiner
- ROI clipper
- Statistics analyzer / Histogram rebinner
- Output writers (HDF5, TIFF)

---

## Installation

### Development Setup

```bash
git clone https://github.com/ornlneutronimaging/NeuNorm.git
cd NeuNorm
git checkout neunorm-2.0-base
pixi install
```

### Quick Start

```python
import NeuNorm
print(NeuNorm.__version__)  # 2.0.0a0
```

---

## Architecture

### Technology Stack

- **Data Models**: Pydantic v2 for validation
- **Array Processing**: scipp with automatic variance propagation
- **TIFF I/O**: scitiff (scipp ecosystem)
- **Performance**: Numba JIT for hot paths
- **Testing**: pytest with hypothesis

### Key Design Principles

1. **Scipp-native**: All processing uses `sc.DataArray` with automatic uncertainty tracking
2. **Modular pipelines**: Each processing step is an independent, testable module
3. **Workflow-driven**: Pipelines are composed based on detector/facility combination
4. **Explicit over implicit**: Configuration via Pydantic models, no hidden defaults

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Branch Strategy

- `main`: Stable NeuNorm 1.x releases
- `neunorm-2.0-base`: Active v2.0 development (target branch for PRs)
- `next`: Integration branch for v2.0

---

## NeuNorm 1.x (Legacy)

For users of NeuNorm 1.x, see the [archived documentation](archive/neunorm-1.x/README.md).

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
