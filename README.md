# NeuNorm 2.0 (Alpha) - Modern Neutron Imaging Data Processing

> **⚠️ DEVELOPMENT BRANCH**: This is the NeuNorm 2.0 development branch featuring a complete architectural rewrite. For the stable NeuNorm 1.x release, see the [`main` branch](https://github.com/neutrons/NeuNorm/tree/main).

[![PyPI version](https://badge.fury.io/py/NeuNorm.svg)](https://badge.fury.io/py/NeuNorm)
[![Anaconda-Server Badge](https://anaconda.org/neutronimaging/neunorm/badges/version.svg)](https://anaconda.org/neutronimaging/neunorm)
[![codecov](https://codecov.io/gh/neutrons/NeuNorm/branch/next/graph/badge.svg)](https://codecov.io/gh/neutrons/NeuNorm)
[![Documentation Status](https://readthedocs.org/projects/neunorm/badge/?version=latest)](http://neunorm.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/97755175.svg)](https://zenodo.org/badge/latestdoi/97755175)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00815/status.svg)](https://doi.org/10.21105/joss.00815)

## What's New in 2.0

NeuNorm 2.0 is a ground-up rewrite designed for modern neutron imaging workflows at ORNL facilities (MARS at HFIR and VENUS at SNS):

- **Time-of-Flight (TOF) Support**: Full 4D data processing (energy, time, x, y)
- **Event-Mode Data**: Direct TPX3/TPX4 event processing with chunked loading
- **In-Situ Workflows**: Time-resolved phase evolution tracking
- **Modern Architecture**: Pydantic data models, scipp-based processing
- **Dual-Facility Support**: Unified workflows for both MARS and VENUS
- **Phase Decomposition**: NMF with Beer-Lambert constraints
- **Resonance/Bragg Edge Detection**: Automated feature detection

### Installation (2.0 Alpha)

```bash
# Using pixi (recommended for development)
git clone https://github.com/neutrons/NeuNorm.git
cd NeuNorm
git checkout neunorm-2.0-base
pixi install

# Using conda (when released)
conda install -c neutrons -c conda-forge neunorm=2.0.0a0

# Using pip (when released)
pip install NeuNorm==2.0.0a0
```

### Quick Start (2.0)

```python
import neunorm
print(neunorm.__version__)  # 2.0.0a0
```

**Full documentation coming soon!**

---

## NeuNorm 1.x Documentation (Legacy)

For users of NeuNorm 1.x, see the [archived documentation](archive/neunorm-1.x/README.md).

### Abstract (1.x)
--------

NeuNorm is an open-source Python library that normalized neutron imaging measurements.

In order to cancel detector electronic noises, source beam fluctuations and other pollution signals from close by beam lines, every data acquired need to be normalized. In order to perform the normalization, one must take, in addition to his data set, either 1 or 2 extra data set. A set of open beam (OB) when sample has been removed but beam is on. An optional set of dark field (DF) is taken when beam is off and sample off. The dark field allows to clean the electronic noises from the images. The principle of normalization can be summarized by the following figure.

![](documentation/source/_static/normalization_principle.png)

which is defined by the following equation

$$
I_{n}(i, j) = \frac{I(i,j) - DF(i,j)}{OB(i,j) - DF(i,j)}
$$

where In is the image normalized, I the raw image, DF the dark field, OB the open beam and i and j the x and y-pixels along the images.

To improve the normalization, the program also allows the user to select a region of interest (ROI) in the sample images in order to match the background of the raw data with the background of the open beam. This is necessary for some beam lines where the fluctuations of the beam are too important to be neglected. The program calculates then, for each raw data, the average counts of this ROI divided by the average counts of the same ROI of the open beams, then apply this ratio to the normalized data.

Input data often contains very hight counts coming from gamma rays. Those are also corrected by the program by doing a median filter around those "gamma" pixels. How those gamma pixels are determined. They are always the highest counts for the input file format.


References
----------

[1] NumPy python library. Stéfan van der Walt and S. Chris Colbert and Gaël Varoquaux, The NumPy Array: A Structure
for Efficient Numerical Computation, Computing in Science \& Engineering. Volume 13, Number 2, (22-30) 2011

[2 SciPy python library. Travis E. Oliphant, SciPy: Open source scientific tools for Python,
Computing in Science and Engineering, Volume 9, (10-20) 2007

Installation
------------
$ conda install -c neutronimaging neunorm

Meta
----

Jean Bilheux - bilheuxjm@ornl.gov

Distributed under the BSD license. See LICENSE.txt for more information


Acknowledgements
----------------

This work is sponsored by the Laboratory Directed Research and
Development Program of Oak Ridge National Laboratory, managed by
UT-Battelle LLC, under Contract No. DE-AC05-00OR22725 with the U.S.
Department of Energy. The United States Government retains and the
publisher, by accepting the article for publication, acknowledges
that the United States Government retains a non-exclusive, paid-up,
irrevocable, worldwide license to publish or reproduce the published
form of this manuscript, or allow others to do so, for United States
Government purposes. The Department of Energy will provide public
access to these results of federally sponsored research in accordance
with the DOE Public Access Plan(http://energy.gov/downloads/doe-public-access-plan).
