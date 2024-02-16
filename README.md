[![PyPI version](https://badge.fury.io/py/NeuNorm.svg)](https://badge.fury.io/py/NeuNorm)
[![Anaconda-Server Badge](https://anaconda.org/neutronimaging/neunorm/badges/version.svg)](https://anaconda.org/neutronimaging/neunorm)
[![codecov](https://codecov.io/gh/scikit-beam/NeuNorm/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-beam/NeuNorm)
[![Documentation Status](https://readthedocs.org/projects/neunorm/badge/?version=latest)](http://neunorm.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/97755175.svg)](https://zenodo.org/badge/latestdoi/97755175)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00815/status.svg)](https://doi.org/10.21105/joss.00815)

Abstract
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
