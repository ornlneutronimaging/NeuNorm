---
title: 'NeuNorm: Neutron Imaging Normalization Library'
tags:
  - normalization
  - neutron imaging
authors:
 - name: Jean Bilheux
   orcid: 0000-0003-2172-6487
   affiliation: 1
affiliations:
 - name: Oak Ridge National Laboratory
   index: 1
   
date: 22 May 2018
bibliography: paper.bib
---

# Summary

NeuNorm is an open-source Python library that normalized neutron imaging measurements. 

In order to cancel detector electronic noises, source beam fluctuations and other pollution signals from close by beam lines, every data acquired need to be normalized. In order to perform the normalization, one must take, in addition to his data set, either 1 or 2 extra data set. A set of open beam (OB) when sample has been removed but beam is on. An optional set of dark field (DF) is taken when beam is off and sample off. The dark field allows to clean the electronic noises from the images. The principle of normalization can be summarized by the following figure. 

![](../documentation/source/_static/normalization_principle.png)

which is defined by the following equation

![](../documentation/source/_static/normalization_equation.png)

where In is the image normalized, I the raw image, DF the dark field, OB the open beam and i and j the x and y-pixels along the images.

To improve the normalization, the program also allows the user to select a region of interest (ROI) in the sample images in order to match the background of the raw data with the background of the open beam. This is necessary for some beam lines where the fluctuations of the beam are too important to be neglected. The program calculates then, for each raw data, the average counts of this ROI divided by the average counts of the same ROI of the open beams, then apply this ratio to the normalized data. 

Input data often contains very hight counts coming from gamma rays. Those are also corrected by the program by doing a median filter around those "gamma" pixels. How those gamma pixels are determined. They are always the highest counts for the input file format.


# Acknowledgements
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

# References
1. NumPy python library. Stéfan van der Walt and S. Chris Colbert and Gaël Varoquaux, The NumPy Array: A Structure
for Efficient Numerical Computation, Computing in Science \& Engineering. Volume 13, Number 2, (22-30) 2011

2. SciPy python library. Travis E. Oliphant, SciPy: Open source scientific tools for Python,
Computing in Science and Engineering, Volume 9, (10-20) 2007

