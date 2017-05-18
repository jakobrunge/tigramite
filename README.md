# TIGRAMITE – Causal discovery for time series datasets
Version 3.0 described in http://arxiv.org/abs/1702.07007

(Python Package)

[Github](https://github.com/jakobrunge/tigramite_v3.git)

[Documentation](https://jakobrunge.github.io/tigramite_v3/)


## General Notes

TIGRAMITE is a time series analysis python module. With flexibly adaptable scripts it allows to reconstruct graphical models (conditional independence graphs) from discrete or continuously-valued time series based on a causal discovery algorithm and create high-quality plots of the results.


## Features

- different conditional independence test statistics adapted to
  continuously-valued or discrete data, and different assumptions about
  linear or nonlinear dependencies
- hyperparameter optimization
- easy parallelization
- handling of masked time series data
- false discovery control and confidence interval estimation


## Required python packages

- numpy, tested with Version 1.10
- scipy, tested with Version 0.17
- sklearn, tested with Version 0.18 (optional, necessary for GPDC and GPACE tests)
- ace python package (https://pypi.python.org/pypi/ace/0.3) OR rpy2 and R-package 'acepack' (optional, necessary for GPACE test)
- matplotlib, tested with Version 1.5
- networkx, tested with Version 1.10
- basemap (only if plotting on a map is needed)
- mpi4py (optional, necessary for using the parallelized implementation)
- cython (optional, necessary for CMIknn and GPDC tests)
- statsmodels, tested with Version 0.6 (optional, necessary for p-value corrections)


## Installation

python setup.py install

This will install tigramite in your path.

To use just the ParCorr and  CMIsymb independence tests, only numpy and scipy are required. For CMIknn, cython can optionally be used for compilation, otherwise the provided *.c file is used. GPDC also is based on cython, and additionally, sklearn is required for Gaussian Process regression.

GPACE requires more work: Firstly, sklearn is required for Gaussian Process regression. Secondly, either the python package 'ace' or the R-package 'acepack' are required for the ACE estimator. The R-package version is much faster. 'ace' can be installed via pip install ace. 'acepack' has to be installed in R first, and can then be accessed by tigramite using the rpy2-interface. Note that GPACE relies on a pre-computed null-distribution stored in a *.npz file. A null distribution has to be computed for each sample size (like a critical value table). To generate the *.npz file, the script generate_gpace_nulldist.py can be used.


## User Agreement

By downloading TIGRAMITE you agree with the following points: The toolbox is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application the toolbox.

You commit to cite TIGRAMITE in your reports or publications if used.


## License

Copyright (C) 2012-2017 Jakob Runge

mpi4py wrapper module "mpi.py" Copyright (C) 2012 Jobst Heitzig

See license.txt for full text.

TIGRAMITE is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. TIGRAMITE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.