# TIGRAMITE – Causal discovery for time series datasets
Version 3.0 described in http://arxiv.org/abs/1702.07007

(Python Package)

[Github](https://github.com/jakobrunge/tigramite.git)

[Documentation](https://jakobrunge.github.io/tigramite/)


## General Notes

Tigramite is a causal time series analysis python package. It allows to efficiently reconstruct causal graphs from high-dimensional time series datasets and model the obtained causal dependencies for causal mediation and prediction analyses. Causal discovery is based on linear as well as non-parametric conditional independence tests applicable to discrete or continuously-valued time series. Also includes functions for high-quality plots of the results. Please cite the following papers depending on which method you use:

## Features

- high detection power even for large-scale time series datasets
- flexible conditional independence test statistics adapted to
  continuously-valued or discrete data, and different assumptions about
  linear or nonlinear dependencies
- automatic hyperparameter optimization
- parallel computing script based on mpi4py
- handling of missing values and masks
- p-value correction and confidence interval estimation
- causal mediation class to analyze causal pathways
- prediction class based on sklearn models including causal feature selection


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

To use just the ParCorr and CMIsymb independence tests, only numpy and scipy are required. For CMIknn, cython can optionally be used for compilation, otherwise the provided *.c file is used. GPDC also is based on cython, and additionally, sklearn is required for Gaussian Process regression.

GPACE requires more work: Firstly, sklearn is required for Gaussian Process regression. Secondly, either the python package 'ace' or the R-package 'acepack' are required for the ACE estimator. The R-package version is much faster. 'ace' can be installed via pip install ace. 'acepack' has to be installed in R first, and can then be accessed by tigramite using the rpy2-interface. 

For GPDC and GPACE we recommend to pre-compute and store the null-distribution for a wide range of expected sample sizes with the function ``generate_and_save_nulldists``. The file containing the null distributions can then be supplied to the class with the keyword null_dist_filename.


## User Agreement

By downloading TIGRAMITE you agree with the following points: The toolbox is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application the toolbox.

You commit to cite TIGRAMITE in your reports or publications if used:

1. J. Runge, S. Flaxman, and D. Sejdinovic (2017): Detecting causal associations in large nonlinear time series datasets. https://arxiv.org/abs/1702.07007

2. J. Runge et al. (2015): Identifying causal gateways and mediators in complex spatio-temporal systems. Nature Communications, 6, 8502. http://doi.org/10.1038/ncomms9502

3. J. Runge (2015): Quantifying information transfer and mediation along causal pathways in complex systems. Phys. Rev. E, 92(6), 62829. http://doi.org/10.1103/PhysRevE.92.062829

4. J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths (2012): Escaping the Curse of Dimensionality in Estimating Multivariate Transfer Entropy. Physical Review Letters, 108(25), 258701. http://doi.org/10.1103/PhysRevLett.108.258701


## License

Copyright (C) Jakob Runge

See license.txt for full text.

TIGRAMITE is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. TIGRAMITE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.