# TIGRAMITE – Causal discovery for time series datasets
Version 4.2

(Python Package)

[Github](https://github.com/jakobrunge/tigramite.git)

[Documentation](https://jakobrunge.github.io/tigramite/)


## General Notes

Tigramite is a causal time series analysis python package. It allows to
 efficiently reconstruct causal graphs from high-dimensional time series datasets and model the obtained causal dependencies for causal mediation and prediction analyses. Causal discovery is based on linear as well as non-parametric conditional independence tests applicable to discrete or continuously-valued time series.  Also includes functions for high-quality plots of the results. Please cite the following papers depending on which method you use:

- PCMCI: J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic, Detecting and quantifying causal associations in large nonlinear time series datasets. Sci. Adv. 5, eaau4996 (2019). https://advances.sciencemag.org/content/5/11/eaau4996
- PCMCI+: J. Runge (2020): Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence, UAI 2020,Toronto, Canada, 2019, AUAI Press, 2020. http://auai.org/uai2020/proceedings/579_main_paper.pdf
- Generally: J. Runge (2018): Causal Network Reconstruction from Time Series: From Theoretical Assumptions to Practical Estimation. Chaos: An Interdisciplinary Journal of Nonlinear Science 28 (7): 075310. https://aip.scitation.org/doi/10.1063/1.5025050
- Nature Communications Perspective paper: https://www.nature.com/articles/s41467-019-10105-3
- Mediation class: J. Runge et al. (2015): Identifying causal gateways and mediators in complex spatio-temporal systems. Nature Communications, 6, 8502. http://doi.org/10.1038/ncomms9502
- Mediation class: J. Runge (2015): Quantifying information transfer and mediation along causal pathways in complex systems. Phys. Rev. E, 92(6), 62829. http://doi.org/10.1103/PhysRevE.92.062829
- CMIknn: J. Runge (2018): Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics. http://proceedings.mlr.press/v84/runge18a.html


## Features

- high detection power even for large-scale time series datasets
- flexible conditional independence test statistics adapted to
  continuously-valued or discrete data, and different assumptions about
  linear or nonlinear dependencies
- automatic hyperparameter optimization for most tests
- parallel computing script based on mpi4py
- handling of missing values and masks
- p-value correction and confidence interval estimation
- causal mediation class to analyze causal pathways
- prediction class based on sklearn models including causal feature selection


## Required python packages

- numpy
- scipy
- scikit-learn   (optional, necessary for GPDC test)
- matplotlib (optional, only for plotting)
- networkx (optional, only for plotting and mediation)
- cython   (optional, necessary for CMIknn and GPDC tests)
- mpi4py   (optional, necessary for using the parallelized implementation)


## Installation

python setup.py install

This will install tigramite in your path.

To use just the ParCorr and CMIsymb independence tests, only numpy and scipy are required. For other independence tests more packages are required:

- CMIknn: cython can optionally be used for compilation, otherwise the provided ``*.c'' file is used 

- GPDC: also based on cython, and additionally, scikit-learn is required for Gaussian Process regression


## User Agreement

By downloading TIGRAMITE you agree with the following points: TIGRAMITE is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of TIGRAMITE.

You commit to cite above papers in your reports or publications.


## License

Copyright (C) 2014-2020 Jakob Runge

See license.txt for full text.

TIGRAMITE is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. TIGRAMITE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.