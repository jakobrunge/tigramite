.. Tigramite documentation master file, created by
   sphinx-quickstart on Thu May 11 18:32:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TIGRAMITE
=========

`Github repo <https://github.com/jakobrunge/tigramite/>`_

Tigramite is a causal time series analysis python package. It allows to efficiently reconstruct causal graphs from high-dimensional time series datasets and model the obtained causal dependencies for causal mediation and prediction analyses. Causal discovery is based on linear as well as non-parametric conditional independence tests applicable to discrete or continuously-valued time series. Also includes functions for high-quality plots of the results. Please cite the following papers depending on which method you use:


0. J. Runge et al. (2019): Inferring causation from time series in Earth system sciences.
   Nature Communications, 10(1):2553.
   https://www.nature.com/articles/s41467-019-10105-3

1. J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic (2019): Detecting and quantifying causal associations in large nonlinear time series datasets.
   Sci. Adv. 5, eaau4996.
   https://advances.sciencemag.org/content/5/11/eaau4996

2. J. Runge et al. (2015): Identifying causal gateways and mediators in complex spatio-temporal systems. 
   Nature Communications, 6, 8502. 
   http://doi.org/10.1038/ncomms9502

3. J. Runge (2015): Quantifying information transfer and mediation along causal pathways in complex systems. 
   Phys. Rev. E, 92(6), 62829. 
   http://doi.org/10.1103/PhysRevE.92.062829

4. J. Runge (2018): Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information.
   In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics.
   http://proceedings.mlr.press/v84/runge18a.html

5. J. Runge (2018): Causal Network Reconstruction from Time Series: From Theoretical Assumptions to Practical Estimation.
   Chaos: An Interdisciplinary Journal of Nonlinear Science 28 (7): 075310.   
   https://aip.scitation.org/doi/10.1063/1.5025050


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. autosummary::

   tigramite.pcmci.PCMCI
   tigramite.independence_tests.CondIndTest
   tigramite.independence_tests.ParCorr
   tigramite.independence_tests.GPDC
   tigramite.independence_tests.CMIknn
   tigramite.independence_tests.CMIsymb  
   tigramite.independence_tests.RCOT  
   tigramite.data_processing
   tigramite.models.Models
   tigramite.models.LinearMediation
   tigramite.models.Prediction
   tigramite.plotting


:mod:`tigramite.pcmci`: PCMCI
===========================================

.. autoclass:: tigramite.pcmci.PCMCI
   :members:

:mod:`tigramite.independence_tests`: Conditional independence tests
=================================================================================

Base class:

.. autoclass:: tigramite.independence_tests.CondIndTest
   :members:

Test statistics:

.. autoclass:: tigramite.independence_tests.ParCorr
   :members:

.. autoclass:: tigramite.independence_tests.GPDC
   :members:

.. autoclass:: tigramite.independence_tests.CMIknn
   :members:
   
.. autoclass:: tigramite.independence_tests.CMIsymb
   :members:

.. autoclass:: tigramite.independence_tests.RCOT
   :members:

:mod:`tigramite.data_processing`: Data processing functions
===========================================================

.. automodule:: tigramite.data_processing
   :members:

:mod:`tigramite.models`: Time series modeling, mediation, and prediction
========================================================================

Base class:

.. autoclass:: tigramite.models.Models
   :members:

Derived classes:

.. autoclass:: tigramite.models.LinearMediation
   :members:

.. autoclass:: tigramite.models.Prediction
   :members:

:mod:`tigramite.plotting`: Plotting functions
=============================================

.. automodule:: tigramite.plotting
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
