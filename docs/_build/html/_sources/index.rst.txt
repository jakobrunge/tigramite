.. Tigramite documentation master file, created by
   sphinx-quickstart on Thu May 11 18:32:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TIGRAMITE
=========

`Github repo <https://github.com/jakobrunge/tigramite/>`_

Tigramite is a causal time series analysis python package. It allows to efficiently estimate causal graphs from high-dimensional time series datasets (causal discovery) and to use these graphs for robust forecasting and the estimation and prediction of direct, total, and mediated effects. Causal discovery is based on linear as well as non-parametric conditional independence tests applicable to discrete or continuously-valued time series. Also includes functions for high-quality plots of the results. Please cite the following papers depending on which method you use:


- PCMCI: J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic, Detecting and quantifying causal associations in large nonlinear time series datasets. Sci. Adv. 5, eaau4996 (2019). https://advances.sciencemag.org/content/5/11/eaau4996

- PCMCI+: J. Runge (2020): Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence, UAI 2020,Toronto, Canada, 2019, AUAI Press, 2020. http://auai.org/uai2020/proceedings/579_main_paper.pdf

- LPCMCI: Gerhardus, A. & Runge, J. High-recall causal discovery for autocorrelated time series with latent confounders Advances in Neural Information Processing Systems, 2020, 33. https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html

- Generally: J. Runge (2018): Causal Network Reconstruction from Time Series: From Theoretical Assumptions to Practical Estimation. Chaos: An Interdisciplinary Journal of Nonlinear Science 28 (7): 075310. https://aip.scitation.org/doi/10.1063/1.5025050

- Nature Communications Perspective paper: https://www.nature.com/articles/s41467-019-10105-3

- Causal effects: J. Runge, Necessary and sufficient graphical conditions for optimal adjustment sets in causal graphical models with hidden variables, Advances in Neural Information Processing Systems, 2021, 34

- Mediation class: J. Runge et al. (2015): Identifying causal gateways and mediators in complex spatio-temporal systems. Nature Communications, 6, 8502. http://doi.org/10.1038/ncomms9502

- Mediation class: J. Runge (2015): Quantifying information transfer and mediation along causal pathways in complex systems. Phys. Rev. E, 92(6), 62829. http://doi.org/10.1103/PhysRevE.92.062829

- CMIknn: J. Runge (2018): Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics. http://proceedings.mlr.press/v84/runge18a.html



.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. autosummary::

   tigramite.pcmci.PCMCI
   tigramite.lpcmci.LPCMCI
   tigramite.independence_tests.CondIndTest
   tigramite.independence_tests.ParCorr
   tigramite.independence_tests.GPDC
   tigramite.independence_tests.GPDCtorch
   tigramite.independence_tests.CMIknn
   tigramite.independence_tests.CMIsymb  
   tigramite.independence_tests.OracleCI
   tigramite.independence_tests.ParCorrMult
   tigramite.causal_effects.CausalEffects
   tigramite.models.Models
   tigramite.models.LinearMediation
   tigramite.models.Prediction
   tigramite.data_processing
   tigramite.toymodels.structural_causal_processes
   tigramite.plotting


:mod:`tigramite.pcmci`: PCMCI
===========================================

.. autoclass:: tigramite.pcmci.PCMCI
   :members:


:mod:`tigramite.lpcmci`: LPCMCI
===========================================

.. autoclass:: tigramite.lpcmci.LPCMCI
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

.. autoclass:: tigramite.independence_tests.GPDCtorch
   :members:

.. autoclass:: tigramite.independence_tests.CMIknn
   :members:
   
.. autoclass:: tigramite.independence_tests.CMIsymb
   :members:

.. autoclass:: tigramite.independence_tests.OracleCI
   :members:

.. autoclass:: tigramite.independence_tests.ParCorrMult
   :members:

:mod:`tigramite.causal_effects`: Causal Effect analysis
===========================================================

.. autoclass:: tigramite.causal_effects.CausalEffects
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


:mod:`tigramite.data_processing`: Data processing functions
===========================================================

.. automodule:: tigramite.data_processing
   :members:


:mod:`tigramite.toymodels`: Toy model generators
===========================================================

.. automodule:: tigramite.toymodels.structural_causal_processes
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
