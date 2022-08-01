"""Tigramite causal discovery for time series."""

# Author: Wang Jincheng <wjcuhk@gmail.com.com>
#

from __future__ import print_function
import warnings
from numba import jit
import numpy as np
import pandas as pd
from cmath import log
import string
import time

from pgmpy.estimators.CITests import g_sq

from .independence_tests_base import CondIndTest

class ChiSquare(CondIndTest):

    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self, n_symbs=None, **kwargs):
        # Setup the member variables
        self._measure = 'chi-square'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.n_symbs = n_symbs
        # Call the parent constructor
        CondIndTest.__init__(self, **kwargs)
        if self.verbosity > 0:
            print("n_symbs = %s" % self.n_symbs)

    def _bincount_hist(self, symb_array, weights=None):
        """Computes histogram from symbolic array.

        The maximum of the symbolic array determines the alphabet / number
        of bins.

        Parameters
        ----------
        symb_array : integer array
            Data array of shape (dim, T).

        weights : float array, optional (default: None)
            Optional weights array of shape (dim, T).

        Returns
        -------
        hist : array
            Histogram array of shape (base, base, base, ...)*number of
            dimensions with Z-dimensions coming first.
        """

        if self.n_symbs is None:
            n_symbs = int(symb_array.max() + 1)
        else:
            n_symbs = self.n_symbs
            if n_symbs < int(symb_array.max() + 1):
                raise ValueError("n_symbs must be >= symb_array.max() + 1 = {}".format(symb_array.max() + 1))

        if 'int' not in str(symb_array.dtype):
            raise ValueError("Input data must of integer type, where each "
                             "number indexes a symbol.")

        dim, T = symb_array.shape
        """
        JC NOTE: When the dim is high (especially because of the number of conditioning variables),
        the computation of histogram is memory-intensive and time-intensive, even for binary variables.
        Unfortunately, currently we can only limit the maximum number of conditioning variables (through parameter specification).
        Future work can be explored to implement a new way to compute the histogram.
        """
        flathist = np.zeros((n_symbs ** dim), dtype='int16')
        multisymb = np.zeros(T, dtype='int64')
        if weights is not None:
            flathist = np.zeros((n_symbs ** dim), dtype='float32')
            multiweights = np.ones(T, dtype='float32')

        for i in range(dim):
            multisymb += symb_array[i, :] * n_symbs ** i
            if weights is not None:
                multiweights *= weights[i, :]

        if weights is None:
            result = np.bincount(multisymb)
        else:
            result = (np.bincount(multisymb, weights=multiweights)
                      / multiweights.sum())

        flathist[:len(result)] += result

        hist = flathist.reshape(tuple([n_symbs, n_symbs] +
                                      [n_symbs for i in range(dim - 2)])).T
        print(hist.shape)
        print(hist[1,:,:])
        return hist
    
    def get_dependence_measure(self, array, xyz):
        dim, T = array.shape
        col_names = (list(string.ascii_lowercase) + list(string.ascii_uppercase))[:dim]
        assert(len(col_names) == dim)
        data = pd.DataFrame(array, columns=col_names)
        statistic_val, p_value, dof = g_sq(X=col_names[0], Y=col_names[1], Z=col_names[2:], data=data, boolean=False)
        return statistic_val, p_value