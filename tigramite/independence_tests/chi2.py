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

    def get_dependence_measure(self, array, xyz):
        arr = array.T
        T, dim = arr.shape
        col_names = (list(string.ascii_lowercase) + list(string.ascii_uppercase))[:dim]
        assert(len(col_names) == dim)
        df = pd.DataFrame(data=arr, columns=col_names)
        statistic_val, p_value, dof = g_sq(X=col_names[0], Y=col_names[1], Z=col_names[2:], data=df, boolean=False)
        return statistic_val, p_value