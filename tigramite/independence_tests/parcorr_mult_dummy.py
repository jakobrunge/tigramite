"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import stats
import numpy as np
import sys
import warnings

from tigramite.independence_tests.parcorr_mult import ParCorrMult
from .independence_tests_base import CondIndTest

class ParCorrMultDummy(ParCorrMult):
    r"""Partial correlation test for multivariate X and Y.

    Adaption of the ParCorMult conditional independence test to be able to deal with one-hot encoded dummy variables.
    Namely, if a dummy variable takes on the role of X or Y, it is excluded from the regression step. Data of dummy
    variables is also not standardized. Importantly, dummy variables are assumed to be exogenous to the system.
    ParCorMult estimates multivariate partial correlation through ordinary least squares (OLS) regression and some
    test for multivariate dependency among the residuals.

    Notes
    -----
    To test :math:`X \perp Y | Z`, first :math:`Z` is regressed out from
    :math:`X` and :math:`Y` assuming the  model

    .. math::  X & =  Z \beta_X + \epsilon_{X} \\
        Y & =  Z \beta_Y + \epsilon_{Y}

    using OLS regression. Then different measures for the dependency among the residuals 
    can be used. Currently only a test for zero correlation on the maximum of the residuals' 
    correlation is performed.

    Parameters
    ----------
    dummy_indices : list
        List of lists of all indices that belong to dummy variables.
    **kwargs :
        Arguments passed on to Parent class CondIndTest.
    """
    # documentation
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self, dummy_indices, **kwargs):
        self._measure = 'par_corr_mult_new'
        self.dummy_indices = dummy_indices
        self.X = []
        self.Y = []

        ParCorrMult.__init__(self, **kwargs)

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max', alpha_or_thres=None):
        self.X = X
        self.Y = Y
        return super().run_test(X, Y, Z, tau_max, cut_off, alpha_or_thres)


    def get_dependence_measure(self, array, xyz):
        """Return multivariate kernel correlation coefficient.

        Estimated as some dependency measure on the
        residuals of a linear OLS regression.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Partial correlation coefficient.
        """

        dim, T = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        if self.X in self.dummy_indices:
            x_vals = np.copy(array[np.where(xyz == 0)[0], :])
        else:
            x_vals = self._get_single_residuals(array, xyz, target_var=0)

        if self.Y in self.dummy_indices:
            y_vals = np.copy(array[np.where(xyz == 1)[0], :])
        else:
            y_vals = self._get_single_residuals(array, xyz, target_var=1)

        array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
        xyz_resid = np.array([index_code for index_code in xyz if index_code != 2])

        val = self.mult_corr(array_resid, xyz_resid)

        return val

    def mult_corr(self, array, xyz, standardize=True):
        """Return multivariate dependency measure.

        Parameters
        ----------
        array : array-like
            data array with X, Y in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        Returns
        -------
        val : float
            Multivariate dependency measure.
        """

        dim, n = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        # only standardize the part of the array that is not the dummy
        x = (xyz != 5)
        y = (xyz != 5)
        if self.X in self.dummy_indices:
            x = (xyz != 0)
        if self.Y in self.dummy_indices:
            y = (xyz != 1)

        # Standardize
        if standardize:
            condition = (np.all((x, y)))
            array[np.where(condition)[0]] -= array[np.where(condition)[0]].mean(axis=1).reshape(condition.sum(), 1)
            std = array[np.where(condition)[0]].std(axis=1)
            for i in np.where(condition)[0]:
                if std[i] != 0.:
                    array[np.where(condition)[0]][i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        x = array[np.where(xyz==0)[0]]
        y = array[np.where(xyz==1)[0]]

        if self.correlation_type == 'max_corr':
            # Get (positive or negative) absolute maximum correlation value
            corr = np.corrcoef(x, y)[:len(x), len(x):].flatten()
            val = corr[np.argmax(np.abs(corr))]

            # val = 0.
            # for x_vals in x:
            #     for y_vals in y:
            #         val_here, _ = stats.pearsonr(x_vals, y_vals)
            #         val = max(val, np.abs(val_here))
        
        # elif self.correlation_type == 'linear_hsci':
        #     # For linear kernel and standardized data (centered and divided by std)
        #     # biased V -statistic of HSIC reduces to sum of squared inner products
        #     # over all dimensions
        #     val = ((x.dot(y.T)/float(n))**2).sum()
        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr' implemented.")

        return val