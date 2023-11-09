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

class ParCorrMultNew(ParCorrMult):
    r"""Partial correlation test for multivariate X and Y.

    Multivariate partial correlation is estimated through ordinary least squares (OLS)
    regression and some test for multivariate dependency among the residuals.

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
    correlation_type : {'max_corr'}
        Which dependency measure to use on residuals.
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
        """Perform conditional independence test.

        Calls the dependence measure and significance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.
        If recycle_residuals is True, also _get_single_residuals must be
        available.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.
        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.
        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples. For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample.  Last,
            'max_lag' uses as much samples as possible.
        alpha_or_thres : float (optional)
            Significance level (if significance='analytic' or 'shuffle_test') or
            threshold (if significance='fixed_thres'). If given, run_test returns
            the test decision dependent=True/False.

        Returns
        -------
        val, pval, [dependent] : Tuple of floats and bool
            The test statistic value and the p-value. If alpha_or_thres is
            given, run_test also returns the test decision dependent=True/False.
        """

        if self.significance == 'fixed_thres' and alpha_or_thres is None:
            raise ValueError("significance == 'fixed_thres' requires setting alpha_or_thres")

        # Get the array to test on
        (array, xyz, XYZ, data_type,
         nonzero_array, nonzero_xyz, nonzero_XYZ, nonzero_data_type) = self._get_array(
            X=X, Y=Y, Z=Z, tau_max=tau_max, cut_off=cut_off,
            remove_constant_data=True, verbosity=self.verbosity)
        X, Y, Z = XYZ
        self.X = X
        self.Y = Y
        nonzero_X, nonzero_Y, nonzero_Z = nonzero_XYZ

        # Record the dimensions
        # dim, T = array.shape

        # Ensure it is a valid array
        if np.any(np.isnan(array)):
            raise ValueError("nans in the array!")

        combined_hash = self._get_array_hash(array, xyz, XYZ)

        # Get test statistic value and p-value [cached if possible]
        if combined_hash in self.cached_ci_results.keys():
            cached = True
            val, pval = self.cached_ci_results[combined_hash]
        else:
            cached = False

            # If all X or all Y are zero, then return pval=1, val=0, dependent=False
            if len(nonzero_X) == 0 or len(nonzero_Y) == 0:
                val = 0.
                pval = None if self.significance == 'fixed_thres' else 1.
            else:
                # Get the dependence measure, reycling residuals if need be
                val = self._get_dependence_measure_recycle(nonzero_X, nonzero_Y, nonzero_Z,
                                                           nonzero_xyz, nonzero_array, nonzero_data_type)
                # Get the p-value (None if significance = 'fixed_thres')
                dim, T = nonzero_array.shape
                pval = self._get_p_value(val=val, array=nonzero_array, xyz=nonzero_xyz, T=T, dim=dim)
            self.cached_ci_results[combined_hash] = (val, pval)

        # Make test decision
        if len(nonzero_X) == 0 or len(nonzero_Y) == 0:
            dependent = False
        else:
            if self.significance == 'fixed_thres':
                if self.two_sided:
                    dependent = np.abs(val) >= np.abs(alpha_or_thres)
                else:
                    dependent = val >= alpha_or_thres
                pval = 0. if dependent else 1.
            else:
                if alpha_or_thres is None:
                    dependent = None
                else:
                    dependent = pval <= alpha_or_thres

        self.ci_results[(tuple(X), tuple(Y), tuple(Z))] = (val, pval, dependent)

        # Return the calculated value(s)
        if self.verbosity > 1:
            self._print_cond_ind_results(val=val, pval=pval, cached=cached, dependent=dependent,
                                         conf=None)

        if alpha_or_thres is None:
            return val, pval
        else:
            return val, pval, dependent

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