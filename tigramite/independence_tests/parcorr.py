"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import stats
import numpy as np
import sys
import warnings

from .independence_tests_base import CondIndTest

class ParCorr(CondIndTest):
    r"""Partial correlation test.

    Partial correlation is estimated through linear ordinary least squares (OLS)
    regression and a test for non-zero linear Pearson correlation on the
    residuals.

    Notes
    -----
    To test :math:`X \perp Y | Z`, first :math:`Z` is regressed out from
    :math:`X` and :math:`Y` assuming the  model

    .. math::  X & =  Z \beta_X + \epsilon_{X} \\
        Y & =  Z \beta_Y + \epsilon_{Y}

    using OLS regression. Then the dependency of the residuals is tested with
    the Pearson correlation test.

    .. math::  \rho\left(r_X, r_Y\right)

    For the ``significance='analytic'`` Student's-*t* distribution with
    :math:`T-D_Z-2` degrees of freedom is implemented.

    Assumes one-dimensional X, Y.Use ParCorrMult for multivariate X, Y.


    Parameters
    ----------
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

    def __init__(self, **kwargs):
        self._measure = 'par_corr'
        self.two_sided = True
        self.residual_based = True

        CondIndTest.__init__(self, **kwargs)

    def _get_single_residuals(self, array, target_var,
                              standardize=True,
                              return_means=False):
        """Returns residuals of linear multiple regression.

        Performs a OLS regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        target_var : {0, 1}
            Variable to regress out conditions from.

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.

        Returns
        -------
        resid [, mean] : array-like
            The residual of the regression and optionally the estimated line.
        """

        dim, T = array.shape
        dim_z = dim - 2

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        y = array[target_var, :]

        if dim_z > 0:
            z = array[2:, :].T.copy()
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        return resid

    def get_dependence_measure(self, array, xyz, data_type=None):
        """Return partial correlation.

        Estimated as the Pearson correlation of the residuals of a linear
        OLS regression.

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

        x_vals = self._get_single_residuals(array, target_var=0)
        y_vals = self._get_single_residuals(array, target_var=1)
        val, _ = stats.pearsonr(x_vals, y_vals)
        return val

    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False,
                                 data_type=None):
        """Returns p-value for shuffle significance test.

        For residual-based test statistics only the residuals are shuffled.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """

        x_vals = self._get_single_residuals(array, target_var=0)
        y_vals = self._get_single_residuals(array, target_var=1)
        array_resid = np.array([x_vals, y_vals])
        xyz_resid = np.array([0, 1])

        null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        pval = (null_dist >= np.abs(value)).mean()

        # Adjust p-value for two-sided measures
        if pval < 1.:
            pval *= 2.

        if return_null_dist:
            return pval, null_dist
        return pval

    def get_analytic_significance(self, value, T, dim, xyz):
        """Returns analytic p-value from Student's t-test for the Pearson
        correlation coefficient.

        Assumes two-sided correlation. If the degrees of freedom are less than
        1, numpy.nan is returned.

        Parameters
        ----------
        value : float
            Test statistic value.

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Get the number of degrees of freedom
        deg_f = T - dim

        if deg_f < 1:
            pval = np.nan
        elif abs(abs(value) - 1.0) <= sys.float_info.min:
            pval = 0.0
        else:
            trafo_val = value * np.sqrt(deg_f/(1. - value*value))
            # Two sided significance level
            pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2

        return pval

    def get_analytic_confidence(self, value, df, conf_lev):
        """Returns analytic confidence interval for correlation coefficient.

        Based on Student's t-distribution.

        Parameters
        ----------
        value : float
            Test statistic value.

        df : int
            degrees of freedom of the test

        conf_lev : float
            Confidence interval, eg, 0.9

        Returns
        -------
        (conf_lower, conf_upper) : Tuple of floats
            Upper and lower confidence bound of confidence interval.
        """
        # Confidence interval is two-sided
        c_int = (1. - (1. - conf_lev) / 2.)

        value_tdist = value * np.sqrt(df) / np.sqrt(1. - value**2)
        conf_lower = (stats.t.ppf(q=1. - c_int, df=df, loc=value_tdist)
                      / np.sqrt(df + stats.t.ppf(q=1. - c_int, df=df,
                                                 loc=value_tdist)**2))
        conf_upper = (stats.t.ppf(q=c_int, df=df, loc=value_tdist)
                      / np.sqrt(df + stats.t.ppf(q=c_int, df=df,
                                                 loc=value_tdist)**2))
        return (conf_lower, conf_upper)


    def get_model_selection_criterion(self, j, parents, tau_max=0, criterion='aic'):
        """Returns model selection criterion modulo constants.

        Fits a linear model of the parents to variable j and returns the
        score. Here used to determine optimal hyperparameters in PCMCI, 
        in particular the pc_alpha value.

        Parameters
        ----------
        j : int
            Index of target variable in data array.

        parents : list
            List of form [(0, -1), (3, -2), ...] containing parents.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        criterion : string
            Scoring criterion among AIC, BIC, or corrected AIC.

        Returns:
        score : float
            Model score.
        """

        Y = [(j, 0)]
        X = [(j, 0)]   # dummy variable here
        Z = parents
        array, xyz, _ = self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                    tau_max=tau_max,
                                                    mask_type=self.mask_type,
                                                    return_cleaned_xyz=False,
                                                    do_checks=True,
                                                    verbosity=self.verbosity)

        dim, T = array.shape

        y = self._get_single_residuals(array, target_var=1, return_means=False)
        # Get RSS
        rss = (y**2).sum()
        # Number of parameters dim includes dummy x, therefore -1 which includes de-meaning 
        p = dim - 1

        # Get AIC
        if criterion == 'corrected_aic':
            score = T * np.log(rss) + 2. * p + (2.*p**2 + 2.*p)/(T - p - 1)
        elif criterion == 'bic':
            score = T * np.log(rss / float(T)) + p * np.log(T)  # BIC = n*log(residual sum of squares/n) + K*log(n)
        elif criterion == 'aic':
            score = T * np.log(rss) + 2. * p
        else:
            raise ValueError("Unknown scoring criterion.")

        return score
