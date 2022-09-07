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

class ParCorrMult(CondIndTest):
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

    def __init__(self, correlation_type='max_corr', **kwargs):
        self._measure = 'par_corr_mult'
        self.two_sided = True
        self.residual_based = True

        self.correlation_type = correlation_type

        if self.correlation_type not in ['max_corr']:
            raise ValueError("correlation_type must be in ['max_corr'].")

        CondIndTest.__init__(self, **kwargs)

    def _get_single_residuals(self, array, xyz, target_var,
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

        xyz : array of ints
            XYZ identifier array of shape (dim,).

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
        dim_z = (xyz == 2).sum()

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.):
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        y = np.fastCopyAndTranspose(array[np.where(xyz==target_var)[0], :])

        if dim_z > 0:
            z = np.fastCopyAndTranspose(array[np.where(xyz==2)[0], :])
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (np.fastCopyAndTranspose(resid), np.fastCopyAndTranspose(mean))

        return np.fastCopyAndTranspose(resid)

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

        x_vals = self._get_single_residuals(array, xyz, target_var=0)
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

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.):
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

    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
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

        dim, T = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        x_vals = self._get_single_residuals(array, xyz, target_var=0)
        y_vals = self._get_single_residuals(array, xyz, target_var=1)

        array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
        xyz_resid = np.array([index_code for index_code in xyz if index_code != 2])


        null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        pval = (null_dist >= np.abs(value)).mean()

        # Adjust p-value for two-sided measures
        if pval < 1.:
            pval *= 2.

        # Adjust p-value for dimensions of x and y (conservative Bonferroni-correction)
        # pval *= dim_x*dim_y

        if return_null_dist:
            return pval, null_dist
        return pval

    def get_analytic_significance(self, value, T, dim, xyz):
        """Returns analytic p-value depending on correlation_type.

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

        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        if self.correlation_type == 'max_corr':
            if deg_f < 1:
                pval = np.nan
            elif abs(abs(value) - 1.0) <= sys.float_info.min:
                pval = 0.0
            else:
                trafo_val = value * np.sqrt(deg_f/(1. - value*value))
                # Two sided significance level
                pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2
        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr' implemented.")

        # Adjust p-value for dimensions of x and y (conservative Bonferroni-correction)
        pval *= dim_x*dim_y

        return pval

    def get_analytic_confidence(self, value, df, conf_lev):
        """
        Base class assumption that this is not implemented.  Concrete classes
        should override when possible.
        """
        raise NotImplementedError("Analytic confidence not"+\
                                  " implemented for %s" % self.measure)

    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """
        Base class assumption that this is not implemented.  Concrete classes
        should override when possible.
        """
        raise NotImplementedError("Model selection not"+\
                                  " implemented for %s" % self.measure)

if __name__ == '__main__':
    
    import tigramite
    from tigramite.data_processing import DataFrame
    # import numpy as np
    import timeit

    # np.random.seed(3)
    cmi = ParCorrMult(
            # significance = 'shuffle_test',
            # sig_samples=1000,
        )

    rate = np.zeros(100)
    for i in range(100):
        print(i)
        data = np.random.randn(100, 6)
        data[:,2] += -0.5*data[:,0]
        # data[:,1] += data[:,2]
        dataframe = DataFrame(data)

        cmi.set_dataframe(dataframe)

        pval = cmi.run_test(
                X=[(0,0)], #, (1,0)], 
                Y=[(2,0)], #, (3, 0)], 
                # Z=[(5,0)]
                Z = []
                )[1]
        
        rate[i] = pval <= 0.1
        # print(cmi.run_test(X=[(0,0),(1,0)], Y=[(2,0), (3, 0)], Z=[(5,0)]))
    print(rate.mean())