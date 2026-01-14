"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import stats, linalg
import numpy as np
import sys
import warnings

from tigramite.independence_tests.independence_tests_base import CondIndTest

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

        if self.correlation_type not in ['max_corr', 'PCCA_WilksLambda']:
            raise ValueError("correlation_type must be in ['max_corr', 'PCCA_WilksLambda']")

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
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        y = array[np.where(xyz==target_var)[0], :].T.copy()

        if dim_z > 0:
            z = (array[np.where(xyz==2)[0], :]).T.copy()
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (np.fastCopyAndTranspose(resid), np.fastCopyAndTranspose(mean))

        return resid.T.copy()

    def get_dependence_measure(self, array, xyz, data_type=None):
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

    def compute_wilks_lambda_cca(self, X, Y, num_components=None):
        """
        Compute Wilks Lambda from canonical correlations between X and Y using NumPy.
        
        Parameters:
        - X, Y: Arrays of shape (n_features, n_samples)
        - num_components: Number of canonical correlations to return (default: all)
        
        Returns:
        - wilks_lambda: Wilks Lambda of canonical correlations
        """

        # Center variables (per feature)
        X = X - X.mean(axis=1, keepdims=True)
        Y = Y - Y.mean(axis=1, keepdims=True)

        # Compute covariance matrices (features are rows, samples are columns)
        Cxx = np.cov(X, rowvar=True)
        Cyy = np.cov(Y, rowvar=True)
        Cxy = np.cov(X, Y, rowvar=True)[:X.shape[0], X.shape[0]:]

        # Ensure matrices are at least 2D
        if Cxx.ndim == 0:
            Cxx = np.array([[Cxx]])
        if Cyy.ndim == 0:
            Cyy = np.array([[Cyy]])
        if Cxy.ndim == 1:
            Cxy = Cxy[:, None]

        # Regularization for stability
        eps = 1e-10
        Cxx += eps * np.eye(Cxx.shape[0])
        Cyy += eps * np.eye(Cyy.shape[0])

        # Solve generalized eigenvalue problem
        M = linalg.inv(Cxx) @ Cxy @ linalg.inv(Cyy) @ Cxy.T

        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.flip(np.real(eigvals))  # sort descending
        rho_sq = np.abs(np.maximum(eigvals, 0.0))

        if num_components is not None:
            rho_sq = rho_sq[:num_components]

        wilks_lambda = np.prod([1 - rho for rho in rho_sq])
        
        return wilks_lambda

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
        elif self.correlation_type == 'PCCA_WilksLambda':
            val = self.compute_wilks_lambda_cca(x, y)

        # elif self.correlation_type == 'linear_hsci':
        #     # For linear kernel and standardized data (centered and divided by std)
        #     # biased V -statistic of HSIC reduces to sum of squared inner products
        #     # over all dimensions
        #     val = ((x.dot(y.T)/float(n))**2).sum()
        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr' and 'PCCA_WilksLambda' implemented.")

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

        # pval = (null_dist >= np.abs(value)).mean()
        pval = float(np.sum(np.abs(null_dist) >= np.abs(value)) + 1) / (self.sig_samples + 1)

        # # Adjust p-value for two-sided measures
        # if pval < 1.:
        #     pval *= 2.

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
        dim_z = dim - dim_x - dim_y

        if self.correlation_type == 'max_corr':
            if deg_f < 1:
                pval = np.nan
            elif abs(abs(value) - 1.0) <= sys.float_info.min:
                pval = 0.0
            else:
                trafo_val = value * np.sqrt(deg_f/(1. - value*value))
                # Two sided significance level
                pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2
            # Adjust p-value for dimensions of x and y (conservative Bonferroni-correction)
            pval *= dim_x*dim_y
        elif self.correlation_type == 'PCCA_WilksLambda':
            if deg_f < 1:
                pval = np.nan
            else:
                trafo_val = -(T - dim_z - 0.5 * (dim_x + dim_y + 1)) * np.log(value)
                dof = dim_x * dim_y
                pval = 1 - stats.chi2.cdf(trafo_val, dof)     
        else:
            raise NotImplementedError("Currently only"
                                      "correlation_type == 'max_corr' and 'PCCA_WilksLambda' implemented.")
        

        return pval

    def get_model_selection_criterion(self, j, parents, tau_max=0, corrected_aic=False):
        """Returns Akaike's Information criterion modulo constants.

        Fits a linear model of the parents to each variable in j and returns
        the average score. Leave-one-out cross-validation is asymptotically
        equivalent to AIC for ordinary linear regression models. Here used to
        determine optimal hyperparameters in PCMCI, in particular the
        pc_alpha value.

        Parameters
        ----------
        j : int
            Index of target variable in data array.

        parents : list
            List of form [(0, -1), (3, -2), ...] containing parents.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

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

        y = self._get_single_residuals(array, xyz, target_var=0)

        n_comps = y.shape[0]
        score = 0.
        for y_component in y:
            # Get RSS
            rss = (y_component**2).sum()
            # Number of parameters
            p = dim - 1
            # Get AIC
            if corrected_aic:
                comp_score = T * np.log(rss) + 2. * p + (2.*p**2 + 2.*p)/(T - p - 1)
            else:
                comp_score = T * np.log(rss) + 2. * p
            score += comp_score

        score /= float(n_comps)
        return score


if __name__ == '__main__':
    
    import tigramite
    from tigramite.data_processing import DataFrame
    # import numpy as np
    import timeit

    seed=None
    random_state = np.random.default_rng(seed=seed)
    cmi = ParCorrMult(
            correlation_type='PCCA_WilksLambda',
            # significance = 'shuffle_test',
            # sig_samples=1000,
        )

    samples=5000
    rate = np.zeros(samples)
    for i in range(samples):
        print(i)
        data = random_state.standard_normal((100, 6))
        # data[:,2] += -0.3*data[:,0]
        # data[:,1] += data[:,2]
        dataframe = DataFrame(data, 
            # vector_vars={0:[(0,0), (1,0)], 1:[(2,0),(3,0)], 2:[(4,0),(5,0)]}
            )

        cmi.set_dataframe(dataframe)

        pval = cmi.run_test(
                X=[(0,0), (1,0), ], 
                Y=[(2,0),(3, 0)], 
                # Z=[(5,0)]
                Z = [(4, 0), (5, 0)]
                )[1]
        
        rate[i] = pval <= 0.05

        # cmi.get_model_selection_criterion(j=0, parents=[(1, 0), (2, 0)], tau_max=0, corrected_aic=False)

        # print(cmi.run_test(X=[(0,0),(1,0)], Y=[(2,0), (3, 0)], Z=[(5,0)]))
    print(rate.mean())
