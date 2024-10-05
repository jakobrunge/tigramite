from __future__ import print_function
import numpy as np
import warnings

from .parcorr import ParCorr
from .robust_parcorr import RobustParCorr
# from tigramite.independence_tests.parcorr import ParCorr
# from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite import data_processing as pp


class ParCorrWLS(ParCorr):
    r"""Weighted partial correlation test.

    Partial correlation is estimated through linear weighted least squares (WLS)
    regression and a test for non-zero linear Pearson correlation on the
    residuals.
    Either the variances, i.e. weights, are known, or they can be estimated using non-parametric regression
    (using k nearest neighbour).

    Notes
    -----
    To test :math:`X \perp Y | Z`, first :math:`Z` is regressed out from
    :math:`X` and :math:`Y` assuming the  model

    .. math::  X & =  Z \beta_X + \epsilon_{X} \\
        Y & =  Z \beta_Y + \epsilon_{Y}

    using WLS regression. Here, we do not assume homoskedasticity of the error terms.
    Then the dependency of the residuals is tested with
    the Pearson correlation test.

    .. math::  \rho\left(r_X, r_Y\right)

    For the ``significance='analytic'`` Student's-*t* distribution with
    :math:`T-D_Z-2` degrees of freedom is implemented.

    Parameters
    ----------
    gt_std_matrix: array-like, optional (default: None)
        Standard deviations of the noise of shape (T, nb_nodes)
    expert_knowledge: string or dict (default: time-dependent heteroskedasticity)
        Either string "time-dependent heteroskedasticity" meaning that every variable only has time-dependent
        heteroskedasticity, or string "homoskedasticity" where we assume homoskedasticity for all variables, or
        dictionary containing expert knowledge about heteroskedastic relationships as list of tuples or strings.
    window_size: int (default: 10)
        Number of nearest neighbours that we are using for estimating the variance function.
    robustify: bool (default: False)
        Indicates whether the robust partial correlation test should be used, i.e. whether the data should be
        transformed to normal marginals before testing
    **kwargs :
        Arguments passed on to Parent class ParCorr.
    """

    # documentation

    def __init__(self, gt_std_matrix=None,
                 expert_knowledge="time-dependent heteroskedasticity",
                 window_size=10, robustify=False, **kwargs):

        self.gt_std_matrix = gt_std_matrix
        self.expert_knowledge = expert_knowledge
        self.window_size = window_size
        self.robustify = robustify

        self.stds = None

        ParCorr.__init__(self,
                         recycle_residuals=False,  # Doesn't work with ParCorrWLS
                         **kwargs)
        self._measure = 'par_corr_wls'

    def _stds_preparation(self, X, Y, Z, tau_max=0, cut_off='2xtau_max', verbosity=0):
        """Helper function to bring expert_knowledge into standard form."""

        if self.expert_knowledge == "time-dependent heteroskedasticity":
            self.expert_knowledge = {variable: ["time-dependent heteroskedasticity"]
                                     for variable in range(self.dataframe.N)}
        elif self.expert_knowledge == "homoskedasticity":
            self.expert_knowledge = {}

    def _get_array(self, X, Y, Z, tau_max=0, cut_off='2xtau_max', verbosity=0, return_cleaned_xyz=True,
                   remove_constant_data=False):
        """Convenience wrapper around construct_array. Simultaneously, construct self.stds which needs to correspond
        to the variables in the array."""

        if self.measure in ['par_corr_wls']:
            if len(X) > 1 or len(Y) > 1:
                raise ValueError("X and Y for %s must be univariate." % self.measure)

        Z_orig = Z.copy()
        expert_knowledge_XY = []
        for var in [X[0][0], Y[0][0]]:
            if type(self.expert_knowledge) != str and var in self.expert_knowledge:
                expert_knowledge_XY += self.expert_knowledge[var]

        # add heteroskedasticity-inducing parents to Z (later these are removed again)
        # to obtain data cleaned the same as X and Y for weight estimation
        for item in expert_knowledge_XY:
            if type(item) == tuple:
                Z += [item]

        # Call the _get_array function of the parent class
        if remove_constant_data:
            array, xyz, XYZ, data_type, nonzero_array, nonzero_xyz, nonzero_XYZ, nonzero_data_type = super()._get_array(
                X=X, Y=Y, Z=Z,
                tau_max=tau_max,
                cut_off=cut_off,
                verbosity=verbosity,
                remove_constant_data=remove_constant_data)

            X, Y, Z = XYZ
            flat_XYZ = X + Y + Z
            counter = None if (len(Z) - len(Z_orig)) <= 0 else -1 * (len(Z) - len(Z_orig))
            data_hs_parent = {}
            for i, item in enumerate(expert_knowledge_XY):
                if type(item) == tuple:
                    data_hs_parent[item] = array[flat_XYZ.index(item), :]

            # stds have to correspond to array without the zero-rows
            nonzero_array_copy = nonzero_array.copy()
            nonzero_X, nonzero_Y, nonzero_Z = nonzero_XYZ
            self._get_std_estimation(nonzero_array_copy, nonzero_X, nonzero_Y, nonzero_Z, tau_max,
                                     cut_off, verbosity, data_hs_parent)

            if data_type:
                data_type = data_type[:counter]
                nonzero_data_type = nonzero_data_type[:counter]

            return array[:counter], xyz[:counter], (X, Y, Z[:counter]), data_type, \
                nonzero_array[:counter], nonzero_xyz[:counter], (nonzero_X, nonzero_Y, nonzero_Z[:counter]), \
                nonzero_data_type

        else:
            array, xyz, XYZ, data_type = super()._get_array(
                X=X, Y=Y, Z=Z,
                tau_max=tau_max,
                cut_off=cut_off,
                verbosity=verbosity,
                remove_constant_data=remove_constant_data)

            X, Y, Z = XYZ
            flat_XYZ = X + Y + Z
            counter = None if (len(Z) - len(Z_orig)) <= 0 else -1 * (len(Z) - len(Z_orig))

            dim, T = array.shape
            # save the data of the heteroskedasticity inducing parents to use for weight estimation
            data_hs_parent = np.zeros((len(expert_knowledge_XY), T))
            for i, item in enumerate(expert_knowledge_XY):
                if type(item) == tuple:
                    data_hs_parent[i, :] = array[flat_XYZ.index(item), :]

            array_copy = array.copy()
            self._get_std_estimation(array_copy, X, Y, Z, tau_max, cut_off, verbosity, data_hs_parent)
            if data_type:
                data_type = data_type[:counter]

            return array[:counter], xyz[:counter], (X, Y, Z[:counter]), data_type

    def _estimate_std_time(self, arr, target_var):
        """
        Estimate the standard deviations of the error terms using the squared-residuals approach. First calculate
        the absolute value of the residuals using OLS, then smooth them using a sliding window while keeping the time
        order of the residuals.
        In this way we can approximate variances that are time-dependent.

        Parameters
        ----------
        arr: array
            Data array of shape (dim, T)
        target_var: {0, 1}
            Variable to regress out conditions from.

        Returns
        -------
        std_est: array
            Standard deviation array of shape (T,)

        """
        dim, T = arr.shape
        dim_z = dim - 2
        # Standardization not necessary for variance estimation
        y = np.copy(arr[target_var, :])

        if dim_z > 0:
            z = arr[2:, :].T.copy()
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = abs(y - mean)
        else:
            resid = abs(y)

        # average variance within window
        std_est = np.concatenate(
            (np.ones(self.window_size - 1), np.convolve(resid, np.ones(self.window_size), 'valid') / self.window_size))
        return std_est

    def _estimate_std_parent(self, arr, target_var, target_lag, H, data_hs_parent):
        """
        Estimate the standard deviations of the error terms using a residual-based approach.
        First calculate the absolute value of the residuals using OLS, then smooth them by averaging over the k ones
        that are closest in H-value. In this way we are able to deal with parent-dependent heteroskedasticity.

        Parameters
        ----------
        arr: array
            Data array of shape (dim, T)
        target_var: {0, 1}
            Variable to obtain noise variance approximation for.
        target_lag: -int
            Lag of the variable to obtain noise variance approximation for.
        H: of the form [(var, -tau)], where var specifies the variable index and tau the time lag
            Variable to use for the sorting of the residuals, i.e. variable that the heteroskedasticity depends on.

        Returns
        -------
        std_est: array
            Standard deviation array of shape (T,)

        """
        dim, T = arr.shape
        dim_z = dim - 2
        y = np.copy(arr[target_var, :])

        if dim_z > 0:
            z = arr[2:, :].T.copy()
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = abs(y - mean)
            lag = H[1] + target_lag

            # order the residuals w.r.t. the heteroskedasticity-inducing parent corresponding to sample h
            h = data_hs_parent[-1 * lag:]

            ordered_z_ind = np.argsort(h)
            ordered_z_ind = ordered_z_ind * (ordered_z_ind > 0)
            revert_argsort = np.argsort(ordered_z_ind)

            truncate_resid = resid[np.abs(lag):]
            sorted_resid = truncate_resid[ordered_z_ind]

            # smooth the nearest neighbour residuals
            variance_est_sorted = np.concatenate(
                (np.ones(self.window_size - 1),
                 np.convolve(sorted_resid, np.ones(self.window_size), 'valid') / self.window_size,))
            std_est = variance_est_sorted[revert_argsort]
            std_est = np.concatenate((std_est, np.ones(np.abs(lag))))
            std_est = np.roll(std_est, np.abs(lag))
        else:
            resid = abs(y)
            std_est = np.concatenate(
                (np.ones(self.window_size - 1),
                 np.convolve(resid, np.ones(self.window_size), 'valid') / self.window_size))

        return std_est

    def _get_std_estimation(self, array, X, Y, Z=[], tau_max=0, cut_off='2xtau_max', verbosity=0, data_hs_parent=None):
        """Use expert knowledge on the heteroskedastic relationships contained in self.expert_knowledge to estimate the
        standard deviations of the error terms.
        The expert knowledge can specify whether there is sampling index / time dependent heteroskedasticity,
        heteroskedasticity with respect to a specified parent, or homoskedasticity.
        
        Parameters
        ----------
        array : array
            Data array of shape (dim, T)
            
        X, Y : list of tuples
            X,Y are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        Return
        ------
        stds: array-like
            Array of standard deviations of error terms for X and Y of shape (2, T).
        """
        self._stds_preparation(X, Y, Z, tau_max, cut_off, verbosity)

        dim, T = array.shape
        if self.gt_std_matrix is not None:
            stds_dataframe = pp.DataFrame(self.gt_std_matrix,
                                          mask=self.dataframe.mask,
                                          missing_flag=self.dataframe.missing_flag,
                                          datatime={0: np.arange(len(self.gt_std_matrix[:, 0]))})
            stds, _, _, _ = stds_dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                           tau_max=tau_max,
                                                           mask_type=self.mask_type,
                                                           return_cleaned_xyz=True,
                                                           do_checks=True,
                                                           remove_overlaps=True,
                                                           cut_off=cut_off,
                                                           verbosity=verbosity)
        else:
            stds = np.ones((2, T))
            for count, variable in enumerate([X[0], Y[0]]):
                # Here we assume that it is known what the heteroskedasticity function depends on for every variable
                if variable[0] in self.expert_knowledge:
                    hs_source = self.expert_knowledge[variable[0]][0]
                    if hs_source == "time-dependent heteroskedasticity":
                        stds[count] = self._estimate_std_time(array, count)
                    elif type(hs_source) is tuple:
                        stds[count] = self._estimate_std_parent(array, count, variable[1],
                                                                hs_source, data_hs_parent[hs_source])

        self.stds = stds
        return stds

    def _get_single_residuals(self, array, target_var,
                              standardize=False,
                              return_means=False):
        """Returns residuals of weighted linear multiple regression.

        Performs a WLS regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns.

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

        x_vals_sum = np.sum(array)
        x_vals_has_nan = np.isnan(x_vals_sum)
        if x_vals_has_nan:
            raise ValueError("array has nans")

        try:
            stds = self.stds[target_var]

        except TypeError:
            warnings.warn("No estimated or ground truth standard deviations supplied for weights. "
                          "Assume homoskedasticity, i.e. all weights are 1.")
            stds = np.ones(T)

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            x_vals_sum = np.sum(array)
            x_vals_has_nan = np.isnan(x_vals_sum)
            if x_vals_has_nan:
                raise ValueError("array has nans")
        y = np.copy(array[target_var, :])
        weights = np.diag(np.reciprocal(stds))

        if dim_z > 0:
            z = array[2:, :].T.copy()
            # include weights in z and y
            zw = np.dot(weights, z)
            yw = np.dot(y, weights)
            beta_hat = np.linalg.lstsq(zw, yw, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = np.dot(y - mean, weights)
            resid_vals_sum = np.sum(resid)
            resid_vals_has_nan = np.isnan(resid_vals_sum)
            if resid_vals_has_nan:
                raise ValueError("resid has nans")
        else:
            # resid = y
            resid = np.dot(y, weights)
            mean = None

        if return_means:
            return resid, mean
        return resid

    def get_dependence_measure(self, array, xyz):
        if self.robustify:
            array = RobustParCorr.trafo2normal(self, array)
        return ParCorr.get_dependence_measure(self, array, xyz)

    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        if self.robustify:
            array = RobustParCorr.trafo2normal(self, array)
        return ParCorr.get_shuffle_significance(self, array, xyz, value,
                                                return_null_dist=False)

    def get_model_selection_criterion(self, j, parents, tau_max=0, corrected_aic=False):
        """Returns Akaike's Information criterion modulo constants.

        Fits a linear model of the parents to variable j and returns the
        score. Leave-one-out cross-validation is asymptotically equivalent to
        AIC for ordinary linear regression models. Here used to determine
        optimal hyperparameters in PCMCI, in particular the pc_alpha value.

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
        X = [(j, 0)]  # dummy variable here
        Z = parents
        array, xyz, _, _ = self._get_array(X, Y, Z, tau_max=tau_max, verbosity=self.verbosity,
                                           return_cleaned_xyz=False)
        dim, T = array.shape

        # Transform to normal marginals
        if self.robustify:
            array = RobustParCorr.trafo2normal(self, array)

        y = self._get_single_residuals(array, target_var=1, return_means=False)
        # Get RSS
        rss = (y ** 2).sum()
        # Number of parameters
        p = dim - 1
        # Get AIC
        if corrected_aic:
            score = T * np.log(rss) + 2. * p + (2. * p ** 2 + 2. * p) / (T - p - 1)
        else:
            score = T * np.log(rss) + 2. * p
        return score

