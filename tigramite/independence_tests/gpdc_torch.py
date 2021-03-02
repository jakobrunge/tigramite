"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0



from __future__ import print_function
import numpy as np
import gpytorch
from gpytorch.kernels import RBFKernel
import torch

from .independence_tests_base import CondIndTest

try:
    from tigramite import tigramite_cython_code
except:
    print("Could not import packages for CMIknn and GPDC estimation")


class GaussProcRegTorch():
    r"""Gaussian processes abstract base class.

    GP is estimated with gpytorch. Note that the kernel's hyperparameters are 
    optimized during fitting.

    When the null distribution is not analytically available, but can be
    precomputed with the function generate_and_save_nulldists(...) which saves
    a \*.npz file containing the null distribution for different sample sizes.
    This file can then be supplied as null_dist_filename.

    Parameters
    ----------
    null_samples : int
        Number of null samples to use

    cond_ind_test : CondIndTest
        Conditional independence test that this Gaussian Proccess Regressor will
        calculate the null distribution for.  This is used to grab the
        get_dependence_measure function.

    null_dist_filename : str, otional (default: None)
        Path to file containing null distribution.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self,
                 null_samples,
                 cond_ind_test,
                 null_dist_filename=None,
                 verbosity=0):
        # Set the dependence measure function
        self.cond_ind_test = cond_ind_test
        # Set member variables
        self.verbosity = verbosity
        # Set the null distribution defaults
        self.null_samples = null_samples
        self.null_dists = {}
        self.null_dist_filename = null_dist_filename
        # Check if we are loading a null distrubtion from a cached file
        if self.null_dist_filename is not None:
            self.null_dists, self.null_samples = \
                self._load_nulldist(self.null_dist_filename)

    def _load_nulldist(self, filename):
        r"""
        Load a precomputed null distribution from a \*.npz file.  This
        distribution can be calculated using generate_and_save_nulldists(...).

        Parameters
        ----------
        filename : strng
            Path to the \*.npz file

        Returns
        -------
        null_dists, null_samples : dict, int
            The null distirbution as a dictionary of distributions keyed by
            sample size, the number of null samples in total.
        """
        null_dist_file = np.load(filename)
        null_dists = dict(zip(null_dist_file['T'],
                              null_dist_file['exact_dist']))
        null_samples = len(null_dist_file['exact_dist'][0])
        return null_dists, null_samples

    def _generate_nulldist(self, df,
                           add_to_null_dists=True):
        """Generates null distribution for pairwise independence tests.

        Generates the null distribution for sample size df. Assumes pairwise
        samples transformed to uniform marginals. Uses get_dependence_measure
        available in class and generates self.sig_samples random samples. Adds
        the null distributions to self.null_dists.

        Parameters
        ----------
        df : int
            Degrees of freedom / sample size to generate null distribution for.
        add_to_null_dists : bool, optional (default: True)
            Whether to add the null dist to the dictionary of null dists or
            just return it.

        Returns
        -------
        null_dist : array of shape [df,]
            Only returned,if add_to_null_dists is False.
        """

        if self.verbosity > 0:
            print("Generating null distribution for df = %d. " % df)
            if add_to_null_dists:
                print("For faster computations, run function "
                      "generate_and_save_nulldists(...) to "
                      "precompute null distribution and load *.npz file with "
                      "argument null_dist_filename")

        xyz = np.array([0, 1])

        null_dist = np.zeros(self.null_samples)
        for i in range(self.null_samples):
            array = np.random.rand(2, df)
            null_dist[i] = self.cond_ind_test.get_dependence_measure(
                array, xyz)

        null_dist.sort()
        if add_to_null_dists:
            self.null_dists[df] = null_dist
        return null_dist

    def _generate_and_save_nulldists(self, sample_sizes, null_dist_filename):
        """Generates and saves null distribution for pairwise independence
        tests.

        Generates the null distribution for different sample sizes. Calls
        generate_nulldist. Null dists are saved to disk as
        self.null_dist_filename.npz. Also adds the null distributions to
        self.null_dists.

        Parameters
        ----------
        sample_sizes : list
            List of sample sizes.

        null_dist_filename : str
            Name to save file containing null distributions.
        """

        self.null_dist_filename = null_dist_filename

        null_dists = np.zeros((len(sample_sizes), self.null_samples))

        for iT, T in enumerate(sample_sizes):
            null_dists[iT] = self._generate_nulldist(
                T, add_to_null_dists=False)
            self.null_dists[T] = null_dists[iT]

        np.savez("%s" % null_dist_filename,
                 exact_dist=null_dists,
                 T=np.array(sample_sizes))


    def _get_single_residuals(self, array, target_var,
                                    return_means=False,
                                    standardize=True,
                                    return_likelihood=False,
                                    training_iter=25,
                                    lr=0.1):
        """Returns residuals of Gaussian process regression.

        Performs a GP regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated mean and the likelihood.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        target_var : {0, 1}
            Variable to regress out conditions from.

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand.

        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.

        return_likelihood : bool, optional (default: False)
            Whether to return the log_marginal_likelihood of the fitted GP

        Returns
        -------
        resid [, mean, likelihood] : array-like
            The residual of the regression and optionally the estimated mean
            and/or the likelihood.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        GPU = torch.cuda.is_available()
        dim, T = array.shape

        if dim <= 2:
            if return_likelihood:
                return array[target_var, :], -np.inf
            return array[target_var, :]

        # Implement using PyTorch
        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            array /= array.std(axis=1).reshape(dim, 1)
            if np.isnan(array).any():
                raise ValueError("Nans after standardizing, "
                                 "possibly constant array!")

        target_series = array[target_var, :]
        z = np.fastCopyAndTranspose(array[2:])
        if np.ndim(z) == 1:
            z = z.reshape(-1, 1)

        train_x = torch.tensor(z).float()
        train_y = torch.tensor(target_series).float()

        if GPU:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        # We will use the simplest form of GP model, exact inference
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(
                    train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()

                # We only use the RBF kernel here, the WhiteNoiseKernel is deprecated
                # and its featured integrated into the Likelihood-Module.
                self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        if GPU:
            likelihood = likelihood.cuda()
            model = model.cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = model(train_x).loc.detach()
            loglik = mll(model(train_x), train_y)*T

        if GPU:
            resid = (train_y - mean).detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()
        else:
            resid = (train_y - mean).detach().numpy()
            mean = mean.detach().numpy()

        if return_means and not return_likelihood:
            return resid, mean
        elif return_likelihood and not return_means:
            return resid, loglik
        elif return_means and return_likelihood:
            return resid, mean, loglik
        return resid

    def _get_model_selection_criterion(self, j, parents, tau_max=0):
        """Returns log marginal likelihood for GP regression.

        Fits a GP model of the parents to variable j and returns the negative
        log marginal likelihood as a model selection score. Is used to determine
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
        X = [(j, 0)]   # dummy variable here
        Z = parents
        array, xyz = \
            self.cond_ind_test.dataframe.construct_array(
                X=X, Y=Y, Z=Z,
                tau_max=tau_max,
                mask_type=self.cond_ind_test.mask_type,
                return_cleaned_xyz=False,
                do_checks=True,
                verbosity=self.verbosity)

        dim, T = array.shape

        _, logli = self._get_single_residuals(array,
                                              target_var=1,
                                              return_likelihood=True)

        score = -logli
        return score


class GPDCtorch(CondIndTest):
    r"""GPDC conditional independence test based on Gaussian processes and
        distance correlation. Here with gpytorch implementation.

    GPDC is based on a Gaussian process (GP) regression and a distance
    correlation test on the residuals [2]_. GP is estimated with gpytorch.
    The distance correlation test is implemented with
    cython. Here the null distribution is not analytically available, but can be
    precomputed with the function generate_and_save_nulldists(...) which saves a
    \*.npz file containing the null distribution for different sample sizes.
    This file can then be supplied as null_dist_filename.

    Notes
    -----

    GPDC is based on a Gaussian process (GP) regression and a distance
    correlation test on the residuals. Distance correlation is described in
    [2]_. To test :math:`X \perp Y | Z`, first :math:`Z` is regressed out from
    :math:`X` and :math:`Y` assuming the  model

    .. math::  X & =  f_X(Z) + \epsilon_{X} \\
        Y & =  f_Y(Z) + \epsilon_{Y}  \\
        \epsilon_{X,Y} &\sim \mathcal{N}(0, \sigma^2)

    using GP regression. Here :math:`\sigma^2` and the kernel bandwidth are
    optimzed using ``gpytorch``. Then the residuals  are transformed to uniform
    marginals yielding :math:`r_X,r_Y` and their dependency is tested with

    .. math::  \mathcal{R}\left(r_X, r_Y\right)

    The null distribution of the distance correlation should be pre-computed.
    Otherwise it is computed during runtime.

    The cython-code for distance correlation is Copyright (c) 2012, Florian
    Finkernagel (https://gist.github.com/ffinkernagel/2960386).

    References
    ----------
    .. [2] Gabor J. Szekely, Maria L. Rizzo, and Nail K. Bakirov: Measuring and
           testing dependence by correlation of distances,
           https://arxiv.org/abs/0803.4101

    Parameters
    ----------
    null_dist_filename : str, otional (default: None)
        Path to file containing null distribution.

    **kwargs :
        Arguments passed on to parent class GaussProcRegTorch.

    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 null_dist_filename=None,
                 **kwargs):
        self._measure = 'gp_dc'
        self.two_sided = False
        self.residual_based = True
        # Call the parent constructor
        CondIndTest.__init__(self, **kwargs)
        # Build the regressor
        self.gauss_pr = GaussProcRegTorch(self.sig_samples,
                                     self,
                                     null_dist_filename=null_dist_filename,
                                     verbosity=self.verbosity)

        if self.verbosity > 0:
            print("null_dist_filename = %s" % self.gauss_pr.null_dist_filename)
            print("")

    def _load_nulldist(self, filename):
        r"""
        Load a precomputed null distribution from a \*.npz file.  This
        distribution can be calculated using generate_and_save_nulldists(...).

        Parameters
        ----------
        filename : strng
            Path to the \*.npz file

        Returns
        -------
        null_dists, null_samples : dict, int
            The null distirbution as a dictionary of distributions keyed by
            sample size, the number of null samples in total.
        """
        return self.gauss_pr._load_nulldist(filename)

    def generate_nulldist(self, df, add_to_null_dists=True):
        """Generates null distribution for pairwise independence tests.

        Generates the null distribution for sample size df. Assumes pairwise
        samples transformed to uniform marginals. Uses get_dependence_measure
        available in class and generates self.sig_samples random samples. Adds
        the null distributions to self.gauss_pr.null_dists.

        Parameters
        ----------
        df : int
            Degrees of freedom / sample size to generate null distribution for.

        add_to_null_dists : bool, optional (default: True)
            Whether to add the null dist to the dictionary of null dists or
            just return it.

        Returns
        -------
        null_dist : array of shape [df,]
            Only returned,if add_to_null_dists is False.
        """
        return self.gauss_pr._generate_nulldist(df, add_to_null_dists)

    def generate_and_save_nulldists(self, sample_sizes, null_dist_filename):
        """Generates and saves null distribution for pairwise independence
        tests.

        Generates the null distribution for different sample sizes. Calls
        generate_nulldist. Null dists are saved to disk as
        self.null_dist_filename.npz. Also adds the null distributions to
        self.gauss_pr.null_dists.

        Parameters
        ----------
        sample_sizes : list
            List of sample sizes.

        null_dist_filename : str
            Name to save file containing null distributions.
        """
        self.gauss_pr._generate_and_save_nulldists(sample_sizes,
                                                   null_dist_filename)


    def _get_single_residuals(self, array, target_var,
                                    return_means=False,
                                    standardize=True,
                                    return_likelihood=False,
                                    training_iter=25,
                                    lr=0.1):
        """Returns residuals of Gaussian process regression.

        Performs a GP regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated mean and the likelihood.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        target_var : {0, 1}
            Variable to regress out conditions from.

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand.

        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.

        return_likelihood : bool, optional (default: False)
            Whether to return the log_marginal_likelihood of the fitted GP

        Returns
        -------
        resid [, mean, likelihood] : array-like
            The residual of the regression and optionally the estimated mean
            and/or the likelihood.
        """
        return self.gauss_pr._get_single_residuals(
            array, target_var,
            return_means,
            standardize,
            return_likelihood,
            training_iter,
            lr)

    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """Returns log marginal likelihood for GP regression.

        Fits a GP model of the parents to variable j and returns the negative
        log marginal likelihood as a model selection score. Is used to determine
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
        return self.gauss_pr._get_model_selection_criterion(j, parents, tau_max)

    def get_dependence_measure(self, array, xyz):
        """Return GPDC measure.

        Estimated as the distance correlation of the residuals of a GP
        regression.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            GPDC test statistic.
        """

        x_vals = self._get_single_residuals(array, target_var=0)
        y_vals = self._get_single_residuals(array, target_var=1)
        val = self._get_dcorr(np.array([x_vals, y_vals]))
        return val

    def _get_dcorr(self, array_resid):
        """Return distance correlation coefficient.

        The variables are transformed to uniform marginals using the empirical
        cumulative distribution function beforehand. Here the null distribution
        is not analytically available, but can be precomputed with the function
        generate_and_save_nulldists(...) which saves a \*.npz file containing
        the null distribution for different sample sizes. This file can then be
        supplied as null_dist_filename.

        Parameters
        ----------
        array_resid : array-like
            data array must be of shape (2, T)

        Returns
        -------
        val : float
            Distance correlation coefficient.
        """
        # Remove ties before applying transformation to uniform marginals
        # array_resid = self._remove_ties(array_resid, verbosity=4)
        x_vals, y_vals = self._trafo2uniform(array_resid)

        _, val, _, _ = tigramite_cython_code.dcov_all(x_vals, y_vals)
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

        x_vals = self._get_single_residuals(array, target_var=0)
        y_vals = self._get_single_residuals(array, target_var=1)
        array_resid = np.array([x_vals, y_vals])
        xyz_resid = np.array([0, 1])

        null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval

    def get_analytic_significance(self, value, T, dim):
        """Returns p-value for the distance correlation coefficient.

        The null distribution for necessary degrees of freedom (df) is loaded.
        If not available, the null distribution is generated with the function
        generate_nulldist(). It is recommended to generate the nulldists for a
        wide range of sample sizes beforehand with the function
        generate_and_save_nulldists(...). The distance correlation coefficient
        is one-sided. If the degrees of freedom are less than 1, numpy.nan is
        returned.

        Parameters
        ----------
        value : float
            Test statistic value.

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        Returns
        -------
        pval : float or numpy.nan
            p-value.
        """

        # GP regression approximately doesn't cost degrees of freedom
        df = T

        if df < 1:
            pval = np.nan
        else:
            # idx_near = (np.abs(self.sample_sizes - df)).argmin()
            if int(df) not in list(self.gauss_pr.null_dists):
                # if np.abs(self.sample_sizes[idx_near] - df) / float(df) > 0.01:
                if self.verbosity > 0:
                    print("Null distribution for GPDC not available "
                          "for deg. of freed. = %d." % df)
                self.generate_nulldist(df)
            null_dist_here = self.gauss_pr.null_dists[int(df)]
            pval = np.mean(null_dist_here > np.abs(value))
        return pval

# def dcov_all(x, y):
#     """Calculate distance covariance, distance correlation,
#     distance variance of x sample and distance variance of y sample"""

#     GPU_FLAG = torch.cuda.is_available()


#     x = np.reshape(x, (x.shape[0], 1))
#     y = np.reshape(y, (y.shape[0], 1))

#     x = torch.tensor(x)
#     y = torch.tensor(y)

#     mean_x_0 = torch.zeros(x.shape[0],1)
#     mean_x_1 = torch.zeros(x.shape[0],1)
#     mean_y_0 = torch.zeros(x.shape[0],1)
#     mean_y_1 = torch.zeros(x.shape[0],1)
#     x_abs_diff = torch.zeros(x.shape[0],1)
#     y_abs_diff = torch.zeros(x.shape[0],1)
#     x_all_mean = 0
#     y_all_mean = 0
#     SUM_VAL = 0

#     if GPU_FLAG:
#         x = x.cuda()
#         y = y.cuda()
#         mean_x_0 = mean_x_0.cuda()
#         mean_x_1 = mean_x_1.cuda()
#         mean_y_0 = mean_y_0.cuda()
#         mean_y_1 = mean_y_1.cuda()
#         x_abs_diff = x_abs_diff.cuda()
#         y_abs_diff = y_abs_diff.cuda()


#     # Calculate means
#     for i in range(x.shape[0]):
#         x_abs_diff = torch.abs(x - x[i,:])
#         y_abs_diff = torch.abs(y - y[i,:])
#         mean_x_0 += x_abs_diff
#         mean_y_0 += y_abs_diff

#         x_all_mean += torch.sum(mean_x_0)
#         y_all_mean += torch.sum(mean_y_0)

#         mean_x_1[i] = torch.mean(x_abs_diff)
#         mean_y_1[i] = torch.mean(y_abs_diff)

#     mean_x_0 = mean_x_0/mean_x_0.shape[0]
#     mean_y_0 = mean_y_0/mean_y_0.shape[0]

#     x_all_mean = x_all_mean/mean_x_0.shape[0]**2
#     y_all_mean = y_all_mean/mean_y_0.shape[0]**2


#     for i in range(x.shape[0]):
#         X_i = torch.abs(x - x[i,:]) - mean_x_0 - mean_x_1[i] + x_all_mean
#         Y_i = torch.abs(y - y[i,:]) - mean_y_0 - mean_y_1[i] + y_all_mean
#         SUM_VAL += torch.sum(torch.mul(X_i, Y_i))

#     SUM_VAL = SUM_VAL/X_i.shape[0]**2

#     # Todo: dCor, dVarX, dVarY
#     # Todo: Better batching?

#     if GPU_FLAG:
#         dCov = 1 # dCov.cpu()

#     return 1 #dCov.numpy()


if __name__ == "__main__":
    import tigramite.data_processing as pp
    import tigramite.plotting as tp
    import matplotlib.pyplot as plt
    from time import time
    import sys


    # Example process to play around with
    # Each key refers to a variable and the incoming links are supplied
    # as a list of format [((var, -lag), coeff, function), ...]

    def lin_f(x):
        return x

    def nonlin_f(x):
        return x + 5.0 * x ** 2 * np.exp(-(x ** 2) / 20.0)

    print(torch.cuda.is_available())

    for samples in [10000, 20000, 50000, 100000]:
        torch.cuda.empty_cache()
        test_a = np.random.normal(0,1, samples)
        test_b = np.random.normal(0,1, samples)

        start = time()
        distance_corr = dcov_all(test_a, test_b)
        end_torch = time() - start

        print("###################", flush=True)
        print(f"T={samples}.", flush=True)
        print(f"PyTorch (GPU) took {end_torch} seconds.\n", flush=True)


        start = time()
        distance_corr = tigramite_cython_code.dcov_all(test_a, test_b)
        end_cython = time() - start

        print(f"Cython took {end_cython} seconds.\n", flush=True)

    sys.exit(0)

    N = 50

    links = {i: [((i, -1),
                    np.random.uniform(low=0.4, high=0.9, size=None),
                    np.random.choice([lin_f, nonlin_f], p=[0.7, 0.3]))]
                for i in range(N)}

    for i in range(1,N):
        links[i].append(((i-1, -1),
                np.random.uniform(low=0.4, high=0.9, size=None),
                np.random.choice([lin_f, nonlin_f], p=[0.7, 0.3])))

    torch_time_lst = []
    sklearn_time_lst = []

    for T in [25000, 40000, 55000]:

        #torch.cuda.empty_cache()
        data, nonstat = pp.structural_causal_process(links, T=T)
        data = data.T
        gpdc = GPDC()


        #start_sklearn = time()
        #get_resid_sklearn = gpdc._get_single_residuals(data, 0,
        #                                               return_means=True,
        #                                               standardize=True,
        #                                               return_likelihood=True)
        #end_sklearn = time() - start_sklearn
        #sklearn_time_lst.append(end_sklearn)

        start_torch = time()
        get_resid_torch = gpdc._get_single_residuals(data, 0,
                                                           return_means=True,
                                                           standardize=True,
                                                           return_likelihood=True)

        end_torch = time() - start_torch
        torch_time_lst.append(end_torch)

        print("###################", flush=True)
        print(f"Data shape: {data.shape}", flush=True)
        #print(f"Sklearn took {end_sklearn} seconds.", flush=True)
        print(f"Torch took {end_torch} seconds.", flush=True)

        print("\n")

        try:
            #print(f"Sklearn loglik: {get_resid_sklearn[2]}")
            print(f"Torch loglik: {get_resid_torch[2]}")
            print(f"\n")
        except IndexError:
            print("The likelihood was not returned.")

    #print(f"Sklearn times: {sklearn_time_lst}")
    print(f"Torch times: {torch_time_lst}")

