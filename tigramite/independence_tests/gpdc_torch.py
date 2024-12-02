"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import json, warnings, os, pathlib
import numpy as np
import gc
import dcor
import torch
import gpytorch
from .LBFGS import FullBatchLBFGS
from .independence_tests_base import CondIndTest

class GaussProcRegTorch():
    r"""Gaussian processes abstract base class.

    GP is estimated with gpytorch. Note that the kernel's hyperparameters are
    optimized during fitting.

    When the null distribution is not analytically available, but can be
    precomputed with the function generate_and_save_nulldists(...) which saves
    a \*.npz file containing the null distribution for different sample sizes.
    This file can then be supplied as null_dist_filename.

    Assumes one-dimensional X, Y. But can be combined with PairwiseMultCI to
    obtain a test for multivariate X, Y.

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
                 checkpoint_size=None,
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
        # Size for batching
        self.checkpoint_size = checkpoint_size

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
            array = self.cond_ind_test.random_state.random((2, df))
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
                                    training_iter=50,
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
            Whether to return the log_marginal_likelihood of the fitted GP.

        training_iter : int, optional (default: 50)
            Number of training iterations.

        lr : float, optional (default: 0.1)
            Learning rate (default: 0.1).

        Returns
        -------
        resid [, mean, likelihood] : array-like
            The residual of the regression and optionally the estimated mean
            and/or the likelihood.
        """

        dim, T = array.shape

        if dim <= 2:
            if return_likelihood:
                return array[target_var, :], -np.inf
            return array[target_var, :]

        # Implement using PyTorch
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
            # if np.isnan(array).any():
            #     raise ValueError("Nans after standardizing, "
            #                      "possibly constant array!")

        target_series = array[target_var, :]
        z = array[2:].T.copy()
        if np.ndim(z) == 1:
            z = z.reshape(-1, 1)

        train_x = torch.tensor(z).float()
        train_y = torch.tensor(target_series).float()

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_device = torch.device(device_type)
        train_x, train_y = train_x.to(output_device), train_y.to(output_device)

        if device_type == 'cuda':
            # If GPU is available, use MultiGPU with Kernel Partitioning
            n_devices = torch.cuda.device_count()
            class mExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood, n_devices):
                    super(mExactGPModel, self).__init__(train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()
                    base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

                    self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                        base_covar_module, device_ids=range(n_devices),
                        output_device=output_device
                    )

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            def mtrain(train_x,
                      train_y,
                      n_devices,
                      output_device,
                      checkpoint_size,
                      preconditioner_size,
                      n_training_iter,
                      ):
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
                model = mExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
                model.train()
                likelihood.train()

                optimizer = FullBatchLBFGS(model.parameters(), lr=lr)
                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
                        gpytorch.settings.max_preconditioner_size(preconditioner_size):

                    def closure():
                        optimizer.zero_grad()
                        output = model(train_x)
                        loss = -mll(output, train_y)
                        return loss

                    loss = closure()
                    loss.backward()

                    for i in range(n_training_iter):
                        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                        loss, _, _, _, _, _, _, fail = optimizer.step(options)

                        '''print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, n_training_iter, loss.item(),
                            model.covar_module.module.base_kernel.lengthscale.item(),
                            model.likelihood.noise.item()
                        ))'''

                        if fail:
                            # print('Convergence reached!')
                            break

                return model, likelihood, mll

            def find_best_gpu_setting(train_x,
                                      train_y,
                                      n_devices,
                                      output_device,
                                      preconditioner_size
                                      ):
                N = train_x.size(0)

                # Find the optimum partition/checkpoint size by decreasing in powers of 2
                # Start with no partitioning (size = 0)
                settings = [0] + [int(n) for n in np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N))))]

                for checkpoint_size in settings:
                    print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
                    try:
                        # Try a full forward and backward pass with this setting to check memory usage
                        _, _, _ = mtrain(train_x, train_y,
                                     n_devices=n_devices, output_device=output_device,
                                     checkpoint_size=checkpoint_size,
                                     preconditioner_size=preconditioner_size, n_training_iter=1)

                        # when successful, break out of for-loop and jump to finally block
                        break
                    except RuntimeError as e:
                        pass
                    except AttributeError as e:
                        pass
                    finally:
                        # handle CUDA OOM error
                        gc.collect()
                        torch.cuda.empty_cache()
                return checkpoint_size

            # Set a large enough preconditioner size to reduce the number of CG iterations run
            preconditioner_size = 100
            if self.checkpoint_size is None:
                self.checkpoint_size = find_best_gpu_setting(train_x, train_y,
                                                        n_devices=n_devices,
                                                        output_device=output_device,
                                                        preconditioner_size=preconditioner_size)

            model, likelihood, mll = mtrain(train_x, train_y,
                                      n_devices=n_devices, output_device=output_device,
                                      checkpoint_size=self.checkpoint_size,
                                      preconditioner_size=100,
                                      n_training_iter=training_iter)

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
                mean = model(train_x).loc.detach()
                loglik = mll(model(train_x), train_y)*T

            resid = (train_y - mean).detach().cpu().numpy()
            mean = mean.detach().cpu().numpy()

        else:
            # If only CPU is available, we will use the simplest form of GP model, exact inference
            class ExactGPModel(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(ExactGPModel, self).__init__(
                        train_x, train_y, likelihood)
                    self.mean_module = gpytorch.means.ConstantMean()

                    # We only use the RBF kernel here, the WhiteNoiseKernel is deprecated
                    # and its featured integrated into the Likelihood-Module.
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x, train_y, likelihood)

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
                loglik = mll(model(train_x), train_y) * T

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
        array, xyz, _ = \
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
    r"""GPDC conditional independence test based on Gaussian processes and distance correlation. Here with gpytorch implementation.

    GPDC is based on a Gaussian process (GP) regression and a distance
    correlation test on the residuals [2]_. GP is estimated with gpytorch.
    The distance correlation test is implemented with the dcor package available
    from pip. Here the null distribution is not analytically available, but can be
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
                                    training_iter=50,
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

        training_iter : int, optional (default: 50)
            Number of training iterations.

        lr : float, optional (default: 0.1)
            Learning rate (default: 0.1).

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

    def get_dependence_measure(self, array, xyz, data_type=None):
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
        generate_and_save_nulldists(...) which saves a *.npz file containing
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
        val = dcor.distance_correlation(x_vals, y_vals, method='AVL')
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

        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval

    def get_analytic_significance(self, value, T, dim, xyz):
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

        xyz : array of ints
            XYZ identifier array of shape (dim,).

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

