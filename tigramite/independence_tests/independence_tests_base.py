"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import warnings
import math
import abc
import random
import numpy as np
import six
from numba import jit
import time
from hashlib import sha1

@jit(nopython=True)
def _permutation_and_shuffle(block_starts, n_blks, x_array):
    blk_starts = np.copy(block_starts)
    random.shuffle(blk_starts)
    x_shuffled = np.zeros(n_blks, dtype=np.int64)
    x_shuffled[0::1] = x_array[blk_starts]
    return x_shuffled

@six.add_metaclass(abc.ABCMeta)
class CondIndTest():
    """Base class of conditional independence tests.

    Provides useful general functions for different independence tests such as
    shuffle significance testing and bootstrap confidence estimation. Also
    handles masked samples. Other test classes can inherit from this class.

    Parameters
    ----------
    seed : int, optional(default = 42)
        Seed for RandomState (default_rng)

    mask_type : str, optional (default = None)
        Must be in {None, 'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence measure
        I(X; Y | Z) the samples should be masked. If None, the mask is not used. 
        Explained in tutorial on masking and missing values.

    significance : str, optional (default: 'analytic')
        Type of significance test to use. In this package 'analytic',
        'fixed_thres' and 'shuffle_test' are available.

    fixed_thres : float, optional (default: 0.1)
        If significance is 'fixed_thres', this specifies the threshold for the
        absolute value of the dependence measure.

    sig_samples : int, optional (default: 1000)
        Number of samples for shuffle significance test.

    sig_blocklength : int, optional (default: None)
        Block length for block-shuffle significance test. If None, the
        block length is determined from the decay of the autocovariance as
        explained in [1]_.

    confidence : str, optional (default: None)
        Specify type of confidence estimation. If False, numpy.nan is returned.
        'bootstrap' can be used with any test, for ParCorr also 'analytic' is
        implemented.

    conf_lev : float, optional (default: 0.9)
        Two-sided confidence interval.

    conf_samples : int, optional (default: 100)
        Number of samples for bootstrap.

    conf_blocklength : int, optional (default: None)
        Block length for block-bootstrap. If None, the block length is
        determined from the decay of the autocovariance as explained in [1]_.

    recycle_residuals : bool, optional (default: False)
        Specifies whether residuals should be stored. This may be faster, but
        can cost considerable memory.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """
    @abc.abstractmethod
    def get_dependence_measure(self, array, xyz):
        """
        Abstract function that all concrete classes must instantiate.
        """
        pass

    @abc.abstractproperty
    def measure(self):
        """
        Abstract property to store the type of independence test.
        """
        pass

    def __init__(self,
                 seed=52,
                 mask_type=None,
                 significance='analytic',
                 fixed_thres=0.1,
                 sig_samples=1000,
                 sig_blocklength=None,
                 confidence=None,
                 conf_lev=0.9,
                 conf_samples=100,
                 conf_blocklength=None,
                 recycle_residuals=False,
                 verbosity=0): 
        # Set the dataframe to None for now, will be reset during pcmci call
        self.dataframe = None
        # Set the options
        self.random_state = np.random.default_rng(seed)
        random.seed(seed)
        self.significance = significance
        self.sig_samples = sig_samples
        self.sig_blocklength = sig_blocklength
        self.fixed_thres = fixed_thres
        self.verbosity = verbosity
        self.cached_ci_results = {}
        # If we recycle residuals, then set up a residual cache
        self.recycle_residuals = recycle_residuals
        if self.recycle_residuals:
            self.residuals = {}
        # If we use a mask, we cannot recycle residuals
        self.set_mask_type(mask_type)

        # Set the confidence type and details
        self.confidence = confidence
        self.conf_lev = conf_lev
        self.conf_samples = conf_samples
        self.conf_blocklength = conf_blocklength

        # Print information about the
        if self.verbosity > 0:
            self.print_info()

    def set_mask_type(self, mask_type):
        """
        Setter for mask type to ensure that this option does not clash with
        recycle_residuals.

        Parameters
        ----------
        mask_type : str
            Must be in {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence measure
            I(X; Y | Z) the samples should be masked. If None, the mask is not used. 
            Explained in tutorial on masking and missing values.
        """
        # Set the mask type
        self.mask_type = mask_type
        # Check if this clashes with residual recycling
        if self.mask_type is not None:
            if self.recycle_residuals is True:
                warnings.warn("Using a mask disables recycling residuals.")
            self.recycle_residuals = False
        # Check the mask type is keyed correctly
        self._check_mask_type()

    def print_info(self):
        """
        Print information about the conditional independence test parameters
        """
        info_str = "\n# Initialize conditional independence test\n\nParameters:"
        info_str += "\nindependence test = %s" % self.measure
        info_str += "\nsignificance = %s" % self.significance
        # Check if we are using a shuffle test
        if self.significance == 'shuffle_test':
            info_str += "\nsig_samples = %s" % self.sig_samples
            info_str += "\nsig_blocklength = %s" % self.sig_blocklength
        # Check if we are using a fixed threshold
        elif self.significance == 'fixed_thres':
            info_str += "\nfixed_thres = %s" % self.fixed_thres
        # Check if we have a confidence type
        if self.confidence:
            info_str += "\nconfidence = %s" % self.confidence
            info_str += "\nconf_lev = %s" % self.conf_lev
        # Check if this confidence type is boostrapping
        if self.confidence == 'bootstrap':
            info_str += "\nconf_samples = %s" % self.conf_samples
            info_str += "\nconf_blocklength = %s" %self.conf_blocklength
        # Check if we use a non-trivial mask type
        if self.mask_type is not None:
            info_str += "\nmask_type = %s" % self.mask_type
        # Check if we are recycling residuals or not
        if self.recycle_residuals:
            info_str += "\nrecycle_residuals = %s" % self.recycle_residuals
        # Print the information string
        print(info_str)

    def _check_mask_type(self):
        """
        mask_type : str, optional (default = None)
            Must be in {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence measure
            I(X; Y | Z) the samples should be masked. If None, the mask is not used. 
            Explained in tutorial on masking and missing values.
        """
        if self.mask_type is not None:
            mask_set = set(self.mask_type) - set(['x', 'y', 'z'])
            if mask_set:
                err_msg = "mask_type = %s," % self.mask_type + " but must be" +\
                          " list containing 'x','y','z', or any combination"
                raise ValueError(err_msg)

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

    def get_analytic_significance(self, value, T, dim):
        """
        Base class assumption that this is not implemented.  Concrete classes
        should override when possible.
        """
        raise NotImplementedError("Analytic significance not"+\
                                  " implemented for %s" % self.measure)

    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        """
        Base class assumption that this is not implemented.  Concrete classes
        should override when possible.
        """
        raise NotImplementedError("Shuffle significance not"+\
                                  " implemented for %s" % self.measure)

    def _get_single_residuals(self, array, target_var,
                              standardize=True, return_means=False):
        """
        Base class assumption that this is not implemented.  Concrete classes
        should override when possible.
        """
        raise NotImplementedError("Residual calculation not"+\
                                  " implemented for %s" % self.measure)

    def set_dataframe(self, dataframe):
        """Initialize and check the dataframe.

        Parameters
        ----------
        dataframe : data object
            Set tigramite dataframe object. It must have the attributes
            dataframe.values yielding a numpy array of shape (observations T,
            variables N) and optionally a mask of the same shape and a missing
            values flag.

        """
        self.dataframe = dataframe
        if self.mask_type is not None:
            dataframe._check_mask(require_mask=True)

    def _keyfy(self, x, z):
        """Helper function to make lists unique."""
        return (tuple(set(x)), tuple(set(z)))

    def _get_array(self, X, Y, Z, tau_max=0, cut_off='2xtau_max',
                   verbosity=0):
        """Convencience wrapper around construct_array."""

        if self.measure in ['par_corr']:
            if len(X) > 1 or len(Y) > 1:
                raise ValueError("X and Y for %s must be univariate." %
                                        self.measure)
        # Call the wrapped function
        return self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                              tau_max=tau_max,
                                              mask_type=self.mask_type,
                                              return_cleaned_xyz=True,
                                              do_checks=True,
                                              cut_off=cut_off,
                                              verbosity=verbosity)

    def _get_array_hash(self, array, xyz, XYZ):
        """Helper function to get hash of array.

        For a CI test X _|_ Y | Z the order of variables within X or Y or Z 
        does not matter and also the order X and Y can be swapped.
        Hence, to compare hashes of the whole array, we order accordingly
        to create a unique, order-independent hash. 

        Parameters
        ----------
        array : Data array of shape (dim, T)
            Data array.
        xyz : array
            Identifier array of shape (dim,) identifying which row in array
            corresponds to X, Y, and Z
        XYZ : list of tuples

        Returns
        -------
        combined_hash : str
            Hash that identifies uniquely an array of XYZ      
        """

        X, Y, Z = XYZ

        # First check whether CI result was already computed
        # by checking whether hash of (xyz, array) already exists
        # Individually sort X, Y, Z since for a CI test it does not matter
        # how they are aranged
        x_orderd = sorted(range(len(X)), key=X.__getitem__)
        arr_x = array[xyz==0][x_orderd]
        x_hash = sha1(np.ascontiguousarray(arr_x)).hexdigest()

        y_orderd = sorted(range(len(Y)), key=Y.__getitem__)
        arr_y = array[xyz==1][y_orderd]
        y_hash = sha1(np.ascontiguousarray(arr_y)).hexdigest()

        z_orderd = sorted(range(len(Z)), key=Z.__getitem__)
        arr_z = array[xyz==2][z_orderd]
        z_hash = sha1(np.ascontiguousarray(arr_z)).hexdigest()

        sorted_xy = sorted([x_hash, y_hash])
        combined_hash = (sorted_xy[0], sorted_xy[1], z_hash)
        return combined_hash
    
    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """Perform conditional independence test.

        Calls the dependence measure and signficicance test functions. The child
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

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """

        # Get the array to test on
        array, xyz, XYZ = self._get_array(X, Y, Z, tau_max, cut_off)
        X, Y, Z = XYZ
        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.any(np.isnan(array)):
            raise ValueError("nans in the array!")

        combined_hash = self._get_array_hash(array, xyz, XYZ)

        if combined_hash in self.cached_ci_results.keys():
            cached = True
            val, pval = self.cached_ci_results[combined_hash]
        else:
            cached = False
            # Get the dependence measure, recycling residuals if need be
            if self.measure in ['chi-square']:
                val, pval = self._get_dependence_measure_recycle(X, Y, Z, xyz, array)
            else:
                val = self._get_dependence_measure_recycle(X, Y, Z, xyz, array)
                # Get the p-value
                pval = self.get_significance(val, array, xyz, T, dim)
            self.cached_ci_results[combined_hash] = (val, pval)

        if self.verbosity > 2:
            self._print_cond_ind_results(val=val, pval=pval, cached=cached,
                                         conf=None)
        # Return the value and the pvalue
        return val, pval

    def run_test_raw(self, x, y, z=None):
        """Perform conditional independence test directly on input arrays x, y, z.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.

        Parameters
        ----------
        x, y, z : arrays
            x,y,z are of the form (samples, dimension).

        Returns
        -------
        val, pval : Tuple of floats

            The test statistic value and the p-value.
        """

        if np.ndim(x) != 2 or np.ndim(y) != 2:
            raise ValueError("x,y must be arrays of shape (samples, dimension)"
                             " where dimension can be 1.")

        if z is not None and np.ndim(z) != 2:
            raise ValueError("z must be array of shape (samples, dimension)"
                             " where dimension can be 1.")

        if z is None:
            # Get the array to test on
            array = np.vstack((x.T, y.T))

            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])])

        else:
            # Get the array to test on
            array = np.vstack((x.T, y.T, z.T))

            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])] +
                           [2 for i in range(z.shape[1])])

        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")
        # Get the dependence measure
        val = self.get_dependence_measure(array, xyz)
        # Get the p-value
        pval = self.get_significance(val, array, xyz, T, dim)
        # Return the value and the pvalue
        return val, pval

    def _get_dependence_measure_recycle(self, X, Y, Z, xyz, array):
        """Get the dependence_measure, optionally recycling residuals

        If self.recycle_residuals is True, also _get_single_residuals must be
        available.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        array : array
            Data array of shape (dim, T)

        Return
        ------
        val : float
            Test statistic
        """
        # Check if we are recycling residuals
        if self.recycle_residuals:
            # Get or calculate the cached residuals
            x_resid = self._get_cached_residuals(X, Z, array, 0)
            y_resid = self._get_cached_residuals(Y, Z, array, 1)
            # Make a new residual array
            array_resid = np.array([x_resid, y_resid])
            xyz_resid = np.array([0, 1])
            # Return the dependence measure
            return self.get_dependence_measure(array_resid, xyz_resid)
        # If not, return the dependence measure on the array and xyz
        return self.get_dependence_measure(array, xyz)

    def _get_cached_residuals(self, x_nodes, z_nodes, array, target_var):
        """
        Retrieve or calculate the cached residuals for the given node sets.

        Parameters
        ----------
            x_nodes : list of tuples
                List of nodes, X or Y normally. Used to key the residual cache
                during lookup

            z_nodes : list of tuples
                List of nodes, Z normally

            target_var : int
                Key to differentiate X from Y.
                x_nodes == X => 0, x_nodes == Y => 1

            array : array
                Data array of shape (dim, T)

        Returns
        -------
            x_resid : array
                Residuals calculated by _get_single_residual
        """
        # Check if we have calculated these residuals
        if self._keyfy(x_nodes, z_nodes) in list(self.residuals):
            x_resid = self.residuals[self._keyfy(x_nodes, z_nodes)]
        # If not, calculate the residuals
        else:
            x_resid = self._get_single_residuals(array, target_var=target_var)
            if z_nodes:
                self.residuals[self._keyfy(x_nodes, z_nodes)] = x_resid
        # Return these residuals
        return x_resid
    
    def get_significance(self, val, array, xyz, T, dim, sig_override=None):
        """
        Returns the p-value from whichever significance function is specified
        for this test.  If an override is used, then it will call a different
        function then specified by self.significance

        Parameters
        ----------
        val : float
            Test statistic value.

        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        sig_override : string
            Must be in 'analytic', 'shuffle_test', 'fixed_thres'

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Defaults to the self.significance member value
        use_sig = self.significance
        if sig_override is not None:
            use_sig = sig_override
        # Check if we are using the analytic significance
        if use_sig == 'analytic':
            pval = self.get_analytic_significance(value=val, T=T, dim=dim)
        # Check if we are using the shuffle significance
        elif use_sig == 'shuffle_test':
            pval = self.get_shuffle_significance(array=array,
                                                 xyz=xyz,
                                                 value=val)
        # Check if we are using the fixed_thres significance
        elif use_sig == 'fixed_thres':
            pval = self.get_fixed_thres_significance(
                    value=val,
                    fixed_thres=self.fixed_thres)
        else:
            raise ValueError("%s not known." % self.significance)
        # Return the calculated value
        return pval

    def get_measure(self, X, Y, Z=None, tau_max=0):
        """Estimate dependence measure.

        Calls the dependence measure function. The child classes must specify
        a function get_dependence_measure.

        Parameters
        ----------
        X, Y [, Z] : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns
        -------
        val : float
            The test statistic value.

        """
        # Make the array
        array, xyz, (X, Y, Z) = self._get_array(X, Y, Z, tau_max)
        D, T = array.shape
        # Check it is valid
        if np.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")
        # Return the dependence measure
        return self._get_dependence_measure_recycle(X, Y, Z, xyz, array)

    def get_confidence(self, X, Y, Z=None, tau_max=0):
        """Perform confidence interval estimation.

        Calls the dependence measure and confidence test functions. The child
        classes can specify a function get_dependence_measure and
        get_analytic_confidence or get_bootstrap_confidence. If confidence is
        False, (numpy.nan, numpy.nan) is returned.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns
        -------
        (conf_lower, conf_upper) : Tuple of floats
            Upper and lower confidence bound of confidence interval.
        """
        # Check if a confidence type has been defined
        if self.confidence:
            # Ensure the confidence level given makes sense
            if self.conf_lev < .5 or self.conf_lev >= 1.:
                raise ValueError("conf_lev = %.2f, " % self.conf_lev +
                                 "but must be between 0.5 and 1")
            half_conf = self.conf_samples * (1. - self.conf_lev)/2.
            if self.confidence == 'bootstrap' and  half_conf < 1.:
                raise ValueError("conf_samples*(1.-conf_lev)/2 is %.2f"
                                 % half_conf + ", must be >> 1")

        if self.confidence:
            # Make and check the array
            array, xyz, _ = self._get_array(X, Y, Z, tau_max, verbosity=0)
            dim, T = array.shape
            if np.isnan(array).sum() != 0:
                raise ValueError("nans in the array!")

            # Check if we are using analytic confidence or bootstrapping it
            if self.confidence == 'analytic':
                val = self.get_dependence_measure(array, xyz)
                (conf_lower, conf_upper) = \
                        self.get_analytic_confidence(df=T-dim,
                                                     value=val,
                                                     conf_lev=self.conf_lev)
            elif self.confidence == 'bootstrap':
                # Overwrite analytic values
                (conf_lower, conf_upper) = \
                        self.get_bootstrap_confidence(
                            array, xyz,
                            conf_samples=self.conf_samples,
                            conf_blocklength=self.conf_blocklength,
                            conf_lev=self.conf_lev, verbosity=self.verbosity)
            else:
                raise ValueError("%s confidence estimation not implemented"
                                 % self.confidence)
        else:
            return None

        # Cache the confidence interval
        self.conf = (conf_lower, conf_upper)
        # Return the confidence interval
        return (conf_lower, conf_upper)

    def _print_cond_ind_results(self, val, pval=None, cached=None, conf=None):
        """Print results from conditional independence test.

        Parameters
        ----------
        val : float
            Test stastistic value.

        pval : float, optional (default: None)
            p-value

        conf : tuple of floats, optional (default: None)
            Confidence bounds.
        """
        printstr = "        val = % .3f" % (val)      
        if pval is not None:
            printstr += " | pval = %.5f" % (pval)
        if conf is not None:
            printstr += " | conf bounds = (%.3f, %.3f)" % (
                conf[0], conf[1])
        if cached is not None:
            printstr += " %s" % ({0:"", 1:"[cached]"}[cached])

        print(printstr)

    def get_bootstrap_confidence(self, array, xyz, dependence_measure=None,
                                 conf_samples=100, conf_blocklength=None,
                                 conf_lev=.95, verbosity=0):
        """Perform bootstrap confidence interval estimation.

        With conf_blocklength > 1 or None a block-bootstrap is performed.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        dependence_measure : function (default = self.get_dependence_measure)
            Dependence measure function must be of form
            dependence_measure(array, xyz) and return a numeric value

        conf_lev : float, optional (default: 0.9)
            Two-sided confidence interval.

        conf_samples : int, optional (default: 100)
            Number of samples for bootstrap.

        conf_blocklength : int, optional (default: None)
            Block length for block-bootstrap. If None, the block length is
            determined from the decay of the autocovariance as explained in
            [1]_.

        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        (conf_lower, conf_upper) : Tuple of floats
            Upper and lower confidence bound of confidence interval.
        """
        # Check if a dependence measure if provided or if to use default
        if not dependence_measure:
            dependence_measure = self.get_dependence_measure

        # confidence interval is two-sided
        c_int = 1. - (1. - conf_lev)/2.
        dim, T = array.shape

        # If not block length is given, determine the optimal block length.
        # This has a maximum of 10% of the time sample length
        if conf_blocklength is None:
            conf_blocklength = \
                    self._get_block_length(array, xyz, mode='confidence')
        # Determine the number of blocks total, rounding up for non-integer
        # amounts
        n_blks = int(math.ceil(float(T)/conf_blocklength))

        # Print some information
        if verbosity > 2:
            print("            block_bootstrap confidence intervals"
                  " with block-length = %d ..." % conf_blocklength)

        # Generate the block bootstrapped distribution
        bootdist = np.zeros(conf_samples)
        for smpl in range(conf_samples):
            # Get the starting indices for the blocks
            blk_strt = self.random_state.integers(0, T - conf_blocklength + 1, n_blks)
            # Get the empty array of block resampled values
            array_bootstrap = \
                    np.zeros((dim, n_blks*conf_blocklength), dtype=array.dtype)
            # Fill the array of block resamples
            for i in range(conf_blocklength):
                array_bootstrap[:, i::conf_blocklength] = array[:, blk_strt + i]
            # Cut to proper length
            array_bootstrap = array_bootstrap[:, :T]

            bootdist[smpl] = dependence_measure(array_bootstrap, xyz)

        # Sort and get quantile
        bootdist.sort()
        conf_lower = bootdist[int((1. - c_int) * conf_samples)]
        conf_upper = bootdist[int(c_int * conf_samples)]
        # Return the confidance limits as a tuple
        return (conf_lower, conf_upper)

    def _get_acf(self, series, max_lag=None):
        """Returns autocorrelation function.

        Parameters
        ----------
        series : 1D-array
            data series to compute autocorrelation from

        max_lag : int, optional (default: None)
            maximum lag for autocorrelation function. If None is passed, 10% of
            the data series length are used.

        Returns
        -------
        autocorr : array of shape (max_lag + 1,)
            Autocorrelation function.
        """
        # Set the default max lag
        if max_lag is None:
            max_lag = int(max(5, 0.1*len(series)))
        # Initialize the result
        autocorr = np.ones(max_lag + 1)
        # Iterate over possible lags
        for lag in range(1, max_lag + 1):
            # Set the values
            y1_vals = series[lag:]
            y2_vals = series[:len(series) - lag]
            # Calculate the autocorrelation
            autocorr[lag] = np.corrcoef(y1_vals, y2_vals, ddof=0)[0, 1]
        return autocorr

    def _get_block_length(self, array, xyz, mode):
        """Returns optimal block length for significance and confidence tests.

        Determine block length using approach in Mader (2013) [Eq. (6)] which
        improves the method of Pfeifer (2005) with non-overlapping blocks In
        case of multidimensional X, the max is used. Further details in [1]_.
        Two modes are available. For mode='significance', only the indices
        corresponding to X are shuffled in array. For mode='confidence' all
        variables are jointly shuffled. If the autocorrelation curve fit fails,
        a block length of 5% of T is used. The block length is limited to a
        maximum of 10% of T.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        mode : str
            Which mode to use.

        Returns
        -------
        block_len : int
            Optimal block length.
        """
        # Inject a dependency on siganal, optimize
        from scipy import signal, optimize
        # Get the shape of the array
        dim, T = array.shape
        # Initiailize the indices
        indices = range(dim)
        if mode == 'significance':
            indices = np.where(xyz == 0)[0]

        # Maximum lag for autocov estimation
        max_lag = int(0.1*T)
        # Define the function to optimize against
        def func(x_vals, a_const, decay):
            return a_const * decay**x_vals

        # Calculate the block length
        block_len = 1
        for i in indices:
            # Get decay rate of envelope of autocorrelation functions
            # via hilbert trafo
            autocov = self._get_acf(series=array[i], max_lag=max_lag)
            autocov[0] = 1.
            hilbert = np.abs(signal.hilbert(autocov))
            # Try to fit the curve
            try:
                popt, _ = optimize.curve_fit(
                    f=func,
                    xdata=np.arange(0, max_lag+1),
                    ydata=hilbert,
                )
                phi = popt[1]
                # Formula of Pfeifer (2005) assuming non-overlapping blocks
                l_opt = (4. * T * (phi / (1. - phi) + phi**2 / (1. - phi)**2)**2
                         / (1. + 2. * phi / (1. - phi))**2)**(1. / 3.)
                block_len = max(block_len, int(l_opt))
            except RuntimeError:
                print("Error - curve_fit failed in block_shuffle, using"
                      " block_len = %d" % (int(.05 * T)))
                block_len = max(int(.05 * T), 2)
        # Limit block length to a maximum of 10% of T
        block_len = min(block_len, int(0.1 * T))
        return block_len

    def _get_shuffle_dist(self, array, xyz, dependence_measure,
                          sig_samples, sig_blocklength=None,
                          verbosity=0):
        """Returns shuffle distribution of test statistic.

        The rows in array corresponding to the X-variable are shuffled using
        a block-shuffle approach.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

       dependence_measure : object
           Dependence measure function must be of form
           dependence_measure(array, xyz) and return a numeric value

        sig_samples : int, optional (default: 100)
            Number of samples for shuffle significance test.

        sig_blocklength : int, optional (default: None)
            Block length for block-shuffle significance test. If None, the
            block length is determined from the decay of the autocovariance as
            explained in [1]_.

        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        null_dist : array of shape (sig_samples,)
            Contains the sorted test statistic values estimated from the
            shuffled arrays.
        """
        dim, T = array.shape

        x_indices = np.where(xyz == 0)[0]
        dim_x = len(x_indices)

        if sig_blocklength is None:
            sig_blocklength = self._get_block_length(array, xyz,
                                                     mode='significance')

        n_blks = int(math.floor(float(T)/sig_blocklength))
        if verbosity > 2:
            print("            Significance test with block-length = %d "
                  "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blks*sig_blocklength:]

        null_dist = np.zeros(sig_samples)

        """
         JC NOTE: Inside the loop, the function first performs permutation-shuffle, then CMI-computation operations.
         Here we use numba to speed up the permutation-shuffle operation. As a result, the speed increases by 20%.
         For the optimization of CMI-computation, please refer to the function get_dependence_measure(self, array, xyz) in cmisymb.py.
        """
        # By doing so, the loop uses a shuffled way to estimate the p-value.
        permutation_shuffle_time_list = []
        histgen_cmicomp_time_list = []
        start = time.time()
        for sam in range(sig_samples):
            shuffle_start = time.time()
            x_shuffled = None
            # JC NOTE: If decide to use the optimization: Set numba_optimize
            numba_optimize = True
            if numba_optimize:
                x_shuffled = _permutation_and_shuffle(block_starts, n_blks, array[x_indices[0]])
                assert(x_shuffled is not None)
                array_shuffled[x_indices[0]] = x_shuffled
            else:
                blk_starts = self.random_state.permutation(block_starts)[:n_blks]
                x_shuffled = np.zeros((dim_x, n_blks*sig_blocklength),
                                      dtype=array.dtype)
                for i, index in enumerate(x_indices): # Randomly rearrange the blocks (consisting of sig_blocklength of records about x) in original array
                    for blk in range(sig_blocklength):
                        x_shuffled[i, blk::sig_blocklength] = \
                                array[index, blk_starts + blk]

                # Insert tail randomly somewhere
                if tail.shape[1] > 0:
                    insert_tail_at = self.random_state.choice(block_starts)
                    x_shuffled = np.insert(x_shuffled, insert_tail_at,
                                           tail.T, axis=1)

                for i, index in enumerate(x_indices):
                    array_shuffled[index] = x_shuffled[i]
            shuffle_end = time.time()
            #print("Permutation + Shuffle time cost: {} seconds".format(shuffle_end - shuffle_start))
            permutation_shuffle_time_list.append(shuffle_end - shuffle_start)
            cmi_comp_start = time.time()
            null_dist[sam] = dependence_measure(array=array_shuffled, xyz=xyz)
            cmi_comp_end = time.time()
            # print("Hist-generation + CMI-computation consumes time : {} seconds".format(cmi_comp_end - cmi_comp_start))
            histgen_cmicomp_time_list.append(cmi_comp_end - cmi_comp_start)
        end = time.time()
        if verbosity > 2:
            print("Average time cost of permutation + shuffle operations: {} seconds".format(1.0 * sum(permutation_shuffle_time_list) / sig_samples))
            print("Average time cost of hist-generation + cmi-computation operations: {} seconds".format(1.0 * sum(histgen_cmicomp_time_list) / sig_samples))
            print("The permutation-shuffle + CMI-computation for single test ends. Consumed time: {} seconds".format( (end - start) * 1.0 ))
        return null_dist

    def get_fixed_thres_significance(self, value, fixed_thres):
        """Returns signficance for thresholding test.

        Returns 0 if numpy.abs(value) is smaller than fixed_thres and 1 else.

        Parameters
        ----------
        value : number
            Value of test statistic for unshuffled estimate.

        fixed_thres : number
            Fixed threshold, is made positive.

        Returns
        -------
        pval : bool
            Returns 0 if numpy.abs(value) is smaller than fixed_thres and 1
            else.

        """
        if np.abs(value) < np.abs(fixed_thres):
            pval = 1.
        else:
            pval = 0.

        return pval

    def _trafo2uniform(self, x):
        """Transforms input array to uniform marginals.

        Assumes x.shape = (dim, T)

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        u : array-like
            array with uniform marginals.
        """

        def trafo(xi):
            xisorted = np.sort(xi)
            yi = np.linspace(1. / len(xi), 1, len(xi))
            return np.interp(xi, xisorted, yi)

        if np.ndim(x) == 1:
            u = trafo(x)
        else:
            u = np.empty(x.shape)
            for i in range(x.shape[0]):
                u[i] = trafo(x[i])
        return u
