"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import warnings
import math
import abc
from scipy import special, stats, spatial
import numpy as np
import six
import sys

try:
    from sklearn import gaussian_process
except:
    print("Could not import sklearn for Gaussian process tests")

try:
    from tigramite import tigramite_cython_code
except:
    print("Could not import packages for CMIknn and GPDC estimation")

try:
    import rpy2
    import rpy2.robjects
    rpy2.robjects.r['options'](warn=-1)
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
except:
    print("Could not import rpy package")

try:
    importr('RCIT')
except:
    print("Could not import r-package RCIT")

@six.add_metaclass(abc.ABCMeta)
class CondIndTest():
    """Base class of conditional independence tests.

    Provides useful general functions for different independence tests such as
    shuffle significance testing and bootstrap confidence estimation. Also
    handles masked samples. Other test classes can inherit from this class.

    Parameters
    ----------
    mask_type : str, optional (default = None)
        Must be in {'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence measure
        I(X; Y | Z) the samples should be masked. If None, 'y' is used, which
        excludes all time slices containing masked samples in Y. Explained in
        [1]_.

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
        self.significance = significance
        self.sig_samples = sig_samples
        self.sig_blocklength = sig_blocklength
        self.fixed_thres = fixed_thres
        self.verbosity = verbosity
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
            Must be in {'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, 'y' is
            used, which excludes all time slices containing masked samples in Y.
            Explained in [1]_.
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
            info_str += "mask_type = %s" % self.mask_type
        # Check if we are recycling residuals or not
        if self.recycle_residuals:
            info_str += "recycle_residuals = %s" % self.recycle_residuals
        # Print the information string
        print(info_str)

    def _check_mask_type(self):
        """
        mask_type : str, optional (default = None)
            Must be in {'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, 'y' is
            used, which excludes all time slices containing masked samples in Y.
            Explained in [1]_.
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

    def _get_array(self, X, Y, Z, tau_max=0, cut_off='2xtau_max', verbosity=None):
        """Convencience wrapper around _construct_array."""
        # Set the verbosity to the default value
        if verbosity is None:
            verbosity=self.verbosity

        if self.measure in ['par_corr']:
            if len(X) > 1 or len(Y) > 1:
                raise ValueError("X and Y for %s must be univariate." %
                                        self.measure)
        # Call the wrapped function
        return self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                              tau_max=tau_max,
                                              mask_type=self.mask_type,
                                              return_cleaned_xyz=True,
                                              do_checks=False,
                                              cut_off=cut_off,
                                              verbosity=verbosity)

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
        if np.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")
        # Get the dependence measure, reycling residuals if need be
        val = self._get_dependence_measure_recycle(X, Y, Z, xyz, array)
        # Get the p-value
        pval = self.get_significance(val, array, xyz, T, dim)
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
        # Defaults to the self.signficance memeber value
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
        elif not self.confidence:
            return None
        else:
            raise ValueError("%s confidence estimation not implemented"
                             % self.confidence)
        # Cache the confidence interval
        self.conf = (conf_lower, conf_upper)
        # Return the confidence interval
        return (conf_lower, conf_upper)

    def _print_cond_ind_results(self, val, pval=None, conf=None):
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

        if pval is not None:
            printstr = "        pval = %.5f | val = %.3f" % (
                pval, val)
            if conf is not None:
                printstr += " | conf bounds = (%.3f, %.3f)" % (
                    conf[0], conf[1])
        else:
            printstr = "        val = %.3f" % val
            if conf is not None:
                printstr += " | conf bounds = (%.3f, %.3f)" % (
                    conf[0], conf[1])
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
            Levelof verbosity.

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
            # Get the starting indecies for the blocks
            blk_strt = np.random.randint(0, T - conf_blocklength + 1, n_blks)
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
        # print 'n_blks ', n_blks
        if verbosity > 2:
            print("            Significance test with block-length = %d "
                  "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blks*sig_blocklength:]

        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):

            blk_starts = np.random.permutation(block_starts)[:n_blks]

            x_shuffled = np.zeros((dim_x, n_blks*sig_blocklength),
                                  dtype=array.dtype)

            for i, index in enumerate(x_indices):
                for blk in range(sig_blocklength):
                    x_shuffled[i, blk::sig_blocklength] = \
                            array[index, blk_starts + blk]

            # Insert tail randomly somewhere
            if tail.shape[1] > 0:
                insert_tail_at = np.random.choice(block_starts)
                x_shuffled = np.insert(x_shuffled, insert_tail_at,
                                       tail.T, axis=1)

            for i, index in enumerate(x_indices):
                array_shuffled[index] = x_shuffled[i]

            null_dist[sam] = dependence_measure(array=array_shuffled,
                                                xyz=xyz)

        null_dist.sort()

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
            array /= array.std(axis=1).reshape(dim, 1)
            if np.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")

        y = array[target_var, :]

        if dim_z > 0:
            z = np.fastCopyAndTranspose(array[2:, :])
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        return resid

    def get_dependence_measure(self, array, xyz):
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

        pval = (null_dist >= np.abs(value)).mean()

        # Adjust p-value for two-sided measures
        if pval < 1.:
            pval *= 2.

        if return_null_dist:
            return pval, null_dist
        return pval

    def get_analytic_significance(self, value, T, dim):
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


    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """Returns Akaike's Information criterion modulo constants.

        Fits a linear model of the parents to variable j and returns the score.
        I used to determine optimal hyperparameters in PCMCI, in particular
        the pc_alpha value.

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
        array, xyz = self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                    tau_max=tau_max,
                                                    mask_type=self.mask_type,
                                                    return_cleaned_xyz=False,
                                                    do_checks=False,
                                                    verbosity=self.verbosity)

        dim, T = array.shape

        y = self._get_single_residuals(array, target_var=1, return_means=False)
        # Get RSS
        rss = (y**2).sum()
        # Number of parameters
        p = dim - 1
        # Get AIC
        score = T * np.log(rss) + 2. * p
        return score

class GaussProcReg():
    r"""Gaussian processes abstract base class.

    GP is estimated with scikit-learn and allows to flexibly specify kernels and
    hyperparameters or let them be optimized automatically. The kernel specifies
    the covariance function of the GP. Parameters can be passed on to
    ``GaussianProcessRegressor`` using the gp_params dictionary. If None is
    passed, the kernel '1.0 * RBF(1.0) + WhiteKernel()' is used with alpha=0 as
    default. Note that the kernel's hyperparameters are optimized during
    fitting.

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

    gp_version : {'new', 'old'}, optional (default: 'new')
        The older GP version from scikit-learn 0.17 was used for the numerical
        simulations in [1]_. The newer version from scikit-learn 0.19 is faster
        and allows more flexibility regarding kernels etc.

    gp_params : dictionary, optional (default: None)
        Dictionary with parameters for ``GaussianProcessRegressor``.

    null_dist_filename : str, otional (default: None)
        Path to file containing null distribution.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """
    def __init__(self,
                 null_samples,
                 cond_ind_test,
                 gp_version='new',
                 gp_params=None,
                 null_dist_filename=None,
                 verbosity=0):
        # Set the dependence measure function
        self.cond_ind_test = cond_ind_test
        # Set member variables
        self.gp_version = gp_version
        self.gp_params = gp_params
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

        xyz = np.array([0,1])

        null_dist = np.zeros(self.null_samples)
        for i in range(self.null_samples):
            array = np.random.rand(2, df)
            null_dist[i] = self.cond_ind_test.get_dependence_measure(array, xyz)

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
            null_dists[iT] = self._generate_nulldist(T, add_to_null_dists=False)
            self.null_dists[T] = null_dists[iT]

        np.savez("%s" % null_dist_filename,
                 exact_dist=null_dists,
                 T=np.array(sample_sizes))

    def _get_single_residuals(self, array, target_var,
                              return_means=False,
                              standardize=True,
                              return_likelihood=False):
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
        dim, T = array.shape

        if self.gp_params is None:
            self.gp_params = {}

        if dim <= 2:
            if return_likelihood:
                return array[target_var, :], -np.inf
            return array[target_var, :]

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            array /= array.std(axis=1).reshape(dim, 1)
            if np.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")

        target_series = array[target_var, :]
        z = np.fastCopyAndTranspose(array[2:])
        if np.ndim(z) == 1:
            z = z.reshape(-1, 1)

        if self.gp_version == 'old':
            # Old GP failed for ties in the data
            def remove_ties(series, verbosity=0):
                # Test whether ties exist and add noise to destroy ties...
                cnt = 0
                while len(np.unique(series)) < np.size(series):
                    series += 1E-6 * np.random.rand(*series.shape)
                    cnt += 1
                    if cnt > 100:
                        break
                return series

            z = remove_ties(z)
            target_series = remove_ties(target_series)

            gp = gaussian_process.GaussianProcess(
                nugget=1E-1,
                thetaL=1E-16,
                thetaU=np.inf,
                corr='squared_exponential',
                optimizer='fmin_cobyla',
                regr='constant',
                normalize=False,
                storage_mode='light')

        elif self.gp_version == 'new':
            # Overwrite default kernel and alpha values
            params = self.gp_params.copy()
            if 'kernel' not in list(self.gp_params):
                kernel = gaussian_process.kernels.RBF() +\
                         gaussian_process.kernels.WhiteKernel()
            else:
                kernel = self.gp_params['kernel']
                del params['kernel']

            if 'alpha' not in list(self.gp_params):
                alpha = 0.
            else:
                alpha = self.gp_params['alpha']
                del params['alpha']

            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                           alpha=alpha,
                                                           **params)

        gp.fit(z, target_series.reshape(-1, 1))

        if self.verbosity > 3 and self.gp_version == 'new':
            print(kernel, alpha, gp.kernel_, gp.alpha)

        if self.verbosity > 3 and self.gp_version == 'old':
            print(gp.get_params)

        if return_likelihood:
            likelihood = gp.log_marginal_likelihood()

        mean = gp.predict(z).squeeze()

        resid = target_series - mean

        if return_means and not return_likelihood:
            return (resid, mean)
        elif return_likelihood and not return_means:
            return (resid, likelihood)
        elif return_means and return_likelihood:
            return resid, mean, likelihood
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
                    do_checks=False,
                    verbosity=self.verbosity)

        dim, T = array.shape

        _, logli = self._get_single_residuals(array,
                                              target_var=1,
                                              return_likelihood=True)

        score = -logli
        return score

class GPDC(CondIndTest):
    r"""GPDC conditional independence test based on Gaussian processes and
        distance correlation.

    GPDC is based on a Gaussian process (GP) regression and a distance
    correlation test on the residuals [2]_. GP is estimated with scikit-learn
    and allows to flexibly specify kernels and hyperparameters or let them be
    optimized automatically. The distance correlation test is implemented with
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
    optimzed using ``sklearn``. Then the residuals  are transformed to uniform
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

    gp_version : {'new', 'old'}, optional (default: 'new')
        The older GP version from scikit-learn 0.17 was used for the numerical
        simulations in [1]_. The newer version from scikit-learn 0.19 is faster
        and allows more flexibility regarding kernels etc.

    gp_params : dictionary, optional (default: None)
        Dictionary with parameters for ``GaussianProcessRegressor``.

    **kwargs :
        Arguments passed on to parent class GaussProcReg.

    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 null_dist_filename=None,
                 gp_version='new',
                 gp_params=None,
                 **kwargs):
        self._measure = 'gp_dc'
        self.two_sided = False
        self.residual_based = True
        # Call the parent constructor
        CondIndTest.__init__(self, **kwargs)
        # Build the regressor
        self.gauss_pr = GaussProcReg(self.sig_samples,
                                     self,
                                     gp_version=gp_version,
                                     gp_params=gp_params,
                                     null_dist_filename=null_dist_filename,
                                     verbosity=self.verbosity)

        if self.verbosity > 0:
            print("null_dist_filename = %s" % self.gauss_pr.null_dist_filename)
            print("gp_version = %s" % self.gauss_pr.gp_version)
            if self.gauss_pr.gp_params is not None:
                for key in  list(self.gauss_pr.gp_params):
                    print("%s = %s" % (key, self.gauss_pr.gp_params[key]))
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
                              return_likelihood=False):
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
            return_likelihood)

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
            P-value.
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

class CMIknn(CondIndTest):
    r"""Conditional mutual information test based on nearest-neighbor estimator.

    Conditional mutual information is the most general dependency measure coming
    from an information-theoretic framework. It makes no assumptions about the
    parametric form of the dependencies by directly estimating the underlying
    joint density. The test here is based on the estimator in  S. Frenzel and B.
    Pompe, Phys. Rev. Lett. 99, 204101 (2007), combined with a shuffle test to
    generate  the distribution under the null hypothesis of independence first
    used in [3]_. The knn-estimator is suitable only for variables taking a
    continuous range of values. For discrete variables use the CMIsymb class.

    Notes
    -----
    CMI is given by

    .. math:: I(X;Y|Z) &= \int p(z)  \iint  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Its knn-estimator is given by

    .. math:: \widehat{I}(X;Y|Z)  &=   \psi (k) + \frac{1}{T} \sum_{t=1}^T
            \left[ \psi(k_{Z,t}) - \psi(k_{XZ,t}) - \psi(k_{YZ,t}) \right]

    where :math:`\psi` is the Digamma function.  This estimator has as a
    parameter the number of nearest-neighbors :math:`k` which determines the
    size of hyper-cubes around each (high-dimensional) sample point. Then
    :math:`k_{Z,},k_{XZ},k_{YZ}` are the numbers of neighbors in the respective
    subspaces.

    :math:`k` can be viewed as a density smoothing parameter (although it is
    data-adaptive unlike fixed-bandwidth estimators). For large :math:`k`, the
    underlying dependencies are more smoothed and CMI has a larger bias,
    but lower variance, which is more important for significance testing. Note
    that the estimated CMI values can be slightly negative while CMI is a non-
    negative quantity.

    This method requires the scipy.spatial.cKDTree package and the tigramite
    cython module.

    References
    ----------
    .. [3] J. Runge (2018): Conditional Independence Testing Based on a
           Nearest-Neighbor Estimator of Conditional Mutual Information.
           In Proceedings of the 21st International Conference on Artificial
           Intelligence and Statistics.
           http://proceedings.mlr.press/v84/runge18a.html

    Parameters
    ----------
    knn : int or float, optional (default: 0.2)
        Number of nearest-neighbors which determines the size of hyper-cubes
        around each (high-dimensional) sample point. If smaller than 1, this is
        computed as a fraction of T, hence knn=knn*T. For knn larger or equal to
        1, this is the absolute number.

    shuffle_neighbors : int, optional (default: 10)
        Number of nearest-neighbors within Z for the shuffle surrogates which
        determines the size of hyper-cubes around each (high-dimensional) sample
        point.

    transform : {'ranks', 'standardize',  'uniform', False}, optional
        (default: 'ranks')
        Whether to transform the array beforehand by standardizing
        or transforming to uniform marginals.

    n_jobs : int (optional, default = -1)
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.

    significance : str, optional (default: 'shuffle_test')
        Type of significance test to use. For CMIknn only 'fixed_thres' and
        'shuffle_test' are available.

    **kwargs :
        Arguments passed on to parent class CondIndTest.
    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 knn=0.2,
                 shuffle_neighbors=5,
                 significance='shuffle_test',
                 transform='ranks',
                 n_jobs=-1,
                 **kwargs):
        # Set the member variables
        self.knn = knn
        self.shuffle_neighbors = shuffle_neighbors
        self.transform = transform
        self._measure = 'cmi_knn'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.n_jobs = n_jobs
        # Call the parent constructor
        CondIndTest.__init__(self, significance=significance, **kwargs)
        # Print some information about construction
        if self.verbosity > 0:
            if self.knn < 1:
                print("knn/T = %s" % self.knn)
            else:
                print("knn = %s" % self.knn)
            print("shuffle_neighbors = %d\n" % self.shuffle_neighbors)

    def _get_nearest_neighbors(self, array, xyz, knn):
        """Returns nearest neighbors according to Frenzel and Pompe (2007).

        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        knn : int or float
            Number of nearest-neighbors which determines the size of hyper-cubes
            around each (high-dimensional) sample point. If smaller than 1, this
            is computed as a fraction of T, hence knn=knn*T. For knn larger or
            equal to 1, this is the absolute number.

        Returns
        -------
        k_xz, k_yz, k_z : tuple of arrays of shape (T,)
            Nearest neighbors in subspaces.
        """

        dim, T = array.shape
        array = array.astype('float')

        # Add noise to destroy ties...
        array += (1E-6 * array.std(axis=1).reshape(dim, 1)
                  * np.random.rand(array.shape[0], array.shape[1]))

        if self.transform == 'standardize':
            # Standardize
            array = array.astype('float')
            array -= array.mean(axis=1).reshape(dim, 1)
            array /= array.std(axis=1).reshape(dim, 1)
            # FIXME: If the time series is constant, return nan rather than
            # raising Exception
            if np.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")
        elif self.transform == 'uniform':
            array = self._trafo2uniform(array)
        elif self.transform == 'ranks':
            array = array.argsort(axis=1).argsort(axis=1).astype('float')


        # Use cKDTree to get distances eps to the k-th nearest neighbors for
        # every sample in joint space XYZ with maximum norm
        tree_xyz = spatial.cKDTree(array.T)
        epsarray = tree_xyz.query(array.T, k=knn+1, p=np.inf,
                                  eps=0., n_jobs=self.n_jobs)[0][:, knn].astype('float')

        # Prepare for fast cython access
        dim_x = int(np.where(xyz == 0)[0][-1] + 1)
        dim_y = int(np.where(xyz == 1)[0][-1] + 1 - dim_x)

        k_xz, k_yz, k_z = \
                tigramite_cython_code._get_neighbors_within_eps_cython(array,
                                                                       T,
                                                                       dim_x,
                                                                       dim_y,
                                                                       epsarray,
                                                                       knn,
                                                                       dim)
        return k_xz, k_yz, k_z

    def get_dependence_measure(self, array, xyz):
        """Returns CMI estimate as described in Frenzel and Pompe PRL (2007).

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """

        dim, T = array.shape

        if self.knn < 1:
            knn_here = max(1, int(self.knn*T))
        else:
            knn_here = max(1, int(self.knn))

        k_xz, k_yz, k_z = self._get_nearest_neighbors(array=array,
                                                      xyz=xyz,
                                                      knn=knn_here)

        val = special.digamma(knn_here) - (special.digamma(k_xz) +
                                           special.digamma(k_yz) -
                                           special.digamma(k_z)).mean()

        return val


    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest niehgbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

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

        # Skip shuffle test if value is above threshold
        # if value > self.minimum threshold:
        #     if return_null_dist:
        #         return 0., None
        #     else:
        #         return 0.

        # max_neighbors = max(1, int(max_neighbor_ratio*T))
        x_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            if self.verbosity > 2:
                print("            nearest-neighbor shuffle significance "
                      "test with n = %d and %d surrogates" % (
                      self.shuffle_neighbors, self.sig_samples))

            # Get nearest neighbors around each sample point in Z
            z_array = np.fastCopyAndTranspose(array[z_indices, :])
            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array,
                                       k=self.shuffle_neighbors,
                                       p=np.inf,
                                       eps=0.)[1].astype('int32')

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):

                # Generate random order in which to go through indices loop in
                # next step
                order = np.random.permutation(T).astype('int32')
                # print(order[:5])
                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = \
                    tigramite_cython_code._get_restricted_permutation_cython(
                        T=T,
                        shuffle_neighbors=self.shuffle_neighbors,
                        neighbors=neighbors,
                        order=order)

                array_shuffled = np.copy(array)
                for i in x_indices:
                    array_shuffled[i] = array[i, restricted_permutation]

                null_dist[sam] = self.get_dependence_measure(array_shuffled,
                                                             xyz)

        else:
            null_dist = \
                    self._get_shuffle_dist(array, xyz,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        # Sort
        null_dist.sort()
        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval


class CMIsymb(CondIndTest):
    r"""Conditional mutual information test based on discrete estimator.

    Conditional mutual information is the most general dependency measure
    coming from an information-theoretic framework. It makes no assumptions
    about the parametric form of the dependencies by directly estimating the
    underlying joint density. The test here is based on directly estimating
    the joint distribution assuming symbolic input, combined with a
    shuffle test to generate  the distribution under the null hypothesis of
    independence. The knn-estimator is suitable only for discrete variables.
    For continuous variables, either pre-process the data using the functions
    in data_processing or, better, use the CMIknn class.

    Notes
    -----
    CMI and its estimator are given by

    .. math:: I(X;Y|Z) &= \sum p(z)  \sum \sum  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Parameters
    ----------
    n_symbs : int, optional (default: None)
        Number of symbols in input data. Should be at least as large as the
        maximum array entry + 1. If None, n_symbs is based on the
        maximum value in the array (array.max() + 1).

    significance : str, optional (default: 'shuffle_test')
        Type of significance test to use. For CMIsymb only 'fixed_thres' and
        'shuffle_test' are available.

    sig_blocklength : int, optional (default: 1)
        Block length for block-shuffle significance test.

    conf_blocklength : int, optional (default: 1)
        Block length for block-bootstrap.

    **kwargs :
        Arguments passed on to parent class CondIndTest.
    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 n_symbs=None,
                 significance='shuffle_test',
                 sig_blocklength=1,
                 conf_blocklength=1,
                 **kwargs):
        # Setup the member variables
        self._measure = 'cmi_symb'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.n_symbs = n_symbs
        # Call the parent constructor
        CondIndTest.__init__(self,
                             significance=significance,
                             sig_blocklength=sig_blocklength,
                             conf_blocklength=conf_blocklength,
                             **kwargs)

        if self.verbosity > 0:
            print("n_symbs = %s" % self.n_symbs)
            print("")

        if self.conf_blocklength is None or self.sig_blocklength is None:
            warnings.warn("Automatic block-length estimations from decay of "
                          "autocorrelation may not be sensical for discrete "
                          "data")

    def _bincount_hist(self, symb_array, weights=None):
        """Computes histogram from symbolic array.

        The maximum of the symbolic array determines the alphabet / number
        of bins.

        Parameters
        ----------
        symb_array : integer array
            Data array of shape (dim, T).

        weights : float array, optional (default: None)
            Optional weights array of shape (dim, T).

        Returns
        -------
        hist : array
            Histogram array of shape (base, base, base, ...)*number of
            dimensions with Z-dimensions coming first.
        """

        if self.n_symbs is None:
            n_symbs = int(symb_array.max() + 1)
        else:
            n_symbs = self.n_symbs
            if n_symbs < int(symb_array.max() + 1):
                raise ValueError("n_symbs must be >= symb_array.max() + 1 = {}".format(symb_array.max() + 1))

        if 'int' not in str(symb_array.dtype):
            raise ValueError("Input data must of integer type, where each "
                             "number indexes a symbol.")

        dim, T = symb_array.shape

        flathist = np.zeros((n_symbs ** dim), dtype='int16')
        multisymb = np.zeros(T, dtype='int64')
        if weights is not None:
            flathist = np.zeros((n_symbs ** dim), dtype='float32')
            multiweights = np.ones(T, dtype='float32')

        for i in range(dim):
            multisymb += symb_array[i, :] * n_symbs ** i
            if weights is not None:
                multiweights *= weights[i, :]

        if weights is None:
            result = np.bincount(multisymb)
        else:
            result = (np.bincount(multisymb, weights=multiweights)
                      / multiweights.sum())

        flathist[:len(result)] += result

        hist = flathist.reshape(tuple([n_symbs, n_symbs] +
                                      [n_symbs for i in range(dim - 2)])).T

        return hist

    def get_dependence_measure(self, array, xyz):
        """Returns CMI estimate based on bincount histogram.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """

        _, T = array.shape

        # High-dimensional histogram
        hist = self._bincount_hist(array, weights=None)

        def _plogp_vector(T):
            """Precalculation of p*log(p) needed for entropies."""
            gfunc = np.zeros(T + 1)
            data = np.arange(1, T + 1, 1)
            gfunc[1:] = data * np.log(data)
            def plogp_func(time):
                return gfunc[time]
            return np.vectorize(plogp_func)

        plogp = _plogp_vector(T)
        hxyz = (-(plogp(hist)).sum() + plogp(T)) / float(T)
        hxz = (-(plogp(hist.sum(axis=1))).sum() + plogp(T)) / float(T)
        hyz = (-(plogp(hist.sum(axis=0))).sum() + plogp(T)) / float(T)
        hz = (-(plogp(hist.sum(axis=0).sum(axis=0))).sum()+plogp(T)) / float(T)
        val = hxz + hyz - hz - hxyz
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

        null_dist = self._get_shuffle_dist(array, xyz,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval

class RCOT(CondIndTest):
    r"""Randomized Conditional Correlation Test.

    Tests conditional independence in the fully non-parametric setting based on
    Kernel measures. For not too small sample sizes, the test can utilize an
    analytic approximation of the null distribution making it very fast. Based
    on r-package ``rcit``. This test is described in [5]_.

    Notes
    -----

    RCOT is a fast variant of the Kernel Conditional Independence Test (KCIT)
    utilizing random Fourier features. Kernel tests measure conditional
    independence in the fully non-parametric setting. In practice, RCOT tests
    scale linearly with sample size and return accurate p-values much faster
    than KCIT in the large sample size context. To use the analytical null
    approximation, the sample size should be at least ~1000.

    The method is fully described in [5]_ and the r-package documentation. The
    free parameters are the approximation of the partial kernel cross-covariance
    matrix and the number of random fourier features for the conditioning set.
    One caveat is that RCOT is, as the name suggests, based on random fourier
    features. To get reproducable results, you should fix the seed (default).

    This class requires the rpy package and the prior installation of ``rcit``
    from https://github.com/ericstrobl/RCIT. This is provided with tigramite
    as an external package.

    References
    ----------
    .. [5] Eric V. Strobl, Kun Zhang, Shyam Visweswaran:
           Approximate Kernel-based Conditional Independence Tests for Fast Non-
           Parametric Causal Discovery.
           https://arxiv.org/abs/1702.03877

    Parameters
    ----------
    num_f : int, optional
        Number of random fourier features for conditioning set. More features
        better approximate highly structured joint densities, but take more
        computational time.

    approx : str, optional
        Which approximation of the partial cross-covariance matrix, options:
        'lpd4' the Lindsay-Pilla-Basak method (default), 'gamma' for the
        Satterthwaite-Welch method, 'hbe' for the Hall-Buckley-Eagleson method,
        'chi2' for a normalized chi-squared statistic, 'perm' for permutation
        testing (warning: this one is slow).

    seed : int or None, optional
        Which random fourier feature seed to use. If None, you won't get
        reproducable results.

    significance : str, optional (default: 'analytic')
        Type of significance test to use.

    **kwargs :
        Arguments passed on to parent class CondIndTest.
    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 num_f=25,
                 approx="lpd4",
                 seed=42,
                 significance='analytic',
                 **kwargs):
        # Set the members
        self.num_f = num_f
        self.approx = approx
        self.seed = seed
        self._measure = 'rcot'
        self.two_sided = False
        self.residual_based = False
        self._pval = None
        # Call the parent constructor
        CondIndTest.__init__(self, significance=significance, **kwargs)

        # Print some information
        if self.verbosity > 0:
            print("num_f = %s" % self.num_f + "\n")
            print("approx = %s" % self.approx + "\n\n")

    def get_dependence_measure(self, array, xyz):
        """Returns RCOT estimate.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            RCOT estimate.
        """
        dim, T = array.shape
        x_vals = array[0]
        y_vals = array[1]
        z_vals = np.fastCopyAndTranspose(array[2:])

        rcot = rpy2.robjects.r['RCoT'](x_vals, y_vals, z_vals,
                                       num_f=self.num_f,
                                       approx=self.approx,
                                       seed=self.seed)

        val = float(rcot.rx2('Sta')[0])
        # Cache the p-value for use later
        self._pval = float(rcot.rx2('p')[0])

        return val

    def get_analytic_significance(self, **args):
        """
        Returns analytic p-value from RCoT test statistic.
        NOTE: Must first run get_dependence_measure, where p-value is determined
        from RCoT test statistic.

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        return self._pval

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

        null_dist = self._get_shuffle_dist(array, xyz,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval
