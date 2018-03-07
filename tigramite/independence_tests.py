"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0

import warnings
import numpy
import sys, os
import math

from scipy import linalg, special, stats

try:
    from sklearn import gaussian_process
except:
    print("Could not import sklearn for GPACE")

try:
    from scipy import spatial
    from tigramite import tigramite_cython_code
except:
    print("Could not import packages for CMIknn and GPDC estimation")


# try:
#     import rpy2
#     import rpy2.robjects
#     rpy2.robjects.r['options'](warn=-1)

#     from rpy2.robjects.packages import importr
#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
# except:
#         print("Could not import rpy package")

try:
    importr('RCIT')
except:
    print("Could not import r-package RCIT")

try:
    importr('acepack')
except:
    print("Could not import r-package acepack for GPACE,"
          " use python ACE package")

try: 
    import ace
except:
    print("Could not import python ACE package for GPACE")



# @staticmethod
def _construct_array(X, Y, Z, tau_max, data,
                     use_mask=False,
                     mask=None, mask_type=None,
                     missing_flag=None,
                     return_cleaned_xyz=False,
                     do_checks=True,
                     cut_off='2xtau_max',
                     verbosity=0):
    """Constructs array from variables X, Y, Z from data.

    Data is of shape (T, N), where T is the time series length and N the
    number of variables.

    Parameters
    ----------
    X, Y, Z : list of tuples
        For a dependence measure I(X;Y|Z), Y is of the form [(varY, 0)],
        where var specifies the variable index. X typically is of the form
        [(varX, -tau)] with tau denoting the time lag and Z can be
        multivariate [(var1, -lag), (var2, -lag), ...] .

    tau_max : int
        Maximum time lag. This may be used to make sure that estimates for
        different lags in X and Z all have the same sample size.

    data : array-like, 
        This is the data input array of shape = (T, N)

    use_mask : bool, optional (default: False)
        Whether a supplied mask should be used.

    mask : boolean array, optional (default: False)
        Mask of data array, marking masked values as 1. Must be of same
        shape as data.

    missing_flag : number, optional (default: None)
        Flag for missing values. Dismisses all time slices of samples where
        missing values occur in any variable and also flags samples for all
        lags up to 2*tau_max. This avoids biases, see section on masking in
        Supplement of [1]_.

    mask_type : {'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence
        measure I(X; Y | Z) the samples should be masked. If None, 'y' is
        used, which excludes all time slices containing masked samples in Y.
        Explained in [1]_.

    return_cleaned_xyz : bool, optional (default: False)
        Whether to return cleaned X,Y,Z, where possible duplicates are 
        removed.

    do_checks : bool, optional (default: True)
        Whether to perform sanity checks on input X,Y,Z

    cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
        How many samples to cutoff at the beginning. The default is '2xtau_max',
        which guarantees that MCI tests are all conducted on the same samples. 
        For modeling, 'max_lag_or_tau_max' can be used, which uses the maximum
        of tau_max and the conditions, which is useful to compare multiple
        models on the same sample. Last, 'max_lag' uses as much samples as
        possible.

    verbosity : int, optional (default: 0)
        Level of verbosity.

    Returns
    -------
    array, xyz [,XYZ] : Tuple of data array of shape (dim, T) and xyz 
        identifier array of shape (dim,) identifying which row in array
        corresponds to X, Y, and Z. For example::
            X = [(0, -1)], Y = [(1, 0)], Z = [(1, -1), (0, -2)]
            yields an array of shape (5, T) and xyz is  
            xyz = numpy.array([0,1,2,2])          
        If return_cleaned_xyz is True, also outputs the cleaned XYZ lists.
     
    """

    def uniq(input):
        output = []
        for x in input:
            if x not in output:
                output.append(x)
        return output

    data_type = data.dtype

    T, N = data.shape

    # Remove duplicates in X, Y, Z
    X = uniq(X)
    Y = uniq(Y)
    Z = uniq(Z)

    if do_checks:
        if len(X) == 0:
            raise ValueError("X must be non-zero")
        if len(Y) == 0:
            raise ValueError("Y must be non-zero")

    # If a node in Z occurs already in X or Y, remove it from Z
    Z = [node for node in Z if (node not in X) and (node not in Y)]

    # Check that all lags are non-positive and indices are in [0,N-1]
    XYZ = X + Y + Z
    dim = len(XYZ)

    if do_checks:
        if numpy.array(XYZ).shape != (dim, 2):
            raise ValueError("X, Y, Z must be lists of tuples in format"
                             " [(var, -lag),...], eg., [(2, -2), (1, 0), ...]")
        if numpy.any(numpy.array(XYZ)[:, 1] > 0):
            raise ValueError("nodes are %s, " % str(XYZ) +
                             "but all lags must be non-positive")
        if (numpy.any(numpy.array(XYZ)[:, 0] >= N)
                or numpy.any(numpy.array(XYZ)[:, 0] < 0)):
            raise ValueError("var indices %s," % str(numpy.array(XYZ)[:, 0]) +
                             " but must be in [0, %d]" % (N - 1))
        if numpy.all(numpy.array(Y)[:, 1] < 0):
            raise ValueError("Y-nodes are %s, " % str(Y) +
                             "but one of the Y-nodes must have zero lag")

    if cut_off == '2xtau_max':
        max_lag = 2*tau_max
    elif cut_off == 'max_lag':
        max_lag = abs(numpy.array(XYZ)[:, 1].min())
    elif cut_off == 'max_lag_or_tau_max':
        max_lag = max(abs(numpy.array(XYZ)[:, 1].min()), tau_max)

    # Setup XYZ identifier
    xyz = numpy.array([0 for i in range(len(X))] +
                      [1 for i in range(len(Y))] +
                      [2 for i in range(len(Z))])

    # Setup and fill array with lagged time series
    array = numpy.zeros((dim, T - max_lag), dtype=data_type)
    for i, node in enumerate(XYZ):
        var, lag = node
        array[i, :] = data[max_lag + lag: T + lag, var]

    if missing_flag is not None or use_mask:
        use_indices = numpy.ones(T - max_lag, dtype='int')
    
    if missing_flag is not None:
        # Dismiss all samples where missing values occur in any variable
        # and for any lag up to max_lag
        missing_anywhere = numpy.any(data==missing_flag, axis=1)
        for tau in range(max_lag+1):
            use_indices[missing_anywhere[tau:T-max_lag+tau]] = 0
   
    if use_mask:
        # Remove samples with mask == 1
        # conditional on which mask_type is used
        array_selector = numpy.zeros((dim, T - max_lag), dtype='int32')
        for i, node in enumerate(XYZ):
            var, lag = node
            array_selector[i, :] = (
                mask[max_lag + lag: T + lag, var] == False)

        # use_indices = numpy.ones(T - max_lag, dtype='int')
        if 'x' in mask_type:
            use_indices *= numpy.prod(array_selector[xyz == 0, :],
                                      axis=0)
        if 'y' in mask_type:
            use_indices *= numpy.prod(array_selector[xyz == 1, :],
                                      axis=0)
        if 'z' in mask_type:
            use_indices *= numpy.prod(array_selector[xyz == 2, :],
                                      axis=0)
    
    if missing_flag is not None or use_mask:
        if use_indices.sum() == 0:
            raise ValueError("No unmasked samples")
        array = array[:, use_indices == 1]

    if verbosity > 2:
        print("            Constructed array of shape " +
              "%s from\n" % str(array.shape) +
              "            X = %s\n" % str(X) +
              "            Y = %s\n" % str(Y) +
              "            Z = %s" % str(Z))
        if use_mask:
            print("            with masked samples in "
                  "%s removed" % mask_type)
        if missing_flag is not None:
            print("            with missing values labeled "
                  "%s removed" % missing_flag)    

    if return_cleaned_xyz:
        return array, xyz, (X, Y, Z)
    else:
        return array, xyz

class CondIndTest(object):
    """Base class of conditional independence tests.

    Provides useful general functions for different independence tests such as
    shuffle significance testing and bootstrap confidence estimation. Also
    handles masked samples. Other test classes can inherit from this class.

    Parameters
    ----------
    use_mask : bool, optional (default: False)
        Whether a supplied mask should be used.

    mask_type : {'y','x','z','xy','xz','yz','xyz'}
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

    confidence : False or str, optional (default: False)
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

    def __init__(self, 
        use_mask=False,
        mask_type=None,

        significance='analytic',
        fixed_thres=0.1,
        sig_samples=1000,
        sig_blocklength=None,

        confidence=False,
        conf_lev=0.9,
        conf_samples=100,
        conf_blocklength=None,

        recycle_residuals=False,
        verbosity=0
        ):

        self.use_mask = use_mask
        self.mask_type = mask_type
        self.significance = significance
        self.sig_samples = sig_samples
        self.sig_blocklength = sig_blocklength
        self.fixed_thres = fixed_thres

        self.confidence = confidence
        self.conf_lev = conf_lev
        self.conf_samples = conf_samples
        self.conf_blocklength = conf_blocklength

        self.verbosity = verbosity

        self.recycle_residuals = recycle_residuals
        if self.recycle_residuals:
            self.residuals = {}

        if self.use_mask:
            self.recycle_residuals = False
        if self.mask_type is None:
            self.mask_type = 'y'


        if self.verbosity > 0:
            print("\n# Initialize conditional independence test\n"
                  "\nParameters:")
            print("independence test = %s" % self.measure
                  + "\nsignificance = %s" % self.significance
                  )
            if self.significance == 'shuffle_test':
                print(""
                + "sig_samples = %s" % self.sig_samples
                + "\nsig_blocklength = %s" % self.sig_blocklength)
            elif self.significance == 'fixed_thres':
                print(""
                + "fixed_thres = %s" % self.fixed_thres)
            if self.confidence:
                print("confidence = %s" % self.confidence
                + "\nconf_lev = %s" % self.conf_lev)
                if self.confidence == 'bootstrap':
                    print(""
                    + "conf_samples = %s" % self.conf_samples
                    + "\nconf_blocklength = %s" % self.conf_blocklength)
            if self.use_mask:
                print(""
                  + "use_mask = %s" % self.use_mask
                  + "\nmask_type = %s" % self.mask_type)
            if self.recycle_residuals:
                print("recycle_residuals = %s" % self.recycle_residuals)

            # print("\n")
        # if use_mask:
        #     if mask_type is None or len(set(mask_type) -
        #                                 set(['x', 'y', 'z'])) > 0:
        #         raise ValueError("mask_type = %s, but must be list containing"
        #                          % mask_type + " 'x','y','z', or any "
        #                          "combination")

    def set_dataframe(self, dataframe):
        """Initialize dataframe.

        Parameters
        ----------
        dataframe : data object
            Set tigramite dataframe object. It must have the attributes
            dataframe.values yielding a numpy array of shape (observations T,
            variables N) and optionally a mask of the same shape and a missing
            values flag.

        """
        self.data = dataframe.values
        self.mask = dataframe.mask
        self.missing_flag = dataframe.missing_flag

    def _keyfy(self, x, z):
        """Helper function to make lists unique."""
        return (tuple(set(x)), tuple(set(z)))

    def _get_array(self, X, Y, Z, tau_max=0, verbosity=None):
        """Convencience wrapper around _construct_array."""
        
        if verbosity is None:
            verbosity=self.verbosity

        return _construct_array(
            X=X, Y=Y, Z=Z,
            tau_max=tau_max,
            data=self.data,
            use_mask=self.use_mask,
            mask=self.mask,
            mask_type=self.mask_type,
            missing_flag=self.missing_flag,
            return_cleaned_xyz=True,
            do_checks=False,
            verbosity=verbosity)

    # @profile
    def run_test(self, X, Y, Z=None, tau_max=0):
        """Perform conditional independence test.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.
        If recycle_residuals is True, also  _get_single_residuals must be
        available.

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
        val, pval : Tuple of floats
        
            The test statistic value and the p-value. These are also made in the
            class as self.val and self.pval.
        
        """

        array, xyz, XYZ = self._get_array(X, Y, Z, tau_max)
        X, Y, Z = XYZ

        dim, T = array.shape

        if numpy.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")

        if self.recycle_residuals:
            if self._keyfy(X, Z) in list(self.residuals):
                x_resid = self.residuals[self._keyfy(X, Z)]
            else:
                x_resid = self._get_single_residuals(array, target_var = 0)
                if len(Z) > 0:
                    self.residuals[self._keyfy(X, Z)] = x_resid

            if self._keyfy(Y, Z) in list(self.residuals):
                y_resid = self.residuals[self._keyfy(Y, Z)]
            else:
                y_resid = self._get_single_residuals(array, target_var = 1)
                if len(Z) > 0:
                    self.residuals[self._keyfy(Y, Z)] = y_resid

            array_resid = numpy.array([x_resid, y_resid])
            xyz_resid = numpy.array([0, 1])

            val = self.get_dependence_measure(array_resid, xyz_resid)

        else:
            val = self.get_dependence_measure(array, xyz)

        if self.significance == 'analytic':
            pval = self.get_analytic_significance(value=val, T=T, dim=dim)

        elif self.significance == 'shuffle_test':
            pval = self.get_shuffle_significance(array=array,
                                                 xyz=xyz,
                                                 value=val)
        elif self.significance == 'fixed_thres':
            pval = self.get_fixed_thres_significance(value=val, 
                                                fixed_thres=self.fixed_thres)
        else:
            raise ValueError("%s not known." % self.significance)

        self.X = X
        self.Y = Y
        self.Z = Z
        self.val = val
        self.pval = pval

        return val, pval

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

        array, xyz, XYZ = self._get_array(X, Y, Z, tau_max)
        X, Y, Z = XYZ

        D, T = array.shape

        if numpy.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")

        if self.recycle_residuals:
            if self._keyfy(X, Z) in list(self.residuals):
                x_resid = self.residuals[self._keyfy(X, Z)]
            else:
                x_resid = self._get_single_residuals(array, target_var = 0)
                if len(Z) > 0:
                    self.residuals[self._keyfy(X, Z)] = x_resid

            if self._keyfy(Y, Z) in list(self.residuals):
                y_resid = self.residuals[self._keyfy(Y, Z)]
            else:
                y_resid = self._get_single_residuals(array,target_var = 1)
                if len(Z) > 0:
                    self.residuals[self._keyfy(Y, Z)] = y_resid

            array_resid = numpy.array([x_resid, y_resid])
            xyz_resid = numpy.array([0, 1])

            val = self.get_dependence_measure(array_resid, xyz_resid)

        else:
            val = self.get_dependence_measure(array, xyz)

        return val

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

        if self.confidence:
            if (self.conf_lev < .5 or self.conf_lev >= 1.):
                raise ValueError("conf_lev = %.2f, " % self.conf_lev +
                                 "but must be between 0.5 and 1")
            if (self.confidence == 'bootstrap'
                    and self.conf_samples * (1. - self.conf_lev) / 2. < 1.):
                raise ValueError("conf_samples*(1.-conf_lev)/2 is %.2f"
                                 % (self.conf_samples * (1. - self.conf_lev) / 2.) +
                                 ", must be >> 1")

        array, xyz, XYZ = self._get_array(X, Y, Z, tau_max, verbosity=0)

        dim, T = array.shape

        if numpy.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")

        if self.confidence == 'analytic':
            val = self.get_dependence_measure(array, xyz)

            (conf_lower, conf_upper) = self.get_analytic_confidence(df=T-dim, 
                                    value=val, conf_lev=self.conf_lev)

        elif self.confidence == 'bootstrap':
            # Overwrite analytic values
            (conf_lower, conf_upper) = self.get_bootstrap_confidence(array, xyz,
                             dependence_measure=self.get_dependence_measure,
                             conf_samples=self.conf_samples, 
                             conf_blocklength=self.conf_blocklength,
                             conf_lev=self.conf_lev, verbosity=self.verbosity)
        elif self.confidence == False:
            return None

        else:
            raise ValueError("%s confidence estimation not implemented" 
                             % self.confidence)

        self.conf = (conf_lower, conf_upper)

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

    # @profile
    def get_bootstrap_confidence(self, array, xyz, dependence_measure,
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

        dependence_measure : object
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

        # confidence interval is two-sided
        c_int = (1. - (1. - conf_lev) / 2.)
        dim, T = array.shape

        if conf_blocklength is None:
            conf_blocklength = self._get_block_length(array, xyz,
                                                     mode='confidence')

        n_blocks = int(math.ceil(float(T) / float(conf_blocklength)))

        if verbosity > 2:
            print("            block_bootstrap confidence intervals"
                  " with block-length = %d ..." % conf_blocklength)

        bootdist = numpy.zeros(conf_samples)
        for sam in range(conf_samples):
            rand_block_starts = numpy.random.randint(0,
                         T - conf_blocklength + 1, n_blocks)
            array_bootstrap = numpy.zeros((dim, n_blocks*conf_blocklength), 
                                          dtype = array.dtype)

            # array_bootstrap = array[:, rand_block_starts]
            for b in range(conf_blocklength):
                array_bootstrap[:, b::conf_blocklength] = array[:, 
                                                          rand_block_starts + b]

            # Cut to proper length
            array_bootstrap = array_bootstrap[:, :T]
            bootdist[sam] = dependence_measure(array_bootstrap, xyz)

        # Sort and get quantile
        bootdist.sort()
        conf_lower = bootdist[int((1. - c_int) * conf_samples)]
        conf_upper = bootdist[int(c_int * conf_samples)]

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
        if max_lag is None:
            max_lag = int(max(5, 0.1*len(series)))

        autocorr = numpy.ones(max_lag + 1)
        for lag in range(1, max_lag + 1):

            y1 = series[lag:]
            y2 = series[:len(series) - lag]

            autocorr[lag] = numpy.corrcoef(y1, y2, ddof=0)[0, 1]

        return autocorr

    def _get_block_length(self, array, xyz, mode):
        """Returns optimal block length for significance and confidence tests.

        Determine block length using approach in Mader (2013) [Eq. (6)] which
        improves the method of Pfeifer (2005) with non-overlapping blocks In
        case of multidimensional X, the max is used. Further details in [1]_.
        Two modes are available. For mode='significance', only the indices
        corresponding to X are shuffled in array. For  mode='confidence' all
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

        from scipy import signal, optimize

        dim, T = array.shape

        if mode == 'significance':
            indices = numpy.where(xyz == 0)[0]
        else:
            indices = range(dim)

        # Maximum lag for autocov estimation
        max_lag = int(0.1*T)

        def func(x, a, decay):
            return a * decay**x

        block_len = 1
        for i in indices:

            # Get decay rate of envelope of autocorrelation functions
            # via hilbert trafo
            autocov = self._get_acf(series=array[i], max_lag=max_lag)

            autocov[0] = 1.
            hilbert = numpy.abs(signal.hilbert(autocov))

            try:
                popt, pcov = optimize.curve_fit(
                    func, range(0, max_lag + 1), hilbert)
                phi = popt[1]

                # Formula of Pfeifer (2005) assuming non-overlapping blocks
                l_opt = (4. * T * (phi / (1. - phi) + phi**2 / (1. - phi)**2)**2
                         / (1. + 2. * phi / (1. - phi))**2)**(1. / 3.)

                block_len = max(block_len, int(l_opt))

            except RuntimeError:
                print(
                    "Error - curve_fit failed in block_shuffle, using"
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

        x_indices = numpy.where(xyz == 0)[0]
        dim_x = len(x_indices)

        if sig_blocklength is None:
            sig_blocklength = self._get_block_length(array, xyz,
                                                     mode='significance')

        n_blocks = int(math.floor(float(T) / float(sig_blocklength)))
        # print 'n_blocks ', n_blocks
        if verbosity > 2:
            print("            Significance test with block-length = %d "
                  "..." % (sig_blocklength))

        array_shuffled = numpy.copy(array)
        block_starts = numpy.arange(0, T - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blocks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blocks*sig_blocklength:]

        null_dist = numpy.zeros(sig_samples)
        for sam in range(sig_samples):

            rand_block_starts = numpy.random.permutation(block_starts)[:n_blocks]

            x_shuffled = numpy.zeros((dim_x, n_blocks*sig_blocklength), 
                                          dtype = array.dtype)

            for i, index in enumerate(x_indices):
                for b in range(sig_blocklength):
                    x_shuffled[i, b::sig_blocklength] = array[index, 
                                            rand_block_starts + b]

            # Insert tail randomly somewhere
            if tail.shape[1] > 0:
                insert_tail_at = numpy.random.choice(block_starts)
                x_shuffled = numpy.insert(x_shuffled, insert_tail_at, 
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
        if numpy.abs(value) < numpy.abs(fixed_thres):
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
            xisorted = numpy.sort(xi)
            yi = numpy.linspace(1. / len(xi), 1, len(xi))
            return numpy.interp(xi, xisorted, yi)

        if numpy.ndim(x) == 1:
            u = trafo(x)
        else:
            u = numpy.empty(x.shape)
            for i in range(x.shape[0]):
                u[i] = trafo(x[i])
        return u

    def generate_nulldist(self, df, add_to_null_dists=True):
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

        xyz = numpy.array([0,1])
        null_dist = numpy.zeros(self.null_samples)
        for i in range(self.null_samples):
            array = numpy.random.rand(2, df)

            null_dist[i] = self.get_dependence_measure(array, xyz)

        null_dist.sort()    
        if add_to_null_dists:
            self.null_dists[df] = null_dist
        else:
            return null_dist

    def generate_and_save_nulldists(self, sample_sizes, null_dist_filename):
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

        null_dists = numpy.zeros((len(sample_sizes), self.sig_samples))
        
        for iT, T in enumerate(sample_sizes):
            null_dists[iT] = self.generate_nulldist(T, add_to_null_dists=False)
            self.null_dists[T] = null_dists[iT]
        
        numpy.savez("%s" % null_dist_filename, exact_dist=null_dists, 
                            T=numpy.array(sample_sizes))


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

    def __init__(self, **kwargs):

        # super(ParCorr, self).__init__(

        self.measure = 'par_corr'
        self.two_sided = True
        self.residual_based = True

        CondIndTest.__init__(self, **kwargs)
        if self.verbosity > 0:
            print("")

    # @profile
    def _get_single_residuals(self, array, target_var, 
                standardize = True,
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
            if numpy.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")

        y = array[target_var, :]

        if dim_z > 0:
            z = numpy.fastCopyAndTranspose(array[2:, :])
            beta_hat = numpy.linalg.lstsq(z, y)[0]
            mean = numpy.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        else:
            return resid

    # @profile
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

        x = self._get_single_residuals(array, target_var = 0)
        y = self._get_single_residuals(array, target_var = 1)

        val, dummy = stats.pearsonr(x, y)

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

        x = self._get_single_residuals(array, target_var = 0)
        y = self._get_single_residuals(array, target_var = 1)
        array_resid = numpy.array([x, y])
        xyz_resid = numpy.array([0,1])

        null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                               self.get_dependence_measure,
                               sig_samples=self.sig_samples, 
                               sig_blocklength=self.sig_blocklength,
                               verbosity=self.verbosity)

        pval = (null_dist >= numpy.abs(value)).mean()
        
        # Adjust p-value for two-sided measures
        if pval < 1.: pval *= 2.

        if return_null_dist:
            return pval, null_dist
        else:
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

        df = T - dim

        if df < 1:
            pval = numpy.nan
        else:
            trafo_val = value * numpy.sqrt(df / (1. - value**2))
            # Two sided significance level
            pval = stats.t.sf(numpy.abs(trafo_val), df) * 2

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

        value_tdist = value * numpy.sqrt(df) / numpy.sqrt(1. - value**2)
        conf_lower = (stats.t.ppf(q=1. - c_int, df=df, loc=value_tdist)
                      / numpy.sqrt(df + stats.t.ppf(q=1. - c_int, df=df,
                                                       loc=value_tdist)**2))
        conf_upper = (stats.t.ppf(q=c_int, df=df, loc=value_tdist)
                      / numpy.sqrt(df + stats.t.ppf(q=c_int, df=df,
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
        array, xyz = _construct_array(
            X=X, Y=Y, Z=Z,
            tau_max=tau_max,
            data=self.data,
            use_mask=self.use_mask,
            mask=self.mask,
            mask_type=self.mask_type,
            return_cleaned_xyz=False,
            do_checks=False,
            verbosity=self.verbosity)

        dim, T = array.shape

        y = self._get_single_residuals(
            array, target_var=1, return_means=False)
        # Get RSS
        rss = (y**2).sum()
        # Number of parameters
        p = dim - 1
        # Get AIC
        score = T * numpy.log(rss) + 2. * p

        return score

class GP():
    r"""Gaussian processes base class.

    GP is estimated with scikit-learn and allows to flexibly specify kernels and
    hyperparameters or let them be optimized automatically. The kernel specifies
    the covariance function of the GP. Parameters can be passed on to
    ``GaussianProcessRegressor`` using the gp_params dictionary. If None is
    passed, the kernel '1.0 * RBF(1.0) + WhiteKernel()' is used with alpha=0 as
    default. Note that the kernel's hyperparameters are optimized during
    fitting.

    Parameters
    ----------
    gp_version : {'new', 'old'}, optional (default: 'new')
        The older GP version from scikit-learn 0.17 was used for the numerical
        simulations in [1]_. The newer version from scikit-learn 0.19 is faster
        and allows more flexibility regarding kernels etc.

    gp_params : dictionary, optional (default: None)
        Dictionary with parameters for ``GaussianProcessRegressor``.
    
    """
    def __init__(self,
                gp_version='new',
                gp_params=None):

        # CondIndTest.__inits__(self,)

        self.gp_version = gp_version
        self.gp_params = gp_params
    
    # @profile
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
                return array[target_var, :], -numpy.inf
            else:
                return array[target_var, :]

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            array /= array.std(axis=1).reshape(dim, 1)
            if numpy.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")

        target_series = array[target_var, :]
        z = numpy.fastCopyAndTranspose(array[2:])
        if numpy.ndim(z) == 1:
            z = z.reshape(-1, 1)

        if self.gp_version == 'old':
            # Old GP failed for ties in the data
            def remove_ties(series, verbosity=0):
                # Test whether ties exist and add noise to destroy ties...
                cnt = 0
                while len(numpy.unique(series)) < numpy.size(series):
                    series += 1E-6 * numpy.random.rand(*series.shape)
                    cnt += 1
                    if cnt > 100: break
                return series

            z = remove_ties(z)  #z += 1E-3 * numpy.random.rand(*z.shape)
            target_series = remove_ties(target_series)  #target_series += 1E-3 * numpy.random.rand(*target_series.shape)
            
            gp = gaussian_process.GaussianProcess(
                nugget=1E-1,
                thetaL=1E-16,
                thetaU=numpy.inf,
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

            gp = gaussian_process.GaussianProcessRegressor(
                kernel=kernel,
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

        if return_means and return_likelihood==False:
            return (resid, mean)
        elif return_means==False and return_likelihood:
            return (resid, likelihood)
        elif return_means and return_likelihood:
            return resid, mean, likelihood
        else:
            return resid

    def get_model_selection_criterion(self, j,
                                      parents,
                                      tau_max=0):
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
        array, xyz = _construct_array(
            X=X, Y=Y, Z=Z,
            tau_max=tau_max,
            data=self.data,
            use_mask=self.use_mask,
            mask=self.mask,
            mask_type=self.mask_type,
            return_cleaned_xyz=False,
            do_checks=False,
            verbosity=self.verbosity)

        dim, T = array.shape

        y, logli = self._get_single_residuals(array,
                            target_var=1, return_likelihood=True)

        score = -logli

        return score

class GPACE(CondIndTest,GP):
    r"""GPACE conditional independence test based on Gaussian processes and
        maximal correlation.

    GPACE is based on a Gaussian process (GP) regression and a maximal
    correlation test on the residuals. GP is estimated with scikit-learn and
    allows to flexibly specify kernels and hyperparameters or let them be
    optimized automatically. The maximal correlation test is implemented with
    the ACE estimator either from a pure python implementation (slow) or, if rpy
    is available, using the R-package 'acepack'. Here the null distribution is
    not analytically available, but can be precomputed with the function
    generate_and_save_nulldists(...) which saves a \*.npz file containing the
    null distribution for different sample sizes. This file can then be supplied
    as null_dist_filename.

    Notes
    -----
    As described in [1]_, GPACE is based on a Gaussian
    process (GP) regression and a maximal correlation test on the residuals. To
    test :math:`X \perp Y | Z`, first  :math:`Z` is regressed out from :math:`X`
    and :math:`Y` assuming the  model

    .. math::  X & =  f_X(Z) + \epsilon_{X} \\
        Y & =  f_Y(Z) + \epsilon_{Y}  \\
        \epsilon_{X,Y} &\sim \mathcal{N}(0, \sigma^2)

    using GP regression. Here :math:`\sigma^2` and the kernel bandwidth are
    optimzed using ``sklearn``. Then the residuals  are transformed to uniform
    marginals yielding :math:`r_X,r_Y` and their dependency is tested with

    .. math::  \max_{g,h}\rho\left(g(r_X),h(r_Y)\right)

    where :math:`g,h` yielding maximal correlation are obtained using the
    Alternating Conditional Expectation (ACE) algorithm. The null distribution
    of the maximal correlation can be pre-computed.

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
    
    ace_version : {'python', 'acepack'}
        Estimator for ACE estimator of maximal correlation to use. 'python'
        loads the very slow pure python version available from
        https://pypi.python.org/pypi/ace/0.3. 'acepack' loads the much faster
        version from the R-package acepack. This requires the R-interface
        rpy2 to be installed and acepack needs to be installed in R beforehand.
        Note that both versions 'python' and 'acepack' may result in different
        results. In [1]_ the acepack version was used.

    **kwargs : 
        Arguments passed on to parent class CondIndTest.

    """
    def __init__(self,
                null_dist_filename=None,
                gp_version='new',
                gp_params=None,
                ace_version='acepack',
                **kwargs):

        GP.__init__(self, 
                    gp_version=gp_version,
                    gp_params=gp_params,
                    )

        self.ace_version = ace_version

        self.measure = 'gp_ace'
        self.two_sided = False
        self.residual_based = True

        # Load null-dist file, adapt if necessary
        self.null_dist_filename = null_dist_filename

        CondIndTest.__init__(self, **kwargs)

        if self.verbosity > 0:
            print("null_dist_filename = %s" % self.null_dist_filename)
            print("gp_version = %s" % self.gp_version)
            if self.gp_params is not None:
                for key in  list(self.gp_params):
                    print("%s = %s" % (key, self.gp_params[key]))
            print("ace_version = %s" % self.ace_version)
            print("")

        if null_dist_filename is None:
            self.null_samples = self.sig_samples
            self.null_dists = {}
        else:
            null_dist_file = numpy.load(null_dist_filename)
            # self.sample_sizes = null_dist_file['T']
            self.null_dists = dict(zip(null_dist_file['T'], 
                                      null_dist_file['exact_dist']))
            # print self.null_dist
            self.null_samples = len(null_dist_file['exact_dist'][0])

    def get_dependence_measure(self, array, xyz):
        """Return GPACE measure.

        Estimated as the maximal correlation of the residuals of a GP
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
            GPACE test statistic.    
        """

        D, T = array.shape

        x = self._get_single_residuals(array, target_var=0)
        y = self._get_single_residuals(array, target_var=1)

        val = self._get_maxcorr(numpy.array([x, y]))

        return val

    # @profile
    def _get_maxcorr(self, array_resid):
        """Return maximal correlation coefficient estimated by ACE.

        Method is described in [1]_. The maximal correlation test is implemented
        with the ACE estimator either from a pure python implementation (slow)
        or, if rpy is available, using the R-package 'acepack'. The variables
        are transformed to uniform marginals using the empirical cumulative
        distribution function beforehand. Here the null distribution is not
        analytically available, but can be precomputed with the function
        generate_and_save_nulldists(...) which saves a \*.npz file containing the
        null distribution for different sample sizes. This file can then be
        supplied as null_dist_filename.

        Parameters 
        ---------- 
        array_resid : array-like     
            data array must be of shape (2, T)

        Returns
        -------
        val : float
            Maximal correlation coefficient.
        """

        # Remove ties before applying transformation to uniform marginals
        # array_resid = self._remove_ties(array_resid, verbosity=4)

        x, y = self._trafo2uniform(array_resid)

        if self.ace_version == 'python':
            class Suppressor(object):
                """Wrapper class to prevent output from ACESolver."""
                def __enter__(self):
                    self.stdout = sys.stdout
                    sys.stdout = self
                def __exit__(self, type, value, traceback):
                    sys.stdout = self.stdout
                def write(self, x): 
                    pass
            myace = ace.ace.ACESolver()
            myace.specify_data_set([x], y)
            with Suppressor():
                myace.solve()
            val = numpy.corrcoef(myace.x_transforms[0], myace.y_transform)[0,1]
        
        elif self.ace_version == 'acepack':
            ace_rpy = rpy2.robjects.r['ace'](x, y)
            val = numpy.corrcoef(numpy.asarray(ace_rpy[8]).flatten(), 
                                 numpy.asarray(ace_rpy[9]))[0, 1]
        else:
            raise ValueError("ace_version must be 'python' or 'acepack'")
        
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

        x = self._get_single_residuals(array, target_var = 0)
        y = self._get_single_residuals(array, target_var = 1)
        array_resid = numpy.array([x, y])
        xyz_resid = numpy.array([0,1])

        null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                               self.get_dependence_measure,
                               sig_samples=self.sig_samples, 
                               sig_blocklength=self.sig_blocklength,
                               verbosity=self.verbosity)

        pval = (null_dist >= value).mean()
        
        if return_null_dist:
            return pval, null_dist
        else:
            return pval

    def get_analytic_significance(self, value, T, dim):
        """Returns p-value for the maximal correlation coefficient.
        
        The null distribution for necessary degrees of freedom (df) is loaded.
        If not available, the null distribution is generated with the function
        generate_nulldist(). It can be precomputed with the function
        generate_and_save_nulldists(...) which saves a \*.npz file containing the
        null distribution for different sample sizes. This file can then be
        supplied as null_dist_filename. The maximal correlation coefficient is
        one-sided. If the degrees of freedom are less than 1, numpy.nan is
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
            pval = numpy.nan
        else:
            # idx_near = (numpy.abs(self.sample_sizes - df)).argmin()

            if int(df) not in list(self.null_dists):
            # numpy.abs(self.sample_sizes[idx_near] - df) / float(df) > 0.01:
                if self.verbosity > 0:
                    print("Null distribution for GPACE not available "
                             "for deg. of freed. = %d."
                             "" % df) 
                self.generate_nulldist(df)

            null_dist_here = self.null_dists[df]
            pval = numpy.mean(null_dist_here > numpy.abs(value))

        return pval

    def get_analytic_confidence(self, value, df, conf_lev):
        """Placeholder function, not available."""
        raise ValueError("Analytic confidence not implemented for %s"
                         "" % self.measure)

class GPDC(CondIndTest,GP):
    r"""GPDC conditional independence test based on Gaussian processes and
        distance correlation.

    GPDC is based on a Gaussian process (GP) regression and a distance
    correlation test on the residuals [2]_. GP is estimated with scikit-learn
    and allows to flexibly specify kernels and hyperparameters or let them be
    optimized automatically. The distance correlation test is implemented with
    cython. Here the null distribution is not analytically available, but can be
    precomputed with the function generate_and_save_nulldists(...) which saves a
    \*.npz file containing the null distribution for different sample sizes. This
    file can then be supplied as null_dist_filename.

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
        Arguments passed on to parent class CondIndTest.

    """
    def __init__(self,
                null_dist_filename=None,
                gp_version='new',
                gp_params=None,
                **kwargs):

        GP.__init__(self, 
                    gp_version=gp_version,
                    gp_params=gp_params,
                    )

        self.measure = 'gp_dc'
        self.two_sided = False
        self.residual_based = True

        # Load null-dist file, adapt if necessary
        self.null_dist_filename = null_dist_filename

        CondIndTest.__init__(self, **kwargs)

        if self.verbosity > 0:
            print("null_dist_filename = %s" % self.null_dist_filename)
            print("gp_version = %s" % self.gp_version)
            if self.gp_params is not None:
                for key in  list(self.gp_params):
                    print("%s = %s" % (key, self.gp_params[key]))
            print("")

        if null_dist_filename is None:
            self.null_samples = self.sig_samples
            self.null_dists = {}
        else:
            null_dist_file = numpy.load(null_dist_filename)
            # self.sample_sizes = null_dist_file['T']
            self.null_dists = dict(zip(null_dist_file['T'], 
                                      null_dist_file['exact_dist']))
            # print self.null_dist
            self.null_samples = len(null_dist_file['exact_dist'][0])

    
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

        D, T = array.shape

        x = self._get_single_residuals(array, target_var=0)
        y = self._get_single_residuals(array, target_var=1)

        val = self._get_dcorr(numpy.array([x, y]))

        return val


    # @profile
    def _get_dcorr(self, array_resid):
        """Return distance correlation coefficient.
        
        The variables are transformed to uniform marginals using the empirical
        cumulative distribution function beforehand. Here the null distribution
        is not analytically available, but can be precomputed with the function
        generate_and_save_nulldists(...) which saves a \*.npz file containing the
        null distribution for different sample sizes. This file can then be
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

        x, y = self._trafo2uniform(array_resid)

        dc, val, dvx, dvy = tigramite_cython_code.dcov_all(x, y)
        
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

        x = self._get_single_residuals(array, target_var = 0)
        y = self._get_single_residuals(array, target_var = 1)
        array_resid = numpy.array([x, y])
        xyz_resid = numpy.array([0,1])

        null_dist = self._get_shuffle_dist(array_resid, xyz_resid,
                               self.get_dependence_measure,
                               sig_samples=self.sig_samples, 
                               sig_blocklength=self.sig_blocklength,
                               verbosity=self.verbosity)

        pval = (null_dist >= value).mean()
        
        if return_null_dist:
            return pval, null_dist
        else:
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
            pval = numpy.nan
        else:
            # idx_near = (numpy.abs(self.sample_sizes - df)).argmin()
            if int(df) not in list(self.null_dists):
            # if numpy.abs(self.sample_sizes[idx_near] - df) / float(df) > 0.01:
                if self.verbosity > 0:
                    print("Null distribution for GPDC not available "
                             "for deg. of freed. = %d."
                             "" % df) 
                    
                self.generate_nulldist(df)

            null_dist_here = self.null_dists[int(df)]
            pval = numpy.mean(null_dist_here > numpy.abs(value))

        return pval

    def get_analytic_confidence(self, value, df, conf_lev):
        """Placeholder function, not available."""
        raise ValueError("Analytic confidence not implemented for %s"
                         "" % self.measure)

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

    This class requires the scipy.spatial.cKDTree package and the tigramite
    cython module.

    References
    ----------
    .. [3] J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths: 
           Escaping the Curse of Dimensionality in Estimating Multivariate 
           Transfer Entropy. Physical Review Letters, 108(25), 258701. 
           http://doi.org/10.1103/PhysRevLett.108.258701

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

    transform : {'standardize', 'ranks',  'uniform', False}, optional 
        (default: 'standardize')
        Whether to transform the array beforehand by standardizing
        or transforming to uniform marginals.

    significance : str, optional (default: 'shuffle_test')
        Type of significance test to use. For CMIknn only 'fixed_thres' and 
        'shuffle_test' are available.

    **kwargs : 
        Arguments passed on to parent class CondIndTest. 
    """
    def __init__(self,
                knn=0.2,
                shuffle_neighbors=5,
                significance='shuffle_test',
                transform='standardize',
                **kwargs):


        self.knn = knn
        self.shuffle_neighbors = shuffle_neighbors
        self.transform = transform

        self.measure = 'cmi_knn'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False

        CondIndTest.__init__(self, significance=significance, **kwargs)

        if self.verbosity > 0:
            if self.knn < 1:
                print("knn/T = %s" % self.knn)
            else:
                print("knn = %s" % self.knn)
            print("shuffle_neighbors = %d" % self.shuffle_neighbors)
            print("")

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
                  * numpy.random.rand(array.shape[0], array.shape[1]))

        if self.transform == 'standardize':
            # Standardize
            array = array.astype('float')
            array -= array.mean(axis=1).reshape(dim, 1)
            array /= array.std(axis=1).reshape(dim, 1)
            # FIXME: If the time series is constant, return nan rather than
            # raising Exception
            if numpy.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")
        elif self.transform == 'uniform':
            array = self._trafo2uniform(array)
        elif self.transform == 'ranks':
            array = array.argsort(axis=1).argsort(axis=1).astype('float')


        # Use cKDTree to get distances eps to the k-th nearest neighbors for
        # every sample in joint space XYZ with maximum norm
        tree_xyz = spatial.cKDTree(array.T)
        epsarray = tree_xyz.query(array.T, k=knn+1, p=numpy.inf,
                                  eps=0.)[0][:,knn].astype('float')

        # Prepare for fast cython access
        dim_x = int(numpy.where(xyz == 0)[0][-1] + 1)
        dim_y = int(numpy.where(xyz == 1)[0][-1] + 1 - dim_x)

        k_xz, k_yz, k_z = \
         tigramite_cython_code._get_neighbors_within_eps_cython(array, T, dim_x,
         dim_y, epsarray, knn, dim)

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

        k_xz, k_yz, k_z = self._get_nearest_neighbors(array=array, xyz=xyz,
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
        x_indices = numpy.where(xyz == 0)[0]
        z_indices = numpy.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            if self.verbosity > 2:
                print("            nearest-neighbor shuffle significance "
                      "test with n = %d and %d surrogates" % (
                        self.shuffle_neighbors,  self.sig_samples))

            # Get nearest neighbors around each sample point in Z
            z_array = numpy.fastCopyAndTranspose(array[z_indices,:])
            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array, 
                        k=self.shuffle_neighbors, 
                        p=numpy.inf, 
                        eps=0.)[1].astype('int32')  
            # print neighbors

            null_dist = numpy.zeros(self.sig_samples)
            for sam in range(self.sig_samples):

                # Generate random order in which to go through indices loop in
                # next step
                order = numpy.random.permutation(T).astype('int32')

                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = \
                    tigramite_cython_code._get_restricted_permutation_cython(
                                T=T, 
                                shuffle_neighbors=self.shuffle_neighbors, 
                                neighbors=neighbors, 
                                order = order)

                array_shuffled = numpy.copy(array)
                for i in x_indices:
                    array_shuffled[i] = array[i, restricted_permutation]

                null_dist[sam] = self.get_dependence_measure(array_shuffled, 
                                                             xyz)
        
        else:
            null_dist = self._get_shuffle_dist(array, xyz,
                               self.get_dependence_measure,
                               sig_samples=self.sig_samples, 
                               sig_blocklength=self.sig_blocklength,
                               verbosity=self.verbosity)

        # Sort
        null_dist.sort()
        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        else:
            return pval


    def get_analytic_significance(self, value, T, dim):
        """Placeholder function, not available."""
        raise ValueError("Analytic significance not implemented for %s"
                         "" % self.measure)

    def get_analytic_confidence(self, value, df, conf_lev):
        """Placeholder function, not available."""
        raise ValueError("Analytic confidence not implemented for %s"
                         "" % self.measure)

    def get_model_selection_criterion(self, j,
                                      parents,
                                      tau_max=0):
        """Placeholder function, not available."""
        raise ValueError("Model selection not implemented for %s"
                         "" % self.measure)

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
    in data_processing or use the CMIknn class.

    Notes
    -----
    CMI and its estimator are given by

    .. math:: I(X;Y|Z) &= \sum p(z)  \sum \sum  p(x,y|z) \log 
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Parameters
    ----------
    n_symbs : int, optional (default: None)
        Number of symbols in input data. If None, n_symbs=data.max()+1

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
    def __init__(self,
                n_symbs=None,
                significance='shuffle_test',
                sig_blocklength=1,
                conf_blocklength=1,
                **kwargs):


        self.measure = 'cmi_symb'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False

        self.n_symbs = n_symbs

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
            self.n_symbs = int(symb_array.max() + 1)

        if 'int' not in str(symb_array.dtype):
            raise ValueError("Input data must of integer type, where each "
                             "number indexes a symbol.")

        dim, T = symb_array.shape

        # Needed because numpy.bincount cannot process longs
        if type(self.n_symbs ** dim) != int:
            raise ValueError("Too many n_symbs and/or dimensions, "
                             "numpy.bincount cannot process longs")
        if self.n_symbs ** dim * 16. / 8. / 1024. ** 3 > 3.:
            raise ValueError("Dimension exceeds 3 GB of necessary "
                             "memory (change this code line if more...)")
        if dim * self.n_symbs ** dim > 2 ** 65:
            raise ValueError("base = %d, D = %d: Histogram failed: "
                             "dimension D*base**D exceeds int64 data type"
                             % (self.n_symbs, dim))

        flathist = numpy.zeros((self.n_symbs ** dim), dtype='int16')
        multisymb = numpy.zeros(T, dtype='int64')
        if weights is not None:
            flathist = numpy.zeros((self.n_symbs ** dim), dtype='float32')
            multiweights = numpy.ones(T, dtype='float32')

        # print numpy.prod(weights, axis=0)
        for i in range(dim):
            multisymb += symb_array[i, :] * self.n_symbs ** i
            if weights is not None:
                multiweights *= weights[i, :]
                # print i, multiweights

        if weights is None:
            result = numpy.bincount(multisymb)
            # print result
        else:
            result = (numpy.bincount(multisymb, weights=multiweights)
                      / multiweights.sum())

        flathist[:len(result)] += result

        hist = flathist.reshape(tuple([self.n_symbs, self.n_symbs] +
                                      [self.n_symbs for i in range(dim - 2)])).T

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

        dim, T = array.shape

        # High-dimensional Histogram
        hist = self._bincount_hist(array, weights=None)

        def _plogp_vector(T):
            """Precalculation of p*log(p) needed for entropies."""
            gfunc = numpy.zeros(T + 1, dtype='float')
            gfunc = numpy.zeros(T + 1)
            gfunc[1:] = numpy.arange(
                1, T + 1, 1) * numpy.log(numpy.arange(1, T + 1, 1))
            def plogp_func(t):
                return gfunc[t]
            return numpy.vectorize(plogp_func)
        
        plogp = _plogp_vector(T)
        
        hxyz = (-(plogp(hist)).sum() + plogp(T)) / float(T)
        hxz = (-(plogp(hist.sum(axis=1))).sum() + plogp(T)) / \
            float(T)
        hyz = (-(plogp(hist.sum(axis=0))).sum() + plogp(T)) / \
            float(T)
        hz = (-(plogp(hist.sum(axis=0).sum(axis=0))).sum() +
              plogp(T)) / float(T)

        # else:
        #     def plogp_func(p):
        #         if p == 0.: return 0.
        #         else: return p*numpy.log(p)
        #     plogp = numpy.vectorize(plogp_func)

        #     hxyz = -plogp(hist).sum()
        #     hxz = -plogp(hist.sum(axis=1)).sum()
        #     hyz = -plogp(hist.sum(axis=0)).sum()
        #     hz = -plogp(hist.sum(axis=0).sum(axis=0)).sum()

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
        else:
            return pval

    def get_analytic_significance(self, value, T, dim):
        """Placeholder function, not available."""
        raise ValueError("Analytic confidence not implemented for %s"
                         "" % self.measure)

    def get_analytic_confidence(self, value, df, conf_lev):
        """Placeholder function, not available."""
        raise ValueError("Analytic confidence not implemented for %s"
                         "" % self.measure)

    def get_model_selection_criterion(self, j,
                                      parents,
                                      tau_max=0):
        """Placeholder function, not available."""
        raise ValueError("Model selection not implemented for %s"
                         "" % self.measure)


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
    from https://github.com/ericstrobl/RCIT.

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
    def __init__(self,
                num_f=25,
                approx="lpd4",
                seed=42,
                significance='analytic',
                **kwargs):

        self.num_f = num_f
        self.approx = approx
        self.seed = seed

        self.measure = 'rcot'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False

        CondIndTest.__init__(self, significance=significance, **kwargs)

        if self.verbosity > 0:
            print("num_f = %s" % self.num_f)
            print("approx = %s" % self.approx)
            print("")

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

        x = array[0]
        y = array[1]
        z = numpy.fastCopyAndTranspose(array[2:])

        rcot = numpy.asarray(rpy2.robjects.r['RCIT'](x, y, z, 
            corr=True, 
            num_f=self.num_f, 
            approx=self.approx,
            seed=self.seed))
        
        val = float(rcot[1])
        self.pval = float(rcot[0])

        return val


    def get_analytic_significance(self, **args): 
        """Returns analytic p-value from RCIT test statistic.
        
        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """

        return self.pval

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
        else:
            return pval

    def get_analytic_confidence(self, value, df, conf_lev):
        """Placeholder function, not available."""
        raise ValueError("Analytic confidence not implemented for %s"
                         "" % self.measure)

    def get_model_selection_criterion(self, j,
                                      parents,
                                      tau_max=0):
        """Placeholder function, not available."""
        raise ValueError("Model selection not implemented for %s"
                         "" % self.measure)


if __name__ == '__main__':

    # Quick test
    import data_processing as pp
    numpy.random.seed(44)
    a = 0.
    c = 0.6
    T = 4000
    # Each key refers to a variable and the incoming links are supplied as a
    # list of format [((driver, lag), coeff), ...]
    links_coeffs = {0: [((0, -1), a)],
                    1: [((1, -1), a), ((0, -1), c)],
                    2: [((2, -1), a), ((1, -1), c)],
                    }

    data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)

    data_mask = numpy.zeros(data.shape)

    # cond_ind_test = ParCorr(
    #     significance='analytic',
    #     sig_samples=100,

    #     confidence='bootstrap', #'bootstrap',
    #     conf_lev=0.9,
    #     conf_samples=100,
    #     conf_blocklength=1,

    #     use_mask=False,
    #     mask_type='y',
    #     recycle_residuals=False,
    #     verbosity=3)


    # cond_ind_test = GPACE(
    #     significance='analytic',
    #     sig_samples=100,

    #     confidence=False, # False  'bootstrap',
    #     conf_lev=0.9,
    #     conf_samples=100,
    #     conf_blocklength=None,

    #     use_mask=False,
    #     mask_type='y',

    #     null_dist_filename=None,
    #     gp_version='new',
    #     ace_version='acepack',
    #     recycle_residuals=False,
    #     verbosity=4)

    # cond_ind_test = GPDC(
    #     significance='analytic',
    #     sig_samples=1000,
    #     sig_blocklength=1,

    #     confidence=False, # False  'bootstrap',
    #     conf_lev=0.9,
    #     conf_samples=100,
    #     conf_blocklength=1,

    #     use_mask=False,
    #     mask_type='y',

    #     null_dist_filename='/home/jakobrunge/test/test.npz', #'/home/tests/test.npz',
    #     gp_version='new',

    #     recycle_residuals=False,
    #     verbosity=4)

    # cond_ind_test.generate_and_save_nulldists( sample_sizes=[100, 250],
    #     null_dist_filename='/home/jakobrunge/test/test.npz')
    # cond_ind_test.null_dist_filename = '/home/jakobrunge/test/test.npz'

    cond_ind_test = CMIknn(
        significance='shuffle_test',
        sig_samples=1000,
        knn=.1,
        transform='ranks',
        shuffle_neighbors=5,
        confidence=False, #'bootstrap',
        conf_lev=0.9,
        conf_samples=100,
        conf_blocklength=None,

        use_mask=False,
        mask_type='y',
        recycle_residuals=False,
        verbosity=3,
        )

    # cond_ind_test = CMIsymb()
    #     significance='shuffle_test',
    #     sig_samples=1000,

    #     confidence='bootstrap', #'bootstrap',
    #     conf_lev=0.9,
    #     conf_samples=100,
    #     conf_blocklength=None,

    #     use_mask=False,
    #     mask_type='y',
    #     recycle_residuals=False,
    #     verbosity=3)


    # cond_ind_test = RCOT(
    #     significance='analytic',
    #     num_f=25,
    #     confidence=False, #'bootstrap', #'bootstrap',
    #     conf_lev=0.9,
    #     conf_samples=100,
    #     conf_blocklength=None,

    #     use_mask=False,
    #     mask_type='y',
    #     recycle_residuals=False,
    #     verbosity=3,
    #     )

    if cond_ind_test.measure == 'cmi_symb':
        data = pp.quantile_bin_array(data, bins=6)

    dataframe = pp.DataFrame(data)
    cond_ind_test.set_dataframe(dataframe)

    tau_max = 5
    X = [(0, -2)]
    Y = [(2, 0)]
    Z = [(1, -1)]  #(2, -1), (1, -1), (0, -3)]  #[(1, -1)]  #[(2, -1), (1, -1), (0, -3)] # [(2, -1), (1, -1), (2, -3)]   [(1, -1)]
    
    # print cond_ind_test._get_shuffle_dist

    val, pval = cond_ind_test.run_test(X, Y, Z, tau_max=tau_max)
    conf_interval = cond_ind_test.get_confidence(X, Y, Z, tau_max=tau_max)

    # print cond_ind_test.get_model_selection_criterion(2,
    #                                   [(0, -2)],
    #                                   tau_max=tau_max)

    print ("I(X,Y|Z) = %.2f | p-value = %.3f " % (val, pval))
    if conf_interval is not None:
        print ("[%.2f, %.2f]" % conf_interval)
