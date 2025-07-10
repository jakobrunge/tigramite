"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import warnings
import math
import abc
import copy
import numpy as np
import six
from hashlib import sha1


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
        Deprecated.

    sig_samples : int, optional (default: 500)
        Number of samples for shuffle significance test.

    sig_meanblocklength : int or float, optional (default: None)
        Mean block length for block-shuffle significance test. If None, the
        mean block length is determined from the autocorrelation as
        explained in Politis and White (2004) and Patton et al. (2009).

    confidence : str, optional (default: None)
        Specify type of confidence estimation. If False, numpy.nan is returned.
        'bootstrap' can be used with any test, for ParCorr also 'analytic' is
        implemented.

    conf_lev : float, optional (default: 0.9)
        Two-sided confidence interval.

    conf_samples : int, optional (default: 100)
        Number of samples for bootstrap.

    conf_meanblocklength : int or float, optional (default: None)
        Mean block length for the stationary block-bootstrap. If None, the 
        mean block length is determined from the decay of the autocorrelation
        as described in Politis and White (2004) and Patton et al. (2009).

    recycle_residuals : bool, optional (default: False)
        Specifies whether residuals should be stored. This may be faster, but
        can cost considerable memory.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """
    @abc.abstractmethod
    def get_dependence_measure(self, array, xyz, data_type=None):
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
                 seed=42,
                 mask_type=None,
                 significance='analytic',
                 fixed_thres=None,
                 sig_samples=500,
                 sig_meanblocklength=None,
                 confidence=None,
                 conf_lev=0.9,
                 conf_samples=100,
                 conf_meanblocklength=None,
                 recycle_residuals=False,
                 verbosity=0):
        # Set the dataframe to None for now, will be reset during pcmci call
        self.dataframe = None
        # Set the options
        self.random_state = np.random.default_rng(seed)
        self.significance = significance
        self.sig_samples = sig_samples
        self.sig_meanblocklength = sig_meanblocklength
        if fixed_thres is not None:
            raise ValueError("fixed_thres is replaced by providing alpha_or_thres in run_test")
        self.verbosity = verbosity
        self.cached_ci_results = {}
        self.ci_results = {}
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
        self.conf_meanblocklength = conf_meanblocklength

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
            info_str += "\nsig_meanblocklength = %s" % self.sig_meanblocklength
        # # Check if we are using a fixed threshold
        # elif self.significance == 'fixed_thres':
        #     info_str += "\nfixed_thres = %s" % self.fixed_thres
        # Check if we have a confidence type
        if self.confidence:
            info_str += "\nconfidence = %s" % self.confidence
            info_str += "\nconf_lev = %s" % self.conf_lev
        # Check if this confidence type is boostrapping
        if self.confidence == 'bootstrap':
            info_str += "\nconf_samples = %s" % self.conf_samples
            info_str += "\nconf_meanblocklength = %s" %self.conf_meanblocklength
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
                                 data_type=None,
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
            if dataframe.mask is None:
                raise ValueError("mask_type is not None, but no mask in dataframe.")
            dataframe._check_mask(dataframe.mask)

    def _keyfy(self, x, z):
        """Helper function to make lists unique."""
        return (tuple(set(x)), tuple(set(z)))

    def _get_array(self, X, Y, Z, tau_max=0, cut_off='2xtau_max',
                   remove_constant_data=False,
                   verbosity=0):
        """Convencience wrapper around construct_array."""

        if self.measure in ['par_corr', 'par_corr_wls', 'robust_par_corr', 'regressionCI', 
                            'gsquared', 'gp_dc']:
            if len(X) > 1 or len(Y) > 1:
                raise ValueError("X and Y for %s must be univariate." %
                                        self.measure)

        if self.dataframe is None:
            raise ValueError("Call set_dataframe first when using CI test outside causal discovery classes.")

        # Call the wrapped function
        array, xyz, XYZ, type_array = self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                              tau_max=tau_max,
                                              mask_type=self.mask_type,
                                              return_cleaned_xyz=True,
                                              do_checks=True,
                                              remove_overlaps=True,
                                              cut_off=cut_off,
                                              verbosity=verbosity)

        if remove_constant_data:
            zero_components = np.where(array.std(axis=1)==0.)[0]

            X, Y, Z = XYZ
            x_indices = np.where(xyz == 0)[0]
            newX = [X[entry] for entry, ind in enumerate(x_indices) if ind not in zero_components]

            y_indices = np.where(xyz == 1)[0]
            newY = [Y[entry] for entry, ind in enumerate(y_indices) if ind not in zero_components]

            z_indices = np.where(xyz == 2)[0]
            newZ = [Z[entry] for entry, ind in enumerate(z_indices) if ind not in zero_components]

            nonzero_XYZ = (newX, newY, newZ)

            nonzero_array = np.delete(array, zero_components, axis=0)
            nonzero_xyz = np.delete(xyz, zero_components, axis=0)
            if type_array is not None:
                nonzero_type_array = np.delete(type_array, zero_components, axis=0)
            else:
                nonzero_type_array = None

            return array, xyz, XYZ, type_array, nonzero_array, nonzero_xyz, nonzero_XYZ, nonzero_type_array

        return array, xyz, XYZ, type_array

    
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
                pval = self._get_p_value(val=val, array=nonzero_array, xyz=nonzero_xyz, T=T, dim=dim,
                                         data_type=nonzero_data_type)
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
                
        self.ci_results[(tuple(X), tuple(Y),tuple(Z))] = (val, pval, dependent)

        # Return the calculated value(s)
        if self.verbosity > 1:
            self._print_cond_ind_results(val=val, pval=pval, cached=cached, dependent=dependent,
                                         conf=None)                             

        if alpha_or_thres is None:
            return val, pval
        else:              
            return val, pval, dependent


    def run_test_raw(self, x, y, z=None, x_type=None, y_type=None, z_type=None, alpha_or_thres=None):
        """Perform conditional independence test directly on input arrays x, y, z.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.

        Parameters
        ----------
        x, y, z : arrays
            x,y,z are of the form (samples, dimension).

        x_type, y_type, z_type : array-like
            data arrays of same shape as x, y and z respectively, which describes whether variables
            are continuous or discrete: 0s for continuous variables and
            1s for discrete variables

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

        if np.ndim(x) != 2 or np.ndim(y) != 2:
            raise ValueError("x,y must be arrays of shape (samples, dimension)"
                             " where dimension can be 1.")

        if z is not None and np.ndim(z) != 2:
            raise ValueError("z must be array of shape (samples, dimension)"
                             " where dimension can be 1.")

        if x_type is not None or y_type is not None or z_type is not None:
            has_data_type = True
        else:
            has_data_type = False

        if x_type is None and has_data_type:
            x_type = np.zeros(x.shape, dtype='int')

        if y_type is None and has_data_type:
            y_type = np.zeros(y.shape, dtype='int')

        if z is None:
            # Get the array to test on
            array = np.vstack((x.T, y.T))
            if has_data_type:
                data_type = np.vstack((x_type.T, y_type.T))

            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])])

        else:
            # Get the array to test on
            array = np.vstack((x.T, y.T, z.T))
            if z_type is None and has_data_type:
                z_type = np.zeros(z.shape, dtype='int')

            if has_data_type:
                data_type = np.vstack((x_type.T, y_type.T, z_type.T))
            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])] +
                           [2 for i in range(z.shape[1])])
        
        if self.significance == 'fixed_thres' and alpha_or_thres is None:
            raise ValueError("significance == 'fixed_thres' requires setting alpha_or_thres")

        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")
        # Get the dependence measure
        if has_data_type:
            val = self.get_dependence_measure(array, xyz, data_type=data_type)
        else:
            val = self.get_dependence_measure(array, xyz)


        # Get the p-value (returns None if significance='fixed_thres')
        if has_data_type:
            pval = self._get_p_value(val=val, array=array, xyz=xyz,
                    T=T, dim=dim, data_type=data_type)
        else:
            pval = self._get_p_value(val=val, array=array, xyz=xyz,
                    T=T, dim=dim)

        # Make test decision
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

        # Return the value and the pvalue
        if alpha_or_thres is None:
            return val, pval
        else:              
            return val, pval, dependent

    def get_dependence_measure_raw(self, x, y, z=None, x_type=None, y_type=None, z_type=None):
        """Return test statistic directly on input arrays x, y, z.

        Calls the dependence measure function. The child classes must specify
        a function get_dependence_measure.

        Parameters
        ----------
        x, y, z : arrays
            x,y,z are of the form (samples, dimension).

        x_type, y_type, z_type : array-like
            data arrays of same shape as x, y and z respectively, which describes whether variables
            are continuous or discrete: 0s for continuous variables and
            1s for discrete variables

        Returns
        -------
        val : float
            The test statistic value.
        """

        if np.ndim(x) != 2 or np.ndim(y) != 2:
            raise ValueError("x,y must be arrays of shape (samples, dimension)"
                             " where dimension can be 1.")

        if z is not None and np.ndim(z) != 2:
            raise ValueError("z must be array of shape (samples, dimension)"
                             " where dimension can be 1.")

        if x_type is not None or y_type is not None or z_type is not None:
            has_data_type = True
        else:
            has_data_type = False

        if x_type is None and has_data_type:
            x_type = np.zeros(x.shape, dtype='int')

        if y_type is None and has_data_type:
            y_type = np.zeros(y.shape, dtype='int')

        if z is None:
            # Get the array to test on
            array = np.vstack((x.T, y.T))
            if has_data_type:
                data_type = np.vstack((x_type.T, y_type.T))

            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])])

        else:
            # Get the array to test on
            array = np.vstack((x.T, y.T, z.T))
            if z_type is None and has_data_type:
                z_type = np.zeros(z.shape, dtype='int')

            if has_data_type:
                data_type = np.vstack((x_type.T, y_type.T, z_type.T))
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
        if has_data_type:
            val = self.get_dependence_measure(array, xyz, data_type=data_type)
        else:
            val = self.get_dependence_measure(array, xyz)
              
        return val

    def _get_dependence_measure_recycle(self, X, Y, Z, xyz, array, data_type=None):
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

       data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.

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
            # data type can only be continuous in this case
            return self.get_dependence_measure(array_resid, xyz_resid)

        # If not, return the dependence measure on the array and xyz
        if data_type is not None:
            return self.get_dependence_measure(array, xyz, 
                                        data_type=data_type)
        else:
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

    def _get_p_value(self, val, array, xyz, T, dim,
                         data_type=None,
                         sig_override=None):
        """
        Returns the p-value from whichever significance function is specified
        for this test. If an override is used, then it will call a different
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
            
       data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.

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
            pval = self.get_analytic_significance(value=val, T=T, dim=dim, xyz=xyz)
        # Check if we are using the shuffle significance
        elif use_sig == 'shuffle_test':
            pval = self.get_shuffle_significance(array=array,
                                                 xyz=xyz,
                                                 value=val,
                                                 data_type=data_type)
        # Check if we are using the fixed_thres significance
        elif use_sig == 'fixed_thres':
            # Determined outside then
            pval = None
            # if self.two_sided:
            #     dependent = np.abs(val) >= np.abs(alpha_or_thres)
            # else:
            #     dependent = val >= alpha_or_thres
            # pval = 0. if dependent else 1.
            # # pval = self.get_fixed_thres_significance(
            # #         value=val,
            # #         fixed_thres=self.fixed_thres)
        else:
            raise ValueError("%s not known." % self.significance)

        # # Return the calculated value(s)
        # if alpha_or_thres is not None:
        #     if use_sig != 'fixed_thres': 
        #         dependent = pval <= alpha_or_thres 
        #     return pval, dependent 
        # else:
        return pval

    def get_measure(self, X, Y, Z=None, tau_max=0, 
                    data_type=None):
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
        
       data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.


        Returns
        -------
        val : float
            The test statistic value.

        """

        # Get the array to test on
        (array, xyz, XYZ, data_type, 
         nonzero_array, nonzero_xyz, nonzero_XYZ, nonzero_data_type) = self._get_array(
                                            X=X, Y=Y, Z=Z, tau_max=tau_max,
                                            remove_constant_data=True, 
                                            verbosity=self.verbosity)
        X, Y, Z = XYZ
        nonzero_X, nonzero_Y, nonzero_Z = nonzero_XYZ

        # Record the dimensions
        # dim, T = array.shape

        # Ensure it is a valid array
        if np.any(np.isnan(array)):
            raise ValueError("nans in the array!")

        # If all X or all Y are zero, then return pval=1, val=0, dependent=False
        if len(nonzero_X) == 0 or len(nonzero_Y) == 0:
            val = 0.
        else:
            # Get the dependence measure, reycling residuals if need be
            val = self._get_dependence_measure_recycle(nonzero_X, nonzero_Y, nonzero_Z, 
                                        nonzero_xyz, nonzero_array, nonzero_data_type)
          
        return val

    def get_confidence(self, X, Y, Z=None, tau_max=0,
                       data_type=None):
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
            
       data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.

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
            array, xyz, _, data_type = self._get_array(X=X, Y=Y, Z=Z, tau_max=tau_max,
                                            remove_constant_data=False, verbosity=0)
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
                            conf_meanblocklength=self.conf_meanblocklength,
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

    def _print_cond_ind_results(self, val, pval=None, cached=None, dependent=None, conf=None):
        """Print results from conditional independence test.

        Parameters
        ----------
        val : float
            Test stastistic value.

        pval : float, optional (default: None)
            p-value

        dependent : bool
            Test decision.

        conf : tuple of floats, optional (default: None)
            Confidence bounds.
        """
        printstr = "        val = % .3f" % (val)      
        if pval is not None:
            printstr += " | pval = %.5f" % (pval)
        if dependent is not None:
            printstr += " | dependent = %s" % (dependent)
        if conf is not None:
            printstr += " | conf bounds = (%.3f, %.3f)" % (
                conf[0], conf[1])
        if cached is not None:
            printstr += " %s" % ({0:"", 1:"[cached]"}[cached])

        print(printstr)
    
    def get_bootstrap_confidence(self, array, xyz, dependence_measure=None,
                                 conf_samples=100, conf_meanblocklength=None,
                                 conf_lev=.95, 
                                 data_type=None,
                                 verbosity=0):
        """Perform bootstrap confidence interval estimation.
        Uses stationary bootstrap procedure of Politis and Romero (1994)

        With conf_meanblocklength >= 1 or None a block-bootstrap is performed.

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

        conf_meanblocklength : int or float, or None 
            Mean block length for the stationary block-bootstrap. If None, the 
            mean block length is determined from the decay of the autocorrelation
            as described in Politis and White (2004) and Patton et al. (2009). 

       data_type : array-like
            Binary data array of same shape as array which describes whether 
            individual samples in a variable (or all samples) are continuous 
            or discrete: 0s for continuous variables and 1s for discrete variables.

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

        # If no mean block length is given, determine the optimal block length.
        if conf_meanblocklength is None:
            conf_meanblocklength = \
                self._get_mean_block_length(array,xyz,mode='confidence')
        
        # Probability of new block at each index #
        pnewblk = 1.0/float(conf_meanblocklength)

        # Print some information
        if verbosity > 2:
            print("            block_bootstrap confidence intervals"
                  " with mean block-length = %d ..." % conf_meanblocklength)
            
        bootdist = np.zeros(conf_samples)
        for smpl in range(conf_samples):
            # Randomly sample the length of each block #
            blkslen = self.random_state.geometric(pnewblk,size=T+1)
            blkslen = blkslen[0:np.where(
                np.cumsum(blkslen)>T)[0][0]+1] #sum of block lengths cut to proper length
            blkslen[-1] = blkslen[-1]-(np.sum(blkslen)-T) #truncate last block to match proper length
            # Get the starting indices for the blocks #
            blk_strt = self.random_state.choice(np.arange(T),len(blkslen),replace=True)
            # Create the random sequence of indices #
            boot_draw = np.concatenate([np.arange(blk_strt[idx],blk_strt[idx]+blkslen[idx])
                                        for idx in range(len(blkslen))]) #the resampled indices
            boot_draw = boot_draw%T #wrap around (Politis and Romero, 1994 below Eq.(1))
            array_bootstrap = np.copy(array[:,boot_draw]) #values at selected indices
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

    def _get_mean_block_length(self, array, xyz, mode):
        """Returns optimal mean block length for significance and confidence tests.
    
        Determine mean block length for stationary bootstrap using approach in
        Politis and White (2004) with correction from Patton et al. (2009).
        For mode ='significance', only the indices corresponding to X are used
        in the computation of the mean block length. For mode='confidence' all
        variables are used.
        The max mean block length across variables is output.
        Adapted from code of Andrew Patton (public.econ.duke.edu/~ap172/code.html)
        See also R documentation (public.econ.duke.edu/~ap172/R_Help.pdf)
        
        Politis, D. N., & White, H. (2004). Automatic Block-Length Selection for 
        the Dependent Bootstrap. Econometric Reviews
        (PW2004)
        
        Patton, A., Politis, D. N., & White, H. (2009). Correction to “Automatic
        block-length selection for the dependent bootstrap” by D. Politis and 
        H. White. Econometric Reviews
        (PPW2009)
    
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns
    
        xyz : array of ints
            XYZ identifier array of shape (dim,).
    
        mode : str
            Which mode to use: 'significance' or 'confidence'
    
        Returns
        -------
        mean_block_len : float
            Optimal mean block length for the stationary bootstrap.
        """
        
        ### Helper functions ###
        def politis_lambda(ttt):
            '''
            lambda(t) equation of PW2004
            '''
            return(((abs(ttt)<(1/2)).astype(int)*1 + \
            (abs(ttt)>=(1/2)).astype(int)*(2*(1-abs(ttt)))) * \
            (abs(ttt)<=1).astype(int))
                
        def mlag(inmat,nlags=1,fill_val=np.nan):
            '''
            Generates a maxtrix of nlags for each variable of xmat
            xmat: n_samples*n_variables matrix (can be passed as a single array)
            nlags: number of lags to include in the output
            fill_val: fill value for the entries prior to the lag
            returns: xmat of dimensions n_samples*(n_variables*nlags)
                     where each lagged variable is a column
            Adapted from matlab code of James P. LeSage 
            '''
            numdims = inmat.ndim
            if(numdims==1): #a single array was passed
                inmat = np.reshape(inmat,(len(inmat),1))
            n_obs,n_vars = np.shape(inmat)
            outmat = fill_val*np.ones((n_obs,n_vars*nlags))
            for ii in range(n_vars):
                outmat[:,ii*nlags:(ii+1)*nlags] = np.column_stack\
                    ([np.append(lag*[np.nan],inmat[0:-1*lag,ii]) for lag in range(1,nlags+1)])
            if(numdims==1 and nlags==1): #outmat is a single array
                outmat = outmat.flatten()
            return(outmat)
        
        # Get the shape of the array
        dim, T = array.shape
        # Initiailize the indices
        indices = range(dim)
        if mode == 'significance':
            indices = np.where(xyz == 0)[0]
        n_vars = len(indices)
        
        ### Fixed parameters ###
        bigkn  = max(5,np.sqrt(np.log10(T))) #footnote c of PW2004
        smallc = 2 #footnote c of PW2004
        m_max  = math.ceil(np.sqrt(T)+bigkn) #Patton code and R documention
        b_max  = math.ceil(min(3*np.sqrt(T),T/3)) #Patton code and R documention
        ### Critical value for significance of autocorrelation ###
        critsignif = smallc*np.sqrt(np.log10(T)/T)
        ### Initialize array of optimal stationary bootstrap block sizes ###
        bsbhat = np.nan*np.ones(n_vars)
        ### Loop through variables ###
        for idx in indices:
            iivals       = np.copy(array[idx,:])
            iivalslagged = mlag(iivals,nlags=m_max)
            iivalslagged = iivalslagged[m_max:,:] #remove nan values from pre-lag entries
            corrcfs   = np.corrcoef(iivalslagged.T)[:,0] #autocorrelation
            is_small  = abs(corrcfs)<=critsignif #non-significant autocorrelation
            temp      = np.sum(np.row_stack([is_small[kk:-bigkn+kk]
                                             for kk in range(bigkn)]),axis=0)
            if(np.any(temp==bigkn)):
                # Find last index before autocorr drops below crit for at least Kn #
                mhat = np.where(temp==bigkn)[0][0]-1
                #note: use -1 to get index before drop
                #see fig.3 of PW2004: they say that mhat should be 1 
                #(not 2 thus we need -1)
            else:
                # Required drop in autocorr does not occur #
                mhat = np.where(is_small==False)[0][-1] #last index of signif autocorr
            bigm = 2*mhat #p59 of PW2004
            bigm = min(bigm,m_max) #apply m_max limit
            if(bigm==0):
                bsbhat[idx] = 1 #single sample block size
            else:
                # Compute autocovariance from 0 to bigm #
                iivalslagged  = mlag(iivals,nlags=bigm+1)
                iivalslagged  = iivalslagged[bigm+1:,:]
                covcfs        = np.cov(iivalslagged.T)[:,0] #autocovariance
                # Compute Ghat of PW2004 #
                termr      = np.append(np.flip(covcfs)[:-1],covcfs) #Rhat(k) of PW2004
                termk      = abs(np.arange(-1*bigm,bigm+1,1)) #abs(k) of PW2004
                termlambda = politis_lambda(np.arange(-1*bigm,bigm+1,1)/bigm) #lambda(k/M) of PW2004
                bigghat    = np.sum(termlambda*termk*termr) #Ghat of PW2004
                # Compute D_SBhat of PW2004 #
                smallg0    = np.sum(termlambda*termr) #ghat(0) of PW2004
                dsbhat     = 2*smallg0**2 #see PPW2009
                # Compute b_opt,sbhat of PW2004 #
                bsbhat[idx] = (((2*bigghat**2)/dsbhat)*T)**(1/3)
        bsbhat = np.maximum(1,np.minimum(b_max,bsbhat))
        # Return largest across-variable optimal mean block length #
        mean_block_len = np.max(bsbhat) 
        return(mean_block_len)

    def _get_shuffle_dist(self, array, xyz, dependence_measure,
                          sig_samples, sig_meanblocklength=None,
                          verbosity=0):
        """Returns shuffle distribution of test statistic.

        The rows in array corresponding to the X-variable are shuffled using
        a block-shuffle approach.

        Uses the stationary bootstrap of Politis and Romano (1994)
        Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
        Journal of the American Statistical association

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

        sig_meanblocklength : int or float, optional (default: None)
            Mean block length for the stationary block-bootstrap. If None, the
            mean block length is determined from the decay of the autocorrelation
            as described in Politis and White (2004) and Patton et al. (2009).

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

        if sig_meanblocklength is None:
            sig_meanblocklength = \
                self._get_mean_block_length(array,xyz,mode='significance')

        if verbosity > 2:
            print("            Significance test with mean block-length = %d "
                  "..." % (sig_meanblocklength))
        
        # Probability of new block at each index #
        pnewblk = 1.0/float(sig_meanblocklength)


        array_shuffled = np.copy(array)
        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):
            # Randomly sample the length of each block #
            blkslen = self.random_state.geometric(pnewblk,size=T+1)
            blkslen = blkslen[0:np.where(
                np.cumsum(blkslen)>T)[0][0]+1] #sum of block lengths cut to proper length
            blkslen[-1] = blkslen[-1]-(np.sum(blkslen)-T) #truncate last block to match proper length
            # Get the starting indices for the blocks #
            blk_strt = self.random_state.choice(np.arange(T),len(blkslen),replace=True)
            # Create the random sequence of indices #
            boot_draw = np.concatenate([np.arange(blk_strt[idx],blk_strt[idx]+blkslen[idx])
                                        for idx in range(len(blkslen))]) #the resampled indices
            boot_draw = boot_draw%T #wrap around (Politis and Romero, 1994 below Eq.(1))
            for i, index in enumerate(x_indices):
                array_shuffled[index,:] = np.copy(array[index,boot_draw])

            null_dist[sam] = dependence_measure(array=array_shuffled,
                                                xyz=xyz)

        return null_dist

    def get_fixed_thres_significance(self, value, fixed_thres):
        """DEPRECATED Returns signficance for thresholding test.
        """
        raise ValueError("fixed_thres is replaced by alpha_or_thres in run_test.")
        # if np.abs(value) < np.abs(fixed_thres):
        #     pval = 1.
        # else:
        #     pval = 0.

        # return pval

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
