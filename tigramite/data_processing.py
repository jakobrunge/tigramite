"""Tigramite data processing functions."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0
from __future__ import print_function
from collections import defaultdict, OrderedDict
import sys
import warnings
import copy
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numba import jit

class DataFrame():
    """Data object containing time series array and optional mask.

    Parameters
    ----------
    data : array-like
        Numpy array of shape (observations T, variables N)
    mask : array-like, optional (default: None)
        Optional mask array, must be of same shape as data

    Attributes
    ----------
    data : array-like
        Numpy array of shape (observations T, variables N)
    mask : array-like, optional (default: None)
        Optional mask array, must be of same shape as data
    missing_flag : number, optional (default: None)
        Flag for missing values in dataframe. Dismisses all time slices of
        samples where missing values occur in any variable and also flags
        samples for all lags up to 2*tau_max. This avoids biases, see
        section on masking in Supplement of [1]_.
    var_names : list of strings, optional (default: range(N))
        Names of variables, must match the number of variables. If None is
        passed, variables are enumerated as [0, 1, ...]
    datatime : array-like, optional (default: None)
        Timelabel array. If None, range(T) is used.
    remove_missing_upto_maxlag : bool, optional (default: False)
        Whether to remove not only missing samples, but also all neighboring
        samples up to max_lag (as given by cut_off in construct_array).
    """
    def __init__(self, data, mask=None, missing_flag=None, var_names=None,
        datatime=None, remove_missing_upto_maxlag=False):

        self.values = data.copy()
        self.mask = mask
        self.missing_flag = missing_flag
        if self.missing_flag is not None:
            self.values[self.values == self.missing_flag] = np.nan
        self.remove_missing_upto_maxlag = remove_missing_upto_maxlag
        T, N = data.shape
        # Set the variable names
        self.var_names = var_names
        # Set the default variable names if none are set
        if self.var_names is None:
            self.var_names = {i: i for i in range(N)}
        else:
            if len(self.var_names) != N:
                raise ValueError("len(var_names) != data.shape[1].")
        # Set datatime
        self.datatime = datatime
        if self.datatime is None:
            self.datatime = np.arange(T)

        # if type(self.values) != np.ndarray:
        #     raise TypeError("data is of type %s, " % type(self.values) +
        #                     "must be np.ndarray")
        if N > T:
            warnings.warn("data.shape = %s," % str(self.values.shape) +
                          " is it of shape (observations, variables) ?")
        # if np.isnan(data).sum() != 0:
        #     raise ValueError("NaNs in the data")
        self._check_mask()

        self.T = T
        self.N = N

        # If PCMCI.run_bootstrap_of is called, then the
        # bootstrap random draw can be set here
        self.bootstrap = None

    def _check_mask(self, mask=None, require_mask=False):
        """Checks that the mask is:
            * The same shape as the data
            * Is an numpy ndarray (or subtype)
            * Does not contain any NaN entrie

        Parameters
        ----------
        require_mask : bool (default : False)
        """
        # Check that there is a mask if required
        _use_mask = mask
        if _use_mask is None:
            _use_mask = self.mask
        if require_mask and _use_mask is None:
            raise ValueError("Expected a mask, but got nothing!")
        # If we have a mask, check it
        if _use_mask is not None:
            # Check the mask inherets from an ndarray
            if not isinstance(_use_mask, np.ndarray):
                raise TypeError("mask is of type %s, " %
                                type(_use_mask) +
                                "must be numpy.ndarray")
            # Check if there is an nan-value in the mask
            if np.isnan(np.sum(_use_mask)):
                raise ValueError("NaNs in the data mask")
            # Check the mask and the values have the same shape
            if self.values.shape != _use_mask.shape:
                raise ValueError("shape mismatch: dataframe.values.shape = %s"
                                 % str(self.values.shape) + \
                                 " but mask.shape = %s, must be identical"
                                 % str(_use_mask.shape))

    def construct_array(self, X, Y, Z, tau_max,
                        extraZ=None,
                        mask=None,
                        mask_type=None,
                        return_cleaned_xyz=False,
                        do_checks=True,
                        cut_off='2xtau_max',
                        verbosity=0):
        """Constructs array from variables X, Y, Z from data.

        Data is of shape (T, N), where T is the time series length and N the
        number of variables.

        Parameters
        ----------
        X, Y, Z, extraZ : list of tuples
            For a dependence measure I(X;Y|Z), X, Y, Z can be multivariate of
            the form [(var1, -lag), (var2, -lag), ...]. At least one varlag in Y 
            has to be at lag zero. extraZ is only used in CausalEffects class.
        tau_max : int
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X and Z all have the same sample size.
        mask : array-like, optional (default: None)
            Optional mask array, must be of same shape as data.  If it is set,
            then it overrides the self.mask assigned to the dataframe. If it is
            None, then the self.mask is used, if it exists.
        mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        return_cleaned_xyz : bool, optional (default: False)
            Whether to return cleaned X,Y,Z, where possible duplicates are
            removed.
        do_checks : bool, optional (default: True)
            Whether to perform sanity checks on input X,Y,Z
        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples. For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample.  Last,
            'max_lag' uses as much samples as possible.
        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        array, xyz [,XYZ] : Tuple of data array of shape (dim, time) and xyz
            identifier array of shape (dim,) identifying which row in array
            corresponds to X, Y, and Z. For example:: X = [(0, -1)], Y = [(1,
            0)], Z = [(1, -1), (0, -2)] yields an array of shape (4, T) and
            xyz is xyz = numpy.array([0,1,2,2]) If return_cleaned_xyz is
            True, also outputs the cleaned XYZ lists.

        """

        # Get the length in time and the number of nodes
        T, N = self.values.shape

        if extraZ is None:
            extraZ = []
        # Remove duplicates in X, Y, Z, extraZ
        X = list(OrderedDict.fromkeys(X))
        Y = list(OrderedDict.fromkeys(Y))
        Z = list(OrderedDict.fromkeys(Z))
        extraZ = list(OrderedDict.fromkeys(extraZ))

        # If a node in Z occurs already in X or Y, remove it from Z
        Z = [node for node in Z if (node not in X) and (node not in Y)]
        extraZ = [node for node in extraZ if (node not in X) and (node not in Y) and (node not in Z)]

        # Check that all lags are non-positive and indices are in [0,N-1]
        XYZ = X + Y + Z + extraZ
        dim = len(XYZ)

        # Ensure that XYZ makes sense
        if do_checks:
            self._check_nodes(Y, XYZ, N, dim)

        # Figure out what cut off we will be using
        if cut_off == '2xtau_max':
            max_lag = 2*tau_max
        elif cut_off == 'max_lag':
            max_lag = abs(np.array(XYZ)[:, 1].min())
        elif cut_off == 'max_lag_or_tau_max':
            max_lag = max(abs(np.array(XYZ)[:, 1].min()), tau_max)
        else:
            raise ValueError("max_lag must be in {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}")

        # Setup XYZ identifier
        index_code = {'x' : 0,
                      'y' : 1,
                      'z' : 2,
                      'e' : 3}
        xyz = np.array([index_code[name]
                        for var, name in zip([X, Y, Z, extraZ], ['x', 'y', 'z', 'e'])
                        for _ in var])

        # Setup and fill array with lagged time series
        time_length = T - max_lag
        array = np.zeros((dim, time_length), dtype=self.values.dtype)
        # Note, lags are negative here
        for i, (var, lag) in enumerate(XYZ):
            if self.bootstrap is None:
                array[i, :] = self.values[max_lag + lag:T + lag, var]
            else:
                array[i, :] = self.values[self.bootstrap + lag, var]

        # Choose which indices to use
        use_indices = np.ones(time_length, dtype='int')

        # Remove all values that have missing value flag, and optionally as well the time
        # slices that occur up to max_lag after
        if self.missing_flag is not None:
            missing_anywhere = np.array(np.where(np.any(np.isnan(array), axis=0))[0])
            if self.remove_missing_upto_maxlag:
                for tau in range(max_lag+1):
                    if self.bootstrap is None:
                        delete = missing_anywhere + tau 
                        delete = delete[delete < time_length]
                        use_indices[delete] = 0
                    else:
                        use_indices[missing_anywhere[self.bootstrap] + tau] = 0
            else:
                if self.bootstrap is None:
                    use_indices[missing_anywhere] = 0
                else:
                    use_indices[missing_anywhere[self.bootstrap]] = 0

        # Use the mask override if needed
        _use_mask = mask
        if _use_mask is None:
            _use_mask = self.mask
        else:
            self._check_mask(mask=_use_mask)

        if _use_mask is not None:
            # Remove samples with mask == 1 conditional on which mask_type is
            # used Create an array selector that is the same shape as the output
            # array
            array_mask = np.zeros((dim, time_length), dtype='int32')
            # Iterate over all nodes named in X, Y, or Z
            for i, (var, lag) in enumerate(XYZ):
                # Transform the mask into the output array shape, i.e. from data
                # mask to array mask
                if self.bootstrap is None:
                    array_mask[i, :] = (_use_mask[max_lag + lag: T + lag, var] == False)
                else:
                    array_mask[i, :] = (_use_mask[self.bootstrap + lag, var] == False)

            # Iterate over defined mapping from letter index to number index,
            # i.e. 'x' -> 0, 'y' -> 1, 'z'-> 2, 'e'-> 3
            for idx, cde in index_code.items():
                # Check if the letter index is in the mask type
                if (mask_type is not None) and (idx in mask_type):
                    # If so, check if any of the data that correspond to the
                    # letter index is masked by taking the product along the
                    # node-data to return a time slice selection, where 0 means
                    # the time slice will not be used
                    slice_select = np.prod(array_mask[xyz == cde, :], axis=0)
                    use_indices *= slice_select

        if (self.missing_flag is not None) or (_use_mask is not None):
            if use_indices.sum() == 0:
                raise ValueError("No unmasked samples")
            array = array[:, use_indices == 1]

        # Print information about the constructed array
        if verbosity > 2:
            self.print_array_info(array, X, Y, Z, self.missing_flag, mask_type, extraZ)

        # Return the array and xyz and optionally (X, Y, Z)
        if return_cleaned_xyz:
            return array, xyz, (X, Y, Z)

        return array, xyz

    def _check_nodes(self, Y, XYZ, N, dim):
        """
        Checks that:
        * The requests XYZ nodes have the correct shape
        * All lags are non-positive
        * All indices are less than N
        * One of the Y nodes has zero lag

        Parameters
        ----------
        Y : list of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.
        XYZ : list of tuples
            List of nodes chosen for current independence test
        N : int
            Total number of listed nodes
        dim : int
            Number of nodes excluding repeated nodes
        """
        if np.array(XYZ).shape != (dim, 2):
            raise ValueError("X, Y, Z must be lists of tuples in format"
                             " [(var, -lag),...], eg., [(2, -2), (1, 0), ...]")
        if np.any(np.array(XYZ)[:, 1] > 0):
            raise ValueError("nodes are %s, " % str(XYZ) +
                             "but all lags must be non-positive")
        if (np.any(np.array(XYZ)[:, 0] >= N)
                or np.any(np.array(XYZ)[:, 0] < 0)):
            raise ValueError("var indices %s," % str(np.array(XYZ)[:, 0]) +
                             " but must be in [0, %d]" % (N - 1))
        # if np.all(np.array(Y)[:, 1] != 0):
        #     raise ValueError("Y-nodes are %s, " % str(Y) +
        #                      "but one of the Y-nodes must have zero lag")

    def print_array_info(self, array, X, Y, Z, missing_flag, mask_type, extraZ=None):
        """
        Print info about the constructed array

        Parameters
        ----------
        array : Data array of shape (dim, T)
            Data array.
        X, Y, Z, extraZ : list of tuples
            For a dependence measure I(X;Y|Z), Y is of the form [(varY, 0)],
            where var specifies the variable index. X typically is of the form
            [(varX, -tau)] with tau denoting the time lag and Z can be
            multivariate [(var1, -lag), (var2, -lag), ...] .
        missing_flag : number, optional (default: None)
            Flag for missing values. Dismisses all time slices of samples where
            missing values occur in any variable and also flags samples for all
            lags up to 2*tau_max. This avoids biases, see section on masking in
            Supplement of [1]_.
        mask_type : {'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        """
        if extraZ is None:
            extraZ = []
        indt = " " * 12
        print(indt + "Constructed array of shape %s from"%str(array.shape) +
              "\n" + indt + "X = %s" % str(X) +
              "\n" + indt + "Y = %s" % str(Y) +
              "\n" + indt + "Z = %s" % str(Z))
        if extraZ is not None:  
            print(indt + "extraZ = %s" % str(extraZ))
        if self.mask is not None and mask_type is not None:
            print(indt+"with masked samples in %s removed" % mask_type)
        if self.missing_flag is not None:
            print(indt+"with missing values = %s removed" % self.missing_flag)



def lowhighpass_filter(data, cutperiod, pass_periods='low'):
    """Butterworth low- or high pass filter.

    This function applies a linear filter twice, once forward and once
    backwards. The combined filter has linear phase.

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    cutperiod : int
        Period of cutoff.
    pass_periods : str, optional (default: 'low')
        Either 'low' or 'high' to act as a low- or high-pass filter

    Returns
    -------
    data : array
        Filtered data array.
    """
    try:
        from scipy.signal import butter, filtfilt
    except:
        print('Could not import scipy.signal for butterworth filtering!')

    fs = 1.
    order = 3
    ws = 1. / cutperiod / (0.5 * fs)
    b, a = butter(order, ws, pass_periods)
    if np.ndim(data) == 1:
        data = filtfilt(b, a, data)
    else:
        for i in range(data.shape[1]):
            data[:, i] = filtfilt(b, a, data[:, i])

    return data


def smooth(data, smooth_width, kernel='gaussian',
           mask=None, residuals=False):
    """Returns either smoothed time series or its residuals.

    the difference between the original and the smoothed time series
    (=residuals) of a kernel smoothing with gaussian (smoothing kernel width =
    twice the sigma!) or heaviside window, equivalent to a running mean.

    Assumes data of shape (T, N) or (T,)
    :rtype: array
    :returns: smoothed/residual data

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    smooth_width : float
        Window width of smoothing, 2*sigma for a gaussian.
    kernel : str, optional (default: 'gaussian')
        Smoothing kernel, 'gaussian' or 'heaviside' for a running mean.
    mask : bool array, optional (default: None)
        Data mask where True labels masked samples.
    residuals : bool, optional (default: False)
        True if residuals should be returned instead of smoothed data.

    Returns
    -------
    data : array-like
        Smoothed/residual data.
    """

    print("%s %s smoothing with " % ({True: "Take residuals of a ",
                                      False: ""}[residuals], kernel) +
          "window width %.2f (2*sigma for a gaussian!)" % (smooth_width))

    totaltime = len(data)
    if kernel == 'gaussian':
        window = np.exp(-(np.arange(totaltime).reshape((1, totaltime)) -
                             np.arange(totaltime).reshape((totaltime, 1))
                             ) ** 2 / ((2. * smooth_width / 2.) ** 2))
    elif kernel == 'heaviside':
        import scipy.linalg
        wtmp = np.zeros(totaltime)
        wtmp[:np.ceil(smooth_width / 2.)] = 1
        window = scipy.linalg.toeplitz(wtmp)

    if mask is None:
        if np.ndim(data) == 1:
            smoothed_data = (data * window).sum(axis=1) / window.sum(axis=1)
        else:
            smoothed_data = np.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = (
                    data[:, i] * window).sum(axis=1) / window.sum(axis=1)
    else:
        if np.ndim(data) == 1:
            smoothed_data = ((data * window * (mask==False)).sum(axis=1) /
                             (window * (mask==False)).sum(axis=1))
        else:
            smoothed_data = np.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = ((
                    data[:, i] * window * (mask==False)[:, i]).sum(axis=1) /
                    (window * (mask==False)[:, i]).sum(axis=1))

    if residuals:
        return data - smoothed_data
    else:
        return smoothed_data


def weighted_avg_and_std(values, axis, weights):
    """Returns the weighted average and standard deviation.

    Parameters
    ---------
    values : array
        Data array of shape (time, variables).
    axis : int
        Axis to average/std about
    weights : array
        Weight array of shape (time, variables).

    Returns
    -------
    (average, std) : tuple of arrays
        Tuple of weighted average and standard deviation along axis.
    """

    values[np.isnan(values)] = 0.
    average = np.ma.average(values, axis=axis, weights=weights)

    variance = np.sum(weights * (values - np.expand_dims(average, axis)
                                    ) ** 2, axis=axis) / weights.sum(axis=axis)

    return (average, np.sqrt(variance))


def time_bin_with_mask(data, time_bin_length, mask=None):
    """Returns time binned data where only about non-masked values is averaged.

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    time_bin_length : int
        Length of time bin.
    mask : bool array, optional (default: None)
        Data mask where True labels masked samples.

    Returns
    -------
    (bindata, T) : tuple of array and int
        Tuple of time-binned data array and new length of array.
    """

    T = len(data)

    time_bin_length = int(time_bin_length)

    if mask is None:
        sample_selector = np.ones(data.shape)
    else:
        # Invert mask
        sample_selector = (mask == False)

    if np.ndim(data) == 1.:
        data.shape = (T, 1)
        mask.shape = (T, 1)

    bindata = np.zeros(
        (T // time_bin_length,) + data.shape[1:], dtype="float32")
    for index, i in enumerate(range(0, T - time_bin_length + 1,
                                    time_bin_length)):
        # print weighted_avg_and_std(fulldata[i:i+time_bin_length], axis=0,
        # weights=sample_selector[i:i+time_bin_length])[0]
        bindata[index] = weighted_avg_and_std(data[i:i + time_bin_length],
                                              axis=0,
                                              weights=sample_selector[i:i +
                                              time_bin_length])[0]

    T, grid_size = bindata.shape

    return (bindata.squeeze(), T)

@jit
def _get_patterns(array, array_mask, patt, patt_mask, weights, dim, step, fac, N, T):
    v = np.zeros(dim, dtype='float')

    start = step * (dim - 1)
    for n in range(0, N):
        for t in range(start, T):
            mask = 1
            ave = 0.
            for k in range(0, dim):
                tau = k * step
                v[k] = array[t - tau, n]
                ave += v[k]
                mask *= array_mask[t - tau, n]
            ave /= dim
            var = 0.
            for k in range(0, dim):
                var += (v[k] - ave) ** 2
            var /= dim
            weights[t - start, n] = var
            if (v[0] < v[1]):
                p = 1
            else:
                p = 0
            for i in range(2, dim):
                for j in range(0, i):
                    if (v[j] < v[i]):
                        p += fac[i]
            patt[t - start, n] = p
            patt_mask[t - start, n] = mask

    return patt, patt_mask, weights

def ordinal_patt_array(array, array_mask=None, dim=2, step=1,
                        weights=False, verbosity=0):
    """Returns symbolified array of ordinal patterns.

    Each data vector (X_t, ..., X_t+(dim-1)*step) is converted to its rank
    vector. E.g., (0.2, -.6, 1.2) --> (1,0,2) which is then assigned to a
    unique integer (see Article). There are faculty(dim) possible rank vectors.

    Note that the symb_array is step*(dim-1) shorter than the original array!

    Reference: B. Pompe and J. Runge (2011). Momentary information transfer as
    a coupling measure of time series. Phys. Rev. E, 83(5), 1-12.
    doi:10.1103/PhysRevE.83.051122

    Parameters
    ----------
    array : array-like
        Data array of shape (time, variables).
    array_mask : bool array
        Data mask where True labels masked samples.
    dim : int, optional (default: 2)
        Pattern dimension
    step : int, optional (default: 1)
        Delay of pattern embedding vector.
    weights : bool, optional (default: False)
        Whether to return array of variances of embedding vectors as weights.
    verbosity : int, optional (default: 0)
        Level of verbosity.

    Returns
    -------
    patt, patt_mask [, patt_time] : tuple of arrays
        Tuple of converted pattern array and new length
    """
    from scipy.misc import factorial

    array = array.astype('float64')

    if array_mask is not None:
        assert array_mask.dtype == 'int32'
    else:
        array_mask = np.zeros(array.shape, dtype='int32')


    if np.ndim(array) == 1:
        T = len(array)
        array = array.reshape(T, 1)
        array_mask = array_mask.reshape(T, 1)

    # Add noise to destroy ties...
    array += (1E-6 * array.std(axis=0)
              * np.random.rand(array.shape[0], array.shape[1]).astype('float64'))


    patt_time = int(array.shape[0] - step * (dim - 1))
    T, N = array.shape

    if dim <= 1 or patt_time <= 0:
        raise ValueError("Dim mist be > 1 and length of delay vector smaller "
                         "array length.")

    patt = np.zeros((patt_time, N), dtype='int32')
    weights_array = np.zeros((patt_time, N), dtype='float64')

    patt_mask = np.zeros((patt_time, N), dtype='int32')

    # Precompute factorial for c-code... patterns of dimension
    # larger than 10 are not supported
    fac = factorial(np.arange(10)).astype('int32')

    # _get_patterns assumes mask=0 to be a masked value
    array_mask = (array_mask == False).astype('int32')

    (patt, patt_mask, weights_array) = _get_patterns(array, array_mask, patt, patt_mask, weights_array, dim, step, fac, N, T)

    weights_array = np.asarray(weights_array)
    patt = np.asarray(patt)
    # Transform back to mask=1 implying a masked value
    patt_mask = np.asarray(patt_mask) == False

    if weights:
        return patt, patt_mask, patt_time, weights_array
    else:
        return patt, patt_mask, patt_time


def quantile_bin_array(data, bins=6):
    """Returns symbolified array with equal-quantile binning.

    Parameters
    ----------
    data : array
        Data array of shape (time, variables).
    bins : int, optional (default: 6)
        Number of bins.

    Returns
    -------
    symb_array : array
        Converted data of integer type.
    """
    T, N = data.shape

    # get the bin quantile steps
    bin_edge = int(np.ceil(T / float(bins)))

    symb_array = np.zeros((T, N), dtype='int32')

    # get the lower edges of the bins for every time series
    edges = np.sort(data, axis=0)[::bin_edge, :].T
    bins = edges.shape[1]

    # This gives the symbolic time series
    symb_array = (data.reshape(T, N, 1) >= edges.reshape(1, N, bins)).sum(
        axis=2) - 1

    return symb_array.astype('int32')

def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0, initial_values=None):
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with possibly more user-friendly input options.

    DEPRECATED. Will be removed in future.
    """
    print("data generating models are now in toymodels folder: "
         "from tigramite.toymodels import structural_causal_processes as toys.")
    return None

def structural_causal_process(links, T, noises=None, 
                        intervention=None, intervention_type='hard',
                        seed=None):
    """Returns a structural causal process with contemporaneous and lagged
    dependencies.

    DEPRECATED. Will be removed in future.
    """
    print("data generating models are now in toymodels folder: "
         "from tigramite.toymodels import structural_causal_processes as toys.")
    return None


if __name__ == '__main__':
    
    from tigramite.toymodels.structural_causal_processes import structural_causal_process
    ## Generate some time series from a structural causal process
    def lin_f(x): return x
    def nonlin_f(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    links = {0: [((0, -1), 0.9, lin_f)],
             1: [((1, -1), 0.8, lin_f), ((0, -1), 0.3, nonlin_f)],
             2: [((2, -1), 0.7, lin_f), ((1, 0), -0.2, lin_f)],
             }
    noises = [np.random.randn, np.random.randn, np.random.randn]
    data, nonstat = structural_causal_process(links,
     T=100, noises=noises)
    print(data.shape)

    frame = DataFrame(data)
