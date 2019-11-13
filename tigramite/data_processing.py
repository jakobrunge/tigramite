"""Tigramite data processing functions."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0
from __future__ import print_function
from collections import defaultdict, OrderedDict
import sys
import warnings
import copy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class DataFrame():
    """Data object containing time series array and optional mask.

    Alternatively, a panda dataframe can be used.

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
    """
    def __init__(self, data, mask=None, missing_flag=None, var_names=None,
        datatime=None):

        self.values = data
        self.mask = mask
        self.missing_flag = missing_flag
        T, N = data.shape
        # Set the variable names
        self.var_names = var_names
        # Set the default variable names if none are set
        if self.var_names is None:
            self.var_names = {i: i for i in range(N)}

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
                                 " but mask.shape = %s,"
                                 % str(_use_mask.shape)) + \
                                 "must identical"

    def construct_array(self, X, Y, Z, tau_max,
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
        X, Y, Z : list of tuples
            For a dependence measure I(X;Y|Z), Y is of the form [(varY, 0)],
            where var specifies the variable index. X typically is of the form
            [(varX, -tau)] with tau denoting the time lag and Z can be
            multivariate [(var1, -lag), (var2, -lag), ...] .

        tau_max : int
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X and Z all have the same sample size.

        mask : array-like, optional (default: None)
            Optional mask array, must be of same shape as data.  If it is set,
            then it overrides the self.mask assigned to the dataframe. If it is
            None, then the self.mask is used, if it exists.

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
        array, xyz [,XYZ] : Tuple of data array of shape (dim, T) and xyz
            identifier array of shape (dim,) identifying which row in array
            corresponds to X, Y, and Z. For example::
                X = [(0, -1)], Y = [(1, 0)], Z = [(1, -1), (0, -2)]
                yields an array of shape (5, T) and xyz is
                xyz = numpy.array([0,1,2,2])
            If return_cleaned_xyz is True, also outputs the cleaned XYZ lists.
        """
        # Get the length in time and the number of nodes
        T, N = self.values.shape

        # Remove duplicates in X, Y, Z
        X = list(OrderedDict.fromkeys(X))
        Y = list(OrderedDict.fromkeys(Y))
        Z = list(OrderedDict.fromkeys(Z))

        # If a node in Z occurs already in X or Y, remove it from Z
        Z = [node for node in Z if (node not in X) and (node not in Y)]

        # Check that all lags are non-positive and indices are in [0,N-1]
        XYZ = X + Y + Z
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
                      'z' : 2}
        xyz = np.array([index_code[name]
                        for var, name in zip([X, Y, Z], ['x', 'y', 'z'])
                        for _ in var])

        # Setup and fill array with lagged time series
        time_length = T - max_lag
        array = np.zeros((dim, time_length), dtype=self.values.dtype)
        # Note, lags are negative here
        for i, (var, lag) in enumerate(XYZ):
            array[i, :] = self.values[max_lag + lag:T + lag, var]

        # Choose which indices to use
        use_indices = np.ones(time_length, dtype='int')

        # Remove all values that have missing value flag, as well as the time
        # slices that occur up to max_lag after
        if self.missing_flag is not None:
            missing_anywhere = np.any(self.values == self.missing_flag, axis=1)
            for tau in range(max_lag+1):
                use_indices[missing_anywhere[tau:T-max_lag+tau]] = 0

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
                array_mask[i, :] = (_use_mask[max_lag + lag: T + lag, var] == False)
            # Iterate over defined mapping from letter index to number index,
            # i.e. 'x' -> 0, 'y' -> 1, 'z'-> 2
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
            self.print_array_info(array, X, Y, Z, self.missing_flag, mask_type)

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
        if np.all(np.array(Y)[:, 1] != 0):
            raise ValueError("Y-nodes are %s, " % str(Y) +
                             "but one of the Y-nodes must have zero lag")

    def print_array_info(self, array, X, Y, Z, missing_flag, mask_type):
        """
        Print info about the constructed array

        Parameters
        ----------
        array : Data array of shape (dim, T)

        X, Y, Z : list of tuples
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
            measure I(X; Y | Z) the samples should be masked. If None, 'y' is
            used, which excludes all time slices containing masked samples in Y.
            Explained in [1]_.
        """
        indt = " " * 12
        print(indt + "Constructed array of shape %s from"%str(array.shape) +
              "\n" + indt + "X = %s" % str(X) +
              "\n" + indt + "Y = %s" % str(Y) +
              "\n" + indt + "Z = %s" % str(Z))
        if self.mask is not None:
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

    # Import cython code
    try:
        import tigramite.tigramite_cython_code as tigramite_cython_code
    except ImportError:
        raise ImportError("Could not import tigramite_cython_code, please"
                          " compile cython code first as described in Readme.")

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

    # _get_patterns_cython assumes mask=0 to be a masked value
    array_mask = (array_mask == False).astype('int32')

    (patt, patt_mask, weights_array) = \
            tigramite_cython_code._get_patterns_cython(array, array_mask,
                                                       patt, patt_mask,
                                                       weights_array, dim,
                                                       step, fac, N, T)

    weights_array = np.asarray(weights_array)
    patt = np.asarray(patt)
    # Transform back to mask=1 implying a masked value
    patt_mask = np.asarray(patt_mask) == False

    if weights:
        return (patt, patt_mask, patt_time, weights_array)
    else:
        return (patt, patt_mask, patt_time)


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

def _generate_noise(covar_matrix, time=1000, use_inverse=False):
    """
    Generate a multivariate normal distribution using correlated innovations.

    Parameters
    ----------
    covar_matrix : array
        Covariance matrix of the random variables

    time : int
        Sample size

    use_inverse : bool, optional
        Negate the off-diagonal elements and invert the covariance matrix
        before use

    Returns
    -------
    noise : array
        Random noise generated according to covar_matrix
    """
    # Pull out the number of nodes from the shape of the covar_matrix
    n_nodes = covar_matrix.shape[0]
    # Make a deep copy for use in the inverse case
    this_covar = covar_matrix
    # Take the negative inverse if needed
    if use_inverse:
        this_covar = copy.deepcopy(covar_matrix)
        this_covar *= -1
        this_covar[np.diag_indices_from(this_covar)] *= -1
        this_covar = np.linalg.inv(this_covar)
    # Return the noise distribution
    return np.random.multivariate_normal(mean=np.zeros(n_nodes),
                                            cov=this_covar,
                                            size=time)

def _check_stability(graph):
    """
    Raises an AssertionError if the input graph corresponds to a non-stationary
    process.

    Parameters
    ----------
    graph : array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    """
    # Get the shape from the input graph
    n_nodes, _, period = graph.shape
    # Set the top section as the horizontally stacked matrix of
    # shape (n_nodes, n_nodes * period)
    stability_matrix = \
        scipy.sparse.hstack([scipy.sparse.lil_matrix(graph[:, :, t_slice])
                             for t_slice in range(period)])
    # Extend an identity matrix of shape
    # (n_nodes * (period - 1), n_nodes * (period - 1)) to shape
    # (n_nodes * (period - 1), n_nodes * period) and stack the top section on
    # top to make the stability matrix of shape
    # (n_nodes * period, n_nodes * period)
    stability_matrix = \
        scipy.sparse.vstack([stability_matrix,
                             scipy.sparse.eye(n_nodes * (period - 1),
                                              n_nodes * period)])
    # Check the number of dimensions to see if we can afford to use a dense
    # matrix
    n_eigs = stability_matrix.shape[0]
    if n_eigs <= 25:
        # If it is relatively low in dimensionality, use a dense array
        stability_matrix = stability_matrix.todense()
        eigen_values, _ = scipy.linalg.eig(stability_matrix)
    else:
        # If it is a large dimensionality, convert to a compressed row sorted
        # matrix, as it may be easier for the linear algebra package
        stability_matrix = stability_matrix.tocsr()
        # Get the eigen values of the stability matrix
        eigen_values = scipy.sparse.linalg.eigs(stability_matrix,
                                                k=(n_eigs - 2),
                                                return_eigenvectors=False)
    # Ensure they all have less than one magnitude
    assert np.all(np.abs(eigen_values) < 1.), \
        "Values given by time lagged connectivity matrix corresponds to a "+\
        " non-stationary process!"

def _check_initial_values(initial_values, shape):
    """
    Raises a AssertionError if the input initial values:
        * Are not a numpy array OR
        * Do not have the shape (n_nodes, max_delay+1)

    Parameters
    ----------
    graph : array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    """
    # Ensure it is a numpy array
    assert isinstance(initial_values, np.ndarray),\
        "User must provide initial_values as a numpy.ndarray"
    # Check the shape is correct
    assert initial_values.shape == shape,\
        "Initial values must be of shape (n_nodes, max_delay+1)"+\
        "\n current shape : " + str(initial_values.shape)+\
        "\n desired shape : " + str(shape)

def _var_network(graph,
                 add_noise=True,
                 inno_cov=None,
                 invert_inno=False,
                 T=100,
                 initial_values=None):
    """Returns a vector-autoregressive process with correlated innovations.

    Useful for testing.

    Example:
        graph=numpy.array([[[0.2,0.,0.],[0.5,0.,0.]],
                           [[0.,0.1,0. ],[0.3,0.,0.]]])

        represents a process

        X_1(t) = 0.2 X_1(t-1) + 0.5 X_2(t-1) + eps_1(t)
        X_2(t) = 0.3 X_2(t-1) + 0.1 X_1(t-2) + eps_2(t)

        with inv_inno_cov being the negative (except for diagonal) inverse
        covariance matrix of (eps_1(t), eps_2(t)) OR inno_cov being
        the covariance. Initial values can also be provided.


    Parameters
    ----------
    graph : array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)

    add_noise : bool, optional (default: True)
        Flag to add random noise or not

    inno_cov : array, optional (default: None)
        Covariance matrix of innovations.

    invert_inno : bool, optional (defualt : False)
        Flag to negate off-diagonal elements of inno_cov and invert it before
        using it as the covariance matrix of innovations

    T : int, optional (default: 100)
        Sample size.

    initial_values : array, optional (defult: None)
        Initial values for each node. Shape is (n_nodes, max_delay+1), i.e. must
        be of shape (graph.shape[1], graph.shape[2]).

    Returns
    -------
    X : array
        Array of realization.
    """
    n_nodes, _, period = graph.shape

    time = T
    # Test stability
    _check_stability(graph)

    # Generate the returned data
    data = np.random.randn(n_nodes, time)
    # Load the initial values
    if initial_values is not None:
        # Check the shape of the initial values
        _check_initial_values(initial_values, data[:, :period].shape)
        # Input the initial values
        data[:, :period] = initial_values

    # Check if we are adding noise
    noise = None
    if add_noise:
        # Use inno_cov if it was provided
        if inno_cov is not None:
            noise = _generate_noise(inno_cov,
                                    time=time,
                                    use_inverse=invert_inno)
        # Otherwise just use uncorrelated random noise
        else:
            noise = np.random.randn(time, n_nodes)

    for a_time in range(period, time):
        data_past = np.repeat(
            data[:, a_time-period:a_time][:, ::-1].reshape(1, n_nodes, period),
            n_nodes, axis=0)
        data[:, a_time] = (data_past*graph).sum(axis=2).sum(axis=1)
        if add_noise:
            data[:, a_time] += noise[a_time]

    return data.transpose()

def _iter_coeffs(parents_neighbors_coeffs):
    """
    Iterator through the current parents_neighbors_coeffs structure.  Mainly to
    save repeated code and make it easier to change this structure.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Yields
    -------
    (node_id, parent_id, time_lag, coeff) : tuple
        Tuple defining the relationship between nodes across time
    """
    # Iterate through all defined nodes
    for node_id in list(parents_neighbors_coeffs):
        # Iterate over parent nodes and unpack node and coeff
        for (parent_id, time_lag), coeff in parents_neighbors_coeffs[node_id]:
            # Yield the entry
            yield node_id, parent_id, time_lag, coeff

def _check_parent_neighbor(parents_neighbors_coeffs):
    """
    Checks to insure input parent-neighbor connectivity input is sane.  This
    means that:
        * all time lags are non-positive
        * all parent nodes are included as nodes themselves
        * all node indexing is contiguous
        * all node indexing starts from zero
    Raises a ValueError if any one of these conditions are not met.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.
    """
    # Initialize some lists for checking later
    all_nodes = set()
    all_parents = set()
    # Iterate through variables
    for j in list(parents_neighbors_coeffs):
        # Cache all node ids to ensure they are contiguous
        all_nodes.add(j)
    # Iterate through all nodes
    for j, i, tau, _ in _iter_coeffs(parents_neighbors_coeffs):
        # Check all time lags are equal to or less than zero
        if tau > 0:
            raise ValueError("Lag between parent {} and node {}".format(i, j)+\
                             " is {} > 0, must be <= 0!".format(tau))
        # Cache all parent ids to ensure they are mentioned as node ids
        all_parents.add(i)
    # Check that all nodes are contiguous from zero
    all_nodes_list = sorted(list(all_nodes))
    if all_nodes_list != list(range(len(all_nodes_list))):
        raise ValueError("Node IDs in input dictionary must be contiguous"+\
                         " and start from zero!\n"+\
                         " Found IDs : [" +\
                         ",".join(map(str, all_nodes_list))+ "]")
    # Check that all parent nodes are mentioned as a node ID
    if not all_parents.issubset(all_nodes):
        missing_nodes = sorted(list(all_parents - all_nodes))
        all_parents_list = sorted(list(all_parents))
        raise ValueError("Parent IDs in input dictionary must also be in set"+\
                         " of node IDs."+\
                         "\n Parent IDs "+" ".join(map(str, all_parents_list))+\
                         "\n Node IDs "+" ".join(map(str, all_nodes_list)) +\
                         "\n Missing IDs " + " ".join(map(str, missing_nodes)))

def _check_symmetric_relations(a_matrix):
    """
    Check if the argument matrix is symmetric.  Raise a value error with details
    about the offending elements if it is not.  This is useful for checking the
    instantaneously linked nodes have the same link strength.

    Parameters
    ----------
    a_matrix : 2D numpy array
        Relationships between nodes at tau = 0. Indexed such that first index is
        node and second is parent, i.e. node j with parent i has strength
        a_matrix[j,i]
    """
    # Check it is symmetric
    if not np.allclose(a_matrix, a_matrix.T, rtol=1e-10, atol=1e-10):
        # Store the disagreement elements
        bad_elems = ~np.isclose(a_matrix, a_matrix.T, rtol=1e-10, atol=1e-10)
        bad_idxs = np.argwhere(bad_elems)
        error_message = ""
        for node, parent in bad_idxs:
            # Check that we haven't already printed about this pair
            if bad_elems[node, parent]:
                error_message += \
                    "Parent {:d} of node {:d}".format(parent, node)+\
                    " has coefficient {:f}.\n".format(a_matrix[node, parent])+\
                    "Parent {:d} of node {:d}".format(node, parent)+\
                    " has coefficient {:f}.\n".format(a_matrix[parent, node])
            # Check if we already printed about this one
            bad_elems[node, parent] = False
            bad_elems[parent, node] = False
        raise ValueError("Relationships between nodes at tau=0 are not"+\
                         " symmetric!\n"+error_message)

def _find_max_time_lag_and_node_id(parents_neighbors_coeffs):
    """
    Function to find the maximum time lag in the parent-neighbors-coefficients
    object, as well as the largest node ID

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    (max_time_lag, max_node_id) : tuple
        Tuple of the maximum time lag and maximum node ID
    """
    # Default maximum lag and node ID
    max_time_lag = 0
    max_node_id = 0
    # Iterate through the keys in parents_neighbors_coeffs
    for j, _, tau, _ in _iter_coeffs(parents_neighbors_coeffs):
        # Find max lag time
        max_time_lag = max(max_time_lag, abs(tau))
        # Find the max node ID
        max_node_id = max(max_node_id, j)
    # Return these values
    return max_time_lag, max_node_id

def _get_true_parent_neighbor_dict(parents_neighbors_coeffs):
    """
    Function to return the dictionary of true parent neighbor causal
    connections in time.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    true_parent_neighbor : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Initialize the returned dictionary of lists
    true_parents_neighbors = defaultdict(list)
    for j, i, tau, coeff in _iter_coeffs(parents_neighbors_coeffs):
        # Add parent node id and lag if non-zero coeff
        if coeff != 0.:
            true_parents_neighbors[j].append((i, tau))
    # Return the true relations
    return true_parents_neighbors

def _get_covariance_matrix(parents_neighbors_coeffs):
    """
    Determines the covariance matrix for correlated innovations

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    covar_matrix : numpy array
        Covariance matrix implied by the parents_neighbors_coeffs.  Used to
        generate correlated innovations.
    """
    # Get the total number of nodes
    _, max_node_id = \
            _find_max_time_lag_and_node_id(parents_neighbors_coeffs)
    n_nodes = max_node_id + 1
    # Initialize the covariance matrix
    covar_matrix = np.identity(n_nodes)
    # Iterate through all the node connections
    for j, i, tau, coeff in _iter_coeffs(parents_neighbors_coeffs):
        # Add to covar_matrix if node connection is instantaneous
        if tau == 0:
            covar_matrix[j, i] = coeff
    return covar_matrix

def _get_lag_connect_matrix(parents_neighbors_coeffs):
    """
    Generates the lagged connectivity matrix from a parent-neighbor
    connectivity dictionary.  Used to generate the input for _var_network

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    connect_matrix : numpy array
        Lagged connectivity matrix. Shape is (n_nodes, n_nodes, max_delay+1)
    """
    # Get the total number of nodes and time lag
    max_time_lag, max_node_id = \
            _find_max_time_lag_and_node_id(parents_neighbors_coeffs)
    n_nodes = max_node_id + 1
    n_times = max_time_lag + 1
    # Initialize full time graph
    connect_matrix = np.zeros((n_nodes, n_nodes, n_times))
    for j, i, tau, coeff in _iter_coeffs(parents_neighbors_coeffs):
        # If there is a non-zero time lag, add the connection to the matrix
        if tau != 0:
            connect_matrix[j, i, -(tau+1)] = coeff
    # Return the connectivity matrix
    return connect_matrix

def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0, initial_values=None):
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with possibly more user-friendly input options.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
            {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...}
        for all variables where vars must be in [0..N-1] and lags <= 0 with
        number of variables N. If lag=0, a nonzero value in the covariance
        matrix (or its inverse) is implied. These should be the same for (i, j)
        and (j, i).

    use : str, optional (default: 'inv_inno_cov')
        Specifier, either 'inno_cov' or 'inv_inno_cov'.
        Any other specifier will result in non-correlated noise.
        For debugging, 'no_noise' can also be specified, in which case random
        noise will be disabled.

    T : int, optional (default: 1000)
        Sample size.

    verbosity : int, optional (default: 0)
        Level of verbosity.

    initial_values : array, optional (default: None)
        Initial values for each node. Shape must be (N, max_delay+1)

    Returns
    -------
    data : array-like
        Data generated from this process
    true_parent_neighbor : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Check the input parents_neighbors_coeffs dictionary for sanity
    _check_parent_neighbor(parents_neighbors_coeffs)
    # Generate the true parent neighbors graph
    true_parents_neighbors = \
        _get_true_parent_neighbor_dict(parents_neighbors_coeffs)
    # Generate the correlated innovations
    innos = _get_covariance_matrix(parents_neighbors_coeffs)
    # Generate the lagged connectivity matrix for _var_network
    connect_matrix = _get_lag_connect_matrix(parents_neighbors_coeffs)
    # Default values as per 'inno_cov'
    add_noise = True
    invert_inno = False
    # Use the correlated innovations
    if use == 'inno_cov':
        if verbosity > 0:
            print("\nInnovation Cov =\n%s" % str(innos))
    # Use the inverted correlated innovations
    elif use == 'inv_inno_cov':
        invert_inno = True
        if verbosity > 0:
            print("\nInverse Innovation Cov =\n%s" % str(innos))
    # Do not use any noise
    elif use == 'no_noise':
        add_noise = False
        if verbosity > 0:
            print("\nInverse Innovation Cov =\n%s" % str(innos))
    # Use decorrelated noise
    else:
        innos = None
    # Ensure the innovation matrix is symmetric if it is used
    if (innos is not None) and add_noise:
        _check_symmetric_relations(innos)
    # Generate the data using _var_network
    data = _var_network(graph=connect_matrix,
                        add_noise=add_noise,
                        inno_cov=innos,
                        invert_inno=invert_inno,
                        T=T,
                        initial_values=initial_values)
    # Return the data
    return data, true_parents_neighbors

class _Logger(object):
    """Class to append print output to a string which can be saved"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = ""       # open("log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log += message  # .write(message)
