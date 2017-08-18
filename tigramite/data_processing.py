"""Tigramite data processing functions."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0


from __future__ import print_function
from collections import defaultdict
import numpy
import sys, warnings


# TODO force usage of pandas DF, do not support own data frame...
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

    """
    def __init__(self, data,
                 mask = None, missing_flag=None):

        self.values = data
        self.mask = mask
        self.missing_flag = missing_flag
        T, N = data.shape

        # if type(self.values) != numpy.ndarray:
        #     raise TypeError("data is of type %s, " % type(self.values) +
        #                     "must be numpy.ndarray")
        if N > T:
            warnings.warn("data.shape = %s," % str(self.values.shape) +
                          " is it of shape (observations, variables) ?")
        # if numpy.isnan(data).sum() != 0:
        #     raise ValueError("NaNs in the data")

        if self.mask is not None:
            if self.mask.shape != self.values.shape:
                raise ValueError("Mask array must of same shape as data array")

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
    if numpy.ndim(data) == 1:
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
        window = numpy.exp(-(numpy.arange(totaltime).reshape((1, totaltime)) -
                             numpy.arange(totaltime).reshape((totaltime, 1))
                             ) ** 2 / ((2. * smooth_width / 2.) ** 2))
    elif kernel == 'heaviside':
        import scipy.linalg
        wtmp = numpy.zeros(totaltime)
        wtmp[:numpy.ceil(smooth_width / 2.)] = 1
        window = scipy.linalg.toeplitz(wtmp)

    if mask is None:
        if numpy.ndim(data) == 1:
            smoothed_data = (data * window).sum(axis=1) / window.sum(axis=1)
        else:
            smoothed_data = numpy.zeros(data.shape)
            for i in range(data.shape[1]):
                smoothed_data[:, i] = (
                    data[:, i] * window).sum(axis=1) / window.sum(axis=1)
    else:
        if numpy.ndim(data) == 1:
            smoothed_data = ((data * window * (mask==False)).sum(axis=1) /
                             (window * (mask==False)).sum(axis=1))
        else:
            smoothed_data = numpy.zeros(data.shape)
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

    values[numpy.isnan(values)] = 0.
    average = numpy.ma.average(values, axis=axis, weights=weights)

    variance = numpy.sum(weights * (values - numpy.expand_dims(average, axis)
                                    ) ** 2, axis=axis) / weights.sum(axis=axis)

    return (average, numpy.sqrt(variance))


def time_bin_with_mask(data, time_bin_length, sample_selector=None):
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

    if sample_selector is None:
        sample_selector = numpy.ones(data.shape)

    if numpy.ndim(data) == 1.:
        data.shape = (T, 1)
        sample_selector.shape = (T, 1)

    bindata = numpy.zeros(
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
    import scipy
    from scipy.misc import factorial

    # Import cython code
    try:
        import tigramite_cython_code
    except ImportError:
        raise ImportError("Could not import tigramite_cython_code, please"
                          " compile cython code first as described in Readme.")

    array = array.astype('float64')

    if array_mask is not None:
        assert array_mask.dtype == 'int32'
    else:
        array_mask = numpy.zeros(array.shape, dtype='int32')


    if numpy.ndim(array) == 1:
        T = len(array)
        array = array.reshape(T, 1)
        array_mask = array_mask.reshape(T, 1)

    # Add noise to destroy ties...
    array += (1E-6 * array.std(axis=0)
              * numpy.random.rand(array.shape[0], array.shape[1]).astype('float64'))


    patt_time = int(array.shape[0] - step * (dim - 1))
    T, N = array.shape

    if dim <= 1 or patt_time <= 0:
        raise ValueError("Dim mist be > 1 and length of delay vector smaller "
                         "array length.")

    patt = numpy.zeros((patt_time, N), dtype='int32')
    weights_array = numpy.zeros((patt_time, N), dtype='float64')

    patt_mask = numpy.zeros((patt_time, N), dtype='int32')

    # Precompute factorial for c-code... patterns of dimension
    # larger than 10 are not supported
    fac = factorial(numpy.arange(10)).astype('int32')

    # _get_patterns_cython assumes mask=0 to be a masked value
    array_mask = (array_mask == False).astype('int32')

    (patt, patt_mask, weights_array) = tigramite_cython_code._get_patterns_cython(
        array, array_mask, patt, patt_mask, weights_array, dim, step, fac, N,
        T)

    weights_array = numpy.asarray(weights_array)
    patt = numpy.asarray(patt)
    # Transform back to mask=1 implying a masked value
    patt_mask = numpy.asarray(patt_mask) == False

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
    bin_edge = int(numpy.ceil(T / float(bins)))

    symb_array = numpy.zeros((T, N), dtype='int32')

    # get the lower edges of the bins for every time series
    edges = numpy.sort(data, axis=0)[::bin_edge, :].T
    bins = edges.shape[1]

    # This gives the symbolic time series
    symb_array = (data.reshape(T, N, 1) >= edges.reshape(1, N, bins)).sum(
        axis=2) - 1

    return symb_array.astype('int32')

def _generate_noise(covar_matrix, use_inverse=False):
    """
    Generate a multivariate normal distribution using correlated innovations.

    Parameters
    ----------
    covar_matrix : array
        Covariance matrix of the random variables

    use_inverse : bool, optional
        Invert the covarience matrix before use

    Returns
    -------
    noise : array
        Random noise generated according to covar_matrix
    """
    if use_inverse == 'inv_inno_cov' and inv_inno_cov is not None:
        #TODO wrap in function
        mult = -numpy.ones((N, N))
        mult[numpy.diag_indices_from(mult)] = 1
        inv_inno_cov *= mult
        noise = numpy.random.multivariate_normal(
            mean=numpy.zeros(N),
            cov=numpy.linalg.inv(inv_inno_cov),
            size=T)
    elif use == 'inno_cov' and inno_cov is not None:
        noise = numpy.random.multivariate_normal(
            mean=numpy.zeros(N), cov=inno_cov, size=T)
    else:
        noise = numpy.random.randn(T, N)



def _var_network(graph,
                 inv_inno_cov=None,
                 inno_cov=None,
                 #TODO inconsistent default values with var_process
                 use='inno_cov',
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

    inv_inno_cov : array, optional (default: None)
        Inverse covariance matrix of innovations.

    inno_cov : array, optional (default: None)
        Covariance matrix of innovations.

    use : str, optional (default: 'inno_cov')
        Specifier, either 'inno_cov' or 'inv_inno_cov'.
        For debugging, 'no_inno' can also be specified, in which case random noise
        will be disabled.

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
    # TODO remove one N value..
    N, N, P = graph.shape

    # Test stability
    # TODO Sparse matrix...  this goes as (N*P)^2
    stabmat = numpy.zeros((N * P, N * P))
    index = 0
    # TODO Use enum instead of index..
    # TODO what is this
    # TODO wrap in function
    for i in range(0, N * P, N):
        stabmat[:N, i:i + N] = graph[:, :, index]
        if index < P - 1:
            stabmat[i + N:i + 2 * N, i:i + N] = numpy.identity(N)
        index += 1

    eig = numpy.linalg.eig(stabmat)[0]
    assert numpy.all(numpy.abs(eig) < 1.), "Nonstationary process!"

    # Generate the returned data
    X = numpy.random.randn(N, T)
    # Load the initial values
    if initial_values is not None:
        # Ensure it is a numpy array
        assert isinstance(initial_values, numpy.ndarray),\
            "User must provide initial_values as a numpy.ndarray"
        # Check the shape is correct
        assert initial_values.shape == X[:, :P].shape,\
            "Initial values must be of shape (n_nodes, max_delay+1)"+\
            "\n current shape : " + str(initial_values.shape)+\
            "\n desired shape : " + str(X[:, :P].shape)
        # Input the initial values
        X[:, :P] = initial_values

    if use == 'inv_inno_cov' and inv_inno_cov is not None:
        #TODO wrap in function
        mult = -numpy.ones((N, N))
        mult[numpy.diag_indices_from(mult)] = 1
        inv_inno_cov *= mult
        noise = numpy.random.multivariate_normal(
            mean=numpy.zeros(N),
            cov=numpy.linalg.inv(inv_inno_cov),
            size=T)
    elif use == 'inno_cov' and inno_cov is not None:
        noise = numpy.random.multivariate_normal(
            mean=numpy.zeros(N), cov=inno_cov, size=T)
    elif use == 'no_inno':
        noise = numpy.zeros((T, N))
    else:
        noise = numpy.random.randn(T, N)

    # TODO what is this
    # TODO further numpy usage may simplify this
    for t in range(P, T):
        Xpast = numpy.repeat(
            X[:, t - P:t][:, ::-1].reshape(1, N, P), N, axis=0)
        X[:, t] = (Xpast * graph).sum(axis=2).sum(axis=1) + noise[t]

    return X.transpose()

def _iter_parents_neighbours_coeffs(parents_neighbors_coeffs):
    """
    Iterator through the current parents_neighbours_coeffs structure.  Mainly to
    save repeated code and make it easier to change this structure.

    Parameters
    ----------
    parents_neighbors_coeffs : dict

        Dictionary of format {..., j:[(var1, lag1), (var2, lag2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N. If lag=0, a nonzero value in the covariance matrix (or
        its inverse) is implied. These should be the same for (i, j) and (j, i).

    Yields
    -------
    (current_node_id, parent_node_id, time_lag, coeff) : tuple
        Tuple defining the relationship between nodes across time
    """
    for j in list(parents_neighbors_coeffs):
        # Iterate over parent nodes and unpack node and coeff
        for (i, tau), coeff in parents_neighbors_coeffs[j]:
            # 
            yield j, i, tau, coeff


def _find_max_time_lag_and_node_id(parents_neighbors_coeffs):
    """
    Function to find the maximum time lag in the parent-neighbours-coefficients
    object, as well as the largest node ID

    Parameters
    ----------
    parents_neighbors_coeffs : dict

        Dictionary of format {..., j:[(var1, lag1), (var2, lag2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N. If lag=0, a nonzero value in the covariance matrix (or
        its inverse) is implied. These should be the same for (i, j) and (j, i).

    Returns
    -------
    (max_time_lag, max_node_id) : tuple
        Tuple of the maximum time lag and maximum node ID
    """
    # Default maximum lag
    max_time_lag = 0
    # Iterate through the keys in parents_neighbors_coeffs
    for j in list(parents_neighbors_coeffs):
        # Extract lag time from each node
        for node, _ in parents_neighbors_coeffs[j]:
            _, tau = node[0], node[1]
            # TODO is this correct?
            assert tau <= 0, \
                "All time lags must be given as non-positive values"
            # Find max lag time
            max_time_lag = max(max_time_lag, abs(tau))
    # Find largest node id
    max_node_id = max(list(parents_neighbors_coeffs)) + 1
    # Return these values
    return max_time_lag, max_node_id

def _get_true_parent_neighbour_dict(parents_neighbors_coeffs):
    """
    Function to return the dictionary of true parent neighbour causal 
    connections in time.

    Parameters
    ----------
    parents_neighbors_coeffs : dict

        Dictionary of format {..., j:[(var1, lag1), (var2, lag2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    true_parent_neighbour : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the 
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Initialize the returned dictionary
    true_parents_neighbors = defaultdict(list)
    for j in list(parents_neighbors_coeffs):
        # Iterate over parent nodes and unpack node and coeff
        for (i, tau), coeff in parents_neighbors_coeffs[j]:
            # Add parent node id and lag if non-zero coeff
            if coeff != 0.:
                true_parents_neighbors[j].append((i, tau))
    # Return the true relations
    return true_parents_neighbors

def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0, initial_values=None):
    #TODO docstring looks wrong about dict input format
    #TODO docstring is wrong about the output, optional output can be used
    #TODO j: [var1, lag1, coeff] is a better format
    #TODO sparse array of j->[var1, lag1, coeff]
    #TODO normal array of j->[var1, lag1, coeff], forcing j to be contiguous from 0
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with possibly more user-friendly input options.

    Parameters
    ----------
    parents_neighbors_coeffs : dict

        Dictionary of format {..., j:[(var1, lag1), (var2, lag2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N. If lag=0, a nonzero value in the covariance matrix (or
        its inverse) is implied. These should be the same for (i, j) and (j, i).

    use : str, optional (default: 'inv_inno_cov')
        Specifier, either 'inno_cov' or 'inv_inno_cov'.
        For debugging, 'no_inno' can also be specified, in which case random noise
        will be disabled.

    T : int, optional (default: 1000)
        Sample size.

    verbosity : int, optional (default: 0)
        Level of verbosity.

    initial_values : array, optional (default: None)
        Initial values for each node. Shape must be (N, max_delay+1)

    Returns
    -------
    X : array-like
        Array of realization.
    """
    # Find the maximum node ID and time lag
    max_time_lag, max_node_id = \
            _find_max_time_lag_and_node_id(parents_neighbors_coeffs)
    N = max_node_id
    # Generate the true parent neighbours graph
    true_parents_neighbors = \
            _get_true_parent_neighbour_dict(parents_neighbors_coeffs)

    # Initialize full time graph
    # TODO scipy sparse could be useful here
    graph = numpy.zeros((N, N, max_time_lag + 1))
    # TODO use numpy identity
    innos = numpy.zeros((N, N))
    innos[range(N), range(N)] = 1.
    # print graph.shape
    for j in list(parents_neighbors_coeffs):
        # Initialize the list of true parents for each node
        true_parents_neighbors[j] = []
        # Iterate over parent nodes and unpack node and coeff
        for node, coeff in parents_neighbors_coeffs[j]:
            i, tau = node[0], -node[1]
            # Add to innos  if zero lag
            if tau == 0:
                innos[j, i] = innos[i, j] = coeff
            # Otherwise add to graph
            else:
                graph[j, i, tau - 1] = coeff

    if verbosity > 0:
        print("VAR graph =\n%s" % str(graph))
        if use == 'inno_cov':
            print("\nInnovation Cov =\n%s" % str(innos))
        elif use == 'inv_inno_cov':
            print("\nInverse Innovation Cov =\n%s" % str(innos))
        elif use == 'no_inno':
            print("\nNo random noise will be applied!\n")

    data = _var_network(graph=graph,
                        inv_inno_cov=innos,
                        inno_cov=innos,
                        use=use,
                        initial_values=initial_values,
                        T=T)

    return data, true_parents_neighbors


class _Logger(object):
    """Class to append print output to a string which can be saved"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = ""       # open("log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log += message  # .write(message)
