"""Tigramite data processing functions."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0


import numpy
import sys, warnings


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


def _var_network(graph,
                inv_inno_cov=None, inno_cov=None, use='inno_cov',
                T=100):
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
        the covariance.

    Parameters
    ----------
    graph : array
        Lagged connectivity matrices.

    inv_inno_cov : array, optional (default: None)
        Inverse covariance matrix of innovations.

    inno_cov : array, optional (default: None)
        Covariance matrix of innovations.

    use : str, optional (default: 'inno_cov')
        Specifier, either 'inno_cov' or 'inv_inno_cov'.

    T : int, optional (default: 100)
        Sample size.

    Returns
    -------
    X : array
        Array of realization.
    """
    N, N, P = graph.shape

    # Test stability
    stabmat = numpy.zeros((N * P, N * P))
    index = 0
    for i in range(0, N * P, N):
        stabmat[:N, i:i + N] = graph[:, :, index]
        if index < P - 1:
            stabmat[i + N:i + 2 * N, i:i + N] = numpy.identity(N)

        index += 1

    eig = numpy.linalg.eig(stabmat)[0]
    assert numpy.all(numpy.abs(eig) < 1.), "Nonstationary process!"

    X = numpy.random.randn(N, T)

    if use == 'inv_inno_cov' and inv_inno_cov is not None:
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

    for t in range(P, T):
        Xpast = numpy.repeat(
            X[:, t - P:t][:, ::-1].reshape(1, N, P), N, axis=0)
        X[:, t] = (Xpast * graph).sum(axis=2).sum(axis=1) + noise[t]

    return X.transpose()


def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0):
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with more user-friendly input options.

    Parameters
    ----------
    parents_neighbors_coeffs : dict 
        Dictionary of format {..., j:[((var1, lag1), coeff), ...], ...} for all
        variables where vars must be in [0..N-1] and lags <= 0 with number of
        variables N. Coeff refers to the coefficient in the linear model. If
        lag=0, a nonzero value in the covariance matrix (or its inverse) is
        implied. These should be the same for (i, j) and (j, i).

    use : str, optional (default: 'inv_inno_cov') 
        Specifier, either 'inno_cov' or 'inv_inno_cov'. If 'inno_cov', lag=0
        entries in parents_neighbors_coeffs are interpreted as entries of the
        innovation noise term's covariance matrix. I 'inv_inno_cov', they are
        interpreted as entries in the inverse covariance matrix.

    T : int, optional (default: 1000) 
        Sample size.

    verbosity : int, optional (default: 0)
        Level of verbosity.

    Returns
    -------
    X : array-like
        Array of realization.

    true_parents_neighbors : dict
        Dictionary of true parents and neighbors
    """

    max_lag = 0
    for j in list(parents_neighbors_coeffs):
        for node, coeff in parents_neighbors_coeffs[j]:
            i, tau = node[0], -node[1]
            max_lag = max(max_lag, abs(tau))
    N = max(list(parents_neighbors_coeffs)) + 1
    graph = numpy.zeros((N, N, max_lag + 1))

    innos = numpy.zeros((N, N))
    innos[range(N), range(N)] = 1.
    true_parents_neighbors = {}
    # print graph.shape
    for j in list(parents_neighbors_coeffs):
        true_parents_neighbors[j] = []
        for node, coeff in parents_neighbors_coeffs[j]:
            i, tau = node[0], -node[1]
            if coeff != 0.:
                true_parents_neighbors[j].append((i, -tau))
            if tau == 0:
                innos[j, i] = innos[i, j] = coeff
            else:
                graph[j, i, tau - 1] = coeff

    if verbosity > 0:
        print("VAR graph =\n%s" % str(graph))
        if use == 'inno_cov':
            print("\nInnovation Cov =\n%s" % str(innos))
        elif use == 'inv_inno_cov':
            print("\nInverse Innovation Cov =\n%s" % str(innos))

    data = _var_network(graph=graph, inv_inno_cov=innos,
                       inno_cov=innos,
                       use=use, T=T)

    return data, true_parents_neighbors


class _Logger(object):
    """Class to append print output to a string which can be saved"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = ""       # open("log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log += message  # .write(message)