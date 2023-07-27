"""Tigramite toymodels."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0
from __future__ import print_function
import sys
import warnings
import copy
import math
import numpy as np

from tigramite.toymodels import structural_causal_processes as toys



def generate_linear_model_from_data(dataframe, parents, tau_max, realizations=100, 
                generate_noise_from='covariance',  
                T_data = None,
                model_params=None,
                data_transform=None,
                mask_type='y',
                boot_blocklength=1,
                seed=42,
                verbosity=0):
    """
    Fits a (contemporaneous and lagged) linear SCM to data, computes
    residuals, and then generates surrogate realizations with noise drawn
    with replacement from residuals or from a multivariate normal.
    
    Parameters
    ----------
    generate_noise_from : {'covariance', 'residuals'}
        Whether to generate the noise from a gaussian with same mean and covariance
        as residuals, or by drawing (with replacement) from the residuals.
    boot_blocklength : int, or in {'cube_root', 'from_autocorrelation'}
        Block length for block-bootstrap, which only applies to
        generate_noise_from='residuals'. If 'from_autocorrelation', the block
        length is determined from the decay of the autocovariance and
        if 'cube_root' it is the cube root of the time series length.
    seed : int, optional(default = None)
        Seed for RandomState (default_rng)
    """

    from tigramite.models import Models
    from sklearn.linear_model import LinearRegression

    assert dataframe.analysis_mode == 'single'

    random_state = np.random.default_rng(seed)

    def lin_f(x): return x

    if model_params is None:
        model_params = {}

    N = dataframe.N
    T = dataframe.T[0]
    if T_data is None:
        T_data = T

    ## Estimate noise covariance matrix of residuals
    model = Models(dataframe=dataframe,
                     model=LinearRegression(**model_params),
                     data_transform=data_transform,
                     mask_type='y',
                     verbosity=0)

    model.fit_full_model(all_parents=parents, tau_max=tau_max, return_data=True)

    links_coeffs = {}
    for j in range(N):
        links_coeffs[j] = []
        if len(parents[j]) > 0:
            for ipar, par in enumerate(parents[j]):
                links_coeffs[j].append(((par[0], int(par[1])), model.fit_results[j]['model'].coef_[ipar], lin_f))
                if verbosity > 0:
                    print(j, ((par[0], int(par[1])), np.round(model.fit_results[j]['model'].coef_[ipar], 2),) )

    pred = model.predict_full_model(
                new_data=None,
                pred_params=None,
                cut_off='max_lag_or_tau_max')

    # Computes cov and mean, but internally also the residuals needed here
    cov, mean = model.get_residuals_cov_mean(new_data=None,
                pred_params=None,)

    if generate_noise_from == 'covariance':
        # cov = np.cov(overlapping_residuals, rowvar=0)
        # mean = np.mean(overlapping_residuals, axis=0)   # residuals should have zero mean due to prediction including constant
        if verbosity > 0:
            print('covariance')
            print(np.round(cov, 2))
            print('mean')
            print(np.round(mean, 2))

    overlapping_residuals = model.residuals
    len_residuals = len(overlapping_residuals)

    ## Construct linear Gaussian structural causal model with this noise structure and generate many realizations with same sample size as data
    transient_fraction = 0.2
    size = T_data + int(math.floor(transient_fraction*T_data))
    # datasets = {}

    if generate_noise_from == 'residuals':
        if boot_blocklength == 'cube_root':
            boot_blocklength = max(1, int(T_data**(1/3)))
        elif boot_blocklength == 'from_autocorrelation':
            boot_blocklength = \
                get_block_length(overlapping_residuals.T, xyz=np.zeros(N), mode='confidence')
        elif type(boot_blocklength) is int and boot_blocklength > 0:
            pass
        else:
            raise ValueError("boot_blocklength must be integer > 0, 'cube_root', or 'from_autocorrelation'")

        # Determine the number of blocks total, rounding up for non-integer
        # amounts
        n_blks = int(math.ceil(float(size) / boot_blocklength))
        if n_blks < 10:
            raise ValueError("Only %d block(s) for block-sampling,"  %n_blks +
                             "choose smaller boot_blocklength!")

    for r in range(realizations):
        if generate_noise_from == 'covariance':
            noises = random_state.multivariate_normal(mean=mean, cov=cov, size=size)
        elif generate_noise_from == 'residuals':

            # Get the starting indices for the blocks
            blk_strt = random_state.choice(np.arange(len_residuals - boot_blocklength), 
                size=n_blks, replace=True)

            # Get the empty array of block resampled values
            boot_draw = np.zeros(n_blks*boot_blocklength, dtype='int')
            # Fill the array of block resamples
            for i in range(boot_blocklength):
                boot_draw[i::boot_blocklength] = np.arange(size)[blk_strt + i]
            # Cut to proper length
            draw = boot_draw[:size]
            # draw = np.random.randint(0, len(overlapping_residuals), size)    
            noises = overlapping_residuals[draw]

        else: raise ValueError("generate_noise_from has to be either 'covariance' or 'residuals'")

        dataset = toys.structural_causal_process(links=links_coeffs, noises=noises, T=T_data, 
                                                     transient_fraction=transient_fraction)[0]
        if np.any(np.isinf(dataset)):
            raise ValueError("Infinite data")

        yield dataset

    # return self   #datasets


def get_acf(series, max_lag=None):
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
        autocorr[lag] = np.corrcoef(y1_vals, y2_vals)[0, 1]
    return autocorr

def get_block_length(array, xyz, mode):
    """Returns optimal block length for significance and confidence tests.

    Determine block length using approach in Mader (2013) [Eq. (6)] which
    improves the method of Pfeifer (2005) with non-overlapping blocks In
    case of multidimensional X, the max is used. Further details in [1]_.
    Two modes are available. For mode='significance', only the indices
    corresponding to X are shuffled in array. For mode='confidence' all
    variables are jointly shuffled. If the autocorrelation curve fit fails,
    a block length of 5% of T is used. The block length is limited to a
    maximum of 10% of T.

    Mader et al., Journal of Neuroscience Methods,
    Volume 219, Issue 2, 15 October 2013, Pages 285-291

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
        autocov = get_acf(series=array[i], max_lag=max_lag)
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
            # Formula of Mader (2013) assuming non-overlapping blocks
            l_opt = (4. * T * (phi / (1. - phi) + phi**2 / (1. - phi)**2)**2
                     / (1. + 2. * phi / (1. - phi))**2)**(1. / 3.)
            block_len = max(block_len, int(l_opt))

            # pylab.plot(np.arange(0, max_lag+1), hilbert)
            # pylab.plot(np.arange(0, max_lag+1), func(np.arange(0, max_lag+1), popt[0], popt[1]))
            # print("block_len ", block_len, int(l_opt))
            # pylab.show()
        except RuntimeError:
            warnings.warn("Error - curve_fit failed for estimating block_shuffle length, using"
                  " block_len = %d" % (int(.05 * T)))
            # block_len = max(int(.05 * T), block_len)

    # print("chosen ", block_len)
    # Limit block length to a maximum of 10% of T
    block_len = min(block_len, int(0.1 * T))
    # print("chosen ", block_len)

    return block_len


if __name__ == '__main__':


    import tigramite
    from tigramite import data_processing as pp

    lin_f = lambda x: x
    links_coeffs = {0: [((0, -1), 0.98, lin_f), ((0, -2), -0.7, lin_f)],
                    1: [((1, -1), 0.9, lin_f), ((0, -1), 0.3, lin_f)],
                    2: [((2, -1), 0.9, lin_f), ((0, -2), -0.5, lin_f)],
                    3: [((3, -1), 0.9, lin_f)], #, ((4, -1), 0.4, lin_f)],
                    4: [((4, -1), 0.9, lin_f), ((3, 0), 0.5, lin_f)], #, ((3, -1), 0.3, lin_f)],
                    }
    T = 50     # time series length
    # Make some noise with different variance, alternatively just noises=None
    noises = None  # np.array([(1. + 0.2*float(j))*np.random.randn((T + int(math.floor(0.2*T)))) 
                       # for j in range(len(links_coeffs))]).T

    data, _ = toys.structural_causal_process(links_coeffs, T=T, noises=noises)
    T, N = data.shape

    # For generality, we include some masking
    mask = np.zeros(data.shape, dtype='int')
    mask[:int(T/2),0] = True
    # mask[int(T/2)+30:,1] = True
    # Create some missing samples at different time points
    data[11,0] = 9999.
    data[22,2] = 9999.
    data[33,3] = 9999.

    tau_max = 4
    # Initialize dataframe object, specify time axis and variable names
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$']
    dataframe = pp.DataFrame(data, 
                             mask=mask,
                             missing_flag = 9999.,
                             datatime = {0:np.arange(len(data))}, 
                             var_names=var_names)
    parents = {}
    for j in links_coeffs:
        parents[j] = []
        for par in links_coeffs[j]:
            parents[j].append(par[0])
    print(parents)
    datasets = list(generate_linear_model_from_data(dataframe, parents=parents, 
                tau_max=tau_max, realizations=100, 
                generate_noise_from='residuals',
                boot_blocklength=3, #'from_autocorrelation',
                verbosity=0))
    print(datasets[0].shape)
