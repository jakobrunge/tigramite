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
                boot_meanblocklength=1,
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
    boot_meanblocklength : float, or in {'cube_root','from_autocorrelation'}
        Mean block length for the stationary block-bootstrap. If 'cube_root' it is
        the cube root of the time series length. If 'from_autocorrelation', the 
        mean block length is determined from the decay of the autocorrelation
        as described in Politis and White (2004) and Patton et al. (2009).
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
        if boot_meanblocklength == 'cube_root':
            boot_meanblocklength = max(1,(T_data**(1/3)))
        elif boot_meanblocklength == 'from_autocorrelation':
            boot_meanblocklength = get_mean_block_length(overlapping_residuals.T)
            # Take maximum mean block length, corresponding to most persistent variable #
            boot_meanblocklength = np.max(boot_meanblocklength)
        elif type(boot_meanblocklength) in [int,float] and boot_meanblocklength >= 1:
            pass
        else:
            raise ValueError("boot_meanblocklength must be integer or float >= 1, 'cube_root', or 'from_autocorrelation'")
        # Probability of new block at each index #
        pnewblk = 1.0/float(boot_meanblocklength)

    for r in range(realizations):
        if generate_noise_from == 'covariance':
            noises = random_state.multivariate_normal(mean=mean, cov=cov, size=size)
        elif generate_noise_from == 'residuals':
            # Randomly sample the length of each block #
            blkslen = random_state.geometric(pnewblk,size=size+1)
            blkslen = blkslen[0:np.where(
                np.cumsum(blkslen)>size)[0][0]+1] #sum of block lengths cut to proper length
            blkslen[-1] = blkslen[-1]-(np.sum(blkslen)-size) #truncate last block to match proper length
            # Get the starting indices for the blocks #
            blk_strt = random_state.choice(np.arange(len_residuals),len(blkslen),replace=True)
            # Create the random sequence of indices #
            boot_draw = np.concatenate([np.arange(blk_strt[idx],blk_strt[idx]+blkslen[idx])
                                        for idx in range(len(blkslen))]) #the resampled indices
            boot_draw = boot_draw%len_residuals #wrap around (Politis and Romero, 1994 below Eq.(1))
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

def get_mean_block_length(array):
    """Returns optimal mean block length for significance and confidence tests.

    Determine mean block length for stationary bootstrap using approach in
    Politis and White (2004) with correction from Patton et al. (2009).
    The mean block length for each variable is output.
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

    Returns
    -------
    bsbhat : array of floats
        Optimal mean block length for the stationary bootstrap
        for each variable.
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

    ### Fixed parameters ###
    bigkn  = max(5,np.sqrt(np.log10(T))) #footnote c of PW2004
    smallc = 2 #footnote c of PW2004
    m_max  = math.ceil(np.sqrt(T)+bigkn) #Patton code and R documention
    b_max  = math.ceil(min(3*np.sqrt(T),T/3)) #Patton code and R documention
    ### Critical value for significance of autocorrelation ###
    critsignif = smallc*np.sqrt(np.log10(T)/T)
    ### Initialize array of optimal stationary bootstrap block sizes ###
    bsbhat = np.nan*np.ones(dim)
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
    return(bsbhat)


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
