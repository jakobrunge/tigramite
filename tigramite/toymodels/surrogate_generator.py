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
                generate_noise_from='covariance',  verbosity=0):
    """
    Fits a (contemporaneous and lagged) linear SCM to data, computes
    residuals, and then generates surrogate realizations with noise drawn
    with replacement from residuals or from a multivariate normal.
    """

    from tigramite.models import Models, Prediction
    from sklearn.linear_model import LinearRegression

    assert dataframe.analysis_mode == 'single'

    if dataframe.mask is not None:
        for j in range(dataframe.N):
            # print(j) #, dataframe.mask[0][:, 0])
            np.testing.assert_allclose(dataframe.mask[0][:, 0], dataframe.mask[0][:, j])
    else:
        dataframe.mask = {0: np.zeros(dataframe.values[0].shape)}

    assert np.any(np.isnan(dataframe.values[0])) == False


    N = dataframe.N
    T = dataframe.T[0]


    ## Fit linear structural causal model to causal parents taken from graph
    def lin_f(x): return x

    # Build the model
    model = Models(
                    dataframe=dataframe,
                    model=LinearRegression(),
                    data_transform=None, #data_transform,
                    mask_type='y',
                    verbosity=0)
    links_coeffs = {}
    for j in range(N):
        links_coeffs[j] = []
        if len(parents[j]) > 0:
            fit_res = model.get_general_fitted_model(
                Y=[(j, 0)], X=list(parents[j]), Z=[],
                conditions=None,
                tau_max=tau_max,
                cut_off='tau_max',
                return_data=False)

            for ipar, par in enumerate(parents[j]):
                links_coeffs[j].append(((par[0], int(par[1])), fit_res[(j,0)]['model'].coef_[ipar], lin_f))
                if verbosity > 0:
                    print(j, ((par[0], int(par[1])), np.round(fit_res[(j,0)]['model'].coef_[ipar], 2),) )

    ## Estimate noise covariance matrix of residuals
    prediction = Prediction(dataframe=dataframe,
                     train_indices=range(T),
                     test_indices=range(T),
                     prediction_model=LinearRegression(),
                     # mask_type='y',
                     verbosity=0)

    prediction.fit(target_predictors=parents, tau_max=tau_max, return_data=True)

    # Get overlapping samples
    overlapping = set(list(range(0, T)))
    for j in parents:
        # print(j, prediction.fitted_model[j])
        if prediction.fitted_model[j] is not None:
            overlapping = overlapping.intersection(set(prediction.fitted_model[j]['used_indices'][0]))

    overlapping = sorted(list(overlapping))

    predicted = prediction.predict(target=[j for j in parents if len(parents[j]) > 0])

    # Residuals only exist after tau_max
    residuals = dataframe.values[0].copy()

    for index, j in enumerate([j for j in parents if len(parents[j]) > 0]):
        residuals[tau_max:][dataframe.mask[0][tau_max:, j]==False, j] -= predicted[index]
    
    overlapping_residuals = residuals[overlapping]

    if generate_noise_from == 'covariance':
        cov = np.cov(overlapping_residuals, rowvar=0)
        mean = np.mean(overlapping_residuals, axis=0)   # residuals should have zero mean due to prediction including constant
        if verbosity > 0:
            print('covariance')
            print(np.round(cov, 2))
            print('mean')
            print(np.round(mean, 2))


    ## Construct linear Gaussian structural causal model with this noise structure and generate many realizations with same sample size as data
    transient_fraction = 0.2
    size = T + int(math.floor(transient_fraction*T))
    datasets = {}
    for r in range(realizations):
        if generate_noise_from == 'covariance':
            noises = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
        else:
            draw = np.random.randint(0, len(overlapping_residuals), size)
            noises = overlapping_residuals[draw]

        datasets[r] = toys.structural_causal_process(links=links_coeffs, noises=noises, T=T, 
                                                     transient_fraction=transient_fraction)[0]
        if np.any(np.isinf(datasets[r])):
            raise ValueError("Infinite data")
    return datasets


if __name__ == '__main__':


    import tigramite
    from tigramite import data_processing as pp
    
    np.random.seed(14)     # Fix random seed
    lin_f = lambda x: x
    links_coeffs = {0: [((0, -1), 0.7, lin_f)],
                    1: [((1, -1), 0.8, lin_f), ((0, -1), 0.3, lin_f)],
                    2: [((2, -1), 0.5, lin_f), ((0, -2), -0.5, lin_f)],
                    3: [((3, -1), 0., lin_f)], #, ((4, -1), 0.4, lin_f)],
                    4: [((4, -1), 0., lin_f), ((3, 0), 0.5, lin_f)], #, ((3, -1), 0.3, lin_f)],
                    }
    T = 200     # time series length
    # Make some noise with different variance, alternatively just noises=None
    noises = np.array([(1. + 0.2*float(j))*np.random.randn((T + int(math.floor(0.2*T)))) 
                       for j in range(len(links_coeffs))]).T

    data, _ = toys.structural_causal_process(links_coeffs, T=T, noises=noises)
    T, N = data.shape

    # For generality, we include some masking
    # mask = np.zeros(data.shape, dtype='int')
    # mask[:int(T/2)] = True
    mask=None
    tau_max = 4
    # Initialize dataframe object, specify time axis and variable names
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$', r'$X^4$']
    dataframe = pp.DataFrame(data, 
                             mask=mask,
                             datatime = {0:np.arange(len(data))}, 
                             var_names=var_names)
    parents = {}
    for j in links_coeffs:
        parents[j] = []
        for par in links_coeffs[j]:
            parents[j].append(par[0])
    print(parents)
    datasets = generate_linear_model_from_data(dataframe, parents=parents, 
                tau_max=tau_max, realizations=100, 
                generate_noise_from='covariance',
                verbosity=0)