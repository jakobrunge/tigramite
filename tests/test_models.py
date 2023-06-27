"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
import numpy as np
import pytest
import sklearn.linear_model

from tigramite.models import LinearMediation
import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.models import Prediction
from tigramite.independence_tests.parcorr import ParCorr

from test_pcmci_calculations import a_chain, gen_data_frame

# Pylint settings
# pylint: disable=redefined-outer-name

# TEST LINK GENERATION #########################################################
@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.9, 0.7), 10000, 21),
    (a_chain(0.9, 0.7), 10000, 42)])
def data_frame_a(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val), links_coeffs

# TEST MODELS ##################################################################
def test_linear_mediation_coeffs(data_frame_a):
    # Build the dataframe and the model
    (dataframe, true_parents), links_coeffs = data_frame_a
    med = LinearMediation(dataframe=dataframe, data_transform=None)
    # Fit the model
    med.fit_model(all_parents=true_parents, tau_max=3)
    # Ensure the results make sense
    for j, i, tau, coeff in toys._iter_coeffs(links_coeffs):
        np.testing.assert_allclose(med.get_coeff(i=i, tau=tau, j=j),
                                   coeff, rtol=1e-1)

def test_linear_med_cause_coeffs(data_frame_a):
    # Build the dataframe and the model
    (dataframe, true_parents), _ = data_frame_a
    med = LinearMediation(dataframe=dataframe, data_transform=None)
    # Fit the model
    med.fit_model(all_parents=true_parents, tau_max=3)
    # Ensure the causal and mediated causal effects make sense
    for causal_coeff in [med.get_ce(i=0, tau=-2, j=2),
                         med.get_mce(i=0, tau=-2, j=2, k=1)]:
        np.testing.assert_allclose(causal_coeff, 0.49, rtol=1e-1)

# TEST PREDICTIONS #############################################################
def test_predictions(data_frame_a):
    # TODO NOTE: This doesn't actually test if the predictions make sense, only 
    # that they work!
    # Get the data
    (dataframe, true_parents), links_coeffs = data_frame_a
    T = dataframe.T[0]
    # Build the prediction
    a_cond_ind_test = ParCorr(significance='analytic')
    pred = Prediction(dataframe=dataframe,
                      cond_ind_test=a_cond_ind_test,
                      prediction_model=sklearn.linear_model.LinearRegression(),
                      train_indices=range(int(0.8*T)),
                      test_indices=range(int(0.8*T), T),
                      verbosity=0)
    # Load some parameters
    tau_max = 3
    steps_ahead = 1
    target = 2
    # Get the predictors from pc_stable
    all_predictors = pred.get_predictors(selected_targets=[target],
                                         selected_links=None,
                                         steps_ahead=steps_ahead,
                                         tau_max=tau_max,
                                         pc_alpha=None,
                                         max_conds_dim=None,
                                         max_combinations=1)
    # Fit the predictors using the ML method
    pred.fit(target_predictors=all_predictors,
             selected_targets=[target],
             tau_max=tau_max,
             return_data=True)
    # Predict the values
    _ = pred.predict(target)
    # TODO test the values make sense
