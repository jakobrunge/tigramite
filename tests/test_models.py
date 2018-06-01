"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
import numpy as np
import pytest

from tigramite.models import LinearMediation
import tigramite.data_processing as pp

from test_pcmci_calculations import a_chain, gen_data_frame

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

    #Link coefficient (0, -2) --> 2:  0.0
    #Causal effect (0, -2) --> 2:  0.250648072987
    #Mediated Causal effect (0, -2) --> 2 through 1:  0.250648072987
    #Average Causal Effect:  [ 0.36897445  0.25718002  0.        ]
    #Average Causal Susceptibility:  [ 0.          0.24365041  0.38250406]
    #Average Mediated Causal Effect:  [ 0.          0.12532404  0.        ]

# TEST LINK GENERATION #########################################################
@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.5, 0.2), 10000, 21),
    (a_chain(0.5, 0.2), 10000, 42)])
def data_frame_a(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val), links_coeffs

# TEST MODEL FUNCTIONALITY #####################################################
def test_linear_mediation_coeffs(data_frame_a):
    # Build the dataframe and the model
    (dataframe, true_parents), links_coeffs = data_frame_a
    med = LinearMediation(dataframe=dataframe)
    # Fit the model
    med.fit_model(all_parents=true_parents, tau_max=3)
    # Ensure the results make sense
    for j, i, tau, coeff in pp._iter_coeffs(links_coeffs):
        np.testing.assert_allclose(med.get_coeff(i=i, tau=tau, j=j),
                                   coeff, rtol=1e-1)

def test_linear_mediation_causal_coeffs(data_frame_a):
    # Build the dataframe and the model
    (dataframe, true_parents), links_coeffs = data_frame_a
    med = LinearMediation(dataframe=dataframe)
    # Fit the model
    med.fit_model(all_parents=true_parents, tau_max=3)
    # Ensure the causal and mediated casual effects make sense
    for causal_coeff in [med.get_ce(i=0, tau=-2, j=2),
                         med.get_mce(i=0, tau=-2, j=2, k=1)]:
        np.testing.assert_allclose(causal_coeff, 0.035, rtol=1e-1)
