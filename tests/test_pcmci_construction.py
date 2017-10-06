"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
from collections import defaultdict
import numpy as np
import pytest

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPACE
import tigramite.data_processing as pp

from test_pcmci_calculations import a_chain

# Define the verbosity at the global scope
VERBOSITY = 10

# CONVENIENCE FUNCTIONS ########################################################
def _get_parent_graph(parents_neighbors_coeffs, exclude=None):
    """
    Iterates through the input parent-neighghbour coefficient dictionary to
    return only parent relations (i.e. where tau != 0)
    """
    graph = defaultdict(list)
    for j, i, tau, _ in pp._iter_coeffs(parents_neighbors_coeffs):
        if tau != 0 and (i, tau) != exclude:
            graph[j].append((i, tau))
    return dict(graph)

def _select_links(link_ids, true_parents):
    """
    Select links given by  from the true parents dictionary
    """
    if link_ids is None:
        return None
    return {par : [true_parents[par][link]] for par in true_parents \
                                            for link in link_ids}

# TEST LINK GENERATION #########################################################
@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,     time, seed_val
    (a_chain(0.5, 0.6, length=10), 1000, 42)])
def a_sample(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Set the random seed
    np.random.seed(seed_val)
    # Generate the data
    data, _ = pp.var_process(links_coeffs, T=time)
    # Get the true parents
    true_parents = _get_parent_graph(links_coeffs)
    return pp.DataFrame(data), true_parents

# PCMCI CONSTRUCTION ###########################################################
@pytest.fixture()
    # Fixture to build and return a parameterized PCMCI.  Different selected
    # variables can be defined here.
def a_pcmci(a_sample, request):
    # Unpack the test data and true parent graph
    dataframe, true_parents = a_sample
    # Build the PCMCI instance
    pcmci = PCMCI(selected_variables=None,
                  dataframe=dataframe,
                  cond_ind_test=ParCorr(verbosity=VERBOSITY),
                  verbosity=VERBOSITY)
    # Return the constructed PCMCI, expected results, and common parameters
    return pcmci, true_parents

# PCMCI CONSTRUCTION TESTING ###################################################
@pytest.fixture(params=[
    # Create a list of selected variables and testing for these variables
    ()])
def a_select_vars_params(request):
    # Return the requested parameters
    return request.param

def test_select_vars(a_pcmci):
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Test the default selected variables are correct
    err_msg = "Default option for _select_variables should return all variables"
    all_vars = pcmci._set_selected_variables(None)
    np.testing.assert_array_equal(all_vars, range(pcmci.N), err_msg=err_msg)
    # Test that both out of range and negative numbers throw an error while
    # good sets ones do not
    for sel_vars in [all_vars[::2], all_vars[1::2]]:
        # Initialize the list
        new_vars = []
        err_msg = "Variables incorrectly selected in _select_variables"
        # Test the good parameter set
        try:
            new_vars = pcmci._set_selected_variables(sel_vars)
        # Ensure no exception is raised
        except:
            pytest.fail("Selected variables fail incorrectly!")
        # Ensure the correct values are returned
        np.testing.assert_array_equal(new_vars,
                                      list(set(sel_vars)),
                                      err_msg=err_msg)
        # Ensure an exception is raised for a bad parameter set
        for bad_val, message in [(-1, "Negative"),
                                 (pcmci.N + 1, "Out of range")]:
            err_msg = message + " selected variables do not fail!"
            with pytest.raises(ValueError, message=err_msg):
                bad_vars = np.array(new_vars) * bad_val
                _ = pcmci._set_selected_variables(bad_vars)
