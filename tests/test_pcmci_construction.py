"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
from collections import defaultdict
import numpy as np
import pytest
from scipy.misc import comb

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPACE
import tigramite.data_processing as pp

from test_pcmci_calculations import a_chain

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

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
N_NODES = 4
@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,     time, seed_val
    (a_chain(0.5, 0.6, length=N_NODES), 1000, 42)])
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

# TEST VARIABLE SELECTION ######################################################
def get_select_vars(some_vars):
    """
    Yield some subsets of the input variables
    """
    for subset in [some_vars[0::2], some_vars[1::2]]:
        yield subset

def test_select_vars_default(a_pcmci):
    """
    Check that _set_selected_variables returns all variables by default.
    """
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Test the default selected variables are correct
    all_vars = pcmci._set_selected_variables(None)
    err_msg = "Default option for _set_selected_variables should return all"+\
              " variables"
    np.testing.assert_array_equal(list(all_vars),
                                  list(range(pcmci.N)),
                                  err_msg=err_msg)

def test_select_vars_setting(a_pcmci):
    """
    Check that _set_selected_variables returns the correct subset of all variables
    when requested.
    """
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Test that both out of range and negative numbers throw an error while
    # good sets ones do not
    for sel_vars in get_select_vars(range(pcmci.N)):
        # Initialize the list
        new_vars = pcmci._set_selected_variables(sel_vars)
        err_msg = "Variables incorrectly selected in _select_variables"
        # Ensure the correct values are returned
        np.testing.assert_array_equal(new_vars,
                                      sorted(list(set(sel_vars))),
                                      err_msg=err_msg)

def test_select_vars_errors(a_pcmci):
    """
    Check that _set_selected_variables throws the correct errors for bad requests
    but does not crash for good requests.
    """
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Test that both out of range and negative numbers throw an error while
    # good sets ones do not
    for sel_vars in get_select_vars(range(pcmci.N)):
        # Initialize the list
        new_vars = []
        # Test the good parameter set
        try:
            new_vars = pcmci._set_selected_variables(sel_vars)
        # Ensure no exception is raised
        except:
            pytest.fail("Selected variables fail incorrectly!")
        # Ensure an exception is raised for a bad parameter set
        for bad_val, message in [(-1, "Negative"),
                                 (pcmci.N + 1, "Out of range")]:
            err_msg = message + " selected variables do not fail!"
            with pytest.raises(ValueError, message=err_msg):
                bad_vars = np.array(new_vars) * bad_val
                _ = pcmci._set_selected_variables(bad_vars)

# TEST LINK SELECTION ##########################################################
TAU_MIN = 1
TAU_MAX = 3
def get_expected_links(a_range, b_range, t_range):
    """
    Helper function to generate the expected links
    """
    return {a_var : [(b_var, -lag) for b_var in b_range for lag in t_range]
            for a_var in a_range}

def test_select_links_default(a_pcmci):
    """
    Check that _set_sel_links returns all links by default.
    """
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Test the default selected variables are correct
    sel_links = pcmci._set_sel_links(None, TAU_MIN, TAU_MAX)
    err_msg = "Default option for _set_sel_links should return all possible "+\
              "combinations"
    good_links = get_expected_links(range(pcmci.N),
                                    range(pcmci.N),
                                    range(TAU_MIN, TAU_MAX + 1))
    np.testing.assert_equal(sel_links, good_links, err_msg=err_msg)

def test_select_links_setting(a_pcmci):
    """
    Check that _set_sel_links returns only links of selected variables.
    """
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Seleect some of the variables
    for sel_vars in get_select_vars(range(pcmci.N)):
        # Initialize the list
        pcmci.selected_variables = pcmci._set_selected_variables(sel_vars)
        # Ensure the correct values are returned
        sel_links = pcmci._set_sel_links(None, TAU_MIN, TAU_MAX)
        good_links = get_expected_links(sel_vars,
                                        range(pcmci.N),
                                        range(TAU_MIN, TAU_MAX + 1))
        # TODO remove when we remove the empty lists from the code
        for idx in range(pcmci.N):
            if idx not in sel_vars:
                good_links[idx] = []
        err_msg = "Links incorrectly set in _set_sel_links"
        np.testing.assert_equal(sel_links, good_links, err_msg=err_msg)

def test_select_links_errors(a_pcmci):
    """
    Check that _set_sel_links throws the correct errors when out of range
    variables are mentioned.
    """
    # Unpack the pcmci instance
    pcmci, _ = a_pcmci
    # Select some of the variables
    for sel_vars in get_select_vars(range(pcmci.N)):
        # Initialize the list
        pcmci.selected_variables = pcmci._set_selected_variables(sel_vars)
        # Ensure the correct values are returned
        test_links = get_expected_links(sel_vars,
                                        range(pcmci.N),
                                        range(TAU_MIN, TAU_MAX + 1))
        # Test the good parameter set
        try:
            _ = pcmci._set_sel_links(test_links, TAU_MIN, TAU_MAX)
        # Ensure no exception is raised
        except:
            pytest.fail("Selected links fail incorrectly!")
        # Ensure an exception is raised for a bad parameter set
        for bad_val, message in [(pcmci.N + 1, "Out of range")]:
            err_msg = message + " selected links do not fail!"
            with pytest.raises(ValueError, message=err_msg):
                test_links[bad_val] = [(bad_val, TAU_MAX)]
                _ = pcmci._set_sel_links(test_links, TAU_MIN, TAU_MAX)

# TEST ITERATORS ###############################################################
@pytest.fixture(params=[
    # Store some parameters for setting maximum conditions
    #max_cond_dim, tau_min, tau_max
    (None,         1,       3),
    (10,           1,       3)])
    #(1,            2,       4)]) TODO address this failing case
def a_iter_cond_param(request):
    return request.param

def test_condition_iterator(a_pcmci, a_iter_cond_param):
    """
    Test if the correct conditions are returned from the iterator.  Checks that:
        * Parent node never included in yielded conditions
        * Yielded conditions of the same dimension are unique
        * The expected total number of conditions are given
    """
    # Unpack the pcmci algorithm
    pcmci, _ = a_pcmci
    # Unpack the iterator conditions parameters
    max_cond_dim, tau_min, tau_max = a_iter_cond_param
    # Get all possible links
    sel_links = pcmci._set_sel_links(None, tau_min, tau_max)
    # Loop over all possible condition dimentions
    # TODO ask jakob about max_conds_dim in source and maybe change this
    # limit to max_cond_dim + 1
    max_cond_dim = pcmci._set_max_condition_dim(None, tau_max)
    if max_cond_dim == pcmci._set_max_condition_dim(max_cond_dim, tau_max):
        max_cond_dim -= 1
    # Loop over all nodes
    for j in range(pcmci.N):
        # Initialise the list of conditions for this node
        total_n_conds = 0
        expected_total_n_conds = 0
        # Parents for this node
        parents = sel_links[j]
        # Loop over all possible dimentionality of conditions
        for cond_dim in range(max_cond_dim + 1):
            # Iterate through all possible pairs (that have not converged yet)
            for par in parents:
                # Number of conditions for this dimension
                expected_n_conds = comb(len(parents) - 1, cond_dim, exact=True)
                expected_total_n_conds += expected_n_conds
                # Create an array for these conditions
                par_conds = list()
                # Iterate through all possible combinations
                for cond in pcmci._iter_conditions(par, j, cond_dim, parents):
                    # Ensure the condition does not contain the parent
                    assert par not in cond, "Parent should not be in condition"
                    # Keep this condition in list of conditions
                    par_conds.append([par] + cond)
                    assert len(cond) == cond_dim,\
                        "Number of conditions should match requested dimension"
                # Check that at least one condition is found
                assert len(par_conds) != 0,\
                    "Expected number of conditions for condition "+\
                    " dimension {}".format(cond_dim)+\
                    " and total n_nodes {}".format(len(parents)-1)+\
                    " is zero!"
                # Check that this iterations has the expected number of
                # conditions
                assert len(par_conds) == expected_n_conds,\
                    "Expected number of conditions for condition "+\
                    " dimension {}".format(cond_dim)+\
                    " and total n_nodes {}".format(len(parents)-1)+\
                    " does not match"
                # Record the total number found
                total_n_conds += len(par_conds)
                # Check that all condition, parent pairs are unique
                par_conds = np.array(par_conds)
                assert np.unique(par_conds, axis=0).shape == par_conds.shape
        # Check that the right number of conditions are found
        assert total_n_conds == expected_total_n_conds, \
            "Expected number of combinations found"
