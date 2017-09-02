"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
from collections import Counter, defaultdict
import numpy
from nose.tools import assert_equal
import pytest

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPACE
import tigramite.data_processing as pp

# Define the verbosity at the global scope
VERBOSITY = 10

# CONVENIENCE FUNCTIONS ########################################################
def assert_graphs_equal(actual, expected):
    """
    Check whether lists in dict have the same elements.
    This ignore the order of the elements in the list.
    """
    for j in list(expected):
        assert_equal(Counter(iter(actual[j])), Counter(iter(expected[j])))

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

def _get_parents_from_results(pcmci, results, alpha_level):
    """
    Select the significant parents from the MCI-like results at a given
    alpha_level
    """
    significant_parents = \
        pcmci._return_significant_parents(pq_matrix=results['p_matrix'],
                                          val_matrix=results['val_matrix'],
                                          alpha_level=alpha_level)
    return significant_parents['parents']

# TEST DATA GENERATION #########################################################
@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # auto_corr, coeff, time, seed_val
    (0.1,        0.9,   1000, 2),
    (0.5,        0.6,   1000, 11),
    (0.5,        0.6,   1000, 42)])
def a_sample(request):
    # Set the parameters
    auto_corr, coeff, time, seed_val = request.param
    # Set the random seed
    numpy.random.seed(seed_val)
    # Define the parent-neighghbour relations
    links_coeffs = {0: [((0, -1), auto_corr)],
                    1: [((1, -1), auto_corr), ((0, -1), coeff)],
                    2: [((2, -1), auto_corr), ((1, -1), coeff)]}
    # Generate the data
    data, _ = pp.var_process(links_coeffs, T=time)
    # Get the true parents
    true_parents = _get_parent_graph(links_coeffs)
    return pp.DataFrame(data), true_parents

# PCMCI CONSTRUCTION ###########################################################
@pytest.fixture(params=[
    # Keep parameters common for all the run_ algorithms here
    # tau_min, tau_max,  sel_link,
     (1,       2,        None),
     (1,       2,        [0])])
def a_common_params(request):
    # Unpack the parameters to return them in a common tuple
    return request.param

@pytest.fixture()
    # Parameterize and return the independence test.
    # Currently just a wrapper for ParCorr, but is extendable
def a_test(request):
    return ParCorr(verbosity=VERBOSITY)

@pytest.fixture(params=[None, [1]])
    # Fixture to build and return a parameterized PCMCI.  Different selected
    # variables can be defined here.
def a_pcmci(a_sample, a_test, a_common_params, request):
    # Unpack the test data and true parent graph
    dataframe, true_parents = a_sample
    # Unpack the common parameters
    tau_min, tau_max, sel_link = a_common_params
    # Get the parameters from this request
    select_vars = request.param
    # Build the PCMCI instance
    pcmci = PCMCI(selected_variables=select_vars,
                  dataframe=dataframe,
                  cond_ind_test=a_test,
                  verbosity=VERBOSITY)
    # If there are selected variables, edit the true parents to reflect this
    if select_vars is not None:
        true_parents = {sel_v : true_parents[sel_v] for sel_v in select_vars}
    # Select the correct links if they are given
    select_links = _select_links(sel_link, true_parents)
    # Ensure we change the true parents to be the same as the selected links
    if select_links is not None:
        true_parents = select_links
    # Return the constructed PCMCI, expected results, and common parameters
    return pcmci, true_parents, tau_min, tau_max, select_links

# PC_STABLE TESTING ############################################################
@pytest.fixture(params=[
    # Keep parameters for the pc_stable algorithm here
    # pc_alpha,  max_conds_dim,  max_comb, save_iterations
     (None,      None,           2,        False),
     (0.05,      None,           1,        False),
     (0.05,      None,           10,       False),
     (0.05,      None,           1,        True),
     (0.05,      2,              1,        False)])
def a_pc_stable_params(request):
    # Return the parameters for the pc_stable test
    return request.param

@pytest.fixture()
def a_run_pc_stable(a_pcmci, a_pc_stable_params):
    # Unpack the pcmci, true parents, and common parameters
    pcmci, true_parents, tau_min, tau_max, select_links = a_pcmci
    # Unpack the pc_stable parameters
    pc_alpha, max_conds_dim, max_combinations, save_iter = a_pc_stable_params
    # Run PC stable
    pcmci.run_pc_stable(selected_links=select_links,
                        tau_min=tau_min,
                        tau_max=tau_max,
                        save_iterations=save_iter,
                        pc_alpha=pc_alpha,
                        max_conds_dim=max_conds_dim,
                        max_combinations=max_combinations)
    # Return the calculated and expected results
    return pcmci.all_parents, true_parents

def test_pc_stable(a_run_pc_stable):
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pc_stable
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)

# MCI TESTING ##################################################################
@pytest.fixture(params=[
    # Keep parameters for the mci algorithm here
    # alpha_level, max_conds_px, max_conds_py
     (0.01,        None,         None)])
def a_mci_params(request):
    # Return the parameters for the mci test
    return request.param

@pytest.fixture()
def a_run_mci(a_pcmci, a_mci_params):
    # Unpack the pcmci and the true parents, and common parameters
    pcmci, true_parents, tau_min, tau_max, select_links = a_pcmci
    # Unpack the MCI parameters
    alpha_level, max_conds_px, max_conds_py = a_mci_params
    # Run the MCI algorithm with the given parameters
    results = pcmci.run_mci(selected_links=select_links,
                            tau_min=tau_min,
                            tau_max=tau_max,
                            parents=true_parents,
                            max_conds_py=max_conds_px,
                            max_conds_px=max_conds_py)
    # Return the calculated and expected results
    return _get_parents_from_results(pcmci, results, alpha_level), true_parents

def test_mci(a_run_mci):
    # Unpack the calculated and true parents
    parents, true_parents = a_run_mci
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)

# PCMCI TESTING ################################################################
@pytest.fixture()
def a_run_pcmci(a_pcmci, a_pc_stable_params, a_mci_params):
    # Unpack the pcmci and the true parents, and common parameters
    pcmci, true_parents, tau_min, tau_max, select_links = a_pcmci
    # Unpack the pc_stable parameters
    pc_alpha, max_conds_dim, max_combinations, save_iter = a_pc_stable_params
    # Unpack the MCI parameters
    alpha_level, max_conds_px, max_conds_py = a_mci_params
    # Run the PCMCI algorithm with the given parameters
    results = pcmci.run_pcmci(selected_links=select_links,
                              tau_min=tau_min,
                              tau_max=tau_max,
                              save_iterations=save_iter,
                              pc_alpha=pc_alpha,
                              max_conds_dim=max_conds_dim,
                              max_combinations=max_combinations,
                              max_conds_px=max_conds_px,
                              max_conds_py=max_conds_py)
    # Return the results and the expected result
    return _get_parents_from_results(pcmci, results, alpha_level), true_parents

def test_pcmci(a_run_pcmci):
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pcmci
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)
