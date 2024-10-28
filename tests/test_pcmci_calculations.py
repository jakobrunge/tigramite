"""
Tests for pcmci.py, including tests for run_pc_stable, run_mci, and run_pcmci.
"""
from __future__ import print_function
from collections import Counter, defaultdict
import itertools
import numpy as np
# from nose.tools import assert_equal
from numpy.testing import assert_equal
import pytest

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.oracle_conditional_independence import OracleCI
import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

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
    for j, i, tau, _ in toys._iter_coeffs(parents_neighbors_coeffs):
        if tau != 0 and (i, tau) != exclude:
            graph[j].append((i, tau))
    return dict(graph)

def _select_links(link_ids, true_parents):
    """
    Select links given from the true parents dictionary
    """
    if link_ids is None:
        return None
    return {par : {true_parents[par][link]:'-->'} for par in true_parents \
                                            for link in link_ids}

def _get_parents_from_results(pcmci, results):
    """
    Select the significant parents from the MCI-like results at a given
    alpha_level
    """
    significant_parents = \
        pcmci.return_parents_dict(graph=results['graph'],
                                val_matrix=results['val_matrix'])
    return significant_parents

def gen_data_frame(links_coeffs, time, seed_val):
    # Set the random seed
    np.random.seed(seed_val)
    # Generate the data
    data, _ = toys.var_process(links_coeffs, T=time)
    # Get the true parents
    true_parents = _get_parent_graph(links_coeffs)
    return pp.DataFrame(data), true_parents

# TEST LINK GENERATION #########################################################
def a_chain(auto_corr, coeff, length=3):
    """
    Generate a simple chain process with the given auto-correlations and
    parents with the given coefficient strength.  A length can also be defined
    to get a longer chain.

    Parameters
    ----------
    auto_corr: float
        Autocorrelation strength for all nodes
    coeff : float
        Parent strength for all relations
    length : int
        Length of the chain
    """
    return_links = dict()
    return_links[0] = [((0, -1), auto_corr)]
    for lnk in range(1, length):
        return_links[lnk] = [((lnk, -1), auto_corr), ((lnk-1, -1), coeff)]
    return return_links

# TODO implement common_driver: return two variables commonly driven by N common
# drivers which are random noise, autocorrelation as parameter
# TODO implement independent drivers, autocorrelated noise
# TODO check common_driver, independent driver cases for current variable sets
# TODO implement USER_INPUT dictionary,
# USER_INPUT = dict()


# TEST DATA GENERATION #########################################################
@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time,  seed_val
    (a_chain(0.1, 0.9),           1000,  2),
    (a_chain(0.5, 0.6),           1000,  11),
    (a_chain(0.5, 0.6, length=5), 10000, 42)])
def a_sample(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)

# PCMCI CONSTRUCTION ###########################################################
@pytest.fixture(params=[
    # Keep parameters common for all the run_ algorithms here
    # tau_min, tau_max,  sel_link,
     (1,       2,        None),
     # (1,       2,        [0])
     ])
def a_common_params(request):
    # Return the requested parameters
    return request.param

@pytest.fixture()
    # Parameterize and return the independence test.
    # Currently just a wrapper for ParCorr, but is extendable
def a_test(request):
    return ParCorr(verbosity=VERBOSITY)

@pytest.fixture(params=[None])
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
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=a_test,
                  verbosity=VERBOSITY)
    # Select the correct links if they are given
    select_links = _select_links(sel_link, true_parents)
    # print(select_links)
    # Ensure we change the true parents to be the same as the selected links
    if select_links is not None:
        true_parents = select_links
    # Return the constructed PCMCI, expected results, and common parameters
    return pcmci, true_parents, tau_min, tau_max, select_links

# PC_STABLE TESTING ############################################################
@pytest.fixture(params=[
    # Keep parameters for the pc_stable algorithm here
    # pc_alpha,  max_conds_dim,  max_comb, save_iterations
     (None,      None,           3,        False),
     (0.05,      None,           1,        False),
     (0.05,      None,           10,       False),
     # (0.05,      None,           1,        True),
     # (0.05,      3,              1,        False)
    ])
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
    pcmci.run_pc_stable(link_assumptions=None,
                        tau_min=tau_min,
                        tau_max=tau_max,
                        save_iterations=save_iter,
                        pc_alpha=pc_alpha,
                        max_conds_dim=max_conds_dim,
                        max_combinations=max_combinations)
    # Return the calculated and expected results
    return pcmci.all_parents, true_parents

def test_pc_stable(a_run_pc_stable):
    """
    Test the pc_stable algorithm and check it calculates the correct parents.
    """
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
    results = pcmci.run_mci(link_assumptions=None,
                            tau_min=tau_min,
                            tau_max=tau_max,
                            parents=true_parents,
                            max_conds_py=max_conds_px,
                            max_conds_px=max_conds_py,
                            alpha_level=alpha_level)
    # Return the calculated and expected results
    return _get_parents_from_results(pcmci, results), true_parents

def test_mci(a_run_mci):
    """
    Test the mci algorithm and check it calculates the correct parents.
    """
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
    results = pcmci.run_pcmci(link_assumptions=None,
                              tau_min=tau_min,
                              tau_max=tau_max,
                              save_iterations=save_iter,
                              pc_alpha=pc_alpha,
                              max_conds_dim=max_conds_dim,
                              max_combinations=max_combinations,
                              max_conds_px=max_conds_px,
                              max_conds_py=max_conds_py,
                              alpha_level=alpha_level)
    # Return the results and the expected result
    return _get_parents_from_results(pcmci, results), true_parents

def test_pcmci(a_run_pcmci):
    """
    Test the pcmci algorithm and check it calculates the correct parents.
    """
    # Unpack the calculated and true parents
    parents, true_parents = a_run_pcmci
    # Ensure they are the same
    assert_graphs_equal(parents, true_parents)

# PCMCIplus TESTING
# ################################################################
def lin_f(x): return x

def setup_nodes(auto_coeff, N):
    link_coeffs = {}
    for j in range(N):
       link_coeffs[j] = [((j, -1), auto_coeff, lin_f)]
    return link_coeffs

def a_collider(auto_coeff, coeff):
    link_coeffs = setup_nodes(auto_coeff, N=3)
    for i in [0, 2]:
        link_coeffs[1].append(((i, 0), coeff, lin_f))
    return link_coeffs

def a_rule1(auto_coeff, coeff):
    link_coeffs = setup_nodes(auto_coeff, N=3)
    link_coeffs[1].append(((0, -1), coeff, lin_f))
    link_coeffs[2].append(((1, 0), coeff, lin_f))
    return link_coeffs

def a_rule2(auto_coeff, coeff):
    link_coeffs = setup_nodes(auto_coeff, N=3)
    link_coeffs[1].append(((1, -1), coeff, lin_f))
    link_coeffs[1].append(((0, 0), coeff, lin_f))
    link_coeffs[2].append(((1, 0), coeff, lin_f))
    link_coeffs[2].append(((0, 0), coeff, lin_f))
    return link_coeffs

def a_rule3(auto_coeff, coeff):
    link_coeffs = setup_nodes(auto_coeff, N=4)
    link_coeffs[1].append(((1, -1), coeff, lin_f))
    link_coeffs[0].append(((1, 0), coeff, lin_f))
    link_coeffs[2].append(((1, 0), coeff, lin_f))
    link_coeffs[2].append(((0, 0), coeff, lin_f))
    link_coeffs[3].append(((0, 0), coeff, lin_f))
    link_coeffs[2].append(((3, 0), coeff, lin_f))

    return link_coeffs

def a_random_process(
    N, L, 
    coupling_coeffs, 
    coupling_funcs, 
    auto_coeffs, 
    tau_max, 
    contemp_fraction=0.,
    num_trials=1000,
    model_seed=None):
    """Generates random structural causal process.
       
       TODO: documentation.

    Returns
    -------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
    """
    
    def lin(x): return x

    random_state = np.random.RandomState(model_seed)

    # print links
    a_len = len(auto_coeffs)
    if type(coupling_coeffs) == float:
        coupling_coeffs = [coupling_coeffs]
    c_len  = len(coupling_coeffs)
    func_len = len(coupling_funcs)

    if contemp_fraction > 0.:
        contemp = True
        L_lagged = int((1.-contemp_fraction)*L)
        L_contemp = L - L_lagged
        if L==1: 
            # Randomly assign a lagged or contemp link
            L_lagged = random_state.randint(0,2)
            L_contemp = int(L_lagged == False)

    else:
        contemp = False
        L_lagged = L
        L_contemp = 0

    # for ir in range(num_trials):
    # Random order
    causal_order = list(random_state.permutation(N))

    links = dict([(i, []) for i in range(N)])

    # Generate auto-dependencies at lag 1
    for i in causal_order:
        a = auto_coeffs[random_state.randint(0, a_len)]

        if a != 0.:
            links[i].append(((int(i), -1), float(a), lin))

    chosen_links = []
    # Create contemporaneous DAG
    contemp_links = []
    for l in range(L_contemp):

        cause = random_state.choice(causal_order[:-1])
        effect = random_state.choice(causal_order)
        while (causal_order.index(cause) >= causal_order.index(effect)
             or (cause, effect) in chosen_links):
            cause = random_state.choice(causal_order[:-1])
            effect = random_state.choice(causal_order)
        
        contemp_links.append((cause, effect))
        chosen_links.append((cause, effect))

    # Create lagged links (can be cyclic)
    lagged_links = []
    for l in range(L_lagged):

        cause = random_state.choice(causal_order)
        effect = random_state.choice(causal_order)
        while (cause, effect) in chosen_links or cause == effect:
            cause = random_state.choice(causal_order)
            effect = random_state.choice(causal_order)
        
        lagged_links.append((cause, effect))
        chosen_links.append((cause, effect))

    # print(chosen_links)
    # print(contemp_links)
    for (i, j) in chosen_links:

        # Choose lag
        if (i, j) in contemp_links:
            tau = 0
        else:
            tau = int(random_state.randint(1, tau_max+1))
        # print tau
        # CHoose coupling
        c = float(coupling_coeffs[random_state.randint(0, c_len)])
        if c != 0:
            func = coupling_funcs[random_state.randint(0, func_len)]

            links[j].append(((int(i), -tau), c, func))

    # # Stationarity check assuming model with linear dependencies at least for large x
    # # if check_stationarity(links)[0]:
    #     # return links
    # X, nonstat = toys.structural_causal_process(links, 
    #     T=100000, noises=None)
    # if nonstat == False:
    #     return links
    # else:
    #     print("Trial %d: Not a stationary model" % ir)


    # print("No stationary models found in {} trials".format(num_trials))
    # return None
    return links

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time,  seed_val
    (a_collider(0., 0.7),           1000,  2),
    (a_collider(0.6, 0.7),          1000,  2),
    (a_rule1(0., 0.7),          1000,  2),
    (a_rule2(0., 0.7),          1000,  2),
    (a_rule3(0., 0.7),          1000,  2),
    (a_rule3(0.5, 0.5),          1000,  2),

    ## Randomly generated structural causal processes, NEED to check whether seed
    ## generates graph that has only ONE member in Markov equivalence class!
    (a_random_process(
     N=5, L=5, coupling_coeffs=[0.7, -0.7],
     coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
     tau_max=5, contemp_fraction=0.3, num_trials=1,
     model_seed=3),             1000, 5),

    (a_random_process(
     N=5, L=5, coupling_coeffs=[0.7, -0.7],
     coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
     tau_max=5, contemp_fraction=0.3, num_trials=1,
     model_seed=4),             1000, 5),

    (a_random_process(
     N=5, L=5, coupling_coeffs=[0.7, -0.7],
     coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
     tau_max=5, contemp_fraction=0.3, num_trials=1,
     model_seed=5),             1000, 5),

    (a_random_process(
        N=10, L=10, coupling_coeffs=[0.7, -0.7],
        coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
        tau_max=5, contemp_fraction=0.3, num_trials=1,
        model_seed=3), 1000, 5),

    (a_random_process(
        N=20, L=20, coupling_coeffs=[0.7, -0.7],
        coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
        tau_max=5, contemp_fraction=0.2, num_trials=1,
        model_seed=4), 1000, 5),

    # (a_random_process(
    #     N=10, L=20, coupling_coeffs=[0.7, -0.7],
    #     coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
    #     tau_max=5, contemp_fraction=0.5, num_trials=1,
    #     model_seed=4), 1000, 6),

    # (a_random_process(
    #     N=8, L=20, coupling_coeffs=[0.7, -0.7],
    #     coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
    #     tau_max=5, contemp_fraction=0.5, num_trials=1,
    #     model_seed=4), 1000, 6),

    # (a_random_process(
    #     N=8, L=20, coupling_coeffs=[0.7, -0.7],
    #     coupling_funcs=[lin_f, lin_f], auto_coeffs=[0., 0.5],
    #     tau_max=5, contemp_fraction=0.5, num_trials=1,
    #     model_seed=5), 1000, 7),
])

def a_pcmciplus(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param

    # Retrieve lags
    tau_min, tau_max = toys._get_minmax_lag(links_coeffs)
    # Generate the data
    data, _ = toys.structural_causal_process(links=links_coeffs, T=time,
                                           noises=None, seed=seed_val)
    # Get the true parents
    # true_parents = toys._get_parents(links_coeffs, exclude_contemp=False)
    true_graph = toys.links_to_graph(links_coeffs, tau_max=tau_max)
    return pp.DataFrame(data), true_graph, links_coeffs, tau_min, tau_max

@pytest.fixture(params=[
    # Keep parameters for the algorithm here
    # pc_alpha, contemp_collider_rule, conflict_resolution, reset, cond_ind_test
    # OracleCI tests
    (0.01, 'majority', True, False, 'oracle_ci'),
    (0.01, 'none', True, False, 'oracle_ci'),
    (0.01, 'conservative', True, False, 'oracle_ci'),
    (0.01, 'majority', False, False, 'oracle_ci'),
    (0.01, 'none', False, False, 'oracle_ci'),
    (0.01, 'conservative', False, False, 'oracle_ci'),
    (0.01, 'majority', True, True, 'oracle_ci'),
    (0.01, 'none', True, True, 'oracle_ci'),
    (0.01, 'conservative', True, True, 'oracle_ci'),
    (0.01, 'majority', False, True, 'oracle_ci'),
    (0.01, 'none', False, True, 'oracle_ci'),
    (0.01, 'conservative', False, True, 'oracle_ci'),

    # ParCorr tests (can have finite sample errors)
    # (0.01, 'majority', True, False, 'par_corr'),
    # (0.01, 'none', True, False, 'par_corr'),
    # (0.01, 'conservative', True, False, 'par_corr'),
    # (0.01, 'majority', False, False, 'par_corr'),
    # (0.01, 'none', False, False, 'par_corr'),
    # (0.01, 'conservative', False, False, 'par_corr'),
    # (0.01, 'majority', True, True, 'par_corr'),
    # (0.01, 'none', True, True, 'par_corr'),
    # (0.01, 'conservative', True, True, 'par_corr'),
    # (0.01, 'majority', False, True, 'par_corr'),
    # (0.01, 'none', False, True, 'par_corr'),
    # (0.01, 'conservative', False, True, 'par_corr'),

])
def a_pcmciplus_params(request):
    # Return the parameters for the mci test
    return request.param

@pytest.fixture()
def a_run_pcmciplus(a_pcmciplus, a_pcmciplus_params):
    # Unpack the pcmci and the true parents, and common parameters
    dataframe, true_graph, links_coeffs, tau_min, tau_max = a_pcmciplus

    # Unpack the parameters
    (pc_alpha,
     contemp_collider_rule, conflict_resolution,
     reset_lagged_links, cond_ind_test_class,
     ) = a_pcmciplus_params

    if cond_ind_test_class == 'oracle_ci':
        cond_ind_test = OracleCI(links=links_coeffs)
    elif cond_ind_test_class == 'par_corr':
        cond_ind_test = ParCorr()

    # Run the PCMCI algorithm with the given parameters
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=2)
    results = pcmci.run_pcmciplus(
                      link_assumptions=None,
                      tau_min=tau_min,
                      tau_max=tau_max,
                      pc_alpha=pc_alpha,
                      contemp_collider_rule=contemp_collider_rule,
                      conflict_resolution=conflict_resolution,
                      reset_lagged_links=reset_lagged_links,
                      max_conds_dim=None,
                      max_conds_py=None,
                      max_conds_px=None,
                      )
    # Print true links
    print("************************")
    print("\nTrue Graph")
    for lag in range(tau_max):
        print("Lag %d = ", lag)
        print(true_graph[:,:,lag])
    # pcmci.print_significant_links(
    #                             p_matrix=(true_graph != ""),
    #                             val_matrix=true_graph,
    #                             conf_matrix=None,
    #                             q_matrix=None,
    #                             graph=true_graph,
    #                             ambiguous_triples=None,
    #                             alpha_level=0.05)
    # Return the results and the expected result
    return results['graph'], true_graph

def test_pcmciplus(a_run_pcmciplus):
    """
    Test the pcmciplus algorithm and check it calculates the correct graph.
    """
    # Unpack the calculated and true graph
    graph, true_graph = a_run_pcmciplus
    # Ensure they are the same
    np.testing.assert_equal(graph, true_graph)


@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time,  seed_val
    (a_random_process(
        N=5, L=15, coupling_coeffs=[0.5, -0.5],
        coupling_funcs=[lin_f], auto_coeffs=[0.5, 0.7],
        tau_max=1, contemp_fraction=0.6, num_trials=100,
        model_seed=2), 30, 5),

    (a_random_process(
        N=5, L=15, coupling_coeffs=[0.5, -0.5],
        coupling_funcs=[lin_f], auto_coeffs=[0.5, 0.7],
        tau_max=1, contemp_fraction=0.6, num_trials=100,
        model_seed=3), 30, 6),
])

def a_pcmciplus_order_independence(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param

    # Retrieve lags
    tau_min, tau_max = toys._get_minmax_lag(links_coeffs)
    # Generate the data
    data, _ = toys.structural_causal_process(links=links_coeffs, T=time,
                                           noises=None, seed=seed_val)
    # Get the true parents
    # true_parents = toys._get_parents(links_coeffs, exclude_contemp=False)
    true_graph = toys.links_to_graph(links_coeffs, tau_max=tau_max)
    return pp.DataFrame(data), true_graph, links_coeffs, tau_min, tau_max

@pytest.fixture(params=[
    # Keep parameters for the algorithm here
    # pc_alpha, contemp_collider_rule, conflict_resolution, reset, cond_ind_test
    # OracleCI tests

    # ParCorr tests (can have finite sample errors)
    (0.2, 'majority', True, False, 'par_corr'),
    (0.2, 'conservative', True, False, 'par_corr'),
    (0.2, 'majority', True, True, 'par_corr'),
    (0.2, 'conservative', True, True, 'par_corr'),
    (None, 'majority', True, False, 'par_corr'),
    (None, 'conservative', True, False, 'par_corr'),
    (None, 'majority', True, True, 'par_corr'),
    (None, 'conservative', True, True, 'par_corr'),
    ])
def a_pcmciplus_params_order_independence(request):
    # Return the parameters for the mci test
    return request.param

# @pytest.fixture()
def test_order_independence_pcmciplus(a_pcmciplus_order_independence,
                    a_pcmciplus_params_order_independence):
    # Unpack the pcmci and the true parents, and common parameters
    dataframe, true_graph, links_coeffs, tau_min, tau_max = \
        a_pcmciplus_order_independence

    data = dataframe.values[0]
    T = dataframe.T[0]
    N = dataframe.N

    # Unpack the parameters
    (pc_alpha,
     contemp_collider_rule, conflict_resolution,
     reset_lagged_links, cond_ind_test_class,
     ) = a_pcmciplus_params_order_independence

    if cond_ind_test_class == 'oracle_ci':
        cond_ind_test = OracleCI(links=links_coeffs)
    elif cond_ind_test_class == 'par_corr':
        cond_ind_test = ParCorr()

    # Run the PCMCI algorithm with the given parameters
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)
    print("************************")
    print("\nTrue Graph")
    for lag in range(tau_max):
        print("Lag %d = ", lag)
        print(true_graph[:,:,lag])
    # pcmci.print_significant_links(
    #                             p_matrix=(true_graph == 0),
    #                             val_matrix=true_graph,
    #                             conf_matrix=None,
    #                             q_matrix=None,
    #                             graph=true_graph,
    #                             ambiguous_triples=None,
    #                             alpha_level=0.05)

    results = pcmci.run_pcmciplus(
                      link_assumptions=None,
                      tau_min=tau_min,
                      tau_max=tau_max,
                      pc_alpha=pc_alpha,
                      contemp_collider_rule=contemp_collider_rule,
                      conflict_resolution=conflict_resolution,
                      reset_lagged_links=reset_lagged_links,
                      max_conds_dim=None,
                      max_conds_py=None,
                      max_conds_px=None,
                      )
    correct_results = results['graph']

    for perm in itertools.permutations(range(N)):

        print(perm)
        data_new = np.copy(data[:,perm])
        dataframe = pp.DataFrame(data_new, var_names=list(perm))
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test,
                      verbosity=1)
        results = pcmci.run_pcmciplus(
                      link_assumptions=None,
                      tau_min=tau_min,
                      tau_max=tau_max,
                      pc_alpha=pc_alpha,
                      contemp_collider_rule=contemp_collider_rule,
                      conflict_resolution=conflict_resolution,
                      reset_lagged_links=reset_lagged_links,
                      max_conds_dim=None,
                      max_conds_py=None,
                      max_conds_px=None,
                      )

        tmp = np.take(correct_results, perm, axis=0)
        back_converted_result = np.take(tmp, perm, axis=1)

        for tau in range(tau_max+1):
            if not np.array_equal(results['graph'][:,:,tau],
                               back_converted_result[:,:,tau]):
                print(tau)
                print(results['graph'][:,:,tau])
                print(back_converted_result[:,:,tau])
                print(back_converted_result[:,:,tau]-results['graph'][:,:,tau])
                print(perm)

        # np.allclose(results['graph'], back_converted_result)
        np.testing.assert_equal(results['graph'], back_converted_result)

