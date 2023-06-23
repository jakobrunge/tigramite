from copy import deepcopy

import pytest
import numpy as np
from numpy.random import MT19937, SeedSequence

from tigramite.jpcmci import J_PCMCIplus
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from tigramite.independence_tests.oracle_conditional_independence import OracleCI

from tigramite.toymodels.context_model import shift_link_entries, ContextModel
import tigramite.data_processing as pp


# J-PCMCIplus TESTING
############### for now only test one model

def lin_f(x):
    return x


@pytest.fixture(params=[
    # Generate a test data sample
    # value, total time length, and random seed to different numbers
    # links,               time, nb_domains, seed_val

    ({0: [((0, -1), 0.3, lin_f), ((3, -1), 0.6, lin_f), ((4, -1), 0.9, lin_f)],
      1: [((1, -1), 0.4, lin_f), ((3, -1), 0.4, lin_f)],
      2: [((2, -1), 0.3, lin_f), ((1, -2), -0.5, lin_f), ((4, -1), 0.5, lin_f), ((5, 0), 0.6, lin_f)],
      3: [], 4: [], 5: []
      }, {
         0: "system",
         1: "system",
         2: "system",
         3: "time_context",
         4: "time_context",
         5: "space_context"
     }, 1000, 200, SeedSequence(12345).spawn(1)[0]),
])
def a_jpcmciplus(request):
    # Set the parameters
    links, node_types, T, nb_domains, seed_val = request.param

    # Retrieve lags
    tau_min, tau_max = toys._get_minmax_lag(links)
    # Generate the data
    contextmodel = ContextModel(links=links, node_classification=node_types, noises=None,
                                seed=seed_val)

    data_ens, nonstationary = contextmodel.generate_data(nb_domains, T)
    assert not nonstationary

    # decide which context variables should be latent, and which are observed
    observed_context_indices = [4, 5]
    system_indices = [node for i, node in enumerate(links.keys()) if node_types[i] == "system"]

    # all system variables are also observed, thus we get the following observed data
    observed_indices = system_indices + observed_context_indices
    observed_indices_time = [el for el in observed_context_indices if node_types[el] == "time"]
    data_observed = {key: data_ens[key][:, observed_indices] for key in data_ens}

    dummy_data_time = np.identity(T)

    data_dict = {}
    for i in range(nb_domains):
        dummy_data_space = np.zeros((T, nb_domains))
        dummy_data_space[:, i] = 1.
        data_dict[i] = np.hstack((data_observed[i], dummy_data_time, dummy_data_space))

    # Define vector-valued variables including dummy variables as well as observed (system and context) variables
    nb_observed_context_nodes = len(observed_context_indices)
    N = len(system_indices)
    observed_temporal_context_nodes = list(range(N, N + len(observed_indices_time)))
    observed_spatial_context_nodes = list(range(N + len(observed_indices_time),
                                                N + len(observed_context_indices)))
    time_dummy_index = N + nb_observed_context_nodes
    space_dummy_index = N + nb_observed_context_nodes + 1
    time_dummy = list(range(time_dummy_index, time_dummy_index + T))
    space_dummy = list(range(time_dummy_index + T, time_dummy_index + T + nb_domains))

    vector_vars = {i: [(i, 0)] for i in
                   system_indices + observed_temporal_context_nodes + observed_spatial_context_nodes}
    vector_vars[time_dummy_index] = [(i, 0) for i in time_dummy]
    vector_vars[space_dummy_index] = [(i, 0) for i in space_dummy]

    dataframe = pp.DataFrame(
        data=data_dict,
        vector_vars=vector_vars,
        analysis_mode='multiple',
    )

    # Get the true parents
    # true_parents = toys._get_parents(links_coeffs, exclude_contemp=False)
    true_graph = toys.links_to_graph(links, tau_max=tau_max)
    # TODO: remove context-context links
    # TODO: add dummy
    return dataframe, true_graph, links, tau_min, tau_max, \
        observed_temporal_context_nodes, observed_spatial_context_nodes, time_dummy_index, space_dummy_index


@pytest.fixture(params=[
    # Keep parameters for the algorithm here
    # pc_alpha, contemp_collider_rule, conflict_resolution, reset, cond_ind_test
    # OracleCI tests
    (0.01, 'majority', True, False, 'oracle_ci'),
    #(0.01, 'none', True, False, 'oracle_ci'),
    #(0.01, 'conservative', True, False, 'oracle_ci'),
    #(0.01, 'majority', False, False, 'oracle_ci'),
    #(0.01, 'none', False, False, 'oracle_ci'),
    #(0.01, 'conservative', False, False, 'oracle_ci'),
    #(0.01, 'majority', True, True, 'oracle_ci'),
    #(0.01, 'none', True, True, 'oracle_ci'),
    #(0.01, 'conservative', True, True, 'oracle_ci'),
    #(0.01, 'majority', False, True, 'oracle_ci'),
    #(0.01, 'none', False, True, 'oracle_ci'),
    #(0.01, 'conservative', False, True, 'oracle_ci'),
])
def a_jpcmciplus_params(request):
    # Return the parameters for the mci test
    return request.param


@pytest.fixture()
def a_run_jpcmciplus(a_jpcmciplus, a_jpcmciplus_params):
    # Unpack the pcmci and the true parents, and common parameters
    dataframe, true_graph, links_coeffs, tau_min, tau_max, \
        observed_temporal_context_nodes, observed_spatial_context_nodes, \
        time_dummy_index, space_dummy_index = a_jpcmciplus

    # Unpack the parameters
    (pc_alpha,
     contemp_collider_rule, conflict_resolution,
     reset_lagged_links, cond_ind_test_class,
     ) = a_jpcmciplus_params

    cond_ind_test = OracleCI(links=links_coeffs)
    if cond_ind_test_class == 'par_corr_mult':
        cond_ind_test = ParCorrMult()

    # Run the PCMCI algorithm with the given parameters
    pcmci = J_PCMCIplus(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=2,
                        time_context_nodes=observed_temporal_context_nodes,
                        space_context_nodes=observed_spatial_context_nodes,
                        time_dummy=time_dummy_index,
                        space_dummy=space_dummy_index
                        )
    results = pcmci.run_jpcmciplus(
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
        print(true_graph[:, :, lag])
    # pcmci.print_significant_links(
    #                             p_matrix=(true_graph != ""),
    #                             val_matrix=true_graph,
    #                             conf_matrix=None,
    #                             q_matrix=None,
    #                             graph=true_graph,
    #                             ambiguous_triples=None,
    #                             alpha_level=0.05)
    # Print true links
    print("************************")
    print("\nJ-PCMCI+ Output")
    for lag in range(tau_max):
        print("Lag %d = ", lag)
        print(results['graph'][:, :, lag])
    # Return the results and the expected result
    return results['graph'], true_graph


def test_jpcmciplus(a_run_jpcmciplus):
    """
    Test the pcmciplus algorithm and check it calculates the correct graph.
    """
    # Unpack the calculated and true graph
    graph, true_graph = a_run_jpcmciplus
    # Ensure they are the same
    np.testing.assert_equal(graph, true_graph)


# ############ Utils
def mergeDictionary(dict_1, dict_2):
    dict_3 = dict(dict_1)
    dict_3.update(dict_2)
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3


def choose_cause_tau(candidates, tau_max, random_state):
    if type(candidates[0]) is tuple:
        cause, tau = random_state.choice(candidates)
    else:
        cause = random_state.choice(candidates)
        tau = 0 if tau_max == 0 else int(random_state.integers(1, tau_max + 1))
    return cause, tau


def get_non_parents(links, tau_max):
    parents = sum(links.values(), [])
    parents = [i[0] for i in parents]
    non_parents = [(var, tau) for var in links.keys() for tau in range(tau_max + 1) if (var, -tau) not in parents]
    return non_parents


def populate_links(links, chosen_links, contemp_links, max_lag, dependency_coeffs, dependency_funcs, random_state):
    for (i, j, tau) in chosen_links:
        # Choose lag
        if tau is None:
            if (i, j) in contemp_links or max_lag == 0:
                tau = 0
            else:
                tau = int(random_state.integers(1, max_lag + 1))

        # Choose dependency
        c = float(random_state.choice(dependency_coeffs))
        if c != 0:
            func = random_state.choice(dependency_funcs)

            links[j].append(((int(i), -int(tau)), c, func))
    return links


def set_links(effect_candidates, L, cause_candidates, tau_max, dependency_funcs, dependency_coeffs, random_state):
    chosen_links = []
    contemp_links = []
    links = {i: [] for i in effect_candidates}
    for l in range(L):
        cause, tau = choose_cause_tau(cause_candidates, tau_max, random_state)
        effect = random_state.choice(effect_candidates)

        while (cause, effect, tau) in chosen_links:
            cause, tau = choose_cause_tau(cause_candidates, tau_max, random_state)
            effect = random_state.choice(effect_candidates)
        if tau == 0:
            contemp_links.append((cause, effect, tau))
            chosen_links.append((cause, effect, tau))
        else:
            chosen_links.append((cause, effect, tau))

        links = populate_links(links, chosen_links, contemp_links, tau_max, dependency_coeffs, dependency_funcs,
                               random_state)

    return links


# J-PCMCIplus TESTING
# ################################################################
def a_generate_random_context_model(N=3,
                                    K_space=2,
                                    K_time=2,
                                    tau_max=2,
                                    model_seed=None):
    #child_seeds = model_seed.spawn(7)
    # use seedsequence instead
    ss = SeedSequence()
    child_seeds = ss.spawn(7)

    dependency_funcs = ['linear']
    dependency_coeffs = [-0.5, 0.5]
    auto_coeffs = [0.5, 0.7]
    contemp_fraction = 0.5
    noise_dists = ['gaussian']
    noise_means = [0.]
    noise_sigmas = [0.5, 2.]

    L = 1 if N == 2 else N
    L_space = 1 if K_space == 2 else K_space
    L_time = 1 if K_time == 2 else K_time

    nodes_sc = list(range(N + K_time, N + K_space + K_time))
    nodes_tc = list(range(N, K_time + N))
    nodes_sys = list(range(N))

    links_tc = {}
    links_sc = {}
    noises_tc = []
    noises_sc = []

    # graph for temporal context vars
    if K_time != 0:
        links_tc, noises_tc = toys.generate_structural_causal_process(len(nodes_tc),
                                                                      L_time,
                                                                      dependency_funcs,
                                                                      dependency_coeffs,
                                                                      auto_coeffs,
                                                                      contemp_fraction,
                                                                      tau_max,
                                                                      noise_dists,
                                                                      noise_means,
                                                                      noise_sigmas,
                                                                      MT19937(child_seeds[0]),
                                                                      MT19937(child_seeds[1]), )
        links_tc = shift_link_entries(links_tc, N)

    if K_space != 0:
        # graph for spatial context vars
        links_sc, noises_sc = toys.generate_structural_causal_process(len(nodes_sc),
                                                                      L_space,
                                                                      dependency_funcs,
                                                                      dependency_coeffs,
                                                                      auto_coeffs,
                                                                      1.,
                                                                      0,
                                                                      noise_dists,
                                                                      noise_means,
                                                                      noise_sigmas,
                                                                      MT19937(child_seeds[2]),
                                                                      MT19937(child_seeds[3]))
        links_sc = shift_link_entries(links_sc, N + K_time)

    # graph for system vars
    links_sys, noises_sys = toys.generate_structural_causal_process(len(nodes_sys),
                                                                    L,
                                                                    dependency_funcs,
                                                                    dependency_coeffs,
                                                                    auto_coeffs,
                                                                    1.,
                                                                    tau_max,
                                                                    noise_dists,
                                                                    noise_means,
                                                                    noise_sigmas,
                                                                    MT19937(child_seeds[4]),
                                                                    MT19937(child_seeds[5]))

    links = dict(links_tc)
    links.update(links_sc)
    links.update(links_sys)
    noises = noises_sys + noises_tc + noises_sc

    # set context-system links
    non_parent_tc = get_non_parents(links_tc, tau_max)
    non_parent_sc = get_non_parents(links_sc, 0)

    # number of links between system and context
    L_context_sys = 2 * (len(non_parent_sc) + len(non_parent_tc))

    context_sys_links = set_links(nodes_sys, L_context_sys, non_parent_sc + non_parent_tc,
                                  tau_max, dependency_funcs, dependency_coeffs,
                                  np.random.default_rng(child_seeds[6]))

    # join all link-dicts to form graph over context and system nodes
    links = mergeDictionary(links, context_sys_links)

    return links, links_tc, links_sc, links_sys, noises

# ################################################################
