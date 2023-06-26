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
        tau = 0 if tau_max == 0 else int(random_state.integers(1, tau_max))
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
                tau = int(random_state.integers(1, max_lag))

        # Choose dependency
        c = float(random_state.choice(dependency_coeffs))
        if c != 0:
            func = random_state.choice(dependency_funcs)
            if func == "linear":
                func = lin_f

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


def nb_latent_before(node, observed_context_indices, node_classification):
    return len(
        [el for el in range(node) if not (el in observed_context_indices or node_classification[el] == "system")])


# J-PCMCIplus TESTING for random context models
# ################################################################
def a_generate_random_context_model(N=3,
                                    K_space=2,
                                    K_time=2,
                                    tau_max=2):
    # use seedsequence
    ss = SeedSequence(12345)
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

    node_classification = {}
    for node in links_sys.keys():
        node_classification[node] = "system"
    for node in links_tc.keys():
        node_classification[node] = "time_context"
    for node in links_sc.keys():
        node_classification[node] = "space_context"

    return links, node_classification, noises


# ################################################################


@pytest.fixture(params=[
    # Generate a test data sample
    # value, total time length, and random seed to different numbers
    # links,               node_classification, noises, time, nb_domains, seed_val

    (a_generate_random_context_model(N=3,
                                    K_space=2,
                                    K_time=2,
                                    tau_max=2), 100, 100, SeedSequence(12345).spawn(1)[0]),
    (a_generate_random_context_model(N=5,
                                    K_space=2,
                                    K_time=2,
                                    tau_max=2), 100, 100, SeedSequence(12345).spawn(1)[0]),
    (a_generate_random_context_model(N=10,
                                    K_space=5,
                                    K_time=5,
                                    tau_max=2), 100, 100, SeedSequence(12345).spawn(1)[0]),
    (({0: [((0, -1), 0.3, lin_f), ((3, -1), 0.6, lin_f), ((4, -1), 0.9, lin_f)],
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
    }, None), 100, 50, SeedSequence(12345).spawn(1)[0]),

])
def a_jpcmciplus(request):
    # Set the parameters
    ct_model, T, nb_domains, seed_val = request.param
    links, node_classification_gt, noises = ct_model

    a_generate_random_context_model(N=5,
                                    K_space=2,
                                    K_time=2,
                                    tau_max=2)

    percent_observed = 0.5

    # Retrieve lags
    tau_min, tau_max = toys._get_minmax_lag(links)
    # Generate the data
    contextmodel = ContextModel(links=links, node_classification=node_classification_gt, noises=noises,
                                seed=seed_val)

    data_ens, nonstationary = contextmodel.generate_data(nb_domains, T)
    assert not nonstationary

    # decide which context variables should be latent, and which are observed
    time_context_indices = [node for node in links.keys() if node_classification_gt[node] == "time_context"]
    space_context_indices = [node for node in links.keys() if node_classification_gt[node] == "space_context"]
    system_indices = [node for node in links.keys() if node_classification_gt[node] == "system"]

    random_state = np.random.default_rng(seed_val.spawn(1)[0])
    observed_indices_time = random_state.choice(time_context_indices,
                                                size=int(percent_observed * len(time_context_indices)),
                                                replace=False).tolist()
    observed_indices_space = random_state.choice(space_context_indices,
                                                 size=int(percent_observed * len(space_context_indices)),
                                                 replace=False).tolist()
    observed_context_indices = observed_indices_time + observed_indices_space

    # all system variables are also observed, thus we get the following observed data
    observed_indices = system_indices + observed_context_indices
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
                                                N + nb_observed_context_nodes))
    time_dummy_index = N + nb_observed_context_nodes
    space_dummy_index = N + nb_observed_context_nodes + 1
    time_dummy = list(range(time_dummy_index, time_dummy_index + T))
    space_dummy = list(range(time_dummy_index + T, time_dummy_index + T + nb_domains))

    vector_vars = {i: [(i, 0)] for i in
                   system_indices + observed_temporal_context_nodes + observed_spatial_context_nodes}
    vector_vars[time_dummy_index] = [(i, 0) for i in time_dummy]
    vector_vars[space_dummy_index] = [(i, 0) for i in space_dummy]

    node_classification = {}
    for node in observed_indices:
        node_classification[node - nb_latent_before(node, observed_context_indices, node_classification_gt)] = \
            node_classification_gt[node]
    node_classification[time_dummy_index] = "time_dummy"
    node_classification[space_dummy_index] = "space_dummy"

    dataframe = pp.DataFrame(
        data=data_dict,
        vector_vars=vector_vars,
        analysis_mode='multiple',
    )

    # Get the true parents
    # augment the true_parents by remove context-context links and adding dummy (i.e. perform dummy projection)
    augmented_links = do_dummy_projection(links, node_classification_gt, observed_context_indices,
                                          time_dummy_index, space_dummy_index)
    augmented_true_graph = toys.links_to_graph(augmented_links, tau_max=tau_max)

    return dataframe, augmented_true_graph, augmented_links, tau_min, tau_max, node_classification


def do_dummy_projection(links, node_classification, observed_context_indices, time_dummy_index, space_dummy_index):
    """
    Helper function to augment the true_parents by remove context-context links and adding dummy
    (i.e. perform dummy projection)
    """

    # remove latent links, shift remaining, add dummy
    augmented_links = {}
    for node in node_classification.keys():
        if node_classification[node] == "system":
            keep_parents = []
            for parent in links[node]:
                if node_classification[parent[0][0]] == "system":
                    keep_parents.append(parent)
                elif node_classification[parent[0][0]] == "time_context":
                    if parent[0][0] in observed_context_indices:
                        keep_parents.append(((parent[0][0] - nb_latent_before(parent[0][0], observed_context_indices,
                                                                              node_classification),
                                              parent[0][1]), parent[1], parent[2]))
                    else:
                        keep_parents.append(((time_dummy_index, 0), 0, "dummy"))
                elif node_classification[parent[0][0]] == "space_context":
                    if parent[0][0] in observed_context_indices:
                        keep_parents.append(((parent[0][0] - nb_latent_before(parent[0][0], observed_context_indices,
                                                                              node_classification), parent[0][1]),
                                             parent[1], parent[2]))
                    else:
                        keep_parents.append(((space_dummy_index, 0), 0, "dummy"))
                augmented_links[node] = list(dict.fromkeys(keep_parents))

        # remove all parents of context nodes
        elif node_classification[node] == "time_context":
            if node in observed_context_indices:
                augmented_links[node - nb_latent_before(node, observed_context_indices, node_classification)] = []
        elif node_classification[node] == "space_context":
            if node in observed_context_indices:
                augmented_links[node - nb_latent_before(node, observed_context_indices, node_classification)] = []

    augmented_links[time_dummy_index] = []
    augmented_links[space_dummy_index] = []
    return augmented_links


@pytest.fixture(params=[
    # Keep parameters for the algorithm here
    # pc_alpha, contemp_collider_rule, conflict_resolution, reset, cond_ind_test
    # OracleCI tests
    (0.01, 'majority', True, False, 'oracle_ci'),
    # (0.01, 'none', True, False, 'oracle_ci'),
    # (0.01, 'conservative', True, False, 'oracle_ci'),
    # (0.01, 'majority', False, False, 'oracle_ci'),
    # (0.01, 'none', False, False, 'oracle_ci'),
    # (0.01, 'conservative', False, False, 'oracle_ci'),
    # (0.01, 'majority', True, True, 'oracle_ci'),
    # (0.01, 'none', True, True, 'oracle_ci'),
    # (0.01, 'conservative', True, True, 'oracle_ci'),
    # (0.01, 'majority', False, True, 'oracle_ci'),
    # (0.01, 'none', False, True, 'oracle_ci'),
    # (0.01, 'conservative', False, True, 'oracle_ci'),
])
def a_jpcmciplus_params(request):
    # Return the parameters for the mci test
    return request.param


@pytest.fixture()
def a_run_jpcmciplus(a_jpcmciplus, a_jpcmciplus_params):
    # Unpack the pcmci and the true parents, and common parameters
    dataframe, true_graph, links_coeffs, tau_min, tau_max, node_classification = a_jpcmciplus

    # Unpack the parameters
    (pc_alpha,
     contemp_collider_rule, conflict_resolution,
     reset_lagged_links, cond_ind_test_class,
     ) = a_jpcmciplus_params

    cond_ind_test = OracleCI(links=links_coeffs)
    if cond_ind_test_class == 'par_corr_mult':
        cond_ind_test = ParCorrMult()

    # Run the PCMCI algorithm with the given parameters
    jpcmci = J_PCMCIplus(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=2,
                         node_classification=node_classification, dummy_ci_test=cond_ind_test
                         )
    results = jpcmci.run_jpcmciplus(
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
        max_combinations=10,
    )
    # Print true links
    print("************************")
    print("\nTrue Graph")
    for lag in range(tau_max + 1):
        print(f"Lag {lag} = ")
        print(true_graph[:, :, lag])

    # Print algorithm output
    print("************************")
    print("\nJ-PCMCI+ Output")
    for lag in range(tau_max + 1):
        print(f"Lag {lag} = ")
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
