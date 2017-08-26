from __future__ import print_function
import unittest
from collections import Counter
import numpy
from nose.tools import assert_equal
import pytest

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPACE
import tigramite.data_processing as pp

def assert_graphs_equal(actual, expected):
    """
    Check whether lists in dict have the same elements.
    This ignore the order of the elements in the list.
    """
    for j in list(expected):
        assert_equal(Counter(iter(actual[j])), Counter(iter(expected[j])))

def _get_parent_graph(nodes, exclude=None):
    """Returns parents"""

    graph = {}
    for j in list(nodes):
        graph[j] = []
        for var, lag in nodes[j]:
            if lag != 0 and (var, lag) != exclude:
                graph[j].append((var, lag))

    return graph


def _get_neighbor_graph(nodes, exclude=None):

    graph = {}
    for j in list(nodes):
        graph[j] = []
        for var, lag in nodes[j]:
            if lag == 0 and (var, lag) != exclude:
                graph[j].append((var, lag))

    return graph


def cmi2parcorr_trafo(cmi):
    return numpy.sqrt(1.-numpy.exp(-2.*cmi))

verbosity = 0

@pytest.fixture()
def a_sample(request):
    # Set the parameters
    auto = 0.5
    coeff = 0.6
    T = 1000
    # Set the random seed
    numpy.random.seed(42)
    # Define the parent-neighghbour relations
    links_coeffs = {0: [((0, -1), auto)],
                    1: [((1, -1), auto), ((0, -1), coeff)],
                    2: [((2, -1), auto), ((1, -1), coeff)]}
    # Generate the data
    data, true_parents_coeffs = pp.var_process(links_coeffs, T=T)
    # Get the true parents
    true_parents = _get_parent_graph(true_parents_coeffs)
    return data, true_parents

def test_pcmci(a_sample):
    # Unpack the test data and true parent graph
    data, true_parents = a_sample
    # Setting up strict test level
    pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tau_max = 2
    alpha_level = 0.01

    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr(
        verbosity=verbosity)

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    results = pcmci.run_pcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
    )

    parents = pcmci._return_significant_parents(
        pq_matrix=results['p_matrix'],
        val_matrix=results['val_matrix'],
        alpha_level=alpha_level)['parents']

    assert_graphs_equal(parents, true_parents)


def test_pc_stable(a_sample):
    # Unpack the test data and true parent graph
    data, true_parents = a_sample
    # Setting up strict test level
    pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tau_max = 2
    alpha_level = 0.01

    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr(
        verbosity=verbosity)

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    pcmci.run_pc_stable( selected_links=None,
                         tau_min=1,
                         tau_max=tau_max,
                         save_iterations=False,
                         pc_alpha=pc_alpha,
                         max_conds_dim=None,
                         max_combinations=1,
                         )

    parents = pcmci.all_parents
    assert_graphs_equal(parents, true_parents)

def test_pc_stable_selected_links(a_sample):
    # Unpack the test data and true parent graph
    data, true_parents = a_sample
    # Setting up strict test level
    pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tau_max = 2
    alpha_level = 0.01

    true_parents_here = {0: [(0, -1)],
                   1: [(1, -1), (0, -1)],
                   2: []
                   }

    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr(
        verbosity=verbosity)

    pcmci = PCMCI(
        selected_variables=None,
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    pcmci.run_pc_stable( selected_links=true_parents_here,
                         tau_min=1,
                         tau_max=tau_max,
                         save_iterations=False,
                         pc_alpha=pc_alpha,
                         max_conds_dim=None,
                         max_combinations=1,
                         )

    parents = pcmci.all_parents
    # print(parents)
    # print(_get_parent_graph(true_parents))
    assert_graphs_equal(parents, true_parents_here)


def test_pc_stable_selected_variables(a_sample):
    # Unpack the test data and true parent graph
    data, true_parents = a_sample
    # Setting up strict test level
    pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tau_max = 2
    alpha_level = 0.01

    true_parents_here = {0: [],
                   1: [(1, -1), (0, -1)],
                   2: []
                   }

    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr(
        verbosity=verbosity)

    pcmci = PCMCI(
        selected_variables=[1],
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    pcmci.run_pc_stable( selected_links=None,
                         tau_min=1,
                         tau_max=tau_max,
                         save_iterations=False,
                         pc_alpha=pc_alpha,
                         max_conds_dim=None,
                         max_combinations=1,
                         )

    parents = pcmci.all_parents
    # print(parents)
    # print(_get_parent_graph(true_parents))
    assert_graphs_equal(parents, true_parents_here)

def test_pc_stable_max_conds_dim(a_sample):
    # Unpack the test data and true parent graph
    data, true_parents = a_sample
    # Setting up strict test level
    pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tau_max = 2
    alpha_level = 0.01

    # true_parents_here = {0: [],
    #                1: [(1, -1), (0, -1)],
    #                2: []
    #                }

    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr(
        verbosity=verbosity)

    pcmci = PCMCI(
        selected_variables=None,
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    pcmci.run_pc_stable( selected_links=None,
                         tau_min=1,
                         tau_max=tau_max,
                         save_iterations=False,
                         pc_alpha=pc_alpha,
                         max_conds_dim=2,
                         max_combinations=1,
                         )

    parents = pcmci.all_parents
    # print(parents)
    # print(_get_parent_graph(true_parents))
    assert_graphs_equal(parents, true_parents)

def test_mci(a_sample):
    # Unpack the test data and true parent graph
    data, true_parents = a_sample
    # Setting up strict test level
    pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    tau_max = 2
    alpha_level = 0.01

    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr(
        verbosity=verbosity)

    pcmci = PCMCI(
        selected_variables=None,
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    results = pcmci.run_mci(
                selected_links=None,
                tau_min=1,
                tau_max=tau_max,
                parents=true_parents,
                max_conds_py=None,
                max_conds_px=None,
                )

    parents = pcmci._return_significant_parents(
                                pq_matrix=results['p_matrix'],
                              val_matrix=results['val_matrix'],
                              alpha_level=alpha_level,
                              )['parents']
    assert_graphs_equal(parents, true_parents)
