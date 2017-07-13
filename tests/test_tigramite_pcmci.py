import numpy

# Make Python see modules in parent package
# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tigramite
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr #, GPACE
import tigramite.data_processing as pp
# import tigramite_plotting 

import nose
import nose.tools as nt
# import unittest


def assert_graphs_equal(actual, expected):
    """Check whether lists in dict are equal"""

    for j in list(expected):
        nt.assert_items_equal(actual[j], expected[j])


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

#
#  Start
#
class TestPCMCI():  #unittest.TestCase):
    # def __init__(self):
    #     pass

    def setUp(self):

       auto = .5
       coeff = 0.6
       T = 1000
       numpy.random.seed(42)
       # True graph
       links_coeffs = {0: [((0, -1), auto)],
                       1: [((1, -1), auto), ((0, -1), coeff)],
                       2: [((2, -1), auto), ((1, -1), coeff)]
                       }

       self.data, self.true_parents_coeffs = pp.var_process(links_coeffs, T=T)
       T, N = self.data.shape 

       self.true_parents = _get_parent_graph(self.true_parents_coeffs)

    def test_pcmci(self):
        # Setting up strict test level
        pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        tau_max = 2
        alpha_level = 0.01

        dataframe = pp.DataFrame(self.data)

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

        # print parents
        # print self.true_parents
        assert_graphs_equal(parents, self.true_parents)

    def test_pc_stable(self):

        # Setting up strict test level
        pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        tau_max = 2
        alpha_level = 0.01

        dataframe = pp.DataFrame(self.data)

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
        # print parents
        # print _get_parent_graph(true_parents)
        assert_graphs_equal(parents, self.true_parents)

    def test_pc_stable_selected_links(self):

        # Setting up strict test level
        pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        tau_max = 2
        alpha_level = 0.01

        true_parents_here = {0: [(0, -1)],
                       1: [(1, -1), (0, -1)],
                       2: []
                       }

        dataframe = pp.DataFrame(self.data)

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
        # print parents
        # print _get_parent_graph(true_parents)
        assert_graphs_equal(parents, true_parents_here)


    def test_pc_stable_selected_variables(self):

        # Setting up strict test level
        pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        tau_max = 2
        alpha_level = 0.01

        true_parents_here = {0: [],
                       1: [(1, -1), (0, -1)],
                       2: []
                       }

        dataframe = pp.DataFrame(self.data)

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
        # print parents
        # print _get_parent_graph(true_parents)
        assert_graphs_equal(parents, true_parents_here)


    def test_pc_stable_max_conds_dim(self):

        # Setting up strict test level
        pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        tau_max = 2
        alpha_level = 0.01

        # true_parents_here = {0: [],
        #                1: [(1, -1), (0, -1)],
        #                2: []
        #                }

        dataframe = pp.DataFrame(self.data)

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
        # print parents
        # print _get_parent_graph(true_parents)
        assert_graphs_equal(parents, self.true_parents)


    def test_mci(self):

        # Setting up strict test level
        pc_alpha = 0.05  #[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        tau_max = 2
        alpha_level = 0.01

        dataframe = pp.DataFrame(self.data)

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
                    parents=self.true_parents,
                    max_conds_py=None,
                    max_conds_px=None,
                    )

        parents = pcmci._return_significant_parents(
                                    pq_matrix=results['p_matrix'],
                                  val_matrix=results['val_matrix'],
                                  alpha_level=alpha_level,
                                  )['parents']
        # print parents
        # print _get_parent_graph(true_parents)
        assert_graphs_equal(parents, self.true_parents)



if __name__ == "__main__":
    # unittest.main()

    ## Individual tests
    # test_pcmci = TestPCMCI()
    # test_pcmci.setUp()
    # test_pcmci.test_pc_stable()
    # test_pcmci.test_pcmci()
    nose.main(module=TestPCMCI())
    # nose.run()  #argv=[sys.argv[0],
    #                         test_tigramite_pcmci,
    #                         '-v'])

    # unittest.main()
