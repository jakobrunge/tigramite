"""Tigramite causal discovery for time series."""

# Authors: Wiebke Günther <wiebke.guenther@dlr.de>, Urmi Ninad, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import numpy as np
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from copy import deepcopy
import itertools


class J_PCMCIplus(PCMCI):
    r"""J-PCMCI+ causal discovery for time series datasets from multiple contexts.
        This class is based on the PCMCI framework as described in [1]_.
        J_PCMCIplus enables causal discovery for time series data from different contexts,
        i.e. datasets, where some of the variables describing the context might be unobserved.
        The method is described in detail in [10]_.
        See the tutorial for guidance in applying the method.

        References
        ----------
        .. [1] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic,
               Detecting and quantifying causal associations in large nonlinear time
               series datasets. Sci. Adv. 5, eaau4996 (2019)
               https://advances.sciencemag.org/content/5/11/eaau4996
        .. [10] W. Günther, U. Ninad, J. Runge,
               Causal discovery for time series from multiple datasets with latent contexts. UAI 2023
        Parameters
        ----------
        time_context_nodes : list
            List with entries from {0, ..., N-1} where N is the number of nodes.
            This is the list of the temporal context nodes which are assumed to be constant over the different datasets.
        space_context_nodes :
            List with entries from {0, ..., N-1} where N is the number of nodes.
            This is the list of the spatial context nodes which are assumed to be constant over time.
        time_dummy : int or None (default: None)
            Node corresponding to the temporal dummy variable.
        space_dummy : int or None (default: None)
            Node corresponding to the spatial dummy variable.
        Attributes
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the conditioning-parents estimated with PC algorithm.
        val_min : dictionary
            Dictionary of form val_min[j][(i, -tau)] = float
            containing the minimum test statistic value for each link estimated in
            the PC algorithm.
        pval_max : dictionary
            Dictionary of form pval_max[j][(i, -tau)] = float containing the maximum
            p-value for each link estimated in the PC algorithm.
        iterations : dictionary
            Dictionary containing further information on algorithm steps.
        N : int
            Number of variables.
        T : dict
            Time series sample length of dataset(s).
        context_parents : dictionary or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the dependence of the system nodes on the observed context and dummy nodes.
        observed_context_parents : dictionary or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the dependence of the system nodes on the observed context nodes.
        nb_system_nodes : int
            Number of system nodes.
        nb_context_nodes : int
            Number of context nodes.
        nb_dummy_nodes : int
            Number of dummy nodes.
        dummy_ci_test : conditional independence test object
            Conditional independence test used to test dependence between system nodes and dummy nodes.
            Currently, ParCorr is used with one-hot encoded dummies.
        """

    # TODO: test this assumption:
    # assume context nodes are at the end

    def __init__(self, time_context_nodes, space_context_nodes, time_dummy=None, space_dummy=None, **kwargs):
        self.time_context_nodes = time_context_nodes
        self.space_context_nodes = space_context_nodes
        self.time_dummy = time_dummy
        self.space_dummy = space_dummy

        self.context_parents = None
        self.observed_context_parents = None
        self.dummy_ci_test = ParCorrMult(significance='analytic')
        self.nb_context_nodes = len(self.time_context_nodes) + len(self.space_context_nodes)
        self.nb_dummy_nodes = sum([time_dummy is not None, space_dummy is not None])
        self.mode = "system_search"

        PCMCI.__init__(self, **kwargs)
        self.nb_system_nodes = self.N - self.nb_context_nodes - self.nb_dummy_nodes

    def run_jpcmciplus(self,
                       contemp_collider_rule='majority',
                       link_assumptions=None,
                       tau_min=0,
                       tau_max=2,
                       pc_alpha=0.05):
        """
        Runs J_PCMCIplus time-lagged and contemporaneous causal discovery for time series from multiple contexts.
        Method described in [10]_:
            W. Günther, U. Ninad, J. Runge,
            Causal discovery for time series from multiple datasets with latent contexts. UAI 2023
        Notes
        -----
        The J_PCMCIplus causal discovery method is described in [10]_, where
        also analytical and numerical results are presented. J_PCMCIplus can identify the joint causal graph
        over multiple datasets containing time series data from different contexts under the standard assumptions
        of Causal Sufficiency, Faithfulness and the Markov condition, as well as some background knowledge assumptions.
        J_PCMCIplus estimates time-lagged and contemporaneous causal links from context to system
        variables and in between system variables by a four-step procedure:

        1.  **Discovery of supersets of the lagged parents of the system and observed temporal context nodes** by
        running the :math:`PC_1` lagged phase on this subset of nodes.

        2.  Next, the **MCI test is run on pairs of system and context nodes conditional on subsets of system
        and context**, i.e. perform MCI tests for pairs :math:`((C^j_{t-\\tau}, X^i_t))_{\\tau > 0}`,
        :math:`(C_t^j, X_t^i)`, :math:`(X_t^i, C_t^j)` for all :math:`i,j`,

        .. math:: C_{t-\\tau}^i \\perp X_t^j | \\mathbf{S}, \\hat{\\mathcal{B}}^-_t(X_t^j)
                    \\setminus \\{ C_{t-\\tau}^i \\}, \\hat{\\mathcal{B}}^-_{t-\\tau}(C_{t-\\tau}^i)

        with :math:`\\mathbf{S}` being a subset of the contemporaneous adjacencies :math:`\\mathcal{A}_t(X_t^j)` and
        :math:`\\hat{\\mathcal{B}}^-_t(X_t^j)` are the lagged adjacencies from step one. If :math:`C` is a spatial context variable,
        we only have to test the contemporaneous pairs :math:`(C_t^j, X_t^i)`, :math:`(X_t^i, C_t^j)` for all :math:`i,j`.
        If :math:`C_t^j` and :math:`X_t^i` are conditionally independent, all lagged links between :math:`C_t^j` and
        :math:`X^j_{t-\\tau}` are also removed for all :math:`\\tau`.

        3.  **Perform MCI tests on all system-dummy pairs conditional on the superset of lagged links, the discovered
        contemporaneous context adjacencies, as well as on subsets of contemporaneous system links**, i.e. test
        for :math:`(D, X_t^i)`, :math:`(X_t^i, D)` for all :math:`i`, i.e.

        .. math:: D \\perp X_t^j | \\mathbf{S}, \\hat{\\mathcal{B}}^C_t(X_t^j),

        where :math:`\\mathbf{S} \\subset \\mathcal{A}_t(X_t^i)` and :math:`\\hat{\\mathcal{B}}^C_t(X_t^j)`
        are the lagged and contextual adjacencies found in the previous step.
        If :math:`D` and :math:`X_t^j` are found to be conditionally independence, links between :math:`D` and
        :math:`X^j_{t-\\tau}` are removed for all :math:`\\tau`.
        By assumption context node is the parent in all system-context links.

        4.  Finally, we **perform  MCI tests on all system pairs conditional on discovered lagged, context and dummy
        adjacencies, as well as on subsets of contemporaneous system links** and **orientation phase**. In more detail,
        we perform MCI test for pairs :math:`((X^j_{t-\\tau}, X_t^i))_{\\tau > 0}`, :math:`(X_t^i, X_t^j)` for all
        :math:`i, j`, i.e.

        .. math:: X^i_{t-\\tau} \\perp X_t^j | \\mathbf{S}, \\hat{\\mathcal{B}}^{CD}_t(X_t^j)
                    \\setminus \\{ X_{t-\\tau}^i \\},\\hat{\\mathcal{B}}^{CD}_t(X_{t-\\tau}^i)

        where :math:`\\mathbf{S} \\subset \\mathcal{A}_t(X_t^i)` and :math:`\\hat{\\mathcal{B}}^{CD}_t(X_t^j)`
        are the lagged, contextual, and dummy adjacencies found in the previous steps.
        Finally, all remaining edges (without expert knowledge) are oriented using the PCMCI+ orientation phase while
        making use of all triples involving one context or dummy variable and two system variables as in the non-time
        series case.

        J_PCMCIplus can be flexibly combined with any kind of conditional
        independence test statistic adapted to the kind of data (continuous
        or discrete) and its assumed dependency types. These are available in
        ``tigramite.independence_tests``.
        See PCMCIplus for a description of the parameters of J_PCMCIplus. Also, guidance on best practices for
        setting these parameters is given there.

        Parameters
        ----------
        link_assumptions : dict
            Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
            assumptions about links. This initializes the graph with entries
            graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
            implies that a directed link from i to j at lag 0 must exist.
            Valid link types are 'o-o', '-->', '<--'. In addition, the middle
            mark can be '?' instead of '-'. Then '-?>' implies that this link
            may not exist, but if it exists, its orientation is '-->'. Link
            assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
            requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
            does not appear in the dictionary, it is assumed absent. That is,
            if link_assumptions is not None, then all links have to be specified
            or the links are assumed absent.
        tau_min : int, optional (default: 0)
            Minimum time lag to test.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        pc_alpha : float or list of floats, default: 0.01
            Significance level in algorithm. If a list or None is passed, the
            pc_alpha level is optimized for every graph across the given
            pc_alpha values ([0.001, 0.005, 0.01, 0.025, 0.05] for None) using
            the score computed in cond_ind_test.get_model_selection_criterion().
        contemp_collider_rule : {'majority', 'conservative', 'none'}
            Rule for collider phase to use. See the paper for details. Only
            'majority' and 'conservative' lead to an order-independent
            algorithm.
        conflict_resolution : bool, optional (default: True)
            Whether to mark conflicts in orientation rules. Only for True
            this leads to an order-independent algorithm.
        reset_lagged_links : bool, optional (default: False)
            Restricts the detection of lagged causal links in Step 2 to the
            significant adjacencies found in the PC1 algorithm in Step 1. For
            True, *all* lagged links are considered again, which improves
            detection power for lagged links, but also leads to larger
            runtimes.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.
        max_conds_py : int, optional (default: None)
            Maximum number of lagged conditions of Y to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px : int, optional (default: None)
            Maximum number of lagged conditions of X to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px_lagged : int, optional (default: None)
            Maximum number of lagged conditions of X when X is lagged in MCI
            tests. If None is passed, this number is equal to max_conds_px.
        fdr_method : str, optional (default: 'none')
            Correction method, default is Benjamini-Hochberg False Discovery
            Rate method.
        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values regarding adjacencies.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values regarding adjacencies.
        sepset : dictionary
            Separating sets. See paper for details.
        ambiguous_triples : list
            List of ambiguous triples, only relevant for 'majority' and
            'conservative' rules, see paper for details.
        """
        if self.time_dummy is None:
            dummy_vars = []
        else:
            dummy_vars = [self.time_dummy]

        if self.space_dummy is not None:
            dummy_vars += [self.space_dummy]
        observed_context_nodes = self.time_context_nodes + self.space_context_nodes

        # initialize / clean link_assumptions
        if link_assumptions is not None:
            _link_assumptions = deepcopy(link_assumptions)
        else:
            _link_assumptions = {j: {(i, -tau): 'o?o' for i in range(self.N)
                                     for tau in range(tau_max + 1)} for j in range(self.N)}

        _link_assumptions = self.assume_exogeneous_context(_link_assumptions, observed_context_nodes)
        _link_assumptions = self.clean_link_assumptions(_link_assumptions, tau_max)

        # steps 0 and 1:
        context_system_results = self.discover_lagged_and_context_system_links(
            _link_assumptions,
            tau_min=tau_min,
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )
        context_results = context_system_results
        self.observed_context_parents = context_results['parents']

        if self.verbosity > 0:
            print("Found observed context parents and lagged parents: ", context_results['lagged_context_parents'])

        if self.time_dummy is not None or self.space_dummy is not None:
            # step 2:
            dummy_system_results = self.discover_dummy_system_links(
                _link_assumptions,
                context_system_results,
                tau_min=tau_min,
                tau_max=tau_max,
                pc_alpha=pc_alpha
            )
            context_results = dummy_system_results

        self.mode = "system_search"

        self.context_parents = context_results['parents']
        if self.verbosity > 0:
            print("Discovered contextual parents: ", self.context_parents)

        # step 3:
        results = self.discover_system_system_links(_link_assumptions, context_results, dummy_vars,
                                                    observed_context_nodes, tau_min, tau_max, pc_alpha,
                                                    contemp_collider_rule)

        # Return the dictionary
        return results

    def assume_exogeneous_context(self, link_assumptions, observed_context_nodes):
        """Helper function to amend the link_assumptions to ensure that all context-system links are oriented
        such that the context variable is the parent."""
        for j in link_assumptions:
            if j in range(self.nb_system_nodes):
                for link in link_assumptions[j]:
                    i, lag = link
                    if i in observed_context_nodes + [self.time_dummy, self.space_dummy]:  # is context var
                        link_type = link_assumptions[j][link]
                        link_assumptions[j][link] = '-' + link_type[1] + '>'
        return link_assumptions

    def clean_link_assumptions(self, link_assumptions, tau_max):
        """Helper function to amend the link_assumptions in the following ways
            * remove any links where dummy is the child
            * remove any lagged links to dummy, and space_context (not to expressive time context)
            * and system - context links where context is the child
            * and any links between spatial and temporal context
        """
        if self.time_dummy is not None: link_assumptions[self.time_dummy] = {}
        if self.space_dummy is not None: link_assumptions[self.space_dummy] = {}

        for j in range(self.nb_system_nodes + self.nb_context_nodes):
            for lag in range(1, tau_max + 1):
                for c in [self.time_dummy, self.space_dummy] + self.space_context_nodes:
                    if (c, -lag) in link_assumptions[j]: link_assumptions[j].pop((c, -lag), None)
        for c in self.space_context_nodes + self.time_context_nodes:
            for j in range(self.nb_system_nodes):
                for lag in range(tau_max + 1):
                    if (j, -lag) in link_assumptions[c]: link_assumptions[c].pop((j, -lag), None)
            if (c, 0) in link_assumptions[c]: link_assumptions[c].pop((c, 0), None)  # remove self-links

        for c in self.space_context_nodes:
            for k in self.time_context_nodes:
                for lag in range(tau_max + 1):
                    if (k, -lag) in link_assumptions[c]: link_assumptions[c].pop((k, -lag), None)
        for c in self.time_context_nodes:
            for k in self.space_context_nodes:
                if (k, 0) in link_assumptions[c]: link_assumptions[c].pop((k, 0), None)

        return link_assumptions

    def remove_dummy_link_assumptions(self, link_assumptions):
        """Helper function to remove any links to dummy from link_assumptions."""
        link_assumptions_wo_dummy = deepcopy(link_assumptions)
        for j in range(self.nb_system_nodes + self.nb_context_nodes):
            if (self.time_dummy, 0) in link_assumptions_wo_dummy[j]:
                link_assumptions_wo_dummy[j].pop((self.time_dummy, 0), None)
            if (self.space_dummy, 0) in link_assumptions_wo_dummy[j]:
                link_assumptions_wo_dummy[j].pop((self.space_dummy, 0), None)
        return link_assumptions_wo_dummy

    def remove_obs_context_link_assumptions(self, link_assumptions, tau_max):
        """Helper function to remove any links to observed context nodes from link_assumptions."""
        link_assumptions_wo_obs_context = deepcopy(link_assumptions)

        for c in self.space_context_nodes + self.time_context_nodes:
            link_assumptions_wo_obs_context[c] = {}
        for j in range(self.nb_system_nodes):
            for c in self.space_context_nodes + self.time_context_nodes:
                for lag in range(tau_max + 1):
                    if (c, -lag) in link_assumptions_wo_obs_context[j]:
                        link_assumptions_wo_obs_context[j].pop((c, -lag), None)
        return link_assumptions_wo_obs_context

    def clean_system_link_assumptions(self, link_assumptions, dummy_vars, observed_context_nodes, tau_max):
        """Helper function to remove any links to dummy and observed context nodes from link_assumptions.
            Add discovered links to contextual parents (from steps 1 and 2) to the link_assumptions.
        """
        system_links = deepcopy(link_assumptions)

        for j in range(self.nb_system_nodes):
            for C in dummy_vars + observed_context_nodes:
                for lag in range(tau_max + 1):
                    if (C, -lag) in system_links[j]:
                        system_links[j].pop((C, -lag), None)

        for j in system_links:
            system_links[j].update({parent: '-->' for parent in self.context_parents[j]})

        for C in observed_context_nodes + dummy_vars:
            # we are not interested in links between context variables (thus system_links[C] = {})
            system_links[C] = {}
        return system_links

    def discover_lagged_and_context_system_links(self, link_assumptions, tau_min=0, tau_max=1, pc_alpha=0.01):
        """
        TODO: add description
        **bla**

        Parameters
        ----------
        link_assumptions
        tau_min
        tau_max
        pc_alpha

        Returns
        -------
        lagged_context_parents : dictionary
        values : dictionary
        parents : dictionary
        """
        # Initializing
        parents = {j: [] for j in range(self.nb_system_nodes + self.nb_context_nodes + self.nb_dummy_nodes)}
        lagged_context_parents = {j: [] for j in
                                  range(self.nb_system_nodes + self.nb_context_nodes + self.nb_dummy_nodes)}
        values = np.zeros((self.nb_system_nodes + self.nb_context_nodes + 2,
                           self.nb_system_nodes + self.nb_context_nodes + 2, tau_max + 1))

        # find links in btw expressive context, and btw expressive context and sys_vars
        # here, we exclude any links to dummy
        _link_assumptions_wo_dummy = self.remove_dummy_link_assumptions(link_assumptions)

        self.mode = "context_search"
        print("##### Discovering context-system links #####")
        # run PCMCI+ on subset of links to discover context-system links
        # (we use simple v-structure orientation rules since the orientation phase is not important at this step)
        skeleton_results = self.run_pcmciplus(
            tau_min=tau_min,
            tau_max=tau_max,
            link_assumptions=_link_assumptions_wo_dummy,
            contemp_collider_rule=None,
            pc_alpha=pc_alpha)
        skeleton_val = skeleton_results['val_matrix']

        self.mode = "system_search"
        skeleton_graph = skeleton_results['graph']

        for j in range(self.nb_system_nodes + self.nb_context_nodes):
            for c in self.space_context_nodes + self.time_context_nodes:
                for k in range(tau_max + 1):
                    if skeleton_graph[c, j, k] == 'o-o' or skeleton_graph[c, j, k] == '-->':
                        parents[j].append((c, -k))
                        lagged_context_parents[j].append((c, -k))
                        values[c, j, k] = skeleton_val[c, j, k]
            for i in range(self.nb_system_nodes):
                for k in range(tau_max + 1):
                    if skeleton_graph[i, j, k] == 'o-o' or skeleton_graph[i, j, k] == '-->':
                        lagged_context_parents[j].append((i, -k))

        return {'lagged_context_parents': lagged_context_parents,
                'values': values,
                'parents': parents
                }

    def discover_dummy_system_links(self, link_assumptions, context_system_results, tau_min=0, tau_max=1,
                                    pc_alpha=0.01):
        """

        Parameters
        ----------
        link_assumptions
        context_system_results
        tau_min
        tau_max
        pc_alpha

        Returns
        -------
        lagged_context_parents : dictionary
        values : dictionary
        parents : dictionary
        """

        lagged_context_parents = context_system_results['lagged_context_parents']
        values = context_system_results['values']
        parents = context_system_results['parents']

        # setup link assumptions without the observed context nodes
        _link_assumptions_wo_obs_context = self.remove_obs_context_link_assumptions(link_assumptions, tau_max)

        self.mode = "dummy_context_search"
        print("#### Discovering dummy-system links ####")
        # run PC algorithm to find links between dummies and system variables
        """
        _int_link_assumptions = self._set_link_assumptions(_link_assumptions_wo_obs_context, tau_min, tau_max)
        links_for_pc = {}
        for j in range(self.N):
            links_for_pc[j] = {}
            for parent in lagged_context_parents[j]:
                if parent in _int_link_assumptions[j].keys() and _int_link_assumptions[j][parent] in ['-?>', '-->']:
                    links_for_pc[j][parent] = _int_link_assumptions[j][parent]

            # Add contemporaneous links
            for link in _int_link_assumptions[j]:
                i, tau = link
                link_type = _int_link_assumptions[j][link]
                if abs(tau) == 0:
                    links_for_pc[j][(i, 0)] = link_type

        initial_graph = self._dict_to_graph(links_for_pc, tau_max)

        print("links_for_pc", links_for_pc)
        skeleton_results_dummy = self._pcalg_skeleton(
            initial_graph=initial_graph,
            lagged_parents=lagged_context_parents,
            pc_alpha=pc_alpha,
            mode='contemp_conds',
            tau_min=tau_min,
            tau_max=tau_max,
            max_conds_dim=self.N,
            max_combinations=np.inf,
            max_conds_py=None,
            max_conds_px=None,
            max_conds_px_lagged=None,
        )
        skeleton_graph_dummy = skeleton_results_dummy['graph']
        skeleton_graph_dummy[skeleton_graph_dummy=='o?o'] = 'o-o'
        skeleton_graph_dummy[skeleton_graph_dummy=='-?>'] = '-->'
        skeleton_graph_dummy[skeleton_graph_dummy=='<?-'] = '<--'
        """

        skeleton_results_dummy = self.run_pcmciplus_phase234(
            lagged_parents=lagged_context_parents,
            tau_min=tau_min,
            tau_max=tau_max,
            contemp_collider_rule=None,
            link_assumptions=_link_assumptions_wo_obs_context,
            pc_alpha=pc_alpha)

        skeleton_graph_dummy = skeleton_results_dummy['graph']
        skeleton_val_dummy = skeleton_results_dummy['val_matrix']

        for j in range(self.nb_system_nodes):
            for k in range(tau_max + 1):
                if skeleton_graph_dummy[self.time_dummy, j, k] == 'o-o' or \
                        skeleton_graph_dummy[self.time_dummy, j, k] == '-->':
                    parents[j].append((self.time_dummy, k))
                    values[self.time_dummy, j, k] = skeleton_val_dummy[self.time_dummy, j, k]
                    lagged_context_parents[j].append((self.time_dummy, k))
                if skeleton_graph_dummy[self.space_dummy, j, k] == 'o-o' or \
                        skeleton_graph_dummy[self.space_dummy, j, k] == '-->':
                    parents[j].append((self.space_dummy, k))
                    lagged_context_parents[j].append((self.space_dummy, k))
                    values[self.space_dummy, j, k] = skeleton_val_dummy[self.space_dummy, j, k]
        return {'lagged_context_parents': lagged_context_parents,
                'values': values,
                'parents': parents
                }

    def discover_system_system_links(self, link_assumptions, context_results, dummy_vars,
                                     observed_context_nodes, tau_min=0, tau_max=1, pc_alpha=0.01,
                                     contemp_collider_rule="majority"):
        """
        Step 4 and orientation phase
        TODO: add description

        Parameters
        ----------
        link_assumptions
        context_results
        dummy_vars
        observed_context_nodes
        tau_min
        tau_max
        pc_alpha
        contemp_collider_rule

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values regarding adjacencies.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values regarding adjacencies.
        sepset : dictionary
            Separating sets. See paper for details.
        ambiguous_triples : list
            List of ambiguous triples, only relevant for 'majority' and
            'conservative' rules, see paper for details.
        """
        lagged_context_parents = context_results['lagged_context_parents']
        context_parents_values = context_results['values']

        # Get the parents from run_pc_stable only on the system links
        system_links = self.clean_system_link_assumptions(link_assumptions, dummy_vars, observed_context_nodes, tau_max)

        print("#### Discovering system-system links ####")
        self.mode = "system_search"
        results = self.run_pcmciplus_phase234(
            lagged_parents=lagged_context_parents,
            tau_min=tau_min,
            tau_max=tau_max,
            contemp_collider_rule=contemp_collider_rule,
            link_assumptions=system_links,
            pc_alpha=pc_alpha)

        for c in observed_context_nodes + dummy_vars:
            for j in list(range(self.nb_system_nodes)) + observed_context_nodes + dummy_vars:
                for lag in range(tau_max + 1):
                    # add context-system links to results
                    results['val_matrix'][c, j, lag] = context_parents_values[c, j, lag]
                    results['val_matrix'][j, c, lag] = context_parents_values[c, j, lag]

        # Return the dictionary
        return results

    def _remaining_pairs(self, graph, adjt, tau_min, tau_max, p):
        """Helper function returning the remaining pairs that still need to be
        tested depending on the J_PCMCIplus step, i.e. discovery of context-system links (step 1),
        dummy-context links (step 2) or system-system links in which case the function of the parent class is called.
        """
        all_context_nodes = self.time_context_nodes + self.space_context_nodes
        if self.mode == "context_search":
            # during discovery of context-system links we are only
            # interested in context-context and context-system pairs
            N = graph.shape[0]
            pairs = []
            for (i, j) in itertools.product(range(N), range(N)):
                for abstau in range(tau_min, tau_max + 1):
                    if (graph[i, j, abstau] != ""
                            and len(
                                [a for a in adjt[j] if a != (i, -abstau)]) >= p
                            and i in all_context_nodes):
                        pairs.append((i, j, abstau))
            return pairs
        elif self.mode == "dummy_context_search":
            # during discovery of dummy-system links we are only
            # interested in dummy-system pairs
            N = graph.shape[0]
            pairs = []
            for (i, j) in itertools.product(range(N), range(N)):
                for abstau in range(tau_min, tau_max + 1):
                    if (graph[i, j, abstau] != ""
                            and len(
                                [a for a in adjt[j] if a != (i, -abstau)]) >= p
                            and i in [self.time_dummy, self.space_dummy]):
                        pairs.append((i, j, abstau))
            return pairs
        else:
            return super()._remaining_pairs(graph, adjt, tau_min, tau_max, p)

    def _run_pcalg_test(self, graph, i, abstau, j, S, lagged_parents, max_conds_py,
                        max_conds_px, max_conds_px_lagged, tau_max):
        """MCI conditional independence tests within PCMCIplus or PC algorithm. Depending on the J_PCMCIplus step
        the setup is adapted slightly. During the discovery of dummy-system links (step 2) we are using
        the dummy_ci_test and condition on the parents found during step 1; during the discovery of system-system links
        (step 3) we are conditioning on the found contextual parents.
        Parameters
        ----------
        graph : array
            ...
        i : int
            Variable index.
        abstau : int
            Time lag (absolute value).
        j : int
            Variable index.
        S : list
            List of contemporaneous conditions.
        lagged_parents : dictionary of lists
            Dictionary of lagged parents for each node.
        max_conds_py : int
            Max number of lagged parents for node j.
        max_conds_px : int
            Max number of lagged parents for lagged node i.
        max_conds_px_lagged : int
            Maximum number of lagged conditions of X when X is lagged in MCI
            tests. If None is passed, this number is equal to max_conds_px.
        tau_max : int
            Maximum time lag.
        Returns
        -------
        val : float
            Test statistic value.
        pval : float
            Test statistic p-value.
        Z : list
            List of conditions.
        """
        if self.mode == 'dummy_context_search':
            # during discovery of dummy-system links we are using the dummy_ci_test and condition on the found
            # contextual parents from step 1.
            if lagged_parents is None:
                cond = list(S)
            else:
                cond = list(S) + lagged_parents[j]
            context_parents_j = self.observed_context_parents[j]
            cond = cond + context_parents_j
            cond = list(dict.fromkeys(cond))

            return self.run_test_dummy([(j, -abstau)], [(i, 0)], cond, tau_max)
        elif self.mode == 'system_search':
            # during discovery of system-system links we are conditioning on the found contextual parents
            if self.time_dummy is None:
                dummy_vars = []
            else:
                dummy_vars = [(self.time_dummy, 0)]

            if self.space_dummy is not None:
                dummy_vars += [(self.space_dummy, 0)]

            if lagged_parents is None:
                cond = list(S)
            else:
                cond = list(S) + lagged_parents[j]
            # always add self.obs_context_parents
            context_parents_j = self.context_parents[j]
            cond = cond + context_parents_j
            cond = list(dict.fromkeys(cond))
            return super()._run_pcalg_test(graph, i, abstau, j, cond, lagged_parents, max_conds_py,
                                           max_conds_px, max_conds_px_lagged, tau_max)
        else:
            return super()._run_pcalg_test(graph, i, abstau, j, S, lagged_parents, max_conds_py,
                                           max_conds_px, max_conds_px_lagged, tau_max)

    def run_test_dummy(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """Helper function to deal with difficulties in constructing the array for CI testing
        that arise due to the one-hot encoding of the dummy."""
        # Get the array to test on
        array, xyz, XYZ, _ = self.dataframe.construct_array(X=X, Y=Y, Z=Z, tau_max=tau_max,
                                                            mask_type=self.dummy_ci_test.mask_type,
                                                            return_cleaned_xyz=True,
                                                            do_checks=True,
                                                            remove_overlaps=True,
                                                            cut_off=cut_off,
                                                            verbosity=0)
        # remove the parts of the array within dummy that are constant zero (ones are cut off)
        mask = np.all(array == 0., axis=1) | np.all(array == 1., axis=1)
        xyz = xyz[~mask]
        array = array[~mask]

        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.any(np.isnan(array)):
            raise ValueError("nans in the array!")

        # combined_hash = self.cond_ind_test._get_array_hash(array, xyz, XYZ)

        if False:  # combined_hash in self.cond_ind_test.cached_ci_results.keys():
            cached = True
            val, pval = self.cond_ind_test.cached_ci_results[combined_hash]
        else:
            cached = False
            # Get the dependence measure, recycling residuals if need be
            val = self.dummy_ci_test._get_dependence_measure_recycle(X, Y, Z, xyz, array)

            # Get the p-value
            pval = self.dummy_ci_test.get_significance(val, array, xyz, T, dim)

        if self.verbosity > 1:
            self.dummy_ci_test._print_cond_ind_results(val=val, pval=pval, cached=cached, conf=None)
        # Return the value and the p-value
        return val, pval, Z

    def run_pcmciplus_phase234(self, lagged_parents, selected_links=None, link_assumptions=None, tau_min=0, tau_max=1,
                               pc_alpha=0.01,
                               contemp_collider_rule='majority',
                               conflict_resolution=True,
                               reset_lagged_links=False,
                               max_conds_dim=None,
                               max_conds_py=None,
                               max_conds_px=None,
                               max_conds_px_lagged=None,
                               fdr_method='none',
                               ):
        """Runs PCMCIplus time-lagged and contemporaneous causal discovery for
                time series without its first phase.
                Method described in [5]_:
                http://www.auai.org/~w-auai/uai2020/proceedings/579_main_paper.pdf
                Parameters
                ----------
                lagged_parents : dictionary of lists
                    Dictionary of lagged parents for each node.
                selected_links : dict or None
                    Deprecated, replaced by link_assumptions
                link_assumptions : dict
                    Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
                    assumptions about links. This initializes the graph with entries
                    graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
                    implies that a directed link from i to j at lag 0 must exist.
                    Valid link types are 'o-o', '-->', '<--'. In addition, the middle
                    mark can be '?' instead of '-'. Then '-?>' implies that this link
                    may not exist, but if it exists, its orientation is '-->'. Link
                    assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
                    requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
                    does not appear in the dictionary, it is assumed absent. That is,
                    if link_assumptions is not None, then all links have to be specified
                    or the links are assumed absent.
                tau_min : int, optional (default: 0)
                    Minimum time lag to test.
                tau_max : int, optional (default: 1)
                    Maximum time lag. Must be larger or equal to tau_min.
                pc_alpha : float or list of floats, default: 0.01
                    Significance level in algorithm. If a list or None is passed, the
                    pc_alpha level is optimized for every graph across the given
                    pc_alpha values ([0.001, 0.005, 0.01, 0.025, 0.05] for None) using
                    the score computed in cond_ind_test.get_model_selection_criterion().
                contemp_collider_rule : {'majority', 'conservative', 'none'}
                    Rule for collider phase to use. See the paper for details. Only
                    'majority' and 'conservative' lead to an order-independent
                    algorithm.
                conflict_resolution : bool, optional (default: True)
                    Whether to mark conflicts in orientation rules. Only for True
                    this leads to an order-independent algorithm.
                reset_lagged_links : bool, optional (default: False)
                    Restricts the detection of lagged causal links in Step 2 to the
                    significant adjacencies found in the PC1 algorithm in Step 1. For
                    True, *all* lagged links are considered again, which improves
                    detection power for lagged links, but also leads to larger
                    runtimes.
                max_conds_dim : int, optional (default: None)
                    Maximum number of conditions to test. If None is passed, this number
                    is unrestricted.
                max_combinations : int, optional (default: 1)
                    Maximum number of combinations of conditions of current cardinality
                    to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
                    a larger number, such as 10, can be used.
                max_conds_py : int, optional (default: None)
                    Maximum number of lagged conditions of Y to use in MCI tests. If
                    None is passed, this number is unrestricted.
                max_conds_px : int, optional (default: None)
                    Maximum number of lagged conditions of X to use in MCI tests. If
                    None is passed, this number is unrestricted.
                max_conds_px_lagged : int, optional (default: None)
                    Maximum number of lagged conditions of X when X is lagged in MCI
                    tests. If None is passed, this number is equal to max_conds_px.
                fdr_method : str, optional (default: 'none')
                    Correction method, default is Benjamini-Hochberg False Discovery
                    Rate method.
                Returns
                -------
                graph : array of shape [N, N, tau_max+1]
                    Resulting causal graph, see description above for interpretation.
                val_matrix : array of shape [N, N, tau_max+1]
                    Estimated matrix of test statistic values regarding adjacencies.
                p_matrix : array of shape [N, N, tau_max+1]
                    Estimated matrix of p-values regarding adjacencies.
                sepset : dictionary
                    Separating sets. See paper for details.
                ambiguous_triples : list
                    List of ambiguous triples, only relevant for 'majority' and
                    'conservative' rules, see paper for details.
                """

        if selected_links is not None:
            raise ValueError("selected_links is DEPRECATED, use link_assumptions instead.")

        # Check if pc_alpha is chosen to optimize over a list
        if pc_alpha is None or isinstance(pc_alpha, (list, tuple, np.ndarray)):
            # Call optimizer wrapper around run_pcmciplus()
            return self._optimize_pcmciplus_alpha(
                link_assumptions=link_assumptions,
                tau_min=tau_min,
                tau_max=tau_max,
                pc_alpha=pc_alpha,
                contemp_collider_rule=contemp_collider_rule,
                conflict_resolution=conflict_resolution,
                reset_lagged_links=reset_lagged_links,
                max_conds_dim=max_conds_dim,
                max_conds_py=max_conds_py,
                max_conds_px=max_conds_px,
                max_conds_px_lagged=max_conds_px_lagged,
                fdr_method=fdr_method)

        # else:
        #     raise ValueError("pc_alpha=None not supported in PCMCIplus, choose"
        #                      " 0 < pc_alpha < 1 (e.g., 0.01)")

        if pc_alpha < 0. or pc_alpha > 1:
            raise ValueError("Choose 0 <= pc_alpha <= 1")

        # For the lagged PC algorithm only the strongest conditions are tested
        max_combinations = 1

        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        # _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
        _int_link_assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)

        p_matrix = self.p_matrix
        val_matrix = self.val_matrix

        # Step 2+3+4: PC algorithm with contemp. conditions and MCI tests
        if self.verbosity > 0:
            print("\n##\n## Step 2: PC algorithm with contemp. conditions "
                  "and MCI tests\n##"
                  "\n\nParameters:")
            if link_assumptions is not None:
                print("\nlink_assumptions = %s" % str(_int_link_assumptions))
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\ncontemp_collider_rule = %s" % contemp_collider_rule
                  + "\nconflict_resolution = %s" % conflict_resolution
                  + "\nreset_lagged_links = %s" % reset_lagged_links
                  + "\nmax_conds_dim = %s" % max_conds_dim
                  + "\nmax_conds_py = %s" % max_conds_py
                  + "\nmax_conds_px = %s" % max_conds_px
                  + "\nmax_conds_px_lagged = %s" % max_conds_px_lagged
                  + "\nfdr_method = %s" % fdr_method
                  )

        # Set the maximum condition dimension for Y and X
        max_conds_py = self._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = self._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)

        if reset_lagged_links:
            # Run PCalg on full graph, ignoring that some lagged links
            # were determined as non-significant in PC1 step
            links_for_pc = deepcopy(_int_link_assumptions)
        else:
            # Run PCalg only on lagged parents found with PC1
            # plus all contemporaneous links
            links_for_pc = {}  # deepcopy(lagged_parents)
            for j in range(self.N):
                links_for_pc[j] = {}
                for parent in lagged_parents[j]:
                    if parent in _int_link_assumptions[j] and _int_link_assumptions[j][parent] in ['-?>', '-->']:
                        links_for_pc[j][parent] = _int_link_assumptions[j][parent]
                # Add Contemporaneous links
                for link in _int_link_assumptions[j]:
                    i, tau = link
                    link_type = _int_link_assumptions[j][link]
                    if abs(tau) == 0:
                        links_for_pc[j][(i, 0)] = link_type

        # self.mode = "standard"
        results = self.run_pcalg(
            link_assumptions=links_for_pc,
            pc_alpha=pc_alpha,
            tau_min=tau_min,
            tau_max=tau_max,
            max_conds_dim=max_conds_dim,
            max_combinations=None,
            lagged_parents=lagged_parents,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            max_conds_px_lagged=max_conds_px_lagged,
            mode='contemp_conds',
            contemp_collider_rule=contemp_collider_rule,
            conflict_resolution=conflict_resolution)

        graph = results['graph']

        # Update p_matrix and val_matrix with values from links_for_pc
        for j in range(self.N):
            for link in links_for_pc[j]:
                i, tau = link
                if links_for_pc[j][link] not in ['<--', '<?-']:
                    p_matrix[i, j, abs(tau)] = results['p_matrix'][i, j, abs(tau)]
                    val_matrix[i, j, abs(tau)] = results['val_matrix'][i, j,
                    abs(tau)]

        # Update p_matrix and val_matrix for indices of symmetrical links
        p_matrix[:, :, 0] = results['p_matrix'][:, :, 0]
        val_matrix[:, :, 0] = results['val_matrix'][:, :, 0]

        ambiguous = results['ambiguous_triples']

        conf_matrix = None
        # TODO: implement confidence estimation, but how?
        # if self.cond_ind_test.confidence is not False:
        #     conf_matrix = results['conf_matrix']

        # Correct the p_matrix if there is a fdr_method
        if fdr_method != 'none':
            p_matrix = self.get_corrected_pvalues(p_matrix=p_matrix, tau_min=tau_min,
                                                  tau_max=tau_max,
                                                  link_assumptions=_int_link_assumptions,
                                                  fdr_method=fdr_method)

        # Store the parents in the pcmci member
        self.all_lagged_parents = lagged_parents

        # Cache the resulting values in the return dictionary
        return_dict = {'graph': graph,
                       'val_matrix': val_matrix,
                       'p_matrix': p_matrix,
                       'ambiguous_triples': ambiguous,
                       'conf_matrix': conf_matrix}
        # Print the results
        if self.verbosity > 0:
            self.print_results(return_dict, alpha_level=pc_alpha)
        # Return the dictionary
        return return_dict
