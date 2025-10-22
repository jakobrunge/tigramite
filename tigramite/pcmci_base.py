"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import scipy.stats
import math
from joblib import Parallel, delayed

class PCMCIbase():
    r"""PCMCI base class.

    Parameters
    ----------
    dataframe : data object
        This is the Tigramite dataframe object. Among others, it has the
        attributes dataframe.values yielding a numpy array of shape (
        observations T, variables N) and optionally a mask of the same shape.
    cond_ind_test : conditional independence test object
        This can be ParCorr or other classes from
        ``tigramite.independence_tests`` or an external test passed as a
        callable. This test can be based on the class
        tigramite.independence_tests.CondIndTest.
    verbosity : int, optional (default: 0)
        Verbose levels 0, 1, ...

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
    """

    def __init__(self, dataframe,
                 cond_ind_test,
                 verbosity=0):
        # Set the data for this iteration of the algorithm
        self.dataframe = dataframe
        # Set the conditional independence test to be used
        self.cond_ind_test = deepcopy(cond_ind_test)
        if isinstance(self.cond_ind_test, type):
            raise ValueError("PCMCI requires that cond_ind_test "
                             "is instantiated, e.g. cond_ind_test =  "
                             "ParCorr().")
        self.cond_ind_test.set_dataframe(self.dataframe)
        # Set the verbosity for debugging/logging messages
        self.verbosity = verbosity
        # Set the variable names 
        self.var_names = self.dataframe.var_names

        # Store the shape of the data in the T and N variables
        self.T = self.dataframe.T
        self.N = self.dataframe.N


    def _reverse_link(self, link):
        """Reverse a given link, taking care to replace > with < and vice versa."""

        if link == "":
            return ""

        if link[2] == ">":
            left_mark = "<"
        else:
            left_mark = link[2]

        if link[0] == "<":
            right_mark = ">"
        else:
            right_mark = link[0]

        return left_mark + link[1] + right_mark

    def _check_cyclic(self, link_dict):
        """Return True if the link_dict has a contemporaneous cycle.

        """

        path = set()
        visited = set()

        def visit(vertex):
            if vertex in visited:
                return False
            visited.add(vertex)
            path.add(vertex)
            for itaui in link_dict.get(vertex, ()):
                i, taui = itaui
                link_type = link_dict[vertex][itaui] 
                if taui == 0 and link_type in ['-->', '-?>']:
                    if i in path or visit(i):
                        return True
            path.remove(vertex)
            return False

        return any(visit(v) for v in link_dict)

    def _set_link_assumptions(self, link_assumptions, tau_min, tau_max,
                       remove_contemp=False):
        """Helper function to set and check the link_assumptions argument

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
        tau_mix : int
            Minimum time delay to test.
        tau_max : int
            Maximum time delay to test.
        remove_contemp : bool
            Whether contemporaneous links (at lag zero) should be removed.

        Returns
        -------
        link_assumptions : dict
            Cleaned links.
        """
        # Copy and pass into the function
        _int_link_assumptions = deepcopy(link_assumptions)
        # Set the default selected links if none are set
        _vars = list(range(self.N))
        _lags = list(range(-(tau_max), -tau_min + 1, 1))
        if _int_link_assumptions is None:
            _int_link_assumptions = {}
            # Set the default as all combinations
            for j in _vars:
                _int_link_assumptions[j] = {}
                for i in _vars:
                    for lag in range(tau_min, tau_max + 1):
                        if not (i == j and lag == 0):
                            if lag == 0:
                                _int_link_assumptions[j][(i, 0)] = 'o?o'
                            else:
                                _int_link_assumptions[j][(i, -lag)] = '-?>'
  
        else:

            if remove_contemp:
                for j in _int_link_assumptions.keys():
                    _int_link_assumptions[j] = {link:_int_link_assumptions[j][link] 
                                        for link in _int_link_assumptions[j]
                                         if link[1] != 0}

        # Make contemporaneous assumptions consistent and orient lagged links
        for j in _vars:
            for link in _int_link_assumptions[j]:
                i, tau = link
                link_type = _int_link_assumptions[j][link]
                if tau == 0:
                    if (j, 0) in _int_link_assumptions[i]:
                        if _int_link_assumptions[j][link] != self._reverse_link(_int_link_assumptions[i][(j, 0)]):
                            raise ValueError("Inconsistent link assumptions for indices %d - %d " %(i, j))
                    else:
                        _int_link_assumptions[i][(j, 0)] = self._reverse_link(_int_link_assumptions[j][link])
                else:
                    # Orient lagged links by time order while leaving the middle mark
                    new_link_type = '-' + link_type[1] + '>'
                    _int_link_assumptions[j][link] = new_link_type

        # Otherwise, check that our assumpions are sane
        # Check that the link_assumptions refer to links that are inside the
        # data range and types
        _key_set = set(_int_link_assumptions.keys())
        valid_entries = _key_set == set(range(self.N))

        valid_types = [
                    'o-o',
                    'o?o',
                    '-->',
                    '-?>',
                    '<--',
                    '<?-',
                        ]

        for links in _int_link_assumptions.values():
            if isinstance(links, dict) and len(links) == 0:
                continue
            for var, lag in links:
                if var not in _vars or lag not in _lags:
                    valid_entries = False
                if links[(var, lag)] not in valid_types:
                    valid_entries = False


        if not valid_entries:
            raise ValueError("link_assumptions"
                             " must be dictionary with keys for all [0,...,N-1]"
                             " variables and contain only links from "
                             "these variables in range [tau_min, tau_max] "
                             "and with link types in %s" %str(valid_types))

        # Check for contemporaneous cycles
        if self._check_cyclic(_int_link_assumptions):
            raise ValueError("link_assumptions has contemporaneous cycle(s).")

        # Return the _int_link_assumptions
        return _int_link_assumptions

    def _dict_to_matrix(self, val_dict, tau_max, n_vars, default=1):
        """Helper function to convert dictionary to matrix format.

        Parameters
        ---------
        val_dict : dict
            Dictionary of form {0:{(0, -1):float, ...}, 1:{...}, ...}.
        tau_max : int
            Maximum lag.
        n_vars : int
            Number of variables.
        default : int
            Default value for entries not part of val_dict.

        Returns
        -------
        matrix : array of shape (N, N, tau_max+1)
            Matrix format of p-values and test statistic values.
        """
        matrix = np.ones((n_vars, n_vars, tau_max + 1))
        matrix *= default

        for j in val_dict.keys():
            for link in val_dict[j].keys():
                k, tau = link
                if tau == 0:
                    matrix[k, j, 0] = matrix[j, k, 0] = val_dict[j][link]
                else:
                    matrix[k, j, abs(tau)] = val_dict[j][link]
        return matrix


    def get_corrected_pvalues(self, p_matrix,
                              fdr_method='fdr_bh',
                              exclude_contemporaneous=True,
                              tau_min=0,
                              tau_max=1,
                              link_assumptions=None,
                              ):
        """Returns p-values corrected for multiple testing.

        Currently implemented is Benjamini-Hochberg False Discovery Rate
        method. Correction is performed either among all links if
        exclude_contemporaneous==False, or only among lagged links.

        Parameters
        ----------
        p_matrix : array-like
            Matrix of p-values. Must be of shape (N, N, tau_max + 1).
        tau_min : int, default: 0
            Minimum time lag. Only used as consistency check of link_assumptions. 
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min. Only used as 
            consistency check of link_assumptions. 
        link_assumptions : dict or None
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
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method.     
        exclude_contemporaneous : bool, optional (default: True)
            Whether to include contemporaneous links in correction.

        Returns
        -------
        q_matrix : array-like
            Matrix of shape (N, N, tau_max + 1) containing corrected p-values.
        """

        def _ecdf(x):
            """No frills empirical cdf used in fdr correction.
            """
            nobs = len(x)
            return np.arange(1, nobs + 1) / float(nobs)

        # Get the shape parameters from the p_matrix
        _, N, tau_max_plusone = p_matrix.shape
        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Include only link_assumptions if given
        if link_assumptions != None:
            # Create a mask for these values
            mask = np.zeros((N, N, tau_max_plusone), dtype='bool')
            _int_link_assumptions = self._set_link_assumptions(link_assumptions, tau_min, tau_max)
            for j, links_ in _int_link_assumptions.items():
                for link in links_:
                    i, lag = link
                    if _int_link_assumptions[j][link] not in ["<--", "<?-"]:    
                        mask[i, j, abs(lag)] = True
        else:
            # Create a mask for these values
            mask = np.ones((N, N, tau_max_plusone), dtype='bool')
        # Ignore values from lag-zero 'autocorrelation' indices
        mask[range(N), range(N), 0] = False
        # Exclude all contemporaneous values if requested
        if exclude_contemporaneous:
            mask[:, :, 0] = False
        # Create the return value
        q_matrix = np.array(p_matrix)
        # Use the multiple tests function
        if fdr_method is None or fdr_method == 'none':
            pass
        elif fdr_method == 'fdr_bh':
            pvs = p_matrix[mask]
            pvals_sortind = np.argsort(pvs)
            pvals_sorted = np.take(pvs, pvals_sortind)

            ecdffactor = _ecdf(pvals_sorted)

            pvals_corrected_raw = pvals_sorted / ecdffactor
            pvals_corrected = np.minimum.accumulate(
                pvals_corrected_raw[::-1])[::-1]
            del pvals_corrected_raw

            pvals_corrected[pvals_corrected > 1] = 1
            pvals_corrected_ = np.empty_like(pvals_corrected)
            pvals_corrected_[pvals_sortind] = pvals_corrected
            del pvals_corrected

            q_matrix[mask] = pvals_corrected_

        else:
            raise ValueError('Only FDR method fdr_bh implemented')

        # Return the new matrix
        return q_matrix


    def _get_adj_time_series(self, graph, include_conflicts=True, sort_by=None):
        """Helper function that returns dictionary of adjacencies from graph.

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph, see description above for interpretation.
        include_conflicts : bool, optional (default: True)
            Whether conflicting links (marked as 2 in graph) should be returned.
        sort_by : dict or none, optional (default: None)
            If not None, the adjacencies are sorted by the absolute values of
            the corresponding entries.

        Returns
        -------
        adjt : dictionary
            Adjacency dictionary.
        """
        N, N, tau_max_plusone = graph.shape
        adjt = {}
        if include_conflicts:
            for j in range(N):
                where = np.where(graph[:, j, :] != "")
                adjt[j] = list(zip(*(where[0], -where[1])))
        else:
            for j in range(N):
                where = np.where(np.logical_and.reduce((graph[:,j,:] != "", 
                                                        graph[:,j,:] != "x-x",
                                                        graph[:,j,:] != "x?x")))
                # where = np.where(graph[:, j, :] == 1)
                adjt[j] = list(zip(*(where[0], -where[1])))

        if sort_by is not None:
            for j in range(N):
                # Get the absolute value for all the test statistics
                abs_values = {k: np.abs(sort_by[j][k]) for k in list(sort_by[j])
                              if k in adjt[j]}
                adjt[j] = sorted(abs_values, key=abs_values.get, reverse=True)

        return adjt

    def _get_adj_time_series_contemp(self, graph, include_conflicts=True,
                                     sort_by=None):
        """Helper function that returns dictionary of contemporaneous
        adjacencies from graph.

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph, see description above for interpretation.
        include_conflicts : bool, optional (default: True)
            Whether conflicting links (marked as 2 in graph) should be returned.
        sort_by : dict or none, optional (default: None)
            If not None, the adjacencies are sorted by the absolute values of
            the corresponding entries.

        Returns
        -------
        adjt : dictionary
            Contemporaneous adjacency dictionary.
        """
        N, N, tau_max_plusone = graph.shape
        adjt = self._get_adj_time_series(graph,
                                         include_conflicts=include_conflicts,
                                         sort_by=sort_by)
        for j in range(N):
            adjt[j] = [a for a in adjt[j] if a[1] == 0]
            # adjt[j] = list(np.where(graph[:,j,0] != 0)[0])

        return adjt


    def _get_simplicial_node(self, circle_cpdag, variable_order):
        """Find simplicial nodes in circle component CPDAG.

        A vertex V is simplicial if all vertices adjacent to V are also adjacent
        to each other (form a clique).

        Parameters
        ----------
        circle_cpdag : array of shape (N, N, tau_max+1)
            Circle component of PCMCIplus graph.
        variable_order : list of length N
            Order of variables in which to search for simplicial nodes.

        Returns
        -------
        (j, adj_j) or None
            First found simplicial node and its adjacencies.
        """

        for j in variable_order:
            adj_j = np.where(np.logical_or(circle_cpdag[:,j,0] == "o-o",
                                           circle_cpdag[:,j,0] == "o?o"))[0].tolist()

            # Make sure the node has any adjacencies
            all_adjacent = len(adj_j) > 0

            # If it has just one adjacency, it's also simplicial
            if len(adj_j) == 1:
                return (j, adj_j)  
            else:
                for (var1, var2) in itertools.combinations(adj_j, 2):
                    if circle_cpdag[var1, var2, 0] == "": 
                        all_adjacent = False
                        break

                if all_adjacent:
                    return (j, adj_j)

        return None

    def _get_dag_from_cpdag(self, cpdag_graph, variable_order):
        """Yields one member of the Markov equivalence class of a CPDAG.

        Removes conflicting edges.

        Used in PCMCI to run model selection on the output of PCMCIplus in order
        to, e.g., optimize pc_alpha.

        Based on Zhang 2008, Theorem 2 (simplified for CPDAGs): Let H be the
        graph resulting from the following procedure applied to a CPDAG:
 
        Consider the circle component of the CPDAG (sub graph consisting of all
        (o-o edges, i.e., only for contemporaneous links), CPDAG^C and turn into
        a DAG with no unshielded colliders. Then (H is a member of the Markov
        equivalence class of the CPDAG.

        We use the approach mentioned in Colombo and Maathuis (2015) Lemma 7.6:
        First note that CPDAG^C is chordal, that is, any cycle of length four or
        more has a chord, which is an edge joining two vertices that are not
        adjacent in the cycle; see the proof of Lemma 4.1 of Zhang (2008b). Any
        chordal graph with more than one vertex has two simplicial vertices,
        that is, vertices V such that all vertices adjacent to V are also
        adjacent to each other. We choose such a vertex V1 and orient any edges
        incident to V1 into V1. Since V1 is simplicial, this does not create
        unshielded colliders. We then remove V1 and these edges from the graph.
        The resulting graph is again chordal and therefore again has at least
        two simplicial vertices. Choose such a vertex V2 , and orient any edges
        incident to V2 into V2. We continue this procedure until all edges are
        oriented. The resulting ordering is called a perfect elimination scheme
        for CPDAG^C. Then the combined graph with the directed edges already
        contained in the CPDAG is returned.

        Parameters
        ----------
        cpdag_graph : array of shape (N, N, tau_max+1)
            Result of PCMCIplus, a CPDAG.
        variable_order : list of length N
            Order of variables in which to search for simplicial nodes.

        Returns
        -------
        dag : array of shape (N, N, tau_max+1)
            One member of the Markov equivalence class of the CPDAG.
        """

        # TODO: Check whether CPDAG is chordal

        # Initialize resulting MAG
        dag = np.copy(cpdag_graph)

        # Turn circle component CPDAG^C into a DAG with no unshielded colliders.
        circle_cpdag = np.copy(cpdag_graph)
        # All lagged links are directed by time, remove them here
        circle_cpdag[:,:,1:] = ""
        # Also remove conflicting links
        circle_cpdag[circle_cpdag=="x-x"] = ""
        # Find undirected links, remove directed links
        for i, j, tau in zip(*np.where(circle_cpdag != "")):
            if circle_cpdag[i,j,0][1] == '?':
                raise ValueError("Invalid middle mark.")
            if circle_cpdag[i,j,0] == "-->":
                circle_cpdag[i,j,0] = ""

        # Iterate through simplicial nodes
        simplicial_node = self._get_simplicial_node(circle_cpdag,
                                                    variable_order)
        while simplicial_node is not None:

            # Choose such a vertex V1 and orient any edges incident to V1 into
            # V1 in the MAG And remove V1 and these edges from the circle
            # component PAG
            (j, adj_j) = simplicial_node
            for var in adj_j:
                dag[var, j, 0] = "-->"
                dag[j, var, 0] = "<--"
                circle_cpdag[var, j, 0] = circle_cpdag[j, var, 0] = "" 

            # Iterate
            simplicial_node = self._get_simplicial_node(circle_cpdag,
                                                    variable_order)

        return dag

    def convert_to_string_graph(self, graph_bool):
        """Converts the 0,1-based graph returned by PCMCI to a string array
        with links '-->'.

        Parameters
        ----------
        graph_bool : array
            0,1-based graph array output by PCMCI.

        Returns
        -------
        graph : array
            graph as string array with links '-->'.
        """

        graph = np.zeros(graph_bool.shape, dtype='<U3')
        graph[:] = ""
        # Lagged links
        graph[:,:,1:][graph_bool[:,:,1:]==1] = "-->"
        # Unoriented contemporaneous links
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==1, 
                                    graph_bool[:,:,0].T==1)] = "o-o"
        # Conflicting contemporaneous links
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==2, 
                                    graph_bool[:,:,0].T==2)] = "x-x"
        # Directed contemporaneous links
        for (i,j) in zip(*np.where(
            np.logical_and(graph_bool[:,:,0]==1, graph_bool[:,:,0].T==0))):
            graph[i,j,0] = "-->"
            graph[j,i,0] = "<--"

        return graph

    def symmetrize_p_and_val_matrix(self, p_matrix, val_matrix, link_assumptions, conf_matrix=None):
        """Symmetrizes the p_matrix, val_matrix, and conf_matrix based on link_assumptions
           and the larger p-value.

        Parameters
        ----------
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values. Set to 1 if val_only=True.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        link_assumptions : dict or None
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
        Returns
        -------
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values. Set to 1 if val_only=True.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """

        # Symmetrize p_matrix and val_matrix and conf_matrix
        for i in range(self.N):
            for j in range(self.N):
                # If both the links are present in link_assumptions, symmetrize using maximum p-value
                # if ((i, 0) in link_assumptions[j] and (j, 0) in link_assumptions[i]):
                if (i, 0) in link_assumptions[j]:
                    if link_assumptions[j][(i, 0)] in ["o-o", 'o?o']:
                        if (p_matrix[i, j, 0]
                                >= p_matrix[j, i, 0]):
                            p_matrix[j, i, 0] = p_matrix[i, j, 0]
                            val_matrix[j, i, 0] = val_matrix[i, j, 0]
                            if conf_matrix is not None:
                                conf_matrix[j, i, 0] = conf_matrix[i, j, 0]

                    # If only one of the links is present in link_assumptions, symmetrize using the p-value of the link present
                    # elif ((i, 0) in link_assumptions[j] and (j, 0) not in link_assumptions[i]):
                    elif link_assumptions[j][(i, 0)] in ["-->", '-?>']:
                        p_matrix[j, i, 0] = p_matrix[i, j, 0]
                        val_matrix[j, i, 0] = val_matrix[i, j, 0]
                        if conf_matrix is not None:
                            conf_matrix[j, i, 0] = conf_matrix[i, j, 0]
                    else:
                        # Links not present in link_assumptions
                        pass

        # Return the values as a dictionary and store in class
        results = {'val_matrix': val_matrix,
                   'p_matrix': p_matrix,
                   'conf_matrix': conf_matrix}
        return results

    def run_sliding_window_of(self, method, method_args, 
                        window_step,
                        window_length,
                        conf_lev = 0.9,
                        ):
        """Runs chosen method on sliding windows taken from DataFrame.

        The function returns summary_results and all_results (containing the
        individual window results). summary_results contains val_matrix_mean
        and val_matrix_interval, the latter containing the confidence bounds for
        conf_lev. If the method also returns a graph, then 'most_frequent_links'
        containing the most frequent link outcome (either 0 or 1 or a specific
        link type) in each entry of graph, as well as 'link_frequency',
        containing the occurence frequency of the most frequent link outcome,
        are returned.

        Parameters
        ----------
        method : str
            Chosen method among valid functions in PCMCI.
        method_args : dict
            Arguments passed to method.
        window_step : int
            Time step of windows.
        window_length : int
            Length of sliding window.
        conf_lev : float, optional (default: 0.9)
            Two-sided confidence interval for summary results.

        Returns
        -------
        Dictionary of results for every sliding window.
        """

        valid_methods = ['run_pc_stable',
                          'run_mci',
                          'get_lagged_dependencies',
                          'run_fullci',
                          'run_bivci',
                          'run_pcmci',
                          'run_pcalg',
                          'run_lpcmci',
                          'run_jpcmciplus',
                          # 'run_pcalg_non_timeseries_data',
                          'run_pcmciplus',]

        if method not in valid_methods:
            raise ValueError("method must be one of %s" % str(valid_methods))

        if self.dataframe.reference_points_is_none is False:
            raise ValueError("Reference points are not accepted in "
                             "sliding windows analysis, align data before and use masking"
                             " and/or missing values.")

        T = self.dataframe.largest_time_step

        if self.cond_ind_test.recycle_residuals:
            # recycle_residuals clashes with sliding windows...
            raise ValueError("cond_ind_test.recycle_residuals must be False.")

        if self.verbosity > 0:
            print("\n##\n## Running sliding window analysis of %s " % method +
                  "\n##\n" +
                  "\nwindow_step = %s \n" % window_step +
                  "\nwindow_length = %s \n" % window_length
                  )

        original_reference_points = deepcopy(self.dataframe.reference_points)

        window_start_points = np.arange(0, T - window_length, window_step)
        n_windows = len(window_start_points)

        if len(window_start_points) == 0:
            raise ValueError("Empty list of windows, check window_length and window_step!")

        window_results = {}
        for iw, w in enumerate(window_start_points):
            if self.verbosity > 0:
                print("\n# Window start %s (%d/%d) \n" %(w, iw+1, len(window_start_points)))                
            # Construct reference_points from window
            time_window = np.arange(w, w + window_length, 1)
            # Remove points beyond T
            time_window = time_window[time_window < T]

            self.dataframe.reference_points = time_window
            window_res = deepcopy(getattr(self, method)(**method_args))

            # Aggregate val_matrix and other arrays to new arrays with
            # windows as first dimension. Lists and other objects
            # are stored in dictionary
            for key in window_res:
                res_item = window_res[key]
                if iw == 0:
                    if type(res_item) is np.ndarray:
                        window_results[key] = np.empty((n_windows,) 
                                                     + res_item.shape,
                                                     dtype=res_item.dtype) 
                    else:
                        window_results[key] = {}
                
                window_results[key][iw] = res_item

        # Reset to original_reference_points data for further analyses
        # self.dataframe.values[0] = original_data
        self.dataframe.reference_points = original_reference_points

        # Generate summary results
        summary_results = self.return_summary_results(results=window_results, 
                                                      conf_lev=conf_lev)

        return {'summary_results': summary_results, 
                'window_results': window_results}

    def run_bootstrap_of(self, method, method_args,
                        boot_samples=100,
                        boot_blocklength=1,
                        conf_lev=0.9, aggregation="majority", seed=None):
        """Runs chosen method on bootstrap samples drawn from DataFrame.

        Bootstraps for tau=0 are drawn from [2xtau_max, ..., T] and all lagged
        variables constructed in DataFrame.construct_array are consistently
        shifted with respect to this bootstrap sample to ensure that lagged
        relations in the bootstrap sample are preserved.

        The function returns summary_results and all_results (containing the
        individual bootstrap results). summary_results contains
        val_matrix_mean and val_matrix_interval, the latter containing the
        confidence bounds for conf_lev. If the method also returns a graph,
        then 'most_frequent_links' containing the most frequent link outcome
        (specific link type) in each entry of graph, as well
        as 'link_frequency', containing the occurence frequency of the most
        frequent link outcome, are returned. Two aggregation methods are
        available for 'most_frequent_links'. By default, "majority"
        provides the most frequent link outcome. Alternatively 
        "no_edge_majority" provides an alternative aggregation strategy.
        As explained in Debeire et al. (2024), in the first step of this 
        alternative approach, the orientation of edges is ignored, and the 
        focus is only on determining the adjacency of each pair of vertices. 
        This is done through majority voting between no edge and all other 
        edge types. In the second step, the adjacencies identified in the
        first step are oriented based on majority voting. This alternative 
        approach ensures that no edge can only be voted on if it appears 
        in more than half of the bootstrap ensemble of graphs.

        Assumes that method uses cond_ind_test.run_test() function with cut_off
        = '2xtau_max'.

        Utilizes parallelization via joblib.

        Parameters
        ----------
        method : str
            Chosen method among valid functions in PCMCI.
        method_args : dict
            Arguments passed to method.
        boot_samples : int
            Number of bootstrap samples to draw.
        boot_blocklength : int, optional (default: 1)
            Block length for block-bootstrap.
        conf_lev : float, optional (default: 0.9)
            Two-sided confidence interval for summary results.
        seed : int, optional(default = None)
            Seed for RandomState (default_rng)
        aggregation : str, optional (default: "majority")
            Chosen aggregation strategy: "majority" or "no_edge_majority".

        Returns
        -------
        Dictionary of summary results and results for every bootstrap sample.
        """

        valid_methods = ['run_pc_stable',
                          'run_mci',
                          'get_lagged_dependencies',
                          'run_fullci',
                          'run_bivci',
                          'run_pcmci',
                          'run_pcalg',
                          'run_pcalg_non_timeseries_data',
                          'run_pcmciplus',
                          'run_lpcmci',
                          'run_jpcmciplus',
                          ]
        if method not in valid_methods:
            raise ValueError("method must be one of %s" % str(valid_methods))

        T = self.dataframe.largest_time_step
        seed_sequence = np.random.SeedSequence(seed)
        #global_random_state = np.random.default_rng(seed)

        # Extract tau_max to construct bootstrap draws
        if 'tau_max' not in method_args:
            raise ValueError("tau_max must be explicitely set in method_args.")
        tau_max = method_args['tau_max']

        if self.cond_ind_test.recycle_residuals:
            # recycle_residuals clashes with bootstrap draws...
            raise ValueError("cond_ind_test.recycle_residuals must be False.")

        if self.verbosity > 0:
            print("\n##\n## Running Bootstrap of %s " % method +
                  "\n##\n" +
                  "\nboot_samples = %s \n" % boot_samples +
                  "\nboot_blocklength = %s \n" % boot_blocklength
                  )

        # Set bootstrap attribute to be passed to dataframe
        self.dataframe.bootstrap = {}
        self.dataframe.bootstrap['boot_blocklength'] = boot_blocklength

        boot_results = {}
        #for b in range(boot_samples):
            # Generate random state for this boot and set it in dataframe
            # which will generate a draw with replacement
            #boot_seed = global_random_state.integers(0, boot_samples, 1)
            #boot_random_state = np.random.default_rng(boot_seed)
            #self.dataframe.bootstrap['random_state'] = boot_random_state

        child_seeds = seed_sequence.spawn(boot_samples)

        aggregated_results = Parallel(n_jobs=-1)(
            delayed(self.parallelized_bootstraps)(method, method_args, boot_seed=child_seeds[b]) for
            b in range(boot_samples))

        for b in range(boot_samples):
            # Aggregate val_matrix and other arrays to new arrays with
            # boot_samples as first dimension. Lists and other objects
            # are stored in dictionary
            boot_res = aggregated_results[b]
            for key in boot_res:
                res_item = boot_res[key]
                if type(res_item) is np.ndarray:
                    if b == 0:
                        boot_results[key] = np.empty((boot_samples,) 
                                                     + res_item.shape,
                                                     dtype=res_item.dtype) 
                    boot_results[key][b] = res_item
                else:
                    if b == 0:
                        boot_results[key] = {}
                    boot_results[key][b] = res_item

        # Generate summary results
        summary_results = self.return_summary_results(results=boot_results, 
                                                      conf_lev=conf_lev,
                                                      aggregation=aggregation)

        # Reset bootstrap to None
        self.dataframe.bootstrap = None

        return {'summary_results': summary_results, 
                'boot_results': boot_results}

    def parallelized_bootstraps(self, method, method_args, boot_seed):
        # Pass seed sequence for this boot and set it in dataframe
        # which will generate a draw with replacement
        boot_random_state = np.random.default_rng(boot_seed)
        self.dataframe.bootstrap['random_state'] = boot_random_state
        boot_res = getattr(self, method)(**method_args)
        return boot_res

    @staticmethod
    def return_summary_results(results, conf_lev=0.9, aggregation="majority"):
        """Return summary results for causal graphs.

        The function returns summary_results of an array of PCMCI(+) results.
        Summary_results contains val_matrix_mean and val_matrix_interval, the latter 
        containing the confidence bounds for conf_lev. If the method also returns a graph,
        then 'most_frequent_links' containing the most frequent link outcome 
        (either 0 or 1 or a specific link type) in each entry of graph, as well 
        as 'link_frequency', containing the occurence frequency of the most 
        frequent link outcome, are returned. Two aggregation methods are
        available for 'most_frequent_links'. By default, "majority"
        provides the most frequent link outcome. Alternatively 
        "no_edge_majority" provides an alternative aggregation strategy.
        As explained in Debeire et al. (2024), in the first step of this 
        alternative approach, the orientation of edges is ignored, and the 
        focus is only on determining the adjacency of each pair of vertices. 
        This is done through majority voting between no edge and all other 
        edge types. In the second step, the adjacencies identified in the
        first step are oriented based on majority voting. This alternative 
        approach ensures that no edge can only be voted on if it appears 
        in more than half of the bootstrap ensemble of graphs.

        Parameters
        ----------
        results : dict
            Results dictionary where the numpy arrays graph and val_matrix are
            of shape (n_results, N, N, tau_max + 1).
        conf_lev : float, optional (default: 0.9)
            Two-sided confidence interval for summary results.
        aggregation : str, optional (default: "majority")
            Chosen aggregation strategy: "majority" or "no_edge_majority".
        Returns
        -------
        Dictionary of summary results.
        """

        valid_aggregations = {"majority", "no_edge_majority"}
        if aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation mode: {aggregation}. Expected one of {valid_aggregations}")

        # Generate summary results
        summary_results = {}

        if 'graph' in results:
            n_results, N, N, tau_max_plusone = results['graph'].shape
            tau_max = tau_max_plusone - 1
            # print(repr(results['graph']))
            summary_results['most_frequent_links'] = np.zeros((N, N, tau_max_plusone),
                                dtype=results['graph'][0].dtype)
            summary_results['link_frequency'] = np.zeros((N, N, tau_max_plusone),
                                dtype='float')
            
            #preferred order in case of ties with the spirit of 
            #keeping the least assertive and most cautious claims in the presence of ties.
            #In case of ties between other link types, a conflicting link "x-x" is assigned
            preferred_order = [
            "",       # No link (most conservative)
            #"o?o",    # No claim made (lag 0 only)
            #"<?>",    # Neither is ancestor
            "x-x",    # Conflict (used to break <--> vs --> vs <-- ties)
            "o-o",    # Undirected link (lag 0 only)
            # "<-o",    # X^i not ancestor, but linked (lag 0 only)
            # "o->",    # X^j not ancestor, but linked
            # rest is solved by conflict
            # "<->",
            # "-->",
            # "<--",
            ]

            for (i, j) in itertools.product(range(N), range(N)):
                for abstau in range(0, tau_max + 1):
                    links, counts = np.unique(results['graph'][:,i,j,abstau], 
                                        return_counts=True)
                    list_of_most_freq = links[counts == counts.max()]
                    if aggregation=="majority":
                        if len(list_of_most_freq) == 1:
                            choice = list_of_most_freq[0]
                        else:
                            ordered_list = [link for link in preferred_order
                                            if link in list_of_most_freq]
                            if len(ordered_list) == 0:
                                choice = "x-x"
                            else:
                                choice = ordered_list[0]
                        summary_results['most_frequent_links'][i,j, abstau] = choice
                        summary_results['link_frequency'][i,j, abstau] = \
                                    counts[counts == counts.max()].sum()/float(n_results)

                    elif aggregation=="no_edge_majority":
                        if counts[links == ""].size == 0: #handle the case where there is no "" in links
                            freq_of_no_edge=0
                        else:
                            # make scalar count (counts[...] returns a 1-element array)
                            freq_of_no_edge = int(counts[links == ""].sum())
                            
                        freq_of_adjacency = n_results - freq_of_no_edge
                        if freq_of_adjacency > freq_of_no_edge:
                            adja_links = np.delete(links,np.where(links == ""))
                            adja_counts = np.delete(counts,np.where(links == ""))
                            list_of_most_freq_adja = adja_links[adja_counts == adja_counts.max()]
                            if len(list_of_most_freq_adja) == 1:
                                choice = list_of_most_freq_adja[0]
                            else:
                                ordered_list = [link for link in preferred_order
                                                if link in list_of_most_freq_adja]
                                if len(ordered_list) == 0:
                                    choice = "x-x"
                                else:
                                    choice = ordered_list[0]
                            summary_results['most_frequent_links'][i,j, abstau] = choice
                            summary_results['link_frequency'][i,j, abstau] = \
                                    adja_counts[adja_counts == adja_counts.max()].sum()/float(n_results)
                        else: 
                            choice= ""
                            summary_results['most_frequent_links'][i,j, abstau] = choice
                            summary_results['link_frequency'][i,j, abstau] = \
                                    freq_of_no_edge/float(n_results)
        # Confidence intervals for val_matrix; interval is two-sided
        c_int = (1. - (1. - conf_lev)/2.)
        summary_results['val_matrix_mean'] = np.mean(
                                    results['val_matrix'], axis=0)

        summary_results['val_matrix_interval'] = np.stack(np.percentile(
                                    results['val_matrix'], axis=0,
                                    q = [100*(1. - c_int), 100*c_int]), axis=3)
        return summary_results

    @staticmethod
    def graph_to_dict(graph):
        """Helper function to convert graph to dictionary of links.

        Parameters
        ---------
        graph : array of shape (N, N, tau_max+1)
            Matrix format of graph in string format.

        Returns
        -------
        links : dict
            Dictionary of form {0:{(0, -1): o-o, ...}, 1:{...}, ...}.
        """
        N = graph.shape[0]

        links = dict([(j, {}) for j in range(N)])

        for (i, j, tau) in zip(*np.where(graph!='')):
            links[j][(i, -tau)] = graph[i,j,tau]

        return links

    # @staticmethod
    def _dict_to_graph(self, links, tau_max=None):
        """Helper function to convert dictionary of links to graph.

        Parameters
        ---------
        links : dict
            Dictionary of form {0:{(0, -1): 'o-o'}, ...}, 1:{...}, ...}.

        Returns
        -------
        graph : array of shape (N, N, tau_max+1)
            Matrix format of graph in string format.
        """

        N = len(links)

        # Get maximum time lag
        max_lag = 0
        for j in range(N):
            for link in links[j]:
                var, lag = link
                if isinstance(links[j], dict):
                    link_type = links[j][link]
                    if link_type != "":
                        max_lag = max(max_lag, abs(lag))
                else:
                    max_lag = max(max_lag, abs(lag))

        if tau_max is None:
            tau_max = max_lag
        else:
            if tau_max < max_lag:
                raise ValueError("maxlag(links) > tau_max")

        graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
        graph[:] = ""
        for j in range(N):
            for link in links[j]:
                i, tau = link
                if isinstance(links[j], dict):
                    link_type = links[j][link]
                    graph[i, j, abs(tau)] = link_type
                else:
                    graph[i, j, abs(tau)] = '-->'

        return graph

    @staticmethod
    def get_graph_from_dict(links, tau_max=None):
        """Helper function to convert dictionary of links to graph array format.

        Parameters
        ---------
        links : dict
            Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
            Also format {0:[(0, -1), ...], 1:[...], ...} is allowed.
        tau_max : int or None
            Maximum lag. If None, the maximum lag in links is used.

        Returns
        -------
        graph : array of shape (N, N, tau_max+1)
            Matrix format of graph with 1 for true links and 0 else.
        """

        def _get_minmax_lag(links):
            """Helper function to retrieve tau_min and tau_max from links.
            """

            N = len(links)

            # Get maximum time lag
            min_lag = np.inf
            max_lag = 0
            for j in range(N):
                for link_props in links[j]:
                    if len(link_props) > 2:
                        var, lag = link_props[0]
                        coeff = link_props[1]
                        # func = link_props[2]
                        if coeff != 0.:
                            min_lag = min(min_lag, abs(lag))
                            max_lag = max(max_lag, abs(lag))
                    else:
                        var, lag = link_props
                        min_lag = min(min_lag, abs(lag))
                        max_lag = max(max_lag, abs(lag))   

            return min_lag, max_lag

        N = len(links)

        # Get maximum time lag
        min_lag, max_lag = _get_minmax_lag(links)

        # Set maximum lag
        if tau_max is None:
            tau_max = max_lag
        else:
            if max_lag > tau_max:
                raise ValueError("tau_max is smaller than maximum lag = %d "
                                 "found in links, use tau_max=None or larger "
                                 "value" % max_lag)

        graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
        for j in links.keys():
            for link_props in links[j]:
                if len(link_props) > 2:
                    var, lag = link_props[0]
                    coeff = link_props[1]
                    if coeff != 0.:
                        graph[var, j, abs(lag)] = "-->"
                        if lag == 0:
                            graph[j, var, 0] = "<--"
                else:
                    var, lag = link_props
                    graph[var, j, abs(lag)] = "-->"
                    if lag == 0:
                        graph[j, var, 0] = "<--"

        return graph

    @staticmethod
    def build_link_assumptions(link_assumptions_absent_link_means_no_knowledge,
                               n_component_time_series,
                               tau_max,
                               tau_min=0):
        
        out = {j: {(i, -tau_i): ("o?>" if tau_i > 0 else "o?o")
             for i in range(n_component_time_series) for tau_i in range(tau_min, tau_max+1)
             if (tau_i > 0 or i != j)} for j in range(n_component_time_series)}
        
        for j, links_j in link_assumptions_absent_link_means_no_knowledge.items():
            for (i, lag_i), link_ij in links_j.items():
                if link_ij == "": 
                    del out[j][(i, lag_i)]
                else:
                    out[j][(i, lag_i)] = link_ij
        return out
