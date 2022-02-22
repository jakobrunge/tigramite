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


def _create_nested_dictionary(depth=0, lowest_type=dict):
    """Create a series of nested dictionaries to a maximum depth.  The first
    depth - 1 nested dictionaries are defaultdicts, the last is a normal
    dictionary.

    Parameters
    ----------
    depth : int
        Maximum depth argument.
    lowest_type: callable (optional)
        Type contained in leaves of tree.  Ex: list, dict, tuple, int, float ...
    """
    new_depth = depth - 1
    if new_depth <= 0:
        return defaultdict(lowest_type)
    return defaultdict(lambda: _create_nested_dictionary(new_depth))


def _nested_to_normal(nested_dict):
    """Transforms the nested default dictionary into a standard dictionaries

    Parameters
    ----------
    nested_dict : default dictionary of default dictionaries of ... etc.
    """
    if isinstance(nested_dict, defaultdict):
        nested_dict = {k: _nested_to_normal(v) for k, v in nested_dict.items()}
    return nested_dict


class PCMCI():
    r"""PCMCI causal discovery for time series datasets.

    PCMCI is a causal discovery framework for large-scale time series
    datasets. This class contains several methods. The standard PCMCI method
    addresses time-lagged causal discovery and is described in [1]_ where
    also further sub-variants are discussed. Lagged as well as contemporaneous
    causal discovery is addressed with PCMCIplus and described in [5]_. See the
    tutorials for guidance in applying these methods.

    PCMCI has:

    * different conditional independence tests adapted to linear or
      nonlinear dependencies, and continuously-valued or discrete data (
      implemented in ``tigramite.independence_tests``)
    * (mostly) hyperparameter optimization
    * easy parallelization (separate script)
    * handling of masked time series data
    * false discovery control and confidence interval estimation


    Notes
    -----

    .. image:: mci_schematic.*
       :width: 200pt

    In the PCMCI framework, the dependency structure of a set of time series
    variables is represented in a *time series graph* as shown in the Figure.
    The nodes of a time series graph are defined as the variables at
    different times and a link indicates a conditional dependency that can be
    interpreted as a causal dependency under certain assumptions (see paper).
    Assuming stationarity, the links are repeated in time. The parents
    :math:`\mathcal{P}` of a variable are defined as the set of all nodes
    with a link towards it (blue and red boxes in Figure).

    The different PCMCI methods estimate causal links by iterative
    conditional independence testing. PCMCI can be flexibly combined with
    any kind of conditional independence test statistic adapted to the kind
    of data (continuous or discrete) and its assumed dependency types.
    These are available in ``tigramite.independence_tests``.

    NOTE: MCI test statistic values define a particular measure of causal
    strength depending on the test statistic used. For example, ParCorr()
    results in normalized values between -1 and 1. However, if you are 
    interested in quantifying causal effects, i.e., the effect of
    hypothetical interventions, you may better look at the causal effect 
    estimation functionality of Tigramite.

    References
    ----------

    .. [1] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic,
           Detecting and quantifying causal associations in large nonlinear time 
           series datasets. Sci. Adv. 5, eaau4996 (2019) 
           https://advances.sciencemag.org/content/5/11/eaau4996

    .. [5] J. Runge,
           Discovering contemporaneous and lagged causal relations in 
           autocorrelated nonlinear time series datasets
           http://www.auai.org/~w-auai/uai2020/proceedings/579_main_paper.pdf

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
    selected_variables : list
        Deprecated, just here to raise Error if not None.
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
    T : int
        Time series sample length.
    """

    def __init__(self, dataframe,
                 cond_ind_test,
                 selected_variables=None,
                 verbosity=0):
        # Set the data for this iteration of the algorithm
        self.dataframe = dataframe
        # Set the conditional independence test to be used
        self.cond_ind_test = cond_ind_test
        if isinstance(self.cond_ind_test, type):
            raise ValueError("PCMCI requires that cond_ind_test "
                             "is instantiated, e.g. cond_ind_test =  "
                             "ParCorr().")
        self.cond_ind_test.set_dataframe(self.dataframe)
        # Set the verbosity for debugging/logging messages
        self.verbosity = verbosity
        # Set the variable names 
        self.var_names = self.dataframe.var_names

        # Raise error if selected_variables is set
        if selected_variables is not None:
            raise ValueError("selected_variables is deprecated, use "
                             "the selected_links parameter in the respective "
                             "functions run_pcmci() etc.")

        # Store the shape of the data in the T and N variables
        self.T = self.dataframe.T
        self.N = self.dataframe.N

    def _set_sel_links(self, selected_links, tau_min, tau_max,
                       remove_contemp=False):
        """Helper function to set and check the selected links argument

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are returned.
        tau_mix : int
            Minimum time delay to test.
        tau_max : int
            Maximum time delay to test.
        remove_contemp : bool
            Whether contemporaneous links (at lag zero) should be removed.

        Returns
        -------
        selected_links : dict
            Cleaned links.
        """
        # Copy and pass into the function
        _int_sel_links = deepcopy(selected_links)
        # Set the default selected links if none are set
        _vars = list(range(self.N))
        _lags = list(range(-(tau_max), -tau_min + 1, 1))
        if _int_sel_links is None:
            _int_sel_links = {}
            # Set the default as all combinations of the selected variables
            for j in _vars:
                _int_sel_links[j] = [(var, -lag) for var in _vars
                                     for lag in range(tau_min, tau_max + 1) 
                                     if not (var == j and lag == 0)]
        else:
            if remove_contemp:
                for j in _int_sel_links.keys():
                    _int_sel_links[j] = [link for link in _int_sel_links[j]
                                         if link[1] != 0]
        # Otherwise, check that our selection is sane
        # Check that the selected links refer to links that are inside the
        # data range
        _key_set = set(_int_sel_links.keys())
        valid_entries = _key_set == set(range(self.N))

        for link in _int_sel_links.values():
            if isinstance(link, list) and len(link) == 0:
                continue
            for var, lag in link:
                if var not in _vars or lag not in _lags:
                    valid_entries = False

        if not valid_entries:
            raise ValueError("selected_links"
                             " must be dictionary with keys for all [0,...,N-1]"
                             " variables and contain only links from "
                             "these variables in range [tau_min, tau_max]")

        # Return the selected links
        return _int_sel_links

    def _iter_conditions(self, parent, conds_dim, all_parents):
        """Yield next condition.

        Yields next condition from lexicographically ordered conditions.

        Parameters
        ----------
        parent : tuple
            Tuple of form (i, -tau).
        conds_dim : int
            Cardinality in current step.
        all_parents : list
            List of form [(0, -1), (3, -2), ...].

        Yields
        -------
        cond :  list
            List of form [(0, -1), (3, -2), ...] for the next condition.
        """
        all_parents_excl_current = [p for p in all_parents if p != parent]
        for cond in itertools.combinations(all_parents_excl_current, conds_dim):
            yield list(cond)

    def _sort_parents(self, parents_vals):
        """Sort current parents according to test statistic values.

        Sorting is from strongest to weakest absolute values.

        Parameters
        ---------
        parents_vals : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.

        Returns
        -------
        parents : list
            List of form [(0, -1), (3, -2), ...] containing sorted parents.
        """
        if self.verbosity > 1:
            print("\n    Sorting parents in decreasing order with "
                  "\n    weight(i-tau->j) = min_{iterations} |val_{ij}(tau)| ")
        # Get the absolute value for all the test statistics
        abs_values = {k: np.abs(parents_vals[k]) for k in list(parents_vals)}
        return sorted(abs_values, key=abs_values.get, reverse=True)

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
                matrix[k, j, abs(tau)] = val_dict[j][link]
        return matrix

    def _print_link_info(self, j, index_parent, parent, num_parents,
                         already_removed=False):
        """Print info about the current link being tested.

        Parameters
        ----------
        j : int
            Index of current node being tested.
        index_parent : int
            Index of the current parent.
        parent : tuple
            Standard (i, tau) tuple of parent node id and time delay
        num_parents : int
            Total number of parents.
        already_removed : bool
            Whether parent was already removed.
        """
        link_marker = {True:"o-o", False:"-->"}

        abstau = abs(parent[1])
        if self.verbosity > 1:
            print("\n    Link (%s % d) %s %s (%d/%d):" % (
                self.var_names[parent[0]], parent[1], link_marker[abstau==0],
                self.var_names[j],
                index_parent + 1, num_parents))

            if already_removed:
                print("    Already removed.")

    def _print_cond_info(self, Z, comb_index, pval, val):
        """Print info about the condition

        Parameters
        ----------
        Z : list
            The current condition being tested.
        comb_index : int
            Index of the combination yielding this condition.
        pval : float
            p-value from this condition.
        val : float
            value from this condition.
        """
        var_name_z = ""
        for i, tau in Z:
            var_name_z += "(%s % .2s) " % (self.var_names[i], tau)
        if len(Z) == 0: var_name_z = "()"
        print("    Subset %d: %s gives pval = %.5f / val = % .3f" %
              (comb_index, var_name_z, pval, val))

    def _print_a_pc_result(self, nonsig, conds_dim, max_combinations):
        """Print the results from the current iteration of conditions.

        Parameters
        ----------
        nonsig : bool
            Indicate non-significance.
        conds_dim : int
            Cardinality of the current step.
        max_combinations : int
            Maximum number of combinations of conditions of current cardinality
            to test.
        """
        # Start with an indent
        print_str = "    "
        # Determine the body of the text
        if nonsig:
            print_str += "Non-significance detected."
        elif conds_dim > max_combinations:
            print_str += "Still subsets of dimension" + \
                         " %d left," % (conds_dim) + \
                         " but q_max = %d reached." % (max_combinations)
        else:
            print_str += "No conditions of dimension %d left." % (conds_dim)
        # Print the message
        print(print_str)

    def _print_converged_pc_single(self, converged, j, max_conds_dim):
        """
        Print statement about the convergence of the pc_stable_single algorithm.

        Parameters
        ----------
        convergence : bool
            true if convergence was reached.
        j : int
            Variable index.
        max_conds_dim : int
            Maximum number of conditions to test.
        """
        if converged:
            print("\nAlgorithm converged for variable %s" %
                  self.var_names[j])
        else:
            print(
                "\nAlgorithm not yet converged, but max_conds_dim = %d"
                " reached." % max_conds_dim)

    def _run_pc_stable_single(self, j,
                              selected_links=None,
                              tau_min=1,
                              tau_max=1,
                              save_iterations=False,
                              pc_alpha=0.2,
                              max_conds_dim=None,
                              max_combinations=1):
        """Lagged PC algorithm for estimating lagged parents of single variable.

        Parameters
        ----------
        j : int
            Variable index.
        selected_links : list, optional (default: None)
            List of form [(0, -1), (3, -2), ...]
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, optional (default: 1)
            Minimum time lag to test. Useful for variable selection in
            multi-step ahead predictions. Must be greater zero.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, optional (default: False)
            Whether to save iteration step results such as conditions used.
        pc_alpha : float or None, optional (default: 0.2)
            Significance level in algorithm. If a list is given, pc_alpha is
            optimized using model selection criteria provided in the
            cond_ind_test class as get_model_selection_criterion(). If None,
            a default list of values is used.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.

        Returns
        -------
        parents : list
            List of estimated parents.
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.
        pval_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        iterations : dict
            Dictionary containing further information on algorithm steps.
        """
        # Initialize the dictionaries for the pval_max, val_min parents_values
        # results
        pval_max = dict()
        val_min = dict()
        parents_values = dict()
        # Initialize the parents values from the selected links, copying to
        # ensure this initial argument is unchanged.
        parents = deepcopy(selected_links)
        val_min = {(p[0], p[1]): None for p in parents}
        pval_max = {(p[0], p[1]): None for p in parents}

        # Define a nested defaultdict of depth 4 to save all information about
        # iterations
        iterations = _create_nested_dictionary(4)
        # Ensure tau_min is at least 1
        tau_min = max(1, tau_min)

        # Loop over all possible condition dimensions
        max_conds_dim = self._set_max_condition_dim(max_conds_dim,
                                                    tau_min, tau_max)
        # Iteration through increasing number of conditions, i.e. from 
        # [0, max_conds_dim] inclusive
        converged = False
        for conds_dim in range(max_conds_dim + 1):
            # (Re)initialize the list of non-significant links
            nonsig_parents = list()
            # Check if the algorithm has converged
            if len(parents) - 1 < conds_dim:
                converged = True
                break
            # Print information about
            if self.verbosity > 1:
                print("\nTesting condition sets of dimension %d:" % conds_dim)

            # Iterate through all possible pairs (that have not converged yet)
            for index_parent, parent in enumerate(parents):
                # Print info about this link
                if self.verbosity > 1:
                    self._print_link_info(j, index_parent, parent, len(parents))
                # Iterate through all possible combinations
                nonsig = False
                for comb_index, Z in \
                        enumerate(self._iter_conditions(parent, conds_dim,
                                                        parents)):
                    # Break if we try too many combinations
                    if comb_index >= max_combinations:
                        break
                    # Perform independence test
                    val, pval = self.cond_ind_test.run_test(X=[parent],
                                                    Y=[(j, 0)],
                                                    Z=Z,
                                                    tau_max=tau_max,
                                                    # verbosity=self.verbosity
                                                    )
                    # Print some information if needed
                    if self.verbosity > 1:
                        self._print_cond_info(Z, comb_index, pval, val)
                    # Keep track of maximum p-value and minimum estimated value
                    # for each pair (across any condition)
                    parents_values[parent] = \
                        min(np.abs(val), parents_values.get(parent,
                                                            float("inf")))

                    if pval_max[parent] is None or pval > pval_max[parent]:
                        pval_max[parent] = pval
                        val_min[parent] = val

                    # Save the iteration if we need to
                    if save_iterations:
                        a_iter = iterations['iterations'][conds_dim][parent]
                        a_iter[comb_index]['conds'] = deepcopy(Z)
                        a_iter[comb_index]['val'] = val
                        a_iter[comb_index]['pval'] = pval
                    # Delete link later and break while-loop if non-significant
                    if pval > pc_alpha:
                        nonsig_parents.append((j, parent))
                        nonsig = True
                        break

                # Print the results if needed
                if self.verbosity > 1:
                    self._print_a_pc_result(nonsig,
                                            conds_dim, max_combinations)

            # Remove non-significant links
            for _, parent in nonsig_parents:
                del parents_values[parent]
            # Return the parents list sorted by the test metric so that the
            # updated parents list is given to the next cond_dim loop
            parents = self._sort_parents(parents_values)
            # Print information about the change in possible parents
            if self.verbosity > 1:
                print("\nUpdating parents:")
                self._print_parents_single(j, parents, parents_values, pval_max)

        # Print information about if convergence was reached
        if self.verbosity > 1:
            self._print_converged_pc_single(converged, j, max_conds_dim)
        # Return the results
        return {'parents': parents,
                'val_min': val_min,
                'pval_max': pval_max,
                'iterations': _nested_to_normal(iterations)}

    def _print_pc_params(self, selected_links, tau_min, tau_max, pc_alpha,
                         max_conds_dim, max_combinations):
        """Print the setup of the current pc_stable run.

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form specifying which links should be tested.
        tau_min : int, default: 1
            Minimum time lag to test.
        tau_max : int, default: 1
            Maximum time lag to test.
        pc_alpha : float or list of floats
            Significance level in algorithm.
        max_conds_dim : int
            Maximum number of conditions to test.
        max_combinations : int
            Maximum number of combinations of conditions to test.
        """
        print("\n##\n## Step 1: PC1 algorithm with lagged conditions\n##"
              "\n\nParameters:")
        if selected_links is not None:
            print("selected_links = %s" % selected_links)
        print("independence test = %s" % self.cond_ind_test.measure
              + "\ntau_min = %d" % tau_min
              + "\ntau_max = %d" % tau_max
              + "\npc_alpha = %s" % pc_alpha
              + "\nmax_conds_dim = %s" % max_conds_dim
              + "\nmax_combinations = %d" % max_combinations)
        print("\n")

    def _print_pc_sel_results(self, pc_alpha, results, j, score, optimal_alpha):
        """Print the results from the pc_alpha selection.

        Parameters
        ----------
        pc_alpha : list
            Tested significance levels in algorithm.
        results : dict
            Results from the tested pc_alphas.
        score : array of floats
            scores from each pc_alpha.
        j : int
            Index of current variable.
        optimal_alpha : float
            Optimal value of pc_alpha.
        """
        print("\n# Condition selection results:")
        for iscore, pc_alpha_here in enumerate(pc_alpha):
            names_parents = "[ "
            for pari in results[pc_alpha_here]['parents']:
                names_parents += "(%s % d) " % (
                    self.var_names[pari[0]], pari[1])
            names_parents += "]"
            print("    pc_alpha=%s got score %.4f with parents %s" %
                  (pc_alpha_here, score[iscore], names_parents))
        print("\n--> optimal pc_alpha for variable %s is %s" %
              (self.var_names[j], optimal_alpha))

    def _check_tau_limits(self, tau_min, tau_max):
        """Check the tau limits adhere to 0 <= tau_min <= tau_max.

        Parameters
        ----------
        tau_min : float
            Minimum tau value.
        tau_max : float
            Maximum tau value.
        """
        if not 0 <= tau_min <= tau_max:
            raise ValueError("tau_max = %d, " % (tau_max) + \
                             "tau_min = %d, " % (tau_min) + \
                             "but 0 <= tau_min <= tau_max")

    def _set_max_condition_dim(self, max_conds_dim, tau_min, tau_max):
        """
        Set the maximum dimension of the conditions. Defaults to self.N*tau_max.

        Parameters
        ----------
        max_conds_dim : int
            Input maximum condition dimension.
        tau_max : int
            Maximum tau.

        Returns
        -------
        max_conds_dim : int
            Input maximum condition dimension or default.
        """
        # Check if an input was given
        if max_conds_dim is None:
            max_conds_dim = self.N * (tau_max - tau_min + 1)
        # Check this is a valid
        if max_conds_dim < 0:
            raise ValueError("maximum condition dimension must be >= 0")
        return max_conds_dim

    def run_pc_stable(self,
                      selected_links=None,
                      tau_min=1,
                      tau_max=1,
                      save_iterations=False,
                      pc_alpha=0.2,
                      max_conds_dim=None,
                      max_combinations=1):
        """Lagged PC algorithm for estimating lagged parents of all variables.

        Parents are made available as self.all_parents

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, default: 1
            Minimum time lag to test. Useful for multi-step ahead predictions.
            Must be greater zero.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, default: False
            Whether to save iteration step results such as conditions used.
        pc_alpha : float or list of floats, default: [0.05, 0.1, 0.2, ..., 0.5]
            Significance level in algorithm. If a list or None is passed, the
            pc_alpha level is optimized for every variable across the given
            pc_alpha values using the score computed in
            cond_ind_test.get_model_selection_criterion().
        max_conds_dim : int or None
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, default: 1
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.

        Returns
        -------
        all_parents : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated parents.
        """
        # Create an internal copy of pc_alpha
        _int_pc_alpha = deepcopy(pc_alpha)
        # Check if we are selecting an optimal alpha value
        select_optimal_alpha = True
        # Set the default values for pc_alpha
        if _int_pc_alpha is None:
            _int_pc_alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        elif not isinstance(_int_pc_alpha, (list, tuple, np.ndarray)):
            _int_pc_alpha = [_int_pc_alpha]
            select_optimal_alpha = False
        # Check the limits on tau_min
        self._check_tau_limits(tau_min, tau_max)
        tau_min = max(1, tau_min)
        # Check that the maximum combinations variable is correct
        if max_combinations <= 0:
            raise ValueError("max_combinations must be > 0")
        # Implement defaultdict for all pval_max, val_max, and iterations
        pval_max = defaultdict(dict)
        val_min = defaultdict(dict)
        iterations = defaultdict(dict)

        if self.verbosity > 0:
            self._print_pc_params(selected_links, tau_min, tau_max,
                              _int_pc_alpha, max_conds_dim,
                              max_combinations)

        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max,
                                             remove_contemp=True)

        # Initialize all parents
        all_parents = dict()
        # Set the maximum condition dimension
        max_conds_dim = self._set_max_condition_dim(max_conds_dim,
                                                    tau_min, tau_max)

        # Loop through the selected variables
        for j in range(self.N):
            # Print the status of this variable
            if self.verbosity > 1:
                print("\n## Variable %s" % self.var_names[j])
                print("\nIterating through pc_alpha = %s:" % _int_pc_alpha)
            # Initialize the scores for selecting the optimal alpha
            score = np.zeros_like(_int_pc_alpha)
            # Initialize the result
            results = {}
            for iscore, pc_alpha_here in enumerate(_int_pc_alpha):
                # Print statement about the pc_alpha being tested
                if self.verbosity > 1:
                    print("\n# pc_alpha = %s (%d/%d):" % (pc_alpha_here,
                                                          iscore + 1,
                                                          score.shape[0]))
                # Get the results for this alpha value
                results[pc_alpha_here] = \
                    self._run_pc_stable_single(j,
                                               selected_links=_int_sel_links[j],
                                               tau_min=tau_min,
                                               tau_max=tau_max,
                                               save_iterations=save_iterations,
                                               pc_alpha=pc_alpha_here,
                                               max_conds_dim=max_conds_dim,
                                               max_combinations=max_combinations)
                # Figure out the best score if there is more than one pc_alpha
                # value
                if select_optimal_alpha:
                    score[iscore] = \
                        self.cond_ind_test.get_model_selection_criterion(
                            j, results[pc_alpha_here]['parents'], tau_max)
            # Record the optimal alpha value
            optimal_alpha = _int_pc_alpha[score.argmin()]
            # Only print the selection results if there is more than one
            # pc_alpha
            if self.verbosity > 1 and select_optimal_alpha:
                self._print_pc_sel_results(_int_pc_alpha, results, j,
                                           score, optimal_alpha)
            # Record the results for this variable
            all_parents[j] = results[optimal_alpha]['parents']
            val_min[j] = results[optimal_alpha]['val_min']
            pval_max[j] = results[optimal_alpha]['pval_max']
            iterations[j] = results[optimal_alpha]['iterations']
            # Only save the optimal alpha if there is more than one pc_alpha
            if select_optimal_alpha:
                iterations[j]['optimal_pc_alpha'] = optimal_alpha
        # Save the results in the current status of the algorithm
        self.all_parents = all_parents
        self.val_matrix = self._dict_to_matrix(val_min, tau_max, self.N, 
                                               default=0.)
        self.p_matrix = self._dict_to_matrix(pval_max, tau_max, self.N,
                                            default=1.)
        self.iterations = iterations
        self.val_min = val_min
        self.pval_max = pval_max
        # Print the results
        if self.verbosity > 0:
            print("\n## Resulting lagged parent (super)sets:")
            self._print_parents(all_parents, val_min, pval_max)
        # Return the parents
        return all_parents

    def _print_parents_single(self, j, parents, val_min, pval_max):
        """Print current parents for variable j.

        Parameters
        ----------
        j : int
            Index of current variable.
        parents : list
            List of form [(0, -1), (3, -2), ...].
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.
        pval_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        """
        if len(parents) < 20 or hasattr(self, 'iterations'):
            print("\n    Variable %s has %d link(s):" % (
                self.var_names[j], len(parents)))
            if (hasattr(self, 'iterations')
                    and 'optimal_pc_alpha' in list(self.iterations[j])):
                print("    [pc_alpha = %s]" % (
                    self.iterations[j]['optimal_pc_alpha']))
            if val_min is None or pval_max is None:
                for p in parents:
                    print("        (%s % .d)" % (
                        self.var_names[p[0]], p[1]))
            else:
                for p in parents:
                    print("        (%s % .d): max_pval = %.5f, min_val = % .3f" % (
                        self.var_names[p[0]], p[1], pval_max[p],
                        val_min[p]))
        else:
            print("\n    Variable %s has %d link(s):" % (
                self.var_names[j], len(parents)))

    def _print_parents(self, all_parents, val_min, pval_max):
        """Print current parents.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the conditioning-parents estimated with PC algorithm.
        val_min : dict
            Dictionary of form {0:{(0, -1):float, ...}} containing the minimum
            test statistic value of a link.
        pval_max : dict
            Dictionary of form {0:{(0, -1):float, ...}} containing the maximum
            p-value of a link across different conditions.
        """
        for j in [var for var in list(all_parents)]:
            if val_min is None or pval_max is None:
                self._print_parents_single(j, all_parents[j],
                                           None, None)
            else:
                self._print_parents_single(j, all_parents[j],
                                           val_min[j], pval_max[j])

    def _mci_condition_to_string(self, conds):
        """Convert the list of conditions into a string.

        Parameters
        ----------
        conds : list
            List of conditions.
        """
        cond_string = "[ "
        for k, tau_k in conds:
            cond_string += "(%s % d) " % (self.var_names[k], tau_k)
        cond_string += "]"
        return cond_string

    def _print_mci_conditions(self, conds_y, conds_x_lagged,
                              j, i, tau, count, n_parents):
        """Print information about the conditions for the MCI algorithm.

        Parameters
        ----------
        conds_y : list
            Conditions on node.
        conds_x_lagged : list
            Conditions on parent.
        j : int
            Current node.
        i : int
            Parent node.
        tau : int
            Parent time delay.
        count : int
            Index of current parent.
        n_parents : int
            Total number of parents.
        """
        # Remove the current parent from the conditions
        conds_y_no_i = [node for node in conds_y if node != (i, tau)]
        # Get the condition string for parent
        condy_str = self._mci_condition_to_string(conds_y_no_i)
        # Get the condition string for node
        condx_str = self._mci_condition_to_string(conds_x_lagged)
        # Formate and print the information
        indent = "\n        "
        print_str = indent + "link (%s % d) " % (self.var_names[i], tau)
        print_str += "--> %s (%d/%d):" % (
            self.var_names[j], count + 1, n_parents)
        print_str += indent + "with conds_y = %s" % (condy_str)
        print_str += indent + "with conds_x = %s" % (condx_str)
        print(print_str)

    def _print_pcmciplus_conditions(self, lagged_parents, i, j, abstau,
                                    max_conds_py, max_conds_px, 
                                    max_conds_px_lagged):
        """Print information about the conditions for PCMCIplus.

        Parameters
        ----------
        lagged_parents : dictionary of lists
            Dictionary of lagged parents for each node.
        j : int
            Current node.
        i : int
            Parent node.
        abstau : int
            Parent time delay.
        max_conds_py : int
            Max number of parents for node j.
        max_conds_px : int
            Max number of parents for lagged node i.
        max_conds_px_lagged : int
            Maximum number of lagged conditions of X when X is lagged in MCI 
            tests. If None is passed, this number is equal to max_conds_px.
        """
        conds_y = lagged_parents[j][:max_conds_py]
        conds_y_no_i = [node for node in conds_y if node != (i, -abstau)]
        if abstau == 0:
            conds_x = lagged_parents[i][:max_conds_px]
        else:
            if max_conds_px_lagged is None:
                conds_x = lagged_parents[i][:max_conds_px]
            else:
                conds_x = lagged_parents[i][:max_conds_px_lagged]

        # Shift the conditions for X by tau
        conds_x_lagged = [(k, -abstau + k_tau) for k, k_tau in conds_x]
        condy_str = self._mci_condition_to_string(conds_y_no_i)
        condx_str = self._mci_condition_to_string(conds_x_lagged)
        print_str = "    with conds_y = %s" % (condy_str)
        print_str += "\n    with conds_x = %s" % (condx_str)
        print(print_str)

    def _get_int_parents(self, parents):
        """Get the input parents dictionary.

        Parameters
        ----------
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.

        Returns
        -------
        int_parents : defaultdict of lists
            Internal copy of parents, respecting default options
        """
        int_parents = deepcopy(parents)
        if int_parents is None:
            int_parents = defaultdict(list)
        else:
            int_parents = defaultdict(list, int_parents)
        return int_parents

    def _iter_indep_conds(self,
                          parents,
                          selected_links,
                          max_conds_py,
                          max_conds_px):
        """Iterate through the conditions dictated by the arguments, yielding
        the needed arguments for conditional independence functions.

        Parameters
        ----------
        parents : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable.
        selected_links : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested.
        max_conds_py : int
            Maximum number of conditions of Y to use.
        max_conds_px : int
            Maximum number of conditions of Z to use.

        Yields
        ------
        i, j, tau, Z : list of tuples
            (i, tau) is the parent node, (j, 0) is the current node, and Z is of
            the form [(var, tau + tau')] and specifies the condition to test
        """
        # Loop over the selected variables
        for j in range(self.N):
            # Get the conditions for node j
            conds_y = parents[j][:max_conds_py]
            # Create a parent list from links seperated in time and by node
            parent_list = [(i, tau) for i, tau in selected_links[j]
                           if (i, tau) != (j, 0)]
            # Iterate through parents (except those in conditions)
            for cnt, (i, tau) in enumerate(parent_list):
                # Get the conditions for node i
                conds_x = parents[i][:max_conds_px]
                # Shift the conditions for X by tau
                conds_x_lagged = [(k, tau + k_tau) for k, k_tau in conds_x]
                # Print information about the mci conditions if requested
                if self.verbosity > 1:
                    self._print_mci_conditions(conds_y, conds_x_lagged, j, i,
                                               tau, cnt, len(parent_list))
                # Construct lists of tuples for estimating
                # I(X_t-tau; Y_t | Z^Y_t, Z^X_t-tau)
                # with conditions for X shifted by tau
                Z = [node for node in conds_y if node != (i, tau)]
                # Remove overlapped nodes between conds_x_lagged and conds_y
                Z += [node for node in conds_x_lagged if node not in Z]
                # Yield these list
                yield j, i, tau, Z

    def _run_mci_or_variants(self,
                             selected_links=None,
                             tau_min=0,
                             tau_max=1,
                             parents=None,
                             max_conds_py=None,
                             max_conds_px=None,
                             val_only=False,
                             alpha_level=0.05,
                             fdr_method='none'):
        """Base function for MCI method and variants.

        Returns the matrices of test statistic values, (optionally corrected) 
        p-values, and (optionally) confidence intervals. Also (new in 4.3)
        returns graph based on alpha_level (and optional FDR-correction).

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.
        max_conds_py : int or None
            Maximum number of conditions of Y to use. If None is passed, this
            number is unrestricted.
        max_conds_px : int or None
            Maximum number of conditions of Z to use. If None is passed, this
            number is unrestricted.
        val_only : bool, default: False
            Option to only compute dependencies and not p-values.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method. 

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """
        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)

        # Set the maximum condition dimension for Y and X
        max_conds_py = self._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = self._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)
        # Get the parents that will be checked
        _int_parents = self._get_int_parents(parents)
        # Initialize the return values
        val_matrix = np.zeros((self.N, self.N, tau_max + 1))
        p_matrix = np.ones((self.N, self.N, tau_max + 1))
        # Initialize the optional return of the confidance matrix
        conf_matrix = None
        if self.cond_ind_test.confidence is not None:
            conf_matrix = np.zeros((self.N, self.N, tau_max + 1, 2))

        # Get the conditions as implied by the input arguments
        for j, i, tau, Z in self._iter_indep_conds(_int_parents,
                                                   _int_sel_links,
                                                   max_conds_py,
                                                   max_conds_px):
            # Set X and Y (for clarity of code)
            X = [(i, tau)]
            Y = [(j, 0)]

            if val_only is False:
                # Run the independence tests and record the results
                val, pval = self.cond_ind_test.run_test(X, Y, Z=Z,
                                                        tau_max=tau_max,
                                                        # verbosity=
                                                        # self.verbosity
                                                        )
                val_matrix[i, j, abs(tau)] = val
                p_matrix[i, j, abs(tau)] = pval
            else:
                val = self.cond_ind_test.get_measure(X, Y, Z=Z, tau_max=tau_max)
                val_matrix[i, j, abs(tau)] = val

            # Get the confidence value, returns None if cond_ind_test.confidence
            # is False
            conf = self.cond_ind_test.get_confidence(X, Y, Z=Z, tau_max=tau_max)
            # Record the value if the conditional independence requires it
            if self.cond_ind_test.confidence:
                conf_matrix[i, j, abs(tau)] = conf

        if val_only:
            results = {'val_matrix':val_matrix,
                       'conf_matrix':conf_matrix}
            self.results = results
            return results

        # Correct the p_matrix if there is a fdr_method
        if fdr_method != 'none':
            p_matrix = self.get_corrected_pvalues(p_matrix=p_matrix, tau_min=tau_min, 
                                                  tau_max=tau_max, 
                                                  selected_links=_int_sel_links,
                                                  fdr_method=fdr_method)

        # Threshold p_matrix to get graph
        final_graph = p_matrix <= alpha_level

        # Convert to string graph representation
        graph = self.convert_to_string_graph(final_graph)

        # Symmetrize p_matrix and val_matrix
        symmetrized_results = self.symmetrize_p_and_val_matrix(
                            p_matrix=p_matrix, 
                            val_matrix=val_matrix, 
                            selected_links=_int_sel_links,
                            conf_matrix=conf_matrix)

        if self.verbosity > 0:
            self.print_significant_links(
                    graph = graph,
                    p_matrix = symmetrized_results['p_matrix'], 
                    val_matrix = symmetrized_results['val_matrix'],
                    conf_matrix = symmetrized_results['conf_matrix'],
                    alpha_level = alpha_level)

        # Return the values as a dictionary and store in class
        results = {
            'graph': graph,
            'p_matrix': symmetrized_results['p_matrix'],
            'val_matrix': symmetrized_results['val_matrix'],
            'conf_matrix': symmetrized_results['conf_matrix'],
                   }
        self.results = results
        return results

    def run_mci(self,
                selected_links=None,
                tau_min=0,
                tau_max=1,
                parents=None,
                max_conds_py=None,
                max_conds_px=None,
                val_only=False,
                alpha_level=0.05,
                fdr_method='none'):
        """MCI conditional independence tests.

        Implements the MCI test (Algorithm 2 in [1]_). 

        Returns the matrices of test statistic values, (optionally corrected) 
        p-values, and (optionally) confidence intervals. Also (new in 4.3)
        returns graph based on alpha_level (and optional FDR-correction).

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.
        max_conds_py : int or None
            Maximum number of conditions of Y to use. If None is passed, this
            number is unrestricted.
        max_conds_px : int or None
            Maximum number of conditions of Z to use. If None is passed, this
            number is unrestricted.
        val_only : bool, default: False
            Option to only compute dependencies and not p-values.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method. 

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """

        if self.verbosity > 0:
            print("\n##\n## Step 2: MCI algorithm\n##"
                  "\n\nParameters:")
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\nmax_conds_py = %s" % max_conds_py
                  + "\nmax_conds_px = %s" % max_conds_px)

        return self._run_mci_or_variants(
            selected_links=selected_links,
            tau_min=tau_min,
            tau_max=tau_max,
            parents=parents,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            val_only=val_only,
            alpha_level=alpha_level,
            fdr_method=fdr_method)

    def get_lagged_dependencies(self,
                                selected_links=None,
                                tau_min=0,
                                tau_max=1,
                                val_only=False,
                                alpha_level=0.05,
                                fdr_method='none'):
        """Unconditional lagged independence tests.

        Implements the unconditional lagged independence test (see [ 1]_).
        
        Returns the matrices of test statistic values, (optionally corrected) 
        p-values, and (optionally) confidence intervals. Also (new in 4.3)
        returns graph based on alpha_level (and optional FDR-correction).

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        val_only : bool, default: False
            Option to only compute dependencies and not p-values.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method. 

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """

        if self.verbosity > 0:
            print("\n##\n## Estimating lagged dependencies \n##"
                  "\n\nParameters:")
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max)

        return self._run_mci_or_variants(
            selected_links=selected_links,
            tau_min=tau_min,
            tau_max=tau_max,
            parents=None,
            max_conds_py=0,
            max_conds_px=0,
            val_only=val_only,
            alpha_level=alpha_level,
            fdr_method=fdr_method)

    def run_fullci(self,
                   selected_links=None,
                   tau_min=0,
                   tau_max=1,
                   val_only=False,
                   alpha_level=0.05,
                   fdr_method='none'):
        """FullCI conditional independence tests.

        Implements the FullCI test (see [1]_). 

        Returns the matrices of test statistic values, (optionally corrected) 
        p-values, and (optionally) confidence intervals. Also (new in 4.3)
        returns graph based on alpha_level (and optional FDR-correction).

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        val_only : bool, default: False
            Option to only compute dependencies and not p-values.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method. 

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """

        if self.verbosity > 0:
            print("\n##\n## Running Tigramite FullCI algorithm\n##"
                  "\n\nParameters:")
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max)

        full_past = dict([(j, [(i, -tau)
                               for i in range(self.N)
                               for tau in range(max(1, tau_min), tau_max + 1)])
                          for j in range(self.N)])

        return self._run_mci_or_variants(
            selected_links=selected_links,
            tau_min=tau_min,
            tau_max=tau_max,
            parents=full_past,
            max_conds_py=None,
            max_conds_px=0,
            val_only=val_only,
            alpha_level=alpha_level,
            fdr_method=fdr_method)

    def run_bivci(self,
                  selected_links=None,
                  tau_min=0,
                  tau_max=1,
                  val_only=False,
                  alpha_level=0.05,
                  fdr_method='none'):
        """BivCI conditional independence tests.

        Implements the BivCI test (see [1]_). 

        Returns the matrices of test statistic values, (optionally corrected) 
        p-values, and (optionally) confidence intervals. Also (new in 4.3)
        returns graph based on alpha_level (and optional FDR-correction).

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        val_only : bool, default: False
            Option to only compute dependencies and not p-values.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method. 

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.
        """

        if self.verbosity > 0:
            print("\n##\n## Running Tigramite BivCI algorithm\n##"
                  "\n\nParameters:")
            print("\nindependence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max)

        auto_past = dict([(j, [(j, -tau)
                               for tau in range(max(1, tau_min), tau_max + 1)])
                          for j in range(self.N)])

        return self._run_mci_or_variants(
            selected_links=selected_links,
            tau_min=tau_min,
            tau_max=tau_max,
            parents=auto_past,
            max_conds_py=None,
            max_conds_px=0,
            val_only=val_only,
            alpha_level=alpha_level,
            fdr_method=fdr_method)

    def get_corrected_pvalues(self, p_matrix,
                              fdr_method='fdr_bh',
                              exclude_contemporaneous=True,
                              tau_min=0,
                              tau_max=1,
                              selected_links=None,
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
            Minimum time lag. Only used as consistency check of selected_links. 
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min. Only used as 
            consistency check of selected_links. 
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
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
        # Include only selected_links if given
        if selected_links != None:
            # Create a mask for these values
            mask = np.zeros((N, N, tau_max_plusone), dtype='bool')
            _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
            for node1, links_ in _int_sel_links.items():
                for node2, lag in links_:
                    mask[node2, node1, abs(lag)] = True
        else:
            # Create a mask for these values
            mask = np.ones((N, N, tau_max_plusone), dtype='bool')
        # Ignore values from autocorrelation indices
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

    def get_graph_from_pmatrix(self, p_matrix, alpha_level, 
            tau_min, tau_max, selected_links=None):
        """Construct graph from thresholding the p_matrix at an alpha-level.

        Allows to take into account selected_links.

        Parameters
        ----------
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        tau_mix : int
            Minimum time delay to test.
        tau_max : int
            Maximum time delay to test.
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        """  

        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)

        if selected_links != None:
            # Create a mask for these values
            mask = np.zeros((self.N, self.N, tau_max + 1), dtype='bool')
            for node1, links_ in _int_sel_links.items():
                for node2, lag in links_:
                    mask[node2, node1, abs(lag)] = True
        else:
            # Create a mask for these values
            mask = np.ones((self.N, self.N, tau_max + 1), dtype='bool')

        # Set all p-values of absent links to 1.
        p_matrix[mask==False] == 1.

        # Threshold p_matrix to get graph
        graph_bool = p_matrix <= alpha_level

        # Convert to string graph representation
        graph = self.convert_to_string_graph(graph_bool)

        # Return the graph
        return graph

    def return_parents_dict(self, graph,
                             val_matrix,
                             include_lagzero_parents=False):
        """Returns dictionary of parents sorted by val_matrix.

        If parents are unclear (link with o or x), then no parent 
        is returned. 

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array-like
            Matrix of test statistic values. Must be of shape (N, N, tau_max +
            1).
        include_lagzero_parents : bool (default: False)
            Whether the dictionary should also return parents at lag
            zero. 

        Returns
        -------
        parents_dict : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated parents.
        """

        # Initialize the return value
        parents_dict = dict()
        for j in range(self.N):
            # Get the good links
            if include_lagzero_parents:
                good_links = np.argwhere(graph[:, j, :] == "-->")
                # Build a dictionary from these links to their values
                links = {(i, -tau): np.abs(val_matrix[i, j, abs(tau)])
                         for i, tau in good_links}
            else:
                good_links = np.argwhere(graph[:, j, 1:] == "-->")
                # Build a dictionary from these links to their values
                links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1])
                         for i, tau in good_links}
            # Sort by value
            parents_dict[j] = sorted(links, key=links.get, reverse=True)
        
        return parents_dict
               

    def return_significant_links(self, pq_matrix,
                                 val_matrix,
                                 alpha_level=0.05,
                                 include_lagzero_links=False):
        """Returns list of significant links as well as a boolean matrix.

        DEPRECATED. Will be removed in future.
        """
        print("return_significant_links() is DEPRECATED: now run_pcmci(), "
              " run_mci()"
              " and all variants directly return the graph based on thresholding "
              "the p_matrix at alpha_level. The graph can also be updated "
              "based on a (potentially further adjusted) p_matrix using "
              "get_graph_from_pmatrix(). "
              "A dictionary of parents can be obtained "
              "with return_parents_dict().")
        return None

    def print_significant_links(self,
                                p_matrix,
                                val_matrix,
                                conf_matrix=None,
                                graph=None,
                                ambiguous_triples=None,
                                alpha_level=0.05):
        """Prints significant links.

        Used for output of PCMCI and PCMCIplus. For the latter also information
        on ambiguous links and conflicts is returned.

        Parameters
        ----------
        alpha_level : float, optional (default: 0.05)
            Significance level.
        p_matrix : array-like
            Must be of shape (N, N, tau_max + 1).
        val_matrix : array-like
            Must be of shape (N, N, tau_max + 1).
        conf_matrix : array-like, optional (default: None)
            Matrix of confidence intervals of shape (N, N, tau_max+1, 2).
        graph : array-like
            Must be of shape (N, N, tau_max + 1).
        ambiguous_triples : list
            List of ambiguous triples.
        """
        if graph is not None:
            sig_links = (graph != "")*(graph != "<--")
        else:
            sig_links = (p_matrix <= alpha_level)

        print("\n## Significant links at alpha = %s:" % alpha_level)
        for j in range(self.N):
            links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                     for p in zip(*np.where(sig_links[:, j, :]))}
            # Sort by value
            sorted_links = sorted(links, key=links.get, reverse=True)
            n_links = len(links)
            string = ("\n    Variable %s has %d "
                      "link(s):" % (self.var_names[j], n_links))
            for p in sorted_links:
                string += ("\n        (%s % d): pval = %.5f" %
                           (self.var_names[p[0]], p[1],
                            p_matrix[p[0], j, abs(p[1])]))
                string += " | val = % .3f" % (
                    val_matrix[p[0], j, abs(p[1])])
                if conf_matrix is not None:
                    string += " | conf = (%.3f, %.3f)" % (
                        conf_matrix[p[0], j, abs(p[1])][0],
                        conf_matrix[p[0], j, abs(p[1])][1])
                if graph is not None:
                    if p[1] == 0 and graph[j, p[0], 0] == "o-o":
                        string += " | unoriented link"
                    if graph[p[0], j, abs(p[1])] == "x-x":
                        string += " | unclear orientation due to conflict"
            print(string)

        # link_marker = {True:"o-o", False:"-->"}

        if ambiguous_triples is not None and len(ambiguous_triples) > 0:
            print("\n## Ambiguous triples (not used for orientation):\n")
            for triple in ambiguous_triples:
                (i, tau), k, j = triple
                print("    [(%s % d), %s, %s]" % (
                    self.var_names[i], tau, 
                    self.var_names[k],
                    self.var_names[j]))

    def print_results(self,
                      return_dict,
                      alpha_level=0.05):
        """Prints significant parents from output of MCI or PCMCI algorithms.

        Parameters
        ----------
        return_dict : dict
            Dictionary of return values, containing keys
                * 'p_matrix'
                * 'val_matrix'
                * 'conf_matrix'

        alpha_level : float, optional (default: 0.05)
            Significance level.
        """
        # Check if conf_matrix is defined
        conf_matrix = None
        conf_key = 'conf_matrix'
        if conf_key in return_dict:
            conf_matrix = return_dict[conf_key]
        # Wrap the already defined function
        if 'graph' in return_dict:
            graph = return_dict['graph']
        else:
            graph = None
        if 'ambiguous_triples' in return_dict:
            ambiguous_triples = return_dict['ambiguous_triples']
        else:
            ambiguous_triples = None
        self.print_significant_links(return_dict['p_matrix'],
                                     return_dict['val_matrix'],
                                     conf_matrix=conf_matrix,
                                     graph=graph,
                                     ambiguous_triples=ambiguous_triples,
                                     alpha_level=alpha_level)

    def run_pcmci(self,
                  selected_links=None,
                  tau_min=0,
                  tau_max=1,
                  save_iterations=False,
                  pc_alpha=0.05,
                  max_conds_dim=None,
                  max_combinations=1,
                  max_conds_py=None,
                  max_conds_px=None,
                  alpha_level=0.05,
                  fdr_method='none'):
        """Runs PCMCI time-lagged causal discovery for time series.

        Wrapper around PC-algorithm function and MCI function.

        Notes
        -----

        The PCMCI causal discovery method is comprehensively described in [
        1]_, where also analytical and numerical results are presented. Here
        we briefly summarize the method.

        PCMCI estimates time-lagged causal links by a two-step procedure:

        1.  Condition-selection: For each variable :math:`j`, estimate a
            *superset* of parents :math:`\\tilde{\mathcal{P}}(X^j_t)` with the
            iterative PC1 algorithm, implemented as ``run_pc_stable``. The
            condition-selection step reduces the dimensionality and avoids
            conditioning on irrelevant variables.

        2.  *Momentary conditional independence* (MCI)

        .. math:: X^i_{t-\\tau} \perp X^j_{t} | \\tilde{\\mathcal{P}}(
                  X^j_t), \\tilde{\mathcal{P}}(X^i_{t-\\tau})

        here implemented as ``run_mci``. This step estimates the p-values and
        test statistic values for all links accounting for common drivers,
        indirect links, and autocorrelation.

        NOTE: MCI test statistic values define a particular measure of causal
        strength depending on the test statistic used. For example, ParCorr()
        results in normalized values between -1 and 1. However, if you are 
        interested in quantifying causal effects, i.e., the effect of
        hypothetical interventions, you may better look at the causal effect 
        estimation functionality of Tigramite.

        PCMCI can be flexibly combined with any kind of conditional
        independence test statistic adapted to the kind of data (continuous
        or discrete) and its assumed dependency types. These are available in
        ``tigramite.independence_tests``.

        The main free parameters of PCMCI (in addition to free parameters of
        the conditional independence test statistic) are the maximum time
        delay :math:`\\tau_{\\max}` (``tau_max``) and the significance
        threshold in the condition-selection step :math:`\\alpha` (
        ``pc_alpha``). The maximum time delay depends on the application and
        should be chosen according to the maximum causal time lag expected in
        the complex system. We recommend a rather large choice that includes
        peaks in the ``get_lagged_dependencies`` function. :math:`\\alpha`
        should not be seen as a significance test level in the
        condition-selection step since the iterative hypothesis tests do not
        allow for a precise assessment. :math:`\\alpha` rather takes the role
        of a regularization parameter in model-selection techniques. If a
        list of values is given or ``pc_alpha=None``, :math:`\\alpha` is
        optimized using model selection criteria implemented in the respective
        ``tigramite.independence_tests``.

        Further optional parameters are discussed in [1]_.

        Examples
        --------
        >>> import numpy
        >>> from tigramite.pcmci import PCMCI
        >>> from tigramite.independence_tests import ParCorr
        >>> import tigramite.data_processing as pp
        >>> from tigramite.toymodels import structural_causal_processes as toys
        >>> numpy.random.seed(7)
        >>> # Example process to play around with
        >>> # Each key refers to a variable and the incoming links are supplied
        >>> # as a list of format [((driver, -lag), coeff), ...]
        >>> links_coeffs = {0: [((0, -1), 0.8)],
                            1: [((1, -1), 0.8), ((0, -1), 0.5)],
                            2: [((2, -1), 0.8), ((1, -2), -0.6)]}
        >>> data, _ = toys.var_process(links_coeffs, T=1000)
        >>> # Data must be array of shape (time, variables)
        >>> print (data.shape)
        (1000, 3)
        >>> dataframe = pp.DataFrame(data)
        >>> cond_ind_test = ParCorr()
        >>> pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        >>> results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
        >>> pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                         val_matrix=results['val_matrix'],
                                         alpha_level=0.05)
        ## Significant parents at alpha = 0.05:

            Variable 0 has 1 link(s):
                (0 -1): pval = 0.00000 | val =  0.588

            Variable 1 has 2 link(s):
                (1 -1): pval = 0.00000 | val =  0.606
                (0 -1): pval = 0.00000 | val =  0.447

            Variable 2 has 2 link(s):
                (2 -1): pval = 0.00000 | val =  0.618
                (1 -2): pval = 0.00000 | val = -0.499


        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        tau_min : int, optional (default: 0)
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        save_iterations : bool, optional (default: False)
            Whether to save iteration step results such as conditions used.
        pc_alpha : float, optional (default: 0.05)
            Significance level in algorithm.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.
        max_conds_py : int, optional (default: None)
            Maximum number of conditions of Y to use. If None is passed, this
            number is unrestricted.
        max_conds_px : int, optional (default: None)
            Maximum number of conditions of Z to use. If None is passed, this
            number is unrestricted.
        alpha_level : float, optional (default: 0.05)
            Significance level at which the p_matrix is thresholded to 
            get graph.
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method. 

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of test statistic values.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values, optionally adjusted if fdr_method is
            not 'none'.
        conf_matrix : array of shape [N, N, tau_max+1,2]
            Estimated matrix of confidence intervals of test statistic values.
            Only computed if set in cond_ind_test, where also the percentiles
            are set.

        """

        # Get the parents from run_pc_stable
        all_parents = self.run_pc_stable(selected_links=selected_links,
                                         tau_min=tau_min,
                                         tau_max=tau_max,
                                         save_iterations=save_iterations,
                                         pc_alpha=pc_alpha,
                                         max_conds_dim=max_conds_dim,
                                         max_combinations=max_combinations)
        # Get the results from run_mci, using the parents as the input
        results = self.run_mci(selected_links=selected_links,
                               tau_min=tau_min,
                               tau_max=tau_max,
                               parents=all_parents,
                               max_conds_py=max_conds_py,
                               max_conds_px=max_conds_px,
                               alpha_level=alpha_level,
                               fdr_method=fdr_method)
    
        # Store the parents in the pcmci member
        self.all_parents = all_parents

        # Print the information
        # if self.verbosity > 0:
        #     self.print_results(results)
        # Return the dictionary
        self.results = results
        return results

    def run_pcmciplus(self,
                      selected_links=None,
                      tau_min=0,
                      tau_max=1,
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
        time series.

        Method described in [5]_: 
        http://www.auai.org/~w-auai/uai2020/proceedings/579_main_paper.pdf

        Notes
        -----

        The PCMCIplus causal discovery method is described in [5]_, where
        also analytical and numerical results are presented. In contrast to
        PCMCI, PCMCIplus can identify the full, lagged and contemporaneous,
        causal graph (up to the Markov equivalence class for contemporaneous
        links) under the standard assumptions of Causal Sufficiency,
        Faithfulness and the Markov condition.

        PCMCIplus estimates time-lagged and contemporaneous causal links by a
        four-step procedure:

        1.  Condition-selection (same as for PCMCI): For each variable
        :math:`j`, estimate a *superset* of lagged parents :math:`\\widehat{
        \\mathcal{B}}_t^-( X^j_t)` with the iterative PC1 algorithm,
        implemented as ``run_pc_stable``. The condition-selection step
        reduces the dimensionality and avoids conditioning on irrelevant
        variables.

        2.   PC skeleton phase with contemporaneous conditions and *Momentary
        conditional independence* (MCI) tests: Iterate through subsets
        :math:`\\mathcal{S}` of contemporaneous adjacencies and conduct MCI
        conditional independence tests:

        .. math:: X^i_{t-\\tau} ~\\perp~ X^j_{t} ~|~ \\mathcal{S},
                  \\widehat{\\mathcal{B}}_t^-(X^j_t),
                  \\widehat{\\mathcal{B}}_{t-\\tau}^-(X^i_{t-{\\tau}})

        here implemented as ``run_pcalg``. This step estimates the p-values and
        test statistic values for all lagged and contemporaneous adjacencies
        accounting for common drivers, indirect links, and autocorrelation.

        3.   PC collider orientation phase: Orient contemporaneous collider
        motifs based on unshielded triples. Optionally apply conservative or
        majority rule (also based on MCI tests).

        4.   PC rule orientation phase: Orient remaining contemporaneous
        links based on PC rules.

        In contrast to PCMCI, the relevant output of PCMCIplus is the
        array ``graph``. Its string entries are interpreted as follows:

        * ``graph[i,j,tau]=-->`` for :math:`\\tau>0` denotes a directed, lagged
          causal link from :math:`i` to :math:`j` at lag :math:`\\tau`

        * ``graph[i,j,0]=-->`` (and ``graph[j,i,0]=<--``) denotes a directed,
          contemporaneous causal link from :math:`i` to :math:`j`

        * ``graph[i,j,0]=o-o`` (and ``graph[j,i,0]=o-o``) denotes an unoriented,
          contemporaneous adjacency between :math:`i` and :math:`j` indicating
          that the collider and orientation rules could not be applied (Markov
          equivalence)

        * ``graph[i,j,0]=x-x`` and (``graph[j,i,0]=x-x``) denotes a conflicting,
          contemporaneous adjacency between :math:`i` and :math:`j` indicating
          that the directionality is undecided due to conflicting orientation
          rules

        Importantly, ``p_matrix`` and ``val_matrix`` for PCMCIplus quantify
        the uncertainty and strength, respectively, only for the
        adjacencies, but not for the directionality of contemporaneous links.
        Note that lagged links are always oriented due to time order.

        PCMCIplus can be flexibly combined with any kind of conditional
        independence test statistic adapted to the kind of data (continuous
        or discrete) and its assumed dependency types. These are available in
        ``tigramite.independence_tests``.

        The main free parameters of PCMCIplus (in addition to free parameters of
        the conditional independence tests) are the maximum time delay
        :math:`\\tau_{\\max}` (``tau_max``) and the significance threshold
        :math:`\\alpha` ( ``pc_alpha``). 

        If a list or None is passed for ``pc_alpha``, the significance level is
        optimized for every graph across the given ``pc_alpha`` values using the
        score computed in ``cond_ind_test.get_model_selection_criterion()``.
        Since PCMCIplus outputs not a DAG, but an equivalence class of DAGs,
        first one member of this class is computed and then the score is
        computed as the average over all models fits for each variable in ``[0,
        ..., N]`` for that member. The score is the same for all members of the
        class.

        The maximum time delay depends on the application and should be chosen
        according to the maximum causal time lag expected in the complex system.
        We recommend a rather large choice that includes peaks in the
        ``get_lagged_dependencies`` function. Another important parameter is
        ``contemp_collider_rule``. Only if set to ``majority`` or
        ``conservative'' and together with ``conflict_resolution=True``,
        PCMCIplus is fully *order independent* meaning that the order of the N
        variables in the dataframe does not matter. Last, the default option
        ``reset_lagged_links=False`` restricts the detection of lagged causal
        links in Step 2 to the significant adjacencies found in Step 1, given by
        :math:`\\widehat{ \\mathcal{B}}_t^-( X^j_t)`. For
        ``reset_lagged_links=True``, *all* lagged links are considered again,
        which improves detection power for lagged links, but also leads to
        larger runtimes.

        Further optional parameters are discussed in [5]_.

        Examples
        --------
        >>> import numpy as np
        >>> from tigramite.pcmci import PCMCI
        >>> from tigramite.independence_tests import ParCorr
        >>> import tigramite.data_processing as pp
        >>> from tigramite.toymodels import structural_causal_processes as toys
        >>> # Example process to play around with
        >>> # Each key refers to a variable and the incoming links are supplied
        >>> # as a list of format [((var, -lag), coeff, function), ...]
        >>> def lin_f(x): return x
        >>> links = {0: [((0, -1), 0.9, lin_f)],
                     1: [((1, -1), 0.8, lin_f), ((0, -1), 0.8, lin_f)],
                     2: [((2, -1), 0.7, lin_f), ((1, 0), 0.6, lin_f)],
                     3: [((3, -1), 0.7, lin_f), ((2, 0), -0.5, lin_f)],
                     }
        >>> data, nonstat = toys.structural_causal_process(links,
                            T=1000, seed=7)
        >>> # Data must be array of shape (time, variables)
        >>> print (data.shape)
        (1000, 4)
        >>> dataframe = pp.DataFrame(data)
        >>> cond_ind_test = ParCorr()
        >>> pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        >>> results = pcmci.run_pcmciplus(tau_min=0, tau_max=2, pc_alpha=0.01)
        >>> pcmci.print_results(results, alpha_level=0.01)
            ## Significant links at alpha = 0.01:

            Variable 0 has 1 link(s):
                (0 -1): pval = 0.00000 | val =  0.676

            Variable 1 has 2 link(s):
                (1 -1): pval = 0.00000 | val =  0.602
                (0 -1): pval = 0.00000 | val =  0.599

            Variable 2 has 2 link(s):
                (1  0): pval = 0.00000 | val =  0.486
                (2 -1): pval = 0.00000 | val =  0.466

            Variable 3 has 2 link(s):
                (3 -1): pval = 0.00000 | val =  0.524
                (2  0): pval = 0.00000 | val = -0.449 

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, 0), (0, -1), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
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

        # Check if pc_alpha is chosen to optimze over a list
        if pc_alpha is None or isinstance(pc_alpha, (list, tuple, np.ndarray)):
            # Call optimizer wrapper around run_pcmciplus()
            return self._optimize_pcmciplus_alpha(
                        selected_links=selected_links,
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

        # For the lagged PC algorithm only the strongest conditions are tested
        max_combinations = 1

        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)

        # Step 1: Get a superset of lagged parents from run_pc_stable
        lagged_parents = self.run_pc_stable(selected_links=selected_links,
                                            tau_min=tau_min,
                                            tau_max=tau_max,
                                            pc_alpha=pc_alpha,
                                            max_conds_dim=max_conds_dim,
                                            max_combinations=max_combinations)

        p_matrix = self.p_matrix
        val_matrix = self.val_matrix

        # Step 2+3+4: PC algorithm with contemp. conditions and MCI tests
        if self.verbosity > 0:
            print("\n##\n## Step 2: PC algorithm with contemp. conditions "
                  "and MCI tests\n##"
                  "\n\nParameters:")
            if selected_links is not None:
                print("\nselected_links = %s" % _int_sel_links)
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
            links_for_pc = deepcopy(_int_sel_links)
        else:
            # Run PCalg only on lagged parents found with PC1 
            # plus all contemporaneous links
            links_for_pc = deepcopy(lagged_parents)
            for j in range(self.N):
                for link in _int_sel_links[j]:
                    i, tau = link
                    if abs(tau) == 0:
                        links_for_pc[j].append((i, 0))

        results = self.run_pcalg(
            selected_links=links_for_pc,
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
                                                  selected_links=_int_sel_links,
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
        self.results = return_dict
        return return_dict

    def run_pcalg(self, selected_links=None, pc_alpha=0.01, tau_min=0,
                  tau_max=1, max_conds_dim=None, max_combinations=None,
                  lagged_parents=None, max_conds_py=None, max_conds_px=None,
                  max_conds_px_lagged=None,
                  mode='standard', contemp_collider_rule='majority',
                  conflict_resolution=True):

        """Runs PC algorithm for time-lagged and contemporaneous causal
        discovery for time series.

        For ``mode='contemp_conds'`` this implements Steps 2-4 of the
        PCMCIplus method described in [5]_. For ``mode='standard'`` this
        implements the standard PC algorithm adapted to time series.

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0: [(3, 0), (0, -1), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.
        lagged_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            additional conditions for each CI test. As part of PCMCIplus
            these are the superset of lagged parents estimated with the PC1
            algorithm.
        mode : {'standard', 'contemp_conds'}
            For ``mode='contemp_conds'`` this implements Steps 2-4 of the
            PCMCIplus method. For ``mode='standard'`` this implements the
            standard PC algorithm adapted to time series.
        tau_min : int, optional (default: 0)
            Minimum time lag to test.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        pc_alpha : float, optional (default: 0.01)
            Significance level.
        contemp_collider_rule : {'majority', 'conservative', 'none'}
            Rule for collider phase to use. See the paper for details. Only
            'majority' and 'conservative' lead to an order-independent
            algorithm.
        conflict_resolution : bool, optional (default: True)
            Whether to mark conflicts in orientation rules. Only for True
            this leads to an order-independent algorithm.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int
            Maximum number of combinations of conditions of current cardinality
            to test.
        max_conds_py : int, optional (default: None)
            Maximum number of lagged conditions of Y to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px : int, optional (default: None)
            Maximum number of lagged conditions of X to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px_lagged : int, optional (default: None)
            Maximum number of lagged conditions of X when X is lagged in MCI 
            tests. If None is passed, this number is equal to max_conds_px.

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
        # TODO: save_iterations

        # Sanity checks
        if pc_alpha is None:
            raise ValueError("pc_alpha=None not supported in PC algorithm, "
                             "choose 0 < pc_alpha < 1 (e.g., 0.01)")

        if mode not in ['contemp_conds', 'standard']:
            raise ValueError("mode must be either 'contemp_conds' or "
                             "'standard'")

        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)

        if max_conds_dim is None:
            if mode == 'standard':
                max_conds_dim = self._set_max_condition_dim(max_conds_dim,
                                                            tau_min, tau_max)
            elif mode == 'contemp_conds':
                max_conds_dim = self.N

        if max_combinations is None:
            max_combinations = np.inf

        # Initialize initial graph from selected_links
        initial_graph = np.zeros((self.N, self.N, tau_max + 1), dtype='uint8')
        for j in range(self.N):
            for link in _int_sel_links[j]:
                i, tau = link
                initial_graph[i, j, abs(tau)] = 1

        skeleton_results = self._pcalg_skeleton(
            initial_graph=initial_graph,
            lagged_parents=lagged_parents,
            mode=mode,
            pc_alpha=pc_alpha,
            tau_min=tau_min,
            tau_max=tau_max,
            max_conds_dim=max_conds_dim,
            max_combinations=max_combinations,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            max_conds_px_lagged=max_conds_px_lagged,
        )

        skeleton_graph = skeleton_results['graph']
        sepset = skeleton_results['sepset']

        colliders_step_results = self._pcalg_colliders(
            graph=skeleton_graph,
            sepset=sepset,
            lagged_parents=lagged_parents,
            mode=mode,
            pc_alpha=pc_alpha,
            tau_max=tau_max,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
            max_conds_px_lagged=max_conds_px_lagged,
            conflict_resolution=conflict_resolution,
            contemp_collider_rule=contemp_collider_rule,
            )

        collider_graph = colliders_step_results['graph']
        ambiguous_triples = colliders_step_results['ambiguous_triples']

        final_graph = self._pcalg_rules_timeseries(
            graph=collider_graph,
            ambiguous_triples=ambiguous_triples,
            conflict_resolution=conflict_resolution,
        )

        # Symmetrize p_matrix and val_matrix
        symmetrized_results = self.symmetrize_p_and_val_matrix(
                            p_matrix=skeleton_results['p_matrix'], 
                            val_matrix=skeleton_results['val_matrix'], 
                            selected_links=_int_sel_links,
                            conf_matrix=None)

        # Convert numerical graph matrix to string
        graph_str = self.convert_to_string_graph(final_graph)

        pc_results = {
            'graph': graph_str,
            'p_matrix': symmetrized_results['p_matrix'],
            'val_matrix': symmetrized_results['val_matrix'],
            'sepset': colliders_step_results['sepset'],
            'ambiguous_triples': colliders_step_results['ambiguous_triples'],
        }

        if self.verbosity > 1:
            print("\n-----------------------------")
            print("PCMCIplus algorithm finished.")
            print("-----------------------------")

        self.pc_results = pc_results
        return pc_results

    def run_pcalg_non_timeseries_data(self, pc_alpha=0.01,
                  max_conds_dim=None, max_combinations=None, 
                  contemp_collider_rule='majority',
                  conflict_resolution=True):

        """Runs PC algorithm for non-time series data.

        Simply calls run_pcalg with tau_min = tau_max = 0.
        Removes lags from ouput dictionaries.

        Parameters
        ----------
        pc_alpha : float, optional (default: 0.01)
            Significance level.
        contemp_collider_rule : {'majority', 'conservative', 'none'}
            Rule for collider phase to use. See the paper for details. Only
            'majority' and 'conservative' lead to an order-independent
            algorithm.
        conflict_resolution : bool, optional (default: True)
            Whether to mark conflicts in orientation rules. Only for True
            this leads to an order-independent algorithm.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int
            Maximum number of combinations of conditions of current cardinality
            to test.

        Returns
        -------
        graph : array of shape [N, N, 1]
            Resulting causal graph, see description above for interpretation.
        val_matrix : array of shape [N, N, 1]
            Estimated matrix of test statistic values regarding adjacencies.
        p_matrix : array of shape [N, N, 1]
            Estimated matrix of p-values regarding adjacencies.
        sepset : dictionary
            Separating sets. See paper for details.
        ambiguous_triples : list
            List of ambiguous triples, only relevant for 'majority' and
            'conservative' rules, see paper for details.
        """

        results = self.run_pcalg(pc_alpha=pc_alpha, tau_min=0, tau_max=0, 
                    max_conds_dim=max_conds_dim, max_combinations=max_combinations,
                  mode='standard', contemp_collider_rule=contemp_collider_rule,
                  conflict_resolution=conflict_resolution)

        # Remove tau-dimension
        # results['graph'] = results['graph'].squeeze()
        # results['val_matrix'] = results['val_matrix'].squeeze()
        # results['p_matrix'] = results['p_matrix'].squeeze()
        old_sepsets = results['sepset'].copy()
        results['sepset'] = {}
        for old_sepset in old_sepsets:
           new_sepset = (old_sepset[0][0], old_sepset[1])
           conds = [cond[0] for cond in old_sepsets[old_sepset]]

           results['sepset'][new_sepset] = conds

        ambiguous_triples = results['ambiguous_triples'].copy()
        results['ambiguous_triples'] = []
        for triple in ambiguous_triples:
           new_triple = (triple[0][0], triple[1], triple[2])

           results['ambiguous_triples'].append(new_triple)
        
        self.pc_results = results
        return results


    def _run_pcalg_test(self, i, abstau, j, S, lagged_parents, max_conds_py,
                        max_conds_px, max_conds_px_lagged, tau_max):
        """MCI conditional independence tests within PCMCIplus or PC algorithm.

        Parameters
        ----------
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

        # Perform independence test adding lagged parents
        if lagged_parents is not None:
            conds_y = lagged_parents[j][:max_conds_py]
            # Get the conditions for node i
            if abstau == 0:
                conds_x = lagged_parents[i][:max_conds_px]
            else:
                if max_conds_px_lagged is None:
                    conds_x = lagged_parents[i][:max_conds_px]
                else:
                    conds_x = lagged_parents[i][:max_conds_px_lagged]

        else:
            conds_y = conds_x = []
        # Shift the conditions for X by tau
        conds_x_lagged = [(k, -abstau + k_tau) for k, k_tau in conds_x]

        Z = [node for node in S]
        Z += [node for node in conds_y if
              node != (i, -abstau) and node not in Z]
        # Remove overlapping nodes between conds_x_lagged and conds_y
        Z += [node for node in conds_x_lagged if node not in Z]

        val, pval = self.cond_ind_test.run_test(X=[(i, -abstau)], Y=[(j, 0)],
                                                Z=Z, tau_max=tau_max,
                                                # verbosity=self.verbosity
                                                )

        return val, pval, Z

    def _print_triple_info(self, triple, index, n_triples):
        """Print info about the current triple being tested.

        Parameters
        ----------
        triple : tuple
            Standard ((i, tau), k, j) tuple of nodes and time delays.
        index : int
            Index of triple.
        n_triples : int
            Total number of triples.
        """
        (i, tau), k, j = triple
        link_marker = {True:"o-o", False:"-->"}

        print("\n    Triple (%s % d) %s %s o-o %s (%d/%d)" % (
            self.var_names[i], tau, link_marker[tau==0], self.var_names[k],
            self.var_names[j], index + 1, n_triples))


    def _tests_remaining(self, i, j, abstau, graph, adjt, p):
        """Helper function returning whether a certain pair still needs to be
        tested."""
        return graph[i, j, abstau] != 0 and len(
            [a for a in adjt[j] if a != (i, -abstau)]) >= p

    def _any_tests_remaining(self, graph, adjt, tau_min, tau_max, p):
        """Helper function returning whether any pair still needs to be
        tested."""
        remaining_pairs = self._remaining_pairs(graph, adjt, tau_min, tau_max,
                                                p)

        if len(remaining_pairs) > 0:
            return True
        else:
            return False

    def _remaining_pairs(self, graph, adjt, tau_min, tau_max, p):
        """Helper function returning the remaining pairs that still need to be
        tested."""
        N = graph.shape[0]
        pairs = []
        for (i, j) in itertools.product(range(N), range(N)):
            for abstau in range(tau_min, tau_max + 1):
                if (graph[i, j, abstau] != 0
                        and len(
                            [a for a in adjt[j] if a != (i, -abstau)]) >= p):
                    pairs.append((i, j, abstau))

        return pairs

    def _pcalg_skeleton(self,
                       initial_graph,
                       lagged_parents,
                       mode,
                       pc_alpha,
                       tau_min,
                       tau_max,
                       max_conds_dim,
                       max_combinations,
                       max_conds_py,
                       max_conds_px,
                       max_conds_px_lagged,
                       ):
        """Implements the skeleton discovery step of the PC algorithm for
        time series.

        Parameters
        ----------
        initial_graph : array of shape (N, N, tau_max+1) or None
            Initial graph.
        lagged_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            additional conditions for each CI test. As part of PCMCIplus
            these are the superset of lagged parents estimated with the PC1
            algorithm.
        mode : {'standard', 'contemp_conds'}
            For ``mode='contemp_conds'`` this implements Steps 2-4 of the
            PCMCIplus method. For ``mode='standard'`` this implements the
            standard PC algorithm adapted to time series.
        tau_min : int, optional (default: 0)
            Minimum time lag to test.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        pc_alpha : float, optional (default: 0.01)
            Significance level.
        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.
        max_combinations : int
            Maximum number of combinations of conditions of current cardinality
            to test.
        max_conds_py : int, optional (default: None)
            Maximum number of lagged conditions of Y to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px : int, optional (default: None)
            Maximum number of lagged conditions of X to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px_lagged : int, optional (default: None)
            Maximum number of lagged conditions of X when X is lagged in MCI 
            tests. If None is passed, this number is equal to max_conds_px.

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
        """
        N = self.N

        # Form complete graph
        if initial_graph is None:
            graph = np.ones((N, N, tau_max + 1), dtype='int')
        else:
            graph = initial_graph

        # Remove lag-zero self-loops
        graph[range(N), range(N), 0] = 0

        # Define adjacencies for standard and contemp_conds mode
        if mode == 'contemp_conds':
            adjt = self._get_adj_time_series_contemp(graph)
        elif mode == 'standard':
            adjt = self._get_adj_time_series(graph)

        val_matrix = np.zeros((N, N, tau_max + 1))
        val_min = dict()
        for j in range(self.N):
            val_min[j] = {(p[0], -p[1]): np.inf
                          for p in zip(*np.where(graph[:, j, :]))}

        # Initialize p-values. Set to 1 if there's no link in the initial graph
        pvalues = np.zeros((N, N, tau_max + 1))
        pvalues[graph == 0] = 1.
        pval_max = dict()
        for j in range(self.N):
            pval_max[j] = {(p[0], -p[1]): 0.
                           for p in zip(*np.where(graph[:, j, :]))}

        # TODO: Remove sepset alltogether?
        # Intialize sepsets that store the conditions that make i and j
        # independent
        sepset = self._get_sepset(tau_min, tau_max)

        if self.verbosity > 1:
            print("\n--------------------------")
            print("Skeleton discovery phase")
            print("--------------------------")

        # Start with zero cardinality conditions
        p = 0
        while (self._any_tests_remaining(graph, adjt, tau_min, tau_max,
                                         p) and p <= max_conds_dim):
            if self.verbosity > 1:
                print(
                    "\nTesting contemporaneous condition sets of dimension "
                    "%d: " % p)

            remaining_pairs = self._remaining_pairs(graph, adjt, tau_min,
                                                    tau_max, p)
            n_remaining = len(remaining_pairs)
            for ir, (i, j, abstau) in enumerate(remaining_pairs):
                # Check if link was not already removed (contemp links)
                if graph[i, j, abstau]:
                    if self.verbosity > 1:
                        self._print_link_info(j=j, index_parent=ir,
                                              parent=(i, -abstau),
                                              num_parents=n_remaining)

                    # Generate all subsets of conditions of cardinality p
                    conditions = list(itertools.combinations(
                        [(k, tauk) for (k, tauk) in adjt[j]
                         if not (k == i and tauk == -abstau)], p))

                    n_conditions = len(conditions)
                    if self.verbosity > 1:
                        print(
                            "    Iterate through %d subset(s) of conditions: "
                            % n_conditions)
                        if lagged_parents is not None:
                            self._print_pcmciplus_conditions(lagged_parents, i,
                                                         j, abstau,
                                                         max_conds_py,
                                                         max_conds_px,
                                                         max_conds_px_lagged)
                    nonsig = False
                    # Iterate through condition sets
                    for q, S in enumerate(conditions):
                        if q > max_combinations:
                            break

                        # Run MCI test
                        val, pval, Z = self._run_pcalg_test(
                            i, abstau, j, S, lagged_parents, max_conds_py,
                            max_conds_px, max_conds_px_lagged, tau_max)

                        # Store minimum test statistic value for sorting adjt
                        # (only internally used)
                        val_min[j][(i, -abstau)] = min(np.abs(val),
                                                       val_min[j].get(
                                                           (i, -abstau)))
                        # Store maximum p-value (only internally used)
                        pval_max[j][(i, -abstau)] = max(pval,
                                                        pval_max[j].get(
                                                            (i, -abstau)))

                        # Store max. p-value and corresponding value to return
                        if pval >= pvalues[i, j, abstau]:
                            pvalues[i, j, abstau] = pval
                            val_matrix[i, j, abstau] = val

                        if self.verbosity > 1:
                            self._print_cond_info(Z=S, comb_index=q, pval=pval,
                                                  val=val)

                        # If conditional independence is found, remove link
                        # from graph and store sepset
                        if pval > pc_alpha:
                            nonsig = True
                            if abstau == 0:
                                graph[i, j, 0] = graph[j, i, 0] = 0
                                sepset[((i, 0), j)] = sepset[
                                    ((j, 0), i)] = list(S)
                            else:
                                graph[i, j, abstau] = 0
                                sepset[((i, -abstau), j)] = list(S)
                            break

                    # Print the results if needed
                    if self.verbosity > 1:
                        self._print_a_pc_result(nonsig,
                                                conds_dim=p,
                                                max_combinations=
                                                max_combinations)
                else:
                    self._print_link_info(j=j, index_parent=ir,
                                          parent=(i, -abstau),
                                          num_parents=n_remaining,
                                          already_removed=True)

            # Increase condition cardinality
            p += 1

            # Re-compute adj and sort by minimum absolute test statistic value
            if mode == 'contemp_conds':
                adjt = self._get_adj_time_series_contemp(graph, sort_by=val_min)
            elif mode == 'standard':
                adjt = self._get_adj_time_series(graph, sort_by=val_min)

            if self.verbosity > 1:
                print("\nUpdated contemp. adjacencies:")
                self._print_parents(all_parents=adjt, val_min=val_min,
                                    pval_max=pval_max)

        if self.verbosity > 1:
            if not (self._any_tests_remaining(graph, adjt, tau_min, tau_max,
                                              p) and p <= max_conds_dim):
                print("\nAlgorithm converged at p = %d." % (p - 1))
            else:
                print(
                    "\nAlgorithm not yet converged, but max_conds_dim = %d"
                    " reached." % max_conds_dim)

        return {'graph': graph,
                'sepset': sepset,
                'p_matrix': pvalues,
                'val_matrix': val_matrix,
                }

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
                where = np.where(graph[:, j, :] != 0)
                adjt[j] = list(zip(*(where[0], -where[1])))
        else:
            for j in range(N):
                where = np.where(graph[:, j, :] == 1)
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

    def _get_sepset(self, tau_min, tau_max):
        """Returns initial sepset.

        Parameters
        ----------
        tau_min : int, optional (default: 0)
            Minimum time lag to test.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.

        Returns
        -------
        sepset : dict
            Initialized sepset.
        """
        sepset = dict([(((i, -tau), j), [])
                       for tau in range(tau_min, tau_max + 1)
                       for i in range(self.N)
                       for j in range(self.N)])

        return sepset

    def _find_unshielded_triples(self, graph):
        """Find unshielded triples i_tau o-(>) k_t o-o j_t with i_tau -/- j_t.

        Excludes conflicting links.

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.

        Returns
        -------
        triples : list
            List of triples.
        """

        N = graph.shape[0]
        adjt = self._get_adj_time_series(graph, include_conflicts=False)

        # Find unshielded triples
        # Find triples i_tau o-(>) k_t o-o j_t with i_tau -/- j_t
        triples = []
        for j in range(N):
            for (k, tauk) in adjt[j]:
                if tauk == 0:
                    for (i, taui) in adjt[k]:
                        if not (k == j or (
                                taui == 0 and (i == k or i == j))):
                            if ((taui == 0 and graph[i, j, 0] == 0 and
                                 graph[j, i, 0] == 0 and graph[j, k, 0] == 1)
                                    or (taui < 0 and graph[j, k, 0] == 1
                                        and graph[i, j, abs(taui)] == 0)):
                                triples.append(((i, taui), k, j))

        return triples

    def _pcalg_colliders(self,
                        graph,
                        sepset,
                        lagged_parents,
                        mode,
                        pc_alpha,
                        tau_max,
                        max_conds_py,
                        max_conds_px,
                        max_conds_px_lagged,
                        contemp_collider_rule,
                        conflict_resolution,
                        ):
        """Implements the collider orientation step of the PC algorithm for
        time series.

        Parameters
        ----------
        graph : array of shape (N, N, tau_max+1)
            Current graph.
        sepset : dictionary
            Separating sets. See paper for details.
        lagged_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            additional conditions for each CI test. As part of PCMCIplus
            these are the superset of lagged parents estimated with the PC1
            algorithm.
        mode : {'standard', 'contemp_conds'}
            For ``mode='contemp_conds'`` this implements Steps 2-4 of the
            PCMCIplus method. For ``mode='standard'`` this implements the
            standard PC algorithm adapted to time series.
        pc_alpha : float, optional (default: 0.01)
            Significance level.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        max_conds_py : int, optional (default: None)
            Maximum number of lagged conditions of Y to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px : int, optional (default: None)
            Maximum number of lagged conditions of X to use in MCI tests. If
            None is passed, this number is unrestricted.
        max_conds_px_lagged : int, optional (default: None)
            Maximum number of lagged conditions of X when X is lagged in MCI 
            tests. If None is passed, this number is equal to max_conds_px.
        contemp_collider_rule : {'majority', 'conservative', 'none'}
            Rule for collider phase to use. See the paper for details. Only
            'majority' and 'conservative' lead to an order-independent
            algorithm.
        conflict_resolution : bool, optional (default: True)
            Whether to mark conflicts in orientation rules. Only for True
            this leads to an order-independent algorithm.

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph, see description above for interpretation.
        sepset : dictionary
            Separating sets. See paper for details.
        ambiguous_triples : list
            List of ambiguous triples, only relevant for 'majority' and
            'conservative' rules, see paper for details.
        """

        if self.verbosity > 1:
            print("\n----------------------------")
            print("Collider orientation phase")
            print("----------------------------")
            print("\ncontemp_collider_rule = %s" % contemp_collider_rule)
            print("conflict_resolution = %s\n" % conflict_resolution)

        # Find unshielded triples
        triples = self._find_unshielded_triples(graph)

        v_structures = []
        ambiguous_triples = []

        if contemp_collider_rule is None or contemp_collider_rule == 'none':
            # Standard collider orientation rule of PC algorithm
            # If k_t not in sepset(i_tau, j_t), then orient
            # as i_tau --> k_t <-- j_t
            for itaukj in triples:
                (i, tau), k, j = itaukj
                if (k, 0) not in sepset[((i, tau), j)]:
                    v_structures.append(itaukj)
        else:
            # Apply 'majority' or 'conservative' rule to orient colliders          
            # Compute all (contemp) subsets of potential parents of i and all 
            # subsets of potential parents of j that make i and j independent
            def subsets(s):
                if len(s) == 0: return []
                subsets = []
                for cardinality in range(len(s) + 1):
                    subsets += list(itertools.combinations(s, cardinality))
                subsets = [list(sub) for sub in list(set(subsets))]
                return subsets

            # We only consider contemporaneous adjacencies because only these
            # can include the (contemp) k. Furthermore, we only need to check
            # adjacencies of i for tau=0
            if mode == 'contemp_conds':
                adjt = self._get_adj_time_series_contemp(graph)
            elif mode == 'standard':
                adjt = self._get_adj_time_series(graph)

            n_triples = len(triples)
            for ir, itaukj in enumerate(triples):
                (i, tau), k, j = itaukj

                if self.verbosity > 1:
                    self._print_triple_info(itaukj, ir, n_triples)

                neighbor_subsets_tmp = subsets(
                    [(l, taul) for (l, taul) in adjt[j]
                     if not (l == i and tau == taul)])
                if tau == 0:
                    # Furthermore, we only need to check contemp. adjacencies
                    # of i for tau=0
                    neighbor_subsets_tmp += subsets(
                        [(l, taul) for (l, taul) in adjt[i]
                         if not (l == j and taul == 0)])

                # Make unique
                neighbor_subsets = []
                for subset in neighbor_subsets_tmp:
                    if subset not in neighbor_subsets:
                        neighbor_subsets.append(subset)

                n_neighbors = len(neighbor_subsets)

                if self.verbosity > 1:
                    print(
                        "    Iterate through %d condition subset(s) of "
                        "neighbors: " % n_neighbors)
                    if lagged_parents is not None:
                        self._print_pcmciplus_conditions(lagged_parents, i, j,
                                         abs(tau), max_conds_py, max_conds_px,
                                         max_conds_px_lagged)

                # Test which neighbor subsets separate i and j
                neighbor_sepsets = []
                for iss, S in enumerate(neighbor_subsets):
                    val, pval, Z = self._run_pcalg_test(
                        i, abs(tau), j, S, lagged_parents, max_conds_py,
                        max_conds_px, max_conds_px_lagged, tau_max)

                    if self.verbosity > 1:
                        self._print_cond_info(Z=S, comb_index=iss, pval=pval,
                                              val=val)

                    if pval > pc_alpha:
                        neighbor_sepsets += [S]

                if len(neighbor_sepsets) > 0:
                    fraction = np.sum(
                        [(k, 0) in S for S in neighbor_sepsets]) / float(
                        len(neighbor_sepsets))

                if contemp_collider_rule == 'conservative':
                    # Triple is labeled as unambiguous if at least one
                    # separating set is found and either k is in ALL
                    # (fraction == 1) or NONE (fraction == 0) of them
                    if len(neighbor_sepsets) == 0:
                        if self.verbosity > 1:
                            print(
                                "    No separating subsets --> ambiguous "
                                "triple found")
                        ambiguous_triples.append(itaukj)
                    else:
                        if fraction == 0:
                            # If (k, 0) is in none of the neighbor_sepsets,
                            # orient as collider
                            v_structures.append(itaukj)
                            if self.verbosity > 1:
                                print(
                                    "    Fraction of separating subsets "
                                    "containing (%s 0) is = 0 --> collider "
                                    "found" % self.var_names[k])
                            # Also delete (k, 0) from sepset (if present)
                            if (k, 0) in sepset[((i, tau), j)]:
                                sepset[((i, tau), j)].remove((k, 0))
                            if tau == 0:
                                if (k, 0) in sepset[((j, tau), i)]:
                                    sepset[((j, tau), i)].remove((k, 0))
                        elif fraction == 1:
                            # If (k, 0) is in all of the neighbor_sepsets,
                            # leave unoriented
                            if self.verbosity > 1:
                                print(
                                    "    Fraction of separating subsets "
                                    "containing (%s 0) is = 1 --> "
                                    "non-collider found" % self.var_names[k])
                            # Also add (k, 0) to sepset (if not present)
                            if (k, 0) not in sepset[((i, tau), j)]:
                                sepset[((i, tau), j)].append((k, 0))
                            if tau == 0:
                                if (k, 0) not in sepset[((j, tau), i)]:
                                    sepset[((j, tau), i)].append((k, 0))
                        else:
                            if self.verbosity > 1:
                                print(
                                    "    Fraction of separating subsets "
                                    "containing (%s 0) is = between 0 and 1 "
                                    "--> ambiguous triple found" %
                                    self.var_names[k])
                            ambiguous_triples.append(itaukj)

                elif contemp_collider_rule == 'majority':

                    if len(neighbor_sepsets) == 0:
                        if self.verbosity > 1:
                            print(
                                "    No separating subsets --> ambiguous "
                                "triple found")
                        ambiguous_triples.append(itaukj)
                    else:
                        if fraction == 0.5:
                            if self.verbosity > 1:
                                print(
                                    "    Fraction of separating subsets "
                                    "containing (%s 0) is = 0.5 --> ambiguous "
                                    "triple found" % self.var_names[k])
                            ambiguous_triples.append(itaukj)
                        elif fraction < 0.5:
                            v_structures.append(itaukj)
                            if self.verbosity > 1:
                                print(
                                    "    Fraction of separating subsets "
                                    "containing (%s 0) is < 0.5 "
                                    "--> collider found" % self.var_names[k])
                            # Also delete (k, 0) from sepset (if present)
                            if (k, 0) in sepset[((i, tau), j)]:
                                sepset[((i, tau), j)].remove((k, 0))
                            if tau == 0:
                                if (k, 0) in sepset[((j, tau), i)]:
                                    sepset[((j, tau), i)].remove((k, 0))
                        elif fraction > 0.5:
                            if self.verbosity > 1:
                                print(
                                    "    Fraction of separating subsets "
                                    "containing (%s 0) is > 0.5 "
                                    "--> non-collider found" %
                                    self.var_names[k])
                            # Also add (k, 0) to sepset (if not present)
                            if (k, 0) not in sepset[((i, tau), j)]:
                                sepset[((i, tau), j)].append((k, 0))
                            if tau == 0:
                                if (k, 0) not in sepset[((j, tau), i)]:
                                    sepset[((j, tau), i)].append((k, 0))

        if self.verbosity > 1 and len(v_structures) > 0:
            print("\nOrienting links among colliders:")

        link_marker = {True:"o-o", False:"-->"}

        # Now go through list of v-structures and (optionally) detect conflicts
        oriented_links = []
        for itaukj in v_structures:
            (i, tau), k, j = itaukj

            if self.verbosity > 1:
                print("\n    Collider (%s % d) %s %s o-o %s:" % (
                    self.var_names[i], tau, link_marker[
                        tau==0], self.var_names[k],
                    self.var_names[j]))

            if (k, j) not in oriented_links and (j, k) not in oriented_links:
                if self.verbosity > 1:
                    print("      Orient %s o-o %s as %s --> %s " % (
                        self.var_names[j], self.var_names[k], self.var_names[j],
                        self.var_names[k]))
                graph[k, j, 0] = 0
                oriented_links.append((j, k))
            else:
                if conflict_resolution is False and self.verbosity > 1:
                    print("      Already oriented")

            if conflict_resolution:
                if (k, j) in oriented_links:
                    if self.verbosity > 1:
                        print(
                            "        Conflict since %s <-- %s already "
                            "oriented: Mark link as `2` in graph" % (
                                self.var_names[j], self.var_names[k]))
                    graph[j, k, 0] = graph[k, j, 0] = 2

            if tau == 0:
                if (i, k) not in oriented_links and (
                        k, i) not in oriented_links:
                    if self.verbosity > 1:
                        print("      Orient %s o-o %s as %s --> %s " % (
                            self.var_names[i], self.var_names[k],
                            self.var_names[i], self.var_names[k]))
                    graph[k, i, 0] = 0
                    oriented_links.append((i, k))
                else:
                    if conflict_resolution is False and self.verbosity > 1:
                        print("      Already oriented")

                if conflict_resolution:
                    if (k, i) in oriented_links:
                        if self.verbosity > 1:
                            print(
                                "        Conflict since %s <-- %s already "
                                "oriented: Mark link as `2` in graph" % (
                                    self.var_names[i], self.var_names[k]))
                        graph[i, k, 0] = graph[k, i, 0] = 2

        if self.verbosity > 1:
            adjt = self._get_adj_time_series(graph)
            print("\nUpdated adjacencies:")
            self._print_parents(all_parents=adjt, val_min=None, pval_max=None)

        return {'graph': graph,
                'sepset': sepset,
                'ambiguous_triples': ambiguous_triples,
                }

    def _find_triples_rule1(self, graph):
        """Find triples i_tau --> k_t o-o j_t with i_tau -/- j_t.

        Excludes conflicting links.

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.

        Returns
        -------
        triples : list
            List of triples.
        """
        adjt = self._get_adj_time_series(graph, include_conflicts=False)

        N = graph.shape[0]
        triples = []
        for j in range(N):
            for (k, tauk) in adjt[j]:
                if tauk == 0 and graph[j, k, 0] == 1:
                    for (i, taui) in adjt[k]:
                        if not (k == j or (taui == 0 and (i == k or i == j))):
                            if ((taui == 0 and graph[i, j, 0] == 0
                                 and graph[j, i, 0] == 0
                                 and graph[k, i, 0] == 0)
                                    or taui < 0 and graph[
                                        i, j, abs(taui)] == 0):
                                triples.append(((i, taui), k, j))
        return triples

    def _find_triples_rule2(self, graph):
        """Find triples i_t --> k_t --> j_t with i_t -- j_t.

        Excludes conflicting links.

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.

        Returns
        -------
        triples : list
            List of triples.
        """

        adjtcont = self._get_adj_time_series_contemp(graph,
                                                     include_conflicts=False)
        N = graph.shape[0]

        triples = []
        for j in range(N):
            for (k, tauk) in adjtcont[j]:
                if (j, 0) not in adjtcont[k]:
                    for (i, taui) in adjtcont[k]:
                        if (k, 0) not in adjtcont[i]:
                            if graph[i, j, 0] == 1 and graph[j, i, 0] == 1:
                                triples.append(((i, 0), k, j))
        return triples

    def _find_chains_rule3(self, graph):
        """Find chains i_t o-o k_t --> j_t and i_t o-o l_t --> j_t with
           i_t o-o j_t and k_t -/- l_t.

        Excludes conflicting links.

        Parameters
        ----------
        graph : array of shape [N, N, tau_max+1]
            Causal graph, see description above for interpretation.

        Returns
        -------
        chains : list
            List of chains.
        """
        N = graph.shape[0]
        adjtcont = self._get_adj_time_series_contemp(graph,
                                                     include_conflicts=False)

        chains = []
        for j in range(N):
            for (i, _) in adjtcont[j]:
                if graph[j, i, 0] == 1:
                    for (k, _) in adjtcont[j]:
                        for (l, _) in adjtcont[j]:
                            # Nodes should not be identical
                            if not ((k == l) or (k == i) or (l == i)):
                                # There should be an arrowhead from k and l to j
                                if (j, 0) not in adjtcont[k] and (j, 0) not \
                                        in adjtcont[l]:
                                    # Check that i is adjacent to k and l
                                    if (k, 0) in adjtcont[i]\
                                            and (l, 0) in adjtcont[i]:
                                        # Check that not both have arrow
                                        # towards i
                                        if (i, 0) in adjtcont[k] \
                                                or (i, 0) in adjtcont[l]:
                                            # k and l should not be adjacent
                                            if (graph[k, l, 0] == 0
                                                    and graph[l, k, 0] == 0):
                                                chains.append((((i, 0), k, j),
                                                               ((i, 0), l, j)))

        return chains

    def _pcalg_rules_timeseries(self,
                                graph,
                                ambiguous_triples,
                                conflict_resolution,
                                ):
        """Implements the rule orientation step of the PC algorithm for
        time series.

        Parameters
        ----------
        graph : array of shape (N, N, tau_max+1)
            Current graph.
        ambiguous_triples : list
            List of ambiguous triples, only relevant for 'majority' and
            'conservative' rules, see paper for details.
        conflict_resolution : bool
            Whether to mark conflicts in orientation rules. Only for True
            this leads to an order-independent algorithm.

        Returns
        -------
        graph : array of shape [N, N, tau_max+1]
            Resulting causal graph, see description above for interpretation.
        """
        N = graph.shape[0]

        def rule1(graph, oriented_links):
            """Find (unambiguous) triples i_tau --> k_t o-o j_t with
               i_tau -/- j_t and orient as i_tau --> k_t --> j_t.
            """
            triples = self._find_triples_rule1(graph)
            triples_left = False

            for itaukj in triples:
                if itaukj not in ambiguous_triples:
                    triples_left = True
                    # Orient as i_tau --> k_t --> j_t
                    (i, tau), k, j = itaukj
                    if (j, k) not in oriented_links and (
                            k, j) not in oriented_links:
                        if self.verbosity > 1:
                            print(
                                "    R1: Found (%s % d) --> %s o-o %s, "
                                "orient as %s --> %s" % (
                                    self.var_names[i], tau, self.var_names[k],
                                    self.var_names[j],
                                    self.var_names[k], self.var_names[j]))
                        graph[j, k, 0] = 0
                        oriented_links.append((k, j))

                    if conflict_resolution:
                        if (j, k) in oriented_links:
                            if self.verbosity > 1:
                                print(
                                    "        Conflict since %s <-- %s already"
                                    " oriented: Mark link as `2` in graph" % (
                                        self.var_names[k], self.var_names[j]))
                            graph[j, k, 0] = graph[k, j, 0] = 2

            return triples_left, graph, oriented_links

        def rule2(graph, oriented_links):
            """Find (unambiguous) triples i_t --> k_t --> j_t with i_t o-o j_t
               and orient as i_t --> j_t.
            """

            triples = self._find_triples_rule2(graph)
            triples_left = False

            for itaukj in triples:
                if itaukj not in ambiguous_triples:
                    # TODO: CHeck whether this is actually needed
                    # since ambiguous triples are always unshielded and here
                    # we look for triples where i and j are connected
                    triples_left = True
                    # Orient as i_t --> j_t
                    (i, tau), k, j = itaukj
                    if (j, i) not in oriented_links and (
                            i, j) not in oriented_links:
                        if self.verbosity > 1:
                            print(
                                "    R2: Found %s --> %s --> %s  with  %s "
                                "o-o %s, orient as %s --> %s" % (
                                    self.var_names[i], self.var_names[k],
                                    self.var_names[j],
                                    self.var_names[i], self.var_names[j],
                                    self.var_names[i], self.var_names[j]))
                        graph[j, i, 0] = 0
                        oriented_links.append((i, j))
                    if conflict_resolution:
                        if (j, i) in oriented_links:
                            if self.verbosity > 1:
                                print(
                                    "        Conflict since %s <-- %s already "
                                    "oriented: Mark link as `2` in graph" % (
                                        self.var_names[i], self.var_names[j]))
                            graph[j, i, 0] = graph[i, j, 0] = 2

            return triples_left, graph, oriented_links

        def rule3(graph, oriented_links):
            """Find (unambiguous) chains i_t o-o k_t --> j_t
               and i_t o-o l_t --> j_t with i_t o-o j_t
               and k_t -/- l_t: Orient as i_t --> j_t.
            """
            # First find all chains i_t -- k_t --> j_t with i_t -- j_t
            # and k_t -/- l_t
            chains = self._find_chains_rule3(graph)

            chains_left = False

            for (itaukj, itaulj) in chains:
                if (itaukj not in ambiguous_triples and
                        itaulj not in ambiguous_triples):
                    # TODO: CHeck whether this is actually needed
                    # since ambiguous triples are always unshielded and here
                    # we look for triples where i and j are connected
                    chains_left = True
                    # Orient as i_t --> j_t
                    (i, tau), k, j = itaukj
                    _       , l, _ = itaulj

                    if (j, i) not in oriented_links and (
                            i, j) not in oriented_links:
                        if self.verbosity > 1:
                            print(
                                "    R3: Found %s o-o %s --> %s and %s o-o "
                                "%s --> %s with %s o-o %s and %s -/- %s, "
                                "orient as %s --> %s" % (
                                    self.var_names[i], self.var_names[k],
                                    self.var_names[j], self.var_names[i],
                                    self.var_names[l], self.var_names[j],
                                    self.var_names[i], self.var_names[j],
                                    self.var_names[k], self.var_names[l],
                                    self.var_names[i], self.var_names[j]))
                        graph[j, i, 0] = 0
                        oriented_links.append((i, j))
                    if conflict_resolution:
                        if (j, i) in oriented_links:
                            if self.verbosity > 1:
                                print(
                                    "        Conflict since %s <-- %s already "
                                    "oriented: Mark link as `2` in graph" % (
                                        self.var_names[i], self.var_names[j]))
                            graph[j, i, 0] = graph[i, j, 0] = 2

            return chains_left, graph, oriented_links

        if self.verbosity > 1:
            print("\n")
            print("----------------------------")
            print("Rule orientation phase")
            print("----------------------------")

        oriented_links = []
        graph_new = np.copy(graph)
        any1 = any2 = any3 = True
        while (any1 or any2 or any3):
            if self.verbosity > 1:
                print("\nTry rule(s) %s" % (
                    np.where(np.array([0, any1, any2, any3]))))
            any1, graph_new, oriented_links = rule1(graph_new, oriented_links)
            any2, graph_new, oriented_links = rule2(graph_new, oriented_links)
            any3, graph_new, oriented_links = rule3(graph_new, oriented_links)

        if self.verbosity > 1:
            adjt = self._get_adj_time_series(graph_new)
            print("\nUpdated adjacencies:")
            self._print_parents(all_parents=adjt, val_min=None, pval_max=None)

        return graph_new

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
            adj_j = np.where(circle_cpdag[:,j,0] == "o-o")[0].tolist()

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

    def _optimize_pcmciplus_alpha(self,
                      selected_links,
                      tau_min,
                      tau_max,
                      pc_alpha,
                      contemp_collider_rule,
                      conflict_resolution,
                      reset_lagged_links,
                      max_conds_dim,
                      max_conds_py,
                      max_conds_px,
                      max_conds_px_lagged,
                      fdr_method,
                      ):
        """Optimizes pc_alpha in PCMCIplus.

        If a list or None is passed for ``pc_alpha``, the significance level is
        optimized for every graph across the given ``pc_alpha`` values using the
        score computed in ``cond_ind_test.get_model_selection_criterion()``

        Parameters
        ----------
        See those for run_pcmciplus()

        Returns
        -------
        Results for run_pcmciplus() for the optimal pc_alpha.
        """

        if pc_alpha is None:
            pc_alpha_list = [0.001, 0.005, 0.01, 0.025, 0.05]
        else:
            pc_alpha_list = pc_alpha

        if self.verbosity > 0:
            print("\n##\n## Optimizing pc_alpha over " + 
                  "pc_alpha_list = %s" % str(pc_alpha_list) +
                  "\n##")

        results = {}
        score = np.zeros_like(pc_alpha_list)
        for iscore, pc_alpha_here in enumerate(pc_alpha_list):
            # Print statement about the pc_alpha being tested
            if self.verbosity > 0:
                print("\n## pc_alpha = %s (%d/%d):" % (pc_alpha_here,
                                                      iscore + 1,
                                                      score.shape[0]))
            # Get the results for this alpha value
            results[pc_alpha_here] = \
                self.run_pcmciplus(selected_links=selected_links,
                                    tau_min=tau_min,
                                    tau_max=tau_max,
                                    pc_alpha=pc_alpha_here,
                                    contemp_collider_rule=contemp_collider_rule,
                                    conflict_resolution=conflict_resolution,
                                    reset_lagged_links=reset_lagged_links,
                                    max_conds_dim=max_conds_dim,
                                    max_conds_py=max_conds_py,
                                    max_conds_px=max_conds_px,
                                    max_conds_px_lagged=max_conds_px_lagged,
                                    fdr_method=fdr_method)

            # Get one member of the Markov equivalence class of the result
            # of PCMCIplus, which is a CPDAG

            # First create order that is based on some feature of the variables
            # to avoid order-dependence of DAG, i.e., it should not matter
            # in which order the variables appear in dataframe
            # Here we use the sum of absolute val_matrix values incident at j
            val_matrix = results[pc_alpha_here]['val_matrix']
            variable_order = np.argsort(
                                np.abs(val_matrix).sum(axis=(0,2)))[::-1]

            dag = self._get_dag_from_cpdag(
                            cpdag_graph=results[pc_alpha_here]['graph'],
                            variable_order=variable_order)
            

            # Compute the best average score when the model selection
            # is applied to all N variables
            for j in range(self.N):
                parents = []
                for i, tau in zip(*np.where(dag[:,j,:] == "-->")):
                    parents.append((i, -tau))
                score[iscore] += \
                    self.cond_ind_test.get_model_selection_criterion(
                        j, parents, tau_max)
            score[iscore] /= float(self.N)

        # Record the optimal alpha value
        optimal_alpha = pc_alpha_list[score.argmin()]

        if self.verbosity > 0:
            print("\n##"+
                  "\n\n## Scores for individual pc_alpha values:\n")
            for iscore, pc_alpha in enumerate(pc_alpha_list):
                print("   pc_alpha = %7s yields score = %.5f" % (pc_alpha, 
                                                                score[iscore]))
            print("\n##\n## Results for optimal " +
                  "pc_alpha = %s\n##" % optimal_alpha)
            self.print_results(results[optimal_alpha], alpha_level=optimal_alpha)

        optimal_results = results[optimal_alpha]
        optimal_results['optimal_alpha'] = optimal_alpha
        return optimal_results

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

    def symmetrize_p_and_val_matrix(self, p_matrix, val_matrix, selected_links, conf_matrix=None):
        """Symmetrizes the p_matrix, val_matrix, and conf_matrix based on selected_links
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
        selected_links : dict or None
            Dictionary of form {0: [(3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested.

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
                # If both the links are present in selected_links, symmetrize using maximum p-value
                if ((i, 0) in selected_links[j] and (j, 0) in selected_links[i]):
                    if (p_matrix[i, j, 0]
                            >= p_matrix[j, i, 0]):
                        p_matrix[j, i, 0] = p_matrix[i, j, 0]
                        val_matrix[j, i, 0] = val_matrix[i, j, 0]
                        if conf_matrix is not None:
                            conf_matrix[j, i, 0] = conf_matrix[i, j, 0]

                # If only one of the links is present in selected_links, symmetrize using the p-value of the link present
                elif ((i, 0) in selected_links[j] and (j, 0) not in selected_links[i]):
                    p_matrix[j, i, 0] = p_matrix[i, j, 0]
                    val_matrix[j, i, 0] = val_matrix[i, j, 0]
                    if conf_matrix is not None:
                        conf_matrix[j, i, 0] = conf_matrix[i, j, 0]
                else:
                    # Links not present in selected_links
                    pass

        # Return the values as a dictionary and store in class
        results = {'val_matrix': val_matrix,
                   'p_matrix': p_matrix,
                   'conf_matrix': conf_matrix}
        return results

    def run_sliding_window_of(self, method, method_args, 
                        window_step,
                        window_length,
                        conf_lev = 0.95,
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
                          # 'run_pcalg_non_timeseries_data',
                          'run_pcmciplus',]

        if method not in valid_methods:
            raise ValueError("method must be one of %s" % str(valid_methods))

        T = self.T

        if self.cond_ind_test.recycle_residuals:
            # recycle_residuals clashes with sliding windows...
            raise ValueError("cond_ind_test.recycle_residuals must be False.")

        if self.verbosity > 0:
            print("\n##\n## Running sliding window analysis of %s " % method +
                  "\n##\n" +
                  "\nwindow_step = %s \n" % window_step +
                  "\nwindow_length = %s \n" % window_length
                  )

        if self.dataframe.missing_flag is None:
            self.dataframe.missing_flag = True

        original_data = deepcopy(self.dataframe.values)

        window_start_points = np.arange(0, T - window_length, window_step)
        n_windows = len(window_start_points)

        window_results = {}
        for iw, w in enumerate(window_start_points):
            # Set values before and after window to np.nan
            data_window = deepcopy(original_data)
            data_window[0:w] = np.nan
            data_window[w + window_length:] = np.nan

            self.dataframe.values = data_window
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

        # Reset to original data for further analyses
        self.dataframe.values = original_data

        # Generate summary results
        summary_results = {}

        if 'graph' in window_results:
            most_frequent_links, counts = scipy.stats.mode(
                        window_results['graph'], axis=0)
            summary_results['most_frequent_links'] =\
                    most_frequent_links[0]  #.squeeze()
            summary_results['link_frequency'] =\
                    counts[0]/float(n_windows)   

        # Confidence intervals for val_matrix; interval is two-sided
        c_int = (1. - (1. - conf_lev)/2.)
        summary_results['val_matrix_mean'] = np.mean(
                                    window_results['val_matrix'], axis=0)

        summary_results['val_matrix_interval'] = np.stack(np.percentile(
                                    window_results['val_matrix'], axis=0,
                                    q = [100*(1. - c_int), 100*c_int]), axis=3)

        return {'summary_results': summary_results, 
                'window_results': window_results}


if __name__ == '__main__':
    from tigramite.independence_tests import ParCorr, CMIknn
    import tigramite.data_processing as pp
    from tigramite.toymodels import structural_causal_processes as toys
    import tigramite.plotting as tp

    np.random.seed(43)

    # Example process to play around with
    # Each key refers to a variable and the incoming links are supplied
    # as a list of format [((var, -lag), coeff, function), ...]
    def lin_f(x): return x
    def nonlin_f(x): return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))

    links = {0: [((0, -1), 0.9, lin_f)],
             1: [((1, -1), 0.8, lin_f), ((0, -1), 0.5, lin_f)],
             # 2: [((2, -1), 0.7, lin_f), ((1, 0), 0.1, lin_f)],
             # 3: [((3, -1), 0.7, lin_f), ((2, 0), -0.1, lin_f)],
             }

    data, nonstat = toys.structural_causal_process(links,
                        T=10000, seed=7)

    # Data must be array of shape (time, variables)
    print(data.shape)
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)
    # results = pcmci.run_pcmciplus(tau_min=0, tau_max=2, pc_alpha=0.01)
    # pcmci.print_results(results, alpha_level=0.01)


    print(pcmci.run_sliding_window_of(method='run_pcmciplus', 
                    method_args={'tau_min':0, 'tau_max':3, 'pc_alpha':0.01}, 
                        window_step=500,
                        window_length=1000,
                        conf_lev = 0.95)['summary_results'])

