import numpy as np
from itertools import product, combinations
from copy import deepcopy

from .pcmci_base import PCMCIbase

class LPCMCI(PCMCIbase):
    r""" LPCMCI is an algorithm for causal discovery in large-scale times series that allows for latent confounders and
    learns lag-specific causal relationships. The algorithm is introduced and explained in:

    [1] Gerhardus, A. & Runge, J. High-recall causal discovery for autocorrelated time series with latent confounders.
    Advances in Neural Information Processing Systems, 2020, 33.
    https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html
    
    NOTE: This method is still EXPERIMENTAL since the default settings of hyperparameters are still being fine-tuned.
    We actually invite feedback on which work best in applications and numerical experiments.
    The main function, which applies the algorithm, is 'run_lpcmci'.

    Parameters passed to the constructor:

    - dataframe: Tigramite dataframe object that contains the the time series dataset \bold{X}
    
    - cond_ind_test: A conditional independence test object that specifies which conditional independence test CI is to be used
    
    - verbosity: Controls the verbose output self.run_lpcmci() and the function it calls.

    Parameters passed to self.run_lpcmci(): 
    Note: The default values are still being tuned and some parameters might be removed in the future.
    
    - link_assumptions: dict or None
        Two-level nested dictionary such that link_assumptions[j][(i, lag_i)], where 0 <= j, i <= N-1 (with N the number of component
        time series) and -tau_max <= lag_i <= -tau_min, is a string which specifies background knowledge about the link from X^i_{t+lag_i} to
        X^j_t. These are the possibilities for this string and the corresponding claim:
            
            '-?>'   : X^i_{t+lag_i} is an ancestor of X^j_t.
            '-->'   : X^i_{t+lag_i} is an ancestor of X^j_t, and there is a link between X^i_{t+lag_i} and X^j_t
            '<?-'   : Only allowed for lag_i = 0. X^j_t is an ancestor of X^i_t.
            '<--'   : Only allowed for lag_i = 0. X^j_t is an ancestor of X^i_t, and there is a link between X^i_t and X^j_t
            '<?>'   : Neither X^i_{t+lag_i} is an ancestor of X^j_t nor the other way around
            '<->'   : Neither X^i_{t+lag_i} is an ancestor of X^j_t nor the other way around, and there is a link between X^i_{t+lag_i} and X^j_t
            'o?>'   : X^j_t is not an ancestor of X^i_{t+lag_i} (for lag_i < 0 this background knowledge is (for the default settings of self.run_lpcmci()) imposed automatically)
            'o->'   : X^j_t is not an ancestor of X^i_{t+lag_i}, and there is a link between X^i_{t+lag_i} and X^j_t
            '<?o'   : Only allowed for lag_i = 0. X^i_t is not an ancestor of X^j_t
            '<-o'   : Only allowed for lag_i = 0. X^i_t is not an ancestor of X^j_t, and there is a link between X^i_t and X^j_t
            'o-o'   : Only allowed for lag_i = 0. There is a link between X^i_t and X^j_t
            'o?o'   : Only allowed for lag_i = 0. No claim is made
            ''      : There is no link between X^i_{t+lag_i} and X^j_t.

        Another way to specify the absent link is if the form of the link between (i, lag_i) and (j, 0) is not specified by the dictionary, that is, if either
        link_assumptions[j] does not exist or link_assumptions[j] does exist but link_assumptions[j][(i, lag_i)] does
        not exist, then the link between (i, lag_i) and (j, 0) is assumed to be absent.
    
    - tau_min: The assumed minimum time lag, i.e., links with a lag smaller
      than tau_min are assumed to be absent.
    
    - tau_max: The maximum considered time lag, i.e., the algorithm learns a
      DPAG on a time window [t-\taumax, t] with \tau_max + 1 time steps. It
      is *not* assumed that in the underlying time series DAG there are no
      links with a lag larger than \tau_max.
    
    - pc_alpha: The significance level of conditional independence tests
    
    - n_preliminary_iterations: Determines the number of iterations in the
      preliminary phase of LPCMCI, corresponding to the 'k' in LPCMCI(k) in
      [1].
    
    - max_cond_px: Consider a pair of variables (X^i_{t-\tau}, X^j_t)
      with \tau > 0. In Algorithm S2 in [1] (here this is
      self._run_ancestral_removal_phase()), the algorithm does not test for
      conditional independence given subsets of apds_t(X^i_{t-\tau}, X^j_t, C
      (G)) of cardinality higher than max_cond_px. In Algorithm S3 in [1]
      (here this is self._run_non_ancestral_removal_phase()), the algorithm
      does not test for conditional independence given subsets of napds_t
      (X^i_{t-\tau}, X^j_t, C(G)) of cardinality higher than max_cond_px.
    
    - max_p_global: Restricts all conditional independence tests to
      conditioning sets with cardinality smaller or equal to max_p_global
    
    - max_p_non_ancestral: Restricts all conditional independence tests in the
      second removal phase (here this is self._run_dsep_removal_phase()) to
      conditioning sets with cardinality smaller or equal to max_p_global
    
    - max_q_global: For each ordered pair (X^i_{t-\tau}, X^j_t) of adjacent
      variables and for each cardinality of the conditioning sets test at
      most max_q_global many conditioning sets (when summing over all tested
      cardinalities more than max_q_global tests may be made)
    
    - max_pds_set: In Algorithm S3 (here this is
      self._run_non_ancestral_removal_phase()), the algorithm tests for
      conditional independence given subsets of the relevant napds_t sets. If
      for a given link the set napds_t(X^j_t, X^i_{t-\tau}, C(G)) has more
      than max_pds_set many elements (or, if the link is also tested in the
      opposite directed, if napds_t(X^i_{t-\tau}, X^j_t, C(G)) has more than
      max_pds_set elements), this link is not tested.
    
    - prelim_with_collider_rules: If True: As in pseudocode If False: Line 22
      of Algorithm S2 in [1] is replaced by line 18 of Algorithm S2 when
      Algorithm S2 is called from the preliminary phase (not in the last
      application of Algorithm S2 directly before Algorithm S3 is applied)
    
    - parents_of_lagged: If True: As in pseudocode If False: The default
      conditioning set is pa(X^j_t, C(G)) rather than pa({X^j_t, X^i_
      {t-\tau}, C(G)) for tau > 0
    
    - prelim_only: If True, stop after the preliminary phase. Can be used for
      detailed performance analysis
    
    - break_once_separated: If True: As in pseudocode If False: The break
      commands are removed from Algorithms S2 and S3 in in [1]
    
    - no_non_ancestral_phase: If True, do not execute Algorithm S3. Can be
      used for detailed performance analysis
    
    - use_a_pds_t_for_majority: If True: As in pseudocode If False: The search
      for separating sets instructed by the majority rule is made given
      subsets adj(X^j_t, C(G)) rather than subsets of apds_t(X^j_t, X^i_
      {t-\tau}, C(G))
    
    - orient_contemp:
        If orient_contemp == 1: As in pseudocode of Algorithm S2 in [1]
        If orient_contemp == 2: Also orient contemporaneous links in line 18 of Algorithm S2
        If orient_comtemp == 0: Also not orient contemporaneous links in line 22 of Algorithm S2
    
    - update_middle_marks:
        If True: As in pseudoce of Algorithms S2 and S3 in [1]
        If False: The MMR rule is not applied
    
    - prelim_rules:
        If prelim_rules == 1: As in pseudocode of Algorithm S2 in [1]
        If prelim_rules == 0: Exclude rules R9^prime and R10^\prime from line 18 in Algorithm S2
    
    - fix_all_edges_before_final_orientation: When one of max_p_global,
      max_p_non_ancestral, max_q_global or max_pds_set is not np.inf, the
      algorithm may terminate although not all middle marks are empty. All
      orientation rules are nevertheless sound, since the rules always check
      for the appropriate middle marks. If
      fix_all_edges_before_final_orientation is True, all middle marks are
      set to the empty middle mark by force, followed by another application
      of the rules.
    
    - auto_first: If True: As in pseudcode of Algorithms S2 and S3 in [1] If
      False: Autodependency links are not prioritized even before
      contemporaneous links
    
    - remember_only_parents:
        If True: As in pseudocode of Algorithm 1
        If False: If X^i_{t-\tau} has been marked as ancestor of X^j_t at any point of a preliminary iteration but the link between
        X^i_{t-\tau} and X^j_t was removed later, the link is nevertheless initialized with a tail at X^i_{t-\tau} in the re-initialization
    
    - no_apr:
        If no_apr == 0: As in pseudcode of Algorithms S2 and S3 in [1]
        If no_apr == 1: The APR is not applied by Algorithm S2, except in line 22 of its last call directly before the call of Algorithm S3
        If no_apr == 2: The APR is never applied

    Return value of self.run_lpcmci():
        graph : array of shape (N, N, tau_max+1)
            Resulting DPAG, representing the learned causal relationships.
        val_matrix : array of shape (N, N, tau_max+1)
            Estimated matrix of test statistic values regarding adjacencies.
        p_matrix : array of shape [N, N, tau_max+1]
            Estimated matrix of p-values regarding adjacencies.

    A note on middle marks: For convenience (to have strings of the same
    lengths) we here internally denote the empty middle mark by '-'.  For
    post-processing purposes all middle marks are set to the empty middle
    mark (here '-').
    
    A note on wildcards: The middle mark wildcard \ast and the edge mark
    wildcard are here represented as '*'', the edge mark wildcard \star
    as '+'.
    """

    def __init__(self, dataframe, cond_ind_test, verbosity = 0):
        """Class constructor. Store:
                i)      data
                ii)     conditional independence test object
                iii)    some instance attributes"""

        # Init base class
        PCMCIbase.__init__(self, dataframe=dataframe, 
                        cond_ind_test=cond_ind_test,
                        verbosity=verbosity)

    def run_lpcmci(self,
                    link_assumptions = None,
                    tau_min = 0,
                    tau_max = 1, 
                    pc_alpha = 0.05,
                    n_preliminary_iterations = 1,
                    max_cond_px = 0,
                    max_p_global = np.inf,
                    max_p_non_ancestral = np.inf,
                    max_q_global = np.inf,
                    max_pds_set = np.inf,
                    prelim_with_collider_rules = True,
                    parents_of_lagged = True,
                    prelim_only = False,
                    break_once_separated = True,
                    no_non_ancestral_phase = False,     
                    use_a_pds_t_for_majority = True,
                    orient_contemp = 1,
                    update_middle_marks = True,
                    prelim_rules = 1,
                    fix_all_edges_before_final_orientation = True,
                    auto_first = True,
                    remember_only_parents = True,
                    no_apr = 0):
        """Run LPCMCI on the dataset and with the conditional independence test passed to the class constructor and with the
        options passed to this function."""

        #######################################################################################################################
        #######################################################################################################################
        # Step 0: Initializations
        self._initialize(link_assumptions, tau_min, tau_max, pc_alpha, n_preliminary_iterations, max_cond_px, max_p_global,
            max_p_non_ancestral, max_q_global, max_pds_set, prelim_with_collider_rules, parents_of_lagged, prelim_only,
            break_once_separated, no_non_ancestral_phase, use_a_pds_t_for_majority, orient_contemp, update_middle_marks,
            prelim_rules, fix_all_edges_before_final_orientation, auto_first, remember_only_parents, no_apr)

        #######################################################################################################################
        #######################################################################################################################
        # Step 1: Preliminary phases
        for i in range(self.n_preliminary_iterations):

            # Verbose output
            if self.verbosity >= 1:
                print("\n=======================================================")
                print("=======================================================")
                print("Starting preliminary phase {:2}".format(i + 1))

            # In the preliminary phases, auto-lag links are tested with first priority. Among the auto-lag links, different lags are
            # not distinguished. All other links have lower priority, among which those which shorter lags have higher priority
            self._run_ancestral_removal_phase(prelim = True)

            # Verbose output
            if self.verbosity >= 1:
                print("\nPreliminary phase {:2} complete".format(i + 1))
                print("\nGraph:\n--------------------------------")
                self._print_graph_dict()
                print("--------------------------------")

            # When the option self.prelim_only is chosen, do not re-initialize in the last iteration
            if i == self.n_preliminary_iterations - 1 and self.prelim_only:
                break

            # Remember ancestorships, re-initialize and re-apply the remembered ancestorships
            def_ancs = self.def_ancs

            if self.remember_only_parents:
                smaller_def_ancs = dict()
                for j in range(self.N):
                    smaller_def_ancs[j] = {(i, lag_i) for (i, lag_i) in def_ancs[j] if self._get_link((i, lag_i), (j, 0)) != ""}
                def_ancs = smaller_def_ancs

            self._initialize_run_memory()
            self._apply_new_ancestral_information(None, def_ancs)

        #######################################################################################################################
        #######################################################################################################################
        # Step 2: Full ancestral phase
        if not self.prelim_only:

            # Verbose output
            if self.verbosity >= 1:
                print("\n=======================================================")
                print("=======================================================")
                print("Starting final ancestral phase")

            # In the standard ancestral phase, links are prioritized in the same as in the preliminary phases
            self._run_ancestral_removal_phase()

            # Verbose output
            if self.verbosity >= 1:
                print("\nFinal ancestral phase complete")
                print("\nGraph:\n--------------------------------")
                self._print_graph_dict()
                print("--------------------------------")

        #######################################################################################################################
        #######################################################################################################################
        # Step 3: Non-ancestral phase
        if (not self.prelim_only) and (not self.no_non_ancestral_phase):

            # Verbose output
            if self.verbosity >= 1:
                print("\n=======================================================")
                print("=======================================================")
                print("Starting non-ancestral phase")

            # In the non-ancestral phase, large lags are prioritized
            self._run_non_ancestral_removal_phase()

            # Verbose output
            if self.verbosity >= 1:
                print("\nNon-ancestral phase complete")
                print("\nGraph:\n--------------------------------")
                self._print_graph_dict()
                print("--------------------------------")

        if self.fix_all_edges_before_final_orientation:

            # Verbose output
            if self.verbosity >= 1:
                print("\n=======================================================")
                print("=======================================================")
                print("Final rule application phase")
                print("\nSetting all middle marks to '-'")

            self._fix_all_edges()
            self._run_orientation_phase(rule_list = self._rules_all, only_lagged = False)

        #######################################################################################################################
        #######################################################################################################################

        # Verbose output
        if self.verbosity >= 1:
            print("\n=======================================================")
            print("=======================================================")
            print("\nLPCMCI has converged")
            print("\nFinal graph:\n--------------------------------")
            print("--------------------------------")
            self._print_graph_dict()
            print("--------------------------------")
            print("--------------------------------\n")

            print("Max search set: {}".format(self.max_na_search_set_found))
            print("Max na-pds set: {}\n".format(self.max_na_pds_set_found))

        # Post processing
        self._fix_all_edges()
        self.graph = self._dict2graph()
        self.pval_max_matrix = self._dict_to_matrix(self.pval_max, self.tau_max, self.N, default = 0)
        self.val_min_matrix = self._dict_to_matrix(self.pval_max_val, self.tau_max, self.N, default = 0)
        self.cardinality_matrix = self._dict_to_matrix(self.pval_max_card, self.tau_max, self.N, default = 0)

        # Build and return the return dictionariy
        return_dict = {"graph": self.graph,
                       "p_matrix": self.pval_max_matrix,
                       "val_matrix": self.val_min_matrix}
        return return_dict


    def _initialize(self, link_assumptions, tau_min, tau_max, pc_alpha, n_preliminary_iterations, max_cond_px, max_p_global,
        max_p_non_ancestral, max_q_global, max_pds_set, prelim_with_collider_rules, parents_of_lagged, prelim_only,
        break_once_separated, no_non_ancestral_phase, use_a_pds_t_for_majority, orient_contemp, update_middle_marks, prelim_rules,
        fix_all_edges_before_final_orientation, auto_first, remember_only_parents, no_apr):
        """Function for
            i)      saving the arguments passed to self.run_lpcmci() as instance attributes
            ii)     initializing various memory variables for storing the current graph, sepsets etc.
            """

        # Save the arguments passed to self.run_lpcmci()
        self.link_assumptions = link_assumptions
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.n_preliminary_iterations = n_preliminary_iterations
        self.max_cond_px = max_cond_px
        self.max_p_global = max_p_global
        self.max_p_non_ancestral = max_p_non_ancestral
        self.max_q_global = max_q_global
        self.max_pds_set = max_pds_set
        self.prelim_with_collider_rules = prelim_with_collider_rules
        self.parents_of_lagged = parents_of_lagged
        self.prelim_only = prelim_only
        self.break_once_separated = break_once_separated
        self.no_non_ancestral_phase = no_non_ancestral_phase
        self.use_a_pds_t_for_majority = use_a_pds_t_for_majority
        self.orient_contemp = orient_contemp
        self.update_middle_marks = update_middle_marks
        self.prelim_rules = prelim_rules
        self.fix_all_edges_before_final_orientation = fix_all_edges_before_final_orientation
        self.auto_first = auto_first
        self.remember_only_parents = remember_only_parents
        self.no_apr = no_apr

        if isinstance(pc_alpha, (list, tuple, np.ndarray)):
                raise ValueError("pc_alpha must be single float in LPCMCI.")
        if pc_alpha < 0. or pc_alpha > 1:
            raise ValueError("Choose 0 <= pc_alpha <= 1")
            
        # Check that validity of tau_min and tau_max
        self._check_tau_min_tau_max()

        # Check the validity of 'link_assumptions'
        if self.link_assumptions is not None:
            self._check_link_assumptions()

        # Rules to be executed at the end of a preliminary phase
        self._rules_prelim_final= [["APR"], ["ER-08"], ["ER-02"], ["ER-01"], ["ER-09"], ["ER-10"]]

        # Rules to be executed within the while loop of a preliminary phase
        self._rules_prelim = [["APR"], ["ER-08"], ["ER-02"], ["ER-01"]] if self.prelim_rules == 0 else self._rules_prelim_final

        # Full list of all rules
        self._rules_all = [["APR"], ["ER-08"], ["ER-02"], ["ER-01"], ["ER-00-d"], ["ER-00-c"], ["ER-03"], ["R-04"], ["ER-09"], ["ER-10"], ["ER-00-b"], ["ER-00-a"]]

        # Initialize various memory variables for storing the current graph, sepsets etc.
        self._initialize_run_memory()

        # Return
        return True

    def _check_tau_min_tau_max(self):
        """Check whether the choice of tau_min and tau_max is valid."""

        if not 0 <= self.tau_min <= self.tau_max:
            raise ValueError("tau_min = {}, ".format(self.tau_min) + \
                             "tau_max = {}, ".format(self.tau_max) + \
                             "but 0 <= tau_min <= tau_max required.")

    def _check_link_assumptions(self):
        """Check the validity of user-specified 'link_assumptions'.

        The checks assert:
        - Valid dictionary keys
        - Valid edge types
        - That no causal cycle is specified
        - That no almost causal cycle is specified

        The checks do not assert that maximality is not violated."""

        # Ancestorship matrices
        ancs_mat_contemp = np.zeros((self.N, self.N), dtype = "int32")
        ancs_mat = np.zeros((self.N*(self.tau_max + 1),
            self.N*(self.tau_max + 1)), dtype = "int32")

        # Run through the outer dictionary
        for j, links_j in self.link_assumptions.items():

            # Check validity of keys of outer dictionary
            if not 0 <= j <= self.N - 1:
                raise ValueError("The argument 'link_assumption' must be a "\
                    "dictionary whose keys are in {0, 1, ..., N-1}, where N "\
                    "is the number of component time series. Here, "\
                    f"N = {self.N}.")

            # Run through the inner dictionary
            for (i, lag_i), link_ij in links_j.items():

                # Check validity of keys of inner dictionary
                if i == j and lag_i == 0:
                    raise ValueError(f"The dictionary 'link_assumptions[{j}] "\
                        f"must not have the key ({j}, 0), because this refers "\
                        "to a self-link.")

                if (not (0 <= i <= self.N - 1)
                    or not (-self.tau_max <= lag_i <= -self.tau_min)):
                    raise ValueError("All values of 'link_assumptions' must "\
                        "be dictionaries whose keys are of the form (i, "\
                        "lag_i), where i in {0, 1, ..., N-1} with N the "\
                        "number of component time series and lag_i in "\
                        "{-tau_max, ..., -tau_min} with tau_max the maximum "\
                        "considered time lag and tau_min the minimum assumed "\
                        f"time lag. Here, N = {self.N} and tau_max = "\
                        f"{self.tau_max} and tau_min = {self.tau_min}.")

                # Check for validity of entries. At the same time mark the
                # ancestorships in ancs_mat_contemp and ancs_mat

                if link_ij == "":

                    # Check for symmetry of lag zero links
                    if lag_i == 0:

                        if (self.link_assumptions.get(i) is None
                            or self.link_assumptions[i].get((j, 0)) is None
                            or self.link_assumptions[i][(j, 0)] != ""):
                            raise ValueError("The lag zero links specified by "\
                                "'link_assumptions' must be symmetric: Because"\
                                f"'link_assumptions'[{j}][({i}, {0})] = '', "\
                                " there must also be "\
                                f"'link_assumptions'[{i}][({j}, {0})] = ''.")
                    continue

                if len(link_ij) != 3:
                    if lag_i < 0:
                        raise ValueError("Invalid link: "\
                            f"'link_assumptions'[{j}][({i}, {lag_i})] = "\
                            f"{link_ij}. Allowed are: '-?>', '-->', '<?>', "\
                            "'<->', 'o?>', 'o->'.")
                    else:
                        raise ValueError("Invalid link: "\
                            f"'link_assumptions'[{j}][({i}, {lag_i})] = "\
                            f"{link_ij}. Allowed are: '-?>', '-->', '<?>', "\
                            "'<->', 'o?>', 'o->', '<?-', '<--', '<?o', '<--', "\
                            "'o-o', 'o?o'.")

                if link_ij[0] == "-":

                    if link_ij[2] != ">":
                        raise ValueError("Invalid link: "\
                            f"'link_assumptions'[{j}][({i}, {lag_i})] = "\
                            f"{link_ij}. The first character '-', which says "\
                            f"that ({i}, {lag_i}) is an ancestor (cause) of "\
                            f"({j}, 0). Hence, ({j}, 0) is a non-ancestor "\
                            f"(non-cause) of ({i}, {lag_i}) and the third "\
                            "character must be '>'.")

                    # Mark the ancestorship
                    if lag_i == 0:
                        ancs_mat_contemp[i, j] = 1
                    for Delta_t in range(0, self.tau_max + 1 - abs(lag_i)):
                        ancs_mat[self.N*(abs(lag_i) + Delta_t) + i,
                            self.N*Delta_t + j] = 1

                elif link_ij[0] in ["<", "o"]:

                    if lag_i < 0:

                        if link_ij[2] != ">":
                            raise ValueError("Invalid link: "\
                                f"'link_assumptions'[{j}][({i}, {lag_i})] = "\
                                f"{link_ij}. Since {lag_i} < 0, ({j}, 0) "\
                                f"cannot be an ancestor (cause) of "\
                                f"({i}, {lag_i}). Hence, the third character "\
                                f"must be '>'.")

                    else:

                        if link_ij[2] not in ["-", ">", "o"]:
                            raise ValueError("Invalid link: "\
                                f"'link_assumptions'[{j}][({i}, {0})] = "\
                                f"{link_ij}. The third character must be one "\
                                "of the following: 1) '-', which says that "\
                                f"({j}, 0) is an ancestor (cause) of "\
                                f"({i}, {0}). 2) '>', which says that "\
                                f"({j}, 0) is a non-ancestor (non-cause) of "\
                                f"({i}, {0}). 3) 'o', which says that it is "\
                                f"unknown whether or not ({j}, {0}) is an "\
                                f"ancestor (cause) of ({i}, {0}).")

                        if link_ij[2] == "-":

                            if link_ij[0] != "<":
                                raise ValueError("Invalid link: "\
                                    f"'link_assumptions'[{j}][({i}, {0})] = "\
                                    f"{link_ij}. The third character is '-', "\
                                    f"which says that ({j}, {0}) is an "\
                                    f"ancestor (cause) of ({i}, 0). Hence, "\
                                    f"({i}, 0) is a non-ancestor (non-cause) "\
                                    f"of ({j}, {0}) and the first character "\
                                    "must be '<'.")

                            # Mark the ancestorship
                            ancs_mat_contemp[j, i] = 1
                            for Delta_t in range(0, self.tau_max + 1):
                                ancs_mat[self.N*Delta_t + j,
                                    self.N*Delta_t + i] = 1

                else:
                    raise ValueError(f"Invalid link: "\
                        f"'link_assumptions'[{j}][({i}, {lag_i})] = "\
                        f"{link_ij}. The first character must be one of the "\
                        f"following: 1) '-', which says that ({i}, {lag_i}) "\
                        f"is an ancestor (cause) of ({j}, 0). 2) '<', which "\
                        f"says that ({i}, {lag_i}) is a non-ancestor "\
                        f"(non-cause) of ({j}, 0). 3) 'o', which says that it"\
                        f"is unknown whether or not ({i}, {lag_i}) is an "\
                        f"ancestor (cause) of ({j}, {0}).")

                if link_ij[1] not in ["-", "?"]:
                    raise ValueError("Invalid link: "\
                        f"'link_assumptions'[{j}][({i}, {lag_i})] = "\
                        f"{link_ij}. The second character must be one of the "\
                        "following: 1) '-', which says that the link "\
                        f"({i}, {lag_i}) {link_ij} ({j}, 0) is definitely "\
                        "part of the graph. 2) '?', which says that link "\
                        "might be but does not need to be part of the graph.")

                # Check for symmetry of lag zero links
                if lag_i == 0:

                    if (self.link_assumptions.get(i) is None
                        or self.link_assumptions[i].get((j, 0)) is None
                        or self.link_assumptions[i][(j, 0)] != self._reverse_link(link_ij)):
                        raise ValueError(f"The lag zero links specified by "\
                            "'link_assumptions' must be symmetric: Because "\
                            f"'link_assumptions'[{j}][({i}, {0})] = "\
                            f"'{link_ij}' there must also be "\
                            f"'link_assumptions'[{i}][({j}, {0})] = "\
                            f"'{self._reverse_link(link_ij)}'.")

        # Check for contemporaneous cycles
        ancs_mat_contemp_to_N = np.linalg.matrix_power(ancs_mat_contemp, self.N)
        if np.sum(ancs_mat_contemp_to_N) != 0:
            raise ValueError("According to 'link_assumptions', there is a "\
                "contemporaneous causal cycle. Causal cycles are not allowed.")

        # Check for almost directed cycles
        ancs_mat_summed = np.linalg.inv(np.eye(ancs_mat.shape[0], dtype = "int32") - ancs_mat)
        for j, links_j in self.link_assumptions.items():
            for (i, lag_i), link_ij in links_j.items():
                if (link_ij != ""
                    and link_ij[0] == "<"
                    and ancs_mat_summed[self.N*abs(lag_i) + i, j] != 0):
                    raise ValueError(f"Inconsistency in 'link_assumptions': "\
                        f"Since 'link_assumptions'[{j}][({i}, {lag_i})] "\
                        f"= {link_ij}, variable ({i}, {lag_i}) is a "\
                        f"non-ancestor (non-cause) of ({j}, 0). At the same "\
                        "time, however, 'link_assumptions' specifies a "\
                        f"directed path (causal path) from ({i}, {lag_i}) to "\
                        f"({j}, 0).")

        # Replace absent entries by ''
        for j in range(self.N):
            if self.link_assumptions.get(j) is None:
                self.link_assumptions[j] = {(i, -tau_i): ""
                    for (i, tau_i) in product(range(self.N), range(self.tau_min, self.tau_max+1))
                    if (tau_i > 0 or i != j)}
            else:
                for (i, tau_i) in product(range(self.N), range(self.tau_min, self.tau_max+1)):
                    if (tau_i > 0 or i != j):
                        if self.link_assumptions[j].get((i, -tau_i)) is None:
                            self.link_assumptions[j][(i, -tau_i)] = ""

    def _initialize_run_memory(self):
        """Function for initializing various memory variables for storing the current graph, sepsets etc."""
        
        # Initialize the nested dictionary for storing the current graph.
        # Syntax: self.graph_dict[j][(i, -tau)] gives the string representing the link from X^i_{t-tau} to X^j_t
        self.graph_dict = {}
        for j in range(self.N):

            self.graph_dict[j] = {(i, 0): "o?o" for i in range(self.N) if j != i}

            if self.max_cond_px == 0 and self.update_middle_marks:
                self.graph_dict[j].update({(i, -tau): "oL>" for i in range(self.N) for tau in range(1, self.tau_max + 1)})
            else:
                self.graph_dict[j].update({(i, -tau): "o?>" for i in range(self.N) for tau in range(1, self.tau_max + 1)})

        # Initialize the nested dictionary for storing separating sets
        # Syntax: self.sepsets[j][(i, -tau)] stores separating sets of X^i_{t-tau} to X^j_t. For tau = 0, i < j.
        self.sepsets = {j: {(i, -tau): set() for i in range(self.N) for tau in range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}

        # Initialize dictionaries for storing known ancestorships, non-ancestorships, and ambiguous ancestorships
        # Syntax: self.def_ancs[j] contains the set of all known ancestors of X^j_t. Equivalently for the others
        self.def_ancs = {j: set() for j in range(self.N)}
        self.def_non_ancs = {j: set() for j in range(self.N)}
        self.ambiguous_ancestorships = {j: set() for j in range(self.N)}

        # Initialize nested dictionaries for saving the maximal p-value among all conditional independence tests of a given
        # pair of variables as well as the corresponding test statistic values and conditioning set cardinalities
        # Syntax: As for self.sepsets
        self.pval_max = {j: {(i, -tau): -np.inf for i in range(self.N) for tau in range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}
        self.pval_max_val = {j: {(i, -tau): np.inf for i in range(self.N) for tau in range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}
        self.pval_max_card = {j: {(i, -tau): -np.inf for i in range(self.N) for tau in range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}                                        
        # Initialize a nested dictionary for caching na-pds-sets
        # Syntax: self._na_pds_t[(i, t_i)][(j, t_j)] stores na_pds_t((i, t_i), (j, t_j))
        self._na_pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}

        # Initialize a variable for remembering the maximal cardinality among all calculated na-pds-sets, as well as the
        # maximial cardinality of any search set in the non-ancestral phase
        self.max_na_search_set_found = -1
        self.max_na_pds_set_found = -1

        # Apply the restriction imposed by tau_min
        self._apply_tau_min_restriction()

        # Apply the background knowledge given by background_knowledge
        if self.link_assumptions is not None:
            self._apply_link_assumptions()

        # Return
        return True

    def _apply_tau_min_restriction(self):
        """Apply the restrictions imposed by a non-zero tau_min:
        - Remove all links of lag smaller than tau_min from self.graph_dict
        - Set the corresponding entries in self.pval_max, self.pval_max_val, and self.pval_max_card to np.inf, -np.inf, np.inf
        """

        for (i, j, tau) in product(range(self.N), range(self.N), range(0, self.tau_min)):
            if tau > 0 or j != i:
                self.graph_dict[j][(i, -tau)] = ""

            if tau > 0 or i < j:
                self.pval_max[j][(i, -tau)] = np.inf
                self.pval_max_val[j][(i, -tau)] = -np.inf
                self.pval_max_card[j][(i, -tau)] = np.inf

    def _apply_link_assumptions(self):
        """Apply the background knowledge specified by 'link_assumptions':
        - Write the specified edge types to self.graph_dict
        - Set the corresponding entries in self.pval_max to np.inf, in self.pval_max_val to -np.inf, and in
        - to self.pval_max_card to np.inf
        """

        for j, links_j in self.link_assumptions.items():
            for (i, lag_i), link in self.link_assumptions[j].items():

                # Apply background knowledge
                if link != "" and link[1] == "?" and lag_i < 0 and self.max_cond_px == 0 and self.update_middle_marks:
                    self.graph_dict[j][(i, lag_i)] = link[0] + "L" + link[2]
                else:
                    self.graph_dict[j][(i, lag_i)] = link

                # If background knowledge amounts to absence of link, set the corresponding entries in
                # self.pval_max to 2, in self.pval_max_val to -np.inf, and in self.pval_max_card to None to np.inf
                if link == "" and (lag_i < 0 or i < j):
                    self.pval_max[j][(i, lag_i)] = np.inf
                    self.pval_max_val[j][(i, lag_i)] = -np.inf
                    self.pval_max_card[j][(i, lag_i)] = np.inf

    def _run_ancestral_removal_phase(self, prelim = False):
        """Run an ancestral edge removal phase, this is Algorithm S2"""

        # Iterate until convergence
        # p_pc is the cardinality of the non-default part of the conditioning sets. The full conditioning sets may have
        # higher cardinality due to default conditioning on known parents
        p_pc = 0
        while_broken = False
        while True:

            ##########################################################################################################
            ### Run the next removal iteration #######################################################################

            # Force-quit while loop when p_pc exceeds the limit put by self.max_p_global
            if p_pc > self.max_p_global:
                while_broken = True
                break

            # Verbose output
            if self.verbosity >= 1:
                if p_pc == 0:
                    print("\nStarting test phase\n")
                print("p = {}".format(p_pc))

            # Variables to memorize the occurence and absence of certain events in the below edge removal phase
            has_converged = True
            any_removal = False

            # Generate the prioritized link list
            if self.auto_first:

                link_list = [product(range(self.N), range(-self.tau_max, 0))]
                link_list = link_list + [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]

            else:

                link_list = [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]


            # Run through all elements of link_list. Each element of link_list specifies ordered pairs of variables whose
            # connecting edges are then subjected to conditional independence tests
            for links in link_list:

                # Memory variables for storing edges that are marked for removal
                to_remove = {j: {} for j in range(self.N)}

                # Iterate through all edges specified by links. Note that since the variables paris are ordered, (A, B) and (B, A)
                # are seen as different pairs.
                for pair in links:

                    # Decode the elements of links into pairs of variables (X, Y)
                    if len(pair) == 2:
                        X = (pair[0], pair[1])
                        Y = (pair[0], 0)
                    else:
                        X = (pair[0], pair[2])
                        Y = (pair[1], 0)

                        # Do not test auto-links twice
                        if self.auto_first and X[0] == Y[0]:
                            continue

                    ######################################################################################################
                    ### Exclusion of links ###############################################################################

                    # Exclude the current link if ...
                    # ... X = Y
                    if X[1] == 0 and X[0] == Y[0]:
                        continue
                    # ... X > Y
                    if self._is_smaller(Y, X):
                        continue

                    # Get the current link
                    link = self._get_link(X, Y)

                    # Moreover exclude the current link if ...
                    # ... X and Y are not adjacent anymore
                    if link == "":
                        continue
                    # ... the link is definitely part of G
                    if link[1] == "-":
                        continue

                    ######################################################################################################
                    ### Determine  which tests the link will be  subjected to  ###########################################

                    # Depending on the middle mark on the link between X and Y as well as on some global options, we may not need
                    # to search for separating set among the potential parents of Y and/or X.
                    test_Y = True if link[1] not in ["R", "!"] else False
                    test_X = True if (link[1] not in ["L", "!"] and (X[1] == 0 or (self.max_cond_px > 0 and self.max_cond_px >= p_pc))) else False
                    
                    ######################################################################################################
                    ### Preparation PC search set and default conditioning set ###########################################

                    if test_Y:
                        S_default_YX, S_search_YX = self._get_default_and_search_sets(Y, X, "ancestral")

                    if test_X:
                        S_default_XY, S_search_XY = self._get_default_and_search_sets(X, Y, "ancestral")

                    ######################################################################################################
                    ### Middle mark updates ##############################################################################

                    any_middle_mark_update = False

                    # Note: Updating the middle marks here, within the for-loop, does not spoil order independence. In fact, this
                    # update does not influence the flow of the for-loop at all
                    if test_Y:
                        if len(S_search_YX) < p_pc:
                            # Note that X is smaller than Y. If S_search_YX exists and has fewer than p elements, X and Y are not
                            # d-separated by S \subset Par(Y). Therefore, the middle mark on the edge between X and Y can be updated
                            # with 'R'
                            self._apply_middle_mark(X, Y, "R")
                        else:
                            # Since S_search_YX exists and has hat least p_pc elements, the link between X and Y will be subjected to
                            # conditional independenc tests. Therefore, the algorithm has not converged yet.
                            has_converged = False

                    if test_X:
                        if len(S_search_XY) < p_pc:
                            # Note that X is smaller than Y. If S_search_XY exists and has fewer than p elements, X and Y are not
                            # d-separated by S \subset Par(X). Therefore, the middle mark on the edge between X and Y can be updated
                            # with 'L'
                            self._apply_middle_mark(X, Y, "L")
                        else:
                            # Since S_search_YX exists and has hat least p_pc elements, the link between X and Y will be subjected to
                            # conditional independenc tests. Therefore, the algorithm has not converged yet.
                            has_converged = False

                    ######################################################################################################

                    ######################################################################################################
                    ### Tests for conditional independence ###############################################################

                    # If option self.break_once_separated is True, the below for-loops will be broken immediately once a separating set
                    # has been found. In conjunction with the modified majority rule employed for orienting links, order independence
                    # (with respect to the index 'i' on X^i_t) then requires that the tested conditioning sets are ordered in an order
                    # independent way. Here, the minimal effect size of previous conditional independence tests serve as an order
                    # independent order criterion.
                    if self.break_once_separated or not np.isinf(self.max_q_global):
                        if test_Y:
                            S_search_YX = self._sort_search_set(S_search_YX, Y)
                        if test_X:
                            S_search_XY = self._sort_search_set(S_search_XY, X)

                    # Run through all cardinality p_pc subsets of S_search_YX
                    if test_Y:

                        q_count = 0
                        for S_pc in combinations(S_search_YX, p_pc):

                            q_count = q_count + 1
                            if q_count > self.max_q_global:
                                break

                            # Build the full conditioning set
                            Z = set(S_pc)
                            Z = Z.union(S_default_YX)

                            # Test conditional independence of X and Y given Z
                            val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                                tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                            if self.verbosity >= 2:
                                print("ANC(Y):    %s _|_ %s  |  S_def = %s, S_pc = %s: val = %.2f / pval = % .4f" %
                                    (X, Y, ' '.join([str(z) for z in S_default_YX]), ' '.join([str(z) for z in S_pc]), val, pval))

                            # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                            # values and conditioning set cardinalities
                            self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                            # Check whether test result was significant
                            if not dependent: #pval > self.pc_alpha:

                                # Mark the edge from X to Y for removal and save sepset
                                to_remove[Y[0]][X] = True
                                self._save_sepset(X, Y, (frozenset(Z), "wm"))
            
                                # Verbose output
                                if self.verbosity >= 1:
                                    print("({},{:2}) {:11} {} given {} union {}".format(X[0], X[1], "independent", Y, S_pc, S_default_YX))

                                if self.break_once_separated:
                                    break

                    # Run through all cardinality p_pc subsets of S_search_XY
                    if test_X:

                        q_count = 0
                        for S_pc in combinations(S_search_XY, p_pc):

                            q_count = q_count + 1
                            if q_count > self.max_q_global:
                                break

                            # Build the full conditioning set
                            Z = set(S_pc)
                            Z = Z.union(S_default_XY)

                            # Test conditional independence of X and Y given Z
                            val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                                tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                            if self.verbosity >= 2:
                                print("ANC(X):    %s _|_ %s  |  S_def = %s, S_pc = %s: val = %.2f / pval = % .4f" %
                                    (X, Y, ' '.join([str(z) for z in S_default_XY]), ' '.join([str(z) for z in S_pc]), val, pval))

                            # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                            # values and conditioning set cardinalities
                            self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                            # Check whether test result was significant
                            if not dependent: # pval > self.pc_alpha:

                                # Mark the edge from X to Y for removal and save sepset
                                to_remove[Y[0]][X] = True
                                self._save_sepset(X, Y, (frozenset(Z), "wm"))
            
                                # Verbose output
                                if self.verbosity >= 1:
                                    print("({},{:2}) {:11} {} given {} union {}".format(X[0], X[1], "independent", Y, S_pc, S_default_XY))

                                if self.break_once_separated:
                                    break

                # for pair in links

                ##########################################################################################################
                ### Remove edges marked for removal in to_remove #########################################################

                # Run through all of the nested dictionary
                for j in range(self.N):
                    for (i, lag_i) in to_remove[j].keys():

                        # Remember that at least one edge has been removed, remove the edge
                        any_removal = True
                        self._write_link((i, lag_i), (j, 0), "", verbosity = self.verbosity)

            # end for links in link_list

            # Verbose output
            if self.verbosity >= 1:
                    print("\nTest phase complete")

            ##############################################################################################################
            ### Orientations and next step ###############################################################################

            if any_removal:
                # At least one edge was removed or at least one middle mark has been updated. Therefore: i) apply the restricted set of
                # orientation rules, ii) restart the while loop at p_pc = 0, unless all edges have converged, then break the while loop

                only_lagged = False if self.orient_contemp == 2 else True
                any_update = self._run_orientation_phase(rule_list = self._rules_prelim, only_lagged = only_lagged)

                # If the orientation phase made a non-trivial update, then restart the while loop. Else increase p_pc by one
                if any_update:
                    if self.max_cond_px == 0 and self.update_middle_marks:
                        self._update_middle_marks()
                    p_pc = 0

                else:
                    p_pc = p_pc + 1

            else:
                # The graph has not changed at all in this iteration of the while loop. Therefore, if all edges have converged, break the
                # while loop. If at least one edge has not yet converged, increase p_pc by one.

                if has_converged:
                    break
                else:
                    p_pc = p_pc + 1

        # end while True

        ##################################################################################################################
        ### Consistency test and middle mark update ######################################################################

        # Run through the entire graph
        for j in range(self.N):
            for (i, lag_i) in self.graph_dict[j].keys():

                X = (i, lag_i)
                Y = (j, 0)

                if self._is_smaller(Y, X):
                    continue

                # Consider only those links that are still part G
                link = self._get_link((i, lag_i), (j, 0))
                if len(link) > 0:

                    # Consistency check
                    if not while_broken:
                        assert link[1] != "?"
                        assert link[1] != "L"
                        assert ((link[1] != "R") or (lag_i < 0 and (self.max_cond_px > 0 or not self.update_middle_marks))
                            or (self.no_apr != 0))


                    # Update all middle marks to '!'
                    if link[1] not in ["-", "!"]:
                        self._write_link((i, lag_i), (j, 0), link[0] + "!" + link[2])
                    

        ##################################################################################################################
        ### Final rule applications ######################################################################################

        if not prelim or self.prelim_with_collider_rules:

            if not prelim:
                self.no_apr = self.no_apr - 1

            any_update = self._run_orientation_phase(rule_list = self._rules_all, only_lagged = False)

            if self.max_cond_px == 0 and self.update_middle_marks and any_update:
                self._update_middle_marks()

        else:

            only_lagged = False if self.orient_contemp >= 1 else True
            any_update = self._run_orientation_phase(rule_list = self._rules_prelim_final, only_lagged = only_lagged)

            if self.max_cond_px == 0 and self.update_middle_marks and any_update:
                self._update_middle_marks()

        # Return
        return True


    def _run_non_ancestral_removal_phase(self):
        """Run the non-ancestral edge removal phase, this is Algorithm S3"""

        # Update of middle marks
        self._update_middle_marks()

        # This function initializeds self._graph_full_dict, a nested dictionary representing the graph including links that are
        # forward in time. This will make the calculcation of na-pds-t sets easier.
        self._initialize_full_graph()

        # Iterate until convergence. Here, p_pc is the cardinality of the non-default part of the conditioning sets. The full
        # conditioning sets may have higher cardinality due to default conditioning on known parents
        p_pc = 0
        while True:

            ##########################################################################################################
            ### Run the next removal iteration #######################################################################

            # Force-quit while loop when p_pc exceeds the limit put by self.max_p_global or self.max_p_non_ancestral
            if p_pc > self.max_p_global or p_pc > self.max_p_non_ancestral:
                break

            # Verbose output
            if self.verbosity >= 1:
                if p_pc == 0:
                    print("\nStarting test phase\n")
                print("p = {}".format(p_pc))

            # Variables to memorize the occurence and absence of certain events in the below edge removal phase
            has_converged = True
            any_removal = False

            # Generate the prioritized link list
            if self.auto_first:

                link_list = [product(range(self.N), range(-self.tau_max, 0))]
                link_list = link_list + [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]

            else:

                link_list = [product(range(self.N), range(self.N), range(-lag, -lag + 1)) for lag in range(0, self.tau_max + 1)]


            # Run through all elements of link_list. Each element of link_list specifies ordered pairs of variables whose connecting
            # edges are then subjected to conditional independence tests
            for links in link_list:

                # Memory variables for storing edges that are marked for removal
                to_remove = {j: {} for j in range(self.N)}

                # Iterate through all edges specified by links. Note that since the variables paris are ordered, (A, B) and (B, A) are
                # seen as different pairs.
                for pair in links:

                    if len(pair) == 2:
                        X = (pair[0], pair[1])
                        Y = (pair[0], 0)
                    else:
                        X = (pair[0], pair[2])
                        Y = (pair[1], 0)

                        # Do not test auto-links twice
                        if self.auto_first and X[0] == Y[0]:
                            continue

                    ######################################################################################################
                    ### Exclusion of links ###############################################################################

                    # Exclude the current link if ...
                    # ... X = Y
                    if X[1] == 0 and X[0] == Y[0]:
                        continue
                    # ... X > Y
                    if self._is_smaller(Y, X):
                        continue

                    # Get the current link
                    link = self._get_link(X, Y)

                    # Exclude the current link if ...
                    if link == "":
                        continue
                    # ... the link is definitely part of G
                    if link[1] == "-":
                        continue

                    ######################################################################################################
                    ### Determine which tests the link will be subjected to  #############################################

                    # The algorithm always searches for separating sets in na-pds-t(Y, X). Depending on whether the X and Y are
                    # contemporaneous on some global options, the algorithm may also search for separating sets in na-pds-t(X, Y)
                    test_X = True if (X[1] == 0 or (self.max_cond_px > 0 and self.max_cond_px >= p_pc)) else False
                    
                    ######################################################################################################
                    ### Preparation of default conditioning sets and PC search sets ######################################

                    # Verbose output
                    if self.verbosity >= 2:
                        print("_get_na_pds_t ")

                    S_default_YX, S_search_YX = self._get_default_and_search_sets(Y, X, "non-ancestral")

                    self.max_na_search_set_found = max(self.max_na_search_set_found, len(S_search_YX))

                    if test_X:
                        S_default_XY, S_search_XY = self._get_default_and_search_sets(X, Y, "non-ancestral")

                        self.max_na_search_set_found = max(self.max_na_search_set_found, len(S_search_XY))

                    # If the search set exceeds the specified bounds, do not test this link
                    if len(S_search_YX) > self.max_pds_set or (test_X and len(S_search_XY) > self.max_pds_set):
                        continue

                    ######################################################################################################

                    ######################################################################################################
                    ### Middle mark updates ##############################################################################

                    # Note: Updating the middle marks here, within the for-loop, does not spoil order independence. In fact, this
                    # update does not influence the flow of the for-loop at all
                    if len(S_search_YX) < p_pc or (test_X and len(S_search_XY) < p_pc):
                        # Mark the link from X to Y as converged, remember the fixation, then continue
                        self._write_link(X, Y, link[0] + "-" + link[2], verbosity = self.verbosity)
                        continue

                    else:
                        has_converged = False


                    ######################################################################################################
                    ### Tests for conditional independence ###############################################################

                    # If option self.break_once_separated is True, the below for-loops will be broken immediately once a separating set
                    # has been found. In conjunction with the modified majority rule employed for orienting links, order independence
                    # (with respect to the index 'i' on X^i_t) then requires that the tested conditioning sets are ordered in an order
                    # independent way. Here, the minimal effect size of previous conditional independence tests serve as an order
                    # independent order criterion.
                    if self.break_once_separated or not np.isinf(self.max_q_global):
                        S_search_YX = self._sort_search_set(S_search_YX, Y)
                        if test_X:
                            S_search_XY = self._sort_search_set(S_search_XY, X)

                    # Verbose output
                    if self.verbosity >= 2:
                        print("for S_pc in combinations(S_search_YX, p_pc)")

                    # Run through all cardinality p_pc subsets of S_search_YX
                    q_count = 0
                    for S_pc in combinations(S_search_YX, p_pc):

                        q_count = q_count + 1
                        if q_count > self.max_q_global:
                            break

                        # Build the full conditioning set
                        Z = set(S_pc)
                        Z = Z.union(S_default_YX)

                        # Test conditional independence of X and Y given Z
                        # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)
                        val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                            tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                        if self.verbosity >= 2:
                            print("Non-ANC(Y):    %s _|_ %s  |  S_def = %s, S_pc = %s: val = %.2f / pval = % .4f" %
                                (X, Y, ' '.join([str(z) for z in S_default_YX]), ' '.join([str(z) for z in S_pc]), val, pval))

                        # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                        # values and conditioning set cardinalities
                        self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                        # Check whether test result was significant
                        if not dependent: # pval > self.pc_alpha:

                            # Mark the edge from X to Y for removal and save sepset
                            to_remove[Y[0]][X] = True
                            self._save_sepset(X, Y, (frozenset(Z), "wm"))
        
                            # Verbose output
                            if self.verbosity >= 1:
                                print("({},{:2}) {:11} {} given {} union {}".format(X[0], X[1], "independent", Y, S_pc, S_default_YX))

                            if self.break_once_separated:
                                break

                    if test_X:

                        # Verbose output
                        if self.verbosity >= 2:
                            print("for S_pc in combinations(S_search_XY, p_pc)")

                        # Run through all cardinality p_pc subsets of S_search_XY
                        q_count = 0
                        for S_pc in combinations(S_search_XY, p_pc):

                            q_count = q_count + 1
                            if q_count > self.max_q_global:
                                break

                            # Build the full conditioning set
                            Z = set(S_pc)
                            Z = Z.union(S_default_XY)

                            # Test conditional independence of X and Y given Z
                            # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)
                            val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                                tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                            if self.verbosity >= 2:
                                print("Non-ANC(X):    %s _|_ %s  |  S_def = %s, S_pc = %s: val = %.2f / pval = % .4f" %
                                    (X, Y, ' '.join([str(z) for z in S_default_XY]), ' '.join([str(z) for z in S_pc]), val, pval))

                            # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                            # values and conditioning set cardinalities
                            self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                            # Check whether test result was significant
                            if not dependent: # pval > self.pc_alpha:

                                # Mark the edge from X to Y for removal and save sepset
                                to_remove[Y[0]][X] = True
                                self._save_sepset(X, Y, (frozenset(Z), "wm"))
            
                                # Verbose output
                                if self.verbosity >= 1:
                                    print("({},{:2}) {:11} {} given {} union {}".format(X[0], X[1], "independent", Y, S_pc, S_default_YX))

                                if self.break_once_separated:
                                    break

                # end for links in link_list

                ##########################################################################################################
                ### Remove edges marked for removal in to_remove #########################################################

                # Check whether there is any removal at all
                any_removal_this = False

                # Run through all of the nested dictionary
                for j in range(self.N):
                    for (i, lag_i) in to_remove[j].keys():

                        # Remember that at least one edge has been removed, remove the edge
                        any_removal = True
                        any_removal_this = True
                        self._write_link((i, lag_i), (j, 0), "", verbosity = self.verbosity)

                # If any_removal_this = True, we need to recalculate full graph dict
                if any_removal_this:
                    self._initialize_full_graph()
                    self._na_pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}


            # end for links in link_list

            # Verbose output
            if self.verbosity >= 1:
                    print("\nTest phase complete")

            ##############################################################################################################
            ### Orientations and next step ###############################################################################

            if any_removal:
                # At least one edge was removed or at least one middle mark has been updated. Therefore: i) apply the full set of
                # orientation rules, ii) restart the while loop at p_pc = 0, unless all edges have converged, then break the while loop

                any_update = self._run_orientation_phase(rule_list = self._rules_all, only_lagged = False)

                if any_update:
                    self._initialize_full_graph()
                    self._na_pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}
                    p_pc = 0

                else:
                    p_pc = p_pc + 1

            else:
                # The graph has not changed at all in this iteration of the while loop. Therefore, if all edges have converged, break
                # the while loop. If at least one edge has not yet converged, increase p_pc by one.

                if has_converged:
                    break
                else:
                    p_pc = p_pc + 1

        # end while True

        ##################################################################################################################
        ### Final rule applications ######################################################################################

        self._run_orientation_phase(rule_list = self._rules_all, only_lagged = False)

        # Return
        return True


    def _run_orientation_phase(self, rule_list, only_lagged = False):
        """Exhaustively apply the rules specified by rule_list, this is Algorithm S4"""

        # Verbose output
        if self.verbosity >= 1:
            print("\nStarting orientation phase")
            print("with rule list: ", rule_list)

        # Remember whether this call to _run_orientation_phase has made any update to G
        restarted_once = False

        # Run through all priority levels of rule_list
        idx = 0
        while idx <= len(rule_list) - 1:

            # Some rule require self._graph_full_dict. Therefore, it is initialized once the while loop (re)-starts at the first
            # prioprity level
            if idx == 0:
                self._initialize_full_graph()

            # Remember whether G will be updated with new useful information ('x' marks are considered not useful)
            restart = False

            ###########################################################################################################
            ### Rule application ######################################################################################

            # Get the current rules
            current_rules = rule_list[idx]

            # Prepare a list to remember marked orientations
            to_orient = []

            # Run through all current rules
            for rule in current_rules:

                # Verbose output
                if self.verbosity >= 1:
                    print("\n{}:".format(rule))

                # Exhaustively apply the rule to the graph...
                orientations = self._apply_rule(rule, only_lagged)

                # Verbose output
                if self.verbosity >= 1:
                    for ((i, j, lag_i), new_link) in set(orientations):
                        print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Marked:", i, lag_i, self._get_link((i, lag_i), (j, 0)), j, 0,i, lag_i, new_link, j, 0))
                    if len(orientations) == 0:
                        print("Found nothing")

                # ... and stage the results for orientation and removal
                to_orient.extend(orientations)

            ###########################################################################################################
            ### Aggregation of marked orientations ####################################################################

            links_to_remove = set()
            links_to_fix = set()
            new_ancs = {j: set() for j in range(self.N)}
            new_non_ancs = {j: set() for j in range(self.N)}

            # Run through all of the nested dictionary
            for ((i, j, lag_i), new_link) in to_orient:

                # The old link
                old_link = self._get_link((i, lag_i), (j, 0))

                # Is the link marked for removal?
                if new_link == "" and len(old_link) > 0:
                    links_to_remove.add((i, j, lag_i))
                    continue

                # Assert that no preceeding variable is marked as an ancestor of later variable
                assert not (lag_i > 0 and new_link[2] == "-")

                # Is the link marked for fixation?
                if new_link[1] == "-" and old_link[1] != "-":
                    links_to_fix.add((i, j, lag_i))

                # New ancestral relation of (i, lag_i) to (j, 0)
                if new_link[0] == "-" and old_link[0] != "-":
                    new_ancs[j].add((i, lag_i))
                elif new_link[0] == "<" and old_link[0] != "<":
                    new_non_ancs[j].add((i, lag_i))
                
                # New ancestral relation of (j, 0) to (i, lag_i == 0)
                if lag_i == 0:
                    if new_link[2] == "-" and old_link[2] != "-":
                        new_ancs[i].add((j, 0))
                    elif new_link[2] == ">" and old_link[2] != ">":
                        new_non_ancs[i].add((j, 0))

            # Resolve conflicts about removal and fixation
            ambiguous_links = links_to_fix.intersection(links_to_remove)
            links_to_fix = links_to_fix.difference(ambiguous_links)
            links_to_remove = links_to_remove.difference(ambiguous_links)

            ###########################################################################################################
            ### Removals, update middle marks, update ancestral information ###########################################

            # Remove links
            for (i, j, lag_i) in links_to_remove:
                self._write_link((i, lag_i), (j, 0), "", verbosity = self.verbosity)
                restart = True

            # Fix links
            for (i, j, lag_i) in links_to_fix:
                old_link = self._get_link((i, lag_i), (j, 0))
                new_link = old_link[0] + "-" + old_link[2]
                self._write_link((i, lag_i), (j, 0), new_link, verbosity = self.verbosity)
                restart = True

            # Mark links as ambiguous
            for (i, j, lag_i) in ambiguous_links:
                old_link = self._get_link((i, lag_i), (j, 0))
                new_link = old_link[0] + "x" + old_link[2]
                self._write_link((i, lag_i), (j, 0), new_link, verbosity = self.verbosity)

            # Update ancestral information. The function called includes conflict resolution
            restart = restart or self._apply_new_ancestral_information(new_non_ancs, new_ancs)

            ###########################################################################################################
            ### Make separating sets of removed links weakly minimal ##################################################

            if len(links_to_remove) > 0:

                # Verbose output
                if self.verbosity >= 1:
                    print("\nLinks were removed by rules\n")

                new_ancs = {j: set() for j in range(self.N)}
                new_non_ancs = {j: set() for j in range(self.N)}

                # Run through all links that have been removed
                for (i, j, lag_i) in links_to_remove:

                    X = (i, lag_i)
                    Y = (j, 0)

                    # Get ancestors of X and Y
                    ancs_XY = self._get_ancs([X, Y]).difference({X, Y})

                    # Read out all separating sets that were found in the rule phase, then consider only those of minimal
                    # cardinality
                    old_sepsets_all = {Z for (Z, _) in self._get_sepsets(X, Y)}
                    min_size = min({len(Z) for Z in old_sepsets_all})
                    old_sepsets_smallest = {Z for Z in old_sepsets_all if len(Z) == min_size}

                    # For all separating sets of minimal cardinality, find weakly minimal separating subsets
                    self._delete_sepsets(X, Y)
                    self._make_sepset_weakly_minimal(X, Y, old_sepsets_smallest, ancs_XY)
                    new_sepsets = self._get_sepsets(X, Y)

                # end for (i, j, lag_i) in links_to_remove
            # end  if len(links_to_remove) > 0

            # If any useful new information was found, go back to idx = 0, else increase idx by 1
            if restart:
                idx = 0
                restarted_once = True
            else:
                idx = idx + 1

        # end while idx <= len(rule_list) - 1

        # Verbose output
        if self.verbosity >= 1:
            print("\nOrientation phase complete")

        # No return value
        return restarted_once

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _get_default_and_search_sets(self, A, B, phase):
        """Return the default conditioning set and PC search set"""

        if phase == "ancestral":

            # This is a-pds-t(A, B)
            S_raw = self._get_a_pds_t(A, B)

            # Determine the default conditioning set
            S_default = self._get_parents(A, B).difference({A, B})

            # Determine the PC search set
            S_search = S_raw.difference(S_default)


        elif phase == "non-ancestral":

            # This is na-pds-t(A, B)
            S_raw = self._get_na_pds_t(A, B)

            self.max_na_pds_set_found = max(self.max_na_pds_set_found, len(S_raw))

            # Determine the default conditioning set
            S_default = S_raw.intersection(self._get_ancs([A, B]))
            S_default = S_default.union(self._get_parents(A, B))
            S_default = S_default.difference({A, B})

            # Determine the PC search set
            S_search = S_raw.difference(S_default)

        # Return
        return S_default, S_search


    def _apply_new_ancestral_information(self, new_non_ancs, new_ancs):
        """Apply the new ancestorships and non-ancestorships specified by new_non_ancs and new_ancs to the current graph. Conflicts
        are resolved by marking. Returns True if any circle mark was turned into a head or tail, else False."""

        #######################################################################################################
        ### Preprocessing #####################################################################################

        # Memory variables
        add_to_def_non_ancs = {j: set() for j in range(self.N)}
        add_to_def_ancs = {j: set() for j in range(self.N)}
        add_to_ambiguous_ancestorships = {j: set() for j in range(self.N)}
        put_head_or_tail = False

        # Default values
        if new_non_ancs is None:
            new_non_ancs = {j: set() for j in range(self.N)}

        if new_ancs is None:
            new_ancs = {j: set() for j in range(self.N)}

        # Marking A as ancestor of B implies that B is marked as a non-ancestor of A. This is only non-trivial for A before B
        for j in range(self.N):
            for (i, lag_i) in new_ancs[j]:
                if lag_i == 0:
                    new_non_ancs[i].add((j, 0))

        #######################################################################################################
        ### Conflict resolution ###############################################################################

        # Iterate through new_non_ancs
        for j in range(self.N):
            for (i, lag_i) in new_non_ancs[j]:
                # X = (i, lag_i), Y = (j, 0)
                # X is marked as non-ancestor for Y

                # Conflict resolution
                if (i, lag_i) in self.ambiguous_ancestorships[j]:
                    # There is a conflict, since it is already marked as ambiguous whether X is an ancestor of Y
                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as non-anc of {} but saved as ambiguous".format("Conflict:", i, lag_i, (j, 0)))

                elif (i, lag_i) in self.def_ancs[j]:
                    # There is a conflict, since X is already marked as ancestor of Y
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))

                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as non-anc of {} but saved as anc".format("Conflict:", i, lag_i, (j, 0)))

                elif (i, lag_i) in new_ancs[j]:
                    # There is a conflict, since X is also marked as a new ancestor of Y
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))

                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as both anc- and non-anc of {}".format("Conflict:", i, lag_i, (j, 0)))

                else:
                    # There is no conflict
                    add_to_def_non_ancs[j].add((i, lag_i))
                    
        # Iterate through new_ancs
        for j in range(self.N):
            for (i, lag_i) in new_ancs[j]:
                # X = (i, lag_i), Y = (j, 0)
                # X is marked as ancestor for Y

                # Conflict resolution
                if (i, lag_i) in self.ambiguous_ancestorships[j]:
                    # There is a conflict, since it is already marked as ambiguous whether X is an ancestor of Y
                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as anc of {} but saved as ambiguous".format("Conflict:", i, lag_i, (j, 0)))

                elif lag_i == 0 and (j, 0) in self.ambiguous_ancestorships[i]:
                    # There is a conflict, since X and Y are contemporaneous and it is already marked ambiguous as whether Y is an
                    # ancestor of X
                    # Note: This is required here, because X being an ancestor of Y implies that Y is not an ancestor of X. This
                    # ambiguity cannot exist when X is before Y
                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as anc of {} but saved as ambiguous".format("Conflict:", i, lag_i, (j, 0)))

                elif (i, lag_i) in self.def_non_ancs[j]:
                    # There is a conflict, since X is already marked as non-ancestor of Y
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))

                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as anc of {} but saved as non-anc".format("Conflict:", i, lag_i, (j, 0)))

                elif (i, lag_i) in new_non_ancs[j]:
                    # There is a conflict, since X is also marked as a new non-ancestor of Y
                    add_to_ambiguous_ancestorships[j].add((i, lag_i))

                    if self.verbosity >= 1:
                        print("{:10} ({}, {:2}) marked as both anc- and non-anc of {}".format("Conflict:", i, lag_i, (j, 0)))

                else:
                    # There is no conflict
                    add_to_def_ancs[j].add((i, lag_i))

        #######################################################################################################

        #######################################################################################################
        ### Apply the ambiguous information ###################################################################

        for j in range(self.N):

            for (i, lag_i) in add_to_ambiguous_ancestorships[j]:

                old_link = self._get_link((i, lag_i), (j, 0))
                if len(old_link) > 0 and old_link[0] != "x":

                    new_link = "x" + old_link[1] + old_link[2]
                    self._write_link((i, lag_i), (j, 0), new_link, verbosity = self.verbosity)

                if self.verbosity >= 1:
                    if (i, lag_i) in self.def_ancs[j]:
                        print("{:10} Removing ({}, {:2}) as anc of {}".format("Update:", i, lag_i, (j, 0)))
                    if (i, lag_i) in self.def_non_ancs[j]:
                        print("{:10} Removing ({}, {:2}) as non-anc of {}".format("Update:", i, lag_i, (j, 0)))

                self.def_ancs[j].discard((i, lag_i))
                self.def_non_ancs[j].discard((i, lag_i))

                if lag_i == 0:

                    if self.verbosity >= 1 and (j, 0) in self.def_ancs[i]:
                        print("{:10} Removing {} as anc of {}".format("Update:", i, lag_i, (j, 0)))

                    self.def_ancs[i].discard((j, 0))
                    # Do we also need the following?
                    # self.def_non_ancs[i].discard((j, 0))

                if self.verbosity >= 1 and (i, lag_i) not in self.ambiguous_ancestorships[j]:
                    print("{:10} Marking ancestorship of ({}, {:2}) to {} as ambiguous".format("Update:", i, lag_i, (j, 0)))

                self.ambiguous_ancestorships[j].add((i, lag_i))

        #######################################################################################################
        ### Apply the unambiguous information #################################################################

        for j in range(self.N):

            for (i, lag_i) in add_to_def_non_ancs[j]:

                old_link = self._get_link((i, lag_i), (j, 0))
                if len(old_link) > 0 and old_link[0] != "<":
                    new_link = "<" + old_link[1] + old_link[2]
                    self._write_link((i, lag_i), (j, 0), new_link, verbosity = self.verbosity)
                    put_head_or_tail = True

                if self.verbosity >= 1 and (i, lag_i) not in self.def_non_ancs[j]:
                    print("{:10} Marking ({}, {:2}) as non-anc of {}".format("Update:", i, lag_i, (j, 0)))  

                self.def_non_ancs[j].add((i, lag_i))


            for (i, lag_i) in add_to_def_ancs[j]:

                old_link = self._get_link((i, lag_i), (j, 0))
                if len(old_link) > 0 and (old_link[0] != "-" or old_link[2] != ">"):
                    new_link = "-" + old_link[1] + ">"
                    self._write_link((i, lag_i), (j, 0), new_link, verbosity = self.verbosity)
                    put_head_or_tail = True

                if self.verbosity >= 1 and (i, lag_i) not in self.def_ancs[j]:
                    print("{:10} Marking ({}, {:2}) as anc of {}".format("Update:", i, lag_i, (j, 0)))

                self.def_ancs[j].add((i, lag_i))

                if lag_i == 0:

                    if self.verbosity >= 1 and (j, 0) not in self.def_non_ancs[i]:
                        print("{:10} Marking {} as non-anc of {}".format("Update:",(j, 0), (i, 0)))

                    self.def_non_ancs[i].add((j, 0))

        #######################################################################################################

        return put_head_or_tail

    def _apply_rule(self, rule, only_lagged):
        """Call the orientation-removal-rule specified by the string argument rule."""

        if rule == "APR":
            return self._apply_APR(only_lagged)
        elif rule == "ER-00-a":
            return self._apply_ER00a(only_lagged)
        elif rule == "ER-00-b":
            return self._apply_ER00b(only_lagged)
        elif rule == "ER-00-c":
            return self._apply_ER00c(only_lagged)
        elif rule == "ER-00-d":
            return self._apply_ER00d(only_lagged)
        elif rule == "ER-01":
            return self._apply_ER01(only_lagged)
        elif rule == "ER-02":
            return self._apply_ER02(only_lagged)
        elif rule == "ER-03":
            return self._apply_ER03(only_lagged)
        elif rule == "R-04":
            return self._apply_R04(only_lagged)
        elif rule == "ER-08":
            return self._apply_ER08(only_lagged)
        elif rule == "ER-09":
            return self._apply_ER09(only_lagged)
        elif rule == "ER-10":
            return self._apply_ER10(only_lagged)


    def _get_na_pds_t(self, A, B):
        """Return the set na_pds_t(A, B), with at least one of them at lag 0"""

        # Unpack A and B, then assert that at least one of them is at lag 0
        var_A, lag_A = A
        var_B, lag_B = B
        assert lag_A == 0 or lag_B == 0

        # If na_pds_t(A, B) is in memory, return immediately
        memo = self._na_pds_t[A].get(B)
        if memo is not None:
            return memo

        # Else, re-compute na_pds_t(A, B) it according to the current graph and cache it.

        # Re-compute na_pds_t_1(A, B) according to the current graph
        na_pds_t_1 = {(var, lag + lag_A)
                    # W = (var, lag + lag_A) is in na_pds_t_1(A, B) if ...
                    for ((var, lag), link) in self.graph_dict[var_A].items()
                    # ... it is a non-future adjacency of A
                    if len(link) > 0
                    # ... and is not B
                    and (var, lag + lag_A) != B
                    # ... and is not before t - tau_max
                    and (lag + lag_A) >= -self.tau_max
                    # ... and is not after both A and B
                    # ... (i.e. is not after time t)
                    and (lag + lag_A) <= 0
                    # ... and is not a definite non-ancestor of A,
                    #     which implies that it is not a definite descendant of A,
                    and link[0] != "<"
                    # ... and is not a definite descendant of B
                    #     (i.e., B is not a definite ancestor of W)
                    and (var_B, lag_B - (lag + lag_A)) not in self.def_ancs[var]
                    }

        # Compute na_pds_t_2(A, B)

        # Find all potential C_1 nodes
        C1_list = set()
        for ((var, lag), link) in self.graph_full_dict[var_A].items():

            node = (var, lag + lag_A)

            # node is added to C1_list if, in addition to being adjacent to A, ...
            # ... it is not B
            if (var, lag + lag_A) == B:
                continue

            # ... it is not before t - tau_max
            if (lag + lag_A) < -self.tau_max:
                continue

            # ... it is not after B
            if (lag + lag_A) > lag_B:
                continue

            # ... it is not a definite ancestor of A
            if link[0] == "-":
                continue

            # ... it is not a definite descendant of A
            if link[2] == "-":
                continue

            # ... it is not a definite non-ancestor of B,
            #     which implies that it is not a definite descendant of B
            if (var, (lag + lag_A) - lag_B) in self.def_non_ancs[var_B]:
                continue

            # If all tests are passed, node is added to C1_list
            C1_list.add(node)

        # end for ((var, lag), link) in self.graph_full_dict[var_A].items()

        # Breath first search to find (a superset of) na_pds_t_2(A, B)

        visited = set()
        start_from = {(C1, A) for C1 in C1_list}

        while start_from:

            new_start_from = set()
            new_do_not_visit = set()

            for (current_node, previous_node) in start_from:

                visited.add((current_node, previous_node))

                for (var, lag) in self.graph_full_dict[current_node[0]]:

                    next_node = (var, lag + current_node[1])

                    if next_node[1] < -self.tau_max:
                        continue
                    if next_node[1] > 0:
                        continue
                    if (next_node, current_node) in visited:
                        continue
                    if next_node == previous_node:
                        continue
                    if next_node == B:
                        continue
                    if next_node == A:
                        continue

                    link_l = self._get_link(next_node, current_node)
                    link_r = self._get_link(previous_node, current_node)

                    if link_l[2] == "-" or link_r[2] == "-":
                        continue
                    if self._get_link(next_node, previous_node) == "" and (link_l[2] == "o" or link_r[2] == "o"):
                        continue
                    if (var_A, lag_A - next_node[1]) in self.def_ancs[next_node[0]] or (var_B, lag_B - next_node[1]) in self.def_ancs[next_node[0]]:
                        continue
                    if ((next_node[1] - lag_A > 0) or (next_node[0], next_node[1] - lag_A) in self.def_non_ancs[var_A]) and ((next_node[1] - lag_B > 0) or (next_node[0], next_node[1] - lag_B) in self.def_non_ancs[var_B]):
                        continue

                    new_start_from.add((next_node, current_node))

            start_from = new_start_from

        # end  while start_from

        na_pds_t_2 = {node for (node, _) in visited}

        self._na_pds_t[A][B] = na_pds_t_1.union(na_pds_t_2).difference({A, B})
        return self._na_pds_t[A][B]


    def _make_sepset_weakly_minimal(self, X, Y, Z_list, ancs):
        """
        X and Y are conditionally independent given Z in Z_list However, it is not yet clear whether any of these Z are minimal
        separating set.

        This function finds weakly minimal separating subsets in an order independent way and writes them to the self.sepsets
        dictionary. Only certainly weakly minimal separating subsets are retained.
        """

        # Assert that all Z in Z_list have the same cardinality
        assert len({len(Z) for Z in Z_list}) == 1

        # Base Case 1:
        # Z in Z_list is weakly minimal if len(Z) <= 1 or Z \subset ancs
        any_weakly_minimal = False

        for Z in Z_list:

            if len(Z) <=1 or Z.issubset(ancs):
                self._save_sepset(X, Y, (frozenset(Z), "wm"))
                any_weakly_minimal = True

        if any_weakly_minimal:
            return None

        # If not Base Case 1, we need to search for separating subsets. We do this for all Z in Z_list, and build a set sepsets_next_call
        # that contains all separating sets for the next recursive call
        sepsets_next_call = set()

        for Z in Z_list:

            # Find all nodes A in Z that are not in ancs
            removable = Z.difference(ancs)

            # Test for removal of all nodes in removable
            new_sepsets = []
            val_values = []

            for A in removable:

                Z_A = [node for node in Z if node != A]

                # Run the conditional independence test
                # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = Z_A, tau_max = self.tau_max)
                val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = Z_A, 
                    tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                if self.verbosity >= 2:
                    print("MakeMin:    %s _|_ %s  |  Z_A = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z_A)]), val, pval))

                # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                # values and conditioning set cardinalities
                self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_A))

                # Check whether the test result was significant
                if not dependent: # pval > self.pc_alpha:
                    new_sepsets.append(frozenset(Z_A))
                    val_values.append(val)

            # If new_sepsets is empty, then Z is already weakly minimal
            if len(new_sepsets) == 0:
                self._save_sepset(X, Y, (frozenset(Z), "wm"))
                any_weakly_minimal = True

            # If we did not yet find a weakly minimal separating set
            if not any_weakly_minimal:

                # Sort all separating sets in new_sepets by their test statistic, then append those separating sets with maximal statistic
                # to sepsets_next_call. This i) guarantees order independence while ii) continuing to test as few as possible separating sets
                new_sepsets = [node for _, node in sorted(zip(val_values, new_sepsets), reverse = True)]

                i = -1
                while i <= len(val_values) - 2 and val_values[i + 1] == val_values[0]:
                    sepsets_next_call.add(new_sepsets[i])
                    i = i + 1

                assert i >= 0

        # If we did not yet find a weakly minimal separating set, make a recursive call
        if not any_weakly_minimal:
            self._make_sepset_weakly_minimal(X, Y, sepsets_next_call, ancs)
        else:
            return None


    def _B_not_in_SepSet_AC(self, A, B, C):
        """Is B in less than half of the sets in SepSets(A, C)?"""

        # Treat A - B - C as the same triple as C - B - A
        # Convention: A is before C or, if they are contemporaneous, the index of A is smaller than that of C
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_not_in_SepSet_AC(C, B, A)

        # Remember all separating sets that we will find
        all_sepsets = set()

        # Get the non-future adjacencies of A and C
        if not self.use_a_pds_t_for_majority:
            adj_A = self._get_non_future_adj([A]).difference({A, C})
            adj_C = self._get_non_future_adj([C]).difference({A, C})
        else:
            adj_A = self._get_a_pds_t(A, C).difference({A, C})
            adj_C = self._get_a_pds_t(C, A).difference({A, C})

        Z_add = self._get_parents(A, C).difference({A, C})

        search_A = adj_A.difference(Z_add)
        search_C = adj_C.difference(Z_add)

        if not np.isinf(self.max_q_global):
            search_A = self._sort_search_set(search_A, A)
            search_C = self._sort_search_set(search_C, C)

        # Test for independence given all subsets of non-future adjacencies of A
        if A[1] < C[1]:
            max_p_A = min([len(search_A), self.max_cond_px, self.max_p_global]) + 1
        else:
            max_p_A = min([len(search_A), self.max_p_global]) + 1

        # Shift lags
        search_A = [(var, lag - C[1]) for (var, lag) in search_A]
        search_C = [(var, lag - C[1]) for (var, lag) in search_C]
        Z_add = {(var, lag - C[1]) for (var, lag) in Z_add}
        X = (A[0], A[1] - C[1])
        Y = (C[0], 0)

        for p in range(max_p_A):

            q_count = 0
            for Z_raw in combinations(search_A, p):

                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break

                # Prepare the conditioning set
                Z = {node for node in Z_raw if node != X and node != Y}
                Z = Z.union(Z_add)

                # Test conditional independence of X and Y given Z
                # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)
                val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                    tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                if self.verbosity >= 2:
                    print("BnotinSepSetAC(A):    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in Z_add]), ' '.join([str(z) for z in {node for node in Z_raw if node != X and node != Y}]), val, pval))

                # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                # values and conditioning set cardinalities
                self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                # Check whether test result was significant
                if not dependent: # pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

        # Test for independence given all subsets of non-future adjacencies of C
        for p in range(min(len(search_C), self.max_p_global) + 1):

            q_count = 0 
            for Z_raw in combinations(search_C, p):

                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break

                # Prepare the conditioning set
                Z = {node for node in Z_raw if node != X and node != Y}
                Z = Z.union(Z_add)

                # Test conditional independence of X and Y given Z
                # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)
                val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                    tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                if self.verbosity >= 2:
                    # print("BnotinSepSetAC(C):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                    #     (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))
                    print("BnotinSepSetAC(C):    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in Z_add]), ' '.join([str(z) for z in {node for node in Z_raw if node != X and node != Y}]), val, pval))

                # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                # values and conditioning set cardinalities
                self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                # Check whether test result was significant
                if not dependent: # pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

        # Append the already known sepset
        all_sepsets = all_sepsets.union({Z for (Z, _) in self._get_sepsets(X, Y)})

        # Count number of sepsets and number of sepsets that contain B
        n_sepsets = len(all_sepsets)
        n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])

        return True if 2*n_sepsets_with_B < n_sepsets else False


    def _B_in_SepSet_AC(self, A, B, C):
        """Is B in more than half of the sets in SepSets(A, C)?"""

        # Treat A - B - C as the same triple as C - B - A
        # Convention: A is before C or, if they are contemporaneous, the index of A is smaller than that of C
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_in_SepSet_AC(C, B, A)

        link_AB = self._get_link(A, B)
        link_CB = self._get_link(C, B)

        if link_AB == "" or link_CB == "" or link_AB[1] != "-" or link_CB[1] != "-":

            # Vote is based on those sets that where found already
            all_sepsets = {Z for (Z, _) in self._get_sepsets(A, C)}

            # Count number of sepsets and number of sepsets that contain B
            n_sepsets = len(all_sepsets)
            n_sepsets_with_B = len([1 for Z in all_sepsets if B in Z])

            return True if 2*n_sepsets_with_B > n_sepsets else False

        else:

            # Remember all separating sets that we will find
            all_sepsets = set()

            # Get the non-future adjacencies of A and C
            if not self.use_a_pds_t_for_majority:
                adj_A = self._get_non_future_adj([A]).difference({A, C})
                adj_C = self._get_non_future_adj([C]).difference({A, C})
            else:
                adj_A = self._get_a_pds_t(A, C).difference({A, C})
                adj_C = self._get_a_pds_t(C, A).difference({A, C})

            Z_add = self._get_parents(A, C).difference({A, C})

            search_A = adj_A.difference(Z_add)
            search_C = adj_C.difference(Z_add)

            if not np.isinf(self.max_q_global):
                search_A = self._sort_search_set(search_A, A)
                search_C = self._sort_search_set(search_C, C)

            # Test for independence given all subsets of non-future adjacencies of A
            if A[1] < C[1]:
                max_p_A = min([len(search_A), self.max_cond_px, self.max_p_global]) + 1
            else:
                max_p_A = min([len(search_A), self.max_p_global]) + 1

            # Shift lags
            search_A = [(var, lag - C[1]) for (var, lag) in search_A]
            search_C = [(var, lag - C[1]) for (var, lag) in search_C]
            Z_add = {(var, lag - C[1]) for (var, lag) in Z_add}
            X = (A[0], A[1] - C[1])
            Y = (C[0], 0)

            for p in range(max_p_A):

                q_count = 0
                for Z_raw in combinations(search_A, p):

                    q_count = q_count + 1
                    if q_count > self.max_q_global:
                        break

                    # Prepare the conditioning set
                    Z = {node for node in Z_raw if node != X and node != Y}
                    Z = Z.union(Z_add)

                    # Test conditional independence of X and Y given Z
                    # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)
                    val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                        tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)
                    
                    if self.verbosity >= 2:
                        # print("BinSepSetAC(A):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        #     (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))
                        print("BinSepSetAC(A):    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in Z_add]), ' '.join([str(z) for z in {node for node in Z_raw if node != X and node != Y}]), val, pval))

                    # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                    # values and conditioning set cardinalities
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                    # Check whether test result was significant
                    if not dependent: # pval > self.pc_alpha:
                        all_sepsets.add(frozenset(Z))

            # Test for independence given all subsets of non-future adjacencies of C
            for p in range(min(len(search_C), self.max_p_global) + 1):

                q_count = 0 
                for Z_raw in combinations(search_C, p):

                    q_count = q_count + 1
                    if q_count > self.max_q_global:
                        break

                    # Prepare the conditioning set
                    Z = {node for node in Z_raw if node != X and node != Y}
                    Z = Z.union(Z_add)

                    # Test conditional independence of X and Y given Z
                    # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)
                    val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), 
                        tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)
                    
                    if self.verbosity >= 2:
                        # print("BinSepSetAC(C):     %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        #     (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))
                        print("BinSepSetAC(C):    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in Z_add]), ' '.join([str(z) for z in {node for node in Z_raw if node != X and node != Y}]), val, pval))

                    # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic
                    # values and conditioning set cardinalities
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z))

                    # Check whether test result was significant
                    if not dependent: # pval > self.pc_alpha:
                        all_sepsets.add(frozenset(Z))

            # Append the already known sepset
            all_sepsets = all_sepsets.union({Z for (Z, _) in self._get_sepsets(X, Y)})

            # Count number of sepsets and number of sepsets that contain B
            n_sepsets = len(all_sepsets)
            n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])

            return True if 2*n_sepsets_with_B > n_sepsets else False


    def _get_parents(self, A, B):
        """Return all known parents of all nodes in node_list"""

        if self.parents_of_lagged or A[1] == B[1]:

            out = {(var, lag + A[1]) for ((var, lag), link) in self.graph_dict[A[0]].items() if len(link) > 0 and link[0] == "-" and lag + A[1] >= -self.tau_max}
            return out.union({(var, lag + B[1]) for ((var, lag), link) in self.graph_dict[B[0]].items() if len(link) > 0 and link[0] == "-" and lag + B[1] >= -self.tau_max})

        else:
            if A[1] < B[1]:
                return {(var, lag + B[1]) for ((var, lag), link) in self.graph_dict[B[0]].items() if len(link) > 0 and link[0] == "-" and lag + B[1] >= -self.tau_max}
            else:
                return {(var, lag + A[1]) for ((var, lag), link) in self.graph_dict[A[0]].items() if len(link) > 0 and link[0] == "-" and lag + A[1] >= -self.tau_max}


    def _apply_middle_mark(self, X, Y, char):
        """Update the middle mark on the link between X and Y with the character char"""

        # Get the old link
        old_link = self._get_link(X, Y)

        # Determine the new link
        if old_link[1] == "?":
            new_link = old_link[0] + char + old_link[2]
        elif (old_link[1] == "L" and char == "R") or (old_link[1] == "R" and char == "L"):
            new_link = old_link[0] + "!" + old_link[2]
        else:
            assert False

        # Write the new link
        self._write_link(X, Y, new_link, verbosity = self.verbosity)

        # Return
        return True


    def _update_middle_marks(self):
        """Apply rule MMR"""

        if self.verbosity >= 1:
            print("\nMiddle mark updates\n")

        # Run through all links
        for j in range(self.N):
            for ((i, lag_i), link) in self.graph_dict[j].items():

                if link == "":
                    continue

                X = (i, lag_i)
                Y = (j, 0)

                # Apply above rule for A = X and B = Y
                link_XY = self._get_link(X, Y)
                smaller_XY = self._is_smaller(X, Y)

                if link_XY[2] == ">":

                    if link_XY[1] == "?":
                        if smaller_XY:
                            new_link = link_XY[0] + "L>"
                        else:
                            new_link = link_XY[0] + "R>"

                        self._write_link(X, Y, new_link, verbosity = self.verbosity)

                    elif (link_XY[1] == "R" and smaller_XY) or (link_XY[1] == "L" and not smaller_XY):

                        new_link = link_XY[0] + "!>"

                        self._write_link(X, Y, new_link, verbosity = self.verbosity)


                # Apply above rule for A = Y and B = X
                link_YX = self._get_link(Y, X)
                smaller_YX = self._is_smaller(Y, X)

                if link_YX[2] == ">":

                    if link_YX[1] == "?":
                        if smaller_YX:
                            new_link = link_YX[0] + "L>"
                        else:
                            new_link = link_YX[0] + "R>"

                        self._write_link(Y, X, new_link, verbosity = self.verbosity)
   

                    elif (link_YX[1] == "R" and smaller_YX) or (link_YX[1] == "L" and not smaller_YX):

                        new_link = link_YX[0] + "!>"

                        self._write_link(Y, X, new_link, verbosity = self.verbosity)
    
    def _is_smaller(self, X, Y):
        """
        A node X is said to be smaller than node Y if
        i)  X is before Y or
        ii) X and Y are contemporaneous and the variable index of X is smaller than that of Y.

        Return True if X is smaller than Y, else return False
        """

        return (X[1] < Y [1]) or (X[1] == Y[1] and X[0] < Y[0])           


    def _get_a_pds_t(self, A, B):
        """Return the set a_pds_t(A, B)"""

        # Unpack A and assert that A is at lag 0
        var_A, lag_A = A

        # Compute a_pds_t(A, B) according to the current graph
        return {(var, lag + lag_A)
                    # W = (var, lag) is in a_pds_t(A, B) if ...
                    for ((var, lag), link) in self.graph_dict[var_A].items()
                    # ... it is a non-future adjacency of A
                    if len(link) > 0
                    # ... and it is not B
                    and (var, lag + lag_A) != B
                    # ... it is not before t - self.tau_max
                    and lag + lag_A >= -self.tau_max
                    # ... and it is not a definite non-ancestor of A
                    and link[0] != "<"
                    }


    def _get_ancs(self, node_list):
        """Return the currently known set of ancestors of all nodes in the list node_list. The nodes are not required to be at
        lag 0"""

        # Build the output set
        out = set()

        # Run through all nodes
        for A in node_list:
            # Unpack the node
            (var_A, lag_A) = A
            # Add the ancestors of node to out
            out = out.union({(var, lag + lag_A) for (var, lag) in self.def_ancs[var_A] if lag + lag_A >= - self.tau_max})

        # Return
        return out


    def _get_non_ancs(self, node_list):
        """Return the currently known set of non-ancestors of all nodes in the list node_list. The nodes are not required to be
        at lag 0"""

        # Build the output set
        out = set()

        # Run through all nodes
        for A in node_list:
            # Unpack the node
            (var_A, lag_A) = A
            # Add the ancestors of node to out
            out = out.union({(var, lag + lag_A) for (var, lag) in self.def_non_ancs[var_A] if lag + lag_A >= - self.tau_max})

        # Return
        return out


    def _fix_all_edges(self):
        """Remove all non-trivial orientations"""

        for j in range(self.N):
            for (i, lag_i) in self.graph_dict[j].keys():

                link = self._get_link((i, lag_i), (j, 0))
                if len(link) > 0:
                    new_link = link[0] + "-" + link[2]
                    self.graph_dict[j][(i, lag_i)] = new_link

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _apply_APR(self, only_lagged):
        """Return all orientations implied by orientation rule APR"""

        # Build the output list
        out = []

        if self.no_apr > 0:
            return out

        # Get and run through all relevant graphical structures
        for j in range(self.N):
            for (i, lag_i) in self.graph_dict[j]:

                A = (i, lag_i)
                B = (j, 0)

                if only_lagged and lag_i == 0:
                    continue

                # Get the link from A to B
                link_AB = self._get_link(A, B)

                if self._match_link(pattern='-!>', link=link_AB) \
                   or (self._match_link(pattern='-R>', link=link_AB) and self._is_smaller(A, B)) \
                   or (self._match_link(pattern='-L>', link=link_AB) and self._is_smaller(B, A)):

                    # Write the new link from A to B to the output list
                    out.append(self._get_pair_key_and_new_link(A, B, "-->"))

        # Return the output list
        return out

    def _apply_ER01(self, only_lagged):
        """Return all orientations implied by orientation rule R1^prime"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='**>', pattern_jk='o*+', pattern_ik='')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if only_lagged and B[1] == C[1]:
                    continue

            if self.verbosity >= 2:
                print("ER01: ", (A, B, C))

            # Check whether the rule applies
            if self._B_in_SepSet_AC(A, B, C):

                if self.verbosity >= 2:
                    print(" --> in sepset ")

                # Prepare the new link from B to C and append it to the output list
                link_BC = self._get_link(B, C)
                new_link_BC = "-" + link_BC[1] + ">"
                out.append(self._get_pair_key_and_new_link(B, C, new_link_BC))

        # Return the output list
        return out

    def _apply_ER02(self, only_lagged):
        """Return all orientations implied by orientation rule R2^prime"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = set(self._find_triples(pattern_ij='-*>', pattern_jk='**>', pattern_ik='+*o'))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='**>', pattern_jk='-*>', pattern_ik='+*o')))

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if only_lagged and A[1] == C[1]:
                    continue

            # The rule applies to all relevant graphical structures. Therefore, prepare the new link and append it to the output list
            link_AC = self._get_link(A, C)
            new_link_AC = link_AC[0] + link_AC[1] + ">"
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))

            # print("Rule 2", A, self._get_link(A, B), B, self._get_link(B, C), C, self._get_link(A, C), new_link_AC)

        # Return the output list
        return out


    def _apply_ER03(self, only_lagged):
        """Return all orientations implied by orientation rule R3^prime"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_quadruples = self._find_quadruples(pattern_ij='**>', pattern_jk='<**', pattern_ik='', 
                                                           pattern_il='+*o', pattern_jl='o*+', pattern_kl='+*o')

        # Run through all appropriate graphical structures
        for (A, B, C, D) in all_appropriate_quadruples:

            if only_lagged and B[1] == D[1]:
                continue

            # Check whether the rule applies
            if self._B_in_SepSet_AC(A, D, C):

                # Prepare the new link from D to B and append it to the output list
                link_DB = self._get_link(D, B)
                new_link_DB = link_DB[0] + link_DB[1] + ">"
                out.append(self._get_pair_key_and_new_link(D, B, new_link_DB))

        # Return the output list
        return out


    def _apply_R04(self, only_lagged):
        """Return all orientations implied by orientation rule R4 (standard FCI rule)"""

        # Build the output list
        out = []

        # Find all relevant triangles W-V-Y
        all_appropriate_triples = self._find_triples(pattern_ij='<-*', pattern_jk='o-+', pattern_ik='-->')

        # Run through all of these triangles
        for triple in all_appropriate_triples:

            (W, V, Y) = triple

            if only_lagged and (V[1] == Y[1] and W[1] == V[1]):
                continue

            # Get the current link from W to V, which we will need below
            link_WV = self._get_link(W, V)

            # Find all discriminating paths for this triangle
            # Note: To guarantee order independence, we check all discriminating paths. Alternatively, we could check the rule for all
            # shortest such paths
            discriminating_paths =  self._get_R4_discriminating_paths(triple, max_length = np.inf)

            # Run through all discriminating paths
            for path in discriminating_paths:

                # Get the end point node
                X_1 = path[-1]

                # Check which of the two cases of the rule we are in, then append the appropriate new links to the output list
                if self._B_in_SepSet_AC(X_1, V, Y):
                    # New link from V to Y
                    out.append(self._get_pair_key_and_new_link(V, Y, "-->"))

                elif link_WV != "<-x" and self._B_not_in_SepSet_AC(X_1, V, Y):
                    # New link from V to Y
                    out.append(self._get_pair_key_and_new_link(V, Y, "<->"))

                    # If needed, also the new link from W to V
                    if link_WV != "<->":
                        out.append(self._get_pair_key_and_new_link(W, V, "<->"))

        # Return the output list
        return out


    def _apply_ER08(self, only_lagged):
        """Return all orientations implied by orientation rule R8^prime"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='-*>', pattern_jk='-*>', pattern_ik='o*+')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if only_lagged and A[1] == C[1]:
                continue

            # The rule applies to all relevant graphical structures. Therefore, prepare the new link and append it to the output list
            link_AC = self._get_link(A, C)
            new_link_AC = "-" + link_AC[1] + ">"
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))

            #print("Rule 8:", A, self._get_link(A, B), B, self._get_link(B, C), C, link_AC, new_link_AC)

        # Return the output list
        return out


    def _apply_ER09(self, only_lagged):
        """Return all orientations implied by orientation rule R9^prime"""

        # Build the output list
        out = []

        # Find unshielded triples B_1 o--*--o A o--*--> C or B_1 <--*--o A o--*--> C or B_1 <--*-- A o--*--> C 
        all_appropriate_triples = set(self._find_triples(pattern_ij='o*o', pattern_jk='o*>', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<*o', pattern_jk='o*>', pattern_ik='')))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<*-', pattern_jk='o*>', pattern_ik='')))

        # Run through all these triples
        for (B_1, A, C) in all_appropriate_triples:

            if only_lagged and A[1] == C[1]:
                continue

            # Check whether A is in SepSet(B_1, C), else the rule does not apply
            if not self._B_in_SepSet_AC(B_1, A, C):
                continue

            # Although we do not yet know whether the rule applies, we here determine the new form of the link from A to C if the rule
            # does apply
            link_AC = self._get_link(A, C)
            new_link_AC = "-" + link_AC[1] + ">"
            pair_key, new_link = self._get_pair_key_and_new_link(A, C, new_link_AC)

            # For the search of uncovered potentially directed paths from B_1 to C, determine the initial pattern as dictated by the link
            # from A to B_1
            first_link = self._get_link(A, B_1)
            if self._match_link(pattern='o*o', link=first_link):
                initial_allowed_patterns = ['-*>', 'o*>', 'o*o']
            elif self._match_link(pattern='o*>', link=first_link) or self._match_link(pattern='-*>', link=first_link):
                initial_allowed_patterns = ['-*>']
            
            # Return all uncovered potentially directed paths from B_1 to C
            #uncovered_pd_paths =  self._find_potentially_directed_paths(B_1, C, initial_allowed_patterns, return_if_any_path_found = False,
            # uncovered=True, reduce_allowed_patterns=True, max_length = np.inf)

            # Find all uncovered potentially directed paths from B_1 to C
            uncovered_pd_paths = self._get_potentially_directed_uncovered_paths(B_1, C, initial_allowed_patterns)

            # Run through all of these paths and check i) whether the node adjacent to B_1 is non-adjacent to A, ii) whether condition iv) of
            # the rule antecedent is true. If there is any such path, then the link can be oriented
            for upd_path in uncovered_pd_paths:

                # Is the node adjacent to B_1 non-adjacent to A (this implies that there are at least three nodes on the path, because else the
                # node adjacent to B_1 is C) and is A not part of the path?
                if len(upd_path) < 3 or A in upd_path or self._get_link(A, upd_path[1]) != "":
                    continue

                # If the link from A to B_1 is into B_1, condition iv) is true
                if first_link[2] == ">":
                    # Mark the link from A to C for orientation, break the for loop to continue with the next triple
                    out.append((pair_key, new_link))
                    break

                # If the link from A to B_1 is not in B_1, we need to check whether B_1 is in SepSet(A, X) where X is the node on upd_path next
                # to B_1
                if not self._B_in_SepSet_AC(A, B_1, upd_path[1]):
                    # Continue with the next upd_path
                    continue

                # Now check whether rule iv) for all triples on upd_path
                path_qualifies = True
                for i in range(len(upd_path) - 2):
                    # We consider the unshielded triples upd_path[i] - upd_path[i+1] - upd_path[i+2]

                    # If the link between upd_path[i] and upd_path[i+1] is into the latter, condition iv) is true
                    left_link = self._get_link(upd_path[i], upd_path[i+1])
                    if left_link[2] == ">":
                        # The path qualifies, break the inner for loop
                        break

                    # If not, then we need to continue with checking whether upd_path[i+1] in SepSet(upd_path[i+1], upd_path[i+2])
                    if not self._B_in_SepSet_AC(upd_path[i], upd_path[i+1], upd_path[i+2]):
                        # The path does not qualifying, break the inner for loop
                        path_qualifies = False
                        break

                # The path qualifies, mark the edge from A to C for orientation and break the outer for loop to continue with the next triple
                if path_qualifies:
                    out.append((pair_key, new_link))
                    break

                # The path does not qualify, continue with the next upd_path

            # end for upd_path in uncovered_pd_paths
        # end for (B_1, A, C) in all_appropriate_triples

        # Return the output list
        return out


    def _apply_ER10(self, only_lagged):
        """Return all orientations implied by orientation rule R10^prime"""

        # Build the output list
        out = []

        # Find all triples A o--> C <-- P_C
        all_appropriate_triples = set(self._find_triples(pattern_ij='o*>', pattern_jk='<*-', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='o*>', pattern_jk='<*-', pattern_ik='***')))

        # Collect all triples for the given pair (A, C)
        triple_sorting_dict = {}
        for (A, C, P_C) in all_appropriate_triples:
            if triple_sorting_dict.get((A, C)) is None:
                triple_sorting_dict[(A, C)] = [P_C]
            else:
                triple_sorting_dict[(A, C)].append(P_C)


        # Run through all (A, C) pairs
        for (A, C) in triple_sorting_dict.keys():

            if only_lagged and A[1] == C[1]:
                continue

            # Find all uncovered potentially directed paths from A to C through any of the P_C nodes
            relevant_paths = []
            for P_C in triple_sorting_dict[(A, C)]:
                for upd_path in self._get_potentially_directed_uncovered_paths(A, P_C, ['-*>', 'o*>', 'o*o']):

                    # Run through all of these paths and check i) whether the second to last element is not adjacent to C (this requires it to
                    # have a least three nodes, because else the second to last element would be A) and ii) whether the left edge of any 3-node
                    # sub-path is into the middle nor or, if not, whether the middle node is in the separating set of the two end-point nodes
                    # (of the 3-node) sub-path and iii) whether C is not element of the path. If path meets these conditions, add its second node
                    # (the adjacent to A) to the set second_nodes

                    if len(upd_path) < 3 or C in upd_path or self._get_link(upd_path[-2], C) != "":
                        continue

                    upd_path.append(C)

                    path_qualifies = True
                    for i in range(len(upd_path) - 2):
                        # We consider the unshielded triples upd_path[i] - upd_path[i+1] - upd_path[i+2]

                        # If the link between upd_path[i] and upd_path[i+1] is into the latter, the path qualifies
                        left_link = self._get_link(upd_path[i], upd_path[i+1])
                        if left_link[2] == ">":
                            # The path qualifies, break the inner for loop
                            break

                        # If not, then we need to continue with checking whether upd_path[i+1] in SepSet(upd_path[i+1], upd_path[i+2])
                        if not self._B_in_SepSet_AC(upd_path[i], upd_path[i+1], upd_path[i+2]):
                            # The path does not qualify, break the inner for loop
                            path_qualifies = False
                            break

                    # The path qualifies, add upd_path[i] to second_nodes and continue with the next upd_path
                    if path_qualifies:
                        relevant_paths.append(upd_path)

                # The path does not qualify, continue with the next upd_path

                # end for path in self._get_potentially_directed_uncovered_paths(A, P_C, ['-*>', 'o*>', 'o*o'])
            # end for P_C in triple_sorting_dict[(A, C)]

            # Find all second nodes on the relevant paths
            second_nodes = list({path[1] for path in relevant_paths})

            # Check whether there is any pair of non-adjacent nodes in second_nodes, such that A is in their separating set. If yes, mark the link
            # from A to C for orientation
            for i, j in product(range(len(second_nodes)), range(len(second_nodes))):

                if i < j and self._get_link(second_nodes[i], second_nodes[j]) == "" and self._B_in_SepSet_AC(second_nodes[i], A, second_nodes[j]):
                    # Append new link and break the for loop
                    link_AC = self._get_link(A, C)
                    new_link_AC = "-" + link_AC[1] + ">"
                    out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))
                    break

        # end for (A, C) in triple_sorting_dict.keys()

        # Return the output list
        return out


    def _apply_ER00a(self, only_lagged):
        """Return all orientations implied by orientation rule R0^prime a"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik='')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # Unpack A, B, C
            (i, lag_i) = A
            (j, lag_j) = B
            (k, lag_k) = C

            if only_lagged and (A[1] == B[1] or B[1] == C[1]):
                continue

            # Get all weakly minimal separating sets in SepSet(A, C)
            # Remark: The non weakly minimal separating sets may be larger, that's why we disfavor them
            sepsets = self._get_sepsets(A, C)
            sepsets = {Z for (Z, status) in sepsets if status == "wm"}

            ###################################################################################
            ### Part 1) of the rule ###########################################################

            remove_AB = False
            link_AB = self._get_link(A, B)

            # i) Middle mark must not be "x" or "-"
            if link_AB[1] not in ['-', 'x']:
                # Test A indep B given union(SepSet(A, C), intersection(def-anc(B), adj(B))) setminus{A, B} setminus{future of both A and B}

                # Conditioning on parents
                Z_add = self._get_parents(A, B).difference({A, B})

                # Shift the lags appropriately
                if lag_i <= lag_j:
                    X = (i, lag_i - lag_j) # A shifted
                    Y = (j, 0) # B shifted
                    delta_lag = lag_j

                else:
                    X = (j, lag_j - lag_i) # B shifted
                    Y = (i, 0) # A shifted
                    delta_lag = lag_i

                # Run through all weakly minimal separating sets of A and C
                for Z in sepsets:      

                    # Construct the conditioning set to test
                    Z_test = Z.union(Z_add).difference({A, B})
                    Z_test = {(var, lag - delta_lag) for (var, lag) in Z_test if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    Z_add2 = {(var, lag - delta_lag) for (var, lag) in Z_add.difference({A, B}) if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}

                    # Test conditional independence of X and Y given Z
                    # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), tau_max = self.tau_max)
                    val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), 
                        tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                    if self.verbosity >= 2:
                        # print("ER00a(part1):    %s _|_ %s  |  Z_test = %s: val = %.2f / pval = % .4f" %
                        #     (X, Y, ' '.join([str(z) for z in list(Z_test)]), val, pval))
                        print("ER00a(part1):    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in Z_add2]), ' '.join([str(z) for z in Z_test]), val, pval))

                    # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic values and
                    # conditioning set cardinalities
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_test))

                    # Check whether test result was significant
                    if not dependent: # pval > self.pc_alpha:

                        # Mark the edge from X to Y for removal and save sepset
                        remove_AB = True
                        self._save_sepset(X, Y, (frozenset(Z_test), "nwm"))

                if remove_AB:

                    # Remember the edge for removal
                    pair_key, new_link = self._get_pair_key_and_new_link(A, B, "")
                    out.append((pair_key, new_link))

            ###################################################################################
            ### Part 2) of the rule ###########################################################

            remove_CB = False
            link_CB = self._get_link(C, B)

            # i) Middle mark must not be "x" or "-"
            if link_CB[1] not in ['-', 'x']:
                # Test C indep B given union(SepSet(A, C), intersection(def-anc(B), adj(B))) setminus{A, B} setminus{future of both C and B}

                # Conditioning on parents
                Z_add = self._get_parents(C, B).difference({C, B})

                # Shift the lags appropriately
                if lag_k <= lag_j:
                    X = (k, lag_k - lag_j)
                    Y = (j, 0)
                    delta_lag = lag_j
                else:
                    X = (j, lag_j - lag_k)
                    Y = (k, 0)
                    delta_lag = lag_k

                # Run through all weakly minimal separating sets of A and C
                for Z in sepsets:

                    # Construct the conditioning set to test
                    Z_test = Z.union(Z_add).difference({C, B})
                    Z_test = {(var, lag - delta_lag) for (var, lag) in Z_test if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    Z_add2 = {(var, lag - delta_lag) for (var, lag) in Z_add.difference({A, B}) if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}

                    # Test conditional independence of X and Y given Z
                    # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), tau_max = self.tau_max)
                    val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), 
                        tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                    if self.verbosity >= 2:
                        # print("ER00a(part2):    %s _|_ %s  |  Z_test = %s: val = %.2f / pval = % .4f" %
                        #     (X, Y, ' '.join([str(z) for z in list(Z_test)]), val, pval))
                        print("ER00a(part2):    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in Z_add2]), ' '.join([str(z) for z in Z_test]), val, pval))

                    # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic values and
                    # conditioning set cardinalities
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_test))

                    # Check whether test result was significant
                    if not dependent: # pval > self.pc_alpha:
                        
                        # Mark the edge from X to Y for removal and save sepset
                        remove_CB = True
                        self._save_sepset(X, Y, (frozenset(Z_test), "nwm"))

                if remove_CB:

                    # Remember the edge for removal
                    pair_key, new_link = self._get_pair_key_and_new_link(C, B, "")
                    out.append((pair_key, new_link))

            ###################################################################################
            ### Part 3) of the rule ###########################################################

            if remove_AB or remove_CB or link_AB[2] in ["-", "x"] or link_CB[2] in ["-", "x"] or link_AB[1] == "x" or link_CB[1] == "x" or (link_AB[2] == ">" and link_CB[2] == ">"):
                continue

            if self._B_not_in_SepSet_AC(A, B, C):

                # Prepare the new links and save them to the output
                if link_AB[2] != ">":
                    new_link_AB = link_AB[0] + link_AB[1] + ">"
                    out.append(self._get_pair_key_and_new_link(A, B, new_link_AB))

                new_link_CB = link_CB[0] + link_CB[1] + ">"
                if link_CB[2] != ">":
                    out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))

        # end for (A, B, C) in all_appropriate_triples

        # Return the output list
        return out


    def _apply_ER00b(self, only_lagged):
        """Return all orientations implied by orientation rule R0^prime b"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        triples_1 = self._find_triples(pattern_ij='**>', pattern_jk='o!+', pattern_ik='')
        triples_2 = [trip for trip in self._find_triples(pattern_ij='**>', pattern_jk='oR+', pattern_ik='') if self._is_smaller(trip[1], trip[2])]
        triples_3 = [trip for trip in self._find_triples(pattern_ij='**>', pattern_jk='oL+', pattern_ik='') if self._is_smaller(trip[2], trip[1])]
        all_appropriate_triples = set(triples_1).union(set(triples_2), set(triples_3))

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # Unpack A, B, C
            (i, lag_i) = A
            (j, lag_j) = B
            (k, lag_k) = C

            if only_lagged and A[1] == B[1]:
                continue

            # Get all weakly minimal separating sets in SepSet(A, C)
            # Remark: The non weakly minimal separating sets may be larger, that's why we disfavor them
            sepsets = self._get_sepsets(A, C)
            sepsets = {Z for (Z, status) in sepsets if status == "wm"}

            ###################################################################################
            ### Part 1) of the rule ###########################################################

            remove_AB = False
            link_AB = self._get_link(A, B)

            # i) Middle mark must not be "x" or "-"
            if link_AB[1] not in ['-', 'x']:
                # Test A indep B given union(SepSet(A, C), intersection(def-anc(B), adj(B))) setminus{A, B} setminus{future of both A and B}

                # Conditioning on parents
                Z_add = self._get_parents(A, B).difference({A, B})

                # Shift the lags appropriately
                if lag_i <= lag_j:
                    X = (i, lag_i - lag_j)
                    Y = (j, 0)
                    delta_lag = lag_j
                else:
                    X = (j, lag_j - lag_i)
                    Y = (i, 0)
                    delta_lag = lag_i

                # Run through all weakly minimal separating sets of A and C
                for Z in sepsets:

                    # Construct the conditioning set to test
                    Z_test = Z.union(Z_add).difference({A, B})
                    Z_test = {(var, lag - delta_lag) for (var, lag) in Z_test if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}
                    Z_add2 = {(var, lag - delta_lag) for (var, lag) in Z_add.difference({A, B}) if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}

                    # Test conditional independence of X and Y given Z
                    # val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), tau_max = self.tau_max)
                    val, pval, dependent = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), 
                        tau_max = self.tau_max, alpha_or_thres=self.pc_alpha)

                    if self.verbosity >= 2:
                        # print("ER00b:    %s _|_ %s  |  Z_test = %s: val = %.2f / pval = % .4f" %
                        #     (X, Y, ' '.join([str(z) for z in list(Z_test)]), val, pval))
                        print("ER00b:    %s _|_ %s  |  Z_add = %s, Z = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in Z_add2]), ' '.join([str(z) for z in Z_test]), val, pval))

                    # Accordingly update dictionaries that keep track of the maximal p-value and the corresponding test statistic values and
                    # conditioning set cardinalities
                    self._update_pval_val_card_dicts(X, Y, pval, val, len(Z_test))

                    # Check whether test result was significant
                    if not dependent: # pval > self.pc_alpha:

                        # Mark the edge from X to Y for removal and save sepset
                        remove_AB = True
                        self._save_sepset(X, Y, (frozenset(Z_test), "nwm"))

                if remove_AB:
                    # Remember the edge for removal
                    pair_key, new_link = self._get_pair_key_and_new_link(A, B, "")
                    out.append((pair_key, new_link))

            ###################################################################################
            ### Part 2) of the rule ###########################################################

            if only_lagged and B[1] == C[1]:
                continue

            if remove_AB or link_AB[1] == "x":
                continue

            if self._B_not_in_SepSet_AC(A, B, C):

                # Prepare the new link and save it to the output
                link_CB = self._get_link(C, B)
                new_link_CB = link_CB[0] + link_CB[1] + ">"
                out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))

        # end for (A, B, C) in all_appropriate_triples

        # Return the output list
        return out


    def _apply_ER00c(self, only_lagged):
        """Return all orientations implied by orientation rule R0^prime c"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        triples_1 = self._find_triples(pattern_ij='*-*', pattern_jk='o!+', pattern_ik='')
        triples_2 = [trip for trip in self._find_triples(pattern_ij='*-*', pattern_jk='oR+', pattern_ik='') if self._is_smaller(trip[1], trip[2])]
        triples_3 = [trip for trip in self._find_triples(pattern_ij='*-*', pattern_jk='oL+', pattern_ik='')
                        if self._is_smaller(trip[2], trip[1])]
        all_appropriate_triples = set(triples_1).union(set(triples_2), set(triples_3))

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if only_lagged and  B[1] == C[1]:
                continue

            # Check whether the rule applies
            if self._B_not_in_SepSet_AC(A, B, C):

                # Prepare the new link and append it to the output
                link_CB = self._get_link(C, B)
                new_link_CB = link_CB[0] + link_CB[1] + ">"
                out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))

        # end for (A, B, C) in all_appropriate_triples

        # Return the output list
        return out


    def _apply_ER00d(self, only_lagged):
        """Return all orientations implied by orientation rule R0^prime d"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        triples_1 = self._find_triples(pattern_ij='*-o', pattern_jk='o-*', pattern_ik='')
        triples_2 = self._find_triples(pattern_ij='*->', pattern_jk='o-*', pattern_ik='')
        all_appropriate_triples = set(triples_1).union(set(triples_2))

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if only_lagged and (A[1] == B[1] and B[1] == C[1]):
                continue

            # Check whether the rule applies
            if self._B_not_in_SepSet_AC(A, B, C):
                # Prepare the new links and append them to the output

                # From C to B
                if not only_lagged or B[1] != C[1]:
                    link_CB = self._get_link(C, B)
                    new_link_CB = link_CB[0] + link_CB[1] + ">"
                    out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))

                # If needed, also fromA to B
                link_AB = self._get_link(A, B)
                if (not only_lagged or A[1] != B[1]) and link_AB[2] == "o":
                    new_link_AB = link_AB[0] + link_AB[1] + ">"
                    out.append(self._get_pair_key_and_new_link(A, B, new_link_AB))

        # end for (A, B, C) in all_appropriate_triples

        # Return the output list
        return out

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _print_graph_dict(self):
        """Print all links in graph_dict"""

        for j in range(self.N):
            for ((i, lag_i), link) in self.graph_dict[j].items():
                if len(link) > 0 and (lag_i < 0 or i < j):
                    print("({},{:2}) {} {}".format(i, lag_i, link, (j, 0)))


    def _get_link(self, A, B):
        """Get the current link from node A to B"""

        (var_A, lag_A) = A
        (var_B, lag_B) = B

        if abs(lag_A - lag_B) > self.tau_max:
            return ""
        elif lag_A <= lag_B:
            return self.graph_dict[var_B][(var_A, lag_A - lag_B)]
        else:
            return self._reverse_link(self.graph_dict[var_A][(var_B, lag_B - lag_A)])


    def _get_non_future_adj(self, node_list):
        """Return all non-future adjacencies of all nodes in node_list"""

        # Build the output starting from an empty set
        out = set()

        # For each node W in node_list ...
        for A in node_list:
            # Unpack A
            (var_A, lag_A) = A
            # Add all (current) non-future adjacencies of A to the set out
            out = out.union({(var, lag + lag_A) for ((var, lag), link) in self.graph_dict[var_A].items() if len(link) > 0 and lag + lag_A >= -self.tau_max})

        # Return the desired set
        return out

    def _update_pval_val_card_dicts(self, X, Y, pval, val, card):
        """If 'pval' is larger than the current maximal p-value across all previous independence tests for X and Y (stored in self.pval_max)
        then: Replace the current values stored in self.pval_max, self.pval_max_val, self.pval_max_card respectively by 'pval', 'val', and 'card'."""

        if X[1] < 0 or X[0] < Y[0]:
            if pval > self.pval_max[Y[0]][X]:
                self.pval_max[Y[0]][X] = pval
                self.pval_max_val[Y[0]][X] = val
                self.pval_max_card[Y[0]][X] = card
        else:
            if pval > self.pval_max[X[0]][Y]:
                self.pval_max[X[0]][Y] = pval
                self.pval_max_val[X[0]][Y] = val
                self.pval_max_card[X[0]][Y] = card

    def _save_sepset(self, X, Y, Z):
        """Save Z as separating sets of X and Y. Y is assumed to be at lag 0"""

        # Unpack X and Y
        (i, lag_i) = X
        (j, lag_j) = Y

        assert lag_j == 0

        # Save the sepset
        if lag_i < 0 or i < j:
            self.sepsets[j][X].add(Z)
        else:
            self.sepsets[i][Y].add(Z)

    def _reverse_link(self, link):
        """Reverse a given link, taking care to replace > with < and vice versa"""

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


    def _write_link(self, A, B, new_link, verbosity = 0):
        """Write the information that the link from node A to node B takes the form of new_link into self.graph_dict. Neither is it assumed
        that at least of the nodes is at lag 0, nor must A be before B. If A and B are contemporaneous, also the link from B to A is written
        as the reverse of new_link"""

        # Unpack A and B
        (var_A, lag_A) = A
        (var_B, lag_B) = B

        # Write the link from A to B
        if lag_A < lag_B:

            if verbosity >= 1:
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_A, lag_A - lag_B, self.graph_dict[var_B][(var_A, lag_A - lag_B)], var_B, 0, var_A, lag_A - lag_B, new_link, var_B, 0))
                #print("Replacing {:3} from ({},{:2}) to {} with {:3}".format(self.graph_dict[var_B][(var_A, lag_A - lag_B)], var_A, lag_A - lag_B, (var_B, 0), new_link))

            self.graph_dict[var_B][(var_A, lag_A - lag_B)] = new_link


        elif lag_A == lag_B:

            if verbosity >= 1:
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_A, lag_A - lag_B, self.graph_dict[var_B][(var_A, 0)], var_B, 0, var_A, 0, new_link, var_B, 0))
                #print("Replacing {:3} from ({},{:2}) to {} with {:3}".format(self.graph_dict[var_B][(var_A, 0)], var_A, 0, (var_B, 0), new_link))
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_B, 0, self.graph_dict[var_A][(var_B, 0)], var_A, 0, var_B, 0, self._reverse_link(new_link), var_A, 0))
                #print("Replacing {:3} from ({},{:2}) to {} with {:3}".format(self.graph_dict[var_A][(var_B, 0)], var_B, 0, (var_A, 0), self._reverse_link(new_link)))

            self.graph_dict[var_B][(var_A, 0)] = new_link
            self.graph_dict[var_A][(var_B, 0)] = self._reverse_link(new_link)

        else:

            if verbosity >= 1:
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_B, lag_B - lag_A, self.graph_dict[var_A][(var_B, lag_B - lag_A)], var_A, 0, var_B, lag_B - lag_A, self._reverse_link(new_link), var_A, 0))
                #print("Replacing {:3} from ({},{:2}) to {} with {:3}".format(self.graph_dict[var_A][(var_B, lag_B - lag_A)], var_B, lag_B - lag_A, (var_A, 0), self._reverse_link(new_link)))

            self.graph_dict[var_A][(var_B, lag_B - lag_A)] = self._reverse_link(new_link)


    def _get_sepsets(self, A, B):
        """For two non-adjacent nodes, get the their separating stored in self.sepsets."""

        (var_A, lag_A) = A
        (var_B, lag_B) = B

        def _shift(Z, lag_B):
            return frozenset([(var, lag + lag_B) for (var, lag) in Z])

        if lag_A < lag_B:
            out = {(_shift(Z, lag_B), status) for (Z, status) in self.sepsets[var_B][(var_A, lag_A - lag_B)]}
        elif lag_A > lag_B:
            out = {(_shift(Z, lag_A), status) for (Z, status) in self.sepsets[var_A][(var_B, lag_B - lag_A)]}
        else:
            out = {(_shift(Z, lag_A), status) for (Z, status) in self.sepsets[max(var_A, var_B)][(min(var_A, var_B), 0)]}

        return out


    def _initialize_full_graph(self):
        """
        The function _get_na_pds_t() needs to know the future adjacencies of a given node, not only the non-future adjacencies that are
        stored in self.graph_dict. To aid this, this function initializes the dictionary graph_full_dict:

        self.graph_full_dict[j][(i, -tau_i)] contains all adjacencies of (j, 0), in particular those for which tau_i < 0.
        """

        # Build from an empty nested dictionary
        self.graph_full_dict = {j: {} for j in range(self.N)}

        # Run through the entire nested dictionary self.graph_dict
        for j in range(self.N):
            for ((var, lag), link) in self.graph_dict[j].items():

                if link != "":
                    # Add non-future adjacencies
                    self.graph_full_dict[j][(var, lag)] = link

                    # Add the future adjacencies 
                    if lag < 0:
                        self.graph_full_dict[var][(j, -lag)] = self._reverse_link(link)

        # Return nothing
        return None


    def _get_pair_key_and_new_link(self, A, B, link_AB):
        """The link from A to B takes the form link_AB. Bring this information into a form appropriate for the output of rule applications"""

        (var_A, lag_A) = A
        (var_B, lag_B) = B

        if lag_A <= lag_B:
            return ((var_A, var_B, lag_A - lag_B), link_AB)
        elif lag_A > lag_B:
            return ((var_B, var_A, lag_B - lag_A), self._reverse_link(link_AB))


    def _match_link(self, pattern, link):
        """Matches pattern including wildcards with link."""
        
        if pattern == '' or link == '':
            return True if pattern == link else False
        else:
            left_mark, middle_mark, right_mark = pattern
            if left_mark != '*':
                if left_mark == '+':
                    if link[0] not in ['<', 'o']: return False
                else:
                    if link[0] != left_mark: return False

            if right_mark != '*':
                if right_mark == '+':
                    if link[2] not in ['>', 'o']: return False
                else:
                    if link[2] != right_mark: return False    
            
            if middle_mark != '*' and link[1] != middle_mark: return False    
                       
            return True


    def _dict2graph(self):
        """Convert self.graph_dict to graph array of shape (N, N, self.tau_max + 1)."""

        graph = np.zeros((self.N, self.N, self.tau_max + 1), dtype='U3')
        for j in range(self.N):
            for adj in self.graph_dict[j]:
                (i, lag_i) = adj
                graph[i, j, abs(lag_i)] = self.graph_dict[j][adj]

        return graph


    def _find_adj(self, graph, node, patterns, exclude=None, ignore_time_bounds=True):
        """Find adjacencies of node matching patterns."""
        
        # Setup
        i, lag_i = node
        if exclude is None: exclude = []
        if type(patterns) == str:
            patterns = [patterns]

        # Init
        adj = []
        # Find adjacencies going forward/contemp
        for k, lag_ik in zip(*np.where(graph[i,:,:])):  
            matches = [self._match_link(patt, graph[i, k, lag_ik]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i + lag_ik)
                if match not in adj and (k, lag_i + lag_ik) not in exclude and (-self.tau_max <= lag_i + lag_ik <= 0 or ignore_time_bounds):
                    adj.append(match)
        
        # Find adjacencies going backward/contemp
        for k, lag_ki in zip(*np.where(graph[:,i,:])):  
            matches = [self._match_link(self._reverse_link(patt), graph[k, i, lag_ki]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i - lag_ki)
                if match not in adj and (k, lag_i - lag_ki) not in exclude and (-self.tau_max <= lag_i - lag_ki <= 0 or ignore_time_bounds):
                    adj.append(match)
     
        return adj
        

    def _is_match(self, graph, X, Y, pattern_ij):
        """Check whether the link between X and Y agrees with pattern_ij"""

        (i, lag_i) = X
        (j, lag_j) = Y
        tauij = lag_j - lag_i
        if abs(tauij) >= graph.shape[2]:
            return False
        return ((tauij >= 0 and self._match_link(pattern_ij, graph[i, j, tauij])) or
               (tauij < 0 and self._match_link(self._reverse_link(pattern_ij), graph[j, i, abs(tauij)])))


    def _find_triples(self, pattern_ij, pattern_jk, pattern_ik):
        """Find triples (i, lag_i), (j, lag_j), (k, lag_k) that match patterns."""
  
        # Graph as array makes it easier to search forward AND backward in time
        graph = self._dict2graph()

        # print(graph[:,:,0])
        # print(graph[:,:,1])
        # print("matching ", pattern_ij, pattern_jk, pattern_ik)

        matched_triples = []
                
        for i in range(self.N):
            # Set lag_i = 0 without loss of generality, will be adjusted at end
            lag_i = 0
            adjacencies_i = self._find_adj(graph, (i, lag_i), pattern_ij)
            # print(i, adjacencies_i)
            for (j, lag_j) in adjacencies_i:

                adjacencies_j = self._find_adj(graph, (j, lag_j), pattern_jk,
                                          exclude=[(i, lag_i)])
                # print(j, adjacencies_j)
                for (k, lag_k) in adjacencies_j:
                    if self._is_match(graph, (i, lag_i), (k, lag_k), pattern_ik):                            
                        # Now use stationarity and shift triple such that the right-most
                        # node (on a line t=..., -2, -1, 0, 1, 2, ...) is at lag 0
                        righmost_lag = max(lag_i, lag_j, lag_k)
                        match = ((i, lag_i - righmost_lag), 
                                 (j, lag_j - righmost_lag),
                                 (k, lag_k - righmost_lag))
                        largest_lag = min(lag_i - righmost_lag, lag_j - righmost_lag, lag_k - righmost_lag)
                        if match not in matched_triples and \
                            -self.tau_max <= largest_lag <= 0:
                            matched_triples.append(match)                       
                
        return matched_triples  


    def _find_quadruples(self, pattern_ij, pattern_jk, pattern_ik, 
                               pattern_il, pattern_jl, pattern_kl):
        """Find quadruples (i, lag_i), (j, lag_j), (k, lag_k), (l, lag_l) that match patterns."""
  
        # We assume this later
        assert pattern_il != ''

        # Graph as array makes it easier to search forward AND backward in time
        graph = self._dict2graph()

        matched_quadruples = []
                
        # First get triple ijk
        ijk_triples = self._find_triples(pattern_ij, pattern_jk, pattern_ik)

        for triple in ijk_triples:
            # Unpack triple
            (i, lag_i), (j, lag_j), (k, lag_k) = triple

            # Search through adjacencies
            adjacencies = set(self._find_adj(graph, (i, lag_i), pattern_il,
                                          exclude=[(j, lag_j), (k, lag_k)]))
            if pattern_jl != '':
                adjacencies = adjacencies.intersection(set(
                                self._find_adj(graph, (j, lag_j), pattern_jl,
                                          exclude=[(i, lag_i), (k, lag_k)])))
            else:
                adjacencies = set([adj for adj in adjacencies 
                                if self._is_match(graph, (j, lag_j), adj, '')])

            if pattern_kl != '':
                adjacencies = adjacencies.intersection(set(
                                self._find_adj(graph, (k, lag_k), pattern_kl,
                                          exclude=[(i, lag_i), (j, lag_j)])))
            else:
                adjacencies = set([adj for adj in adjacencies 
                                if self._is_match(graph, (k, lag_k), adj, '')])

            for adj in adjacencies:
                (l, lag_l) = adj
                    
                # Now use stationarity and shift quadruple such that the right-most
                # node (on a line t=..., -2, -1, 0, 1, 2, ...) is at lag 0
                righmost_lag = max(lag_i, lag_j, lag_k, lag_l)
                match = ((i, lag_i - righmost_lag), 
                         (j, lag_j - righmost_lag),
                         (k, lag_k - righmost_lag),
                         (l, lag_l - righmost_lag),
                         )
                largest_lag = min(lag_i - righmost_lag, 
                                  lag_j - righmost_lag, 
                                  lag_k - righmost_lag,
                                  lag_l - righmost_lag,
                                  )
                if match not in matched_quadruples and \
                    -self.tau_max <= largest_lag <= 0:
                    matched_quadruples.append(match)                       
                
        return matched_quadruples 


    def _get_R4_discriminating_paths(self, triple, max_length = np.inf):
        """Find all discriminating paths starting from triple"""

        def _search(path_taken, max_length):

            # Get the last visited node and its link to Y
            last_node = path_taken[-1]
            link_to_Y = self._get_link(last_node, path_taken[0])

            # Base Case: If the current path is a discriminating path, return it as single entry of a list
            if len(path_taken) > 3 and link_to_Y == "":
                return [path_taken]            

            # If the current path is not a discriminating path, continue the path
            paths = []

            if self._get_link(last_node, path_taken[-2])[0] == "<" and link_to_Y == "-->" and len(path_taken) < max_length:

                # Search through all adjacencies of the last node
                for (var, lag) in self.graph_full_dict[last_node[0]].keys():

                    # Build the next node and get its link to the previous
                    next_node = (var, lag + last_node[1])
                    next_link = self._get_link(next_node, last_node)

                    # Check whether this node can be visited
                    if next_node[1] <= 0 and next_node[1] >= -self.tau_max and next_node not in path_taken and self._match_link("*->", next_link):

                        # Recursive call
                        paths.extend(_search(path_taken[:] + [next_node], max_length))

            # Return the list of discriminating paths
            return paths

        # Unpack the triple
        (W, V, Y) = triple

        # Return all discriminating paths starting at this triple
        return _search([Y, V, W], max_length)


    def _get_potentially_directed_uncovered_paths(self, start_node, end_node, initial_allowed_patterns):
        """Find all potentiall directed uncoverged paths from start_node to end_node whose first link takes one the forms specified by
        initial_allowed_patters"""

        assert start_node != end_node

        # Function for recursive search of potentially directed uncovered paths
        def _search(end_node, path_taken, allowed_patterns):

            # List for outputting potentially directed uncovered paths
            paths = []

            # The last visited note becomes the new start_node
            start_node = path_taken[-1]

            # Base case: End node has been reached
            if start_node == end_node:
                paths.append(path_taken)

            # Recursive build case
            else:
                # Run through the adjacencies of start_node
                #for next_node in self.graph_full_dict[start_node[0]]:
                for (var, lag) in self.graph_full_dict[start_node[0]].keys():

                    next_node = (var, lag + start_node[1])

                    # Consider only nodes that ...
                    # ... are within the allowed time frame
                    if next_node[1] < -self.tau_max or next_node[1] > 0:
                        continue
                    # ... have not been visited yet
                    if next_node in path_taken:
                        continue
                    # ... are non-adjacent to the node before start_node
                    if len(path_taken) >= 2 and self._get_link(path_taken[-2], next_node) != "":
                        continue
                    # ... whose link with start_node matches one of the allowed patters
                    link = self._get_link(start_node, next_node)
                    if not any([self._match_link(pattern = pattern, link = link) for pattern in allowed_patterns]):
                        continue

                    # Determine the allowed patters for the next recursive call
                    if self._match_link(pattern='o*o', link=link):
                        new_allowed_patters = ["o*o", "o*>", "-*>"]
                    elif self._match_link(pattern='o*>', link=link) or self._match_link(pattern='-*>', link=link):
                        new_allowed_patters = ["-*>"]

                    # Determine the new path taken
                    new_path_taken = path_taken[:] + [next_node]

                    # Recursive call
                    paths.extend(_search(end_node, new_path_taken, new_allowed_patters))

            # Output list of potentially directed uncovered paths
            return paths

        # end def _search(end_node, path_taken, allowed_patterns)

        # Output potentially directed uncovered paths
        paths = _search(end_node, [start_node], initial_allowed_patterns)
        return [path for path in paths if len(path) > 2]


    def _sort_search_set(self, search_set, reference_node):
        """Sort the nodes in search_set by their values in self.pval_max_val with respect to the reference_node. Nodes with higher absolute
        values appear earlier"""

        sort_by_potential_minus_infs = [self._get_pval_max_val(node, reference_node) for node in search_set]
        sort_by = [(np.abs(value) if value != -np.inf else 0) for value in sort_by_potential_minus_infs]

        return [x for _, x in sorted(zip(sort_by, search_set), reverse = True)]

    def _get_pval_max_val(self, X, Y):
        """Return the test statistic value of that independence test for X and Y which, among all such tests, has the largest p-value."""

        if X[1] < 0 or X[0] < Y[0]:
            return self.pval_max_val[Y[0]][X]
        else:
            return self.pval_max_val[X[0]][Y]    

    def _delete_sepsets(self, X, Y):
        """Delete all separating sets of X and Y. Y is assumed to be at lag 0"""

        # Unpack X and Y
        (i, lag_i) = X
        (j, lag_j) = Y

        assert lag_j == 0

        # Save the sepset
        if lag_i < 0 or i < j:
            self.sepsets[j][X] = set()
        else:
            self.sepsets[i][Y] = set()


if __name__ == '__main__':

    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as pp
    from tigramite.toymodels import structural_causal_processes as toys
    import tigramite.plotting as tp
    from matplotlib import pyplot as plt

    # Example process to play around with
    # Each key refers to a variable and the incoming links are supplied
    # as a list of format [((var, -lag), coeff, function), ...]
    def lin_f(x): return x
    def nonlin_f(x): return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))

    links = {0: [((0, -1), 0.9, lin_f), ((3, -1), -0.6, lin_f)],
             1: [((1, -1), 0.9, lin_f), ((3, -1), 0.6, lin_f)],
             2: [((2, -1), 0.9, lin_f), ((1, -1), 0.6, lin_f)],
             3: [],
             }

    full_data, nonstat = toys.structural_causal_process(links,
                        T=1000, seed=7)
    
    # We now remove variable 3 which plays the role of a hidden confounder
    data = full_data[:, [0, 1, 2]]

    # Data must be array of shape (time, variables)
    print(data.shape)
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr(significance='fixed_thres')
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = lpcmci.run_lpcmci(tau_max=2, pc_alpha=0.01)

    # # For a proper causal interpretation of the graph see the paper!
    # print(results['graph'])
    # tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'])
    # plt.show()

    # results = lpcmci.run_sliding_window_of(
    #     window_step=499, window_length=500,
    #     method='run_lpcmci', method_args={'tau_max':1})
