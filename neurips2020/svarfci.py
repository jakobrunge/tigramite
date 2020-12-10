import numpy as np
from itertools import product, combinations
import os

class SVARFCI():
    r"""
    This class implements the SVAR-FCI algorithm introduced in:

        Malinsky, D. and Spirtes, P. (2018). Causal Structure Learning from Multivariate Time Series in Settings with Unmeasured Confounding. In Le, T. D., Zhang, K., Kıcıman, E., Hyvärinen, A., and Liu, L., editors, Proceedings of 2018 ACM SIGKDD Workshop on Causal Disocvery, volume 92 of Proceedings of Machine Learning Research, pages 23–47, London, UK. PMLR.

    Our implementation applies several modifications:
        1) It assumes the absence of selection variables.
        2) It guarantees order-independence by i) using the majority rule to decide whether a given node is in a given separating set and ii) applying a rule to the entire graph and resolving potential conflicts among the proposed orientations by means of the conflict mark 'x' before modifing the graph.
        3) It allows for the following conclusion: If X^i_{t-\tau} and X^j_t for \tau > 0 are not m-separated by any subset of D-Sep(X^j_t, X^i_{t-\tau}, \mathcal{M}(\mathcal{G})) then these variables are adjacent in \mathcal{M}(\mathcal{G}). In particular, this conclusions does not require that X^i_{t-\tau} and X^j_t are moreover not m-separated by any subset of D-Sep(X^i_{t-\tau}, X^j_t, \mathcal{M}(\mathcal{G}))
        4) Several control parameters apply further modifications, see below.

    Parameters passed to the constructor:
    - dataframe:
        Tigramite dataframe object that contains the the time series dataset \bold{X}
    - cond_ind_test:
        A conditional independence test object that specifies which conditional independence test CI is to be used

    Parameters passed to self.run_svarfci():
    - tau_max:
        The maximum considered time lag tau_max
    - pc_alpha:
        The significance level \alpha of conditional independence tests
    - max_cond_px:
        Consider a pair of variables (X^i_{t-\tau}, X^j_t) with \tau > 0. In the first removal phase (here this is self._run_pc_removal_phase()), the algorithm does not test for conditional independence given subsets of X^i_{t-\tau} of cardinality higher than max_cond_px. In the second removal phase (here this is self._run_dsep_removal_phase()), the algorithm does not test for conditional independence given subsets of pds_t(X^i_{t-\tau}, X^j_t) of cardinality higher than max_cond_px.
    - max_p_global:
        Restricts all conditional independence tests to conditioning sets with cardinality smaller or equal to max_p_global
    - max_p_dsep:
        Restricts all conditional independence tests in the second removal phase (here this is self._run_dsep_removal_phase()) to conditioning sets with cardinality smaller or equal to max_p_global
    - max_q_global:
        For each ordered pair (X^i_{t-\tau}, X^j_t) of adjacent variables and for each cardinality of the conditioning sets test at most max_q_global many conditioning sets (when summing over all tested cardinalities more than max_q_global tests may be made)
    - max_pds_set:
        In the second removal phase (here this is self._run_dsep_removal_phase()), the algorithm tests for conditional independence given subsets of the pds_t sets defined in the above reference. If for a given link the set pds_t(X^j_t, X^i_{t-\tau}) has more than max_pds_set many elements (or, if the link is also tested in the opposite directed, if pds_t(X^i_{t-\tau}, X^j_t) has more than max_pds_set elements), this link is not tested.
    - fix_all_edges_before_final_orientation:
        When one of the four previous parameters is not np.inf, the edge removals may terminate before we can be sure that all remaining edges are indeed part of the true PAG. However, soundness of the FCI orientation rules requires that they be applied only once the correct skeleton has been found. Therefore, the rules are only applied to those edges for which we are sure that they are part of the PAG. This can lead to quite uninformative results. If fix_all_edges_before_final_orientation is True, this precaution is overruled and the orientation rules are nevertheless applied to all edges.
    - verbosity:
        Controls the verbose output self.run_svarfci() and the function it calls.

    Return value of self.run_svarfci():
        The estimated graph in form of a link matrix. This is a numpy array of shape (self.N, self.N, self.tau_max + 1), where the entry array[i, j, \tau] is a string that visualizes the estimated link from X^i_{i-\tau} to X^j_t. For example, if array[0, 2, 1] = 'o->', then the estimated graph contains the link X^i_{t-1} o-> X^j_t. This numpy array is also saved as instance attribute self.graph. Note that self.N is the number of observed time series and self.tau_max the maximal considered time lag.

    A note on middle marks:
        In order to distinguish edges that are in the PAG for sure from edges that may not be in the PAG, we use the notion of middle marks that we introduced for LPCMCI. This becomes useful for the precaution discussed in the explanation of the parameter 'fix_all_edges_before_final_orientation', see above. In particular, we use the middle marks '?' and '' (empty). For convenience (to have strings of the same lengths) we here internally denote the empty middle mark by '-'. For post-processing purposes all middle marks are nevertheless set to the empty middle mark (here '-') in line 99, but if verbosity >= 1 a graph with the middle marks will be printed out before.
    
    A note on wildcards:
        The middle mark wildcard \ast and the edge mark wildcard are here represented as *, the edge mark wildcard \star as +
    """

    def __init__(self, dataframe, cond_ind_test):
        """Class constructor. Store:
                i)      data
                ii)     conditional independence test object
                iii)    some instance attributes"""

        # Save the time series data that the algorithm operates on
        self.dataframe = dataframe

        # Set the conditional independence test to be used
        self.cond_ind_test = cond_ind_test
        self.cond_ind_test.set_dataframe(self.dataframe)

        # Store the shape of the data in the T and N variables
        self.T, self.N = self.dataframe.values.shape


    def run_svarfci(self, 
                tau_max = 1, 
                pc_alpha = 0.05,
                max_cond_px = 0,
                max_p_global = np.inf,
                max_p_dsep = np.inf,
                max_q_global = np.inf,
                max_pds_set = np.inf,
                fix_all_edges_before_final_orientation = True,
                verbosity = 0):
        """Run the SVAR-FCI algorithm on the dataset and with the conditional independence test passed to the class constructor and with the options passed to this function."""

        # Step 0: Initializations
        self._initialize(tau_max, pc_alpha, max_cond_px, max_p_global, max_p_dsep, max_q_global, max_pds_set, fix_all_edges_before_final_orientation, verbosity)

        # Step 1: PC removal phase
        self._run_pc_removal_phase()

        # Step 2: D-Sep removal phase (including preliminary collider orientation phase)
        self._run_dsep_removal_phase()

        # Step 3: FCI orientation phase
        if self.fix_all_edges_before_final_orientation:
            self._fix_all_edges()

        self._run_fci_orientation_phase()   

        # Post processing
        if self.verbosity >= 1:
            print("Ambiguous triples", self.ambiguous_triples)
            print("Max pds set: {}\n".format(self.max_pds_set_found))

        self._fix_all_edges()
        self.graph = self._dict2graph()
        self.val_min_matrix = self._dict_to_matrix(self.val_min, self.tau_max, self.N, default = 0)
        self.cardinality_matrix = self._dict_to_matrix(self.max_cardinality, self.tau_max, self.N, default = 0)

        # Return the estimated graph
        return self.graph


    def _initialize(self,
                    tau_max,
                    pc_alpha,
                    max_cond_px,
                    max_p_global,
                    max_p_dsep,
                    max_q_global,
                    max_pds_set,
                    fix_all_edges_before_final_orientation,
                    verbosity):
        """Function for
            i)      saving the arguments passed to self.run_svarfci() as instance attributes
            ii)     initializing various memory variables for storing the current graph, sepsets etc.
            """

        # Save the arguments passed to self.run_svarfci()
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.max_cond_px = max_cond_px
        self.max_p_global = max_p_global
        self.max_p_dsep = max_p_dsep
        self.max_q_global = max_q_global
        self.max_pds_set = max_pds_set
        self.fix_all_edges_before_final_orientation = fix_all_edges_before_final_orientation
        self.verbosity = verbosity
        
        # Initialize the nested dictionary for storing the current graph.
        # Syntax: self.graph_dict[j][(i, -tau)] gives the string representing the link from X^i_{t-tau} to X^j_t
        self.graph_dict = {}
        for j in range(self.N):
            self.graph_dict[j] = {(i, 0): "o?o" for i in range(self.N) if j != i}
            self.graph_dict[j].update({(i, -tau): "o?>" for i in range(self.N) for tau in range(1, self.tau_max + 1)})

        # Initialize the nested dictionary for storing separating sets
        # Syntax: self.sepsets[j][(i, -tau)] stores separating sets of X^i_{t-tau} to X^j_t. For tau = 0, i < j.
        self.sepsets = {j: {(i, -tau): set() for i in range(self.N) for tau in range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}

        # Initialize dictionaries for storing known ancestorships, non-ancestorships, and ambiguous ancestorships
        # Syntax: self.def_ancs[j] contains the set of all known ancestors of X^j_t. Equivalently for the others
        self.def_ancs = {j: set() for j in range(self.N)}
        self.def_non_ancs = {j: set() for j in range(self.N)}
        self.ambiguous_ancestorships = {j: set() for j in range(self.N)}

        # Initialize nested dictionaries for saving the minimum test statistic among all conditional independence tests of a given pair of variables, the maximum p-values, as well as the maximal cardinality of the known separating sets.
        # Syntax: As for self.sepsets
        self.val_min = {j: {(i, -tau): float("inf") for i in range(self.N) for tau in
                                range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}
        self.pval_max = {j: {(i, -tau): 0 for i in range(self.N) for tau in
                                range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}
        self.max_cardinality = {j: {(i, -tau): 0 for i in range(self.N) for tau in
                                range(self.tau_max + 1) if (tau > 0 or i < j)} for j in range(self.N)}
                                                  
        # Initialize a nested dictionary for caching pds-sets
        # Syntax: As for self.sepsets
        self._pds_t = {(j, -tau_j): {} for j in range(self.N) for tau_j in range(self.tau_max + 1)}

        # Initialize a set for memorizing ambiguous triples
        self.ambiguous_triples = set()

        # Initialize a variable for remembering the maximal cardinality among all calculated pds-sets
        self.max_pds_set_found = -1

        ################################################################################################
        # Only relevant for use with oracle CI
        self._oracle = False
        ################################################################################################

        # Return
        return True


    def _run_pc_removal_phase(self):
        """Run the first removal phase of the FCI algorithm adapted to stationary time series. This is essentially the skeleton phase of the PC algorithm"""

        # Verbose output
        if self.verbosity >= 1:
            print("\n=======================================================")
            print("=======================================================")
            print("Starting preliminary removal phase")

        # Iterate until convergence
        # p_pc is the cardinality of the conditioning set
        p_pc = 0
        while True:

            ##########################################################################################################
            ### Run the next removal iteration #######################################################################

            # Verbose output
            if self.verbosity >= 1:
                if p_pc == 0:
                    print("\nStarting test phase\n")
                print("p = {}".format(p_pc))

            # Variable to check for convergence
            has_converged = True

            # Variable for keeping track of edges marked for removal
            to_remove = {j: {} for j in range(self.N)}

            # Iterate through all links
            for (i, j, lag_i) in product(range(self.N), range(self.N), range(-self.tau_max, 1)):

                # Decode the triple (i, j, lag_i) into pairs of variables (X, Y)
                X = (i, lag_i)
                Y = (j, 0)

                ######################################################################################################
                ### Exclusion of links ###############################################################################

                # Exclude the current link if ...
                # ... X = Y
                if lag_i == 0 and i == j:
                    continue
                # ... X > Y (so, in fact, we don't distinguish between both directions of the same edge)
                if self._is_smaller(Y, X):
                    continue

                # Get the current link from X to Y
                link = self._get_link(X, Y)

                # Also exclude the current link if ...
                # ... X and Y are not adjacent anymore
                if link == "":
                    continue
                
                ######################################################################################################
                ### Preparation of PC search sets ####################################################################

                # Search for separating sets in the non-future adjacencies of X, without X and Y themselves
                S_search_YX = self._get_non_future_adj([Y]).difference({X, Y})

                # Search for separating sets in the non-future adjacencies of Y, without X and Y themselves, always if X and Y are contemporaneous or if specified by self.max_cond_px
                test_X = True if (lag_i == 0 or (self.max_cond_px > 0 and self.max_cond_px >= p_pc)) else False
                if test_X:

                    S_search_XY = self._get_non_future_adj([X]).difference({X, Y})

                ######################################################################################################
                ### Check whether the link needs testing #############################################################

                # If there are less than p_pc elements in the search sets, the link does not need further testing
                if len(S_search_YX) < p_pc and (not test_X or len(S_search_XY) < p_pc):
                    continue

                # Force-quit while leep when p_pc exceeds the specified limits
                if p_pc > self.max_p_global:
                    continue

                # This link does need testing. Therfore, the algorithm has not converged yet
                has_converged = False
            
                ######################################################################################################
                ### Tests for conditional independence ###############################################################

                # If self.max_q_global is finite, the below for loop may be broken earlier. To still guarantee order independence, the set from which the potential separating sets are created is ordered in an order independent way. Here, the elements of S_search_YX are ordered according to their minimal test statistic with Y
                if not np.isinf(self.max_q_global):
                    S_search_YX = self._sort_search_set(S_search_YX, Y)

                # q_count counts the number of conditional independence tests made for subsets of S_search_YX
                q_count = 0

                # Run through all cardinality p_pc subsets of S_search_YX
                for Z in combinations(S_search_YX, p_pc):

                    # Stop testing if the number of tests exceeds the bound specified by self.max_q_global
                    q_count = q_count + 1
                    if q_count > self.max_q_global:
                        break

                    # Test conditional independence of X and Y given Z. Correspondingly updateself.val_min, self.pval_max, and self.cardinality 
                    val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                    if self.verbosity >= 2:
                        print("    %s _|_ %s  |  S_pc = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                    self._update_val_min(X, Y, val)
                    self._update_pval_max(X, Y, pval)
                    self._update_cardinality(X, Y, len(Z))

                    # Check whether the test result was significant
                    if pval > self.pc_alpha:

                        # Mark the edge from X to Y for removal, save Z as separating set
                        to_remove[Y[0]][X] = True
                        self._save_sepset(X, Y, (frozenset(Z), ""))
    
                        # Verbose output
                        if self.verbosity >= 1:
                            print("({},{:2}) {:11} {} given {}".format(X[0], X[1], "independent", Y, Z))

                        # Break the for loop
                        break
            
                # Run through all cardinality p_pc subsets of S_search_XY
                if test_X:

                    if not np.isinf(self.max_q_global):
                        S_search_XY = self._sort_search_set(S_search_XY, X)

                    q_count = 0
                    for Z in combinations(S_search_XY, p_pc):

                        q_count = q_count + 1
                        if q_count > self.max_q_global:
                            break

                        val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                        if self.verbosity >= 2:
                            print("    %s _|_ %s  |  S_pc = %s: val = %.2f / pval = % .4f" %
                                (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                        self._update_val_min(X, Y, val)
                        self._update_pval_max(X, Y, pval)
                        self._update_cardinality(X, Y, len(Z))

                        if pval > self.pc_alpha:

                            to_remove[Y[0]][X] = True
                            self._save_sepset(X, Y, (frozenset(Z), ""))
        
                            if self.verbosity >= 1:
                                print("({},{:2}) {:11} {} given {}".format(X[0], X[1], "independent", Y, Z))

                            break

            # end for (i, j, lag_i) in product(range(self.N), range(self.N), range(-self.tau_max, 1))

            ##########################################################################################################
            ### Remove edges marked for removal in to_remove #########################################################

            # Remove edges
            for j in range(self.N):
                for (i, lag_i) in to_remove[j].keys():

                    self._write_link((i, lag_i), (j, 0), "", verbosity = self.verbosity)

            # Verbose output
            if self.verbosity >= 1:
                    print("\nTest phase complete")

            ##########################################################################################################
            ### Check for convergence ################################################################################
        
            if has_converged:
                # If no link needed testing, this algorithm has converged. Therfore, break the while loop
                break
            else:
                # At least one link needed testing, this algorithm has not yet converged. Therefore, increase p_pc
                p_pc = p_pc + 1

        # end while True

        # Verbose output
        if self.verbosity >= 1:
            print("\nPreliminary removal phase complete")
            print("\nGraph:\n--------------------------------")
            self._print_graph_dict()
            print("--------------------------------")

        # Return
        return True


    def _run_dsep_removal_phase(self):
        """Run the second removal phase of the FCI algorithm, including the preliminary collider orientation that is necessary for determining pds-sets"""

        # Verbose output
        if self.verbosity >= 1:
            print("\n=======================================================")
            print("=======================================================")
            print("Starting final removal phase")

        # Make the preliminary orientations that are necessary for determining pds_t sets
        self._run_orientation_phase(rule_list = [["R-00-d"]], voting = "Majority-Preliminary")

        # Remember all edges that have not been fully tested due to self.max_pds_set, self.max_q_global or self.max_p_global
        self._cannot_fix = set()

        # Iterate until convergence
        # p_pc is the cardinality of the conditioning set
        p_pc = 0
        while True:

            ##########################################################################################################
            ### Run the next removal iteration #######################################################################

            # Verbose output
            if self.verbosity >= 1:
                if p_pc == 0:
                    print("\nStarting test phase\n")
                print("p = {}".format(p_pc))

            # Variable to check for convergence
            has_converged = True

            # Variable for keeping track of edges marked for removal
            to_remove = {j: {} for j in range(self.N)}

            # Iterate through all links
            for (i, j, lag_i) in product(range(self.N), range(self.N), range(-self.tau_max, 1)):

                # Decode the triple (i, j, lag_i) into pairs of variables (X, Y)
                X = (i, lag_i)
                Y = (j, 0)

                ######################################################################################################
                ### Exclusion of links ###############################################################################

                # Exclude the current link if ...
                # ... X = Y
                if lag_i == 0 and i == j:
                    continue
                # ... X > Y
                if self._is_smaller(Y, X):
                    continue

                # Get the current link
                link = self._get_link(X, Y)

                # Also exclude the current link if ...
                # ... X and Y are not adjacent anymore
                if link == "":
                    continue
                # ... X and Y are adjacent in the true MAG
                if link[1] == "-":
                    continue

                ######################################################################################################
                ### Preparation of PC search sets ####################################################################

                # Verbose output
                if self.verbosity >= 2:
                    print("_get_pds_t ")

                # Search for separating sets in pds_t(Y, X)
                S_search_YX = self._get_pds_t(Y, X)

                # Search for separating sets in pds_t(X, Y) always if X and Y are contemporaneous or if specified by self.max_cond_px
                test_X = True if (lag_i == 0 or (self.max_cond_px > 0 and self.max_cond_px >= p_pc)) else False
                if test_X:
                    S_search_XY = self._get_pds_t(X, Y)

                # If the pds_t sets exceed the specified bounds, do not test this link. Remember that the link has not been fully tested
                if len(S_search_YX) > self.max_pds_set or (test_X and len(S_search_XY) > self.max_pds_set):
                    self._cannot_fix.add((X, Y))
                    continue

                ######################################################################################################
                ### Check whether the link needs testing #############################################################

                # If there are less than p_pc elements in the search set(s), the link does not need further testing. X and Y are adjacent in the true MAG, unless the link has not been fully tested
                if len(S_search_YX) < p_pc and (not test_X or len(S_search_XY) < p_pc):
                    if (X, Y) not in self._cannot_fix:
                        self._write_link(X, Y, link[0] + "-" + link[2], verbosity = self.verbosity)
                    continue

                # Force-quit while leep when p_pc exceeds the specified limits
                if p_pc > self.max_p_global or p_pc > self.max_p_dsep:
                    continue

                # Since this link does need testing, the algorithm has not converged yet
                has_converged = False

                ######################################################################################################
                ### Tests for conditional independence ###############################################################

                # Verbose output
                if self.verbosity >= 1:
                    print("for S_pc in combinations(S_search_YX, p_pc)")
            
                # If self.max_q_global is finite, the below for loop may be broken earlier. To still guarantee order independence, the set from which the potential separating sets are created is ordered in an order independent way. Here, the elements of S_search_YX are ordered according to their minimal test statistic with Y
                if not np.isinf(self.max_q_global):
                    S_search_YX = self._sort_search_set(S_search_YX, Y)

                # q_count counts the number of conditional independence tests made for subsets of S_search_YX
                q_count = 0

                # Run through all cardinality p_pc subsets of S_search_YX
                for Z in combinations(S_search_YX, p_pc):

                    # Stop testing if the number of tests exceeds the bound specified by self.max_q_global. Remember that the link hast not been fully tested
                    q_count = q_count + 1
                    if q_count > self.max_q_global:
                        self._cannot_fix.add((X, Y))
                        break

                    # Test conditional independence of X and Y given Z. Correspondingly updateself.val_min, self.pval_max, and self.cardinality 
                    val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                    if self.verbosity >= 2:
                        print("    %s _|_ %s  |  S_pc = %s: val = %.2f / pval = % .4f" %
                            (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                    self._update_val_min(X, Y, val)
                    self._update_pval_max(X, Y, pval)
                    self._update_cardinality(X, Y, len(Z))

                    # Check whether the test result was significant
                    if pval > self.pc_alpha:

                        # Mark the edge from X to Y for removal and save sepset
                        to_remove[Y[0]][X] = True
                        self._save_sepset(X, Y, (frozenset(Z), ""))
    
                        # Verbose output
                        if self.verbosity >= 1:
                            print("({},{:2}) {:11} {} given {}".format(X[0], X[1], "independent", Y, Z))

                        # Break the for loop
                        break

                if test_X:

                    if self.verbosity >= 1:
                        print("for S_pc in combinations(S_search_XY, p_pc)")

                    if not np.isinf(self.max_q_global):
                        S_search_XY = self._sort_search_set(S_search_XY, X)

                    q_count = 0
                    for Z in combinations(S_search_XY, p_pc):

                        q_count = q_count + 1
                        if q_count > self.max_q_global:
                            self._cannot_fix.add((X, Y))
                            break

                        val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                        if self.verbosity >= 2:
                            print("    %s _|_ %s  |  S_pc = %s: val = %.2f / pval = % .4f" %
                                (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                        # Update val_min and pval_max
                        self._update_val_min(X, Y, val)
                        self._update_pval_max(X, Y, pval)
                        self._update_cardinality(X, Y, len(Z))

                        if pval > self.pc_alpha:

                            to_remove[Y[0]][X] = True
                            self._save_sepset(X, Y, (frozenset(Z), ""))
        
                            if self.verbosity >= 1:
                                print("({},{:2}) {:11} {} given {}".format(X[0], X[1], "independent", Y, Z))

                            break

            # end for (i, j, lag_i) in product(range(self.N), range(self.N), range(-(tau_max + 1), 1))

            ##########################################################################################################
            ### Remove edges marked for removal in to_remove #########################################################

            # Remove edges
            for j in range(self.N):
                for (i, lag_i) in to_remove[j].keys():

                    self._write_link((i, lag_i), (j, 0), "", verbosity = self.verbosity)

            # Verbose output
            if self.verbosity >= 1:
                    print("\nTest phase complete")

            ##########################################################################################################
            ### Check for convergence ################################################################################
        
            if has_converged:
                # If no link needed testing, this algorithm has converged. Therfore, break the while loop
                break
            else:
                # At least one link needed testing, this algorithm has not yet converged. Therefore, increase p_pc
                p_pc = p_pc + 1

        # end while True

        # Undo all preliminary collider orientations
        self._unorient_all_edges()
        self.def_non_ancs = {j: set() for j in range(self.N)}

        # Verbose output
        if self.verbosity >= 1:
            print("\nFinal removal phase complete")
            print("\nGraph:\n--------------------------------")
            self._print_graph_dict()
            print("--------------------------------")

        # Return
        return True


    def _run_fci_orientation_phase(self):
        """Run the final orientation phase the FCI algorithm"""

        # Verbose output
        if self.verbosity >= 1:
            print("\n=======================================================")
            print("=======================================================")
            print("Starting FCI orientation phase")

        # Orient colliders colliders
        self._run_orientation_phase(rule_list = [["R-00-d"]], voting = "Majority-Final")

        # Exhaustively apply the other relevant orientation rules. Rules 5, 6 and 7 are not relevant because by assumption there are no selection variables
        self._run_orientation_phase(rule_list = [["R-01"], ["R-02"], ["R-03"], ["R-04"], ["R-08"], ["R-09"], ["R-10"]], voting = "Majority-Final")

        # Verbose output
        if self.verbosity >= 1:
            print("\nFCI orientation phase complete")
            print("\nFinal graph:\n--------------------------------")
            print("--------------------------------")
            self._print_graph_dict()
            print("--------------------------------")
            print("--------------------------------\n")

        # Return
        return True

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _run_orientation_phase(self, rule_list, voting):
        """Function for exhaustive application of the orientation rules specified by rule_list. The argument voting specifies the rule with which it is decided whether B is in the separating set of A and C, where A - B - C is an unshielded triple"""

        # Verbose output
        if self.verbosity >= 1:
            print("\nStarting orientation phase")
            print("with rule list: ", rule_list)

        # Run through all priority levels of rule_list
        idx = 0
        while idx <= len(rule_list) - 1:

            # Some rule require that self._graph_full_dict is updated. Therefore, initialize this variable once the while loop (re)-starts at the first prioprity level
            if idx == 0:
                self._initialize_full_graph()

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
                orientations = self._apply_rule(rule, voting)

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

            new_ancs = {j: set() for j in range(self.N)}
            new_non_ancs = {j: set() for j in range(self.N)}

            # Run through all of the nested dictionary
            for ((i, j, lag_i), new_link) in to_orient:

                # The old link
                old_link = self._get_link((i, lag_i), (j, 0))

                # Assert that no preceeding variable is marked as an ancestor of later variable
                assert not (lag_i > 0 and new_link[2] == "-")

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

            ###########################################################################################################
            ### Update ancestral information and determine next step ##################################################

            # Update ancestral information. The function called includes conflict resolution
            restart = self._apply_new_ancestral_information(new_non_ancs, new_ancs)

            # If any useful new information was found, go back to idx = 0, else increase idx by 1
            idx = 0 if restart == True else idx + 1

        # end  while i <= len(self.rule_list) - 1
        # The algorithm has converged

        # Verbose output
        if self.verbosity >= 1:
            print("\nOrientation phase complete")

        # Return
        return True


    def _get_pds_t(self, A, B):
        """Return pds_t(A, B) according to the current graph"""

        # Unpack A and B, then assert that at least one of them is at lag 0
        var_A, lag_A = A
        var_B, lag_B = B
        assert lag_A == 0 or lag_B == 0

        # If pds_t(A, B) is in memory, return from memory
        memo = self._pds_t[A].get(B) 
        if memo is not None:
            return memo

        # Else, re-compute it with breath-first search according to the current graph
        visited = set()
        start_from = {((var, lag + lag_A), A) for (var, lag) in self.graph_full_dict[var_A].keys() if lag + lag_A >= -self.tau_max and lag + lag_A <= 0}

        while start_from:

            new_start_from = set()

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
                    if self._get_link(next_node, previous_node) == "" and (self._get_link(previous_node, current_node)[2] == "o" or self._get_link(next_node, current_node)[2] == "o"):
                        continue

                    new_start_from.add((next_node, current_node))

            start_from = new_start_from

        # Cache results and return
        res = {node for (node, _) in visited if node != A and node != B}
        self.max_pds_set_found = max(self.max_pds_set_found, len(res))
        self._pds_t[A][B] = res
        return self._pds_t[A][B]


    def _unorient_all_edges(self):
        """Remove all orientations, except the non-ancestorships implied by time order"""

        for j in range(self.N):
                for (i, lag_i) in self.graph_dict[j].keys():

                    link = self._get_link((i, lag_i), (j, 0))
                    if len(link) > 0:
                        if lag_i == 0:
                            new_link = "o" + link[1] + "o"
                        else:
                            new_link = "o" + link[1] + ">"
                        self.graph_dict[j][(i, lag_i)] = new_link


    def _fix_all_edges(self):
        """Set the middle mark of all links to '-'"""

        for j in range(self.N):
                for (i, lag_i) in self.graph_dict[j].keys():

                    link = self._get_link((i, lag_i), (j, 0))
                    if len(link) > 0:
                        new_link = link[0] + "-" + link[2]
                        self.graph_dict[j][(i, lag_i)] = new_link

    
    def _apply_new_ancestral_information(self, new_non_ancs, new_ancs):
        """Apply the new ancestorships and non-ancestorships specified by new_non_ancs and new_ancs to the current graph. Conflicts are resolved by marking. Returns True if any circle mark was turned into a head or tail, else False."""

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
                    # There is a conflict, since X and Y are contemporaneous and it is already marked ambiguous as whether Y is an ancestor of Y
                    # Note: This is required here, because X being an ancestor of Y implies that Y is not an ancestor of X. This ambiguity cannot exist when X is before Y
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


    def _apply_rule(self, rule, voting):
        """Call the orientation-removal-rule specified by the string argument rule. Pass on voting."""

        if rule == "R-00-d":
            return self._apply_R00(voting)
        elif rule == "R-01":
            return self._apply_R01(voting)
        elif rule == "R-02":
            return self._apply_R02()
        elif rule == "R-03":
            return self._apply_R03(voting)
        elif rule == "R-04":
            return self._apply_R04(voting)
        elif rule == "R-08":
            return self._apply_R08()
        elif rule == "R-09":
            return self._apply_R09(voting)
        elif rule == "R-10":
            return self._apply_R10(voting)


    def _B_not_in_SepSet_AC(self, A, B, C, voting):
        """Return True if B is not in the separating set of A and C. If voting = 'Majority-Final', this is done according to the standard majority rule. If voting = 'Majority-Final', for A-B-C that would be marked as ambiguous triples by 'Majority-Final', also return True."""

        # Treat A - B - C as the same triple as C - B - A
        # Convention: A is before C or, if they are contemporaneous, the index of A is smaller than that of C
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_not_in_SepSet_AC(C, B, A, voting)

        ################################################################################################
        # Only relevant for use with oracle CI
        if self._oracle:
            return self._B_not_in_SepSet_AC_given_answers[((A[0], A[1] - C[1]), (B[0], B[1] - C[1]), (C[0], 0))]
        ################################################################################################

        # If the triple is ambiguous, immediately return False
        if (A, B, C) in self.ambiguous_triples or (C, B, A) in self.ambiguous_triples:
            return False

        # Remember all separating sets that we will find
        all_sepsets = set()

        # Test for independence given all subsets of non-future adjacencies of A
        adj_A = self._get_non_future_adj([A]).difference({A, C})
        adj_C = self._get_non_future_adj([C]).difference({A, C})

        # Depending on the self.max_cond_px and self.max_p_global, determine the maximal cardinality of subsets of adj_A that are tested
        if A[1] < C[1]:
            max_p_A = min([len(adj_A), self.max_cond_px, self.max_p_global]) + 1
        else:
            max_p_A = min([len(adj_A), self.max_p_global]) + 1

        # If self.max_q_global is finite, order adj_A and adj_C according to self.val_min to guarantee order independence
        if not np.isinf(self.max_q_global):
            adj_A = self._sort_search_set(adj_A, A)
            adj_C = self._sort_search_set(adj_C, C)

        # Shift lags
        adj_A = [(var, lag - C[1]) for (var, lag) in adj_A]
        adj_C = [(var, lag - C[1]) for (var, lag) in adj_C]
        X = (A[0], A[1] - C[1])
        Y = (C[0], 0)

        # Test for independence given subsets of non-future adjacencies of A
        for p in range(max_p_A):

            # Count the number of tests made at this value of p
            q_count = 0

            for Z_raw in combinations(adj_A, p):

                # Break if the maximal number of tests specified by self.max_q_global has been exceeded
                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break

                # Prepare the conditioning set
                Z = {node for node in Z_raw if node != X and node != Y}

                # Test for conditional independence of X and Y given Z
                val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                if self.verbosity >= 2:
                    print("BnotinSepSetAC(A):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                # Check whether the test result was significant. If yes, remember Z as separating set
                if pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

        # Test for independence given subsets of non-future adjacencies of C
        for p in range(min(len(adj_C), self.max_p_global) + 1):

            q_count = 0
            for Z_raw in combinations(adj_C, p):

                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break

                Z = {node for node in Z_raw if node != X and node != Y}

                val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                if self.verbosity >= 2:
                    print("BnotinSepSetAC(C):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                if pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

        # Count number of sepsets and number of sepsets that contain B
        n_sepsets = len(all_sepsets)
        n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])

        # Determine the answer
        if voting == "Majority-Preliminary":

            # Return True if no separating set was found or if at least one separating set was found and B is in less than half of them, else False
            return True if (n_sepsets == 0 or 2*n_sepsets_with_B < n_sepsets) else False

        elif voting == "Majority-Final":

            # Return True if at least one separating set was found and B is in less than half of them, False if at least one separating set has been found and B is in more than half of them, else mark the triple as ambiguous
            if n_sepsets == 0 or 2*n_sepsets_with_B == n_sepsets:

                #######################################################
                #for (Z, _) in self._get_sepsets(A, C):
                #    return False if B in Z else True
                #######################################################

                self.ambiguous_triples.add((A, B, C))
                return False
            elif 2*n_sepsets_with_B < n_sepsets:
                return True
            else:
                return False

        else:

            assert False


    def _B_in_SepSet_AC(self, A, B, C, voting):
        """Return True if B is in the separating set of A and C. This is done according to the standard majority rule"""

        # Treat A - B - C as the same triple as C - B - A
        # Convention: A is before C or, if they are contemporaneous, the index of A is smaller than that of C
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_in_SepSet_AC(C, B, A, voting)

        ################################################################################################
        # Only relevant for use with oracle CI
        if self._oracle:
            return not self._B_not_in_SepSet_AC_given_answers[((A[0], A[1] - C[1]), (B[0], B[1] - C[1]), (C[0], 0))]
        ################################################################################################

        if (A, B, C) in self.ambiguous_triples or (C, B, A) in self.ambiguous_triples:
            return False

        # This function must only be called from the final orientation phase
        if voting != "Majority-Final":
            assert False

        # Remember all separating sets that we will find
        all_sepsets = set()

        # Get the non-future adjacencies of A and C
        adj_A = self._get_non_future_adj([A]).difference({A, C})
        adj_C = self._get_non_future_adj([C]).difference({A, C})

        # Depending on the self.max_cond_px and self.max_p_global, determine the maximal cardinality of subsets of adj_A that are tested
        if A[1] < C[1]:
            max_p_A = min([len(adj_A), self.max_cond_px, self.max_p_global]) + 1
        else:
            max_p_A = min([len(adj_A), self.max_p_global]) + 1

        if not np.isinf(self.max_q_global):
            adj_A = self._sort_search_set(adj_A, A)
            adj_C = self._sort_search_set(adj_C, C)

        # Shift lags
        adj_A = [(var, lag - C[1]) for (var, lag) in adj_A]
        adj_C = [(var, lag - C[1]) for (var, lag) in adj_C]
        X = (A[0], A[1] - C[1])
        Y = (C[0], 0)

        # Test for independence given subsets of non-future adjacencies of A
        for p in range(max_p_A):

            # Count the number of tests made at this value of p
            q_count = 0

            for Z_raw in combinations(adj_A, p):

                # Break if the maximal number of tests specified by self.max_q_global has been exceeded
                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break

                # Prepare the conditioning set
                Z = {node for node in Z_raw if node != X and node != Y}

                # Test conditional independence of X and Y given Z
                val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                if self.verbosity >= 2:
                    print("BinSepSetAC(A):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                # Check whether the test result was significant. If yes, remember Z as separating set
                if pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

        # Test for independence given subsets of non-future adjacencies of C
        for p in range(min(len(adj_C), self.max_p_global) + 1):

            q_count = 0
            for Z_raw in combinations(adj_C, p):

                q_count = q_count + 1
                if q_count > self.max_q_global:
                    break

                Z = {node for node in Z_raw if node != X and node != Y}

                val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                if self.verbosity >= 2:
                    print("BinSepSetAC(C):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                if pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

        # Count number of sepsets and number of sepsets that contain B
        n_sepsets = len(all_sepsets)
        n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])

        # Return False if at least one separating set was found and B is in less than half of them, True if at least one separating set has been found and B is in more than half of them, else mark the triple as ambiguous
        if n_sepsets == 0 or 2*n_sepsets_with_B == n_sepsets:

            #######################################################
            #for (Z, _) in self._get_sepsets(A, C):
            #    return True if B in Z else False
            #######################################################

            self.ambiguous_triples.add((A, B, C))
            return  False
        elif 2*n_sepsets_with_B < n_sepsets:
            return False
        else:
            return True

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _apply_R00(self, voting):
        """Return all orientations implied by orientation rule R-00-d"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        if voting == "Majority-Preliminary":
            triples_1 = self._find_triples(pattern_ij='**o', pattern_jk='o**', pattern_ik='')
            triples_2 = self._find_triples(pattern_ij='**>', pattern_jk='o**', pattern_ik='')
            all_appropriate_triples = set(triples_1).union(set(triples_2))
        else:
            triples_1 = self._find_triples(pattern_ij='*-o', pattern_jk='o-*', pattern_ik='')
            triples_2 = self._find_triples(pattern_ij='*->', pattern_jk='o-*', pattern_ik='')
            all_appropriate_triples = set(triples_1).union(set(triples_2))            

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if self.verbosity >= 2:
                print("R00: ", (A, B, C))

            # Check whether the rule applies
            if self._B_not_in_SepSet_AC(A, B, C, voting):
                # Prepare the new links and append them to the output

                if self.verbosity >= 2:
                    print("   --> not in sepset ")

                # From C to B
                link_CB = self._get_link(C, B)
                new_link_CB = link_CB[0] + link_CB[1] + ">"
                out.append(self._get_pair_key_and_new_link(C, B, new_link_CB))

                # If needed, also fromA to B
                link_AB = self._get_link(A, B)
                if link_AB[2] == "o":
                    new_link_AB = link_AB[0] + link_AB[1] + ">"
                    out.append(self._get_pair_key_and_new_link(A, B, new_link_AB))

        # Return the output list
        return out


    def _apply_R01(self, voting):
        """Return all orientations implied by orientation rule R-01"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='*->', pattern_jk='o-+', pattern_ik='')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            if self.verbosity >= 2:
                print("R01: ", (A, B, C))

            # Check whether the rule applies
            if self._B_in_SepSet_AC(A, B, C, voting):

                if self.verbosity >= 2:
                    print(" --> in sepset ")

                # Prepare the new link from B to C and append it to the output list
                link_BC = self._get_link(B, C)
                new_link_BC = "-" + link_BC[1] + ">"
                out.append(self._get_pair_key_and_new_link(B, C, new_link_BC))

        # Return the output list
        return out


    def _apply_R02(self):
        """Return all orientations implied by orientation rule R-02"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = set(self._find_triples(pattern_ij='-->', pattern_jk='*->', pattern_ik='+-o'))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='*->', pattern_jk='-->', pattern_ik='+-o')))

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # The rule applies to all relevant graphical structures. Therefore, prepare the new link and append it to the output list
            link_AC = self._get_link(A, C)
            new_link_AC = link_AC[0] + link_AC[1] + ">"
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))

        # Return the output list
        return out


    def _apply_R03(self, voting):
        """Return all orientations implied by orientation rule R-03"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_quadruples = self._find_quadruples(pattern_ij='*->', pattern_jk='<-*', pattern_ik='', 
                                                           pattern_il='+-o', pattern_jl='o-+', pattern_kl='+-o')

        # Run through all appropriate graphical structures
        for (A, B, C, D) in all_appropriate_quadruples:

            # Check whether the rule applies
            if self._B_in_SepSet_AC(A, D, C, voting):

                # Prepare the new link from D to B and append it to the output list
                link_DB = self._get_link(D, B)
                new_link_DB = link_DB[0] + link_DB[1] + ">"
                out.append(self._get_pair_key_and_new_link(D, B, new_link_DB))

        # Return the output list
        return out


    def _apply_R04(self, voting):
        """Return all orientations implied by orientation rule R-04"""

        # Build the output list
        out = []

        # Find all relevant triangles W-V-Y
        all_appropriate_triples = self._find_triples(pattern_ij='<-*', pattern_jk='o-+', pattern_ik='-->')

        # Run through all of these triangles
        for triple in all_appropriate_triples:

            (W, V, Y) = triple

            # Get the current link from W to V, which we will need below
            link_WV = self._get_link(W, V)

            # Find all discriminating paths for this triangle
            # Note: To guarantee order independence, we check all discriminating paths. Alternatively, we could check the rule for all shortest such paths
            discriminating_paths =  self._get_R4_discriminating_paths(triple, max_length = np.inf)

            # Run through all discriminating paths
            for path in discriminating_paths:

                # Get the end point node
                X_1 = path[-1]

                # Check which of the two cases of the rule we are in, then append the appropriate new links to the output list
                if self._B_in_SepSet_AC(X_1, V, Y, voting):
                    # New link from V to Y
                    out.append(self._get_pair_key_and_new_link(V, Y, "-->"))

                elif link_WV != "<-x" and self._B_not_in_SepSet_AC(X_1, V, Y, voting):
                    # New link from V to Y
                    out.append(self._get_pair_key_and_new_link(V, Y, "<->"))

                    # If needed, also the new link from W to V
                    if link_WV != "<->":
                        out.append(self._get_pair_key_and_new_link(W, V, "<->"))

        # Return the output list
        return out

    def _apply_R08(self):
        """Return all orientations implied by orientation rule R-08"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='-->', pattern_jk='-->', pattern_ik='o-+')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # The rule applies to all relevant graphical structures. Therefore, prepare the new link and append it to the output list
            link_AC = self._get_link(A, C)
            new_link_AC = "-" + link_AC[1] + ">"
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))

        # Return the output list
        return out


    def _apply_R09(self, voting):
        """Return all orientations implied by orientation rule R-09"""

        # Build the output list
        out = []

        # Find unshielded triples B_1 o--*--o A o--*--> C or B_1 <--*--o A o--*--> C or B_1 <--*-- A o--*--> C 
        all_appropriate_triples = set(self._find_triples(pattern_ij='o-o', pattern_jk='o->', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<-o', pattern_jk='o->', pattern_ik='')))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<--', pattern_jk='o->', pattern_ik='')))

        # Run through all these triples
        for (B_1, A, C) in all_appropriate_triples:

            # Check whether A is in SepSet(B_1, C), else the rule does not apply
            if not self._B_in_SepSet_AC(B_1, A, C, voting):
                continue

            # Although we do not yet know whether the rule applies, we here determine the new form of the link from A to C if the rule does apply
            link_AC = self._get_link(A, C)
            new_link_AC = "-" + link_AC[1] + ">"
            pair_key, new_link = self._get_pair_key_and_new_link(A, C, new_link_AC)

            # For the search of uncovered potentially directed paths from B_1 to C, determine the initial pattern as dictated by the link from A to B_1
            first_link = self._get_link(A, B_1)
            if self._match_link(pattern='o*o', link=first_link):
                initial_allowed_patterns = ['-->', 'o->', 'o-o']
            elif self._match_link(pattern='o->', link=first_link) or self._match_link(pattern='-->', link=first_link):
                initial_allowed_patterns = ['-->']

            # Find all uncovered potentially directed paths from B_1 to C
            uncovered_pd_paths = self._get_potentially_directed_uncovered_paths_fci(B_1, C, initial_allowed_patterns)

            # Run through all of these paths and check i) whether the node adjacent to B_1 is non-adjacent to A, ii) whether condition iv) of the rule antecedent is true. If there is any such path, then the link can be oriented
            for upd_path in uncovered_pd_paths:

                # Is the node adjacent to B_1 non-adjacent to A (this implies that there are at least three nodes on the path, because else the node adjacent to B_1 is C) and is A not part of the path?
                if len(upd_path) < 3 or A in upd_path or self._get_link(A, upd_path[1]) != "":
                    continue

                # If the link from A to B_1 is into B_1, condition iv) is true
                if first_link[2] == ">":
                    # Mark the link from A to C for orientation, break the for loop to continue with the next triple
                    out.append((pair_key, new_link))
                    break

                # If the link from A to B_1 is not in B_1, we need to check whether B_1 is in SepSet(A, X) where X is the node on upd_path next to B_1
                if not self._B_in_SepSet_AC(A, B_1, upd_path[1], voting):
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
                    if not self._B_in_SepSet_AC(upd_path[i], upd_path[i+1], upd_path[i+2], voting):
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


    def _apply_R10(self, voting):
        """Return all orientations implied by orientation rule R-10"""

        # Build the output list
        out = []

        # Find all triples A o--> C <-- P_C
        all_appropriate_triples = set(self._find_triples(pattern_ij='o->', pattern_jk='<--', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='o->', pattern_jk='<--', pattern_ik='***')))

        # Collect all triples for the given pair (A, C)
        triple_sorting_dict = {}
        for (A, C, P_C) in all_appropriate_triples:
            if triple_sorting_dict.get((A, C)) is None:
                triple_sorting_dict[(A, C)] = [P_C]
            else:
                triple_sorting_dict[(A, C)].append(P_C)

        # Run through all (A, C) pairs
        for (A, C) in triple_sorting_dict.keys():

            # Find all uncovered potentially directed paths from A to C through any of the P_C nodes
            relevant_paths = []
            for P_C in triple_sorting_dict[(A, C)]:
                for upd_path in self._get_potentially_directed_uncovered_paths_fci(A, P_C, ['-->', 'o->', 'o-o']):

                    # Run through all of these paths and check i) whether the second to last element is not adjacent to C (this requires it to have a least three nodes, because else the second to last element would be A) and ii) whether the left edge of any 3-node sub-path is into the middle nor or, if not, whether the middle node is in the separating set of the two end-point nodes (of the 3-node) sub-path and iii) whether C is not element of the path. If path meets these conditions, add its second node (the adjacent to A) to the set second_nodes

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
                        if not self._B_in_SepSet_AC(upd_path[i], upd_path[i+1], upd_path[i+2], voting):
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

            # Check whether there is any pair of non-adjacent nodes in second_nodes, such that A is in their separating set. If yes, mark the link from A to C for orientation
            for i, j in product(range(len(second_nodes)), range(len(second_nodes))):

                if i < j and self._get_link(second_nodes[i], second_nodes[j]) == "" and self._B_in_SepSet_AC(second_nodes[i], A, second_nodes[j], voting):
                    # Append new link and break the for loop
                    link_AC = self._get_link(A, C)
                    new_link_AC = "-" + link_AC[1] + ">"
                    out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))
                    break

        # end for (A, C) in triple_sorting_dict.keys()

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


    def _is_smaller(self, X, Y):
        """
        A node X is said to be smaller than node Y if
        i)  X is before Y or
        ii) X and Y are contemporaneous and the variable index of X is smaller than that of Y.

        Return True if X is smaller than Y, else return False
        """

        return (X[1] < Y [1]) or (X[1] == Y[1] and X[0] < Y[0])


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


    def _update_val_min(self, X, Y, val):
        """Some conditional independence test for X and Y has given the test statistic value val. Update the val_min dictionary accordingly"""

        if X[1] < 0 or X[0] < Y[0]:
            self.val_min[Y[0]][X] = min(self.val_min[Y[0]][X], np.abs(val))
        else:
            self.val_min[X[0]][Y] = min(self.val_min[X[0]][Y], np.abs(val))


    def _get_val_min(self, X, Y):
        """Return the value stored in self.val_min for the variable pair (X, Y)"""

        if X[1] < 0 or X[0] < Y[0]:
            return self.val_min[Y[0]][X]
        else:
            return self.val_min[X[0]][Y]


    def _update_cardinality(self, X, Y, cardinality):
        """X and Y were found conditionally independent given a separating set of cardinality cardinality. Update the self.cardinality accordingly"""

        if X[1] < 0 or X[0] < Y[0]:
            self.max_cardinality[Y[0]][X] = max(self.max_cardinality[Y[0]][X], cardinality)
        else:
            self.max_cardinality[X[0]][Y] = max(self.max_cardinality[X[0]][Y], cardinality)


    def _update_pval_max(self, X, Y, pval):
        """Some conditional independence test for X and Y has given the p-value val. Update the pval_max dictionary accordingly"""

        if X[1] < 0 or X[0] < Y[0]:
            self.pval_max[Y[0]][X] = max(self.pval_max[Y[0]][X], pval)
        else:
            self.pval_max[X[0]][Y] = max(self.pval_max[X[0]][Y], pval)


    def _sort_search_set(self, search_set, reference_node):
        """Sort the nodes in search_set by their val_min value with respect to the reference_node. Nodes with higher values appear earlier""" 

        sort_by = [self._get_val_min(reference_node, node) for node in search_set]
        return [x for _, x in sorted(zip(sort_by, search_set), reverse = True)]


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
        """Write the information that the link from node A to node B takes the form of new_link into self.graph_dict. Neither is it assumed that at least of the nodes is at lag 0, nor must A be before B. If A and B are contemporaneous, also the link from B to A is written as the reverse of new_link"""

        # Unpack A and B
        (var_A, lag_A) = A
        (var_B, lag_B) = B

        # Write the link from A to B
        if lag_A < lag_B:

            if verbosity >= 1:
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_A, lag_A - lag_B, self.graph_dict[var_B][(var_A, lag_A - lag_B)], var_B, 0, var_A, lag_A - lag_B, new_link, var_B, 0))

            self.graph_dict[var_B][(var_A, lag_A - lag_B)] = new_link


        elif lag_A == lag_B:

            if verbosity >= 1:
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_A, lag_A - lag_B, self.graph_dict[var_B][(var_A, 0)], var_B, 0, var_A, 0, new_link, var_B, 0))

                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_B, 0, self.graph_dict[var_A][(var_B, 0)], var_A, 0, var_B, 0, self._reverse_link(new_link), var_A, 0))

            self.graph_dict[var_B][(var_A, 0)] = new_link
            self.graph_dict[var_A][(var_B, 0)] = self._reverse_link(new_link)

        else:

            if verbosity >= 1:
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Writing:", var_B, lag_B - lag_A, self.graph_dict[var_A][(var_B, lag_B - lag_A)], var_A, 0, var_B, lag_B - lag_A, self._reverse_link(new_link), var_A, 0))

            self.graph_dict[var_A][(var_B, lag_B - lag_A)] = self._reverse_link(new_link)


    def _get_sepsets(self, A, B):
        """For two non-adjacent nodes, get the their separating stored in self.sepsets"""

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
        """Initialize self.graph_full_dict. This nested dictionary represents the graph and as opposed to self.graph_dict also contains forward links"""

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


    def _get_potentially_directed_uncovered_paths_fci(self, start_node, end_node, initial_allowed_patterns):
        """Find all potentiall directed uncoverged paths from start_node to end_node whose first link takes one the forms specified by initial_allowed_patters"""

        assert start_node != end_node

        # Function for recursive search of potentially directed uncovered paths
        def _search(end_node, path_taken, allowed_patterns):

            # print(path_taken)

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
                    # ... are not part of an ambiguous triple
                    if len(path_taken) >= 2 and ((path_taken[-2], start_node, next_node) in self.ambiguous_triples or (next_node, start_node, path_taken[-2]) in self.ambiguous_triples):
                        continue
                    # ... whose link with start_node matches one of the allowed patters
                    link = self._get_link(start_node, next_node)
                    if not any([self._match_link(pattern = pattern, link = link) for pattern in allowed_patterns]):
                        continue

                    # Determine the allowed patters for the next recursive call
                    if self._match_link(pattern='o-o', link=link):
                        new_allowed_patters = ["o-o", "o->", "-->"]
                    elif self._match_link(pattern='o->', link=link) or self._match_link(pattern='-->', link=link):
                        new_allowed_patters = ["-->"]

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


    def _dict_to_matrix(self, val_dict, tau_max, n_vars, default=1):
        """Convert a dictionary to matrix format"""

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