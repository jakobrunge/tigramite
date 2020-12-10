import numpy as np
from itertools import product, combinations
import os

class SVARRFCI():
    r"""
    This class implements an adapation of the RFCI algorithm to stationary time series with the assumption of no selection variables. The RFCI algorithm was introduced in:

        Colombo, D., Maathuis, M. H., Kalisch, M., and Richardson, T. S. (2012). Learning high-dimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics, 40:294–321.

    We note the following:
        1) The algorithm is fully order-independence. This is achieved by two things. First, we use the majority rule to decide whether a given node is in a given separating set. Since the unshielded triple rule (given in Lemma 3.1 in the above reference) demands minimal separating set, the majority vote is restricted to separating sets of minimal cardinality (this implies minimality). Second, we apply an orientation rule to the entire graph and resolve potential conflicts among the proposed orientations by means of the conflict mark 'x' before modifing the graph. This also applies to the discriminating path rule (given in Lemma 3.2 in the above reference)
        2) Several control parameters apply modifications, see below.

    Parameters passed to the constructor:
    - dataframe:
        Tigramite dataframe object that contains the the time series dataset \bold{X}
    - cond_ind_test:
        A conditional independence test object that specifies which conditional independence test CI is to be used

    Parameters passed to self.run_svarrfci():
    - tau_max:
        The maximum considered time lag tau_max
    - pc_alpha:
        The significance level \alpha of conditional independence tests
    - max_cond_px:
        Consider a pair of variables (X^i_{t-\tau}, X^j_t) with \tau > 0. In the edge removal phase (here this is self._run_pc_removal_phase()), the algorithm does not test for conditional independence given subsets of X^i_{t-\tau} of cardinality higher than max_cond_px.
    - max_p_global:
        Restricts all conditional independence tests to conditioning sets with cardinality smaller or equal to max_p_global
    - max_q_global:
        For each ordered pair (X^i_{t-\tau}, X^j_t) of adjacent variables and for each cardinality of the conditioning sets test at most max_q_global many conditioning sets (when summing over all tested cardinalities more than max_q_global tests may be made)
    - fix_all_edges_before_final_orientation (will be removed)
    - verbosity:
        Controls the verbose output self.run_svarrfci() and the function it calls.

    Return value of self.run_svarrfci():
        The estimated graphin form of a link matrix. This is a numpy array of shape (self.N, self.N, self.tau_max + 1), where the entry array[i, j, \tau] is a string that visualizes the estimated link from X^i_{i-\tau} to X^j_t. For example, if array[0, 2, 1] = 'o->', then the estimated graph contains the link X^i_{t-1} o-> X^j_t. This numpy array is also saved as instance attribute self.graph. Note that self.N is the number of observed time series and self.tau_max the maximal considered time lag.

    A note on middle marks:
        Even if both max_p_global and max_q_global are np.inf, RFCI does not guarantee that all edges that remain in its graph after convergence (this is an RFCI-PAG) are also in the PAG. However, it does guarantee this for all edges that have a tail. We use the middle marks that we introduced for LPCMCI to explicate this distinction. In particular, we use the middle marks '?' and '' (empty). For convenience (to have strings of the same lengths) we here internally denote the empty middle mark by '-'. For post-processing purposes all middle marks are nevertheless set to the empty middle mark (here '-') in line 80, but if verbosity >= 1 a graph with the middle marks will be printed out before.
    
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


    def run_svarrfci(self, 
                tau_max = 1, 
                pc_alpha = 0.05,
                max_cond_px = 0,
                max_p_global = np.inf,
                max_q_global = np.inf,
                fix_all_edges_before_final_orientation = True,
                verbosity = 0):
        """Run an adaption of the RFCI algorithm to stationary time series without selection variables on the dataset and with the conditional independence test passed to the class constructor and with the options passed to this function."""

        # Step 0: Intializations
        self._initialize(tau_max, pc_alpha, max_cond_px, max_p_global, max_q_global, fix_all_edges_before_final_orientation, verbosity)

        # Step 1: PC removal phase
        self._run_pc_removal_phase()

        # Step 2: RFCI orientation phase
        self._run_rfci_orientation_phase()

        # Post processing
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
                    max_q_global,
                    fix_all_edges_before_final_orientation,
                    verbosity):
        """Function for
            i)      saving the arguments passed to self.run_svarrfci() as instance attributes
            ii)     initializing various memory variables for storing the current graph, sepsets etc.
            """

        # Save the arguments passed to self.run_svarrfci()
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.max_cond_px = max_cond_px
        self.max_p_global = max_p_global
        self.max_q_global = max_q_global
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

        # Return
        return True


    def _run_pc_removal_phase(self):
        """Run the removal phase of the RFCI algorithm adapted to time series. This is essentially the skeleton phase of the PC algorithm"""

        # Verbose output
        if self.verbosity >= 1:
            print("\n=======================================================")
            print("=======================================================")
            print("Starting removal phase")

        # Remember all edges that are fully tested, even for finite max_p_global and max_q_global. Remember all edges that have not been fully tested
        self._can_fix = set()
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

                # If there are less than p_pc elements in both search sets, the link does not need further testing. If the pair (X, Y) has been fully tested, i.e., it has not been added to self._cannot_fix, we add it to self._can_fix. Then, later, in case one edge mark is set to a tail, we know that the link is part of the True MAG
                if len(S_search_YX) < p_pc and (not test_X or len(S_search_XY) < p_pc):
                    if (X, Y) not in self._cannot_fix:
                        self._can_fix.add((X, Y))
                    continue

                # Force-quit while leep when p_pc exceeds the specified limits
                if p_pc > self.max_p_global:
                    continue

                # Since this link does need testing, the algorithm has not converged yet
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

                        # Mark the edge from X to Y for removal, save Z as minimal separating set
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
                            self._cannot_fix.add((X, Y))
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
            print("\nRemoval phase complete")
            print("\nGraph:\n--------------------------------")
            self._print_graph_dict()
            print("--------------------------------")

        # Return
        return True


    def _run_rfci_orientation_phase(self):
        """Run the orientation phase of the RFCI algorithm: Steps 2 and 3 of algorithm 3.2 in the RFCI paper"""

        # Verbose output
        if self.verbosity >= 1:
            print("\n=======================================================")
            print("=======================================================")
            print("Starting RFCI orientation phase")

        # Run the RFCI unshielded triple rule
        M = set(self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik=''))
        self._run_rfci_utr_rule(M)

        # Remember whether the middle marks of all links are put to '-' by force. This is done once in the last iteration of the while loop in case self.fix_all_edges_before_final_orientations is True
        fixed_all = False

        # Run until convergence
        changed = True
        while changed:

            # Remember the current graph
            old_graph_dict = {}
            for j in range(self.N):
                old_graph_dict[j] = {k: v for (k, v) in self.graph_dict[j].items()}

            # Run Rules 1 - 3
            self._run_orientation_phase(rule_list = [["R-01"], ["R-02"], ["R-03"]])

            # Run the RFCI discriminating path rule
            self._run_rfci_dpr_rule()

            # Run Rules 8 - 10
            self._run_orientation_phase(rule_list = [["R-08"], ["R-09"], ["R-10"]])

            # Check whether there was a change
            changed = False
            for j in range(self.N):
                for (k, v)in self.graph_dict[j].items():
                    if v != old_graph_dict[j][k]:
                        changed = True

            # In case the corresonponding option is chosen and graph does not change anymore, set all middle marks to '-'
            if not changed and self.fix_all_edges_before_final_orientation and not fixed_all:

                self._fix_all_edges()
                changed = True
                fixed_all = True

        # Fix all edges that have a tail
        self._fix_edges_with_tail()

        # Verbose output
        if self.verbosity >= 1:
            print("\nRFCI orientation phase complete")
            print("\nFinal graph:\n--------------------------------")
            print("--------------------------------")
            self._print_graph_dict()
            print("--------------------------------")
            print("--------------------------------\n")

        # Return True
        return True


    def _run_rfci_utr_rule(self, M):
        """Run the RFCI unshielded triple rule: Algorithm 4.4 of the RFCI supplement paper"""

        # Verbose output
        if self.verbosity >= 1:
            print("\nStarting RFCI UTR-Rule:")

        # Take care that not both (A, B, C) and (C, B, A) appear in M
        M_unique = set()
        for (A, B, C) in M:
            if not (C, B, A) in M_unique:
                M_unique.add((A, B, C))
        M = M_unique

        # Make a list of triples that will bee tested for orientation ('L' in RFCI paper)
        L = set()

        # Run as long as there are unshielded triples in M
        while len(M) > 0:

            # Remember all unshielded triples
            old_unshielded_triples = set(self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik=''))

            # Make a list of edges that are marked for removal
            to_remove = set()

            # Run through all unshielded triples in M
            for (A, B, C) in M:

                 # Unpack A, B, C
                (i, lag_i) = A
                (j, lag_j) = B
                (k, lag_k) = C

                # Get all minimal separating sets in SepSet(A, C)
                sepsets = self._get_sepsets(A, C)
                sepsets = {Z for (Z, status) in sepsets if status == "m"}

                ###############################################################################################################
                ###############################################################################################################

                remove_AB = False
                link_AB = self._get_link(A, B)

                # Test A indep B given union(SepSet(A, C), intersection(def-anc(B), adj(B))) setminus{A, B} setminus{future of both A and B}

                # Shift the lags appropriately
                if lag_i <= lag_j:
                    X = (i, lag_i - lag_j) # A shifted
                    Y = (j, 0) # B shifted
                    delta_lag = lag_j

                else:
                    X = (j, lag_j - lag_i) # B shifted
                    Y = (i, 0) # A shifted
                    delta_lag = lag_i

                # Run through all minimal separating sets of A and C
                for Z in sepsets:      

                    # Construct the conditioning set to test
                    # Take out future elements
                    Z_test = {(var, lag - delta_lag) for (var, lag) in Z if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}

                    # Test conditional independence of X and Y given Z,
                    val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), tau_max = self.tau_max)

                    if self.verbosity >= 2:
                        print("UTR:    %s _|_ %s  |  Z_test = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z_test)]), val, pval))

                    # Update val_min and pval_max
                    self._update_val_min(X, Y, val)
                    self._update_pval_max(X, Y, pval)
                    self._update_cardinality(X, Y, len(Z_test))

                    # Check whether the test result was significant
                    if pval > self.pc_alpha:
                        # Mark the edge from X to Y for removal
                        remove_AB = True
                        # Store Z as a non-weakly-minimal separating set of X and Y
                        self._save_sepset(X, Y, (frozenset(Z_test), "nm"))

                if remove_AB:
                    # Remember the edge for removal
                    pair_key, new_link = self._get_pair_key_and_new_link(A, B, "")
                    to_remove.add((X, Y))

                ###############################################################################################################
                ###############################################################################################################

                remove_CB = False
                link_CB = self._get_link(C, B)

                # Test C indep B given union(SepSet(A, C), intersection(def-anc(B), adj(B))) setminus{A, B} setminus{future of both C and B}

                # Shift the lags appropriately
                if lag_k <= lag_j:
                    X = (k, lag_k - lag_j)
                    Y = (j, 0)
                    delta_lag = lag_j
                else:
                    X = (j, lag_j - lag_k)
                    Y = (k, 0)
                    delta_lag = lag_k

                # Run through all minimal separating sets of A and C
                for Z in sepsets:

                    # Construct the conditioning set to test
                    # Take out future elements
                    Z_test = {(var, lag - delta_lag) for (var, lag) in Z if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}

                    # Test conditional independence of X and Y given Z,
                    val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z_test), tau_max = self.tau_max)

                    if self.verbosity >= 2:
                        print("UTR:    %s _|_ %s  |  Z_test = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z_test)]), val, pval))

                    # Update val_min and pval_max
                    self._update_val_min(X, Y, val)
                    self._update_pval_max(X, Y, pval)
                    self._update_cardinality(X, Y, len(Z_test))

                    # Check whether the test result was significant
                    if pval > self.pc_alpha:
                        # Mark the edge from X to Y for removal
                        remove_CB = True
                        # Store Z as a non-weakly-minimal separating set of X and Y
                        self._save_sepset(X, Y, (frozenset(Z_test), "nm"))

                if remove_CB:
                    # Remember the edge for removal
                    pair_key, new_link = self._get_pair_key_and_new_link(C, B, "")
                    to_remove.add((X, Y))

                ###############################################################################################################
                ###############################################################################################################

                if not remove_AB and not remove_CB and not link_AB[2] in ["-", "x"] and not link_CB[2] in ["-", "x"] and not (link_AB[2] == ">" and link_CB[2] == ">"):
                    
                    L.add((A, B, C))

            # end for (A, B, C) in M

            ###################################################################################################################
            ###################################################################################################################

            # Remove edges marked for removal
            for (X, Y) in to_remove:
                self._write_link(X, Y, "", verbosity = self.verbosity)

            # Make sepsets minimal (here, this agrees with minimal)
            for (X, Y) in to_remove:

                # Read out all separating sets that were found in the rule phase, then consider only those of minimal cardinality
                old_sepsets_all = {Z for (Z, _) in self._get_sepsets(X, Y)}
                min_size = min({len(Z) for Z in old_sepsets_all})
                old_sepsets_smallest = {Z for Z in old_sepsets_all if len(Z) == min_size}

                # For all separating sets of minimal cardinality, find minimal separating subsets in an order independent way
                self._delete_sepsets(X, Y)
                self._make_sepset_minimal(X, Y, old_sepsets_smallest)

            # Find new unshielded triples and determine the new "M"
            new_unshielded_triples = set(self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik=''))
            M = new_unshielded_triples.difference(old_unshielded_triples)

            # Take care that not both (A, B, C) and (C, B, A) appear in M
            M_unique = set()
            for (A, B, C) in M:
                if not (C, B, A) in M_unique:
                    M_unique.add((A, B, C))
            M = M_unique

        # end while len(M) > 0

        #######################################################################################################################
        #######################################################################################################################

        # Remove all elements from L that are no langer part of an unshielded triple
        L_final = {(A, B, C) for (A, B, C) in L if self._get_link(A, B) != "" and self._get_link(C, B) != ""}

        # Run through all these triples and test for orientation as collider
        to_orient = []
        for (A, B, C) in L_final:

            if self._B_not_in_SepSet_AC(A, B, C):

                link_AB = self._get_link(A, B)
                link_CB = self._get_link(C, B)

                # Prepare the new links and save them to the output
                if link_AB[2] != ">":
                    new_link_AB = link_AB[0] + link_AB[1] + ">"
                    to_orient.append(self._get_pair_key_and_new_link(A, B, new_link_AB))

                new_link_CB = link_CB[0] + link_CB[1] + ">"
                if link_CB[2] != ">":
                    to_orient.append(self._get_pair_key_and_new_link(C, B, new_link_CB))

        # Verbose output
        if self.verbosity >= 1:
            print("\nUTR")
            for ((i, j, lag_i), new_link) in set(to_orient):
                print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Marked:", i, lag_i, self._get_link((i, lag_i), (j, 0)), j, 0,i, lag_i, new_link, j, 0))
            if len(to_orient) == 0:
                print("Found nothing")

        # Return if no orientations were found
        if len(to_orient) == 0:
        	return False

        # Aggreate orienations
        new_ancs = {j: set() for j in range(self.N)}
        new_non_ancs = {j: set() for j in range(self.N)}

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

        # Make the orientations
        self._apply_new_ancestral_information(new_non_ancs, new_ancs)

        # Return True
        return True


    def _run_rfci_dpr_rule(self):
        """Run the RFCI discriminating path rule: Lines 3 - 29 in algorithm 4.5 of the RFCI supplement paper"""

        # Verbose output
        if self.verbosity >= 1:
            print("\nStarting RFCI DPR-Rule:")

        # Find all relevant triangles W-V-Y
        triangles = set(self._find_triples(pattern_ij='<**', pattern_jk='o*+', pattern_ik='-*>'))

        # Verbose output
        if self.verbosity >= 1 and len(triangles) == 0:
            print("\nFound no suitable triangles")

        # Run through all triangles
        while len(triangles) > 0:

            # Remember all paths that qualify for the orientation test
            paths_to_test_for_orientation = dict()

            # Remember edges marked for removal
            to_remove = set()

            # Run through all of these triangles
            for (W, V, Y_path) in triangles:

                # Find all discriminating paths for this triangle, then consider only the shortest paths
                discriminating_paths = self._get_R4_discriminating_paths_rfci((W, V, Y_path), max_length = np.inf)

                # If there is no discriminating path, continue with the next triple
                if len(discriminating_paths) == 0:
                    continue

                # Only consider shortests discrimintating paths
                min_len = min([len(path) for path in discriminating_paths])
                shortest_discriminating_paths = [path for path in discriminating_paths if len(path) == min_len]

                # Run through all shortests discriminating paths
                for path in shortest_discriminating_paths:

                    path_disqualified = False

                    # Get the separating set between the end points
                    X_1 = path[-1]
                    all_sepsets = {Z for (Z, _) in self._get_sepsets(X_1, Y_path)}

                    # Run through all pairs of adjancent variables on path
                    for i in range(min_len - 1):

                        # Read out the current pair of adjacent variables
                        (var_A, lag_A) = path[i]
                        (var_B, lag_B) = path[i + 1]

                        # Time shift accordingly
                        if lag_A <= lag_B:
                            X = (var_A, lag_A - lag_B) # A shifted
                            Y = (var_B, 0) # B shifted
                            delta_lag = lag_B

                        else:
                            X = (var_B, lag_B - lag_A) # B shifted
                            Y = (var_A, 0) # A shifted
                            delta_lag = lag_A

                        # Run through all sepsets
                        for S_ik in all_sepsets:

                            # Time shift the separating set
                            S_ik_shift = {(var, lag - delta_lag) for (var, lag) in S_ik if lag - delta_lag <= 0 and lag - delta_lag >= -self.tau_max}

                            # Run through all subsets of S_ik
                            for p in range(len(S_ik) + 1):
                                for Z in combinations(S_ik_shift, p):

                                    # HACK
                                    val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = list(Z), tau_max = self.tau_max)

                                    if self.verbosity >= 2:
                                        print("DPR:    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                                            (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                                    # Update val_min and pval_max
                                    self._update_val_min(X, Y, val)
                                    self._update_pval_max(X, Y, pval)
                                    self._update_cardinality(X, Y, len(Z))

                                    # Check whether the test result was significant
                                    if pval > self.pc_alpha:
                                        # Mark the edge from X to Y for removal and store Z as a weakly-minimal separating set of X and Y
                                        to_remove.add((X, Y))
                                        self._save_sepset(X, Y, (frozenset(Z), "m"))

                                        # Break the inner most for loops
                                        path_disqualified = True
                                        break

                                if path_disqualified:
                                    break

                                # end for Z in combinations(S_ik_shift, p)
                            # end for p in range(len(S_ik) + 1)
                        # end for S_ik in all_sepsets

                        # If the path has been disqualifed, break the for loop through adjacent pairs on the path
                        if path_disqualified:
                            break

                    # end for i in range(min_len - 1)
                    if not path_disqualified:
                        if (W, V, Y_path) in paths_to_test_for_orientation.keys():
                            paths_to_test_for_orientation[(W, V, Y_path)].append(path)
                        else:
                            paths_to_test_for_orientation[(W, V, Y_path)] = [path]

                # end for path in shortest_discriminating_paths
            # end for (W, V, Y) in triangles

            # Remember unshielded triples at this point
            old_unshielded_triples = set(self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik=''))

            # Delete all edges that are marked for removal
            for (X, Y) in to_remove:
                self._write_link(X, Y, "", verbosity = self.verbosity)

            # Determine the unshielded triples
            new_unshielded_triples = set(self._find_triples(pattern_ij='***', pattern_jk='***', pattern_ik=''))
            new_unshielded_triples = new_unshielded_triples.difference(old_unshielded_triples)

            # Run the RFCI unshielded triple rule on the new unshielded triples
            restart = self._run_rfci_utr_rule(new_unshielded_triples)

            # Keep only those qualfied paths that are still paths
            final_paths = dict()
            for (key, path_list) in paths_to_test_for_orientation.items():

                disqualifed = False

                for path in path_list:

                    for i in range(len(path) - 1):
                        if self._get_link(path[i], path[i+1]) == "":
                            disqualifed = True
                            break
                    if disqualifed:
                        continue

                    if key in final_paths.keys():
                        final_paths[key].append(path)
                    else:
                        final_paths[key] = [path]

            # Subject the surviving paths to the orientation test
            to_orient = []
            for (key, path_list) in final_paths.items():
                for path in path_list:

                    # Get the initial triangle
                    Y = path[0]
                    V = path[1]
                    W = path[2]

                    # Get the end point node
                    X_1 = path[-1]

                    # Get the current link from W to V, which we will need below
                    link_WV = self._get_link(W, V)

                    # Check which of the two cases of the rule we are in, then append the appropriate new links to the output list
                    if self._B_in_SepSet_AC(X_1, V, Y):
                        # New link from V to Y
                        to_orient.append(self._get_pair_key_and_new_link(V, Y, "-->"))

                    elif link_WV != "<-x" and self._B_not_in_SepSet_AC(X_1, V, Y):
                        # New link from V to Y
                        to_orient.append(self._get_pair_key_and_new_link(V, Y, "<->"))

                        # If needed, also the new link from W to V
                        if link_WV != "<->":
                            to_orient.append(self._get_pair_key_and_new_link(W, V, "<->"))

            # Verbose output
            if self.verbosity >= 1:
                print("\nDPR")
                for ((i, j, lag_i), new_link) in set(to_orient):
                    print("{:10} ({},{:2}) {:3} ({},{:2}) ==> ({},{:2}) {:3} ({},{:2}) ".format("Marked:", i, lag_i, self._get_link((i, lag_i), (j, 0)), j, 0,i, lag_i, new_link, j, 0))
                if len(to_orient) == 0:
                    print("Found nothing")

            # Return if neither UTR nor DPR found anything
            if not restart and len(to_orient) == 0:
            	return True

            # Aggreate orienations
            new_ancs = {j: set() for j in range(self.N)}
            new_non_ancs = {j: set() for j in range(self.N)}

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

            # Make the orientations
            self._apply_new_ancestral_information(new_non_ancs, new_ancs)

            # Check for the new relevant triangles
            new_triangles = set(self._find_triples(pattern_ij='<**', pattern_jk='o*+', pattern_ik='-*>'))
            triangles = new_triangles.difference(triangles)

        # end while len(triangles) > 0


    def _make_sepset_minimal(self, X, Y, Z_list):
        """
        X and Y are conditionally independent given Z in Z_list However, it is not yet clear whether any of these Z are minimal separating set.

        This function finds minimal separating subsets in an order independent way and writes them to the self.sepsets dictionary. Only those sets which are minimal separating sets are kept.
        """

        # Base Case 1:
        # Z in Z_list is minimal if len(Z) <= 1 or Z \subset ancs
        any_minimal = False

        for Z in Z_list:

            if len(Z) <=1:
                self._save_sepset(X, Y, (frozenset(Z), "m"))
                any_minimal = True

        if any_minimal:
            return None

        # If not Base Case 1, we need to search for separating subsets. We do this for all Z in Z_list, and build a set sepsets_next_call that contains all separating sets for the next recursive call
        sepsets_next_call = set()

        for Z in Z_list:

            # Test for removal of all nodes in removable
            new_sepsets = []
            val_values = []

            for A in Z:

                Z_A = [node for node in Z if node != A]

                # Run the conditional independence test
                val, pval = self.cond_ind_test.run_test(X = [X], Y = [Y], Z = Z_A, tau_max = self.tau_max)

                if self.verbosity >= 2:
                    print("MakeMin:    %s _|_ %s  |  Z_A = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z_A)]), val, pval))

                # Check whether the test result was significant
                if pval > self.pc_alpha:
                    new_sepsets.append(frozenset(Z_A))
                    val_values.append(val)

            # If new_sepsets is empty, then Z is already minimal
            if len(new_sepsets) == 0:
                self._save_sepset(X, Y, (frozenset(Z), "m"))
                any_minimal = True

            # If we did not yet find a minimal separating set
            if not any_minimal:

                # Sort all separating sets in new_sepets by their test statistic, then append those separating sets with maximal statistic to sepsets_next_call. This i) guarantees order independence while ii) continuing to test as few as possible separating sets
                new_sepsets = [node for _, node in sorted(zip(val_values, new_sepsets), reverse = True)]

                i = -1
                while i <= len(val_values) - 2 and val_values[i + 1] == val_values[0]:
                    sepsets_next_call.add(new_sepsets[i])
                    i = i + 1

                assert i >= 0

        # If we did not yet find a minimal separating set, make a recursive call
        if not any_minimal:
            self._make_sepset_minimal(X, Y, sepsets_next_call)
        else:
            return None

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _run_orientation_phase(self, rule_list):
        """Function for exhaustive application of the orientation rules specified by rule_list."""

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
                orientations = self._apply_rule(rule)

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

    def _fix_all_edges(self):
        """Set the middle mark of all links to '-'"""

        for j in range(self.N):
                for (i, lag_i) in self.graph_dict[j].keys():

                    link = self._get_link((i, lag_i), (j, 0))
                    if len(link) > 0:
                        new_link = link[0] + "-" + link[2]
                        self.graph_dict[j][(i, lag_i)] = new_link


    def _fix_edges_with_tail(self):
        """Set the middle mark of all edges with a tail to '-', provided they are in self._can_fix. For an explanation of self._can_fix see _run_pc_removal_phase()"""

        for j in range(self.N):
                for (i, lag_i) in self.graph_dict[j].keys():

                    link = self._get_link((i, lag_i), (j, 0))
                    if len(link) > 0 and (link[0] == "-" or link[2] == "-") and (((i, lag_i), (j, 0)) in self._can_fix or ((j, 0), (i, lag_i)) in self._can_fix):
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


    def _apply_rule(self, rule):
        """Call the orientation-removal-rule specified by the string argument rule"""

        if rule == "R-01":
            return self._apply_R01()
        elif rule == "R-02":
            return self._apply_R02()
        elif rule == "R-03":
            return self._apply_R03()
        elif rule == "R-08":
            return self._apply_R08()
        elif rule == "R-09":
            return self._apply_R09()
        elif rule == "R-10":
            return self._apply_R10()


    def _B_not_in_SepSet_AC(self, A, B, C):
        """Return True if B is not in the separating set of A and C according to the standard majority rule."""

        # Treat A - B - C as the same triple as C - B - A
        # Convention: A is before C or, if they are contemporaneous, the index of A is smaller than that of C
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_not_in_SepSet_AC(C, B, A)

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

        # Return True if any separating set was found and B is in less than half of them
        return True if n_sepsets > 0 and 2*n_sepsets_with_B < n_sepsets else False


    def _B_in_SepSet_AC(self, A, B, C):
        """Return True if B is not in the separating set of A and C according to the standard majority rule on minimal separating sets"""

        # Treat A - B - C as the same triple as C - B - A
        # Convention: A is before C or, if they are contemporaneous, the index of A is smaller than that of C
        if C[1] < A[1] or (C[1] == A[1] and C[0] < A[0]):
            return self._B_not_in_SepSet_AC(C, B, A)

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

        # Remember whether any separating set is found in the below for loop
        sepset_found = False

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
                    print("BinSepSetAC(A):    %s _|_ %s  |  Z = %s: val = %.2f / pval = % .4f" %
                        (X, Y, ' '.join([str(z) for z in list(Z)]), val, pval))

                # Check whether the test result was significant. If yes, remember Z as separating set
                if pval > self.pc_alpha:
                    all_sepsets.add(frozenset(Z))

                    # To guarantee minimality of all separating sets, the for loop needs to be broken after this iteration
                    sepset_found = True

            # If a separating set has already been found, break the foor loop
            if sepset_found:
                break

        # Remember whether any separating set is found in the below for loop
        sepset_found = False

        # Test for independence given subsets of non-future adjacencies of A
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
                    sepset_found = True

            if sepset_found:
                break

        # Count number of sepsets and number of sepsets that contain B
        n_sepsets = len(all_sepsets)
        n_sepsets_with_B = len([1 for Z in all_sepsets if (B[0], B[1] - C[1]) in Z])

        # Return True if any separating set was found and B is in more than half of them
        return True if n_sepsets > 0 and 2*n_sepsets_with_B > n_sepsets else False

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    def _apply_R01(self):
        """Return all orientations implied by orientation rule R-01"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='*?>', pattern_jk='o?+', pattern_ik='')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # Check whether the rule applies
            if self._B_in_SepSet_AC(A, B, C):

                # Prepare the new link from B to C and append it to the output list
                link_BC = self._get_link(B, C)
                new_link_BC = "-" + link_BC[1] + ">"
                out.append(self._get_pair_key_and_new_link(B, C, new_link_BC))

        return out


    def _apply_R02(self):
        """Return all orientations implied by orientation rule R-02"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = set(self._find_triples(pattern_ij='-?>', pattern_jk='*?>', pattern_ik='+?o'))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='*?>', pattern_jk='-?>', pattern_ik='+?o')))

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # The rule applies to all relevant graphical structures. Therefore, prepare the new link and append it to the output list
            link_AC = self._get_link(A, C)
            new_link_AC = link_AC[0] + link_AC[1] + ">"
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))

        # Return the output list
        return out


    def _apply_R03(self):
        """Return all orientations implied by orientation rule R-03"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_quadruples = self._find_quadruples(pattern_ij='*?>', pattern_jk='<?*', pattern_ik='', 
                                                           pattern_il='+?o', pattern_jl='o?+', pattern_kl='+?o')

        # Run through all appropriate graphical structures
        for (A, B, C, D) in all_appropriate_quadruples:

            # Check whether the rule applies
            if self._B_in_SepSet_AC(A, D, C):

                # Prepare the new link from D to B and append it to the output list
                link_DB = self._get_link(D, B)
                new_link_DB = link_DB[0] + link_DB[1] + ">"
                out.append(self._get_pair_key_and_new_link(D, B, new_link_DB))

        # Return the output list
        return out


    def _apply_R08(self):
        """Return all orientations implied by orientation rule R-08"""

        # Build the output list
        out = []

        # Find all graphical structures that the rule applies to
        all_appropriate_triples = self._find_triples(pattern_ij='-?>', pattern_jk='-?>', pattern_ik='o?+')

        # Run through all appropriate graphical structures
        for (A, B, C) in all_appropriate_triples:

            # The rule applies to all relevant graphical structures. Therefore, prepare the new link and append it to the output list
            link_AC = self._get_link(A, C)
            new_link_AC = "-" + link_AC[1] + ">"
            out.append(self._get_pair_key_and_new_link(A, C, new_link_AC))

        # Return the output list
        return out


    def _apply_R09(self):
        """Return all orientations implied by orientation rule R-09"""

        # Build the output list
        out = []

        # Find unshielded triples B_1 o--*--o A o--*--> C or B_1 <--*--o A o--*--> C or B_1 <--*-- A o--*--> C 
        all_appropriate_triples = set(self._find_triples(pattern_ij='o?o', pattern_jk='o?>', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<?o', pattern_jk='o?>', pattern_ik='')))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='<?-', pattern_jk='o?>', pattern_ik='')))

        # Run through all these triples
        for (B_1, A, C) in all_appropriate_triples:

            # Check whether A is in SepSet(B_1, C), else the rule does not apply
            if not self._B_in_SepSet_AC(B_1, A, C):
                continue

            # Although we do not yet know whether the rule applies, we here determine the new form of the link from A to C if the rule does apply
            link_AC = self._get_link(A, C)
            new_link_AC = "-" + link_AC[1] + ">"
            pair_key, new_link = self._get_pair_key_and_new_link(A, C, new_link_AC)

            # For the search of uncovered potentially directed paths from B_1 to C, determine the initial pattern as dictated by the link from A to B_1
            first_link = self._get_link(A, B_1)
            if self._match_link(pattern='o?o', link=first_link):
                initial_allowed_patterns = ['-?>', 'o?>', 'o?o']
            elif self._match_link(pattern='o?>', link=first_link) or self._match_link(pattern='-?>', link=first_link):
                initial_allowed_patterns = ['-?>']
            
            # Find all uncovered potentially directed paths from B_1 to C
            uncovered_pd_paths = self._get_potentially_directed_uncovered_paths_rfci(B_1, C, initial_allowed_patterns)

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


    def _apply_R10(self):
        """Return all orientations implied by orientation rule R-10"""

        # Build the output list
        out = []

        # Find all triples A o--> C <-- P_C
        all_appropriate_triples = set(self._find_triples(pattern_ij='o?>', pattern_jk='<?-', pattern_ik=''))
        all_appropriate_triples = all_appropriate_triples.union(set(self._find_triples(pattern_ij='o?>', pattern_jk='<?-', pattern_ik='***')))

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
                for upd_path in self._get_potentially_directed_uncovered_paths_rfci(A, P_C, ['-?>', 'o?>', 'o?o']):

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

            # Check whether there is any pair of non-adjacent nodes in second_nodes, such that A is in their separating set. If yes, mark the link from A to C for orientation
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


    def _get_R4_discriminating_paths_rfci(self, triple, max_length = np.inf):
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

            if self._get_link(last_node, path_taken[-2])[0] == "<" and link_to_Y == "-?>" and len(path_taken) < max_length:

                # Search through all adjacencies of the last node
                for (var, lag) in self.graph_full_dict[last_node[0]].keys():

                    # Build the next node and get its link to the previous
                    next_node = (var, lag + last_node[1])
                    next_link = self._get_link(next_node, last_node)

                    # Check whether this node can be visited
                    if next_node[1] <= 0 and next_node[1] >= -self.tau_max and next_node not in path_taken and self._match_link("*?>", next_link):

                        # Recursive call
                        paths.extend(_search(path_taken[:] + [next_node], max_length))

            # Return the list of discriminating paths
            return paths

        # Unpack the triple
        (W, V, Y) = triple

        # Return all discriminating paths starting at this triple
        return _search([Y, V, W], max_length)


    def _get_potentially_directed_uncovered_paths_rfci(self, start_node, end_node, initial_allowed_patterns):
        """Find all potentiall directed uncoverged paths from start_node to end_node whose first link takes one the forms specified by initial_allowed_patters"""

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
                    if self._match_link(pattern='o?o', link=link):
                        new_allowed_patters = ["o?o", "o?>", "-?>"]
                    elif self._match_link(pattern='o?>', link=link) or self._match_link(pattern='-?>', link=link):
                        new_allowed_patters = ["-?>"]

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