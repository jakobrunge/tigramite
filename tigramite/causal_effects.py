"""Tigramite causal inference for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

import numpy as np
import math
import itertools
from copy import deepcopy
from collections import defaultdict
from tigramite.models import Models
from tigramite.graphs import Graphs
import struct

class CausalEffects(Graphs):
    r"""Causal effect estimation.

    Methods for the estimation of linear or non-parametric causal effects 
    between (potentially multivariate) X and Y (potentially conditional 
    on S) by (generalized) backdoor adjustment. Various graph types are 
    supported, also including hidden variables.
    
    Linear and non-parametric estimators are based on sklearn. For the 
    linear case without hidden variables also an efficient estimation 
    based on Wright's path coefficients is available. This estimator 
    also allows to estimate mediation effects.

    See the corresponding paper [6]_ and tigramite tutorial for an 
    in-depth introduction. 

    References
    ----------

    .. [6] J. Runge, Necessary and sufficient graphical conditions for
           optimal adjustment sets in causal graphical models with 
           hidden variables, Advances in Neural Information Processing
           Systems, 2021, 34 
           https://proceedings.neurips.cc/paper/2021/hash/8485ae387a981d783f8764e508151cd9-Abstract.html


    Parameters
    ----------
    graph : array of either shape [N, N], [N, N, tau_max+1], or [N, N, tau_max+1, tau_max+1]
        Different graph types are supported, see tutorial.
    X : list of tuples
        List of tuples [(i, -tau), ...] containing cause variables.
    Y : list of tuples
        List of tuples [(j, 0), ...] containing effect variables.
    S : list of tuples
        List of tuples [(i, -tau), ...] containing conditioned variables.  
    graph_type : str
        Type of graph.
    hidden_variables : list of tuples
        Hidden variables in format [(i, -tau), ...]. The internal graph is 
        constructed by a latent projection.
    check_SM_overlap : bool
        Whether to check whether S overlaps with M.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self,
                 graph,
                 graph_type,
                 X,
                 Y,
                 S=None,
                 hidden_variables=None,
                 check_SM_overlap=True,
                 verbosity=0):
        
        self.verbosity = verbosity
        self.N = graph.shape[0]

        if S is None:
            S = []

        self.listX = list(X)
        self.listY = list(Y)
        self.listS = list(S)

        self.X = set(X)
        self.Y = set(Y)
        self.S = set(S)    

        # TODO?: check that masking aligns with hidden samples in variables
        if hidden_variables is None:
            hidden_variables = []


        # Determine tau_max
        if graph_type in ['dag', 'admg']: 
            self.tau_max = 0

        elif graph_type in ['tsg_dag', 'tsg_admg']:
            # tau_max is implicitely derived from
            # the dimensions 
            self.tau_max = graph.shape[2] - 1

        elif graph_type in ['stationary_dag', 'stationary_admg']:
            # For a stationary DAG without hidden variables it's sufficient to consider
            # a tau_max that includes the parents of X, Y, M, and S. A conservative
            # estimate thereof is simply the lag-dimension of the stationary DAG plus
            # the maximum lag of XYS.
            statgraph_tau_max = graph.shape[2] - 1
            maxlag_XYS = 0
            for varlag in self.X.union(self.Y).union(self.S):
                maxlag_XYS = max(maxlag_XYS, abs(varlag[1]))
            self.tau_max = maxlag_XYS + statgraph_tau_max

        self.hidden_variables = set(hidden_variables)
        if len(self.hidden_variables.intersection(self.X.union(self.Y).union(self.S))) > 0:
            raise ValueError("XYS overlaps with hidden_variables!")

        # self.tau_max is needed in the Graphs class
        Graphs.__init__(self, 
                        graph=graph,
                        graph_type=graph_type,
                        tau_max=self.tau_max,
                        hidden_variables=self.hidden_variables,
                        verbosity=verbosity)

        self._check_XYS()

        self.ancX = self._get_ancestors(X)
        self.ancY = self._get_ancestors(Y)
        self.ancS = self._get_ancestors(S)

        # If X is not in anc(Y), then no causal link exists
        if self.ancY.intersection(set(X)) == set():
            self.no_causal_path = True
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
        else:
            self.no_causal_path = False

        # Get mediators
        mediators = self.get_mediators(start=self.X, end=self.Y) 

        M = set(mediators)
        self.M = M

        self.listM = list(self.M)

        for varlag in self.X.union(self.Y).union(self.S):
            if abs(varlag[1]) > self.tau_max:
                raise ValueError("X, Y, S must have time lags inside graph.")

        if len(self.X.intersection(self.Y)) > 0:
            raise ValueError("Overlap between X and Y.")

        if len(self.S.intersection(self.Y.union(self.X))) > 0:
            raise ValueError("Conditions S overlap with X or Y.")

        # # TODO: need to prove that this is sufficient for non-identifiability!
        # if len(self.X.intersection(self._get_descendants(self.M))) > 0:
        #     raise ValueError("Not identifiable: Overlap between X and des(M)")

        if check_SM_overlap and len(self.S.intersection(self.M)) > 0:
            raise ValueError("Conditions S overlap with mediators M.")

        self.desX = self._get_descendants(self.X)
        self.desY = self._get_descendants(self.Y)
        self.desM = self._get_descendants(self.M)
        self.descendants = self.desY.union(self.desM)

        # Define forb as X and descendants of YM
        self.forbidden_nodes = self.descendants.union(self.X)  #.union(S)

        # Define valid ancestors
        self.vancs = self.ancX.union(self.ancY).union(self.ancS) - self.forbidden_nodes

        if self.verbosity > 0:
            if len(self.S.intersection(self.desX)) > 0:
                print("Warning: Potentially outside assumptions: Conditions S overlap with des(X)")

        # Here only check if S overlaps with des(Y), leave the option that S
        # contains variables in des(M) to the user
        if len(self.S.intersection(self.desY)) > 0:
            raise ValueError("Not identifiable: Conditions S overlap with des(Y).")

        if self.verbosity > 0:
            print("\n##\n## Initializing CausalEffects class\n##"
                  "\n\nInput:")
            print("\ngraph_type = %s" % graph_type
                  + "\nX = %s" % self.listX
                  + "\nY = %s" % self.listY
                  + "\nS = %s" % self.listS
                  + "\nM = %s" % self.listM
                  )
            if len(self.hidden_variables) > 0:
                print("\nhidden_variables = %s" % self.hidden_variables
                      ) 
            print("\n\n")
            if self.no_causal_path:
                print("No causal path from X to Y exists!")


    def _check_XYS(self):
        """Check whether XYS are sober.
        """

        XYS = self.X.union(self.Y).union(self.S)
        for xys in XYS:
            var, lag = xys 
            if var < 0 or var >= self.N:
                raise ValueError("XYS vars must be in [0...N]")
            if lag < -self.tau_max or lag > 0:
                raise ValueError("XYS lags must be in [-taumax...0]")


    def check_XYS_paths(self):
        """Check whether one can remove nodes from X and Y with no proper causal paths.

        Returns
        -------
        X, Y : cleaned lists of X and Y with irrelevant nodes removed.
        """

        # TODO: Also check S...
        oldX = self.X.copy()
        oldY = self.Y.copy()

        # anc_Y = self._get_ancestors(self.Y)
        # anc_S = self._get_ancestors(self.S)

        # Remove first from X those nodes with no causal path to Y or S
        X = set([x for x in self.X if x in self.ancY.union(self.ancS)])
        
        # Remove from Y those nodes with no causal path from X
        # des_X = self._get_descendants(X)

        Y = set([y for y in self.Y if y in self.desX])

        # Also require that all x in X have proper path to Y or S,
        # that is, the first link goes out of x 
        # and into path nodes
        mediators_S = self.get_mediators(start=self.X, end=self.S)
        path_nodes = list(self.M.union(Y).union(mediators_S)) 
        X = X.intersection(self._get_all_parents(path_nodes))

        if set(oldX) != set(X) and self.verbosity > 0:
            print("Consider pruning X = %s to X = %s " %(oldX, X) +
                  "since only these have causal path to Y")

        if set(oldY) != set(Y) and self.verbosity > 0:
            print("Consider pruning Y = %s to Y = %s " %(oldY, Y) +
                  "since only these have causal path from X")

        return (list(X), list(Y))


    def get_optimal_set(self, 
        alternative_conditions=None,
        minimize=False,
        return_separate_sets=False,
        ):
        """Returns optimal adjustment set.
        
        See Runge NeurIPS 2021.

        Parameters
        ----------
        alternative_conditions : set of tuples
            Used only internally in optimality theorem. If None, self.S is used.
        minimize : {False, True, 'colliders_only'} 
            Minimize optimal set. If True, minimize such that no subset 
            can be removed without making it invalid. If 'colliders_only',
            only colliders are minimized.
        return_separate_sets : bool
            Whether to return tuple of parents, colliders, collider_parents, and S.
        
        Returns
        -------
        Oset_S : False or list or tuple of lists
            Returns optimal adjustment set if a valid set exists, otherwise False.
        """


        # Needed for optimality theorem where Osets for alternative S are tested
        if alternative_conditions is None:
            S = self.S.copy()
            vancs = self.vancs.copy()
        else:
            S = alternative_conditions
            newancS = self._get_ancestors(S)
            vancs = self.ancX.union(self.ancY).union(newancS) - self.forbidden_nodes

            # vancs = self._get_ancestors(list(self.X.union(self.Y).union(S))) - self.forbidden_nodes

        # descendants = self._get_descendants(self.Y.union(self.M))

        # Sufficient condition for non-identifiability
        if len(self.X.intersection(self.descendants)) > 0:
            return False  # raise ValueError("Not identifiable: Overlap between X and des(M)")

        ##
        ## Construct O-set
        ##

        # Start with parents 
        parents = self._get_all_parents(self.Y.union(self.M)) # set([])

        # Remove forbidden nodes
        parents = parents - self.forbidden_nodes

        # Construct valid collider path nodes
        colliders = set([])
        for w in self.Y.union(self.M):
            j, tau = w 
            this_level = [w]
            non_suitable_nodes = []
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    suitable_spouses = set(self._get_spouses(varlag)) - set(non_suitable_nodes)
                    for spouse in suitable_spouses:
                        i, tau = spouse
                        if spouse in self.X:
                            return False

                        if (# Node not already in set
                            spouse not in colliders  #.union(parents)
                            # not forbidden
                            and spouse not in self.forbidden_nodes 
                            # in time bounds
                            and (-self.tau_max <= tau <= 0) # or self.ignore_time_bounds)
                            and (spouse in vancs
                                or not self._check_path(#graph=self.graph, 
                                    start=self.X, end=[spouse], 
                                                    conditions=list(parents.union(vancs)) + list(S),
                                                    ))
                                ):
                                colliders = colliders.union(set([spouse]))
                                next_level.append(spouse)
                        else:
                            if spouse not in colliders:
                                non_suitable_nodes.append(spouse)


                this_level = set(next_level) - set(non_suitable_nodes)  

        # Add parents and raise Error if not identifiable
        collider_parents = self._get_all_parents(colliders)
        if len(self.X.intersection(collider_parents)) > 0:
            return False

        colliders_and_their_parents = colliders.union(collider_parents)

        # Add valid collider path nodes and their parents
        Oset = parents.union(colliders_and_their_parents)


        if minimize: 
            removable = []
            # First remove all those that have no path from X
            sorted_Oset =  Oset
            if minimize == 'colliders_only':
                sorted_Oset = [node for node in sorted_Oset if node not in parents]

            for node in sorted_Oset:
                if (not self._check_path(#graph=self.graph, 
                    start=self.X, end=[node], 
                                conditions=list(Oset - set([node])) + list(S))):
                    removable.append(node) 

            Oset = Oset - set(removable)
            if minimize == 'colliders_only':
                sorted_Oset = [node for node in Oset if node not in parents]

            removable = []
            # Next remove all those with no direct connection to Y
            for node in sorted_Oset:
                if (not self._check_path(#graph=self.graph, 
                    start=[node], end=self.Y, 
                            conditions=list(Oset - set([node])) + list(S) + list(self.X),
                            ends_with=['**>', '**+'])): 
                    removable.append(node) 

            Oset = Oset - set(removable)

        Oset_S = Oset.union(S)

        if return_separate_sets:
            return parents, colliders, collider_parents, S
        else:
            return list(Oset_S)


    def _get_collider_paths_optimality(self, source_nodes, target_nodes,
        condition, 
        inside_set=None, 
        start_with_tail_or_head=False, 
        ):
        """Returns relevant collider paths to check optimality.

        Iterates over collider paths within O-set via depth-first search

        """

        for w in source_nodes:
            # Only used to return *all* collider paths 
            # (needed in optimality theorem)
            
            coll_path = []

            queue = [(w, coll_path)]

            non_valid_subsets = []

            while queue:

                varlag, coll_path = queue.pop()

                coll_path = coll_path + [varlag]

                suitable_nodes = set(self._get_spouses(varlag))

                if start_with_tail_or_head and coll_path == [w]:
                    children = set(self._get_children(varlag))
                    suitable_nodes = suitable_nodes.union(children)
 
                for node in suitable_nodes:
                    i, tau = node
                    if ((-self.tau_max <= tau <= 0) # or self.ignore_time_bounds)
                        and node not in coll_path):

                        if condition == 'II' and node not in target_nodes and node not in self.vancs:
                            continue

                        if node in inside_set:
                            if condition == 'I':
                                non_valid = False
                                for pathset in non_valid_subsets[::-1]:
                                    if set(pathset).issubset(set(coll_path + [node])):
                                        non_valid = True
                                        break
                                if non_valid is False:
                                    queue.append((node, coll_path)) 
                                else:
                                    continue
                            elif condition == 'II':
                                queue.append((node, coll_path))

                        if node in target_nodes:  
                            # yield coll_path
                            # collider_paths[node].append(coll_path) 
                            if condition == 'I':         
                                # Construct OπiN
                                Sprime = self.S.union(coll_path)
                                OpiN = self.get_optimal_set(alternative_conditions=Sprime)
                                if OpiN is False:
                                    queue = [(q_node, q_path) for (q_node, q_path) in queue if set(coll_path).issubset(set(q_path + [q_node])) is False]
                                    non_valid_subsets.append(coll_path)
                                else:
                                    return False

                            elif condition == 'II':
                                return True
                                # yield coll_path
 
        if condition == 'I':
            return True
        elif condition == 'II':
            return False
        # return collider_paths


    def check_optimality(self):
        """Check whether optimal adjustment set exists according to Thm. 3 in Runge NeurIPS 2021.

        Returns
        -------
        optimality : bool
            Returns True if an optimal adjustment set exists, otherwise False.
        """

        # Cond. 0: Exactly one valid adjustment set exists
        cond_0 = (self._get_all_valid_adjustment_sets(check_one_set_exists=True))

        #
        # Cond. I
        #
        parents, colliders, collider_parents, _ = self.get_optimal_set(return_separate_sets=True)
        Oset = parents.union(colliders).union(collider_parents)
        n_nodes = self._get_all_spouses(self.Y.union(self.M).union(colliders)) - self.forbidden_nodes - Oset - self.S - self.Y - self.M - colliders

        if (len(n_nodes) == 0):
            # # (1) There are no spouses N ∈ sp(YMC) \ (forbOS)
            cond_I = True
        else:
            
            # (2) For all N ∈ N and all its collider paths i it holds that 
            # OπiN does not block all non-causal paths from X to Y
            # cond_I = True
            cond_I = self._get_collider_paths_optimality(
                source_nodes=list(n_nodes), target_nodes=list(self.Y.union(self.M)),
                condition='I', 
                inside_set=Oset.union(self.S), start_with_tail_or_head=False,
                )

        #
        # Cond. II
        #
        e_nodes = Oset.difference(parents)
        cond_II = True
        for E in e_nodes:
            Oset_minusE = Oset.difference(set([E]))
            if self._check_path(#graph=self.graph, 
                start=list(self.X), end=[E], 
                                conditions=list(self.S) + list(Oset_minusE)):
                   
                cond_II = self._get_collider_paths_optimality(
                    target_nodes=self.Y.union(self.M), 
                    source_nodes=list(set([E])),
                    condition='II', 
                    inside_set=list(Oset.union(self.S)),
                    start_with_tail_or_head = True)
               
                if cond_II is False:
                    if self.verbosity > 1:
                        print("Non-optimal due to E = ", E)
                    break

        optimality = (cond_0 or (cond_I and cond_II))
        if self.verbosity > 0:
            print("Optimality = %s with cond_0 = %s, cond_I = %s, cond_II = %s"
                    %  (optimality, cond_0, cond_I, cond_II))
        return optimality

    def _check_validity(self, Z):
        """Checks whether Z is a valid adjustment set."""

        # causal_children = list(self.M.union(self.Y))
        backdoor_path = self._check_path(#graph=self.graph, 
            start=list(self.X), end=list(self.Y), 
                            conditions=list(Z), 
                            # causal_children=causal_children,
                            path_type = 'non_causal')

        if backdoor_path:
            return False
        else:
            return True
    
    def _get_adjust_set(self, 
        minimize=False,
        ):
        """Returns Adjust-set.
        
        See van der Zander, B.; Liśkiewicz, M. & Textor, J.
        Separators and adjustment sets in causal graphs: Complete 
        criteria and an algorithmic framework 
        Artificial Intelligence, Elsevier, 2019, 270, 1-40

        """

        vancs = self.vancs.copy()

        if minimize:
            # Get removable nodes by computing minimal valid set from Z
            if minimize == 'keep_parentsYM':
                minimize_nodes = vancs - self._get_all_parents(list(self.Y.union(self.M)))

            else:
                minimize_nodes = vancs

            # Zprime2 = Zprime
            # First remove all nodes that have no unique path to X given Oset
            for node in minimize_nodes:
                # path = self.oracle.check_shortest_path(X=X, Y=[node], 
                #     Z=list(vancs - set([node])), 
                #     max_lag=None, 
                #     starts_with=None, #'arrowhead', 
                #     forbidden_nodes=None, #list(Zprime - set([node])), 
                #     return_path=False)
                path = self._check_path(#graph=self.graph, 
                    start=self.X, end=[node], 
                    conditions=list(vancs - set([node])), 
                     )
  
                if path is False:
                    vancs = vancs - set([node])

            if minimize == 'keep_parentsYM':
                minimize_nodes = vancs - self._get_all_parents(list(self.Y.union(self.M)))
            else:
                minimize_nodes = vancs

            # print(Zprime2) 
            # Next remove all nodes that have no unique path to Y given Oset_min
            # Z = Zprime2
            for node in minimize_nodes:

                path = self._check_path(#graph=self.graph, 
                    start=[node], end=self.Y, 
                    conditions=list(vancs - set([node])) + list(self.X),
                    )

                if path is False:
                   vancs = vancs - set([node])  

        if self._check_validity(list(vancs)) is False:
            return False
        else:
            return list(vancs)


    def _get_all_valid_adjustment_sets(self, 
        check_one_set_exists=False, yield_index=None):
        """Constructs all valid adjustment sets or just checks whether one exists.
        
        See van der Zander, B.; Liśkiewicz, M. & Textor, J.
        Separators and adjustment sets in causal graphs: Complete 
        criteria and an algorithmic framework 
        Artificial Intelligence, Elsevier, 2019, 270, 1-40

        """

        cond_set = set(self.S)
        all_vars = [(i, -tau) for i in range(self.N)
                    for tau in range(0, self.tau_max + 1)]

        all_vars_set = set(all_vars) - self.forbidden_nodes


        def find_sep(I, R):
            Rprime = R - self.X - self.Y
            # TODO: anteriors and NOT ancestors where
            # anteriors include --- links in causal paths
            # print(I)
            XYI = list(self.X.union(self.Y).union(I))
            # print(XYI)
            ancs = self._get_ancestors(list(XYI))
            Z = ancs.intersection(Rprime)
            if self._check_validity(Z) is False:
                return False
            else:
                return Z


        def list_sep(I, R):
            # print(find_sep(X, Y, I, R))
            if find_sep(I, R) is not False:
                # print(I,R)
                if I == R: 
                    # print('--->', I)
                    yield I
                else:
                    # Pick arbitrary node from R-I
                    RminusI = list(R - I)
                    # print(R, I, RminusI)
                    v = RminusI[0]
                    # print("here ", X, Y, I.union(set([v])), R)
                    yield from list_sep(I.union(set([v])), R)
                    yield from list_sep(I, R - set([v]))

        # print("all ", X, Y, cond_set, all_vars_set)
        all_sets = []
        I = cond_set
        R = all_vars_set
        for index, valid_set in enumerate(list_sep(I, R)):
            # print(valid_set)
            all_sets.append(list(valid_set))
            if check_one_set_exists and index > 0:
                break

            if yield_index is not None and index == yield_index:
                return valid_set

        if yield_index is not None:
            return None

        if check_one_set_exists:
            if len(all_sets) == 1:
                return True
            else:
                return False

        return all_sets


    def fit_total_effect(self,
        dataframe, 
        estimator,
        adjustment_set='optimal',
        conditional_estimator=None,  
        data_transform=None,
        mask_type=None,
        ignore_identifiability=False,
        ):
        """Returns a fitted model for the total causal effect of X on Y 
           conditional on S.

        Parameters
        ----------
        dataframe : data object
            Tigramite dataframe object. It must have the attributes dataframe.values
            yielding a numpy array of shape (observations T, variables N) and
            optionally a mask of the same shape and a missing values flag.
        estimator : sklearn model object
            For example, sklearn.linear_model.LinearRegression() for a linear
            regression model.
        adjustment_set : str or list of tuples
            If 'optimal' the Oset is used, if 'minimized_optimal' the minimized Oset,
            and if 'colliders_minimized_optimal', the colliders-minimized Oset.
            If a list of tuples is passed, this set is used.
        conditional_estimator : sklearn model object, optional (default: None)
            Used to fit conditional causal effects in nested regression. 
            If None, the same model as for estimator is used.
        data_transform : sklearn preprocessing object, optional (default: None)
            Used to transform data prior to fitting. For example,
            sklearn.preprocessing.StandardScaler for simple standardization. The
            fitted parameters are stored.
        mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        ignore_identifiability : bool
            Only applies to adjustment sets supplied by user. Ignores if that 
            set leads to a non-identifiable effect.
        """

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return self

        self.dataframe = dataframe
        self.conditional_estimator = conditional_estimator

        # if self.dataframe.has_vector_data:
        #     raise ValueError("vector_vars in DataFrame cannot be used together with CausalEffects!"
        #                      " You can estimate vector-valued effects by using multivariate X, Y, S."
        #                      " Note, however, that this requires assuming a graph at the level "
        #                      "of the components of X, Y, S, ...")

        if self.N != self.dataframe.N:
            raise ValueError("Dataset dimensions inconsistent with number of variables in graph.")

        if adjustment_set == 'optimal':
            # Check optimality and use either optimal or colliders_only set
            adjustment_set = self.get_optimal_set()
        elif adjustment_set == 'colliders_minimized_optimal':
            adjustment_set = self.get_optimal_set(minimize='colliders_only')
        elif adjustment_set == 'minimized_optimal':
            adjustment_set = self.get_optimal_set(minimize=True)
        else:
            if ignore_identifiability is False and self._check_validity(adjustment_set) is False:
                raise ValueError("Chosen adjustment_set is not valid.")

        if adjustment_set is False:
            raise ValueError("Causal effect not identifiable via adjustment.")

        self.adjustment_set = adjustment_set

        # Fit model of Y on X and Z (and conditions)
        # Build the model
        self.model = Models(
                        dataframe=dataframe,
                        model=estimator,
                        conditional_model=conditional_estimator,
                        data_transform=data_transform,
                        mask_type=mask_type,
                        verbosity=self.verbosity)      

        self.model.get_general_fitted_model(
                Y=self.listY, X=self.listX, Z=list(self.adjustment_set),
                conditions=self.listS,
                tau_max=self.tau_max,
                cut_off='tau_max',
                return_data=False)

        return self

    def predict_total_effect(self, 
        intervention_data, 
        conditions_data=None,
        pred_params=None,
        return_further_pred_results=False,
        aggregation_func=np.mean,
        transform_interventions_and_prediction=False,
        ):
        """Predict effect of intervention with fitted model.

        Uses the model.predict() function of the sklearn model.

        Parameters
        ----------
        intervention_data : numpy array
            Numpy array of shape (time, len(X)) that contains the do(X) values.
        conditions_data : data object, optional
            Numpy array of shape (time, len(S)) that contains the S=s values.
        pred_params : dict, optional
            Optional parameters passed on to sklearn prediction function.
        return_further_pred_results : bool, optional (default: False)
            In case the predictor class returns more than just the expected value,
            the entire results can be returned.
        aggregation_func : callable
            Callable applied to output of 'predict'. Default is 'np.mean'.
        transform_interventions_and_prediction : bool (default: False)
            Whether to perform the inverse data_transform on prediction results.
        
        Returns
        -------
        Results from prediction: an array of shape  (time, len(Y)).
        If estimate_confidence = True, then a tuple is returned.
        """

        def get_vectorized_length(W):
            return sum([len(self.dataframe.vector_vars[w[0]]) for w in W])

        # lenX = len(self.listX)
        # lenS = len(self.listS)

        lenX = get_vectorized_length(self.listX)
        lenS = get_vectorized_length(self.listS)

        if intervention_data.shape[1] != lenX:
            raise ValueError("intervention_data.shape[1] must be len(X).")

        if conditions_data is not None and lenS > 0:
            if conditions_data.shape[1] != lenS:
                raise ValueError("conditions_data.shape[1] must be len(S).")
            if conditions_data.shape[0] != intervention_data.shape[0]:
                raise ValueError("conditions_data.shape[0] must match intervention_data.shape[0].")
        elif conditions_data is not None and lenS == 0:
            raise ValueError("conditions_data specified, but S=None or empty.")
        elif conditions_data is None and lenS > 0:
            raise ValueError("S specified, but conditions_data is None.")


        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return np.zeros((len(intervention_data), len(self.listY)))

        effect = self.model.get_general_prediction(
            intervention_data=intervention_data,
            conditions_data=conditions_data,
            pred_params=pred_params,
            return_further_pred_results=return_further_pred_results,
            transform_interventions_and_prediction=transform_interventions_and_prediction,
            aggregation_func=aggregation_func,) 

        return effect

    def fit_wright_effect(self,
        dataframe, 
        mediation=None,
        method='parents',
        links_coeffs=None,  
        data_transform=None,
        mask_type=None,
        ):
        """Returns a fitted model for the total or mediated causal effect of X on Y 
           potentially through mediator variables.

        Parameters
        ----------
        dataframe : data object
            Tigramite dataframe object. It must have the attributes dataframe.values
            yielding a numpy array of shape (observations T, variables N) and
            optionally a mask of the same shape and a missing values flag.
        mediation : None, 'direct', or list of tuples
            If None, total effect is estimated, if 'direct' then only the direct effect is estimated,
            else only those causal paths are considerd that pass at least through one of these mediator nodes.
        method : {'parents', 'links_coeffs', 'optimal'}
            Method to use for estimating Wright's path coefficients. If 'optimal', 
            the Oset is used, if 'links_coeffs', the coefficients in links_coeffs are used,
            if 'parents', the parents are used (only valid for DAGs).
        links_coeffs : dict
            Only used if method = 'links_coeffs'.
            Dictionary of format: {0:[((i, -tau), coeff),...], 1:[...],
            ...} for all variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float.
        data_transform : None
            Not implemented for Wright estimator. Complicated for missing samples.
        mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        """

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return self

        if data_transform is not None:
            raise ValueError("data_transform not implemented for Wright estimator."
                             " You can preprocess data yourself beforehand.")

        import sklearn.linear_model

        self.dataframe = dataframe
        if self.dataframe.has_vector_data:
            raise ValueError("vector_vars in DataFrame cannot be used together with Wright method!"
                             " You can either 1) estimate vector-valued effects by using multivariate (X, Y, S)"
                             " together with assuming a graph at the level of the components of (X, Y, S), "
                             " or 2) use vector_vars together with fit_total_effect and an estimator"
                             " that supports multiple outputs.")

        estimator = sklearn.linear_model.LinearRegression()

        # Fit model of Y on X and Z (and conditions)
        # Build the model
        self.model = Models(
                        dataframe=dataframe,
                        model=estimator,
                        data_transform=None, #data_transform,
                        mask_type=mask_type,
                        verbosity=self.verbosity)

        mediators = self.M  # self.get_mediators(start=self.X, end=self.Y)

        if mediation == 'direct':
            causal_paths = {}         
            for w in self.X:
                causal_paths[w] = {}
                for z in self.Y:
                    if w in self._get_parents(z):
                        causal_paths[w][z] = [[w, z]]
                    else:
                        causal_paths[w][z] = []
        else:
            causal_paths = self._get_causal_paths(source_nodes=self.X, 
                target_nodes=self.Y, mediators=mediators, 
                mediated_through=mediation, proper_paths=True)

        if method == 'links_coeffs':
            coeffs = {}
            max_lag = 0
            for medy in [med for med in mediators] + [y for y in self.listY]:
                coeffs[medy] = {}
                j, tauj = medy
                for ipar, par_coeff in enumerate(links_coeffs[medy[0]]):
                    par, coeff, _ = par_coeff
                    i, taui = par
                    taui_shifted = taui + tauj
                    max_lag = max(abs(par[1]), max_lag)
                    coeffs[medy][(i, taui_shifted)] = coeff #self.fit_results[j][(j, 0)]['model'].coef_[ipar]

            self.model.tau_max = max_lag
            # print(coeffs)

        elif method == 'optimal':
            # all_parents = {}
            coeffs = {}
            for medy in [med for med in mediators] + [y for y in self.listY]:
                coeffs[medy] = {}
                mediator_parents = self._get_all_parents([medy]).intersection(mediators.union(self.X).union(self.Y)) - set([medy])
                all_parents = self._get_all_parents([medy]) - set([medy])
                for par in mediator_parents:
                    Sprime = set(all_parents) - set([par, medy])
                    causal_effects = CausalEffects(graph=self.graph, 
                                        X=[par], Y=[medy], S=Sprime,
                                        graph_type=self.graph_type,
                                        check_SM_overlap=False,
                                        )
                    oset = causal_effects.get_optimal_set()
                    # print(medy, par, list(set(all_parents)), oset)
                    if oset is False:
                        raise ValueError("Not identifiable via Wright's method.")
                    fit_res = self.model.get_general_fitted_model(
                        Y=[medy], X=[par], Z=oset,
                        tau_max=self.tau_max,
                        cut_off='tau_max',
                        return_data=False)
                    coeffs[medy][par] = fit_res['model'].coef_[0]

        elif method == 'parents':
            coeffs = {}
            for medy in [med for med in mediators] + [y for y in self.listY]:
                coeffs[medy] = {}
                # mediator_parents = self._get_all_parents([medy]).intersection(mediators.union(self.X)) - set([medy])
                all_parents = self._get_all_parents([medy]) - set([medy])
                if 'dag' not in self.graph_type:
                    spouses = self._get_all_spouses([medy]) - set([medy])
                    if len(spouses) != 0:
                        raise ValueError("method == 'parents' only possible for "
                                         "causal paths without adjacent bi-directed links!")

                # print(j, all_parents[j])
                # if len(all_parents[j]) > 0:
                # print(medy, list(all_parents))
                fit_res = self.model.get_general_fitted_model(
                    Y=[medy], X=list(all_parents), Z=[],
                    conditions=None,
                    tau_max=self.tau_max,
                    cut_off='tau_max',
                    return_data=False)

                for ipar, par in enumerate(list(all_parents)):
                    # print(par, fit_res['model'].coef_)
                    coeffs[medy][par] = fit_res['model'].coef_[0][ipar]

        else:
            raise ValueError("method must be 'optimal', 'links_coeffs', or 'parents'.")
        
        # Effect is sum over products over all path coefficients
        # from x in X to y in Y
        effect = {}
        for (x, y) in itertools.product(self.listX, self.listY):
            effect[(x, y)] = 0.
            for causal_path in causal_paths[x][y]:
                effect_here = 1.
                # print(x, y, causal_path)
                for index, node in enumerate(causal_path[:-1]):
                    i, taui = node
                    j, tauj = causal_path[index + 1]
                    # tau_ij = abs(tauj - taui)
                    # print((j, tauj), (i, taui))
                    effect_here *= coeffs[(j, tauj)][(i, taui)]

                effect[(x, y)] += effect_here
               
        # Make fitted coefficients available as attribute
        self.coeffs = coeffs

        # Modify and overwrite variables in self.model
        self.model.Y = self.listY
        self.model.X = self.listX  
        self.model.Z = []
        self.model.conditions = [] 
        self.model.cut_off = 'tau_max' # 'max_lag_or_tau_max'

        class dummy_fit_class():
            def __init__(self, y_here, listX_here, effect_here):
                dim = len(listX_here)
                self.coeff_array = np.array([effect_here[(x, y_here)] for x in listX_here]).reshape(dim, 1)
            def predict(self, X):
                return np.dot(X, self.coeff_array).squeeze()

        fit_results = {}
        for y in self.listY:
            fit_results[y] = {}
            fit_results[y]['model'] = dummy_fit_class(y, self.listX, effect)
            fit_results[y]['data_transform'] = deepcopy(data_transform)

        # self.effect = effect
        self.model.fit_results = fit_results
        return self
 
    def predict_wright_effect(self, 
        intervention_data, 
        pred_params=None,
        ):
        """Predict linear effect of intervention with fitted Wright-model.

        Parameters
        ----------
        intervention_data : numpy array
            Numpy array of shape (time, len(X)) that contains the do(X) values.
        pred_params : dict, optional
            Optional parameters passed on to sklearn prediction function.

        Returns
        -------
        Results from prediction: an array of shape  (time, len(Y)).
        """

        lenX = len(self.listX)
        lenY = len(self.listY)

        if intervention_data.shape[1] != lenX:
            raise ValueError("intervention_data.shape[1] must be len(X).")

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return np.zeros((len(intervention_data), len(self.Y)))

        intervention_T, _ = intervention_data.shape


        predicted_array = np.zeros((intervention_T, lenY))
        pred_dict = {}
        for iy, y in enumerate(self.listY):
            # Print message
            if self.verbosity > 1:
                print("\n## Predicting target %s" % str(y))
                if pred_params is not None:
                    for key in list(pred_params):
                        print("%s = %s" % (key, pred_params[key]))
            # Default value for pred_params
            if pred_params is None:
                pred_params = {}
            # Check this is a valid target
            if y not in self.model.fit_results:
                raise ValueError("y = %s not yet fitted" % str(y))

            # data_transform is too complicated for Wright estimator
            # Transform the data if needed
            # fitted_data_transform = self.model.fit_results[y]['fitted_data_transform']
            # if fitted_data_transform is not None:
            #     intervention_data = fitted_data_transform['X'].transform(X=intervention_data)

            # Now iterate through interventions (and potentially S)
            for index, dox_vals in enumerate(intervention_data):
                # Construct XZS-array
                intervention_array = dox_vals.reshape(1, lenX) 
                predictor_array = intervention_array

                predicted_vals = self.model.fit_results[y]['model'].predict(
                X=predictor_array, **pred_params)
                predicted_array[index, iy] = predicted_vals.mean()

                # data_transform is too complicated for Wright estimator
                # if fitted_data_transform is not None:
                #     rescaled = fitted_data_transform['Y'].inverse_transform(X=predicted_array[index, iy].reshape(-1, 1))
                #     predicted_array[index, iy] = rescaled.squeeze()

        return predicted_array


    def fit_bootstrap_of(self, method, method_args, 
                        boot_samples=100,
                        boot_blocklength=1,
                        seed=None):
        """Runs chosen method on bootstrap samples drawn from DataFrame.
        
        Bootstraps for tau=0 are drawn from [max_lag, ..., T] and all lagged
        variables constructed in DataFrame.construct_array are consistently
        shifted with respect to this bootsrap sample to ensure that lagged
        relations in the bootstrap sample are preserved.

        This function fits the models, predict_bootstrap_of can then be used
        to get confidence intervals for the effect of interventions.

        Parameters
        ----------
        method : str
            Chosen method among valid functions in this class.
        method_args : dict
            Arguments passed to method.
        boot_samples : int
            Number of bootstrap samples to draw.
        boot_blocklength : int, optional (default: 1)
            Block length for block-bootstrap.
        seed : int, optional(default = None)
            Seed for RandomState (default_rng)
        """

        # if dataframe.analysis_mode != 'single':
        #     raise ValueError("CausalEffects class currently only supports single "
        #                      "datasets.")

        valid_methods = ['fit_total_effect',
                         'fit_wright_effect',
                          ]

        if method not in valid_methods:
            raise ValueError("method must be one of %s" % str(valid_methods))

        # First call the method on the original dataframe 
        # to make available adjustment set etc
        getattr(self, method)(**method_args)

        self.original_model = deepcopy(self.model)

        if self.verbosity > 0:
            print("\n##\n## Running Bootstrap of %s " % method +
                  "\n##\n" +
                  "\nboot_samples = %s \n" % boot_samples +
                  "\nboot_blocklength = %s \n" % boot_blocklength
                  )

        method_args_bootstrap = deepcopy(method_args)
        self.bootstrap_results = {}

        for b in range(boot_samples):
            # # Replace dataframe in method args by bootstrapped dataframe
            # method_args_bootstrap['dataframe'].bootstrap = boot_draw
            if seed is None:
                random_state = np.random.default_rng(None)
            else:
                random_state = np.random.default_rng(seed*boot_samples + b)

            method_args_bootstrap['dataframe'].bootstrap = {'boot_blocklength':boot_blocklength,
                                                            'random_state':random_state}

            # Call method and save fitted model
            getattr(self, method)(**method_args_bootstrap)
            self.bootstrap_results[b] = deepcopy(self.model)

        # Reset model
        self.model = self.original_model

        return self


    def predict_bootstrap_of(self, method, method_args, 
                        conf_lev=0.9,
                        return_individual_bootstrap_results=False):
        """Predicts with fitted bootstraps.

        To be used after fitting with fit_bootstrap_of. Only uses the 
        expected values of the predict function, not potential other output.

        Parameters
        ----------
        method : str
            Chosen method among valid functions in this class.
        method_args : dict
            Arguments passed to method.
        conf_lev : float, optional (default: 0.9)
            Two-sided confidence interval.
        return_individual_bootstrap_results : bool
            Returns the individual bootstrap predictions.

        Returns
        -------
        confidence_intervals : numpy array
        """

        valid_methods = ['predict_total_effect',
                         'predict_wright_effect',
                          ]

        if method not in valid_methods:
            raise ValueError("method must be one of %s" % str(valid_methods))

        # def get_vectorized_length(W):
        #     return sum([len(self.dataframe.vector_vars[w[0]]) for w in W])

        lenX = len(self.listX)
        lenS = len(self.listS)
        lenY = len(self.listY)

        intervention_T, _ = method_args['intervention_data'].shape

        boot_samples = len(self.bootstrap_results)
        # bootstrap_predicted_array = np.zeros((boot_samples, intervention_T, lenY))
        
        for b in range(boot_samples): #self.bootstrap_results.keys():
            self.model = self.bootstrap_results[b]
            boot_effect = getattr(self, method)(**method_args)

            if isinstance(boot_effect, tuple):
                boot_effect = boot_effect[0]
            
            if b == 0:
                bootstrap_predicted_array = np.zeros((boot_samples, ) + boot_effect.shape, 
                                            dtype=boot_effect.dtype)
            bootstrap_predicted_array[b] = boot_effect

        # Reset model
        self.model = self.original_model

        # Confidence intervals for val_matrix; interval is two-sided
        c_int = (1. - (1. - conf_lev)/2.)
        confidence_interval = np.percentile(
                bootstrap_predicted_array, axis=0,
                q = [100*(1. - c_int), 100*c_int])   #[:,:,0]

        if return_individual_bootstrap_results:
            return bootstrap_predicted_array, confidence_interval

        return confidence_interval


if __name__ == '__main__':
    
    # Consider some toy data
    import tigramite
    import tigramite.toymodels.structural_causal_processes as toys
    import tigramite.data_processing as pp
    import tigramite.plotting as tp
    from matplotlib import pyplot as plt
    import sys

    import sklearn
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor


    # def lin_f(x): return x
    # coeff = .5
 
    # links_coeffs = {0: [((0, -1), 0.5, lin_f)],
    #          1: [((1, -1), 0.5, lin_f), ((0, -1), 0.5, lin_f)],
    #          2: [((2, -1), 0.5, lin_f), ((1, 0), 0.5, lin_f)]
    #          }
    # T = 1000
    # data, nonstat = toys.structural_causal_process(
    #     links_coeffs, T=T, noises=None, seed=7)
    # dataframe = pp.DataFrame(data)

    # graph = CausalEffects.get_graph_from_dict(links_coeffs)

    # original_graph = np.array([[['', ''],
    #     ['-->', ''],
    #     ['-->', ''],
    #     ['', '']],

    #    [['<--', ''],
    #     ['', '-->'],
    #     ['-->', ''],
    #     ['-->', '']],

    #    [['<--', ''],
    #     ['<--', ''],
    #     ['', '-->'],
    #     ['-->', '']],

    #    [['', ''],
    #     ['<--', ''],
    #     ['<--', ''],
    #     ['', '-->']]], dtype='<U3')
    # graph = np.copy(original_graph)

    # # Add T <-> Reco and T 
    # graph[2,3,0] = '+->' ; graph[3,2,0] = '<-+'
    # graph[1,3,1] = '<->' #; graph[2,1,0] = '<--'

    # added = np.zeros((4, 4, 1), dtype='<U3')
    # added[:] = ""
    # graph = np.append(graph, added , axis=2)


    # X = [(1, 0)]
    # Y = [(3, 0)]

    # # # Initialize class as `stationary_dag`
    # causal_effects = CausalEffects(graph, graph_type='stationary_admg', 
    #                             X=X, Y=Y, S=None, 
    #                             hidden_variables=None, 
    #                             verbosity=0)

    # print(causal_effects.get_optimal_set())

    # tp.plot_time_series_graph(
    #     graph = graph,
    #     save_name='Example_graph_in.pdf',
    #     # special_nodes=special_nodes,
    #     # var_names=var_names,
    #     figsize=(6, 4),
    #     )

    # tp.plot_time_series_graph(
    #     graph = causal_effects.graph,
    #     save_name='Example_graph_out.pdf',
    #     # special_nodes=special_nodes,
    #     # var_names=var_names,
    #     figsize=(6, 4),
    #     )

    # causal_effects.fit_wright_effect(dataframe=dataframe, 
    #                         # links_coeffs = links_coeffs,
    #                         # mediation = [(1, 0), (1, -1), (1, -2)]
    #                         )

    # intervention_data = 1.*np.ones((1, 1))
    # y1 = causal_effects.predict_wright_effect( 
    #         intervention_data=intervention_data,
    #         )

    # intervention_data = 0.*np.ones((1, 1))
    # y2 = causal_effects.predict_wright_effect( 
    #         intervention_data=intervention_data,
    #         )

    # beta = (y1 - y2)
    # print("Causal effect is %.5f" %(beta))

    # tp.plot_time_series_graph(
    #     graph = causal_effects.graph,
    #     save_name='Example_graph.pdf',
    #     # special_nodes=special_nodes,
    #     var_names=var_names,
    #     figsize=(8, 4),
    #     )

    T = 10000
    def lin_f(x): return x

    auto_coeff = 0.
    coeff = 2.

    links = {
            0: [((0, -1), auto_coeff, lin_f)], 
            1: [((1, -1), auto_coeff, lin_f)], 
            2: [((2, -1), auto_coeff, lin_f), ((0, 0), coeff, lin_f)],
            3: [((3, -1), auto_coeff, lin_f)], 
            }
    data, nonstat = toys.structural_causal_process(links, T=T, 
                                noises=None, seed=7)


    # # Create some missing values
    # data[-10:,:] = 999.
    # var_names = range(2)

    dataframe = pp.DataFrame(data,
                    vector_vars={0:[(0,0), (1,0)], 
                                 1:[(2,0), (3,0)]}
                    )

    # # Construct expert knowledge graph from links here 
    aux_links = {0: [(0, -1)],
                 1: [(1, -1), (0, 0)],
              }
    # # Use staticmethod to get graph
    graph = CausalEffects.get_graph_from_dict(aux_links, tau_max=2)
    # graph = np.array([['', '-->'],
    #                   ['<--', '']], dtype='<U3')
    
    # # We are interested in lagged total effect of X on Y
    X = [(0, 0), (0, -1)]
    Y = [(1, 0), (1, -1)]

    # # Initialize class as `stationary_dag`
    causal_effects = CausalEffects(graph, graph_type='stationary_dag', 
                                X=X, Y=Y, S=None, 
                                hidden_variables=None, 
                                verbosity=1)

    # print(data)
    # # Optimal adjustment set (is used by default)
    # # print(causal_effects.get_optimal_set())

    # # # Fit causal effect model from observational data
    causal_effects.fit_total_effect(
        dataframe=dataframe, 
        # mask_type='y',
        estimator=LinearRegression(),
        )

    # # Fit causal effect model from observational data
    # causal_effects.fit_bootstrap_of(
    #     method='fit_total_effect',
    #     method_args={'dataframe':dataframe,  
    #     # mask_type='y',
    #     'estimator':LinearRegression()
    #     },
    #     boot_samples=3,
    #     boot_blocklength=1,
    #     seed=5
    #     )


    # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    lenX = 4 # len(dataframe.vector_vars[X[0][0]])
    dox_vals = np.linspace(0., 1., 3)
    intervention_data = np.tile(dox_vals.reshape(len(dox_vals), 1), lenX)

    intervention_data = np.array([[1., 0., 0., 0.]])

    print(intervention_data)

    pred_Y = causal_effects.predict_total_effect( 
            intervention_data=intervention_data)
    print(pred_Y, pred_Y.shape)





    # # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    # # dox_vals = np.array([1.]) #np.linspace(0., 1., 1)
    # intervention_data = np.tile(dox_vals.reshape(len(dox_vals), 1), len(X))
    # conf = causal_effects.predict_bootstrap_of(
    #     method='predict_total_effect',
    #     method_args={'intervention_data':intervention_data})
    # print(conf, conf.shape)



    # # # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    # # dox_vals = np.array([1.]) #np.linspace(0., 1., 1)
    # # intervention_data = dox_vals.reshape(len(dox_vals), len(X))
    # # pred_Y = causal_effects.predict_total_effect( 
    # #         intervention_data=intervention_data)
    # # print(pred_Y)



    # # Fit causal effect model from observational data
    # causal_effects.fit_wright_effect(
    #     dataframe=dataframe, 
    #     # mask_type='y',
    #     # estimator=LinearRegression(),
    #     # data_transform=StandardScaler(),
    #     )

    # # # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    # dox_vals = np.linspace(0., 1., 5)
    # intervention_data = dox_vals.reshape(len(dox_vals), len(X))
    # pred_Y = causal_effects.predict_wright_effect( 
    #         intervention_data=intervention_data)
    # print(pred_Y)
