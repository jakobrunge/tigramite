"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

import numpy as np
import itertools
from copy import deepcopy
from tigramite.models import Models

class CausalEffects:
    r"""Causal effect analysis for time series models.

    Handles the estimation of causal effects given a causal graph.

    Parameters
    ----------
    graph : array of shape [N, N, tau_max+1]
        Causal graph with link types `-->` (direct causal links),
        `<->` (counfounding), or `+->` (both, only for ADMGs).
    X : list of tuples
        List of tuples [(i, -tau), ...] containing cause variables.
    Y : list of tuples
        List of tuples [(j, 0), ...] containing effect variables.
    S : list of tuples
        List of tuples [(i, -tau), ...] containing conditioned variables.
    tau_max : int, optional (default: None)
        Maximum time lag. If None, graph.shape[2] - 1 is used.   
    graph_type : str
        Type of graph. Currently 'admg' and 'dag' are supported. 
    prune_X : bool
        Whether or not to remove nodes from X with no proper causal path to Y.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self,
                 graph,
                 X,
                 Y,
                 S=None,
                 tau_max=None,
                 graph_type='admg',
                 prune_X=True,
                 verbosity=0):
        
        self.graph_type = graph_type
        self.graph = graph
        self.N = graph.shape[0]

        if tau_max is None:
            self.tau_max = graph.shape[2] - 1
        else:
            self.tau_max = tau_max

        self.ignore_time_bounds = False

        self.verbosity = verbosity

        if self.graph_type == 'pag':
            self.possible = True 
            self.definite_status = True
        else:
            self.possible = False
            self.definite_status = False

        if S is None:
            S = []

        X = set(X)
        Y = set(Y)
        S = set(S)       

        anc_Y = self._get_ancestors(Y)

        # If X is not in anc(Y), then no causal link exists
        if anc_Y.intersection(set(X)) == set():
            raise ValueError("No causal path from X to Y exists.")

        self.X = X
        self.Y = Y
        self.S = S

        # Get mediators
        mediators = self.get_mediators() 

        M = set(mediators)
        self.M = M

        if prune_X:
            oldX = X.copy()

            # Remove from X those nodes with no causal path to Y
            X = set([x for x in self.X if x in anc_Y])

            # Also require that all x in X have proper path,
            # that is, the first link goes out of x 
            # and into causal children
            causal_children = list(self.M.union(self.Y)) 
            self.X = X.intersection(self._get_all_parents(causal_children))

            if verbosity > 0:
                print("Pruning X = %s to X=%s " %(oldX, self.X) +
                      "since only these have causal path to Y")

        if len(self.X.intersection(self.Y)) > 0:
            raise ValueError("Overlap between X and Y")

        if len(S.intersection(self.Y.union(self.X))) > 0:
            raise ValueError("Conditions S overlap with X or Y")

        # if len(self.S.intersection(self.M)) > 0:
        #     raise ValueError("Conditions S overlap with mediators M")

        descendants = self._get_descendants(self.Y.union(self.M))
        
        # Remove X and descendants of YM
        self.forbidden_nodes = descendants.union(self.X)  #.union(S)

        self.vancs = self._get_ancestors(list(self.X.union(self.Y).union(self.S))) - self.forbidden_nodes

        if len(self.S.intersection(self._get_descendants(self.X))) > 0:
            if self.verbosity > 0:
                print("Potentially outside assumptions: Conditions S overlap with des(X)")

        if len(self.S.intersection(self._get_descendants(self.Y))) > 0:
            raise ValueError("Not identifiable: Conditions S overlap with des(Y)")

        self.listX = list(self.X)
        self.listY = list(self.Y)
        self.listS = list(self.S)

    def get_mediators(self,):
        """for proper causal paths from X to Y"""
        # if conditions is None: conditions = []
        # anc_Y = self._get_ancestors(Y)
        des_X = self._get_descendants(self.X)
        # mediators = anc_Y.intersection(des_X) - set(Y) - set(X) #- set(conditions)

        mediators = set()

        # Walk along proper causal paths backwards from Y to X
        potential_mediators = set()
        for y in self.Y:
            j, tau = y 
            this_level = [y]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for parent in self._get_parents(varlag):
                        i, tau = parent
                        if (parent in des_X
                            and parent not in mediators
                            # and parent not in potential_mediators
                            and parent not in self.X
                            and parent not in self.Y
                            and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)):
                            mediators = mediators.union(set([parent]))
                            next_level.append(parent)
                            
                this_level = next_level  

        return mediators

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

    def _match_link(self, pattern, link):
        """Matches pattern including wildcards with link.
           
           In an ADMG we have edge types ["-->", "<--", "<->", "+->", "<-+"].
           Here +-> corresponds to having both "-->" and "<->".

           In a MAG we have edge types   ["-->", "<--", "<->", "---"].
        """
        
        if pattern == '' or link == '':
            return True if pattern == link else False
        else:
            left_mark, middle_mark, right_mark = pattern
            if left_mark != '*':
                if link[0] != '+':
                    if link[0] != left_mark: return False

            if right_mark != '*':
                if link[2] != '+':
                    if link[2] != right_mark: return False 
            
            if middle_mark != '*' and link[1] != middle_mark: return False    
                       
            return True

    def _find_adj(self, node, patterns, exclude=None):
        """Find adjacencies of node matching patterns."""
        
        graph = self.graph

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
                if match not in adj and (k, lag_i + lag_ik) not in exclude and (-self.tau_max <= lag_i + lag_ik <= 0 or self.ignore_time_bounds):
                    adj.append(match)
        
        # Find adjacencies going backward/contemp
        for k, lag_ki in zip(*np.where(graph[:,i,:])):  
            matches = [self._match_link(self._reverse_link(patt), graph[k, i, lag_ki]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i - lag_ki)
                if match not in adj and (k, lag_i - lag_ki) not in exclude and (-self.tau_max <= lag_i - lag_ki <= 0 or self.ignore_time_bounds):
                    adj.append(match)
     
        return adj

    def _is_match(self, nodei, nodej, pattern_ij):
        """Check whether the link between X and Y agrees with pattern_ij"""

        graph = self.graph

        (i, lag_i) = nodei
        (j, lag_j) = nodej
        tauij = lag_j - lag_i
        if abs(tauij) >= graph.shape[2]:
            return False
        return ((tauij >= 0 and self._match_link(pattern_ij, graph[i, j, tauij])) or
               (tauij < 0 and self._match_link(self._reverse_link(pattern_ij), graph[j, i, abs(tauij)])))


    def _get_children(self, varlag):
        """Returns set of children (varlag --> ...) for (lagged) varlag."""
        if self.possible:
            patterns=['-*>', 'o*o', 'o*>']
        else:
            patterns=['-*>']
        return self._find_adj(node=varlag, patterns=patterns)

    def _get_parents(self, varlag):
        """Returns set of parents (varlag <-- ...)) for (lagged) varlag."""
        if self.possible:
            patterns=['<*-', 'o*o', '<*o']
        else:
            patterns=['<*-']
        return self._find_adj(node=varlag, patterns=patterns)

    def _get_spouses(self, varlag):
        """Returns set of spouses (varlag <-> ...))  for (lagged) varlag."""
        return self._find_adj(node=varlag, patterns=['<*>'])

    def _get_neighbors(self, varlag):
        """Returns set of neighbors (varlag --- ...)) for (lagged) varlag."""
        return self._find_adj(node=varlag, patterns=['-*-'])

    def _get_ancestors(self, W):
        """Get ancestors of nodes in W up to time tau_max.
        
        Includes the nodes themselves.
        """

        ancestors = set(W)

        for w in W:
            j, tau = w 
            this_level = [w]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:

                    for par in self._get_parents(varlag):
                        i, tau = par
                        if par not in ancestors and -self.tau_max <= tau <= 0:
                            ancestors = ancestors.union(set([par]))
                            next_level.append(par)

                this_level = next_level       

        return ancestors

    def _get_all_parents(self, W):
        """Get parents of nodes in W up to time tau_max.
        
        Includes the nodes themselves.
        """

        parents = set(W)

        for w in W:
            j, tau = w 
            for par in self._get_parents(w):
                i, tau = par
                if par not in parents and -self.tau_max <= tau <= 0:
                    parents = parents.union(set([par]))

        return parents

    def _get_all_spouses(self, W):
        """Get spouses of nodes in W up to time tau_max.
        
        Includes the nodes themselves.
        """

        spouses = set(W)

        for w in W:
            j, tau = w 
            for spouse in self._get_spouses(w):
                i, tau = spouse
                if spouse not in spouses and -self.tau_max <= tau <= 0:
                    spouses = spouses.union(set([spouse]))

        return spouses

    def _get_descendants(self, W):
        """Get descendants of nodes in W up to time t.
        
        Includes the nodes themselves.
        """

        descendants = set(W)

        for w in W:
            j, tau = w 
            this_level = [w]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for child in self._get_children(varlag):
                        i, tau = child
                        if (child not in descendants 
                            and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)):
                            descendants = descendants.union(set([child]))
                            next_level.append(child)

                this_level = next_level       

        return descendants

    def _get_collider_path_nodes(self, W, descendants):
        """Get non-descendant collider path nodes of nodes in W up to time t.
        
        """

        collider_path_nodes = set([])
        # print("descendants ", descendants)
        for w in W:
            # print(w)
            j, tau = w 
            this_level = [w]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    # print("\t", varlag, self._get_spouses(varlag))
                    for spouse in self._get_spouses(varlag):
                        # print("\t\t", spouse)
                        i, tau = spouse
                        if (spouse not in collider_path_nodes
                            and spouse not in descendants 
                            and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)):
                            collider_path_nodes = collider_path_nodes.union(set([spouse]))
                            next_level.append(spouse)

                this_level = next_level       

        # Add parents
        for w in collider_path_nodes:
            for par in self._get_parents(w):
                if (par not in collider_path_nodes
                    and par not in descendants
                    and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)):
                    collider_path_nodes = collider_path_nodes.union(set([par]))

        return collider_path_nodes

    def _get_adjacents(self, node, patterns, max_lag=0, exclude=None):
        """Find adjacencies of node matching patterns."""
        
        graph = self.graph

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
                if (graph[i, k, lag_ik], match) not in adj and (k, lag_i + lag_ik) not in exclude and (-max_lag <= lag_i + lag_ik <= 0): # or self.ignore_time_bounds):
                    adj.append((graph[i, k, lag_ik], match))
        
        # Find adjacencies going backward/contemp
        for k, lag_ki in zip(*np.where(graph[:,i,:])):  
            matches = [self._match_link(self._reverse_link(patt), graph[k, i, lag_ki]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i - lag_ki)
                if (graph[i, k, lag_ik], match) not in adj and (k, lag_i - lag_ki) not in exclude and (-max_lag <= lag_i - lag_ki <= 0): # or self.ignore_time_bounds):
                    adj.append((self._reverse_link(graph[k, i, lag_ki]), match))

        return adj

    def _get_maximum_possible_lag(self, XYZ):
        """See Thm. XXXX"""

        def _repeating(link, seen_links):
            """Returns True if a link or its time-shifted version is already
            included in seen_links."""
            i, taui = link[0]
            j, tauj = link[1]

            for seen_link in seen_links:
                seen_i, seen_taui = seen_link[0]
                seen_j, seen_tauj = seen_link[1]

                if (i == seen_i and j == seen_j
                    and abs(tauj - taui) == abs(seen_tauj - seen_taui)):
                    return True

            return False

        if self.possible:
            patterns=['<*-', '<*o', 'o*o'] 
        else:
            patterns=['<*-'] 

        max_lag = 0
        for node in XYZ:
            j, tau = node   # tau <= 0
            max_lag = max(max_lag, abs(tau))
            
            seen_ancs = []
            seen_links = []
            this_level = [node]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for link_par in self._get_adjacents(node=varlag, patterns=patterns, 
                                        max_lag=max_lag, exclude=seen_ancs):
                        _, par = link_par
                        i, tau = par
                        # if par not in seen_ancs:
                        if not _repeating((par, varlag), seen_links):
                            max_lag = max(max_lag, abs(tau))
                            seen_ancs.append(par)
                            next_level.append(par)
                            seen_links.append((par, varlag))

                this_level = next_level

        return max_lag


    def check_path(self, start, end, conditions=None, 
        starts_with=None,
        ends_with=None,
        only_non_causal_paths=False,
        causal_children=None,
        ):
        """ 
        
        Includes checks of the optimality-theorem. Cond1-related checks test the existence of
        a collider path, COnd2-related checks the negation of a certain path as stated

        """

        # assert not (check_optimality_path == True and only_collider_paths == True)

        if conditions is None:
            conditions = set([])
        # if conditioned_variables is None:
        #     S = []

        start = set(start)
        end = set(end)
        conditions = set(conditions)
        
        # Get maximal possible time lag of a connecting path
        # See Thm. XXXX
        XYZ = start.union(end).union(conditions)
        max_lag = self._get_maximum_possible_lag(XYZ)

        if causal_children is None:
            causal_children = []
       
        if starts_with is None:
            starts_with = '***'

        if ends_with is None:
            ends_with = '***'

        #
        # Breadth-first search to find connection
        #
        start_from = set()
        for x in start:
            for link_neighbor in self._get_adjacents(node=x, patterns=starts_with, 
                                        max_lag=max_lag, exclude=list(start)):
                link, neighbor = link_neighbor
                if only_non_causal_paths:
                    # By amenability every proper possibly directed causal path starts with -*>
                    if (neighbor in causal_children and self._match_link('-*>', link) 
                        and not self._match_link('+*>', link)):
                        continue
                # start_from.add((link, neighbor))
                start_from.add((x, link, neighbor))

        visited = set()
        for (varlag_i, link_ik, varlag_k) in start_from:
            visited.add((link_ik, varlag_k))

        # Traversing through motifs i *-* k *-* j
        while start_from:

            # for (link_ik, varlag_k) in start_from:
            for (varlag_i, link_ik, varlag_k) in start_from:

                if varlag_k in end and self._match_link(ends_with, link_ik):
                    # print("Connected ", link_ik, varlag_k)
                    return True

            # Get any neighbor from starting nodes
            # link_ik, varlag_k = start_from.pop()
            varlag_i, link_ik, varlag_k = start_from.pop()

            # print("Get k = ", link_ik, varlag_k)
            # print("start_from ", start_from)
            # print("visited    ", visited)

            for link_neighbor in self._get_adjacents(node=varlag_k, patterns='***', 
                                        max_lag=max_lag, exclude=X):
                link_kj, varlag_j = link_neighbor
                # print("Get j = ", link_kj, varlag_j)

                # print ("visited ", (link_kj, varlag_j), visited)
                if (link_kj, varlag_j) in visited:
                # if (varlag_i, link_kj, varlag_j) in visited:
                    # print("in visited")
                    continue
                # print("Not in visited")

                # If motif  i *-* k *-* j is open, 
                # then add link_kj, varlag_j to visited and start_from
                left_mark = link_ik[2]
                right_mark = link_kj[0]
                # print(left_mark, right_mark)

                if self.definite_status:
                    # Exclude paths that are not definite_status implying that any of the following
                    # motifs occurs:
                    # i *-> k o-* j
                    if (left_mark == '>' and right_mark == 'o'):
                        continue
                    # i *-o k <-* j
                    if (left_mark == 'o' and right_mark == '<'):
                        continue
                    # i *-o k o-* j and i and j are adjacent
                    if (left_mark == 'o' and right_mark == 'o'
                        and self._is_match(varlag_i, varlag_j, "***")):
                        continue

                    # If k is in conditions and motif is *-o k o-*, then motif is blocked since
                    # i and j are non-adjacent due to the check above
                    if varlag_k in conditions and (left_mark == 'o' and right_mark == 'o'):
                        # print("Motif closed ", link_ik, varlag_k, link_kj, varlag_j )
                        continue  # [('>', '<'), ('>', '+'), ('+', '<'), ('+', '+')]

                # If k is in conditions and left or right mark is tail '-', then motif is blocked
                if varlag_k in conditions and (left_mark == '-' or right_mark == '-'):
                    # print("Motif closed ", link_ik, varlag_k, link_kj, varlag_j )
                    continue  # [('>', '<'), ('>', '+'), ('+', '<'), ('+', '+')]

                # If k is not in conditions and left and right mark are heads '><', then motif is blocked
                if varlag_k not in conditions and (left_mark == '>' and right_mark == '<'):
                    # print("Motif closed ", link_ik, varlag_k, link_kj, varlag_j )
                    continue  # [('>', '<'), ('>', '+'), ('+', '<'), ('+', '+')]

                # Motif is open
                # print("Motif open ", link_ik, varlag_k, link_kj, varlag_j )
                # start_from.add((link_kj, varlag_j))
                visited.add((link_kj, varlag_j))
                start_from.add((varlag_k, link_kj, varlag_j))
                # visited.add((varlag_k, link_kj, varlag_j))


        # print("Separated")
        # sys.exit(0)
        return False


    def get_optimal_set(self, 
        alternative_conditions=None,
        minimize=False,
        return_separate_sets=False,
        ):
        """Constructs optimal adjustment set
        
        See Runge NeurIPS 2021.

        minimize=False: full O-set
        minimize=True: full minimized O-set
        minimize='by_score': remove based on ci_test-model selection, but only up to minimal set

        """


        # Needed for optimality theorem where Osets for alternative S are tested
        if alternative_conditions is None:
            S = self.S.copy()
            vancs = self.vancs.copy()
        else:
            S = alternative_conditions
            vancs = self._get_ancestors(list(self.X.union(self.Y).union(S))) - self.forbidden_nodes

        descendants = self._get_descendants(self.Y.union(self.M))

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
                            if self.verbosity > 0:
                                print ("Causal effect not identifiable (valid collider node in X)")
                            return False

                        if (# Node not already in set
                            spouse not in colliders  #.union(parents)
                            # not forbidden
                            and spouse not in self.forbidden_nodes 
                            # in time bounds
                            and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)
                            and (spouse in vancs
                                or not self.check_path(start=self.X, end=[spouse], 
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
            if self.verbosity > 0:
                print ("Causal effect not identifiable (parent of valid collider node in X)")
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
                if (not self.check_path(start=X, end=[node], 
                                conditions=list(Oset - set([node])) + list(S))):
                    removable.append(node) 

            Oset = Oset - set(removable)
            if minimize == 'colliders_only':
                sorted_Oset = [node for node in Oset if node not in parents]

            removable = []
            # Next remove all those with no direct connection to Y
            for node in sorted_Oset:
                if (not self.check_path(start=[node], end=self.Y, 
                            conditions=list(Oset - set([node])) + list(S) + list(self.X),
                            ends_with='**>')): 
                    removable.append(node) 

            Oset = Oset - set(removable)

        Oset_S = Oset.union(S)

        if return_separate_sets:
            return parents, colliders, collider_parents, S
        else:
            return list(Oset_S)


    def get_collider_paths_optimality(self, source_nodes, target_nodes,
        condition, 
        inside_set=None, 
        start_with_tail_or_head=False, 
        # possible=False
        ):
        """Iterates over collider paths within O-set via depth-first search

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
                    if ((-self.tau_max <= tau <= 0 or self.ignore_time_bounds)
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
        """TODO"""

        # Cond. 0: Exactly one valid adjustment set exists
        cond_0 = (self.get_all_valid_adjustment_sets(check_one_set_exists=True))

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
            cond_I = self.get_collider_paths_optimality(
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
            if self.check_path(start=list(self.X), end=[E], 
                                conditions=list(self.S) + list(Oset_minusE)):
                   
                cond_II = self.get_collider_paths_optimality(
                    target_nodes=self.Y.union(self.M), 
                    source_nodes=list(set([E])),
                    condition='II', 
                    inside_set=list(Oset.union(self.S)),
                    start_with_tail_or_head = True)
               
                if cond_II is False:
                    break
   
        # print("Optimality = ", cond_0, cond_I, cond_II)
        optimality = (cond_0 or (cond_I and cond_II))
        return optimality

    def _check_validity(self, Z):
        """"""

        causal_children = list(self.M.union(self.Y))
        backdoor_path = self.check_path(start=list(self.X), end=list(self.Y), 
                            conditions=list(Z), 
                            causal_children=causal_children,
                            only_non_causal_paths=True)

        if backdoor_path:
            return False
        else:
            return True
    
    def get_adjust_set(self, 
        minimize=False,
        ):
        """Checks whether any valid adjustment set exist
        
        based on van der Zander, Textor...

        """
        #             for tau in range(0, self.tau_max + 1)]

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
                path = self.check_path(start=self.X, end=[node], 
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

                path = self.check_path(start=[node], end=self.Y, 
                    conditions=list(vancs - set([node])) + list(self.X),
                    )

                if path is False:
                   vancs = vancs - set([node])  

        if self._check_validity(list(vancs)) is False:
            return False
        else:
            return list(vancs)


    def check_backdoor_identifiability(self):
        """Checks whether any valid adjustment set exist
        
        based on van der Zander, Textor...

        """
        Z = self.get_adjust_set()
        if Z is False:
            return False
        else:
            return True

    def get_all_valid_adjustment_sets(self, 
        check_one_set_exists=False, yield_index=None):
        """Constructs all valid adjustment sets
        
        See Runge UAI 2021.

        based on van der Laan, Textor...

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


    def get_causal_paths(self, source_nodes, target_nodes,
        mediators=None,
        mediated_through=None,
        proper_paths=True,
        ):
        """Returns causal paths via depth-first search

        """

        source_nodes = set(source_nodes)
        target_nodes = set(target_nodes)

        if mediators is None:
            mediators = set()
        else:
            mediators = set(mediators)

        if mediated_through is not None:
            mediated_through = set(mediated_through)

        if proper_paths:
             inside_set = mediators.union(target_nodes) - source_nodes
        else:
             inside_set = mediators.union(target_nodes).union(source_nodes)

        all_causal_paths = {}         
        for w in source_nodes:
            all_causal_paths[w] = {}
            for z in target_nodes:
                all_causal_paths[w][z] = []

        for w in source_nodes:
            
            causal_path = []
            queue = [(w, causal_path)]

            while queue:

                varlag, causal_path = queue.pop()
                causal_path = causal_path + [varlag]
                suitable_nodes = set(self._get_children(varlag)
                    ).intersection(inside_set)

                for node in suitable_nodes:
                    i, tau = node
                    if ((-self.tau_max <= tau <= 0 or self.ignore_time_bounds)
                        and node not in causal_path):

                        queue.append((node, causal_path)) 
 
                        if node in target_nodes:  

                            if mediated_through is not None and len(set(causal_path).intersection(mediated_through)) == 0:
                                continue
                            else:
                                all_causal_paths[w][node].append(causal_path + [node]) 

        return all_causal_paths


    def fit_total_effect(self,
        dataframe, 
        estimator_model,
        adjustment_set='optimal',  
        data_transform=None,
        mask_type=None,
        ):
        """Returns a fitted model for the total causal effect of X on Y 
           conditional on S.
        """

        self.dataframe = dataframe


        if adjustment_set == 'optimal':
            # Check optimality and use either optimal or colliders_only set
            adjustment_set = self.get_optimal_set()
        elif adjustment_set == 'colliders_minimized_optimal':
            adjustment_set = self.get_optimal_set(minimize='colliders_only')
        elif adjustment_set == 'minimized_optimal':
            adjustment_set = self.get_optimal_set(minimize=True)
        else:
            # TODO: Check validity
            if self._check_validity(adjustment_set) is False:
                raise ValueError("Chosen adjustment_set is not valid.")

        self.adjustment_set = adjustment_set

        # Fit model of Y on X and Z (and conditions)
        # Build the model
        self.model = Models(
                        dataframe=dataframe,
                        model=estimator_model,
                        data_transform=data_transform,
                        mask_type=mask_type,
                        verbosity=self.verbosity)

        self.model.get_general_fitted_model(
                Y=self.listY, X=self.listX, Z=list(self.adjustment_set),
                conditions=self.listS,
                tau_max=self.tau_max,
                cut_off='max_lag_or_tau_max',
                return_data=False)

        return self

    def predict_total_effect(self, 
        intervention_data=None, 
        conditions_data=None,
        pred_params=None,
        ):
        """Returns a fitted model for the total causal effect of X on Y 
           conditional on S.
        """

        if intervention_data is not None:
            if intervention_data.values.shape != self.dataframe.values.shape:
                raise ValueError("intervention_data must be of same shape as "
                                 "fitted model dataframe.")

        if conditions_data is not None:
            if conditions_data.values.shape != self.dataframe.values.shape:
                raise ValueError("intervention_data must be of same shape as "
                                 "fitted model dataframe.")

        effect = self.model.get_general_prediction(
                Y=self.listY, X=self.listX, Z=list(self.adjustment_set),
                intervention_data=intervention_data,
                conditions=self.listS,
                conditions_data=conditions_data,
                pred_params=pred_params,
                cut_off='max_lag_or_tau_max')

        return effect

    def fit_wrights_effect(self,
        dataframe, 
        mediated_through=None,
        method=None,
        links_coeffs=None,  
        data_transform=None,
        mask_type=None,
        ):
        """Returns a fitted model for the total causal effect of X on Y 
           conditional on S.
        """
        import sklearn.linear_model

        self.dataframe = dataframe
        estimator_model = sklearn.linear_model.LinearRegression()

        # Fit model of Y on X and Z (and conditions)
        # Build the model
        self.model = Models(
                        dataframe=dataframe,
                        model=estimator_model,
                        data_transform=data_transform,
                        mask_type=mask_type,
                        verbosity=self.verbosity)

        mediators = self.get_mediators()
        causal_paths = self.get_causal_paths(source_nodes=self.X, 
                target_nodes=self.Y, mediators=mediators, 
                mediated_through=mediated_through, proper_paths=True)

        if method == 'links_coeffs':
            coeffs = {}
            max_lag = 0
            for j in [med[0] for med in mediators] + [y[0] for y in self.listY]:
                coeffs[j] = {}
                for ipar, par_coeff in enumerate(links_coeffs[j]):
                    par, coeff, _ = par_coeff
                    max_lag = max(abs(par[1]), max_lag)
                    coeffs[j][par] = coeff #self.fit_results[j][(j, 0)]['model'].coef_[ipar]

            self.model.tau_max = max_lag

        elif method == 'optimal':
            # all_parents = {}
            coeffs = {}
            for j in [med[0] for med in mediators] + [y[0] for y in self.listY]:
                coeffs[j] = {}
                all_parents = self._get_all_parents([(j, 0)]) - set([(j, 0)])
                for par in all_parents:
                    Sprime = set(all_parents) - set([par])
                    # print(j, par, Sprime)
                    causal_effects = CausalEffects(graph=self.graph, X=[par], Y=[(j, 0)],
                                        S=Sprime,
                                        tau_max=self.tau_max, graph_type=self.graph_type,
                                            prune_X=False)
                    oset = causal_effects.get_optimal_set()
                    # print(j, all_parents[j])
                    if oset is False:
                        raise ValueError("Not identifiable via Wright's method.")
                    fit_res = self.model.get_general_fitted_model(
                        Y=[(j, 0)], X=[par], Z=oset,
                        tau_max=self.tau_max,
                        cut_off='max_lag_or_tau_max',
                        return_data=False)
                    coeffs[j][par] = fit_res[(j, 0)]['model'].coef_[0]

        elif method == 'parents':
            if self.graph_type != 'dag':
                raise ValueError("method == 'parents' only possible for DAGs")

            coeffs = {}
            for j in [med[0] for med in mediators] + [y[0] for y in self.listY]:
                coeffs[j] = {}
                all_parents = self._get_all_parents([(j, 0)]) - set([(j, 0)])
                # print(j, all_parents[j])
                # if len(all_parents[j]) > 0:
                fit_res = self.model.get_general_fitted_model(
                    Y=[(j, 0)], X=list(all_parents), Z=[],
                    conditions=None,
                    tau_max=self.tau_max,
                    cut_off='max_lag_or_tau_max',
                    return_data=False)

                for ipar, par in enumerate(all_parents):
                    coeffs[j][par] = fit_res[(j, 0)]['model'].coef_[ipar]

        # Effect is sum over products over all path coefficients
        # from x in X to y in Y
        effect = {}
        for (x, y) in itertools.product(self.listX, self.listY):
            effect[(x, y)] = 0.
            for causal_path in causal_paths[x][y]:
                effect_here = 1.
                for index, node in enumerate(causal_path[:-1]):
                    i, taui = node
                    j, tauj = causal_path[index + 1]
                    tau_ij = abs(tauj - taui)
                    effect_here *= coeffs[j][(i, -tau_ij)]

                effect[(x, y)] += effect_here
                    
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

    
    def predict_wrights_effect(self, 
        intervention_data=None, 
        # conditions_data=None,
        pred_params=None,
        ):
        """Returns a fitted model for the total causal effect of X on Y 
           conditional on S.
        """

        if intervention_data is not None:
            if intervention_data.values.shape != self.dataframe.values.shape:
                raise ValueError("intervention_data must be of same shape as "
                                 "fitted model dataframe.")

        effect = self.model.get_general_prediction(
                Y=self.listY, X=self.listX, Z=[],
                intervention_data=intervention_data,
                conditions=[],
                conditions_data=None,
                pred_params=pred_params,
                cut_off='max_lag_or_tau_max')

        return effect


if __name__ == '__main__':
   
    import tigramite.data_processing as pp
    import tigramite.plotting as tp
    from tigramite.independence_tests import OracleCI
    from tigramite.data_processing import DataFrame

    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    

    # Example from NeurIPS 2021 paper Fig. 1A
    coeff = .5
    conf_coeff = 2.
    conf_coeff2 = 1.
    def lin_f(x): return x

    # Non-time series example
    # links = {
    #         0: [((3, 0), conf_coeff, lin_f), ((6, 0), conf_coeff, lin_f)], 
    #         1: [((0, 0), coeff, lin_f), ((4, 0), conf_coeff, lin_f)], 
    #         2: [((0, 0), coeff, lin_f), ((1, 0), coeff, lin_f), ((4, 0), conf_coeff, lin_f), ((7, 0), conf_coeff, lin_f)], #, ((1, 0), coeff, lin_f)], 
    #         3: [((6, 0), conf_coeff, lin_f)],
    #         4: [((3, 0), conf_coeff2, lin_f)],
    #         5: [((4, 0), conf_coeff, lin_f), ((7, 0), conf_coeff, lin_f)],
    #         6: [],
    #         7: []}

    # Same example with time-structure
    auto_coeff = 0.3
    links = {
            0: [((0, -1), auto_coeff, lin_f), ((3, 0), conf_coeff, lin_f), ((6, 0), conf_coeff, lin_f)], 
            1: [((1, -1), auto_coeff, lin_f), ((0, -1), coeff, lin_f), ((4, 0), conf_coeff, lin_f)], 
            2: [((2, -1), auto_coeff, lin_f), ((0, -2), coeff, lin_f), ((1, -1), coeff, lin_f), ((4, -1), conf_coeff, lin_f), ((7, 0), conf_coeff, lin_f)], #, ((1, 0), coeff, lin_f)], 
            3: [((3, -1), auto_coeff, lin_f), ((6, 0), conf_coeff, lin_f)],
            4: [((4, -1), auto_coeff, lin_f), ((3, -1), conf_coeff2, lin_f)],
            5: [((5, -1), auto_coeff, lin_f), ((4, -1), conf_coeff, lin_f), ((7, 0), conf_coeff, lin_f)], #, ((8, -1), conf_coeff, lin_f), ((8, 0), conf_coeff, lin_f)],
            6: [],
            7: [],
            8: []}

    # DAG version of Non-time series example
    # links = {
    #         0: [((3, 0), conf_coeff, lin_f)], 
    #         1: [((0, 0), coeff, lin_f), ((4, 0), conf_coeff, lin_f)], 
    #         2: [((0, 0), coeff, lin_f), ((1, 0), coeff, lin_f), ((4, 0), conf_coeff, lin_f)], #, ((1, 0), coeff, lin_f)], 
    #         3: [],
    #         4: [((3, 0), conf_coeff2, lin_f)],
    #         5: [((4, 0), conf_coeff, lin_f)],}

    # observed_vars = [0,   1,   2,   3,    4,    5]
    # var_names =    ['X', 'M', 'Y', 'Z1', 'Z2', 'Z3']
    # X = [(0, 0)] #, (0, -2)]
    # Y = [(2, 0)]
    # conditions = []   # called 'S' in paper

    # DAG version of time series example
    # auto_coeff = 0.3 
    # links = {
    #         0: [((0, -1), auto_coeff, lin_f),((3, 0), conf_coeff, lin_f)], 
    #         1: [((1, -1), auto_coeff, lin_f),((0, -1), coeff, lin_f), ((4, 0), conf_coeff, lin_f)], 
    #         2: [((2, -1), auto_coeff, lin_f),((0, -2), coeff, lin_f), ((1, -1), coeff, lin_f), ((4, -1), conf_coeff, lin_f), ((5, 0), coeff, lin_f)], 
    #         3: [((3, -1), auto_coeff, lin_f),],
    #         4: [((4, -1), auto_coeff, lin_f),((3, -1), conf_coeff2, lin_f)],
    #         5: [((5, -1), auto_coeff, lin_f),((4, -1), conf_coeff, lin_f)],}

    observed_vars = [0,   1,   2,   3,    4,    5]
    var_names =    ['X', 'M', 'Y', 'Z1', 'Z2', 'Z3']
    X = [(0, -1), (0, -2), (0, -3)]
    Y = [(2, 0), (2, -1)]
    conditions = []   # called 'S' in paper


    # if tau_max is None, graph.shape[2]-1 will be used
    tau_max = 4  # 4 for time series version

    oracle = OracleCI(links=links, observed_vars=observed_vars)
    graph = oracle.graph
    # tau_max = graph.shape[2] - 1

    T = 10000
    data, nonstat = pp.structural_causal_process(links, T=T, noises=None, seed=7)
    dataframe = pp.DataFrame(data)

    # Initialize class
    causal_effects = CausalEffects(graph=graph, X=X, Y=Y, S=None,
                                    graph_type='admg',
                                    tau_max = tau_max,
                                    verbosity=0)


    optimality = causal_effects.check_optimality()
    print("(Graph, X, Y, S) fulfills optimality: ", optimality)

    # Adjust-set
    adjust = causal_effects.get_adjust_set()
    print("\nAdjust / Ancs set")
    print([(var_names[v[0]], v[1]) for v in adjust])

    # Minimized Adjust-set
    adjust_min = causal_effects.get_adjust_set(minimize=True)
    print("\nMin Ancs set")
    print([(var_names[v[0]], v[1]) for v in adjust_min])

    # ParX-minimized Ancs-set
    adjust_pxmin = causal_effects.get_adjust_set(minimize='keep_parentsYM')
    print("\nMinParX Ancs set")
    print([(var_names[v[0]], v[1]) for v in adjust_pxmin])

    # Optimal adjustment set
    opt = causal_effects.get_optimal_set()
    print("\nOset")
    print([(var_names[v[0]], v[1]) for v in opt])

    # Minimized adjustment set
    opt_min = causal_effects.get_optimal_set(minimize=True)
    print("\nMin Oset")
    print([(var_names[v[0]], v[1]) for v in opt_min])

    opt_cmin = causal_effects.get_optimal_set(minimize='colliders_only')
    print("\nMinColl Oset")
    print([(var_names[v[0]], v[1]) for v in opt_cmin])


    # Plot graph
    if tau_max is not None:
        graph_plot = np.zeros((len(observed_vars), 
            len(observed_vars), tau_max+1), dtype='<U3')
        graph_plot[:,:, :graph.shape[2]] = graph
        graph_plot[:,:, graph.shape[2]:] = ""
        # print(graph_plot.shape)
        # print(graph.shape)
    else:
        graph_plot = graph

    special_nodes = {}
    for node in X:
        special_nodes[node] = 'red'
    for node in Y:
        special_nodes[node] = 'blue'
    for node in opt:
        special_nodes[node] = 'orange'
    for node in causal_effects.get_mediators():
        special_nodes[node] = 'lightblue'
  
    tp.plot_graph(graph = graph_plot, var_names=var_names, 
            save_name='Example-Fig1A.pdf',
            figsize = (15, 15), node_size=0.2,
            special_nodes=special_nodes)
    if tau_max is not None:
        tp.plot_time_series_graph(graph = graph_plot, var_names=var_names, 
            save_name='Example-Fig1A-TSG.pdf',
            figsize = (15, 15),
            special_nodes=special_nodes)

    #
    estimator_model = LinearRegression()
    # estimator_model = KNeighborsRegressor(n_neighbors=3)
    # estimator_model = MLPRegressor(max_iter=200)

    # causal_effects.fit_total_effect(
    #     dataframe=dataframe, 
    #     estimator_model=estimator_model,
    #     adjustment_set='optimal',  
    #     data_transform=None,
    #     mask_type=None,
    #     )

    # # Causal effect in observational data
    # ce_obs = causal_effects.predict_total_effect( 
    #     intervention_data=None, 
    #     conditions_data=None,
    #     )

    # # Causal effect for interventional data 
    # # with + 1 added
    intervention_data = data.copy()
    intervention_data[:, X[0]] += 1.
    intervention_data = pp.DataFrame(intervention_data)

    # ce_int = causal_effects.predict_total_effect( 
    #     intervention_data=intervention_data, 
    #     conditions_data=None,
    #     )

    causal_effects.fit_wrights_effect(dataframe=dataframe, 
        links_coeffs=links,
        method = 'optimal',
         mediated_through=[(1, -2)],
         # data_transform=sklearn.preprocessing.StandardScaler()
         )
    ce_obs = causal_effects.predict_wrights_effect(intervention_data=None)
    ce_int = causal_effects.predict_wrights_effect(intervention_data=intervention_data)


    ## Expected change corresponds to linear regression coefficient
    ## for linear models
    for y in Y:
        beta = (ce_int[y] - ce_obs[y]).mean()
        print("\nLinear causal effect on %s = %.2f" %(y, beta))





