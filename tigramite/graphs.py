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
import struct

class Graphs():
    r"""Graph class.

    Methods for dealing with causal graphs. Various graph types are 
    supported, also including hidden variables.
    

    Parameters
    ----------
    graph : array of either shape [N, N], [N, N, tau_max+1], or [N, N, tau_max+1, tau_max+1]
        Different graph types are supported, see tutorial.
    graph_type : str
        Type of graph.
    hidden_variables : list of tuples
        Hidden variables in format [(i, -tau), ...]. The internal graph is 
        constructed by a latent projection.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self,
                 graph,
                 graph_type,
                 tau_max,
                 hidden_variables=None,
                 verbosity=0):
        
        self.verbosity = verbosity
        self.N = graph.shape[0]
        self.tau_max = tau_max

        # 
        # Checks regarding graph type
        #
        supported_graphs = ['dag', 
                            'admg',
                            'tsg_dag',
                            'tsg_admg',
                            'stationary_dag',
                            'stationary_admg',

                            # 'mag',
                            # 'tsg_mag',
                            # 'stationary_mag',
                            # 'pag',
                            # 'tsg_pag',
                            # 'stationary_pag',
                            ]
        if graph_type not in supported_graphs:
            raise ValueError("Only graph types %s supported!" %supported_graphs)

        # TODO?: check that masking aligns with hidden samples in variables
        if hidden_variables is None:
            hidden_variables = []

        # Only needed for later extension to MAG/PAGs
        if 'pag' in graph_type:
            self.possible = True 
            self.definite_status = True
        else:
            self.possible = False
            self.definite_status = False

        # Not needed for now...
        # self.ignore_time_bounds = False

        # Construct internal graph from input graph depending on graph type
        # and hidden variables
        self._construct_graph(graph=graph, graph_type=graph_type,
                              hidden_variables=hidden_variables)

        self._check_graph(self.graph)


    def _construct_graph(self, graph, graph_type, hidden_variables):
        """Construct internal graph object based on input graph and hidden variables.

           Uses the latent projection operation.
        """


        if graph_type in ['dag', 'admg']: 
            if graph.ndim != 2:
                raise ValueError("graph_type in ['dag', 'admg'] assumes graph.shape=(N, N).")
            # Convert to shape [N, N, 1, 1] with dummy dimension
            # to process as tsg_dag or tsg_admg with potential hidden variables
            self.graph = np.expand_dims(graph, axis=(2, 3))
            
            # tau_max needed in _get_latent_projection_graph
            # self.tau_max = 0

            if len(hidden_variables) > 0:
                self.graph = self._get_latent_projection_graph() # stationary=False)
                self.graph_type = "tsg_admg"
            else:
                # graph = self.graph
                self.graph_type = 'tsg_' + graph_type

        elif graph_type in ['tsg_dag', 'tsg_admg']:
            if graph.ndim != 4:
                raise ValueError("tsg-graph_type assumes graph.shape=(N, N, tau_max+1, tau_max+1).")

            # Then tau_max is implicitely derived from
            # the dimensions 
            self.graph = graph
            # self.tau_max = graph.shape[2] - 1

            if len(hidden_variables) > 0:
                self.graph = self._get_latent_projection_graph() #, stationary=False)
                self.graph_type = "tsg_admg"
            else:
                self.graph_type = graph_type   

        elif graph_type in ['stationary_dag', 'stationary_admg']:
            # Currently only stationary_dag without hidden variables is supported
            if graph.ndim != 3:
                raise ValueError("stationary graph_type assumes graph.shape=(N, N, tau_max+1).")
            
            # # TODO: remove if theory for stationary ADMGs is clear
            # if graph_type == 'stationary_dag' and len(hidden_variables) > 0:
            #     raise ValueError("Hidden variables currently not supported for "
            #                      "stationary_dag.")

            # For a stationary DAG without hidden variables it's sufficient to consider
            # a tau_max that includes the parents of X, Y, M, and S. A conservative
            # estimate thereof is simply the lag-dimension of the stationary DAG plus
            # the maximum lag of XYS.
            # statgraph_tau_max = graph.shape[2] - 1
            # maxlag_XYS = 0
            # for varlag in self.X.union(self.Y).union(self.S):
            #     maxlag_XYS = max(maxlag_XYS, abs(varlag[1]))

            # self.tau_max = maxlag_XYS + statgraph_tau_max

            stat_graph = deepcopy(graph)

            #########################################		
            # Use this tau_max and construct ADMG by assuming paths of
            # maximal lag 10*tau_max... TO BE REVISED!
            self.graph = graph
            self.graph = self._get_latent_projection_graph(stationary=True)
            self.graph_type = "tsg_admg"
            #########################################

            # Also create stationary graph extended to tau_max
            self.stationary_graph = np.zeros((self.N, self.N, self.tau_max + 1), dtype='<U3')
            self.stationary_graph[:, :, :stat_graph.shape[2]] = stat_graph

            # allowed_edges = ["-->", "<--"]

            # # Construct tsg_graph
            # graph = np.zeros((self.N, self.N, self.tau_max + 1, self.tau_max + 1), dtype='<U3')
            # graph[:] = ""
            # for (i, j) in itertools.product(range(self.N), range(self.N)):
            #     for jt, tauj in enumerate(range(0, self.tau_max + 1)):
            #         for it, taui in enumerate(range(tauj, self.tau_max + 1)):
            #             tau = abs(taui - tauj)
            #             if tau == 0 and j == i:
            #                 continue
            #             if tau > statgraph_tau_max:
            #                 continue                        

            #             # if tau == 0:
            #             #     if stat_graph[i, j, tau] == '-->':
            #             #         graph[i, j, taui, tauj] = "-->" 
            #             #         graph[j, i, tauj, taui] = "<--" 

            #             #     # elif stat_graph[i, j, tau] == '<--':
            #             #     #     graph[i, j, taui, tauj] = "<--"
            #             #     #     graph[j, i, tauj, taui] = "-->" 
            #             # else:
            #             if stat_graph[i, j, tau] == '-->':
            #                 graph[i, j, taui, tauj] = "-->" 
            #                 graph[j, i, tauj, taui] = "<--" 
            #             elif stat_graph[i, j, tau] == '<--':
            #                 pass
            #             elif stat_graph[i, j, tau] == '':
            #                 pass
            #             else:
            #                 edge = stat_graph[i, j, tau]
            #                 raise ValueError("Invalid graph edge %s. " %(edge) +
            #                      "For graph_type = %s only %s are allowed." %(graph_type, str(allowed_edges)))
      
            #             # elif stat_graph[i, j, tau] == '<--':
            #             #     graph[i, j, taui, tauj] = "<--"
            #             #     graph[j, i, tauj, taui] = "-->" 

            # self.graph_type = 'tsg_dag'
            # self.graph = graph


        # return (graph, graph_type, self.tau_max, hidden_variables)

            # max_lag = self._get_maximum_possible_lag(XYZ=list(X.union(Y).union(S)), graph=graph)

            # stat_mediators = self._get_mediators_stationary_graph(start=X, end=Y, max_lag=max_lag)
            # self.tau_max = self._get_maximum_possible_lag(XYZ=list(X.union(Y).union(S).union(stat_mediators)), graph=graph)
            # self.tau_max = graph_taumax
            # for varlag in X.union(Y).union(S):
            #     self.tau_max = max(self.tau_max, abs(varlag[1]))

            # if verbosity > 0:
            #     print("Setting tau_max = ", self.tau_max)

            # if tau_max is None:
            #     self.tau_max = graph_taumax
            #     for varlag in X.union(Y).union(S):
            #         self.tau_max = max(self.tau_max, abs(varlag[1]))

            #     if verbosity > 0:
            #         print("Setting tau_max = ", self.tau_max)
            # else:
                # self.tau_max = graph_taumax
                # # Repeat hidden variable pattern 
                # # if larger tau_max is given
                # if self.tau_max > graph_taumax:
                #     for lag in range(graph_taumax + 1, self.tau_max + 1):
                #         for j in range(self.N):
                #             if (j, -(lag % (graph_taumax+1))) in self.hidden_variables:
                #                 self.hidden_variables.add((j, -lag))
            # print(self.hidden_variables)

        #     self.graph = self._get_latent_projection_graph(self.graph, stationary=True)
        #     self.graph_type = "tsg_admg"
        # else:

    def _check_graph(self, graph):
        """Checks that graph contains no invalid entries/structure.

        Assumes graph.shape = (N, N, tau_max+1, tau_max+1)
        """

        allowed_edges = ["-->", "<--"]
        if 'admg' in self.graph_type:
            allowed_edges += ["<->", "<-+", "+->"]
        elif 'mag' in self.graph_type:
            allowed_edges += ["<->"]
        elif 'pag' in self.graph_type:
            allowed_edges += ["<->", "o-o", "o->", "<-o"]                         # "o--",
                        # "--o",
                        # "x-o",
                        # "o-x",
                        # "x--",
                        # "--x",
                        # "x->",
                        # "<-x",
                        # "x-x",
                    # ]

        graph_dict = defaultdict(list)
        for i, j, taui, tauj in zip(*np.where(graph)):
            edge = graph[i, j, taui, tauj]
            # print((i, -taui), edge, (j, -tauj), graph[j, i, tauj, taui])
            if edge != self._reverse_link(graph[j, i, tauj, taui]):
                raise ValueError(
                    "graph needs to have consistent edges (eg"
                    " graph[i,j,taui,tauj]='-->' requires graph[j,i,tauj,taui]='<--')"
                )

            if edge not in allowed_edges:
                raise ValueError("Invalid graph edge %s. " %(edge) +
                                 "For graph_type = %s only %s are allowed." %(self.graph_type, str(allowed_edges)))

            if edge == "-->" or edge == "+->":
                # Map to (i,-taui, j, tauj) graph
                indexi = i * (self.tau_max + 1) + taui
                indexj = j * (self.tau_max + 1) + tauj

                graph_dict[indexj].append(indexi)

        # Check for cycles
        if self._check_cyclic(graph_dict):
            raise ValueError("graph is cyclic.")

        # if MAG: check for almost cycles
        # if PAG???

    def _check_cyclic(self, graph_dict):
        """Return True if the graph_dict has a cycle.

        graph_dict must be represented as a dictionary mapping vertices to
        iterables of neighbouring vertices. For example:

        >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
        True
        >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
        False
        """

        path = set()
        visited = set()

        def visit(vertex):
            if vertex in visited:
                return False
            visited.add(vertex)
            path.add(vertex)
            for neighbour in graph_dict.get(vertex, ()):
                if neighbour in path or visit(neighbour):
                    return True
            path.remove(vertex)
            return False

        return any(visit(v) for v in graph_dict)

    def get_mediators(self, start, end):
        """Returns mediator variables on proper causal paths.

        Parameters
        ----------
        start : set
            Set of start nodes.
        end : set
            Set of end nodes.

        Returns
        -------
        mediators : set
            Mediators on causal paths from start to end.
        """

        des_X = self._get_descendants(start)

        mediators = set()

        # Walk along proper causal paths backwards from Y to X
        # potential_mediators = set()
        for y in end:
            j, tau = y 
            this_level = [y]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for parent in self._get_parents(varlag):
                        i, tau = parent
                        # print(varlag, parent, des_X)
                        if (parent in des_X
                            and parent not in mediators
                            # and parent not in potential_mediators
                            and parent not in start
                            and parent not in end
                            and (-self.tau_max <= tau <= 0)): # or self.ignore_time_bounds)):
                            mediators = mediators.union(set([parent]))
                            next_level.append(parent)
                            
                this_level = next_level  

        return mediators

    def _get_mediators_stationary_graph(self, start, end, max_lag):
        """Returns mediator variables on proper causal paths
           from X to Y in a stationary graph."""

        des_X = self._get_descendants_stationary_graph(start, max_lag)

        mediators = set()

        # Walk along proper causal paths backwards from Y to X
        potential_mediators = set()
        for y in end:
            j, tau = y 
            this_level = [y]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for _, parent in self._get_adjacents_stationary_graph(graph=self.graph, 
                                node=varlag, patterns=["<*-", "<*+"], max_lag=max_lag, exclude=None):
                        i, tau = parent
                        if (parent in des_X
                            and parent not in mediators
                            # and parent not in potential_mediators
                            and parent not in start
                            and parent not in end
                            # and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)
                            ):
                            mediators = mediators.union(set([parent]))
                            next_level.append(parent)
                            
                this_level = next_level  

        return mediators

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
                # if link[0] != '+':
                    if link[0] != left_mark: return False

            if right_mark != '*':
                # if link[2] != '+':
                    if link[2] != right_mark: return False 
            
            if middle_mark != '*' and link[1] != middle_mark: return False    
                       
            return True

    def _find_adj(self, node, patterns, exclude=None, return_link=False):
        """Find adjacencies of node that match given patterns."""
        
        graph = self.graph

        if exclude is None:
            exclude = []
        #     exclude = self.hidden_variables
        # else:
        #     exclude = set(exclude).union(self.hidden_variables)

        # Setup
        i, lag_i = node
        lag_i = abs(lag_i)

        if exclude is None: exclude = []
        if type(patterns) == str:
            patterns = [patterns]

        # Init
        adj = []
        # Find adjacencies going forward/contemp
        for k, lag_ik in zip(*np.where(graph[i,:,lag_i,:])):
            # print((k, lag_ik), graph[i,k,lag_i,lag_ik]) 
            # matches = [self._match_link(patt, graph[i,k,lag_i,lag_ik]) for patt in patterns]
            # if np.any(matches):
            for patt in patterns:
                if self._match_link(patt, graph[i,k,lag_i,lag_ik]):
                    match = (k, -lag_ik)
                    if match not in exclude:
                        if return_link:
                            adj.append((graph[i,k,lag_i,lag_ik], match))
                        else:
                            adj.append(match)
                    break

        
        # Find adjacencies going backward/contemp
        for k, lag_ki in zip(*np.where(graph[:,i,:,lag_i])):  
            # print((k, lag_ki), graph[k,i,lag_ki,lag_i]) 
            # matches = [self._match_link(self._reverse_link(patt), graph[k,i,lag_ki,lag_i]) for patt in patterns]
            # if np.any(matches):
            for patt in patterns:
                if self._match_link(self._reverse_link(patt), graph[k,i,lag_ki,lag_i]):
                    match = (k, -lag_ki)
                    if match not in exclude:
                        if return_link:
                            adj.append((self._reverse_link(graph[k,i,lag_ki,lag_i]), match))
                        else:
                            adj.append(match)
                    break
     
        adj = list(set(adj))
        return adj

    def _is_match(self, nodei, nodej, pattern_ij):
        """Check whether the link between X and Y agrees with pattern."""

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
            patterns=['-*>', '+*>']
        return self._find_adj(node=varlag, patterns=patterns)

    def _get_parents(self, varlag):
        """Returns set of parents (varlag <-- ...) for (lagged) varlag."""
        if self.possible:
            patterns=['<*-', 'o*o', '<*o']
        else:
            patterns=['<*-', '<*+']
        return self._find_adj(node=varlag, patterns=patterns)

    def _get_spouses(self, varlag):
        """Returns set of spouses (varlag <-> ...)  for (lagged) varlag."""
        return self._find_adj(node=varlag, patterns=['<*>', '+*>', '<*+'])

    def _get_neighbors(self, varlag):
        """Returns set of neighbors (varlag --- ...) for (lagged) varlag."""
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

    def _get_descendants_stationary_graph(self, W, max_lag):
        """Get descendants of nodes in W up to time t in stationary graph.
        
        Includes the nodes themselves.
        """

        descendants = set(W)

        for w in W:
            j, tau = w 
            this_level = [w]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for _, child in self._get_adjacents_stationary_graph(graph=self.graph, 
                                node=varlag, patterns=["-*>", "-*+"], max_lag=max_lag, exclude=None):
                        i, tau = child
                        if (child not in descendants 
                            # and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)
                            ):
                            descendants = descendants.union(set([child]))
                            next_level.append(child)

                this_level = next_level       

        return descendants

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
                            and (-self.tau_max <= tau <= 0)): # or self.ignore_time_bounds)):
                            descendants = descendants.union(set([child]))
                            next_level.append(child)

                this_level = next_level       

        return descendants

    def _get_collider_path_nodes(self, W, descendants):
        """Get non-descendant collider path nodes and their parents of nodes in W up to time t.
        
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
                            and (-self.tau_max <= tau <= 0)): # or self.ignore_time_bounds)):
                            collider_path_nodes = collider_path_nodes.union(set([spouse]))
                            next_level.append(spouse)

                this_level = next_level       

        # Add parents
        for w in collider_path_nodes:
            for par in self._get_parents(w):
                if (par not in collider_path_nodes
                    and par not in descendants
                    and (-self.tau_max <= tau <= 0)): # or self.ignore_time_bounds)):
                    collider_path_nodes = collider_path_nodes.union(set([par]))

        return collider_path_nodes

    def _get_adjacents_stationary_graph(self, graph, node, patterns, 
        max_lag=0, exclude=None):
        """Find adjacencies of node matching patterns in a stationary graph."""
        
        # graph = self.graph

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
                if (k, lag_i + lag_ik) not in exclude and (-max_lag <= lag_i + lag_ik <= 0): # or self.ignore_time_bounds):
                    adj.append((graph[i, k, lag_ik], match))
        
        # Find adjacencies going backward/contemp
        for k, lag_ki in zip(*np.where(graph[:,i,:])):  
            matches = [self._match_link(self._reverse_link(patt), graph[k, i, lag_ki]) for patt in patterns]
            if np.any(matches):
                match = (k, lag_i - lag_ki)
                if (k, lag_i - lag_ki) not in exclude and (-max_lag <= lag_i - lag_ki <= 0): # or self.ignore_time_bounds):
                    adj.append((self._reverse_link(graph[k, i, lag_ki]), match))         
        
        adj = list(set(adj))
        return adj

    def _get_canonical_dag_from_graph(self, graph):
        """Constructs canonical DAG as links_coeffs dictionary from graph.

        For every <-> link further latent variables are added.
        This corresponds to a canonical DAG (Richardson Spirtes 2002).

        Can be used to evaluate d-separation.
        """

        N, N, tau_maxplusone = graph.shape
        tau_max = tau_maxplusone - 1

        links = {j: [] for j in range(N)}

        # Add further latent variables to accommodate <-> links
        latent_index = N
        for i, j, tau in zip(*np.where(graph)):

            edge_type = graph[i, j, tau]

            # Consider contemporaneous links only once
            if tau == 0 and j > i:
                continue

            if edge_type == "-->":
                links[j].append((i, -tau))
            elif edge_type == "<--":
                links[i].append((j, -tau))
            elif edge_type == "<->":
                links[latent_index] = []
                links[i].append((latent_index, 0))
                links[j].append((latent_index, -tau))
                latent_index += 1
            # elif edge_type == "---":
            #     links[latent_index] = []
            #     selection_vars.append(latent_index)
            #     links[latent_index].append((i, -tau))
            #     links[latent_index].append((j, 0))
            #     latent_index += 1
            elif edge_type == "+->":
                links[j].append((i, -tau))
                links[latent_index] = []
                links[i].append((latent_index, 0))
                links[j].append((latent_index, -tau))
                latent_index += 1
            elif edge_type == "<-+":
                links[i].append((j, -tau))
                links[latent_index] = []
                links[i].append((latent_index, 0))
                links[j].append((latent_index, -tau))
                latent_index += 1

        return links


    def _get_maximum_possible_lag(self, XYZ, graph):
        """Construct maximum relevant time lag for d-separation in stationary graph.

        TO BE REVISED!

        """

        def _repeating(link, seen_path):
            """Returns True if a link or its time-shifted version is already
            included in seen_links."""
            i, taui = link[0]
            j, tauj = link[1]

            for index, seen_link in enumerate(seen_path[:-1]):
                seen_i, seen_taui = seen_link
                seen_j, seen_tauj = seen_path[index + 1]

                if (i == seen_i and j == seen_j
                    and abs(tauj-taui) == abs(seen_tauj-seen_taui)):
                    return True

            return False

        # TODO: does this work with PAGs?
        # if self.possible:
        #     patterns=['<*-', '<*o', 'o*o'] 
        # else:
        #     patterns=['<*-'] 

        canonical_dag_links = self._get_canonical_dag_from_graph(graph)

        max_lag = 0
        for node in XYZ:
            j, tau = node   # tau <= 0
            max_lag = max(max_lag, abs(tau))

            causal_path = []
            queue = [(node, causal_path)]

            while queue:
                varlag, causal_path = queue.pop()
                causal_path = [varlag] + causal_path

                var, lag = varlag
                for partmp in canonical_dag_links[var]:
                    i, tautmp = partmp
                    # Get shifted lag since canonical_dag_links is at t=0
                    tau = tautmp + lag
                    par = (i, tau)

                    if (par not in causal_path):
                    
                        if len(causal_path) == 1:
                            queue.append((par, causal_path))
                            continue

                        if (len(causal_path) > 1) and not _repeating((par, varlag), causal_path):
                            
                                max_lag = max(max_lag, abs(tau))
                                queue.append((par, causal_path))

        return max_lag

    def _get_latent_projection_graph(self, stationary=False):
        """For DAGs/ADMGs uses the Latent projection operation (Pearl 2009).

           Assumes a normal or stationary graph with potentially unobserved nodes.
           Also allows particular time steps to be unobserved. By stationarity
           that pattern of unobserved nodes is repeated into -infinity.

           Latent projection operation for latents = nodes before t-tau_max or due to <->:
           (i)  auxADMG contains (i, -taui) --> (j, -tauj) iff there is a directed path 
                (i, -taui) --> ... --> (j, -tauj) on which
                every non-endpoint vertex is in hidden variables (= not in observed_vars)
                here iff (i, -|taui-tauj|) --> j in graph
           (ii) auxADMG contains (i, -taui) <-> (j, -tauj) iff there exists a path of the 
                form (i, -taui) <-- ... --> (j, -tauj) on
                which every non-endpoint vertex is non-collider AND in L (=not in observed_vars)
                here iff (i, -|taui-tauj|) <-> j OR there is path 
                (i, -taui) <-- nodes before t-tau_max --> (j, -tauj)
        """
        
        # graph = self.graph

        # if self.hidden_variables is None:
        #     hidden_variables_here = []
        # else:
        hidden_variables_here = self.hidden_variables

        aux_graph = np.zeros((self.N, self.N, self.tau_max + 1, self.tau_max + 1), dtype='<U3')
        aux_graph[:] = ""
        for (i, j) in itertools.product(range(self.N), range(self.N)):
            for jt, tauj in enumerate(range(0, self.tau_max + 1)):
                for it, taui in enumerate(range(0, self.tau_max + 1)):
                    tau = abs(taui - tauj)
                    if tau == 0 and j == i:
                        continue
                    if (i, -taui) in hidden_variables_here or (j, -tauj) in hidden_variables_here:
                        continue
                    # print("\n")
                    # print((i, -taui), (j, -tauj))

                    cond_i_xy = (
                            # tau <= graph_taumax 
                        # and (graph[i, j, tau] == '-->' or graph[i, j, tau] == '+->') 
                        #     )
                          # and 
                          self._check_path( #graph=graph,
                                                start=[(i, -taui)],
                                                 end=[(j, -tauj)],
                                                 conditions=None,
                                                 starts_with=['-*>', '+*>'],
                                                 ends_with=['-*>', '+*>'],
                                                 path_type='causal',
                                                 hidden_by_taumax=False,
                                                 hidden_variables=hidden_variables_here,
                                                 stationary_graph=stationary,
                                                 ))
                    cond_i_yx = (
                        # tau <= graph_taumax 
                        # and (graph[i, j, tau] == '<--' or graph[i, j, tau] == '<-+') 
                        #     )
                        # and 
                        self._check_path( #graph=graph,
                                              start=[(j, -tauj)],
                                               end=[(i, -taui)],
                                               conditions=None,
                                               starts_with=['-*>', '+*>'],
                                               ends_with=['-*>', '+*>'],
                                               path_type='causal',
                                               hidden_by_taumax=False,
                                               hidden_variables=hidden_variables_here,
                                               stationary_graph=stationary,
                                               ))
                    if stationary:
                        hidden_by_taumax_here = True
                    else:
                        hidden_by_taumax_here = False

                    cond_ii = (
                        # tau <= graph_taumax 
                                # and 
                                (
                                #     graph[i, j, tau] == '<->' 
                                # or graph[i, j, tau] == '+->' or graph[i, j, tau] == '<-+')) 
                                    self._check_path( #graph=graph,
                                                start=[(i, -taui)],
                                                 end=[(j, -tauj)],
                                                 conditions=None,
                                                 starts_with=['<**', '+**'],
                                                 ends_with=['**>', '**+'],
                                                 path_type='any',
                                                 hidden_by_taumax=hidden_by_taumax_here,
                                                 hidden_variables=hidden_variables_here,
                                                 stationary_graph=stationary,
                                                 )))

                    if cond_i_xy and not cond_i_yx and not cond_ii:
                        aux_graph[i, j, taui, tauj] = "-->"  #graph[i, j, tau]
                        # if tau == 0:
                        aux_graph[j, i, tauj, taui] = "<--"  # graph[j, i, tau]
                    elif not cond_i_xy and cond_i_yx and not cond_ii:
                        aux_graph[i, j, taui, tauj] = "<--"  #graph[i, j, tau]
                        # if tau == 0:
                        aux_graph[j, i, tauj, taui] = "-->"  # graph[j, i, tau]
                    elif not cond_i_xy and not cond_i_yx and cond_ii:
                        aux_graph[i, j, taui, tauj] = '<->'
                        # if tau == 0:
                        aux_graph[j, i, tauj, taui] = '<->'
                    elif cond_i_xy and not cond_i_yx and cond_ii:
                        aux_graph[i, j, taui, tauj] = '+->'
                        # if tau == 0:
                        aux_graph[j, i, tauj, taui] = '<-+'                        
                    elif not cond_i_xy and cond_i_yx and cond_ii:
                        aux_graph[i, j, taui, tauj] = '<-+'
                        # if tau == 0:
                        aux_graph[j, i, tauj, taui] = '+->' 
                    elif cond_i_xy and cond_i_yx:
                        raise ValueError("Cycle between %s and %s!" %(str(i, -taui), str(j, -tauj)))
                    # print(aux_graph[i, j, taui, tauj])

                    # print((i, -taui), (j, -tauj), cond_i_xy, cond_i_yx, cond_ii, aux_graph[i, j, taui, tauj], aux_graph[j, i, tauj, taui])

        return aux_graph

    def _check_path(self, 
        # graph, 
        start, end,
        conditions=None, 
        starts_with=None,
        ends_with=None,
        path_type='any',
        # causal_children=None,
        stationary_graph=False,
        hidden_by_taumax=False,
        hidden_variables=None,
        ):
        """Check whether an open/active path between start and end given conditions exists.
        
        Also allows to restrict start and end patterns and to consider causal/non-causal paths

        hidden_by_taumax and hidden_variables are relevant for the latent projection operation.
        """


        if conditions is None:
            conditions = set([])
        # if conditioned_variables is None:
        #     S = []

        start = set(start)
        end = set(end)
        conditions = set(conditions)
        
        # Get maximal possible time lag of a connecting path
        # See Thm. XXXX - TO BE REVISED!
        XYZ = start.union(end).union(conditions)
        if stationary_graph:
            max_lag = 10*self.tau_max  # TO BE REVISED! self._get_maximum_possible_lag(XYZ, self.graph)
            causal_children = list(self._get_mediators_stationary_graph(start, end, max_lag).union(end))
        else:
            max_lag = None
            causal_children = list(self.get_mediators(start, end).union(end))
       
        # if hidden_variables is None:
        #     hidden_variables = set([])

        if hidden_by_taumax:
            if hidden_variables is None:
                hidden_variables = set([])
            hidden_variables = hidden_variables.union([(k, -tauk) for k in range(self.N) 
                                            for tauk in range(self.tau_max+1, max_lag + 1)])

        # print("causal_children ", causal_children)

        if starts_with is None:
            starts_with = ['***']
        elif type(starts_with) == str:
            starts_with = [starts_with]

        if ends_with is None:
            ends_with = ['***']
        elif type(ends_with) == str:
            ends_with = [ends_with]
        #
        # Breadth-first search to find connection
        #
        # print("\nstart, starts_with, ends_with, end ", start, starts_with, ends_with, end)
        # print("hidden_variables ", hidden_variables)
        start_from = set()
        for x in start:
            if stationary_graph:
                link_neighbors = self._get_adjacents_stationary_graph(graph=self.graph, node=x, patterns=starts_with, 
                                        max_lag=max_lag, exclude=list(start))
            else:
                link_neighbors = self._find_adj(node=x, patterns=starts_with, exclude=list(start), return_link=True)
            
            for link_neighbor in link_neighbors:
                link, neighbor = link_neighbor

                # if before_taumax and neighbor[1] >= -self.tau_max:
                #     continue

                if (hidden_variables is not None and neighbor not in end
                                    and neighbor not in hidden_variables):
                    continue

                if path_type == 'non_causal':
                    if (neighbor in causal_children and self._match_link('-*>', link) 
                        and not self._match_link('+*>', link)):
                        continue
                elif path_type == 'causal':
                    if (neighbor not in causal_children): # or self._match_link('<**', link)):
                        continue                    
                start_from.add((x, link, neighbor))

        # print("start, end, start_from ", start, end, start_from)

        visited = set()
        for (varlag_i, link_ik, varlag_k) in start_from:
            visited.add((link_ik, varlag_k))

        # Traversing through motifs i *-* k *-* j
        while start_from:

            # print("Continue ", start_from)
            # for (link_ik, varlag_k) in start_from:
            removables = []
            for (varlag_i, link_ik, varlag_k) in start_from:

                # print("varlag_k in end ", varlag_k in end, link_ik)
                if varlag_k in end:
                    if np.any([self._match_link(patt, link_ik) for patt in ends_with]):
                        # print("Connected ", varlag_i, link_ik, varlag_k)
                        return True
                    else:
                        removables.append((varlag_i, link_ik, varlag_k))

            for removable in removables:
                start_from.remove(removable)
            if len(start_from)==0:
                return False

            # Get any neighbor from starting nodes
            # link_ik, varlag_k = start_from.pop()
            varlag_i, link_ik, varlag_k = start_from.pop()

            # print("Get k = ", link_ik, varlag_k)
            # print("start_from ", start_from)
            # print("visited    ", visited)

            if stationary_graph:
                link_neighbors = self._get_adjacents_stationary_graph(graph=self.graph, node=varlag_k, patterns='***', 
                                        max_lag=max_lag, exclude=list(start))
            else:
                link_neighbors = self._find_adj(node=varlag_k, patterns='***', exclude=list(start), return_link=True)
            
            # print("link_neighbors ", link_neighbors)
            for link_neighbor in link_neighbors:
                link_kj, varlag_j = link_neighbor
                # print("Walk ", link_ik, varlag_k, link_kj, varlag_j)

                # print ("visited ", (link_kj, varlag_j), visited)
                if (link_kj, varlag_j) in visited:
                # if (varlag_i, link_kj, varlag_j) in visited:
                    # print("in visited")
                    continue
                # print("Not in visited")

                if path_type == 'causal':
                    if not (self._match_link('-*>', link_kj) or self._match_link('+*>', link_kj)):
                        continue 

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

                # if (before_taumax and varlag_j not in end 
                #     and varlag_j[1] >= -self.tau_max):
                #     # print("before_taumax ", varlag_j)
                #     continue

                if (hidden_variables is not None and varlag_j not in end
                                    and varlag_j not in hidden_variables):
                    continue

                # Motif is open
                # print("Motif open ", link_ik, varlag_k, link_kj, varlag_j )
                # start_from.add((link_kj, varlag_j))
                visited.add((link_kj, varlag_j))
                start_from.add((varlag_k, link_kj, varlag_j))
                # visited.add((varlag_k, link_kj, varlag_j))


        # print("Separated")
        return False


    def _get_causal_paths(self, source_nodes, target_nodes,
        mediators=None,
        mediated_through=None,
        proper_paths=True,
        ):
        """Returns causal paths via depth-first search.

        Allows to restrict paths through mediated_through.

        """

        source_nodes = set(source_nodes)
        target_nodes = set(target_nodes)

        if mediators is None:
            mediators = set()
        else:
            mediators = set(mediators)

        if mediated_through is None:
            mediated_through = []
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
                    if ((-self.tau_max <= tau <= 0) # or self.ignore_time_bounds)
                        and node not in causal_path):

                        queue.append((node, causal_path)) 
 
                        if node in target_nodes:  
                            if len(mediated_through) > 0 and len(set(causal_path).intersection(mediated_through)) == 0:
                                continue
                            else:
                                all_causal_paths[w][node].append(causal_path + [node]) 

        return all_causal_paths

    @staticmethod
    def get_dict_from_graph(graph, parents_only=False):
        """Helper function to convert graph to dictionary of links.

        Parameters
        ---------
        graph : array of shape (N, N, tau_max+1)
            Matrix format of graph in string format.

        parents_only : bool
            Whether to only return parents ('-->' in graph)

        Returns
        -------
        links : dict
            Dictionary of form {0:{(0, -1): o-o, ...}, 1:{...}, ...}.
        """
        N = graph.shape[0]

        links = dict([(j, {}) for j in range(N)])

        if parents_only:
            for (i, j, tau) in zip(*np.where(graph=='-->')):
                links[j][(i, -tau)] = graph[i,j,tau]
        else:
            for (i, j, tau) in zip(*np.where(graph!='')):
                links[j][(i, -tau)] = graph[i,j,tau]

        return links

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
