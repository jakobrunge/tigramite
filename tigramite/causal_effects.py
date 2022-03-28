"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

import numpy as np
import itertools
from copy import deepcopy
from collections import defaultdict
from tigramite.models import Models
import struct

class CausalEffects():
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

        # 
        # Checks regarding graph type
        #
        supported_graphs = ['dag', 
                            'admg',
                            'tsg_dag',
                            'tsg_admg',
                            'stationary_dag',
                            # 'stationary_admg',

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
        
        self.hidden_variables = set(hidden_variables)
        if len(self.hidden_variables.intersection(self.X.union(self.Y).union(self.S))) > 0:
            raise ValueError("XYS overlaps with hidden_variables!")



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
        # (self.graph, self.graph_type, 
        #  self.tau_max, self.hidden_variables) = 

        self._construct_graph(graph=graph, graph_type=graph_type,
                              hidden_variables=hidden_variables)

        # print(self.graph.shape)
        self._check_graph(self.graph)

        self.ancX = self._get_ancestors(X)
        self.ancY = self._get_ancestors(Y)
        self.ancS = self._get_ancestors(S)

        # If X is not in anc(Y), then no causal link exists
        if self.ancY.intersection(set(X)) == set():
            self.no_causal_path = True
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")

            # raise ValueError("No causal path from X to Y exists.")
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
            raise ValueError("Overlap between X and Y")

        if len(self.S.intersection(self.Y.union(self.X))) > 0:
            raise ValueError("Conditions S overlap with X or Y")

        # # TODO: need to prove that this is sufficient for non-identifiability!
        # if len(self.X.intersection(self._get_descendants(self.M))) > 0:
        #     raise ValueError("Not identifiable: Overlap between X and des(M)")

        if check_SM_overlap and len(self.S.intersection(self.M)) > 0:
            raise ValueError("Conditions S overlap with mediators M!")

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
            raise ValueError("Not identifiable: Conditions S overlap with des(Y)")

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
            self.tau_max = 0

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
            self.tau_max = graph.shape[2] - 1

            if len(hidden_variables) > 0:
                self.graph = self._get_latent_projection_graph() #, stationary=False)
                self.graph_type = "tsg_admg"
            else:
                self.graph_type = graph_type   

        elif graph_type in ['stationary_dag']:
            # Currently only stationary_dag without hidden variables is supported
            if graph.ndim != 3:
                raise ValueError("stationary graph_type assumes graph.shape=(N, N, tau_max+1).")
            # TODO: remove if theory for stationary ADMGs is clear
            if graph_type == 'stationary_dag' and len(hidden_variables) > 0:
                raise ValueError("Hidden variables currently not supported for "
                                 "stationary_dag.")

            # For a stationary DAG without hidden variables it's sufficient to consider
            # a tau_max that includes the parents of X, Y, M, and S. A conservative
            # estimate thereof is simply the lag-dimension of the stationary DAG plus
            # the maximum lag of XYS.
            statgraph_tau_max = graph.shape[2] - 1
            maxlag_XYS = 0
            for varlag in self.X.union(self.Y).union(self.S):
                maxlag_XYS = max(maxlag_XYS, abs(varlag[1]))

            self.tau_max = maxlag_XYS + statgraph_tau_max

            stat_graph = deepcopy(graph)

            allowed_edges = ["-->", "<--"]

            # Construct tsg_graph
            graph = np.zeros((self.N, self.N, self.tau_max + 1, self.tau_max + 1), dtype='<U3')
            graph[:] = ""
            for (i, j) in itertools.product(range(self.N), range(self.N)):
                for jt, tauj in enumerate(range(0, self.tau_max + 1)):
                    for it, taui in enumerate(range(tauj, self.tau_max + 1)):
                        tau = abs(taui - tauj)
                        if tau == 0 and j == i:
                            continue
                        if tau > statgraph_tau_max:
                            continue                        

                        # if tau == 0:
                        #     if stat_graph[i, j, tau] == '-->':
                        #         graph[i, j, taui, tauj] = "-->" 
                        #         graph[j, i, tauj, taui] = "<--" 

                        #     # elif stat_graph[i, j, tau] == '<--':
                        #     #     graph[i, j, taui, tauj] = "<--"
                        #     #     graph[j, i, tauj, taui] = "-->" 
                        # else:
                        if stat_graph[i, j, tau] == '-->':
                            graph[i, j, taui, tauj] = "-->" 
                            graph[j, i, tauj, taui] = "<--" 
                        elif stat_graph[i, j, tau] == '<--':
                            pass
                        elif stat_graph[i, j, tau] == '':
                            pass
                        else:
                            edge = stat_graph[i, j, tau]
                            raise ValueError("Invalid graph edge %s. " %(edge) +
                                 "For graph_type = %s only %s are allowed." %(graph_type, str(allowed_edges)))
      


                        # elif stat_graph[i, j, tau] == '<--':
                        #     graph[i, j, taui, tauj] = "<--"
                        #     graph[j, i, tauj, taui] = "-->" 

            self.graph_type = 'tsg_dag'
            self.graph = graph


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
        potential_mediators = set()
        for y in end:
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
                                node=varlag, patterns="<*-", max_lag=max_lag, exclude=None):
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
                if link[0] != '+':
                    if link[0] != left_mark: return False

            if right_mark != '*':
                if link[2] != '+':
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

    # @profile
    def _get_children(self, varlag):
        """Returns set of children (varlag --> ...) for (lagged) varlag."""
        if self.possible:
            patterns=['-*>', 'o*o', 'o*>']
        else:
            patterns=['-*>']
        return self._find_adj(node=varlag, patterns=patterns)

    def _get_parents(self, varlag):
        """Returns set of parents (varlag <-- ...) for (lagged) varlag."""
        if self.possible:
            patterns=['<*-', 'o*o', '<*o']
        else:
            patterns=['<*-']
        return self._find_adj(node=varlag, patterns=patterns)

    def _get_spouses(self, varlag):
        """Returns set of spouses (varlag <-> ...)  for (lagged) varlag."""
        return self._find_adj(node=varlag, patterns=['<*>'])

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
                                node=varlag, patterns="-*>", max_lag=max_lag, exclude=None):
                        i, tau = child
                        if (child not in descendants 
                            # and (-self.tau_max <= tau <= 0 or self.ignore_time_bounds)
                            ):
                            descendants = descendants.union(set([child]))
                            next_level.append(child)

                this_level = next_level       

        return descendants

    # @profile
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
                                                 starts_with='-*>',
                                                 ends_with='-*>',
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
                                               starts_with='-*>',
                                               ends_with='-*>',
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
                                                 starts_with='<**',
                                                 ends_with='**>',
                                                 path_type='any',
                                                 hidden_by_taumax=hidden_by_taumax_here,
                                                 hidden_variables=hidden_variables_here,
                                                 stationary_graph=stationary,
                                                 )))

                    # print((i, -taui), (j, -tauj), cond_i_xy, cond_i_yx, cond_ii)

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

        return aux_graph

    # @profile
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
            max_lag = self._get_maximum_possible_lag(XYZ, self.graph)
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

        # print("hidden_variables ", hidden_variables)
        if starts_with is None:
            starts_with = '***'

        if ends_with is None:
            ends_with = '***'

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
            
            # print("link_neighbors ", link_neighbors)
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
                    if (neighbor not in causal_children or self._match_link('<**', link)):
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
                    if self._match_link(ends_with, link_ik):
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
                    if not self._match_link('-*>', link_kj):
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

    # @profile
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
            self.vancs = self.ancX.union(self.ancY).union(newancS) - self.forbidden_nodes

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
                            ends_with='**>')): 
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
   
        # print("Optimality = ", cond_0, cond_I, cond_II)
        optimality = (cond_0 or (cond_I and cond_II))
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

    # @profile
    def fit_total_effect(self,
        dataframe, 
        estimator,
        adjustment_set='optimal',
        conditional_estimator=None,  
        data_transform=None,
        mask_type=None,
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
        """

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return self

        self.dataframe = dataframe
        self.conditional_estimator = conditional_estimator

        if adjustment_set == 'optimal':
            # Check optimality and use either optimal or colliders_only set
            adjustment_set = self.get_optimal_set()
        elif adjustment_set == 'colliders_minimized_optimal':
            adjustment_set = self.get_optimal_set(minimize='colliders_only')
        elif adjustment_set == 'minimized_optimal':
            adjustment_set = self.get_optimal_set(minimize=True)
        else:
            if self._check_validity(adjustment_set) is False:
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
                cut_off='max_lag_or_tau_max',
                return_data=False)

        return self

    # @profile
    def predict_total_effect(self, 
        intervention_data, 
        conditions_data=None,
        pred_params=None,
        return_further_pred_results=False,
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

        Returns
        -------
        Results from prediction: an array of shape  (time, len(Y)).
        """

        if intervention_data.shape[1] != len(self.listX):
            raise ValueError("intervention_data.shape[1] must be len(X).")

        if conditions_data is not None:
            if conditions_data.shape[1] != len(self.listS):
                raise ValueError("conditions_data.shape[1] must be len(S).")
            if conditions_data.shape[0] != intervention_data.shape[0]:
                raise ValueError("conditions_data.shape[0] must match intervention_data.shape[0].")

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return np.zeros((len(intervention_data), len(self.listY)))

        effect = self.model.get_general_prediction(
            intervention_data=intervention_data,
            conditions_data=conditions_data,
            pred_params=pred_params,
            return_further_pred_results=return_further_pred_results) 

        return effect

    # @profile
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
        data_transform : sklearn preprocessing object, optional (default: None)
            Used to transform data prior to fitting. For example,
            sklearn.preprocessing.StandardScaler for simple standardization. The
            fitted parameters are stored.
        mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
            Masking mode: Indicators for which variables in the dependence
            measure I(X; Y | Z) the samples should be masked. If None, the mask
            is not used. Explained in tutorial on masking and missing values.
        """

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return self

        import sklearn.linear_model

        self.dataframe = dataframe
        estimator = sklearn.linear_model.LinearRegression()

        # Fit model of Y on X and Z (and conditions)
        # Build the model
        self.model = Models(
                        dataframe=dataframe,
                        model=estimator,
                        data_transform=data_transform,
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
                for ipar, par_coeff in enumerate(links_coeffs[medy[0]]):
                    par, coeff, _ = par_coeff
                    max_lag = max(abs(par[1]), max_lag)
                    coeffs[medy][par] = coeff #self.fit_results[j][(j, 0)]['model'].coef_[ipar]

            self.model.tau_max = max_lag

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
                    if oset is False:
                        raise ValueError("Not identifiable via Wright's method.")
                    fit_res = self.model.get_general_fitted_model(
                        Y=[medy], X=[par], Z=oset,
                        tau_max=self.tau_max,
                        cut_off='max_lag_or_tau_max',
                        return_data=False)
                    coeffs[medy][par] = fit_res[medy]['model'].coef_[0]
                    # print(mediators, par, medy, coeffs[medy][par])

        elif method == 'parents':
            if 'dag' not in self.graph_type:
                raise ValueError("method == 'parents' only possible for DAGs")

            coeffs = {}
            for medy in [med for med in mediators] + [y for y in self.listY]:
                coeffs[medy] = {}
                # mediator_parents = self._get_all_parents([medy]).intersection(mediators.union(self.X)) - set([medy])
                all_parents = self._get_all_parents([medy]) - set([medy])
                # print(j, all_parents[j])
                # if len(all_parents[j]) > 0:
                fit_res = self.model.get_general_fitted_model(
                    Y=[medy], X=list(all_parents), Z=[],
                    conditions=None,
                    tau_max=self.tau_max,
                    cut_off='max_lag_or_tau_max',
                    return_data=False)

                for ipar, par in enumerate(all_parents):
                    coeffs[medy][par] = fit_res[medy]['model'].coef_[ipar]

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
                    effect_here *= coeffs[(j, tauj)][(i, taui)]

                effect[(x, y)] += effect_here
               
        # Make fitted coefficients available as attribute
        self.coeffs = coeffs

        # Modify and overwrite variables in self.model
        self.model.Y = self.listY
        self.model.X = self.listX  
        self.model.Z = []
        self.model.conditions = [] 
        self.model.cut_off = 'max_lag_or_tau_max'

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

    # @profile 
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
        if intervention_data.shape[1] != len(self.X):
            raise ValueError("intervention_data.shape[1] must be len(X).")

        if self.no_causal_path:
            if self.verbosity > 0:
                print("No causal path from X to Y exists.")
            return np.zeros((len(intervention_data), len(self.Y)))

        intervention_T, lenX = intervention_data.shape

        lenY = len(self.listY)

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

            # Transform the data if needed
            a_transform = self.model.fit_results[y]['data_transform']
            if a_transform is not None:
                intervention_data = a_transform.transform(X=intervention_data)
        

            # Now iterate through interventions (and potentially S)
            for index, dox_vals in enumerate(intervention_data):
                # Construct XZS-array
                intervention_array = dox_vals.reshape(1, lenX) 
                predictor_array = intervention_array

                predicted_vals = self.model.fit_results[y]['model'].predict(
                X=predictor_array, **pred_params)
                predicted_array[index, iy] = predicted_vals.mean()

        return predicted_array

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

    import sklearn
    from sklearn.linear_model import LinearRegression

    T = 1000
    def lin_f(x): return x
    auto_coeff = 0.3
    coeff = 2.
    links = {
            0: [((0, -1), auto_coeff, lin_f)], 
            1: [((1, -1), auto_coeff, lin_f), ((0, -1), coeff, lin_f)], 
            2: [((2, -1), auto_coeff, lin_f), ((1, 0), coeff, lin_f)],
            }
    data, nonstat = toys.structural_causal_process(links, T=T, 
                                noises=None, seed=7)

    # Create some missing values
    data[-10:,:] = 999.
    var_names = range(5)
    dataframe = pp.DataFrame(data, var_names=var_names,
     missing_flag=999.) 


    # Construct expert knowledge graph from links here 
    links = {0: [(0, -1)],
             1: [(1, -1), (0, -1)],
             2: [(2, -1), (1, 0),],
             }
    # Use staticmethod to get graph
    graph = CausalEffects.get_graph_from_dict(links, tau_max=None)
    
    # We are interested in lagged total effect of X on Y
    X = [(0, -1)]
    Y = [(2, 0)]

    # Initialize class as `stationary_dag`
    causal_effects = CausalEffects(graph, graph_type='stationary_dag', 
                                X=X, Y=Y, S=None, 
                                hidden_variables=None, 
                                verbosity=5)

    print(data)
    # Optimal adjustment set (is used by default)
    # print(causal_effects.get_optimal_set())

    # # Fit causal effect model from observational data
    # causal_effects.fit_total_effect(
    #     dataframe=dataframe, 
    #     # mask_type='y',
    #     estimator=LinearRegression(),
    #     )

    # # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    # dox_vals = np.linspace(0., 1., 5)
    # intervention_data = dox_vals.reshape(len(dox_vals), len(X))
    # pred_Y = causal_effects.predict_total_effect( 
    #         intervention_data=intervention_data)
    # print(pred_Y)

    # Fit causal effect model from observational data
    causal_effects.fit_wright_effect(
        dataframe=dataframe, 
        # mask_type='y',
        # estimator=LinearRegression(),
        )

    # Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
    dox_vals = np.linspace(0., 1., 5)
    intervention_data = dox_vals.reshape(len(dox_vals), len(X))
    pred_Y = causal_effects.predict_wright_effect( 
            intervention_data=intervention_data)
    print(pred_Y)