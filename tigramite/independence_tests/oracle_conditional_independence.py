"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import numpy as np

from collections import defaultdict, OrderedDict
from itertools import combinations, permutations


class OracleCI:
    r"""Oracle of conditional independence test X _|_ Y | Z given a graph.

    Class around link_coeff causal ground truth. X _|_ Y | Z is based on
    assessing whether X and Y are d-separated given Z in the graph.

    Class can be used just like a Tigramite conditional independence class
    (e.g., ParCorr). The main use is for unit testing of PCMCI methods.

    Parameters
    ----------
    graph : array of shape [N, N, tau_max+1]
        Causal graph.
    links : dict
        Dictionary of form {0:[(0, -1), ...], 1:[...], ...}.
        Alternatively can also digest {0: [((0, -1), coeff, func)], ...}.
    observed_vars : None or list, optional (default: None)
        Subset of keys in links definining which variables are 
        observed. If None, then all variables are observed.
    selection_vars : None or list, optional (default: None)
        Subset of keys in links definining which variables are 
        selected (= always conditioned on at every time lag).
        If None, then no variables are selected.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    # documentation
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 links=None,
                 observed_vars=None,
                 selection_vars=None,
                 graph=None,
                 graph_is_mag=False,
                 tau_max=None,
                 verbosity=0):

        self.tau_max = tau_max
        self.graph_is_mag = graph_is_mag

        if links is None:
            if graph is None:
                raise ValueError("Either links or graph must be specified!")
            else:
                # Get canonical DAG from graph, potentially interpreted as MAG
                # self.tau_max = graph.shape[2]
                (links, 
                 observed_vars, 
                 selection_vars) = self.get_links_from_graph(graph)
                # # TODO make checks and tau_max?
                # self.graph = graph


        self.verbosity = verbosity
        self._measure = 'oracle_ci'
        self.confidence = None
        self.links = links
        self.N = len(links)
        # self.tau_max = self._get_minmax_lag(self.links)

        # Initialize already computed dsepsets of X, Y, Z
        self.dsepsets = {}

        # Initialize observed vars
        self.observed_vars = observed_vars
        if self.observed_vars is None:
            self.observed_vars = range(self.N)
        else:
            if not set(self.observed_vars).issubset(set(range(self.N))):
                raise ValueError("observed_vars must be subset of range(N).")
            if self.observed_vars != sorted(self.observed_vars):
                raise ValueError("observed_vars must ordered.")
            if len(self.observed_vars) != len(set(self.observed_vars)):
                raise ValueError("observed_vars must not contain duplicates.")

        self.selection_vars = selection_vars

        if self.selection_vars is not None:
            if not set(self.selection_vars).issubset(set(range(self.N))):
                raise ValueError("selection_vars must be subset of range(N).")
            if self.selection_vars != sorted(self.selection_vars):
                raise ValueError("selection_vars must ordered.")
            if len(self.selection_vars) != len(set(self.selection_vars)):
                raise ValueError("selection_vars must not contain duplicates.")
        else:
            self.selection_vars = []

        # ToDO: maybe allow to use user-tau_max, otherwise deduced from links
        self.graph = self.get_graph_from_links(tau_max=tau_max)

        self.ci_results = {}

    def set_dataframe(self, dataframe):
        """Dummy function."""
        pass

    def _check_XYZ(self, X, Y, Z):
        """Checks variables X, Y, Z.

        Parameters
        ----------
        X, Y, Z : list of tuples
            For a dependence measure I(X;Y|Z), Y is of the form [(varY, 0)],
            where var specifies the variable index. X typically is of the form
            [(varX, -tau)] with tau denoting the time lag and Z can be
            multivariate [(var1, -lag), (var2, -lag), ...] .

        Returns
        -------
        X, Y, Z : tuple
            Cleaned X, Y, Z.
        """
        # Get the length in time and the number of nodes
        N = self.N

        # Remove duplicates in X, Y, Z
        X = list(OrderedDict.fromkeys(X))
        Y = list(OrderedDict.fromkeys(Y))
        Z = list(OrderedDict.fromkeys(Z))

        # If a node in Z occurs already in X or Y, remove it from Z
        Z = [node for node in Z if (node not in X) and (node not in Y)]

        # Check that all lags are non-positive and indices are in [0,N-1]
        XYZ = X + Y + Z
        dim = len(XYZ)
        # Ensure that XYZ makes sense
        if np.array(XYZ).shape != (dim, 2):
            raise ValueError("X, Y, Z must be lists of tuples in format"
                             " [(var, -lag),...], eg., [(2, -2), (1, 0), ...]")
        if np.any(np.array(XYZ)[:, 1] > 0):
            raise ValueError("nodes are %s, " % str(XYZ) +
                             "but all lags must be non-positive")
        if (np.any(np.array(XYZ)[:, 0] >= N)
                or np.any(np.array(XYZ)[:, 0] < 0)):
            raise ValueError("var indices %s," % str(np.array(XYZ)[:, 0]) +
                             " but must be in [0, %d]" % (N - 1))
        if np.all(np.array(Y)[:, 1] != 0):
            raise ValueError("Y-nodes are %s, " % str(Y) +
                             "but one of the Y-nodes must have zero lag")

        return (X, Y, Z)

    def _get_lagged_parents(self, var_lag, exclude_contemp=False,
                only_non_causal_paths=False, X=None, causal_children=None):
        """Helper function to yield lagged parents for var_lag from
        self.links_coeffs.

        Parameters
        ----------
        var_lag : tuple
            Tuple of variable and lag which is assumed <= 0.
        exclude_contemp : bool
            Whether contemporaneous links should be exluded.

        Yields
        ------
        Next lagged parent.
        """

        var, lag = var_lag

        for link_props in self.links[var]:
            if len(link_props) == 3:
                i, tau = link_props[0]
                coeff = link_props[1]
            else:
                i, tau = link_props
                coeff = 1.
            if coeff != 0.:
                if not (exclude_contemp and lag == 0):
                    if only_non_causal_paths:
                        if not ((i, lag + tau) in X and var_lag in causal_children):
                            yield (i, lag + tau)
                    else:
                        yield (i, lag + tau)

    def _get_children(self):
        """Helper function to get children from links.

        Note that for children the lag is positive.

        Returns
        -------
        children : dict
            Dictionary of form {0:[(0, 1), (3, 0), ...], 1:[], ...}.
        """

        N = len(self.links)
        children = dict([(j, []) for j in range(N)])

        for j in range(N):
            for link_props in self.links[j]:
                if len(link_props) == 3:
                    i, tau = link_props[0]
                    coeff = link_props[1]
                else:
                    i, tau = link_props
                    coeff = 1.
                if coeff != 0.:
                    children[i].append((j, abs(tau)))

        return children

    def _get_lagged_children(self, var_lag, children, exclude_contemp=False,
           only_non_causal_paths=False, X=None, causal_children=None):
        """Helper function to yield lagged children for var_lag from children.

        Parameters
        ----------
        var_lag : tuple
            Tuple of variable and lag which is assumed <= 0.
        children : dict
            Dictionary of form {0:[(0, 1), (3, 0), ...], 1:[], ...}.
        exclude_contemp : bool
            Whether contemporaneous links should be exluded.

        Yields
        ------
        Next lagged child.
        """

        var, lag = var_lag
        # lagged_parents = []

        for child in children[var]:
            k, tau = child
            if not (exclude_contemp and tau == 0):
                # lagged_parents.append((i, lag + tau))
                if only_non_causal_paths:
                    if not (var_lag in X and (k, lag + tau) in causal_children):
                        yield (k, lag + tau)
                else:
                    yield (k, lag + tau)

    def _get_non_blocked_ancestors(self, Y, conds=None, mode='non_repeating',
                                    max_lag=None):
        """Helper function to return the non-blocked ancestors of variables Y.

        Returns a dictionary of ancestors for every y in Y. y is a tuple (
        var, lag) where lag <= 0. All ancestors with directed paths towards y
        that are not blocked by conditions in conds are included. In mode
        'non_repeating' an ancestor X^i_{t-\tau_i} with link X^i_{t-\tau_i}
        --> X^j_{ t-\tau_j} is only included if X^i_{t'-\tau_i} --> X^j_{
        t'-\tau_j} is not already part of the ancestors. The most lagged
        ancestor for every variable X^i defines the maximum ancestral time
        lag, which is also returned. In mode 'max_lag' ancestors are included
        up to the maximum time lag max_lag.

        It's main use is to return the maximum ancestral time lag max_lag of
        y in Y for every variable in self.links_coeffs.

        Parameters
        ----------
        Y : list of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.
        conds : list of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.
        mode : {'non_repeating', 'max_lag'}
            Whether repeating links should be excluded or ancestors should be
            followed up to max_lag.
        max_lag : int
            Maximum time lag to include ancestors.

        Returns
        -------
        ancestors : dict
            Includes ancestors for every y in Y.
        max_lag : int
            Maximum time lag to include ancestors.
        """

        def _repeating(link, seen_links):
            """Returns True if a link or its time-shifted version is already
            included in seen_links."""
            i, taui = link[0]
            j, tauj = link[1]

            for seen_link in seen_links:
                seen_i, seen_taui = seen_link[0]
                seen_j, seen_tauj = seen_link[1]

                if (i == seen_i and j == seen_j
                    and abs(tauj-taui) == abs(seen_tauj-seen_taui)):
                    return True

            return False

        if conds is None:
            conds = []

        conds = [z for z in conds if z not in Y]

        N = len(self.links)

        # Initialize max. ancestral time lag for every N
        if mode == 'non_repeating':
            max_lag = 0
        else:
            if max_lag is None:
                raise ValueError("max_lag must be set in mode = 'max_lag'")

        if self.selection_vars is not None:
            for selection_var in self.selection_vars:
                # print (selection_var, conds)
                # print([(selection_var, -tau_sel) for tau_sel in range(0, max_lag + 1)])
                conds += [(selection_var, -tau_sel) for tau_sel in range(0, max_lag + 1)]

        ancestors = dict([(y, []) for y in Y])

        for y in Y:
            j, tau = y   # tau <= 0
            if mode == 'non_repeating':
                max_lag = max(max_lag, abs(tau))
            seen_links = []
            this_level = [y]
            while len(this_level) > 0:
                next_level = []
                for varlag in this_level:
                    for par in self._get_lagged_parents(varlag):
                        i, tau = par
                        if par not in conds and par not in ancestors[y]:
                            if ((mode == 'non_repeating' and
                                not _repeating((par, varlag), seen_links)) or
                                (mode == 'max_lag' and
                                 abs(tau) <= abs(max_lag))):
                                    ancestors[y].append(par)
                                    if mode == 'non_repeating':
                                        max_lag = max(max_lag,
                                                         abs(tau))
                                    next_level.append(par)
                                    seen_links.append((par, varlag))

                this_level = next_level

        return ancestors, max_lag

    def _get_maximum_possible_lag(self, XYZ):
        """Helper function to return the maximum time lag of any confounding path.

        This is still based on a conjecture!

        The conjecture states that if and only if X and Y are d-connected given Z
        in a stationary DAG, then there exists a confounding path with a maximal
        time lag (i.e., the node on that path with maximal lag) given as follows:
        For any node in XYZ consider all non-repeating causal paths from the past
        to that node, where non-repeating means that a link X^i_{t-\tau_i}
        --> X^j_{ t-\tau_j} is only traversed if X^i_{t'-\tau_i} --> X^j_{
        t'-\tau_j} is not already part of that path. The most lagged
        ancestor for every variable node in XYZ defines the maximum ancestral time
        lag, which is returned.

        Parameters
        ----------
        XYZ : list of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.

        Returns
        -------
        max_lag : int
            Maximum time lag of non-repeating causal path ancestors.
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

        N = len(self.links)

        # Initialize max. ancestral time lag for every N
        max_lag = 0
    
        # Not sure whether this is relevant!
        # if self.selection_vars is not None:
        #     for selection_var in self.selection_vars:
        #         # print (selection_var, conds)
        #         # print([(selection_var, -tau_sel) for tau_sel in range(0, max_lag + 1)])
        #         conds += [(selection_var, -tau_sel) for tau_sel in range(0, max_lag + 1)]

        # ancestors = dict([(y, []) for y in Y])

        for y in XYZ:
            j, tau = y   # tau <= 0
            max_lag = max(max_lag, abs(tau))
                
            causal_path = []
            queue = [(y, causal_path)]

            while queue:
                varlag, causal_path = queue.pop()
                causal_path = [varlag] + causal_path

                for node in self._get_lagged_parents(varlag):
                    i, tau = node

                    if (node not in causal_path):
                    
                        if len(causal_path) == 1:
                            queue.append((node, causal_path))
                            continue

                        if (len(causal_path) > 1) and not _repeating((node, varlag), causal_path):
                            
                                max_lag = max(max_lag, abs(tau))
                                queue.append((node, causal_path))

        if self.verbosity > 0:
            print("Max. non-repeated ancestral time lag: ", max_lag)

        # ATTENTION: this may not find correct common ancestors, therefore multiply by 10
        # until the problem is solved
        max_lag *= 10

        return max_lag

    def _get_descendants(self, W, children, max_lag, ignore_time_bounds=False):
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
                    for child in self._get_lagged_children(varlag, children):
                        i, tau = child
                        if (child not in descendants 
                            and (-max_lag <= tau <= 0 or ignore_time_bounds)):
                            descendants = descendants.union(set([child]))
                            next_level.append(child)

                this_level = next_level       

        return list(descendants)

    def _has_any_path(self, X, Y, conds, max_lag=None, 
        starts_with=None, ends_with=None,
        directed=False,
        forbidden_nodes=None,
        only_non_causal_paths=False,
        check_optimality_cond=False,
        optimality_cond_des_YM=None,
        optimality_cond_Y=None,
        only_collider_paths_with_vancs=False,
        XYS=None,
        return_path=False):
        """Returns True if X and Y are d-connected by any open path.

        Does breadth-first search from both X and Y and meets in the middle.
        Paths are walked according to the d-separation rules where paths can
        only traverse motifs <-- v <-- or <-- v --> or --> v --> or
        --> [v] <-- where [.] indicates that v is conditioned on.
        Furthermore, paths nodes (v, t) need to fulfill max_lag <= t <= 0
        and links cannot be traversed backwards.

        Parameters
        ----------
        X, Y : lists of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.
        conds : list of tuples
            Of the form [(var, -tau)], where var specifies the variable
            index and tau the time lag.
        max_lag : int
            Maximum time lag.
        starts_with : {None, 'tail', 'arrohead'}
            Whether to only consider paths starting with particular mark at X.
        ends_with : {None, 'tail', 'arrohead'}
            Whether to only consider paths ending with particular mark at Y.
        """
        if max_lag is None:
            if conds is None:
                conds = []
            max_lag = self._get_maximum_possible_lag(X+Y+conds)

        def _walk_to_parents(v, fringe, this_path, other_path):
            """Helper function to update paths when walking to parents."""
            found_connection = False
            for w in self._get_lagged_parents(v, 
                only_non_causal_paths=only_non_causal_paths, X=X, 
                causal_children=causal_children):
                # Cannot walk into conditioned parents and
                # cannot walk beyond t or max_lag
                i, t = w

                if w == x and starts_with == 'arrowhead':
                    continue

                if w == y and ends_with == 'arrowhead':
                    continue

                if (w not in conds and w not in forbidden_nodes and
                    # (w, v) not in seen_links and
                    t <= 0 and abs(t) <= max_lag):
                    # if ((w, 'tail') not in this_path and 
                    #     (w, None) not in this_path):
                    if (w not in this_path or 
                        ('tail' not in this_path[w] and None not in this_path[w])):
                        if self.verbosity > 1:
                            print("Walk parent: %s --> %s  " %(v, w))
                        fringe.append((w, 'tail'))
                        if w not in this_path:
                            this_path[w] = {'tail' : (v, 'arrowhead')}
                        else:
                            this_path[w]['tail'] = (v, 'arrowhead')
                        # seen_links.append((v, w))
                    # Determine whether X and Y are connected
                    # (w, None) indicates the start or end node X/Y
                    # if ((w, 'tail') in other_path 
                    #    or (w, 'arrowhead') in other_path
                    #    or (w, None) in other_path):
                    if w in other_path:
                        found_connection = (w, 'tail') 
                        if self.verbosity > 1:
                            print("Found connection: ", found_connection)  
                        break
            return found_connection, fringe, this_path

        def _walk_to_children(v, fringe, this_path, other_path):
            """Helper function to update paths when walking to children."""
            found_connection = False
            for w in self._get_lagged_children(v, children, 
                only_non_causal_paths=only_non_causal_paths, X=X, 
                causal_children=causal_children):
                # You can also walk into conditioned children,
                # but cannot walk beyond t or max_lag
                i, t = w

                if w == x and starts_with == 'tail':
                    continue

                if w == y and ends_with == 'tail':
                    continue

                if (w not in forbidden_nodes and
                    # (w, v) not in seen_links and
                    t <= 0 and abs(t) <= max_lag):
                    # if ((w, 'arrowhead') not in this_path and 
                    #     (w, None) not in this_path):
                    if (w not in this_path or 
                        ('arrowhead' not in this_path[w] and None not in this_path[w])):
                        if self.verbosity > 1:
                            print("Walk child:  %s --> %s  " %(v, w))
                        fringe.append((w, 'arrowhead'))
                        # this_path[(w, 'arrowhead')] = (v, 'tail')
                        if w not in this_path:
                            this_path[w] = {'arrowhead' : (v, 'tail')}
                        else:
                            this_path[w]['arrowhead'] = (v, 'tail')
                        # seen_links.append((v, w))
                    # Determine whether X and Y are connected
                    # If the other_path contains w with a tail, then w must
                    # NOT be conditioned on. Alternatively, if the other_path
                    # contains w with an arrowhead, then w must be
                    # conditioned on.
                    # if (((w, 'tail') in other_path and w not in conds)
                    #    or ((w, 'arrowhead') in other_path and w in conds)
                    #    or (w, None) in other_path):
                    if w in other_path:
                        if (('tail' in other_path[w] and w not in conds) or
                            ('arrowhead' in other_path[w] and w in conds) or
                            (None in other_path[w])):
                            found_connection = (w, 'arrowhead') 
                            if self.verbosity > 1:
                                print("Found connection: ", found_connection) 
                            break
            return found_connection, fringe, this_path

        def _walk_fringe(this_level, fringe, this_path, other_path):
            """Helper function to walk each fringe, i.e., the path from X and Y,
            respectively."""
            found_connection = False

            if starts_with == 'arrowhead':
                if len(this_level) == 1 and this_level[0] == (x, None):
                    (found_connection, fringe,
                         this_path) = _walk_to_parents(x, fringe, 
                                                       this_path, other_path)
                    return found_connection, fringe, this_path, other_path

            elif starts_with == 'tail':
                if len(this_level) == 1 and this_level[0] == (x, None):
                    (found_connection, fringe,
                         this_path) = _walk_to_children(x, fringe, 
                                                       this_path, other_path)
                    return found_connection, fringe, this_path, other_path 

            if ends_with == 'arrowhead':
                if len(this_level) == 1 and this_level[0] == (y, None):
                    (found_connection, fringe,
                         this_path) = _walk_to_parents(y, fringe, 
                                                       this_path, other_path)
                    return found_connection, fringe, this_path, other_path

            elif ends_with == 'tail':
                if len(this_level) == 1 and this_level[0] == (y, None):
                    (found_connection, fringe,
                         this_path) = _walk_to_children(y, fringe, 
                                                       this_path, other_path)
                    return found_connection, fringe, this_path, other_path 

            for v, mark in this_level:
                if v in conds:
                    if (mark == 'arrowhead' or mark == None) and directed is False:
                        # Motif: --> [v] <--
                        # If standing on a condition and coming from an
                        # arrowhead, you can only walk into parents
                        (found_connection, fringe,
                         this_path) = _walk_to_parents(v, fringe, 
                                                       this_path, other_path)
                        if found_connection: break            
                else:
                    if only_collider_paths_with_vancs:
                        continue

                    if (mark == 'tail' or mark == None):
                        # Motif: <-- v <-- or <-- v -->
                        # If NOT standing on a condition and coming from
                        # a tail mark, you can walk into parents or 
                        # children
                        (found_connection, fringe,
                         this_path) = _walk_to_parents(v, fringe, 
                                                       this_path, other_path)
                        if found_connection: break 
                        
                        if not directed:
                            (found_connection, fringe,
                             this_path) = _walk_to_children(v, fringe, 
                                                           this_path, other_path)
                            if found_connection: break 
                      
                    elif mark == 'arrowhead':
                        # Motif: --> v -->
                        # If NOT standing on a condition and coming from
                        # an arrowhead mark, you can only walk into
                        # children
                        (found_connection, fringe,
                         this_path) = _walk_to_children(v, fringe, 
                                                       this_path, other_path)
                        if found_connection: break

                        if check_optimality_cond and v[0] in self.observed_vars:
                            # if v is not descendant of YM
                            # and v is not connected to Y given X OS\Cu
                            # print("v = ", v)
                            cond4a = v not in optimality_cond_des_YM
                            cond4b = not self._has_any_path(X=[v], Y=optimality_cond_Y, 
                                conds=conds + X, 
                                max_lag=None, 
                                starts_with=None,
                                ends_with=None,  
                                forbidden_nodes=None, #list(prelim_Oset), 
                                return_path=False)
                            # print(cond4a, cond4b)
                            if cond4a and cond4b:
                                (found_connection, fringe,
                                 this_path) = _walk_to_parents(v, fringe, 
                                                               this_path, other_path)
                                # print(found_connection)
                                if found_connection: break

            if self.verbosity > 1:
                print("Updated fringe: ", fringe)
            return found_connection, fringe, this_path, other_path

        def backtrace_path():
            """Helper function to get path from start point, end point, 
            and connection found."""

            path = [found_connection[0]]
            node, mark = found_connection

            if 'tail' in pred[node]:
                mark = 'tail'
            else:
                mark = 'arrowhead'
            # print(found_connection)
            while path[-1] != x:
                # print(path, node, mark, pred[node])
                prev_node, prev_mark = pred[node][mark]
                path.append(prev_node)
                if prev_mark == 'arrowhead':
                    if prev_node not in conds:
                        # if pass_through_colliders:
                        #     if 'tail' in pred[prev_node] and pred[prev_node]['tail'] != (node, mark):
                        #         mark = 'tail'
                        #     else:
                        #         mark = 'arrowhead'
                        # else:
                            mark = 'tail'
                    elif prev_node in conds:
                        mark = 'arrowhead'
                elif prev_mark == 'tail':
                    if 'tail' in pred[prev_node] and pred[prev_node]['tail'] != (node, mark):
                        mark = 'tail'
                    else:
                        mark = 'arrowhead' 
                node = prev_node

            path.reverse()

            node, mark = found_connection
            if 'tail' in succ[node]:
                mark = 'tail'
            else:
                mark = 'arrowhead'

            while path[-1] != y:
                next_node, next_mark = succ[node][mark]
                path.append(next_node)
                if next_mark == 'arrowhead':
                    if next_node not in conds:
                        # if pass_through_colliders:
                        #     if 'tail' in succ[next_node] and succ[next_node]['tail'] != (node, mark):
                        #         mark = 'tail'
                        #     else:
                        #         mark = 'arrowhead'
                        # else:
                            mark = 'tail'
                    elif next_node in conds:
                        mark = 'arrowhead'
                elif next_mark == 'tail':
                    if 'tail' in succ[next_node] and succ[next_node]['tail'] != (node, mark):
                        mark = 'tail'
                    else:
                        mark = 'arrowhead' 
                node = next_node

            return path


        if conds is None:
            conds = []

        if forbidden_nodes is None:
            forbidden_nodes = []

        conds = [z for z in conds if z not in Y and z not in X]
        # print(X, Y, conds)

        if self.selection_vars is not None:
            for selection_var in self.selection_vars:
                conds += [(selection_var, -tau_sel) for tau_sel in range(0, max_lag + 1)]


        N = len(self.links)
        children = self._get_children()

        if only_non_causal_paths:
            anc_Y_dict = self._get_non_blocked_ancestors(Y=Y, conds=None, mode='max_lag',
                                    max_lag=max_lag)[0]
            # print(anc_Y_dict)
            anc_Y = []
            for y in Y:
                anc_Y += anc_Y_dict[y]
            des_X = self._get_descendants(X, children=children, max_lag=max_lag)
            mediators = set(anc_Y).intersection(set(des_X)) - set(Y) - set(X)

            causal_children = list(mediators) + Y
        else:
            causal_children = None

        if only_collider_paths_with_vancs:
            vancs_dict = self._get_non_blocked_ancestors(Y=XYS, conds=None, mode='max_lag',
                                    max_lag=max_lag)[0]
            vancs = set()
            for xys in XYS:
                vancs = vancs.union(set(vancs_dict[xys]))
            vancs = list(vancs) + XYS
            conds = vancs
        # else:
        #     vancs = None

        # Iterate through nodes in X and Y
        for x in X:
          for y in Y:

            # seen_links = []
            # predecessor and successors in search
            # (x, None) where None indicates start/end nodes, later (v,
            # 'tail') or (w, 'arrowhead') indicate how a link ends at a node
            pred = {x : {None: None}}
            succ = {y : {None: None}}

            # initialize fringes, start with forward from X
            forward_fringe = [(x, None)]
            reverse_fringe = [(y, None)]

            while forward_fringe and reverse_fringe:
                if len(forward_fringe) <= len(reverse_fringe):
                    if self.verbosity > 1:
                        print("Walk from X since len(X_fringe)=%d "
                              "<= len(Y_fringe)=%d" % (len(forward_fringe), 
                                len(reverse_fringe)))
                    this_level = forward_fringe
                    forward_fringe = []    
                    (found_connection, forward_fringe, pred, 
                     succ) = _walk_fringe(this_level, forward_fringe, pred, 
                                                succ)

                    # print(pred)
                    if found_connection: 
                        if return_path:
                            backtraced_path = backtrace_path()
                            return [(self.observed_vars.index(node[0]), node[1]) 
                                    for node in backtraced_path 
                                    if node[0] in self.observed_vars]
                        else: 
                            return True
                else:
                    if self.verbosity > 1:
                        print("Walk from Y since len(X_fringe)=%d "
                              "> len(Y_fringe)=%d" % (len(forward_fringe), 
                                len(reverse_fringe)))
                    this_level = reverse_fringe
                    reverse_fringe = []
                    (found_connection, reverse_fringe, succ, 
                     pred) = _walk_fringe(this_level, reverse_fringe, succ, 
                                                pred)

                    if found_connection: 
                        if return_path:
                            backtraced_path = backtrace_path()
                            return [(self.observed_vars.index(node[0]), node[1]) 
                                    for node in backtraced_path 
                                    if node[0] in self.observed_vars]
                        else: 
                            return True

                if self.verbosity > 1:
                    print("X_fringe = %s \n" % str(forward_fringe) +
                          "Y_fringe = %s" % str(reverse_fringe))           

        return False

    def _is_dsep(self, X, Y, Z, max_lag=None):
        """Returns whether X and Y are d-separated given Z in the graph.

        X, Y, Z are of the form (var, lag) for lag <= 0. D-separation is
        based on:

        1. Assessing the maximum time lag max_lag possible for any confounding
        path (see _get_maximum_possible_lag(...)).

        2. Using the time series graph truncated at max_lag we then test
        d-separation between X and Y conditional on Z using breadth-first
        search of non-blocked paths according to d-separation rules.

        Parameters
        ----------
        X, Y, Z : list of tuples
            List of variables chosen for current independence test.
        max_lag : int, optional (default: None)
            Used here to constrain the _is_dsep function to the graph
            truncated at max_lag instead of identifying the max_lag from
            ancestral search.

        Returns
        -------
        dseparated : bool, or path
            True if X and Y are d-separated given Z in the graph.
        """

        N = len(self.links)

        if self.verbosity > 0:
            print("Testing X=%s d-sep Y=%s given Z=%s in TSG" %(X, Y, Z))

        if Z is None:
            Z = []

        if max_lag is not None:
            # max_lags = dict([(j, max_lag) for j in range(N)])
            if self.verbosity > 0:
                print("Set max. time lag to: ", max_lag)
        else:
            max_lag = self._get_maximum_possible_lag(X+Y+Z)

        # Store overall max. lag
        self.max_lag = max_lag

        # _has_any_path is the main function that searches open paths
        any_path = self._has_any_path(X, Y, conds=Z, max_lag=max_lag)

        if any_path:
            dseparated = False
        else:
            dseparated = True

        return dseparated

    def check_shortest_path(self, X, Y, Z,
                 max_lag=None,  # compute_ancestors=False, 
                 starts_with=None, ends_with=None, 
                 forbidden_nodes=None,
                 directed=False,
                 only_non_causal_paths=False,
                 check_optimality_cond=False,
                 optimality_cond_des_YM=None,
                 optimality_cond_Y=None,
                 return_path=False):
        """Returns path between X and Y given Z in the graph.

        X, Y, Z are of the form (var, lag) for lag <= 0. D-separation is
        based on:

        1. Assessing maximum time lag max_lag of last ancestor of any X, Y, Z
        with non-blocked (by Z), non-repeating directed path towards X, Y, Z
        in the graph. 'non_repeating' means that an ancestor X^i_{ t-\tau_i}
        with link X^i_{t-\tau_i} --> X^j_{ t-\tau_j} is only included if
        X^i_{t'-\tau_i} --> X^j_{ t'-\tau_j} for t'!=t is not already part of
        the ancestors.

        2. Using the time series graph truncated at max_lag we then test
        d-separation between X and Y conditional on Z using breadth-first
        search of non-blocked paths according to d-separation rules including
        selection variables.

        Optionally only considers paths starting/ending with specific marks)
        and makes available the ancestors up to max_lag of X, Y, Z. This may take 
        a very long time, however.

        Parameters
        ----------
        X, Y, Z : list of tuples
            List of variables chosen for testing paths.
        max_lag : int, optional (default: None)
            Used here to constrain the has_path function to the graph
            truncated at max_lag instead of identifying the max_lag from
            ancestral search.
        compute_ancestors : bool
            Whether to also make available the ancestors for X, Y, Z as
            self.anc_all_x, self.anc_all_y, and self.anc_all_z, respectively.
        starts_with : {None, 'tail', 'arrohead'}
            Whether to only consider paths starting with particular mark at X.
        ends_with : {None, 'tail', 'arrohead'}
            Whether to only consider paths ending with particular mark at Y.

        Returns
        -------
        path : list or False
            Returns path or False if no path exists.
        """

        N = len(self.links)

        # Translate from observed_vars index to full variable set index
        X = [(self.observed_vars[x[0]], x[1]) for x in X]
        Y = [(self.observed_vars[y[0]], y[1]) for y in Y]
        Z = [(self.observed_vars[z[0]], z[1]) for z in Z]

        # print(X)
        # print(Y)
        # print(Z)

        if check_optimality_cond:
            optimality_cond_des_YM = [(self.observed_vars[x[0]], x[1]) 
                                        for x in optimality_cond_des_YM]
            optimality_cond_Y = [(self.observed_vars[x[0]], x[1]) 
                                    for x in optimality_cond_Y]

        # Get the array to test on
        X, Y, Z = self._check_XYZ(X, Y, Z)

        if self.verbosity > 0:
            print("Testing X=%s d-sep Y=%s given Z=%s in TSG" %(X, Y, Z))

        if max_lag is not None:
            # max_lags = dict([(j, max_lag) for j in range(N)])
            if self.verbosity > 0:
                print("Set max. time lag to: ", max_lag)
        else:
            max_lag = self._get_maximum_possible_lag(X+Y+Z)

        # Store overall max. lag
        self.max_lag = max_lag

        # _has_any_path is the main function that searches open paths
        any_path = self._has_any_path(X, Y, conds=Z, max_lag=max_lag, 
                                      starts_with=starts_with, ends_with=ends_with,
                                      return_path=return_path,
                                      directed=directed,
                                      only_non_causal_paths=only_non_causal_paths,
                                      check_optimality_cond=check_optimality_cond,
                                      optimality_cond_des_YM=optimality_cond_des_YM,
                                      optimality_cond_Y=optimality_cond_Y,
                                      forbidden_nodes=forbidden_nodes)

        if any_path:
            if return_path:
                any_path_observed = [(self.observed_vars.index(node[0]), node[1]) for node in any_path 
                             if node[0] in self.observed_vars]
            else:
                any_path_observed = True
        else: 
            any_path_observed = False

        if self.verbosity > 0:
            print("_has_any_path     = ", any_path)
            print("_has_any_path_obs = ", any_path_observed)


        # if compute_ancestors:
        #     if self.verbosity > 0:
        #         print("Compute ancestors.")

        #     # Get ancestors up to maximum ancestral time lag incl. repeated
        #     # links
        #     self.anc_all_x, _ = self._get_non_blocked_ancestors(X, conds=Z,
        #                                     mode='max_lag', max_lag=max_lag)
        #     self.anc_all_y, _ = self._get_non_blocked_ancestors(Y, conds=Z,
        #                                     mode='max_lag', max_lag=max_lag)
        #     self.anc_all_z, _ = self._get_non_blocked_ancestors(Z, conds=Z,
        #                                     mode='max_lag', max_lag=max_lag)

        return any_path_observed

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max', alpha_or_thres=None,
                 verbosity=0):
        """Perform oracle conditional independence test.

        Calls the d-separation function.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index in the observed_vars and tau the time lag.
        tau_max : int, optional (default: 0)
            Not used here.
        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            Not used here.
        alpha_or_thres : float
            Not used here.

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """

        if Z is None:
            Z = []

        # Translate from observed_vars index to full variable set index
        X = [(self.observed_vars[x[0]], x[1]) for x in X]
        Y = [(self.observed_vars[y[0]], y[1]) for y in Y]
        Z = [(self.observed_vars[z[0]], z[1]) for z in Z]

        # Get the array to test on
        X, Y, Z = self._check_XYZ(X, Y, Z)

        if not str((X, Y, Z)) in self.dsepsets:
            self.dsepsets[str((X, Y, Z))] = self._is_dsep(X, Y, Z)

        if self.dsepsets[str((X, Y, Z))]:
            val = 0.
            pval = 1.
            dependent = False
        else:
            val = 1.
            pval = 0.
            dependent = True

        # Saved here, but not currently used
        self.ci_results[(tuple(X), tuple(Y),tuple(Z))] = (val, pval, dependent)
        
        if verbosity > 1:
            self._print_cond_ind_results(val=val, pval=pval, cached=False,
                                         conf=None)
        # Return the value and the pvalue
        if alpha_or_thres is None:
            return val, pval
        else:
            return val, pval, dependent

    def get_measure(self, X, Y, Z=None, tau_max=0):
        """Returns dependence measure.

        Returns 0 if X and Y are d-separated given Z in the graph and 1 else.

        Parameters
        ----------
        X, Y [, Z] : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index in the observed_vars and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns
        -------
        val : float
            The test statistic value.

        """

        # Translate from observed_vars index to full variable set index
        X = [(self.observed_vars[x[0]], x[1]) for x in X]
        Y = [(self.observed_vars[y[0]], y[1]) for y in Y]
        Z = [(self.observed_vars[z[0]], z[1]) for z in Z]

        # Check XYZ
        X, Y, Z = _check_XYZ(X, Y, Z)

        if not str((X, Y, Z)) in self.dsepsets:
            self.dsepsets[str((X, Y, Z))] = self._is_dsep(X, Y, Z)

        if self.dsepsets[str((X, Y, Z))]:
            return 0.
        else:
            return 1.

    def _print_cond_ind_results(self, val, pval=None, cached=None, conf=None):
        """Print results from conditional independence test.

        Parameters
        ----------
        val : float
            Test stastistic value.
        pval : float, optional (default: None)
            p-value
        conf : tuple of floats, optional (default: None)
            Confidence bounds.
        """
        printstr = "        val = %.3f" % (val)      
        if pval is not None:
            printstr += " | pval = %.5f" % (pval)
        if conf is not None:
            printstr += " | conf bounds = (%.3f, %.3f)" % (
                conf[0], conf[1])
        if cached is not None:
            printstr += " %s" % ({0:"", 1:"[cached]"}[cached])

        print(printstr)

    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """
        Base class assumption that this is not implemented.  Concrete classes
        should override when possible.
        """
        raise NotImplementedError("Model selection not"+\
                                  " implemented for %s" % self.measure)

    def _reverse_patt(self, patt):
        """Inverts a link pattern"""

        if patt == "":
            return ""

        left_mark, middle_mark, right_mark = patt[0], patt[1], patt[2]
        if left_mark == "<":
            new_right_mark = ">"
        else:
            new_right_mark = left_mark
        if right_mark == ">":
            new_left_mark = "<"
        else:
            new_left_mark = right_mark

        return new_left_mark + middle_mark + new_right_mark


    def get_links_from_graph(self, graph):
        """
        Constructs links_coeffs dictionary, observed_vars, 
        and selection_vars from graph array (MAG or DAG).

        In the case of MAGs, for every <-> or --- link further
        latent and selection variables, respectively, are added.
        This corresponds to a canonical DAG (Richardson Spirtes 2002).

        For ADMGs "---" are not supported, but also links of type "+->"
        exist, which corresponds to having both "-->" and "<->".

        Can be used to evaluate d-separation in MAG/DAGs.

        """

        if "U3" not in str(graph.dtype):
            raise ValueError("graph must be of type '<U3'!")

        if self.graph_is_mag:
            edge_types = ["-->", "<--", "<->", "---"]
        else:
            edge_types = ["-->", "<--", "<->", "+->", "<-+"] #, "--+", "+--"]


        N, N, tau_maxplusone = graph.shape
        tau_max = tau_maxplusone - 1

        observed_vars = list(range(N))

        selection_vars = []

        links = {j: [] for j in observed_vars }

        # Add further latent variables to accommodate <-> and --- links
        latent_index = N
        for i, j, tau in zip(*np.where(graph)):

            edge_type = graph[i, j, tau]

            if edge_type not in edge_types:
                raise ValueError(
                    "Links can only be in %s " %str(edge_types)
                )

            if tau == 0:
                if edge_type != self._reverse_patt(graph[j, i, 0]):
                    raise ValueError(
                        "graph needs to have consistent lag-zero patterns (eg"
                        " graph[i,j,0]='-->' requires graph[j,i,0]='<--')"
                    )

                # Consider contemporaneous links only once
                if j > i:
                    continue

            # Restrict lagged links
            else: 
                if edge_type not in ["-->", "<->", "---", "+->"]: #, "--+"]:
                    raise ValueError(
                        "Lagged links can only be in ['-->', '<->', '---', '+->']"
                    )

            if edge_type == "-->":
                links[j].append((i, -tau))
            elif edge_type == "<--":
                links[i].append((j, -tau))
            elif edge_type == "<->":
                links[latent_index] = []
                links[i].append((latent_index, 0))
                links[j].append((latent_index, -tau))
                latent_index += 1
            elif edge_type == "---":
                links[latent_index] = []
                selection_vars.append(latent_index)
                links[latent_index].append((i, -tau))
                links[latent_index].append((j, 0))
                latent_index += 1
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
            # elif edge_type == "+--":
            #     links[i].append((j, -tau))
            #     links[latent_index] = []
            #     selection_vars.append(latent_index)
            #     links[latent_index].append((i, -tau))
            #     links[latent_index].append((j, 0))
            #     latent_index += 1
            # elif edge_type == "--+":
            #     links[j].append((i, -tau))
            #     links[latent_index] = []
            #     selection_vars.append(latent_index)
            #     links[latent_index].append((i, -tau))
            #     links[latent_index].append((j, 0))
            #     latent_index += 1

        return links, observed_vars, selection_vars

    def _get_minmax_lag(self, links):
        """Helper function to retrieve tau_min and tau_max from links
        """

        N = len(links)

        # Get maximum time lag
        min_lag = np.inf
        max_lag = 0
        for j in range(N):
            for link_props in links[j]:
                if len(link_props) == 3:
                    i, lag = link_props[0]
                    coeff = link_props[1]
                else:
                    i, lag = link_props
                    coeff = 1.                
                # func = link_props[2]
                if coeff != 0.:
                    min_lag = min(min_lag, abs(lag))
                    max_lag = max(max_lag, abs(lag))
        return min_lag, max_lag

    def get_graph_from_links(self, tau_max=None):
        """
        Constructs graph (DAG or MAG or ADMG) from links, observed_vars, 
        and selection_vars.

        For ADMGs uses the Latent projection operation (Pearl 2009).

        """

        # TODO: use MAG from DAG construction procedure (lecture notes)
        # issues with tau_max?
        if self.graph_is_mag is False and len(self.selection_vars) > 0:
            raise ValueError("ADMGs do not support selection_vars.")

        N_all = len(self.links)

        # If tau_max is None, compute from links_coeffs
        _, max_lag_links = self._get_minmax_lag(self.links)
        if tau_max is None:
            tau_max = max_lag_links
        else:
            if max_lag_links > tau_max:
                raise ValueError("tau_max must be >= maximum lag in links_coeffs; choose tau_max=None")

        # print("max_lag_links ", max_lag_links)

        N = len(self.observed_vars)

        # Init graph
        graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
        graph[:] = ""
        # We will enumerate the observed variables with (i,j) which refers to the index in MAG graph
        # while x, y iterates through the variables in the underlying DAG

        # Loop over the observed variables
        for j, y in enumerate(self.observed_vars):
          for i, x in enumerate(self.observed_vars):
            for tau in range(0, tau_max + 1):
              if (x, -tau) != (y, 0):

                if self.graph_is_mag:
                    dag_anc_y, _ = self._get_non_blocked_ancestors(Y=[(y, 0)], conds=None, 
                                            mode='max_lag',
                                            max_lag=tau_max)
                    # Only consider observed ancestors
                    mag_anc_y = [anc for anc in dag_anc_y[(y, 0)] 
                                if anc[0] in self.observed_vars]

                    dag_anc_x, _ = self._get_non_blocked_ancestors(Y=[(x, -tau)], 
                                    conds=None, mode='max_lag',
                                    max_lag=tau_max)
                    
                    # Only consider observed ancestors
                    mag_anc_x = [anc for anc in dag_anc_x[(x, -tau)] 
                                if anc[0] in self.observed_vars]   

                    # Add selection variable ancestors
                    dag_anc_s = set()
                    for s in self.selection_vars:
                        dag_anc_s_here, _ = self._get_non_blocked_ancestors(Y=[(s, 0)], 
                                    conds=None, mode='max_lag',
                                    max_lag=tau_max)
                        dag_anc_s = dag_anc_s.union(set(dag_anc_s_here[(s, 0)]))
                    
                    dag_anc_s = list(dag_anc_s)
                    # Only consider observed ancestors
                    mag_anc_s = [anc for anc in dag_anc_s 
                                if anc[0] in self.observed_vars]

                    Z = set([z for z in mag_anc_y + mag_anc_x + mag_anc_s if z != (y, 0) and z != (x, -tau)])
                    Z = list(Z)
                    
                    separated = self._is_dsep(X=[(x, -tau)], Y=[(y, 0)], Z=Z, max_lag=None)
                    
                    # If X and Y are connected given Z, mark a link
                    if not separated:
                        # (i, -tau) --> j
                        if (x, -tau) in dag_anc_y[(y, 0)] + dag_anc_s and (y, 0) not in dag_anc_x[(x, -tau)] + dag_anc_s:
                            graph[i, j, tau] = "-->"
                            if tau == 0:
                                graph[j, i, 0] = "<--"

                        elif (x, -tau) not in dag_anc_y[(y, 0)] + dag_anc_s and (y, 0) not in dag_anc_x[(x, -tau)] + dag_anc_s:
                            graph[i, j, tau] = "<->"
                            if tau == 0:
                                graph[j, i, 0] = "<->"

                        elif (x, -tau) in dag_anc_y[(y, 0)] + dag_anc_s and (y, 0) in dag_anc_x[(x, -tau)] + dag_anc_s:
                            graph[i, j, tau] = "---"
                            if tau == 0:
                                graph[j, i, 0] = "---"
                else:
                    if tau == 0 and j >= i:
                        continue
                    # edge_types = ["-->", "<->", "+->"]
                    # Latent projection operation:
                    # (i)  ADMG contains i --> j iff there is a directed path x --> ... --> y on which
                    #      every non-endpoint vertex is in hidden variables (= not in observed_vars)
                    # (ii) ADMG contains i <-> j iff there exists a path of the form x <-- ... --> y on
                    #      which every non-endpoint vertex is non-collider AND in L (=not in observed_vars)
                    observed_varslags = set([(v, -lag) for v in self.observed_vars
                                                   for lag in range(0, tau_max + 1)]) - set([(x, -tau), (y, 0)])
                    cond_one_xy = self._has_any_path(X=[(x, -tau)], Y=[(y, 0)], 
                        conds=[], 
                        max_lag=None, 
                        starts_with='tail',
                        ends_with='arrowhead',
                        directed=True,  
                        forbidden_nodes=list(observed_varslags), 
                        return_path=False)
                    if tau == 0:
                        cond_one_yx = self._has_any_path(X=[(y, 0)], Y=[(x, 0)], 
                            conds=[], 
                            max_lag=None, 
                            starts_with='tail',
                            ends_with='arrowhead',
                            directed=True,  
                            forbidden_nodes=list(observed_varslags), 
                            return_path=False)
                    else:
                        cond_one_yx = False
                    cond_two = self._has_any_path(X=[(x, -tau)], Y=[(y, 0)], 
                        conds=[], 
                        max_lag=None, 
                        starts_with='arrowhead',
                        ends_with='arrowhead',
                        directed=False,  
                        forbidden_nodes=list(observed_varslags), 
                        return_path=False)
                    if cond_one_xy and cond_one_yx:
                        raise ValueError("Cyclic graph!")
                    # print((x, -tau), y, cond_one_xy, cond_one_yx, cond_two)

                    # Only (i) holds: i --> j
                    if cond_one_xy and not cond_two:
                        graph[i, j, tau] = "-->"
                        if tau == 0:
                            graph[j, i, 0] = "<--"
                    elif cond_one_yx and not cond_two:
                        graph[i, j, tau] = "<--"
                        if tau == 0:
                            graph[j, i, 0] = "-->"

                    # Only (ii) holds: i <-> j
                    elif not cond_one_xy and not cond_one_yx and cond_two:
                        graph[i, j, tau] = "<->"
                        if tau == 0:
                            graph[j, i, 0] = "<->"

                    # Both (i) and (ii) hold: i +-> j
                    elif cond_one_xy and cond_two:
                        graph[i, j, tau] = "+->"
                        if tau == 0:
                            graph[j, i, 0] = "<-+"
                    elif cond_one_yx and cond_two:
                        graph[i, j, tau] = "<-+"
                        if tau == 0:
                            graph[j, i, 0] = "+->"
                    # print((i, -tau), j, cond_one_xy, cond_one_yx, cond_two)

        return graph

    def get_confidence(self, X, Y, Z=None, tau_max=0):
        """For compatibility with PCMCI.

        Returns
        -------
        None
        """
        return None

if __name__ == '__main__':

    import tigramite.plotting as tp
    from matplotlib import pyplot as plt
    def lin_f(x): return x

    # Define the stationary DAG
    links = {0 : [(0, -3), (1, 0)], 1: [(2, -2)], 2: [(1, -2)]}
    observed_vars = [0, 1, 2]

    oracle = OracleCI(links=links, 
        observed_vars=observed_vars, 
        graph_is_mag=True,
        #         selection_vars=selection_vars,
        #     verbosity=2
        )
    graph = oracle.graph
    print(graph[:,:,0])

    tp.plot_time_series_graph(graph=graph, var_names=None, figsize=(5, 5),
                save_name="tsg.pdf")

    X = [(0, 0)]
    Y = [(2, 0)]
    Z = []
    # node = (3, 0)
    # prelim_Oset = set([(3, 0)])
    # S = set([])
    # collider_path_nodes = set([])
    path = oracle._has_any_path(X=X, Y=Y, 
                            conds=Z, 
                            max_lag=8, 
                            starts_with='arrowhead',
                            ends_with='arrowhead',  
                            forbidden_nodes=None, 
                            return_path=True)
    print(path)

    print("-------------------------------")
    print(oracle._get_maximum_possible_lag(X+Z)) #(X = X, Y = Y, Z = Z))

    # cond_ind_test = OracleCI(graph=graph)
    # links, observed_vars, selection_vars = cond_ind_test.get_links_from_graph(graph)
    # print("{")
    # for j in links.keys():
    #     parents = repr([(p, 'coeff', 'lin_f') for p in links[j]])
    #     print(f"{j: 1d}" ":"  f"{parents:s},")
    # print(repr(observed_vars))
    # cond_ind_test = OracleCI(graph=graph, verbosity=2)

    # X = [(0, 0)]
    # Y = [(2, 0)]
    # Z = [(7, 0), (3, 0), (6, 0), (5, 0), (4, 0)] #(1, -3), (1, -2), (0, -2), (0, -1), (0, -3)]
    # #(j, -2) for j in range(N)] + [(j, 0) for j in range(N)]

    # # print(oracle._get_non_blocked_ancestors(Z, Z=None, mode='max_lag',
    # #                                     max_lag=2))
    # # cond_ind_test = OracleCI(links, observed_vars=observed_vars, verbosity=2)

    # print(cond_ind_test.get_shortest_path(X=X, Y=Y, Z=Z,
    #              max_lag=None, compute_ancestors=False, 
    #              backdoor=True))
   
    # anc_x=None  #oracle.anc_all_x[X[0]]
    # anc_y=None #oracle.anc_all_y[Y[0]]
    # anc_xy=None # []
    # # # for z in Z:
    # # #     anc_xy += oracle.anc_all_z[z]
    
    # fig, ax = tp.plot_tsg(links, 
    #             X=[(observed_vars[x[0]], x[1]) for x in X], 
    #             Y=[(observed_vars[y[0]], y[1]) for y in Y], 
    #             Z=[(observed_vars[z[0]], z[1]) for z in Z],
    #     anc_x=anc_x, anc_y=anc_y, 
    #     anc_xy=anc_xy)

    # fig.savefig("/home/rung_ja/Downloads/tsg.pdf")
