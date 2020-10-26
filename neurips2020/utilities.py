import numpy as np
from itertools import product
from collections import OrderedDict
import tigramite.data_processing as pp
from svarfci import SVARFCI

class OracleCI:
    r"""Oracle of conditional independence test X _|_ Y | Z given a graph.

    Class around link_coeff causal ground truth. X _|_ Y | Z is based on
    assessing whether X and Y are d-separated given Z in the graph.

    Class can be used like a Tigramite conditional independence class
    (e.g., ParCorr).

    Parameters
    ----------
    link_coeffs : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
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
                 link_coeffs,
                 observed_vars=None,
                 verbosity=0):
        self.verbosity = verbosity
        self._measure = 'oracle_ci'
        self.confidence = None
        self.link_coeffs = link_coeffs
        self.N = len(link_coeffs)

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

    def _get_lagged_parents(self, var_lag, exclude_contemp=False):
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

        for link_props in self.link_coeffs[var]:
            i, tau = link_props[0]
            coeff = link_props[1]
            if coeff != 0.:
                if not (exclude_contemp and lag == 0):
                    yield (i, lag + tau)

    def _get_children(self):
        """Helper function to get children from links.

        Note that for children the lag is positive.

        Returns
        -------
        children : dict
            Dictionary of form {0:[(0, 1), (3, 0), ...], 1:[], ...}.
        """

        N = len(self.link_coeffs)
        children = dict([(j, []) for j in range(N)])

        for j in range(N):
            for link_props in self.link_coeffs[j]:
                        i, tau = link_props[0]
                        coeff = link_props[1]
                        if coeff != 0.:
                            children[i].append((j, abs(tau)))

        return children

    def _get_lagged_children(self, var_lag, children, exclude_contemp=False):
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

        N = len(self.link_coeffs)

        # Initialize max. ancestral time lag for every N
        if mode == 'non_repeating':
            max_lag = 0
        else:
            if max_lag is None:
                raise ValueError("max_lag must be set in mode = 'max_lag'")

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

    def _has_any_path(self, X, Y, conds, max_lag):
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

        """

        def _walk_to_parents(v, fringe, this_path, other_path):
            """Helper function to update paths when walking to parents."""
            found_path = False
            for w in self._get_lagged_parents(v):
                # Cannot walk into conditioned parents and
                # cannot walk beyond t or max_lag
                i, t = w
                if (w not in conds and
                    # (w, v) not in seen_links and
                    t <= 0 and abs(t) <= max_lag):
                    if ((w, 'tail') not in this_path and 
                        (w, None) not in this_path):
                        if self.verbosity > 1:
                            print("Walk parent: %s --> %s  " %(v, w))
                        fringe.append((w, 'tail'))
                        this_path[(w, 'tail')] = (v, 'arrowhead')
                        # seen_links.append((v, w))
                    # Determine whether X and Y are connected
                    # (w, None) indicates the start or end node X/Y
                    if ((w, 'tail') in other_path 
                       or (w, 'arrowhead') in other_path
                       or (w, None) in other_path):
                        if self.verbosity > 1:
                            print("Found connection: ", w)
                        found_path = True   
                        break
            return found_path, fringe, this_path

        def _walk_to_children(v, fringe, this_path, other_path):
            """Helper function to update paths when walking to children."""
            found_path = False
            for w in self._get_lagged_children(v, children):
                # You can also walk into conditioned children,
                # but cannot walk beyond t or max_lag
                i, t = w
                if (
                    # (w, v) not in seen_links and
                    t <= 0 and abs(t) <= max_lag):
                    if ((w, 'arrowhead') not in this_path and 
                        (w, None) not in this_path):
                        if self.verbosity > 1:
                            print("Walk child:  %s --> %s  " %(v, w))
                        fringe.append((w, 'arrowhead'))
                        this_path[(w, 'arrowhead')] = (v, 'tail')
                        # seen_links.append((v, w))
                    # Determine whether X and Y are connected
                    # If the other_path contains w with a tail, then w must
                    # NOT be conditioned on. Alternatively, if the other_path
                    # contains w with an arrowhead, then w must be
                    # conditioned on.
                    if (((w, 'tail') in other_path and w not in conds)
                       or ((w, 'arrowhead') in other_path and w in conds)
                       or (w, None) in other_path):
                        if self.verbosity > 1:
                            print("Found connection: ", w)
                        found_path = True   
                        break
            return found_path, fringe, this_path

        def _walk_fringe(this_level, fringe, this_path, other_path):
            """Helper function to walk each fringe, i.e., the path from X and Y,
            respectively."""
            found_path = False
            for v, mark in this_level:
                if v in conds:
                    if (mark == 'arrowhead' or mark == None):
                        # Motif: --> [v] <--
                        # If standing on a condition and coming from an
                        # arrowhead, you can only walk into parents
                        (found_path, fringe,
                         this_path) = _walk_to_parents(v, fringe, 
                                                       this_path, other_path)
                        if found_path: break            
                else:
                    if (mark == 'tail' or mark == None):
                        # Motif: <-- v <-- or <-- v -->
                        # If NOT standing on a condition and coming from
                        # a tail mark, you can walk into parents or 
                        # children
                        (found_path, fringe,
                         this_path) = _walk_to_parents(v, fringe, 
                                                       this_path, other_path)
                        if found_path: break 
                        
                        (found_path, fringe,
                         this_path) = _walk_to_children(v, fringe, 
                                                       this_path, other_path)
                        if found_path: break 
                      
                    elif mark == 'arrowhead':
                        # Motif: --> v -->
                        # If NOT standing on a condition and coming from
                        # an arrowhead mark, you can only walk into
                        # children
                        (found_path, fringe,
                         this_path) = _walk_to_children(v, fringe, 
                                                       this_path, other_path)
                        if found_path: break 
            if self.verbosity > 1:
                print("Updated fringe: ", fringe)
            return found_path, fringe, this_path, other_path

        if conds is None:
            conds = []
        conds = [z for z in conds if z not in Y and z not in X]

        N = len(self.link_coeffs)
        children = self._get_children()

        # Iterate through nodes in X and Y
        for x in X:
          for y in Y:

            seen_links = []
            # predecessor and successors in search
            # (x, None) where None indicates start/end nodes, later (v,
            # 'tail') or (w, 'arrowhead') indicate how a link ends at a node
            pred = {(x, None): None}
            succ = {(y, None): None}

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
                    (found_path, forward_fringe, pred, 
                     succ) = _walk_fringe(this_level, forward_fringe, pred, 
                                                succ)
                    # print(pred)
                    if found_path: return True
                else:
                    if self.verbosity > 1:
                        print("Walk from Y since len(X_fringe)=%d "
                              "> len(Y_fringe)=%d" % (len(forward_fringe), 
                                len(reverse_fringe)))
                    this_level = reverse_fringe
                    reverse_fringe = []
                    (found_path, reverse_fringe, succ, 
                     pred) = _walk_fringe(this_level, reverse_fringe, succ, 
                                                pred)
                    if found_path: return True

                if self.verbosity > 1:
                    print("X_fringe = %s \n" % str(forward_fringe) +
                          "Y_fringe = %s" % str(reverse_fringe))           

        return False

    def _is_dsep(self, X, Y, Z, max_lag=None, compute_ancestors=False):
        """Returns whether X and Y are d-separated given Z in the graph.

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
        search of non-blocked paths according to d-separation rules.

        Optionally makes available the ancestors up to max_lag of X, Y,
        Z. This may take a very long time, however.

        Parameters
        ----------
        X, Y, Z : list of tuples
            List of variables chosen for current independence test.
        max_lag : int, optional (default: None)
            Used here to constrain the _is_dsep function to the graph
            truncated at max_lag instead of identifying the max_lag from
            ancestral search.
        compute_ancestors : bool
            Whether to also make available the ancestors for X, Y, Z as
            self.anc_all_x, self.anc_all_y, and self.anc_all_z, respectively.

        Returns
        -------
        dseparated : bool
            True if X and Y are d-separated given Z in the graph.
        """

        N = len(self.link_coeffs)

        if self.verbosity > 0:
            print("Testing X=%s d-sep Y=%s given Z=%s in TSG" %(X, Y, Z))

        if max_lag is not None:
            # max_lags = dict([(j, max_lag) for j in range(N)])
            if self.verbosity > 0:
                print("Set max. time lag to: ", max_lag)

        else:
            # Get maximum non-repeated ancestral time lag
            _, max_lag_X = self._get_non_blocked_ancestors(X, conds=Z, 
                                                           mode='non_repeating')
            _, max_lag_Y = self._get_non_blocked_ancestors(Y, conds=Z, 
                                                           mode='non_repeating')
            _, max_lag_Z = self._get_non_blocked_ancestors(Z, conds=Z, 
                                                           mode='non_repeating')

            # Get max time lag among the ancestors
            max_lag = max(max_lag_X, max_lag_Y, max_lag_Z)

            if self.verbosity > 0:
                print("Max. non-repeated ancestral time lag: ", max_lag)

        # Store overall max. lag 
        self.max_lag = max_lag


        # _has_any_path is the main function that searches open paths
        any_path = self._has_any_path(X, Y, conds=Z, max_lag=max_lag)
        if self.verbosity > 0:
            print("_has_any_path = ", any_path)

        if any_path:
            dseparated = False
        else:
            dseparated = True

        if compute_ancestors:
            if self.verbosity > 0:
                print("Compute ancestors.")

            # Get ancestors up to maximum ancestral time lag incl. repeated
            # links
            self.anc_all_x, _ = self._get_non_blocked_ancestors(X, conds=Z,
                                            mode='max_lag', max_lag=max_lag)
            self.anc_all_y, _ = self._get_non_blocked_ancestors(Y, conds=Z,
                                            mode='max_lag', max_lag=max_lag)
            self.anc_all_z, _ = self._get_non_blocked_ancestors(Z, conds=Z,
                                            mode='max_lag', max_lag=max_lag)

        return dseparated

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max',
                 compute_ancestors=False,
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

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """

        # Translate from observed_vars index to full variable set index
        X = [(self.observed_vars[x[0]], x[1]) for x in X]
        Y = [(self.observed_vars[y[0]], y[1]) for y in Y]
        Z = [(self.observed_vars[z[0]], z[1]) for z in Z]

        # Get the array to test on
        X, Y, Z = self._check_XYZ(X, Y, Z)

        if not str((X, Y, Z)) in self.dsepsets:
            self.dsepsets[str((X, Y, Z))] = self._is_dsep(X, Y, Z, 
                max_lag=None,
                compute_ancestors=compute_ancestors)

        if self.dsepsets[str((X, Y, Z))]:
            val = 0.
            pval = 1.
        else:
            val = 1.
            pval = 0.

        if verbosity > 1:
            self._print_cond_ind_results(val=val, pval=pval, cached=False,
                                         conf=None)
        # Return the value and the pvalue
        return val, pval

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
            self.dsepsets[str((X, Y, Z))] = self._is_dsep(X, Y, Z, 
                max_lag=None)

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

def _get_minmax_lag(links):
    """Helper function to retrieve tau_min and tau_max from links
    """

    N = len(links)

    # Get maximum time lag
    min_lag = np.inf
    max_lag = 0
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            # func = link_props[2]
            if coeff != 0.:
                min_lag = min(min_lag, abs(lag))
                max_lag = max(max_lag, abs(lag))
    return min_lag, max_lag

def _get_pag_from_dag(links_coeffs, observed_vars=None, tau_max=None, verbosity=0):
    """Computes PAG over observed variables from DAG on full variable set.
    
       Uses OracleCI tests based on ancestors in DAG to obtain skeleton and sepsets.
       Then applies FCI rules (including collider rule).
    """

    if verbosity > 0:
        print("Running _get_pag_from_dag:\n\n1. Ancestors search")

    N_all = len(links_coeffs)

    # If tau_max is None, compute from links_coeffs
    _, max_lag_links = _get_minmax_lag(links_coeffs)
    if tau_max is None:
        tau_max = max_lag_links
    else:
        if max_lag_links > tau_max:
            raise ValueError("tau_max must be >= maximum lag in links_coeffs; choose tau_max=None")

    if observed_vars is None:
        observed_vars = range(N_all)
    else:
        if not set(observed_vars).issubset(set(range(N_all))):
            raise ValueError("observed_vars must be subset of range(N_all).")

    N = len(observed_vars)
    # Init cond_ind_test class
    cond_ind_test = OracleCI(links_coeffs)

    # Init graph and sepsets
    graph_dict = {j: {(i, -tau): "o-o" for i in range(N) for tau in range(tau_max + 1) if tau > 0 or j != i} for j in range(N)}
    sepsets = {j: {(i, -tau): {} for i in range(N) for tau in range(tau_max + 1) if (tau > 0 or i < j)} for j in range(N)}

    sepset_answers = {}

    # We will enumerate the observed variables with (i,j) which refers to the index in pag_graph
    # while x, y iterates through the oberved variables in the underlying DAG

    # Loop over the observed variables
    for j, y in enumerate(observed_vars):

        for i, x in enumerate(observed_vars):
            for tau in range(0, tau_max + 1):
                if (x, -tau) != (y, 0):   

                    dag_anc_y, _ = cond_ind_test._get_non_blocked_ancestors(Y=[(y, 0)], conds=None, 
                                            mode='max_lag',
                                            max_lag=tau_max)
                    # Only consider observed ancestors
                    pag_anc_y = [anc for anc in dag_anc_y[(y, 0)] 
                                if anc[0] in observed_vars]

                    dag_anc_x, _ = cond_ind_test._get_non_blocked_ancestors(Y=[(x, -tau)], 
                                    conds=None, mode='max_lag',
                                    max_lag=tau_max)
                    
                    # Only consider observed ancestors
                    pag_anc_x = [anc for anc in dag_anc_x[(x, -tau)] 
                                if anc[0] in observed_vars]   

                    Z = list(set([z for z in pag_anc_y + pag_anc_x if z != (y, 0) and z != (x, -tau)]))
                    
                    separated = cond_ind_test._is_dsep(X=[(x, -tau)], Y=[(y, 0)], Z=Z, max_lag=None)
                    
                    # If X and Y are connected given Z, mark a link
                    if not separated and tau==0:
                        graph_dict[j][(i, -tau)] = "o-o"
                    elif not separated and tau > 0:
                        graph_dict[j][(i, -tau)] = "o->"
                    # If X and Y are separated given Z, mark absence of links and store sepset
                    else:
                        graph_dict[j][(i, -tau)] = ""

                        # Translate sepset to (i,j)-space
                        S = frozenset((observed_vars.index(cond[0]), cond[1]) for cond in Z)
                        #sepsets[j][(i, -tau)] = {(S, "wm")}

                        sepsets[j][(i, -tau)] = {(S, "")}
                        if tau == 0:
                            #sepsets[i][(j, 0)] = {(S, "wm")}
                            sepsets[i][(j, 0)] = {(S, "")}

                        if tau > 0 or (tau == 0 and i < j):
                            X_type = (i, -tau)
                            Y_type = (j, 0)
                        else:
                            X_type = (j, 0)
                            Y_type = (i, 0)

                        for s in S:
                            sepset_answers[(X_type, s, Y_type)] = False
           
                        for k, tau in product(range(N), range(0, tau_max + 1)):
                            if sepset_answers.get((X_type, (k, -tau), Y_type)) is None:
                                sepset_answers[(X_type, (k, - tau), Y_type)] = True


    if verbosity > 0:
        print("2. FCI orientation rules")

    # Initialize SVARFCI with dummy data
    svarfci = SVARFCI(dataframe=pp.DataFrame(np.zeros((N+1, N))),cond_ind_test=cond_ind_test)
    svarfci._initialize(tau_max = tau_max, pc_alpha = 0.01, max_cond_px = np.inf, max_p_global = np.inf, max_p_dsep = np.inf, max_q_global = np.inf, max_pds_set = np.inf, fix_all_edges_before_final_orientation = False, verbosity = verbosity)
    svarfci._oracle = True

    # Update graph_dict and sepsets
    svarfci.graph_dict = graph_dict
    svarfci.sepsets = sepsets
    
    # Run *all* rules
    svarfci._B_not_in_SepSet_AC_given_answers = sepset_answers
    svarfci._run_fci_orientation_phase()   

    # Also return array version of pag graph
    pag_graph = svarfci._dict2graph()

    return svarfci.graph_dict, pag_graph
    