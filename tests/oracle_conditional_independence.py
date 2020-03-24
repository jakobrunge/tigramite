"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import numpy as np
import sys

from collections import defaultdict, OrderedDict
import itertools
from copy import deepcopy

import networkx as nx
import tigramite.data_processing as pp


class OracleCI:
    r"""Oracle of conditional independence test X _|_ Y | Z given a graph.

    Class around link_coeff causal ground truth. X _|_ Y | Z is based on
    assessing whether X and Y are d-separated given Z in the graph. To this 
    end a modified "conditioned graph" is created for the set Z and then
    ancestral relations are used to decide whether X and Y are connected
    by a directed path or by a common driver path.

    Class can be used just like a Tigramite conditional independence class
    (e.g., ParCorr). The main use is for unit testing of PCMCI methods.

    Parameters
    ----------
    links : dict
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
                 verbosity=0):
        self.verbosity = verbosity
        self._measure = 'oracle_ci'
        self.confidence = None
        self.link_coeffs = link_coeffs
        self.N = len(link_coeffs)

        # Create time series graph from link dictionary
        self.tsg_orig = self._links_to_tsg(self.link_coeffs, 
                                           max_lag=None)

        # Initialize already computed dsepsets of X, Y, Z
        self.dsepsets = {}

    def set_dataframe(self, dataframe):
        """Dummy function."""
        pass

    def varlag2node(self, var, lag):
        """Translate from (var, lag) notation to node in TSG.

        lag must be <= 0.
        """
        return var * self.max_lag + lag
    
    def node2varlag(self, node):
        """Translate from node in TSG to (var, -tau) notation.

        Here tau is <= 0.
        """
        var = node // self.max_lag
        tau = node % (self.max_lag) - (self.max_lag - 1)
        return var, tau

    def _links_to_tsg(self, link_coeffs, max_lag=None):
        """Transform link_coeffs to time series graph.

        TSG is of shape (N*max_lag, N*max_lag).
        """
        # Get maximum lag
        min_lag_links, max_lag_links = pp._get_minmax_lag(link_coeffs)

        # max_lag of TSG is max lag in links + 1 for the zero lag.
        if max_lag is None:
            self.max_lag = max_lag_links + 1
        else:
            self.max_lag = max_lag
        
        tsg = np.zeros((self.N * self.max_lag, self.N * self.max_lag))

        for j in range(self.N):
            for link_props in link_coeffs[j]:
                i, lag = link_props[0]
                tau = abs(lag)
                coeff = link_props[1]
                # func = link_props[2]
                if coeff != 0.:
                    for t in range(self.max_lag):
                        if (0 <= self.varlag2node(i, t - tau) and
                            self.varlag2node(i, t - tau) % self.max_lag 
                            <= self.varlag2node(j, t) % self.max_lag):
                            tsg[self.varlag2node(i, t - tau), 
                            self.varlag2node(j, t)] = 1.
        
        return tsg        

    def _check_nodes(self, Y, XYZ, N, dim):
        """
        Checks that:
            * The requests XYZ nodes have the correct shape
            * All lags are non-positive
            * All indices are less than N
            * One of the Y nodes has zero lag

        Parameters
        ----------
            Y : list of tuples
                Of the form [(var, -tau)], where var specifies the variable
                index and tau the time lag.
            XYZ : list of tuples
                List of nodes chosen for current independence test
            N : int
                Total number of listed nodes
            dim : int
                Number of nodes excluding repeated nodes
        """
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

    def _check_XYZ(self, X, Y, Z, tau_max,
                        verbosity=0):
        """Checks variables X, Y, Z.

        Parameters
        ----------
        X, Y, Z : list of tuples
            For a dependence measure I(X;Y|Z), Y is of the form [(varY, 0)],
            where var specifies the variable index. X typically is of the form
            [(varX, -tau)] with tau denoting the time lag and Z can be
            multivariate [(var1, -lag), (var2, -lag), ...] .
        tau_max : int
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X and Z all have the same sample size.
        verbosity : int, optional (default: 0)
            Level of verbosity.

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
        self._check_nodes(Y, XYZ, N, dim)

        return (X, Y, Z)

    def is_dsep(self, X, Y, Z):
        """Returns whether X and Y are d-separated given Z in the graph.

        Based on constructing modified time series graph for a set Z as follows:

        0. Start with original TSG (this is a directed graph)
        1. Remove all outgoing edges from conditioned nodes (in Z)
        2. For all pairs (m, n) of ancestors of conditioned nodes
           - if abs(tau(m)) > abs(tau(n)):
                    add an edge m --> n 
           - if abs(tau(m)) < abs(tau(n)):
                    add an edge n --> m 
           - if abs(tau(m)) == abs(tau(n)):
                    add both edges m --> n and n --> m

        This creates a graph tsg_cond that is not necessarily acyclic anymore.

        Then ancestral relations between X and Y are used to decide whether there
        is a directed path in tsg_cond from X to Y or vice versa, or whether both
        have overlapping ancestral sets, indicating that they have common drivers.
        
        Parameters
        ----------
            XYZ : list of tuples
                List of nodes chosen for current independence test

        Returns
        -------
        dseparated : bool
            True if X and Y are d-separated given Z in the graph.
        """
        if self.verbosity > 0:
            print("Testing X=%s d-sep Y=%s given Z=%s in TSG" %(X, Y, Z))

        # Check max time lag and update TSG construction if needed
        max_lag_xyz = abs(np.array(X + Y + Z)[:, 1].min()) + 1
        if max_lag_xyz > self.max_lag:
            if self.verbosity > 0:
                print("Update max_lag = ", max_lag_xyz)
            self.tsg_orig = self._links_to_tsg(self.link_coeffs, max_lag=max_lag_xyz)
            self.max_lag = max_lag_xyz

        self.tsg_cond = self.tsg_orig.copy()

        #
        # Construct modified TSG
        #
        
        # 1. Remove all outgoing edges from conditions
        for varlag in Z:
            k, tauk = varlag
            self.tsg_cond[self.varlag2node(k, self.max_lag-1 - abs(tauk)), :] = 0.
        Gcond = nx.DiGraph(self.tsg_cond)
    
        # 2. Connect all ancestors of conditioned nodes by edges depending on lag relation
        for varlag in Z:
            k, tauk = varlag
            ancestors = nx.ancestors(Gcond, self.varlag2node(k, self.max_lag-1 - abs(tauk)))    
            for (m, n) in itertools.combinations(ancestors, 2):
                # var_m, tau_m = self.node2varlag(m)
                # var_n, tau_n = self.node2varlag(n)
                # if abs(tau_m) > abs(tau_n):
                #     self.tsg_cond[m, n] = 0.5
                # elif abs(tau_m) < abs(tau_n):
                #     self.tsg_cond[n, m] = 0.5
                # else:
                self.tsg_cond[m, n] = self.tsg_cond[n, m] = 0.5
                
        Gcond = nx.DiGraph(self.tsg_cond)
        
        # Initialize indicator
        dseparated = True

        # 3. Use the modified graph to check whether paths exist by computing ancestors
        # Iterate through all nodes in X and Y
        for x in X:
            for y in Y:         

                # Get node and varlag representations
                i, taui = x
                i_node = self.varlag2node(i, self.max_lag-1 - abs(taui))
                j, tauj = y
                j_node = self.varlag2node(j, self.max_lag-1 - abs(tauj)) 
                 
                # 3.1 Either X is ancestor of Y or vice versa 
                # Compute ancestors
                anc_x = nx.ancestors(Gcond, i_node) 
                anc_y = nx.ancestors(Gcond, j_node)
                if self.verbosity > 0:
                    print("In conditioned Graph:")
                    print("an(%s) = %s" % (self.node2varlag(i_node), [self.node2varlag(node) for node in anc_x] ))
                    print("an(%s) = %s" % (self.node2varlag(j_node), [self.node2varlag(node) for node in anc_y] ))
                
                if j_node in anc_x:
                    dseparated = False
                    if self.verbosity > 0:
                        print("Directed/Collider path from Y=%s to X=%s " %(self.node2varlag(j_node), self.node2varlag(i_node)))
                if i_node in anc_y:
                    dseparated = False
                    if self.verbosity > 0:
                        print("Directed/Collider path from X=%s to Y=%s " %(self.node2varlag(i_node), self.node2varlag(j_node)))
                    
                # 3.2 Or X and Y have common ancestors
                common_anc = set(anc_x).intersection(set(anc_y))
                if len(common_anc) > 0:
                    dseparated = False
                    if self.verbosity > 0:
                        print("Detected common parents of X=%s and Y=%s: %s " %(self.node2varlag(i_node), self.node2varlag(j_node),
                                                                            [self.node2varlag(node) for node in common_anc]))

        return dseparated

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max',
                 verbosity=0):
        """Perform Oracle conditional independence test.

        Calls the d-separation function

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Not used here.

        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            Not used here.

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """

        # Get the array to test on
        X, Y, Z = self._check_XYZ(X, Y, Z, tau_max)

        if not str((X, Y, Z)) in self.dsepsets:
            self.dsepsets[str((X, Y, Z))] = self.is_dsep(X, Y, Z)

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
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns
        -------
        val : float
            The test statistic value.

        """
        # Check XYZ
        X, Y, Z = _check_XYZ(X, Y, Z, tau_max)

        if self.dsepsets[(X, Y, Z)]:
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

def plot_tsg(tsg, N, max_lag):
    """Plots TSG that is input in format (N*max_lag, N*max_lag).

       Compared to the tigramite plotting function here links X^i_{t-tau} --> X^j_t
       can be missing for different t'. Helpful to visualize the conditioned TSG.
    """

    from matplotlib import pyplot as plt
    import matplotlib.transforms as transforms

    G = nx.DiGraph(tsg)
    
    figsize=(3, 3)
    link_colorbar_label='MCI'
    arrow_linewidth=20.
    vmin_edges=-1
    vmax_edges=1.
    edge_ticks=.4
    cmap_edges='RdBu_r'
    order=None
    node_size=10
    arrowhead_size=20
    curved_radius=.2
    label_fontsize=10
    alpha=1.
    node_label_size=10
    label_space_left=0.1
    label_space_top=0.
    network_lower_bound=0.2
    undirected_style='dashed'
    

    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, frame_on=False)
    var_names = range(N)
    order = range(N)

    # list of all strengths for color map
    all_strengths = []
    # Add attributes, contemporaneous and directed links are handled separately
    for (u, v, dic) in G.edges(data=True):
        if u != v:
            if tsg[u, v] and tsg[v, u]:
                dic['undirected'] = True
                dic['directed'] = False
            else:
                dic['undirected'] = False
                dic['directed'] = True
                
            dic['undirected_alpha'] = alpha
            dic['undirected_color'] = tsg[u, v]

            dic['undirected_width'] = arrow_linewidth
            dic['undirected_attribute'] = dic['directed_attribute'] = None

            all_strengths.append(dic['undirected_color'])
            dic['directed_alpha'] = alpha
            dic['directed_width'] = dic['undirected_width'] = arrow_linewidth

            # value at argmax of average
            dic['directed_color'] = tsg[u, v]

            all_strengths.append(dic['directed_color'])
            dic['label'] = None

        dic['directed_edge'] = False
        dic['directed_edgecolor'] = None
        dic['undirected_edge'] = False
        dic['undirected_edgecolor'] = None

    # If no links are present, set value to zero
    if len(all_strengths) == 0:
        all_strengths = [0.]

    posarray = np.zeros((N * max_lag, 2))
    for i in range(N * max_lag):
        posarray[i] = np.array([(i % max_lag), (1. - i // max_lag)])

    pos_tmp = {}
    for i in range(N * max_lag):
        pos_tmp[i] = np.array([((i % max_lag) - posarray.min(axis=0)[0]) /
                                  (posarray.max(axis=0)[0] -
                                   posarray.min(axis=0)[0]),
                                  ((1. - i // max_lag) -
                                   posarray.min(axis=0)[1]) /
                                  (posarray.max(axis=0)[1] -
                                   posarray.min(axis=0)[1])])
        pos_tmp[i][np.isnan(pos_tmp[i])] = 0.

    pos = {}
    for n in range(N):
        for tau in range(max_lag):
            pos[n * max_lag + tau] = pos_tmp[order[n] * max_lag + tau]

    node_rings = {0: {'sizes': None, 'color_array': None,
                      'label': '', 'colorbar': False,
                      }
                  }

    # ] for v in range(max_lag)]
    node_labels = ['' for i in range(N * max_lag)]

    tp._draw_network_with_curved_edges(
        fig=fig, ax=ax,
        G=deepcopy(G), pos=pos,
        # dictionary of rings: {0:{'sizes':(N,)-array, 'color_array':(N,)-array
        # or None, 'cmap':string,
        node_rings=node_rings,
        # 'vmin':float or None, 'vmax':float or None, 'label':string or None}}
        node_labels=node_labels, node_label_size=node_label_size,
        node_alpha=alpha, standard_size=node_size,
        standard_cmap='OrRd', standard_color='lightgrey',
        log_sizes=False,
        cmap_links=cmap_edges, links_vmin=vmin_edges,
        links_vmax=vmax_edges, links_ticks=edge_ticks,

        cmap_links_edges='YlOrRd', links_edges_vmin=-1., links_edges_vmax=1.,
        links_edges_ticks=.2, link_edge_colorbar_label='link_edge',

        arrowstyle='simple', arrowhead_size=arrowhead_size,
        curved_radius=curved_radius, label_fontsize=label_fontsize,
        label_fraction=.5,
        link_colorbar_label=link_colorbar_label, undirected_curved=True,
        network_lower_bound=network_lower_bound,
        undirected_style=undirected_style, show_colorbar=False,
        )

    for i in range(N):
        trans = transforms.blended_transform_factory(
            fig.transFigure, ax.transData)
        ax.text(label_space_left, pos[order[i] * max_lag][1],
                '%s' % str(var_names[order[i]]), fontsize=label_fontsize,
                horizontalalignment='left', verticalalignment='center',
                transform=trans)

    for tau in np.arange(max_lag - 1, -1, -1):
        trans = transforms.blended_transform_factory(
            ax.transData, fig.transFigure)
        if tau == max_lag - 1:
            ax.text(pos[tau][0], 1.-label_space_top, r'$t$',
                    fontsize=int(label_fontsize*0.7),
                    horizontalalignment='center',
                    verticalalignment='top', transform=trans)
        else:
            ax.text(pos[tau][0], 1.-label_space_top,
                    r'$t-%s$' % str(max_lag - tau - 1),
                    fontsize=int(label_fontsize*0.7),
                    horizontalalignment='center', verticalalignment='top',
                    transform=trans)

    # fig.subplots_adjust(left=0.1, right=.98, bottom=.25, top=.9)
    # savestring = os.path.expanduser(save_name)
    
    # return fig, ax

if __name__ == '__main__':

    import tigramite.plotting as tp
    from matplotlib import pyplot as plt
    def lin_f(x): return x

    # links = {0: [((0, -1), 0.5)],
    #          1: [((1, -1), 0.5)],
    #          2: [((2, -1), 0.5), ((1, 0), 0.6), ((0, 0), 0.6)],
    #          3: [((3, -1), 0.5), ((2, 0), -0.5)],
    #          }
    links = {0: [((0, -1), 0.9)],
             1: [((1, -1), 0.8, lin_f), ((0, -1), 0.8, lin_f)],
             2: [((2, -1), 0.7, lin_f), ((1, 0), 0.6, lin_f)],
             3: [((3, -1), 0.7, lin_f), ((2, 0), -0.5, lin_f)],
             }

    oracle = OracleCI(links, verbosity=1)
    print (oracle.max_lag)

    X = [(0, 0)]
    Y = [(1, 0)]
    Z = [(3, 0)]

    oracle.run_test(X, Y, Z, verbosity=2)
    print (oracle.tsg_orig.shape, oracle.max_lag)
    plot_tsg(oracle.tsg_orig, N=oracle.N, max_lag=oracle.max_lag)
    plot_tsg(oracle.tsg_cond, N=oracle.N, max_lag=oracle.max_lag)
    plt.show()