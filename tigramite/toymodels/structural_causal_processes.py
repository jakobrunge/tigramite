"""Tigramite toymodels."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0
from __future__ import print_function
from collections import defaultdict, OrderedDict
import sys
import warnings
import copy
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import itertools

def _generate_noise(covar_matrix, time=1000, use_inverse=False):
    """
    Generate a multivariate normal distribution using correlated innovations.

    Parameters
    ----------
    covar_matrix : array
        Covariance matrix of the random variables
    time : int
        Sample size
    use_inverse : bool, optional
        Negate the off-diagonal elements and invert the covariance matrix
        before use

    return_eigenvectors
    -------
    noise : array
        Random noise generated according to covar_matrix
    """
    # Pull out the number of nodes from the shape of the covar_matrix
    n_nodes = covar_matrix.shape[0]
    # Make a deep copy for use in the inverse case
    this_covar = covar_matrix
    # Take the negative inverse if needed
    if use_inverse:
        this_covar = copy.deepcopy(covar_matrix)
        this_covar *= -1
        this_covar[np.diag_indices_from(this_covar)] *= -1
        this_covar = np.linalg.inv(this_covar)
    # Return the noise distribution
    return np.random.multivariate_normal(mean=np.zeros(n_nodes),
                                            cov=this_covar,
                                            size=time)

def _check_stability(graph):
    """
    Raises an AssertionError if the input graph corresponds to a non-stationary
    process.

    Parameters
    ----------
    graph : array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    """
    # Get the shape from the input graph
    n_nodes, _, period = graph.shape
    # Set the top section as the horizontally stacked matrix of
    # shape (n_nodes, n_nodes * period)
    stability_matrix = \
        scipy.sparse.hstack([scipy.sparse.lil_matrix(graph[:, :, t_slice])
                             for t_slice in range(period)])
    # Extend an identity matrix of shape
    # (n_nodes * (period - 1), n_nodes * (period - 1)) to shape
    # (n_nodes * (period - 1), n_nodes * period) and stack the top section on
    # top to make the stability matrix of shape
    # (n_nodes * period, n_nodes * period)
    stability_matrix = \
        scipy.sparse.vstack([stability_matrix,
                             scipy.sparse.eye(n_nodes * (period - 1),
                                              n_nodes * period)])
    # Check the number of dimensions to see if we can afford to use a dense
    # matrix
    n_eigs = stability_matrix.shape[0]
    if n_eigs <= 25:
        # If it is relatively low in dimensionality, use a dense array
        stability_matrix = stability_matrix.todense()
        eigen_values, _ = scipy.linalg.eig(stability_matrix)
    else:
        # If it is a large dimensionality, convert to a compressed row sorted
        # matrix, as it may be easier for the linear algebra package
        stability_matrix = stability_matrix.tocsr()
        # Get the eigen values of the stability matrix
        eigen_values = scipy.sparse.linalg.eigs(stability_matrix,
                                                k=(n_eigs - 2),
                                                return_eigenvectors=False)
    # Ensure they all have less than one magnitude
    assert np.all(np.abs(eigen_values) < 1.), \
        "Values given by time lagged connectivity matrix corresponds to a "+\
        " non-stationary process!"

def _check_initial_values(initial_values, shape):
    """
    Raises a AssertionError if the input initial values:
        * Are not a numpy array OR
        * Do not have the shape (n_nodes, max_delay+1)

    Parameters
    ----------
    graph : array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    """
    # Ensure it is a numpy array
    assert isinstance(initial_values, np.ndarray),\
        "User must provide initial_values as a numpy.ndarray"
    # Check the shape is correct
    assert initial_values.shape == shape,\
        "Initial values must be of shape (n_nodes, max_delay+1)"+\
        "\n current shape : " + str(initial_values.shape)+\
        "\n desired shape : " + str(shape)

def _var_network(graph,
                 add_noise=True,
                 inno_cov=None,
                 invert_inno=False,
                 T=100,
                 initial_values=None):
    """
    Returns a vector-autoregressive process with correlated innovations.

    Useful for testing.

    Example:
        graph=numpy.array([[[0.2,0.,0.],[0.5,0.,0.]],
                           [[0.,0.1,0. ],[0.3,0.,0.]]])

        represents a process

        X_1(t) = 0.2 X_1(t-1) + 0.5 X_2(t-1) + eps_1(t)
        X_2(t) = 0.3 X_2(t-1) + 0.1 X_1(t-2) + eps_2(t)

        with inv_inno_cov being the negative (except for diagonal) inverse
        covariance matrix of (eps_1(t), eps_2(t)) OR inno_cov being
        the covariance. Initial values can also be provided.


    Parameters
    ----------
    graph : array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    add_noise : bool, optional (default: True)
        Flag to add random noise or not
    inno_cov : array, optional (default: None)
        Covariance matrix of innovations.
    invert_inno : bool, optional (defualt : False)
        Flag to negate off-diagonal elements of inno_cov and invert it before
        using it as the covariance matrix of innovations
    T : int, optional (default: 100)
        Sample size.

    initial_values : array, optional (defult: None)
        Initial values for each node. Shape is (n_nodes, max_delay+1), i.e. must
        be of shape (graph.shape[1], graph.shape[2]).

    Returns
    -------
    X : array
        Array of realization.
    """

    n_nodes, _, period = graph.shape

    time = T
    # Test stability
    _check_stability(graph)

    # Generate the returned data
    data = np.random.randn(n_nodes, time)
    # Load the initial values
    if initial_values is not None:
        # Check the shape of the initial values
        _check_initial_values(initial_values, data[:, :period].shape)
        # Input the initial values
        data[:, :period] = initial_values

    # Check if we are adding noise
    noise = None
    if add_noise:
        # Use inno_cov if it was provided
        if inno_cov is not None:
            noise = _generate_noise(inno_cov,
                                    time=time,
                                    use_inverse=invert_inno)
        # Otherwise just use uncorrelated random noise
        else:
            noise = np.random.randn(time, n_nodes)

    for a_time in range(period, time):
        data_past = np.repeat(
            data[:, a_time-period:a_time][:, ::-1].reshape(1, n_nodes, period),
            n_nodes, axis=0)
        data[:, a_time] = (data_past*graph).sum(axis=2).sum(axis=1)
        if add_noise:
            data[:, a_time] += noise[a_time]

    return data.transpose()

def _iter_coeffs(parents_neighbors_coeffs):
    """
    Iterator through the current parents_neighbors_coeffs structure.  Mainly to
    save repeated code and make it easier to change this structure.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Yields
    -------
    (node_id, parent_id, time_lag, coeff) : tuple
        Tuple defining the relationship between nodes across time
    """
    # Iterate through all defined nodes
    for node_id in list(parents_neighbors_coeffs):
        # Iterate over parent nodes and unpack node and coeff
        for (parent_id, time_lag), coeff in parents_neighbors_coeffs[node_id]:
            # Yield the entry
            yield node_id, parent_id, time_lag, coeff

def _check_parent_neighbor(parents_neighbors_coeffs):
    """
    Checks to insure input parent-neighbor connectivity input is sane.  This
    means that:
        * all time lags are non-positive
        * all parent nodes are included as nodes themselves
        * all node indexing is contiguous
        * all node indexing starts from zero
    Raises a ValueError if any one of these conditions are not met.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.
    """
    # Initialize some lists for checking later
    all_nodes = set()
    all_parents = set()
    # Iterate through variables
    for j in list(parents_neighbors_coeffs):
        # Cache all node ids to ensure they are contiguous
        all_nodes.add(j)
    # Iterate through all nodes
    for j, i, tau, _ in _iter_coeffs(parents_neighbors_coeffs):
        # Check all time lags are equal to or less than zero
        if tau > 0:
            raise ValueError("Lag between parent {} and node {}".format(i, j)+\
                             " is {} > 0, must be <= 0!".format(tau))
        # Cache all parent ids to ensure they are mentioned as node ids
        all_parents.add(i)
    # Check that all nodes are contiguous from zero
    all_nodes_list = sorted(list(all_nodes))
    if all_nodes_list != list(range(len(all_nodes_list))):
        raise ValueError("Node IDs in input dictionary must be contiguous"+\
                         " and start from zero!\n"+\
                         " Found IDs : [" +\
                         ",".join(map(str, all_nodes_list))+ "]")
    # Check that all parent nodes are mentioned as a node ID
    if not all_parents.issubset(all_nodes):
        missing_nodes = sorted(list(all_parents - all_nodes))
        all_parents_list = sorted(list(all_parents))
        raise ValueError("Parent IDs in input dictionary must also be in set"+\
                         " of node IDs."+\
                         "\n Parent IDs "+" ".join(map(str, all_parents_list))+\
                         "\n Node IDs "+" ".join(map(str, all_nodes_list)) +\
                         "\n Missing IDs " + " ".join(map(str, missing_nodes)))

def _check_symmetric_relations(a_matrix):
    """
    Check if the argument matrix is symmetric.  Raise a value error with details
    about the offending elements if it is not.  This is useful for checking the
    instantaneously linked nodes have the same link strength.

    Parameters
    ----------
    a_matrix : 2D numpy array
        Relationships between nodes at tau = 0. Indexed such that first index is
        node and second is parent, i.e. node j with parent i has strength
        a_matrix[j,i]
    """
    # Check it is symmetric
    if not np.allclose(a_matrix, a_matrix.T, rtol=1e-10, atol=1e-10):
        # Store the disagreement elements
        bad_elems = ~np.isclose(a_matrix, a_matrix.T, rtol=1e-10, atol=1e-10)
        bad_idxs = np.argwhere(bad_elems)
        error_message = ""
        for node, parent in bad_idxs:
            # Check that we haven't already printed about this pair
            if bad_elems[node, parent]:
                error_message += \
                    "Parent {:d} of node {:d}".format(parent, node)+\
                    " has coefficient {:f}.\n".format(a_matrix[node, parent])+\
                    "Parent {:d} of node {:d}".format(node, parent)+\
                    " has coefficient {:f}.\n".format(a_matrix[parent, node])
            # Check if we already printed about this one
            bad_elems[node, parent] = False
            bad_elems[parent, node] = False
        raise ValueError("Relationships between nodes at tau=0 are not"+\
                         " symmetric!\n"+error_message)

def _find_max_time_lag_and_node_id(parents_neighbors_coeffs):
    """
    Function to find the maximum time lag in the parent-neighbors-coefficients
    object, as well as the largest node ID

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    (max_time_lag, max_node_id) : tuple
        Tuple of the maximum time lag and maximum node ID
    """
    # Default maximum lag and node ID
    max_time_lag = 0
    max_node_id = len(parents_neighbors_coeffs.keys()) - 1
    # Iterate through the keys in parents_neighbors_coeffs
    for j, _, tau, _ in _iter_coeffs(parents_neighbors_coeffs):
        # Find max lag time
        max_time_lag = max(max_time_lag, abs(tau))
        # Find the max node ID
        # max_node_id = max(max_node_id, j)
    # Return these values
    return max_time_lag, max_node_id

def _get_true_parent_neighbor_dict(parents_neighbors_coeffs):
    """
    Function to return the dictionary of true parent neighbor causal
    connections in time.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    true_parent_neighbor : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Initialize the returned dictionary of lists
    true_parents_neighbors = defaultdict(list)
    for j in parents_neighbors_coeffs:
        for link_props in parents_neighbors_coeffs[j]:
            i, tau = link_props[0]
            coeff = link_props[1]
            # Add parent node id and lag if non-zero coeff
            if coeff != 0.:
                true_parents_neighbors[j].append((i, tau))
    # Return the true relations
    return true_parents_neighbors

def _get_covariance_matrix(parents_neighbors_coeffs):
    """
    Determines the covariance matrix for correlated innovations

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    covar_matrix : numpy array
        Covariance matrix implied by the parents_neighbors_coeffs.  Used to
        generate correlated innovations.
    """
    # Get the total number of nodes
    _, max_node_id = \
            _find_max_time_lag_and_node_id(parents_neighbors_coeffs)
    n_nodes = max_node_id + 1
    # Initialize the covariance matrix
    covar_matrix = np.identity(n_nodes)
    # Iterate through all the node connections
    for j, i, tau, coeff in _iter_coeffs(parents_neighbors_coeffs):
        # Add to covar_matrix if node connection is instantaneous
        if tau == 0:
            covar_matrix[j, i] = coeff
    return covar_matrix

def _get_lag_connect_matrix(parents_neighbors_coeffs):
    """
    Generates the lagged connectivity matrix from a parent-neighbor
    connectivity dictionary.  Used to generate the input for _var_network

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    connect_matrix : numpy array
        Lagged connectivity matrix. Shape is (n_nodes, n_nodes, max_delay+1)
    """
    # Get the total number of nodes and time lag
    max_time_lag, max_node_id = \
            _find_max_time_lag_and_node_id(parents_neighbors_coeffs)
    n_nodes = max_node_id + 1
    n_times = max_time_lag + 1
    # Initialize full time graph
    connect_matrix = np.zeros((n_nodes, n_nodes, n_times))
    for j, i, tau, coeff in _iter_coeffs(parents_neighbors_coeffs):
        # If there is a non-zero time lag, add the connection to the matrix
        if tau != 0:
            connect_matrix[j, i, -(tau+1)] = coeff
    # Return the connectivity matrix
    return connect_matrix

def var_process(parents_neighbors_coeffs, T=1000, use='inv_inno_cov',
                verbosity=0, initial_values=None):
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with possibly more user-friendly input options.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format: {..., j:[((var1, lag1), coef1), ((var2, lag2),
        coef2), ...], ...} for all variables where vars must be in [0..N-1]
        and lags <= 0 with number of variables N. If lag=0, a nonzero value
        in the covariance matrix (or its inverse) is implied. These should be
        the same for (i, j) and (j, i).
    use : str, optional (default: 'inv_inno_cov')
        Specifier, either 'inno_cov' or 'inv_inno_cov'.
        Any other specifier will result in non-correlated noise.
        For debugging, 'no_noise' can also be specified, in which case random
        noise will be disabled.
    T : int, optional (default: 1000)
        Sample size.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    initial_values : array, optional (default: None)
        Initial values for each node. Shape must be (N, max_delay+1)

    Returns
    -------
    data : array-like
        Data generated from this process
    true_parent_neighbor : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Check the input parents_neighbors_coeffs dictionary for sanity
    _check_parent_neighbor(parents_neighbors_coeffs)
    # Generate the true parent neighbors graph
    true_parents_neighbors = \
        _get_true_parent_neighbor_dict(parents_neighbors_coeffs)
    # Generate the correlated innovations
    innos = _get_covariance_matrix(parents_neighbors_coeffs)
    # Generate the lagged connectivity matrix for _var_network
    connect_matrix = _get_lag_connect_matrix(parents_neighbors_coeffs)
    # Default values as per 'inno_cov'
    add_noise = True
    invert_inno = False
    # Use the correlated innovations
    if use == 'inno_cov':
        if verbosity > 0:
            print("\nInnovation Cov =\n%s" % str(innos))
    # Use the inverted correlated innovations
    elif use == 'inv_inno_cov':
        invert_inno = True
        if verbosity > 0:
            print("\nInverse Innovation Cov =\n%s" % str(innos))
    # Do not use any noise
    elif use == 'no_noise':
        add_noise = False
        if verbosity > 0:
            print("\nInverse Innovation Cov =\n%s" % str(innos))
    # Use decorrelated noise
    else:
        innos = None
    # Ensure the innovation matrix is symmetric if it is used
    if (innos is not None) and add_noise:
        _check_symmetric_relations(innos)
    # Generate the data using _var_network
    data = _var_network(graph=connect_matrix,
                        add_noise=add_noise,
                        inno_cov=innos,
                        invert_inno=invert_inno,
                        T=T,
                        initial_values=initial_values)
    # Return the data
    return data, true_parents_neighbors

class _Graph():
    r"""Helper class to handle graph properties.

    Parameters
    ----------
    vertices : list
        List of nodes.
    """
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices 
  
    def addEdge(self,u,v):
        """Adding edge to graph."""
        self.graph[u].append(v) 
  
    def isCyclicUtil(self, v, visited, recStack): 
        """Utility function to return whether graph is cyclic."""
        # Mark current node as visited and
        # adds to recursion stack 
        visited[v] = True
        recStack[v] = True
  
        # Recur for all neighbours 
        # if any neighbour is visited and in  
        # recStack then graph is cyclic 
        for neighbour in self.graph[v]: 
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                return True
  
        # The node needs to be poped from  
        # recursion stack before function ends 
        recStack[v] = False
        return False
  
    def isCyclic(self):
        """Returns whether graph is cyclic."""
        visited = [False] * self.V 
        recStack = [False] * self.V 
        for node in range(self.V): 
            if visited[node] == False: 
                if self.isCyclicUtil(node,visited,recStack) == True: 
                    return True
        return False
  
    def topologicalSortUtil(self,v,visited,stack):
        """A recursive function used by topologicalSort ."""
        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)

        # Push current vertex to stack which stores result
        stack.insert(0,v)

    def topologicalSort(self):
        """A sorting function. """
        # Mark all the vertices as not visited 
        visited = [False]*self.V 
        stack =[] 

        # Call the recursive helper function to store Topological 
        # Sort starting from all vertices one by one 
        for i in range(self.V): 
          if visited[i] == False: 
              self.topologicalSortUtil(i, visited,stack) 

        return stack

def structural_causal_process_ensemble(realizations=10, ensemble_seed=None, **kwargs):
    """Returns an ensemble of time series generated from a structural causal process.

    This adds an ensemble dimension to the output of structural_causal_process.

    See docstring of structural_causal_process for details.

    Parameters
    ----------
    ensemble_seed : int, optional (default: None)
        Random seed for entire ensemble.
    ** kwargs : 
        Arguments of structural_causal_process.

    Returns
    -------
    data : array-like
        Data generated from this process, shape (M, T, N).
    nonvalid : bool
        Indicates whether data has NaNs or infinities.

    """

    nonvalid = False
    for m in range(realizations):

        # Set ensemble seed
        if ensemble_seed is None:
            seed_here = None
        else:
            seed_here = realizations * ensemble_seed + m

        # Get data
        data, nonvalid_here = structural_causal_process(seed=seed_here, **kwargs)
        
        # Update non-validity
        if nonvalid_here:
            nonvalid = True

        if m == 0:
            data_ensemble = np.zeros((realizations,) + data.shape, dtype='float32')

        data_ensemble[m] = data

    return data_ensemble, nonvalid

def structural_causal_process(links, T, noises=None, 
                        intervention=None, intervention_type='hard',
                        transient_fraction=0.2,
                        seed=None):
    """Returns a time series generated from a structural causal process.

    Allows lagged and contemporaneous dependencies and includes the option
    to have intervened variables or particular samples.

    The interventional data is in particular useful for generating ground
    truth for the CausalEffects class.

    In more detail, the method implements a generalized additive noise model process of the form

    .. math:: X^j_t = \\eta^j_t + \\sum_{X^i_{t-\\tau}\\in \\mathcal{P}(X^j_t)}
              c^i_{\\tau} f^i_{\\tau}(X^i_{t-\\tau})

    Links have the format ``{0:[((i, -tau), coeff, func),...], 1:[...],
    ...}`` where ``func`` can be an arbitrary (nonlinear) function provided
    as a python callable with one argument and coeff is the multiplication
    factor. The noise distributions of :math:`\\eta^j` can be specified in
    ``noises``.

    Through the parameters ``intervention`` and ``intervention_type`` the model
    can also be generated with intervened variables.

    Parameters
    ----------
    links : dict
        Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
        ...} for all variables where i must be in [0..N-1] and tau >= 0 with
        number of variables N. coeff must be a float and func a python
        callable of one argument.
    T : int
        Sample size.
    noises : list of callables or array, optional (default: 'np.random.randn')
        Random distribution function that is called with noises[j](T). If an array,
        it must be of shape ((transient_fraction + 1)*T, N).
    intervention : dict
        Dictionary of format: {1:np.array, ...} containing only keys of intervened
        variables with the value being the array of length T with interventional values.
        Set values to np.nan to leave specific time points of a variable un-intervened.
    intervention_type : str or dict
        Dictionary of format: {1:'hard',  3:'soft', ...} to specify whether intervention is 
        hard (set value) or soft (add value) for variable j. If str, all interventions have 
        the same type.
    transient_fraction : float
        Added percentage of T used as a transient. In total a realization of length
        (transient_fraction + 1)*T will be generated, but then transient_fraction*T will be
        cut off.
    seed : int, optional (default: None)
        Random seed.

    Returns
    -------
    data : array-like
        Data generated from this process, shape (T, N).
    nonvalid : bool
        Indicates whether data has NaNs or infinities.

    """
    random_state = np.random.RandomState(seed)

    N = len(links.keys())
    if noises is None:
        noises = [random_state.randn for j in range(N)]

    if N != max(links.keys())+1:
        raise ValueError("links keys must match N.")

    if isinstance(noises, np.ndarray):
        if noises.shape != (T + int(math.floor(transient_fraction*T)), N):
            raise ValueError("noises.shape must match ((transient_fraction + 1)*T, N).")            
    else:
        if N != len(noises):
            raise ValueError("noises keys must match N.")

    # Check parameters
    max_lag = 0
    contemp_dag = _Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1: 
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort() 

    if intervention is not None:
        if intervention_type is None:
            intervention_type = {j:'hard' for j in intervention}
        elif isinstance(intervention_type, str):
            intervention_type = {j:intervention_type for j in intervention}
        for j in intervention.keys():
            if len(np.atleast_1d(intervention[j])) != T:
                raise ValueError("intervention array for j=%s must be of length T = %d" %(j, T))
            if j not in intervention_type.keys():        
                raise ValueError("intervention_type dictionary must contain entry for %s" %(j))

    transient = int(math.floor(transient_fraction*T))

    data = np.zeros((T+transient, N), dtype='float32')
    for j in range(N):
        if isinstance(noises, np.ndarray):
            data[:, j] = noises[:, j]
        else:
            data[:, j] = noises[j](T+transient)

    for t in range(max_lag, T+transient):
        for j in causal_order:

            if (intervention is not None and j in intervention and t >= transient
                and np.isnan(intervention[j][t - transient]) == False):
                if intervention_type[j] == 'hard':
                    data[t, j] = intervention[j][t - transient]
                    # Move to next j and skip link_props-loop from parents below 
                    continue
                else:
                    data[t, j] += intervention[j][t - transient]

            # This loop is only entered if intervention_type != 'hard'
            for link_props in links[j]:
                var, lag = link_props[0]
                coeff = link_props[1]
                func = link_props[2]
                data[t, j] += coeff * func(data[t + lag, var])

    data = data[transient:]

    nonvalid = (np.any(np.isnan(data)) or np.any(np.isinf(data)))

    return data, nonvalid

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
                if not isinstance(coeff, float) or coeff != 0.:
                    min_lag = min(min_lag, abs(lag))
                    max_lag = max(max_lag, abs(lag))
            else:
                var, lag = link_props
                min_lag = min(min_lag, abs(lag))
                max_lag = max(max_lag, abs(lag))   

    return min_lag, max_lag

def _get_parents(links, exclude_contemp=False):
    """Helper function to parents from links
    """

    N = len(links)

    # Get maximum time lag
    parents = {}
    for j in range(N):
        parents[j] = []
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            # func = link_props[2]
            if coeff != 0.:
                if not (exclude_contemp and lag == 0):
                    parents[j].append((var, lag))

    return parents

def _get_children(parents):
    """Helper function to children from parents
    """

    N = len(parents)
    children = dict([(j, []) for j in range(N)])

    for j in range(N):
        for par in parents[j]:
            i, tau = par
            children[i].append((j, abs(tau)))

    return children

def links_to_graph(links, tau_max=None):
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

def dag_to_links(dag):
    """Helper function to convert DAG graph to dictionary of parents.

    Parameters
    ---------
    dag : array of shape (N, N, tau_max+1)
        Matrix format of graph in string format. Must be DAG.

    Returns
    -------
    parents : dict
        Dictionary of form {0:[(0, -1), ...], 1:[...], ...}.
    """
    N = dag.shape[0]

    parents = dict([(j, []) for j in range(N)])

    if np.any(dag=='o-o') or np.any(dag=='x-x'):
        raise ValueError("graph must be DAG.")

    for (i, j, tau) in zip(*np.where(dag=='-->')):
        parents[j].append((i, -tau))

    return parents

def generate_structural_causal_process(
                N=2, 
                L=1, 
                dependency_funcs=['linear'], 
                dependency_coeffs=[-0.5, 0.5], 
                auto_coeffs=[0.5, 0.7], 
                contemp_fraction=0.,
                max_lag=1, 
                noise_dists=['gaussian'],
                noise_means=[0.],
                noise_sigmas=[0.5, 2.],
                noise_seed=None,
                seed=None):
    """"Randomly generates a structural causal process based on input characteristics.

    The process has the form 

    .. math:: X^j_t = \\eta^j_t + a^j X^j_{t-1} + \\sum_{X^i_{t-\\tau}\\in pa(X^j_t)}
              c^i_{\\tau} f^i_{\\tau}(X^i_{t-\\tau})

    where ``j = 1, ..., N``. Here the properties of :math:`\\eta^j_t` are
    randomly frawn from the noise parameters (see below), :math:`pa
    (X^j_t)` are the causal parents drawn randomly such that in total ``L``
    links occur out of which ``contemp_fraction`` are contemporaneous and
    their time lags are drawn from ``[0 or 1..max_lag]``, the
    coefficients :math:`c^i_{\\tau}` are drawn from
    ``dependency_coeffs``, :math:`a^j` are drawn from ``auto_coeffs``,
    and :math:`f^i_{\\tau}` are drawn from ``dependency_funcs``.

    The returned dictionary links has the format 
    ``{0:[((i, -tau), coeff, func),...], 1:[...], ...}`` 
    where ``func`` can be an arbitrary (nonlinear) function provided
    as a python callable with one argument and coeff is the multiplication
    factor. The noise distributions of :math:`\\eta^j` are returned in
    ``noises``, see specifics below.

    The process might be non-stationary. In case of asymptotically linear
    dependency functions and no contemporaneous links this can be checked with
    ``check_stationarity(...)``. Otherwise check by generating a large sample
    and test for np.inf.

    Parameters
    ---------
    N : int
        Number of variables.
    L : int
        Number of cross-links between two different variables.
    dependency_funcs : list
        List of callables or strings 'linear' or 'nonlinear' for a linear and a specific nonlinear function
        that is asymptotically linear.
    dependency_coeffs : list
        List of floats from which the coupling coefficients are randomly drawn.
    auto_coeffs : list
        List of floats from which the lag-1 autodependencies are randomly drawn.
    contemp_fraction : float [0., 1]
        Fraction of the L links that are contemporaneous (lag zero).
    max_lag : int
        Maximum lag from which the time lags of links are drawn.
    noise_dists : list
        List of noise functions. Either in
        {'gaussian', 'weibull', 'uniform'} or user-specified, in which case
        it must be parametrized just by the size parameter. E.g. def beta
        (T): return np.random.beta(a=1, b=0.5, T)
    noise_means : list
        Noise mean. Only used for noise in {'gaussian', 'weibull', 'uniform'}.
    noise_sigmas : list
        Noise standard deviation. Only used for noise in {'gaussian', 'weibull', 'uniform'}.   
    seed : int
        Random seed to draw the above random functions from.
    noise_seed : int
        Random seed for noise function random generator.

    Returns
    -------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
    noises : list
        List of N noise functions to call by noise(T) where T is the time series length.
    """
    
    # Init random states
    random_state = np.random.RandomState(seed)
    random_state_noise = np.random.RandomState(noise_seed)

    def linear(x): return x
    def nonlinear(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    if max_lag == 0:
        contemp_fraction = 1.

    if contemp_fraction > 0.:
        ordered_pairs = list(itertools.combinations(range(N), 2))
        max_poss_links = min(L, len(ordered_pairs))
        L_contemp = int(contemp_fraction*max_poss_links)
        L_lagged = max_poss_links - L_contemp
    else:
        L_lagged = L
        L_contemp = 0

    # Random order
    causal_order = list(random_state.permutation(N))

    # Init link dict
    links = dict([(i, []) for i in range(N)])

    # Generate auto-dependencies at lag 1
    if max_lag > 0:
        for i in causal_order:
            a = random_state.choice(auto_coeffs)
            if a != 0.:
                links[i].append(((int(i), -1), float(a), linear))

    # Non-cyclic contemp random pairs of links such that
    # index of cause < index of effect
    # Take up to (!) L_contemp links
    ordered_pairs = list(itertools.combinations(range(N), 2))
    random_state.shuffle(ordered_pairs)
    contemp_links = [(causal_order[pair[0]], causal_order[pair[1]]) 
                    for pair in ordered_pairs[:L_contemp]]

    # Possibly cyclic lagged random pairs of links 
    # where we remove already chosen contemp links
    # Take up to (!) L_contemp links
    unordered_pairs = list(itertools.permutations(range(N), 2))
    unordered_pairs = list(set(unordered_pairs) - set(ordered_pairs[:L_contemp]))
    random_state.shuffle(unordered_pairs)
    lagged_links = [(causal_order[pair[0]], causal_order[pair[1]]) 
                    for pair in unordered_pairs[:L_lagged]]

    chosen_links = lagged_links + contemp_links

    # Populate links
    for (i, j) in chosen_links:

        # Choose lag
        if (i, j) in contemp_links:
            tau = 0
        else:
            tau = int(random_state.randint(1, max_lag+1))

        # Choose dependency
        c = float(random_state.choice(dependency_coeffs))
        if c != 0:
            func = random_state.choice(dependency_funcs)
            if func == 'linear':
                func = linear
            elif func == 'nonlinear':
                func = nonlinear

            links[j].append(((int(i), -tau), c, func))

    # Now generate noise functions
    # Either choose among pre-defined noise types or supply your own
    class NoiseModel:
        def __init__(self, mean=0., sigma=1.):
            self.mean = mean
            self.sigma = sigma
        def gaussian(self, T):
            # Get zero-mean unit variance gaussian distribution
            return self.mean + self.sigma*random_state_noise.randn(T)
        def weibull(self, T): 
            # Get zero-mean sigma variance weibull distribution
            a = 2
            mean = scipy.special.gamma(1./a + 1)
            variance = scipy.special.gamma(2./a + 1) - scipy.special.gamma(1./a + 1)**2
            return self.mean + self.sigma*(random_state_noise.weibull(a=a, size=T) - mean)/np.sqrt(variance)
        def uniform(self, T): 
            # Get zero-mean sigma variance uniform distribution
            mean = 0.5
            variance = 1./12.
            return self.mean + self.sigma*(random_state_noise.uniform(size=T) - mean)/np.sqrt(variance)

    noises = []
    for j in links:
        noise_dist = random_state.choice(noise_dists)
        noise_mean = random_state.choice(noise_means)
        noise_sigma = random_state.choice(noise_sigmas)

        if noise_dist in ['gaussian', 'weibull', 'uniform']:
            noise = getattr(NoiseModel(mean = noise_mean, sigma = noise_sigma), noise_dist)
        else:
            noise = noise_dist
        
        noises.append(noise)

    return links, noises

def check_stationarity(links):
    """Returns stationarity according to a unit root test.

    Assumes an at least asymptotically linear vector autoregressive process
    without contemporaneous links.

    Parameters
    ---------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
        Also format {0:[(0, -1), ...], 1:[...], ...} is allowed.

    Returns
    -------
    stationary : bool
        True if VAR process is stationary.
    """

    N = len(links)
    # Check parameters
    max_lag = 0

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            # coeff = link_props[1]
            # coupling = link_props[2]

            max_lag = max(max_lag, abs(lag))

    graph = np.zeros((N,N,max_lag))
    couplings = []

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff    = link_props[1]
            coupling = link_props[2]
            if abs(lag) > 0:
                graph[j,var,abs(lag)-1] = coeff
            couplings.append(coupling)

    stabmat = np.zeros((N*max_lag,N*max_lag))
    index = 0

    for i in range(0,N*max_lag,N):
        stabmat[:N,i:i+N] = graph[:,:,index]
        if index < max_lag-1:
            stabmat[i+N:i+2*N,i:i+N] = np.identity(N)
        index += 1

    eig = np.linalg.eig(stabmat)[0]

    if np.all(np.abs(eig) < 1.):
        stationary = True
    else:
        stationary = False

    return stationary
    # if len(eig) == 0:
    #     return stationary, 0.
    # else:
    #     return stationary, np.abs(eig).max()

class _Logger(object):
    """Class to append print output to a string which can be saved"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = ""       # open("log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log += message  # .write(message)


if __name__ == '__main__':
    
    ## Generate some time series from a structural causal process
    def lin_f(x): return x
    def nonlin_f(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    links, noises = generate_structural_causal_process(seed=1, noise_seed=1)
    
    # data, nonstat = structural_causal_process(links, seed=1,
    #          T=10, noises=noises)

    data, nonstat = structural_causal_process_ensemble(realizations=2, ensemble_seed=0, 
        links=links, T=2, noises=noises)

    print(data)
    print(data.shape)


    # links = {0: [((0, -1), 0.9, lin_f)],
    #          1: [((1, -1), 0.8, lin_f), ((0, -1), 0.3, nonlin_f)],
    #          2: [((2, -1), 0.7, lin_f), ((1, 0), -0.2, lin_f)],
    #          }
    # noises = [np.random.randn, np.random.randn, np.random.randn]

    # data, nonstat = structural_causal_process(links,
    #  T=100, noises=noises)

