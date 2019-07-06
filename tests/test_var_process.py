"""
Testing var_process using exponential decay examples.
"""
from __future__ import print_function
import copy
import pytest
import numpy as np
import tigramite.data_processing as pp

# Pylint settings
# pylint: disable=redefined-outer-name

# TEST DATA GENERATION #########################################################
def gen_decay(init_val, decay_const, n_times, delay):
    """
    Generate the initial values for the exponential decay test cases
    """
    init_arr = np.ones(n_times)*init_val
    for time in range(delay, n_times):
        init_arr[time] = init_arr[time-delay] * decay_const
    return init_arr

@pytest.fixture(params=[
    #Generates <dim> decoupled exponential decays. Decay starts from initial
    # value <init> with decay constant <decay> appreciating every unit of time
    # for a total of <n_times>.  The value at time <t> is taken as the:
    #     [value at (<t> - <delay>)]*<decay>
    #Define parameters to use for exponential decay cases.
    #init,  decay, n_times, delay, dim, name
    (100.,  0.99,  1000,    1,     1,   "Default Exp Decay"),
    (-100., 0.99,  1000,    1,     1,   "Negative Exp Decay"),
    (100.,  0.99,  10,      1,     1,   "Short Exp Decay"),
    (100.,  0.99,  1000,    10,    1,   "Delayed Exp Decay"),
    (100.,  0.99,  1000,    1,     2,   "2D Exp Decay"),])
def decoupled_exp_decay_process(request):
    # Get the parameters
    init, decay, n_times, delay, dim, name = request.param
    # Generate the initial values
    init_vals = np.array([gen_decay(init, decay, delay+1, delay) \
                          for _ in range(dim)])
    # Generate the coefficients
    coefs = {}
    for node in range(dim):
        coefs[node] = [((node, -delay), decay)]
    # Generate the expected values
    expect = np.vstack([gen_decay(init, decay, n_times, delay)\
                       for _ in range(dim)]).T
    return name, init_vals, coefs, expect

@pytest.fixture()
# Couples all the decays by creating negative versions of each variable, then
# summing all variables to create a variable that is always zero.
def coupled_exp_decay_process(decoupled_exp_decay_process):
    # Get the decoupled version
    name, init_vals, coefs, expect = decoupled_exp_decay_process
    # Reflect all conditions and expectations to a negative version
    init_vals = np.vstack([init_vals, -init_vals])
    expect = np.hstack([expect, -expect])
    # Create reflected, decoupled nodes
    n_nodes = len(coefs)
    for key in list(coefs.keys()):
        # Grab the delay and coefficient
        delay  = coefs[key][0][0][1]
        a_coef = coefs[key][0][1]
        # Make a new, decoupled node
        coefs[key + n_nodes] = [((key+n_nodes, delay), a_coef)]
    # Couple the last node as the sum of all the other nodes
    all_nodes = []
    for coef_val in coefs.values():
        all_nodes += coef_val
    # Add in the last node
    coefs[len(coefs)] = all_nodes
    # Sum all initial values to get the initial value for the last, coupled node
    init_vals = np.vstack([init_vals, np.sum(init_vals, axis=0)])
    # Sum all expected values to get the expected value for the last, coupled
    # node
    expect = np.hstack([expect, np.sum(expect, axis=-1).reshape(-1, 1)])
    return "Coupled "+name, init_vals, coefs, expect

def gen_process(a_process):
    """
    Calls var_process for the process fixtures
    """
    # Get the initial values and setup for the decay process
    _, init_vals, coefs, expect = a_process
    # Deducte the max time from the expected answer shape
    max_time = expect.shape[0]
    # Generate the data
    data, true_parents_neighbors = pp.var_process(coefs,
                                                  T=max_time,
                                                  initial_values=init_vals,
                                                  use="no_noise")
    return data, true_parents_neighbors

def check_process_data(name, coefs, data, expect):
    """
    Checks the data is as expected
    """
    # Strip the coefficients from the input parameters
    error_message = "PARAM SET: " + name + " FAILED\n"+\
                    "Bad parameter set for process\n"+\
                    " "+str(coefs)+"\n"\
                    "Data should match expected value"+\
                    "\n Data[:5]:\n "+str(data[:5])+\
                    "\n Data[-5:]:\n "+str(data[-5:])+\
                    "\n Expect[:5]:\n "+str(expect[:5])+\
                    "\n Expect[-5:]:\n "+str(expect[-5:])
    # Check they are the same
    np.testing.assert_allclose(data,
                               expect,
                               rtol=1e-10,
                               verbose=False,
                               err_msg=error_message)

def check_process_parent_neighbours(return_links, coefs):
    """
    Checks the returned process parent-neighbour graphs
    """
    # Strip the coefficients from the input parameters
    true_node_links = dict()
    for node_id, all_node_links in coefs.items():
        true_node_links[node_id] = [link for link, _ in all_node_links]
    # Check this matches the given parameters
    assert return_links == true_node_links

def check_process(a_process):
    """
    Checks var_process
    """
    # Unpack the process
    name, _, coefs, expect = a_process
    # Generate the data
    data, return_links = gen_process(a_process)
    # Ensure the correct parent links are returned
    check_process_parent_neighbours(return_links, coefs)
    # Ensure the data is also correct
    check_process_data(name, coefs, data, expect)

def test_decoupled_process_data(decoupled_exp_decay_process):
    """
    Test that the decoupled processes work
    """
    # Generate and check the data
    check_process(decoupled_exp_decay_process)

def test_coupled_process_data(coupled_exp_decay_process):
    """
    Test that the coupled process work
    """
    # Generate and check the data
    check_process(coupled_exp_decay_process)

# TEST PARAMETER CHECKING ######################################################
@pytest.fixture(params=[
    #Returns a good parameter set along with a modified, bad parameter set that
    #should raise an error in var_process
    #Bad parameter sets are created by defining a link using:
    #node,  parent, delay, error_message
    (0,     0,      1,     "Positive time delay"),
    (100,   0,      -1,    "Non-contiguous node ID"),
    (-10,   0,      -1,    "Node IDs not starting from zero"),
    (0,     100,    -1,    "Non-existant node as parent"),
    (1,     0,      0,     "Non-symmetric instantaneous relation")])
def bad_parameter_sets(request):
    # Define a good parameter set
    default_coef = 0.5
    good_params = {}
    good_params[0] = [((0, -1), default_coef)]
    good_params[1] = [((1, -1), default_coef)] + good_params[0]
    good_params[2] = [((2, -1), default_coef)] + good_params[1]
    # Define a bad parameter set
    node, parent, delay, message = request.param
    bad_params = copy.deepcopy(good_params)
    bad_params[node] = [((parent, delay), default_coef)]
    return good_params, bad_params, message

def test_bad_parameters(bad_parameter_sets):
    """
    Test that the correct exceptions are raised for bad input connectivity
    dictionaries
    """
    # Unpack the parameter set fixture
    good_params, bad_params, message = bad_parameter_sets
    error_message = message + " should trigger a value error for var_process "+\
                              "parent-neighbour dictionary"
    # Test the good parameter set
    try:
        pp._check_parent_neighbor(good_params)
        covar = pp._get_covariance_matrix(good_params)
        pp._check_symmetric_relations(covar)
    # Ensure no exception is raised
    except:
        pytest.fail("Good parameter set triggers exception incorrectly!")
    # Ensure an exception is raised for a bad parameter set
    with pytest.raises(ValueError):
        pp._check_parent_neighbor(bad_params)
        covar = pp._get_covariance_matrix(bad_params)
        pp._check_symmetric_relations(covar)
        pytest.fail(error_message)

# TEST STABILITY CHECKING ######################################################
@pytest.fixture(params=[
    #Returns a stable parameter set along with a modified, unstable parameter 
    #set that should raise an error in var_process
    #Parameter sets are created by autocorrelated nodes, with the node ID equal
    #to the time delay:
    #max_delay,  coefficient,  error_message
    (4,          10.0,         "Low-dimensionality, non-stationary process"),
    (10,         10.0,         "High-dimensionality, non-stationary process")])
def unstable_parameter_sets(request):
    # Define a stable parameter set
    stab_coef = 0.1
    stab_params = dict()
    # Define the unstable parameter set
    unst_params = dict()
    # Get the dimensionality and coefficient
    max_delay, unst_coef, message = request.param
    # Loop over all possible delays from this max delay
    for delay in range(max_delay):
        for a_params, a_coef in zip([stab_params, unst_params],
                                    [stab_coef, unst_coef]):
            node_id = delay
            # Correlate each node with the 0th node, using a stable coefficient
            a_params[node_id] = [((0, -delay), stab_coef)]
            # Autocorrelate each node N with delay N and the given coefficient
            a_params[node_id] = [((node_id, -delay), a_coef)]
    # Return the stable and unstable coefficients
    return pp._get_lag_connect_matrix(stab_params),\
           pp._get_lag_connect_matrix(unst_params),\
           message

def test_stability_parameters(unstable_parameter_sets):
    """
    Test that the correct exceptions are raised for unstable lagged connectivity 
    matricies
    """
    # Unpack the parameter set fixture
    stab_matrix, unst_matrix, message = unstable_parameter_sets
    error_message = message + " should trigger an assertion error for "+\
                              "_var_network graph."
    # Test the good parameter set
    try:
        pp._check_stability(stab_matrix)
    # Ensure no exception is raised
    except:
        pytest.fail("Stable matrix set triggers exception incorrectly!")
    # Ensure an exception is raised for a bad parameter set
    with pytest.raises(AssertionError):
        pp._check_stability(unst_matrix)
        pytest.fail(error_message)

# TEST NOISE GENERATION ########################################################
@pytest.fixture()
def covariance_parameters(request):
    """
    Define a good parameter set with no time delays at all to induce
    a noise-only sample and return the resulting covariance matrix
    """
    default_coef = 0.1
    good_params = {}
    good_params[0] = [((1, 0), default_coef * 1.),
                      ((2, 0), default_coef * 3.)]
    good_params[1] = [((2, 0), default_coef * 2.),
                      ((0, 0), default_coef * 1.)]
    good_params[2] = [((0, 0), default_coef * 3.),
                      ((1, 0), default_coef * 2.)]
    good_params[3] = [((3, 0), default_coef * 4.)]
    # Get the innovation matrix
    covar_matrix = pp._get_covariance_matrix(good_params)
    return good_params, covar_matrix

def test_symmetric_covariance(covariance_parameters):
    """
    Test the random noise generation in var_process
    """
    # Unpack the covariance matrix
    _, covar_matrix = covariance_parameters
    # Test the matrix is symmetric
    err_message = "Covariance matrix must be symmetric!"
    np.testing.assert_allclose(covar_matrix,
                               covar_matrix.T,
                               rtol=1e-10,
                               verbose=True,
                               err_msg=err_message)

def test_covariance_construction(covariance_parameters):
    """
    Test the random noise covariance matrix construction from a set of
    parameters
    """
    # Unpack the covariance matrix and parameters
    good_params, covar_matrix = covariance_parameters
    # Check the values are passed correctly
    for j, i, _, coeff in pp._iter_coeffs(good_params):
        covar_coeff = covar_matrix[j, i]
        err_message = "Node {} and parent node {} have".format(j, i)+\
                        " coefficient {} for tau == 0,\nbut the".format(coeff)+\
                        " coefficient in the covariance matrix"+\
                        " is {} ".format(covar_coeff)
        np.testing.assert_approx_equal(coeff, covar_coeff, err_msg=err_message)

def test_noise_generation(covariance_parameters):
    """
    Ensure the covariance parameters are respected when the noise is generated
    """
    # Unpack the parameters and covariance matrix
    good_params, covar_matrix = covariance_parameters
    # Generate noise-only from this parameter set
    data, _ = pp.var_process(good_params, T=10000, use='inno_cov',
                             verbosity=0, initial_values=None)
    # Get the covariance of the data set
    covar_result = np.cov(data.T)
    err_message = "Covariance of data does not match covariance implied by "+\
                  " parameter set"
    np.testing.assert_allclose(covar_matrix, covar_result,
                               rtol=1e-1, atol=0.025,
                               verbose=True, err_msg=err_message)
