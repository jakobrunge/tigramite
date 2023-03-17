"""
Tests for independence_tests.py.
"""
from __future__ import print_function
from collections import OrderedDict
import numpy as np
import pytest
import sys

import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

# CONSTRUCT ARRAY TESTING FUNCTIONS ############################################
def rand_node(t_min, n_max, t_max=0, n_min=0):
    """
    Generate a random node to test
    """
    rand_node = np.random.randint(n_min, n_max)
    rand_time = np.random.randint(t_min, t_max)
    return (rand_node, rand_time)

def gen_nodes(n_nodes, seed, t_min, n_max):
    """
    Generate some random nodes to tests
    """
    # Set the seed if needed
    np.random.seed(seed)
    # Y nodes are always at (0, 0)
    y_nds = [(0, 0)]
    # X nodes is only one node
    x_nds = [rand_node(t_min, n_max)]
    # Z nodes are multiple nodes
    z_nds = [rand_node(t_min, n_max) for _ in range(n_nodes)]
    return x_nds, y_nds, z_nds


# CONSTRUCT ARRAY TESTING ######################################################
@pytest.fixture(params=[
    # Parameterize the array construction
    #(X, Y, Z) nodes,        t_max, m_val, mask_type
    (gen_nodes(3, 0, -3, 9), 3,     False, None),           # Few nodes
    (gen_nodes(7, 1, -3, 9), 3,     False, None),           # More nodes
    (gen_nodes(7, 2, -3, 3), 3,     False, None),           # Repeated nodes
    (gen_nodes(7, 3, -3, 9), 3,     True,  None),           # Missing vals
    (gen_nodes(7, 4, -3, 9), 3,     True,  ['x']),          # M-val + masked x
    (gen_nodes(7, 4, -3, 9), 3,     True,  ['x','y']),      # M-val + masked xy
    (gen_nodes(3, 5, -4, 9), 2,     False, ['x']),          # masked x
    (gen_nodes(3, 6, -4, 9), 2,     False, ['y']),          # masked y
    (gen_nodes(3, 7, -4, 9), 2,     False, ['z']),          # masked z
    (gen_nodes(3, 7, -4, 9), 2,     False, ['x','y','z'])]) # mask xyz
def cstrct_array_params(request):
    return request.param

def test_construct_array(cstrct_array_params):
    # Unpack the parameters
    (x_nds, y_nds, z_nds), tau_max, missing_vals, mask_type =\
        cstrct_array_params
    # Make some fake data
    data = np.arange(1000).reshape(10, 100).T.astype('float')
    # Get the needed parameters from the data
    T, N = data.shape
    max_lag = 2*tau_max
    n_times = T - max_lag

    # When testing masking and missing value flags, we will remove time slices,
    # starting with the earliest slice.  This counter keeps track of how many
    # rows have been masked.
    n_rows_masked = 0

    # Make a fake mask
    data_mask = np.zeros_like(data, dtype='bool')
    if mask_type is not None:
        for var, nodes in zip(['x', 'y', 'z'], [x_nds, y_nds, z_nds]):
            if var in mask_type:
                # Get the first node
                a_nd, a_tau = nodes[0]
                # Mask the first value of this node
                data_mask[a_tau - n_times + n_rows_masked, a_nd] = True
                n_rows_masked += 1

    # Choose fake missing value as the earliest time entry in the first z-node
    # from the original (non-shifted) datathat is not cutoff by max_lag or
    # masked values from the first z-node
    missing_flag = None
    if missing_vals:
        # Get the node index
        a_nd, _ = z_nds[0]
        # Select the earliest non-cutoff entry from the unshifted data set
        earliest_time = max_lag + n_rows_masked
        missing_flag = data[earliest_time, a_nd]
        # Record that the row with this value and all rows up to max_lag after
        # this value have been cut off as well
        # n_rows_masked += 1

    # Construct the array
    data_f = pp.DataFrame(data, data_mask, missing_flag)
    array, xyz, _ = data_f.construct_array(x_nds, y_nds, z_nds,
                                        tau_max=tau_max,
                                        mask_type=mask_type,
                                        verbosity=VERBOSITY)
    # Ensure x_nds, y_nds, z_ndes are unique
    x_nds = list(OrderedDict.fromkeys(x_nds))
    y_nds = list(OrderedDict.fromkeys(y_nds))
    z_nds = list(OrderedDict.fromkeys(z_nds))
    z_nds = [node for node in z_nds
             if (node not in x_nds) and (node not in y_nds)]

    # Get the expected results
    expect_array = np.array([list(np.arange(data[time-n_times, node],
                                        data[time-n_times, node]+n_times))
                             for node, time in x_nds + y_nds + z_nds]).astype('float')
    expect_xyz = np.array([0 for _ in x_nds] +\
                          [1 for _ in y_nds] +\
                          [2 for _ in z_nds])
    # Apply the mask, which always blocks the latest time of the 0th node of the
    # masked variable, which removes the first n time slices in the returned
    # array
    expect_array = expect_array[:, n_rows_masked:]
    if missing_vals:
        missing_anywhere_base = np.array(np.where(np.any(expect_array==missing_flag, axis=0))[0])
        missing_anywhere = list(missing_anywhere_base)
        # for tau in range(1, max_lag+1):
        #     missing_anywhere += list(np.array(missing_anywhere_base) + tau)
        expect_array = np.delete(expect_array, missing_anywhere, axis=1)

    # Test the results
    np.testing.assert_almost_equal(array, expect_array)
    np.testing.assert_almost_equal(xyz, expect_xyz)
