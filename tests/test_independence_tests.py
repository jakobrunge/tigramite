"""
Tests for independence_tests.py.
"""
from __future__ import print_function
from collections import OrderedDict
import numpy as np
import pytest

from tigramite.independence_tests import ParCorr, GPDC, CMIsymb, CMIknn #, GPACE
import tigramite.data_processing as pp

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale."""
    return -0.5 * np.log(1. - par_corr**2)


# INDEPENDENCE TEST GENERATION #################################################
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

def gen_data_sample(seed, corr_val, T):
    # Set the random seed
    np.random.seed(seed)
    # Define a symmetric covariance matrix
    cov = np.array([[1., corr_val],
                    [corr_val, 1.]])
    # Generate some random data using the above covariance relation
    dim = cov.shape[0]
    array = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov, size=T).T
    # Get the correlation coefficient
    val = np.corrcoef(array)[0, 1]
    # Get a value from the array
    xyz = np.array([0, 1])
    # Return the array, the value, and the xyz array
    return array, val, corr_val, xyz, dim, T

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
    data = np.arange(1000).reshape(10, 100).T
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
        n_rows_masked += max_lag + 1

    # Construct the array
    data_f = pp.DataFrame(data, data_mask, missing_flag)
    array, xyz = data_f.construct_array(x_nds, y_nds, z_nds,
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
    expect_array = np.array([list(range(data[time-n_times, node],
                                        data[time-n_times, node]+n_times))
                             for node, time in x_nds + y_nds + z_nds])
    expect_xyz = np.array([0 for _ in x_nds] +\
                          [1 for _ in y_nds] +\
                          [2 for _ in z_nds])
    # Apply the mask, which always blocks the latest time of the 0th node of the
    # masked variable, which removes the first n time slices in the returned
    # array
    expect_array = expect_array[:, n_rows_masked:]
    # Test the results
    np.testing.assert_almost_equal(array, expect_array)
    np.testing.assert_almost_equal(xyz, expect_xyz)

# PARTIAL CORRELATION TESTING ##################################################
@pytest.fixture()
def par_corr(request):
    # Generate the par_corr independence test
    return ParCorr(mask_type=None,
                   significance='analytic',
                   fixed_thres=0.1,
                   sig_samples=10000,
                   sig_blocklength=3,
                   confidence='analytic',
                   conf_lev=0.9,
                   conf_samples=10000,
                   conf_blocklength=1,
                   recycle_residuals=False,
                   verbosity=0)

@pytest.fixture(params=[
    # Generate the sample to be used for confidance interval comparison
    #seed, corr_val, T
    (5,    0.3,      1000),  # Default
    (6,    0.3,      1000),  # New Seed
    (1,    0.9,      1000)]) # Strong Correlation
def data_sample_a(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_data_sample(seed, corr_val, T)

def test_bootstrap_conf_parcorr(par_corr, data_sample_a):
    # Get the data sample values
    array, val, _, xyz, dim, T = data_sample_a
    # Get the analytic confidence interval
    conf_a = par_corr.get_analytic_confidence(df=T-dim,
                                              value=val,
                                              conf_lev=par_corr.conf_lev)
    # Bootstrap the confidence interval
    conf_b = par_corr.get_bootstrap_confidence(
        array,
        xyz,
        dependence_measure=par_corr.get_dependence_measure,
        conf_samples=par_corr.conf_samples,
        conf_blocklength=par_corr.conf_blocklength,
        conf_lev=par_corr.conf_lev)
    # Ensure the two intervals are the same
    np.testing.assert_allclose(np.array(conf_a), np.array(conf_b), atol=0.01)

# TODO should val here be get_dependence_measure?
# TODO test null distribution
def test_shuffle_sig_parcorr(par_corr, data_sample_a):
    # Get the data sample values
    array, val, _, xyz, dim, T = data_sample_a
    # Get the analytic significance
    pval_a = par_corr.get_analytic_significance(value=val, T=T, dim=dim)
    # Get the shuffle significance
    pval_s = par_corr.get_shuffle_significance(array, xyz, val)
    # Adjust p-value for two-sided measures
    np.testing.assert_allclose(np.array(pval_a), np.array(pval_s), atol=0.01)

# TODO how does this test work?
# TODO test standardize, return_means as well
@pytest.mark.parametrize("seed", [5, 29, 135, 170, 174, 284, 342, 363, 425])
def test_parcorr_residuals(par_corr, seed):
    # Set the random seed
    np.random.seed(seed)
    # Set the target value and the true residuals
    target_var = 0
    true_res = np.random.randn(4, 1000)
    # Copy the true residuals to a new array
    array = np.copy(true_res)
    # Manipulate the array
    array[0] += 0.5*array[2:].sum(axis=0)
    # Estimate the residuals
    est_res = par_corr._get_single_residuals(array, target_var,
                                             standardize=False,
                                             return_means=False)
    np.testing.assert_allclose(est_res, true_res[0], rtol=1e-5, atol=0.02)

# TODO how does this test work?
def test_parcorr(par_corr, data_sample_a):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_a
    # Generate the full array
    dim = 5
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5* array[2:].sum(axis=0)
    array[1] += 0.7* array[2:].sum(axis=0)
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = par_corr.get_dependence_measure(array, xyz)
    # Compare to the true value
    np.testing.assert_allclose(np.array(corr_val), np.array(val_est), atol=0.02)

# GPDC TESTING #################################################################
# TODO need test for GPDC get_dependence_measure
@pytest.fixture()
def gpdc(request):
    return GPDC(mask_type=None,
                significance='analytic',
                fixed_thres=0.1,
                sig_samples=1000,
                sig_blocklength=1,
                confidence='bootstrap',
                conf_lev=0.9,
                conf_samples=100,
                conf_blocklength=None,
                recycle_residuals=False,
                verbosity=0)

@pytest.fixture(params=[
    # Generate the sample to be used for confidence interval comparison
    #seed, corr_val, T
    (5,    0.3,      250),  # Default
    (6,    0.3,      250),  # New Seed
    (1,    0.9,      250)]) # Strong Correlation
def data_sample_b(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_data_sample(seed, corr_val, T)

@pytest.mark.parametrize("seed", list(range(10)))
def test_gpdc_residuals(gpdc, seed):
    # Set the random seed
    np.random.seed(seed)
    c_val = .3
    T = 1000
    # Define the function to check against
    def func(x_arr, c_val=1.):
        # TODO check x**0
        # return x * (1. - 4. * x**0 * np.exp(-x**2 / 2.))
        return c_val*x_arr*(1. - 4.*np.exp(-x_arr*x_arr/2.))
    # Generate the array
    array = np.random.randn(3, T)
    # Manipulate the array
    array[1] += func(array[2], c_val)
    # Set the target value and the target results
    target_var = 1
    target_res = np.copy(array[2])
    # Calculate the residuals
    (_, pred) = gpdc._get_single_residuals(array, target_var,
                                           standardize=False,
                                           return_means=True)
    # Testing that the fit matches in the centre
    cntr = np.where(np.abs(target_res) < .7)[0]
    np.testing.assert_allclose(pred[cntr],
                               func(target_res[cntr], c_val),
                               atol=0.2)

# TODO should val here be the val given by the data_sample fixture?
def test_shuffle_sig_gpdc(gpdc, data_sample_b):
    # Get the data sample
    array, _, _, xyz, dim, T = data_sample_b
    # Trim the data sample down, time goes as T^2
    T = int(T/4.)
    array = array[:, :T]
    # Get the value of the dependence measurement
    val = gpdc.get_dependence_measure(array, xyz)
    pval_a = gpdc.get_analytic_significance(value=val, T=T, dim=dim)
    pval_s = gpdc.get_shuffle_significance(array, xyz, val)
    np.testing.assert_allclose(np.array(pval_a), np.array(pval_s), atol=0.05)

# TODO how does this test work?
def test_trafo2uniform(gpdc, data_sample_a):
    # Get the data sample
    array, _, _, _, _, T = data_sample_a
    # Make the transformation
    uniform = gpdc._trafo2uniform(array)
    # Set the number of bins
    bins = 10
    for i in range(array.shape[0]):
        hist, _ = np.histogram(uniform[i], bins=bins, density=True)
        np.testing.assert_allclose(np.ones(bins)/float(bins),
                                   hist/float(bins),
                                   atol=0.01)

# CMIknn TESTING ###############################################################
@pytest.fixture()
def cmi_knn(request):
    return CMIknn(mask_type=None,
                  significance='shuffle_test',
                  fixed_thres=None,
                  sig_samples=10000,
                  sig_blocklength=3,
                  knn=10,
                  confidence='bootstrap',
                  conf_lev=0.9,
                  conf_samples=10000,
                  conf_blocklength=1,
                  verbosity=0)

@pytest.fixture(params=[
    # Generate the sample to be used for confidence interval comparison
    #seed, corr_val, T
    (5,    0.3,      10000),  # Default
    (6,    0.3,      10000),  # New Seed
    (1,    0.6,      10000)]) # Strong Correlation
def data_sample_c(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_data_sample(seed, corr_val, T)

# TODO how does this test work?
def test_cmi_knn(cmi_knn, data_sample_c):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_c
    # Generate the full array
    dim = 5
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5* array[2:].sum(axis=0)
    array[1] += 0.7* array[2:].sum(axis=0)
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = cmi_knn.get_dependence_measure(array, xyz)
    np.testing.assert_allclose(np.array(_par_corr_to_cmi(corr_val)),
                               np.array(val_est),
                               atol=0.02)

# CMIsymb TESTING ##############################################################
@pytest.fixture()
def cmi_symb(request):
    return CMIsymb(mask_type=None,
                   significance='shuffle_test',
                   fixed_thres=0.1,
                   sig_samples=10000,
                   sig_blocklength=3,
                   confidence='bootstrap',
                   conf_lev=0.9,
                   conf_samples=10000,
                   conf_blocklength=1,
                   verbosity=0)

@pytest.fixture(params=[
    # Generate the sample to be used for confidence interval comparison
    #seed, corr_val, T
    (5,    0.3,      100000),  # Default
    (6,    0.3,      100000),  # New Seed
    (7,    0.6,      100000)]) # Strong Correlation
def data_sample_d(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_data_sample(seed, corr_val, T)

def test_cmi_symb(cmi_symb, data_sample_d):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_d
    # Generate the full array
    dim = 3
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5* array[2:].sum(axis=0)
    array[1] += 0.7* array[2:].sum(axis=0)
    # Transform to symbolic data
    array = pp.quantile_bin_array(array.T, bins=16).T
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = cmi_symb.get_dependence_measure(array, xyz)
    np.testing.assert_allclose(np.array(_par_corr_to_cmi(corr_val)),
                               np.array(val_est),
                               atol=0.02)
