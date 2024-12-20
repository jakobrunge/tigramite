"""
Tests for independence_tests.py.
"""
from __future__ import print_function
import numpy as np
import pytest
from scipy import stats

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmiknn_mixed import CMIknnMixed
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI

import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

from test_pcmci_calculations import (a_chain, mixed_confounder,
                                     gen_data_frame, 
                                     gen_chain_data_frame_mixed, 
                                     gen_confounder_data_frame_mixed)

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale."""
    return -0.5 * np.log(1. - par_corr**2)

# INDEPENDENCE TEST DATA GENERATION ############################################
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

def generate_parent_dependent_stds(Z, T):
    stds = np.ones(T) + 5*(1 + Z[:T])*(1+Z[:T]>0)
    return stds

def generate_time_dependent_stds(Z, T):
    stds = np.array([1 + 0.018*t for t in range(T)])
    return stds
def gen_heteroskedastic_data_sample(seed, corr_val, T, dependence_type):
    random_state = np.random.RandomState(seed)
    # Generate the data
    dim = 3
    array = np.zeros((T, dim))
    Z = random_state.standard_normal(T + 1)
    E = random_state.standard_normal(T)
    array[:, 2] = Z[1:]
    stds_matrix = np.ones((T, dim))
    if dependence_type == "parent":
        stds = generate_parent_dependent_stds(Z, T)
    else:
        stds = generate_time_dependent_stds(Z, T)

    stds_matrix[:, 0] = stds
    stds_matrix[:, 1] = stds
    noise_X = random_state.normal(0, stds, T)
    noise_Y = random_state.normal(0, stds, T)
    array[:, 0] = corr_val[0] * Z[:T] + corr_val[2] * E + noise_X
    array[:, 1] = corr_val[1] * Z[:T] + corr_val[2] * E + noise_Y
    # Get the correlation coefficient
    val = np.corrcoef(array)[0, 1]
    # Get a value from the array
    xyz = np.array([0, 1])
    # Return the array, the value, and the xyz array
    return array, val, corr_val[2], xyz, dim, T, stds_matrix

# INDEPENDENCE TEST COMMON TESTS ###############################################
def check_get_array(ind_test, sample):
    # Get the data sample values
    dataframe, true_parents = sample
    # Set the dataframe of the test object
    ind_test.set_dataframe(dataframe)
    # Generate some nodes
    y_nds = [(0, 0)]
    x_nds = true_parents[0]
    z_nds = true_parents[1]
    tau_max = 3
    # Get the array using the wrapper function
    a_array, a_xyz, a_xyz_nodes, _ = \
            ind_test._get_array(x_nds, y_nds, z_nds, tau_max)
    # Get the array directly from the dataframe
    b_array, b_xyz, b_xyz_nodes, _ = \
            dataframe.construct_array(x_nds, y_nds, z_nds,
                                      tau_max=tau_max,
                                      mask_type=ind_test.mask_type,
                                      return_cleaned_xyz=True,
                                      do_checks=False,
                                      verbosity=ind_test.verbosity)
    # Check the values are the same
    np.testing.assert_allclose(a_array, b_array)
    np.testing.assert_allclose(a_xyz, b_xyz)
    np.testing.assert_allclose(a_xyz_nodes, b_xyz_nodes)

def check_run_test(ind_test, sample):
    # Get the data sample values
    dataframe, true_parents = sample
    # Set the dataframe of the test object
    ind_test.set_dataframe(dataframe)
    # Generate some nodes
    y_nds = [(0, 0)]
    x_nds = true_parents[0]
    z_nds = true_parents[1]
    tau_max = 3
    alpha_or_thres = 0.1
    # Run the test
    val, pval, dependent = ind_test.run_test(X=x_nds, Y=y_nds, Z=z_nds, 
        tau_max=tau_max, alpha_or_thres=alpha_or_thres)

    # Get the array the test is running on
    array, xyz, _, data_type = ind_test._get_array(x_nds, y_nds, z_nds, tau_max)
    dim, T = array.shape
    # Get the correct dependence measure
    val_expt = ind_test.get_dependence_measure(array, xyz, data_type=data_type)
    pval_expt = ind_test._get_p_value(val, array, xyz, T, dim, data_type=data_type)
    if ind_test.significance == 'fixed_thres':
        dependent = val_expt >= alpha_or_thres
        pval_expt = 0. if dependent else 1.
    # Check the values are close
    np.testing.assert_allclose(np.array(val), np.array(val_expt), atol=1e-2)
    np.testing.assert_allclose(np.array(pval), np.array(pval_expt), atol=1e-2)

def check_get_measure(ind_test, sample):
    # Get the data sample values
    dataframe, true_parents = sample
    # Set the dataframe of the test object
    ind_test.set_dataframe(dataframe)
    # Generate some nodes
    y_nds = [(0, 0)]
    x_nds = true_parents[0]
    z_nds = true_parents[1]
    tau_max = 3
    # Run the test
    val = ind_test.get_measure(x_nds, y_nds, z_nds, tau_max)
    # Get the array the test is running on
    array, xyz, _, data_type = ind_test._get_array(x_nds, y_nds, z_nds, tau_max)
    # Get the correct dependence measure
    val_expt = ind_test.get_dependence_measure(array, xyz, data_type=data_type)
    # Check the values are close
    np.testing.assert_allclose(np.array(val), np.array(val_expt), atol=1e-2)

def check_get_confidence(ind_test, sample):
    """
    Compares results of current confidence interval evaluation method against
    bootstrapped confidence interval method.  Implicitly tests
    get_bootstrap_confidence as well.

    NOTE: this is a computationally expensive test
    """
    # Get the data sample values
    dataframe, true_parents = sample
    # Set the dataframe of the test object
    ind_test.set_dataframe(dataframe)
    # Generate some nodes
    y_nds = [(0, 0)]
    x_nds = true_parents[0]
    z_nds = true_parents[1]
    tau_max = 3
    # Get the confidence interval
    conf_a = ind_test.get_confidence(x_nds, y_nds, z_nds, tau_max)
    # Get the array the test is running on
    array, xyz, _, _ = ind_test._get_array(x_nds, y_nds, z_nds, tau_max)
    # Test current confidence interval against bootstrapped interval
    conf_b = ind_test.get_bootstrap_confidence(
        array,
        xyz,
        dependence_measure=ind_test.get_dependence_measure,
        conf_samples=ind_test.conf_samples,
        conf_blocklength=ind_test.conf_blocklength,
        conf_lev=ind_test.conf_lev)
    # Check the values are close
    np.testing.assert_allclose(np.array(conf_a), np.array(conf_b), atol=1e-2)

def check_parcorr_wls_on_heteroskedastic_data(ind_test, sample):
    # Get the data sample values
    array, _, corr_val, xyz, dim, T, true_stds = sample
    array = array[:, :-1].transpose()
    ind_test.data = array
    # compare to calculations on pre-transformed version of the data (which is now homoskedastic)
    ind_test.stds = np.ones((dim, T))
    homoskedastic_array = np.multiply(array, 1/true_stds[:,:-1].transpose())
    homoskedastic_val = ind_test.get_dependence_measure(homoskedastic_array, xyz)
    # prepare the weights
    ind_test.stds = true_stds.transpose()
    # Get the estimated value
    val_est = ind_test.get_dependence_measure(array, xyz)
    # Compare to value obtained for homoskedastic version of the data
    np.testing.assert_allclose(np.array(homoskedastic_val), np.array(val_est), atol=0.03)


def check_std_approximation(ind_test, sample, xlag, ylag):
    # Get the data sample values
    data, _, corr_val, xyz, dim, T, true_stds = sample
    data = data.transpose()
    ind_test.data = data
    X = [(0, xlag)]
    Y = [(1, ylag)]
    tau_max = 3
    # shift the true weights to align with the lagged versions of X and Y
    min_lag = min(xlag, ylag)
    true_stds_shifted = np.zeros((T + min_lag, 2))
    true_stds_shifted[:, 0] = true_stds[np.abs(min_lag)+xlag:xlag or None, 0]
    true_stds_shifted[:, 1] = true_stds[np.abs(min_lag)+ylag:ylag or None, 1]
    # also shift the data
    array = np.zeros((3, T + min_lag))
    array[0, :] = data[0, np.abs(min_lag) + xlag:xlag or None]
    array[1, :] = data[1, np.abs(min_lag) + ylag:ylag or None]
    array[2, :] = data[2, np.abs(min_lag):]
    # Get the estimated value
    val_est = ind_test._get_std_estimation(array, X, Y)
    # Compare to the true value
    np.testing.assert_allclose(true_stds_shifted[2 * tau_max:-2 * tau_max, :].transpose(),
                               np.array(val_est[:, 2 * tau_max:-2 * tau_max]), rtol=2.)

# PARTIAL CORRELATION TESTING ##################################################
@pytest.fixture(params=[
    # Generate par_corr test instances
    # sig,            recycle, confidence
    ('analytic', True, 'analytic'),
    ('analytic', False, 'analytic'),
    ('analytic', False, 'bootstrap'),
    ('shuffle_test', False, 'analytic'),
    ('fixed_thres', False, 'analytic'),
    ])
def par_corr(request):
    # Unpack the parameters
    sig, recycle, conf = request.param
    # Generate the par_corr independence test
    return ParCorr(mask_type=None,
                   significance=sig,
                   sig_samples=10000,
                   sig_blocklength=3,
                   confidence=conf,
                   conf_lev=0.9,
                   conf_samples=10000,
                   conf_blocklength=1,
                   recycle_residuals=recycle,
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

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.1, 0.9),           1000, 2),
    (a_chain(0.5, 0.6),           1000, 11),
    (a_chain(0.5, 0.6, length=5), 1000, 42)])
def data_frame_a(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)

def test_get_array_parcorr(par_corr, data_frame_a):
    # Check the get_array function
    check_get_array(par_corr, data_frame_a)

def test_run_test_parcorr(par_corr, data_frame_a):
    # Check the run_test function
    check_run_test(par_corr, data_frame_a)

def test_get_measure_parcorr(par_corr, data_frame_a):
    # Check the get_measure function
    check_get_measure(par_corr, data_frame_a)

def test_get_confidence_parcorr(par_corr, data_frame_a):
    # Check the get_confidence function
    check_get_confidence(par_corr, data_frame_a)

# TODO test null distribution
def test_shuffle_sig_parcorr(par_corr, data_sample_a):
    # Get the data sample values
    array, val, _, xyz, dim, T = data_sample_a
    # Get the analytic significance
    pval_a = par_corr.get_analytic_significance(value=val, T=T, dim=dim, xyz=xyz)
    # Get the shuffle significance
    pval_s = par_corr.get_shuffle_significance(array, xyz, val)
    # Adjust p-value for two-sided measures
    np.testing.assert_allclose(np.array(pval_a), np.array(pval_s), atol=0.01)

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

def test_parcorr(par_corr, data_sample_a):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_a
    # Generate the full array
    dim = 5
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5 * array[2:].sum(axis=0)
    array[1] += 0.7 * array[2:].sum(axis=0)
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = par_corr.get_dependence_measure(array, xyz)
    # Compare to the true value
    np.testing.assert_allclose(np.array(corr_val), np.array(val_est), atol=0.02)

# WLS PARTIAL CORRELATION TESTING ##################################################
@pytest.fixture(params=[
    # Generate par_corr_wls test instances
    # test ParCorr-WLS on homoskedastic data
    # sig,      recycle, confidence, std_matrix, expert_knowledge
    # ('analytic', True, 'bootstrap', None, "homoskedasticity", False),
    # ('analytic', True, 'bootstrap', np.ones((1000, 2)), "homoskedasticity", False),
    ('analytic', False, 'bootstrap', None, "homoskedasticity", False),
    ('shuffle_test', False, 'bootstrap', None, "homoskedasticity", False),
    ('fixed_thres', False, 'bootstrap', None, "homoskedasticity", False),
    # ('analytic', True, 'bootstrap', None, "homoskedasticity", True),
])

def par_corr_wls(request):
    # Unpack the parameters
    sig, recycle, conf, std_matrix, expert_knowledge, robustify = request.param
    # Generate the par_corr_wls independence test
    return ParCorrWLS(gt_std_matrix=std_matrix,
                      expert_knowledge=expert_knowledge,
                      window_size=100,
                      mask_type=None,
                      significance=sig,
                      sig_samples=10000,
                      sig_blocklength=3,
                      confidence=conf,
                      conf_lev=0.9,
                      conf_samples=10000,
                      conf_blocklength=1,
                      verbosity=0,
                      robustify=robustify)

@pytest.fixture(params=[
    # Generate par_corr test instances
    # basically test ParCorr-WLS on homoskedastic data
    # sig,      recycle, confidence, expert_knowledge, robustify
    ('analytic', False, 'bootstrap', None, {0: [(2, -1)], 1: [(2, -1)]}, False),
    ('analytic', False, 'bootstrap', None, {0: [(2, -1)], 1: [(2, -1)]}, True),
])

def par_corr_wls_expert(request):
    # Unpack the parameters
    sig, recycle, conf, std_matrix, expert_knowledge, robustify = request.param
    # Generate the par_corr independence test
    return ParCorrWLS(expert_knowledge=expert_knowledge,
                      window_size=50,
                      mask_type=None,
                      significance=sig,
                      sig_samples=10000,
                      sig_blocklength=3,
                      confidence=conf,
                      conf_lev=0.9,
                      conf_samples=10000,
                      conf_blocklength=1,
                      # recycle_residuals=recycle,
                      verbosity=0,
                      robustify=robustify)

@pytest.fixture(params=[
    # Generate par_corr test instances
    # basically test ParCorr-WLS on homoskedastic data
    # sig,      recycle, confidence, expert_knowledge, robustify
    ('analytic', True, 'bootstrap', None, {0: ["time-dependent heteroskedasticity"],
                                           1: ["time-dependent heteroskedasticity"]}, False),
    ('analytic', True, 'bootstrap', None, {0: ["time-dependent heteroskedasticity"],
                                           1: ["time-dependent heteroskedasticity"]}, True),
])

def par_corr_wls_expert_time(request):
    # Unpack the parameters
    sig, recycle, conf, std_matrix, expert_knowledge, robustify = request.param
    # Generate the par_corr independence test
    return ParCorrWLS(expert_knowledge=expert_knowledge,
                      window_size=50,
                      mask_type=None,
                      significance=sig,
                      sig_samples=10000,
                      sig_blocklength=3,
                      confidence=conf,
                      conf_lev=0.9,
                      conf_samples=10000,
                      conf_blocklength=1,
                      # recycle_residuals=recycle,
                      verbosity=0,
                      robustify=robustify)

@pytest.fixture(params=[
    # Generate the sample to be used for confidence interval comparison
    # seed, corr_val, T
    (5, 0.3, 1000),  # Default
    (6, 0.3, 1000),  # New Seed
    (1, 0.9, 1000)])  # Strong Correlation
def data_sample_a2(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_data_sample(seed, corr_val, T)


@pytest.fixture(params=[
    # Generate the sample to be used for variance estimation
    # seed, corr_val, T
    (5, [0.3, 0.3, 0.5], 1000),  # Default
    (6, [0.3, 0.3, 0], 1000),  # New Seed
    (1, [0.9, 0.9, 0], 1000),  # Strong Correlation
    (5, [0.3, 0.3, 0.9], 1000),   # Strong Correlation between X and Y
])
def data_sample_hs_parent(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_heteroskedastic_data_sample(seed, corr_val, T, "parent")

@pytest.fixture(params=[
    # Generate the sample to be used for variance estimation
    # seed, corr_val, T
    (5, [0.3, 0.3, 0], 1000),  # Default
    (6, [0.3, 0.6, 0], 1000),  # New Seed
    (1, [0.9, 0.9, 0], 1000)   # Strong Correlation
])
def data_sample_hs_time(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_heteroskedastic_data_sample(seed, corr_val, T, "time")

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.1, 0.9), 1000, 2),
    (a_chain(0.5, 0.6), 1000, 11),
    (a_chain(0.5, 0.6, length=5), 1000, 42)])
def data_frame_a2(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)

def test_get_array_parcorr_wls(par_corr_wls, data_frame_a2):
    # Check the get_array function
    check_get_array(par_corr_wls, data_frame_a2)


def test_run_test_parcorr_wls(par_corr_wls, data_frame_a2):
    # Check the run_test function
    check_run_test(par_corr_wls, data_frame_a2)


def test_get_measure_parcorr_wls(par_corr_wls, data_frame_a2):
    # Check the get_measure function
    check_get_measure(par_corr_wls, data_frame_a2)

def test_get_confidence_parcorr_wls(par_corr_wls, data_frame_a2):
    # Check the get_confidence function
    check_get_confidence(par_corr_wls, data_frame_a2)


def test_shuffle_sig_parcorr_wls(par_corr_wls, data_sample_a2):
    # Get the data sample values
    array, val, _, xyz, dim, T = data_sample_a2
    # Get the analytic significance
    pval_a = par_corr_wls.get_analytic_significance(value=val, T=T, dim=dim, xyz=xyz)
    # Get the shuffle significance
    pval_s = par_corr_wls.get_shuffle_significance(array, xyz, val)
    # Adjust p-value for two-sided measures
    np.testing.assert_allclose(np.array(pval_a), np.array(pval_s), atol=0.01)


#@pytest.mark.parametrize("seed", [5, 29, 135, 170, 174, 284, 342, 363, 425])
@pytest.mark.parametrize("seed", [5, 29, 425])
def test_parcorr_wls_residuals(par_corr_wls, seed):
    # Set the random seed
    np.random.seed(seed)
    # Set the target value and the true residuals
    target_var = 0
    true_res = np.random.randn(4, 1000)
    # Copy the true residuals to a new array
    array = np.copy(true_res)
    # Manipulate the array
    array[0] += 0.5 * array[2:].sum(axis=0)
    # Estimate the residuals
    est_res = par_corr_wls._get_single_residuals(array, target_var,
                                                 standardize=False,
                                                 return_means=False)
    np.testing.assert_allclose(est_res, true_res[0], rtol=1e-5, atol=0.02)


def test_parcorr_wls(par_corr_wls, data_sample_a2):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_a2
    # Generate the full array
    dim = 5
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5 * array[2:].sum(axis=0)
    array[1] += 0.7 * array[2:].sum(axis=0)
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = par_corr_wls.get_dependence_measure(array, xyz)
    # Compare to the true value
    np.testing.assert_allclose(np.array(corr_val), np.array(val_est), atol=0.03)

def test_parcorr_wls_on_heteroskedastic_data(par_corr_wls_expert, data_sample_hs_parent):
    # Get the data sample values
    check_parcorr_wls_on_heteroskedastic_data(par_corr_wls_expert, data_sample_hs_parent)

def test_parcorr_wls_on_heteroskedastic_data(par_corr_wls_expert_time, data_sample_hs_time):
    # Get the data sample values
    check_parcorr_wls_on_heteroskedastic_data(par_corr_wls_expert_time, data_sample_hs_time)

@pytest.mark.parametrize("x_lag", [0, -1])
@pytest.mark.parametrize("y_lag", [0, -1])
def test_std_approximation(par_corr_wls_expert, data_sample_hs_parent, x_lag, y_lag):
    # Get the data sample values
    check_std_approximation(par_corr_wls_expert, data_sample_hs_parent, x_lag, y_lag)

@pytest.mark.parametrize("x_lag", [0, -1])
@pytest.mark.parametrize("y_lag", [0, -1])
def test_std_approximation(par_corr_wls_expert_time, data_sample_hs_time, x_lag, y_lag):
    # Get the data sample values
    check_std_approximation(par_corr_wls_expert_time, data_sample_hs_time, x_lag, y_lag)

# GPDC TESTING #################################################################
@pytest.fixture()
def gpdc(request):
    return GPDC(mask_type=None,
                significance='analytic',
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
    (5,    0.3,      200),  # Default
    (6,    0.3,      200),  # New Seed
    (1,    0.9,      200)]) # Strong Correlation
def data_sample_b(request):
    # Unpack the parameters
    seed, corr_val, T = request.param
    # Return the data sample
    return gen_data_sample(seed, corr_val, T)

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.1, 0.9),           250, 2),
    (a_chain(0.5, 0.6),           250, 11),
    (a_chain(0.5, 0.6, length=5), 250, 42)])
def data_frame_b(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)

def test_get_array_gpdc(gpdc, data_frame_b):
    # Check the get_array function
    check_get_array(gpdc, data_frame_b)

def test_get_measure_gpdc(gpdc, data_frame_b):
    # Check the get_measure function
    check_get_measure(gpdc, data_frame_b)

def test_get_confidence_gpdc(gpdc, data_frame_b):
    # Skip if just checking boostrap vs. bootstrap
    if not gpdc.confidence == 'bootstrap':
        # Check the get_confidence function
        check_get_confidence(gpdc, data_frame_b)

@pytest.mark.parametrize("seed", list(range(10)))
def test_gpdc_residuals(gpdc, seed):
    # Set the random seed
    np.random.seed(seed)
    c_val = .3
    T = 1000
    # Define the function to check against
    def func(x_arr, c_val=1.):
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

def test_shuffle_sig_gpdc(gpdc, data_sample_b):
    # Get the data sample
    array, _, _, xyz, dim, T = data_sample_b
    # Trim the data sample down, time goes as T^2
    T = int(T / 4.)
    array = array[:, :T]
    # Get the value of the dependence measurement
    val = gpdc.get_dependence_measure(array, xyz)
    pval_a = gpdc.get_analytic_significance(value=val, T=T, dim=dim, xyz=xyz)
    pval_s = gpdc.get_shuffle_significance(array, xyz, val)
    np.testing.assert_allclose(np.array(pval_a), np.array(pval_s), atol=0.05)

def test_trafo2uniform(gpdc, data_sample_a):
    # Get the data sample
    array, _, _, _, _, T = data_sample_a
    # Make the transformation
    uniform = gpdc._trafo2uniform(array)
    # Set the number of bins
    bins = 10
    for i in range(array.shape[0]):
        hist, _ = np.histogram(uniform[i], bins=bins, density=True)
        np.testing.assert_allclose(np.ones(bins) / float(bins),
                                   hist / float(bins),
                                   atol=0.01)

# GPDCtorch TESTING #################################################################
@pytest.fixture()
def gpdc_torch(request):
    return GPDCtorch(mask_type=None,
                     significance='analytic',
                     sig_samples=1000,
                     sig_blocklength=1,
                     confidence='bootstrap',
                     conf_lev=0.9,
                     conf_samples=100,
                     conf_blocklength=None,
                     recycle_residuals=False,
                     verbosity=0)

# RE-USING SETUP FROM GPDC

# @pytest.fixture(params=[
#     # Generate the sample to be used for confidence interval comparison
#     #seed, corr_val, T
#     (5,    0.3,      200),  # Default
#     (6,    0.3,      200),  # New Seed
#     (1,    0.9,      200)]) # Strong Correlation
# def data_sample_b(request):
#     # Unpack the parameters
#     seed, corr_val, T = request.param
#     # Return the data sample
#     return gen_data_sample(seed, corr_val, T)

# @pytest.fixture(params=[
#     # Generate a test data sample
#     # Parameterize the sample by setting the autocorrelation value, coefficient
#     # value, total time length, and random seed to different numbers
#     # links_coeffs,               time, seed_val
#     (a_chain(0.1, 0.9),           250, 2),
#     (a_chain(0.5, 0.6),           250, 11),
#     (a_chain(0.5, 0.6, length=5), 250, 42)])
# def data_frame_b(request):
#     # Set the parameters
#     links_coeffs, time, seed_val = request.param
#     # Generate the dataframe
#     return gen_data_frame(links_coeffs, time, seed_val)

def test_get_array_gpdc_torch(gpdc_torch, data_frame_b):
    # Check the get_array function
    check_get_array(gpdc_torch, data_frame_b)

def test_get_measure_gpdc_torch(gpdc_torch, data_frame_b):
    # Check the get_measure function
    check_get_measure(gpdc_torch, data_frame_b)

def test_get_confidence_gpdc_torch(gpdc_torch, data_frame_b):
    # Skip if just checking boostrap vs. bootstrap
    if not gpdc_torch.confidence == 'bootstrap':
        # Check the get_confidence function
        check_get_confidence(gpdc_torch, data_frame_b)

@pytest.mark.parametrize("seed", list(range(10)))
def test_gpdc_torch_residuals(gpdc_torch, seed):
    # Set the random seed
    np.random.seed(seed)
    c_val = .3
    T = 1000
    # Define the function to check against
    def func(x_arr, c_val=1.):
        return c_val * x_arr * (1. - 4. * np.exp(-x_arr * x_arr / 2.))

    # Generate the array
    array = np.random.randn(3, T)
    # Manipulate the array
    array[1] += func(array[2], c_val)
    # Set the target value and the target results
    target_var = 1
    target_res = np.copy(array[2])
    # Calculate the residuals
    (_, pred) = gpdc_torch._get_single_residuals(array, target_var,
                                           standardize=False,
                                           return_means=True)
    # Testing that the fit matches in the centre
    cntr = np.where(np.abs(target_res) < .7)[0]
    np.testing.assert_allclose(pred[cntr],
                               func(target_res[cntr], c_val),
                               atol=0.2, rtol=1e-01)

def test_shuffle_sig_gpdc_torch(gpdc_torch, data_sample_b):
    # Get the data sample
    array, _, _, xyz, dim, T = data_sample_b
    # Trim the data sample down, time goes as T^2
    T = int(T / 4.)
    array = array[:, :T]
    # Get the value of the dependence measurement
    val = gpdc_torch.get_dependence_measure(array, xyz)
    pval_a = gpdc_torch.get_analytic_significance(value=val, T=T, dim=dim, xyz=xyz)
    pval_s = gpdc_torch.get_shuffle_significance(array, xyz, val)
    np.testing.assert_allclose(np.array(pval_a), np.array(pval_s), atol=0.05)

def test_trafo2uniform_torch(gpdc_torch, data_sample_a):
    # Get the data sample
    array, _, _, _, _, T = data_sample_a
    # Make the transformation
    uniform = gpdc_torch._trafo2uniform(array)
    # Set the number of bins
    bins = 10
    for i in range(array.shape[0]):
        hist, _ = np.histogram(uniform[i], bins=bins, density=True)
        np.testing.assert_allclose(np.ones(bins) / float(bins),
                                   hist / float(bins),
                                   atol=0.01)

# CMIknn TESTING ###############################################################
@pytest.fixture()
def cmi_knn(request):
    return CMIknn(mask_type=None,
                  significance='shuffle_test',
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

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.1, 0.9),           100, 2),
    (a_chain(0.5, 0.6),           100, 11),
    (a_chain(0.5, 0.6, length=5), 100, 42)])
def data_frame_c(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)

def test_get_array_cmi_knn(cmi_knn, data_frame_c):
    # Check the get_array function
    check_get_array(cmi_knn, data_frame_c)

def test_run_test_cmi_knn(cmi_knn, data_frame_c):
    # Check the run_test function
    check_run_test(cmi_knn, data_frame_c)

def test_get_measure_cmi_knn(cmi_knn, data_frame_c):
    # Check the get_measure function
    check_get_measure(cmi_knn, data_frame_c)

def test_get_confidence_cmi_knn(cmi_knn, data_frame_c):
    # Skip if just checking boostrap vs. bootstrap
    if not cmi_knn.confidence == 'bootstrap':
        # Check the get_confidence function
        check_get_confidence(cmi_knn, data_frame_c)

def test_cmi_knn(cmi_knn, data_sample_c):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_c
    # Generate the full array
    dim = 5
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5 * array[2:].sum(axis=0)
    array[1] += 0.7 * array[2:].sum(axis=0)
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = cmi_knn.get_dependence_measure(array, xyz)
    np.testing.assert_allclose(np.array(_par_corr_to_cmi(corr_val)),
                               np.array(val_est),
                               atol=0.02)


# CMIknnMixed TESTING ##############################################################

# Here we only test the main functionality of CMIknnMixed, as the rest of the 
# functions are the same as for the continuous CMIknn test

@pytest.fixture()
def cmi_knn_mixed(request):
    return CMIknnMixed(mask_type=None,
                       significance='shuffle_test',
                       sig_samples=20,
                       sig_blocklength=3,
                       knn=0.3,
                       verbosity=2)

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (mixed_confounder(0.1, 0.9), 1000, 2),
    (mixed_confounder(0.5, 0.6), 1000, 11),
    (mixed_confounder(0.5, 0.6), 1000, 42)])
def data_frame_conf_mixed(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_confounder_data_frame_mixed(links_coeffs, time, seed_val)

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.1, 0.9), 10, 2),
    (a_chain(0.5, 0.6), 10, 11),
    (a_chain(0.5, 0.6, length=5), 10, 42)])
def data_frame_chain_mixed(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_chain_data_frame_mixed(links_coeffs, time, seed_val)

def test_get_measure_cmi_knn_mixed_chain(cmi_knn_mixed, data_frame_chain_mixed):
    # Check the get_measure function
    check_get_measure(cmi_knn_mixed, data_frame_chain_mixed)

def test_get_measure_cmi_knn_mixed_confounder(cmi_knn_mixed, data_frame_conf_mixed):
    # Check the get_measure function, aditionally check the type matrix 
    check_get_measure(cmi_knn_mixed, data_frame_conf_mixed)

def test_run_test_cmi_knn_mixed(cmi_knn_mixed, data_frame_chain_mixed):
    # Check the run_test function
    check_run_test(cmi_knn_mixed, data_frame_chain_mixed)


# CMIsymb TESTING ##############################################################
@pytest.fixture()
def cmi_symb(request):
    return CMIsymb(mask_type=None,
                   significance='shuffle_test',
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

@pytest.fixture(params=[
    # Generate a test data sample
    # Parameterize the sample by setting the autocorrelation value, coefficient
    # value, total time length, and random seed to different numbers
    # links_coeffs,               time, seed_val
    (a_chain(0.1, 0.9),           100, 2),
    (a_chain(0.5, 0.6),           100, 11),
    (a_chain(0.5, 0.6, length=5), 100, 42)])
def data_frame_d(request):
    # Set the parameters
    links_coeffs, time, seed_val = request.param
    # Generate the dataframe
    return gen_data_frame(links_coeffs, time, seed_val)

# TODO does not work
# def test_run_test_cmi_symb(cmi_symb, data_frame_d):
#    # Make the data frame integer values
#    df, parents = data_frame_d
#    df.values = (df.values * 1000).astype(int)
#    # Check the run_test function
#    check_run_test(cmi_symb, (df, parents))

# TODO does not work
# def test_run_test_cmi_symb(cmi_symb, data_frame_d):
#    # Make the data frame integer values
#    df, parents = data_frame_d
#    df.values = (df.values * 1000).astype(int)
#    # Check the run_test function
#    check_get_measure(cmi_symb, (df, parents))

def test_cmi_symb(cmi_symb, data_sample_d):
    # Get the data sample values
    small_array, _, corr_val, xyz, dim, T = data_sample_d
    # Generate the full array
    dim = 3
    array = np.random.randn(dim, T)
    array[:2, :] = small_array
    # Generate some confounding
    array[0] += 0.5 * array[2:].sum(axis=0)
    array[1] += 0.7 * array[2:].sum(axis=0)
    # Transform to symbolic data
    array = pp.quantile_bin_array(array.T, bins=16).T
    # Reset the dimension
    xyz = np.array([0, 1, 2])  #, 2, 2])
    # Get the estimated value
    val_est = cmi_symb.get_dependence_measure(array, xyz)
    np.testing.assert_allclose(np.array(_par_corr_to_cmi(corr_val)),
                               np.array(val_est),
                               atol=0.02)

# OTHER TESTS ##################################################################
@pytest.mark.parametrize("mask_type,expected", [
    ('x', True),    # single item in acceptable list
    ('y', True),    # single item in acceptable list
    ('z', True),    # single item in acceptable list
    ('xy', True),   # multiple items in acceptable list
    ('xz', True),   # multiple items in acceptable list
    ('yz', True),   # multiple items in acceptable list
    ('yx', True),   # multiple items in acceptable list, swapped order
    ('xyz', True),  # all items in acceptable list
    ('xzy', True),  # all items in acceptable list, swapped order
    ('a', False),   # single unacceptable item
    ('ax', False),  # single unacceptable item and acceptable item
    ('ab', False)]) # multiple unacceptable items
def test_check_mask(par_corr, mask_type, expected):
    # Set the mask type
    par_corr.mask_type = mask_type
    # Test the good parameter set
    if expected:
        try:
            par_corr._check_mask_type()
        # Ensure no exception is raised
        except:
            pytest.fail("Acceptable mask type "+mask_type+\
                        " is incorrectly throwing an error")
    # Test the bad parameter set
    else:
        err_msg = "Unacceptable mask type "+mask_type+\
                  " is incorrectly NOT throwing an error"
        with pytest.raises(ValueError):
            par_corr._check_mask_type()
            pytest.fail(err_msg)
