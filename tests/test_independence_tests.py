"""
Tests for independence_tests.py.
"""
from __future__ import print_function
from collections import OrderedDict
import numpy as np
import pytest

from tigramite.independence_tests import ParCorr, GPDC, CMIsymb, CMIknn, GPACE
import tigramite.data_processing as pp

from test_pcmci_calculations import a_chain, gen_data_frame

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale."""
    return -0.5 * np.log(1. - par_corr**2)

### CHECK LIST ###

# Concrete classes
   # ParCorr
   # GPDC
   # GPACE
   # CMIknn
   # CMIsymb
   # RCOT

# CondIntTest (for all concrete)
   # _check_mask_type
   # _get_array
   # run_test
   # get_measure
   # get_confidence
   # get_bootstrap_confidence
   # _get_acf
   # _get_block_length
   # _get_shuffle_dist
   # get_fixed_thresh_sig
   # tranfo2uniform
# GausProc
    # TEST _get_single_residuals
    # TEST generate_and_save_nulldists
    # TEST generate_nulldist
    # TEST _load_nulldist
# GPDC
    # TEST _get_max_corr
    # TEST get_shuffle_sig
    # TEST get_analytic_sig
    # TEST get_dependence_measure


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

# INDEPENDENCE TEST COMMON TESTS ###############################################
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
    # Run the test
    val, pval = ind_test.run_test(x_nds, y_nds, z_nds, tau_max)
    # Get the array the test is running on
    array, xyz, _ = ind_test._get_array(x_nds, y_nds, z_nds, tau_max)
    dim, T = array.shape
    # Get the correct dependence measure
    val_expt = ind_test.get_dependence_measure(array, xyz)
    pval_expt = ind_test.get_significance(val, array, xyz, T, dim)
    # Check the values are close
    np.testing.assert_allclose(np.array(val), np.array(val_expt), atol=1e-3)
    np.testing.assert_allclose(np.array(pval), np.array(pval_expt), atol=1e-3)

# PARTIAL CORRELATION TESTING ##################################################
@pytest.fixture(params=[
    # Generate par_corr test instances
    #sig,            recycle,
    ('analytic',     True),
    ('analytic',     False),
    ('shuffle_test', False),
    ('fixed_thres',  False)])
def par_corr(request):
    # Unpack the parameters
    sig, recycle = request.param
    # Generate the par_corr independence test
    return ParCorr(mask_type=None,
                   significance=sig,
                   fixed_thres=0.1,
                   sig_samples=10000,
                   sig_blocklength=3,
                   confidence='analytic',
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

   # get_measure
   # get_confidence
   # get_bootstrap_confidence
   # get_fixed_thresh_sig
   # tranfo2uniform
   # _check_mask_type
   # _get_array
   # _get_acf
   # _get_block_length
   # _get_shuffle_dist

def test_run_test_parcorr(par_corr, data_frame_a):
    # Check the run_test function
    check_run_test(par_corr, data_frame_a)

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
    array[0] += 0.5* array[2:].sum(axis=0)
    array[1] += 0.7* array[2:].sum(axis=0)
    # Reset the dimension
    xyz = np.array([0, 1, 2, 2, 2])
    # Get the estimated value
    val_est = par_corr.get_dependence_measure(array, xyz)
    # Compare to the true value
    np.testing.assert_allclose(np.array(corr_val), np.array(val_est), atol=0.02)

# GPDC TESTING #################################################################
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

def test_run_test_gpdc(gpdc, data_frame_b):
    # Check the run_test function
    check_run_test(gpdc, data_frame_b)

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
    T = int(T/4.)
    array = array[:, :T]
    # Get the value of the dependence measurement
    val = gpdc.get_dependence_measure(array, xyz)
    pval_a = gpdc.get_analytic_significance(value=val, T=T, dim=dim)
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

def test_run_test_cmi_knn(cmi_knn, data_frame_c):
    # Check the run_test function
    check_run_test(cmi_knn, data_frame_c)

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
#def test_run_test_cmi_symb(cmi_symb, data_frame_d):
#    # Make the data frame integer values
#    df, parents = data_frame_d
#    df.values = (df.values * 1000).astype(int)
#    # Check the run_test function
#    check_run_test(cmi_symb, (df, parents))

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
