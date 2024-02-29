"""
Tests for independence_tests.py.
"""
from __future__ import print_function
import numpy as np
import pytest
from scipy import stats

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.pairwise_CI import PairwiseMultCI
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.regressionCI import RegressionCI
# PairwiseMultCI TESTING ##################################################
@pytest.fixture(params=[
    # Generate PairwiseMultCI test instances
    # test PairwiseMultCI for different combinations of:
    # learn_augmented_cond_sets, significance (of the univariate tests),  alpha_pre, pre_step_sample_fraction, cond_ind_test, cond_ind_test_thres, cond_ind_test_thres_pre
    ('True', 'fixed_thres', 0.5, 0.2, ParCorr, 1, 0.5),
    ('True', 'fixed_thres', 0.5, 0.5, ParCorr, 1, 0.5),
    ('True', 'fixed_thres', 0.8, 0.2, ParCorr, 1, 0.5),
    ('True', 'fixed_thres', 0.8, 0.5, ParCorr, 1, 0.5),
    ('True', 'analytic', 0.5, 0.2, ParCorr, None, None),
    ('True', 'shuffle_test', 0.5, 0.2, ParCorr, None, None),
    ('True', 'analytic', 0.5, 0.5, ParCorr, None, None),
    ('True', 'shuffle_test', 0.5, 0.5, ParCorr, None, None),
    ('True', 'analytic', 0.8, 0.2, ParCorr, None, None),
    ('True', 'shuffle_test', 0.8, 0.2, ParCorr, None, None),
    ('True', 'analytic', 0.8, 0.5, ParCorr, None, None),
    ('True', 'shuffle_test', 0.8, 0.5, ParCorr, None, None),
    ('True', 'analytic', 0.5, 0.2, RobustParCorr, None, None),
    ('True', 'shuffle_test', 0.5, 0.2, RobustParCorr, None, None),
    ('True', 'analytic', 0.5, 0.5, RobustParCorr, None, None),
    ('True', 'shuffle_test', 0.5, 0.5, RobustParCorr, None, None),
    ('True', 'analytic', 0.8, 0.2, RobustParCorr, None, None),
    ('True', 'shuffle_test', 0.8, 0.2, RobustParCorr, None, None),
    ('True', 'analytic', 0.8, 0.5, RobustParCorr, None, None),
    ('True', 'shuffle_test', 0.8, 0.5, RobustParCorr, None, None),
    ('True', 'shuffle_test', 0.5, 0.2, CMIknn, None, None),
    ('True', 'shuffle_test', 0.5, 0.5, CMIknn, None, None),
    ('True', 'shuffle_test', 0.8, 0.2, CMIknn, None, None),
    ('True', 'shuffle_test', 0.8, 0.5, CMIknn, None, None),
    ('False', 'fixed_thres', 0.0, 0.0, ParCorr, 1, 0.5),
    ('False', 'analytic', 0.0, 0.0, ParCorr, None, None),
    ('False', 'shuffle_test', 0.0, 0.0, ParCorr, None, None),
    ('False', 'analytic', 0.0, 0.0, RobustParCorr, None, None),
    ('False', 'shuffle_test', 0.0, 0.0, RobustParCorr, None, None),
    ('False', 'shuffle_test', 0.0, 0.0, CMIknn, None, None),
])

def pairwise_mult_ci(request):
    # Unpack the parameters
    learn_augmented_cond_sets, sig, alpha_pre, pre_step_sample_fraction, cond_ind_test, cond_ind_test_thres, cond_ind_test_thres_pre = request.param
    # Generate the par_corr_wls independence test
    if sig != "fixed_thres":
        return PairwiseMultCI(learn_augmented_cond_sets = learn_augmented_cond_sets, cond_ind_test = cond_ind_test(significance = sig),
                          alpha_pre = alpha_pre,
                          pre_step_sample_fraction = pre_step_sample_fraction)
    else:
        return PairwiseMultCI(learn_augmented_cond_sets = learn_augmented_cond_sets, cond_ind_test = cond_ind_test(significance = sig),
                          alpha_pre = None,
                          pre_step_sample_fraction=pre_step_sample_fraction,
                          significance= sig,
                          fixed_thres_pre = 2
                          )

@pytest.fixture(params=[
    # Generate PairwiseMultCI test instances
    # test PairwiseMultCI for different combinations of:
    # seed,  true_dep, T (=sample size)
    (123, 0, 100),
    (123, 0, 1000),
    (123, 0.2, 100),
    (123, 0.2, 1000),
    (123, 0.5, 100),
    (123, 0, 1000),
    (46, 0, 100),
    (46, 0, 1000),
    (46, 0.2, 100),
    (46, 0.2, 1000),
    (46, 0.5, 100),
    (46, 0, 1000),
])

def data_sample_c_cc_cc(request):
    # Set the random seed
    seed, true_dep, T = request.param
    np.random.seed(seed)
    z1 = np.random.normal(0, 1, T).reshape(T, 1)
    z2 = np.random.normal(0, 1, T).reshape(T, 1)
    z = np.hstack((z1,z2))
    x = np.random.normal(0, 1, T).reshape(T, 1) + z1 + z2
    y1 = np.random.normal(0, 1, T).reshape(T, 1) + z1 + z2
    y2 = true_dep * x + y1 + 0.3 * np.random.normal(0, 1, T).reshape(T, 1)
    y = np.hstack((y1,y2))

    # Return data xyz
    return x, y, z

def test_pairwise_mult_ci(pairwise_mult_ci, data_sample_c_cc_cc):
    # Get the data sample values
    x, y, z = data_sample_c_cc_cc
    # Get the analytic significance
    test_result = pairwise_mult_ci.run_test_raw(x = x, y = y, z = z, alpha_or_thres=1)
    val = test_result[0]
    pval = test_result[1]
    np.testing.assert_allclose(pval, 0.5, atol=0.5)

@pytest.fixture(params=[
    # Generate PairwiseMultCI test instances
    # test PairwiseMultCI for different combinations of:
    # learn_augmented_cond_sets, significance (of the univariate tests),  alpha_pre, pre_step_sample_fraction, cond_ind_test, cond_ind_test_thres, cond_ind_test_thres_pre
    ('False', 'fixed_thres', 0.0, 0.0, ParCorr, 1, 0.5),
    ('False', 'analytic', 0.0, 0.0, ParCorr, None, None),
    ('False', 'shuffle_test', 0.0, 0.0, ParCorr, None, None),
    ('False', 'analytic', 0.0, 0.0, RobustParCorr, None, None),
    ('False', 'shuffle_test', 0.0, 0.0, RobustParCorr, None, None),
    ('False', 'shuffle_test', 0.0, 0.0, CMIknn, None, None),
])

def pairwise_mult_ci2(request):
    # Unpack the parameters
    learn_augmented_cond_sets, sig, alpha_pre, pre_step_sample_fraction, cond_ind_test, cond_ind_test_thres, cond_ind_test_thres_pre = request.param
    # Generate the par_corr_wls independence test
    if sig != "fixed_thres":
        return PairwiseMultCI(learn_augmented_cond_sets = learn_augmented_cond_sets, cond_ind_test = cond_ind_test(significance = sig),
                          alpha_pre = alpha_pre,
                          pre_step_sample_fraction = pre_step_sample_fraction), cond_ind_test(significance = sig)
    else:
        return PairwiseMultCI(learn_augmented_cond_sets = learn_augmented_cond_sets, cond_ind_test = cond_ind_test(significance = sig),
                          alpha_pre = None,
                          pre_step_sample_fraction=pre_step_sample_fraction,
                          significance= sig,
                          fixed_thres_pre = 2
                          ), cond_ind_test(significance = sig)


"""def compare_ci(request):
    # Unpack the parameters
    learn_augmented_cond_sets, sig, alpha_pre, pre_step_sample_fraction, cond_ind_test, cond_ind_test_thres, cond_ind_test_thres_pre = request.param
    # Generate the par_corr_wls independence test
    return cond_ind_test(significance = sig)
"""
@pytest.fixture(params=[
    # Generate PairwiseMultCI test instances
    # test PairwiseMultCI for different combinations of:
    # seed,  true_dep, T (=sample size)
    (123, 0, 100),
    (123, 0.2, 100),
    (123, 0.5, 100),
    (123, 0, 1000),
    (46, 0, 100),
    (46, 0.2, 100),
    (46, 0.5, 100),
    (46, 0, 100),
])


def data_sample_c_cc_cc2(request):
    # Set the random seed
    seed, true_dep, T = request.param
    np.random.seed(seed)
    z1 = np.random.normal(0, 1, T).reshape(T, 1)
    z2 = np.random.normal(0, 1, T).reshape(T, 1)
    z = np.hstack((z1,z2))
    x = np.random.normal(0, 1, T).reshape(T, 1) + z1 + z2
    y = np.random.normal(0, 1, T).reshape(T, 1) + z1 + z2 + true_dep * x


    # Return data xyz
    return x, y, z

def test_compare_pairwise_mult_ci(pairwise_mult_ci2, data_sample_c_cc_cc2):
    # Get the data sample values
    x, y, z = data_sample_c_cc_cc2
    # Get the analytic significance
    test_result = pairwise_mult_ci2[0].run_test_raw(x = x, y = y, z = z, alpha_or_thres=1)
    test_result2 = pairwise_mult_ci2[1].run_test_raw(x=x, y=y, z=z, alpha_or_thres=1)
    val = test_result[0]
    pval = test_result[1]
    val2 = test_result2[0]
    pval2 = test_result2[1]
    np.testing.assert_allclose(val, val2, rtol=0.2)
    np.testing.assert_allclose(pval, pval2, atol=0.2)