"""
Tests for independence_tests.py.
"""
from __future__ import print_function
from collections import Counter, defaultdict
import numpy as np
from nose.tools import assert_equal
import pytest

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIsymb, CMIknn #, GPACE
from tigramite.independence_tests import _construct_array
import tigramite.data_processing as pp

# Pylint settings
# pylint: disable=redefined-outer-name

# Define the verbosity at the global scope
VERBOSITY = 1

def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale."""
    return -0.5 * np.log(1. - par_corr**2)


# INDEPENDENCE TEST GENERATION #################################################
@pytest.fixture()
    # Generate the independence test
def a_test(request):
    return ParCorr(use_mask=False,
                   mask_type=None,
                   significance='analytic',
                   fixed_thres=None,
                   sig_samples=10000,
                   sig_blocklength=3,
                   confidence='analytic',
                   conf_lev=0.9,
                   conf_samples=10000,
                   conf_blocklength=1,
                   recycle_residuals=False,
                   verbosity=0)

def test_construct_array():
    # Make some fake data
    np.random.seed(42)
    data = np.arange(100).reshape(10,10).T
    # Make a fake mask
    data_mask = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='bool')
    # Set the node values
    X = [(1, -1)]
    Y = [(0, 0)]
    Z = [(0, -1), (1, -2), (2, 0)]
    # Set tau_max
    tau_max = 2

    # Test with no masking
    array, xyz = _construct_array(X=X, Y=Y, Z=Z,
                                  tau_max=tau_max,
                                  use_mask=False,
                                  data=data,
                                  mask=data_mask,
                                  missing_flag=None,
                                  mask_type=None,
                                  verbosity=VERBOSITY)
    # Get the expected results
    N, T = data.shape
    n_times = T - (2 * tau_max)
    # Ensure that this is respected
    # TODO pick up things here
    expect_array = np.array([list(range(13, 19)),
                             list(range(4, 10)),
                             list(range(3, 9)),
                             list(range(12, 18)),
                             list(range(24, 30))])
    expect_xyz = np.array([0, 1, 2, 2, 2,])
    # Test the results
    np.testing.assert_almost_equal(array, expect_array)
    np.testing.assert_almost_equal(xyz, expect_xyz)

    # masking y
    array, xyz = _construct_array(X=X, Y=Y, Z=Z,
                                  tau_max=tau_max,
                                  use_mask=True,
                                  data=data,
                                  mask=data_mask,
                                  mask_type=['y'],
                                  verbosity=VERBOSITY)
    # Test the results
    np.testing.assert_almost_equal(array, expect_array)
    np.testing.assert_almost_equal(xyz, expect_xyz)

    # masking all
    array, xyz = _construct_array(X=X, Y=Y, Z=Z,
                                  tau_max=tau_max,
                                  use_mask=True,
                                  data=data,
                                  mask=data_mask,
                                  mask_type=['x', 'y', 'z'],
                                  verbosity=VERBOSITY)
    # Test the results
    np.testing.assert_almost_equal(array, expect_array)
    np.testing.assert_almost_equal(xyz, expect_xyz)

def test_missing_values():
    np.random.seed(42)
    data = np.array([[0, 10, 20, 30],
                        [1, 11, 21, 31],
                        [2, 12, 22, 32],
                        [3, 13, 999, 33],
                        [4, 14, 24, 34],
                        [5, 15, 25, 35],
                        [6, 16, 26, 36]])
    data_mask = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype='bool')

    X = [(1, -2)]
    Y = [(0, 0)]
    Z = [(2, -1)]

    tau_max = 1

    # Missing values
    res = _construct_array(
        X=X, Y=Y, Z=Z,
        tau_max=tau_max,
        use_mask=False,
        data=data,
        mask=data_mask,
        missing_flag=999,
        mask_type=['y'], verbosity=VERBOSITY)

    # print(res[0])
    np.testing.assert_almost_equal(res[0], np.array([[10, 14],
                                                     [ 2,  6],
                                                     [21, 25]]))

def test_bootstrap_vs_analytic_confidence_parcorr(a_test):

    np.random.seed(1)
    ci_par_corr = a_test

    cov = np.array([[1., 0.3],[0.3, 1.]])
    array = np.random.multivariate_normal(mean=np.zeros(2),
                    cov=cov, size=150).T
    val = np.corrcoef(array)[0,1]
    # print(val)
    dim, T = array.shape
    xyz = np.array([0,1])
    conf_ana = ci_par_corr.get_analytic_confidence(df=T-dim,
                            value=val,
                            conf_lev=ci_par_corr.conf_lev)
    conf_boots = ci_par_corr.get_bootstrap_confidence(
        array, xyz,
        dependence_measure=ci_par_corr.get_dependence_measure,
        conf_samples=ci_par_corr.conf_samples,
        conf_blocklength=ci_par_corr.conf_blocklength,
        conf_lev=ci_par_corr.conf_lev)
    np.testing.assert_allclose(np.array(conf_ana),
                               np.array(conf_boots),
                               atol=0.01)

def test_shuffle_vs_analytic_significance_parcorr(a_test):
    np.random.seed(3)
    ci_par_corr = a_test

    cov = np.array([[1., 0.04],[0.04, 1.]])
    array = np.random.multivariate_normal(mean=np.zeros(2),
                    cov=cov, size=250).T
    # array = np.random.randn(3, 10)
    val = np.corrcoef(array)[0,1]
    # print(val)
    dim, T = array.shape
    xyz = np.array([0,1])

    pval_ana = ci_par_corr.get_analytic_significance(value=val, T=T, dim=dim)

    pval_shuffle = ci_par_corr.get_shuffle_significance(array, xyz, val)
    # Adjust p-value for two-sided measures

    np.testing.assert_allclose(np.array(pval_ana),
                               np.array(pval_shuffle),
                               atol=0.01)

def test_parcorr_get_single_residuals(a_test):
    np.random.seed(5)
    ci_par_corr = a_test
    target_var = 0  #np.array([True, False, False, False])
    true_residual = np.random.randn(4, 1000)

    array = np.copy(true_residual)

    array[0] += 0.5*array[2:].sum(axis=0)

    est_residual = ci_par_corr._get_single_residuals(array, target_var,
            standardize=False, return_means=False)
    np.testing.assert_allclose(est_residual, true_residual[0],
                               rtol=1e-5, atol=0.02)

def test_par_corr(a_test):
    np.random.seed(42)
    ci_par_corr = a_test
    val_ana = 0.6
    T = 1000
    array = np.random.randn(5, T)

    cov = np.array([[1., val_ana],[val_ana, 1.]])
    array[:2, :] = np.random.multivariate_normal(mean=np.zeros(2),
                    cov=cov, size=T).T

    # Generate some confounding
    array[0] += 0.5* array[2:].sum(axis=0)
    array[1] += 0.7* array[2:].sum(axis=0)

    dim, T = array.shape
    xyz = np.array([0,1,2,2,2])

    val_est = ci_par_corr.get_dependence_measure(array, xyz)

    np.testing.assert_allclose(np.array(val_ana),
                               np.array(val_est),
                               atol=0.02)

def test_gpdc_get_single_residuals():

    np.random.seed(42)
    ci_test = GPDC(significance='analytic',
                   sig_samples=1000,
                   sig_blocklength=1,
                   confidence='bootstrap',
                   conf_lev=0.9,
                   conf_samples=100,
                   conf_blocklength=None,
                   use_mask=False,
                   mask_type='y',
                   recycle_residuals=False,
                   verbosity=0)

    c = .3
    T = 1000

    np.random.seed(42)

    def func(x):
        return x * (1. - 4. * x**0 * np.exp(-x**2 / 2.))

    array = np.random.randn(3, T)
    array[1] += c*func(array[2])   #.sum(axis=0)
    xyz = np.array([0,1] + [2 for i in range(array.shape[0]-2)])

    target_var = 1

    dim, T = array.shape
    # array -= array.mean(axis=1).reshape(dim, 1)
    c_std = c  #/array[1].std()
    # array /= array.std(axis=1).reshape(dim, 1)
    array_orig = np.copy(array)

    (est_residual, pred) = ci_test._get_single_residuals(
                    array, target_var,
                    standardize=False,
                    return_means=True)

    # Testing that in the center the fit is good
    center = np.where(np.abs(array_orig[2]) < .7)[0]

    np.testing.assert_allclose(pred[center],
        c_std*func(array_orig[2][center]), atol=0.2)

def test_gpdc_get_single_residuals_2(a_test):
    np.random.seed(42)
    ci_test = GPDC(significance='analytic',
                   sig_samples=1000,
                   sig_blocklength=1,
                   confidence='bootstrap',
                   conf_lev=0.9,
                   conf_samples=100,
                   conf_blocklength=None,
                   use_mask=False,
                   mask_type='y',
                   recycle_residuals=False,
                   verbosity=0)

    ci_par_corr = a_test 
    a = 0.
    c = .3
    T = 500
    # Each key refers to a variable and the incoming links are supplied as a
    # list of format [((driver, lag), coeff), ...]
    links_coeffs = {0: [((0, -1), a)],
                    1: [((1, -1), a), ((0, -1), c)],
                    }

    np.random.seed(42)
    data, true_parents_neighbors = pp.var_process(
        links_coeffs,
        use='inv_inno_cov', T=T)
    dataframe = pp.DataFrame(data)
    ci_test.set_dataframe(dataframe)
    # ci_test.set_tau_max(1)

    # X=[(1, -1)]
    # Y=[(1, 0)]
    # Z=[(0, -1)] + [(1, -tau) for tau in range(1, 2)]
    # array, xyz, XYZ = ci_test.get_array(X, Y, Z,
    #     verbosity=0)]
    # ci_test.run_test(X, Y, Z,)
    def func(x):
        return x * (1. - 4. * x**0 * np.exp(-x**2 / 2.))

    true_residual = np.random.randn(3, T)
    array = np.copy(true_residual)
    array[1] += c*func(array[2])   #.sum(axis=0)
    xyz = np.array([0,1] + [2 for i in range(array.shape[0]-2)])

    print('xyz ', xyz, np.where(xyz == 1))
    target_var = 1

    dim, T = array.shape
    # array -= array.mean(axis=1).reshape(dim, 1)
    c_std = c  #/array[1].std()
    # array /= array.std(axis=1).reshape(dim, 1)
    array_orig = np.copy(array)

    import matplotlib
    from matplotlib import pyplot
    (est_residual, pred) = ci_test._get_single_residuals(
                    array, target_var,
                    standardize=False,
                    return_means=True)
    (resid_, pred_parcorr) = ci_par_corr._get_single_residuals(
                    array, target_var,
                    standardize=False,
                    return_means=True)

    #fig = pyplot.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(array_orig[2], array_orig[1])
    #ax.scatter(array_orig[2], pred, color='red')
    #ax.scatter(array_orig[2], pred_parcorr, color='green')
    #ax.plot(np.sort(array_orig[2]), c_std*func(np.sort(array_orig[2])), color='black')
    #pyplot.savefig('.')

def test_shuffle_vs_analytic_significance_gpdc_2():
    np.random.seed(42)
    ci_gpdc = GPDC(significance='analytic',
                   sig_samples=1000,
                   sig_blocklength=1,
                   confidence='bootstrap',
                   conf_lev=0.9,
                   conf_samples=100,
                   conf_blocklength=None,
                   use_mask=False,
                   mask_type='y',
                   recycle_residuals=False,
                   verbosity=0)

    cov = np.array([[1., 0.2], [0.2, 1.]])
    array = np.random.multivariate_normal(mean=np.zeros(2), cov=cov, size=245).T

    dim, T = array.shape
    xyz = np.array([0,1])

    val = ci_gpdc.get_dependence_measure(array, xyz)
    pval_ana = ci_gpdc.get_analytic_significance(value=val, T=T, dim=dim)
    pval_shf = ci_gpdc.get_shuffle_significance(array, xyz, val)
    np.testing.assert_allclose(np.array(pval_ana), np.array(pval_shf), atol=0.05)

def test_shuffle_vs_analytic_significance_gpdc():
    np.random.seed(42)
    ci_gpdc = GPDC(significance='analytic',
                   sig_samples=1000,
                   sig_blocklength=1,
                   confidence='bootstrap',
                   conf_lev=0.9,
                   conf_samples=100,
                   conf_blocklength=None,
                   use_mask=False,
                   mask_type='y',
                   recycle_residuals=False,
                   verbosity=0)

    cov = np.array([[1., 0.01], [0.01, 1.]])
    array = np.random.multivariate_normal(mean=np.zeros(2),
                    cov=cov, size=300).T

    dim, T = array.shape
    xyz = np.array([0,1])

    val = ci_gpdc.get_dependence_measure(array, xyz)
    pval_ana = ci_gpdc.get_analytic_significance(value=val, T=T, dim=dim)
    pval_shf = ci_gpdc.get_shuffle_significance(array, xyz, val)
    np.testing.assert_allclose(np.array(pval_ana), np.array(pval_shf), atol=0.05)

def test_cmi_knn():

    np.random.seed(42)
    ci_cmi_knn = CMIknn(use_mask=False,
                        mask_type=None,
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


    # ci_cmi_knn._trafo2uniform(self, x)

    val_ana = 0.6
    T = 10000
    np.random.seed(42)
    array = np.random.randn(5, T)

    cov = np.array([[1., val_ana],[val_ana, 1.]])
    array[:2, :] = np.random.multivariate_normal(
                    mean=np.zeros(2),
                    cov=cov, size=T).T

    # Generate some confounding
    if len(array) > 2:
        array[0] += 0.5* array[2:].sum(axis=0)
        array[1] += 0.7* array[2:].sum(axis=0)

    # print(np.corrcoef(array)[0,1])
    # print(val)
    dim, T = array.shape
    xyz = np.array([0,1,2,2,2])

    val_est = ci_cmi_knn.get_dependence_measure(array, xyz)

    print(val_est)
    print(_par_corr_to_cmi(val_ana))

    np.testing.assert_allclose(np.array(_par_corr_to_cmi(val_ana)),
                               np.array(val_est),
                               atol=0.02)

def test_trafo2uniform():

    T = 1000
    # np.random.seed(None)
    np.random.seed(42)
    array = np.random.randn(2, T)

    bins = 10
    ci_gpdc = GPDC(significance='analytic',
                   sig_samples=1000,
                   sig_blocklength=1,
                   confidence='bootstrap',
                   conf_lev=0.9,
                   conf_samples=100,
                   conf_blocklength=None,
                   use_mask=False,
                   mask_type='y',
                   recycle_residuals=False,
                   verbosity=0)


    uniform = ci_gpdc._trafo2uniform(array)
    # print(uniform)

    # import matplotlib
    # from matplotlib import pylab
    for i in range(array.shape[0]):
        print(uniform[i].shape)
        hist, edges = np.histogram(uniform[i], bins=bins,
                                  density=True)
        # pylab.figure()
        # pylab.hist(uniform[i], color='grey', alpha=0.3)
        # pylab.hist(array[i], alpha=0.3)
        # pylab.show()
        print(hist/float(bins))  #, edges
        np.testing.assert_allclose(np.ones(bins)/float(bins),
                                      hist/float(bins),
                                       atol=0.01)

def test_cmi_symb():

    np.random.seed(42)
    ci_cmi_symb = CMIsymb(use_mask=False,
                        mask_type=None,
                        significance='shuffle_test',
                        fixed_thres=None,
                        sig_samples=10000,
                        sig_blocklength=3,

                        confidence='bootstrap',
                        conf_lev=0.9,
                        conf_samples=10000,
                        conf_blocklength=1,

                        verbosity=0)

    val_ana = 0.6
    T = 100000
    np.random.seed(None)
    array = np.random.randn(3, T)

    cov = np.array([[1., val_ana],[val_ana, 1.]])
    array[:2, :] = np.random.multivariate_normal(
                    mean=np.zeros(2),
                    cov=cov, size=T).T

    # Generate some confounding
    if len(array) > 2:
        array[0] += 0.5* array[2:].sum(axis=0)
        array[1] += 0.7* array[2:].sum(axis=0)

    # Transform to symbolic data
    array = pp.quantile_bin_array(array.T, bins=16).T

    dim, T = array.shape
    xyz = np.array([0,1,2,2,2])

    val_est = ci_cmi_symb.get_dependence_measure(array, xyz)

    print(val_est)
    print(_par_corr_to_cmi(val_ana))

    np.testing.assert_allclose(np.array(_par_corr_to_cmi(val_ana)),
                               np.array(val_est),
                               atol=0.02)
