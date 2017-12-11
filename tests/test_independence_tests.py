"""
Tests for independence_tests.py.
"""
from __future__ import print_function
from collections import Counter, defaultdict, OrderedDict
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

# TEST NODES
TST_X = [(1, -1)]
TST_Y = [(0, 0)]
TST_Z = [(0, -1), (1, -2), (2, 0)]

def rand_node(t_min, n_max, t_max=0, n_min=0):
    """
    Generate a random node to test
    """
    rand_node = np.random.randint(n_min, n_max)
    rand_time = np.random.randint(t_min, t_max)
    return (rand_node, rand_time)

def gen_nodes(n_nodes, seed, t_min=-2, n_max=2):
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

@pytest.fixture(params=[
    # Generate the independence test
    #(X, Y, Z) nodes,                    t_max, m_val, mask_type
    (gen_nodes(3, 0, t_min=-3, n_max=9), 3,     False, None), # Few nodes
    (gen_nodes(7, 1, t_min=-3, n_max=9), 3,     False, None), # More nodes
    (gen_nodes(7, 2, t_min=-3, n_max=3), 3,     False, None), # Repeated nodes
    (gen_nodes(7, 3, t_min=-3, n_max=9), 3,     True,  None), # Missing vals
    (gen_nodes(7, 4, t_min=-3, n_max=9), 3,     True,  ['x']), # M-val + mask
    (gen_nodes(3, 5, t_min=-4, n_max=9), 2,     False, ['x']), # masked x
    (gen_nodes(3, 6, t_min=-4, n_max=9), 2,     False, ['y']), # masked y
    (gen_nodes(3, 7, t_min=-4, n_max=9), 2,     False, ['z']), # masked z
    (gen_nodes(3, 7, t_min=-4, n_max=9), 2,     False, ['x','y','z'])])#mask xyz
def cstrct_array_params(request):
    return request.param

def test_construct_array(cstrct_array_params):
    # Unpack the parameters
    (x_nds, y_nds, z_nds), tau_max, missing_vals, mask_type =\
        cstrct_array_params
    # Make some fake data
    data = np.arange(150).reshape(10, 15).T
    # Get the needed parameters from the data
    T, N = data.shape
    max_lag = 2*tau_max
    n_times = T - max_lag

    # When testing masking and missing value flags, we will remove time slices,
    # starting with the earliest slice.  This counter keeps track of how many
    # rows have been masked.
    n_rows_masked = 0

    # Make a fake mask
    use_mask = False
    data_mask = np.zeros_like(data, dtype='bool')
    if mask_type is not None:
        use_mask = True
        for var, nodes in zip(['x', 'y', 'z'], [x_nds, y_nds, z_nds]):
            if var in mask_type:
                # Get the first node
                a_nd, a_tau = nodes[0]
                # Mask the first value of this node
                data_mask[a_tau - n_times + n_rows_masked, a_nd] = True
                n_rows_masked += 1

    # Choose fake missing value as the first entry from the value that would be
    # returned from the first z-node
    missing_flag = None
    if missing_vals:
        # Take a value that would appear in from the z-node selection and make
        # it the missing value flag
        a_nd, a_tau = z_nds[0]
        missing_flag = data[a_tau - n_times + n_rows_masked, a_nd]
        n_rows_masked += max_lag + 1

    # Construct the array
    array, xyz = _construct_array(x_nds, y_nds, z_nds,
                                  tau_max=tau_max,
                                  use_mask=use_mask,
                                  data=data,
                                  mask=data_mask,
                                  missing_flag=missing_flag,
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
