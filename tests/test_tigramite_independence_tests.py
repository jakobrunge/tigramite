import numpy

# Make Python see modules in parent package
# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tigramite.independence_tests import ParCorr, GPACE, GPDC, CMIknn, CMIsymb, _construct_array
import tigramite.data_processing as pp

import nose
import nose.tools as nt
# import unittest


verbosity = 0

def _par_corr_to_cmi(par_corr):
    """Transformation of partial correlation to CMI scale."""

    return -0.5 * numpy.log(1. - par_corr**2)

#
#  Start
#
class TestCondInd():  #unittest.TestCase):
    # def __init__(self):
    #     pass

    def setUp(self):

       auto = 0.6
       coeff = 0.6
       T = 1000
       numpy.random.seed(42)
       # True graph
       links_coeffs = {0: [((0, -1), auto)],
                       1: [((1, -1), auto), ((0, -1), coeff)],
                       2: [((2, -1), auto), ((1, -1), coeff)]
                       }

       self.data, self.true_parents_coeffs = pp.var_process(links_coeffs, T=T)
       T, N = self.data.shape 

       self.ci_par_corr = ParCorr(use_mask=False,
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


       self.ci_gpdc = GPDC(
                            significance='analytic',
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

    def test_construct_array(self):

        data = numpy.array([[0, 10, 20, 30],
                            [1, 11, 21, 31],
                            [2, 12, 22, 32],
                            [3, 13, 23, 33],
                            [4, 14, 24, 34],
                            [5, 15, 25, 35],
                            [6, 16, 26, 36]])
        data_mask = numpy.array([[0, 1, 1, 0],
                                 [0, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype='bool')

        X = [(1, -1)]
        Y = [(0, 0)]
        Z = [(0, -1), (1, -2), (2, 0)]

        tau_max = 2

        # No masking
        res = _construct_array(
            X=X, Y=Y, Z=Z,
            tau_max=tau_max,
            use_mask=False,
            data=data,
            mask=data_mask,
            missing_flag=None,
            mask_type=None, verbosity=verbosity)
        print res[0]
        numpy.testing.assert_almost_equal(res[0],
                                          numpy.array([[13, 14, 15],
                                                     [ 4,  5,  6],
                                                     [ 3,  4,  5],
                                                     [12, 13, 14],
                                                     [24, 25, 26]]))
        numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

        # masking y
        res = _construct_array(
            X=X, Y=Y, Z=Z,
            tau_max=tau_max,
            use_mask=True,
            data=data,
            mask=data_mask,
            mask_type=['y'], verbosity=verbosity)
        print res[0]

        numpy.testing.assert_almost_equal(res[0],
                                          numpy.array([[13, 14, 15],
                                                     [ 4,  5,  6],
                                                     [ 3,  4,  5],
                                                     [12, 13, 14],
                                                     [24, 25, 26]]))

        numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

        # masking all
        res = _construct_array(
            X=X, Y=Y, Z=Z,
            tau_max=tau_max,
            use_mask=True,
            data=data,
            mask=data_mask,
            mask_type=['x', 'y', 'z'], verbosity=verbosity)
        print res[0]

        numpy.testing.assert_almost_equal(res[0],
                                          numpy.array([[13, 14, 15],
                                                     [ 4,  5,  6],
                                                     [ 3,  4,  5],
                                                     [12, 13, 14],
                                                     [24, 25, 26]]))

        numpy.testing.assert_almost_equal(res[1], numpy.array([0, 1, 2, 2, 2]))

    def test_missing_values(self):

        data = numpy.array([[0, 10, 20, 30],
                            [1, 11, 21, 31],
                            [2, 12, 22, 32],
                            [3, 13, 999, 33],
                            [4, 14, 24, 34],
                            [5, 15, 25, 35],
                            [6, 16, 26, 36],
                            ])
        data_mask = numpy.array([[0, 0, 0, 0],
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
            mask_type=['y'], verbosity=verbosity)

        # print res[0]
        numpy.testing.assert_almost_equal(res[0],
                                          numpy.array([[10, 14],
                                                     [ 2,  6],
                                                     [21, 25]]))


    def test_bootstrap_vs_analytic_confidence_parcorr(self):

        cov = numpy.array([[1., 0.3],[0.3, 1.]])
        array = numpy.random.multivariate_normal(mean=numpy.zeros(2),
                        cov=cov, size=150).T

        val = numpy.corrcoef(array)[0,1]
        # print val
        dim, T = array.shape
        xyz = numpy.array([0,1])

        conf_ana = self.ci_par_corr.get_analytic_confidence(df=T-dim, 
                                value=val, 
                                conf_lev=self.ci_par_corr.conf_lev)

        conf_boots = self.ci_par_corr.get_bootstrap_confidence(
            array, xyz,
            dependence_measure=self.ci_par_corr.get_dependence_measure,
            conf_samples=self.ci_par_corr.conf_samples, 
            conf_blocklength=self.ci_par_corr.conf_blocklength,
            conf_lev=self.ci_par_corr.conf_lev, 
            )
        
        print conf_ana
        print conf_boots

        numpy.testing.assert_allclose(numpy.array(conf_ana), 
                                   numpy.array(conf_boots),
                                   atol=0.01)


    def test_shuffle_vs_analytic_significance_parcorr(self):

        cov = numpy.array([[1., 0.04],[0.04, 1.]])
        array = numpy.random.multivariate_normal(mean=numpy.zeros(2),
                        cov=cov, size=250).T
        # array = numpy.random.randn(3, 10)
        val = numpy.corrcoef(array)[0,1]
        # print val
        dim, T = array.shape
        xyz = numpy.array([0,1])

        pval_ana = self.ci_par_corr.get_analytic_significance(value=val,
                                                             T=T, dim=dim)

        pval_shuffle = self.ci_par_corr.get_shuffle_significance(array, xyz,
                               val)
        # Adjust p-value for two-sided measures
        
        print pval_ana
        print pval_shuffle

        numpy.testing.assert_allclose(numpy.array(pval_ana), 
                                   numpy.array(pval_shuffle),
                                   atol=0.01)


    def test__parcorr_get_single_residuals(self):

        target_var = 0  #numpy.array([True, False, False, False])
        true_residual = numpy.random.randn(4, 1000)

        array = numpy.copy(true_residual)

        array[0] += 0.5*array[2:].sum(axis=0)

        est_residual = self.ci_par_corr._get_single_residuals(array, target_var, 
                standardize=False, return_means=False)

        # print est_residual[:10]
        # print true_residual[0, :10]
        numpy.testing.assert_allclose(est_residual, true_residual[0], atol=0.01)


    def test_par_corr(self):

        val_ana = 0.6
        T = 1000
        array = numpy.random.randn(5, T)

        cov = numpy.array([[1., val_ana],[val_ana, 1.]])
        array[:2, :] = numpy.random.multivariate_normal(mean=numpy.zeros(2),
                        cov=cov, size=T).T

        # Generate some confounding
        array[0] += 0.5* array[2:].sum(axis=0)
        array[1] += 0.7* array[2:].sum(axis=0)

        # print numpy.corrcoef(array)[0,1]
        # print val
        dim, T = array.shape
        xyz = numpy.array([0,1,2,2,2])

        val_est = self.ci_par_corr.get_dependence_measure(array, xyz)
        
        print val_est
        print val_ana

        numpy.testing.assert_allclose(numpy.array(val_ana), 
                                   numpy.array(val_est),
                                   atol=0.02)

    def test__gpdc_get_single_residuals(self):


        ci_test = self.ci_gpdc
        # ci_test = self.ci_par_corr

        c = .3
        T = 1000

        numpy.random.seed(42)

        def func(x):
            return x * (1. - 4. * x**0 * numpy.exp(-x**2 / 2.))

        array = numpy.random.randn(3, T)
        array[1] += c*func(array[2])   #.sum(axis=0)
        xyz = numpy.array([0,1] + [2 for i in range(array.shape[0]-2)])

        target_var = 1    

        dim, T = array.shape
        # array -= array.mean(axis=1).reshape(dim, 1)
        c_std = c  #/array[1].std()
        # array /= array.std(axis=1).reshape(dim, 1)
        array_orig = numpy.copy(array)

        (est_residual, pred) = ci_test._get_single_residuals(
                        array, target_var, 
                        standardize=False, 
                        return_means=True)

        # Testing that in the center the fit is good
        center = numpy.where(numpy.abs(array_orig[2]) < .7)[0]
        print (pred[center][:10]).round(2)
        print (c_std*func(array_orig[2][center])[:10]).round(2)
        numpy.testing.assert_allclose(pred[center], 
            c_std*func(array_orig[2][center]), atol=0.2)

    def plot__gpdc_get_single_residuals(self):


        #######
        ci_test = self.ci_gpdc
        # ci_test = self.ci_par_corr

        a = 0.
        c = .3
        T = 500
        # Each key refers to a variable and the incoming links are supplied as a
        # list of format [((driver, lag), coeff), ...]
        links_coeffs = {0: [((0, -1), a)],
                        1: [((1, -1), a), ((0, -1), c)],
                        }

        numpy.random.seed(42)
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
            return x * (1. - 4. * x**0 * numpy.exp(-x**2 / 2.))

        true_residual = numpy.random.randn(3, T)
        array = numpy.copy(true_residual)
        array[1] += c*func(array[2])   #.sum(axis=0)
        xyz = numpy.array([0,1] + [2 for i in range(array.shape[0]-2)])

        print 'xyz ', xyz, numpy.where(xyz==1)
        target_var = 1    

        dim, T = array.shape
        # array -= array.mean(axis=1).reshape(dim, 1)
        c_std = c  #/array[1].std()
        # array /= array.std(axis=1).reshape(dim, 1)
        array_orig = numpy.copy(array)

        import matplotlib
        from matplotlib import pyplot
        (est_residual, pred) = ci_test._get_single_residuals(
                        array, target_var, 
                        standardize=False, 
                        return_means=True)
        (resid_, pred_parcorr) = self.ci_par_corr._get_single_residuals(
                        array, target_var, 
                        standardize=False, 
                        return_means=True)

        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.scatter(array_orig[2], array_orig[1])
        ax.scatter(array_orig[2], pred, color='red')
        ax.scatter(array_orig[2], pred_parcorr, color='green')
        ax.plot(numpy.sort(array_orig[2]), c_std*func(numpy.sort(array_orig[2])), color='black')

        pyplot.savefig('/home/jakobrunge/test/gpdctest.pdf')


    def test_shuffle_vs_analytic_significance_gpdc(self):

        cov = numpy.array([[1., 0.2], [0.2, 1.]])
        array = numpy.random.multivariate_normal(mean=numpy.zeros(2),
                        cov=cov, size=245).T

        dim, T = array.shape
        xyz = numpy.array([0,1])

        val = self.ci_gpdc.get_dependence_measure(array, xyz)

        pval_ana = self.ci_gpdc.get_analytic_significance(value=val,
                                                             T=T, dim=dim)

        pval_shuffle = self.ci_gpdc.get_shuffle_significance(array, xyz,
                               val)

        print pval_ana
        print pval_shuffle

        numpy.testing.assert_allclose(numpy.array(pval_ana), 
                                   numpy.array(pval_shuffle),
                                   atol=0.05)


    def test_shuffle_vs_analytic_significance_gpdc(self):

        cov = numpy.array([[1., 0.01], [0.01, 1.]])
        array = numpy.random.multivariate_normal(mean=numpy.zeros(2),
                        cov=cov, size=300).T

        dim, T = array.shape
        xyz = numpy.array([0,1])

        val = self.ci_gpdc.get_dependence_measure(array, xyz)

        pval_ana = self.ci_gpdc.get_analytic_significance(value=val,
                                                             T=T, dim=dim)

        pval_shuffle = self.ci_gpdc.get_shuffle_significance(array, xyz,
                               val)
        print pval_ana
        print pval_shuffle

        numpy.testing.assert_allclose(numpy.array(pval_ana), 
                                   numpy.array(pval_shuffle),
                                   atol=0.05)

    def test_cmi_knn(self):

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
        numpy.random.seed(42)
        array = numpy.random.randn(5, T)

        cov = numpy.array([[1., val_ana],[val_ana, 1.]])
        array[:2, :] = numpy.random.multivariate_normal(
                        mean=numpy.zeros(2),
                        cov=cov, size=T).T

        # Generate some confounding
        if len(array) > 2:
            array[0] += 0.5* array[2:].sum(axis=0)
            array[1] += 0.7* array[2:].sum(axis=0)

        # print numpy.corrcoef(array)[0,1]
        # print val
        dim, T = array.shape
        xyz = numpy.array([0,1,2,2,2])

        val_est = ci_cmi_knn.get_dependence_measure(array, xyz)
        
        print val_est
        print _par_corr_to_cmi(val_ana)

        numpy.testing.assert_allclose(numpy.array(_par_corr_to_cmi(val_ana)), 
                                   numpy.array(val_est),
                                   atol=0.02)

    def test_trafo2uniform(self):

        T = 1000
        # numpy.random.seed(None)
        array = numpy.random.randn(2, T)

        bins = 10

        uniform = self.ci_gpdc._trafo2uniform(array)
        # print uniform

        # import matplotlib
        # from matplotlib import pylab
        for i in range(array.shape[0]):
            print uniform[i].shape
            hist, edges = numpy.histogram(uniform[i], bins=bins, 
                                      density=True)
            # pylab.figure()
            # pylab.hist(uniform[i], color='grey', alpha=0.3)
            # pylab.hist(array[i], alpha=0.3)
            # pylab.show()
            print hist/float(bins)  #, edges
            numpy.testing.assert_allclose(numpy.ones(bins)/float(bins), 
                                          hist/float(bins),
                                           atol=0.01)

    def test_cmi_symb(self):

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
        numpy.random.seed(None)
        array = numpy.random.randn(3, T)

        cov = numpy.array([[1., val_ana],[val_ana, 1.]])
        array[:2, :] = numpy.random.multivariate_normal(
                        mean=numpy.zeros(2),
                        cov=cov, size=T).T

        # Generate some confounding
        if len(array) > 2:
            array[0] += 0.5* array[2:].sum(axis=0)
            array[1] += 0.7* array[2:].sum(axis=0)

        # Transform to symbolic data
        array = pp.quantile_bin_array(array.T, bins=16).T

        dim, T = array.shape
        xyz = numpy.array([0,1,2,2,2])

        val_est = ci_cmi_symb.get_dependence_measure(array, xyz)
        
        print val_est
        print _par_corr_to_cmi(val_ana)

        numpy.testing.assert_allclose(numpy.array(_par_corr_to_cmi(val_ana)), 
                                   numpy.array(val_est),
                                   atol=0.02)

if __name__ == "__main__":
    # unittest.main()

    # tci = TestCondInd()  #unittest.TestCase)

    # tci.setUp()
    # tci.test_construct_array()
    # tci.test_missing_values()
    # tci.test_shuffle_vs_analytic_significance_gpdc()
    # tci.test_trafo2uniform()
    # tci.test_cmi_symb()
    # tci.test_bootstrap_vs_analytic_confidence()
    # tci.test_shuffle_vs_analytic_significance_gpdc()
    # tci.test__gpdc_get_single_residuals()
    # unittest.main()
    nose.run()
