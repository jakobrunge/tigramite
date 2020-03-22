"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import warnings
import numpy as np

from .independence_tests_base import CondIndTest

class CMIsymb(CondIndTest):
    r"""Conditional mutual information test based on discrete estimator.

    Conditional mutual information is the most general dependency measure
    coming from an information-theoretic framework. It makes no assumptions
    about the parametric form of the dependencies by directly estimating the
    underlying joint density. The test here is based on directly estimating
    the joint distribution assuming symbolic input, combined with a
    shuffle test to generate  the distribution under the null hypothesis of
    independence. The knn-estimator is suitable only for discrete variables.
    For continuous variables, either pre-process the data using the functions
    in data_processing or, better, use the CMIknn class.

    Notes
    -----
    CMI and its estimator are given by

    .. math:: I(X;Y|Z) &= \sum p(z)  \sum \sum  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Parameters
    ----------
    n_symbs : int, optional (default: None)
        Number of symbols in input data. Should be at least as large as the
        maximum array entry + 1. If None, n_symbs is based on the
        maximum value in the array (array.max() + 1).

    significance : str, optional (default: 'shuffle_test')
        Type of significance test to use. For CMIsymb only 'fixed_thres' and
        'shuffle_test' are available.

    sig_blocklength : int, optional (default: 1)
        Block length for block-shuffle significance test.

    conf_blocklength : int, optional (default: 1)
        Block length for block-bootstrap.

    **kwargs :
        Arguments passed on to parent class CondIndTest.
    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 n_symbs=None,
                 significance='shuffle_test',
                 sig_blocklength=1,
                 conf_blocklength=1,
                 **kwargs):
        # Setup the member variables
        self._measure = 'cmi_symb'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.n_symbs = n_symbs
        # Call the parent constructor
        CondIndTest.__init__(self,
                             significance=significance,
                             sig_blocklength=sig_blocklength,
                             conf_blocklength=conf_blocklength,
                             **kwargs)

        if self.verbosity > 0:
            print("n_symbs = %s" % self.n_symbs)
            print("")

        if self.conf_blocklength is None or self.sig_blocklength is None:
            warnings.warn("Automatic block-length estimations from decay of "
                          "autocorrelation may not be sensical for discrete "
                          "data")

    def _bincount_hist(self, symb_array, weights=None):
        """Computes histogram from symbolic array.

        The maximum of the symbolic array determines the alphabet / number
        of bins.

        Parameters
        ----------
        symb_array : integer array
            Data array of shape (dim, T).

        weights : float array, optional (default: None)
            Optional weights array of shape (dim, T).

        Returns
        -------
        hist : array
            Histogram array of shape (base, base, base, ...)*number of
            dimensions with Z-dimensions coming first.
        """

        if self.n_symbs is None:
            n_symbs = int(symb_array.max() + 1)
        else:
            n_symbs = self.n_symbs
            if n_symbs < int(symb_array.max() + 1):
                raise ValueError("n_symbs must be >= symb_array.max() + 1 = {}".format(symb_array.max() + 1))

        if 'int' not in str(symb_array.dtype):
            raise ValueError("Input data must of integer type, where each "
                             "number indexes a symbol.")

        dim, T = symb_array.shape

        flathist = np.zeros((n_symbs ** dim), dtype='int16')
        multisymb = np.zeros(T, dtype='int64')
        if weights is not None:
            flathist = np.zeros((n_symbs ** dim), dtype='float32')
            multiweights = np.ones(T, dtype='float32')

        for i in range(dim):
            multisymb += symb_array[i, :] * n_symbs ** i
            if weights is not None:
                multiweights *= weights[i, :]

        if weights is None:
            result = np.bincount(multisymb)
        else:
            result = (np.bincount(multisymb, weights=multiweights)
                      / multiweights.sum())

        flathist[:len(result)] += result

        hist = flathist.reshape(tuple([n_symbs, n_symbs] +
                                      [n_symbs for i in range(dim - 2)])).T

        return hist

    def get_dependence_measure(self, array, xyz):
        """Returns CMI estimate based on bincount histogram.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """

        _, T = array.shape

        # High-dimensional histogram
        hist = self._bincount_hist(array, weights=None)

        def _plogp_vector(T):
            """Precalculation of p*log(p) needed for entropies."""
            gfunc = np.zeros(T + 1)
            data = np.arange(1, T + 1, 1)
            gfunc[1:] = data * np.log(data)
            def plogp_func(time):
                return gfunc[time]
            return np.vectorize(plogp_func)

        plogp = _plogp_vector(T)
        hxyz = (-(plogp(hist)).sum() + plogp(T)) / float(T)
        hxz = (-(plogp(hist.sum(axis=1))).sum() + plogp(T)) / float(T)
        hyz = (-(plogp(hist.sum(axis=0))).sum() + plogp(T)) / float(T)
        hz = (-(plogp(hist.sum(axis=0).sum(axis=0))).sum()+plogp(T)) / float(T)
        val = hxz + hyz - hz - hxyz
        return val

    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        """Returns p-value for shuffle significance test.

        For residual-based test statistics only the residuals are shuffled.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """

        null_dist = self._get_shuffle_dist(array, xyz,
                                           self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval