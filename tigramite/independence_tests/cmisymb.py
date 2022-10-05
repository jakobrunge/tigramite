"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import warnings
import numpy as np
from scipy.stats.contingency import crosstab
# from numba import jit   # could make it even faster, also acticate @jit(forceobj=True)

from .independence_tests_base import CondIndTest

class CMIsymb(CondIndTest):
    r"""Conditional mutual information test for discrete/categorical data.

    Conditional mutual information is the most general dependency measure
    coming from an information-theoretic framework. It makes no assumptions
    about the parametric form of the dependencies by directly estimating the
    underlying joint density. The test here is based on directly estimating
    the joint distribution assuming symbolic input, combined with a
    local shuffle test to generate  the distribution under the null hypothesis of
    independence. This estimator is suitable only for discrete variables.
    For continuous variables use the CMIknn class and for mixed-variable
    datasets the CMIknnMixed class (including mixed-type variables).

    Assumes one-dimensional X, Y.

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
                          "autocorrelation may not be correct for discrete "
                          "data")

    # # @jit  #(forceobj=True)
    def _bincount_hist(self, symb_array, weights=None):
        """Computes histogram from symbolic array.

        The maximum of the symbolic array determines the alphabet / number
        of bins.

        Parameters
        ----------
        symb_array : integer array
            Data array of shape (dim, T). If a float is passed, it will be converted to int.

        weights : float array, optional (default: None)
            Optional weights array of shape (dim, T).

        Returns
        -------
        hist : array
            Histogram array of shape (base, base, base, ...)*number of
            dimensions with Z-dimensions coming first.
        """

        if 'int' not in str(symb_array.dtype):
            # raise ValueError("Input data must of integer type, where each "
            #                  "number indexes a symbol.")
            warnings.warn("Input data should be of integer type, where each "
                          "number indexes a symbol. If you provide a float,"
                          " then the array will still be converted to int.")
            symb_array = symb_array.astype('int')

        if self.n_symbs is None:
            n_symbs = int(symb_array.max() + 1)   # + 1 accounts for the '0'
        else:
            n_symbs = self.n_symbs
            if n_symbs < int(symb_array.max() + 1):
                raise ValueError("n_symbs must be >= symb_array.max() + 1 = {}".format(symb_array.max() + 1))

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

        # High-dimensional histogram, ALSO SWAP summation below when using
        # bincount!
        # hist = self._bincount_hist(array, weights=None)  
        array_flip = np.flipud(array)
        _, hist = crosstab(*tuple(np.split(array_flip, len(xyz), axis=0)), 
            levels=None, sparse=False)

        def _plogp_vector(T):
            """Precalculation of p*log(p) needed for entropies."""
            gfunc = np.zeros(T + 1)
            data = np.arange(1, T + 1, 1)
            gfunc[1:] = data * np.log(data)
            def plogp_func(time):
                return gfunc[time]
            return np.vectorize(plogp_func)

        # Dimensions are (Z^dz, .... Z^1, Y, X) using crosstab, otherwise
        # (X, Y, Z....)
        plogp = _plogp_vector(T)
        hxyz = (-(plogp(hist)).sum() + plogp(T)) / float(T)
        hxz = (-(plogp(hist.sum(axis=-1))).sum() + plogp(T)) / float(T)
        hyz = (-(plogp(hist.sum(axis=-2))).sum() + plogp(T)) / float(T)
        hz = (-(plogp(hist.sum(axis=-1).sum(axis=-1))).sum()+plogp(T)) / float(T)
        val = hxz + hyz - hz - hxyz
        return val

    # @jit  #(forceobj=True)
    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        """Returns p-value for shuffle significance test.

        Performes a local permutation test: x_i values are only permuted with
        those x_j for which z_i = z_j. Samples are drawn without replacement
        as much as possible.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for original (unshuffled) estimate.

        Returns
        -------
        pval : float
            p-value.
        """

        dim, T = array.shape
        x_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0:
            # Get neighbors around each sample point in z
            z_array = array[z_indices, :].T
            # Unique combinations of z in the data (z1, z2, z3 ...)
            z_comb = np.unique(z_array, axis=0)

            # Create neighbor indices of length z_comb with default as -1.
            neighbors = np.full((len(z_comb), T), -1)
            # Neighborhood indices for each unique combination in z_comb.
            for i in range(len(z_comb)):
                neighbor_indices = np.where((z_array == z_comb[i]).all(axis=1))[0]
                neighbors[i, :len(neighbor_indices)] = neighbor_indices

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):
                # Generate random order in which to go through samples.
                order = self.random_state.permutation(T).astype('int32')
                restricted_permutation = np.zeros(T, dtype='int32')
                # A global list of used indices across time samples and combinations.
                # Since there are no repetitive (z) indices across combinations, a global list can be used.
                used = np.array([], dtype='int32')
                for sample_index in order:
                    # Get the index of the z combination for sample_index in z_comb
                    z_choice_index = np.where((z_comb == array[z_indices, sample_index]).all(axis=1))[0][0]
                    neighbors_choices = neighbors[z_choice_index][neighbors[z_choice_index] > -1]
                    # Shuffle neighbors in-place to randomize the choice of indices
                    self.random_state.shuffle(neighbors_choices)
                    # Permuting indices
                    m = 0
                    use = neighbors_choices[m]
                    while ((use in used) and (m < len(neighbors_choices))):
                        m += 1
                        use = neighbors_choices[m]

                    restricted_permutation[sample_index] = use
                    used = np.append(used, use)

                array_shuffled = np.copy(array)
                for i in x_indices:
                    array_shuffled[i] = array[i, restricted_permutation]

                null_dist[sam] = self.get_dependence_measure(array_shuffled,
                                                             xyz)
        else:
            null_dist = \
                self._get_shuffle_dist(array, xyz,
                                       self.get_dependence_measure,
                                       sig_samples=self.sig_samples,
                                       sig_blocklength=self.sig_blocklength,
                                       verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval


if __name__ == '__main__':
    
    import tigramite
    from tigramite.data_processing import DataFrame
    import tigramite.data_processing as pp
    import numpy as np

    np.random.seed(42)
    cmi = CMIsymb(sig_samples=100, seed=4)

    T = 1000
    dimz = 10
    z = np.random.binomial(n=1, p=0.5, size=(T, dimz)).reshape(T, dimz)
    x = np.empty(T).reshape(T, 1)
    y = np.empty(T).reshape(T, 1)
    for t in range(T):
        val = z[t, 0].squeeze()
        prob = 0.2+val*0.6
        x[t] = np.random.choice([0,1], p=[prob, 1.-prob])
        y[t] = np.random.choice([0,1, 2], p=[prob, (1.-prob)/2., (1.-prob)/2.])

    print('start')
    print(cmi.run_test_raw(x, y, z=None))
    print(cmi.run_test_raw(x, y, z=z))