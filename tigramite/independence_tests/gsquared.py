"""Tigramite causal discovery for time series."""

# Author: Sagar Nagaraj Simha, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import special, spatial
import numpy as np

from scipy.stats import chi2
from scipy.special import xlogy
from scipy.stats.contingency import crosstab
from scipy.stats.contingency import expected_freq
from scipy.stats.contingency import margins
from .independence_tests_base import CondIndTest

class Gsquared(CondIndTest):
    r"""G-squared conditional independence test for categorical data.

    Uses Chi2 as the null distribution and the method from [7]_ to
    adjust the degrees of freedom. Valid only asymptotically, recommended are
    above 1000-2000 samples (depends on data). For smaller sample sizes use the
    CMIsymb class which includes a local permutation test.

    Assumes one-dimensional X, Y. But can be combined with PairwiseMultCI to
    obtain a test for multivariate X, Y.

    This method requires the scipy.stats package.

    Notes
    -----
    The general formula is

    .. math:: G(X;Y|Z) &= 2 n \sum p(z)  \sum \sum  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)}

    where :math:`n` is the sample size. This is simply :math:`2 n CMI(X;Y|Z)`.

    References
    ----------

    .. [7] Bishop, Y.M.M., Fienberg, S.E. and Holland, P.W. (1975) Discrete
           Multivariate Analysis: Theory and Practice. MIT Press, Cambridge.

    Parameters
    ----------
    n_symbs : int, optional (default: None)
        Number of symbols in input data. Should be at least as large as the
        maximum array entry + 1. If None, n_symbs is inferred by scipy's crosstab

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
                 **kwargs):
        
        # Setup the member variables
        self._measure = 'gsquared'
        self.n_symbs = n_symbs
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        CondIndTest.__init__(self, **kwargs)

        if self.verbosity > 0:
            print("n_symbs = %s" % self.n_symbs)
            print("")

    def get_dependence_measure(self, array, xyz, data_type=None):
        """Returns Gsquared/G-test test statistic.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            G-squared estimate.
        """
        _, T = array.shape
        z_indices = np.where(xyz == 2)[0]

        # Flip 2D-array so that order is ([zn...z0, ym...y0, xk...x0], T). The
        # contingency table is built in this order to ease creating subspaces
        # of Z=z.
        array_flip = np.flipud(array)

        # When n_symbs is given, levels=range(0, n_symbs). If data does not
        # have a symbol in levels, then count=0 in the corresponding N-D
        # position of contingency table. If levels does not contain a certain
        # symbol that is present in the data, then the symbol from data is
        # ignored. If None, then levels are inferred from data (default).

        if self.n_symbs is None:
            levels = None
        else:
            levels = np.tile(np.arange(self.n_symbs), (len(xyz), 1))  
            # Assuming same list of levels for (z, y, x).

        _, observed = crosstab(*(np.asarray(np.split(array_flip, len(xyz), axis=0)).reshape((-1, T))), levels=levels,
                           sparse=False)

        observed_shape = observed.shape

        gsquare = 0.0
        dof = 0

        # The following loop is over the z-subspace to sum over the G-squared
        # statistic and count empty entries to adjust the degrees of freedom.

        # TODO: Can be further optimized to operate entirely on observed array
        # without 'for', to operate only within slice of z. sparse=True can
        # also optimize further.

        # For each permutation of z = (zn ... z1, z0). Example - (0...1,0,1)
        for zs in np.ndindex(observed_shape[:len(z_indices)]):
            observedYX = observed[zs]
            mY, mX = margins(observedYX)

            if(np.sum(mY)!=0):
                expectedYX = expected_freq(observedYX)
                gsquare += 2 * np.sum(xlogy(observedYX, observedYX) 
                                      - xlogy(observedYX, expectedYX))

                # Check how many rows and columns are all-zeros. i.e. how may
                # marginals are zero in expected-frq
                nzero_rows = np.sum(~expectedYX.any(axis=1)) 
                nzero_cols = np.sum(~expectedYX.any(axis=0))

                # Compute dof. Reduce by 1 dof for every marginal row & column=
                # 0 and add to global degrees of freedom [adapted from
                # Bishop, 1975].
                cardYX = observedYX.shape
                dof += ((cardYX[0] - 1 - nzero_rows) * (cardYX[1] - 1 - nzero_cols))

        # dof cannot be lesser than 1
        dof = max(dof, 1)
        self._temp_dof = dof
        return gsquare

    def get_analytic_significance(self, value, T, dim, xyz):
        """Return the p_value of test statistic value 'value', according to a
           chi-square distribution with 'dof' degrees of freedom."""
                      
        # Calculate the p_value
        p_value = chi2.sf(value, self._temp_dof)
        del self._temp_dof

        return p_value


if __name__ == '__main__':
    
    import tigramite
    from tigramite.data_processing import DataFrame
    import tigramite.data_processing as pp
    import numpy as np

    seed=42
    random_state = np.random.default_rng(seed=seed)
    cmi = Gsquared()

    T = 1000
    dimz = 3
    z = random_state.binomial(n=1, p=0.5, size=(T, dimz)).reshape(T, dimz)
    x = np.empty(T).reshape(T, 1)
    y = np.empty(T).reshape(T, 1)
    for t in range(T):
        val = z[t, 0].squeeze()
        prob = 0.2+val*0.6
        x[t] = random_state.choice([0,1], p=[prob, 1.-prob])
        y[t] = random_state.choice([0,1, 2], p=[prob, (1.-prob)/2., (1.-prob)/2.])

    print('start')
    print(cmi.run_test_raw(x, y, z=None))
    print(cmi.run_test_raw(x, y, z=z))
