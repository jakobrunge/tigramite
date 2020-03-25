"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import special, stats, spatial
import numpy as np

from .independence_tests_base import CondIndTest

try:
    from tigramite import tigramite_cython_code
except:
    print("Could not import packages for CMIknn and GPDC estimation")


class CMIknn(CondIndTest):
    r"""Conditional mutual information test based on nearest-neighbor estimator.

    Conditional mutual information is the most general dependency measure coming
    from an information-theoretic framework. It makes no assumptions about the
    parametric form of the dependencies by directly estimating the underlying
    joint density. The test here is based on the estimator in  S. Frenzel and B.
    Pompe, Phys. Rev. Lett. 99, 204101 (2007), combined with a shuffle test to
    generate  the distribution under the null hypothesis of independence first
    used in [3]_. The knn-estimator is suitable only for variables taking a
    continuous range of values. For discrete variables use the CMIsymb class.

    Notes
    -----
    CMI is given by

    .. math:: I(X;Y|Z) &= \int p(z)  \iint  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Its knn-estimator is given by

    .. math:: \widehat{I}(X;Y|Z)  &=   \psi (k) + \frac{1}{T} \sum_{t=1}^T
            \left[ \psi(k_{Z,t}) - \psi(k_{XZ,t}) - \psi(k_{YZ,t}) \right]

    where :math:`\psi` is the Digamma function.  This estimator has as a
    parameter the number of nearest-neighbors :math:`k` which determines the
    size of hyper-cubes around each (high-dimensional) sample point. Then
    :math:`k_{Z,},k_{XZ},k_{YZ}` are the numbers of neighbors in the respective
    subspaces.

    :math:`k` can be viewed as a density smoothing parameter (although it is
    data-adaptive unlike fixed-bandwidth estimators). For large :math:`k`, the
    underlying dependencies are more smoothed and CMI has a larger bias,
    but lower variance, which is more important for significance testing. Note
    that the estimated CMI values can be slightly negative while CMI is a non-
    negative quantity.

    This method requires the scipy.spatial.cKDTree package and the tigramite
    cython module.

    References
    ----------

    .. [3] J. Runge (2018): Conditional Independence Testing Based on a
           Nearest-Neighbor Estimator of Conditional Mutual Information.
           In Proceedings of the 21st International Conference on Artificial
           Intelligence and Statistics.
           http://proceedings.mlr.press/v84/runge18a.html

    Parameters
    ----------
    knn : int or float, optional (default: 0.2)
        Number of nearest-neighbors which determines the size of hyper-cubes
        around each (high-dimensional) sample point. If smaller than 1, this is
        computed as a fraction of T, hence knn=knn*T. For knn larger or equal to
        1, this is the absolute number.

    shuffle_neighbors : int, optional (default: 10)
        Number of nearest-neighbors within Z for the shuffle surrogates which
        determines the size of hyper-cubes around each (high-dimensional) sample
        point.

    transform : {'ranks', 'standardize',  'uniform', False}, optional
        (default: 'ranks')
        Whether to transform the array beforehand by standardizing
        or transforming to uniform marginals.

    n_jobs : int (optional, default = -1)
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.

    significance : str, optional (default: 'shuffle_test')
        Type of significance test to use. For CMIknn only 'fixed_thres' and
        'shuffle_test' are available.

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
                 knn=0.2,
                 shuffle_neighbors=5,
                 significance='shuffle_test',
                 transform='ranks',
                 n_jobs=-1,
                 **kwargs):
        # Set the member variables
        self.knn = knn
        self.shuffle_neighbors = shuffle_neighbors
        self.transform = transform
        self._measure = 'cmi_knn'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.n_jobs = n_jobs
        # Call the parent constructor
        CondIndTest.__init__(self, significance=significance, **kwargs)
        # Print some information about construction
        if self.verbosity > 0:
            if self.knn < 1:
                print("knn/T = %s" % self.knn)
            else:
                print("knn = %s" % self.knn)
            print("shuffle_neighbors = %d\n" % self.shuffle_neighbors)

    def _get_nearest_neighbors(self, array, xyz, knn):
        """Returns nearest neighbors according to Frenzel and Pompe (2007).

        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        knn : int or float
            Number of nearest-neighbors which determines the size of hyper-cubes
            around each (high-dimensional) sample point. If smaller than 1, this
            is computed as a fraction of T, hence knn=knn*T. For knn larger or
            equal to 1, this is the absolute number.

        Returns
        -------
        k_xz, k_yz, k_z : tuple of arrays of shape (T,)
            Nearest neighbors in subspaces.
        """

        dim, T = array.shape
        array = array.astype('float')

        # Add noise to destroy ties...
        array += (1E-6 * array.std(axis=1).reshape(dim, 1)
                  * np.random.rand(array.shape[0], array.shape[1]))

        if self.transform == 'standardize':
            # Standardize
            array = array.astype('float')
            array -= array.mean(axis=1).reshape(dim, 1)
            array /= array.std(axis=1).reshape(dim, 1)
            # FIXME: If the time series is constant, return nan rather than
            # raising Exception
            if np.isnan(array).sum() != 0:
                raise ValueError("nans after standardizing, "
                                 "possibly constant array!")
        elif self.transform == 'uniform':
            array = self._trafo2uniform(array)
        elif self.transform == 'ranks':
            array = array.argsort(axis=1).argsort(axis=1).astype('float')


        # Use cKDTree to get distances eps to the k-th nearest neighbors for
        # every sample in joint space XYZ with maximum norm
        tree_xyz = spatial.cKDTree(array.T)
        epsarray = tree_xyz.query(array.T, k=knn+1, p=np.inf,
                                  eps=0., n_jobs=self.n_jobs)[0][:, knn].astype('float')

        # Prepare for fast cython access
        dim_x = int(np.where(xyz == 0)[0][-1] + 1)
        dim_y = int(np.where(xyz == 1)[0][-1] + 1 - dim_x)

        k_xz, k_yz, k_z = \
                tigramite_cython_code._get_neighbors_within_eps_cython(array,
                                                                       T,
                                                                       dim_x,
                                                                       dim_y,
                                                                       epsarray,
                                                                       knn,
                                                                       dim)
        return k_xz, k_yz, k_z

    def get_dependence_measure(self, array, xyz):
        """Returns CMI estimate as described in Frenzel and Pompe PRL (2007).

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

        dim, T = array.shape

        if self.knn < 1:
            knn_here = max(1, int(self.knn*T))
        else:
            knn_here = max(1, int(self.knn))

        k_xz, k_yz, k_z = self._get_nearest_neighbors(array=array,
                                                      xyz=xyz,
                                                      knn=knn_here)

        val = special.digamma(knn_here) - (special.digamma(k_xz) +
                                           special.digamma(k_yz) -
                                           special.digamma(k_z)).mean()

        return val


    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False):
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest niehgbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

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
        dim, T = array.shape

        # Skip shuffle test if value is above threshold
        # if value > self.minimum threshold:
        #     if return_null_dist:
        #         return 0., None
        #     else:
        #         return 0.

        # max_neighbors = max(1, int(max_neighbor_ratio*T))
        x_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            if self.verbosity > 2:
                print("            nearest-neighbor shuffle significance "
                      "test with n = %d and %d surrogates" % (
                      self.shuffle_neighbors, self.sig_samples))

            # Get nearest neighbors around each sample point in Z
            z_array = np.fastCopyAndTranspose(array[z_indices, :])
            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array,
                                       k=self.shuffle_neighbors,
                                       p=np.inf,
                                       eps=0.)[1].astype('int32')

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):

                # Generate random order in which to go through indices loop in
                # next step
                order = np.random.permutation(T).astype('int32')
                # print(order[:5])
                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = \
                    tigramite_cython_code._get_restricted_permutation_cython(
                        T=T,
                        shuffle_neighbors=self.shuffle_neighbors,
                        neighbors=neighbors,
                        order=order)

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

        # Sort
        null_dist.sort()
        pval = (null_dist >= value).mean()

        if return_null_dist:
            return pval, null_dist
        return pval

