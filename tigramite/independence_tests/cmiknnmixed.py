from __future__ import print_function
import warnings
import math

from scipy import special, spatial
import numpy as np
from numba import jit, njit, prange

from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
import warnings

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils.extmath import cartesian

from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI

import time

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


class CMIknnMixed(CondIndTest):
    r"""Conditional mutual information test based on nearest-neighbor estimator.

    Conditional mutual information is the most general dependency measure coming
    from an information-theoretic framework. It makes no assumptions about the
    parametric form of the dependencies by directly estimating the underlying
    joint density. The tests here are based on the entropy estimation using 
    k-nearest neighbors. We implement three different approaches:
    
    (1) Mesner & Shalizi [1]
    (2) Conditional variant [2]
    (3) Our variant: TODO add link

    building upon the tigramite package.
    
    These approaches differ in how the distance metrics are defined when searching 
    for neighbors: 
    
    (1) The distance on the discrete dimensions for unequal values is 1, otherwise 0.
    (2) This approach splits the space into clusters for which discrete values are 
        all equal, then computes distances between thsoe points (which now have 
        only continuous values). 
    (2) This approach uses the approach from [1], but defines the distance for
        points with unequal discrete values as infinite, and ignores all 
        neighbors that have infinite distances. 

    Combined with a shuffle test to
    generate the distribution under the null hypothesis of independence,
    also described in [2].
    The knn-estimator is suitable for heterogeneous variables
    (mixed-type, multivariate with discrete and continuous dimensions). 
    For mixture-type variables, use only (1) or (3). 
    
    Notes
    -----
    These estimators have as a
    parameter the number of nearest-neighbors :math:`k` which determines the
    size of hyper-cubes around each (high-dimensional) sample point.
    
    For variants (2) and (3), k is used locally, meaning that it defines
    how many neighbors from a respective subsample should be considered.

    :math:`k` can be viewed as a density smoothing parameter (although it is
    data-adaptive unlike fixed-bandwidth estimators). For large :math:`k`, the
    underlying dependencies are more smoothed and CMI has a larger bias,
    but lower variance, which is more important for significance testing. Note
    that the estimated CMI values can be slightly negative while CMI is a non-
    negative quantity.
   
    This method requires the scipy package.

    References
    ----------
    .. [1] Mesner, O.C., & Shalizi, C.R. (2019): Conditional Mutual Information
           Estimation for Mixed Discrete and Continuous Variables with 
           Nearest Neighbors. arXiv: Statistics Theory.
           https://arxiv.org/abs/1912.03387
           
    .. [2] Zan, L., Meynaoui, A., Assaad, C.K., Devijver, E., & Gaussier, 
           É. (2022): A Conditional Mutual Information Estimator for 
           Mixed Data and an Associated Conditional Independence Test. 
           Entropy, 24.
           https://www.mdpi.com/1099-4300/24/9/1234/html
    Parameters
    ----------
    knn : int or float, optional (default: 0.2)
        Number of nearest-neighbors which determines the size of hyper-cubes
        around each (high-dimensional) sample point. If smaller than 1, this is
        computed as a fraction of T, hence knn=knn*T. For knn larger or equal to
        1, this is the absolute number.

    knn_type: string, optional (default: 'local')
        The type of heuristic to be used for setting the k-hyperparameter
        using our approach (3).
        'local': this approach computes the smallest cluster size, and 
        sets k as knn times the smallest cluster size (for all points
        in the dataset)
        'global': if a neighbor is at infinite distance, k is set as 
        the knn times the cluster size of the respective point
        'cluster-size': if a neighbor is at infinite distance, k is set as 
        the cluster size of the respective point
        
    estimator : string, optional (default: 'MS')
        The type of estimator to be used. Three options are available:
        approach (1) (Mesner and Shalizi (2021) [1]): 'MS', 
        approach (2) (Zao et.al. (2022) [2]): 'ZMADG',
        approach (3) (Mesner and Shalizi (2021) [1]) with 
        infinite distance for points from different categories): 'MSinf'
        
    shuffle_neighbors : int, optional (default: 5)
        Number of nearest-neighbors within Z for the shuffle surrogates which
        determines the size of hyper-cubes around each (high-dimensional) sample
        point.

    transform : {'ranks', 'standardize',  'scale', False}, optional
        (default: 'standardize')
        Whether to transform the array beforehand by transforming to ranks,
        standardizing or scaling to (a,b)
    
    scale : tuple, optional (default: (0,1))
        the scale (a,b) to use if the transform is set 'scale'
        
    perc : float, optional (default: None)
        the value to be used as percentage of the cluster size for the realization
        of a discrete value when using the 'MSinf' method. If set to None, 
        it is the same as the knn value.

    workers : int (optional, default = -1)
        Number of workers to use for parallel processing. If -1 is given
        all processors are used. Default: -1.
        
    rho: list of float, optional (default: [np.inf])
        Hyperparameters used for weighting the discrete variable distances. 
        If not initialized, the distance will be set to np.inf, such that discrete
        variables with different values will never be considered neighbors. 
        Otherwise the rho
        ...

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
                 knn=0.1,
                 knn_type='local',
                 estimator='MSinf',
                 shuffle_neighbors=5,
                 significance='shuffle_test',
                 transform='standardize',
                 scale_range=(0, 1),
                 max_with_0=False,
                 workers=-1,
                 model_selection_folds=3,
                 **kwargs):
        # Set the member variables
        self.knn = knn
        self.knn_type = knn_type
        self.estimator = estimator
        self.shuffle_neighbors = shuffle_neighbors
        self.transform = transform
        self.max_with_0 = max_with_0
        self.scale_range = scale_range
        self._measure = 'cmi_knn_mixed'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.workers = workers
        self.model_selection_folds = model_selection_folds
        self.eps = 1e-5

        # Call the parent constructor
        CondIndTest.__init__(self, significance=significance, **kwargs)
        # Print some information about construction
        if self.verbosity > 0:
            if self.knn < 1:
                print("knn/T = %s" % self.knn)
            else:
                print("knn = %s" % self.knn)
            print("shuffle_neighbors = %d\n" % self.shuffle_neighbors)

    def _standardize_array(self, array, dim):
        """Standardizes a given array with dimensions dim.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        dim: int
            number of dimensions of the data.
        
        Returns
        -------
        array : array-like
            The standardized array.
        """
        array = array.astype(np.float64)
        array -= array.mean(axis=1).reshape(dim, 1)
        std = array.std(axis=1)
        for i in range(dim):
            if std[i] != 0.:
                array[i] /= std[i]
        if np.any(std == 0.):
            warnings.warn("Possibly constant array!")
        return array

    def _scale_array(self, array, minmax=(0, 1)):
        """Scales a given array to range minmax dimension-wise.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        minmax: tuple (a, b)
            the min and the max values (a, b) for the scaling
        
        Returns
        -------
        array : array-like
            The scaled array.
        """
        scaler = MinMaxScaler(minmax)
        return scaler.fit_transform(array.T).T

    def _rank_array(self, array):
        """Transform a given array to ranks.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns
        
        Returns
        -------
        array : array-like
            The scaled array.
        """
        return array.argsort(axis=1).argsort(axis=1).astype(np.float64)

    def _transform_mixed_data(self, array, type_mask=None, add_noise=False):
        """Applies data transformations to the continuous dimensions of the given data.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        add_noise : bool (default False)
            Defines whether to add small normal noise to the continuous data.
            
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        array : array-like
            The array with the continuous data transformed. 
            
        """
        continuous_idxs = np.where(np.all(type_mask == 0, axis=1))[0]
        cont_dim = len(continuous_idxs)

        if add_noise:
            array[continuous_idxs, :] += (1E-6 * array[continuous_idxs, :].std(axis=1).reshape(cont_dim, 1)
                                          * self.random_state.random(
                        (array[continuous_idxs, :].shape[0], array[continuous_idxs, :].shape[1])))
        if self.transform == 'standardize':
            array[continuous_idxs, :] = self._standardize_array(array[continuous_idxs, :], cont_dim)
        elif self.transform == 'scale':
            array[continuous_idxs, :] = self._scale_array(array[continuous_idxs, :], minmax=self.scale_range)
        elif self.transform == 'ranks':
            array[continuous_idxs, :] = self._rank_array(array[continuous_idxs, :])
        elif self.transform == 'none':
            pass
        else:
            warnings.warn('Unknown transform')

        return array

    def _transform_to_one_hot_mixed(self, array, xyz, type_mask,
                                    zero_inf=False):
        """Applies one-hot encoding to the discrete dimensions of the array.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : list 
            List that indicates which dimensions belong to which variable, e.g.
            for X, Y, Z one-dimensional xyz = [0, 1, 2]
            
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables
        
        zero_inf : bool, optional (default: False)
            defines whether to set infinite distances between points with different
            values for the discrete dimensions

        Returns
        -------
        array : array-like
            The array with the continuous data transformed. 
            
        """

        discrete_idx_list = np.where(np.all(type_mask == 1, axis=0), 1, 0)
        mixed_idx_list = np.where(np.any(type_mask == 1, axis=0), 1, 0)

        narray = np.copy(array)
        nxyz = np.copy(xyz)
        ntype_mask = np.copy(type_mask)

        appended_columns = 0
        for i in range(len(discrete_idx_list)):
            if discrete_idx_list[i] == 1:
                encoder = OneHotEncoder(handle_unknown='ignore')
                i += appended_columns
                data = narray[:, i]
                encoder_df = encoder.fit_transform(data.reshape(-1, 1)).toarray()
                if zero_inf:
                    encoder_df = np.where(encoder_df == 1, 9999999, 0)

                xyz_val = [nxyz[i]] * encoder_df.shape[-1]
                narray = np.concatenate([narray[:, :i], encoder_df, narray[:, i + 1:]], axis=-1)

                nxyz = np.concatenate([nxyz[:i], xyz_val, nxyz[i + 1:]])
                ntype_mask = np.concatenate([ntype_mask[:, :i],
                                             np.ones(encoder_df.shape),
                                             ntype_mask[:, i + 1:]],
                                            axis=-1)
                appended_columns += encoder_df.shape[-1] - 1

            elif mixed_idx_list[i] == 1 and zero_inf == True:
                i += appended_columns
                data = narray[:, i]
                xyz_val = nxyz[i]

                # find categories 
                categories = np.unique(narray[:, i] * ntype_mask[:, i])
                categories = np.delete(categories, categories == 0.)
                cont_vars = np.unique(narray[:, i] * (1 - ntype_mask[:, i]))

                encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore')
                xyz_val = nxyz[i]
                encoder_df = encoder.fit_transform(data.reshape(-1, 1)).toarray()
                if zero_inf:
                    encoder_df = np.where(encoder_df == 1, 9999999, 0)

                xyz_val = [nxyz[i]] * (encoder_df.shape[-1] + 1)
                cont_column = np.expand_dims(narray[:, i] * (1 - ntype_mask[:, i]), -1)
                narray = np.concatenate([narray[:, :i], cont_column, encoder_df, narray[:, i + 1:]], axis=-1)

                nxyz = np.concatenate([nxyz[:i], xyz_val, nxyz[i + 1:]])
                ntype_mask = np.concatenate([ntype_mask[:, :i],
                                             np.zeros(cont_column.shape),
                                             np.ones(encoder_df.shape),
                                             ntype_mask[:, i + 1:]],
                                            axis=-1)
                appended_columns += encoder_df.shape[-1]

        ndiscrete_idx_list = np.where(np.any(ntype_mask == 1, axis=0), 1, 0)

        return narray, nxyz, ntype_mask, ndiscrete_idx_list

    def get_smallest_cluster_size(self, array, type_mask=None):
        """Computes the smallest number of samples for each realization 
        of the discrete variables. 
        Used for computation of the "local" knn. 
        
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns
        
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables
        Returns
        -------
        min_nc : integer
            The smallest number of samples in a cluster.
        """
        discrete_idx_list = np.where(np.any(type_mask == 1, axis=0), 1, 0)
        discrete_xyz_idx = np.where(np.asarray(discrete_idx_list) == 1)[0]
        # if both are fully continuous, return 0.1 * n
        if len(discrete_xyz_idx) == 0:
            min_nc = array.shape[0]
        else:
            num_xyz_classes = [np.unique(array[:, index]) for index in range(len(discrete_idx_list)) if
                               (discrete_idx_list[index] == 1)]

            xyz_cartesian_product = []

            if len(num_xyz_classes) > 1:
                xyz_cartesian_product = cartesian(num_xyz_classes)
            elif len(num_xyz_classes) > 0:
                xyz_cartesian_product = num_xyz_classes[0]

            min_nc = array.shape[0]

            if len(xyz_cartesian_product) > 0:
                for i, entry in enumerate(xyz_cartesian_product):
                    current_array = array[np.sum(array[:, discrete_xyz_idx] == entry,
                                                 axis=-1) == len(discrete_xyz_idx)]
                    if current_array.shape[0] > 0 and current_array.shape[0] < min_nc:
                        min_nc = current_array.shape[0]

        return min_nc

    @jit(forceobj=True)
    def _get_nearest_neighbors_zeroinf_onehot(self, array, xyz, knn,
                                              type_mask=None):
        """Returns CMI estimate according to [1] with an
        altered distance metric: the 0-inf metric, which attributes 
        infinite distance to points where the values for the discrete dimensions
        do not coincide.
        
        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ. Uses the 0-inf metric for 
        discrete variables.
        
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
        
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables
        Returns
        -------
        k_xz, k_yz, k_z : tuple of arrays of shape (T,)
            Nearest neighbors in subspaces.
        """
        dim, T = array.shape

        array = array.astype(np.float64)
        xyz = xyz.astype(np.int32)

        array = self._transform_mixed_data(array, type_mask)

        array = array.T
        type_mask = type_mask.T

        narray, nxyz, ntype_mask, discrete_idx_list = self._transform_to_one_hot_mixed(array, xyz, type_mask,
                                                                                       zero_inf=True)

        # Subsample indices
        x_indices = np.where(nxyz == 0)[0]
        y_indices = np.where(nxyz == 1)[0]
        z_indices = np.where(nxyz == 2)[0]
        xz_indices = np.concatenate([x_indices, z_indices])
        yz_indices = np.concatenate([y_indices, z_indices])

        # Fit trees
        tree_xyz = spatial.cKDTree(narray)
        neighbors = tree_xyz.query(narray, k=knn + 1, p=np.inf,
                                   distance_upper_bound=9999999)

        n, k = neighbors[0].shape

        epsarray = np.zeros(n)
        for i in range(n):
            if neighbors[0][i, knn] == np.inf:
                # number of non-inf neighbors
                replacement_idx = np.where(neighbors[0][i] != np.inf)[0][-1]
                if self.knn_type == 'global':
                    # look at at least one neighbor
                    r = max(int(replacement_idx * self.perc), 1)
                elif self.knn_type == 'cluster_size' or self.knn_type == 'local':
                    r = replacement_idx
                epsarray[i] = neighbors[0][i, r]
            else:
                epsarray[i] = neighbors[0][i, knn]


        neighbors_radius_xyz = tree_xyz.query_ball_point(narray, epsarray, p=np.inf)

        k_tilde = [
            len(neighbors_radius_xyz[i]) - 1 if len(neighbors_radius_xyz[i]) > 1 else len(neighbors_radius_xyz[i]) for i
            in range(len(neighbors_radius_xyz))]

        # compute entropies
        xz = narray[:, xz_indices]
        tree_xz = spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(xz, r=epsarray, p=np.inf, return_length=True)

        yz = narray[:, yz_indices]
        tree_yz = spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(yz, r=epsarray, p=np.inf, return_length=True)

        if len(z_indices) > 0:
            z = narray[:, z_indices]
            tree_z = spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(z, r=epsarray, p=np.inf, return_length=True)
        else:
            # Number of neighbors is T when z is empty.
            k_z = np.full(T, T, dtype='float')

        end = time.time()

        k_xz = np.asarray([i - 1 if i > 1 else i for i in k_xz])
        k_yz = np.asarray([i - 1 if i > 1 else i for i in k_yz])
        k_z = np.asarray([i - 1 if i > 1 else i for i in k_z])

        return k_tilde, k_xz, k_yz, k_z

    def get_dependence_measure_MSinf(self, array, xyz,
                                     type_mask=None):
        """Returns CMI estimate according to [1] with an
        altered distance metric: the 0-inf metric, which attributes 
        infinite distance to points where the values for the discrete dimensions
        do not coincide.
        
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).
        
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """
        dim, T = array.shape
        # compute knn according to knn type
        if self.knn < 1:
            if self.knn_type == 'global':
                # compute knn
                knn = max(1, int(self.knn * T))
                self.perc = self.knn
            elif self.knn_type == 'cluster_size':
                knn = max(1, int(self.knn * T))
            elif self.knn_type == 'local':
                min_nc = self.get_smallest_cluster_size(array.T, type_mask.T)
                knn = max(1, int(self.knn * min_nc))
        else:
            raise ValueError("MSinf needs knn value as percentage (value < 1), not number of neighbors!")

        knn_tilde, k_xz, k_yz, k_z = self._get_nearest_neighbors_zeroinf_onehot(array=array,
                                                                                xyz=xyz,
                                                                                knn=knn,
                                                                                type_mask=type_mask)
        val = (special.digamma(knn_tilde) - special.digamma(k_xz) -
               special.digamma(k_yz) +
               special.digamma(k_z))

        val = val[np.isfinite(val)].mean()

        if self.max_with_0 and val < 0.:
            val = 0.

        return val

    @jit(forceobj=True)
    def _get_nearest_neighbors_MS_one_hot(self, array, xyz,
                                          knn, type_mask=None):
        """Returns nearest neighbors according to [1].

        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ. Uses a custom-defined metric for 
        discrete variables.

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
            
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        k_tilde, k_xz, k_yz, k_z : tuple of arrays of shape (T,)
            Nearest neighbors in XYZ, XZ, YZ, and Z subspaces.
        """

        dim, T = array.shape

        array = array.astype(np.float64)
        xyz = xyz.astype(np.int32)

        array = self._transform_mixed_data(array, type_mask)

        array = array.T
        type_mask = type_mask.T

        narray, nxyz, ntype_mask, discrete_idx_list = self._transform_to_one_hot_mixed(array,
                                                                                       xyz,
                                                                                       type_mask)

        # Subsample indices
        x_indices = np.where(nxyz == 0)[0]
        y_indices = np.where(nxyz == 1)[0]
        z_indices = np.where(nxyz == 2)[0]

        xz_indices = np.concatenate([x_indices, z_indices])
        yz_indices = np.concatenate([y_indices, z_indices])

        # Fit trees
        tree_xyz = spatial.cKDTree(narray)
        neighbors = tree_xyz.query(narray, k=knn + 1, p=np.inf, workers=self.workers)

        epsarray = neighbors[0][:, -1].astype(np.float64)

        neighbors_radius_xyz = tree_xyz.query_ball_point(narray, epsarray, p=np.inf,
                                                         workers=self.workers)

        # search again for neighbors in the radius to find all of them
        # in the discrete case k_tilde can be larger than the given knn
        k_tilde = np.asarray(
            [len(neighbors_radius_xyz[i]) - 1 if len(neighbors_radius_xyz[i]) > 1 else len(neighbors_radius_xyz[i]) for
             i in range(len(neighbors_radius_xyz))])

        # compute entropies
        xz = narray[:, xz_indices]
        tree_xz = spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(xz, r=epsarray, p=np.inf,
                                        workers=self.workers, return_length=True)

        yz = narray[:, yz_indices]
        tree_yz = spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(yz, r=epsarray, p=np.inf,
                                        workers=self.workers, return_length=True)

        if len(z_indices) > 0:
            z = narray[:, z_indices]
            tree_z = spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(z, r=epsarray, p=np.inf,
                                          workers=self.workers, return_length=True)

        else:
            # Number of neighbors is T when z is empty.
            k_z = np.full(T, T, dtype='float')

        k_xz = np.asarray([i - 1 if i > 1 else i for i in k_xz])
        k_yz = np.asarray([i - 1 if i > 1 else i for i in k_yz])
        k_z = np.asarray([i - 1 if i > 1 else i for i in k_z])

        return k_tilde, k_xz, k_yz, k_z

    def get_dependence_measure_MS(self, array, xyz,
                                  type_mask=None):

        """Returns CMI estimate as described in [1].
        
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).
        
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """
        dim, T = array.shape

        # compute knn
        if self.knn < 1:
            knn = max(1, int(self.knn * T))
        else:
            knn = max(1, self.knn)

        knn_tilde, k_xz, k_yz, k_z = self._get_nearest_neighbors_MS_one_hot(array=array,
                                                                            xyz=xyz,
                                                                            knn=knn,
                                                                            type_mask=type_mask)

        val = (special.digamma(knn_tilde) - special.digamma(k_xz) -
               special.digamma(k_yz) +
               special.digamma(k_z))

        val = val[np.isfinite(val)].mean()

        if self.max_with_0 and val < 0.:
            val = 0.

        return val

    def _compute_entropies_for_discrete_entry(self, array,
                                              discrete_values,
                                              discrete_idxs,
                                              continuous_idxs,
                                              total_num_samples,
                                              knn):
        # select data for which the discrete values are the given ones
        current_array = array[np.sum(array[:, discrete_idxs] == discrete_values,
                                     axis=-1) == len(discrete_idxs)]

        # if we do not have samples, we cannot estimate CMI
        if np.size(current_array) == 0:
            return 0., 0.

        T, dim = current_array.shape

        # if we have more samples than knns and samples are not purely discrete, we can
        # compute CMI
        if len(continuous_idxs) > 0 and T > knn:
            val_continuous_entropy = self._compute_continuous_entropy(current_array[:, continuous_idxs], knn)
        else:
            val_continuous_entropy = 0.

        prob = float(T) / total_num_samples

        # multiply by probabilities of occurence
        val_continuous_entropy *= prob
        # compute entropy for that occurence
        val_discrete_entropy = prob * np.log(prob)

        return val_continuous_entropy, val_discrete_entropy

    def _compute_continuous_entropy(self, array, knn):
        T, dim = array.shape
        if T == 1:
            return 0.

        if knn < 1:
            knn = max(np.rint(knn * T), 1)

        tree = spatial.cKDTree(array)
        epsarray = tree.query(array, k=[knn + 1], p=np.inf,
                              workers=self.workers,
                              eps=0.)[0][:, 0].astype(np.float64)
        epsarray = epsarray[epsarray != 0]
        num_non_zero = len(epsarray)
        if num_non_zero == 0:
            cmi_hat = 0.
        else:
            avg_dist = float(array.shape[-1]) / float(num_non_zero) * np.sum(np.log(2 * epsarray))
            cmi_hat = special.digamma(num_non_zero) - special.digamma(knn) + avg_dist

        return cmi_hat

    def get_dependence_measure_ZMADG(self, array, xyz,
                                     type_mask=None):
        """Returns CMI estimate as described in [2].
        
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).
            
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """

        dim, T = array.shape

        if self.knn > 1:
            raise ValueError("ZMADG needs knn value as percentage (value < 1), not number of neighbors!")
        else:
            knn = self.knn

        array = array.astype(np.float64)
        xyz = xyz.astype(np.int32)

        array = self._transform_mixed_data(array, type_mask)

        array = array.T
        type_mask = type_mask.T

        discrete_idx_list = np.where(np.any(type_mask == 1, axis=0), 1, 0)

        if np.sum(discrete_idx_list) == 0:
            raise ValueError("Variables are continuous, cannot use CMIknnMixed ZMADG!")

        # Subsample indices
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]
        xz_indices = np.concatenate([x_indices, z_indices])
        yz_indices = np.concatenate([y_indices, z_indices])

        discrete_xz_indices = discrete_idx_list[xz_indices]
        discrete_yz_indices = discrete_idx_list[yz_indices]
        discrete_z_indices = discrete_idx_list[z_indices]

        discrete_xyz_idx = np.where(np.asarray(discrete_idx_list) == 1)[0]
        discrete_xz_idx = np.where(np.asarray(discrete_xz_indices) == 1)[0]
        discrete_yz_idx = np.where(np.asarray(discrete_yz_indices) == 1)[0]
        discrete_z_idx = np.where(np.asarray(discrete_z_indices) == 1)[0]

        continuous_xyz_idx = np.where(np.asarray(discrete_idx_list) == 0)[0]
        continuous_xz_idx = np.where(np.asarray(discrete_xz_indices) == 0)[0]
        continuous_yz_idx = np.where(np.asarray(discrete_yz_indices) == 0)[0]
        continuous_z_idx = np.where(np.asarray(discrete_z_indices) == 0)[0]

        # get the number of unique values for each category of the discrete variable
        # add empty set for code not to break when accessing [0]
        num_xz_classes = [np.unique(array[:, xz_indices][:, index]) for index in range(len(discrete_xz_indices)) if
                          (discrete_xz_indices[index] == 1)]
        num_yz_classes = [np.unique(array[:, yz_indices][:, index]) for index in range(len(discrete_yz_indices)) if
                          (discrete_yz_indices[index] == 1)]
        num_z_classes = [np.unique(array[:, z_indices][:, index]) for index in range(len(discrete_z_indices)) if
                         (discrete_z_indices[index] == 1)]
        num_xyz_classes = [np.unique(array[:, index]) for index in range(len(discrete_idx_list)) if
                           (discrete_idx_list[index] == 1)]

        xyz_cartesian_product = []
        xz_cartesian_product = []
        yz_cartesian_product = []
        z_cartesian_product = []

        if len(num_xyz_classes) > 1:
            xyz_cartesian_product = cartesian(num_xyz_classes)
        elif len(num_xyz_classes) > 0:
            xyz_cartesian_product = num_xyz_classes[0]

        if len(num_xz_classes) > 1:
            xz_cartesian_product = cartesian(num_xz_classes)
        elif len(num_xz_classes) > 0:
            xz_cartesian_product = num_xz_classes[0]

        if len(num_yz_classes) > 1:
            yz_cartesian_product = cartesian(num_yz_classes)
        elif len(num_yz_classes) > 0:
            yz_cartesian_product = num_yz_classes[0]

        if len(num_z_classes) > 1:
            z_cartesian_product = cartesian(num_z_classes)
        elif len(num_z_classes) > 0:
            z_cartesian_product = num_z_classes[0]

        ####### start computing entropies 

        if len(xyz_cartesian_product) > 0:
            xyz_cmi = 0.
            xyz_entropy = 0.

            for i, entry in enumerate(xyz_cartesian_product):
                xyz_cont_entropy, xyz_disc_entropy = self._compute_entropies_for_discrete_entry(array, entry,
                                                                                                discrete_xyz_idx,
                                                                                                continuous_xyz_idx,
                                                                                                T, knn)
                xyz_cmi += xyz_cont_entropy
                xyz_entropy -= xyz_disc_entropy
        else:
            xyz_cmi = self._compute_continuous_entropy(array, knn)
            xyz_entropy = 0.

        h_xyz = xyz_cmi + xyz_entropy

        if len(xz_cartesian_product) > 0:
            xz_cmi = 0.
            xz_entropy = 0.

            for i, entry in enumerate(xz_cartesian_product):
                xz_cont_entropy, xz_disc_entropy = self._compute_entropies_for_discrete_entry(array[:, xz_indices],
                                                                                              entry,
                                                                                              discrete_xz_idx,
                                                                                              continuous_xz_idx,
                                                                                              T, knn)
                xz_cmi += xz_cont_entropy
                xz_entropy -= xz_disc_entropy
        else:
            xz_cmi = self._compute_continuous_entropy(array[:, xz_indices], knn)
            xz_entropy = 0.

        h_xz = xz_cmi + xz_entropy

        # compute entropies in Xy subspace
        if len(yz_cartesian_product) > 0:
            yz_cmi = 0.
            yz_entropy = 0.

            for i, entry in enumerate(yz_cartesian_product):
                yz_cont_entropy, yz_disc_entropy = self._compute_entropies_for_discrete_entry(array[:, yz_indices],
                                                                                              entry,
                                                                                              discrete_yz_idx,
                                                                                              continuous_yz_idx,
                                                                                              T, knn)
                yz_cmi += yz_cont_entropy
                yz_entropy -= yz_disc_entropy
        else:
            yz_cmi = self._compute_continuous_entropy(array[:, yz_indices], knn)
            yz_entropy = 0.

        h_yz = yz_cmi + yz_entropy

        # compute entropies in Z subspace
        if len(z_cartesian_product) > 0:
            z_cmi = 0.
            z_entropy = 0.

            for i, entry in enumerate(z_cartesian_product):
                z_cont_entropy, z_disc_entropy = self._compute_entropies_for_discrete_entry(array[:, z_indices],
                                                                                            entry,
                                                                                            discrete_z_idx,
                                                                                            continuous_z_idx,
                                                                                            T, knn)
                z_cmi += z_cont_entropy
                z_entropy -= z_disc_entropy
        else:
            z_cmi = self._compute_continuous_entropy(array[:, z_indices], knn)
            z_entropy = 0.

        h_z = z_cmi + z_entropy

        # put it all together for the CMI estimation
        val = h_xz + h_yz - h_xyz - h_z

        if self.max_with_0:
            if val < 0.:
                val = 0.

        return val

    def get_dependence_measure(self, array, xyz,
                               data_type=None):
        """Calls the appropriate function to estimate CMI.
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,)
            
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """
        # check that data is really mixed
        if data_type is None:
            raise ValueError("Type mask cannot be none for CMIknnMixed!")
        if np.sum(data_type) > data_type.size:
            raise ValueError("Type mask contains other values than 0 and 1!")

        if self.estimator == 'MS':
            return self.get_dependence_measure_MS(array,
                                                  xyz,
                                                  data_type)
        elif self.estimator == 'ZMADG':
            return self.get_dependence_measure_ZMADG(array,
                                                     xyz,
                                                     data_type)
        elif self.estimator == 'MSinf':
            return self.get_dependence_measure_MSinf(array,
                                                     xyz,
                                                     data_type)
        else:
            raise ValueError('No such estimator available!')

    # @jit(forceobj=True)
    def get_restricted_permutation(self, T, shuffle_neighbors, neighbors, order):

        restricted_permutation = np.zeros(T, dtype=np.int32)
        used = np.array([], dtype=np.int32)

        for sample_index in order:
            neighbors_to_use = neighbors[sample_index]
            m = 0
            use = neighbors_to_use[m]
            while ((use in used) and (m < shuffle_neighbors - 1)):
                m += 1
                use = neighbors_to_use[m]
            restricted_permutation[sample_index] = use
            used = np.append(used, use)

        return restricted_permutation


    @jit(forceobj=True)
    def _generate_random_permutation(self, array, neighbors, x_indices, type_mask):

        T, dim = array.shape
        # Generate random order in which to go through indices loop in
        # next step
        order = self.random_state.permutation(T).astype(np.int32)

        n = np.empty(neighbors.shape[0], dtype=object)

        for i in range(neighbors.shape[0]):
            v = np.unique(neighbors[i])
            # Shuffle neighbor indices for each sample index
            self.random_state.shuffle(v)
            n[i] = v

        # Select a series of neighbor indices that contains as few as
        # possible duplicates
        restricted_permutation = self.get_restricted_permutation(
            T=T,
            shuffle_neighbors=self.shuffle_neighbors,
            neighbors=n,
            order=order)

        array_shuffled = np.copy(array)
        type_mask_shuffled = np.copy(type_mask)

        for i in x_indices:
            array_shuffled[:, i] = array[restricted_permutation, i]
            type_mask_shuffled[:, i] = type_mask[restricted_permutation, i]

        return array_shuffled, type_mask_shuffled

    # @jit(forceobj=True)
    def compute_perm_null_dist(self, array, xyz,
                               type_mask=None, value=0.05):
        # compute valid neighbors
        narray, nxyz, ntype_mask, discrete_idx_list = self._transform_to_one_hot_mixed(array,
                                                                                       xyz,
                                                                                       type_mask,
                                                                                       zero_inf=True)

        z_indices = np.where(nxyz == 2)[0]

        if self.verbosity > 2:
            print("            nearest-neighbor shuffle significance "
                  "test with n = %d and %d surrogates" % (
                      self.shuffle_neighbors, self.sig_samples))
        # Get nearest neighbors around each sample point in Z
        z_array = np.array(narray[:, z_indices])
        tree_xyz = spatial.cKDTree(z_array)
        neighbors = tree_xyz.query(z_array,
                                   k=self.shuffle_neighbors + 1,
                                   p=np.inf,
                                   workers=self.workers,
                                   distance_upper_bound=9999999,
                                   eps=0.)

        # remove all neighbors with distance infinite -> from another class 
        # for those that are discrete 
        valid_neighbors = np.ones(neighbors[1].shape)
        # fill valid neighbors with point -> if infinite, the neighbor will 
        # be the point itself
        valid_neighbors = np.multiply(valid_neighbors, np.expand_dims(np.arange(valid_neighbors.shape[0]), axis=-1))

        valid_neighbors[neighbors[0] != np.inf] = neighbors[1][neighbors[0] != np.inf]

        null_dist = np.zeros(self.sig_samples)

        for sam in range(self.sig_samples):
            # permute un-encoded array using the valud neighbors list
            array_shuffled, type_mask_shuffled = self._generate_random_permutation(array,
                                                                                   valid_neighbors,
                                                                                   x_indices=np.where(xyz == 0)[0],
                                                                                   type_mask=type_mask)

            # use array instead of narray to avoid double encoding
            null_dist[sam] = self.get_dependence_measure(array_shuffled.T,
                                                         xyz,
                                                         data_type=type_mask_shuffled.T)
        return null_dist

        # return p

    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False,
                                 data_type=None):

        """Returns p-value for nearest-neighbor permutation test.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.
        
        data_type : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables
        
        Returns
        -------
        pval : float
            p-value
        """

        dim, T = array.shape

        array = array.T
        data_type = data_type.T

        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:

            null_dist = self.compute_perm_null_dist(array, xyz, data_type, value)
            # pval = null_dist / self.sig_samples

        else:
            null_dist = \
                self._get_shuffle_dist(array.T, xyz,
                                       sig_samples=self.sig_samples,
                                       sig_blocklength=self.sig_blocklength,
                                       type_mask=data_type.T,
                                       verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            # Sort
            null_dist.sort()
            return pval, null_dist
        return pval

    def _get_shuffle_dist(self, array, xyz,
                          sig_samples, sig_blocklength=None,
                          type_mask=None,
                          verbosity=0):
        """Returns shuffle distribution of test statistic.

        The rows in array corresponding to the X-variable are shuffled using
        a block-shuffle approach.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        dependence_measure : object
           Dependence measure function must be of form
           dependence_measure(array, xyz) and return a numeric value

        sig_samples : int, optional (default: 100)
            Number of samples for shuffle significance test.

        sig_blocklength : int, optional (default: None)
            Block length for block-shuffle significance test. If None, the
            block length is determined from the decay of the autocovariance as
            explained in [1]_.
            
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        null_dist : array of shape (sig_samples,)
            Contains the sorted test statistic values estimated from the
            shuffled arrays.
        """
        dim, T = array.shape

        x_indices = np.where(xyz == 0)[0]
        dim_x = len(x_indices)

        if sig_blocklength is None:
            sig_blocklength = self._get_block_length(array, xyz,
                                                     mode='significance')

        n_blks = int(math.floor(float(T) / sig_blocklength))
        # print 'n_blks ', n_blks
        if verbosity > 2:
            print("            Significance test with block-length = %d "
                  "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        type_mask_shuffled = np.copy(type_mask)
        block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blks * sig_blocklength:]
        tail_type = type_mask_shuffled[x_indices, n_blks * sig_blocklength:]

        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):

            blk_starts = self.random_state.permutation(block_starts)[:n_blks]

            x_shuffled = np.zeros((dim_x, n_blks * sig_blocklength),
                                  dtype=array.dtype)
            type_x_shuffled = np.zeros((dim_x, n_blks * sig_blocklength),
                                       dtype=array.dtype)

            for i, index in enumerate(x_indices):
                for blk in range(sig_blocklength):
                    x_shuffled[i, blk::sig_blocklength] = \
                        array[index, blk_starts + blk]

                    type_x_shuffled[i, blk::sig_blocklength] = \
                        type_mask[index, blk_starts + blk]

            # Insert tail randomly somewhere
            if tail.shape[1] > 0:
                insert_tail_at = self.random_state.choice(block_starts)
                x_shuffled = np.insert(x_shuffled, insert_tail_at,
                                       tail.T, axis=1)
                type_x_shuffled = np.insert(type_x_shuffled, insert_tail_at,
                                            tail_type.T, axis=1)

            for i, index in enumerate(x_indices):
                array_shuffled[index] = x_shuffled[i]
                type_mask_shuffled[index] = type_x_shuffled[i]

            null_dist[sam] = self.get_dependence_measure(array=array_shuffled,
                                                         xyz=xyz,
                                                         data_type=type_mask_shuffled)

        return null_dist

    def get_model_selection_criterion(self, j, parents, tau_max=0):
        """Returns a cross-validation-based score for nearest-neighbor estimates.

        Fits a nearest-neighbor model of the parents to variable j and returns
        the score. The lower, the better the fit. Here used to determine
        optimal hyperparameters in PCMCI(pc_alpha or fixed thres).

        Parameters
        ----------
        j : int
            Index of target variable in data array.

        parents : list
            List of form [(0, -1), (3, -2), ...] containing parents.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        Returns:
        score : float
            Model score.
        """

        import sklearn
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        Y = [(j, 0)]
        X = [(j, 0)]  # dummy variable here
        Z = parents

        array, xyz, data_type = self.dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                               tau_max=tau_max,
                                                               mask_type=self.mask_type,
                                                               return_cleaned_xyz=False,
                                                               do_checks=True,
                                                               verbosity=self.verbosity)
        dim, T = array.shape

        # apply transform to continuous variables
        array = self._transform_mixed_data(array, data_type)

        predictor_indices = list(np.where(xyz == 2)[0])
        predictor_array = array[predictor_indices, :].T
        # Target is only first entry of Y, ie [y]
        target_array = array[np.where(xyz == 1)[0][0], :]

        if predictor_array.size == 0:
            # Regressing on ones if empty parents
            predictor_array = np.ones(T).reshape(T, 1)

        if self.knn < 1:
            knn_here = max(1, int(self.knn * T))
        else:
            knn_here = max(1, int(self.knn))

        # if there are any categoricals in the predictors, encode them to dummies
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder_df = encoder.fit_transform(predictor_array.reshape(-1, 1)).toarray()

        # if the target is discrete, do a classification
        if 1 in data_type.T[:, np.where(xyz == 1)[0][0]]:
            knn_model = KNeighborsClassifier(n_neighbors=knn_here)
        # otherwise do a regression
        else:
            knn_model = KNeighborsRegressor(n_neighbors=knn_here)

        scores = cross_val_score(estimator=knn_model,
                                 X=predictor_array, y=target_array, cv=self.model_selection_folds, n_jobs=self.workers)

        return -scores.mean()


if __name__ == '__main__':
    # Generate some mixed-type data with a binary variable (can also be multinomial) causing two continuous ones.
    random_state = np.random.default_rng(42)
    T = 1000
    data = np.zeros((T, 3))
    data[:, 1] = random_state.binomial(n=1, p=0.5, size=T)
    for t in range(2, T):
        data[t, 0] = 0.5 * data[t - 1, 0] + random_state.normal(0.2 + data[t - 1, 1] * 0.6, 1)
        data[t, 2] = 0.4 * data[t - 1, 2] + random_state.normal(0.2 + data[t - 2, 1] * 0.6, 1)

    data_type = np.zeros(data.shape, dtype='int')
    # X0 is continuous, encoded as 0 in data_type
    data_type[:, 0] = 0
    # X1 is discrete, encoded as 1 in data_type
    data_type[:, 1] = 1
    # X2 is continuous, encoded as 0 in data_type
    data_type[:, 2] = 0

    var_names = [r'$X^0$', r'$X^1$', r'$X^2$']

    dataframe = pp.DataFrame(data,
                             data_type=data_type,
                             var_names=var_names)

    cmi_knn_mixed = CMIknnMixed(significance='shuffle_test',
                                sig_samples=500,
                                estimator='MSinf',
                                knn=0.1)

    Y = [(1, 0)]
    X = [(2, 0)]  # dummy variable here
    Z = [(0, 0)]

    array, xyz, data_type = dataframe.construct_array(X=X, Y=Y, Z=Z,
                                                           tau_max=0)

    start = time.time()
    sig = cmi_knn_mixed.get_shuffle_significance(array, xyz, 0.05, data_type=data_type)
    end = time.time()

    print(end-start, sig)


    # print(cmi_knn_mixed.get_model_selection_criterion(j=2, parents=[(1, -1), (1, 0)], tau_max=0))
