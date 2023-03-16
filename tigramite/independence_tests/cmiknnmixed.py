"""Tigramite causal discovery for time series."""

# Author: Oana Popescu, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from scipy import special, spatial
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils.extmath import cartesian
import numpy as np
import math
from .independence_tests_base import CondIndTest
from numba import jit
import warnings

# profiling
import cProfile, pstats, io
from pstats import SortKey


class CMIknnMixed(CondIndTest):
    r"""Conditional mutual information test based on nearest-neighbor estimator.

    Conditional mutual information is the most general dependency measure coming
    from an information-theoretic framework. It makes no assumptions about the
    parametric form of the dependencies by directly estimating the underlying
    joint density. The test here is based on the estimator in  S. Frenzel and B.
    Pompe, Phys. Rev. Lett. 99, 204101 (2007), combined with a shuffle test to
    generate  the distribution under the null hypothesis of independence first
    used in the reference below. The knn-estimator is suitable only for variables taking a
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
    
    For the case of mixed variables, the distance metric changes from the L-inf 
    norm to ...

    This method requires the scikit-learn package.

    References
    ----------

    J. Runge (2018): Conditional Independence Testing Based on a
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
        
    estimator : string, optional (default: 'MS')
        The type of estimator to be used. Three options are available:
        Mesner and Shalizi (2021): 'MS', Frenzel and Pompe (2007) with 
        infinite distance for points from different categories: 'FPinf',
        and Zao et.al. (2022) where entropies are computed conditional on 
        the discrete dimensions of X,Y and Z. 

    shuffle_neighbors : int, optional (default: 5)
        Number of nearest-neighbors within Z for the shuffle surrogates which
        determines the size of hyper-cubes around each (high-dimensional) sample
        point.

    transform : {'ranks', 'standardize',  'uniform', False}, optional
        (default: 'ranks')
        Whether to transform the array beforehand by standardizing
        or transforming to uniform marginals.

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
                 estimator='MS',
                 use_local_knn=False,
                 shuffle_neighbors=5,
                 significance='shuffle_test',
                 transform='standardize',
                 scale_range=(0, 1),
                 perc=None,
                 workers=-1,
                 **kwargs):
        # Set the member variables
        self.knn = knn
        self.estimator = estimator
        self.use_local_knn = use_local_knn
        self.shuffle_neighbors = shuffle_neighbors
        self.transform = transform
        if perc is None:
            self.perc = self.knn
        else:
            self.perc = perc
        self.scale_range = scale_range
        self._measure = 'cmi_knn_mixed'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.workers = workers
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
        # array /= array.std(axis=1).reshape(dim, 1)
        # FIXME: If the time series is constant, return nan rather than
        # raising Exception
        if np.any(std == 0.):
            warnings.warn("Possibly constant array!")
            # raise ValueError("nans after standardizing, "
            #                  "possibly constant array!")
        return array
    
    def _scale_array(self, array, minmax=(0, 1)):
        scaler = MinMaxScaler(minmax)
        return scaler.fit_transform(array.T).T
            
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
            # Add noise to destroy ties
            array[continuous_idxs] += (1E-6 * array[continuous_idxs].std(axis=1).reshape(cont_dim, 1)
                  * self.random_state.random((array[continuous_idxs].shape[0], array[continuous_idxs].shape[1])))

        if self.transform == 'standardize':
            array[continuous_idxs] = self._standardize_array(array[continuous_idxs], cont_dim)
        elif self.transform == 'scale':
            array[continuous_idxs] = self._scale_array(array[continuous_idxs], minmax=self.scale_range)
        else:
            warnings.warn('Unknown transform')

        return array
         
    
    def _transform_to_one_hot_mixed(self, array, xyz, 
                                    type_mask,
                                    zero_inf=False):
        
        discrete_idx_list = np.where(np.all(type_mask == 1, axis=0), 1, 0)
        mixed_idx_list = np.where(np.any(type_mask == 1, axis=0), 1, 0)

        narray = np.copy(array)
        nxyz = np.copy(xyz)
        ntype_mask = np.copy(type_mask)

        appended_columns = 0
        for i in range(len(discrete_idx_list)):
            # print(i)
            if discrete_idx_list[i] == 1:
                encoder = OneHotEncoder(handle_unknown='ignore')
                i += appended_columns
                data = narray[:, i]
                xyz_val = nxyz[i]
                encoder_df = encoder.fit_transform(data.reshape(-1, 1)).toarray()
                if zero_inf:
                    encoder_df = np.where(encoder_df == 1, 9999999, 0)

                xyz_val = [nxyz[i]] * encoder_df.shape[-1]
                narray = np.concatenate([narray[:, :i], encoder_df, narray[:, i+1:]], axis=-1)

                nxyz = np.concatenate([nxyz[:i], xyz_val, nxyz[i+1:]])
                ntype_mask = np.concatenate([ntype_mask[:, :i],
                                             np.ones(encoder_df.shape), 
                                             ntype_mask[:, i+1:]], 
                                            axis=-1)
                appended_columns += encoder_df.shape[-1] - 1

            elif mixed_idx_list[i] == 1:
                i += appended_columns
                data = narray[:, i]
                xyz_val = nxyz[i]

                # print(i, narray[:, i], ntype_mask[:, i])
                # find categories 
                categories = np.unique(narray[:, i] * ntype_mask[:, i])
                cont_vars = np.unique(narray[:, i] * (1 - ntype_mask[:, i]))

                encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore')
                xyz_val = nxyz[i]
                encoder_df = encoder.fit_transform(data.reshape(-1, 1)).toarray()
                if zero_inf:
                    encoder_df = np.where(encoder_df == 1, 9999999 + np.max(cont_vars), 0)

                xyz_val = [nxyz[i]] * (encoder_df.shape[-1] + 1)
                cont_column = np.expand_dims(narray[:, i] * (1 - ntype_mask[:, i]), -1)
                narray = np.concatenate([narray[:, :i], cont_column, encoder_df, narray[:, i+1:]], axis=-1)

                nxyz = np.concatenate([nxyz[:i], xyz_val, nxyz[i+1:]])
                ntype_mask = np.concatenate([ntype_mask[:, :i], 
                                             np.zeros(cont_column.shape), 
                                             np.ones(encoder_df.shape), 
                                             ntype_mask[:, i+1:]], 
                                            axis=-1)
                appended_columns += encoder_df.shape[-1]

        ndiscrete_idx_list = np.where(np.any(ntype_mask == 1, axis=0), 1, 0)

        return narray, nxyz, ntype_mask, ndiscrete_idx_list         

        

    def run_test(self, X, Y, Z=None, tau_max=0, cut_off='2xtau_max'):
        """Perform conditional independence test.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.
        If recycle_residuals is True, also _get_single_residuals must be
        available.

        Parameters
        ----------
        X, Y, Z : list of tuples
            X,Y,Z are of the form [(var, -tau)], where var specifies the
            variable index and tau the time lag.

        tau_max : int, optional (default: 0)
            Maximum time lag. This may be used to make sure that estimates for
            different lags in X, Z, all have the same sample size.

        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples. For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample.  Last,
            'max_lag' uses as much samples as possible.

        Returns
        -------
        val, pval : Tuple of floats
            The test statistic value and the p-value.
        """
        # Get the array to test on
        array, xyz, XYZ, type_mask = self._get_array(X, Y, Z, tau_max, cut_off)
        X, Y, Z = XYZ

        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.any(np.isnan(array)):
            raise ValueError("nans in the array!")

        combined_hash = self._get_array_hash(array, xyz, XYZ)

        if combined_hash in self.cached_ci_results.keys():
            cached = True
            val, pval = self.cached_ci_results[combined_hash]
        else:
            cached = False
            # Get the dependence measure, reycling residuals if need be
            val, _ = self.get_dependence_measure(array, xyz,
                                              type_mask=type_mask) 
            # Get the p-value
            pval = self.get_significance(val, array, xyz, T, dim,
                                        type_mask=type_mask)
            
            self.cached_ci_results[combined_hash] = (val, pval)

        if self.verbosity > 1:
            self._print_cond_ind_results(val=val, pval=pval, cached=cached,
                                         conf=None)
        # Return the value and the pvalue
        return val, pval

    def run_test_raw(self, x, y, z=None, x_type=None, y_type=None, z_type=None, val_only=False):
        """Perform conditional independence test directly on input arrays x, y, z.

        Calls the dependence measure and signficicance test functions. The child
        classes must specify a function get_dependence_measure and either or
        both functions get_analytic_significance and  get_shuffle_significance.

        Parameters
        ----------
        x, y, z : arrays
            x,y,z are of the form (samples, dimension).
            
        type_mask : array-like
            data array of same shape as [x,y,z] which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        Returns
        -------
        val, pval : Tuple of floats

            The test statistic value and the p-value.
        """

        if np.ndim(x) != 2 or np.ndim(y) != 2:
            raise ValueError("x,y must be arrays of shape (samples, dimension)"
                             " where dimension can be 1.")

        if z is not None and np.ndim(z) != 2:
            raise ValueError("z must be array of shape (samples, dimension)"
                             " where dimension can be 1.")

        if x_type is None or y_type is None:
            raise ValueError("x_type and y_type must be set.")

        if z is None:
            # Get the array to test on
            array = np.vstack((x.T, y.T))
            type_mask = np.vstack((x_type.T, y_type.T))

            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])])

        else:
            # Get the array to test on
            array = np.vstack((x.T, y.T, z.T))
            type_mask = np.vstack((x_type.T, y_type.T, z_type.T))

            # xyz is the dimension indicator
            xyz = np.array([0 for i in range(x.shape[1])] +
                           [1 for i in range(y.shape[1])] +
                           [2 for i in range(z.shape[1])])

        # Record the dimensions
        dim, T = array.shape
        # Ensure it is a valid array
        if np.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")
        # Get the dependence measure
        val, _ = self.get_dependence_measure(array, xyz, type_mask=type_mask)

        if val_only:
            return val
        # Get the p-value
        pval = self.get_significance(val, array, xyz, T, dim, type_mask=type_mask)
        # Return the value and the pvalue
        return val, pval
    
    def get_significance(self, val, array, xyz, T, dim,
                         type_mask=None,
                         sig_override=None):
        """
        Returns the p-value from whichever significance function is specified
        for this test.  If an override is used, then it will call a different
        function then specified by self.significance

        Parameters
        ----------
        val : float
            Test statistic value.

        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables

        sig_override : string
            Must be in 'analytic', 'shuffle_test', 'fixed_thres'

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Defaults to the self.significance member value
        use_sig = self.significance
        if sig_override is not None:
            use_sig = sig_override
        # Check if we are using the analytic significance
        if use_sig == 'analytic':
            raise ValueError("Analytic significance not defined for CMIknnMixed!")
        # Check if we are using the shuffle significance
        elif use_sig == 'shuffle_test':
            pval = self.get_shuffle_significance(array=array,
                                                 xyz=xyz,
                                                 value=val,
                                                 type_mask=type_mask)
        # Check if we are using the fixed_thres significance
        elif use_sig == 'fixed_thres':
            pval = self.get_fixed_thres_significance(
                    value=val,
                    fixed_thres=self.fixed_thres)
        else:
            raise ValueError("%s not known." % self.significance)
        # Return the calculated value
        return pval
    
    
    def _compute_discrete_entropy(self, array, disc_values, discrete_idxs, num_samples):
        current_array = array[np.sum(array[:, discrete_idxs] == disc_values, axis=-1) == len(discrete_idxs)]

        count, dim = current_array.shape

        if count == 0:
            return 0.

        prob = float(count) / num_samples
        # print(prob)
        disc_entropy = prob * np.log(prob)
        # print('d', disc_entropy)
        return disc_entropy
    
    
    def compute_discrete_entropy(self, array, disc_values, discrete_idxs, num_samples):
        current_array = array[np.sum(array[:, discrete_idxs] == disc_values, axis=-1) == len(discrete_idxs)]

        count, dim = current_array.shape

        if count == 0:
            return 0.

        prob = float(count) / num_samples
        disc_entropy = prob * np.log(prob)
        return disc_entropy
    
    @jit(forceobj=True)
    def _get_nearest_neighbors_zeroinf_onehot(self, array, xyz, knn,
                                   type_mask=None):
        """Returns nearest neighbors according to Frenzel and Pompe (2007).

        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ. Accepts points as neighbors only 
        if the points are not at infinite distance. 
        Two points have infinite distance when the values for the discrete 
        dimensions of the points do not match.

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
        
        array, xyz, type_mask, discrete_idx_list = self._transform_to_one_hot_mixed(array, 
                                                                                    xyz, 
                                                                                    type_mask,
                                                                                    zero_inf=True)
            
        # Subsample indices
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]
        xz_indices = np.concatenate([x_indices, z_indices])
        yz_indices = np.concatenate([y_indices, z_indices])
     
        # Fit trees
        tree_xyz = spatial.cKDTree(array)
        neighbors = tree_xyz.query(array, k=knn+1, p=np.inf,
                                   distance_upper_bound=9999999)
        
        n, k = neighbors[0].shape
        
        
        epsarray = np.zeros(n)
        for i in range(n):
            if neighbors[0][i, knn] == np.inf:
                replacement_idx = np.where(neighbors[0][i] != np.inf)[0][-1]
                r = max(int(replacement_idx * self.perc), 1)
                epsarray[i] = neighbors[0][i, r]
            else:
                epsarray[i] = neighbors[0][i, knn]
                
        
        neighbors_radius_xyz = tree_xyz.query_ball_point(array, epsarray, p=np.inf)
        
        k_tilde = [len(neighbors_radius_xyz[i]) - 1 if len(neighbors_radius_xyz[i]) > 1 else len(neighbors_radius_xyz[i]) for i in range(len(neighbors_radius_xyz))]
            
        # compute entropies
        xz = array[:, xz_indices]
        tree_xz = spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(xz, r=epsarray, p=np.inf, return_length=True)
        
        yz = array[:, yz_indices]
        tree_yz = spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(yz, r=epsarray, p=np.inf, return_length=True)
            
        if len(z_indices) > 0:
            z = array[:, z_indices]
            tree_z = spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(z, r=epsarray, p=np.inf, return_length=True)
        else:
            # Number of neighbors is T when z is empty.
            k_z = np.full(T, T, dtype='float')
            
        k_xz = np.asarray([i - 1 if i > 1 else i for i in k_xz])
        k_yz = np.asarray([i - 1 if i > 1 else i for i in k_yz])
        k_z = np.asarray([i - 1 if i > 1 else i for i in k_z])

        return k_tilde, k_xz, k_yz, k_z
    
    def get_dependence_measure_zeroinf(self, array, xyz, 
                                   type_mask=None):
        """Returns CMI estimate according to Frenzel and Pompe with an
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

        # compute knn
        if self.knn < 1:
            knn = max(1, int(self.knn*T))
        else:
            knn = max(1, self.knn)
        

        knn_tilde, k_xz, k_yz, k_z = self._get_nearest_neighbors_zeroinf_onehot(array=array,
                                                                                 xyz=xyz,
                                                                                 knn=knn,
                                                                                 type_mask=type_mask)
        non_zero = knn_tilde - k_xz - k_yz + k_z
        
        non_zero_count = np.count_nonzero(non_zero) / len(non_zero)
        
        val = (special.digamma(knn_tilde) - special.digamma(k_xz) -
                                           special.digamma(k_yz) +
                                           special.digamma(k_z))
        
        val = val[np.isfinite(val)].mean() 

        return val, non_zero_count
    
    @jit(forceobj=True)
    def _get_nearest_neighbors_MS_one_hot(self, array, xyz, 
                                          knn, type_mask=None):
        """Returns nearest neighbors according to Messner and Shalizi (2021).

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
        
        discrete_idx_list = np.where(np.all(type_mask == 1, axis=0), 1, 0)
        
        array, xyz, type_mask, discrete_idx_list = self._transform_to_one_hot_mixed(array, 
                                                                                    xyz, 
                                                                                    type_mask)
        
        # Subsample indices
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]

        xz_indices = np.concatenate([x_indices, z_indices])
        yz_indices = np.concatenate([y_indices, z_indices])
            
        # Fit trees
        tree_xyz = spatial.cKDTree(array)
        neighbors = tree_xyz.query(array, k=knn+1, p=np.inf, workers=self.workers)
        
        
        epsarray = neighbors[0][:, -1].astype(np.float64)
        
        neighbors_radius_xyz = tree_xyz.query_ball_point(array, epsarray, p=np.inf, 
                                                         workers=self.workers)
        
        # search again for neighbors in the radius to find all of them
        # in the discrete case k_tilde can be larger than the given knn
        k_tilde = np.asarray([len(neighbors_radius_xyz[i]) - 1 if len(neighbors_radius_xyz[i]) > 1 else len(neighbors_radius_xyz[i]) for i in range(len(neighbors_radius_xyz))])
            
        # compute entropies
        xz = array[:, xz_indices]
        tree_xz = spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(xz, r=epsarray, p=np.inf,
                                        workers=self.workers, return_length=True)
        
        yz = array[:, yz_indices]
        tree_yz = spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(yz, r=epsarray, p=np.inf, 
                                        workers=self.workers, return_length=True)

        if len(z_indices) > 0:
            z = array[:, z_indices]
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
        
        """Returns CMI estimate as described in Messner and Shalizi (2021).
        
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
            knn = max(1, int(self.knn*T))
        else:
            knn = max(1, self.knn)
        
        
        knn_tilde, k_xz, k_yz, k_z = self._get_nearest_neighbors_MS_one_hot(array=array,
                                                                            xyz=xyz,
                                                                            knn=knn,
                                                                            type_mask=type_mask)
        
        non_zero = knn_tilde - k_xz - k_yz + k_z
        
        non_zero_count = np.count_nonzero(non_zero) / len(non_zero)
        
        val = (special.digamma(knn_tilde) - special.digamma(k_xz) -
                                           special.digamma(k_yz) +
                                           special.digamma(k_z))
        val = val[np.isfinite(val)].mean() 

        return val, non_zero_count

    @jit(forceobj=True)
    def _compute_continuous_entropy(self, array, knn):
        """Returns entropy estimate as described by Kozachenko and Leonenko (1987).
        
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        knn : int
            number of nearest-neighbors to use.

        Returns
        -------
        val : float
            Conditional mutual information estimate.
        """
        T, dim = array.shape
        if T == 1:
            return 0.

        if knn < 1:
            knn = max(np.rint(knn * T), 1)

        tree = spatial.cKDTree(array)
        epsarray = tree.query(array, k=[knn+1], p=np.inf, 
                              workers=self.workers,
                              eps=0.)[0][:, 0].astype(np.float64)
        
        epsarray = epsarray[epsarray != 0]
        num_non_zero = len(epsarray)

        if num_non_zero ==  0: 
            cmi_hat = 0.
        else:
            avg_dist = float(array.shape[-1]) / float(num_non_zero) * np.sum(np.log(2 * epsarray))
            cmi_hat = special.digamma(num_non_zero) - special.digamma(knn) + avg_dist

        return cmi_hat
    
    def _compute_entropies_for_discrete_entry(self, array, 
                                              discrete_values, 
                                              discrete_idxs, 
                                              continuous_idxs, 
                                              total_num_samples, 
                                              knn, 
                                              use_local_knn=False):
        """Returns entropy estimates for a given array as described in ... add citation.
        
        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        discrete_values : tuple of dimension (len(discrete_idxs))
            values of discrete variables for which the entropy is computed
        
        discrete_idxs : array of ints
            indices of the dimensions with discrete data
        
        continuous_idxs : array of ints
            indices of the dimensions with continuous data
        
        total_num_samples : int
            total number of samples
        
        knn : int or float
            if int, number of nearest-neighbors to use
            if float, percentage of the number of samples 
            
        use_local_knn : bool (default False)
            if True, the knn is computed as a percentage of the number of samples
            for one realization of the discrete values in each subspace, 
            otherwise the same knn is used for all subspaces.

        Returns
        -------
        val_continuous entropy, val_discrete_entropy : float, float
            Tuple consisting of estimate for the entropy term for the continuous variables,
            and the estimate for the entropy term for the discrete variables.
        """
        
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

    def get_dependence_measure_conditional(self, array, xyz,
                                           type_mask=None):
        """Returns CMI estimate as described in ....
        
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
        if self.knn < 1 and self.use_local_knn == False:
            knn = max(1, int(self.knn*T))
        else:
            knn = self.knn
            
        array = array.astype(np.float64)
        xyz = xyz.astype(np.int32)
        
        array = self._transform_mixed_data(array, type_mask)
        
        array = array.T
        type_mask = type_mask.T
        
        #TODO
        
        # continue working with discrete idx list
        discrete_idx_list = np.where(np.any(type_mask == 1, axis=0), 1, 0)
        
        if np.sum(discrete_idx_list) == 0:
            raise ValueError("Variables are continuous, cannot use CMIknnMixed conditional!")
            
#         if np.sum(discrete_idx_list) != np.sum(any_discrete_idx_list):
#             raise ValueError("Variables contain mixtures, cannot use CMIknnMixed conditional!")
            
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
        num_xz_classes = [np.unique(array[:, xz_indices][:, index]) for index in range(len(discrete_xz_indices)) if (discrete_xz_indices[index] == 1)]
        num_yz_classes = [np.unique(array[:, yz_indices][:, index]) for index in range(len(discrete_yz_indices)) if (discrete_yz_indices[index] == 1)]
        num_z_classes = [np.unique(array[:, z_indices][:, index]) for index in range(len(discrete_z_indices)) if (discrete_z_indices[index] == 1)]
        num_xyz_classes = [np.unique(array[:, index]) for index in range(len(discrete_idx_list)) if (discrete_idx_list[index] == 1)]
        
        # print('num classes', num_xyz_classes, num_xz_classes, num_yz_classes, num_z_classes)siz

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
    
        # print('cartesian', xyz_cartesian_product)
        # , xz_cartesian_product, yz_cartesian_product, z_cartesian_product)
                
        # compute entropies in XYZ subspace 
        if len(xyz_cartesian_product) > 0:
            xyz_cmi = 0.
            xyz_entropy = 0.

            for i, entry in enumerate(xyz_cartesian_product):
                xyz_cont_entropy, xyz_disc_entropy = self._compute_entropies_for_discrete_entry(array, entry,
                                                                       discrete_xyz_idx, 
                                                                       continuous_xyz_idx, 
                                                                       T, knn,
                                                                       self.use_local_knn)
                xyz_cmi += xyz_cont_entropy
                xyz_entropy -= xyz_disc_entropy
        else:
            xyz_cmi = self._compute_continuous_entropy(array, knn)
            xyz_entropy = 0.
            
        # print(xyz_cmi, xyz_entropy)
        
        # compute entropies in XZ subspace
        if len(xz_cartesian_product) > 0:
            xz_cmi = 0.
            xz_entropy = 0.

            for i, entry in enumerate(xz_cartesian_product):
                xz_cont_entropy, xz_disc_entropy = self._compute_entropies_for_discrete_entry(array[:, xz_indices], entry, 
                                                                     discrete_xz_idx, 
                                                                     continuous_xz_idx, 
                                                                     T, knn, 
                                                                     self.use_local_knn)
                xz_cmi += xz_cont_entropy
                xz_entropy -= xz_disc_entropy
        else:
            xz_cmi = self._compute_continuous_entropy(array[:, xz_indices], knn)
            xz_entropy = 0.
            
        # compute entropies in Xy subspace
        if len(yz_cartesian_product) > 0:
            yz_cmi = 0.
            yz_entropy = 0.

            for i, entry in enumerate(yz_cartesian_product):
                yz_cont_entropy, yz_disc_entropy = self._compute_entropies_for_discrete_entry(array[:, yz_indices], entry, 
                                                                     discrete_yz_idx, 
                                                                     continuous_yz_idx, 
                                                                     T, knn, 
                                                                     self.use_local_knn)
                yz_cmi += yz_cont_entropy
                yz_entropy -= yz_disc_entropy
        else:
            yz_cmi = self._compute_continuous_entropy(array[:, yz_indices], knn)
            yz_entropy = 0.
            

        # compute entropies in Z subspace
        if len(z_cartesian_product) > 0:
            z_cmi = 0.
            z_entropy = 0.

            for i, entry in enumerate(z_cartesian_product):
                z_cont_entropy, z_disc_entropy = self._compute_entropies_for_discrete_entry(array[:, z_indices], 
                                                                   entry, 
                                                                   discrete_z_idx, 
                                                                   continuous_z_idx, 
                                                                   T, knn, 
                                                                   self.use_local_knn)
                z_cmi += z_cont_entropy
                z_entropy -= z_disc_entropy
        else:
            z_cmi = self._compute_continuous_entropy(array[:, z_indices], knn)
            z_entropy = 0.
            
        # put it all together for the CMI estimation
        val = xz_cmi + yz_cmi - xyz_cmi - z_cmi + xz_entropy + yz_entropy - xyz_entropy - z_entropy
        
        entropies = (xz_cmi, yz_cmi, xyz_cmi, z_cmi, xz_entropy, yz_entropy, xyz_entropy, z_entropy)
            
        return val, entropies
    
    def get_dependence_measure(self, array, xyz, 
                               type_mask=None):
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
        if type_mask is None:
            raise ValueError("Type mask cannot be none for CMIknnMixed!")
        if np.sum(type_mask) > type_mask.size:
            raise ValueError("Type mask contains other values than 0 and 1!")

        if self.estimator == 'MS':
            return self.get_dependence_measure_MS(array,
                                                  xyz,
                                                  type_mask)
        elif self.estimator == 'cond':
            return self.get_dependence_measure_conditional(array,
                                                           xyz,
                                                           type_mask)
        elif self.estimator == 'FPinf':
            return self.get_dependence_measure_zeroinf(array,
                                                   xyz,
                                                   type_mask)
        else:
            raise ValueError('No such estimator available!')
            
    @jit(forceobj=True)
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

    @jit(forceobj=True)
    def get_shuffle_significance(self, array, xyz, value,
                                 return_null_dist=False,
                                 type_mask=None):
        
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest neighbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.
        
        type_mask : array-like
            data array of same shape as array which describes whether variables
            are continuous or discrete: 0s for continuous variables and 
            1s for discrete variables
        
        Returns
        -------
        pval : float
            p-value
        """
            
        dim, T = array.shape
        z_indices = np.where(xyz == 2)[0]
        
        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            
            array = array.T
            type_mask = type_mask.T

            # discrete_idx_list = np.where(np.all(type_mask == 1, axis=0), 1, 0)

            array, xyz, type_mask, discrete_idx_list = self._transform_to_one_hot_mixed(array, xyz, type_mask,
                                                                                        zero_inf=True)

            # max_neighbors = max(1, int(max_neighbor_ratio*T))
            x_indices = np.where(xyz == 0)[0]
            z_indices = np.where(xyz == 2)[0]
        
            if self.verbosity > 2:
                print("            nearest-neighbor shuffle significance "
                      "test with n = %d and %d surrogates" % (
                      self.shuffle_neighbors, self.sig_samples))
            # Get nearest neighbors around each sample point in Z
            z_array = array[:, z_indices]
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
                array_shuffled, type_mask_shuffled = self._generate_random_permutation(array, 
                                                                                       valid_neighbors, 
                                                                                       x_indices,
                                                                                       type_mask)
                null_dist[sam], _ = self.get_dependence_measure(array_shuffled.T,
                                                                xyz,
                                                                type_mask=type_mask_shuffled.T)

        else:
            null_dist = \
                    self._get_shuffle_dist(array, xyz,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           type_mask=type_mask,
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
            
        n_blks = int(math.floor(float(T)/sig_blocklength))
        
        # print 'n_blks ', n_blks
        if verbosity > 2:
            print("            Significance test with block-length = %d "
                  "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        type_mask_shuffled = np.copy(type_mask)
        # block_starts = np.arange(0, T - sig_blocklength, sig_blocklength)
        block_starts = np.arange(0, n_blks * sig_blocklength, sig_blocklength)
        
    
        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blks*sig_blocklength:]
        
        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):

            blk_starts = self.random_state.permutation(block_starts)[:n_blks]

            x_shuffled = np.zeros((dim_x, n_blks*sig_blocklength),
                                  dtype=array.dtype)
            type_x_shuffled = np.zeros((dim_x, n_blks*sig_blocklength),
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
                                       tail.T, axis=1)
                
    
            for i, index in enumerate(x_indices):
                array_shuffled[index] = x_shuffled[i]
                type_mask_shuffled[index] = type_x_shuffled[i]
                
            null_dist[sam], _ = self.get_dependence_measure(array=array_shuffled,
                                                         xyz=xyz,
                                                         type_mask=type_mask_shuffled)

        return null_dist


if __name__ == '__main__':
    
    import tigramite
    from tigramite.data_processing import DataFrame
    import tigramite.data_processing as pp
    from tigramite.independence_tests import CMIknn
    import numpy as np

    random_state_ = np.random.default_rng(seed=seed)
    cmi = CMIknnMixed(mask_type=None,
                       significance='shuffle_test',
                       # estimator='cond',
                       use_local_knn=True,
                       fixed_thres=None,
                       sig_samples=500,
                       sig_blocklength=1,
                       transform='scale',
                       knn=0.1,
                       verbosity=0)

    # cmiknn = CMIknn(mask_type=None,
    #                    significance='shuffle_test',
    #                    # estimator='FPinf',
    #                    # use_local_knn=True,
    #                    fixed_thres=None,
    #                    sig_samples=500,
    #                    sig_blocklength=1,
    #                    transform='none',
    #                    knn=0.1,
    #                    verbosity=0)


    T = 1000
    dimz = 1

    # Discrete data
    z = random_state_.binomial(n=1, p=0.5, size=(T, dimz)).reshape(T, dimz)
    x = np.empty(T).reshape(T, 1)
    y = np.empty(T).reshape(T, 1)
    for t in range(T):
        val = z[t, 0].squeeze()
        prob = 0.2 + val*0.6
        x[t] = random_state_.choice([0,1], p=[prob, 1.-prob])
        y[t] = random_state_.choice([0,1, 2], p=[prob, (1.-prob)/2., (1.-prob)/2.])

    # Continuous data
    z = random_state_.standard_normal((T, dimz))
    x = (0.5*z[:,0] + random_state_.standard_normal(T)).reshape(T, 1)
    y = (0.5*z[:,0] + random_state_.standard_normal(T)).reshape(T, 1)

    z2 = random_state_.binomial(n=1, p=0.5, size=(T, dimz)).reshape(T, dimz)
    zfull = np.concatenate((z, z2), axis=1)

    print('X _|_ Y')
    print(cmi.run_test_raw(x, y, z=zfull, 
                    x_type=np.zeros(T, dtype='bool'),
                    y_type=np.zeros(T, dtype='bool'),
                    z_type=np.concatenate((np.zeros((T, dimz), dtype='bool'), np.ones((T, dimz), dtype='bool')), axis=1), 
                    # val_only=True)
                    ))

    # print(cmiknn.run_test_raw(x, y, z=None))
        #   
    # print('X _|_ Y | Z')
    # print(cmi.run_test_raw(x, y, z=z, 
    #                 x_type=np.zeros(T, dtype='bool'), 
    #                 y_type=np.zeros(T, dtype='bool'),
    #                 z_type=np.zeros(T, dtype='bool')))

