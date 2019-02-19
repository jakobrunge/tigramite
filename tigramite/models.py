"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from copy import deepcopy

import numpy as np

from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI

try:
    import sklearn
    import sklearn.linear_model
except:
    print("Could not import sklearn...")

try:
    import networkx
except:
    print("Could not import networkx, LinearMediation plots not possible...")



class Models():
    """Base class for time series models.

    Allows to fit any model from sklearn to the parents of a target variable.
    Also takes care of missing values, masking and preprocessing.

    Parameters
    ----------
    dataframe : data object
        Tigramite dataframe object. It must have the attributes dataframe.values
        yielding a numpy array of shape (observations T, variables N) and
        optionally a mask of the same shape and a missing values flag.

    model : sklearn model object
        For example, sklearn.linear_model.LinearRegression() for a linear
        regression model.

    data_transform : sklearn preprocessing object, optional (default: None)
        Used to transform data prior to fitting. For example,
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored.

    mask_type : {'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence measure
        I(X; Y | Z) the samples should be masked. If None, 'y' is used, which
        excludes all time slices containing masked samples in Y. Explained in
        [1]_.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self,
                 dataframe,
                 model,
                 data_transform=sklearn.preprocessing.StandardScaler(),
                 mask_type=None,
                 verbosity=0):
        # Set the mask type and dataframe object
        self.mask_type = mask_type
        self.dataframe = dataframe
        # Get the number of nodes for this dataset
        self.N = self.dataframe.values.shape[1]
        # Set the model to be used
        self.model = model
        # Set the data_transform object and verbosity
        self.data_transform = data_transform
        self.verbosity = verbosity
        # Initialize the object that will be set later
        self.all_parents = None
        self.selected_variables = None
        self.tau_max = None
        self.fit_results = None

    def get_fit(self, all_parents,
                selected_variables=None,
                tau_max=None,
                cut_off='max_lag_or_tau_max',
                return_data=False):
        """Fit time series model.

        For each variable in selected_variables, the sklearn model is fitted
        with :math:`y` given by the target variable, and :math:`X` given by its
        parents. The fitted model class is returned for later use.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the parents estimated with PCMCI.

        selected_variables : list of integers, optional (default: range(N))
            Specify to estimate parents only for selected variables. If None is
            passed, parents are estimated for all variables.

        tau_max : int, optional (default: None)
            Maximum time lag. If None, the maximum lag in all_parents is used.

        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            'max_lag_or_tau_max', which uses the maximum of tau_max and the
            conditions. This is useful to compare multiple models on the same
            sample. Other options are '2xtau_max', which guarantees that MCI
            tests are all conducted on the same samples.  For modeling, can be
            used, Last, 'max_lag' uses as much samples as possible.

        return_data : bool, optional (default: False)
            Whether to save the data array.

        Returns
        -------
        fit_results : dictionary of sklearn model objects for each variable
            Returns the sklearn model after fitting. Also returns the data
            transformation parameters.

        """
        # Initialize the fit by setting the instance's all_parents attribute
        self.all_parents = all_parents
        # Set the default selected variables to all variables and check if this
        # should be overwritten
        self.selected_variables = range(self.N)
        if selected_variables is not None:
            self.selected_variables = selected_variables
        # Find the maximal parents lag
        max_parents_lag = 0
        for j in self.selected_variables:
            if all_parents[j]:
                this_parent_lag = np.abs(np.array(all_parents[j])[:, 1]).max()
                max_parents_lag = max(max_parents_lag, this_parent_lag)
        # Set the default tau max and check if it shoudl be overwritten
        self.tau_max = max_parents_lag
        if tau_max is not None:
            self.tau_max = tau_max
            if self.tau_max < max_parents_lag:
                raise ValueError("tau_max = %d, but must be at least "
                                 " max_parents_lag = %d"
                                 "" % (self.tau_max, max_parents_lag))
        # Initialize the fit results
        fit_results = {}
        for j in self.selected_variables:
            Y = [(j, 0)]
            X = [(j, 0)] # dummy
            Z = self.all_parents[j]
            array, xyz = \
                self.dataframe.construct_array(X, Y, Z,
                                               tau_max=self.tau_max,
                                               mask_type=self.mask_type,
                                               cut_off=cut_off,
                                               verbosity=self.verbosity)
            # Get the dimensions out of the constructed array
            dim, T = array.shape
            dim_z = dim - 2
            # Transform the data if needed
            if self.data_transform is not None:
                array = self.data_transform.fit_transform(X=array.T).T
            # Fit the model if there are any parents for this variable to fit
            if dim_z > 0:
                # Copy and fit the model
                a_model = deepcopy(self.model)
                a_model.fit(X=array[2:].T, y=array[1])
                # Cache the results
                fit_results[j] = {}
                fit_results[j]['model'] = a_model
                # Cache the data transform
                fit_results[j]['data_transform'] = deepcopy(self.data_transform)
                # Cache the data if needed
                if return_data:
                    fit_results[j]['data'] = array
            # If there are no parents, skip this variable
            else:
                fit_results[j] = None

        # Cache and return the fit results
        self.fit_results = fit_results
        return fit_results

    def get_coefs(self):
        """Returns dictionary of coefficients for linear models.

        Only for models from sklearn.linear_model

        Returns
        -------
        coeffs : dictionary
            Dictionary of dictionaries for each variable with keys given by the
            parents and the regression coefficients as values.
        """
        coeffs = {}
        for j in self.selected_variables:
            coeffs[j] = {}
            for ipar, par in enumerate(self.all_parents[j]):
                coeffs[j][par] = self.fit_results[j]['model'].coef_[ipar]
        return coeffs


class LinearMediation(Models):
    r"""Linear mediation analysis for time series models.

    Fits linear model to parents and provides functions to return measures such
    as causal effect, mediated causal effect, average causal effect, etc. as
    described in [4]_.

    Notes
    -----
    This class implements the following causal mediation measures introduced in
    [4]_:

      * causal effect (CE)
      * mediated causal effect (MCE)
      * average causal effect (ACE)
      * average causal susceptibility (ACS)
      * average mediated causal effect (AMCE)

    Consider a simple model of a causal chain as given in the Example with

    .. math:: X_t &= \eta^X_t \\
              Y_t &= 0.5 X_{t-1} +  \eta^Y_t \\
              Z_t &= 0.5 Y_{t-1} +  \eta^Z_t

    Here the link coefficient of :math:`X_{t-2} \to Z_t` is zero while the
    causal effect is 0.25. MCE through :math:`Y` is 0.25 implying that *all*
    of the the CE is explained by :math:`Y`. ACE from :math:`X` is 0.37 since it
    has CE 0.5 on :math:`Y` and 0.25 on :math:`Z`.

    Examples
    --------
    >>> numpy.random.seed(42)
    >>> links_coeffs = {0: [], 1: [((0, -1), 0.5)], 2: [((1, -1), 0.5)]}
    >>> data, true_parents = pp.var_process(links_coeffs, T=1000)
    >>> dataframe = pp.DataFrame(data)
    >>> med = LinearMediation(dataframe=dataframe)
    >>> med.fit_model(all_parents=true_parents, tau_max=3)
    >>> print "Link coefficient (0, -2) --> 2: ", med.get_coeff(i=0, tau=-2, j=2)
    >>> print "Causal effect (0, -2) --> 2: ", med.get_ce(i=0, tau=-2, j=2)
    >>> print "Mediated Causal effect (0, -2) --> 2 through 1: ", med.get_mce(i=0, tau=-2, j=2, k=1)
    >>> print "Average Causal Effect: ", med.get_all_ace()
    >>> print "Average Causal Susceptibility: ", med.get_all_acs()
    >>> print "Average Mediated Causal Effect: ", med.get_all_amce()
    Link coefficient (0, -2) --> 2:  0.0
    Causal effect (0, -2) --> 2:  0.250648072987
    Mediated Causal effect (0, -2) --> 2 through 1:  0.250648072987
    Average Causal Effect:  [ 0.36897445  0.25718002  0.        ]
    Average Causal Susceptibility:  [ 0.          0.24365041  0.38250406]
    Average Mediated Causal Effect:  [ 0.          0.12532404  0.        ]

    References
    ----------
    .. [4]  J. Runge et al. (2015): Identifying causal gateways and mediators in
            complex spatio-temporal systems.
            Nature Communications, 6, 8502. http://doi.org/10.1038/ncomms9502

    Parameters
    ----------
    dataframe : data object
        Tigramite dataframe object. It must have the attributes dataframe.values
        yielding a numpy array of shape (observations T, variables N) and
        optionally a mask of the same shape and a missing values flag.

    model_params : dictionary, optional (default: None)
        Optional parameters passed on to sklearn model

    data_transform : sklearn preprocessing object, optional (default: None)
        Used to transform data prior to fitting. For example,
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored.

    mask_type : {'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence measure
        I(X; Y | Z) the samples should be masked. If None, 'y' is used, which
        excludes all time slices containing masked samples in Y. Explained in
        [1]_.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """
    def __init__(self,
                 dataframe,
                 model_params=None,
                 data_transform=sklearn.preprocessing.StandardScaler(),
                 mask_type=None,
                 verbosity=0):
        # Initialize the member variables to None
        self.phi = None
        self.psi = None
        self.all_psi_k = None

        # Build the model using the parameters
        if model_params is None:
            model_params = {}
        this_model = sklearn.linear_model.LinearRegression(**model_params)
        Models.__init__(self,
                        dataframe=dataframe,
                        model=this_model,
                        data_transform=data_transform,
                        mask_type=mask_type,
                        verbosity=verbosity)

    def fit_model(self, all_parents, tau_max=None):
        """Fit linear time series model.

        Fits a sklearn.linear_model.LinearRegression model to the parents of
        each variable and computes the coefficient matrices :math:`\Phi` and
        :math:`\Psi` as described in [4]_.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the parents estimated with PCMCI.

        tau_max : int, optional (default: None)
            Maximum time lag. If None, the maximum lag in all_parents is used.

        """
        # Fit the model using the base class
        self.fit_results = self.get_fit(all_parents=all_parents,
                                        selected_variables=None,
                                        tau_max=tau_max)
        # Cache the results in the member variables
        coeffs = self.get_coefs()
        self.phi = self._get_phi(coeffs)
        self.psi = self._get_psi(self.phi)
        self.all_psi_k = self._get_all_psi_k(self.phi)

    def _check_sanity(self, X, Y, k=None):
        """Checks validity of some parameters."""

        if len(X) != 1 or len(Y) != 1:
            raise ValueError("X must be of form [(i, -tau)] and Y = [(j, 0)], "
                             "but are X = %s, Y=%s" % (X, Y))

        i, tau = X[0]

        if abs(tau) > self.tau_max:
            raise ValueError("X must be of form [(i, -tau)] with"
                             " tau <= tau_max")

        if k is not None and (k < 0 or k >= self.N):
            raise ValueError("k must be in [0, N)")

    def _get_phi(self, coeffs):
        """Returns the linear coefficient matrices for different lags.different

        Parameters
        ----------
        coeffs : dictionary
            Dictionary of coefficients for each parent.

        Returns
        -------
        phi : array-like, shape (tau_max + 1, N, N)
            Matrices of coefficients for each time lag.
        """

        phi = np.zeros((self.tau_max + 1, self.N, self.N))
        phi[0] = np.identity(self.N)

        for j in list(coeffs):
            for par in list(coeffs[j]):
                i, tau = par
                phi[abs(tau), j, i] = coeffs[j][par]

        return phi

    def _get_psi(self, phi):
        """Returns the linear causal effect matrices for different lags.

        Parameters
        ----------
        phi : array-like
            Coefficient matrices at different lags.

        Returns
        -------
        psi : array-like, shape (tau_max + 1, N, N)
            Matrices of causal effects for each time lag.
        """

        psi = np.zeros((self.tau_max + 1, self.N, self.N))

        psi[0] = np.identity(self.N)
        for n in range(1, self.tau_max + 1):
            psi[n] = np.zeros((self.N, self.N))
            for s in range(1, n+1):
                psi[n] += np.dot(phi[s], psi[n-s])

        return psi

    def _get_psi_k(self, phi, k):
        """Returns the linear causal effect matrices excluding variable k.

        Parameters
        ----------
        phi : array-like
            Coefficient matrices at different lags.

        k : int
            Variable index to exclude causal effects through.

        Returns
        -------
        psi_k : array-like, shape (tau_max + 1, N, N)
            Matrices of causal effects excluding k.
        """

        psi_k = np.zeros((self.tau_max + 1, self.N, self.N))

        psi_k[0] = np.identity(self.N)
        phi_k = np.copy(phi)
        phi_k[1:, k, :] = 0.
        for n in range(1, self.tau_max + 1):
            psi_k[n] = np.zeros((self.N, self.N))
            for s in range(1, n+1):
                psi_k[n] += np.dot(phi_k[s], psi_k[n-s])

        return psi_k

    def _get_all_psi_k(self, phi):
        """Returns the linear causal effect matrices excluding variables.

        Parameters
        ----------
        phi : array-like
            Coefficient matrices at different lags.

        Returns
        -------
        all_psi_k : array-like, shape (N, tau_max + 1, N, N)
            Matrices of causal effects where for each row another variable is
            excluded.
        """

        all_psi_k = np.zeros((self.N, self.tau_max + 1, self.N, self.N))

        for k in range(self.N):
            all_psi_k[k] = self._get_psi_k(phi, k)

        return all_psi_k

    def get_val_matrix(self,):
        """Returns the matrix of linear coefficients.

        Format is val_matrix[i, j, tau] denotes coefficient of link
        i --tau--> j.

        Returns
        -------
        val_matrix : array
            Matrix of linear coefficients, shape (N, N, tau_max + 1).
        """

        return self.phi.transpose()


    def net_to_tsg(self, row, lag, max_lag):
        """Helper function to translate from network to time series graph."""
        return row*max_lag + lag

    def tsg_to_net(self, node, max_lag):
        """Helper function to translate from time series graph to network."""
        row = node // max_lag
        lag = node % max_lag
        return (row, -lag)

    def get_tsg(self, link_matrix, val_matrix=None, include_neighbors = False):
        """Returns time series graph matrix.

        Constructs a matrix of shape (N*tau_max, N*tau_max) from link_matrix.
        This matrix can be used for plotting the time series graph and analyzing
        causal pathways.

        link_matrix : bool array-like, optional (default: None)
            Matrix of significant links. Must be of same shape as val_matrix. Either
            sig_thres or link_matrix has to be provided.

        val_matrix : array_like
            Matrix of shape (N, N, tau_max+1) containing test statistic values.

        include_neighbors : bool, optional (default: False)
            Whether to include causal paths emanating from neighbors of i

        Returns
        -------
        tsg : array of shape (N*tau_max, N*tau_max)
            Time series graph matrix.
        """

        N = len(link_matrix)
        max_lag = link_matrix.shape[2] + 1

        # Create TSG
        tsg = np.zeros((N*max_lag, N*max_lag))
        for i, j, tau in np.column_stack(np.where(link_matrix)):
            if tau > 0 or include_neighbors:
                for t in range(max_lag):
                    link_start = self.net_to_tsg(i, t-tau, max_lag)
                    link_end = self.net_to_tsg(j, t, max_lag)
                    if (0 <= link_start and
                        (link_start % max_lag) <= (link_end  % max_lag)):
                        if val_matrix is not None:
                            tsg[link_start, link_end] = val_matrix[i,j,tau]
                        else:
                            tsg[link_start, link_end] = 1
        return tsg


    def get_mediation_graph_data(self,  i, tau, j, include_neighbors=False):
        r"""Returns link and node weights for mediation analysis.

        Returns array with non-zero entries for links that are on causal
        paths between :math:`i` and :math:`j` at lag :math:`\tau`.
        ``path_val_matrix`` contains the corresponding path coefficients and
        ``path_node_array`` the MCE values. ``tsg_path_val_matrix`` contains the
        corresponding values in the time series graph format.

        Parameters
        ----------
        i : int
            Index of cause variable.

        tau : int
            Lag of cause variable.

        j : int
            Index of effect variable.

        include_neighbors : bool, optional (default: False)
            Whether to include causal paths emanating from neighbors of i

        Returns
        -------
        graph_data : dictionary
            Dictionary of matrices for coloring mediation graph plots.

        """

        path_link_matrix = np.zeros((self.N, self.N, self.tau_max + 1))
        path_val_matrix = np.zeros((self.N, self.N, self.tau_max + 1))

        # Get mediation of path variables
        path_node_array = (self.psi.reshape(1, self.tau_max + 1, self.N, self.N)
                         - self.all_psi_k)[:,abs(tau), j, i]

        # Get involved links
        val_matrix = self.phi.transpose()
        link_matrix = val_matrix != 0.

        max_lag = link_matrix.shape[2] + 1

        # include_neighbors = False because True would allow
        # --> o -- motifs in networkx.all_simple_paths as paths, but
        # these are blocked...
        tsg = self.get_tsg(link_matrix, val_matrix=val_matrix,
                            include_neighbors = False)

        if include_neighbors:
            # Add contemporaneous links only at source node
            for m, n in zip(*np.where(link_matrix[:,:,0])):
                # print m,n
                if m != n:
                    tsg[self.net_to_tsg(m, max_lag-tau-1, max_lag),
                        self.net_to_tsg(n, max_lag-tau-1, max_lag)
                        ] = val_matrix[m, n, 0]

        tsg_path_val_matrix = np.zeros(tsg.shape)

        graph = networkx.DiGraph(tsg)
        pathways = []

        for path in networkx.all_simple_paths(graph,
                    source=self.net_to_tsg(i, max_lag-tau-1, max_lag),
                    target=self.net_to_tsg(j, max_lag-0-1, max_lag)):
            pathways.append([self.tsg_to_net(p, max_lag) for p in path])
            for ip, p in enumerate(path[1:]):
                tsg_path_val_matrix[path[ip], p] = tsg[path[ip], p]

                k, tau_k = self.tsg_to_net(p, max_lag)
                link_start = self.tsg_to_net(path[ip], max_lag)
                link_end = self.tsg_to_net(p, max_lag)
                delta_tau = abs(link_end[1] - link_start[1])
                path_val_matrix[link_start[0],
                                link_end[0],
                                delta_tau] = val_matrix[link_start[0],
                                                        link_end[0],
                                                        delta_tau]

        graph_data = {'path_node_array':path_node_array,
                      'path_val_matrix':path_val_matrix,
                      'tsg_path_val_matrix':tsg_path_val_matrix,}

        return graph_data


    def get_coeff(self, i, tau, j):
        """Returns link coefficient.

        This is the causal effect for a particular link (i, tau) --> j.

        Parameters
        ----------
        i : int
            Index of cause variable.

        tau : int
            Lag of cause variable.

        j : int
            Index of effect variable.

        Returns
        -------
        coeff : float
        """
        return self.phi[abs(tau), j, i]

    def get_ce(self, i, tau, j):
        """Returns the causal effect.

        This is the causal effect for  (i, tau) -- --> j.

        Parameters
        ----------
        i : int
            Index of cause variable.

        tau : int
            Lag of cause variable.

        j : int
            Index of effect variable.

        Returns
        -------
        ce : float
        """
        return self.psi[abs(tau), j, i]

    def get_ce_max(self, i, j):
        """Returns the causal effect.

        This is the maximum absolute causal effect for  i --> j across all lags.

        Parameters
        ----------
        i : int
            Index of cause variable.

        j : int
            Index of effect variable.

        Returns
        -------
        ce : float
        """
        return np.abs(self.psi[1:, j, i]).max()

    def get_mce(self, i, tau, j, k):
        """Returns the mediated causal effect.

        This is the causal effect for  i --> j minus the causal effect not going
        through k.

        Parameters
        ----------
        i : int
            Index of cause variable.

        tau : int
            Lag of cause variable.

        j : int
            Index of effect variable.

        k : int
            Index of mediator variable.

        Returns
        -------
        mce : float
        """
        mce = self.psi[abs(tau), j, i] - self.all_psi_k[k, abs(tau), j, i]
        return mce

    def get_ace(self, i, lag_mode='absmax', exclude_i=True):
        """Returns the average causal effect.

        This is the average causal effect (ACE) emanating from variable i to any
        other variable. With lag_mode='absmax' this is based on the lag of
        maximum CE for each pair.

        Parameters
        ----------
        i : int
            Index of cause variable.

        lag_mode : {'absmax', 'all_lags'}
            Lag mode. Either average across all lags between each pair or only
            at the lag of maximum absolute causal effect.

        exclude_i : bool, optional (default: True)
            Whether to exclude causal effects on the variable itself at later
            lags.

        Returns
        -------
        ace :float
            Average Causal Effect.
        """

        all_but_i = np.ones(self.N, dtype='bool')
        if exclude_i:
            all_but_i[i] = False

        if lag_mode == 'absmax':
            return np.abs(self.psi[1:, all_but_i, i]).max(axis=0).mean()
        elif lag_mode == 'all_lags':
            return np.abs(self.psi[1:, all_but_i, i]).mean()
        else:
            raise ValueError("lag_mode = %s not implemented" % lag_mode)

    def get_all_ace(self, lag_mode='absmax', exclude_i=True):
        """Returns the average causal effect for all variables.

        This is the average causal effect (ACE) emanating from variable i to any
        other variable. With lag_mode='absmax' this is based on the lag of
        maximum CE for each pair.

        Parameters
        ----------
        lag_mode : {'absmax', 'all_lags'}
            Lag mode. Either average across all lags between each pair or only
            at the lag of maximum absolute causal effect.

        exclude_i : bool, optional (default: True)
            Whether to exclude causal effects on the variable itself at later
            lags.

        Returns
        -------
        ace : array of shape (N,)
            Average Causal Effect for each variable.
        """

        ace = np.zeros(self.N)
        for i in range(self.N):
            ace[i] = self.get_ace(i, lag_mode=lag_mode, exclude_i=exclude_i)

        return ace

    def get_acs(self, j, lag_mode='absmax', exclude_j=True):
        """Returns the average causal susceptibility.

        This is the Average Causal Susceptibility (ACS) affecting a variable j
        from any other variable. With lag_mode='absmax' this is based on the lag
        of maximum CE for each pair.

        Parameters
        ----------
        j : int
            Index of variable.

        lag_mode : {'absmax', 'all_lags'}
            Lag mode. Either average across all lags between each pair or only
            at the lag of maximum absolute causal effect.

        exclude_j : bool, optional (default: True)
            Whether to exclude causal effects on the variable itself at previous
            lags.

        Returns
        -------
        acs : float
            Average Causal Susceptibility.
        """

        all_but_j = np.ones(self.N, dtype='bool')
        if exclude_j:
            all_but_j[j] = False

        if lag_mode == 'absmax':
            return np.abs(self.psi[1:, j, all_but_j]).max(axis=0).mean()
        elif lag_mode == 'all_lags':
            return np.abs(self.psi[1:, j, all_but_j]).mean()
        else:
            raise ValueError("lag_mode = %s not implemented" % lag_mode)

    def get_all_acs(self, lag_mode='absmax', exclude_j=True):
        """Returns the average causal susceptibility.

        This is the Average Causal Susceptibility (ACS) for each variable from
        any other variable. With lag_mode='absmax' this is based on the lag of
        maximum CE for each pair.

        Parameters
        ----------
        lag_mode : {'absmax', 'all_lags'}
            Lag mode. Either average across all lags between each pair or only
            at the lag of maximum absolute causal effect.

        exclude_j : bool, optional (default: True)
            Whether to exclude causal effects on the variable itself at previous
            lags.

        Returns
        -------
        acs : array of shape (N,)
            Average Causal Susceptibility.
        """

        acs = np.zeros(self.N)
        for j in range(self.N):
            acs[j] = self.get_acs(j, lag_mode=lag_mode, exclude_j=exclude_j)

        return acs

    def get_amce(self, k, lag_mode='absmax',
                    exclude_k=True, exclude_self_effects=True):
        """Returns the average mediated causal effect.

        This is the Average Mediated Causal Effect (AMCE) through a variable k
        With lag_mode='absmax' this is based on the lag of maximum CE for each
        pair.

        Parameters
        ----------
        k : int
            Index of variable.

        lag_mode : {'absmax', 'all_lags'}
            Lag mode. Either average across all lags between each pair or only
            at the lag of maximum absolute causal effect.

        exclude_k : bool, optional (default: True)
            Whether to exclude causal effects through the variable itself at
            previous lags.

        exclude_self_effects : bool, optional (default: True)
            Whether to exclude causal self effects of variables on themselves.

        Returns
        -------
        amce : float
            Average Mediated Causal Effect.
        """

        all_but_k = np.ones(self.N, dtype='bool')
        if exclude_k:
            all_but_k[k] = False
            N_new = self.N - 1
        else:
            N_new = self.N

        if exclude_self_effects:
            weights = np.identity(N_new) == False
        else:
            weights = np.ones((N_new, N_new), dtype = 'bool')

        if self.tau_max < 2:
            raise ValueError("Mediation only nonzero for tau_max >= 2")

        all_mce = self.psi[2:, :, :] - self.all_psi_k[k, 2:, :, :]
        # all_mce[:, range(self.N), range(self.N)] = 0.

        if lag_mode == 'absmax':
            return np.average(np.abs(all_mce[:, all_but_k, :]
                                                  [:, :, all_but_k]
                                                ).max(axis=0), weights=weights)
        elif lag_mode == 'all_lags':
            return np.abs(all_mce[:, all_but_k, :][:,:, all_but_k]).mean()
        else:
            raise ValueError("lag_mode = %s not implemented" % lag_mode)


    def get_all_amce(self, lag_mode='absmax',
                    exclude_k=True, exclude_self_effects=True):
        """Returns the average mediated causal effect.

        This is the Average Mediated Causal Effect (AMCE) through all variables
        With lag_mode='absmax' this is based on the lag of maximum CE for each
        pair.

        Parameters
        ----------
        lag_mode : {'absmax', 'all_lags'}
            Lag mode. Either average across all lags between each pair or only
            at the lag of maximum absolute causal effect.

        exclude_k : bool, optional (default: True)
            Whether to exclude causal effects through the variable itself at
            previous lags.

        exclude_self_effects : bool, optional (default: True)
            Whether to exclude causal self effects of variables on themselves.

        Returns
        -------
        amce : array of shape (N,)
            Average Mediated Causal Effect.
        """
        amce = np.zeros(self.N)
        for k in range(self.N):
            amce[k] = self.get_amce(k,
                                    lag_mode=lag_mode,
                                    exclude_k=exclude_k,
                                    exclude_self_effects=exclude_self_effects)

        return amce


class Prediction(Models, PCMCI):
    r"""Prediction class for time series models.

    Allows to fit and predict from any sklearn model. The optimal predictors can
    be estimated using PCMCI. Also takes care of missing values, masking and
    preprocessing.

    Parameters
    ----------
    dataframe : data object
        Tigramite dataframe object. It must have the attributes dataframe.values
        yielding a numpy array of shape (observations T, variables N) and
        optionally a mask of the same shape and a missing values flag.

    train_indices : array-like
        Either boolean array or time indices marking the training data.

    test_indices : array-like
        Either boolean array or time indices marking the test data.

    prediction_model : sklearn model object
        For example, sklearn.linear_model.LinearRegression() for a linear
        regression model.

    cond_ind_test : Conditional independence test object, optional
        Only needed if predictors are estimated with causal algorithm.
        The class will be initialized with masking set to the training data.

    data_transform : sklearn preprocessing object, optional (default: None)
        Used to transform data prior to fitting. For example,
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """
    def __init__(self,
                 dataframe,
                 train_indices,
                 test_indices,
                 prediction_model,
                 cond_ind_test=None,
                 data_transform=None,
                 verbosity=0):

        # Default value for the mask
        mask = dataframe.mask
        if mask is None:
            mask = np.zeros(dataframe.values.shape, dtype='bool')
        # Get the dataframe shape
        T = len(dataframe.values)
        # Have the default dataframe be the training data frame
        train_mask = np.copy(mask)
        train_mask[[t for t in range(T) if t not in train_indices]] = True
        self.dataframe = DataFrame(dataframe.values,
                                   mask=train_mask,
                                   missing_flag=dataframe.missing_flag)
        # Initialize the models baseclass with the training dataframe
        Models.__init__(self,
                        dataframe=self.dataframe,
                        model=prediction_model,
                        data_transform=data_transform,
                        mask_type='y',
                        verbosity=verbosity)

        # Build the testing dataframe as well
        self.test_mask = np.copy(mask)
        self.test_mask[[t for t in range(T) if t not in test_indices]] = True

        # Setup the PCMCI instance
        if cond_ind_test is not None:
            # Force the masking
            cond_ind_test.set_mask_type('y')
            cond_ind_test.verbosity = verbosity
            PCMCI.__init__(self,
                           dataframe=self.dataframe,
                           cond_ind_test=cond_ind_test,
                           selected_variables=None,
                           verbosity=verbosity)

        # Set the member variables
        self.cond_ind_test = cond_ind_test
        # Initialize member varialbes that are set outside
        self.target_predictors = None
        self.selected_targets = None
        self.fitted_model = None
        self.test_array = None

    def get_predictors(self,
                       selected_targets=None,
                       selected_links=None,
                       steps_ahead=1,
                       tau_max=1,
                       pc_alpha=0.2,
                       max_conds_dim=None,
                       max_combinations=1):
        """Estimate predictors using PC1 algorithm.

        Wrapper around PCMCI.run_pc_stable that estimates causal predictors.
        The lead time can be specified by ``steps_ahead``.

        Parameters
        ----------
        selected_targets : list of ints, optional (default: None)
            List of variables to estimate predictors of. If None, predictors of
            all variables are estimated.

        selected_links : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested

        steps_ahead : int, default: 1
            Minimum time lag to test. Useful for multi-step ahead predictions.

        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.

        pc_alpha : float or list of floats, default: 0.2
            Significance level in algorithm. If a list or None is passed, the
            pc_alpha level is optimized for every variable across the given
            pc_alpha values using the score computed in
            cond_ind_test.get_model_selection_criterion()

        max_conds_dim : int or None
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.

        max_combinations : int, default: 1
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.

        Returns
        -------
        predictors : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated predictors.
        """
        # Ensure an independence model is given
        if self.cond_ind_test is None:
            raise ValueError("No cond_ind_test given!")
        # Set the selected variables
        self.selected_variables = range(self.N)
        if selected_targets is not None:
            self.selected_variables = selected_targets
        predictors = self.run_pc_stable(selected_links=selected_links,
                                        tau_min=steps_ahead,
                                        tau_max=tau_max,
                                        save_iterations=False,
                                        pc_alpha=pc_alpha,
                                        max_conds_dim=max_conds_dim,
                                        max_combinations=max_combinations)
        return predictors

    def fit(self, target_predictors,
            selected_targets=None, tau_max=None, return_data=False):
        r"""Fit time series model.

        Wrapper around ``Models.get_fit()``. To each variable in
        ``selected_targets``, the sklearn model is fitted with :math:`y` given
        by the target variable, and :math:`X` given by its predictors. The
        fitted model class is returned for later use.

        Parameters
        ----------
        target_predictors : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the predictors estimated with PCMCI.

        selected_targets : list of integers, optional (default: range(N))
            Specify to fit model only for selected targets. If None is
            passed, models are estimated for all variables.

        tau_max : int, optional (default: None)
            Maximum time lag. If None, the maximum lag in target_predictors is
            used.

        return_data : bool, optional (default: False)
            Whether to save the data array.

        Returns
        -------
        self : instance of self
        """

        self.target_predictors = target_predictors

        if selected_targets is None:
            self.selected_targets = range(self.N)
        else:
            self.selected_targets = selected_targets

        for target in self.selected_targets:
            if target not in list(self.target_predictors):
                raise ValueError("No predictors given for target %s" % target)

        self.fitted_model = \
            self.get_fit(all_parents=self.target_predictors,
                         selected_variables=self.selected_targets,
                         tau_max=tau_max,
                         return_data=return_data)
        return self

    def predict(self, target,
                new_data=None,
                pred_params=None,
                cut_off='max_lag_or_tau_max'):
        r"""Predict target variable with fitted model.

        Uses the model.predict() function of the sklearn model.

        Parameters
        ----------
        target : int
            Index of target variable.

        new_data : data object, optional
            New Tigramite dataframe object with optional new mask.

        pred_params : dict, optional
            Optional parameters passed on to sklearn prediction function.

        cut_off : {'2xtau_max', 'max_lag', 'max_lag_or_tau_max'}
            How many samples to cutoff at the beginning. The default is
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples.  For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample. Last,
            'max_lag' uses as much samples as possible.

        Returns
        -------
        Results from prediction.
        """
        # Print message
        if self.verbosity > 0:
            print("\n##\n## Predicting target %s\n##" % target)
            if pred_params is not None:
                for key in list(pred_params):
                    print("%s = %s" % (key, pred_params[key]))
        # Default value for pred_params
        if pred_params is None:
            pred_params = {}
        # Check this is a valid target
        if target not in self.selected_targets:
            raise ValueError("Target %s not yet fitted" % target)
        # Construct the array form of the data
        Y = [(target, 0)]
        X = [(target, 0)] # dummy
        Z = self.target_predictors[target]
        # Check if we've passed a new dataframe object
        test_array = None
        if new_data is not None:
            test_array, _ = new_data.construct_array(X, Y, Z,
                                                     tau_max=self.tau_max,
                                                     mask_type=self.mask_type,
                                                     cut_off=cut_off,
                                                     verbosity=self.verbosity)
        # Otherwise use the default values
        else:
            test_array, _ = \
                self.dataframe.construct_array(X, Y, Z,
                                               tau_max=self.tau_max,
                                               mask=self.test_mask,
                                               mask_type=self.mask_type,
                                               cut_off=cut_off,
                                               verbosity=self.verbosity)
        # Transform the data if needed
        a_transform = self.fitted_model[target]['data_transform']
        if a_transform is not None:
            test_array = a_transform.transform(X=test_array.T).T
        # Cache the test array
        self.test_array = test_array
        # Run the predictor
        pred = self.fitted_model[target]['model'].predict(X=test_array[2:].T,
                                                          **pred_params)
        return pred

    def get_train_array(self, j):
        """Returns training array."""
        return self.fitted_model[j]['data']

    def get_test_array(self):
        """Returns test array."""
        return self.test_array
