"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0

import numpy
import sys, os
import math

from scipy import linalg, special, stats
from copy import deepcopy

from tigramite.independence_tests import _construct_array

try:
    import sklearn
    import sklearn.linear_model
except:
    print("Could not import sklearn...")

try:
    import networkx
except:
    print("Could not import networkx, LinearMediation plots not possible...")


from tigramite.pcmci import PCMCI

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
        For example, sklearn.linear_model.LinearRegression for a linear
        regression model.

    model_params : dictionary, optional (default: None)
        Optional parameters passed on to sklearn model

    data_transform : sklearn preprocessing object, optional (default: None)
        Used to transform data prior to fitting. For example, 
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored.
    
    use_mask : bool, optional (default: False)
        Whether a supplied mask should be used.

    mask_type : {'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence measure
        I(X; Y | Z) the samples should be masked. If None, 'y' is used, which
        excludes all time slices containing masked samples in Y. Explained in
        [1]_.

    missing_flag : number, optional (default: None)
        Flag for missing values. Dismisses all time slices of samples where
        missing values occur in any variable and also flags samples for all lags
        up to 2*tau_max. This avoids biases, see section on masking in
        Supplement of [1]_.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self, 
            dataframe,
            model,
            model_params=None,
            data_transform=None,
            use_mask=False,
            mask_type=None,
            missing_flag=None,
            verbosity=0,
            ):

        self.use_mask = use_mask
        self.mask_type = mask_type
        self.missing_flag = missing_flag

        self.dataframe = dataframe
        self.data = dataframe.values
        self.mask = dataframe.mask

        self.N = self.data.shape[1]

        self.model = model
        self.model_params = model_params
        if self.model_params is None:
            self.model_params = {}

        self.data_transform = data_transform

        self.verbosity = verbosity


    def get_fit(self, all_parents, selected_variables=None,
                 tau_max=None, cut_off='max_lag_or_tau_max',
                 return_data=False):
        """Fit time series model.

        To each variable in selected_variables, the sklearn model is fitted
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
            '2xtau_max', which guarantees that MCI tests are all conducted on
            the same samples.  For modeling, 'max_lag_or_tau_max' can be used,
            which uses the maximum of tau_max and the conditions, which is
            useful to compare multiple models on the same sample. Last,
            'max_lag' uses as much samples as possible.

        return_data : bool, optional (default: False)
            Whether to save the data array.

        Returns
        -------
        fit_results : dictionary of sklearn model objects for each variable
            Returns the sklearn model after fitting. Also returns the data
            transformation parameters.

        """

        self.all_parents = all_parents

        if selected_variables is None:
            self.selected_variables = range(self.N)
        else:
            self.selected_variables = selected_variables

        max_parents_lag = 0
        for j in self.selected_variables:
            # print self.all_parents[j]
            if len(all_parents[j]) > 0:
                max_parents_lag = max(max_parents_lag, numpy.abs(numpy.array(
                        all_parents[j])[:,1]).max())

        if tau_max is None:
            self.tau_max = max_parents_lag
        else:
            self.tau_max = tau_max
            if self.tau_max < max_parents_lag:
                raise ValueError("tau_max = %d, but must be at least "
                                 " max_parents_lag = %d"
                                 "" % (self.tau_max, max_parents_lag))

        fit_results = {}
        for j in self.selected_variables:
            Y = [(j, 0)]
            X = [(j, 0)] # dummy
            Z = self.all_parents[j]
            array, xyz = _construct_array(X, Y, Z, 
                                    tau_max=self.tau_max,
                                    data=self.data,
                                    use_mask=self.use_mask,
                                    mask=self.mask, 
                                    mask_type=self.mask_type,
                                    missing_flag=self.missing_flag,
                                    cut_off=cut_off,
                                    verbosity=self.verbosity)

            dim, T = array.shape
            dim_z = dim - 2

            if self.data_transform is not None:
                array = self.data_transform.fit_transform(X=array.T).T

            if dim_z > 0:
                model = self.model(**self.model_params)

                # print array[2:].T
                # print array[1].reshape(T, 1)
                model.fit(X=array[2:].T, y=array[1])

                fit_results[j] = {}
                fit_results[j]['model'] = model

                if self.data_transform is not None:
                    fit_results[j]['data_transform'] = deepcopy(self.data_transform)

                if return_data:
                    fit_results[j]['data'] = array
                # print j, model.coef_, model.intercept_

            else:
                fit_results[j] = None

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
                # print ipar, par, self.fit_results[j].coef_ , self.fit_results[j].coef_.shape
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
    
    use_mask : bool, optional (default: False)
        Whether a supplied mask should be used.

    mask_type : {'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence measure
        I(X; Y | Z) the samples should be masked. If None, 'y' is used, which
        excludes all time slices containing masked samples in Y. Explained in
        [1]_.

    missing_flag : number, optional (default: None)
        Flag for missing values. Dismisses all time slices of samples where
        missing values occur in any variable and also flags samples for all lags
        up to 2*tau_max. This avoids biases, see section on masking in
        Supplement of [1]_.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self, 
            dataframe,
            model_params=None,
            data_transform=None,
            use_mask=False,
            mask_type=None,
            missing_flag=None,
            verbosity=0,
            ):


        if data_transform is None:
            data_transform=sklearn.preprocessing.StandardScaler()
        elif data_transform == False:
            class noscaler():
                def __init__(self):
                    pass
                def fit_transform(self, X):
                    return X
                def transform(self, X):
                    return X
            data_transform = noscaler()

        Models.__init__(self, 
            dataframe=dataframe,
            model=sklearn.linear_model.LinearRegression,
            model_params=model_params,
            data_transform=data_transform,
            use_mask=use_mask,
            mask_type=mask_type,
            missing_flag=missing_flag,
            verbosity=verbosity)


    def fit_model(self, all_parents,
                     tau_max=None,):
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

        self.fit_results = self.get_fit(all_parents=all_parents,
                                        selected_variables=None,
                                        tau_max=tau_max
                                        )

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

        phi = numpy.zeros((self.tau_max + 1, self.N, self.N))
        phi[0] = numpy.identity(self.N)

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

        psi = numpy.zeros((self.tau_max + 1, self.N, self.N))

        psi[0] = numpy.identity(self.N)
        for n in range(1, self.tau_max + 1):
            psi[n] = numpy.zeros((self.N, self.N)) 
            for s in range(1, n+1):
                psi[n] += numpy.dot(phi[s], psi[n-s])

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

        psi_k = numpy.zeros((self.tau_max + 1, self.N, self.N))

        psi_k[0] = numpy.identity(self.N)
        phi_k = numpy.copy(phi)
        # print phi[1]
        phi_k[1:, k, :] = 0.
        # print phi_k[1]
        for n in range(1, self.tau_max + 1):
            psi_k[n] = numpy.zeros((self.N, self.N)) 
            for s in range(1, n+1):
                psi_k[n] += numpy.dot(phi_k[s], psi_k[n-s])

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

        all_psi_k = numpy.zeros((self.N, self.tau_max + 1, self.N, self.N))

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
        tsg = numpy.zeros((N*max_lag, N*max_lag))
        for i, j, tau in numpy.column_stack(numpy.where(link_matrix)):
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

        path_link_matrix = numpy.zeros((self.N, self.N, self.tau_max + 1))
        path_val_matrix = numpy.zeros((self.N, self.N, self.tau_max + 1))

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
            for m, n in zip(*numpy.where(link_matrix[:,:,0])):
                # print m,n
                if m != n:
                    tsg[self.net_to_tsg(m, max_lag-tau-1, max_lag),
                        self.net_to_tsg(n, max_lag-tau-1, max_lag)
                        ] = val_matrix[m, n, 0]

        tsg_path_val_matrix = numpy.zeros(tsg.shape)

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
        return numpy.abs(self.psi[1:, j, i]).max()

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

        all_but_i = numpy.ones(self.N, dtype='bool')
        if exclude_i:
            all_but_i[i] = False

        if lag_mode == 'absmax':
            return numpy.abs(self.psi[1:, all_but_i, i]).max(axis=0).mean()
        elif lag_mode == 'all_lags':
            return numpy.abs(self.psi[1:, all_but_i, i]).mean()
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

        ace = numpy.zeros(self.N)
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

        all_but_j = numpy.ones(self.N, dtype='bool')
        if exclude_j:
            all_but_j[j] = False

        if lag_mode == 'absmax':
            return numpy.abs(self.psi[1:, j, all_but_j]).max(axis=0).mean()
        elif lag_mode == 'all_lags':
            return numpy.abs(self.psi[1:, j, all_but_j]).mean()
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

        acs = numpy.zeros(self.N)
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

        all_but_k = numpy.ones(self.N, dtype='bool')
        if exclude_k:
            all_but_k[k] = False
            N_new = self.N - 1
        else:
            N_new = self.N

        if exclude_self_effects:
            weights = numpy.identity(N_new) == False
        else:
            weights = numpy.ones((N_new, N_new), dtype = 'bool')

        if self.tau_max < 2:
            raise ValueError("Mediation only nonzero for tau_max >= 2")

        all_mce = self.psi[2:, :, :] - self.all_psi_k[k, 2:, :, :] 
        # all_mce[:, range(self.N), range(self.N)] = 0.

        if lag_mode == 'absmax':
            return numpy.average(numpy.abs(all_mce[:, all_but_k, :]
                                                  [:, :, all_but_k]
                                                ).max(axis=0), weights=weights)
        elif lag_mode == 'all_lags':
            return numpy.abs(all_mce[:, all_but_k, :][:,:, all_but_k]).mean()
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
        amce = numpy.zeros(self.N)
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
        For example, sklearn.linear_model.LinearRegression for a linear
        regression model.

    prediction_model_params : dictionary, optional (default: None)
        Optional parameters passed on to sklearn model

    cond_ind_model : Conditional independence test class object, optional
        Only needed if predictors are estimated with causal algorithm.
        The class will be initialized with masking set to the training data. 
        Further parameters can be supplied by cond_ind_params.

    cond_ind_params : dictionary, optional
        Parameters for conditional independence test.

    data_transform : sklearn preprocessing object, optional (default: None)
        Used to transform data prior to fitting. For example, 
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored.

    missing_flag : number, optional (default: None)
        Flag for missing values. Dismisses all time slices of samples where
        missing values occur in any variable and also flags samples for all lags
        up to 2*tau_max. This avoids biases, see section on masking in
        Supplement of [1]_.

    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self, 
            dataframe,
            train_indices,
            test_indices,
            prediction_model,
            prediction_model_params=None,
            cond_ind_model=None,
            cond_ind_params=None,
            data_transform=None,
            missing_flag=None,
            verbosity=0,
            ):


        Models.__init__(self, 
            dataframe=dataframe,
            model=prediction_model,
            model_params=prediction_model_params,
            data_transform=data_transform,
            use_mask=True,
            mask_type='y',
            missing_flag=missing_flag,
            verbosity=verbosity)

        if cond_ind_params is None:
            cond_ind_params = {}

        if cond_ind_model is not None:
            cond_ind_test = cond_ind_model(
                    use_mask=True,
                    mask_type='y',
                    verbosity=verbosity,
                    **cond_ind_params)
        else:
            cond_ind_test = None

        if dataframe.mask is None:
            mask = numpy.zeros(dataframe.values.shape, dtype='bool')
        else:
            mask = dataframe.mask

        T = len(dataframe.values)

        self.train_mask = numpy.copy(mask)
        self.train_mask[[t for t in range(T) if t not in train_indices]] = True

        self.test_mask = numpy.copy(mask)
        self.test_mask[[t for t in range(T) if t not in test_indices]] = True

        dataframe_here = deepcopy(dataframe)
        dataframe_here.mask = self.train_mask

        if cond_ind_test is not None:
            PCMCI.__init__(self,
                dataframe=dataframe_here,
                cond_ind_test=cond_ind_test,
                selected_variables=None,
                var_names=None,
                verbosity=verbosity)

        self.cond_ind_model = cond_ind_model
        self.prediction_model = prediction_model
        self.prediction_model_params = prediction_model_params

    def get_predictors(self,
                      selected_targets=None,
                      selected_links=None,
                      steps_ahead=1,
                      tau_max=1,
                      pc_alpha=0.2,
                      max_conds_dim=None,
                      max_combinations=1,
                      ):
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

        if self.cond_ind_model is None:
            raise ValueError("No cond_ind_model given!")

        if selected_targets is None:
            self.selected_variables = range(self.N)
        else:      
            self.selected_variables = selected_targets

        self.mask = self.train_mask

        predictors = self.run_pc_stable(
                      selected_links=selected_links,
                      tau_min=steps_ahead,
                      tau_max=tau_max,
                      save_iterations=False,
                      pc_alpha=pc_alpha,
                      max_conds_dim=max_conds_dim,
                      max_combinations=max_combinations,
                      )

        return predictors



    def fit(self, target_predictors, selected_targets=None,
                        tau_max=None, return_data=False):
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

        # Set mask to train_indices
        self.mask = self.train_mask

        self.fitted_model = self.get_fit(
                                all_parents=self.target_predictors,
                                selected_variables=self.selected_targets,
                                tau_max=tau_max,
                                return_data=return_data,
                                )

        return self

    
    def predict(self, target, new_data=None, pred_params=None,
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

        if self.verbosity > 0:
            print("\n##\n## Predicting target %s\n##" % target)
            if pred_params is not None:
                for key in list(pred_params):
                    print("%s = %s" % (key, pred_params[key]))

        if pred_params is None:
            pred_params = {}

        if new_data is None:
            data = self.dataframe.values
            # Set mask to test_indices
            mask = self.test_mask
        else:
            data = new_data.values
            if new_data.mask is not None:
                mask = new_data.mask
            else:
                mask = numpy.zeros(data.shape)


        if target not in self.selected_targets:
            raise ValueError("Target %s not yet fitted" % target)

        Y = [(target, 0)]
        X = [(target, 0)] # dummy
        Z = self.target_predictors[target]

        array, xyz = _construct_array(X, Y, Z, 
                                tau_max=self.tau_max,
                                data=data,
                                use_mask=self.use_mask,
                                mask=mask, 
                                mask_type=self.mask_type,
                                missing_flag=self.missing_flag,
                                cut_off=cut_off,
                                verbosity=self.verbosity)

        if self.data_transform is not None:
            # print array.shape
            # print self.fitted_model[target]['data_transform'].mean_
            array = self.fitted_model[target]['data_transform'].transform(X=array.T).T

        self.test_array = array
        # print array.shape

        pred = self.fitted_model[target]['model'].predict(X=array[2:].T, **pred_params)

        return pred


    def get_train_array(self, j):
        """Returns training array."""
        return self.fitted_model[j]['data']

    def get_test_array(self):
        """Returns test array."""
        return self.test_array


if __name__ == '__main__':


    import tigramite.data_processing as pp

    # ###
    # ### Models class
    # ###

    # numpy.random.seed(44)
    # a = 0.9
    # c = 0.5
    # T = 1000

    # verbosity = 0
    # # Each key refers to a variable and the incoming links are supplied as a
    # # list of format [((driver, lag), coeff), ...]
    # links_coeffs = {0: [((0, -1), a)],
    #                 1: [((1, -1), a), ((0, -1), c)],
    #                 2: [((2, -1), a), ((1, -1), c), ((0, -1), c)],
    #                 }

    # data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)

    # # Assume true parents are known
    # all_parents = true_parents_neighbors
    # print all_parents

    # # data_mask = numpy.zeros(data.shape)
    # dataframe = pp.DataFrame(data)


    # import sklearn
    # import sklearn.linear_model

    # model = Models(dataframe=dataframe, 
    #     model = sklearn.linear_model.LinearRegression,
    #     # model = sklearn.gaussian_process.GaussianProcessRegressor,
    #     # model_params={'fit_intercept':False},
    #     # model_params = {'alpha':0., 'kernel':sklearn.gaussian_process.kernels.RBF() +
    #     #                                     sklearn.gaussian_process.kernels.WhiteKernel()},
    #     # data_transform=sklearn.preprocessing.StandardScaler(),

    #     )
    
    # results = model.get_fit(all_parents=all_parents, 
    #                         )

    # for j in list(results):
    #     print results[j]['model']  #.coef_


    ###
    ### Linear Mediation
    ###
    numpy.random.seed(42)
    links_coeffs = {0: [((0, -1), 0.6), ((1, 0), 0.6)],
                    1: [((1, -1), 0.8), ((0, -1), 0.5), ((0, 0), 0.6)],
                    2: [((2, -1), 0.9), ((1, -1), 0.5), ((0, -2), 0.)],
                    }

    data, true_parents = pp.var_process(links_coeffs, T=1000)
    dataframe = pp.DataFrame(data)
    med = LinearMediation(dataframe=dataframe, data_transform=False)
    med.fit_model(all_parents=true_parents, tau_max=4)
    print ("Link coefficient (0, -2) --> 2: ", med.get_coeff(i=0, tau=-2, j=2))
    print ("Causal effect (0, -2) --> 2: ", med.get_ce(i=0, tau=-2, j=2))
    print ("Mediated Causal effect (0, -2) --> 2 through 1: ", med.get_mce(i=0, tau=-2, j=2, k=1))
    print ("Average Causal Effect: ", med.get_all_ace())
    print ("Average Causal Susceptibility: ", med.get_all_acs())
    print ("Average Mediated Causal Effect: ", med.get_all_amce())

    i=0; tau=4; j=2

    graph_data = med.get_mediation_graph_data(i=i, tau=tau, j=j, 
                                            include_neighbors=True)

    import plotting as tp


    tp.plot_mediation_graph(
                var_names=['X', 'Y', 'Z'],
                path_val_matrix=graph_data['path_val_matrix'], 
               path_node_array=graph_data['path_node_array'],
               save_name="/home/jakobrunge/test/test_graph.pdf"
                )

    tp.plot_mediation_time_series_graph(
        var_names=['X', 'Y', 'Z'],
        path_node_array=graph_data['path_node_array'],
        tsg_path_val_matrix=graph_data['tsg_path_val_matrix'],
        # vmin_edges=-0.5, vmax_edges=0.5, edge_ticks=0.1,
        # vmin_nodes=-0.5, vmax_nodes=0.5, node_ticks=0.1,
        save_name="/home/jakobrunge/test/test_tsg_graph.pdf"
        )

    # ##
    # ## Prediction
    # ##
    # import pylab

    # numpy.random.seed(44)
    # a = 0.4
    # c = 0.6
    # T = 200

    # verbosity = 0
    # # Each key refers to a variable and the incoming links are supplied as a
    # # list of format [((driver, lag), coeff), ...]
    # links_coeffs = {0: [((0, -1), a)],
    #                 1: [((1, -1), a), ((0, -1), c)],
    #                 2: [((2, -1), a), ((1, -1), c)],  # ((0, -1), c)],
    #                 }

    # data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)

    # # print data
    # # print data.mean(axis=0), data.std(axis=0)
    # data_mask = numpy.zeros(data.shape)

    # dataframe = pp.DataFrame(data)

    # from tigramite.independence_tests import ParCorr
    # # import sklearn
    # # import sklearn.preprocessing
    # # import sklearn.neighbors
    # # import sklearn.linear_model

    # pred = Prediction(dataframe=dataframe,
    #         cond_ind_model=ParCorr,
    #         cond_ind_params = {'significance':'analytic',
    #                            'fixed_thres':0.01},
    #         prediction_model = sklearn.linear_model.LinearRegression,
    #         # prediction_model = sklearn.gaussian_process.GaussianProcessRegressor,
    #         # prediction_model = sklearn.neighbors.KNeighborsRegressor,
    #     # prediction_model_params={'fit_intercept':False},
    #     # prediction_model_params = {'n_neighbors':5},
    #     # prediction_model_params = {'alpha':0., 'kernel':sklearn.gaussian_process.kernels.RBF() +
    #     #                                     sklearn.gaussian_process.kernels.WhiteKernel()},
    #     # data_transform=sklearn.preprocessing.StandardScaler(),
    #     train_indices= range(int(0.8*T)),
    #     test_indices= range(int(0.8*T), T),
    #     verbosity=0
    #     )

    # tau_max = 25
    # steps_ahead = 1
    # target = 2

    # all_predictors = pred.get_predictors(
    #                   selected_targets=[target],
    #                   selected_links=None,
    #                   steps_ahead=steps_ahead,
    #                   tau_max=tau_max,
    #                   pc_alpha=None,
    #                   max_conds_dim=None,
    #                   max_combinations=1,
    #                   )

    # print all_predictors
    
    # pred.fit(target_predictors=all_predictors, 
    #                 selected_targets=[target],
    #                     tau_max=tau_max,
    #                     return_data=True)

    # # print all_predictors[target]

    # new_data = pp.DataFrame(pp.var_process(links_coeffs, T=100)[0])

    # predicted = pred.predict(target,
    #                         # new_data=new_data,
    #                         # cut_off = 'max_lag',
    #                         # pred_params = {'return_std':True}
    #                         )
    # # print predicted[1]

    # true_data = pred.get_test_array()[0]
    # train_data = pred.get_train_array(target)
    # print pred.get_test_array()[0, :10]
    # print train_data[0, :10]

    # print "NRMSE = %.2f" % (numpy.abs(true_data - predicted).mean()/true_data.std())
    # pylab.scatter(true_data, predicted)
    # pylab.title("NRMSE = %.2f" % (numpy.abs(true_data - predicted).mean()/true_data.std()))

    # # pylab.errorbar(true_data, predicted[0], yerr=predicted[1], fmt='o')

    # pylab.plot(true_data, true_data, 'k-')
    # pylab.show()

    # pylab.figure()
    # pred_bad = Prediction(
    #     dataframe=dataframe,
    #     cond_ind_model=ParCorr,
    #     prediction_model = sklearn.linear_model.LinearRegression,
    #     data_transform=sklearn.preprocessing.StandardScaler(),
    #     train_indices= range(int(0.8*T)),
    #     test_indices= range(int(0.8*T), T),
    #     verbosity=0
    #     )
    # all_predictors = {2:[(i, -tau) for i in range(3) for tau in range(1, tau_max+1)]}
    # pred_bad.fit(target_predictors=all_predictors, 
    #                 selected_targets=[target],
    #                     tau_max=tau_max)

    # predicted = pred_bad.predict(target)
    # true_data = pred_bad.get_test_array()[0]
    # print "NRMSE = %.2f" % (numpy.abs(true_data - predicted).mean()/true_data.std())

    # pylab.scatter(true_data, predicted)
    # pylab.title("\nNRMSE = %.2f" % (numpy.abs(true_data - predicted).mean()/true_data.std()))
    # pylab.plot(true_data, true_data, 'k-')
    # pylab.xlabel('True test data')
    # pylab.ylabel('Predicted test data')

    # pylab.show()
