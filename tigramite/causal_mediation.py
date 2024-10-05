"""Tigramite causal inference for time series."""

# Authors: Martin Rabel, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KernelDensity
from functools import partial
import tigramite.toymodels.non_additive as toy_setup
from tigramite.causal_effects import CausalEffects
from tigramite.toymodels.non_additive import _Fct_on_grid, _Fct_smoothed



"""-------------------------------------------------------------------------------------------
-----------------------------   Helpers for Mixed Data Fitting   -----------------------------
-------------------------------------------------------------------------------------------"""


class MixedData:
    """Namescope for mixed-data helpers.

    Mostly for internal use.
    """

    @staticmethod
    def IsPurelyCategorical(vars_and_data):
        """Check if all variables are categorical

        Parameters
        ----------
        vars_and_data : dictionary< toy_setup.VariableDescription, np.array(N) >
            Samples with meta-data.

        Returns
        -------
        purely categorical : bool
            If all variables in vars_and_data are categorical, return true, otherwise false.
        """
        for var in vars_and_data.keys():
            if not var.Id().is_categorical:
                return False
        return True

    @staticmethod
    def _FixZeroDim(continuous_data, categorical_data):
        """[INTERNAL] Ensure consistent shape (0,N) for batches containing zero categorical or continuous variables"""
        if len(continuous_data) == 0:
            assert len(categorical_data) != 0
            continuous_data = np.empty([0, np.shape(categorical_data)[1]], dtype="float64")
            categorical_data = np.array(categorical_data)
        elif len(categorical_data) == 0:
            categorical_data = np.empty([0, np.shape(continuous_data)[1]], dtype="int32")
            continuous_data = np.array(continuous_data)
        else:
            continuous_data = np.array(continuous_data)
            categorical_data = np.array(categorical_data)
        return continuous_data, categorical_data

    @staticmethod
    def Get_data_via_var_ids(x, var_ids):
        """For a subset of variables, get separate batches for continuous and categorical data, and category-shape

        Parameters
        ----------
        x : dictionary< toy_setup.VariableDescription, np.array(N) >
            Samples and meta-data.
        var_ids : *iterable* <toy_setup.VariableDescription>
            Subset of variables to extract data for.

        Returns
        -------
        continuous_data : np.array( # continuous variables in var_ids, # samples )
            Continuous entries of the data.
        categorical_data : np.array( # categorical variables in var_ids, # samples )
            Categorical entries of the data.
        category_shape : list <uint>
            List of the number of categories for each categorical variable in var_ids in the same ordering.
        """
        category_shape = []
        categorical_data = []
        continuous_data = []
        for desc in var_ids:
            if desc.is_categorical:
                category_shape.append(desc.categories)
                categorical_data.append(x[desc])
            else:
                continuous_data.append(x[desc])
        return MixedData._FixZeroDim(continuous_data, categorical_data) + (category_shape,)

    @classmethod
    def Split_data_into_categorical_and_continuous(cls, x):
        """Get separate batches for continuous and categorical data, and category-shape

        (see Get_data_via_var_ids)
        """
        return cls.Get_data_via_var_ids(x, x.keys())

    @staticmethod
    def CategoryCount(category_shape):
        """Get total number of categories in shape.

        Parameters
        ----------
        category_shape : list<uint>
            Number of categories per variable.

        Returns
        -------
        total : uint
            Total number of categories in the product.
        """
        result = 1
        for c in category_shape:
            result *= c
        return result

    @staticmethod
    def Get_Len(continuous, categorical):
        """Extract number of samples from data"""
        if np.shape(continuous)[0] != 0:
            return np.shape(continuous)[1]
        else:
            assert np.shape(categorical)[0] != 0
            return np.shape(categorical)[1]

    @staticmethod
    def SimplifyIfTrivialVector(x):
        """Make shapes consistent."""
        result = np.squeeze(x)
        if np.shape(result) == ():
            return result[()]  # turn a scalar from array(x) of shape () to an actual scalar
        else:
            return result

    @classmethod
    def Call_map(cls, f, x):
        """Execute call consistently."""
        if hasattr(f, '__iter__'):
            result = []
            for coord in f:
                result.append(cls.Call_map(coord, x))
            return np.array(result)
        if callable(f):
            return cls.SimplifyIfTrivialVector(f(x))
        else:
            return cls.SimplifyIfTrivialVector(f.predict(x))


class FitProvider_Continuous_Default:
    r"""Helper for fitting continuous maps.

    See "Technical Appendix B" of the Causal Mediation Tutorial.

    Parameters
    ----------
    fit_map : *callable* (np.array(N, dim x), np.array(N))
        A callable, that given data (x,y) fits a regressor f s.t. y = f(x)
    fit_map_1d : *None* or *callable* (np.array(N, 1), np.array(N))
        overwrite fit_map in the case of a 1-dimensional domain.
    """

    def __init__(self, fit_map, fit_map_1d=None):
        self.fit_map = fit_map
        self.fit_map_1d = fit_map_1d

    @classmethod
    def UseSklearn(cls, neighbors=10, weights='uniform'):
        """Use an sci-kit learn KNeighborsRegressor

        Parameters
        ----------
        neighbors : uint
            The number of neighbors to consider
        weights : string
            Either 'uniform' or 'radius', see sci-kit learn.

        Returns
        -------
        K-Neighbors Fit-Provider : FitProvider_Continuous_Default
            Fit-Provider for use with FitSetup.
        """
        return cls(lambda x, y: KNeighborsRegressor(n_neighbors=neighbors, weights=weights).fit(x, y))

    @classmethod
    def UseSklearn_GC(cls, **kwargs):
        """Use an sci-kit learn GaussianProcessRegressor

        Parameters
        ----------
        ... : any-type
            forwarded to sci-kit

        Returns
        -------
        GC Fit-Provider : FitProvider_Continuous_Default
            Fit-Provider for use with FitSetup.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        return cls(lambda x, y: GaussianProcessRegressor(**kwargs).fit(x, y))

    def Get_Fit_Continuous(self, x, y):
        """Produce a fit

        Parameters
        ----------
        x : np.array(N, dim x)
            Predictor values
        y : np.array(N)
            Training values

        Returns
        -------
        Continuous Fit : *callable* (x)
            The fitted predictor.
        """
        dim_of_domain = np.shape(x)[1]
        if dim_of_domain == 0:
            return lambda x_dim_zero: np.mean(y)
        elif dim_of_domain == 1 and self.fit_map_1d is not None:
            return self.fit_map_1d(x, y)  # use splines?
        else:
            return self.fit_map(x, y)


class FitProvider_Density_Default:
    r"""Helper for fitting continuous densities.

   See "Technical Appendix B" of the Causal Mediation Tutorial.

   Parameters
   ----------
   fit_density : *callable* (np.array(N, dim x))
       A callable, that given data (x) fits a density estimator p s.t. x ~ p
   """

    def __init__(self, fit_density):
        self.fit_density = fit_density

    @classmethod
    def UseSklearn(cls, kernel='gaussian', bandwidth=0.2):
        """Use an sci-kit learn KernelDensity

        Parameters
        ----------
        kernel : string
            E.g. 'gaussian', see sci-kit learn documentation.
        bandwidth : float
            See sci-kit learn documentation.

        Returns
        -------
        Density Estimator : FitProvider_Density_Default
            Fit-Provider for use with FitSetup.
        """
        return cls(lambda x: KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x))

    def Get_Fit_Density(self, x_train):
        """Produce a fit

        Parameters
        ----------
        x_train : np.array(N, dim x)
            Training samples.

        Returns
        -------
        Density Estimate : *callable* (x)
            The fitted predictor.
        """
        dim_of_domain = np.shape(x_train)[1]
        if dim_of_domain == 0:
            return lambda x_dim_zero: np.ones(np.shape(x_dim_zero)[0])
        elif np.shape(x_train)[0] > 0:
            model = self.fit_density(x_train)
            return lambda x_predict: np.exp(model.score_samples(x_predict))
        else:
            return lambda x_predict: np.zeros(np.shape(x_predict)[0])


class MixedMap:
    """Helper to evaluate fitted maps on mixed data.

    Used internally, should not normally be required in user-code.
    """
    error_value = float('nan')

    def __init__(self, maps, var_ids, dtype):
        self.maps = maps
        self.var_ids = var_ids
        self.dtype = dtype

    def predict(self, x):
        """Given x, predict y."""
        return self(x)

    def __call__(self, x):
        """Given x, predict y."""
        continuous_data, categorical_data, category_shape = MixedData.Get_data_via_var_ids(x, self.var_ids)
        category_count = MixedData.CategoryCount(category_shape)
        if category_count == 1:  # avoid errors in ravel_multi_index and improve performance by treating separately
            return MixedData.Call_map(self.maps, continuous_data.T)
        else:
            categories = np.ravel_multi_index(categorical_data, category_shape)
            result = np.empty(MixedData.Get_Len(continuous_data, categorical_data), dtype=self.dtype)
            for cX in range(category_count):
                filter_x = (categories == cX)
                if np.count_nonzero(filter_x) > 0:
                    # Avoid errors if asking for prediction on data with cX not occurring
                    # (Queries must be validated elsewhere)
                    filtered_x = continuous_data[:, filter_x]
                    if self.maps[cX] is not None:
                        result[filter_x] = MixedData.Call_map(self.maps[cX], filtered_x.T)
                    else:
                        result[filter_x] = self.error_value
            return result


class MixedMarkovKernel:
    """Helper to evaluate fitted densities on mixed data.

    Used internally, should not normally be required in user-code.
    """
    def __init__(self, transfers, var_ids, dtype, output_category_count):
        self.transfers = transfers
        self.var_ids = var_ids
        self.dtype = dtype
        self.output_category_count = output_category_count

    def predict(self, x):
        """Given np.array x, predict probability of each sample."""
        return self(x)

    def __call__(self, x):
        """Given np.array x, predict probability of each sample."""
        continuous_data, categorical_data, category_shape = MixedData.Get_data_via_var_ids(x, self.var_ids)
        category_count = MixedData.CategoryCount(category_shape)
        categories = None
        if category_count == 1:  # avoid errors in ravel_multi_index and improve performance by treating separately
            categories = np.zeros(np.shape(continuous_data)[1])
        else:
            categories = np.ravel_multi_index(categorical_data, category_shape)

        values = np.zeros([self.output_category_count, MixedData.Get_Len(continuous_data, categorical_data)])
        for cX in range(category_count):
            filter_x = (categories == cX)
            if np.count_nonzero(filter_x) > 0:
                # Avoid errors if asking for prediction on data with cX not occurring
                # (Queries must be validated elsewhere)
                filtered_x = continuous_data[:, filter_x]
                for cY in range(self.output_category_count):
                    transfer = self.transfers[cX][cY]
                    values[cY][filter_x] = transfer(filtered_x.T)
        normalize = np.sum(values, axis=0)
        return (values / normalize).T


class FitSetup:
    """Helper to fit mixed data.

   See "Technical Appendix B" of the Causal Mediation Tutorial.
    """
    def __init__(self, fit_map=FitProvider_Continuous_Default.UseSklearn(),
                 fit_density=FitProvider_Density_Default.UseSklearn()):
        self.fit_map = fit_map
        self.fit_density = fit_density

    def Fit(self, domain, target, dont_wrap_1d_y_in_vector=True):
        """Fit a mixed data mapping.

        See "Technical Appendix B" of the Causal Mediation Tutorial.
        """
        dont_wrap = dont_wrap_1d_y_in_vector and len(target) == 1
        result = {}
        for y, train in target.items():
            _next = None
            if y.is_categorical:
                _next = self.Fit_CategoricalTarget_MarkovKernel(domain, train, y.categories)
            else:
                _next = self.Fit_ContinuousTarget(domain, train, y.dimension)
            if dont_wrap:
                return _next
            else:
                result[y] = _next
        return result

    def Fit_ContinuousTarget(self, x, y, dim_y):
        """Fit a mixed-domain mapping with continuous target."""
        if dim_y == 1:
            return self.Fit_ContinuousTarget_1D(x, y)
        else:
            result = []
            for i in range(dim_y):
                result.append(self.Fit_ContinuousTarget_1D(x, y[:, i]))
            return result

    def Fit_ContinuousTarget_1D(self, x, y):
        """Fit a mixed-domain mapping with 1-dimensional continuous target."""
        continuous_data, categorical_data, category_shape = MixedData.Split_data_into_categorical_and_continuous(x)

        category_count = MixedData.CategoryCount(category_shape)
        if category_count == 1:
            # Avoid errors in np.ravel_multi_index and improve performance by treating this case separately
            _map = self.fit_map.Get_Fit_Continuous(continuous_data.T, y)
            return MixedMap(_map, var_ids=x.keys(), dtype=y.dtype)
        else:
            # Fit once per category
            category_index = np.ravel_multi_index(categorical_data, category_shape)
            maps = {}
            for cX in range(category_count):
                filter_x = (category_index == cX)
                filtered_y = y[filter_x]
                filtered_x = continuous_data[:, filter_x]
                if np.count_nonzero(filter_x) > 1:
                    maps[cX] = self.fit_map.Get_Fit_Continuous(filtered_x.T, filtered_y)
                else:
                    maps[cX] = None
            return MixedMap(maps, var_ids=x.keys(), dtype=y.dtype)

    def Fit_CategoricalTarget_MarkovKernel(self, x, y, y_category_count):
        """Fit a mixed-domain probabilistic mapping (a Markov-kernel) with categorical target."""
        continuous_data, categorical_data, category_shape = MixedData.Split_data_into_categorical_and_continuous(x)

        category_count = MixedData.CategoryCount(category_shape)
        category_index = None
        if category_count == 1:
            # Avoid errors in np.ravel_multi_index and improve performance by treating this case separately
            category_index = np.zeros(np.shape(continuous_data)[1])
        else:
            category_index = np.ravel_multi_index(categorical_data, category_shape)

        # Compute the transfer-matrix
        transfer_matrix = self._Fit_Enriched_TransferMatrix(continuous_data,
                                                            category_index, category_count,
                                                            y, y_category_count)
        return MixedMarkovKernel(transfer_matrix, var_ids=x, dtype=y.dtype, output_category_count=y_category_count)

    @staticmethod
    def _Fit_TransferMatrix(x_categorical, x_category_count, y_categorical, y_category_count):
        transfers = np.zeros([x_category_count, y_category_count])
        for cX in range(x_category_count):
            filter_x = (x_categorical == cX)
            filtered_y = y_categorical[filter_x]
            normalization = np.count_nonzero(filter_x)
            if normalization != 0:
                for cY in range(y_category_count):
                    transfers[cX, cY] = np.count_nonzero(filtered_y == cY) / normalization
            else:
                transfers[cX, cY] = MixedMap.error_value
        return transfers

    def _Fit_Enriched_TransferMatrix(self, x_continuous,
                                     x_categorical, x_category_count,
                                     y_categorical, y_category_count):
        transfers = []
        for cX in range(x_category_count):
            filter_x = (x_categorical == cX)
            normalization = np.count_nonzero(filter_x)
            transfers_from_cX = []
            transfers.append(transfers_from_cX)
            if normalization != 0:
                for cY in range(y_category_count):
                    # Use Bayes' Theorem to avoid fitting densities conditional on continuous variables
                    filter_both = np.logical_and(x_categorical == cX, y_categorical == cY)
                    assert len(filter_both.shape) == 1
                    p_y_given_discrete_x = np.count_nonzero(filter_both) / normalization
                    p_continuous_x_given_discrete_x_and_y = self.fit_density.Get_Fit_Density(
                        x_continuous[:, filter_both].T)
                    # Avoid weird bugs in lambda-captures combined with for-loop scopes by using "functools->partial"
                    transfers_from_cX.append(partial(self._EvalFromBayes, p_y_given_discrete_x=p_y_given_discrete_x,
                                                     p_continuous_x_given_discrete_x_and_y=p_continuous_x_given_discrete_x_and_y))
                    # transfers_from_cX.append(lambda x_c: self._EvalFromBayes(x_c, p_y_given_discrete_x,
                    #                                                        p_continuous_x_given_discrete_x_and_y))
            else:
                for cY in range(y_category_count):
                    transfers_from_cX.append(lambda x_c: np.full(np.shape(x_c)[0], MixedMap.error_value))
        return transfers

    @staticmethod
    def _EvalFromBayes(x_c, p_y_given_discrete_x, p_continuous_x_given_discrete_x_and_y):
        # P( y | x_c, x_d ) = P( x_c | x_d, y ) P( Y | x_d ) / normalization independent of y
        return p_continuous_x_given_discrete_x_and_y(x_c) * p_y_given_discrete_x





"""-------------------------------------------------------------------------------------------
--------------------------------   Natural Effect Estimation   -------------------------------
-------------------------------------------------------------------------------------------"""

class NaturalEffects_StandardMediationSetup:
    r"""Setup for estimating natural effects in a "standard" mediation (triangle) graph.

    Methods for the non-parametric estimation of mediation effects.
    For (more) general graphs, see NaturalEffects_GraphMediation.

    Actual fit-models can be chosen independently, for details see
    technical appendix B in the mediation-tutorial.

    See references and tigramite tutorial for an in-depth introduction.

    References
    ----------

    J. Pearl. Direct and indirect effects. Proceedings of the Seventeenth Conference
    on Uncertainty in Artificial intelligence, 2001.

    J. Pearl. Interpretation and identification of causal mediation. Psychological
    methods, 19(4):459, 2014.

    I. Shpitser and T. J. VanderWeele. A complete graphical criterion for the adjust-
    ment formula in mediation analysis. The international journal of biostatistics,
    7(1), 2011.

    Parameters
    ----------
    fit_setup : fit model
        A fit model to use internally, e.g. mixed_fit.FitSetup(). See there or technical
        appendix B of the tutorial.
    source : toy_setup.VariableDescription
        The (description of, e.g. toy_setup.ContinuousVariable()) the effect-source.
    target : toy_setup.VariableDescription
        The (description of, e.g. toy_setup.ContinuousVariable()) the effect-target.
    mediator : toy_setup.VariableDescription
        The (description of, e.g. toy_setup.ContinuousVariable()) the effect-mediator.
    data : dictionary ( keys=toy_setup.VariableDescription, values=np.array(N) )
        The data as map variable-description -> samples; e.g. toy_setup.world.Observables(),
        or toy_setup.VariablesFromDataframe( tigramite dataframe )
    """

    def __init__(self, fit_setup, source, target, mediator, data):
        self.fit_setup = fit_setup
        self.X = source
        self.Y = target
        self.M = mediator
        self.x = data[source]
        self.y = data[target]
        self.m = data[mediator]
        X = source
        Y = target
        M = mediator
        x = data[source]
        y = data[target]
        m = data[mediator]

        self._E_Y_XM = None
        self._P_Y_XM = None
        self._P_M_X = None

    def E_Y_XM_fixed_x_obs_m(self, x):
        """[internal] Provide samples Y( fixed x, observed m ) from fit of E[Y|X,M]

        Parameters
        ----------
        x : single value of same type as single sample for X (float, int, or bool)
            The fixed value of X.

        Returns
        -------
        Y( fixed x, observed m ) : np.array(N)
            Samples of Y estimated from observations and fit.
        """
        if self._E_Y_XM is None:
            self._E_Y_XM = self.fit_setup.Fit({self.X: self.x, self.M: self.m}, {self.Y: self.y})
        return self._E_Y_XM({self.X: np.full_like(self.x, x), self.M: self.m})

    def E_Y_XM_fixed_x_all_m(self, x):
        """For categorical M, provide samples Y( fixed x, m_i ) for all categories m_i of M from fit of E[Y|X,M]

        Parameters
        ----------
        x : single value of same type as single sample for X (float, int, or bool)
            The fixed value of X.

        Returns
        -------
        Y( fixed x, all  m_i ) : np.array(# of categories of M)
            Samples of Y estimated from fit for all categories of the mediator M.
        """
        assert self.M.is_categorical
        if self._E_Y_XM is None:
            self._E_Y_XM = self.fit_setup.Fit({self.X: self.x, self.M: self.m}, {self.Y: self.y})
        return self._E_Y_XM({self.X: np.full(self.M.categories, x), self.M: np.arange(self.M.categories)})

    def P_Y_XM_fixed_x_obs_m(self, x):
        """Provide for all samples of M the likelihood P( Y | fixed x, observed m ) from fit of P[Y|X,M]

        Parameters
        ----------
        x : single value of same type as single sample for X (float, int, or bool)
            The fixed value of X.

        Returns
        -------
        P( Y | fixed x, observed m ) : np.array(N, # categories of Y)
            Likelihood of each category of Y given x and observations of m from fit.
        """
        if self._P_Y_XM is None:
            self._P_Y_XM = self.fit_setup.Fit({self.X: self.x, self.M: self.m}, {self.Y: self.y})
        return self._P_Y_XM({self.X: np.full_like(self.x, x), self.M: self.m})

    def P_M_X(self, x):
        """Provide the likelihood P( M | fixed x ) from fit of P[M|X]

        Parameters
        ----------
        x : single value of same type as single sample for X (float, int, or bool)
            The fixed value of X.

        Returns
        -------
        P( M | fixed x ) : np.array( # categories of M )
            Likelihood of each category of M given x and observations of m from fit.
        """
        if self._P_M_X is None:
            self._P_M_X = self.fit_setup.Fit({self.X: self.x}, {self.M: self.m})
        return self._P_M_X({self.X: x})

    def _ValidateRequest(self, change_from, change_to):
        """Sanity-check parameters passed to effect-estimation

        Parameters
        ----------
        cf. NDE, NIE

        Throws
        ------
        Raises and exception if parameters are not meaningful
        """
        if self.X.is_categorical:
            # both_int = isinstance(change_from, int) and isinstance(change_to, int) does not work with numpy
            both_int = change_from % 1 == 0.0 and change_to % 1 == 0.0
            if not both_int:
                raise Exception("Categorical variables can only be intervened to integer values.")
            if change_from >= self.X.categories or change_to >= self.X.categories or change_from < 0 or change_to < 0:
                raise Exception("Intervention on categorical variable was outside of valid category-range.")

    def NDE(self, change_from, change_to, fct_of_nde=None):
        """Compute Natural Direct Effect (NDE)

        Parameters
        ----------
        change_from : single value of same type as single sample for X (float, int, or bool)
            Reference-value to which X is set by intervention in the world seen by the mediator.

        change_to : single value of same type as single sample for X (float, int, or bool)
            Post-intervention-value to which X is set by intervention in the world seen by the effect (directly).

        fct_of_nde : *callable* or None
            Also in the case of a continuous Y the distribution, not just the expectation is identified.
            However, a density-fit is not usually reasonable to do in practise. However, instead of
            computing E[Y] can compute E[f(Y)] efficiently for any f. Assume f=id if None.

        Throws
        ------
        Raises and exception if parameters are not meaningful

        Returns
        -------
        NDE : If Y is categorical -> np.array( # categories Y, 2 )
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.

        NDE : If Y is continuous -> float
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.
        """
        self._ValidateRequest(change_from, change_to)
        if self.Y.is_categorical:
            assert fct_of_nde is None, "Categorical estimate returns full density-estimate anyway."
            return self._NDE_categorical_target(change_from, change_to)
        else:
            return self._NDE_continuous_target(change_from, change_to, fct_of_nde)

    def NIE(self, change_from, change_to):
        """Compute Natural Indirect Effect (NIE)

        Parameters
        ----------
        change_from : single value of same type as single sample for X (float, int, or bool)
           Reference-value to which X is set by intervention in the world seen by the effect (directly).

        change_to : single value of same type as single sample for X (float, int, or bool)
            Post-intervention-value to which X is set by intervention in the world seen by the mediator.

        Throws
        ------
        Raises and exception if parameters are not meaningful

        Returns
        -------
        NIE : If Y is categorical -> np.array( # categories Y, 2 )
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by M from change_from to change_to, while keeping the value as (directly) seen by Y,
            as if X remained at change_from.

        NIE : If Y is continuous -> float
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by M from change_from to change_to, while keeping the value as (directly) seen by Y,
            as if X remained at change_from.
        """
        self._ValidateRequest(change_from, change_to)
        if self.Y.is_categorical:
            return self._NIE_categorical_target(change_from, change_to)
        else:
            return self._NIE_continuous_target(change_from, change_to)

    def _NDE_continuous_target(self, change_from, change_to, fct_of_nde=None):
        """Compute NDE (continuous Y)

        See 'NDE' above.

        Computed from mediation-formula
        (see [Pearl 2001] or [Shpitser, VanderWeele], see references above)
        by "double"-regression.
        """
        difference = self.E_Y_XM_fixed_x_obs_m(change_to) - self.E_Y_XM_fixed_x_obs_m(change_from)

        if fct_of_nde is not None:
            difference = fct_of_nde(difference)

        # sklearn predicts nan if too far from data ... (but might be irrelevant, check separately)
        valid_samples = np.isfinite(difference)
        Difference = toy_setup.ContinuousVariable()
        X = self.X
        x = self.x
        E_Difference_X = self.fit_setup.Fit({X: x[valid_samples]}, {Difference: difference[valid_samples]})
        return E_Difference_X({X: [change_from]})

    def _NDE_categorical_target_full_density(self, actual_value, counterfactual_value):
        """Compute NDE as full density (categorical Y)

        See 'NDE' above.

        Computed from mediation-formula
        (see [Pearl 2001] or [Shpitser, VanderWeele], see references above)
        by "double"-regression.

        Note: According to (see p.13)
        [Shpitser, VanderWeele: A Complete Graphical Criterion for theAdjustment Formula in Mediation Analysis]
        not just the expectation-value, but the full counterfactual distribution can be obtained via mediation-formula.
        """
        P_Y = toy_setup.ContinuousVariable(dimension=self.Y.categories)
        P_Y_samples = self.P_Y_XM_fixed_x_obs_m(actual_value)
        P_Y_X = self.fit_setup.Fit({self.X: self.x}, {P_Y: P_Y_samples})

        return MixedData.Call_map(P_Y_X, {self.X: [counterfactual_value]})

    def _NDE_categorical_target(self, actual_value, counterfactual_value):
        """Compute NDE by evaluating density (categorical Y)

        See 'NDE' and '_NDE_categorical_target_full_density' above.
        """
        # returns (counterfactual probabilities, total effect)
        return np.array([self._NDE_categorical_target_full_density(counterfactual_value, actual_value),
                         self._NDE_categorical_target_full_density(actual_value, actual_value)]).T

    def _NIE_continuous_target(self, change_from, change_to):
        """Compute NIE (continuous Y)

        See 'NIE' above.

        Computed from mediation-formula
        (see [Pearl 2001] or [Shpitser, VanderWeele], see references above)
        by "double"-regression.

        If M is categorical, after fixing X=x, the fitted P( Y | X=x, M=m ), is actually
        categorical (a transfer matrix), because it takes values only in
        im( P ) = { P( Y | X=x, M=m_0 ), ...,  P( Y | X=x, M=m_k ) } where
        m_0, ..., m_k are the categories of M. This is clearly a finite set.
        Since the distribution over this finite subset of the continuous Val(Y)
        is very non-gaussian, "max likelihood by least square estimation" can fail horribly.
        Hence we fit a transfer-matrix instead.
        """
        X = self.X
        M = self.M
        x = self.x
        m = self.m

        if M.is_categorical:
            # if the image of the mapping in double-regression is finite, use a density fit instead
            p_M_X01 = self.P_M_X([change_from, change_to])
            E_YX0 = self.E_Y_XM_fixed_x_all_m(change_from)
            # sum over finite M:
            return np.dot(E_YX0, p_M_X01[1] - p_M_X01[0])

        else:
            Estimate = toy_setup.ContinuousVariable()
            y_at_original_x = self.E_Y_XM_fixed_x_obs_m(change_from)
            ModifiedY = self.fit_setup.Fit({X: x}, {Estimate: y_at_original_x})
            return (MixedData.Call_map(ModifiedY, {X: [change_to]})
                    - MixedData.Call_map(ModifiedY, {X: [change_from]}))

    def _NIE_categorical_target(self, change_from, change_to):
        """Compute NIE (continuous Y)

        See 'NIE' above.

        Computed from mediation-formula
        (see [Pearl 2001] or [Shpitser, VanderWeele], see references above)
        by "double"-regression.

        Similar to before (see _NIE_continuous_target), treat categorical M differently.
        """
        X = self.X
        M = self.M
        x = self.x
        m = self.m
        py_at_original_x = self.P_Y_XM_fixed_x_obs_m(change_from)

        if M.is_categorical:
            # if the image of the mapping in double-regression is finite, use a density fit instead
            p_M_X01 = self.P_M_X([change_from, change_to])
            # sum over finite M:
            result = np.zeros([self.Y.categories, 2])  # 2 is for TE & CF
            for cM in range(M.categories):
                result += np.outer(np.mean(py_at_original_x[m == cM], axis=0), p_M_X01[:, cM])
            return result

        else:
            Estimate = toy_setup.ContinuousVariable(dimension=self.Y.categories)
            ModifiedY = self.fit_setup.Fit({X: x}, {Estimate: py_at_original_x})
            return np.array([MixedData.Call_map(ModifiedY, {X: [change_to]}),
                             MixedData.Call_map(ModifiedY, {X: [change_from]})]).T

    def NDE_grid(self, list_of_points, cf_delta=0.5, normalize_by_delta=False, fct_of_nde=None):
        """Compute NDE as grided (unsmoothed) function

        Parameters
        ----------
        list_of_points : np.array( K )
            List of reference-values at which to estimate NDEs

        cf_delta : float
            The change from reference-value to effect-value (change_from=reference, change_to=ref + delta)

        normalize_by_delta : bool
            Normalize the effect by dividing by cf_delta.

        fct_of_nde : *callable* or None
            Also in the case of a continuous Y the distribution, not just the expectation is identified.
            However, a density-fit is not usually reasonable to do in practise. However, instead of
            computing E[Y] can compute E[f(Y)] efficiently for any f. Assume f=id if None.

        Throws
        ------
        Raises and exception if parameters are not meaningful

        Returns
        -------
        NDE : If Y is categorical -> np.array( K = # grid points, # categories Y, 2 )
            For each grid-point:
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.

        NDE : If Y is continuous -> np.array( K = # grid points )
            For each grid-point:
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.
        """
        return _Fct_on_grid(self.NDE, list_of_points, cf_delta, normalize_by_delta, fct_of_nde=fct_of_nde)

    def NIE_grid(self, list_of_points, cf_delta=0.5, normalize_by_delta=False):
        """Compute NIE as grided (unsmoothed) function

        Parameters
        ----------
        list_of_points : np.array( K )
            List of reference-values at which to estimate NIEs

        cf_delta : float
            The change from reference-value to effect-value (change_from=reference, change_to=ref + delta)

        normalize_by_delta : bool
            Normalize the effect by dividing by cf_delta.

        Throws
        ------
        Raises and exception if parameters are not meaningful

        Returns
        -------
        NIE : If Y is categorical -> np.array( K = # grid points, # categories Y, 2 )
            For each grid-point:
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by M from change_from to change_to, while keeping the value of X "seen" (directly)
            by Y as if X remained at change_from.

        NIE : If Y is continuous -> np.array( K = # grid points )
            For each grid-point:
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by M from change_from to change_to, while keeping the value of X "seen" (directly)
            by Y as if X remained at change_from.
        """
        return _Fct_on_grid(self.NIE, list_of_points, cf_delta, normalize_by_delta)

    def NDE_smoothed(self, min_x, max_x, cf_delta=0.5, steps=100, smoothing_gaussian_sigma_in_steps=5,
                     normalize_by_delta=False, fct_of_nde=None):
        """Compute NDE as smoothed function

        Parameters
        ----------
        min_x : float
            Lower bound of interval on which reference-values for X are taken

        max_x : float
            Upper bound of interval on which reference-values for X are taken

        cf_delta : float
            The change from reference-value to effect-value (change_from=reference, change_to=ref + delta)

        steps : uint
            Number of intermediate values to compute in the interval [min_x, max_x]

        smoothing_gaussian_sigma_in_steps : uint
            The width of the Gauß-kernel used for smoothing, given in steps.

        normalize_by_delta : bool
            Normalize the effect by dividing by cf_delta.

        fct_of_nde : *callable* or None
            Also in the case of a continuous Y the distribution, not just the expectation is identified.
            However, a density-fit is not usually reasonable to do in practise. However, instead of
            computing E[Y] can compute E[f(Y)] efficiently for any f. Assume f=id if None.

        Throws
        ------
        Raises and exception if parameters are not meaningful

        Returns
        -------
        NDE : If Y is categorical -> np.array( # steps, # categories Y, 2 )
            For each grid-point:
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.

        NDE : If Y is continuous -> np.array( # steps )
            For each grid-point:
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.
        """
        return _Fct_smoothed(self.NDE, min_x, max_x, cf_delta, steps, smoothing_gaussian_sigma_in_steps,
                             normalize_by_delta, fct_of_nde=fct_of_nde)

    def NIE_smoothed(self, min_x, max_x, cf_delta=0.5, steps=100, smoothing_gaussian_sigma_in_steps=5,
                     normalize_by_delta=False):
        """Compute NIE as smoothed function

        Parameters
        ----------
        min_x : float
            Lower bound of interval on which reference-values for X are taken

        max_x : float
            Upper bound of interval on which reference-values for X are taken

        cf_delta : float
            The change from reference-value to effect-value (change_from=reference, change_to=ref + delta)

        steps : uint
            Number of intermediate values to compute in the interval [min_x, max_x]

        smoothing_gaussian_sigma_in_steps : uint
            The width of the Gauß-kernel used for smoothing, given in steps.

        normalize_by_delta : bool
            Normalize the effect by dividing by cf_delta.

        Throws
        ------
        Raises and exception if parameters are not meaningful

        Returns
        -------
        NIE : If Y is categorical -> np.array( # steps, # categories Y, 2 )
            For each grid-point:
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by M from change_from to change_to, while keeping the value of X "seen" (directly)
            by Y as if X remained at change_from.

        NDE : If Y is continuous -> np.array( # steps )
            For each grid-point:
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by M from change_from to change_to, while keeping the value of X "seen" (directly)
            by Y as if X remained at change_from.
        """
        return _Fct_smoothed(self.NIE, min_x, max_x, cf_delta, steps, smoothing_gaussian_sigma_in_steps,
                             normalize_by_delta)


class NaturalEffects_GraphMediation:
    r"""Setup for estimating natural effects in a (general) causal graph.

    Methods for the non-parametric estimation of mediation effects by adjustment.

    Actual fit-models can be chosen independently, for details see
    technical appendix B in the mediation-tutorial.

    See references and tigramite tutorial for an in-depth introduction.

    References
    ----------

    J. Pearl. Direct and indirect effects. Proceedings of the Seventeenth Conference
    on Uncertainty in Artificial intelligence, 2001.

    J. Pearl. Interpretation and identification of causal mediation. Psychological
    methods, 19(4):459, 2014.

    I. Shpitser and T. J. VanderWeele. A complete graphical criterion for the adjust-
    ment formula in mediation analysis. The international journal of biostatistics,
    7(1), 2011.

    Parameters
    ----------
    graph : np.array( [N, N] or [N, N, tau_max+1] depending on graph_type ) of 3-character patterns
        The causal graph, see 'Causal Effects' tutorial. E.g. returned by causal discovery method
        (see "Tutorials/Causal Discovery/CD Overview") or by a toymodel (see toy_setup.Model.GetGroundtruthGraph or
        the Mediation tutorial).
    graph_type : string
        The type of graph, tested for 'dag' and 'stationary_dag' (time-series). See 'Causal Effects' tutorial.
    tau_max : uint
        Maximum lag to be considered (can be relevant for adjustment sets, passed to 'Causal Effects' class).
    fit_setup : fit model
        A fit model to use internally, e.g. mixed_fit.FitSetup(). See there or technical
        appendix B of the tutorial.
    observations_data : dictionary ( keys=toy_setup.VariableDescription, values=np.array(N) )
        The data as map variable-description -> samples; e.g. toy_setup.world.Observables(),
        or toy_setup.VariablesFromDataframe( tigramite dataframe )
    effect_source : toy_setup.VariableDescription or (idx, -lag)
        The (description of, e.g. toy_setup.ContinuousVariable()) the effect-source.
    effect_target : toy_setup.VariableDescriptionor (idx, -lag)
        The (description of, e.g. toy_setup.ContinuousVariable()) the effect-target.
    blocked_mediators : iterable of Variable-descriptions or 'all'
        Which mediators to 'block' (consider indirect), *un*\ blocked mediators are considered as
        contributions to the *direct* effect.
    adjustment_set : iterable of Variable-descriptions or 'auto'
        Adjustment-set to use. Will be validated if specified explicitly, if 'auto', will try
        to use an 'optimal' set, fall back to [Perkovic et al]'s adjustment-set (which should always
        work if single-set adjustment as in [Shpitser, VanderWeele] is possible; this follows
        from combining results of [Shpitser, VanderWeele] and [Perkovic et al]).
        See 'Causal Effects' and its tutorial for more info and references on (optimal) adjustment.
    only_check_validity : bool
        If True, do not set up an estimator, only check if an optimal adjustment-set exists (or the
        explicitly specified one is valid). Call this.Valid() to extract the result.
    fall_back_to_total_effect : bool
        If True, if no mediators are blocked, use mediation implementation to estimate the total effect.
        In this case, estimating the total effect through the 'Causal Effects' class might be easier,
        however, for comparison to other estimates, using this option might yield more consistent results.
    _internal_provide_cfx : *None* or tigramite.CausalEffects
        Set to None. Used when called from CausalMediation, which already has a causal-effects class.
    enable_dataframe_based_preprocessing : bool
        Enable (and enforce) data-preprocessing through the tigramite::dataframe, makes missing-data
        and other features available to the mediation analysis. Custom (just in time) handling
        of missing data might be more sample-efficient.
    """

    def __init__(self, graph, graph_type, tau_max, fit_setup, observations_data, effect_source, effect_target,
                 blocked_mediators="all", adjustment_set="auto", only_check_validity=False,
                 fall_back_to_total_effect=False, _internal_provide_cfx=None, enable_dataframe_based_preprocessing=True):

        data = toy_setup.DataHandler(observations_data, dataframe_based_preprocessing=enable_dataframe_based_preprocessing)

        self.Source = data.GetVariableAuto(effect_source, "Source")
        self.Target = data.GetVariableAuto(effect_target, "Target")

        if blocked_mediators != "all":
            blocked_mediators = data.GetVariablesAuto(blocked_mediators, "Mediator")
        if adjustment_set != "auto":
            adjustment_set = data.GetVariablesAuto(adjustment_set, "Adjustment")

        X = data[self.Source]
        Y = data[self.Target]

        # Use tigramite's total effect estimation utility to help generate adjustment-sets and mediators
        cfx_xy = _internal_provide_cfx if _internal_provide_cfx is not None else\
            CausalEffects(graph, graph_type=graph_type, X=[X], Y=[Y])

        # ----- MEDIATORS -----
        # Validate "blocked mediators" are actually mediators, or find "all"
        all_mediators = blocked_mediators == "all"
        if all_mediators:
            M = cfx_xy.M
            blocked_mediators = data.ReverseLookupMulti(M, "Mediator")
        else:
            M = data[blocked_mediators]
            if not set(M) <= set(cfx_xy.M):
                raise Exception("Blocked mediators, if specified, must actually be mediators, try using"
                                "set-intersection with causal_effects_instance_xy.M instead.")
        if len(M) == 0 and not fall_back_to_total_effect:
            raise Exception("There are no mediators, use total-effect estimation instead or set "
                            "fall_back_to_total_effect=True!")

        # ----- ADJUSTMENT -----
        # Use tigramite's total effect estimation utility to help validate adjustment-sets and mediators
        cfx_xm = CausalEffects(graph, graph_type=graph_type, X=[X], Y=M)
        cfx_xm_y = CausalEffects(graph, graph_type=graph_type, X=[X] + list(M), Y=[Y])

        def valid(S):
            return cfx_xm._check_validity(S) and cfx_xm_y._check_validity(S)

        adjustment_set_auto = (adjustment_set == "auto")

        if adjustment_set_auto:
            # first try optimal set (for small cf_delta, optimality for the causal effect should imply
            # optimality for the nde by continuity for the estimator variance)
            Z = cfx_xy.get_optimal_set()

            if not valid(Z) and adjustment_set_auto:
                # fall back to adjust, which should work if any single adjustmentset works
                Z = cfx_xy._get_adjust_set()

            adjustment_set = data.ReverseLookupMulti(Z, "Adjustment")

        else:
            Z = data[adjustment_set]

        self.valid = valid(Z)
        if only_check_validity:
            return

        # Output appropriate error msgs if not valid
        if not self.valid:
            if adjustment_set_auto:
                raise Exception(
                    "The graph-effect you are trying to estimate is not identifiable via one-step adjustment, "
                    "try using a different query "
                    "or refine the causal graph by expert knowledge. "
                    "For the implemented method, there must be an adjustment-set, valid for both X u M -> Y and "
                    "X -> Y. If such a set exists, Perkovic's Adjust(X,Y) is valid, which was tried as "
                    "fallback because adjustment-set='auto' was used.")
            else:
                raise Exception(
                    "The adjustment-set specified is not valid for one-step adjustment, "
                    "try using or a different adjustment-set (or set it to 'auto'), a different query "
                    "or refine the causal graph by expert knowledge. "
                    "For the implemented method, there must be an adjustment-set, valid for both X u M -> Y and "
                    "X -> Y. If such a set exists, Perkovic's Adjust(X,Y) is valid, which is tried as "
                    "fallback if adjustment-set='auto' is used.")

        
        # lock in mediators and adjustment for preprocessing
        self.BlockedMediators = data.ReverseLookupMulti(M)
        self.AdjustmentSet = data.ReverseLookupMulti(Z)

        # ----- STORE RESULTS ON INSTANCE -----
        self.X, self.sources = data.Get("Source", [X], tau_max=tau_max)
        self.X = self.X[0]  # currently univariate anyway
        self.Y, self.targets = data.Get("Target", [Y], tau_max=tau_max)
        self.Y = self.Y[0]
        if len(M) > 0:
            self.M_ids, self.mediators = data.Get("Mediator", M, tau_max=tau_max)
        else:
            self.M_ids = None
            self.mediators = {}


        if len(Z) > 0:
            self.Z_ids, self.adjustment = data.Get("Adjustment", Z, tau_max=tau_max)
        else:
            self.Z_ids = None
            self.adjustment = {}


        self.fit_setup = fit_setup
        self._E_Y_XMZ = None
        self._P_Y_XMZ = None

    def E_Y_XMZ_fixed_x_obs_mz(self, x):
        """Provide samples Y( fixed x, observed m, z ) from fit of E[Y|X,M,Z]

        Parameters
        ----------
        x : single value of same type as single sample for X (float, int, or bool)
            The fixed value of X.

        Returns
        -------
        Y( fixed x, observed m, z ) : np.array(N)
            Samples of Y estimated from observations and fit.
        """
        assert not self.Y.is_categorical
        if self._E_Y_XMZ is None:
            self._E_Y_XMZ = self.fit_setup.Fit({**self.sources, **self.mediators, **self.adjustment}, self.targets)

        return self._E_Y_XMZ({self.X: np.full_like(self.sources[self.X], x),
                              **self.mediators, **self.adjustment})

    def P_Y_XMZ_fixed_x_obs_mz(self, x):
        """Provide for all samples of M, Z the likelihood P( Y | fixed x, observed m, z ) from fit of P[Y|X,M,Z]

        Parameters
        ----------
        x : single value of same type as single sample for X (float, int, or bool)
            The fixed value of X.

        Returns
        -------
        P( Y | fixed x, observed m, z ) : np.array(N, # categories of Y)
            Likelihood of each category of Y given x and observations of m, z from fit.
        """
        assert self.Y.is_categorical
        if self._P_Y_XMZ is None:
            self._P_Y_XMZ = self.fit_setup.Fit({**self.sources, **self.mediators, **self.adjustment}, self.targets)
        return self._P_Y_XMZ({self.X: np.full_like(self.sources[self.X], x),
                              **self.mediators, **self.adjustment})

    def Valid(self):
        """Get validity of adjustment-set

        Returns
        -------
        Valid : bool
            Validity of adjustment set. (see constructor-parameter 'only_check_validity')
        """
        return self.valid

    def NDE(self, change_from, change_to):
        """Compute Natural Direct Effect (NDE)

        Parameters
        ----------
        change_from : single value of same type as single sample for X (float, int, or bool)
            Reference-value to which X is set by intervention in the world seen by the (blocked) mediators.

        change_to : single value of same type as single sample for X (float, int, or bool)
            Post-intervention-value to which X is set by intervention in the world seen by the effect (directly).

        Throws
        ------
        Raises and exception if parameters are not meaningful or if adjustment-set is not valid.

        Returns
        -------
        NDE : If Y is categorical -> np.array( # categories Y, 2 )
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.

        NDE : If Y is continuous -> float
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.
        """
        if not self.valid:
            raise Exception("Valid adjustment-set is required!")
        if not (self.X.ValidValue(change_from) and self.X.ValidValue(change_to)):
            raise Exception("NDE change must be at valid values of the source-variable (e.g. categorical, and within "
                            "the range [0,num-categories).")
        if self.Y.is_categorical:
            return np.array([self._NDE_categorical_target_full_density(change_to, change_from),
                             self._NDE_categorical_target_full_density(change_from, change_from)]).T
        else:
            return self._NDE_continuous_target(change_from, change_to)

    def _NDE_continuous_target(self, change_from, change_to):
        """Compute NDE (continuous Y)

        See 'NDE' above.

        Computed from mediation-formula
        (see [Pearl 2001] or [Shpitser, VanderWeele], see references above)
        by "triple"-regression.
        """
        difference = (self.E_Y_XMZ_fixed_x_obs_mz(change_to)
                      - self.E_Y_XMZ_fixed_x_obs_mz(change_from))

        Difference = toy_setup.ContinuousVariable()
        E_Difference_X = self.fit_setup.Fit({**self.sources, **self.adjustment},
                                            {Difference: difference})

        E_NDE_per_c = E_Difference_X({self.X: np.full_like(self.sources[self.X], change_from), **self.adjustment})

        return np.mean(E_NDE_per_c)

    def _NDE_categorical_target_full_density(self, cf_x, reference_x):
        """Compute NDE as full density (categorical Y)

        See 'NDE' above.

        Computed from mediation-formula
        (see [Pearl 2001] or [Shpitser, VanderWeele], see references above)
        by "double"-regression.

        Note: According to (see p.13)
        [Shpitser, VanderWeele: A Complete Graphical Criterion for theAdjustment Formula in Mediation Analysis]
        not just the expectation-value, but the full counterfactual distribution can be obtained via mediation-formula.

        Note: If all M and Z are categorical, after fixing X=x, the fitted P( Y | X=x, M=m ), is actually
        categorical (a transfer matrix), because it takes values only in
        im( P ) = { P( Y | X=x, M=m_0 ), ...,  P( Y | X=x, M=m_k ) } where
        m_0, ..., m_k are the categories of M. This is clearly a finite set.
        Since the distribution over this finite subset of the continuous Val(Y)
        is very non-gaussian, "max likelihood by least square estimation" can fail horribly.
        Hence we fit a transfer-matrix instead.
        """
        p_y_values = self.P_Y_XMZ_fixed_x_obs_mz(cf_x)

        if (MixedData.IsPurelyCategorical({**self.mediators, **self.adjustment})
                and len({**self.mediators, **self.adjustment}) > 0):
            # If there are mediators or adjustment
            # and they are purely categorical, then the mapping (M u Z) -> P_Y
            # has finite image, treating it as categorical gives better results

            # different numpy-versions behave differently wrt this call:
            # https://numpy.org/devdocs/release/2.0.0-notes.html#np-unique-return-inverse-shape-for-multi-dimensional-inputs
            # see also https://github.com/numpy/numpy/issues/26738
            labels_y, transformed_y_numpy_version_dependent = np.unique(p_y_values, return_inverse=True, axis=0)
            transformed_y = transformed_y_numpy_version_dependent.squeeze()

            P_Y = toy_setup.CategoricalVariable(categories=labels_y)
            P_P_Y_xz = self.fit_setup.Fit({**self.sources, **self.adjustment}, {P_Y: transformed_y})

            C_NDE_per_c = MixedData.Call_map(P_P_Y_xz,
                                                       {self.X: np.full_like(self.sources[self.X], reference_x),
                                                        **self.adjustment})
            print(labels_y)
            print(C_NDE_per_c.shape)
            print(labels_y.shape)
            P_NDE_per_c = np.matmul(C_NDE_per_c, labels_y)
            return np.mean(P_NDE_per_c, axis=0)  # Axis 1 is p of different categories

        else:
            P_Y = toy_setup.ContinuousVariable(dimension=self.Y.categories)
            E_P_Y_xz = self.fit_setup.Fit({**self.sources, **self.adjustment}, {P_Y: p_y_values})

            P_NDE_per_c = MixedData.Call_map(E_P_Y_xz,
                                                       {self.X: np.full_like(self.sources[self.X], reference_x),
                                                        **self.adjustment})

            return np.mean(P_NDE_per_c, axis=1)  # Axis 0 is p of different categories

    def NDE_smoothed(self, min_x, max_x, cf_delta=0.5, steps=100, smoothing_gaussian_sigma_in_steps=5,
                     normalize_by_delta=False):
        """Compute NDE as smoothed function

        Parameters
        ----------
        min_x : float
            Lower bound of interval on which reference-values for X are taken

        max_x : float
            Upper bound of interval on which reference-values for X are taken

        cf_delta : float
            The change from reference-value to effect-value (change_from=reference, change_to=ref + delta)

        steps : uint
            Number of intermediate values to compute in the interval [min_x, max_x]

        smoothing_gaussian_sigma_in_steps : uint
            The width of the Gauß-kernel used for smoothing, given in steps.

        normalize_by_delta : bool
            Normalize the effect by dividing by cf_delta.

        Throws
        ------
        Raises and exception if parameters are not meaningful, adjustment-set is not valid or
        normalization requested makes no sense (normalizing probabilites by delta).

        Returns
        -------
        NDE : If Y is categorical -> np.array( # steps, # categories Y, 2 )
            For each grid-point:
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.

        NDE : If Y is continuous -> np.array( # steps )
            For each grid-point:
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.
        """
        if self.Y.is_categorical and normalize_by_delta:
            raise Exception("Do not normalize categorical output-probabilities by delta. (They are probabilities, "
                            "so normalizing them in this way makes no sense.) Normalize the difference instead.")
        return _Fct_smoothed(self.NDE, min_x, max_x, cf_delta, steps, smoothing_gaussian_sigma_in_steps,
                             normalize_by_delta)

    def PrintInfo(self, detail=1):
        """Print info about estimator.

        Helper to quickly print blocked mediators, adjustment-set used and source->target.
        """
        print(f"Estimator for the effect of {self.Source.Info(detail)} on {self.Target.Info(detail)}")
        self.PrintMediators(detail)
        self.PrintAdjustmentSet(detail)

    def PrintMediators(self, detail=1):
        """ Print info about blocked mediators. """
        print("Blocked Mediators:")
        for m in self.BlockedMediators:
            print(" - " + m.Info(detail))

    def PrintAdjustmentSet(self, detail=1):
        """ Print info about adjustment-set. """
        print("Adjustment Set:")
        for z in self.AdjustmentSet:
            print(" - " + z.Info(detail))




"""-------------------------------------------------------------------------------------------
-------------------------------   Expose Tigramite Interface   -------------------------------
-------------------------------------------------------------------------------------------"""



class CausalMediation(CausalEffects):
    """Non-linear, non-additive causal mediation analysis.

    See the tutorial on Causal Mediation.

    Extends the tigramite.CausalEffects class by natural-effect estimation for counter-factual mediation analysis.
    Effects are estimated by adjustment, where adjustment-sets can be generated automatically (if they exist).

    Actual fit-models can be chosen independently, for details see
    technical appendix B in the mediation-tutorial.

    See references and tigramite tutorial for an in-depth introduction.

    References
    ----------
    J. Pearl. Direct and indirect effects. Proceedings of the Seventeenth Conference
    on Uncertainty in Artificial intelligence, 2001.

    J. Pearl. Interpretation and identification of causal mediation. Psychological
    methods, 19(4):459, 2014.

    I. Shpitser and T. J. VanderWeele. A complete graphical criterion for the adjust-
    ment formula in mediation analysis. The international journal of biostatistics,
    7(1), 2011.

    Parameters
    ----------
    graph : np.array( [N, N] or [N, N, tau_max+1] depending on graph_type ) of 3-character patterns
        The causal graph, see 'Causal Effects' tutorial. E.g. returned by causal discovery method
        (see "Tutorials/Causal Discovery/CD Overview") or by a toymodel (see toy_setup.Model.GetGroundtruthGraph or
        the Mediation tutorial).
    graph_type : string
        The type of graph, tested for 'dag' and 'stationary_dag' (time-series). See 'Causal Effects' tutorial.
    X : (idx, -lag)
        Index of the effect-source.
    Y : (idx, -lag)
        Index of the effect-target.
    S : None
        Reserved. Must be None in current version.
    hidden_variables : None
        Reserved. Must be None in current version.
    verbosity : uint
        Tigramite.CausalEffects verbosity setting.
    """
    def __init__(self, graph, graph_type, X, Y, S=None, hidden_variables=None, verbosity=0):
        super().__init__(graph, graph_type, X, Y, S, hidden_variables, False, verbosity)
        self.BlockedMediators = None
        self.MediationEstimator = None
        assert hidden_variables is None

    def fit_natural_direct_effect(self, dataframe, mixed_data_estimator=FitSetup(),
                                  blocked_mediators='all', adjustment_set='auto',
                                  use_mediation_impl_for_total_effect_fallback=False,
                                  enable_dataframe_based_preprocessing=True):
        """Fit a natural direct effect.

        Parameters
        ----------
        dataframe : tigramite.Dataframe
            Observed data.
        mixed_data_estimator : mixed_fit.FitSetup
            The fit-configuration to use. See mixed_fit.FitSetup and the Mediation tutorial, Appendix B.
        blocked_mediators : 'all' or *iterable* of < (idx, -lag) >
            Which mediators to 'block' (consider indirect), *un*\ blocked mediators are considered as
            contributions to the *direct* effect.
        adjustment_set : 'auto' or None or *iterable* < (idx, -lag) >
            Adjustment-set to use. Will be validated if specified explicitly, if 'auto' or None, will try
            to use an 'optimal' set, fall back to [Perkovic et al]'s adjustment-set (which should always
            work if single-set adjustment as in [Shpitser, VanderWeele] os possible; this follows
            from combining results of [Shpitser, VanderWeele] and [Perkovic et al]).
            See 'Causal Effects' and its tutorial for more info and references on (optimal) adjustment.
        use_mediation_impl_for_total_effect_fallback : bool
            If True, if no mediators are blocked, use mediation implementation to estimate the total effect.
            In this case, estimating the total effect through the 'Causal Effects' class might be easier,
            however, for comparison to other estimates, using this option might yield more consistent results.        
        enable_dataframe_based_preprocessing : bool
            Enable (and enforce) data-preprocessing through the tigramite::dataframe, makes missing-data
            and other features available to the mediation analysis. Custom (just in time) handling
            of missing data might be more sample-efficient.

        Returns
        -------
        estimator : NaturalEffects_GraphMediation
            Typically, use predict_natural_direct_effect or predict_natural_direct_effect_function
            to use the fitted estimator.
            The internal NaturalEffects_GraphMediation (if needed).
        """
        if adjustment_set is None:
            adjustment_set = 'auto'
        self.BlockedMediators = blocked_mediators
        if len(self.X) != 1:
            raise NotImplementedError("Currently only implemented for univariate effects (source).")
        if len(self.Y) != 1:
            raise NotImplementedError("Currently only implemented for univariate effects (target).")
        [source] = self.X
        [target] = self.Y
        self.MediationEstimator = NaturalEffects_GraphMediation(
            graph=self.graph, graph_type=self.graph_type, tau_max=self.tau_max,
            fit_setup=mixed_data_estimator, observations_data=dataframe,
            effect_source=source, effect_target=target,
            blocked_mediators=self.BlockedMediators, adjustment_set=adjustment_set, only_check_validity=False,
            fall_back_to_total_effect=use_mediation_impl_for_total_effect_fallback,
            _internal_provide_cfx=self, enable_dataframe_based_preprocessing=enable_dataframe_based_preprocessing)
        # return a NDE_Graph Estimator, but also remember it for predict_nde
        return self.MediationEstimator

    def predict_natural_direct_effect(self, reference_value_x, cf_intervention_value_x):
        """*After fitting* a natural direct effect, predict its value at a specific point.
        See also predict_natural_direct_effect_function.

        Parameters
        ----------
        reference_value_x : single value of same type as single sample for X (float, int, or bool)
            Reference-value to which X is set by intervention in the world seen by the (blocked) mediators.

        cf_intervention_value_x : single value of same type as single sample for X (float, int, or bool)
            Post-intervention-value to which X is set by intervention in the world seen by the effect (directly).

        Throws
        ------
        Raises and exception if parameters are not meaningful or if adjustment-set is not valid.

        Returns
        -------
        NDE : If Y is categorical -> np.array( # categories Y, 2 )
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.

        NDE : If Y is continuous -> float
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.
        """
        if self.MediationEstimator is None:
            raise Exception("Call fit_natural_direct_effect_x before using predict_natural_direct_effect.")
        return self.MediationEstimator.NDE(change_from=reference_value_x, change_to=cf_intervention_value_x)

    def predict_natural_direct_effect_function(self, min_x, max_x, cf_delta=0.5,
                                               steps=100, smoothing_gaussian_sigma_in_steps=5,
                                               normalize_by_delta=False):
        """Compute NDE as *smoothed* function

        Parameters
        ----------
        min_x : float
            Lower bound of interval on which reference-values for X are taken

        max_x : float
            Upper bound of interval on which reference-values for X are taken

        cf_delta : float
            The change from reference-value to effect-value (change_from=reference, change_to=ref + delta)

        steps : uint
            Number of intermediate values to compute in the interval [min_x, max_x]

        smoothing_gaussian_sigma_in_steps : uint
            The width of the Gauß-kernel used for smoothing, given in steps.

        normalize_by_delta : bool
            Normalize the effect by dividing by cf_delta.

        Throws
        ------
        Raises and exception if parameters are not meaningful, adjustment-set is not valid or
        normalization requested makes no sense (normalizing probabilites by delta).

        Returns
        -------
        NDE : If Y is categorical -> np.array( # steps, # categories Y, 2 )
            For each grid-point:
            The probabilities the categories of Y (after, before) changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.

        NDE : If Y is continuous -> np.array( # steps )
            For each grid-point:
            The change in the expectation-value of Y induced by changing the interventional value of X
            as "seen" by Y from change_from to change_to, while keeping (blocked) M as if X remained at change_from.
        """
        if self.MediationEstimator is None:
            raise Exception("Call fit_natural_direct_effect before using predict_natural_direct_effect_x.")
        return self.MediationEstimator.NDE_smoothed(min_x, max_x, cf_delta,
                                                    steps, smoothing_gaussian_sigma_in_steps,
                                                    normalize_by_delta)
