"""Tigramite causal inference for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
from copy import deepcopy
import json, warnings, os, pathlib
import numpy as np
import sklearn
import sklearn.linear_model
import networkx
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI

class Models():
    """Base class for time series models.

    Allows to fit any model from sklearn to the parents of a target variable.
    Also takes care of missing values, masking and preprocessing. If the
    target variable is multivariate, a model that supports multi-output
    regression must be used. Note that
    sklearn.multioutput.MultiOutputRegressor allows to extend single-output
    models.

    Parameters
    ----------
    dataframe : data object
        Tigramite dataframe object. It must have the attributes dataframe.values
        yielding a numpy array of shape (observations T, variables N) and
        optionally a mask of the same shape and a missing values flag.
    model : sklearn model object
        For example, sklearn.linear_model.LinearRegression() for a linear
        regression model.
    conditional_model : sklearn model object, optional (default: None)
        Used to fit conditional causal effects in nested regression. 
        If None, model is used.
    data_transform : sklearn preprocessing object, optional (default: None)
        Used to transform data prior to fitting. For example,
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored. Note that the inverse_transform is then
        applied to the predicted data.
    mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence
        measure I(X; Y | Z) the samples should be masked. If None, the mask
        is not used. Explained in tutorial on masking and missing values.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    """

    def __init__(self,
                 dataframe,
                 model,
                 conditional_model=None,
                 data_transform=None,
                 mask_type=None,
                 verbosity=0):
        # Set the mask type and dataframe object
        self.mask_type = mask_type
        self.dataframe = dataframe
        # Get the number of nodes and length for this dataset
        self.N = self.dataframe.N
        self.T = self.dataframe.T
        # Set the model to be used
        self.model = model
        if conditional_model is None:
            self.conditional_model = model
        else:
            self.conditional_model = conditional_model
        # Set the data_transform object and verbosity
        self.data_transform = data_transform
        self.verbosity = verbosity
        # Initialize the object that will be set later
        self.all_parents = None
        self.selected_variables = None
        self.tau_max = None
        self.fit_results = None

    # @profile    
    def get_general_fitted_model(self, 
                Y, X, Z=None,
                conditions=None,
                tau_max=None,
                cut_off='max_lag_or_tau_max',
                empty_predictors_function=np.mean,
                return_data=False):
        """Fit time series model.

        For each variable in selected_variables, the sklearn model is fitted
        with :math:`y` given by the target variable(s), and :math:`X` given by its
        parents. The fitted model class is returned for later use.

        Parameters
        ----------
        X, Y, Z : lists of tuples
            List of variables for estimating model Y = f(X,Z)
        conditions : list of tuples.
            Conditions for estimating conditional causal effects.
        tau_max : int, optional (default: None)
            Maximum time lag. If None, the maximum lag in all_parents is used.
        cut_off : {'max_lag_or_tau_max', '2xtau_max', 'max_lag'}
            How many samples to cutoff at the beginning. The default is
            'max_lag_or_tau_max', which uses the maximum of tau_max and the
            conditions. This is useful to compare multiple models on the same
            sample. Other options are '2xtau_max', which guarantees that MCI
            tests are all conducted on the same samples. Last, 'max_lag' uses
            as much samples as possible.
        empty_predictors_function : function
            Function to apply to y if no predictors are given.
        return_data : bool, optional (default: False)
            Whether to save the data array.

        Returns
        -------
        fit_results : dictionary of sklearn model objects
            Returns the sklearn model after fitting. Also returns the data
            transformation parameters.
        """

        def get_vectorized_length(W):
            return sum([len(self.dataframe.vector_vars[w[0]]) for w in W])

        self.X = X 
        self.Y = Y

        if conditions is None:
            conditions = []
        self.conditions = conditions

        if Z is not None:
            Z = [z for z in Z if z not in conditions]

        self.Z = Z

        # lenX = len(self.X)
        # lenS = len(self.conditions)
        self.lenX = get_vectorized_length(self.X)
        self.lenS = get_vectorized_length(self.conditions)

        self.cut_off = cut_off

        # Find the maximal conditions lag
        max_lag = 0
        for y in self.Y:
            this_lag = np.abs(np.array(self.X + self.Z + self.conditions)[:, 1]).max()
            max_lag = max(max_lag, this_lag)
        # Set the default tau max and check if it should be overwritten
        if tau_max is None:
            self.tau_max = max_lag
        else:
            self.tau_max = tau_max
            if self.tau_max < max_lag:
                raise ValueError("tau_max = %d, but must be at least "
                                 " max_lag = %d"
                                 "" % (self.tau_max, max_lag))

        # Construct array of shape (var, time)
        array, xyz, _ = \
            self.dataframe.construct_array(X=self.X, Y=self.Y,  
                                           Z=self.conditions,
                                           extraZ=self.Z,
                                           tau_max=self.tau_max,
                                           mask_type=self.mask_type,
                                           cut_off=self.cut_off,
                                           remove_overlaps=True,
                                           verbosity=self.verbosity)

        # Transform the data if needed
        self.fitted_data_transform = None
        if self.data_transform is not None:
            # Fit only X, Y, and S for later use in transforming input
            X_transform = deepcopy(self.data_transform)
            x_indices = list(np.where(xyz==0)[0])
            X_transform.fit(array[x_indices, :].T)
            self.fitted_data_transform = {'X': X_transform}
            Y_transform = deepcopy(self.data_transform)
            y_indices = list(np.where(xyz==1)[0])
            Y_transform.fit(array[y_indices, :].T)
            self.fitted_data_transform['Y'] = Y_transform
            if len(self.conditions) > 0:
                S_transform = deepcopy(self.data_transform)
                s_indices = list(np.where(xyz==2)[0])
                S_transform.fit(array[s_indices, :].T) 
                self.fitted_data_transform['S'] = S_transform

            # Now transform whole array
            # TODO: Rather concatenate transformed arrays
            all_transform = deepcopy(self.data_transform)
            array = all_transform.fit_transform(X=array.T).T

        # Fit the model 
        # Copy and fit the model
        a_model = deepcopy(self.model)

        predictor_indices =  list(np.where(xyz==0)[0]) \
                           + list(np.where(xyz==3)[0]) \
                           + list(np.where(xyz==2)[0])
        predictor_array = array[predictor_indices, :].T
        target_array = array[np.where(xyz==1)[0], :].T

        if predictor_array.size == 0:
            # Just fit default (eg, mean)
            class EmptyPredictorModel:
                def fit(self, X, y):
                    if y.ndim == 1:
                        self.result = empty_predictors_function(y)
                    else:
                        self.result = empty_predictors_function(y, axis=0)
                def predict(self, X):
                    return self.result
            a_model = EmptyPredictorModel()
        
        a_model.fit(X=predictor_array, y=target_array)
        
        # Cache the results
        fit_results = {}
        fit_results['observation_array'] = array
        fit_results['xyz'] = xyz
        fit_results['model'] = a_model
        # Cache the data transform
        fit_results['fitted_data_transform'] = self.fitted_data_transform

        # Cache and return the fit results
        self.fit_results = fit_results
        return fit_results

    # @profile
    def get_general_prediction(self,
                intervention_data,
                conditions_data=None,
                pred_params=None,
                transform_interventions_and_prediction=False,
                return_further_pred_results=False,
                aggregation_func=np.mean,
                intervention_type='hard',
                ):
        r"""Predict effect of intervention with fitted model.

        Uses the model.predict() function of the sklearn model.

        Parameters
        ----------
        intervention_data : numpy array
            Numpy array of shape (n_interventions, len(X)) that contains the do(X) values.
        conditions_data : data object, optional
            Numpy array of shape (n_interventions, len(S)) that contains the S=s values.
        pred_params : dict, optional
            Optional parameters passed on to sklearn prediction function (model and
            conditional_model).
        transform_interventions_and_prediction : bool (default: False)
            Whether to perform the inverse data_transform on prediction results.
        return_further_pred_results : bool, optional (default: False)
            In case the predictor class returns more than just the expected value,
            the entire results can be returned.
        aggregation_func : callable
            Callable applied to output of 'predict'. Default is 'np.mean'.
        intervention_type : {'hard', 'soft'}
            Specify whether intervention is 'hard' (set value) or 'soft' 
            (add value to observed data).
  
        Returns
        -------
        Results from prediction.
        """

        n_interventions, _ = intervention_data.shape

        if intervention_data.shape[1] != self.lenX:
            raise ValueError("intervention_data.shape[1] must be len(X).")

        if conditions_data is not None:
            if conditions_data.shape[1] != len(self.conditions):
                raise ValueError("conditions_data.shape[1] must be len(S).")
            if conditions_data.shape[0] != intervention_data.shape[0]:
                raise ValueError("conditions_data.shape[0] must match intervention_data.shape[0].")

        # Print message
        if self.verbosity > 1:
            print("\n## Predicting target %s" % str(self.Y))
            if pred_params is not None:
                for key in list(pred_params):
                    print("%s = %s" % (key, pred_params[key]))

        # Default value for pred_params
        if pred_params is None:
            pred_params = {}

        # Check the model is fitted.
        if self.fit_results is None:
            raise ValueError("Model not yet fitted.")

        # Transform the data if needed
        fitted_data_transform = self.fit_results['fitted_data_transform']
        if transform_interventions_and_prediction and fitted_data_transform is not None:
            intervention_data = fitted_data_transform['X'].transform(X=intervention_data)
            if self.conditions is not None and conditions_data is not None:
                conditions_data = fitted_data_transform['S'].transform(X=conditions_data)

        # Extract observational Z from stored array
        z_indices = list(np.where(self.fit_results['xyz']==3)[0])
        z_array = self.fit_results['observation_array'][z_indices, :].T  
        Tobs = len(self.fit_results['observation_array'].T) 

        if intervention_type == 'soft':
            x_indices = list(np.where(self.fit_results['xyz']==0)[0])
            x_array = self.fit_results['observation_array'][x_indices, :].T   

        if self.conditions is not None and conditions_data is not None:
            s_indices = list(np.where(self.fit_results['xyz']==2)[0])
            s_array = self.fit_results['observation_array'][s_indices, :].T  

        pred_dict = {}

        # Now iterate through interventions (and potentially S)
        for index, dox_vals in enumerate(intervention_data):
            # Construct XZS-array
            intervention_array = dox_vals.reshape(1, self.lenX) * np.ones((Tobs, self.lenX))
            if intervention_type == 'soft':
                intervention_array += x_array

            predictor_array = intervention_array

            if  len(self.Z) > 0:
                predictor_array = np.hstack((predictor_array, z_array))

            if self.conditions is not None and conditions_data is not None:
                conditions_array = conditions_data[index].reshape(1, self.lenS) * np.ones((Tobs, self.lenS))  
                predictor_array = np.hstack((predictor_array, conditions_array))

            predicted_vals = self.fit_results['model'].predict(
                                                    X=predictor_array, **pred_params)

            if self.conditions is not None and conditions_data is not None:

                a_conditional_model = deepcopy(self.conditional_model)
                
                if type(predicted_vals) is tuple:
                    predicted_vals_here = predicted_vals[0]
                else:
                    predicted_vals_here = predicted_vals
                
                a_conditional_model.fit(X=s_array, y=predicted_vals_here)
                self.fit_results['conditional_model'] = a_conditional_model

                predicted_vals = a_conditional_model.predict(
                    X=conditions_data[index].reshape(1, self.lenS), **pred_params)   # was conditions_data before

            if transform_interventions_and_prediction and fitted_data_transform is not None:
                predicted_vals = fitted_data_transform['Y'].inverse_transform(X=predicted_vals).squeeze()

            pred_dict[index] = predicted_vals

            # Apply aggregation function
            if type(predicted_vals) is tuple:
                aggregated_pred = aggregation_func(predicted_vals[0], axis=0)
            else:
                aggregated_pred = aggregation_func(predicted_vals, axis=0)

            aggregated_pred = aggregated_pred.squeeze()

            if index == 0:
                predicted_array = np.zeros((n_interventions, ) + aggregated_pred.shape, 
                                        dtype=aggregated_pred.dtype)

            predicted_array[index] = aggregated_pred

            # if fitted_data_transform is not None:
            #     rescaled = fitted_data_transform['Y'].inverse_transform(X=predicted_array[index, iy].reshape(-1, 1))
            #     predicted_array[index, iy] = rescaled.squeeze()

        if return_further_pred_results:
            return predicted_array, pred_dict
        else:
            return predicted_array


    def fit_full_model(self, all_parents,
                selected_variables=None,
                tau_max=None,
                cut_off='max_lag_or_tau_max',
                empty_predictors_function=np.mean,
                return_data=False):
        """Fit time series model.

        For each variable in selected_variables, the sklearn model is fitted
        with :math:`y` given by the target variable, and :math:`X` given by its
        parents. The fitted model class is returned for later use.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, 0), ...], 1:[], ...} containing
            the parents estimated with PCMCI.
        selected_variables : list of integers, optional (default: range(N))
            Specify to estimate parents only for selected variables. If None is
            passed, parents are estimated for all variables.
        tau_max : int, optional (default: None)
            Maximum time lag. If None, the maximum lag in all_parents is used.
        cut_off : {'max_lag_or_tau_max', '2xtau_max', 'max_lag'}
            How many samples to cutoff at the beginning. The default is
            'max_lag_or_tau_max', which uses the maximum of tau_max and the
            conditions. This is useful to compare multiple models on the same
            sample. Other options are '2xtau_max', which guarantees that MCI
            tests are all conducted on the same samples. Last, 'max_lag' uses
            as much samples as possible.
        empty_predictors_function : function
            Function to apply to y if no predictors are given.
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
        # Set the default tau_max and check if it should be overwritten
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
            X = [(j, 0)]  # dummy
            Z = self.all_parents[j]
            array, xyz, _ = \
                self.dataframe.construct_array(X, Y, Z,
                                               tau_max=self.tau_max,
                                               mask_type=self.mask_type,
                                               cut_off=cut_off,
                                               remove_overlaps=True,
                                               verbosity=self.verbosity)
            # Get the dimensions out of the constructed array
            dim, T = array.shape
            dim_z = dim - 2
            # Transform the data if needed
            if self.data_transform is not None:
                array = self.data_transform.fit_transform(X=array.T).T
            # Cache the results
            fit_results[j] = {}
            # Cache the data transform
            fit_results[j]['data_transform'] = deepcopy(self.data_transform)

            if return_data:
                # Cache the data if needed
                fit_results[j]['data'] = array
                fit_results[j]['used_indices'] = self.dataframe.use_indices_dataset_dict
            # Copy and fit the model if there are any parents for this variable to fit
            a_model = deepcopy(self.model)
            if dim_z > 0:
                a_model.fit(X=array[2:].T, y=array[1])
            else:
                # Just fit default (eg, mean)
                class EmptyPredictorModel:
                    def fit(self, X, y):
                        self.result = empty_predictors_function(y)
                    def predict(self, X):
                        return self.result
                a_model = EmptyPredictorModel()
                # a_model = empty_predictors_model(array[1])
                a_model.fit(X=array[2:].T, y=array[1])

            fit_results[j]['model'] = a_model

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

    def get_val_matrix(self):
        """Returns the coefficient array for different lags for linear model.

        Requires fit_model() before. An entry val_matrix[i,j,tau] gives the
        coefficient of the link from i to j at lag tau, including tau=0.

        Returns
        -------
        val_matrix : array-like, shape (N, N, tau_max + 1)
            Array of coefficients for each time lag, including lag-zero.
        """

        coeffs = self.get_coefs()
        val_matrix = np.zeros((self.N, self.N, self.tau_max + 1, ))

        for j in list(coeffs):
            for par in list(coeffs[j]):
                i, tau = par
                val_matrix[i,j,abs(tau)] = coeffs[j][par]

        return val_matrix

    def predict_full_model(self,
                new_data=None,
                pred_params=None,
                cut_off='max_lag_or_tau_max'):
        r"""Predict target variable with fitted model.

        Uses the model.predict() function of the sklearn model.

        A list of predicted time series for self.selected_variables is returned. 

        Parameters
        ----------
        new_data : data object, optional
            New Tigramite dataframe object with optional new mask. Note that
            the data will be cut off according to cut_off, see parameter
            `cut_off` below.
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

        if hasattr(self, 'selected_variables'):
            target_list = self.selected_variables
        else:
            raise ValueError("Model not yet fitted.")

        pred_list = []
        self.stored_test_array = {}
        for target in target_list:
            # Default value for pred_params
            if pred_params is None:
                pred_params = {}

            # Construct the array form of the data
            Y = [(target, 0)]  # dummy
            X = [(target, 0)]  # dummy
            Z = self.all_parents[target]

            # Check if we've passed a new dataframe object
            if new_data is not None:
                # if new_data.mask is None:
                #     # if no mask is supplied, use the same mask as for the fitted array
                #     new_data_mask = self.test_mask
                # else:
                new_data_mask = new_data.mask
                test_array, _, _ = new_data.construct_array(X, Y, Z,
                                                         tau_max=self.tau_max,
                                                         mask=new_data_mask,
                                                         mask_type=self.mask_type,
                                                         cut_off=cut_off,
                                                         remove_overlaps=True,
                                                         verbosity=self.verbosity)
            # Otherwise use the default values
            else:
                test_array, _, _ = \
                    self.dataframe.construct_array(X, Y, Z,
                                                   tau_max=self.tau_max,
                                                   mask_type=self.mask_type,
                                                   cut_off=cut_off,
                                                   remove_overlaps=True,
                                                   verbosity=self.verbosity)
            # Transform the data if needed
            a_transform = self.fit_results[target]['data_transform']
            if a_transform is not None:
                test_array = a_transform.transform(X=test_array.T).T
            # Cache the test array
            self.stored_test_array[target] = test_array
            # Run the predictor
            predicted = self.fit_results[target]['model'].predict(
                X=test_array[2:].T, **pred_params)

            if test_array[2:].size == 0:
                # If there are no predictors, return the value of 
                # empty_predictors_function, which is np.mean 
                # and expand to the test array length
                predicted = predicted * np.ones(test_array.shape[1])

            pred_list.append(predicted)

        return pred_list


    def get_residuals_cov_mean(self, new_data=None, pred_params=None):
        r"""Returns covariance and means of residuals from fitted model.

        Residuals are available as self.residuals.

        Parameters
        ----------
        new_data : data object, optional
            New Tigramite dataframe object with optional new mask. Note that
            the data will be cut off according to cut_off, see parameter
            `cut_off` below.
        pred_params : dict, optional
            Optional parameters passed on to sklearn prediction function.

        Returns
        -------
        Results from prediction.
        """

        assert self.dataframe.analysis_mode == 'single'

        N = self.dataframe.N
        T = self.dataframe.T[0]

        # Get overlapping samples
        used_indices = {}
        overlapping = set(list(range(0, T)))
        for j in self.all_parents:
            if self.fit_results[j] is not None:
                if 'used_indices' not in self.fit_results[j]:
                    raise ValueError("Run ")
                used_indices[j] = set(self.fit_results[j]['used_indices'][0])
                overlapping = overlapping.intersection(used_indices[j])

        overlapping = sorted(list(overlapping))

        if len(overlapping) <= 10:
            raise ValueError("Less than 10 overlapping samples due to masking and/or missing values,"
                             " cannot compute residual covariance!")

        predicted = self.predict_full_model(new_data=new_data,
                                            pred_params=pred_params,
                                            cut_off='max_lag_or_tau_max')

        # Residuals only exist after tau_max
        residuals = self.dataframe.values[0].copy()

        for index, j in enumerate([j for j in self.all_parents]): # if len(parents[j]) > 0]):
            residuals[list(used_indices[j]), j] -= predicted[index]
        
        overlapping_residuals = residuals[overlapping]

        len_residuals = len(overlapping_residuals)

        cov = np.cov(overlapping_residuals, rowvar=0)
        mean = np.mean(overlapping_residuals, axis=0)   # residuals should have zero mean due to prediction including constant

        self.residuals = overlapping_residuals

        return cov, mean

class LinearMediation(Models):
    r"""Linear mediation analysis for time series models.

    Fits linear model to parents and provides functions to return measures such
    as causal effect, mediated causal effect, average causal effect, etc. as
    described in [4]_. Also allows for contemporaneous links.

    For general linear and nonlinear causal effect analysis including latent
    variables and further functionality use the CausalEffects class.

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
    >>> links_coeffs = {0: [], 1: [((0, -1), 0.5)], 2: [((1, -1), 0.5)]}
    >>> data, true_parents = toys.var_process(links_coeffs, T=1000, seed=42)
    >>> dataframe = pp.DataFrame(data)
    >>> med = LinearMediation(dataframe=dataframe)
    >>> med.fit_model(all_parents=true_parents, tau_max=3)
    >>> print "Link coefficient (0, -2) --> 2: ", med.get_coeff(
    i=0, tau=-2, j=2)
    >>> print "Causal effect (0, -2) --> 2: ", med.get_ce(i=0, tau=-2, j=2)
    >>> print "Mediated Causal effect (0, -2) --> 2 through 1: ", med.get_mce(
    i=0, tau=-2, j=2, k=1)
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
    data_transform : sklearn preprocessing object, optional (default: StandardScaler)
        Used to transform data prior to fitting. For example,
        sklearn.preprocessing.StandardScaler for simple standardization. The
        fitted parameters are stored.
    mask_type : {None, 'y','x','z','xy','xz','yz','xyz'}
        Masking mode: Indicators for which variables in the dependence
        measure I(X; Y | Z) the samples should be masked. If None, the mask
        is not used. Explained in tutorial on masking and missing values.
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
        self.dataframe = dataframe
        self.mask_type = mask_type
        self.data_transform = data_transform
        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params

        self.bootstrap_available = False

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

    def fit_model(self, all_parents, tau_max=None, return_data=False):
        r"""Fit linear time series model.

        Fits a sklearn.linear_model.LinearRegression model to the parents of
        each variable and computes the coefficient matrices :math:`\Phi` and
        :math:`\Psi` as described in [4]_. Does accept contemporaneous links.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, 0), ...], 1:[], ...} containing
            the parents estimated with PCMCI.
        tau_max : int, optional (default: None)
            Maximum time lag. If None, the maximum lag in all_parents is used.
        return_data : bool, optional (default: False)
            Whether to save the data array. Needed to get residuals.
        """

        # Fit the model using the base class
        self.fit_results = self.fit_full_model(all_parents=all_parents,
                                        selected_variables=None,
                                        return_data=return_data,
                                        tau_max=tau_max)
        # Cache the results in the member variables
        coeffs = self.get_coefs()
        self.phi = self._get_phi(coeffs)
        self.psi = self._get_psi(self.phi)
        self.all_psi_k = self._get_all_psi_k(self.phi)

        self.all_parents = all_parents
        # self.tau_max = tau_max

    def fit_model_bootstrap(self, 
            boot_blocklength=1,
            seed=None,
            boot_samples=100):
        """Fits boostrap-versions of Phi, Psi, etc.

        Random draws are generated

        Parameters
        ----------
        boot_blocklength : int, or in {'cube_root', 'from_autocorrelation'}
            Block length for block-bootstrap. If 'cube_root' it is the cube 
            root of the time series length.
        seed : int, optional(default = None)
            Seed for RandomState (default_rng)
        boot_samples : int
            Number of bootstrap samples.
        """

        self.phi_boots = np.empty((boot_samples,) + self.phi.shape)
        self.psi_boots = np.empty((boot_samples,) + self.psi.shape)
        self.all_psi_k_boots = np.empty((boot_samples,) + self.all_psi_k.shape)

        if self.verbosity > 0:
            print("\n##\n## Generating bootstrap samples of Phi, Psi, etc "  +
                  "\n##\n" +
                  "\nboot_samples = %s \n" % boot_samples +
                  "\nboot_blocklength = %s \n" % boot_blocklength
                  )


        for b in range(boot_samples):
            # # Replace dataframe in method args by bootstrapped dataframe
            # method_args_bootstrap['dataframe'].bootstrap = boot_draw
            if seed is None:
                random_state = np.random.default_rng(None)
            else:
                random_state = np.random.default_rng(seed+b)

            dataframe_here = deepcopy(self.dataframe)

            dataframe_here.bootstrap = {'boot_blocklength':boot_blocklength,
                                        'random_state':random_state}
            model = Models(dataframe=dataframe_here,
                           model=sklearn.linear_model.LinearRegression(**self.model_params),
                           data_transform=self.data_transform,
                           mask_type=self.mask_type,
                           verbosity=0)

            model.fit_full_model(all_parents=self.all_parents,
                           tau_max=self.tau_max)
            # Cache the results in the member variables
            coeffs = model.get_coefs()
            phi = self._get_phi(coeffs)
            self.phi_boots[b] = phi
            self.psi_boots[b] = self._get_psi(phi)
            self.all_psi_k_boots[b] = self._get_all_psi_k(phi)

        self.bootstrap_available = True

        return self

    def get_bootstrap_of(self, function, function_args, conf_lev=0.9):
        """Applies bootstrap-versions of Phi, Psi, etc. to any function in 
        this class.

        Parameters
        ----------
        function : string
            Valid function from LinearMediation class
        function_args : dict
            Optional function arguments.
        conf_lev : float
            Confidence interval.

        Returns
        -------
        Upper/Lower confidence interval of function.
        """

        valid_functions = [
            'get_coeff',
            'get_ce',
            'get_ce_max',
            'get_joint_ce',
            'get_joint_ce_matrix',
            'get_mce',
            'get_conditional_mce',
            'get_joint_mce',
            'get_ace',
            'get_all_ace',
            'get_acs',
            'get_all_acs',
            'get_amce',
            'get_all_amce',
            'get_val_matrix',
            ]

        if function not in valid_functions:
            raise ValueError("function must be in %s" %valid_functions)

        realizations = self.phi_boots.shape[0]

        original_phi = deepcopy(self.phi)
        original_psi = deepcopy(self.psi)
        original_all_psi_k = deepcopy(self.all_psi_k)

        for r in range(realizations):
            self.phi = self.phi_boots[r]
            self.psi = self.psi_boots[r]
            self.all_psi_k = self.all_psi_k_boots[r]

            boot_effect = getattr(self, function)(**function_args)

            if r == 0:
                bootstrap_result = np.empty((realizations,) + boot_effect.shape)

            bootstrap_result[r] = boot_effect

        # Confidence intervals for val_matrix; interval is two-sided
        c_int = (1. - (1. - conf_lev)/2.)
        confidence_interval = np.percentile(
                bootstrap_result, axis=0,
                q = [100*(1. - c_int), 100*c_int])

        self.phi = original_phi
        self.psi = original_psi 
        self.all_psi_k = original_all_psi_k 
        self.bootstrap_result = bootstrap_result

        return confidence_interval


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
        """Returns the linear coefficient matrices for different lags.

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
        # phi[0] = np.identity(self.N)

        # Also includes contemporaneous lags
        for j in list(coeffs):
            for par in list(coeffs[j]):
                i, tau = par
                phi[abs(tau), j, i] = coeffs[j][par]

        return phi

    def _get_psi(self, phi):
        """Returns the linear causal effect matrices for different lags incl
        lag zero.

        Parameters
        ----------
        phi : array-like
            Coefficient matrices at different lags.

        Returns
        -------
        psi : array-like, shape (tau_max + 1, N, N)
            Matrices of causal effects for each time lag incl contemporaneous links.
        """

        psi = np.zeros((self.tau_max + 1, self.N, self.N))

        psi[0] = np.linalg.pinv(np.identity(self.N) - phi[0])
        for tau in range(1, self.tau_max + 1):
            for s in range(1, tau + 1):
                psi[tau] += np.matmul(psi[0], np.matmul(phi[s], psi[tau - s]) ) 

        # Lagged-only effects:
        # psi = np.zeros((self.tau_max + 1, self.N, self.N))

        # psi[0] = np.identity(self.N)
        # for n in range(1, self.tau_max + 1):
        #     psi[n] = np.zeros((self.N, self.N))
        #     for s in range(1, n + 1):
        #         psi[n] += np.dot(phi[s], psi[n - s])

        return psi

    def _get_psi_k(self, phi, k):
        """Returns the linear causal effect matrices excluding variable k.

        Essentially, this blocks all path through parents of variable k
        at any lag.

        Parameters
        ----------
        phi : array-like
            Coefficient matrices at different lags.
        k : int or list of ints
            Variable indices to exclude causal effects through.

        Returns
        -------
        psi_k : array-like, shape (tau_max + 1, N, N)
            Matrices of causal effects excluding k.
        """

        psi_k = np.zeros((self.tau_max + 1, self.N, self.N))
        
        phi_k = np.copy(phi)
        if isinstance(k, int):
            phi_k[:, k, :] = 0.
        else:
            for k_here in k:
                phi_k[:, k_here, :] = 0.


        psi_k[0] = np.linalg.pinv(np.identity(self.N) - phi_k[0])
        for tau in range(1, self.tau_max + 1):
            # psi_k[tau] = np.matmul(psi_k[0], np.matmul(phi_k[tau], psi_k[0]))
            for s in range(1, tau + 1):
                psi_k[tau] += np.matmul(psi_k[0], np.matmul(phi_k[s], psi_k[tau - s])) 


        # psi_k[0] = np.identity(self.N)
        # phi_k = np.copy(phi)
        # phi_k[:, k, :] = 0.
        # for n in range(1, self.tau_max + 1):
        #     psi_k[n] = np.zeros((self.N, self.N))
        #     for s in range(1, n + 1):
        #         psi_k[n] += np.dot(phi_k[s], psi_k[n - s])

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

    def get_coeff(self, i, tau, j):
        """Returns link coefficient.

        This is the direct causal effect for a particular link (i, -tau) --> j.

        Parameters
        ----------
        i : int
            Index of cause variable.
        tau : int
            Lag of cause variable (incl lag zero).
        j : int
            Index of effect variable.

        Returns
        -------
        coeff : float
        """
        return self.phi[abs(tau), j, i]

    def get_ce(self, i, tau, j):
        """Returns the causal effect.

        This is the causal effect for  (i, -tau) -- --> j.

        Parameters
        ----------
        i : int
            Index of cause variable.
        tau : int
            Lag of cause variable (incl lag zero).
        j : int
            Index of effect variable.

        Returns
        -------
        ce : float
        """
        return self.psi[abs(tau), j, i]

    def get_ce_max(self, i, j):
        """Returns the causal effect.

        This is the maximum absolute causal effect for  i --> j across all
        lags (incl lag zero).

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
        argmax = np.abs(self.psi[:, j, i]).argmax()
        return self.psi[:, j, i][argmax]

    def get_joint_ce(self, i, j):
        """Returns the joint causal effect.

        This is the causal effect from all lags [t, ..., t-tau_max]
        of i on j at time t. Note that the joint effect does not
        count links passing through parents of i itself.

        Parameters
        ----------
        i : int
            Index of cause variable.
        j : int
            Index of effect variable.

        Returns
        -------
        joint_ce : array of shape (tau_max + 1)
            Causal effect from each lag [t, ..., t-tau_max] of i on j.
        """
        joint_ce = self.all_psi_k[i, :, j, i]
        return joint_ce

    def get_joint_ce_matrix(self, i, j):
        """Returns the joint causal effect matrix of i on j.

        This is the causal effect from all lags [t, ..., t-tau_max]
        of i on j at times [t, ..., t-tau_max]. Note that the joint effect does not
        count links passing through parents of i itself.

        An entry (taui, tauj) stands for the effect of i at t-taui on j at t-tauj.

        Parameters
        ----------
        i : int
            Index of cause variable.
        j : int
            Index of effect variable.

        Returns
        -------
        joint_ce_matrix : 2d array of shape (tau_max + 1, tau_max + 1)
            Causal effect matrix from each lag of i on each lag of j.
        """
        joint_ce_matrix = np.zeros((self.tau_max + 1, self.tau_max + 1))
        for tauj in range(self.tau_max + 1):
            joint_ce_matrix[tauj:, tauj] = self.all_psi_k[i, tauj:, j, i][::-1]

        return joint_ce_matrix

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
        k : int or list of ints
            Indices of mediator variables.

        Returns
        -------
        mce : float
        """
        if isinstance(k, int):
            effect_without_k = self.all_psi_k[k, abs(tau), j, i]
        else:
            effect_without_k = self._get_psi_k(self.phi, k=k)[abs(tau), j, i]

        mce = self.psi[abs(tau), j, i] - effect_without_k
        return mce

    def get_conditional_mce(self, i, tau, j, k, notk):
        """Returns the conditional mediated causal effect.

        This is the causal effect for  i --> j for all paths going through k, but not through notk.

        Parameters
        ----------
        i : int
            Index of cause variable.
        tau : int
            Lag of cause variable.
        j : int
            Index of effect variable.
        k : int or list of ints
            Indices of mediator variables.
        notk : int or list of ints
            Indices of mediator variables to exclude.

        Returns
        -------
        mce : float
        """
        if isinstance(k, int):
            k = set([k])
        else:
            k = set(k)
        if isinstance(notk, int):
            notk = set([notk])
        else:
            notk = set(notk)

        bothk = list(k.union(notk))
        notk = list(notk)
  
        effect_without_bothk = self._get_psi_k(self.phi, k=bothk)[abs(tau), j, i]
        effect_without_notk = self._get_psi_k(self.phi, k=notk)[abs(tau), j, i]

        # mce = self.psi[abs(tau), j, i] - effect_without_k
        mce = effect_without_notk - effect_without_bothk

        return mce


    def get_joint_mce(self, i, j, k):
        """Returns the joint causal effect mediated through k.

        This is the mediated causal effect from all lags [t, ..., t-tau_max]
        of i on j at time t for paths through k. Note that the joint effect
        does not count links passing through parents of i itself.

        Parameters
        ----------
        i : int
            Index of cause variable.
        j : int
            Index of effect variable.
        k : int or list of ints
            Indices of mediator variables.

        Returns
        -------
        joint_mce : array of shape (tau_max + 1)
            Mediated causal effect from each lag [t, ..., t-tau_max] of i on j through k.
        """
        if isinstance(k, int):
            k_here = [k]

        effect_without_k = self._get_psi_k(self.phi, k=[i] + k_here)

        joint_mce = self.all_psi_k[i, :, j, i] - effect_without_k[:, j, i]
        return joint_mce

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
            return np.abs(self.psi[:, all_but_i, i]).max(axis=0).mean()
        elif lag_mode == 'all_lags':
            return np.abs(self.psi[:, all_but_i, i]).mean()
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
            return np.abs(self.psi[:, j, all_but_j]).max(axis=0).mean()
        elif lag_mode == 'all_lags':
            return np.abs(self.psi[:, j, all_but_j]).mean()
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
            weights = np.ones((N_new, N_new), dtype='bool')

        # if self.tau_max < 2:
        #     raise ValueError("Mediation only nonzero for tau_max >= 2")

        all_mce = self.psi[:, :, :] - self.all_psi_k[k, :, :, :]
        # all_mce[:, range(self.N), range(self.N)] = 0.

        if lag_mode == 'absmax':
            return np.average(np.abs(all_mce[:, all_but_k, :]
                                     [:, :, all_but_k]
                                     ).max(axis=0), weights=weights)
        elif lag_mode == 'all_lags':
            return np.abs(all_mce[:, all_but_k, :][:, :, all_but_k]).mean()
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


    def get_val_matrix(self, symmetrize=False):
        """Returns the matrix of linear coefficients.

        Requires fit_model() before. An entry val_matrix[i,j,tau] gives the
        coefficient of the link from i to j at lag tau. Lag=0 is always set
        to zero for LinearMediation, use Models class for contemporaneous 
        models.

        Parameters
        ----------
        symmetrize : bool
            If True, the lag-zero entries will be symmetrized such that
            no zeros appear. Useful since other parts of tigramite 
            through an error for non-symmetric val_matrix, eg plotting.

        Returns
        -------
        val_matrix : array
            Matrix of linear coefficients, shape (N, N, tau_max + 1).
        """
        val_matrix = np.copy(self.phi.transpose())
        N = val_matrix.shape[0]

        if symmetrize:
            # Symmetrize since otherwise other parts of tigramite through an error
            for i in range(N):
                for j in range(N):
                    if val_matrix[i,j, 0] == 0.:
                        val_matrix[i,j, 0] = val_matrix[j,i, 0]

        return val_matrix

    def net_to_tsg(self, row, lag, max_lag):
        """Helper function to translate from network to time series graph."""
        return row * max_lag + lag

    def tsg_to_net(self, node, max_lag):
        """Helper function to translate from time series graph to network."""
        row = node // max_lag
        lag = node % max_lag
        return (row, -lag)

    def get_tsg(self, link_matrix, val_matrix=None, include_neighbors=False):
        """Returns time series graph matrix.

        Constructs a matrix of shape (N*tau_max, N*tau_max) from link_matrix.
        This matrix can be used for plotting the time series graph and analyzing
        causal pathways.

        Parameters
        ----------
        link_matrix : bool array-like, optional (default: None)
            Matrix of significant links. Must be of same shape as val_matrix.
            Either sig_thres or link_matrix has to be provided.
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
        tsg = np.zeros((N * max_lag, N * max_lag))
        for i, j, tau in np.column_stack(np.where(link_matrix)):
            # if tau > 0 or include_neighbors:
                for t in range(max_lag):
                    link_start = self.net_to_tsg(i, t - tau, max_lag)
                    link_end = self.net_to_tsg(j, t, max_lag)
                    if (0 <= link_start and
                            (link_start % max_lag) <= (link_end % max_lag)):
                        if val_matrix is not None:
                            tsg[link_start, link_end] = val_matrix[i, j, tau]
                        else:
                            tsg[link_start, link_end] = 1
        return tsg

    def get_mediation_graph_data(self, i, tau, j, include_neighbors=False):
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
                           - self.all_psi_k)[:, abs(tau), j, i]

        # Get involved links
        val_matrix = self.phi.transpose()
        link_matrix = val_matrix != 0.

        max_lag = link_matrix.shape[2] + 1

        # include_neighbors = False because True would allow
        # --> o -- motifs in networkx.all_simple_paths as paths, but
        # these are blocked...
        tsg = self.get_tsg(link_matrix, val_matrix=val_matrix,
                           include_neighbors=False)

        if include_neighbors:
            # Add contemporaneous links only at source node
            for m, n in zip(*np.where(link_matrix[:, :, 0])):
                # print m,n
                if m != n:
                    tsg[self.net_to_tsg(m, max_lag - tau - 1, max_lag),
                        self.net_to_tsg(n, max_lag - tau - 1, max_lag)
                    ] = val_matrix[m, n, 0]

        tsg_path_val_matrix = np.zeros(tsg.shape)

        graph = networkx.DiGraph(tsg)
        pathways = []

        for path in networkx.all_simple_paths(graph,
                                              source=self.net_to_tsg(i,
                                                                     max_lag - tau - 1,
                                                                     max_lag),
                                              target=self.net_to_tsg(j,
                                                                     max_lag - 0 - 1,
                                                                     max_lag)):
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

        graph_data = {'path_node_array': path_node_array,
                      'path_val_matrix': path_val_matrix,
                      'tsg_path_val_matrix': tsg_path_val_matrix}

        return graph_data


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

        if dataframe.analysis_mode != 'single':
            raise ValueError("Prediction class currently only supports single "
                             "datasets.")

        # dataframe.values = {0: dataframe.values[0]}

        # Default value for the mask
        if dataframe.mask is not None:
            mask = {0: dataframe.mask[0]}
        else:
            mask = {0: np.zeros(dataframe.values[0].shape, dtype='bool')}
        # Get the dataframe shape
        T = dataframe.T[0]

        # Have the default dataframe be the training data frame
        train_mask = deepcopy(mask)
        train_mask[0][[t for t in range(T) if t not in train_indices]] = True
        self.dataframe = deepcopy(dataframe)
        self.dataframe.mask = train_mask
        self.dataframe._initialized_from = 'dict'
                 # = DataFrame(dataframe.values[0],
                 #                   mask=train_mask,
                 #                   missing_flag=dataframe.missing_flag)
        # Initialize the models baseclass with the training dataframe
        Models.__init__(self,
                        dataframe=self.dataframe,
                        model=prediction_model,
                        data_transform=data_transform,
                        mask_type='y',
                        verbosity=verbosity)

        # Build the testing dataframe as well
        self.test_mask = deepcopy(mask)
        self.test_mask[0][[t for t in range(T) if t not in test_indices]] = True

        self.train_indices = train_indices
        self.test_indices = test_indices

        # Setup the PCMCI instance
        if cond_ind_test is not None:
            # Force the masking
            cond_ind_test.set_mask_type('y')
            cond_ind_test.verbosity = verbosity
            # PCMCI.__init__(self,
            #                dataframe=self.dataframe,
            #                cond_ind_test=cond_ind_test,
            #                verbosity=verbosity)
            self.pcmci = PCMCI(dataframe=self.dataframe,
                               cond_ind_test=cond_ind_test,
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

        if selected_links is not None:
            link_assumptions = {}
            for j in selected_links.keys():
                link_assumptions[j] = {(i, -tau):"-?>" for i in range(self.N) for tau in range(1, tau_max+1)}
        else:
            link_assumptions = None

        # Ensure an independence model is given
        if self.cond_ind_test is None:
            raise ValueError("No cond_ind_test given!")
        # Set the selected variables
        self.selected_variables = range(self.N)
        if selected_targets is not None:
            self.selected_variables = selected_targets
        
        predictors = self.pcmci.run_pc_stable(link_assumptions=link_assumptions,
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

        Wrapper around ``Models.fit_full_model()``. To each variable in
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

        if selected_targets is None:
            self.selected_targets = range(self.N)
        else:
            self.selected_targets = selected_targets

        if tau_max is None:
            # Find the maximal parents lag
            max_parents_lag = 0
            for j in self.selected_targets:
                if target_predictors[j]:
                    this_parent_lag = np.abs(np.array(target_predictors[j])[:, 1]).max()
                    max_parents_lag = max(max_parents_lag, this_parent_lag)
        else:
            max_parents_lag = tau_max

        if len(set(np.array(self.test_indices) - max_parents_lag)
                .intersection(self.train_indices)) > 0:
            if self.verbosity > 0:
                warnings.warn("test_indices - maxlag(predictors) [or tau_max] "
                "overlaps with train_indices: Choose test_indices "
                "such that there is a gap of max_lag to train_indices!")

        self.target_predictors = target_predictors

        for target in self.selected_targets:
            if target not in list(self.target_predictors):
                raise ValueError("No predictors given for target %s" % target)

        self.fitted_model = \
            self.fit_full_model(all_parents=self.target_predictors,
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

        If target is an int, the predicted time series is returned. If target
        is a list of integers, then a list of predicted time series is returned.
        If the list of integers equals range(N), then an array of shape (T, N)
        of the predicted series is returned.

        Parameters
        ----------
        target : int or list of integers
            Index or indices of target variable(s).
        new_data : data object, optional
            New Tigramite dataframe object with optional new mask. Note that
            the data will be cut off according to cut_off, see parameter
            `cut_off` below.
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

        if isinstance(target, int):
            target_list = [target]
        elif isinstance(target, list):
            target_list = target
        else:
            raise ValueError("target must be either int or list of integers "
                             "indicating the index of the variables to "
                             "predict.")

        if target_list == list(range(self.N)):
            return_type = 'array'
        elif len(target_list) == 1:
            return_type = 'series'
        else:
            return_type = 'list'

        pred_list = []
        self.stored_test_array = {}
        for target in target_list:
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
            Y = [(target, 0)]  # dummy
            X = [(target, 0)]  # dummy
            Z = self.target_predictors[target]

            # Check if we've passed a new dataframe object
            if new_data is not None:
                # if new_data.mask is None:
                #     # if no mask is supplied, use the same mask as for the fitted array
                #     new_data_mask = self.test_mask
                # else:
                new_data_mask = new_data.mask
                test_array, _, _ = new_data.construct_array(X, Y, Z,
                                                         tau_max=self.tau_max,
                                                         mask=new_data_mask,
                                                         mask_type=self.mask_type,
                                                         cut_off=cut_off,
                                                         remove_overlaps=True,
                                                         verbosity=self.verbosity)
            # Otherwise use the default values
            else:
                test_array, _, _ = \
                    self.dataframe.construct_array(X, Y, Z,
                                                   tau_max=self.tau_max,
                                                   mask=self.test_mask,
                                                   mask_type=self.mask_type,
                                                   cut_off=cut_off,
                                                   remove_overlaps=True,
                                                   verbosity=self.verbosity)
            # Transform the data if needed
            a_transform = self.fitted_model[target]['data_transform']
            if a_transform is not None:
                test_array = a_transform.transform(X=test_array.T).T
            # Cache the test array
            self.stored_test_array[target] = test_array
            # Run the predictor
            predicted = self.fitted_model[target]['model'].predict(
                X=test_array[2:].T, **pred_params)

            if test_array[2:].size == 0:
                # If there are no predictors, return the value of 
                # empty_predictors_function, which is np.mean 
                # and expand to the test array length
                predicted = predicted * np.ones(test_array.shape[1])

            pred_list.append(predicted)

        if return_type == 'series':
            return pred_list[0]
        elif return_type == 'list':
            return pred_list
        elif return_type == 'array':
            return np.array(pred_list).transpose()

    def get_train_array(self, j):
        """Returns training array for variable j."""
        return self.fitted_model[j]['data']

    def get_test_array(self, j):
        """Returns test array for variable j."""
        return self.stored_test_array[j]

if __name__ == '__main__':
   
    import tigramite
    import tigramite.data_processing as pp
    from tigramite.toymodels import structural_causal_processes as toys
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.plotting as tp

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.multioutput import MultiOutputRegressor

    def lin_f(x): return x
 

    T = 1000
    def lin_f(x): return x
    auto_coeff = 0.
    coeff = 2.
    links = {
            0: [((0, -1), auto_coeff, lin_f)], 
            1: [((1, -1), auto_coeff, lin_f), ((0, 0), coeff, lin_f)],
            }
    data, nonstat = toys.structural_causal_process(links, T=T, 
                                noises=None, seed=7)

    # data[:,1] = data[:,1] > 0.

    # # Create some missing values
    # data[-10:,:] = 999.
    # var_names = range(2)

    # graph = np.array([['', '-->'],
    #                   ['<--', '']], 
    #                   dtype='<U3')
    print(data, data.mean(axis=0))
    dataframe = pp.DataFrame(data,
                    # vector_vars={0:[(0,0), (1,0)], 1:[(2,0), (3,0)]}
                    ) 
    graph = toys.links_to_graph(links, tau_max=4)
    
    # # We are interested in lagged total effect of X on Y
    X = [(0, 0), (0, -1)]
    Y = [(1, 0), (1, -1)]

    model = Models(dataframe=dataframe, 
        model = LinearRegression(),
        # model = LogisticRegression(),
        # model = MultiOutputRegressor(LogisticRegression()),

        )

    model.get_general_fitted_model( 
                    Y=Y, X=X, Z=[(0, -2)],
                    conditions=[(0, -3)],
                    tau_max=7,
                    cut_off='tau_max',
                    empty_predictors_function=np.mean,
                    return_data=False)

    # print(model.fit_results[(1, 0)]['model'].coef_)

    dox_vals = np.array([0.])   #np.linspace(-1., 1., 1)
    intervention_data = np.tile(dox_vals.reshape(len(dox_vals), 1), len(X))

    conditions_data = np.tile(1. + dox_vals.reshape(len(dox_vals), 1), 1)

    def aggregation_func(x, axis=0, bins=2):
        x = x.astype('int64')
        return np.apply_along_axis(np.bincount, axis=axis, arr=x, minlength=bins).T
    aggregation_func = np.mean

    pred = model.get_general_prediction(
                intervention_data=intervention_data,
                conditions_data=conditions_data,
                pred_params=None,
                transform_interventions_and_prediction=False,
                return_further_pred_results=False,
                aggregation_func=aggregation_func,
                )

    print("\n", pred)

    # T = 1000
    
    # links = {0: [((0, -1), 0.9, lin_f)],
    #          1: [((1, -1), 0.9, lin_f), ((0, 0), -0.8, lin_f)],
    #          2: [((2, -1), 0.9, lin_f), ((0, 0), 0.9, lin_f),  ((1, 0), 0.8, lin_f)],
    #          # 3: [((3, -1), 0.9, lin_f), ((1, 0), 0.8, lin_f),  ((2, 0), -0.9, lin_f)]
    #          }
    # # noises = [np.random.randn for j in links.keys()]
    # data, nonstat = toys.structural_causal_process(links, T=T, noises=None, seed=7)

    # missing_flag = 999
    # for i in range(0, 20):
    #     data[i::100] = missing_flag

    # # mask = data>0

    # parents = toys._get_true_parent_neighbor_dict(links)
    # dataframe = pp.DataFrame(data,  missing_flag = missing_flag)



    # model = LinearRegression()
    # model.fit(X=np.random.randn(10,2), y=np.random.randn(10))
    # model.predict(X=np.random.randn(10,2)[:,2:])
    # sys.exit(0)

    # med = LinearMediation(dataframe=dataframe, #mask_type='y',
    #     data_transform=None)
    # med.fit_model(all_parents=parents, tau_max=None,  return_data=True)

    # print(med.get_residuals_cov_mean())

    # med.fit_model_bootstrap( 
    #             boot_blocklength='cube_root',
    #             seed = 42,
    #             )

    # # print(med.get_val_matrix())

    # print (med.get_ce(i=0, tau=0,  j=3))
    # print(med.get_bootstrap_of(function='get_ce', 
    #     function_args={'i':0, 'tau':0,   'j':3}, conf_lev=0.9))

    # print (med.get_coeff(i=0, tau=-2, j=1))

    # print (med.get_ce_max(i=0, j=2))
    # print (med.get_ce(i=0, tau=0, j=3))
    # print (med.get_mce(i=0, tau=0, k=[2], j=3))
    # print (med.get_mce(i=0, tau=0, k=[1,2], j=3) - med.get_mce(i=0, tau=0, k=[1], j=3))
    # print (med.get_conditional_mce(i=0, tau=0, k=[2], notk=[1], j=3))
    # print (med.get_bootstrap_of('get_conditional_mce', {'i':0, 'tau':0, 'k':[2], 'notk':[1], 'j':3}))

    # print(med.get_joint_ce(i=0, j=2))
    # print(med.get_joint_mce(i=0, j=2, k=1))

    # print(med.get_joint_ce_matrix(i=0, j=2))

    # i=0; tau=4; j=2
    # graph_data = med.get_mediation_graph_data(i=i, tau=tau, j=j)
    # tp.plot_mediation_time_series_graph(
    #     # var_names=var_names,
    #     path_node_array=graph_data['path_node_array'],
    #     tsg_path_val_matrix=graph_data['tsg_path_val_matrix']
    #     )
    # tp.plot_mediation_graph(
    #                     # var_names=var_names,
    #                     path_val_matrix=graph_data['path_val_matrix'], 
    #                     path_node_array=graph_data['path_node_array'],
    #                     ); 
    # plt.show()

    # print ("Average Causal Effect X=%.2f, Y=%.2f, Z=%.2f " % tuple(med.get_all_ace()))
    # print ("Average Causal Susceptibility X=%.2f, Y=%.2f, Z=%.2f " % tuple(med.get_all_acs()))
    # print ("Average Mediated Causal Effect X=%.2f, Y=%.2f, Z=%.2f " % tuple(med.get_all_amce()))
    # med = Models(dataframe=dataframe, model=sklearn.linear_model.LinearRegression(), data_transform=None)
    # # Fit the model
    # med.get_fit(all_parents=true_parents, tau_max=3)

    # print(med.get_val_matrix())

    # for j, i, tau, coeff in toys._iter_coeffs(links):
    #     print(i, j, tau, coeff, med.get_coeff(i=i, tau=tau, j=j))

    # for causal_coeff in [med.get_ce(i=0, tau=-2, j=2),
    #                      med.get_mce(i=0, tau=-2, j=2, k=1)]:
    #     print(causal_coeff)


    # pred = Prediction(dataframe=dataframe,
    #         cond_ind_test=ParCorr(),   #CMIknn ParCorr
    #         prediction_model = sklearn.linear_model.LinearRegression(),
    # #         prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
    #         # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
    #     data_transform=sklearn.preprocessing.StandardScaler(),
    #     train_indices= list(range(int(0.8*T))),
    #     test_indices= list(range(int(0.8*T), T)),
    #     verbosity=0
    #     )

    # # predictors = pred.get_predictors(
    # #                        selected_targets=[2],
    # #                        selected_links=None,
    # #                        steps_ahead=1,
    # #                        tau_max=1,
    # #                        pc_alpha=0.2,
    # #                        max_conds_dim=None,
    # #                        max_combinations=1)
    # predictors = {0: [], # [(0, -1)],
    #              1: [(1, -1), (0, -1)],
    #              2: [(2, -1), (1, 0)]}
    # pred.fit(target_predictors=predictors,
    #         selected_targets=None, tau_max=None, return_data=False)

    # res = pred.predict(target=0,
    #             new_data=None,
    #             pred_params=None,
    #             cut_off='max_lag_or_tau_max')

    # print(data[:,2])
    # print(res)


