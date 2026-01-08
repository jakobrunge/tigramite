"""Tigramite causal discovery for time series."""

# Authors: Elena Saggioro, Sagar Simha, Matthias Bruhns, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from copy import deepcopy
import numpy as np
import sklearn
from joblib import Parallel, delayed
from ortools.linear_solver import pywraplp
import traceback

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.data_processing import DataFrame
from tigramite.models import Prediction
from tigramite.pcmci import PCMCI

class RPCMCI(PCMCI):
    r"""RPCMCI class for extracting causal regimes and the associated graphs from
        time series data.

        Notes
        -----
        The Regime-PCMCI causal discovery method is described in: 

        Elena Saggioro, Jana de Wiljes, Marlene Kretschmer, Jakob Runge;
        Reconstructing regime-dependent causal relationships from observational
        time series. Chaos 1 November 2020; 30 (11): 113115.
        https://doi.org/10.1063/5.0020538

        The method iterates between two phases --a regime learning phase
        (optimization-based) and a causal discovery phase (PCMCI)-- to identify
        regime dependent causal relationships. A persistent discrete regime
        variable is assumed that leads to a finite number of regimes within which
        stationarity can be assumed.

        Parameters
        ----------
        dataframe : data object
            This is the Tigramite dataframe object. It has the attributes
            dataframe.values yielding a numpy array of shape ( observations T,
            variables N). For RPCMCI the mask will be ignored. You may use the
            missing_flag to indicate missing values.
        cond_ind_test : conditional independence test object
            This can be ParCorr or other classes from
            ``tigramite.independence_tests`` or an external test passed as a
            callable. This test can be based on the class
            tigramite.independence_tests.CondIndTest.
        prediction_model : sklearn model object
            For example, sklearn.linear_model.LinearRegression() for a linear
            regression model. This should be consistent with cond_ind_test, ie, 
            use ParCorr() with a linear model and, eg, GPDC() with a 
            GaussianProcessRegressor model, or CMIknn with NearestNeighbors model.
        seed : int
            Random seed for annealing step.
        verbosity : int, optional (default: -1)
            Verbose levels -1, 0, 1, ...
        """

    def __init__(self, dataframe, cond_ind_test=None, 
                    prediction_model=None, seed=None, verbosity=-1):

        self.verbosity = verbosity

        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 1000)

        # Set prediction model to be used in optimization
        self.prediction_model = prediction_model
        if self.prediction_model is None:
            self.prediction_model = sklearn.linear_model.LinearRegression()

        # Set conditional independence test
        if cond_ind_test is None:
            cond_ind_test = ParCorr()
        cond_ind_test.set_mask_type('y')

        if dataframe.analysis_mode != 'single':
            raise ValueError("Only single time series data allowed for RPCMCI.")
   
        if dataframe.has_vector_data:
            raise ValueError("Only scalar data allowed for RPCMCI.")
        
               
        # Masking is not available in RPCMCI, but missing values can be specified
        dataframe.mask = {0:np.zeros(dataframe.values[0].shape, dtype='bool')}
        self.missing_flag = dataframe.missing_flag

        # Init base class
        PCMCI.__init__(self, dataframe=dataframe,
                        cond_ind_test=cond_ind_test,
                        verbosity=0)

    def run_rpcmci(self,
                   num_regimes,
                   max_transitions,
                   switch_thres=0.05,
                   num_iterations=20,
                   max_anneal=10,
                   tau_min=1,
                   tau_max=1,
                   pc_alpha=0.2,
                   alpha_level=0.01,
                   n_jobs=-1,
                   ):

        """Run RPCMCI method for extracting causal regimes and the associated graphs from
            time series data.

        Parameters
        ----------
        num_regimes : int
            Number of assumed regimes.
        max_transitions : int
            Maximum number of transitions within a single regime (persistency parameter).
        switch_thres : float
            Switch threshold.
        num_iterations : int
            Optimization iterations.
        max_anneal : int
            Maximum annealing runs.
        tau_min : int, optional (default: 0)
            Minimum time lag to test.
        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.
        pc_alpha : float, optional (default: 0.2)
            Significance level in PCMCI.
        alpha_level : float, optional (default: 0.05)
            Significance level in PCMCI at which the p_matrix is thresholded to
            get graph.
        n_jobs : int, optional (default: -1)
            Number of CPUs to use in joblib parallization. Default n_jobs=-1
            uses all available.

        Returns
        -------
        regimes : array of shape (n_regimes, T)
            One-hot encoded regime variable.
        causal_results: dictionary
            Contains result of run_pcmci() after convergence.
        diff_g_f : tuple
            Difference between two consecutive optimizations for all annealings and
            the optimal one with minimum objective value (see paper).
        error_free_annealings : int
            Number of annealings that converged without error.
        """

        count_saved_ann = 0
        # initialize residuals (objective value) of MIP optimize
        objmip_ann = [None] * max_anneal
        parents_ann = [None] * max_anneal
        causal_prediction = [None] * max_anneal
        links_ann = [None] * max_anneal
        gamma_ann = [None] * max_anneal
        diff_g_ann = [None] * max_anneal
        q_break_cycle = 5

        data = self.dataframe.values[0]

        def _pcmci(tau_min, tau_max, pc_alpha, alpha_level):
            """Wrapper around running PCMCI."""
            results = self.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha_level)
            graph = results['graph'] 
            pcmci_parents = self.return_parents_dict(graph=graph, val_matrix=results['val_matrix'])
            return results, graph, pcmci_parents

        def _optimize_gamma(resid_sq, max_transitions):
            r"""
            Solves the following optimization problem :

                minimize       c * x

            where c = resid_sq , flattened along num_regimes dimension
                  x = Gamma , flattened along num_regimes dimension

            with Constraints:
                    (1) [\sum_{k=1,num_regimes}gamma^k(t) ]= 1
                                            forall t    : uniqueness
                    (2) [\sum_{t=1:T-1} | gamma^k(t+1) - gamma^k(t) | ] <= max_transitions
                                            forall k    : persistence


            Inputs:
             resid_sq ( np.shape = (num_regimes,T) )
             max_transitions = max number of switchings allowed

            Returns:
            Gamma_updated ( np.shape = (num_regimes,T) ))
            """

            num_regimes, T = resid_sq.shape

            # Create the linear solver with the GLOP backend.
            solver = pywraplp.Solver.CreateSolver("GLOP")
            infinity = solver.infinity()

            # Define vector of integer variables in the interval [0,1].
            G = [solver.NumVar(0, 1, f"x_{i}") for i in range(num_regimes * T)]

            # Define eta, auxiliary vars for constr. (2).
            E = [solver.NumVar(0, infinity, f"eta_{i}") for i in range(num_regimes * T - 1)]
            X = G + E
            solver.Minimize(
                sum([resid_sq[k, t] * X[k * T + t] for k in range(num_regimes) for t in range(T)])
            )

            con_lst = [sum([X[k * T + t] for k in range(num_regimes)]) for t in range(T)]
            for t in range(T):
                solver.Add(con_lst[t] == 1)

            for k in range(num_regimes):
                for t in range(T - 1):
                    # (2.1)
                    solver.Add(
                        (X[k * T + t + 1] - X[k * T + t] - X[k * T + t + num_regimes * T] <= 0)
                    )
                    # (2.2)
                    solver.Add(
                        (
                            (
                                    -1 * X[k * T + t + 1]
                                    + X[k * T + t]
                                    - X[k * T + t + num_regimes * T]
                                    <= 0
                            )
                        )
                    )
                    # (2.3)
                    solver.Add(
                        ((sum([X[k * T + t + num_regimes * T] for t in range(T - 1)]) <= max_transitions))
                    )

            status = solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                if self.verbosity > 0:
                    print("\nOptimal objective: reached.")
                gamma = np.reshape([g.solution_value() for g in G], (num_regimes, T))
                obj_value = solver.Objective().Value()
                return gamma, obj_value
            else:
                # if self.verbosity > -1:
                #     print("The problem does not have an optimal solution. Please change hyperparameters.")
                raise ValueError("The problem does not have an optimal solution. Please change hyperparameters.")

        def one_annealing_step(a):
            """Executes one annealing step. The random seed is self.seed + a."""

            if self.verbosity > 0:
                print(f"\n################# Annealing iteration a = {a} ####################\n")

            T = self.dataframe.T[0]

            # Initialise gamma_0 as random matrix of 1s and 0s
            random_state = np.random.default_rng(self.seed + a)
            gamma_opt = random_state.uniform(0, 1, size=(num_regimes, T))  # range is [0,1)!

            parents_opt = {} # [None] * num_regimes
            results_opt = {} # [None] * num_regimes
            links_opt = {} # [None] * num_regimes
            objective_opt = 0

            # Difference between two consecutive optimizations
            diff_g = []

            #
            # Iteration over 1. causal discovery and 2. constrained optimization
            #
            error_flag = False
            for q in range(num_iterations):
                if self.verbosity > 0:
                    print(f"\n###### Optimization step q = {q}")

                # Initialize to 0
                residuals = np.zeros((num_regimes, T, self.N))

                gamma_temp = deepcopy(gamma_opt)

                #
                # 1. Causal discovery and prediction
                # 

                # Iterate over regimes
                for k in range(num_regimes):
                    if self.verbosity > 0:
                        print(f"{16 * '#'} Regime k = {k}")

                    # Select sample according to gamma_opt, is a bool vector
                    selected_samples_k = (gamma_temp[k, :] > switch_thres)

                    mask_of_k = np.ones(data.shape, dtype="bool")
                    mask_of_k[selected_samples_k] = False

                    # df_of_k = pp.DataFrame(data, mask=mask_of_k, missing_flag=self.missing_flag,
                    #              var_names=self.var_names)

                    # Change mask in dataframe for this step
                    self.dataframe.mask[0] = mask_of_k

                    if np.any((mask_of_k == False).sum(axis=0) <= 5):
                        error_flag = True
                        if self.verbosity > 0:
                            print(f"*****Regime with too few samples in annealing a = {a} at iteration q = {q}.*****\n")
                        if self.verbosity > 0:
                            print("***** Break k-loop of regimes *****\n ")
                        break  # from k-loop         

                    try: 
                        # cond_ind_test = getattr(self, method)(**method_args)
                        # pcmci = PCMCI(dataframe=df_of_k, 
                        #     cond_ind_test=self.cond_ind_test, 
                        #     verbosity=0)
                        results_temp, link_temp, parents_temp = _pcmci(
                                        # pcmci,
                                                                        tau_max=int(tau_max),
                                                                        pc_alpha=pc_alpha,
                                                                        alpha_level=alpha_level,
                                                                        tau_min=tau_min,)
                    except Exception:
                        traceback.print_exc()
                        error_flag = True
                        print(f"*****Value error in causal discovery for annealing a = {a} at iteration q = {q}.*****\n")
                        print("***** Break k-loop of regimes *****\n ")
                        break  # from k-loop

                    parents_opt[k] = parents_temp
                    results_opt[k] = results_temp
                    links_opt[k] = link_temp

                    try: 
                        # Prediction with causal parents
                        pred = Prediction(
                            dataframe=self.dataframe,
                            prediction_model=self.prediction_model,
                            data_transform=sklearn.preprocessing.StandardScaler(),
                            train_indices=range(T),
                            test_indices=range(T),
                            verbosity=0,
                        )

                        pred.fit(
                            target_predictors=parents_temp,
                            selected_targets=range(self.N),
                            tau_max=int(tau_max),
                        ) 
                        # print(parents_temp)
                        # Compute the predicted residuals for each variable
                        predicted = pred.predict(
                            target=list(range(self.N)), 
                            new_data=DataFrame(data, missing_flag=self.missing_flag)
                        )

                        original_data = np.zeros(predicted.shape)
                        for target in range(self.N):
                            # print(data.shape, predicted.shape, original_data.shape, pred.get_test_array(target).shape, mask_of_k.sum(axis=0))
                            # print(pred.get_test_array(target)[0].flatten().std())
                            original_data[:, target] = pred.get_test_array(target)[0].flatten()

                    except Exception:
                        traceback.print_exc()
                        error_flag = True
                        print(f"*****Value error in prediction for annealing a = {a} at iteration q = {q}.*****\n")
                        print("***** Break k-loop of regimes *****\n ")
                        break  # from k-loop


                    # Get residuals
                    residuals[k, int(tau_max):, :] = original_data - predicted
                    # print(np.abs(residuals[k, int(tau_max):, :]).mean(axis=0))

                if error_flag:
                    if self.verbosity > 0:
                        print(f"***** Break q-loop of optimization iterations for Annealing a = {a} at iteration q = {q}." 
                            " Go to next annealing step. *****\n")
                    break

                #
                # 2. Regime optimization step with side constraints
                #

                # Comute the resid_sq
                res_sq = np.square(residuals).sum(axis=-1)
                # print(res_sq.shape)

                try:
                    # Optimization
                    gamma_opt, objective_opt = _optimize_gamma(res_sq, max_transitions)

                except Exception:
                    traceback.print_exc()
                    error_flag = True
                    print(f"*****Value error in optimization for annealing a = {a} at iteration q = {q}.*****\n")
                    break  

                diff_g.append(np.sum(np.abs(gamma_opt - gamma_temp)))

                if self.verbosity > 0:
                    print(f"Difference in abs value between the previous and current gamma "
                        f"(shape num_regimesxT) : {diff_g[q]}")

                # Break conditions
                if diff_g[-1] == 0:
                    if self.verbosity > 0:
                        print("Two consecutive gammas are equal: (local) minimum reached. "
                            "Go to next annealing.\n")
                    break

                if (q >= q_break_cycle) and (diff_g[-1] <= (2 * num_regimes * T // 100)):
                    if self.verbosity > 0:
                        print(f"Iteration larger than {q_break_cycle} and two consecutive gammas are too similar. "
                        f"Go to next annealing.\n")
                    break

            if error_flag:
                if self.verbosity > 0:
                    print(f"*****Annealing a = {a} failed****\n")

                return None

            return a, objective_opt, parents_opt, results_opt, links_opt, gamma_opt, diff_g

        # Parallelizing over annealing steps
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(one_annealing_step)(a) for a in range(max_anneal))

        # all_results = []
        # for a in range(max_anneal):
        #     all_results.append(one_annealing_step(a))

        error_free_annealings = 0
        for result in all_results:
            if result is not None:
                error_free_annealings += 1
                a, objective_opt, parents_opt, results_opt, links_opt, gamma_opt, diff_g = result
                
                # Save annealing results
                objmip_ann[a] = objective_opt  
                parents_ann[a] = parents_opt  
                causal_prediction[a] = results_opt
                links_ann[a] = links_opt
                gamma_ann[a] = gamma_opt 
                diff_g_ann[a] = diff_g 

        if error_free_annealings == 0:
            print("No annealings have converged. Run failed.")
            return None

        # If annealing values are larger than the default. 
        # Can happen for long time series and high dimensionality
        min_obj_val = np.min([a for a in objmip_ann if a is not None])
        i_best = objmip_ann.index(min_obj_val)

        # Final results based on best
        # parents_f = parents_ann[i_best]
        results_f = causal_prediction[i_best]
        # links_f = links_ann[i_best]
        gamma_f = gamma_ann[i_best]
        # Convergence optimization
        diff_g_f = diff_g_ann, diff_g_ann[i_best]

        final_results = {'regimes': gamma_f,
                         'causal_results':results_f,
                         'diff_g_f':diff_g_f,
                         'error_free_annealings':error_free_annealings}
        
        return final_results