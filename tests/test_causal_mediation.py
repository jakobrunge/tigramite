"""Tigramite causal discovery for time series."""

# Authors: Martin Rabel, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

import pytest
import numpy as np
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
import tigramite.toymodels.non_additive as toy_setup
import tigramite.causal_mediation as mediation
from tigramite.causal_mediation import CausalMediation



"""-------------------------------------------------------------------------------------------
-------------------------------   Test Mixed Data Fit Helpers   ------------------------------
-------------------------------------------------------------------------------------------"""


class ConstantMap:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(np.shape(x)[0], self.value)


class MapAddFirstDim:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(np.shape(x)[0], self.value) + x[:, 0]


def test_mixed_map_0_dim():
    X = toy_setup.CategoricalVariable(categories=2)
    f = mediation.MixedMap([ConstantMap(1), ConstantMap(2)], [X], dtype=np.dtype("float32"))
    r = f({X: np.array([0, 0, 0, 1, 1, 0, 1])})
    assert np.all(r == np.array([1, 1, 1, 2, 2, 1, 2]))


def test_mixed_map_1_dim():
    X = toy_setup.CategoricalVariable(categories=2)
    Z = toy_setup.ContinuousVariable()
    f = mediation.MixedMap([MapAddFirstDim(1), MapAddFirstDim(2)], [X, Z], dtype=np.dtype("float32"))
    r = f({X: np.array([0, 0, 0, 1, 1, 0, 1]), Z: np.arange(7)})
    assert np.all(r == np.array([1, 1, 1, 2, 2, 1, 2] + np.arange(7)))


def test_mixed_fit_pure_cat():
    fit = mediation.FitSetup()
    X = toy_setup.CategoricalVariable(categories=5)
    Y = toy_setup.ContinuousVariable()
    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: rng.binomial(X.categories - 1, 0.7, N),
        Y: lambda rng, N: rng.standard_normal(N)
    }, N=1000, seed=55)
    model = toy_setup.Model(sem={
        X: lambda value: value[X.Noise()],
        Y: lambda value: value[Y.Noise()] + 1.75 * value[X] + value[X] * value[X]
    })

    world_train = toy_setup.World(env, model)
    f = fit.Fit_ContinuousTarget_1D({X: world_train.data[X]}, world_train.data[Y])

    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    err = f({X: world_test.data[X]}) - world_test.data[Y]
    err_m = np.mean(err)
    err_std = np.std(err)
    assert err_m < 0.02, "Mean of prediction-error should be 0, found " + str(err_m)
    assert 0.9 < err_std < 1.1, "Std of prediction-error should be 1 (std of noise on Y), found " + str(err_std)


def test_mixed_fit_true_mixed():
    fit = mediation.FitSetup()
    X1 = toy_setup.CategoricalVariable(categories=5)
    X2 = toy_setup.ContinuousVariable()
    Y = toy_setup.ContinuousVariable()
    env = toy_setup.Environment(exogenous_noise={
        X1: lambda rng, N: rng.binomial(X1.categories - 1, 0.7, N),
        X2: lambda rng, N: rng.standard_normal(N),
        Y: lambda rng, N: rng.standard_normal(N)
    }, N=2000, seed=55)
    model = toy_setup.Model(sem={
        X1: lambda value: value[X1.Noise()],
        X2: lambda value: value[X2.Noise()],
        Y: lambda value: value[Y.Noise()] + 1.75 * value[X1] + value[X1] * value[X1] + np.sin(value[X2])
    })

    world_train = toy_setup.World(env, model)
    f = fit.Fit_ContinuousTarget_1D({X1: world_train.data[X1], X2: world_train.data[X2]}, world_train.data[Y])

    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    err = f({X1: world_test.data[X1], X2: world_test.data[X2]}, ) - world_test.data[Y]
    err_m = np.mean(err)
    err_std = np.std(err)
    assert err_m < 0.05, "Mean of prediction-error should be 0, found " + str(err_m)
    assert 0.9 < err_std < 1.1, "Std of prediction-error should be 1 (std of noise on Y), found " + str(err_std)


def test_mixed_fit_pure_continuous():
    fit = mediation.FitSetup()
    X1 = toy_setup.ContinuousVariable()
    X2 = toy_setup.ContinuousVariable()
    Y = toy_setup.ContinuousVariable()
    env = toy_setup.Environment(exogenous_noise={
        X1: lambda rng, N: rng.standard_normal(N),
        X2: lambda rng, N: rng.standard_normal(N),
        Y: lambda rng, N: rng.standard_normal(N)
    }, N=2000, seed=55)
    model = toy_setup.Model(sem={
        X1: lambda value: value[X1.Noise()],
        X2: lambda value: value[X2.Noise()],
        Y: lambda value: value[Y.Noise()] + 1.75 * value[X1] + value[X1] * value[X1] + np.sin(value[X2])
    })

    world_train = toy_setup.World(env, model)
    f = fit.Fit_ContinuousTarget_1D({X1: world_train.data[X1], X2: world_train.data[X2]}, world_train.data[Y])

    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    err = f({X1: world_test.data[X1], X2: world_test.data[X2]}, ) - world_test.data[Y]
    err_m = np.mean(err)
    err_std = np.std(err)
    assert err_m < 0.05, "Mean of prediction-error should be 0, found " + str(err_m)
    assert 0.9 < err_std < 1.15, "Std of prediction-error should be 1 (std of noise on Y), found " + str(err_std)


def test_mixed_density_pure_cat():
    fit = mediation.FitSetup()
    X = toy_setup.CategoricalVariable(categories=5)
    Y = toy_setup.CategoricalVariable(categories=3)
    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: rng.binomial(X.categories - 1, 0.4, N),
        Y: lambda rng, N: rng.binomial(Y.categories - 1, 0.7, N),
    }, N=1000, seed=55)
    model = toy_setup.Model(sem={
        X: lambda value: value[X.Noise()],
        Y: lambda value: np.clip(value[Y.Noise()] + 1.75 * value[X] + value[X] * value[X],
                                 0, Y.categories-1).astype(dtype="int32")
    })

    world_train = toy_setup.World(env, model)
    f = fit.Fit_CategoricalTarget_MarkovKernel({X: world_train.data[X]}, world_train.data[Y], Y.categories)

    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    predictions = np.mean(f({X: world_test.data[X]}), axis=0)
    print("Predictions: " + str(predictions))
    actual = np.zeros(Y.categories)
    for cY in range(Y.categories):
        actual[cY] = np.count_nonzero(world_test.data[Y] == cY) / env.N
    print("Actual: " + str(actual))
    err_abs_max = np.max(np.abs(predictions - actual))
    assert err_abs_max < 0.01, "Prediction did not agree with ground-truth " + str(np.abs(predictions - actual))


def test_mixed_density_true_mixed():
    fit = mediation.FitSetup()
    X1 = toy_setup.CategoricalVariable(categories=2)
    X2 = toy_setup.ContinuousVariable()
    Y = toy_setup.CategoricalVariable(categories=3)
    env = toy_setup.Environment(exogenous_noise={
        X1: lambda rng, N: rng.binomial(X1.categories - 1, 0.4, N),
        X2: lambda rng, N: rng.standard_normal(N),
        Y: lambda rng, N: rng.binomial(Y.categories - 1, 0.7, N),
    }, N=1000, seed=55)
    model = toy_setup.Model(sem={
        X1: lambda value: value[X1.Noise()],
        X2: lambda value: value[X2.Noise()],
        Y: lambda value: np.clip(value[Y.Noise()] + 1.75 * value[X1] + value[X2] * value[X2],
                                 0, Y.categories-1).astype(dtype="int32")
    })

    world_train = toy_setup.World(env, model)
    f = fit.Fit_CategoricalTarget_MarkovKernel({X1: world_train.data[X1],
                                                X2: world_train.data[X2]}, world_train.data[Y], Y.categories)

    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    predictions = np.mean(f({X1: world_test.data[X1], X2: world_train.data[X2]}), axis=0)
    print("Predictions: " + str(predictions))
    actual = np.zeros(Y.categories)
    for cY in range(Y.categories):
        actual[cY] = np.count_nonzero(world_test.data[Y] == cY) / env.N
    print("Actual: " + str(actual))
    err_abs_max = np.max(np.abs(predictions - actual))
    assert err_abs_max < 0.01, "Prediction did not agree with ground-truth " + str(np.abs(predictions - actual))


def test_mixed_density_true_mixed_xaida_ga_example():
    X = toy_setup.CategoricalVariable("Clouded", categories=2)
    M1 = toy_setup.ContinuousVariable("Temperature")
    M2 = toy_setup.CategoricalVariable("Drought", categories=2)
    Y = toy_setup.CategoricalVariable("Wild-Fire", categories=2)

    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: 1.0 * rng.standard_normal(N),
        M1: lambda rng, N: 1.0 * rng.standard_normal(N),
        M2: lambda rng, N: 1.0 * rng.standard_normal(N),
        Y: lambda rng, N: 1.0 * rng.standard_normal(N)
    }, N=5000, seed=11)
    autocorr = 0.3
    model = toy_setup.Model(sem={
        X: lambda value: autocorr * value[X.Lag(1)] + value[X.Noise()] > 1.0,
        M1: lambda value: autocorr * value[M1.Lag(1)] + (2.0 * value[X.Lag(1)] - 1.0) + value[M1.Noise()],
        M2: lambda value: autocorr * value[M2.Lag(1)] + value[X.Lag(1)] + value[M1.Lag(1)] + value[M2.Noise()] > 1.5,
        Y: lambda value: np.logical_or(autocorr * value[Y.Lag(1)] + value[M1.Lag(1)] + value[Y.Noise()] > 2.5,
                                       np.logical_and(value[M2.Lag(1)],
                                                      autocorr * value[Y.Lag(1)] + 5.0 * value[M1.Lag(1)] + value[
                                                          Y.Noise()] > 2.5))
    })


    fit = mediation.FitSetup(mediation.FitProvider_Continuous_Default.UseSklearn(20))

    world_train = toy_setup.World(env, model)

    M1_past = toy_setup.ContinuousVariable("T_past")
    M2_past = toy_setup.CategoricalVariable("D_past", categories=2)
    Y_past = toy_setup.CategoricalVariable("F_past", categories=2)
    obs = world_train.Observables()
    f = fit.Fit_CategoricalTarget_MarkovKernel({M1_past: obs[M1][0:-1],
                                                 M2_past:   obs[M2][0:-1],
                                                 Y_past:   obs[Y][0:-1]},
                                               obs[Y][1:], Y.categories)


    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    obs = world_test.Observables()
    predictions = np.mean(f({M1_past: obs[M1][0:-1], M2_past: obs[M2][0:-1], Y_past: obs[Y][0:-1]}), axis=0)

    print("Predictions: " + str(predictions))

    actual = np.zeros(Y.categories)
    for cY in range(Y.categories):
        actual[cY] = np.count_nonzero(obs[Y] == cY) / env.N
    print("Actual: " + str(actual))
    err_abs_max = np.max(np.abs(predictions - actual))
    assert err_abs_max < 0.01, "Prediction did not agree with ground-truth " + str(np.abs(predictions - actual))


def test_mixed_density_pure_cont():
    fit = mediation.FitSetup()
    X1 = toy_setup.ContinuousVariable()
    X2 = toy_setup.ContinuousVariable()
    Y = toy_setup.CategoricalVariable(categories=3)
    env = toy_setup.Environment(exogenous_noise={
        X1: lambda rng, N: rng.standard_normal(N),
        X2: lambda rng, N: rng.standard_normal(N),
        Y: lambda rng, N: rng.binomial(Y.categories - 1, 0.7, N),
    }, N=1000, seed=55)
    model = toy_setup.Model(sem={
        X1: lambda value: value[X1.Noise()],
        X2: lambda value: value[X2.Noise()],
        Y: lambda value: np.clip(value[Y.Noise()] + 1.75 * value[X1] + value[X2] * value[X2],
                                 0, Y.categories-1).astype(dtype="int32")
    })

    world_train = toy_setup.World(env, model)
    f = fit.Fit_CategoricalTarget_MarkovKernel({X1: world_train.data[X1],
                                                X2: world_train.data[X2]}, world_train.data[Y], Y.categories)

    env.ResetWithNewSeed(77)
    world_test = toy_setup.World(env, model)
    predictions = np.mean(f({X1: world_test.data[X1], X2: world_train.data[X2]}), axis=0)
    print("Predictions: " + str(predictions))
    actual = np.zeros(Y.categories)
    for cY in range(Y.categories):
        actual[cY] = np.count_nonzero(world_test.data[Y] == cY) / env.N
    print("Actual: " + str(actual))
    err_abs_max = np.max(np.abs(predictions - actual))
    assert err_abs_max < 0.01, "Prediction did not agree with ground-truth " + str(np.abs(predictions - actual))




"""-------------------------------------------------------------------------------------------
-----------------------------   Test Natural Effect Estimators   -----------------------------
-------------------------------------------------------------------------------------------"""



def test_nde():
    A = toy_setup.ContinuousVariable(name="Treatment")
    M = toy_setup.CategoricalVariable(name="Mediator", categories=5)
    Y = toy_setup.ContinuousVariable(name="Target")
    env = toy_setup.Environment(exogenous_noise={
        A: lambda rng, N: rng.standard_normal(N),
        M: lambda rng, N: rng.binomial(M.categories - 1, 0.7, N),
        Y: lambda rng, N: rng.standard_normal(N)
    }, N=10000, seed=23)
    model = toy_setup.Model(sem={
        A: lambda value: value[A.Noise()],
        M: lambda value: np.clip(np.round(value[M.Noise()] + value[A]).astype(dtype=np.dtype("int32")), 0,
                                 M.categories - 1),
        Y: lambda value: value[Y.Noise()] + 1.5 * value[A] + value[M] * value[A] + 0.7 * value[M]
    })

    nde_ground_truth = toy_setup.GroundTruth_NDE(0.0, 1.0, A, Y, M, env, model)
    print("Ground truth for NDE = " + str(nde_ground_truth))

    world = toy_setup.World(env, model)
    nde_estimator = mediation.NaturalEffects_StandardMediationSetup(mediation.FitSetup(), A, Y, M, world.Observables())
    nde_est = nde_estimator.NDE(0.0, 1.0)
    print("Estimation from data for NDE = " + str(nde_est))

    assert np.abs(nde_ground_truth - nde_est) < 0.2, "Estimate of NDE and ground-truth should agree"

    grid = np.arange(-2.0, 2.4, 0.4)
    print("Estimating NDE as function of x on grid: " + str(grid))
    nde_fct_ground_truth = toy_setup.GroundTruth_NDE_fct(A, Y, M, env, model, grid)
    print("Ground truth for NDE(x) = " + str(nde_fct_ground_truth))
    nde_fct_est = nde_estimator.NDE_grid(grid)
    print("Estimation from data for NDE(x) = " + str(nde_fct_est))

    assert np.all(np.abs(nde_fct_ground_truth - nde_fct_est) < 1.3), \
        ("Estimate of NDE as function of x and ground-truth should agree, difference was "
         + str(np.abs(nde_fct_ground_truth - nde_fct_est)))

    nie_ground_truth = toy_setup.GroundTruth_NIE(0.0, 1.0, A, Y, M, env, model)
    print("Ground truth for NIE = " + str(nie_ground_truth))
    nie_est = nde_estimator.NIE(0.0, 1.0)
    print("Estimation from data for NIE = " + str(nie_est))

    assert np.abs(nie_ground_truth - nie_est) < 0.2, "Estimate of NIE and ground-truth should agree"


def test_nde_all_categorical():
    A = toy_setup.CategoricalVariable(name="Treatment", categories=3)
    M = toy_setup.CategoricalVariable(name="Mediator", categories=4)
    Y = toy_setup.CategoricalVariable(name="Target", categories=2)
    env = toy_setup.Environment(exogenous_noise={
        A: lambda rng, N: rng.binomial(A.categories - 1, 0.2, N),
        M: lambda rng, N: rng.binomial(M.categories - 1, 0.5, N),
        Y: lambda rng, N: rng.standard_normal(N)
    }, N=10000, seed=23)
    model = toy_setup.Model(sem={
        A: lambda value: value[A.Noise()],
        M: lambda value: value[M.Noise()],
        # np.clip(np.round(value[M.Noise()] * value[A]), 0, M.categories-1).astype(dtype="int32"),
        Y: lambda value: (value[A] + 2.0 * value[M] + 1.0 * value[Y.Noise()]) > 3.0
    })

    nde_ground_truth = toy_setup.GroundTruth_NDE(1, 2, A, Y, M, env, model)
    print("Ground truth for NDE = " + str(nde_ground_truth))
    nde_change_ground_truth = nde_ground_truth.T[0] - nde_ground_truth.T[1]
    print("Ground truth for NDE change = " + str(nde_change_ground_truth))

    world = toy_setup.World(env, model)
    nde_estimator = mediation.NaturalEffects_StandardMediationSetup(mediation.FitSetup(), A, Y, M, world.Observables())
    nde_est = nde_estimator.NDE(1, 2)
    print("Estimation from data for NDE = " + str(nde_est))

    assert np.all(np.abs(nde_ground_truth - nde_est) < 0.01), \
        "(Indirect) estimate of NDE and ground-truth should agree"

    # Technically there is no reason to treat A as ordinal ...
    grid = np.arange(A.categories - 1)
    nde_ground_truth_fct = toy_setup.GroundTruth_NDE_fct(A, Y, M, env, model, grid, cf_delta=1)
    print("Ground Truth of NDE as function of (ordinal) A is: " + str(nde_ground_truth_fct))

    nde_est_fct = nde_estimator.NDE_grid(grid, cf_delta=1)
    print("Estimate of NDE as function of (ordinal) A is: " + str(nde_est_fct))

    assert np.all(np.abs(nde_ground_truth_fct - nde_est_fct) < 0.01), \
        "Estimate of NDE and ground-truth as function should agree"

    nie_ground_truth_fct = toy_setup.GroundTruth_NIE_fct(A, Y, M, env, model, grid, cf_delta=1)
    print("Ground Truth of NIE as function of (ordinal) A is: " + str(nie_ground_truth_fct))
    nie_est = nde_estimator.NIE_grid(grid, cf_delta=1)
    print("Estimate of NIE as function of (ordinal) A is: " + str(nie_est))

    print("Maximum discrepancy was " + str(np.max(np.abs(nie_ground_truth_fct - nie_est))))

    assert np.all(np.abs(nie_ground_truth_fct - nie_est) < 0.012), \
        "Estimate of NIE and ground-truth as function should agree"


def test_NDE_example2():
    X = toy_setup.ContinuousVariable()
    Y = toy_setup.CategoricalVariable(categories=3)
    M = toy_setup.ContinuousVariable()

    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: 2.0 * rng.standard_normal(N),
        M: lambda rng, N: rng.standard_normal(N),
        Y: lambda rng, N: rng.standard_normal(N)
    }, N=1000, seed=12345)
    model = toy_setup.Model(sem={
        X: lambda value: value[X.Noise()],
        M: lambda value: 0.5 * value[X] + value[M.Noise()],
        Y: lambda value: np.round(np.clip(
            0.2 * value[Y.Noise()] + value[X] + value[M],
            a_min=-0.49, a_max=2.49))
    })

    world = toy_setup.World(env, model)

    fit_setup = mediation.FitSetup()
    estimator = mediation.NaturalEffects_StandardMediationSetup(fit_setup, source=X, target=Y, mediator=M,
                                                          data=world.Observables())

    grid = np.arange(-3.0, 3.6, 0.6)
    nde_fct_est = estimator.NDE_grid(grid, cf_delta=0.6)
    nde_fct_ground_truth = toy_setup.GroundTruth_NDE_fct(X, Y, M, env, model, grid, cf_delta=0.6)

    print("Estimate: " + str(nde_fct_est))
    print("Ground Truth: " + str(nde_fct_ground_truth))
    print("Max. Difference: " + str(np.max(np.abs(nde_fct_est - nde_fct_ground_truth))))
    # Use a liberal limit, as these are a lot (12 * 2 * 3 = 72) values
    assert np.all(np.abs(nde_fct_est - nde_fct_ground_truth) < 0.25), \
        "NDE estimation and ground-truth should agree"

    nie_ground_truth_fct = toy_setup.GroundTruth_NIE_fct(X, Y, M, env, model, grid, cf_delta=0.6)
    print("Ground Truth of NIE as function of (ordinal) A is: " + str(nie_ground_truth_fct))
    nie_est = estimator.NIE_grid(grid, cf_delta=0.6)
    print("Estimate of NIE as function of (ordinal) A is: " + str(nie_est))

    print("Maximum discrepancy was " + str(np.max(np.abs(nie_ground_truth_fct - nie_est))))
    # Use a liberal limit, as these are a lot (12 * 2 * 3 = 72) values
    assert np.all(np.abs(nie_ground_truth_fct - nie_est) < 0.2), \
        "Estimate of NIE and ground-truth as function should agree"


def test_graph_nde():
    X = toy_setup.ContinuousVariable("X")
    M = toy_setup.CategoricalVariable("M", categories=2)
    coeff = 0.8

    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: 1.0 * rng.standard_normal(N),
        M: lambda rng, N: 1.0 * rng.standard_normal(N),
    }, N=5000, seed=11)
    model = toy_setup.Model(sem={
        X: lambda value: value[X.Noise()] + 0.5 * value[M.Lag(1)] + coeff * value[X.Lag(1)],
        M: lambda value: value[X] + value[M.Noise()] + 0.7 * value[M.Lag(1)] > 1.0  # M is binary, ie true or false
    })

    # The Groundtruth Graph
    graph, graph_type = model.GetGroundtruthGraph()

    # Generate data
    world = toy_setup.World(env, model)
    obs = world.Observables()

    # Fit & Estimator Setup
    fit_setup = mediation.FitSetup(mediation.FitProvider_Continuous_Default.UseSklearn(20))
    tau_max = 2

    print("\nAutomatically setting up an estimator")
    estimator = mediation.NaturalEffects_GraphMediation(graph, graph_type, tau_max, fit_setup, world.Observables(),
                                                  effect_source=X.Lag(1), effect_target=X,
                                                  blocked_mediators="all", adjustment_set="auto")
    estimator.PrintInfo()

    print("\nManually setting up an estimator")
    estimator2 = mediation.NaturalEffects_GraphMediation(graph, graph_type, tau_max, fit_setup, world.Observables(),
                                                   effect_source=X.Lag(1), effect_target=X,
                                                   blocked_mediators=[M.Lag(1)], adjustment_set=[M.Lag(2)])
    estimator2.PrintInfo()
    print("")

    # Compute Single-Value Estimates
    nde01 = estimator.NDE(0.0, 1.0)
    nde01_ = estimator2.NDE(0.0, 1.0)
    nde01_gt = toy_setup.GroundTruth_NDE(0.0, 1.0, X.Lag(1), X, [M.Lag(1)], env, model)
    print(f"Estimated NDE at x=0 is {nde01:.3f} (auto) = {nde01_:.3f} (manual), analytical NDE is {coeff:.3f}.")
    print(f"[Numerical Ground-Truth is {nde01_gt:.3f}]")

    # Both estimators should be identical
    assert np.all(nde01 == nde01_)

    print(f"Estimated NDE is {nde01}, analytical NDE is {coeff}, numerical NDE is {nde01_gt}.")
    assert abs(nde01 - coeff) < 0.15  # require est to be somewhat accurate
    assert abs(nde01_gt - coeff) < 0.001  # require numerical est to be much more accurate


def test_graph_nde_fully_binary():
    X = toy_setup.CategoricalVariable("X", categories=2)
    M = toy_setup.CategoricalVariable("M", categories=2)

    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: 1.0 * rng.standard_normal(N),
        M: lambda rng, N: 1.0 * rng.standard_normal(N),
    }, N=5000, seed=11)

    model = toy_setup.Model(sem={
        X: lambda value: value[X.Noise()] + 0.5 * value[M.Lag(1)] + 0.8 * value[X.Lag(1)] > 1.0,
        M: lambda value: value[X] + value[M.Noise()] + 0.7 * value[M.Lag(1)] > 1.0
    })

    graph, graph_type = model.GetGroundtruthGraph()

    world = toy_setup.World(env, model)
    obs = world.Observables()

    fit_setup = mediation.FitSetup()

    tau_max = 2
    estimator = mediation.NaturalEffects_GraphMediation(graph, graph_type, tau_max, fit_setup, obs,
                                                  effect_source=X, effect_target=M,
                                                  blocked_mediators="all", adjustment_set="auto",
                                                  fall_back_to_total_effect=True)
    nde01 = estimator.NDE(False, True)

    nde01_gt = toy_setup.GroundTruth_NDE_auto(False, True, estimator, env, model)

    print(f"Estimated (contemporaneous) NDE is {nde01}, numerical ground-truth is {nde01_gt}.")
    #  Require est to be close to ground-truth
    assert np.all(np.abs(nde01 - nde01_gt) < 0.03)


def test_graph_example_breakout_session():
    X = toy_setup.CategoricalVariable("Clouded", categories=2)
    M1 = toy_setup.ContinuousVariable("Temperature")
    M2 = toy_setup.CategoricalVariable("Drought", categories=2)
    Y = toy_setup.CategoricalVariable("Wild-Fire", categories=2)

    env = toy_setup.Environment(exogenous_noise={
        X: lambda rng, N: 1.0 * rng.standard_normal(N),
        M1: lambda rng, N: 1.0 * rng.standard_normal(N),
        M2: lambda rng, N: 1.0 * rng.standard_normal(N),
        Y: lambda rng, N: 1.0 * rng.standard_normal(N)
    }, N=5000, seed=11)
    model = toy_setup.Model(sem={
        X: lambda value: value[X.Noise()] > 1.0,
        M1: lambda value: (value[X] + 1.0) * value[M1.Noise()],
        M2: lambda value: value[X] + value[M1] + value[M2.Noise()] > 1.0,
        Y: lambda value: np.logical_or(value[M1] + value[Y.Noise()] > 3.0,
                                       np.logical_and(value[M2], value[M1] + value[Y.Noise()] > 1.5))
    })

    graph, graph_type = model.GetGroundtruthGraph()
    graph = graph.reshape(graph.shape[0:2])

    world = toy_setup.World(env, model)

    fit_setup = mediation.FitSetup(mediation.FitProvider_Continuous_Default.UseSklearn(20))
    tau_max = 2
    estimator = mediation.NaturalEffects_GraphMediation(graph, graph_type, tau_max, fit_setup, world.Observables(),
                                                  effect_source=X, effect_target=Y,
                                                  blocked_mediators="all", adjustment_set="auto")

    nde01 = estimator.NDE(False, True)
    nde01_gt = toy_setup.GroundTruth_NDE_auto(False, True, estimator, env, model)

    print(f"Estimated NDE at x=0 is {nde01}, ground-truth is {nde01_gt}.")
    assert np.all(np.abs(nde01 - nde01_gt) < 0.005)




"""-------------------------------------------------------------------------------------------
-------------------------------   Test Non-Additive Toymodels   ------------------------------
-------------------------------------------------------------------------------------------"""




@pytest.fixture(scope="module")
def med_vars():
    class StandardMediationSetup:
        A = toy_setup.ContinuousVariable(name="Treatment")
        M = toy_setup.CategoricalVariable(name="Mediator", categories=5)
        Y = toy_setup.ContinuousVariable(name="Target")

    return StandardMediationSetup()


@pytest.fixture(scope="module")
def env(med_vars):
    return toy_setup.Environment(exogenous_noise={
        med_vars.A: lambda rng, N: rng.standard_normal(N),
        med_vars.M: lambda rng, N: rng.binomial(med_vars.M.categories - 1, 0.7, N),
        med_vars.Y: lambda rng, N: rng.standard_normal(N)
    }, seed=1)


def test_environments(env, med_vars):
    assert len(env.noise) == 3
    assert np.shape(env.GetNoise()[med_vars.A.Noise()]) == (env.N,)
    with pytest.raises(Exception) as e:
        env.Reset()
    assert str(e.value) == ("Cannot reset a fixed-seed environment, use seed=None or factory/lambda instead for"
                            "ensemble-creation.")


def test_model_validation(med_vars):
    A = med_vars.A
    M = med_vars.M
    Y = med_vars.Y
    with pytest.raises(Exception):
        toy_setup.Model(sem={
            A: lambda value: value[A.Noise()],
            Y: lambda value: value[Y.Noise()] + value[A] + value[M],
            M: lambda value: value[M.Noise()] + value[A]
        })


@pytest.fixture(scope="module")
def model(med_vars):
    A = med_vars.A
    M = med_vars.M
    Y = med_vars.Y
    return toy_setup.Model(sem={
        A: lambda value: value[A.Noise()],
        M: lambda value: value[M.Noise()] + value[A],
        Y: lambda value: value[Y.Noise()] + value[A] + value[M]
    })


@pytest.fixture(scope="module")
def model0(model, med_vars):
    return model.Intervene({med_vars.A: 0.0})


@pytest.fixture(scope="module")
def model1(model, med_vars):
    return model.Intervene({med_vars.A: 1.0})


@pytest.fixture(scope="module")
def world(env, model):
    return toy_setup.World(env, model)


@pytest.fixture(scope="module")
def world0(env, model0):
    return toy_setup.World(env, model0)


@pytest.fixture(scope="module")
def world1(env, model1):
    return toy_setup.World(env, model1)


def test_interventions(med_vars, env, world, world0, world1):
    # in the "real" world A is noise, thus not always 0
    assert np.any(world.Observables()[med_vars.A] != 0.0)
    # After intervention A=0, a should always be 0
    assert np.all(world0.Observables()[med_vars.A] == 0.0)
    # After intervention A=0, a should always be 1
    assert np.all(world1.Observables()[med_vars.A] == 1.0)


@pytest.fixture(scope="module")
def cfworld(med_vars, env, model, world, world0, world1):
    cf_world = toy_setup.CounterfactualWorld(env, model)
    cf_world.TakeVariablesFromWorld(world0, [med_vars.M])
    cf_world.TakeVariablesFromWorld(world1, [med_vars.A])
    cf_world.Compute()
    return cf_world


def test_cfworld(med_vars, cfworld, world, world0, world1):
    assert np.all(cfworld.Observables()[med_vars.A] == world1.Observables()[med_vars.A] )
    assert np.all(cfworld.Observables()[med_vars.M] == world0.Observables()[med_vars.M] )
    assert np.all(cfworld.data[med_vars.A.Noise()] == world.data[med_vars.A.Noise()] )
    assert np.all(cfworld.data[med_vars.M.Noise()] == world.data[med_vars.M.Noise()] )
    assert np.all(cfworld.data[med_vars.Y.Noise()] == world.data[med_vars.Y.Noise()] )




"""-------------------------------------------------------------------------------------------
--------------------------------   Test Tigramite Interface   --------------------------------
-------------------------------------------------------------------------------------------"""


def test_tutorial_example0():
    graph =  np.array([[['', '-->', ''],
                    ['', '', ''],
                    ['', '', '']],
                   [['', '-->', ''],
                    ['', '-->', ''],
                    ['-->', '', '-->']],
                   [['', '', ''],
                    ['<--', '', ''],
                    ['', '-->', '']]], dtype='<U3')

    X = [(1,-2)]
    Y = [(2,0)]
    causal_effects = CausalMediation(graph, graph_type='stationary_dag', X=X, Y=Y,
                                    S=None, # (currently S must be None)
                                    hidden_variables=None, # (currently hidden must be None)
                                    verbosity=1)
    var_names = ['$X^0$', '$X^1$', '$X^2$']

    opt = causal_effects.get_optimal_set()

    from tigramite import data_processing as pp
    from tigramite.toymodels import structural_causal_processes as toys

    coeff = .5
    direct_eff = 0.5
    def lin_f(x): return x
    links_coeffs = {
                    0: [((0, -1), coeff, lin_f), ((1, -1), coeff, lin_f)], 
                    1: [((1, -1), coeff, lin_f),], 
                    2: [((2, -1), coeff, lin_f), ((1, 0), coeff, lin_f), ((1,-2), direct_eff, lin_f)],
                    }
    # Observational data
    T = 1000
    data, nonstat = toys.structural_causal_process(links_coeffs, T=T, noises=None, seed=42)
    normalization = []
    data_normalized = np.empty_like(data)
    for v in range(0,3):
        m = np.std(data[:,v])
        normalization.append(m)
        data_normalized[:,v] = data[:,v] / m
    dataframe = pp.DataFrame(data, var_names=var_names)
    dataframe_normalized = pp.DataFrame(data_normalized, var_names=var_names)
    fit_setup = mediation.FitSetup(mediation.FitProvider_Continuous_Default.UseSklearn(20))

    # unnormalized data
    causal_effects.fit_natural_direct_effect(dataframe, blocked_mediators='all',
                                    mixed_data_estimator=fit_setup).PrintInfo()

    nde_est = causal_effects.predict_natural_direct_effect(0.0, 1.0)

    # normalized data
    causal_effects.fit_natural_direct_effect(dataframe_normalized, blocked_mediators='all',
                                    mixed_data_estimator=fit_setup)

    nde_est_from_normalized = causal_effects.predict_natural_direct_effect(0.0, 1.0) * normalization[2] / normalization[1]

    # print results
    print( f"Estimate of the NDE is:\n{nde_est} from unnormalized data, "
        + f"\n{nde_est_from_normalized} from normalized data,\nground-truth is {direct_eff}." )
    
    assert 0.3 < nde_est_from_normalized < 0.6


def test_tutorial_example():    
    graph = np.array([[['', '-->', ''],
                       ['', '', ''],
                       ['', '', '']],
                      [['', '-->', ''],
                       ['', '-->', ''],
                       ['-->', '', '-->']],
                      [['', '', ''],
                       ['<--', '', ''],
                       ['', '-->', '']]], dtype='<U3')

    X = [(1, -2)]
    Y = [(2, 0)]

    causal_effects = CausalMediation(graph, graph_type='stationary_dag', X=X, Y=Y,
                                     S=None,  # (currently S must be None)
                                     hidden_variables=None,  # (currently hidden must be None)
                                     verbosity=1)
    var_names = ['$X^0$', '$X^1$', '$X^2$']


    coeff = .5
    direct_eff = 500.0

    def lin_f(x): return x

    links_coeffs = {
        0: [((0, -1), coeff, lin_f), ((1, -1), coeff, lin_f)],
        1: [((1, -1), coeff, lin_f), ],
        2: [((2, -1), coeff, lin_f), ((1, 0), coeff, lin_f), ((1, -2), direct_eff, lin_f)],
    }
    # Observational data
    T = 10000
    data, nonstat = toys.structural_causal_process(links_coeffs, T=T, noises=None, seed=42)

    normalization = []
    data_normalized = np.empty_like(data)
    for v in range(0, 3):
        m = np.std(data[:, v])
        normalization.append(m)
        data_normalized[:, v] = data[:, v] / m

    dataframe = pp.DataFrame(data_normalized, var_names=var_names)

    fit_setup = mediation.FitSetup(mediation.FitProvider_Continuous_Default.UseSklearn(20))
    causal_effects.fit_natural_direct_effect(dataframe, blocked_mediators='all',
                                             mixed_data_estimator=fit_setup).PrintInfo()
    nde_est = (causal_effects.predict_natural_direct_effect(0.0, 1.0)
               * normalization[2] / normalization[1])
    print(f"Estimate of the NDE is {nde_est}, ground-truth is {direct_eff}.")

    assert 0.8 < nde_est / direct_eff < 1.1




class FitProvider_Continous_Filtered:
    def __init__(self, underlying_fit_provider, filter_to_use):
        self.filter = filter_to_use
        self.underlying_fit_provider = underlying_fit_provider
    def Get_Fit_Continuous(self,x_train,y_train):
        return self.underlying_fit_provider.Get_Fit_Continuous(*self.filter.apply(x_train,y_train))
    
class FitProvider_Density_Filtered:
    def __init__(self, underlying_fit_provider, filter_to_use):
        self.filter = filter_to_use
        self.underlying_fit_provider = underlying_fit_provider
    def Get_Fit_Density(self, x_train):
        return self.underlying_fit_provider.Get_Fit_Density(self.filter.apply(x_train))

class FilterMissingValues:
    def __init__(self, missing_value_flag):
        self.missing_value_flag = missing_value_flag
    def apply(self,x,y=None):
        missing_in_any_x = np.any( x==self.missing_value_flag, axis=1 )
        if y is None:
            valid = np.logical_not( missing_in_any_x )
            return x[valid]
        else:
            missing_in_y = ( y==self.missing_value_flag )
            valid = np.logical_not( np.logical_or(missing_in_any_x, missing_in_y) )
            return x[valid], y[valid]

def apply_filter_to_all_inputs(fit_setup, filter_to_apply):
    # Assume the fit_setup can be contructed from map & density fit and has corresponding members
    # (for all implementations based on the FitSetup class in the mediation-module
    #  of tigramite this is the case; see tutorial on mediation, appendix B)
    return fit_setup.__class__(
        fit_map=FitProvider_Continous_Filtered(fit_setup.fit_map, filter_to_apply),
        fit_density=FitProvider_Density_Filtered(fit_setup.fit_density, filter_to_apply),
    )


def test_tutorial_example_custom_fit():
    graph =  np.array([[['', '-->', ''],
                    ['', '', ''],
                    ['', '', '']],
                   [['', '-->', ''],
                    ['', '-->', ''],
                    ['-->', '', '-->']],
                   [['', '', ''],
                    ['<--', '', ''],
                    ['', '-->', '']]], dtype='<U3')

    X = [(1,-2)]
    Y = [(2,0)]
    var_names = ['$X^0$', '$X^1$', '$X^2$']

    from tigramite import data_processing as pp
    from tigramite.toymodels import structural_causal_processes as toys

    coeff = .5
    direct_eff = 0.5
    def lin_f(x): return x
    links_coeffs = {
                    0: [((0, -1), coeff, lin_f), ((1, -1), coeff, lin_f)], 
                    1: [((1, -1), coeff, lin_f),], 
                    2: [((2, -1), coeff, lin_f), ((1, 0), coeff, lin_f), ((1,-2), direct_eff, lin_f)],
                    }
    # Observational data
    T = 1000
    data, nonstat = toys.structural_causal_process(links_coeffs, T=T, noises=None, seed=None)
    normalization = []
    data_normalized = np.empty_like(data)
    for v in range(0,3):
        m = np.std(data[:,v])
        normalization.append(m)
        data_normalized[:,v] = data[:,v] / m
    dataframe = pp.DataFrame(data, var_names=var_names)
    dataframe_normalized = pp.DataFrame(data_normalized, var_names=var_names)


    seed = 12345
    gap_count = 10
    gap_min_len = 10
    gap_max_len = 20
    rng = np.random.default_rng(seed)
    var_idx = rng.integers(0, data_normalized.shape[1], gap_count)
    offset = rng.integers(0, data_normalized.shape[0]-gap_max_len, gap_count)
    missing_count = rng.integers(10, 20, gap_count)

    modified_data = data_normalized
    for gap in range(gap_count):
        modified_data[offset[gap]:offset[gap]+missing_count[gap], var_idx] = 999

    fit_setup = mediation.FitSetup(mediation.FitProvider_Continuous_Default.UseSklearn(20))
    fit_setup2 = apply_filter_to_all_inputs(fit_setup, FilterMissingValues(999))
    dataframe_unmarked_missing = pp.DataFrame(data_normalized, var_names=var_names) #missing_flag=999)

    causal_effects = CausalMediation(graph, graph_type='stationary_dag', X=X, Y=Y,
                                    S=None, # (currently S must be None)
                                    hidden_variables=None, # (currently hidden must be None)
                                    verbosity=1)
    # normalized data
    causal_effects.fit_natural_direct_effect(dataframe_unmarked_missing, blocked_mediators='all',
                                    mixed_data_estimator=fit_setup2, # set the new fit_setup
                                    enable_dataframe_based_preprocessing=False)

    nde_est = causal_effects.predict_natural_direct_effect(0.0, 1.0) * normalization[2] / normalization[1]

    # print results
    print( f"Estimate of the NDE is:\n{nde_est} with missing values,\nground-truth is {direct_eff}." )