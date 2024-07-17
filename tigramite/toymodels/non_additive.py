"""Tigramite causal discovery for time series."""

# Authors: Martin Rabel, Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0


import numpy as np
from tigramite.data_processing import DataFrame
from tigramite.causal_effects import CausalEffects


class VariableDescription:
    r"""Variable descrption (base-class)

    Used for mixed variable fitting and mediation, see Mediation-tutorial.

    Parameters
    ----------
    name : string
        The display-name of the variable
    observable : bool
        Is the variable observable or hidden? (used by toy-models)
    exogenous : bool
        Is the variable assumed exogenous (used on noise-terms internally, keep default)
    """

    def __init__(self, name="(unnamed)", observable=True, exogenous=False):
        """Use CategoricalVariable or ContinuousVariable instead."""
        self.noise_term = None
        self.observable = observable
        self.exogenous = exogenous
        self.name = name

    def Noise(self):
        """
        Get the variable-description for the noise-term on this variable.
        Used e.g. for describing SEMs in noise-models

        Returns
        -------
        noise : VariableDescription
            The variable-description for the noise-term on this variable.
        """
        if self.noise_term is None:
            self.noise_term = ExogenousNoiseVariable(self)
        return self.noise_term

    def Lag(self, offset):
        """
        Get the variable-description for the past of this variable.
        Used e.g. for describing SEMs in noise-models

        Parameters
        ----------
        offset : uint
            The lag to introduce (only non-negative values, always in the past).

        Returns
        -------
        lagged-self : VariableDescription
            This variable, at lag "offset"
        """
        if offset == 0:
            return self
        else:
            return LaggedVariable(self, offset)

    def Info(self, detail=0):
        """
        Get an Info-string for this variable.
        (Overridden by derived classes for discrete, continuous, lagged, ...)

        Parameters
        ----------
        detail : uint
            The amount of detail to include.

        Returns
        -------
        info : string
            A description of this variable.
        """
        return self.name

    def PrintInfo(self):
        """Print Info-string (see Info)"""
        print(self.Info())

    def Id(self):
        """Get the 'vanilla' (unlagged etc) description of this variable
        (Overridden by derived classes for discrete, continuous, lagged, ...)

        Returns
        -------
        id : VariableDescription
            A description of the underlying variable of this description.
        """
        return self

    def LagValue(self):
        """Get the time-lag (non-negative, positive values are in the past) of the
        variable described by this indicator.
        (Overridden by derived classes for discrete, continuous, lagged, ...)

        Returns
        -------
        lag : uint
            The time-lag (non-negative, positive values are in the past) of this variable.
        """
        return 0


class LaggedVariable(VariableDescription):
    """Describes a lagged variable (see VariableDescription)"""

    def __init__(self, var, lag):
        """Use var-description.Lag(offset) instead."""
        self.var = var
        self.lag = lag
        self.exogenous = False

    def Id(self):
        return self.var

    def LagValue(self):
        return self.lag

    def Info(self, detail=0):
        return self.var.Info(detail) + " at lag " + str(self.lag)

    def PrintInfo(self):
        print(self.Info())

    def DefaultValue(self):
        return self.var.DefaultValue()


class ExogenousNoiseVariable(VariableDescription):
    """Describes an exogenous noise variable (see VariableDescription)"""

    def __init__(self, attached_to):
        """Use var-description.Noise() instead."""
        super().__init__(name="Noise of " + attached_to.name, observable=False, exogenous=True)
        self.associated_system_variable = attached_to

    def DefaultValue(self):
        return self.associated_system_variable.DefaultValue()


class CategoricalVariable(VariableDescription):
    """Describes a categorical variable (see VariableDescription)

    Parameters
    ----------
    name : string
        The display-name of the variable.
    categories : uint or *iterable*
        The number of categories in which the variable may take values, or
        an iterable containing the possible values.
    dtype : string
        Name of the numpy dtype ('uint32', 'bool', ...) or 'auto'
    """

    def __init__(self, name="(unnamed)", categories=2, dtype="auto"):
        super().__init__(name=name)
        self.is_categorical = True
        if hasattr(categories, '__iter__'):
            self.categories = len(categories)
            self.category_values = categories
        else:
            self.categories = categories
            self.category_values = None
        if dtype == "auto":
            self.dtype = "uint32" if self.categories > 2 else "bool"
        else:
            self.dtype = dtype

    def Info(self, detail=0):
        if detail > 0:
            return self.name
        else:
            return self.name + f"( Categorical variable with {self.categories} categories )"

    def Empty(self, N):
        """Build an empty np.array with N samples."""
        return np.zeros(N, self.dtype)

    def ValidValue(self, x):
        """Validate a value"""
        is_integer = x % 1 == 0.0  # this works for all of: np.[u]int; built-in [u]int; np.bool; built-in boolean
        return is_integer and 0 <= x < self.categories

    def DefaultValue(self):
        """Return a 'default-value' to use in toy-models at the start of a time-series
        (if otherwise undefined by the SEM)"""
        # used for lagged values not available in time-series
        return 0  # could be user specified & per variable ...

    def CopyInfo(self, name="(unnamed)"):
        """Create a new variable-id with the same parameters"""
        return CategoricalVariable(name, self.categories, self.dtype)


class ContinuousVariable(VariableDescription):
    """Describes a continuous variable (see VariableDescription)

       Parameters
       ----------
       name : string
           The display-name of the variable.
       dtype : string
           Name of the numpy dtype ('float32', 'float64', ...)
       dimension : uint
           The number of dimensions, can be >1 for fitting.
       """

    def __init__(self, name="(unnamed)", dtype="float32", dimension=1):
        super().__init__(name=name)
        self.dtype = dtype
        self.is_categorical = False
        self.dimension = dimension  # fitting probabilities for multiple categories might be higher dim (handled by fit)

    def Info(self, detail=0):
        if detail > 0:
            return self.name
        else:
            return self.name + f"( Continuous variable of dimension {self.dimension} )"

    def Empty(self, N):
        """Build an empty np.array with N samples."""
        if self.dimension == 1:
            return np.zeros(N, dtype=self.dtype)
        else:
            return np.zeros([self.dimension, N], dtype=self.dtype)

    def ValidValue(self, x):
        return True

    def DefaultValue(self):
        """Return a 'default-value' to use in toy-models at the start of a time-series
        (if otherwise undefined by the SEM)"""
        # used for lagged values not available in time-series
        return 0.0  # could be user specified & per variable ...

    def CopyInfo(self, name="(unnamed)"):
        """Create a new variable-id with the same parameters"""
        return ContinuousVariable(name, self.dtype, self.dimension)


class Environment:
    """An (exogenous) environment to use with a model

    Counterfactual models may create different worlds with different models from the same environment.

    Parameters
    ----------
    exogenous_noise : dictionary< VariableDescription, *callable* (np.random.generator, sample-count) >
        For each variable, a callable generating noise-values.
    N : uint
        The initial number of samples to generate
    seed : uint or None
        Seed to use for the random-generator.
    """

    def __init__(self, exogenous_noise, N=1000, seed=None):
        self.rng = np.random.default_rng(seed)
        self.can_reset = seed is None
        self.exogenous_noise = exogenous_noise
        self.noise = {}
        self.N = N
        self._ForceReset()

    def ResetWithNewSeed(self, new_seed, N=None):
        """Reset with a new random-seed."""
        self.rng = np.random.default_rng(new_seed)
        self._ForceReset(N)

    def Reset(self, N=None):
        """Reset with a random new random-seed."""
        if not self.can_reset:
            raise Exception("Cannot reset a fixed-seed environment, use seed=None or factory/lambda instead for"
                            "ensemble-creation.")
        self._ForceReset(N)
        return self

    def _ForceReset(self, N=None):
        """[internal] Enfore a reset."""
        if N is not None:
            self.N = N
        for var, _noise in self.exogenous_noise.items():
            self.noise[var.Noise()] = _noise(self.rng, self.N)

    def GetNoise(self):
        """Get a copy of the noise-values.

        Returns
        -------
        noises : dictionary< Variable-Description, samples >
            Returns a shallow copy of the samples generated.
            Ids are 'exogenous_noise.Noise()' (see constructor).
        """
        return self.noise.copy()  # return a SHALLOW copy of the noise-data (copy the dict, not the data)


class DataPointForValidation:
    """[INTERNAL] Used to validate causal ordering on SEM for toy-model"""

    def __init__(self):
        self.known = {}
        self.is_timeseries = False
        self.max_lag = 0

    def Set(self, var, value):
        self.known[var] = True

    def __getitem__(self, key):
        if key.__class__ == LaggedVariable:
            self.is_timeseries = True
            self.max_lag = max(self.max_lag, key.lag)
            return key.DefaultValue()

        if not key.exogenous and key not in self.known:
            raise Exception("SEM must be causally ordered, such that contemporaneous parents are listed "
                            "before their children.")
        return key.DefaultValue()


class DataPointForParents:
    """[INTERNAL] Used to extract causal parents (for ground-truth graph) from SEM for toy-model"""

    def __init__(self):
        self.parents = []

    def __getitem__(self, key):
        if not key.exogenous:
            self.parents.append(key)
        return key.DefaultValue()


class DataPointView:
    """[INTERNAL] Used to run SEMs on toy-models."""

    def __init__(self, data, known=None):
        self.data = data
        self.index = 0

    def Next(self):
        self.index += 1

    def Set(self, var, value):
        self.data[var][self.index] = value

    def __getitem__(self, key):
        if key.__class__ == LaggedVariable:
            if self.index < key.lag:
                return key.DefaultValue()
            else:
                return self.data[key.var][self.index - key.lag]
        else:
            return self.data[key][self.index]


class Model:
    """Describes a Toy-model via an SEM.

    Validates only the causal ordering, for more validation, see tigramite.toymodels.
    However, this implementation can also generate non-additive models.

    Parameters
    ----------
    sem : dictionary< VariableDescription, *callable* ( sample-view : var-desc->np.array or scalar )
        For variables in causal order, a *callable* which is passed a
        view v of the data-samples, access data by v[var-description],
        and return the value(s) of the described (key) variable.
        For non-time-series, v[var-description] is an np.array,
        for time-series, v[var-description] is a scalar.
    """

    def __init__(self, sem):
        self.SEM = sem
        self.is_timeseries = False
        self.max_lag = 0
        self.Validate()

    def GetGroundtruthLinks(self):
        """Get the causal links to parents (ground-truth).

        Does not validate faithfullness.

        Returns
        -------
        links, indices : dictionaries< VariableDescription, ...>
            For links: Values are the lists of parents.
            For indices: Values are variable indices (see GetGroundtruthLinksRaw)
        """
        self.Validate()
        links = {}
        indices = {}
        idx = 0
        for var, eq in self.SEM.items():
            data_pt = DataPointForParents()
            self.SEM[var](data_pt)
            links[var] = data_pt.parents
            indices[var] = idx
            idx += 1
        return links, indices

    def GetGroundtruthLinksRaw(self):
        """Get unformatted causal links to parents (ground-truth).

        Does not validate faithfullness.

        Returns
        -------
        links : tigramite-format/raw indices + lags
        """
        links, indices = self.GetGroundtruthLinks()
        links_raw = {}
        for var in self.SEM.keys():
            parents_raw = []
            for p in links[var]:
                if p.__class__ == LaggedVariable:
                    parents_raw.append((indices[p.var], -p.lag))
                else:
                    parents_raw.append((indices[p], 0))
            links_raw[indices[var]] = parents_raw
        return links_raw

    def GetGroundtruthGraph(self):
        """Get the (ground-truth) graph.

        Returns
        -------
        graph, graph-type : see e.g. CausalEffects tutorial, string
        """
        graph = CausalEffects.get_graph_from_dict(self.GetGroundtruthLinksRaw())
        if self.is_timeseries:
            return graph, 'stationary_dag'
        else:
            return graph, 'dag'

    def Validate(self):
        """Validate causal ordering (necessary for consistent data-generation), called automatically."""
        data_pt = DataPointForValidation()
        for var, eq in self.SEM.items():
            data_pt.Set(var, self.SEM[var](data_pt))
        self.max_lag = max(self.max_lag, data_pt.max_lag)
        self.is_timeseries = data_pt.is_timeseries

    def ApplyWithExogenousNoise(self, environment, partial_data=None):
        """Apply to environment

        Parameters
        ----------
        environment : Environment
            Exogenous noise-samples given as environment.
        partial_data : None
            [INTERNAL USE] Leave to default.
        """
        if self.is_timeseries:
            # This may be very slow (as it cannot be parallelized or be dispatched efficiently to native code via numpy)
            return self._GenerateAsTimeseries(environment, partial_data)

        data = environment.GetNoise()
        vars = []
        for var, eq in self.SEM.items():
            if partial_data is not None and var in partial_data:
                data[var] = partial_data[var]
            else:
                vars.append(var)

        for var in vars:
            data[var] = self.SEM[var](data)
        return data

    def _GenerateAsTimeseries(self, environment, partial_data=None):
        """[INTERNAL] Generate time-series data.

        Generation of time-series data cannot be parallelized of efficiently dispatched to native code.
        """
        data = environment.GetNoise()
        vars = []
        for var, eq in self.SEM.items():
            if partial_data is not None and var in partial_data:
                data[var] = partial_data[var]
            else:
                data[var] = var.Empty(environment.N)
                vars.append(var)

        data_pt = DataPointView(data, partial_data)
        for x in range(environment.N):
            for var in vars:
                data_pt.Set(var, self.SEM[var](data_pt))
            data_pt.Next()
        return data

    def Intervene(self, changes):
        """Get and intervened model.

        Parameters
        ----------
        changes : dictionary< VariableDescription, ...>
            For each variable to intervene on, either a scalar (hard intervention),
            or a *callable* replacing the equation in the SEM (see constructor).

        Returns
        -------
        intervened model : Model
            The intervened model.
        """
        # Return a model, that describes the intervened system

        new_sem = self.SEM.copy()
        for var, eq in changes.items():
            if callable(eq):
                new_sem[var] = eq  # intervene by function
            else:
                new_sem[var] = lambda data: eq  # return a constant
        return Model(new_sem)


class World:
    """A 'world' instantiation.

    Generate observations from an environment (exogenous noise) and a model (SEM).

    Parameters
    ----------
    environment : Environment
        The environment with exogenous noise samples.
    model : Model
        The SEM describing the system.
    """

    def __init__(self, environment, model):
        self.environment = environment  # to check agreement in counterfactual worlds
        if model is not None:
            self.data = model.ApplyWithExogenousNoise(environment)
        else:
            self.data = {}

    def Observables(self):
        """Get all observables

        Returns
        -------
        observables : dictonary< VariableDescription, np.array( environment.N ) >
            The samples for each observable variable.
        """
        obs = {}
        for var, values in self.data.items():
            if var.observable:
                obs[var] = values
        return obs


class CounterfactualWorld(World):
    """A 'counterfactual' world.

    Generate observations from one environment (exogenous noise) and multiple models (SEM).

    Parameters
    ----------
    environment : Environment
        The environment with exogenous noise samples.
    base-model : Model
        The base-model to generate data from in the end.
        Call TakeVariablesFromWorld overwrite some variables with values from another 'world',
        then call Compute.
    """

    def __init__(self, environment, model):
        super().__init__(environment, None)
        self.model = model

    def TakeVariablesFromWorld(self, world, vars):
        """Take variables from a world.

        Parameters
        ----------
        world : World
            Take samples from here.
        vars : VariableDescription or *iterable* <VariableDescription>
            One (or a iterable of many) variable to overwrite.
        """
        if world.environment != self.environment:
            raise Exception("Counterfactual Worlds must share exogenous noise terms.")

        if hasattr(vars, '__iter__'):
            for var in vars:
                self.TakeVariablesFromWorld(world, var)
        else:
            self.data[vars] = world.data[vars]

    def Compute(self):
        """Compute the counterfactual world observations."""
        self.data = self.model.ApplyWithExogenousNoise(self.environment, partial_data=self.data)


class DataPointView_TS_Window:
    """[INTERNAL] Injected into SEM callbacks to compute values for counterfactual windows in parallel."""

    def __init__(self, env, stationary_data, window_data, offset_in_window, max_lag):
        self.stationary_data = stationary_data
        self.environment = env
        self.window_data = window_data
        self.offset_in_window = offset_in_window
        self.max_lag = max_lag

    def __getitem__(self, key):
        if key.exogenous:
            return self.environment.noise[key][self.max_lag:]
        else:
            if self.offset_in_window < key.LagValue():
                return self.stationary_data[key.Id()][self.max_lag - key.LagValue():-key.LagValue()]
            else:
                return self.window_data[self.offset_in_window - key.LagValue()][key.Id()]


class CounterfactualTimeseries:
    """[INTERNAL] Used to generate ground-truth for time-series NDE.

    Computes the effect of an intervention at a SINGLE point in time.
    Use GroundTruth_* functions instead.
    """

    def __init__(self, environment, base_model, max_lag_in_interventions):
        self.environment = environment
        self.base_model = base_model
        self.max_lag = max_lag_in_interventions

        self.stationary_data = self.base_model.ApplyWithExogenousNoise(self.environment)

    def ComputeIntervention(self, output_var, interventions, take):
        output_windows = []
        for var, value in interventions:
            intervened_window = []
            output_windows.append(intervened_window)
            found_in_window = False
            for delta_t in range(self.max_lag + 1):
                data_pts = DataPointView_TS_Window(self.environment, self.stationary_data,
                                                   intervened_window, delta_t, self.max_lag)
                current_time = {}
                intervened_window.append(current_time)
                for v, eq in self.base_model.SEM.items():
                    if v.Id() == var.Id() and self.max_lag - delta_t == var.LagValue():
                        found_in_window = True
                        current_time[v] = value  # this is a (single) value, but numpy can broadcast it
                    else:
                        current_time[v] = eq(data_pts)
            if not found_in_window:
                raise Exception(f"Intervention {var.Info()}={value} was not found in time-series window.")
        # default to window 0
        cf_window = output_windows[0]
        for m in take:
            cf_window[self.max_lag - m.LagValue()][m.Id()] = output_windows[1][self.max_lag - m.LagValue()][m.Id()]
        data_pts = DataPointView_TS_Window(self.environment, self.stationary_data,
                                           cf_window, self.max_lag, self.max_lag)
        return self.base_model.SEM[output_var](data_pts)


def Ensemble(shared_setup, payloads, runs=1000):
    """Helper to run e.g. estimator vs ground-truth on an ensemble of model-realizations"""

    results = np.zeros([len(payloads), runs])
    get_next = shared_setup
    if not callable(shared_setup):
        environment = shared_setup
        get_next = lambda: environment.Reset()

    for r in range(runs):
        environment = get_next()
        p_idx = 0
        for p in payloads:
            results[p_idx, r] = p(environment)
            p_idx += 1
    return results


def _Fct_on_grid(fct, list_of_points, cf_delta=0.5, normalize_by_delta=False, **kwargs):
    """Helper to evaluate 'fct' on a grid of points"""

    result = []
    for pt in list_of_points:
        result.append(fct(pt, pt + cf_delta, **kwargs))

    if normalize_by_delta:
        return np.array(result) / cf_delta
    else:
        return np.array(result)


def _Fct_smoothed(fct, min_x, max_x, cf_delta=0.5, steps=100, smoothing_gaussian_sigma_in_steps=5,
                  normalize_by_delta=False, boundary_effects="extend range", **kwargs):
    """Helper to evaluate 'fct' on a grid of points with subsequent Gauß-smoothing"""

    stepsize = (max_x - min_x) / steps
    # Extend the window to run on (numpy would extend by zeros if mode="same")
    if boundary_effects == "extend range":
        steps += 6 * smoothing_gaussian_sigma_in_steps - 1
        min_x -= 3 * smoothing_gaussian_sigma_in_steps * stepsize
    else:
        raise Exception("Currently only smoothing mode for boundary-effects is 'extend range'")

    x_values = []  # np.arange with floats is unstable wrt len of output
    result = []
    for i in range(steps + 1):
        pt = stepsize * i + min_x
        x_values.append(pt)
        value = fct(pt, pt + cf_delta, **kwargs)
        result.append(value)

    if normalize_by_delta:
        result = np.array(result) / cf_delta
    else:
        result = np.array(result)

    # cut off convolution-kernel at 3 sigma
    gx = np.arange(-3 * smoothing_gaussian_sigma_in_steps, 3 * smoothing_gaussian_sigma_in_steps)
    gaussian = (np.exp(-(gx / smoothing_gaussian_sigma_in_steps) ** 2 / 2)
                / np.sqrt(2.0 * np.pi) / smoothing_gaussian_sigma_in_steps)
    if len(np.shape(result)) == 1:  # values = (samples)
        smoothed = np.convolve(result, gaussian, mode="valid")
    elif len(np.shape(result)) == 3:  # densities = (samples, categories, cf/te)
        smoothed = np.empty_like(
            result[3 * smoothing_gaussian_sigma_in_steps:1 - 3 * smoothing_gaussian_sigma_in_steps, :, :])
        for cf_te in range(2):
            for p_category in range(np.shape(result)[1]):
                smoothed[:, p_category, cf_te] = np.convolve(result[:, p_category, cf_te], gaussian, mode="valid")
    else:
        raise Exception(f"Invalid result-shape {result.shape}")

    return np.array(x_values)[3 * smoothing_gaussian_sigma_in_steps:1 - 3 * smoothing_gaussian_sigma_in_steps], smoothed


def PlotInfo(**va_arg_dict):
    """Helper to add info to plot-setup"""
    return va_arg_dict


def PlotAbsProbabilities(plt, target_var, data, labels):
    """Helper to plot Effects on Categorical variables (typically set plt=your-pylot-module-name)"""

    fig = plt.figure(figsize=[12.0, 4.8], dpi=75.0, layout='constrained')
    fig.suptitle(labels["title"])
    shared_ax = None
    has_printed_legend = False
    for cY in range(target_var.categories):
        if shared_ax is None:
            shared_ax = plt.subplot(131 + cY)  # digits are rows, cols, index+1
        else:
            plt.subplot(131 + cY, sharey=shared_ax)  # digits are rows, cols, index+1
        for d in data:
            plt.plot(d["x"], d["y"][:, cY, 1], color=d["colorTE"], label=d["labelTE"])
            plt.plot(d["x"], d["y"][:, cY, 0], color=d["colorCF"], label=d["labelCF"])
        plt.xlabel(labels["x"])
        plt.ylabel(labels["y"].format(cY=cY))
        if not has_printed_legend:
            has_printed_legend = True
            fig.legend()
    return fig


def PlotChangeInProbabilities(plt, target_var, data, labels):
    """Helper to plot Effects on Categorical variables (typically set plt=your-pylot-module-name)"""

    fig = plt.figure(figsize=[12.0, 4.8], dpi=75.0, layout='constrained')
    fig.suptitle(labels["title"])
    shared_ax = None
    has_printed_legend = False
    for cY in range(target_var.categories):
        if shared_ax is None:
            shared_ax = plt.subplot(131 + cY)  # digits are rows, cols, index+1
        else:
            plt.subplot(131 + cY, sharey=shared_ax)  # digits are rows, cols, index+1
        for d in data:
            plt.plot(d["x"], d["y"][:, cY, 0] - d["y"][:, cY, 1], color=d["color"], label=d["label"])
        plt.xlabel(labels["x"])
        plt.ylabel(labels["y"].format(cY=cY))
        if not has_printed_legend:
            has_printed_legend = True
            fig.legend()
    return fig


def FindMaxLag(*va_args):
    """[INTERNAL] Finds the max lag in a collection of variables."""
    max_lag = 0
    for var_group in va_args:
        if hasattr(var_group, '__iter__'):
            for var in var_group:
                max_lag = max(max_lag, var.LagValue())
        else:
            var = var_group
            max_lag = max(max_lag, var.LagValue())
    return max_lag


def GroundTruth_NDE_auto(change_from, change_to, estimator, env, model):
    """Ground-Truth computation from toy-model for NDE.

    GroundTruth_*_auto functions extract source, target, blocked-mediators from an estimator.

    Parameters
    ----------
    change_from : single value of same type as single sample for X (float, int, or bool)
        Reference-value to which X is set by intervention in the world seen by the mediator.
    change_to : single value of same type as single sample for X (float, int, or bool)
        Post-intervention-value to which X is set by intervention in the world seen by the effect (directly).
    estimator : NaturalEffects_GraphMediation
        Extract source, target, blocked-mediators from estimator.
    env : Environment
        The environment used.
    model : Model
        The toy-model.

    Returns
    -------
    NDE : If Y is categorical -> np.array( # categories Y, 2 )
        The probabilities the categories of Y (after, before) changing the interventional value of X
        as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.

    NDE : If Y is continuous -> float
        The change in the expectation-value of Y induced by changing the interventional value of X
        as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.
    """
    return GroundTruth_NDE(change_from, change_to, estimator.Source, estimator.Target, estimator.BlockedMediators,
                           env, model)


def GroundTruth_NDE(change_from, change_to, source, target, mediators, env, model):
    """Ground-Truth computation from toy-model for NDE.

    Note: GroundTruth_*_auto functions extract source, target, blocked-mediators from an estimator.

    Parameters
    ----------
    change_from : single value of same type as single sample for X (float, int, or bool)
        Reference-value to which X is set by intervention in the world seen by the mediator.
    change_to : single value of same type as single sample for X (float, int, or bool)
        Post-intervention-value to which X is set by intervention in the world seen by the effect (directly).
    source : VariableDescription
        Effect source.
    target : VariableDescription
        Effect target.
    mediators : *iterable* <VariableDesciption>
        Blocked mediators.
    env : Environment
        The environment used.
    model : Model
        The toy-model.

    Returns
    -------
    NDE : If Y is categorical -> np.array( # categories Y, 2 )
        The probabilities the categories of Y (after, before) changing the interventional value of X
        as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.

    NDE : If Y is continuous -> float
        The change in the expectation-value of Y induced by changing the interventional value of X
        as "seen" by Y from change_from to change_to, while keeping M as if X remained at change_from.
    """

    if target.LagValue() != 0:
        raise Exception("Do not use lagged targets, it is always possible to shift everything, so that the"
                        "effect is on an unlagged variable.")

    if model.is_timeseries:
        # Generate groundtruth for intervention at a SINGLE point in time

        # Window-size (at each point) must be at least the
        max_lag_in_interventions = FindMaxLag(source, target, mediators)
        # shouldn't be necessary, but fixes some issues with offsets in stationary data:
        max_lag_in_interventions = max(max_lag_in_interventions, model.max_lag)

        ts = CounterfactualTimeseries(env, model, max_lag_in_interventions=max_lag_in_interventions)

        y_cf = ts.ComputeIntervention(target, [(source, change_to), (source, change_from)], mediators)
        y_real = ts.ComputeIntervention(target, [(source, change_from)], [])

    else:
        # Ground-Truth for non-timeseries is straight-forward:

        modelA = model.Intervene(changes={source.Id(): change_from})
        modelB = model.Intervene(changes={source.Id(): change_to})
        worldA = World(env, modelA)
        worldB = World(env, modelB)
        cf_world = CounterfactualWorld(env, model)
        cf_world.TakeVariablesFromWorld(worldA, mediators)
        cf_world.TakeVariablesFromWorld(worldB, source)
        cf_world.Compute()

        y_cf = cf_world.Observables()[target]
        y_real = worldA.Observables()[target]

    if target.is_categorical:
        result = []
        for cY in range(target.categories):
            result.append([np.count_nonzero(y_cf == cY), np.count_nonzero(y_real == cY)])
        return np.array(result) / env.N
    else:
        return np.mean(y_cf - y_real)


def GroundTruth_NDE_fct_auto(x_min, x_max, estimator, env, model, cf_delta=0.5, normalize_by_delta=False,
                             grid_stepping=0.1):
    """See GroundTruth_NDE_auto and NaturalEffects_GraphMediation.NDE_smoothed."""
    source = estimator.Source
    target = estimator.Target
    blocked_mediators = estimator.BlockedMediators
    grid = np.arange(x_min, x_max, grid_stepping)
    return grid, GroundTruth_NDE_fct(source, target, blocked_mediators, env, model, grid, cf_delta, normalize_by_delta)


def GroundTruth_NDE_fct(source, target, mediators, env, model, list_of_points, cf_delta=0.5, normalize_by_delta=False):
    """See GroundTruth_NDE and NaturalEffects_GraphMediation.NDE_smoothed."""
    return _Fct_on_grid(GroundTruth_NDE, list_of_points, cf_delta, normalize_by_delta,
                        source=source, target=target, mediators=mediators, env=env, model=model)


def GroundTruth_NIE_fct(source, target, mediator, env, model, list_of_points, cf_delta=0.5, normalize_by_delta=False):
    """See GroundTruth_NIE and NaturalEffects_StandardMediation.NIE_smoothed."""
    return _Fct_on_grid(GroundTruth_NIE, list_of_points, cf_delta, normalize_by_delta,
                        source=source, target=target, mediator=mediator, env=env, model=model)


def GroundTruth_NIE(change_from, change_to, source, target, mediator, env, model):
    """Ground-Truth computation from toy-model for NIE.

    Standard-mediation setup only.

    Parameters
    ----------
    change_from : single value of same type as single sample for X (float, int, or bool)
        Reference-value to which X is set by intervention in the world seen by the mediator.
    change_to : single value of same type as single sample for X (float, int, or bool)
        Post-intervention-value to which X is set by intervention in the world seen by the effect (directly).
    source : VariableDescription
        Effect source.
    target : VariableDescription
        Effect target.
    mediator : VariableDesciption
        Effect mediator.
    env : Environment
        The environment used.
    model : Model
        The toy-model.

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

    modelA = model.Intervene(changes={source: change_from})
    modelB = model.Intervene(changes={source: change_to})
    worldA = World(env, modelA)
    worldB = World(env, modelB)
    cf_world = CounterfactualWorld(env, model)
    # this time, take the mediator from the B-world
    cf_world.TakeVariablesFromWorld(worldA, [source])
    cf_world.TakeVariablesFromWorld(worldB, [mediator])
    cf_world.Compute()

    y_cf = cf_world.Observables()[target]
    y_real = worldA.Observables()[target]

    if target.is_categorical:
        result = []
        for cY in range(target.categories):
            result.append([np.count_nonzero(y_cf == cY), np.count_nonzero(y_real == cY)])
        return np.array(result) / env.N
    else:
        return np.mean(y_cf - y_real)


class DataHandler:
    """[INTERNAL] Implement some helper functions and generate time-series data via tigramite's data-frames."""

    def __init__(self, observables, dataframe_based_preprocessing=True):
        if isinstance(observables, DataFrame):
            self._from_dataframe = True
            self._data = VariablesFromDataframe(observables)
            self._dataframe = observables
            self.use_dataframe_for_preprocessing = dataframe_based_preprocessing
            self.data_selection_virtual_ids = {}
            self._info = {}
            self.preprocessed_data = False
            self.data_selection_frozen = False # filter all data for all fits for missing values, lock after first data-access (see GetVariableAuto and operator [] / __getitem__)
        else:
            self._from_dataframe = False
            self._data = observables
            self._dataframe = DataframeFromVariables(observables)            
            self.use_dataframe_for_preprocessing = False

        self._indices = {}
        self._keys = list(self._data.keys())
        idx = 0
        for var in self._data.keys():
            self._indices[var] = idx
            idx += 1


    def GetIdFor(self, idx, lag, display_name):
        # self._keys[idx] contains info (eg catgorical or continuous, 'original name') for variable idx, attach display name (eg Mediator)
        # also add 'original name' eg 'temperature' in square brackets as well as lag
        base_id =  self._keys[idx]
        return base_id.CopyInfo(display_name + f"[{base_id.Info(detail=1)} at lag {lag}]")
    
    def RequirePreprocessingFor(self, var, info):
        if self.use_dataframe_for_preprocessing:            
            if var not in self.data_selection_virtual_ids:
                # if variable locked also ok after frozen
                if self.data_selection_frozen:
                    raise Exception("Cannot add variables to preprocessing after variable-set has been locked-in")
                assert info is not None
                var_id = self.GetIdFor(var[0], var[1], info)
                self.data_selection_virtual_ids[var] = var_id
                self._indices[var_id] = var
                self._info[var_id] = info

    def GetVariableAuto(self, var, info="Other"):
        if self._from_dataframe:
            return self.ReverseLookupSingle(var, info)
        else:
            return var

    def GetVariablesAuto(self, vars, info="Other"):
        result = []
        for var in vars:
            result.append(self.GetVariableAuto(var, info))
        return result

    def DataFrame(self):
        return self._dataframe

    def __getitem__(self, key):
        if hasattr(key, '__iter__'):
            return [self[entry] for entry in key]
        
        if self.use_dataframe_for_preprocessing:
            return self._indices[key]

        if key.__class__ == LaggedVariable:
            return self._indices[key.var], -key.lag
        else:
            return self._indices[key], 0

    def _PreprocessData(self, **kwargs):
        self.data_selection_frozen = True
        self.preprocessed_data = True
        X = []
        Y = []
        Z = []
        M = []
        for var_lag_index, var_id in self.data_selection_virtual_ids.items():
            info = self._info[var_id]
            if info == "Source":
                X.append(var_lag_index)
            elif info == "Target":
                Y.append(var_lag_index)
            elif info == "Mediator":
                M.append(var_lag_index)
            elif info == "Adjustment":
                Z.append(var_lag_index)
            else:
                raise Exception("Unknown Variable-Interpretation")
        data_preprocessed, xyz, data_type = self.DataFrame().construct_array(X=X, Y=Y, Z=Z, extraZ=M, **kwargs) # kw-args forwards eg tau-max
        self._data = {}
        i = 0
        for x in X:
            self._data[self.data_selection_virtual_ids[x]] = data_preprocessed[i]
            assert xyz[i] == 0
            i += 1
        for y in Y:
            self._data[self.data_selection_virtual_ids[y]] = data_preprocessed[i]
            assert xyz[i] == 1
            i += 1
        for z in Z:
            self._data[self.data_selection_virtual_ids[z]] = data_preprocessed[i]
            assert xyz[i] == 2
            i += 1
        for m in M:
            self._data[self.data_selection_virtual_ids[m]] = data_preprocessed[i]
            assert xyz[i] == 3
            i += 1
        assert i == data_preprocessed.shape[0]
            
    def GetPreprocessed(self, vars, **kwargs):
        if not self.preprocessed_data:
            self._PreprocessData(**kwargs)
        result = {}
        for var in vars:
            assert var in self.data_selection_virtual_ids
            var_id = self.data_selection_virtual_ids[var]
            result[var_id] = self._data[var_id]
        return list(result.keys()), result
            

    def Get(self, name, vars, **kwargs):
        if self.use_dataframe_for_preprocessing:
            return self.GetPreprocessed(vars, **kwargs)
        ids = []
        i = 1
        for idx, lag in vars:
            # assemble names for this variable-group, eg "Mediator5"
            the_name = name if len(vars) == 1 else name + str(i)
            ids.append( self.GetIdFor(idx, lag, the_name) )
            i += 1
        data, xyz, data_type = self.DataFrame().construct_array(X=vars, Y=[], Z=[], **kwargs)
        assert data.shape[0] == len(ids)
        result = {}
        i = 0
        for elem in ids:
            result[elem] = data[i].astype(dtype=np.dtype(elem.dtype))
            i += 1
        return ids, result

    def ReverseLookupSingle(self, index, info):
        if self.use_dataframe_for_preprocessing:
            self.RequirePreprocessingFor(index, info)
            return self.data_selection_virtual_ids[index]
        else:
            return self._keys[index[0]].Lag(-index[1])

    def ReverseLookupMulti(self, index_set, info=None):
        return [self.ReverseLookupSingle(index, info) for index in index_set]


def VariablesFromDataframe(dataframe):
    """Extract Category-Information from tigramite::dataframe

    Parameters
    ----------
    dataframe : tigramite.data_processing.DataFrame
        Dataframe to extract data from.


    Returns
    -------
    variables : dictionary< VariableDescription, np.array(N) >
        The variable-meta-data and data.
    """

    data_types = dataframe.data_type
    if dataframe.data_type is not None:
        # Require data-types to be constant
        first_elements = data_types[0, :]
        if not np.all(first_elements == data_types):
            raise NotImplementedError("Natural Effect Framework currently only supports per variable"
                                      "data-types, this dataframe contains variables with changing"
                                      "(over time) data-types.")

    result = {}  # var i will be in result.items()[i]
    data = dataframe.values[0]
    node_count = np.shape(data)[1]
    for node in range(node_count):
        if data_types is None or data_types[0, node] == 0:
            var = ContinuousVariable(name=str(dataframe.var_names[node]))
            result[var] = data[:, node]
        else:
            labels, transformed = np.unique(data[:, node], return_inverse=True)
            var = CategoricalVariable(name=str(dataframe.var_names[node]), categories=labels)
            result[var] = transformed
    return result


def DataframeFromVariables(data_dict):
    """Convert Category-Information to tigramite::dataframe

    Parameters
    ----------
    data_dict : dictionary< VariableDescription, np.array(N) >
        The variable-meta-data and data.

    Returns
    -------
    dataframe : tigramite.data_processing.DataFrame
        Dataframe containing the raw data.
    """

    data_len = 0
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    if keys[0].is_categorical or keys[0].dimension == 1:
        data_len = np.shape(values[0])[0]
    else:
        data_len = np.shape(values[0])[1]
    dimensions = 0
    for var, data in data_dict.items():
        if var.is_categorical:
            dimensions += 1
        else:
            dimensions += var.dimension

    data_out = np.zeros([data_len, dimensions])
    data_type = np.zeros([data_len, dimensions])
    names = []
    i = 0
    for var, data in data_dict.items():
        if var.is_categorical:
            names.append(var.name)
            if var.category_values is not None:
                data_out[:, i] = np.take(var.category_values, data)
            else:
                data_out[:, i] = data
            data_type[:, i] = np.ones(data_len)
            i += 1
        else:
            if var.dimension > 1:
                for j in range(var.dimension):
                    if var.dimension == 1:
                        names.append(var.name)
                    else:
                        names.append(var.name + "$_" + str(j) + "$")
                    data_out[:, i + j] = data[j]
                    data_type[:, i + j] = np.zeros(data_len)
            else:
                names.append(var.name)
                data_out[:, i] = data
                data_type[:, i] = np.zeros(data_len)
            i += var.dimension
    return DataFrame(data=data_out, data_type=data_type, var_names=names)
