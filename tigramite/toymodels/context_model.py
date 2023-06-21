import numpy as np
from numpy.random import MT19937

from tigramite.toymodels import structural_causal_processes as toys


def shift_link_entries(links, const):
    shifted_links = {}
    for key in links.keys():
        shifted_key = key + const
        values = links[key]
        shifted_values = [((item + const, lag), c, f) for ((item, lag), c, f) in values]
        shifted_links[shifted_key] = shifted_values
    return shifted_links


class ContextModel:
    """
    TODO: adapt
    Returns a time series generated from a structural causal process.

        Allows lagged and contemporaneous dependencies and includes the option
        to have intervened variables or particular samples.

        The interventional data is in particular useful for generating ground
        truth for the CausalEffects class.

        In more detail, the method implements a generalized additive noise model process of the form

        .. math:: X^j_t = \\eta^j_t + \\sum_{X^i_{t-\\tau}\\in \\mathcal{P}(X^j_t)}
                  c^i_{\\tau} f^i_{\\tau}(X^i_{t-\\tau})

        Links have the format ``{0:[((i, -tau), coeff, func),...], 1:[...],
        ...}`` where ``func`` can be an arbitrary (nonlinear) function provided
        as a python callable with one argument and coeff is the multiplication
        factor. The noise distributions of :math:`\\eta^j` can be specified in
        ``noises``.

        Through the parameters ``intervention`` and ``intervention_type`` the model
        can also be generated with intervened variables.

        Parameters
        ----------
        links : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument.
        T : int
            Sample size.
        noises : list of callables or array, optional (default: 'np.random.randn')
            Random distribution function that is called with noises[j](T). If an array,
            it must be of shape ((transient_fraction + 1)*T, N).
        intervention : dict
            Dictionary of format: {1:np.array, ...} containing only keys of intervened
            variables with the value being the array of length T with interventional values.
            Set values to np.nan to leave specific time points of a variable un-intervened.
        intervention_type : str or dict
            Dictionary of format: {1:'hard',  3:'soft', ...} to specify whether intervention is
            hard (set value) or soft (add value) for variable j. If str, all interventions have
            the same type.
        transient_fraction : float
            Added percentage of T used as a transient. In total a realization of length
            (transient_fraction + 1)*T will be generated, but then transient_fraction*T will be
            cut off.
        seed : int, optional (default: None)
            Random seed.

        Returns
        -------
        data : array-like
            Data generated from this process, shape (T, N).
        nonvalid : bool
            Indicates whether data has NaNs or infinities.

        """
    def __init__(self, links_tc={}, links_sc={}, links_sys={}, noises=None, seed=None):
        self.N = len(links_sys.keys())
        self.links_tc = links_tc
        self.links_sc = links_sc
        self.links_sys = links_sys
        self.noises = noises
        self.seed = seed

    def temporal_random(self, links_tc, T, seed):
        if self.noises is not None:
            noises_tc = [self.noises[key] for key in links_tc.keys()]
        else:
            noises_tc = None
        shifted_links_tc = shift_link_entries(links_tc, -self.N)
        data_tc, nonstat_tc = toys.structural_causal_process(shifted_links_tc, T=T, noises=noises_tc,
                                                             seed=MT19937(seed))
        data_tc = {i: data_tc[:, i - self.N] for i in links_tc.keys()}
        return data_tc, nonstat_tc

    def spatial_random(self, links_sc, M, shift, seed):
        shifted_links_sc = shift_link_entries(links_sc, -shift)
        if self.noises is not None:
            noises_sc = [self.noises[key] for key in links_sc.keys()]
        else:
            noises_sc = None
        data_sc, nonstat_sc = toys.structural_causal_process(shifted_links_sc, T=M, noises=noises_sc,
                                                             seed=MT19937(seed))
        return data_sc, nonstat_sc

    def generate_data(self, M, T, transient_fraction):

        links = {**self.links_tc, **self.links_sc, **self.links_sys}

        K_time = len(self.links_tc.keys())
        K_space = len(self.links_sc.keys())

        data = {}
        data_tc = {}
        data_sc = {}
        nonstat_tc = False
        nonstat_sc = False
        nonstationary = []

        child_seeds = self.seed.spawn(3)

        # first generate data for temporal context nodes
        if K_time != 0:
            data_tc, nonstat_tc = self.temporal_random(self.links_tc, T, child_seeds[0])
        data_tc_list = [data_tc for m in range(M)]

        # generate spatial context data (constant in time)
        if K_space != 0:
            data_sc, nonstat_sc = self.spatial_random(self.links_sc, M, K_time + self.N, child_seeds[1])

        for m in range(M):  # assume that this is a given order of datasets
            data_sc_m = {i: np.repeat(data_sc[m, i - self.N - K_time], T) for i in self.links_sc.keys()}

            data_context = dict(data_tc_list[m])
            data_context.update(data_sc_m)

            # generate system data that varies over space and time
            data_m, nonstat = toys.structural_causal_process(links, T=T, intervention=data_context,
                                                             seed=MT19937(child_seeds[2]), noises=self.noises)
            data[m] = data_m
            nonstationary.append(nonstat or nonstat_tc or nonstat_sc)
        return data, np.any(nonstationary)
