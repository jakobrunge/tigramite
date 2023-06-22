import numpy as np
from numpy.random import MT19937

from tigramite.toymodels import structural_causal_processes as toys


def shift_link_entries(links, const):
    """
    Helper Function to shift keys and values of a link dictionary by an integer constant.
    """
    shifted_links = {}
    for key in links.keys():
        shifted_key = key + const
        values = links[key]
        shifted_values = [((item + const, lag), c, f) for ((item, lag), c, f) in values]
        shifted_links[shifted_key] = shifted_values
    return shifted_links


class ContextModel:
    """Allows to sample from a joint structural causal model over different spatial
        and temporal contexts. Restricts temporal and spatial context nodes to be constant over datasets or
        time, respectively.

        Parameters
        ----------
        links_tc : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all temporal context variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument. The temporal context variables are assumed exogenous
            to the system variables. They cannot interact with the spatial context variables due to the assumption
            that they are constant across datasets.
        links_sc : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all spatial context variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument. The spatial context variables are assumed exogenous
            to the system variables. They cannot interact with the temporal context variables due to the assumption
            that they are time-independent, i.e. constant across time.
        links_sys : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all system variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument.
        noises : list of callables or array, optional (default: None)
            Random distribution function that is called with noises[j](T). If an array,
            it must be of shape ((transient_fraction + 1)*T, N).
        seed : int, optional (default: None)
            Random seed.

        Attributes
        -------
        links_tc : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all temporal context variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument. The temporal context variables are assumed exogenous
            to the system variables. They cannot interact with the spatial context variables due to the assumption
            that they are constant across datasets.
        links_sc : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all spatial context variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument. The spatial context variables are assumed exogenous
            to the system variables. They cannot interact with the temporal context variables due to the assumption
            that they are time-independent, i.e. constant across time.
        links_sys : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all system variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument.
        noises : list of callables or array, optional (default: None)
            Random distribution function that is called with noises[j](T). If an array,
            it must be of shape ((transient_fraction + 1)*T, N).
        seed : int, optional (default: None)
            Random seed.

        """

    def __init__(self, links_tc={}, links_sc={}, links_sys={}, noises=None, seed=None):
        self.N = len(links_sys.keys())
        self.links_tc = links_tc
        self.links_sc = links_sc
        self.links_sys = links_sys
        self.noises = noises
        self.seed = seed

    def constant_over_space(self, data_tc, M):
        data_tc_list = [data_tc for _ in range(M)]
        return data_tc_list

    def constant_over_time(self, data_sc, T, M, shift):
        data_sc_list = [{i: np.repeat(data_sc[m, i-shift], T) for i in self.links_sc.keys()} for m in
                        range(M)]
        return data_sc_list

    def _generate_temporal_context_data(self, links_tc, T, M, seed):
        """
        Helper Function to generate data for the temporal context nodes. It essentially is a
        wrapper around toys.structural_causal_process to generate data that is random across time
        but constant across datasets.
        """
        if self.noises is not None:
            noises_tc = [self.noises[key] for key in links_tc.keys()]
        else:
            noises_tc = None
        shifted_links_tc = shift_link_entries(links_tc, -self.N)
        data_tc, nonstat_tc = toys.structural_causal_process(shifted_links_tc, T=T, noises=noises_tc,
                                                             seed=MT19937(seed))
        data_tc = {i: data_tc[:, i - self.N] for i in links_tc.keys()}

        data_tc_list = self.constant_over_space(data_tc, M)
        return data_tc_list, np.any(nonstat_tc)

    def _generate_spatial_context_data(self, links_sc, T, M, shift, seed):
        """
        Helper Function to generate data for the spatial context nodes. It essentially is a
        wrapper around toys.structural_causal_process to generate data that is random across datasets
        but constant across time.
        """
        shifted_links_sc = shift_link_entries(links_sc, -shift)
        if self.noises is not None:
            noises_sc = [self.noises[key] for key in links_sc.keys()]
        else:
            noises_sc = None
        data_sc, nonstat_sc = toys.structural_causal_process(shifted_links_sc, T=M, noises=noises_sc,
                                                             seed=MT19937(seed))

        data_sc_list = self.constant_over_time(data_sc, T, M, shift)
        return data_sc_list, np.any(nonstat_sc)

    def generate_data(self, M, T):
        """
        Generates M datasets of time series generated from a joint structural causal model over different spatial
        and temporal contexts.

        Returns
        ----------
        data : dictionary with array-like values
            Datasets generated from this process, each dataset has the shape (T, N).
        nonvalid : bool
            Indicates whether data has NaNs or infinities.
        """

        links = {**self.links_tc, **self.links_sc, **self.links_sys}

        K_time = len(self.links_tc.keys())

        data = {}
        nonstationary = []

        child_seeds = self.seed.spawn(3)

        # first generate data for temporal context nodes
        data_tc_list, nonstat_tc = self._generate_temporal_context_data(self.links_tc, T, M, child_seeds[0])

        # generate spatial context data (constant in time)
        data_sc_list, nonstat_sc = self._generate_spatial_context_data(self.links_sc,
                                                                       T, M,
                                                                       K_time + self.N,
                                                                       child_seeds[1])
        for m in range(M):
            data_context = dict(data_tc_list[m])
            data_context.update(data_sc_list[m])

            # generate system data that varies over space and time
            data_m, nonstat = toys.structural_causal_process(links, T=T, intervention=data_context,
                                                             seed=MT19937(child_seeds[2]), noises=self.noises)
            data[m] = data_m
            nonstationary.append(nonstat or nonstat_tc or nonstat_sc)
        return data, np.any(nonstationary)
