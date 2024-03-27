import numpy as np

from tigramite.toymodels import structural_causal_processes as toys


def _nb_latent_before(node, observed_context_indices, node_classification):
    return len(
        [el for el in range(node) if not (el in observed_context_indices or node_classification[el] == "system")])

def _do_dummy_projection(links, node_classification, observed_context_indices, time_dummy_index, space_dummy_index):
    """
    Helper function to augment the true_parents by remove context-context links and adding dummy
    (i.e. perform dummy projection)

    links : dictionary
        Ground truth links
    node_classification : dictionary
        Corresponds to ground truth links
    """

    # remove latent links, shift remaining, add dummy
    augmented_links = {}
    for node in node_classification.keys():
        if node_classification[node] == "system":
            keep_parents = []
            for parent in links[node]:
                if node_classification[parent[0][0]] == "system":
                    keep_parents.append(parent)
                elif node_classification[parent[0][0]] == "time_context":
                    if parent[0][0] in observed_context_indices:
                        keep_parents.append(((parent[0][0] - _nb_latent_before(parent[0][0], observed_context_indices,
                                                                               node_classification),
                                              parent[0][1]), parent[1], parent[2]))
                    else:
                        keep_parents.append(((time_dummy_index, 0), 1., "dummy"))
                elif node_classification[parent[0][0]] == "space_context":
                    if parent[0][0] in observed_context_indices:
                        keep_parents.append(((parent[0][0] - _nb_latent_before(parent[0][0], observed_context_indices,
                                                                               node_classification), parent[0][1]),
                                             parent[1], parent[2]))
                    else:
                        keep_parents.append(((space_dummy_index, 0), 1., "dummy"))
                augmented_links[node] = list(dict.fromkeys(keep_parents))

        # remove all parents of context nodes
        elif node_classification[node] == "time_context":
            if node in observed_context_indices:
                augmented_links[node - _nb_latent_before(node, observed_context_indices, node_classification)] = []
        elif node_classification[node] == "space_context":
            if node in observed_context_indices:
                augmented_links[node - _nb_latent_before(node, observed_context_indices, node_classification)] = []

    augmented_links[time_dummy_index] = []
    augmented_links[space_dummy_index] = []
    return augmented_links

def _shift_link_entries(links, const):
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

def _group_links(links, node_types, node_type):
    return {i: links[i] for i in links.keys() if node_types[i] == node_type}


class ContextModel:
    """Allows to sample from a joint structural causal model over different spatial
        and temporal contexts. Restricts temporal and spatial context nodes to be constant over datasets or
        time, respectively.

        Parameters
        ----------
        links : dict
            Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
            ...} for all variables where i must be in [0..N-1] and tau >= 0 with
            number of variables N. coeff must be a float and func a python
            callable of one argument.
        node_classification : dictionary
            Classification of nodes into system, or context nodes.
            Keys of the dictionary are from {0, ..., N-1} where N is the number of nodes.
            Options for the values are "system", "time_context", "space_context". The temporal context variables are
            assumed exogenous to the system variables. They cannot interact with the spatial context variables due
            to the assumption that they are constant across datasets.
        transient_fraction : float
            Added percentage of T used as a transient. In total a realization of length
            (transient_fraction + 1)*T will be generated, but then transient_fraction*T will be
            cut off.
        noises : list of callables or list of arrays, optional (default: None)
            Random distribution function that is called with noises[j](T) for system and time-context nodes, or
            noises[j](M) for space-context nodes where M is the number of datasets. If list of arrays,
            for noises corresponding to time-context and system variables the array needs to be of
            shape ((transient_fraction + 1)*T, ), for space-context variables it needs to be of shape (M, )
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
        transient_fraction : float
            Added percentage of T used as a transient. In total a realization of length
            (transient_fraction + 1)*T will be generated, but then transient_fraction*T will be
            cut off.
        noises : list of callables or list of arrays, optional (default: None)
            Random distribution function that is called with noises[j](T) for system and time-context nodes, or
            noises[j](M) for space-context nodes where M is the number of datasets. If list of arrays,
            for noises corresponding to time-context and system variables the array needs to be of
            shape ((transient_fraction + 1)*T, ), for space-context variables it needs to be of shape (M, )
        seed : int, optional (default: None)
            Random seed.

        """

    def __init__(self, links={}, node_classification={}, transient_fraction=0.2, noises=None, seed=None):
        self.links_tc = _group_links(links, node_classification, "time_context")
        self.links_sc = _group_links(links, node_classification, "space_context")
        self.links_sys = _group_links(links, node_classification, "system")

        self.N = len(self.links_sys.keys())
        self.noises = noises
        self.seed = seed
        self.transient_fraction = transient_fraction



    def _constant_over_space(self, data_tc, M):
        data_tc_list = [data_tc for _ in range(M)]
        return data_tc_list

    def _constant_over_time(self, data_sc, T, M, shift):
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
            if np.all([isinstance(el, np.ndarray) for el in noises_tc]):
                noises_tc = np.stack(noises_tc).transpose()
        else:
            noises_tc = None
        shifted_links_tc = _shift_link_entries(links_tc, -self.N)
        data_tc, nonstat_tc = toys.structural_causal_process(shifted_links_tc, T=T, noises=noises_tc,
                                                             transient_fraction=self.transient_fraction,
                                                             seed=seed)
        data_tc = {i: data_tc[:, i - self.N] for i in links_tc.keys()}

        data_tc_list = self._constant_over_space(data_tc, M)
        return data_tc_list, np.any(nonstat_tc)

    def _generate_spatial_context_data(self, links_sc, T, M, shift, seed):
        """
        Helper Function to generate data for the spatial context nodes. It essentially is a
        wrapper around toys.structural_causal_process to generate data that is random across datasets
        but constant across time.
        """
        shifted_links_sc = _shift_link_entries(links_sc, -shift)
        if self.noises is not None:
            noises_sc = [self.noises[key] for key in links_sc.keys()]
            if np.all([isinstance(el, np.ndarray) for el in noises_sc]):
                noises_sc = np.stack(noises_sc).transpose()
        else:
            noises_sc = None

        data_sc, nonstat_sc = toys.structural_causal_process(shifted_links_sc, T=M, noises=noises_sc,
                                                             transient_fraction=0.,
                                                             seed=seed)

        data_sc_list = self._constant_over_time(data_sc, T, M, shift)
        return data_sc_list, np.any(nonstat_sc)

    def generate_data(self, M, T):
        """
        Generates M datasets of time series generated from a joint structural causal model over different spatial
        and temporal contexts.

         Parameters
         ----------
         M : int
            Number of datasets.
         T : int
            Sample size.

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

        time_seed = [1, self.seed]
        space_seed = [2, self.seed]
        system_seed = [3, self.seed]

        # first generate data for temporal context nodes
        data_tc_list, nonstat_tc = self._generate_temporal_context_data(self.links_tc, T, M, time_seed)

        # generate spatial context data (constant in time)
        data_sc_list, nonstat_sc = self._generate_spatial_context_data(self.links_sc,
                                                                       T, M,
                                                                       K_time + self.N,
                                                                       space_seed)
        for m in range(M):
            data_context = dict(data_tc_list[m])
            data_context.update(data_sc_list[m])

            if self.noises is not None:
                noises_filled = self.noises
                if np.all([isinstance(el, np.ndarray) for el in self.noises]):
                    noises_filled = np.copy(self.noises)
                    for key in self.links_sc.keys():
                        # fill up any space-context noise to have T entries, then convert to numpy array
                        noises_filled[key] = np.random.standard_normal(len(self.noises[list(self.links_sys.keys())[0]]))
                    noises_filled = np.stack(noises_filled).transpose()
            else:
                noises_filled = None

            # generate system data that varies over space and time
            data_m, nonstat = toys.structural_causal_process(links, T=T, intervention=data_context,
                                                             transient_fraction=self.transient_fraction,
                                                             seed=system_seed, noises=noises_filled)
            data[m] = data_m
            nonstationary.append(nonstat or nonstat_tc or nonstat_sc)
        return data, np.any(nonstationary)
