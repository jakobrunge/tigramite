from __future__ import print_function
import pytest
import numpy

import tigramite.data_processing as pp

# Generate typical data
TYPICAL_AUTO = 0.5
TYPICAL_LINC = 0.6
TYPICAL_TIME = 1000
TYPICAL_INNO = 'inv_inno_cov'
TYPICAL_INIT = None
TYPICAL_COEF = {
    0: [((0, -1), TYPICAL_AUTO)],
    1: [((1, -1), TYPICAL_AUTO), ((0, -1), TYPICAL_LINC)],
    2: [((2, -1), TYPICAL_AUTO), ((1, -1), TYPICAL_LINC)]
    }
TYPICAL_PARA = (TYPICAL_COEF,
                TYPICAL_TIME,
                TYPICAL_INIT,
                TYPICAL_INNO)

# Generate exponential decay data
EXP_DECAY_AUTO = 0.99
EXP_DECAY_COEF = {
    0:[((0, -1), EXP_DECAY_AUTO)]
    }
EXP_DECAY_TIME = 1000
EXP_DECAY_INIT = numpy.array([[1000, 1000*EXP_DECAY_AUTO]])
EXP_DECAY_INNO = 'no_inno'
EXP_DECAY_PARA = (EXP_DECAY_COEF,
                  EXP_DECAY_TIME,
                  EXP_DECAY_INIT,
                  EXP_DECAY_INNO)
# Generate the expected value
EXP_DECAY_EXPT = numpy.empty(EXP_DECAY_TIME)
EXP_DECAY_EXPT[:2] = EXP_DECAY_INIT[0]
for i in range(2, EXP_DECAY_TIME):
    EXP_DECAY_EXPT[i] = EXP_DECAY_EXPT[i-1] * EXP_DECAY_AUTO

def gen_process(params):
    # Set up the parameters
    all_coeff, max_time, initial_values, use_val = params
    # Generate the data
    data, true_parents_neighbors = pp.var_process(all_coeff,
                                                  T=max_time,
                                                  initial_values=initial_values,
                                                  use=use_val)
    return data, true_parents_neighbors, all_coeff

@pytest.mark.parametrize('params', [TYPICAL_PARA, EXP_DECAY_PARA])
def test_var_process_parent_neighbours(params):
    _, return_node_links, all_coeff = gen_process(params)
    # Strip the coefficients from the input parameters
    true_node_links = dict()
    for node_id, all_node_links in all_coeff.items():
        true_node_links[node_id] = [link for link, _ in all_node_links]
    # Check this matches the given parameters
    assert return_node_links == true_node_links

@pytest.mark.parametrize('params, expect', [
    (EXP_DECAY_PARA, EXP_DECAY_EXPT)])
def test_var_process_data(params, expect):
    # Generate the data
    data, _, all_coeff = gen_process(params)
    # Strip the coefficients from the input parameters
    numpy.testing.assert_array_equal(data[:,0], expect)
