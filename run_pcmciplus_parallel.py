#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tigramite causal discovery for time series: Parallization script implementing 
the PCMCIplus method based on mpi4py. 

Parallelization is done across variables j for ONLY the PC condition-selection
step
"""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0


from mpi4py import MPI
import numpy
import os, sys, pickle
from copy import deepcopy

from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]


def run_pc_stable_parallel(j):
    """Wrapper around PCMCI.run_pc_stable estimating the parents for a single 
    variable j.

    Parameters
    ----------
    j : int
        Variable index.

    Returns
    -------
    j, pcmci_of_j, parents_of_j : tuple
        Variable index, PCMCI object, and parents of j
    """

    # CondIndTest is initialized globally below
    # Further parameters of PCMCI as described in the documentation can be
    # supplied here:
    pcmci_of_j = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    parents_of_j = pcmci_of_j.run_pc_stable(
        selected_links=selected_links_parallelized[j],
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
    )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j


# Example data, here the real dataset can be loaded as a numpy array of shape
# (T, N)
numpy.random.seed(42)     # Fix random seed
def lin_f(x):
    return x
links_coeffs = {0: [((0, -1), 0.7, lin_f)],
                1: [((1, -1), 0.8, lin_f), ((0, -1), 0.8, lin_f)],
                2: [((2, -1), 0.5, lin_f), ((1, 0), 0.5, lin_f)],
                }

T = 1000     # time series length
data, true_parents_neighbors = toys.structural_causal_process(links_coeffs, T=T, seed=7)
T, N = data.shape

# Optionally specify variable names
var_names = [r'$X^0$', r'$X^1$', r'$X^2$']

# Initialize dataframe object
dataframe = pp.DataFrame(data, var_names=var_names)

# Significance level in condition-selection step.
# In this parallelized version it only supports a float,
# not a list or None. But you can can run this script
# for different pc_alpha and then choose the optimal
# pc_alpha as done in "_optimize_pcmciplus_alpha"
pc_alpha = 0.01

# Maximum time lag
tau_max = 3

# Optional minimum time lag
tau_min = 0

# PCMCIplus specific parameters (see docs)
contemp_collider_rule='majority'
conflict_resolution=True
reset_lagged_links=False

# Maximum cardinality of conditions in PC condition-selection step. The
# recommended default choice is None to leave it unrestricted.
max_conds_dim = None

# Maximum number of parents of X/Y to condition on in MCI step, leave this to None
# to condition on all estimated parents.
max_conds_px = None
max_conds_py = None
max_conds_px_lagged = None

# Selected links may be used to restricted estimation to given links.
selected_links = None

# Verbosity level. Note that slaves will ouput on top of each other.
verbosity = 0

# Chosen conditional independence test
cond_ind_test = ParCorr()  #confidence='analytic')

# FDR control applied to resulting p_matrix
fdr_method = 'none'

# Store results in file
file_name = os.path.expanduser('~') + '/test_results.dat'

# Create master PCMCI object
pcmci_master = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)

_int_sel_links = pcmci_master._set_sel_links(selected_links, tau_min, tau_max)

# Used to tell pcmci.run_pc_stable() to only search for links into j variable.
selected_links_parallelized = {n: {m: _int_sel_links[m] if m == n else [] 
                          for m in range(N)} for n in range(N)}


#
#  Start of the script
#
if COMM.rank == 0:
    # Only the master node (rank=0) runs this
    if verbosity > -1:
        pcmci_master._print_pc_params(selected_links, tau_min, tau_max,
                          pc_alpha, max_conds_dim,
                          1)

    # Split selected_variables into however many cores are available.
    splitted_jobs = split(list(range(N)), COMM.size)
    if verbosity > -1:
        print("Splitted selected_variables = ", splitted_jobs)
else:
    splitted_jobs = None


##
## Step 1: Get a superset of lagged parents from run_pc_stable
##
# Scatter jobs across cores.
scattered_jobs = COMM.scatter(splitted_jobs, root=0)

# Now each rank just does its jobs and collects everything in a results list.
results = []
for j in scattered_jobs:
    # Estimate conditions
    (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(j)

    results.append((j, pcmci_of_j, parents_of_j))

# Gather results on rank 0.
results = MPI.COMM_WORLD.gather(results, root=0)


if COMM.rank == 0:
    # Collect all results in dictionaries and
    lagged_parents = {}
    p_matrix = numpy.ones((N, N, tau_max + 1))
    val_matrix = numpy.zeros((N, N, tau_max + 1))
    # graph = numpy.zeros((N, N, tau_max + 1), dtype='<U3')
    # graph[:] = ""

    for res in results:
        for (j, pcmci_of_j, parents_of_j) in res:
            lagged_parents[j] = parents_of_j[j]
            p_matrix[:, j, :] = pcmci_of_j.p_matrix[:, j, :]
            val_matrix[:, j, :] = pcmci_of_j.val_matrix[:, j, :]

    if verbosity > -1:
        print("\n\n## Resulting lagged condition sets:")
        for j in [var for var in lagged_parents.keys()]:
            pcmci_master._print_parents_single(j, lagged_parents[j],
                                                   None,
                                                   None)


    # Step 2+3+4: PC algorithm with contemp. conditions and MCI tests
    ##
    ##  This step is currently NOT parallelized, all run on master
    ##
    if verbosity > -1:
        print("\n##\n## Step 2: PC algorithm with contemp. conditions "
              "and MCI tests\n##"
              "\n\nParameters:")
        if selected_links is not None:
            print("\nselected_links = %s" % _int_sel_links)
        print("\nindependence test = %s" % cond_ind_test.measure
              + "\ntau_min = %d" % tau_min
              + "\ntau_max = %d" % tau_max
              + "\npc_alpha = %s" % pc_alpha
              + "\ncontemp_collider_rule = %s" % contemp_collider_rule
              + "\nconflict_resolution = %s" % conflict_resolution
              + "\nreset_lagged_links = %s" % reset_lagged_links
              + "\nmax_conds_dim = %s" % max_conds_dim
              + "\nmax_conds_py = %s" % max_conds_py
              + "\nmax_conds_px = %s" % max_conds_px
              + "\nmax_conds_px_lagged = %s" % max_conds_px_lagged
              + "\nfdr_method = %s" % fdr_method
              )

    # lagged_parents = all_results['lagged_parents']
    # p_matrix = all_results['p_matrix']
    # val_matrix = all_results['val_matrix']
    # graph = all_results['graph']
    # if verbosity > -1:
    #     print(all_results['graph'])


    # Set the maximum condition dimension for Y and X
    max_conds_py = pcmci_master._set_max_condition_dim(max_conds_py,
                                               tau_min, tau_max)
    max_conds_px = pcmci_master._set_max_condition_dim(max_conds_px,
                                               tau_min, tau_max)

    if reset_lagged_links:
        # Run PCalg on full graph, ignoring that some lagged links
        # were determined as non-significant in PC1 step
        links_for_pc = deepcopy(_int_sel_links)
    else:
        # Run PCalg only on lagged parents found with PC1 
        # plus all contemporaneous links
        links_for_pc = deepcopy(lagged_parents)
        for j in range(N):
            for link in _int_sel_links[j]:
                i, tau = link
                if abs(tau) == 0:
                    links_for_pc[j].append((i, 0))

    results = pcmci_master.run_pcalg(
        selected_links=links_for_pc,
        pc_alpha=pc_alpha,
        tau_min=tau_min,
        tau_max=tau_max,
        max_conds_dim=max_conds_dim,
        max_combinations=None,
        lagged_parents=lagged_parents,
        max_conds_py=max_conds_py,
        max_conds_px=max_conds_px,
        max_conds_px_lagged=max_conds_px_lagged,
        mode='contemp_conds',
        contemp_collider_rule=contemp_collider_rule,
        conflict_resolution=conflict_resolution)

    graph = results['graph']

    # Update p_matrix and val_matrix with values from links_for_pc
    for j in range(N):
        for link in links_for_pc[j]:
            i, tau = link
            p_matrix[i, j, abs(tau)] = results['p_matrix'][i, j, abs(tau)]
            val_matrix[i, j, abs(tau)] = results['val_matrix'][i, j, 
                                                               abs(tau)]

    # Update p_matrix and val_matrix for indices of symmetrical links
    p_matrix[:, :, 0] = results['p_matrix'][:, :, 0]
    val_matrix[:, :, 0] = results['val_matrix'][:, :, 0]

    ambiguous = results['ambiguous_triples']

    conf_matrix = None

    # Correct the p_matrix if there is a fdr_method
    if fdr_method != 'none':
        p_matrix = pcmci_master.get_corrected_pvalues(p_matrix=p_matrix, tau_min=tau_min, 
                                              tau_max=tau_max, 
                                              selected_links=_int_sel_links,
                                              fdr_method=fdr_method)

    # Cache the resulting values in the return dictionary
    return_dict = {'graph': graph,
                   'val_matrix': val_matrix,
                   'p_matrix': p_matrix,
                   'ambiguous_triples': ambiguous,
                   'conf_matrix': conf_matrix}

    # Print the results
    if verbosity > -1:
        pcmci_master.print_results(return_dict, alpha_level=pc_alpha)
    
    # Save the dictionary

    if verbosity > -1:
        print("Pickling to ", file_name)
    file = open(file_name, 'wb')
    pickle.dump(return_dict, file, protocol=-1)
    file.close()

