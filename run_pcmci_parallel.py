#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tigramite causal discovery for time series: Parallization script implementing 
the PCMCI method based on mpi4py. 

Parallelization is done across variables j for both the PC condition-selection
step and the MCI step.
"""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0


from mpi4py import MPI
import numpy
import os, sys, pickle

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb


# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a the range of selected variables (or range(N)) 
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
        selected_variables=[j],
        verbosity=verbosity)

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    parents_of_j = pcmci_of_j.run_pc_stable(
                      selected_links=selected_links,
                      tau_max=tau_max,
                      pc_alpha=pc_alpha,
            )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j


def run_mci_parallel(j, pcmci_of_j, all_parents):
    """Wrapper around PCMCI.run_mci step.


    Parameters
    ----------
    j : int
        Variable index.

    pcmci_of_j : object
        PCMCI object for variable j. This may contain pre-computed results 
        (such as residuals or null distributions).

    all_parents : dict
        Dictionary of parents for all variables. Needed for MCI independence
        tests.

    Returns
    -------
    j, results_in_j : tuple
        Variable index and results dictionary containing val_matrix, p_matrix,
        and optionally conf_matrix with non-zero entries only for
        matrix[:,j,:].
    """
    results_in_j = pcmci_of_j.run_mci(
                selected_links=selected_links,
                tau_min=tau_min,
                tau_max=tau_max,
                parents=all_parents,
                max_conds_px=max_conds_px,
            )

    return j, results_in_j


# Example data, here the real dataset can be loaded as a numpy array of shape
# (T, N)
numpy.random.seed(42)     # Fix random seed
links_coeffs = {0: [((0, -1), 0.7)],
                1: [((1, -1), 0.8), ((0, -1), 0.8)],
                2: [((2, -1), 0.5), ((1, -2), 0.5)],
                }

T = 500     # time series length
data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
T, N = data.shape

# Optionally specify variable names
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']

# Initialize dataframe object
dataframe = pp.DataFrame(data, var_names=var_names)

# Significance level in condition-selection step. If a list of levels is is
# provided or pc_alpha=None, the optimal pc_alpha is automatically chosen via
# model-selection.
pc_alpha = 0.2  # [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
selected_variables = range(N)  #[2] # [2]  # [2]

# Maximum time lag
tau_max = 3

# Optional minimum time lag in MCI step (in PC-step this is 1)
tau_min = 0

# Maximum cardinality of conditions in PC condition-selection step. The
# recommended default choice is None to leave it unrestricted.
max_conds_dim = None

# Maximum number of parents of X to condition on in MCI step, leave this to None
# to condition on all estimated parents.
max_conds_px = None

# Selected links may be used to restricted estimation to given links.
selected_links = None

# Alpha level for MCI tests (just used for printing since all p-values are 
# stored anyway)
alpha_level = 0.05

# Verbosity level. Note that slaves will ouput on top of each other.
verbosity = 0

# Chosen conditional independence test
cond_ind_test = ParCorr()  #confidence='analytic')

# Store results in file
file_name = os.path.expanduser('~') + '/test_results.dat'


#
#  Start of the script
#
if COMM.rank == 0:
    # Only the master node (rank=0) runs this
    if verbosity > -1:
        print("\n##\n## Running Parallelized Tigramite PC algorithm\n##"
              "\n\nParameters:")
        print("\nindependence test = %s" % cond_ind_test.measure
              + "\ntau_min = %d" % tau_min
              + "\ntau_max = %d" % tau_max
              + "\npc_alpha = %s" % pc_alpha
              + "\nmax_conds_dim = %s" % max_conds_dim)
        print("\n")

    # Split selected_variables into however many cores are available.
    splitted_jobs = split(selected_variables, COMM.size)
    if verbosity > -1:
        print("Splitted selected_variables = "), splitted_jobs
else:
    splitted_jobs = None


##
##  PC algo condition-selection step
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
    # Collect all results in dictionaries and send results to workers
    all_parents = {}
    pcmci_objects = {}
    for res in results:
        for (j, pcmci_of_j, parents_of_j) in res:
            all_parents[j] = parents_of_j[j]
            pcmci_objects[j] = pcmci_of_j

    if verbosity > -1:
        print("\n\n## Resulting condition sets:")
        for j in [var for var in all_parents.keys()]:
            pcmci_objects[j]._print_parents_single(j, all_parents[j],
                                    pcmci_objects[j].val_min[j], 
                                    pcmci_objects[j].p_max[j])

    if verbosity > -1:
        print("\n##\n## Running Parallelized Tigramite MCI algorithm\n##"
              "\n\nParameters:")

        print("\nindependence test = %s" % cond_ind_test.measure
              + "\ntau_min = %d" % tau_min
              + "\ntau_max = %d" % tau_max
              + "\nmax_conds_px = %s" % max_conds_px)
        
        print("Master node: Sending all_parents and pcmci_objects to workers.")
    
    for i in range(1, COMM.size):
        COMM.send((all_parents, pcmci_objects), dest=i)

else:
    if verbosity > -1:
        print("Slave node %d: Receiving all_parents and pcmci_objects..."
              "" % COMM.rank)
    (all_parents, pcmci_objects) = COMM.recv(source=0)


##
##   MCI step
##
# Scatter jobs again across cores.
scattered_jobs = COMM.scatter(splitted_jobs, root=0)

# Now each rank just does its jobs and collects everything in a results list.
results = []
for j in scattered_jobs:
    (j, results_in_j) = run_mci_parallel(j, pcmci_objects[j], all_parents)
    results.append((j, results_in_j))

# Gather results on rank 0.
results = MPI.COMM_WORLD.gather(results, root=0)


if COMM.rank == 0:
    # Collect all results in dictionaries
    # 
    if verbosity > -1:
        print("\nCollecting results...")
    all_results = {}
    for res in results:
        for (j, results_in_j) in res:
            for key in results_in_j.keys():
                if results_in_j[key] is None:  
                    all_results[key] = None
                else:
                    if key not in all_results.keys():
                        if key == 'p_matrix':
                            all_results[key] = numpy.ones(results_in_j[key].shape)
                        else:
                            all_results[key] = numpy.zeros(results_in_j[key].shape)
                        all_results[key][:,j,:] =  results_in_j[key][:,j,:]
                    else:
                        all_results[key][:,j,:] =  results_in_j[key][:,j,:]


    p_matrix=all_results['p_matrix']
    val_matrix=all_results['val_matrix']
    conf_matrix=all_results['conf_matrix']

    sig_links = (p_matrix <= alpha_level)

    if verbosity > -1:
        print("\n## Significant links at alpha = %s:" % alpha_level)
        for j in selected_variables:

            links = dict([((p[0], -p[1] ), numpy.abs(val_matrix[p[0], 
                            j, abs(p[1])]))
                          for p in zip(*numpy.where(sig_links[:, j, :]))])

            # Sort by value
            sorted_links = sorted(links, key=links.get, reverse=True)

            n_links = len(links)

            string = ""
            string = ("\n    Variable %s has %d "
                      "link(s):" % (var_names[j], n_links))
            for p in sorted_links:
                string += ("\n        (%s %d): pval = %.5f" %
                           (var_names[p[0]], p[1], 
                            p_matrix[p[0], j, abs(p[1])]))

                string += " | val = %.3f" % (
                    val_matrix[p[0], j, abs(p[1])])

                if conf_matrix is not None:
                    string += " | conf = (%.3f, %.3f)" % (
                        conf_matrix[p[0], j, abs(p[1])][0], 
                        conf_matrix[p[0], j, abs(p[1])][1])

            print (string)


    if verbosity > -1:
        print("Pickling to ", file_name)
    file = open(file_name, 'wb')
    pickle.dump(all_results, file, protocol=-1)        
    file.close()
    # PCMCI._print_significant_links(
    #        p_matrix=all_results['p_matrix'],
    #        val_matrix=all_results['val_matrix'],
    #        alpha_level=alpha_level,
    #        conf_matrix=all_results['conf_matrix'])
