"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0

import numpy
import os, sys

try: 
    from tigramite import tigramite_cython_code
except:
    print("Could not import cython code")

def generate(
    file_name,
    null_dist_samples=1000, 
    values_T=None,
    ):

    """Generates the null distribution of the distance correlation test.
    
    This script generates the nulldistribution of the distance correlation test
    statistic under the assumption of independent uniformly distributed
    variables x and y for different sample sizes T. The values are sorted and
    stored as  a npz-file which can be loaded within
    tigramite_independence_tests.py

    """

    if values_T is None:
        values_T = range(50, 251, 1) #+  \
               # range(250, 1001, 3) #+\
               # range(1000, 3001, 10) +\
               # range(3000, 10001, 50) +\
               # range(10000, 50001, 200)

    exact_dist = numpy.zeros((len(values_T), null_dist_samples))

    for iT, T in enumerate(values_T):

        print("Generating null distristbution for T = %d" % T)

        for i in range(null_dist_samples):

            x, y = numpy.random.rand(2, T)

            dc, val, dvx, dvy = tigramite_cython_code.dcov_all(x, y)

            exact_dist[iT, i] = val

        exact_dist[iT].sort()

    numpy.savez(file_name, exact_dist=exact_dist, T=numpy.array(values_T))


if __name__ == '__main__':

    # Generate null dist for python_version of ACE
    generate(null_dist_samples=1000, 
            values_T=[245],
            file_name = 'dcorr_nulldists.npz')