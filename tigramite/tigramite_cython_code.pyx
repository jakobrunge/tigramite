import numpy
cimport numpy
import cython
from libc.math cimport abs

cdef inline double max(double a, double b): return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_neighbors_within_eps_cython(
            double[:,:] array, 
            int T, 
            int dim_x, 
            int dim_y, 
            double[:] epsarray,
            int k,
            int dim):

    cdef int[:] k_xz = numpy.zeros(T, dtype='int32')
    cdef int[:] k_yz = numpy.zeros(T, dtype='int32')
    cdef int[:] k_z = numpy.zeros(T, dtype='int32')

    cdef int i, j, d, kz, kxz, kyz
    cdef double dz, dy, dx, epsmax

    # Loop over time points
    for i in range(T):

        # Epsilon of k-th nearest neighbor in joint space
        epsmax = epsarray[i]

        # Count neighbors within epsmax in subspaces, since the reference
        # point is included, all neighbors are at least 1
        kz = 0
        kxz = 0
        kyz = 0
        for j in range(T):

            # Z-subspace, if empty, dz stays 0
            dz = 0.
            for d in range(dim_x+dim_y, dim):
                dz = max( abs(array[d, i] - array[d, j]), dz)
 
            # For no conditions, kz is counted up to T
            if (dz < epsmax):
                kz += 1

                # Only now check Y- and X-subspaces

                # Y-subspace, the loop is only entered for dim_y > 1
                dy = abs(array[dim_x, i] - array[dim_x, j])
                for d in range(dim_x+1, dim_x+dim_y):
                    dy = max( abs(array[d, i] - array[d, j]), dy)
                
                if (dy < epsmax):
                    kyz += 1

                # X-subspace, the loop is only entered for dim_x > 1
                dx = abs(array[0, i] - array[0, j])
                for d in range(1, dim_x):
                    dx = max( abs(array[d, i] - array[d, j]), dx)
   
                if (dx < epsmax):
                    kxz += 1
               
        # Write to arrays
        k_xz[i] = kxz
        k_yz[i] = kyz
        k_z[i] = kz


    return numpy.asarray(k_xz), numpy.asarray(k_yz), numpy.asarray(k_z)


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_patterns_cython(
    double[:,:] array,
    int[:,:] array_mask,
    int[:,:] patt,
    int[:,:] patt_mask,
    double[:,:] weights,
    int dim,
    int step,
    int[:] fac,
    int N, 
    int T):

    cdef int n, t, k, i, j, p, tau, start, mask
    cdef double ave, var
    cdef double[:] v = numpy.zeros(dim, dtype='float')

    start = step*(dim-1)
    for n in range(0, N):
        for t in range(start, T):
            mask = 1
            ave = 0.
            for k in range(0, dim):
                tau = k*step
                v[k] = array[t - tau, n]
                ave += v[k]
                mask *= array_mask[t - tau, n]
            ave /= dim
            var = 0.
            for k in range(0, dim):
                var += (v[k] - ave)**2
            var /= dim
            weights[t-start, n] = var
            if( v[0] < v[1]):
                p = 1
            else:
                p = 0
            for i in range(2, dim):
                for j in range(0, i):
                    if( v[j] < v[i]):
                        p += fac[i]
            patt[t-start, n] = p
            patt_mask[t-start, n] = mask

    return (patt, patt_mask, weights)


cdef inline bint isvalueinarray(
    int val,
    int[:] arr,
    int size):

    cdef int i

    for i in range(size):
        if (arr[i] == val):
            return True

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_restricted_permutation_cython(
    int T,
    int shuffle_neighbors,
    int[:, :] neighbors,
    int[:] order
    ):

    cdef int[:] restricted_permutation = numpy.zeros(T, dtype='int32')
    cdef int[:] used = numpy.zeros(T, dtype='int32')
    #cdef int[:] perm = numpy.zeros(shuffle_neighbors, dtype='int32')

    cdef int i, index, count, use

    # Shuffle neighbor indices for each sample index
    for i in range(T):
        numpy.random.shuffle(neighbors[i])

    for i in range(T):

        index = order[i];
        count = 0

        use = neighbors[index, count]
        while(isvalueinarray(use, used, i) and (count < shuffle_neighbors - 1)):
            count += 1
            use = neighbors[index, count]

        restricted_permutation[index] = use
        
        used[i] = use

    return numpy.asarray(restricted_permutation)


# Copyright (c) 2012, Florian Finkernagel. All right reserved.
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:

##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.

##     * Redistributions in binary form must reproduce the above
##       copyright notice, this list of conditions and the following
##       disclaimer in the documentation and/or other materials provided
##       with the distribution.

##     * Neither the name of Florian Finkernagel nor the names of its
##       contributors may be used to endorse or promote products derived
##       from this software without specific prior written permission.

## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
"""An implementation of Distance Correlation (see http://en.wikipedia.org/wiki/Distance_correlation )
that is not quadratic in space requirements (only in runtime)."""


import unittest
import timeit
import sys
import random
import numpy as np
cimport numpy as np
import cython

ctypedef np.double_t DTYPE_t

#purely for speed reasons
cdef inline double myabs(double a) : return a if a >= 0. else -1 * a

def dcov_all(x, y):
    'Calculate distance covariance, distance correlation, distance variance of x sample and distance variance of y sample'
    x = np.array(x, dtype=np.double)
    y = np.array(y, dtype=np.double)
    dnx = D_N(x)
    dny = D_N(y)

    denom = float(dnx.dim * dnx.dim)
    dc = dnx.product_sum(dny) / denom
    dvx = dnx.squared_sum() / denom
    dvy = dny.squared_sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dc, dr, dvx, dvy


class D_N: 
    """Inner helper of dcov_all. Cache different means that are required for calculating 
    the matrix members on the fly"""

    def __init__(self, x):
        self.x = np.array(x)
        self.dim = x.shape[0]
        self.calculate_means()

    @cython.boundscheck(False)
    def calculate_means(self):
        cdef int dim = self.dim
        cdef DTYPE_t value
        cdef DTYPE_t sum_total = 0
        cdef np.ndarray[DTYPE_t, ndim=1] sum_0 = np.zeros(dim, dtype=np.double)
        cdef np.ndarray[DTYPE_t, ndim=1] sum_1 = np.zeros(dim, dtype=np.double)
        cdef np.ndarray[DTYPE_t, ndim=1] x = self.x
        cdef unsigned int ii
        cdef unsigned int jj
        for ii in range(dim):
            for jj in range(dim):
                value = myabs(x[jj] - x[ii])
                sum_total += value
                sum_1[jj] += value
                sum_0[ii] += value
        self.mean = sum_total / (self.dim**2)
        self.mean_0 = sum_0 / (self.dim)
        self.mean_1 = sum_1 / (self.dim)
        return

    @cython.boundscheck(False)
    def squared_sum(self):
        cdef np.ndarray[DTYPE_t, ndim=1] mean_0 = self.mean_0
        cdef np.ndarray[DTYPE_t, ndim=1] mean_1 = self.mean_1
        cdef DTYPE_t mean = self.mean
        cdef DTYPE_t squared_sum = 0
        cdef DTYPE_t dist
        cdef DTYPE_t d
        cdef np.ndarray[DTYPE_t, ndim=1] x = self.x
        cdef unsigned int dim = self.dim
        cdef unsigned int ii
        cdef unsigned int jj
        for ii in range(dim):
            for jj in range(dim): 
                dist = myabs(x[jj] - x[ii])
                d = dist - mean_0[ii] - mean_1[jj] + mean
                squared_sum += d * d
        return squared_sum
   
    @cython.boundscheck(False)
    def product_sum(self, other):
        cdef np.ndarray[DTYPE_t, ndim=1] mean_0_here = self.mean_0
        cdef np.ndarray[DTYPE_t, ndim=1] mean_1_here = self.mean_1
        cdef DTYPE_t mean_here = self.mean
        cdef np.ndarray[DTYPE_t, ndim=1] mean_0_there = other.mean_0
        cdef np.ndarray[DTYPE_t, ndim=1] mean_1_there = other.mean_1
        cdef DTYPE_t mean_there = other.mean
        cdef DTYPE_t d_here
        cdef DTYPE_t d_there
        cdef DTYPE_t product_sum = 0
        cdef np.ndarray[DTYPE_t, ndim=1] x = self.x
        cdef np.ndarray[DTYPE_t, ndim=1] y = other.x

        cdef unsigned int dim = self.dim
        cdef unsigned int ii
        cdef unsigned int jj
        for ii in range(dim):
            for jj in range(dim): 
                d_here = myabs(x[jj] - x[ii]) - mean_0_here[ii] - mean_1_here[jj] + mean_here
                d_there = myabs(y[jj] - y[ii]) - mean_0_there[ii] - mean_1_there[jj] + mean_there
                product_sum += d_here * d_there
        return product_sum
