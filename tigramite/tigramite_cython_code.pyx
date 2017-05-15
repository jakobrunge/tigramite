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