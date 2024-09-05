# cython: language_level=3
from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_lapack cimport dposv
 
@boundscheck(False)
@wraparound(False)
def PostGauss(cnp.ndarray[cnp.float64_t, ndim=2] G,
              cnp.ndarray[cnp.float64_t, ndim=1] d,
              cnp.ndarray[cnp.float64_t, ndim=2] Sigma_e,
              cnp.ndarray[cnp.float64_t, ndim=1] mu_m,
              cnp.ndarray[cnp.float64_t, ndim=2] Sigma_m):
   
    # Check that the input matrices are in Fortran order for optimal BLAS usage
    if not G.flags['F_CONTIGUOUS']:
        G = np.asfortranarray(G)
    if not Sigma_e.flags['F_CONTIGUOUS']:
        Sigma_e = np.asfortranarray(Sigma_e)
    if not Sigma_m.flags['F_CONTIGUOUS']:
        Sigma_m = np.asfortranarray(Sigma_m)
 
    # Compute Sigma_d
    Sigma_d = G.dot(Sigma_m).dot(G.T) + Sigma_e
 
    # Compute the inverse of Sigma_d
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Sigma_d_inv = np.linalg.inv(Sigma_d)
 
    # Compute m
    m = Sigma_d_inv.dot(d - G.dot(mu_m))
 
    # Compute posterior mean
    mu_post = mu_m + Sigma_m.dot(G.T).dot(m)
 
    # Compute N
    cdef cnp.ndarray[cnp.float64_t, ndim=2] N = np.dot(np.dot(Sigma_d, G), Sigma_m)
    # Compute posterior covariance
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Sigma_post = Sigma_m - np.dot(np.dot(Sigma_m, G.T), N)
    return mu_post, Sigma_post