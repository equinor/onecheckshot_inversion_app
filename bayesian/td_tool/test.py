
import sys
import os
sys.path.append(os.getcwd())
import bayesian.td_tool.post_gauss.PostGauss as pg
import numpy as np
 
import timeit
 
 
def PostGauss_py(G_py, d_py, Sigma_e_py, mu_m_py, Sigma_m_py):
    Sigma_d_py = G_py.dot(Sigma_m_py).dot(G_py.T) + Sigma_e_py
 
    m_py = np.linalg.inv(Sigma_d_py).dot(d_py - G_py.dot(mu_m_py))
 
    mu_post_py = mu_m_py + Sigma_m_py.dot(G_py.T).dot(m_py)
 
    N_py = np.linalg.inv(Sigma_d_py).dot(G_py.dot(Sigma_m_py))
 
    Sigma_post_py = Sigma_m_py - Sigma_m_py.dot(G_py.T).dot(N_py)
 
    return mu_post_py, Sigma_post_py
 
 
setup_code = """
from __main__ import pg, PostGauss_py, G, d, Sigma_e, mu_m, Sigma_m
"""
 
cpp_statement = "mu_post, Sigma_post = pg.PostGauss(G, d, Sigma_e, mu_m, Sigma_m)"
py_statement = "py_mu_post, py_Sigma_post = PostGauss_py(G, d, Sigma_e, mu_m, Sigma_m)"
 
np.random.seed(0)
 
for i in range(0, 100):
    G = np.random.rand(50, 50) * 1e12
    d = np.random.rand(50) * 1.5e12
    Sigma_e = np.eye(50) * 1e11
    mu_m = np.zeros(50)
    Sigma_m = np.eye(50) * 1e12
 
    py_time = timeit.timeit(stmt=py_statement, setup=setup_code, number=1000)
 
    cpp_time = timeit.timeit(stmt=cpp_statement, setup=setup_code, number=1000)
 
    print(f"Iteration {i+1}:")
    print(f"Python implementation time: {py_time} seconds")
    print(f"C++ implementation time: {cpp_time} seconds")