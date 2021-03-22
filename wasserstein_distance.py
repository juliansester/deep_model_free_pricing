# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:32:31 2020

@author: Julian
"""

from scipy.stats import wasserstein_distance
def wasserstein(p,q):   
    n = len(p)
    m = len(q)
    n_half = int(n/2)
    m_half = int(m/2)
    return wasserstein_distance(p[:n_half],q[:m_half])+wasserstein_distance(p[n_half:],q[m_half:])
    