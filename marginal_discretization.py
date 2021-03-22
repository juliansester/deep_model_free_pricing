# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:29:11 2020

@author: Julian
"""


import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.interpolate




def discretize_density(density,min=1e-5, max=20,N=10000,n=20):
    vv = np.linspace(min,max,N)
    epsilon = 1/n
    #Distribution Function:
    distr = np.cumsum(density(vv))/(np.cumsum(density(vv)))[-1]
    ordering = np.argsort(vv)
    quantile = scipy.interpolate.interp1d(distr, vv, bounds_error=False, fill_value='extrapolate')
    support = np.linspace(1e-5,1,n+1)
    v=[]
    for i in range(n):
        v.append(0.5*(quantile(support[i+1])+quantile(support[i])))
    p = [1/n]*n
    return v,p


def u_discretization_normal_distribution(mu,sigma,n,epsilon = 10e-5):
    grid = np.linspace(epsilon,1-epsilon,n+1)
    v=[]
    for i in range(n):
        v.append(0.5*(norm.ppf(grid[i+1],loc=mu,scale=sigma)+norm.ppf(grid[i],loc=mu,scale=sigma)))
    p = [1/n]*n
    return v,p

def u_discretization_lognormal_distribution(mu,sigma,n,epsilon = 10e-2):
    grid = np.linspace(epsilon,1-epsilon,n+1)
    v=[]
    for i in range(n):
        v.append(0.5*(np.exp(mu+sigma*norm.ppf(grid[i+1],loc=0,scale=1))+np.exp(mu+sigma*norm.ppf(grid[i],loc=0,scale=1))))
    p = [1/n]*n
    return v,p

def u_discretization_uniform_distribution(a,b,n):
    grid = np.linspace(a,b,n+1)
    v=[]
    for i in range(n):
        v.append((1/n)*((i+0.5)*(b-a))+a)
    p = [1/n]*n
    return v,p

def density1(x):
    return (0.5)*(x>= -1)*(x<=1)
def density2(x):
    return (1/3)*(5/8)*(x>= -2)*(x< -1)+ (1/3)*(3/8)*(x>= 1)*(x<= 4)