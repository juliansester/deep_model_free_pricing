# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:26:16 2020

@author: Julian
"""
import numpy as np

def check_convex_order(values1,prob1,values2,prob2):
    """
    Function to check whether two distributions are in convex order.

    Parameters
    ----------
    values1 : numpy array or list
        of values with positive support of the first marginal
    prob1 : numpy array or list
        of values with probabilites of the first marginal.
    values2 : numpy array or list
        of values with positive support of the second marginal.
    prob2 : numpy array or list
        of values with probabilites of the second marginal.

    Returns
    -------
    Bool
        True if in convex order, False if not.

    """
    def call_payoff(x,K):
        return max(x-K,0)
    values = np.concatenate((values1,values2))
    #First Check that expectations are the same
    exp1 = sum([p*v for p,v in zip(prob1,values1)])
    exp2 = sum([p*v for p,v in zip(prob2,values2)])
    if exp1 != exp2:
        return False
    #Second check the expectations w.r.t. call function
    for strike in values:
        exp1 = sum([p*call_payoff(v,strike) for p,v in zip(prob1,values1)])
        exp2 = sum([p*call_payoff(v,strike) for p,v in zip(prob2,values2)])
        if exp2 < exp1:
            return False
    return True


