# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:00:42 2020

@author: Julian
"""
import numpy as np

from Functions.check_convex_order import *
from Functions.MOT_2dim import *
from Functions.marginal_discretization import *


def generate_samples(payoff,N=100000,nr_support = 20,variance = False, nr_of_variances = 20):
    x = []
    y = []
    def variance_payoff(x,y):
        return (y/x)**2-1
    while(len(y)<N):
        try:
            #######################
            # Log-Normal Distribution
            ######################
            mu1 = np.random.uniform(-2,2,1)
            sd1 = np.random.uniform(0,0.5,1)
            sd2 = np.random.uniform(1,2)*sd1
            
            values1, prob1 = u_discretization_lognormal_distribution(mu1-0.5*sd1**2,sd1,nr_support)
            values2, prob2 = u_discretization_lognormal_distribution(mu1-0.5*sd2**2,sd2,nr_support)
            
            mean1 = np.sum([v*p for v,p in zip(values1,prob1)])
            mean2 = np.sum([v*p for v,p in zip(values2,prob2)])
            values1 = np.array(values1)
            values2 = np.array(values2)
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            values1 = [v[0] for v in values1]
            values2 = [v[0] for v in values2]
    
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1+values2)
                        y.append([min_val,max_val])
                        if len(y) % 100 == 0:
                            print("{} Scenarios created.".format(len(y)))
                else:
                    # Compute prices
                    min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True)[0]
                    max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False)[0]
                    
                    x.append(values1+values2)
                    y.append([min_val,max_val])
                    if len(y) % 100 == 0:
                        print("{} Scenarios created.".format(len(y)))
                
            ########################
            # Uniform Distribution
            ###########################
            
            mu = np.random.uniform(10,20,1)
            a = np.random.uniform(0,5,1)
            b = np.random.uniform(a,a+5,1)
            
            values1,prob1 = u_discretization_uniform_distribution(mu-a,mu+a,nr_support)
            values2,prob2 = u_discretization_uniform_distribution(mu-b,mu+b,nr_support)
            
            mean1 = np.sum([v*p for v,p in zip(values1,prob1)])
            mean2 = np.sum([v*p for v,p in zip(values2,prob2)])
            values1 = np.array(values1)
            values2 = np.array(values2)
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            values1 = [v[0] for v in values1]
            values2 = [v[0] for v in values2]
    
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1+values2)
                        y.append([min_val,max_val])
                        if len(y) % 100 == 0:
                            print("{} Scenarios created.".format(len(y)))
                else:
                    # Compute prices
                    min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True)[0]
                    max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False)[0]
                    
                    x.append(values1+values2)
                    y.append([min_val,max_val])
                    if len(y) % 100 == 0:
                        print("{} Scenarios created.".format(len(y)))
                        
            # ########################
            # # Uniform Distribution: Continuous and Discrete
            # ###########################
            
            mu = np.random.uniform(5,10,1)
            a = np.random.uniform(0,5,1)
            
            values1, prob1 = u_discretization_uniform_distribution(mu-a,mu+a,nr_support)
            values2 = [mu-a]*int(nr_support/2)+[mu+a]*int(nr_support/2)
            values1 = np.array(values1)
            values2 = np.array(values2)
            prob2 = [1/(nr_support)]*nr_support
            
            mean1 = np.sum([v*p for v,p in zip(values1,prob1)])
            mean2 = np.sum([v*p for v,p in zip(values2,prob2)])
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            values1 = [v[0] for v in values1]
            values2 = [v[0] for v in values2]
            # Try to catch error, if not in convex_order.
            # Adjust mean:
            # Write Function to check whether in convex order?
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1+values2)
                        y.append([min_val,max_val])
                        if len(y) % 100 == 0:
                            print("{} Scenarios created.".format(len(y)))
                else:
                    # Compute prices
                    min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True)[0]
                    max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False)[0]
                    
                    x.append(values1+values2)
                    y.append([min_val,max_val])
                    if len(y) % 100 == 0:
                        print("{} Scenarios created.".format(len(y)))
    
                
                
            ########################
            # Triangular Distribution
            ###########################
            left_tri = np.random.uniform(0,5,1)
            mode = np.random.uniform(left_tri,left_tri+10,1)
            right_tri = np.random.uniform(mode,mode+10,1)
            
            def density(x):
                a = left_tri[0]
                b = right_tri[0]
                c = mode[0]
                value = (2*(x-a)/((b-a)*(c-a)))*(a<=x)*(x<c)
                value += (2/(b-a))*(x==c)
                value +=  (2*(b-x)/((b-a)*(b-c)))*(c<x)*(x<=b)
                return value
            
            values1, prob1 = u_discretization_uniform_distribution(mode-left_tri/2,mode+left_tri/2,nr_support)
            values2, prob2 = discretize_density(density,n = nr_support)
            # Try to catch error, if not in convex_order.
            # Adjust mean:
            mean1 = np.sum([v*p for v,p in zip(values1,prob1)])
            mean2 = np.sum([v*p for v,p in zip(values2,prob2)])
            values1 = np.array(values1)
            values2 = np.array(values2)
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            values1 = [v[0] for v in values1]
            values2 = [v for v in values2]
            # Write Function to check whether in convex order?
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1+values2)
                        y.append([min_val,max_val])
                        if len(y) % 100 == 0:
                            print("{} Scenarios created.".format(len(y)))
                else:
                    # Compute prices
                    min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True)[0]
                    max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False)[0]
                    
                    x.append(values1.tolist()+values2.tolist())
                    y.append([min_val,max_val])
                    if len(y) % 100 == 0:
                        print("{} Scenarios created.".format(len(y)))      
        except:
           print("Error")
        
    return x,y
    