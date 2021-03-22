# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:47:56 2020

@author: Julian
"""
from Functions.check_convex_order import *
from Functions.MOT_2dim import *
import numpy as np


def generate_samples(payoff,N=1000,nr_support = 20,variance = False, nr_of_variances = 20):
    x = []
    y = []
    def variance_payoff(x,y):
        return (y/x)**2-1
    while(len(y)<N):
        try:
            #######################
            # Normal Distribution
            ######################
            mu1 = np.random.uniform(-5,5,1)
            sd1 = np.random.uniform(0,5,1)
            sd2 = np.random.uniform(1,5)*sd1
            
            values1 = np.random.normal(mu1,sd1,nr_support)
            values1 = np.sort(values1)
            prob1 = [1/(nr_support)]*nr_support
            values2 = np.random.normal(mu1,sd2,nr_support)
            values2 = np.sort(values2)
            prob2 = [1/(nr_support)]*nr_support
            # Try to catch error, if not in convex_order.
            # Adjust mean:
            mean1 = np.sum(values1*prob1)
            mean2 = np.sum(values2*prob2)
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            # Write Function to check whether in convex order?
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1.tolist()+values2.tolist())
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
                
            ########################
            # Uniform Distribution
            ###########################
            mu2 = np.random.uniform(-5,5,1)
            bound1 = np.random.uniform(0,5,1)
            bound2 = np.random.uniform(bound1,bound1+10,1)
            
            values1 = np.random.uniform(mu2-bound1,mu2+bound1,nr_support)
            values1 = np.sort(values1)
            prob1 = [1/(nr_support)]*nr_support
            values2 = np.random.uniform(mu2-bound2,mu2+bound2,nr_support)
            values2 = np.sort(values2)
            prob2 = [1/(nr_support)]*nr_support
            # Try to catch error, if not in convex_order.
            # Adjust mean:
            mean1 = np.sum(values1*prob1)
            mean2 = np.sum(values2*prob2)
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            # Write Function to check whether in convex order?
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1.tolist()+values2.tolist())
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

                
                
            ########################
            # Triangular Distribution
            ###########################
            left_tri = np.random.uniform(-5,5,1)
            mode = np.random.uniform(left_tri,left_tri+10,1)
            right_tri = np.random.uniform(mode,mode+10,1)
            
            values1 = np.random.uniform(mode-left_tri/2,mode+left_tri/2,nr_support)
            values1 = np.sort(values1)
            prob1 = [1/(nr_support)]*nr_support
            values2 = np.random.triangular(left_tri,mode,right_tri,nr_support)
            values2 = np.sort(values2)
            prob2 = [1/(nr_support)]*nr_support
            # Try to catch error, if not in convex_order.
            # Adjust mean:
            mean1 = np.sum(values1*prob1)
            mean2 = np.sum(values2*prob2)
            values1 = values1 + 0.5*(mean2-mean1)
            values2 = values2 + 0.5*(mean1-mean2)
            # Write Function to check whether in convex order?
            if check_convex_order(values1,prob1,values2,prob2):
                if variance:
                    min_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=True)[0]
                    max_var = mot_2dim(values1,prob1,values2,prob2,variance_payoff,minimize=False)[0]
                    for var in np.linspace(min_var,max_var,nr_of_variances):
                        min_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=True,variance = var)[0]
                        max_val = mot_2dim(values1,prob1,values2,prob2,payoff,minimize=False,variance = var)[0]
                        x.append([(var-min_var)/(max_var-min_var)]+values1.tolist()+values2.tolist())
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
    