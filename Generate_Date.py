# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:42:57 2020

@author: Julian
"""


import numpy as np
import pandas as pd
import os

# User defined Functions
from Functions.check_convex_order import *
from Functions.marginal_discretization import *
from Functions.generate_samples_2 import *
from Functions.MOT_2dim import *


# Examplary Payoff Function: |x-y|
def payoff(x,y):
    return abs(y-x)

def generate_and_save_data(payoff,N =1000,nr_support = 20, variance = False, nr_of_variances = 20):

    # Compute Lower Price Bounds and generate Samples
    if variance == False:
        x,y = generate_samples(payoff,N,nr_support)
        # Save the sample in a csv File
        
        file_dir = os.path.dirname(os.path.abspath(__file__))
        csv_folder = "csv"
        file_path = os.path.join(file_dir, csv_folder, 'generated_marginals.csv')
        df = [yy + xx for xx,yy in zip(x,y)]
        header_string = ["Price_min"]+["Price_max"]+["1stMarginal("+str(i+1)+")" for i in range(nr_support)]+["2ndMarginal("+str(i+1)+")" for i in range(nr_support)]
        df = pd.DataFrame(df, columns = header_string)
        df.to_csv(file_path, index = False, header=True)
        print("Sample created and saved")
    else:
        x,y = generate_samples(payoff,N,nr_support,variance, nr_of_variances)        # Save the sample in a csv File
        file_dir = os.path.dirname(os.path.abspath(__file__))
        csv_folder = "csv"
        file_path = os.path.join(file_dir, csv_folder, 'generated_marginals_variance_20pts.csv')
        df = [yy + xx for xx,yy in zip(x,y)]
        header_string = ["Price_min"]+["Price_max"]+["Variance"]+["1stMarginal("+str(i+1)+")" for i in range(nr_support)]+["2ndMarginal("+str(i+1)+")" for i in range(nr_support)]
        df = pd.DataFrame(df, columns = header_string)
        df.to_csv(file_path, index = False, header=True)
        print("Sample created and saved")
    
    
generate_and_save_data(payoff,N =500000,nr_support = 20, variance = True, nr_of_variances = 20)
