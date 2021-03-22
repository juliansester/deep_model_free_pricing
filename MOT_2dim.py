"""
Martingale Transport, 2 Dimensional 

@author: Julian Sester
"""


def mot_2dim(values1,prob1,values2,prob2,func,minimize=True,method ="gurobi",variance = False):
    # Importing necessary packages
    import numpy as np
    from scipy.optimize import linprog
    
    # make numpy arrays
    values1 = np.array(values1)
    values2 = np.array(values2)
    prob1 = np.array(prob1)
    prob2 = np.array(prob2)
    
    # Define necessary Variables: Length of the vectors
    n1 = len(values1)
    n2 = len(values2)
    
    # Print Error if Probability Vectors have different length than Value Vectors
    if ((n1 != len(prob1)) | (n2 != len(prob2))):
        print('Length of Probability Vectors and Value Vectors do not coincide!')
        return()
    
    # Scaling probabilities to 1
    prob1 = prob1 / np.sum(prob1)
    prob2 = prob2 / np.sum(prob2)
    
    # Scaling to the same mean
    mean1 = np.sum(values1*prob1)
    mean2 = np.sum(values2*prob2)
    values1 = values1 + 0.5*(mean2-mean1)
    values2 = values2 + 0.5*(mean1-mean2)
    
    if method != "gurobi":
        # Define R.H.S vector
        r = np.concatenate((values1*prob1,prob1,prob2))
        
        # L.H.S. Vector / Setting the size
        A = np.zeros((2*n1+n2,n1*n2))
        
        # Martingale Conditions
        for i in range(n1):
            a = np.zeros((n1,n2))
            for j in range(n2):
                a[i,j] = values2[j]
            A[i,:] = np.reshape(a,n1*n2)
                
        # Marginal Conditions
        for i in range(n1):
            a = np.zeros((n1,n2))
            a[i,:] = 1
            A[n1+i,:] = np.reshape(a,n1*n2)
        
        for i in range(n2):
            a = np.zeros((n1,n2))
            a[:,i] = 1
            A[2*n1+i,:] = np.reshape(a,n1*n2)
    elif method == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model("m")
        m.setParam( 'OutputFlag', False )
        x = m.addMVar(shape=np.int(n1*n2),lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="x")
        # Martingale Conditions
        for i in range(n1):
            a = np.zeros((n1,n2))
            for j in range(n2):
                a[i,j] = values2[j]
            m.addConstr(np.reshape(a,n1*n2) @ x == np.array(values1[i]*prob1[i]))         
        # Marginal Conditions
        for i in range(n1):
            a = np.zeros((n1,n2))
            a[i,:] = 1
            m.addConstr(np.reshape(a,n1*n2) @ x == np.array(prob1[i]))         
        for i in range(n2):
            a = np.zeros((n1,n2))
            a[:,i] = 1
            m.addConstr(np.reshape(a,n1*n2) @ x == np.array(prob2[i]))
            
        if variance:
            a = np.zeros((n1,n2))
            for i in range(n1):
                for j in range(n2):
                    a[i,j] = (values2[j]/values1[i])**2-1
            m.addConstr(np.reshape(a,n1*n2) @ x == variance)
       
    # Define Payoff/Cost Array
    costs = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            costs[i][j] = func(values1[i],values2[j])
    costs = np.reshape(costs,n1*n2)
    # Solve Linear System
    if method != "gurobi":
        if(minimize == True):
            res = linprog(costs, A_eq=A, b_eq=r, bounds=(0,1),  options={"disp": False})
        else:
            res = linprog(-costs, A_eq=A, b_eq=r, bounds=(0,1),  options={"disp": False})
        # print out optimal q and optimal price
        q = res["x"]
        if(minimize == True):
            price = res["fun"]
        else:
           price = -res["fun"]
    elif method == "gurobi":
        if minimize == True:
            m.setObjective(costs @ x, GRB.MINIMIZE)
        elif minimize == False:
            m.setObjective(costs @ x, GRB.MAXIMIZE)
        m.optimize()
        price = m.objVal
        q = m
    
    
    return price, q

# N = 100
# v1 = np.linspace(-1,1,N)
# p1 = np.repeat(1/N,N)
# v2 = np.linspace(-2,2,N)
# p2 = np.repeat(1/N,N)
# def payoff(x,y):
#     return abs(x-y)
    

# print(mot_2dim(v1,p1,v2,p2,payoff,minimize=True,method ="gurobi",variance = False))