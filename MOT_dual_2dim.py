import numpy as np
import gurobipy as gp
from gurobipy import GRB

def dual_2dim_prices(strikes1,prices1_bid,prices1_ask,strikes2,prices2_bid,prices2_ask,func,s0=100,lower_bound=True,stepsize = 1,
                      from_discrete= 0, to_discrete = 2, kappa = 0.1):
    
    # Conversion in numpy arrays
    prices1_bid = np.array(prices1_bid)
    prices1_ask = np.array(prices1_ask)
    prices2_bid = np.array(prices2_bid)
    prices2_ask = np.array(prices2_ask)    
    strikes1 = np.array(strikes1)
    strikes2 = np.array(strikes2)
    
    n1 = len(strikes1)
    n2 = len(strikes2)
    
    # Setting the Grid where we evaluate the function
    values1 = np.arange(from_discrete*s0,to_discrete*s0,stepsize)
    values2 = np.arange(from_discrete*s0,to_discrete*s0,stepsize)
    dp = len(values1)
    # Defining the Cost Function    
    costs = np.zeros((dp,dp))
    for i in range(dp):
        for j in range(dp):
            costs[i,j] = func(values1[i],values2[j])
            if lower_bound:
                costs[i,j] = -costs[i,j]
    # Define the gurobi model
    m = gp.Model("m")
    m.setParam( 'OutputFlag', True)
    x = m.addMVar(shape=1+2*(n1+n2)+1+dp,
                  lb = np.concatenate((np.repeat(-GRB.INFINITY,1),
                                      np.repeat(0,2*(n1+n2)),
                                      np.repeat(-GRB.INFINITY,1),
                                      np.repeat(-GRB.INFINITY,dp))),
                  ub = np.repeat(GRB.INFINITY,1+2*(n1+n2)+1+dp),            
                                      vtype=GRB.CONTINUOUS)
                  
    for i in range(dp):
        for j in range(dp):          
            val1_ask = [max(values1[i]-strikes1[l],0)-prices1_ask[l] for l in range(n1)]
            val1_bid = [-max(values1[i]-strikes1[l],0)+prices1_bid[l] for l in range(n1)]
            val2_ask = [max(values2[j]-strikes2[l],0)-prices2_ask[l] for l in range(n2)]
            val2_bid = [-max(values2[j]-strikes2[l],0)+prices2_bid[l] for l in range(n2)]
            val_dynamic_0 = [values1[i]-s0]
            val_dynamic_1 = [(values2[j]-values1[k])*(k==i) for k in range(dp)]
            lhs =  np.concatenate(([1],val1_ask,val1_bid,val2_ask,val2_bid,val_dynamic_0,val_dynamic_1))
            transaction_costs = kappa
            m.addConstr(lhs @ x >=  costs[i,j]+transaction_costs)
                    
            
    #print(np.unique(A, return_counts=True))
    # Solve Linear System
    #####################
    objective = np.concatenate((np.array([1]),np.zeros(1+2*(n1+n2)+dp)))
 
#    objective = np.concatenate((np.array([1]),prices1_ask,-prices1_bid,prices2_ask,
#                                -prices2_bid,np.zeros(1+dp)))


    m.setObjective(objective @ x, GRB.MINIMIZE)
    m.optimize()
    if lower_bound: 
        param = [-val for val in m.x]
        price = -m.ObjVal
    else:
        param = m.x
        price = m.ObjVal
    a = param[0]
    opt1_buy = param[1:(n1+1)]
    opt1_sell = param[(n1+1):(2*n1+1)]
    opt2_buy = param[(2*n1+1):(2*n1+n2+1)]
    opt2_sell = param[(2*n1+n2+1):(2*n1+2*n2+1)]
    Delta_0 = param[(2*n1+2*n2+1):(2*n1+2*n2+2)]
    Delta_1 = param[(2*n1+2*n2+2):(2*n1+2*n2+2+dp)]
    print("a: {}".format(a))
    print("Option1-Buy: {}".format(opt1_buy))
    print("Option1-Sell: {}".format(opt1_sell))
    print("Option2-Buy: {}".format(opt2_buy))
    print("Option2-Sell: {}".format(opt2_sell))
    print("Delta0: {}".format(Delta_0))
    print("Delta1: {}".format(Delta_1))
    return price, m, param


strikes1 = np.array([100.])
prices1_bid = np.array([5.])
prices1_ask= np.array([8.])
strikes2 = np.array([120.])
prices2_bid = np.array([3.])
prices2_ask= np.array([5.])
s0=100
def func(x,y):
    return 5
s0 = 100
dual_2dim_prices(strikes1,prices1_bid,prices1_ask,strikes2,prices2_bid,prices2_ask,
                        func,s0,lower_bound=True,stepsize = 10,
                      from_discrete= 0, to_discrete = 2, kappa = 0)