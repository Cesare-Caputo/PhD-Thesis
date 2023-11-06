# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:48:37 2020

@author: cesa_
"""
from stochasticdp import StochasticDP
import numpy as np
import pandas as pd
import copy
from garage_DP_helper import *
# from garage_demand import demand_stochastic_less, demand_stochastic, demand_stochastic_series, demand_static
from garage_cost import Exp_cost, opex
import math
import numpy as np
import pandas as pd
from garage_demand import *
from matplotlib import pyplot as plt
from scipy.optimize import *
# Parameters
T=20 #years
cc = 16000# Construction cost per parking space
cl = 3600000# Annual leasing land cost
#p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cr = 2000# Operating cost per parking space
#ct = []# Total construction cost
gc = 0.10# Growth in construction cost per floor above two floors
n0 = 200# Initial number of parking space per floor
p = 10000# Price per parking space
r = 0.12# Discount rate
fmin = 2# Minimum number of floors built
fmax = 8# Maximum number of floors built

D_1 = 750 # Projected year 1 demand
D_10 = 750 #additional demand by year 10
D_F= 250 # additional demand after year 10
T = 20 # project duration in years
alpha = D_10 + D_F; # Parameter for demand model showing difference between initial and final demand values
beta = -math.log(D_F/alpha)/(T/2 - 1) # Parameter for demand model showing growth speed of demand curve



DV_D = np.array([1, 1, 1, 0, 2, 2, 4])

# Design variables
#flex = []# Use the flexible design: 1 = "yes" 0 = "no"
#a1_4 = []# Expansion allowed in years 1 to 4: 1 = "yes" 0 = "no"
#a9_12 = []# Expansion allowed in years 9 to 12: 1 = "yes" 0 = "no"
#17_20 = []# Expansion allowed in years 17 to 20: 1 = "yes" 0 = "no"
#dr = []# Expansion rule: previous years with demand > capacity before expansion
#ft = []# Expansion rule: number of floors expanded by at year t
#f0 = []# Number of initial floors at year 0
#k = []
#E_cost = []

params = (20,1600,360000,2000, .1, 200, 10000, .12, 2, 8)

n_scenarios = 100

def cc_initial(f0):
    if f0 > 2:
        cc_start = cc * n0 * ((((1+gc)**(f0-1) - (1+gc))/gc)) + (2*n0*cc)
    else : 
        cc_start= f0*n0*cc
    return cc_start

def demand_series(T):
    years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
    demand_projections = pd.Series(index=years)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    for i in range(0,T+1): #initializing all ks to initial capacity
        demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s, rDfs)
    return demand_projections
    


def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df    

scenario_df = scenario_generator(n_scenarios)

 
    
a=NPV_garage_GA_DR(DV_D) 
bounds = [(1,1),(1,1), (0, 1), (0, 0), (0, 3), (0, 3), (0, 4)]
result = differential_evolution(func = NPV_garage_GA_DR_scenarios, bounds = bounds , maxiter=100)

Ga_opt_dv = np.array([ (result.x[0]) , (result.x[1]) , (result.x[2]) , (result.x[3]) , (result.x[4]) , (result.x[5]), (result.x[6])])
NPV_GA_opt, model_ga_opt = NPV_garage_GA_DR(Ga_opt_dv)
NPV_GA_opt_dv = NPV_garage_GA_DR(DV_D)
print("NPV for optimal policy with GA_DR Million $", NPV_GA_opt*(10**-6))
print("NPV for optimal prefound policy with GA_DR Million $", NPV_GA_opt_dv*(10**-6))
