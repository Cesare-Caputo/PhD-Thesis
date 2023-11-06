# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:48:37 2020

@author: cesa_
"""
import numpy as np
import pandas as pd
import copy
# from garage_demand import demand_stochastic_less, demand_stochastic, demand_stochastic_series, demand_static
from garage_cost import Exp_cost, opex
import math
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
fmax = 9# Maximum number of floors built

D_1 = 750 # Projected year 1 demand
D_10 = 750 #additional demand by year 10
D_F= 250 # additional demand after year 10
T = 20 # project duration in years
alpha = D_10 + D_F; # Parameter for demand model showing difference between initial and final demand values
beta = -math.log(D_F/alpha)/(T/2 - 1) # Parameter for demand model showing growth speed of demand curve

import math
import numpy as np
import pandas as pd
from garage_demand import demand_static, demand_stochastic, demand_stochastic_less
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# Parameters
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
    

def NPV_garage_inflex(f0,  demand = ''):
        k_inflex = f0*n0 
        cc_start = cc_initial(f0)
        NPV = 0
        years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
        k = pd.Series(index=years)
        revenue = pd.Series(index=years)
        CF = pd.Series(index=years)
        demand_projections = pd.Series(index=years)
        opex = pd.Series(index=years)
        fixed_costs = pd.Series(index=years)
        #set random demand curve profile
        rD0s = np.random.random_sample() # Realised demand in year 0
        rD10s = np.random.random_sample() # Realised additional demand by year 10
        rDfs =np.random.random_sample()# Realised additional demand after year 10
                            
        for i in range(0,T+1): #initializing all ks to initial capacity
            k[i] = f0*n0
            k[0] = 0
            if demand == 'stochastic':
                demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s,rDfs)
            else:
                demand_projections[i] = demand_static(i)
       
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            elif i==0:
                fixed_costs[i] = cc_start + cl
                revenue[i] = 0
                opex[i] = 0
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
            
        for i in range(0,T+1) :
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] 
            NPV += CF[i]/((1+r)**i)
        return NPV

def NPV_garage_inflex_scenarios(f0, demand_scenario,   demand = ''):
        k_inflex = f0*n0 
        cc_start = cc_initial(f0)
        NPV = 0
        years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
        k = pd.Series(index=years)
        revenue = pd.Series(index=years)
        CF = pd.Series(index=years)
        demand_projections = pd.Series(index=years)
        opex = pd.Series(index=years)
        fixed_costs = pd.Series(index=years)
        #set random demand curve profile
        rD0s = np.random.random_sample() # Realised demand in year 0
        rD10s = np.random.random_sample() # Realised additional demand by year 10
        rDfs =np.random.random_sample()# Realised additional demand after year 10
                            
        for i in range(0,T+1): #initializing all ks to initial capacity
            k[i] = f0*n0
            k[0] = 0
            if demand == 'stochastic':
                demand_projections = demand_scenario
            else:
                demand_projections[i] = demand_static(i)
       
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            elif i==0:
                fixed_costs[i] = cc_start + cl
                revenue[i] = 0
                opex[i] = 0
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
            
        for i in range(0,T+1) :
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] 
            NPV += CF[i]/((1+r)**i)
        return NPV


    

def ENPV_MC_inflex(nsim, f0):
    ENPV_res =[]
    for i in  range(nsim):
        ENPV2 = NPV_garage_inflex(f0, demand = 'stochastic')
        ENPV_res.append(ENPV2)
        
    return np.mean(ENPV_res)

def ENPV_MC_inflex_CDF(nsim, f0):
    ENPV_res =[]
    for i in range(nsim):
        ENPV2 = NPV_garage_inflex(f0, demand = 'stochastic')
        ENPV_res.append(ENPV2)
    EPV = np.array(ENPV_res)
    ENPV = np.mean(ENPV_res)
    fig, ax = plt.subplots(figsize=(8, 4))
    cdf = ax.hist(EPV, 100, density=True, histtype='step',
                   cumulative=True, label='Inflexible 6 Floor Design Performance')
    plt.axvline(ENPV, color='tab:blue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(ENPV*1.05, max_ylim*0.9, 'ENPV: {:.2f} million $'.format(ENPV/1000000))
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_title('CDF of design solution')
    ax.set_xlabel('NPV of plan($)')
    ax.set_ylabel('Probability')    
    return np.mean(ENPV_res) , np.std(EPV)




def CF_model_out_inflex(f0, demand ='') : 
        k_inflex = f0*n0
        if f0 > 2:
            cc_start = cc * n0 * ((((1+gc)**(f0-1) - (1+gc))/gc)) + (2*n0*cc)
        else : 
            cc_start= f0*n0*cc
        
        years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
        k = pd.Series(index=years)
        revenue = pd.Series(index=years)
        CF = pd.Series(index=years)
        demand_projections = pd.Series(index=years)
        demand_projections_stoch = pd.Series(index=years)
        opex = pd.Series(index=years)
        fixed_costs = pd.Series(index=years)

                            
        for i in range(0,T+1): #initializing all ks to initial capacity
            k[i] = f0*n0
            if demand == 'stochastic':
                demand_projections[i] = demand_stochastic(i)
            else:
                demand_projections[i] = demand_static(i)
                demand_projections_stoch[i] = demand_stochastic(i)
        k[0] = 0
       
        for i in range(0,T+1):
            if i ==T:
                fixed_costs[i] = 0 #no leasing paid in last year
            elif i==0:
                fixed_costs[i] = cc_start + cl
            else : 
                fixed_costs[i] = cl # leasing paid all years including 0CF
            
        for i in range(0,T+1) :
            revenue[i] = np.minimum(demand_projections[i], k[i])*p
            opex[i] = k[i] * cr
            CF[i] = revenue[i] - opex [i] - fixed_costs[i] 
            CF_model_0 = pd.DataFrame({'Capacity': k,
                          'Demand': demand_projections,
                          'Stoch_Demand' : demand_projections_stoch,
                          'Revenue': revenue,
                          'Cash Flow': CF,
                          'OPEX' : opex,
                          ' Fixed cost' : fixed_costs},
                          index=years)   
    
        return CF_model_0    
 
    
 
