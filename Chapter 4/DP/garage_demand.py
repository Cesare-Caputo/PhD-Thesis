# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:02 2020

@author: cesa_
"""
import math
import numpy as np
import pandas as pd


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

years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']

# Demand Variables

D_1 = 750 # Projected year 1 demand
D_10 = 750 #additional demand by year 10
D_F= 250 # additional demand after year 10
T = 20 # project duration in years
alpha = D_10 + D_F; # Parameter for demand model showing difference between initial and final demand values
beta = -math.log(D_F/alpha)/(T/2 - 1) # Parameter for demand model showing growth speed of demand curve


def demand_static(t): # t represents year
        D_stat = D_1 + D_10 + D_F -alpha * math.exp(-beta*(t-1))
        return D_stat
    
  
 # Stochastic demand variables

 

    
def demand_stochastic(t) : 
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    offD0 = 0.50 # Realised demand in yr 1 within "x" perccentage of demand projection
    offD10 = 0.50 # Additional demand by year 10 within "x" percentage of demand projection
    offDf = 0.50 # Additional demand after year 10 within "x" percentage of demand projection
    vol = 0.15 # Annual volatility of demand growth within "x" percentage of growth projection
    rD0 = (1-offD0)*D_1 + np.random.random_sample()*2*offD0*D_1 # Realised demand in year 0
    rD10 = (1-offD10)*D_10 +np.random.random_sample()*2*offD10*D_10 # Realised additional demand by year 10
    rDf = (1-offDf)*D_F + np.random.random_sample()*2*offDf*D_F# Realised additional demand after year 10
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(T/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-1)) # projected demand vector
    D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-2)) # projected demand vector shifted by one period to right
    D_g_proj = (D_stoc1/D_stoc2) -1
    R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
    D_stoc = D_stoc2 *(1 + R_g)
    return D_stoc

def demand_stochastic_series(t) : #t is years
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    demand_projections = pd.Series(index=years)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    for i in range(0,t+1):
        demand_projections[i] = demand_stochastic_less(i,rD0s,rD10s,rDfs)
    return demand_projections




def demand_stochastic_less(t,rD0s, rD10s, rDfs) : #this is for RL environment so random sample does not get recalculated every time
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    offD0 = 0.50 # Realised demand in yr 1 within "x" perccentage of demand projection
    offD10 = 0.50 # Additional demand by year 10 within "x" percentage of demand projection
    offDf = 0.50 # Additional demand after year 10 within "x" percentage of demand projection
    vol = 0.15 # Annual volatility of demand growth within "x" percentage of growth projection
    rD0 = (1-offD0)*D_1 + rD0s*2*offD0*D_1 # Realised demand in year 0
    rD10 = (1-offD10)*D_10 +rD10s*2*offD10*D_10 # Realised additional demand by year 10
    rDf = (1-offDf)*D_F + rDfs*2*offDf*D_F# Realised additional demand after year 10
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(T/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-1)) # projected demand vector
    D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-2)) # projected demand vector shifted by one period to right
    D_g_proj = (D_stoc1/D_stoc2) -1
    R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
    D_stoc = D_stoc2 *(1 + R_g)
    return D_stoc

def demand_stochastic_less_series(t, rD0, rD10, rDf) : #this is for RL environment so random sample does not get recalculated every time
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
    demand_projections = pd.Series(index=years)
    T=20
    offD0 = 0.50 # Realised demand in yr 1 within "x" perccentage of demand projection
    offD10 = 0.50 # Additional demand by year 10 within "x" percentage of demand projection
    offDf = 0.50 # Additional demand after year 10 within "x" percentage of demand projection
    vol = 0.15 # Annual volatility of demand growth within "x" percentage of growth projection
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(T/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-1)) # projected demand vector
    D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-2)) # projected demand vector shifted by one period to right
    D_g_proj = (D_stoc1/D_stoc2) -1
    R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
    D_stoc = D_stoc2 *(1 + R_g)
    return D_stoc



def cc_start(f0):
    if f0 > 2:
        cc_i = cc * n0 * ((((1+gc)**(f0-1) - (1+gc))/gc)) + (2*n0*cc)
    else : 
        cc_i= f0*n0*cc
    return cc_i
