# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:02 2020

@author: cesa_
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# Parameters
T=30 #years
Tm = 360 # months in 20 years
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
hours = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20', '21' , '22', '23'] # hour 0 is midnight
hourly_demand = [0 , 0, 0, 0, 0, 0, 20, 20, 120, 120, 100,100, 100, 100, 100, 100, 100, 100,100, 120, 120, 120, 120, 120 ] # in Watts
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241
years =   list(range(0,T + 1))
  
def daily_electricity_load(hourly_demand):
    loads = []
    hours = len(hourly_demand)
    for i in range(hours):
        load_hr = hourly_demand[i] * 1#1 hour
        loads.append(load_hr)
    tot_load = sum(loads)
    return tot_load # in Wh



def monthly_electricity_load(hourly_demand):
    tot_daily = daily_electricity_load(hourly_demand)
    tot_monthly = tot_daily *30*.001 # assuming average month is 30 days
    return tot_monthly # in kWh

def yearly_electricity_load(hourly_demand):
    tot_daily = daily_electricity_load(hourly_demand)
    tot_yearly = tot_daily*365 *.001
    return tot_yearly # this in kWh


# Demand Variables , all in kWh as yearly here

D_1 = yearly_electricity_load(hourly_demand) # Projected year 1 demand
D_10 = 350 #additional demand by year 10
D_F= 250 # additional demand after year 10
T = 30 # project duration in years
alpha = D_10 + D_F # Parameter for demand model showing difference between initial and final demand values
beta = -math.log(D_F/alpha)/(T/2 - 1) # Parameter for demand model showing growth speed of demand curve
offD0 = 0.10 # Realised demand in yr 1 within "x" perccentage of demand projection
offD10 = 0.10 # Additional demand by year 10 within "x" percentage of demand projection
offDf = 0.10 # Additional demand after year 10 within "x" percentage of demand projection
vol = 0.15 # Annual volatility of demand growth within "x" percentage of growth projection


# Demand Variables , all in kWh as but on MONTHLY BASIS
#PROBABLY SHOULD EVALUATE LARGER INCREASES FOR THIS APPROACH TO BE WORTH IT

D_1m = monthly_electricity_load(hourly_demand) # Projected month 1 demand, equal to 53.4
D_120m = 35  #additional demand by year 10, or month120
D_Fm= 25 # additional demand after year 10
Tm = 360 # project duration in months
alpha_m = D_120m + D_Fm # Parameter for demand model showing difference between initial and final demand values
beta_m = -math.log(D_Fm/alpha_m)/(Tm/2 - 1) # Parameter for demand model showing growth speed of demand curve
offD0 = 0.10 # Realised demand in yr 1 within "x" perccentage of demand projection
offD10 = 0.10 # Additional demand by year 10 within "x" percentage of demand projection
offDf = 0.10 # Additional demand after year 10 within "x" percentage of demand projection
vol = 0.15 # Annual volatility of demand growth within "x" percentage of growth projection


def demand_static_series(t) : 
    demand_projections = pd.Series(index=years)
    for i in range(0,t+1):
        demand_projections[i] =  D_1 + D_10 + D_F -alpha * math.exp(-beta*(i-1))
    return demand_projections

def demand_static_series_months(t) : 
    demand_projections = pd.Series(index=months , dtype = 'float64' )
    for i in range(0,t+1):
        demand_projections[i] =  ( D_1m + D_120m + D_Fm - alpha_m * math.exp(-beta_m*(i-1)))     
    return demand_projections


def demand_static_series_ngers(t, n_gers) : 
    demand_projections = pd.Series(index=years)
    for i in range(0,t+1):
        demand_projections[i] =  D_1 + D_10 + D_F -alpha * math.exp(-beta*(i-1))
    return demand_projections

def demand_static_series_months_ngers(t, n_gers) : 
    demand_projections = pd.Series(index=months , dtype = 'float64' )
    for i in range(0,t+1):
        demand_projections[i] = n_gers * ( D_1m + D_120m + D_Fm - alpha_m * math.exp(-beta_m*(i-1)))     
    return demand_projections



a = demand_static_series(30)
b = demand_static_series_months(360)

b_yr = b.groupby(b.index //12).sum()

# n_gers = 1
# test = demand_static_series_months_ngers(Tm , n_gers)
# print(test)

def electricity_monthly_demand_stochastic_less_ngers(n_gers, t,rD0s, rD10s, rDfs) : #this is for RL environment so random sample does not get recalculated every time
    rD0 = (1-offD0)*D_1m + rD0s*2*offD0*D_1m # Realised demand in year 0
    rD10 = (1-offD10)*D_120m +rD10s*2*offD10*D_120m # Realised additional demand by year 10
    rDf = (1-offDf)*D_Fm + rDfs*2*offDf*D_Fm# Realised additional demand after year 10
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(Tm/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-1)) # projected demand vector
    D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-2)) # projected demand vector shifted by one period to right
    D_g_proj = (D_stoc1/D_stoc2) -1
    R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
    D_stoc = D_stoc2 *(1 + R_g)
    d_stoch_gers = D_stoc * n_gers
    return d_stoch_gers

def electricity_monthly_demand_stochastic_less_series_ngers(t, n_gers) : #this is for RL environment so random sample does not get recalculated every time
    demand_projections = pd.Series(index=months)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    for i in range(0,t+1):
        demand_projections[i] = electricity_monthly_demand_stochastic_less_ngers(n_gers, i,rD0s,rD10s,rDfs)
    return demand_projections 


    
def electricity_yearly_demand_stochastic(t) : 
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
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

def electricity_yearly_demand_stochastic_less_series(t) : #t is years
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    demand_projections = pd.Series(index=years)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    for i in range(0,t+1):
        demand_projections[i] = electricity_yearly_demand_stochastic_less(i,rD0s,rD10s,rDfs)
    return demand_projections

def electricity_yearly_demand_stochastic_more_series(t) : #t is years
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    demand_projections = pd.Series(index=years)
    for i in range(0,t+1):
        demand_projections[i] = electricity_yearly_demand_stochastic(i)
    return demand_projections



def electricity_yearly_demand_stochastic_less(t,rD0s, rD10s, rDfs) : #this is for RL environment so random sample does not get recalculated every time
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

def electricity_yearly_demand_stochastic_less_ngers(n_gers, t,  rD0s, rD10s, rDfs) : #this is for RL environment so random sample does not get recalculated every time
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
    d_stoc_nger = D_stoc * n_gers
    return d_stoc_nger

def electricity_yearly_demand_stochastic_less_series_ngers(t, n_gers) : #t is years
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    demand_projections = pd.Series(index=years)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    for i in range(0,t+1):
        demand_projections[i] = electricity_yearly_demand_stochastic_less_ngers(n_gers, i,rD0s,rD10s,rDfs)
    return demand_projections



# # #plotting evoltion of optimal solution
# static_demand = demand_static_series_ngers(T , 1)
# stochastic_demand_1 = electricity_yearly_demand_stochastic_less_series(T)
# stochastic_demand_2 = electricity_yearly_demand_stochastic_less_series(T)
# stochastic_demand_3 = electricity_yearly_demand_stochastic_less_series(T)
# # stochastic_demand_extreme = electricity_yearly_demand_stochastic_more_series(T)
# # stochastic_demand_extreme_2 = electricity_yearly_demand_stochastic_more_series(T)


# plt.figure(figsize=(12,5))
# plt.xlabel('Years')
# plt.ylabel('Demand (kWh/year)')
# plt.title('Projected Electricity Demand Evolution for 1 Ger')
# ax1=static_demand.plot(color='blue', grid=True, label=' Determinstic')
# ax2=stochastic_demand_1.plot(color='orange', grid=True, label=' Stochastic 1')
# ax3=stochastic_demand_2.plot(color='red', grid=True, label=' Stochastic 2')
# ax4=stochastic_demand_3.plot(color='green', grid=True, label=' Stochastic 3')
# # ax5=stochastic_demand_extreme.plot(color='yellow', grid=True, label=' Stochastic more 1')
# # ax5=stochastic_demand_extreme_2.plot(color='magenta', grid=True, label=' Stochastic more 2')
# ax1.legend(loc=2)
# ax2.legend(loc=2)
# ax3.legend(loc=2)
# plt.show()
