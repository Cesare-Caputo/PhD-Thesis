# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:38:25 2021

@author: cesa_
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy import *


# mongolian migration patterns
# assume distance required will be increasing over time
ger_cluster_radius = 3000 # meters, radius of circle encompassing cluster and defining cable lenght required


# Migration variables all in meters 
r_1 = 6.85 # meters, radius of  10 ger cluster in first month
r_120 = 1.0 # meters, additional radius of  10 ger cluster by year 10
r_f = 0.5 # meters, additional radius of  10 ger cluster after year 10
T = 30
Tm = 360 # project duration in years
alpha = r_120 + r_f # Parameter for migration model showing difference between initial and final cluster radius values
beta = -math.log(r_f/alpha)/(Tm/2 - 1) # Parameter for radius cluster model showing growth speed of demand curve
offD0 = 0.10 # Realised cluster radius in yr 1 within "x" perccentage of  projection
offD10 = 0.10 # Additional demand by year 10 within "x" percentage of  projection
offDf = 0.10 # Additional demand after year 10 within "x" percentage of projection
vol = 0.12 # Annual volatility of radius growth within "x" percentage of growth projection

# yearly baseline profile

years = list(range(0,T + 1))
alpha_y = r_120 + r_f # Parameter for migration model showing difference between initial and final cluster radius values
beta_y = -math.log(r_f/alpha)/(T/2 - 1) # Parameter for radius cluster model showing growth speed of demand curve
def migration_cluster_radius_static_series(t) : 
    radius_projections = pd.Series(index=years, dtype = 'float64' )
    for i in range(0,t+1):
        radius_projections[i] = ( r_1 + r_120 + r_f - alpha_y * math.exp(-beta_y*(i-1)))     
    return radius_projections


det = migration_cluster_radius_static_series(30)


from heating_demand import monthly_index_fromtimestep
mig_months = 4
months = list(range(0,Tm + 1))
month_temp = np.array([-26, -24, -14, -4 , 4, 6, 12, 10, 3, -6, -15, -22])
avg_heat_season_temp = 4.3 # celsius
def seasonal_cluster_radius_static(t):
    det = migration_cluster_radius_static_series(30)
    monthly_mig = pd.Series(index =months)
    win1 = list(range(4))
    sum1 = list(range(4, 8))
    win2 = list(range(8,12))
    for m in range(Tm+1):
        year_index = m//12
        month_index = monthly_index_fromtimestep(m)
        yearly_mig = det[year_index]
        if month_index in win1:
            monthly_mig[m] = 0
        if month_index in sum1:
            monthly_mig[m] =np.abs(((month_temp[month_index] /avg_heat_season_temp)))*yearly_mig
        if month_index in win2:
            monthly_mig[m] = 0
    return monthly_mig



### using onnline example ###########

T=30
mu=0.005
sigma=0.04
S0=6.85
dt=1





from GBM import gbm_sim

S = gbm_sim (S0, mu, sigma, T, dt)
S1 = gbm_sim (S0, mu, sigma, T, dt)
S2 = gbm_sim (S0, mu, sigma, T, dt)
S3 = gbm_sim (S0, mu, sigma, T, dt)
S4 = gbm_sim (S0, mu, sigma, T, dt)


#Sd = gbm_sim (S0, mu, 0, T, dt)
## make sure all start at same point##

S[0] = S0
S1[0] = S0
S2[0] = S0
S3[0] = S0
S4[0] = S0

t = list(range(30))
#plt.figure(figsize=(12,5))
plt.figure(figsize=(8,5))
#plt.plot(t, Sd)
plt.plot(t, det[1:31], '-.', label = 'Determinstic Projections')
plt.plot(t, S, label = 'Stochastic Scenario Example 1')
plt.plot(t, S3 ,label = 'Stochastic Scenario Example 2')
plt.plot(t, S4 ,label = 'Stochastic Scenario Example 3')
# plt.plot(t, S3)
# plt.plot(t, S4)
plt.xlabel('Year')
plt.ylabel('Average Annual Cluster Radius (km)')
plt.title('Projected vs Realised 18-ger System Migration Evolution ')
plt.legend(loc = 'upper left')




def seasonal_cluster_radius_stoch(t):
    det = gbm_sim (S0, mu, sigma, T, dt)
    monthly_mig = pd.Series(index =months)
    win1 = list(range(4))
    sum1 = list(range(4, 8))
    win2 = list(range(8,12))
    for m in range(Tm):
        year_index = m//12
        month_index = monthly_index_fromtimestep(m)
        yearly_mig = det[year_index]
        if month_index in win1:
            monthly_mig[m] = 0
        if month_index in sum1:
            monthly_mig[m] =yearly_mig*np.random.uniform(.95,1.05)
        if month_index in win2:
            monthly_mig[m] = 0
    return monthly_mig

# a = seasonal_cluster_radius_stoch(360)

# a.to_csv('seasonal_mig_cluster.csv')

# yearly parameters

alpha_y = r_120 + r_f # Parameter for migration model showing difference between initial and final cluster radius values
beta_y = -math.log(r_f/alpha)/(T/2 - 1) # Parameter for radius cluster model showing growth speed of demand curve




months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241
years = list(range(0,T + 1))



def migration_cluster_radius(t,rD0s, rD10s, rDfs) : #this is for RL environment so random sample does not get recalculated every time
    rD0 = (1-offD0)*r_1 + rD0s*2*offD0*r_1 # Realised radius in year 0
    rD10 = (1-offD10)*r_120 +rD10s*2*offD10*r_120 # Realised radius by year 10
    rDf = (1-offDf)*r_f + rDfs*2*offDf*r_f# Realised radius after year 10
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(Tm/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-1)) # projected demand vector
    D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(t-2)) # projected demand vector shifted by one period to right
    D_g_proj = (D_stoc1/D_stoc2) -1
    R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
    D_stoc = D_stoc2 *(1 + R_g)
    return D_stoc

def migration_cluster_radius_series(t) : #this is for RL environment so random sample does not get recalculated every time
    radius_projections = pd.Series(index=months)
    rD0s = np.random.random_sample() 
    rD10s = np.random.random_sample() 
    rDfs =np.random.random_sample()
    for i in range(0,t+1):
        radius_projections[i] = migration_cluster_radius(i,rD0s, rD10s, rDfs)
    return radius_projections


def migration_cluster_radius_series_yearly(t) : #this is for RL environment so random sample does not get recalculated every time
    radius_projections = pd.Series(index=years)
    rD0s = np.random.random_sample() 
    rD10s = np.random.random_sample() 
    rDfs =np.random.random_sample()
    for i in range(0,t+1):
        radius_projections[i] = migration_cluster_radius(i,rD0s, rD10s, rDfs)
    return radius_projections


test = migration_cluster_radius_series_yearly(T)





def migration_cluster_radius_static_series(t) : 
    radius_projections = pd.Series(index=months , dtype = 'float64' )
    for i in range(0,t+1):
        radius_projections[i] = ( r_1 + r_120 + r_f - alpha * math.exp(-beta*(i-1)))     
    return radius_projections


# def herder_migration_stochastic_less_series(t) : #t is years
#     #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
#     demand_projections = pd.Series(index=years)
#     rD0s = np.random.random_sample() # Realised demand in year 0
#     rD10s = np.random.random_sample() # Realised additional demand by year 10
#     rDfs =np.random.random_sample()# Realised additional demand after year 10
#     for i in range(0,t+1):
#         demand_projections[i] = electricity_yearly_demand_stochastic_less(i,rD0s,rD10s,rDfs)
#     return demand_projections






test = migration_cluster_radius_static_series(30)



# test1 = migration_cluster_radius_series(Tm)
# test2 = migration_cluster_radius_series(Tm)
# test3 = migration_cluster_radius_series(Tm)



# plt.figure(figsize=(12,5))
# plt.xlabel('Months')
# plt.ylabel('Migration Cluster Radius(m)')
# plt.title('Projected Migration Cluster radius for 10 ger group')
# ax1=test.plot(color='blue', grid=True, label=' Deterministic')
# ax2=test1.plot(color='orange', grid=True, label='  Stochastic 1')
# ax3=test2.plot(color='red', grid=True, label=' Stochastic 2')
# ax4=test3.plot(color='green', grid=True, label=' Stochastic 3')
# # ax5=stochastic_demand_extreme.plot(color='yellow', grid=True, label=' Stochastic more 1')
# # ax5=stochastic_demand_extreme_2.plot(color='magenta', grid=True, label=' Stochastic more 2')
# ax1.legend(loc=2)
# # ax2.legend(loc=2)
# # ax3.legend(loc=2)
# plt.show()