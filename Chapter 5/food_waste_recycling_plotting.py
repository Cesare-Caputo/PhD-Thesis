# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:02 2020

@author: cesa_
"""
import math
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

# Parameters
T=15 #years
r = .08 # discount rate
#years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20' , '21', '22' , '23' , '24' ,'25', '26','27' ,'28' ,'29' ,'30' ]
years = list(range(16))

# recycling rate variables
r_0 = .10 # % recycling rate , initially same in all sectors
r_vol = .20 #% limit volatility
r_max_det = .60 # %
r_sharp_vol = .40 # %
fw_vol = .3# %
beta_det = .2
r_vol_annual = .10 # anual volatility of recucling rate


# define list encompassing parameters for all sectors
r_max_s = [60, 49 , 52, 50, 48, 68 , 60, 63, 54]
alpha_r_s = [5 , 4, 4, 4, 4, 6, 5, 5, 4 ]
beta_r_s = [.2 , .1637 , .2057 , .1211 , .2589 , .1985 , .2694 , .1981 , .1611]




# food waste recycling variables
fw_growth = .123 
fw_vol = .25
n_plants = 6
tot_fw_recycled_0 = 274 # tonnes per day
fw_recycled_per_plant_0 = tot_fw_recycled_0 / n_plants



# def recycling_rate_static_s1(t):
#     recycling_t = r_max_s1 / (1+ (alpha_r_s1 * math.exp(-beta_r_s1 * t)))
#     return recycling_t

#print(recycling_rate_static_s1(7))

# the first one here is sector 1 hence why substract by 1 below
# def recycling_rate_static_sector(t , n_sec):
#     r_max = r_max_s[n_sec - 1]
#     alpha_r = alpha_r_s[n_sec - 1]
#     beta_r = beta_r_s[n_sec -1]
#     recycling_t = r_max / (1+ alpha_r * math.exp(-beta_r * t))
#     return recycling_t


def recycling_rate_stochastic_less(t, rand_r_max, rand_b) : #this is for RL environment so random sample does not get recalculated every time
    #np.random.seed(7) # set consdtant seed for simulations to for standardized comparison
    r_max = (1 - r_vol) * (r_max_det) + (2*r_max_det * r_vol * rand_r_max)
    beta_stoc = (1-r_sharp_vol) * (beta_det) + (2*r_sharp_vol * beta_det * rand_b )
    alpha_stoc = (r_max / r_0) - 1
    recycling_t_1 = r_max / (1+ alpha_stoc * math.exp(-beta_stoc * t))
    recycling_t_2 = r_max / (1+ alpha_stoc * math.exp(-beta_stoc * (t-1)))
    r_g_proj = (recycling_t_1 - recycling_t_2) / recycling_t_2
    r_g_act =  r_g_proj +  r_vol_annual* st.norm.ppf(np.random.random_sample())
    r_stoc_t = recycling_t_2 * (1+r_g_act)
    return r_stoc_t 

# r_test1 = np.random.random_sample()
# r_test_2 = np.random.random_sample()
# print(r_test1)
# print(r_test_2)


# print(recyling_rate_stochastic_less(1,r_test1, r_test_2))


def recycling_rate_series(T):
    recycling_rate_projections = pd.Series(index=years)
    rand_r_max = np.random.random_sample() 
    rand_b = np.random.random_sample() 
    for i in range(0,T+1): #initializing all ks to initial capacity
        recycling_rate_projections[i] = recycling_rate_stochastic_less(i,rand_r_max , rand_b)
    return recycling_rate_projections


 # NOW COMPUTE ACTUAL FOOD WASTE RECYCLES
 
def recycled_fw_per_sector_series(T):
    recycling_fw_rate_projections = pd.Series(index=years)
    recycling_fw_rate_projections[0] = fw_recycled_per_plant_0
    for i in range(1, T+1):
        rand_factor = st.norm.ppf(np.random.random_sample())
        growth_act = fw_growth  +  (fw_vol * rand_factor)
        recycling_fw_rate_projections[i] = recycling_fw_rate_projections[i-1] * (1+growth_act)
    return recycling_fw_rate_projections 


 
def recycled_fw_per_sector(t, fw_t_min_1):
    recycling_fw_rate_projections = pd.Series(index=years)
    rand_factor = st.norm.ppf(np.random.random_sample())
    growth_act = fw_growth  +  (fw_vol * rand_factor)
    recycling_fw_rate_projections_t = fw_t_min_1 * (1+growth_act)
    return recycling_fw_rate_projections_t

#print(recycling_rate_series(30))
print(recycled_fw_per_sector_series(15))
print(recycled_fw_per_sector(1, fw_recycled_per_plant_0))


## simulation of demand per sector evolution
s_1_demand_df = recycled_fw_per_sector_series(15)
s_2_demand_df = recycled_fw_per_sector_series(15)
s_3_demand_df = recycled_fw_per_sector_series(15)
s_4_demand_df = recycled_fw_per_sector_series(15)
s_5_demand_df = recycled_fw_per_sector_series(15)
s_6_demand_df = recycled_fw_per_sector_series(15)


mark_f_h = .05

fig = plt.figure(figsize = (10,5))
plt.plot(s_1_demand_df, ':xb' , markevery = mark_f_h , label = 'S1'  )
plt.plot(s_2_demand_df , 'y', markevery = mark_f_h , label = 'S2 ' )
plt.plot(s_3_demand_df, '-.sg' , markevery = mark_f_h ,label = 'S3' )
plt.plot(s_4_demand_df,  '--r', marker = 'd',markevery = mark_f_h ,label = 'S4' )
plt.plot(s_5_demand_df,':' ,  color = 'orange' , marker = '*'  ,markevery = mark_f_h , label = 'S5' )
plt.plot( s_6_demand_df, '-' , color = 'sienna', marker = 'o', markevery = mark_f_h ,label = 'S6' )


plt.legend(loc='upper left' , ncol = 1 , fontsize= 12)
plt.title('Example Stochastic Scenario of Recycled Food Waste per Sector' , fontsize= 14)
plt.xlabel('Year' , fontsize= 12)
plt.ylabel('FW Demand (tpd)' , fontsize= 12)
plt.xticks([i for i in range (16)] )
plt.xlim(0,15)

sim = s_1_demand_df+ s_2_demand_df+ s_3_demand_df+ s_4_demand_df+ s_5_demand_df+ s_6_demand_df
########################## DATA GENERATION #######

SCENARIOS = 2000
conf_level = 1.96 # 95th percentile confidence

vals = []

for i in range(SCENARIOS):
    s_1_demand_df = recycled_fw_per_sector_series(15)
    s_2_demand_df = recycled_fw_per_sector_series(15)
    s_3_demand_df = recycled_fw_per_sector_series(15)
    s_4_demand_df = recycled_fw_per_sector_series(15)
    s_5_demand_df = recycled_fw_per_sector_series(15)
    s_6_demand_df = recycled_fw_per_sector_series(15)
    sim = s_1_demand_df+ s_2_demand_df+ s_3_demand_df+ s_4_demand_df+ s_5_demand_df+ s_6_demand_df
    vals.append(sim)
pc4_df = pd.DataFrame(vals)
pc4 = pc4_df.mean(axis = 0)
pc4_std = pc4_df.std(axis = 0) * conf_level

## plotting
plt.figure(figsize=(10,5))
plt.xlabel('Year')
plt.ylabel('Total Recycled FW (tpd) ')
plt.title('Demand Evolution Uncertainty over Project Lifetime', fontsize = 14)

plt.plot(pc4_df.iloc[3], color = 'orange' , label = 'Example Low Demand Stochastic Scenario',ls='--')
#plt.plot(pc4_df.quantile(.5, axis = 1), color = 'orange' , label = 'Example Low Demand Stochastic Scenario',ls='--')
plt.plot(pc4_df.iloc[1997] , color = 'blue' , label = 'Example Medium Demand Stochastic Scenario', ls='-.')
plt.plot(pc4_df.iloc[1996], color = 'green' , label = 'Example High Demand Stochastic Scenario', ls=':')




y = pc4
yerr = pc4_std
plt.fill_between(list(range(16)), y - yerr, y+yerr, color = 'blue', alpha = .1)

plt.legend(loc='upper left')
plt.xticks([i for i in range (16)] )
plt.xlim(0,15)