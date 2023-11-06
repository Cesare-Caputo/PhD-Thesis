# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:02 2020

@author: cesa_
"""
import math
import numpy as np
import pandas as pd
import scipy.stats as st


# Parameters
T=30 #years
r = .08 # discount rate
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20' , '21', '22' , '23' , '24' ,'25', '26','27' ,'28' ,'29' ,'30' ]


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
print(recycled_fw_per_sector_series(30))
print(recycled_fw_per_sector(1, fw_recycled_per_plant_0))
