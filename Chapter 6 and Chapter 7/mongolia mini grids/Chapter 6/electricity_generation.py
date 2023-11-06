# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:32:02 2020

@author: cesa_
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt 
import random
import numpy as np

# Parameters
T=20 #years
Tm = 240
initial_capacity = 363 #W
initial_capacity_kw = .363 #kW

years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
hours = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20', '21' , '22', '23'] # hour 0 is midnight
hourly_demand = [0 , 0, 0, 0, 0, 0, 20, 20, 120, 120, 100,100, 100, 100, 100, 100, 100, 100,100, 120, 120, 120, 120, 120 ] # in Watts
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241


CF_pv_avg = .20 # average capacity factor for mongolia solar , using 25 as this is value used in HOMER calculations even though ADB report states around 18.5
CF_pv_dev = .01 # standard deviation of CF, assumed here but can be confirmed later
CF_wind_avg = .35 # average capacity factor for mongolia wind
CF_wind_dev = .01 # standard deviation of CF, assumed here but can be confirmed later    


CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]


# all of these are in kW
homer_pv_avg_output_monthtly = np.array([.08, .1 , .1 , .1 , .08, .08, .09, .1, .1, .09, .08 ])
# average deviation is defined here as difference between 25th and 75th percentile (25th is zero in all)
homer_pv_avg_deviation= np.array([.34, .33, .31, .35, .27, .27, .29, .31, .33, .35, .34, .33])




#LOOK AT RIGHT VALUE TO BE USED HERE AS 75TH PERCENTILE CAUSES VERY LOW CF IN SOME PERIODS, MAY EVEN GO NEGATIVE
z_75th_pct = .675
z_90th_pct = 1.282

homer_cf_pv_monthly = homer_pv_avg_output_monthtly/ initial_capacity_kw
homer_avg_cf = np.mean(homer_cf_pv_monthly)
homer_cf_pv_monthly_stdev = (homer_pv_avg_deviation - homer_avg_cf )/ (z_75th_pct)
# print(homer_cf_pv_monthly)
# print(np.mean(homer_cf_pv_monthly))


CF_wind_monthly = [ .255, .262, .295, .370, .385, .322, .310, .224, .273, .241, .256, .278]
CF_wind_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]

def randomised_cf_norm_month(month_index, CF_series, CF_series_dev): # this returns a randomized CF centered around normal distribuion and mean for values of each month
    data_normal = norm.rvs(size=100,loc=CF_series[month_index], scale= CF_series_dev[month_index])
    cf = random.choice(data_normal)
    return cf 

def randomised_cf_norm(CF_avg, CF_dev): # this returns a randomized CF centered around normal distribuion and mean for values of each month
    data_normal = norm.rvs(size=100, loc=CF_avg, scale= CF_dev)
    cf = random.choice(data_normal)
    return cf 



def randomised_cf_norm_month_series( CF_series, CF_series_dev): # this returns a randomized CF centered around normal distribuion and mean for values of each month
    CF_projections = pd.Series(index=months)
    for i in range(Tm + 1):
        month_index = monthly_index_fromtimestep(i)
        data_normal = norm.rvs(size=100, loc=CF_series[month_index], scale= CF_series_dev[month_index])
        CF_projections[i] = random.choice(data_normal)    
    return CF_projections

# a = randomised_cf_norm_month_series(CF_wind_monthly, CF_wind_monthly_dev)

# a_test= randomised_cf_norm_month(1,homer_cf_pv_monthly,homer_cf_pv_monthly_stdev  )
# print(a_test)

# predefine monthly indexes here to be used in function below
Tm = 360 # months

# predefine monthly indexes here to be used in function below
jan_indexes = list(range(0,Tm, 12))
feb_indexes = list(range(1,Tm, 12))
march_indexes = list(range(2,Tm, 12))
april_indexes = list(range(3,Tm, 12))
may_indexes = list(range(4,Tm, 12))
june_indexes = list(range(5,Tm, 12))
july_indexes = list(range(6,Tm, 12))
aug_indexes = list(range(7,Tm, 12))
sep_indexes = list(range(8,Tm, 12))
oct_indexes = list(range(9,Tm, 12))
nov_indexes = list(range(10,Tm, 12))
dec_indexes = list(range(11,Tm, 12))


def monthly_index_fromtimestep(time_step):
    i = 0
    if time_step in jan_indexes:
        i = 0
    elif time_step in feb_indexes:
        i = 1
    elif time_step in march_indexes:
        i = 2
    elif time_step in april_indexes:
        i = 3
    elif time_step in may_indexes:
        i = 4
    elif time_step in june_indexes:
        i = 5
    elif time_step in july_indexes:
        i = 6
    elif time_step in aug_indexes:
        i = 7
    elif time_step in sep_indexes:
        i = 8
    elif time_step in oct_indexes:
        i = 9
    elif time_step in nov_indexes:
        i = 10
    elif time_step in dec_indexes:
        i = 11
    return i 

# a = randomised_cf_norm_month(1, CF_pv_monthly,CF_pv_monthly_dev)


def randomised_cf_norm_series(CF_avg, CF_dev): # this returns a randomized CF centered around normal distribuion and mean 
    data_normal = norm.rvs(size=100,loc=CF_avg, scale= CF_dev)
    CF_projections = pd.Series(index=years)
    for i in range(0, T+1): 
        CF_projections[i] = random.choice(data_normal)
    return CF_projections



def normalised_yearly_stoch_electricity_generation_kwh (cf, capacity): # capacity given in W
    generated_yearly = capacity * 24 * 365 * cf * .001 # generation given in kWh
    return generated_yearly


def normalised_monthly_stoch_electricity_generation_kwh (cf_month, capacity): # capacity given in W
    generated_monthly = capacity * 24 * 30 * cf_month * .001 # generation given in kWh
    return generated_monthly


def total_monthly_system_cost_ngers_static(n_gers,electricity, heating, cf_pv):
    a = n_gers * electricity * cf_pv
    return a
    



# assuming here that if cf are above avg no need for transmitting, only in period where one below
# this is significant simplification and should find better way to do this
# not even including diesel but implied i suppose
def standardised_res_cf_difference(pv_cf, wind_cf): # should probly update this so uses average of EVERY MONTH instead, can do later
    #this is for estimating tranmission requirement, LOOK AGAIN
    max_tranmission_ratio = .2
    pv_cf_ratio = pv_cf / CF_pv_avg
    wind_cf_ratio = wind_cf / CF_wind_avg
    if  pv_cf_ratio < 1 or wind_cf_ratio < 1:
        larger_mismatch = np.min(np.array([pv_cf_ratio,wind_cf_ratio] )) # the minimum will be furthest from its average
        power = (1 - larger_mismatch)
        if power >= max_tranmission_ratio:
            power = max_tranmission_ratio
        else: power = power
    else : power = 0 
    return power 

#standardised_res_cf_difference(.23, .38)


# a = np.min(np.array([1,2]))

def cf_histogram(CF_avg, CF_dev):
    data_normal = norm.rvs(size=1000,loc=CF_avg, scale= CF_dev)
    fig, bx = plt.subplots(figsize=(8, 4))
    a= bx.hist(data_normal, bins = 100)
    bx.set_xlabel('Annual Average Capacity Factor')
    bx.set_ylabel('# occurences out of 1000')
    bx.set_title('Capacity Factor Probability Distribution Mongolia')

def electricity_provided_diesel(diesel_capacity, shortage_over5pct):
    max_theorethical_yearly_production = diesel_capacity * 8760 * .001 # convert from Wh to kWh
    if diesel_capacity == 0:
        cf = 0 
        provided = 0
    elif shortage_over5pct > max_theorethical_yearly_production:
        provided = max_theorethical_yearly_production
        cf = 1
    else : 
        provided = shortage_over5pct
        cf = shortage_over5pct/ max_theorethical_yearly_production
    return provided , cf 

def electricity_provided_diesel_monthly(diesel_capacity, shortage_over5pct):
    max_theorethical_monthly_production = diesel_capacity * 24 * 30 * .001 # convert from Wh to kWh
    if diesel_capacity == 0:
        cf = 0 
        provided = 0
    elif shortage_over5pct > max_theorethical_monthly_production:
        provided = max_theorethical_monthly_production
        cf = 1
    else : 
        provided = shortage_over5pct
        cf = shortage_over5pct/ max_theorethical_monthly_production
    return provided , cf 

# # generate determinstic CF plots

# months_graph = np.array(list(range(0,12)))

# CF_wind_monthly = np.array([ .255, .262, .295, .370, .385, .322, .310, .224, .273, .241, .256, .278])
# CF_wind_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]

# CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
# CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]

# fig, ax = plt.subplots()


# ax.errorbar(months_graph, CF_wind_monthly,
#             xerr=0,
#             yerr=CF_wind_monthly_dev,
#             fmt='-o', label = 'Wind')

# ax.errorbar(months_graph, CF_pv_monthly,
#             xerr=0,
#             yerr=CF_pv_monthly_dev,
#             fmt='-o', label = 'Solar PV')
# ax.set_xlabel('Month')
# ax.set_ylabel('Estimated capacity Factor')
# ax.set_title('Wind and PV capacity factor seasonal variation')
# ax.legend(loc=1)

# plt.show()