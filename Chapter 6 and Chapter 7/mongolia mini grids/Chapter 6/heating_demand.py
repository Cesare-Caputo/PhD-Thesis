# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:10:45 2021

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
months_inyr = list(range(0,12))



# average NIGHTIME temperatures Iin celsius to determine relative heat demand each month
#essential assuming heat demand is proportional to nightime temp, reasonable
month_temp = np.array([-26, -24, -14, -4 , 4, 6, 12, 10, 3, -6, -15, -22])
avg_heat_season_temp = -13.4 # celsius

n_gers = 10
# going to look at 5 wall units only for now, maybe can include 8 too later
n_lattice_walls = 5
area_ger_5_walls = 29.3 # m2
area_ger_8_walls = 72.6 # m2
yearly_demand_standard_ger_perm2 = 393 #/ kWh / m2/ yr
yearly_demand_improved_ger_perm2 = 206 #/ kWh / m2/ yr
heating_months = 8 # September 1 to May 1 each year

monthly_demand_standard_ger_perm2 = yearly_demand_standard_ger_perm2 / heating_months

capex_insulation_improved_ger = 2700 # $ for all materials and construction
capex_night_storage = 1800 # USD


#estimation of monthly demand
def monthly_heat_demand_fromtemp_perm2(month_temp , yearly_demand):
    monthly_ger_heat_demand = yearly_demand / heating_months
    monthly_heat_demand = pd.Series(index =months_inyr)
    for i in range(4):
        monthly_heat_demand[i] = np.abs(((month_temp[i]  /avg_heat_season_temp)) * monthly_ger_heat_demand)
    for i in range(4,8):
        monthly_heat_demand[i] = 0
    for i in range(8,12):
        monthly_heat_demand[i] = np.abs(((month_temp[i] ) /avg_heat_season_temp) * monthly_ger_heat_demand)
    return monthly_heat_demand


def monthly_heat_demand_fromtemp_per_ger(month_temp , yearly_demand, ger_area):
    monthly_ger_heat_demand = yearly_demand / heating_months
    monthly_heat_demand = pd.Series(index =months_inyr)
    for i in range(4):
        monthly_heat_demand[i] = np.abs(((month_temp[i]  /avg_heat_season_temp)) * monthly_ger_heat_demand)
    for i in range(4,8):
        monthly_heat_demand[i] = 0
    for i in range(8,12):
        monthly_heat_demand[i] = np.abs(((month_temp[i] ) /avg_heat_season_temp) * monthly_ger_heat_demand)
    return monthly_heat_demand * ger_area


def monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand, ger_area , n_gers):
    monthly_ger_heat_demand = yearly_demand / heating_months
    monthly_heat_demand = pd.Series(index =months_inyr)
    for i in range(4):
        monthly_heat_demand[i] = np.abs(((month_temp[i]  /avg_heat_season_temp)) * monthly_ger_heat_demand)
    for i in range(4,8):
        monthly_heat_demand[i] = 0
    for i in range(8,12):
        monthly_heat_demand[i] = np.abs(((month_temp[i] ) /avg_heat_season_temp) * monthly_ger_heat_demand)
    return monthly_heat_demand * ger_area * n_gers



# gonna do gbm around s trend curve for simplicity
#yearly demand in terms of kwh per m2 as other units already in this
g_yr = .01
d_1 = yearly_demand_standard_ger_perm2
d_15 = yearly_demand_standard_ger_perm2 * ((1 + g_yr)**15)
d_30  = yearly_demand_standard_ger_perm2 * ((1 + g_yr)**15)
alpha_h = d_15 + d_30 # Parameter for demand model showing difference between initial and final demand values
beta_h = -math.log(d_30/alpha_h)/(Tm/2 - 1) # Parameter for demand model showing growth speed of demand curve
offD0 = 0.05 # Realised demand in yr 1 within "x" perccentage of demand projection
offD10 = 0.05 # Additional demand by year 10 within "x" percentage of demand projection
offDf = 0.05 # Additional demand after year 10 within "x" percentage of demand projection
vol = 0.05 # Annual volatility of demand growth within "x" percentage of growth projection





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

def monthly_heat_demand_fromtemp_ngers_stoch_series(ger_area , n_gers):
    #random curve parameters
    years = list(range(31))
    yearly_heat_demand = pd.Series(index =years)
    monthly_heat_demand = pd.Series(index =months)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    rD0 = (1-offD0)*d_1 + rD0s*2*offD0*d_1 # Realised demand in year 0
    rD10 = (1-offD10)*d_15 +rD10s*2*offD10*d_15 # Realised additional demand by year 10
    rDf = (1-offDf)*d_30 + rDfs*2*offDf*d_30# Realised additional demand after year 10
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(Tm/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    for i in range (T+1):
        D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(i-1)) # projected demand vector
        D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(i-2)) # projected demand vector shifted by one period to right
        D_g_proj = (D_stoc1/D_stoc2) -1
        R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
        D_stoc = D_stoc2 *(1 + R_g)
        yearly_heat_demand[i] = D_stoc
    win1 = list(range(4))
    sum1 = list(range(4, 8))
    win2 = list(range(8,12))
    for m in range(Tm+1):
        year_index = m//12
        monthly_ger_heat_demand = yearly_heat_demand[year_index] / heating_months
        month_index = monthly_index_fromtimestep(m)
        if month_index in win1:
            monthly_heat_demand[m] = np.abs(((month_temp[month_index]  /avg_heat_season_temp)) * monthly_ger_heat_demand)
        if month_index in sum1:
            monthly_heat_demand[m] = 0
        if month_index in win2:
            monthly_heat_demand[m] = np.abs(((month_temp[month_index]  /avg_heat_season_temp)) * monthly_ger_heat_demand)
    return monthly_heat_demand * ger_area * n_gers

def yearly_heat_demand_fromtemp_ngers_stoch_series(ger_area , n_gers):
    #random curve parameters
    years = list(range(31))
    yearly_heat_demand = pd.Series(index =years)
    rD0s = np.random.random_sample() # Realised demand in year 0
    rD10s = np.random.random_sample() # Realised additional demand by year 10
    rDfs =np.random.random_sample()# Realised additional demand after year 10
    rD0 = (1-offD0)*d_1 + rD0s*2*offD0*d_1 # Realised demand in year 0
    rD10 = (1-offD10)*d_15 +rD10s*2*offD10*d_15 # Realised additional demand by year 10
    rDf = (1-offDf)*d_30 + rDfs*2*offDf*d_30# Realised additional demand after year 10
    alpha_stoc = rD10 + rDf # Parameter for demand model showing difference between initial and final demand values
    beta_stoc = -math.log(rDf/alpha_stoc)/(Tm/2 - 1)#  Parameter for demand model showing growth speed of demand curve
    for i in range (T+1):
        D_stoc1 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(i-1)) # projected demand vector
        D_stoc2 = rD0 + rD10 + rDf  -alpha_stoc * math.exp(-beta_stoc *(i-2)) # projected demand vector shifted by one period to right
        D_g_proj = (D_stoc1/D_stoc2) -1
        R_g = D_g_proj - vol +   np.random.random_sample()*2*vol
        D_stoc = D_stoc2 *(1 + R_g)
        yearly_heat_demand[i] = D_stoc
    return yearly_heat_demand


yr_test = yearly_heat_demand_fromtemp_ngers_stoch_series(area_ger_5_walls , n_gers)
yr_test2 = yearly_heat_demand_fromtemp_ngers_stoch_series(area_ger_5_walls , n_gers)
yr_test3 = yearly_heat_demand_fromtemp_ngers_stoch_series(area_ger_5_walls , n_gers)


plt.figure(figsize=(8,5))
plt.xlabel('Years')
plt.ylabel('Heating Demand (kWh/year)')
plt.title('Projected Long Term Stochastic Heating Demand Evolution: 10 gers')
ax2=yr_test.plot(color='orange', grid=True, label=' Scenario 1')
ax1=yr_test2.plot(color='blue', grid=True, label=' Scenario 2')
ax3=yr_test3.plot(color='red',  grid=True, label=' Scenario 3')
ax2.legend(loc=2)
ax1.legend(loc=2)
ax3.legend(loc=2)
plt.show()



n_gers = 18
#test =monthly_heat_demand_fromtemp_per_ger(month_temp, yearly_demand_standard_ger_perm2,area_ger_5_walls)
test2 = monthly_heat_demand_fromtemp_ngers_stoch_series(area_ger_5_walls , n_gers)
test = monthly_heat_demand_fromtemp_ngers_stoch_series(area_ger_5_walls , n_gers)
test3 = monthly_heat_demand_fromtemp_ngers_stoch_series(area_ger_5_walls , n_gers)
# test_2 =monthly_heat_demand_fromtemp_per_ger(month_temp, yearly_demand_improved_ger_perm2,area_ger_5_walls)
months_in_yr = list(range(1,13))

a= np.array(test2[348:360])
b = np.array(test2[0:12])
c = np.array(test3[348:360])
d = np.array(test[348:360])
yr30 = pd.Series(a, index = months_in_yr)
yr0 = pd.Series(b, index = months_in_yr)
yr302 = pd.Series(c, index = months_in_yr)
yr303 = pd.Series(d, index = months_in_yr)

plt.figure(figsize=(12,5))
#plt.xlabel('Month')
plt.ylabel('Heating Demand (kWh/month)')
plt.title('Projected Seasonal Heating Demand Evolution Profile: 18 gers')
ax2=yr0.plot(color='orange', grid=True, label=' Year 1')
ax1=yr302.plot(color='blue', grid=True, label=' Year 30 - Scenario 1')
ax3=yr30.plot(color='green',  grid=True, label=' Year 30 - Scenario 2')
ax4=yr303.plot(color='magenta',  grid=True, label=' Year 30 - Scenario 3')
# # ax3=stochastic_demand_2.plot(color='red', grid=True, label=' Stochastic 2')
# # ax4=stochastic_demand_3.plot(color='green', grid=True, label=' Stochastic 3')
# # ax5=stochastic_demand_extreme.plot(color='yellow', grid=True, label=' Stochastic more 1')
# # ax5=stochastic_demand_extreme_2.plot(color='magenta', grid=True, label=' Stochastic more 2')
ax2.legend(loc=3)
ax1.legend(loc=3)
ax3.legend(loc=3)
plt.xticks([2,4,6,8,10,12], ['February' , 'April' ,'June' , 'August' , 'October' , 'December'])
plt.show()