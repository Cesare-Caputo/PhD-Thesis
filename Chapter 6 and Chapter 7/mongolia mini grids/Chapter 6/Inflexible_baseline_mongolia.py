# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:33:19 2021

@author: cesa_
"""

from electricity_demand import *
from minigrid_cost_rl import *
from electricity_generation import *
import pandas as pd
import numpy as np

# Parameters
T=20 #years
r = 0.06# Discount rate
static_capacity_pv = 363 #W of installef flat plate pv as starting capacity
static_capacity_battery = 2000 #Wh 
CF_avg = .20 # average capacity factor for mongolia solar from adb report
CF_dev = .01 # standard deviation of CF, assumed here but can be confirmed later
pv_salvage_value_yr20 = salvage_pv(static_capacity_pv, 20)

years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']


#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 1000

# Scenario generation
def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = electricity_yearly_demand_stochastic_less_series(T)
    return scenario_df


scenario_df = scenario_generator(n_scenarios)

def CF_yearly_generator(scenarios): # generates yearly capacity factors to be used in analysis
    CF_df = pd.DataFrame()
    for i in range(scenarios):
        CF_df[i] = randomised_cf_norm(CF_avg, CF_dev)
    return CF_df   



def total_system_cost_static(scenario): 
    yr_cost = pd.Series(index=years)
    cost = 0 
    for i in range(T+1):  
        CF_yr = randomised_cf_norm(CF_avg, CF_dev)
        electricity_generated_yearly_kwh = normalised_yearly_stoch_electricity_generation_kwh(CF_yr, static_capacity_pv)
        mismatch = mismatch_penalty(electricity_generated_yearly_kwh, scenario[i])
        if i == T+1:
            yr_cost[i] = mismatch - pv_salvage_value_yr20 + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        else:
            yr_cost[i] = mismatch + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        cost += yr_cost[i] / ((1+r)**i) # discounting using discount rate above
    return cost

def yearly_system_cost_series_scenario(scenario):
    yr_cost = pd.Series(index=years)
    for i in range(T+1):  
        CF_yr = randomised_cf_norm(CF_avg, CF_dev)
        electricity_generated_yearly_kwh = normalised_yearly_stoch_electricity_generation_kwh(CF_yr, static_capacity)
        mismatch = mismatch_penalty(electricity_generated_yearly_kwh, scenario[i])
        if i == T+1:
            yr_cost[i] = mismatch - pv_salvage_value_yr20
        else:
            yr_cost[i] = mismatch
    return yr_cost

def average_system_cost_scenarios(scenario_df, scenarios): # this gives you average cost found in all these scenarios
    cost_scenario = []
    for i in range(scenarios):
        tot_cost = total_system_cost_static(scenario_df[i])
        cost_scenario.append(tot_cost)
    return np.mean(cost_scenario)

def total_system_mismatch_static(scenario): 
    yr_mismatch = pd.Series(index=years)
    tot_mismatch = 0 
    for i in range(T+1):  
        CF_yr = randomised_cf_norm(CF_avg, CF_dev)
        electricity_generated_yearly_kwh = normalised_yearly_stoch_electricity_generation_kwh(CF_yr, static_capacity_pv)
        yr_mismatch[i] = shortage_over5pct(electricity_generated_yearly_kwh, scenario[i])
        tot_mismatch += yr_mismatch[i]
    return tot_mismatch

def average_system_mismatch_scenarios(scenario_df, scenarios): # this gives you average cost found in all these scenarios
    mismatch_scenario = []
    for i in range(scenarios):
        tot_mismatch = total_system_mismatch_static(scenario_df[i])
        mismatch_scenario.append(tot_mismatch)
    return np.mean(mismatch_scenario)


# scenario_test = scenario_df[3]
# print(total_system_cost_static(scenario_test))

avg_cost_test = average_system_cost_scenarios(scenario_df, n_scenarios)
print("Average resuluting system cost from unmet load are", avg_cost_test)

avg_mismatch_test = average_system_mismatch_scenarios(scenario_df, n_scenarios)
print("Average kWh mismatch above 5% allowable shortage is" , avg_mismatch_test)




# mismatch_test = total_system_mismatch_static(scenario_test)
# print(mismatch_test)