# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:33:19 2021

@author: cesa_
"""

# from electricity_demand import *
# from minigrid_cost_rl import *
# from electricity_generation import *
# from heating_demand import *
# from herder_migration import *
# import pandas as pd
# import numpy as np
# from mongolia_plotting import *

# Parameters
T=20 #years
Tm = 240 # months in 20 years
r_yr = 0.06# Discount rate
r_month = ((1+r_yr)**(1/12)) - 1 
static_capacity_pv = 363 #W of installef flat plate pv as starting capacity
static_capacity_battery = 2000 #Wh 
CF_avg = .20 # average capacity factor for mongolia solar from adb report
CF_dev = .01 # standard deviation of CF, assumed here but can be confirmed later
pv_salvage_value_yr20 = salvage_pv(static_capacity_pv, 20)


#defining indexes
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241

#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 10
n_gers = 10



# parameters for electrcity generation predictions
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]

Tm = 240 # 240 months in 20 years


# heating parameters

# coal stove parameters
coal_HV_mj = 14.7 # MJ per kg
coal_HV_kj = 14.7 *1000 # kJ per kg
eff_trad_stove = .25
eff_improved_stove_h = .77 
eff_improved_stove_l = .54
kw_2_kj_perh = 3600 
coal_c02_emission_factor = 1.37 # tonnes of co2 per metric tonne coal burned
# these next values are in kj per kw hr to determine coal requirements
heat_input_trad = kw_2_kj_perh / eff_trad_stove
heat_input_improved_h = kw_2_kj_perh / eff_improved_stove_h
heat_input_improved_l = kw_2_kj_perh / eff_improved_stove_l




#parameters for financial evaluation
coal_price = 40 # $/tonne
coal_price_per_kg = coal_price*.001 # $/tonne
coal_trad_stove_capex = 13 # $
coal_improved_stove_capex_l = 129.5 
coal_improved_stove_capex_m = 151
coal_improved_stove_capex_h = 181


# ELECTRICAL HEATER TECHNICAL SPECS
eff_electric_stove = 1
avg_co2_emission_factor_grid = .7111 # tonnes Co2 per year
eh_lifetime = 13 # years USE LATER TO LOOK AT REPLACEMENT COSTS
eh_capex_kw = 152 # $ per kW
eh_capex_w = 152000 # $ per kW



n_lattice_walls = 5
area_ger_5_walls = 29.3 # m2
area_ger_8_walls = 72.6 # m2
yearly_demand_standard_ger_perm2 = 393 #/ kWh / m2/ yr
yearly_demand_improved_ger_perm2 = 206 #/ kWh / m2/ yr
heating_months = 8 # September 1 to May 1 each year

# generate deterministic demand and migration cluster radius profile and heating , demand values are foe 10 gers
demand_projections = demand_static_series_months_ngers(Tm, n_gers)
cluster_radius_projections = migration_cluster_radius_static_series(Tm)
heating_demand_projections =  monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand_standard_ger_perm2, area_ger_5_walls, n_gers)





def total_system_cost_static_monthly (scenario): 
    yr_cost = pd.Series(index=years, dtype= 'float64')
    cost = 0 
    for i in range(T+1):  
        CF_yr = randomised_cf_norm(CF_avg, CF_dev)
        electricity_generated_yearly_kwh = normalised_yearly_stoch_electricity_generation_kwh(CF_yr, static_capacity_pv)
        mismatch = mismatch_penalty(electricity_generated_yearly_kwh, scenario[i])
        if i == T+1:
            yr_cost[i] = mismatch - pv_salvage_value_yr20 + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        else:
            yr_cost[i] = mismatch + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        cost += yr_cost[i] / ((1+r_yr)**i) # discounting using discount rate above
    return cost

#CONSISTENCY WITH NUMBER OF GERS IN DEMAND AND GENERATION


# test = total_system_cost_static_monthly_scenario(scenario_df[3] ,cf_scenario_df[3] )
# test_per_ger = test/ n_gers
# print(test_per_ger)


# THIS FOR N_GERS ABOVE
def average_system_cost_scenarios_monthly(demand_scenario , cf_scenario, n_scenarios , n_gers): # this gives you average cost found in all these scenarios
    cost_scenario = []
    for i in range(n_scenarios):
        tot_cost = total_system_cost_static_monthly_scenario(demand_scenario[i] , cf_scenario[i] , n_gers)
        cost_scenario.append(tot_cost)
    return np.mean(cost_scenario)



# test_avg = average_system_cost_scenarios_monthly(scenario_df, cf_scenario_df, n_scenarios, n_gers)
# print("average system cost for" ,n_gers," gers over " , n_scenarios, " scenarios is ", test_avg)

def total_system_cost_static(scenario): 
    yr_cost = pd.Series(index=years, dtype= 'float64')
    cost = 0 
    for i in range(T+1):  
        CF_yr = randomised_cf_norm(CF_avg, CF_dev)
        electricity_generated_yearly_kwh = normalised_yearly_stoch_electricity_generation_kwh(CF_yr, static_capacity_pv)
        mismatch = mismatch_penalty(electricity_generated_yearly_kwh, scenario[i])
        if i == T+1:
            yr_cost[i] = mismatch - pv_salvage_value_yr20 + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        else:
            yr_cost[i] = mismatch + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        cost += yr_cost[i] / ((1+r_yr)**i) # discounting using discount rate above
    return cost


def total_system_cost_static_df(scenario):
    cost_df = pf.Dataframe(index = years, columns = ['Mismatch' , 'Opex' , 'Capex' ,'Coal' ])
    yr_cost = pd.Series(index=years, dtype= 'float64')
    cost = 0 
    for i in range(T+1):  
        CF_yr = randomised_cf_norm(CF_avg, CF_dev)
        electricity_generated_yearly_kwh = normalised_yearly_stoch_electricity_generation_kwh(CF_yr, static_capacity_pv)
        mismatch = mismatch_penalty(electricity_generated_yearly_kwh, scenario[i])
        if i == T+1:
            yr_cost[i] = mismatch - pv_salvage_value_yr20 + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        else:
            yr_cost[i] = mismatch + yearly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        cost += yr_cost[i] / ((1+r_yr)**i) # discounting using discount rate above
    return cost




def yearly_system_cost_series_scenario(scenario):
    yr_cost = pd.Series(index=years , dtype= 'float64')
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
    yr_mismatch = pd.Series(index=years , dtype= 'float64')
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

# avg_cost_test = average_system_cost_scenarios(scenario_df, n_scenarios)
# print("Average resuluting system cost from unmet load are", avg_cost_test)

# avg_mismatch_test = average_system_mismatch_scenarios(scenario_df, n_scenarios)
# print("Average kWh mismatch above 5% allowable shortage is" , avg_mismatch_test)


# Scenario generation
def scenario_generator_monthly_stoch(scenarios, n_gers):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = electricity_monthly_demand_stochastic_less_series_ngers(Tm, n_gers)
    return scenario_df

# def scenario_generator_monthly_det(scenarios, n_gers):
#     scenario_df = pd.DataFrame()
#     for i in range(scenarios):
#         scenario_df[i] = electricity_monthly_demand_stochastic_less_series_ngers(Tm, n_gers)
#     return scenario_df



# scenario_df = scenario_generator_monthly(n_scenarios, n_gers)



def CF_monthly_generator(scenarios): # generates yearly capacity factors to be used in analysis
    CF_df = pd.DataFrame()
    for i in range(scenarios):
        CF_df[i] = randomised_cf_norm_month_series(CF_pv_monthly, CF_pv_monthly_dev)
    return CF_df   

# cf_scenario_df = CF_monthly_generator(n_scenarios)


# mismatch_test = total_system_mismatch_static(scenario_test)
# print(mismatch_test)

month_temp = np.array([-26, -24, -14, -4 , 4, 6, 12, 10, 3, -6, -15, -22])
avg_heat_season_temp = -13.4 # celsius

  
# going to look at 5 wall units only for now, maybe can include 8 too later


def total_system_cost_determinstic_monthly_inflex (demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers):
    #MAYBE SHOULD BUILD THIS AS DATAFRAME SO WE HAVE CASH F;LOW MODEL FOR DEBUGGING
    #NOTE THAT ALL COSTS PRESENTED ARE UNDISCOUNTED
    cost_df = pd.DataFrame(index = months, columns = [ 'Total' ,'Capex' ,'Mismatch' , 'Opex' ,'Coal' ])
    eh_heat = 0 # inflexible , no electric heaters available
    cost = 0 
    for i in range(Tm + 1):
        month_index = monthly_index_fromtimestep(i)
        electricity_demand = demand_series[i]
        heat_demand = heating_demand_series[month_index]
        CF_yr = cf_pv_month[month_index]
        coal_cost = monthly_coal_expenditure(heat_demand, eh_heat, coal_HV_kj,  eff_trad_stove, coal_price)
        static_capacity_pv_ngers = static_capacity_pv * n_gers
        electricity_generated_monthly_kwh = normalised_monthly_stoch_electricity_generation_kwh (CF_yr, static_capacity_pv_ngers)
        mismatch = mismatch_penalty(electricity_generated_monthly_kwh, demand_series[i])
        cost_df['Capex'][i] = 0
        cost_df['Mismatch'][i] = mismatch
        cost_df['Opex'][i] = monthly_opex_inflexible(static_capacity_pv, static_capacity_battery)
        cost_df['Coal'][i] = coal_cost
        if i == Tm + 1:
            cost_df['Total'][i] = mismatch - pv_salvage_value_yr20 + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery) + coal_cost
        else:
            cost_df['Total'][i] = mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery) + coal_cost
        #cost += month_cost[i] / ((1+r_month)**i) # discounting using discount rate above
        
    return cost_df




A = total_system_cost_determinstic_monthly_inflex (demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)


inflex_monthly_cf_plot(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)
