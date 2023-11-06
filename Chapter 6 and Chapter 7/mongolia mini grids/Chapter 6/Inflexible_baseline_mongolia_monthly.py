# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:33:19 2021

@author: cesa_
"""

from electricity_demand import *
from minigrid_cost_rl import *
from electricity_generation import *
from heating_demand import *
from herder_migration import *
from heating_generation import *
import pandas as pd
import numpy as np
#from mongolia_plotting import *
import scipy.optimize

# Parameters
T=30 #years
Tm = 360 # months in 20 years
r_yr = 0.06# Discount rate
r_month = ((1+r_yr)**(1/12)) - 1 
# static_capacity_pv = 363 #W of installef flat plate pv as starting capacity
# static_capacity_battery = 2000 #Wh 
# static_capacitY_inverter = 2630 # W
CF_avg = .20 # average capacity factor for mongolia solar from adb report
CF_dev = .01 # standard deviation of CF, assumed here but can be confirmed later


#defining indexes
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241

#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 10
n_gers = 10



# parameters for electrcity generation predictions
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]




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
        cost_df['Opex'][i] = monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter)
        cost_df['Coal'][i] = coal_cost
        if i == Tm + 1:
            cost_df['Total'][i] = mismatch - pv_salvage_value_yr20 + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery) + coal_cost
        else:
            cost_df['Total'][i] = mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter) + coal_cost    
    return cost_df

def total_system_cost_determinstic_monthly_inflex_disc (demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers):
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
        cost_df['Opex'][i] = 10* monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter)
        cost_df['Coal'][i] = coal_cost
        if i == Tm + 1:
            cost_df['Total'][i] = (mismatch - pv_salvage_value_yr20 + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter) + coal_cost)/ ((1+r_month)**i)
        else:
            cost_df['Total'][i] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter) + coal_cost)  / ((1+r_month)**i)
    return cost_df


# generate deterministic demand and migration cluster radius profile and heating , demand values are foe 10 gers
demand_projections = demand_static_series_months_ngers(Tm, n_gers)
#cluster_radius_projections = migration_cluster_radius_static_series(Tm)
heating_demand_projections =  monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand_standard_ger_perm2, area_ger_5_walls, n_gers)
# parameters for electrcity generation predictions
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]




def inflex_mongolia_df(initial_design):
    n_gers = 10
    starting_pv_capacity = initial_design[0] * n_gers
    starting_eh_capacity = initial_design[1] * n_gers
    money_weighing_factor = 1
    emission_weighing_factor = 0
    coal_c02_emission_factor = 1.37 # tonnes of co2 per metric tonne coal burned
    cost_df = pd.DataFrame(index = months, columns = [ 'Total' , 'Capex' ,'Mismatch' ,'EH(kWh)' , 'Grid($)' , 'Grid(kWh)', 'Opex' ,'Coal' , 'Coal Cost', 'CO2'])
    #eh_heat = 0 # inflexible , no electric heaters available
    cost = 0 
    demand_series = demand_projections
    heating_demand_series = heating_demand_projections
    cf_pv_month = CF_pv_monthly
    coal_price = coal_price_per_kg
    static_capacity_pv = starting_pv_capacity
    static_capacity_battery = battery_per_pv_inflex(static_capacity_pv)
    static_capacity_inverter , inverter_capex = expansion_impact_inverter(static_capacity_pv)
    #these times are in months, corresponding to expected replacement years
    batt_repl_yr = list(range(48, 360, 48))
    pv_repl_yr = [240]
    inverter_repl_yr =[181]
    eh_repl_yr = [180]
    for i in range(Tm + 1):
        month_index = monthly_index_fromtimestep(i)
        electricity_demand = demand_series[i]
        heat_demand = heating_demand_series[month_index]
        CF_yr = cf_pv_month[month_index]
        static_capacity_pv_ngers = static_capacity_pv * n_gers
        electricity_generated_monthly_kwh = normalised_monthly_stoch_electricity_generation_kwh (CF_yr, static_capacity_pv)
        extra_electricity = np.max([(electricity_generated_monthly_kwh - electricity_demand) , 0])
        eh_heat_res, leftover_posteh_electricity = output_electric_heater_monthly_kwh(heat_demand, starting_eh_capacity, extra_electricity)
        grid_eh_purchase, eh_heat_grid = grid_electricity_interaction_inflex(eh_heat_res, starting_eh_capacity, heat_demand , i)
        eh_heat_total = eh_heat_res + eh_heat_grid
        #coal_cost = monthly_coal_expenditure(heat_demand, eh_heat, coal_HV_kj,  eff_trad_stove, coal_price)
        coal_mass = monthly_coal_requirement(heat_demand, eh_heat_total, coal_HV_kj,  eff_trad_stove)
        coal_cost = coal_mass * coal_price_per_kg
        mismatch = mismatch_penalty(electricity_generated_monthly_kwh, demand_series[i])
        cost_df.loc[i, 'Grid($)'] = grid_eh_purchase
        cost_df.loc[i, 'Mismatch'] = mismatch
        cost_df.loc[i, 'Opex'] = monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) 
        cost_df.loc[i, 'Coal Cost'] = coal_cost
        cost_df.loc[i, 'Coal']= coal_mass
        cost_df.loc[i, 'CO2'] = carbon_footprint_inflex(coal_mass,eh_heat_grid) 
        cost_df.loc[i, 'EH(kWh)']= eh_heat_total
        cost_df.loc[i, 'Grid(kWh)'] = eh_heat_grid
        if i == Tm + 1:
            pv_salvage_value = salvage_pv(static_capacity_pv, 10)
            cost_df.loc[i, 'Total'] = (mismatch - pv_salvage_value + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter) + coal_cost)/ ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = - pv_salvage_value
        elif i ==0 :
            cost_df.loc[i, 'Total'] = pv_capex_inflex(starting_pv_capacity) + eh_capex_inflex(starting_eh_capacity) + battery_capex_inflex(static_capacity_battery) + inverter_capex_inflex(static_capacity_inverter)
            cost_df.loc[i, 'Capex'] = pv_capex_inflex(starting_pv_capacity) + eh_capex_inflex(starting_eh_capacity) + battery_capex_inflex(static_capacity_battery) + inverter_capex_inflex(static_capacity_inverter)
        elif i in batt_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + battery_capex_inflex(static_capacity_battery))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = battery_capex_inflex(static_capacity_battery)
        elif i in pv_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + pv_capex_inflex(starting_pv_capacity))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = pv_capex_inflex(static_capacity_pv)
        elif i in inverter_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + inverter_capex_inflex(static_capacity_inverter))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = inverter_capex_inflex(static_capacity_inverter)
        elif i in eh_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + eh_capex_inflex(starting_eh_capacity))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = eh_capex_inflex(starting_eh_capacity)      
        else:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost)  / ((1+r_month)**i)
        cost_df = cost_df.fillna(0)
    return cost_df






def inflex_mongolia_capacity_opt(initial_design):
    cost_df = inflex_mongolia_df(initial_design)
    lcc = cost_df["Total"].sum()
    return lcc





def inflex_mongolia_c02_opt(initial_design):
    cost_df = inflex_mongolia_df(initial_design)
    lcc = cost_df["CO2"].sum()
    return lcc



def inflex_mongolia_df_mo(initial_design):
    n_gers = 10
    starting_pv_capacity = initial_design[0] * n_gers
    starting_eh_capacity = initial_design[1] * n_gers
    money_weighing_factor = 1
    emission_weighing_factor = 0
    coal_c02_emission_factor = 1.37 # tonnes of co2 per metric tonne coal burned
    cost_df = pd.DataFrame(index = months, columns = [ 'Total' , 'Capex' ,'Mismatch' ,'EH(kWh)' , 'Grid($)' , 'Grid(kWh)', 'Opex' ,'Coal' , 'Coal Cost', 'CO2'])
    #eh_heat = 0 # inflexible , no electric heaters available
    cost = 0 
    demand_series = demand_projections
    heating_demand_series = heating_demand_projections
    cf_pv_month = CF_pv_monthly
    coal_price = coal_price_per_kg
    static_capacity_pv = starting_pv_capacity
    static_capacity_battery = battery_per_pv_inflex(static_capacity_pv)
    static_capacity_inverter , inverter_capex = expansion_impact_inverter(static_capacity_pv)
    #these times are in months, corresponding to expected replacement years
    batt_repl_yr = list(range(48, 360, 48))
    pv_repl_yr = [240]
    inverter_repl_yr =[181]
    eh_repl_yr = [180]
    for i in range(Tm + 1):
        month_index = monthly_index_fromtimestep(i)
        electricity_demand = demand_series[i]
        heat_demand = heating_demand_series[month_index]
        CF_yr = cf_pv_month[month_index]
        static_capacity_pv_ngers = static_capacity_pv * n_gers
        electricity_generated_monthly_kwh = normalised_monthly_stoch_electricity_generation_kwh (CF_yr, static_capacity_pv)
        extra_electricity = np.max([(electricity_generated_monthly_kwh - electricity_demand) , 0])
        eh_heat_res, leftover_posteh_electricity = output_electric_heater_monthly_kwh(heat_demand, starting_eh_capacity, extra_electricity)
        grid_eh_purchase, eh_heat_grid = grid_electricity_interaction_inflex(eh_heat_res, starting_eh_capacity, heat_demand , i)
        eh_heat_total = eh_heat_res + eh_heat_grid
        #coal_cost = monthly_coal_expenditure(heat_demand, eh_heat, coal_HV_kj,  eff_trad_stove, coal_price)
        coal_mass = monthly_coal_requirement(heat_demand, eh_heat_total, coal_HV_kj,  eff_trad_stove)
        coal_cost = coal_mass * coal_price_per_kg
        mismatch = mismatch_penalty(electricity_generated_monthly_kwh, demand_series[i])
        cost_df.loc[i, 'Grid($)'] = grid_eh_purchase
        cost_df.loc[i, 'Mismatch'] = mismatch
        cost_df.loc[i, 'Opex'] = monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) 
        cost_df.loc[i, 'Coal Cost'] = coal_cost
        cost_df.loc[i, 'Coal']= coal_mass
        cost_df.loc[i, 'CO2'] = carbon_footprint_inflex(coal_mass,eh_heat_grid) 
        cost_df.loc[i, 'EH(kWh)']= eh_heat_total
        cost_df.loc[i, 'Grid(kWh)'] = eh_heat_grid
        if i == Tm + 1:
            pv_salvage_value = salvage_pv(static_capacity_pv, 10)
            cost_df.loc[i, 'Total'] = (mismatch - pv_salvage_value + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacitY_inverter) + coal_cost)/ ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = - pv_salvage_value
        elif i ==0 :
            cost_df.loc[i, 'Total'] = pv_capex_inflex(starting_pv_capacity) + eh_capex_inflex(starting_eh_capacity) + battery_capex_inflex(static_capacity_battery) + inverter_capex_inflex(static_capacity_inverter)
            cost_df.loc[i, 'Capex'] = pv_capex_inflex(starting_pv_capacity) + eh_capex_inflex(starting_eh_capacity) + battery_capex_inflex(static_capacity_battery) + inverter_capex_inflex(static_capacity_inverter)
        elif i in batt_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + battery_capex_inflex(static_capacity_battery))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = battery_capex_inflex(static_capacity_battery)
        elif i in pv_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + pv_capex_inflex(starting_pv_capacity))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = pv_capex_inflex(static_capacity_pv)
        elif i in inverter_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + inverter_capex_inflex(static_capacity_inverter))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = inverter_capex_inflex(static_capacity_inverter)
        elif i in eh_repl_yr:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost + eh_capex_inflex(starting_eh_capacity))  / ((1+r_month)**i)
            cost_df.loc[i, 'Capex'] = eh_capex_inflex(starting_eh_capacity)      
        else:
            cost_df.loc[i, 'Total'] = (mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery, static_capacity_inverter) + coal_cost)
        mo_formulation = (cost_df["Total"].sum() * money_weighing_factor) + (cost_df["CO2"].sum() * emission_weighing_factor)
    return mo_formulation



# pv0_homer = 363
# pv0_ga = 8850
# eh_0_test = 1000

# x0 = np.array([pv0_ga, 1000])
# x1 = np.array([0, 4000])



# # #inflex_mongolia_df(x0)

# # x_mo = np.array([0, 10000])

# a = inflex_mongolia_df(x1)
#print(inflex_mongolia_capacity_opt(x2))


# inflex_df_mo_0 = inflex_mongolia_df_mo(x0)
# # inflex_df_mo_1 = inflex_mongolia_df_mo(x1)
# print(inflex_mongolia_capacity_opt(x0))



# print(inflex_df_mo_0)
# print(inflex_df_mo_1)






# print(inflex_mongolia_capacity_opt(x0))
# print(inflex_mongolia_c02_opt(x0))

x0 = np.array([380, 40])
x1 = np.array([863, 5000])
x2 = np.array([281, 0])
x3 = np.array([181, 0])



# from scipy.optimize import differential_evolution, minimize
# results = dict()
# bounds = [(200,1000) , (0,1000)]
# bnds = ((200,3000))
# results['GA'] = differential_evolution(inflex_mongolia_capacity_opt, bounds)


 


# results['Powell']=  minimize(inflex_mongolia_capacity_opt, x0, method= 'Powell', bounds = bounds,  
#                             options={'disp':True,'maxiter':2001})


# print(results['Powell'])
# print(results['GA'])



# z_test_1 = np.array([215, 0])
# z_test_2 = np.array([225, 0])


# print(inflex_mongolia_capacity_opt(z_test_1))
# print(inflex_mongolia_capacity_opt(z_test_2))


# print(inflex_mongolia_capacity_opt(results['GA'].x))
# print(inflex_mongolia_capacity_opt(x3))
# print(inflex_mongolia_capacity_opt(x2))
# print(inflex_mongolia_capacity_opt(x1))
# print(inflex_mongolia_capacity_opt(x0))
# print(inflex_mongolia_capacity_opt(results['Powell'].x))



# inflex_mongolia_df(x0)
# inflex_mongolia_df(x2)






# x_old = np.array([215, 300])


# # # # # test outputs

# print(inflex_mongolia_df_mo(results['trust-constr'].x))
# print(inflex_mongolia_df_mo(results['GA'].x))
# print(inflex_mongolia_df_mo(x_old))





# a = inflex_mongolia_df(results['trust-constr'].x)
# b = inflex_mongolia_df(results['GA'].x)
# c = inflex_mongolia_df(x_old)











#%% MULTI OBJECTIVE OPTIMIZATION OF INFLEXIBLE BASELINE

# from pymoo.algorithms.nsga2 import NSGA2
# from pymoo.model.problem import Problem
# from pymoo.optimize import minimize
# from pymoo.visualization.scatter import Scatter
# from pymoo.util.misc import stack

# class MyProblem(Problem):

#     def __init__(self):
#         super().__init__(n_var=2,
#                           n_obj=2,
#                           n_constr=0,
#                           xl=np.array([500, 10000]),
#                           xu=np.array([500, 20000]),
#                           elementwise_evaluation=True)
#     def _evaluate(self, x, out, *args, **kwargs):
#         f1 = inflex_mongolia_capacity_opt(x)
#         f2 = inflex_mongolia_c02_opt(x)

#         out["F"] = [f1, f2]


# problem = MyProblem()

# algorithm = NSGA2(pop_size=100)

# res = minimize(problem,
#                 algorithm,
#                 ("n_gen", 100),
#                 verbose=True,
#                 seed=1)

# # get the pareto-set and pareto-front for plotting
# ps = problem.pareto_set(use_cache=False, flatten=False)
# pf = problem.pareto_front(use_cache=False, flatten=False)


# # Objective Space
# plot = Scatter(title = "Objective Space")
# plot.add(res.F)
# if pf is not None:
#     plot.add(pf, plot_type="line", color="black", alpha=0.7)
# plot.show()




# print("Baseline CO2 emissions are in tonnes " , inflex_mongolia_co2_opt(pv0_ga))

#inflex_df = inflex_mongolia_df(x0)

# total_coal = inflex_df["Coal"].sum()
# total_c02 = inflex_df["CO2"].sum()
# print("Total Carbon used in kg is " , total_coal)
# print("Resulting in tonnes CO2 emitted of " , total_c02)

# from pymoo.util.running_metric import RunningMetric


# running = RunningMetric(delta_gen=10,
#                         n_plots=4,
#                         only_if_n_plots=True,
#                         key_press=False,
#                         do_show=True)

# for algorithm in res.history:
#     running.notify(algorithm)




#OTHER OLD OPTIMIZATION CODE FROM GAGARAGE EXMAPLE

#results['SLSQP']=  minimize(NPV_garage, plan_bh ,method='SLSQP', bounds = bnds, options={'disp':True,'maxiter':1001,'ftol':1E-5})   
# #results['shgo'] = optimize.shgo(NPV_garage, bnds)
# #results['DA'] = dual_annealing(NPV_garage, bounds=bnds, seed=1234)
# #results['Dual annealing '] = np.rint(results['DA'].x)
# results_GA = differential_evolution(ENPV_MC, bnds)
# results['GA round2'] = np.rint(results_GA.x)
# #results_GA_stoc = differential_evolution(ENPV_MC, bnds)
# #results['GA round stoc'] = np.rint(results_GA_stoc.x)

# #results_GA_BH = basinhopping(ENPV_MC, plan_bh, niter =100)
                                         
# #resbf =  optimize.brute(NPV_obj_array, ranges, finish =None)
# #results['Brute force'] = resbf
# #print(res3)
# #print ("Optimized DV is :" , result.x )
# #print( "leading to NPV of: " , result.fun)
# #print(result)

# NPV_GA_opt = NPV_garage(results['GA round2'])
# # print("leading to NPV of: " , NPV_GA_opt)
# print("leading to ENPV of: " , ENPV_MC(results['GA round2']))





# a =  inflex_mongolia_opt(500)





# cost_df1 = total_system_cost_determinstic_monthly_inflex (demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)

# cost_df2 = total_system_cost_determinstic_monthly_inflex_disc (demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)

# n_yr = 12
# cost_df_yearly = cost_df1.groupby(cost_df1.index //n_yr).sum()

# #HERE DEFINE YEAR 0 COSTS TO FILL PANDAS DATAFRAME AND MAKE MORE SENSE FROM GRAPH
# yr0_row_inflex = pd.DataFrame({'Total' : [0] , 'Capex' : [24300] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# #yr0_row_inflex2 = pd.DataFrame({'Total' : [0] , 'Capex' : [2430] , 'Mismatch' : [0], 'Opex' : [0] , 'Coal' : [0] })
# full_cost_df = pd.concat([yr0_row_inflex, cost_df_yearly]).reset_index(drop = True)



# full_cost_df.drop('Total' , inplace = True, axis = 1)
# full_cost_df.drop(full_cost_df.index[21] , inplace = True, axis = 0)


#full_cost_df["Coal"] = full_cost_df["Coal"]/1.1575

# full_cost_df["Capex"] = full_cost_df["Capex"]*2

# a = full_cost_df.plot.bar(stacked = True)
# a.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.5 , -.4))
# a.set_xlabel('Years')
# a.set_ylabel('Cost ($)')
# a.set_title('Scenario 1: Cumulative Cost Breakdown- Determinstic Baseline - Full System')


# full_cost_df["Capex"][10] = 1205
# full_cost_df["Capex"][15] = 13510


#inflex_monthly_cf_plot(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)
#inflex_yearly_cf_bar_det(demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)
