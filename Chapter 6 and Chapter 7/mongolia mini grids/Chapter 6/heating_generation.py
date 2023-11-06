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
from electricity_distribution import *

# Parameters
T=30 #years
Tm = 360
initial_capacity = 363 #W
initial_capacity_kw = .363 #kW

years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
hours = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20', '21' , '22', '23'] # hour 0 is midnight
hourly_demand = [0 , 0, 0, 0, 0, 0, 20, 20, 120, 120, 100,100, 100, 100, 100, 100, 100, 100,100, 120, 120, 120, 120, 120 ] # in Watts
months = list(range(0,Tm + 1)) # CHECK later if this should be increased to 241

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


def max_output_electric_heater_monthly(heater_capacity): # this is assuming Watts input from RL, output is in kWh to remain consitent
    output = heater_capacity * 24 * 30 * .001
    return output 

def eh_heat_inflex( eh_capacity, electricity_generated , electricity_demand):
    max_eh_output = max_output_electric_heater_monthly(eh_capacity)
    if electricity_generated > electricity_demand:
        eh_heat = electricity_generated - electricity_demand
    else: eh_heat = 0
    return eh_heat


#ensure coal HHV is in kj
def monthly_coal_requirement(heat_demand, eh_heat,  coal_HHV,  stove_efficiency):
    heat_input = kw_2_kj_perh / stove_efficiency # kj per kw hr
    coal_mass_per_kwh = heat_input / coal_HHV
    if eh_heat < heat_demand:
        coal_heat_demand = heat_demand - eh_heat
    else : coal_heat_demand = 0
    coal_mass_total = coal_mass_per_kwh * coal_heat_demand
    return coal_mass_total # in kilograms


def monthly_coal_expenditure(heat_demand, eh_heat, coal_HHV,  stove_efficiency, coal_price):
    coal_input = monthly_coal_requirement(heat_demand, eh_heat, coal_HHV,  stove_efficiency)
    coal_cost = coal_input * coal_price  
    return coal_cost # in kilograms

def coal_mass_perkwh(coal_HHV,  stove_efficiency):
    heat_input = kw_2_kj_perh / stove_efficiency # kj per kw hr
    coal_mass_per_kwh = heat_input / coal_HHV
    return coal_mass_per_kwh # in kilograms



def coal_kg_to_kwh(coal_mass):
    c_kg_to_kwh = coal_mass_perkwh(coal_HV_kj,  eff_trad_stove)
    coal_kwh = coal_mass / c_kg_to_kwh
    return coal_kwh


# a = coal_mass_perkwh(coal_HV_kj,  eff_trad_stove)
# print(a)
# a2 = coal_mass_perkwh(coal_HV_kj,  eff_improved_stove_l)
# print(a2)
# a3 = coal_mass_perkwh(coal_HV_kj,  eff_improved_stove_h)
# print(a3)





#CALACULATE PRODUCTION FROM ELECTRIC HEATERSA, MINUS ELECTRICITY FOR OTHER USES
#DETERMINE UNMET HEATING DEMAND
#DURING WINTER MONTHS CAN BUY ELECTRICITY FROM GER DISTRICT
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




def output_electric_heater_monthly_kwh(heat_demand, heater_capacity , extra_electricity): # this is assuming Watts input capacity from RL, output anfd other inputs is in kWh to remain consitent
    max_output_eh = heater_capacity * 24 * 30 * .001
    if extra_electricity >= 0:
        if heat_demand > max_output_eh :
            output = np.min(np.array([max_output_eh , extra_electricity]))
            leftover_posteh_electricity = 0
        else: 
            output = heat_demand 
            leftover_posteh_electricity = extra_electricity - heat_demand
    else: output , leftover_posteh_electricity = 0, 0 
    return output , leftover_posteh_electricity


# a = max_output_electric_heater_monthly(500)
# a1 = output_electric_heater_monthly_kwh(1000 ,500, 10 )
# print(a)
# print(a1)



# a = randomised_cf_norm_month(1, CF_pv_monthly,CF_pv_monthly_dev)

#CREATE FUNCTION FOR PLUG AND PLAY ELECTRICITY PURCHASE
summer_indexes = [4,5,6,7]

def grid_electricity_interaction(heat_generated_eh, leftover_heat_demand,  leftover_posteh_electricity, eh_capacity,   time_step):
    max_output_eh = eh_capacity * 24 * 30 * .001
    pp_tariff_offgrid = .015
    pp_tariff_grid = .07
    ub_kwh_purchase_price = .049
    month_index =  monthly_index_fromtimestep(time_step)
    line_capacity  =2 # kW
    max_line_output = line_capacity  * 8 * 30 # assuming draewing heating power 6 hours per day
    remaining_eh_capacity  =  np.max([max_output_eh - heat_generated_eh , 0])
    simulated_load_shed = 0
    if month_index in summer_indexes:
        purchase = 0
        sellback = leftover_posteh_electricity * pp_tariff_offgrid # #ENSURE SAME UNITS USED HERE
        net_import = 0
        net_cf = sellback - purchase
    else:
        if leftover_heat_demand > 0:
            net_import = np.min([leftover_heat_demand, max_line_output, remaining_eh_capacity ])
            simulated_load_shed = load_shedding_simulation()
            net_import = (1-simulated_load_shed) * net_import 
            purchase = net_import * ub_kwh_purchase_price
            sellback  = leftover_posteh_electricity * pp_tariff_grid
            net_cf = sellback - purchase
        else: 
            net_import = 0
            purchase= 0
            sellback = leftover_posteh_electricity * pp_tariff_offgrid
            net_cf = sellback - purchase
    return net_cf, net_import, simulated_load_shed



# eh heat res here refers to how much capacity been used
def output_electric_heater_monthly_kwh_inflex( heater_capacity , eh_heat_res , leftover_heat_demand_postres): # this is assuming Watts input capacity from RL, output anfd other inputs is in kWh to remain consitent
    max_output_eh = heater_capacity * 24 * 30 * .001
    line_capacity  =1.5 # kW
    max_line_output = line_capacity  * 6 * 30 # assuming draewing heating power 6 hours per day i nkWh
    eh_grid_output_max = max_output_eh - eh_heat_res
    leftover_heat_demand_postgrid = np.max([(leftover_heat_demand_postres - eh_grid_output_max) , 0])
    return eh_grid_output_max , leftover_heat_demand_postgrid


def output_electric_heater_monthly_kwh_inflex_2( heater_capacity , extra_electricity , leftover_heat_demand_postres): # this is assuming Watts input capacity from RL, output anfd other inputs is in kWh to remain consitent
    max_output_eh = heater_capacity * 24 * 30 * .001
    line_capacity_1ger  =1.5 # kW
    line_capacity = line_capacity_1ger*10
    max_line_output = line_capacity  * 6 * 30 # assuming draewing heating power 6 hours per day i nkWh
    eh_grid_output_max = np.max([max_output_eh - extra_electricity , 0])
    eh_grid_output = np.min([ max_line_output, eh_grid_output_max, leftover_heat_demand_postres ])
    leftover_heat_demand_postgrid = np.max([(leftover_heat_demand_postres - eh_grid_output) , 0])
    return eh_grid_output , leftover_heat_demand_postgrid
    
 # purchase kept positive as dealing with all costs
def grid_electricity_interaction_inflex(eh_heat_res, starting_eh_capacity, heating_demand , time_step ):
    line_capacity  =1.5 # kW
    max_line_output = line_capacity  * 6 * 30 # assuming draewing heating power 6 hours per day
    pp_tariff_offgrid = .015
    pp_tariff_grid = .07
    ub_kwh_purchase_price = .049 #THIS SUPER HIGH BUT JUST SEEING IMPACT ON OPTIMIZATION RESULT
    month_index =  monthly_index_fromtimestep(time_step)
    leftover_heat_demand_postres = np.max([heating_demand - eh_heat_res , 0]) # after meeting with renewable microgrid capacity
    if month_index in summer_indexes:
        purchase = 0
        net_import = 0
        net_cf =  purchase
    else:
        eh_grid_output , leftover_heat_demand_postgrid = output_electric_heater_monthly_kwh_inflex(starting_eh_capacity , eh_heat_res , leftover_heat_demand_postres)
        purchase = leftover_heat_demand_postgrid * ub_kwh_purchase_price
        net_import = np.min([eh_grid_output , max_line_output])
        net_cf =  purchase
    return net_cf, net_import


 # purchase kept positive as dealing with all costs
def grid_electricity_interaction_inflex_2(extra_produced_electricity, starting_eh_capacity, heating_demand , time_step ):
    max_output_eh = starting_eh_capacity * 24 * 30 * .001
    # print(max_output_eh)
    line_capacity_1ger  =1.5 # kW
    line_capacity = line_capacity_1ger*10
    max_line_output = line_capacity  * 6 * 30 # assuming draewing heating power 6 hours per day i nkWh
    pp_tariff_offgrid = .015
    pp_tariff_grid = .07
    ub_kwh_purchase_price = .049 
    month_index =  monthly_index_fromtimestep(time_step) # after meeting with renewable microgrid capacity
    net_cf, net_import,net_eh_res, simulated_load_shed = 0,0 , 0, 0
    if extra_produced_electricity > heating_demand:
        net_cf, net_import, net_eh_res = 0, 0 , heating_demand
    else:
        if month_index in summer_indexes:
            purchase = 0
            net_import = 0
            net_cf =  purchase
            net_eh_res = 0
        else:
            if extra_produced_electricity > 0 :
                net_eh_res = np.min([extra_produced_electricity, max_output_eh])
                leftover_heat_demand_postres = heating_demand - extra_produced_electricity 
                leftover_eh_capacity = np.max([max_output_eh - extra_produced_electricity  , 0])
                net_import = np.min([max_line_output , leftover_eh_capacity,  max_output_eh, leftover_heat_demand_postres ])
                simulated_load_shed = load_shedding_simulation()
                net_import = (1-simulated_load_shed) * net_import
                purchase = net_import * ub_kwh_purchase_price
            else: 
                net_import = np.min([max_line_output ,  max_output_eh, heating_demand])
                simulated_load_shed = load_shedding_simulation()
                net_import = (1-simulated_load_shed) * net_import
                purchase = net_import * ub_kwh_purchase_price
                net_cf =  purchase
                net_eh_res = 0
    return net_cf, net_import , net_eh_res ,simulated_load_shed

     

#print(grid_electricity_interaction(150, 0, 200))



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






# assuming here that if cf are above avg no need for transmitting, only in period where one below
# this is significant simplification and should find better way to do this
# not even including diesel but implied i suppose
# def standardised_res_cf_difference(pv_cf, wind_cf): # should probly update this so uses average of EVERY MONTH instead, can do later
#     #this is for estimating tranmission requirement, LOOK AGAIN
#     max_tranmission_ratio = .2
#     pv_cf_ratio = pv_cf / CF_pv_avg
#     wind_cf_ratio = wind_cf / CF_wind_avg
#     if  pv_cf_ratio < 1 or wind_cf_ratio < 1:
#         larger_mismatch = np.min(np.array([pv_cf_ratio,wind_cf_ratio] )) # the minimum will be furthest from its average
#         power = (1 - larger_mismatch)
#         if power >= max_tranmission_ratio:
#             power = max_tranmission_ratio
#         else: power = power
#     else : power = 0 
#     return power 

#standardised_res_cf_difference(.23, .38)


# a = np.min(np.array([1,2]))

# def cf_histogram(CF_avg, CF_dev):
#     data_normal = norm.rvs(size=1000,loc=CF_avg, scale= CF_dev)
#     fig, bx = plt.subplots(figsize=(8, 4))
#     a= bx.hist(data_normal, bins = 100)
#     bx.set_xlabel('Annual Average Capacity Factor')
#     bx.set_ylabel('# occurences out of 1000')
#     bx.set_title('Capacity Factor Probability Distribution Mongolia')

# def electricity_provided_diesel(diesel_capacity, shortage_over5pct):
#     max_theorethical_yearly_production = diesel_capacity * 8760 * .001 # convert from Wh to kWh
#     if diesel_capacity == 0:
#         cf = 0 
#         provided = 0
#     elif shortage_over5pct > max_theorethical_yearly_production:
#         provided = max_theorethical_yearly_production
#         cf = 1
#     else : 
#         provided = shortage_over5pct
#         cf = shortage_over5pct/ max_theorethical_yearly_production
#     return provided , cf 

# def electricity_provided_diesel_monthly(diesel_capacity, shortage_over5pct):
#     max_theorethical_monthly_production = diesel_capacity * 24 * 30 * .001 # convert from Wh to kWh
#     if diesel_capacity == 0:
#         cf = 0 
#         provided = 0
#     elif shortage_over5pct > max_theorethical_monthly_production:
#         provided = max_theorethical_monthly_production
#         cf = 1
#     else : 
#         provided = shortage_over5pct
#         cf = shortage_over5pct/ max_theorethical_monthly_production
#     return provided , cf 



