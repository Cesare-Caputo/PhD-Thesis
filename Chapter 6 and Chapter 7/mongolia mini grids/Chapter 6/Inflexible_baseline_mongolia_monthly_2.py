# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:33:19 2021

@author: cesa_
"""
#THIS VERSION WAS CREATED A SORIGINAL WAS DOING SOME VERY WEIRD BUGS


from electricity_demand import *
from minigrid_cost_rl import *
from electricity_generation import *
from heating_demand import *
from heating_generation import*
import pandas as pd
import numpy as np

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

# average NIGHTIME temperatures Iin celsius to determine relative heat demand each month
#essential assuming heat demand is proportional to nightime temp, reasonable
month_temp = np.array([-26, -24, -14, -4 , 4, 6, 12, 10, 3, -6, -15, -22])
avg_heat_season_temp = -13.4 # celsius

  
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


# generate deterministic demand and migration cluster radius profile and heating , demand values are foe 10 gers
demand_projections = demand_static_series_months_ngers(Tm, n_gers)
# cluster_radius_projections = migration_cluster_radius_static_series(Tm)

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



heating_demand_projections = monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand_standard_ger_perm2, area_ger_5_walls,10)



def total_system_cost_determinstic_monthly (demand_series, heating_demand_series, cf_pv_month, coal_price, n_gers):
    #MAYBE SHOULD BUILD THIS AS DATAFRAME SO WE HAVE CASH F;LOW MODEL FOR DEBUGGING
    month_cost = pd.Series(index=months , dtype= 'float64')
    eh_heat = 0 # inflexible , no electric heaters available
    cost = 0 
    coal_cost_s = pd.Series(index=months , dtype= 'float64')
    mismatch_cost_s = pd.Series(index=months , dtype= 'float64')
    mismatch_kwh = 0
    coal_mass = 0 
    for i in range(Tm + 1):
        month_index = monthly_index_fromtimestep(i)
        electricity_demand = demand_series[i]
        heat_demand = heating_demand_series[month_index]
        CF_yr = cf_pv_month[month_index]
        coal_mass += monthly_coal_requirement(heat_demand, eh_heat,  coal_HV_kj,  eff_improved_stove_l)
        coal_cost = monthly_coal_expenditure(heat_demand, eh_heat, coal_HV_kj,  eff_improved_stove_l, coal_price)
        static_capacity_pv_ngers = static_capacity_pv * n_gers
        electricity_generated_monthly_kwh = normalised_monthly_stoch_electricity_generation_kwh (CF_yr, static_capacity_pv_ngers)
        mismatch_kwh += shortage_over5pct(electricity_generated_monthly_kwh, electricity_demand)
        mismatch = mismatch_penalty(electricity_generated_monthly_kwh, electricity_demand)
        mismatch_cost_s[i] = mismatch
        coal_cost_s[i]= coal_cost
        if i == Tm + 1:
            month_cost[i] = mismatch - pv_salvage_value_yr20 + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery) + coal_cost
        else:
            month_cost[i] = mismatch + monthly_opex_inflexible(static_capacity_pv, static_capacity_battery) + coal_cost
        cost += month_cost[i] / ((1+r_month)**i) # discounting using discount rate above
    return cost , coal_cost_s , coal_mass, mismatch_cost_s , mismatch_kwh



total_cost, coal_cost, coal_mass, mismatch_cost , mismatch_amount = total_system_cost_determinstic_monthly (demand_projections, heating_demand_projections, CF_pv_monthly, coal_price_per_kg, n_gers)





print(total_cost)
print(mismatch_amount)
print(coal_mass)

plt.figure(figsize=(12,5))
plt.xlabel('Months')
plt.ylabel('Monthly Cost(USD)')
plt.title('Projected cost contributions from Coal and Unmet Load for 10 gers(@ Penalty of .3417 USD/kWh) ')
ax1=coal_cost.plot(color='blue', grid=True, label=' Coal ')
ax2=mismatch_cost.plot(color='orange', grid=True, label=' Mismatch')
# ax3=test2.plot(color='red', grid=True, label=' Stochastic 2')
# ax4=test3.plot(color='green', grid=True, label=' Stochastic 3')
# ax5=stochastic_demand_extreme.plot(color='yellow', grid=True, label=' Stochastic more 1')
# ax5=stochastic_demand_extreme_2.plot(color='magenta', grid=True, label=' Stochastic more 2')
ax1.legend(loc=2)
# ax2.legend(loc=2)
# ax3.legend(loc=2)
plt.show()

