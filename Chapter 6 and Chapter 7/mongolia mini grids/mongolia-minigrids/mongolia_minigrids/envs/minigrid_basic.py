# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:12 2020

@author: cesa_
"""
import gym
from gym import spaces
from electricity_demand import *
from electricity_distribution import *
from minigrid_cost_rl import *
from electricity_generation import *
from herder_migration import *
from heating_demand import *
from heating_generation import *
#from garage_ENPV_obj_arrayinput import cc_start
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding

# define here number of gers considered in system analysis
n_gers = 10

# Parameters
T=20 #years
Tm = 240 # months in 20 years
T_monthly = 240
n_month_peryr = 12
r_yr = 0.06# Discount rate
r_month = ((1+r_yr)**(1/12)) - 1 
CF_pv_avg = .20 # average capacity factor for mongolia solar , using 25 as this is value used in HOMER calculations even though ADB report states around 18.5
CF_pv_dev = .01 # standard deviation of CF, assumed here but can be confirmed later
CF_wind_avg = .35 # average capacity factor for mongolia wind
CF_wind_dev = .01 # standard deviation of CF, assumed here but can be confirmed later    
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152]
CF_wind_monthly = [ .255, .262, .295, .370, .385, .322, .310, .224, .273, .241, .256, .278]

# heating demand parameters FOCUS ON STANDARD FIRST, THEN INCLUDE DECISION RULE FOR NOT
month_temp = np.array([-26, -24, -14, -4 , 4, 6, 12, 10, 4, -6, -15, -22])
# going to look at 5 wall units only for now, maybe can include 8 too later
n_lattice_walls = 5
area_ger_5_walls = 29.3 # m2
area_ger_8_walls = 72.6 # m2
yearly_demand_standard_ger_perm2 = 393 #/ kWh / m2/ yr
yearly_demand_improved_ger_perm2 = 206 #/ kWh / m2/ yr
heating_months = 8 # September 1 to May 1 each year





years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
k = pd.Series(index=years)

     
#Actions , each expansion refers to 500W increase
#could also have one to buy better cabling to reduce losses
NOTHING = 0
EXPAND_PV = 1
EXPAND_WIND = 2
EXPAND_BATTERY = 3
EXPAND_DIESEL = 4
EXPAND_EH = 5
# EXPAND_INSULATION = 6
# WAIT TO INTEGRATE INSULATION EXPANSION TILL LATER WHEN EH SECTION WORKING


# generate deterministic demand and migration cluster radius profile and heating , demand values are foe 10 gers
demand_projections = demand_static_series_months_ngers(Tm, n_gers)
cluster_radius_projections = migration_cluster_radius_static_series(Tm)
heating_demand_projections =  monthly_heat_demand_fromtemp_ngers(month_temp , yearly_demand_standard_ger_perm2, area_ger_5_walls, n_gers)




# define cost of cabling here as using arbitrary value for now
cost_cable_perm = 1 # USD per meter, arbitrary


# coal stove technical parameters
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




# environment data
max_monthly_demand = np.max(demand_projections)# in kWh
min_monthly_demand = np.min(demand_projections)# in kWh
max_monthly_heat_demand = np.max(heating_demand_projections)# in kWh
min_monthly_heat_demand = np.min(heating_demand_projections)# in kWh
max_eh_capacity = 15000 * n_gers #  in Watts
min_eh_capacity = 0 * n_gers # in Watts
max_pv_capacity = 1800 * n_gers #  in Watts
min_pv_capacity = 363 * n_gers # in Watts
max_battery_capacity = 20000 * n_gers #  in Watts
min_battery_capacity = 2000 * n_gers # in Watts
max_wind_capacity = 20000 * n_gers #  in Watts
min_wind_capacity = 0 * n_gers # in Watts
max_diesel_capacity = 20000 * n_gers #  in Watts
min_diesel_capacity = 0 * n_gers # in Watts
max_inverter_capacity = 2000 * n_gers #  in Watts
min_inverter_capacity = 237 * n_gers # in Watts
max_time_left_eps = 240 # months left to project end at starting period 
min_time_left_eps = 0 # months left in last peripod
min_cluster_radius =  np.min(cluster_radius_projections)# meters for area encompassing 10 gers
max_cluster_radius = np.max(cluster_radius_projections)# meters for area encompassing 10 gers
min_cabling_lenght = cable_lenght_from_radius(min_cluster_radius) # cable lenght to connect 10 gers based on min cluster radius
max_cabling_lenght = cable_lenght_from_radius( max_cluster_radius) # cable lenght to connect 10 gers based on max cluster 

# HERE WE ARE DEFINING THE PROBLEM BASED ON A CLUSTER OF 10 MINIGRIDS TOGETHER, SO DEMAND AND CAPACITIES ALL MULTIPLIED BY 10
# this version of environment is fully DETERMINISTIC


#STILL NEED TO IMPLEMENT HEATING IN ENVIRONMENT AND OTHER 

class MinigridBasic(gym.Env):
    def __init__(self , mismatch_cost): # remember to update all other cost formulas, including in inflexible baseline to reflect this
        # Demand and capacity data
        self.low = np.array([min_monthly_demand, min_monthly_heat_demand , min_eh_capacity,  min_pv_capacity, min_battery_capacity, min_wind_capacity,  min_diesel_capacity, min_inverter_capacity, min_time_left_eps, min_cluster_radius, min_cabling_lenght], dtype=np.float32)
        self.high = np.array([max_monthly_demand, max_monthly_heat_demand , max_eh_capacity, max_pv_capacity, max_battery_capacity, max_wind_capacity,  max_diesel_capacity, max_inverter_capacity, max_time_left_eps, max_cluster_radius, max_cabling_lenght], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        # self.observation_space = spaces.Dict({"demand": spaces.Discrete(2), "pv_capacity": spaces.Discrete(3)})
        self.action_space = spaces.Discrete(6) # actions listed above, inverter capacity not considered direct action
        # LATER could look to see advantages of buying oversized inverter from project start
        # ALSO IN NEXT ITERATION IMPLEMENT MULTI DISCRETE SO FOR EACH TECHNOLOGY AT LEAST 3 EXPANSION LEVELS PRESENT
        self.count = 0
        self.time_steps = 0
        self.starting_time_left = max_time_left_eps
        self.starting_pv_capacity = min_pv_capacity 
        self.starting_eh_capacity = min_eh_capacity
        self.starting_battery_capacity = min_battery_capacity 
        self.starting_wind_capacity = min_wind_capacity 
        self.starting_diesel_capacity = min_diesel_capacity 
        self.starting_diesel_capacity = min_diesel_capacity 
        self.starting_inverter_capacity = min_inverter_capacity
        self.starting_cluster_radius = min_cluster_radius
        self.starting_cable_lenght = min_cabling_lenght
        #set random demand curve evolution profile
        self.starting_heat_demand = heating_demand_projections[self.time_steps]
        self.starting_demand = demand_projections[self.time_steps]
        self.state = self.starting_demand ,self.starting_heat_demand,  self.starting_eh_capacity, self.starting_pv_capacity , self.starting_battery_capacity , self.starting_wind_capacity, self.starting_diesel_capacity, self.starting_inverter_capacity, self.starting_time_left, self.starting_cluster_radius , self.starting_cable_lenght

    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        
        current_demand, current_heat_demand, current_eh_capacity, current_pv_capacity, current_battery_capacity, current_wind_capacity, current_diesel_capacity, current_invert_capacity, current_time_left, current_cluster_radius, current_cable_lenght = self.state
        if action == 0:
            E_cost = 0 # this stands for expansion cost
        elif action == 1:
           current_pv_capacity = current_pv_capacity + 500
           E_cost = expansion_cost_pv (action) 
        elif action == 2:
           current_battery_capacity = current_battery_capacity + 500
           E_cost = expansion_cost_battery (action) 
        elif action == 3:
           current_wind_capacity = current_wind_capacity + 500
           E_cost = expansion_cost_wind (action) 
        elif action == 4:
           current__diesel_capacity = current_diesel_capacity + 500
           E_cost = expansion_cost_diesel (action) 
        elif action == 5:
           current_eh_capacity = current_eh_capacity + 500
           E_cost = expansion_cost_eh (action) 
        
        #if action not battery increase or nothiung
        if action != 0 or 2 or 4 or 5 : 
            capacity_expanded_total = 500 # W
            extra_inverter_capacity, extra_inverter_cost = expansion_impact_inverter(capacity_expanded_total)
            current_invert_capacity = current_invert_capacity + extra_inverter_capacity
            E_cost = E_cost + extra_inverter_cost
        
        
        
        
        #bring forward time step to assess impact of action on year t operation
        self.time_steps +=1 
        current_time_left = max_time_left_eps - self.time_steps
        month_index = monthly_index_fromtimestep(self.time_steps)
        # evaluate uncertain parameters
        self.current_demand = demand_projections[self.time_steps] # bring forward timestep, action chosen before this realization but rewards determined by this
        self.current_heating_demand = heating_demand_projections[month_index]
        self.current_cluster_radius = cluster_radius_projections[self.time_steps] 
        # determine need for extra cabling and cost associated
        extra_cabling = check_additional_cabling_reqs(self.current_cluster_radius, current_cable_lenght) # this calculates any additional cabling requirmeents based on change in cluster radius
        cabling_exp_cost = extra_cabling_cost (extra_cabling, cost_cable_perm) 
        self.current_cable_lenght = current_cable_lenght + extra_cabling # update 
        
        
        # assume negligible build times for capacity increase, decision made at start of month and thus that capacity available for generation that month
        self.current_eh_capacity, self.current_pv_capacity, self.current_battery_capacity, self.current_wind_capacity, self.current_diesel_capacity, self.current_invert_capacity, self.time_left = current_eh_capacity, current_pv_capacity, current_battery_capacity, current_wind_capacity, current_diesel_capacity, current_invert_capacity, current_time_left
        self.state = (self.current_demand, self.current_heating_demand, self.current_eh_capacity, self.current_pv_capacity, self.current_battery_capacity, self. current_wind_capacity, self.current_diesel_capacity, self.current_invert_capacity, self.time_left, self.current_cluster_radius, self.current_cable_lenght)
        
        
        
        
        # this capacity factor section needs to be updated with right values but functinality is GOOD
        pv_cf = CF_pv_monthly[month_index]
        wind_cf = CF_wind_monthly[month_index]
        electricity_generated_monthly_kwh_pv = normalised_monthly_stoch_electricity_generation_kwh(pv_cf, self.current_pv_capacity)
        electricity_generated_monthly_kwh_wind =  normalised_monthly_stoch_electricity_generation_kwh(wind_cf, self.current_wind_capacity)
        electricity_generated_monthly_kwh_RES = electricity_generated_monthly_kwh_pv + electricity_generated_monthly_kwh_wind
        avg_tranmission_req = standardised_res_cf_difference(pv_cf, wind_cf) * self.current_demand # this is in kWh so that losses also in kWh right away
        tranmission_losses = cable_energy_losses_3ph(avg_tranmission_req, self.current_cable_lenght) # this also comes out in kWh
        net_electricity_deliverd_monthly_RES = electricity_generated_monthly_kwh_RES - tranmission_losses
        # for now this is simply difference between renewable generation and demand
        # need to look more closely at impact of batteries on this
        #ASSUMING DIESEL NOT BE USED FOR ELECTRIC HEATING AS VERY INEFICCIENT, ONLY FOR UNMET ELECTRIC LOAD
        # THIS PROBABLY NEEDS TO BE CHANGED LATER TO BE MORE REFLECTIBVE OF CONDITIONS
        electricity_required_diesel = shortage_over5pct(net_electricity_deliverd_monthly_RES, self.current_demand)
        electricity_generated_diesel, cf_diesel = electricity_provided_diesel_monthly(self.current_diesel_capacity, electricity_required_diesel)
        total_electricity_delivered = electricity_generated_diesel + net_electricity_deliverd_monthly_RES
        mismatch = self.current_demand - total_electricity_delivered #REMEMBER NEGATIVE MISMATCH MEANS MORE PRODUCTION THAN DEMAND
        shortage = shortage_over5pct(total_electricity_delivered, self.current_demand)
        mismatch_pct = mismatch / self.current_demand
        mismatch_cost = mismatch_penalty(total_electricity_delivered, self.current_demand)
        extra_electricity = total_electricity_delivered - self.current_demand
        
        # look at heating requirements here
        heat_generated_eh = output_electric_heater_monthly_kwh(self.current_heating_demand, self.current_eh_capacity, extra_electricity  )
        # next calculate amount of coal required to satisfy leftover heating demand
        coal_requirement_for_heat = monthly_coal_requirement(self.current_heating_demand, heat_generated_eh,  coal_HV_kj,  eff_trad_stove)
        
        

        system_opex = monthly_opex_system (self.current_pv_capacity, self.current_battery_capacity, self.current_wind_capacity,  electricity_generated_diesel, self.current_invert_capacity, coal_requirement_for_heat)
        coal_expenditure = coal_price_per_kg * coal_requirement_for_heat
        
        
       # calculate total costs for this year    
        tot_cost = mismatch_cost + E_cost + system_opex + cabling_exp_cost + coal_expenditure
        reward = - (tot_cost / ((1+r_month) ** self.time_steps))
        

        
        if self.time_steps == T_monthly:
            done = True
            salvage_value_pv = salvage_pv(self.current_pv_capacity, (self.time_steps /n_month_peryr) ) #the .1 is added to see if it helps agent ovetrbuild less
            salvage_value_wind = salvage_wind(self.current_wind_capacity,( self.time_steps / n_month_peryr))
            # this is over simplification assuming all PV/wind has been used for WHOLE project duration, meaning it is underestimating salvage value
            reward = reward + salvage_value_pv + salvage_value_wind 
        else :
            done = False
    
        info = {"CF PV":pv_cf , "CF Wind": wind_cf , "Electric Heater usage(kWh)": heat_generated_eh,   "Tranmission losses (kWh)" : tranmission_losses , "Distributed Energy(kWh)" : avg_tranmission_req, "Shortage amount (kWh)" : shortage ,   "Mismatch Cost" : mismatch_cost, "Diesel use (kWh)" :  electricity_generated_diesel , "Coal Used(kg)" : coal_requirement_for_heat , "System cost" : tot_cost , "Coal cost" : coal_expenditure ,"Expansion Capex" : E_cost , "Opex" : system_opex}
        #observation returned is capacity at end of period, but reward calculation done with previous one
        # however can probably assume negligible expansion time here since just purchase of equipment
        return np.array(self.state), reward, done, info
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.time_steps = 0
        # self.current_demand, self.current_heating_demand, self.current_eh_capacity, self.current_pv_capacity, self.current_battery_capacity, self. current_wind_capacity, self.current_diesel_capacity, self.current_invert_capacity, self.time_left, self.current_cluster_radius, self.current_cable_lenght = self.starting_pv_capacity , self.starting_battery_capacity , self.starting_wind_capacity, self.starting_diesel_capacity, self.starting_inverter_capacity
        # self.current_demand = demand_projections[self.time_steps]
        self.state = (self.starting_demand ,self.starting_heat_demand,  self.starting_eh_capacity, self.starting_pv_capacity , self.starting_battery_capacity , self.starting_wind_capacity, self.starting_diesel_capacity, self.starting_inverter_capacity, self.starting_time_left, self.starting_cluster_radius , self.starting_cable_lenght)
        return np.array(self.state)
    
    

    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy either 0 or 1.
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1, 2, 3, 4 , 5])
    
    return chosen_action 




