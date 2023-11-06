# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:12 2020

@author: cesa_
"""
import gym
from gym import spaces
#from garage_ENPV_obj_arrayinput import cc_start
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding
import gym
from gym import spaces
from electricity_demand import *
from electricity_distribution import *
from minigrid_cost_rl import *
from electricity_generation import *
from herder_migration import *
#from garage_ENPV_obj_arrayinput import cc_start
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding


years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
k = pd.Series(index=years)


# define here number of gers considered in system analysis
n_gers = 10

# Parameters
T_yr=20 #years
n_month_peryr = 12
T_monthly = T_yr * n_month_peryr
r_yr = 0.06# Discount rate
r_month = ((1+r_yr)**(1/12)) - 1 
CF_pv_avg = .20 # average capacity factor for mongolia solar , using 25 as this is value used in HOMER calculations even though ADB report states around 18.5
CF_pv_dev = .01 # standard deviation of CF, assumed here but can be confirmed later
CF_wind_avg = .35 # average capacity factor for mongolia wind
CF_wind_dev = .01 # standard deviation of CF, assumed here but can be confirmed later    
CF_pv_monthly = [ .155, .16, .162, .178, .193, .234, .221, .224, .193, .174, .165, .152] # reconfirm these values before final submission
CF_pv_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]
CF_wind_monthly = [ .255, .262, 295, .370, .385, .322, .310, .224, .273, .241, .256, .278]
CF_wind_monthly_dev = [.05, .04, .035, .03, .015, .01, .01, .011, .013, .027, .029, .032]
     
#Actions , each expansion refers to 500W increase
NOTHING = 0
EXPAND_PV = 1
EXPAND_WIND = 2
EXPAND_BATTERY = 3
EXPAND_DIESEL = 4

# environment data
max_monthly_demand = 180  *n_gers# in kWh
min_monthly_demand = 40 * n_gers# in kWh
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
min_cluster_radius = 100 # meters for area encompassing 10 gers
max_cluster_radius = 2000 # meters for area encompassing 10 gers
min_cabling_lenght = cable_lenght_from_radius(min_cluster_radius) # cable lenght to connect 10 gers based on min cluster radius
max_cabling_lenght = cable_lenght_from_radius( max_cluster_radius) # cable lenght to connect 10 gers based on max cluster radius



# define cost of cabling here as using arbitrary value for now
cost_cable_perm = 1 # USD per meter, arbitrary

#INCLUDE NEGATIV REWARD PENALTIES FOR ILLEGAL ACTIONS

# this is the same as minigrid basic except that time resolution is increased to monthly

class MinigridMedium(gym.Env):

    def __init__(self, mismatch_cost):
        # Demand and capacity data
        self.low = np.array([min_monthly_demand,min_pv_capacity, min_battery_capacity, min_wind_capacity,  min_diesel_capacity, min_inverter_capacity, min_time_left_eps, min_cluster_radius, min_cabling_lenght], dtype=np.float32)
        self.high = np.array([max_monthly_demand,max_pv_capacity, max_battery_capacity, max_wind_capacity,  max_diesel_capacity, max_inverter_capacity, max_time_left_eps, max_cluster_radius, max_cabling_lenght], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        # self.observation_space = spaces.Dict({"demand": spaces.Discrete(2), "pv_capacity": spaces.Discrete(3)})
        self.action_space = spaces.Discrete(5) # actions listed above, inverter capacity not considered direct action
        # LATER could look to see advantages of buying oversized inverter from project start
        # ALSO IN NEXT ITERATION IMPLEMENT MULTI DISCRETE SO FOR EACH TECHNOLOGY AT LEAST 3 EXPANSION LEVELS PRESENT
        self.time_steps = 0
        self.starting_pv_capacity = min_pv_capacity 
        self.starting_battery_capacity = min_battery_capacity 
        self.starting_wind_capacity = min_wind_capacity 
        self.starting_diesel_capacity = min_diesel_capacity 
        self.starting_diesel_capacity = min_diesel_capacity 
        self.starting_inverter_capacity = min_inverter_capacity
        self.starting_time_left = max_time_left_eps
        #set random demand curve evolution profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        self.starting_demand = electricity_monthly_demand_stochastic_less_ngers(n_gers, self.time_steps ,self.rD0s, self.rD10s, self.rDfs)
        # set uncertain cluster radius migration profile just to make sure demand and radius are indipendent, this determines evolution over time
        self.rD0r = np.random.random_sample()
        self.rD10r = np.random.random_sample()
        self.rDfr = np.random.random_sample()
        self.starting_cluster_radius = migration_cluster_radius(self.time_steps ,self.rD0r, self.rD10r, self.rDfr)
        self.starting_cable_lenght = cable_lenght_from_radius(self.starting_cluster_radius)
        self.state = self.starting_demand , self.starting_pv_capacity , self.starting_battery_capacity , self.starting_wind_capacity, self.starting_diesel_capacity, self.starting_inverter_capacity, self.starting_time_left, self.starting_cluster_radius , self.starting_cable_lenght

    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        current_demand, current_pv_capacity, current_battery_capacity, current_wind_capacity, current_diesel_capacity, current_invert_capacity, current_time_left, current_cluster_radius, current_cable_lenght = self.state
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
        
        #if action not battery increase or nothiung
        if action != 0 or 2 or 4 : 
            capacity_expanded_total = 500 # W
            extra_inverter_capacity, extra_inverter_cost = expansion_impact_inverter(capacity_expanded_total)
            current_invert_capacity = current_invert_capacity + extra_inverter_capacity
            E_cost = E_cost + extra_inverter_cost
        
        
        
        
        #bring forward time step to assess impact of action on year t operation
        self.time_steps +=1 
        current_time_left = max_time_left_eps - self.time_steps
        # evaluate uncertain parameters
        self.current_demand = electricity_monthly_demand_stochastic_less_ngers(n_gers, self.time_steps,self.rD0s, self.rD10s, self.rDfs) # bring forward timestep, action chosen before this realization but rewards determined by this
        self.current_cluster_radius = migration_cluster_radius(self.time_steps ,self.rD0r, self.rD10r, self.rDfr)
        # determine need for extra cabling and cost associated
        extra_cabling = check_additional_cabling_reqs(self.current_cluster_radius, current_cable_lenght) # this calculates any additional cabling requirmeents based on change in cluster radius
        cabling_exp_cost = extra_cabling_cost (extra_cabling, cost_cable_perm) 
        self.current_cable_lenght = current_cable_lenght + extra_cabling # update 
        # assume negligible build times for capacity increase, decision made at start of month and thus that capacity available for generation that month
        self.current_pv_capacity, self.current_battery_capacity, self.current_wind_capacity, self.current_diesel_capacity, self.current_invert_capacity, self.time_left = current_pv_capacity, current_battery_capacity, current_wind_capacity, current_diesel_capacity, current_invert_capacity, current_time_left
        self.state = (self.current_demand, self.current_pv_capacity, self.current_battery_capacity, self. current_wind_capacity, self.current_diesel_capacity, self.current_invert_capacity, self.time_left, self.current_cluster_radius, self.current_cable_lenght)
        month_index = monthly_index_fromtimestep(self.time_steps)
        
        
        
        # this capacity factor section needs to be updated with right values but functinality is GOOD
        pv_cf = randomised_cf_norm_month(month_index, CF_pv_monthly,CF_pv_monthly_dev)
        wind_cf = randomised_cf_norm_month(month_index, CF_wind_monthly,CF_wind_monthly_dev)
        electricity_generated_monthly_kwh_pv = normalised_monthly_stoch_electricity_generation_kwh(pv_cf, self.current_pv_capacity)
        electricity_generated_monthly_kwh_wind =  normalised_monthly_stoch_electricity_generation_kwh(wind_cf, self.current_wind_capacity)
        electricity_generated_monthly_kwh_RES = electricity_generated_monthly_kwh_pv + electricity_generated_monthly_kwh_wind
        avg_tranmission_req = standardised_res_cf_difference(pv_cf, wind_cf) * self.current_demand # this is in kWh so that losses also in kWh right away
        tranmission_losses = cable_energy_losses_3ph(avg_tranmission_req, self.current_cable_lenght) # this also comes out in kWh
        net_electricity_deliverd_monthly_RES = electricity_generated_monthly_kwh_RES - tranmission_losses
        # for now this is simply difference between renewable generation and demand
        # need to look more closely at impact of batteries on this
        electricity_required_diesel = shortage_over5pct(net_electricity_deliverd_monthly_RES, self.current_demand)
        electricity_generated_diesel, cf_diesel = electricity_provided_diesel_monthly(self.current_diesel_capacity, electricity_required_diesel)
        total_electricity_delivered = electricity_generated_diesel + net_electricity_deliverd_monthly_RES
        system_opex = monthly_opex_system (self.current_pv_capacity, self.current_battery_capacity, self.current_wind_capacity,  electricity_generated_diesel, self. current_invert_capacity)
        mismatch = self.current_demand - total_electricity_delivered #REMEMBER NEGATIVE MISMATCH MEANS MORE PRODUCTION THAN DEMAND
        mismatch_pct = mismatch / self.current_demand
        mismatch_cost = mismatch_penalty(total_electricity_delivered, self.current_demand)
        
        
       # calculate total costs for this year    
        tot_cost = mismatch_cost + E_cost + system_opex + cabling_exp_cost
        reward = - (tot_cost / ((1+r_month) ** self.time_steps))
        

        
        if self.time_steps == T_monthly:
            done = True
            salvage_value_pv = salvage_pv(self.current_pv_capacity, (self.time_steps /n_month_peryr) ) #the .1 is added to see if it helps agent ovetrbuild less
            salvage_value_wind = salvage_wind(self.current_wind_capacity,( self.time_steps / n_month_peryr))
            # this is over simplification assuming all PV/wind has been used for WHOLE project duration, meaning it is underestimating salvage value
            reward = reward + salvage_value_pv + salvage_value_wind 
        else :
            done = False
    
        info = {"CF PV":pv_cf , "CF Wind": wind_cf , "Tranmission losses (kWh)" : tranmission_losses , "Distributed Energy(kWh)" : avg_tranmission_req, "Mismatch(kWh)" : mismatch ,   "Mismatch Pct" : mismatch_pct, "Diesel use (kWh)" :  electricity_generated_diesel , "Diesel CF" : cf_diesel}
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
        #self.current_pv_capacity, self.current_battery_capacity, self.current_wind_capacity, self.current_diesel_capacity, self.current_invert_capacity , self.time_left , self.current_cluster_radius, self.current_cable_lenght= self.starting_pv_capacity , self.starting_battery_capacity , self.starting_wind_capacity, self.starting_diesel_capacity, self.starting_inverter_capacity, self.starting_time_left, self.starting_cluster_radius, self.starting_cable_lenght    
        #set random demand curve evolution profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        self.starting_demand = electricity_monthly_demand_stochastic_less_ngers(n_gers, self.time_steps ,self.rD0s, self.rD10s, self.rDfs)
        # set uncertain cluster radius migration profile just to make sure demand and radius are indipendent, this determines evolution over time
        self.rD0r = np.random.random_sample()
        self.rD10r = np.random.random_sample()
        self.rDfr = np.random.random_sample()
        self.starting_cluster_radius = migration_cluster_radius(self.time_steps ,self.rD0r, self.rD10r, self.rDfr)
        self.starting_cable_lenght = cable_lenght_from_radius(self.starting_cluster_radius)
        #self.current_demand = electricity_monthly_demand_stochastic_less_ngers(n_gers, 0 ,self.rD0s, self.rD10s, self.rDfs)
        self.state = (self.starting_demand, self.starting_pv_capacity , self.starting_battery_capacity , self.starting_wind_capacity, self.starting_diesel_capacity, self.starting_inverter_capacity, self.starting_time_left, self.starting_cluster_radius, self.starting_cable_lenght)
        
        return np.array(self.state)
    
    

    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy either 0 or 1.
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1, 2, 3, 4])
    
    return chosen_action 




