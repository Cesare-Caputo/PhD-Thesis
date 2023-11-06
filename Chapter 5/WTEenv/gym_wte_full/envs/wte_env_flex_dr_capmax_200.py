# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:12 2020

@author: cesa_
"""
import gym
from gym import spaces
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding
from wte_costs import *
from wte_revenues import *
from food_waste_recycling import *
from decimal import Decimal


# Parameters
T=15 #years # CHECK INDEX IN OTHER FUNCTION
r = .08 # discount rate

#Actions
NOTHING = 0
EXPAND_1_unit_s1 = 1 # this is the same as a centralised expansion
EXPAND_1_unit_s2 = 2
EXPAND_1_unit_s3 = 3
EXPAND_1_unit_s4 = 4
EXPAND_1_unit_s5 = 5
EXPAND_1_unit_s6 = 6


s1_actions = [EXPAND_1_unit_s1 ]
s2_actions = [EXPAND_1_unit_s2 ]
s3_actions = [EXPAND_1_unit_s3 ]
s4_actions = [EXPAND_1_unit_s4 ]
s5_actions = [EXPAND_1_unit_s5 ]
s6_actions = [EXPAND_1_unit_s6 ]

# define edges pof box observation space here
# starting sector 1 capacity should be 200 as in original implementation

n_plants = 6
tot_fw_recycled_0 = 274 # tonnes per day
fw_recycled_per_plant_0 = tot_fw_recycled_0 / n_plants


#REMEMBER TO ALSO ADD TIME LEFT AS STATE VARIABLE ONCE WORKING
class WTE_EnvFull_capmax(gym.Env):

    def __init__(self):
        # Demand and capacity data
        max_capacity = 600
        min_capacity = 0
        min_capacity_s1 = 000
        max_demand = 3000
        min_demand = 20
        min_time_remaining = 0
        max_time_remaining = 16
        # capaicty 1, demand 1 then capacity 2, demand 2
        self.low = np.array([min_capacity_s1, min_demand , min_capacity, 
                             min_demand, min_capacity, min_demand, min_capacity, 
                             min_demand,min_capacity, min_demand,
                             min_capacity, min_demand , min_time_remaining ], dtype=np.float32)
        self.high = np.array([max_capacity, max_demand , max_capacity, max_demand,
                              max_capacity, max_demand, max_capacity, max_demand,
                              max_capacity, max_demand, max_capacity, max_demand, max_time_remaining], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(7) # No expansion, expand 1,2, or 3 floors
        self.count = 0
        self.time_steps = 0
        self.time_to_end = T + 1
        # initialize capacities in various sectors
        self.current_capacity_s1 = 200 
        self.current_capacity_s2 = 0
        self.current_capacity_s3 = 0
        self.current_capacity_s4 = 0
        self.current_capacity_s5 = 0
        self.current_capacity_s6 = 0
        
        #initialize demand to year 0 predefined levels
        self.fw_demand_s1 = fw_recycled_per_plant_0
        self.fw_demand_s2 = fw_recycled_per_plant_0
        self.fw_demand_s3 = fw_recycled_per_plant_0
        self.fw_demand_s4 = fw_recycled_per_plant_0
        self.fw_demand_s5 = fw_recycled_per_plant_0
        self.fw_demand_s6 = fw_recycled_per_plant_0
        
        
        #set state
        self.state = np.array([self.current_capacity_s1, self.fw_demand_s1 ,
                               self.current_capacity_s2, self.fw_demand_s2 ,
                               self.current_capacity_s3, self.fw_demand_s3 , 
                               self.current_capacity_s4, self.fw_demand_s4 , 
                               self.current_capacity_s5, self.fw_demand_s5 , 
                               self.current_capacity_s6, self.fw_demand_s6 ,
                               self.time_to_end])
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # extract current capacities and demand from state variable
        time_left = self.state[12]
        self.current_capacity_s1 = self.state[0]
        fw_demand_s1 = self.state[1]
        self.current_capacity_s2 = self.state[2]
        fw_demand_s2 = self.state[3]        
        self.current_capacity_s3 = self.state[4]
        fw_demand_s3 = self.state[5]
        self.current_capacity_s4 = self.state[6]
        fw_demand_s4 = self.state[7]        
        self.current_capacity_s5 = self.state[8]
        fw_demand_s5 = self.state[9]        
        self.current_capacity_s6 = self.state[10]
        fw_demand_s6 = self.state[11]        
       
        
        if action == 0:
            capacity_expansion = 0
            E_cost = 0 
        elif action == 1:      
           E_cost = expansion_cost(200)
           self.current_capacity_s1 += 200
        elif action == 2:    
           E_cost = expansion_cost(200)
           self.current_capacity_s2 += 200
        elif action == 3:    
           E_cost = expansion_cost(200)
           self.current_capacity_s3 += 200
        elif action == 4:    
           E_cost = expansion_cost(200)
           self.current_capacity_s4 += 200
        elif action == 5:    
           E_cost = expansion_cost(200)
           self.current_capacity_s5 += 200
        elif action == 6:    
           E_cost = expansion_cost(200)
           self.current_capacity_s6 += 200
           
        #calculate total capacity resulting from previous action
        self.total_installed_capacity = self.current_capacity_s1 + self.current_capacity_s2 + self.current_capacity_s3 + self.current_capacity_s4 + self.current_capacity_s5 + self.current_capacity_s6
        

    
        # bring forward one timestep to calculate revenues and costs for the period
        self.time_steps +=1 
        self.time_to_end = time_left - 1
        self.fw_demand_s1 = recycled_fw_per_sector(self.time_steps, fw_demand_s1)
        self.fw_demand_s2 = recycled_fw_per_sector(self.time_steps, fw_demand_s2)        
        self.fw_demand_s3 = recycled_fw_per_sector(self.time_steps, fw_demand_s3)        
        self.fw_demand_s4 = recycled_fw_per_sector(self.time_steps, fw_demand_s4)        
        self.fw_demand_s5 = recycled_fw_per_sector(self.time_steps, fw_demand_s5)
        self.fw_demand_s6 = recycled_fw_per_sector(self.time_steps, fw_demand_s6)        
        self.total_fw_demand = self.fw_demand_s1 + self.fw_demand_s2 + self.fw_demand_s3 + self.fw_demand_s4 + self.fw_demand_s5 +self.fw_demand_s6
        
        # calculate transportation costs in specific year for each sector
        self.transport_cost_s1 = transport_cost_s1(self.fw_demand_s1 , self.current_capacity_s1)
        self.transport_cost_s2 = transport_cost_s2(self.fw_demand_s2 , self.current_capacity_s2)
        self.transport_cost_s3 = transport_cost_s3(self.fw_demand_s3 , self.current_capacity_s3)
        self.transport_cost_s4 = transport_cost_s4(self.fw_demand_s4 , self.current_capacity_s4)
        self.transport_cost_s5 = transport_cost_s5(self.fw_demand_s5 , self.current_capacity_s5)
        self.transport_cost_s6 = transport_cost_s6(self.fw_demand_s6 , self.current_capacity_s6)
        self.total_transport_cost = self.transport_cost_s1 + self.transport_cost_s2 +self.transport_cost_s3 +self.transport_cost_s4 + self.transport_cost_s5 +self.transport_cost_s6

        # calculate other costs for aggregated system
        self.tot_disposal_costs = disposal_cost(self.total_fw_demand ,self.total_installed_capacity)
        self.opex = opex_tot(self.total_installed_capacity)
        self.land_use_cost = land_cost(self.total_installed_capacity)


        # calculate revenues
                # calculate revenues including penalty for contraints
        total_capacity = self.current_capacity_s1 + self.current_capacity_s2 +self.current_capacity_s3 + self.current_capacity_s4 + self.current_capacity_s5 +self.current_capacity_s6
        #imaginary penalty for overbuilding bhevaiours
        if total_capacity >= 600  and action != 0:
            revenue = -1
        else:
            elec_revenue  = electricity_revenue(self.total_fw_demand ,self.total_installed_capacity)
            refuse_revenue = refuse_collection_revenue(self.total_fw_demand)
            revenue = elec_revenue + refuse_revenue
        # calculate net cash flow
        ncf_t = (revenue) - (self.total_transport_cost + self.tot_disposal_costs + self.opex + self.land_use_cost + E_cost)
        reward = ncf_t / (1.08 ** self.time_steps) if self.time_steps !=0  else ncf_t

        if self.time_steps == T + 1: 
            done = True
        else :
            done = False
   
        self.state =  np.array([self.current_capacity_s1, self.fw_demand_s1 ,
                               self.current_capacity_s2, self.fw_demand_s2 ,
                               self.current_capacity_s3, self.fw_demand_s3 , 
                               self.current_capacity_s4, self.fw_demand_s4 , 
                               self.current_capacity_s5, self.fw_demand_s5 , 
                               self.current_capacity_s6, self.fw_demand_s6 ,
                               self.time_to_end])
        info = {"tot transport cost" : self.total_transport_cost , "Income $" : revenue , "disposal cost" : self.tot_disposal_costs , "Expansion Cost" : E_cost}
        return self.state, reward, done, info
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.time_steps = 0
        self.time_to_end = T + 1
        # initialize capacities in various sectors
        self.current_capacity_s1 = 200 
        self.current_capacity_s2 = 0
        self.current_capacity_s3 = 0
        self.current_capacity_s4 = 0
        self.current_capacity_s5 = 0
        self.current_capacity_s6 = 0
        #initialize demand to year 0 predefined levels
        self.fw_demand_s1 = fw_recycled_per_plant_0
        self.fw_demand_s2 = fw_recycled_per_plant_0
        self.fw_demand_s3 = fw_recycled_per_plant_0
        self.fw_demand_s4 = fw_recycled_per_plant_0
        self.fw_demand_s5 = fw_recycled_per_plant_0
        self.fw_demand_s6 = fw_recycled_per_plant_0
        #reinitialize random demand curve profile for EACH SECTOR to be used in STEP
        #seems she actually did not do this right in her model... continue with her method
        self.rand_r_max_s1 = np.random.random_sample()
        self.rand_b_s1 = np.random.random_sample()
        self.rand_r_max_s2 = np.random.random_sample()
        self.rand_b_s2 = np.random.random_sample()        
        self.rand_r_max_s3 = np.random.random_sample()
        self.rand_b_s3 = np.random.random_sample()        
        self.rand_r_max_s4 = np.random.random_sample()
        self.rand_b_s4 = np.random.random_sample()        
        self.rand_r_max_s5 = np.random.random_sample()
        self.rand_b_s5 = np.random.random_sample()
        self.rand_r_max_s6 = np.random.random_sample()
        self.rand_b_s6 = np.random.random_sample()
        #set state
        self.state = np.array([self.current_capacity_s1, self.fw_demand_s1 ,
                               self.current_capacity_s2, self.fw_demand_s2 ,
                               self.current_capacity_s3, self.fw_demand_s3 , 
                               self.current_capacity_s4, self.fw_demand_s4 , 
                               self.current_capacity_s5, self.fw_demand_s5 , 
                               self.current_capacity_s6, self.fw_demand_s6,
                               self.time_to_end])
        return self.state
    
    

    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy either 0 or 1.
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1,2,3, 4, 5 ,6])
    
    return chosen_action 



