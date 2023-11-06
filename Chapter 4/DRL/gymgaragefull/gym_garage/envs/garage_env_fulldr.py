# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:12 2020

@author: cesa_
"""
import gym
from gym import spaces
from garage_demand import demand_static, demand_stochastic, demand_stochastic_less
#from garage_ENPV_obj_arrayinput import cc_start
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding



# Parameters
T=20 #years
cc = 16000# Construction cost per parking space
cl = 3600000# Annual leasing land cost
#p = 0.00# Percentage of construction cost to acquire flexibility (i.e. fatter columns). Note that this is set to 0 to determine how much the flexibility is worth. The real value can be found by subracting the acquiistion cost from the NPV for a particular design.
cr = 2000# Operating cost per parking space
#ct = []# Total construction cost
gc = 0.10# Growth in construction cost per floor above two floors
n0 = 200# Initial number of parking space per floor
p = 10000# Price per parking space
r = 0.12# Discount rate
fmin = 2# Minimum number of floors built
fmax = 9# Maximum number of floors built

years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']


def cc_start(f0):
    if f0 > 2:
        cc_i = cc * n0 * ((((1+gc)**(f0-1) - (1+gc))/gc)) + (2*n0*cc)
    else : 
        cc_i= f0*n0*cc
    return cc_i

def Exp_cost(k, ft):
    if k == 0:
        Ex_cost = cc_start(ft)
    else:
        Ex_cost = n0*cc*((((1+gc)**(ft))-1)/(gc))*((1+gc)**((k/n0)-1))
    return Ex_cost



def opex(k):
    return cr*k
     
#Actions
NOTHING = 0
EXPAND_1FLOOR = 1
EXPAND_2FLOOR = 2
EXPAND_3FLOOR = 3

#stochastic demand curve applied here

class GarageEnvFull(gym.Env):

    def __init__(self):
        # Demand and capacity data
        max_capacity = 1800
        min_capacity = 200
        max_demand = 3000
        min_demand = 0
        self.low = np.array([min_capacity, min_demand], dtype=np.float32)
        self.high = np.array([max_capacity, max_demand], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(4) # No expansion, expand 1,2, or 3 floors
        self.count = 0
        self.time_steps = 0
        self.fixed_cost = 3600000
        self.current_capacity = 0 # maybe here have action determine this, imlement other decision rule
        #set random demand curve profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        
        #set state
        self.current_demand = demand_stochastic_less(0,self.rD0s, self.rD10s, self.rDfs) 
        self.state = self.current_capacity, self.current_demand

    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        current_capacity, current_demand = self.state
        if action == 0:
            possible_next_capacity = current_capacity
            E_cost = 0 
        elif action == 1:
           if current_capacity + 200 < 1800:         
              E_cost = Exp_cost(current_capacity,1)
              possible_next_capacity = current_capacity + 200
           else : 
              E_cost = 0
              possible_next_capacity = 1800
        elif action == 2:
            if current_capacity + 400 < 1800:         
               E_cost = Exp_cost(current_capacity,2)
               possible_next_capacity = current_capacity + 400
            else : 
               E_cost = 0
               possible_next_capacity = 1800 
        elif action == 3:
            if current_capacity + 600 < 1800:         
               E_cost = Exp_cost(current_capacity,3)
               possible_next_capacity = current_capacity + 600
            else : 
               E_cost = 0
               possible_next_capacity = 1800      
 
    
        if possible_next_capacity == 1800 and action != 0:
            revenue = -1 #tring imaginary penalty for overbuilding to prevent bhaviours, taken out in testing environment
        elif self.time_steps ==0:
            revenue = 0
            #reward = - (self.fixed_cost + E_cost)
        else:
            revenue = np.minimum(self.current_capacity, self.current_demand)*p 
            
        reward = (revenue - opex(self.current_capacity) - self.fixed_cost -E_cost) /((1+r)**self.time_steps)
# Bring forward to next time step in episode  
#check wether should actually have this above like earlier
#but maybe actually better to keep so that reward on each step is revenue in next time period
#still non anticipative because action is chosen before observation but better matching of good actions to right period and reward
        
        if self.time_steps == T:
            done = True
        else :
            done = False

        self.time_steps +=1       
        self.current_capacity = possible_next_capacity
        self.current_demand = demand_stochastic_less(self.time_steps,self.rD0s, self.rD10s, self.rDfs) # bring forward timestep
        self.state = (self.current_capacity, self.current_demand)

        return np.array(self.state), reward, done, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.time_steps = 0
        self.current_capacity = 0
        #reinitialize random demand curve profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        #set state
        self.current_demand = demand_stochastic_less(0 ,self.rD0s, self.rD10s, self.rDfs)
        self.state = (self.current_capacity, self.current_demand)
        return np.array(self.state)
    
    

    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy either 0 or 1.
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1,2,3])
    
    return chosen_action 



