# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:46:40 2020

@author: cesa_
"""

import gym
from gym import spaces
from garage_demand import demand_static, demand_stochastic, demand_stochastic_less , demand_stochastic_series
from garage_cost import Exp_cost, opex
#from garage_ENPV_obj_arrayinput import cc_start
import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding
from gym import utils
from gym import envs
from gym.envs.toy_text import discrete
import copy
import pandas as pd
# from plot_utils import plot_values


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
r = 0.12# Discount rate referred to in stochastic versions
fmin = 2# Minimum number of floors built
fmax = 9# Maximum number of floors built
fixed_cost = 3600000
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= [0,1,2,3]


# Scenario generation

def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df
        
     
# n_scenarios = 10 # this sets number of indipendantly scenarios generated for sample average approximation algorithm
# scenario_df = scenario_generator(n_scenarios)

def cc_start(f0):
    if f0 > 2:
        cc_i = cc * n0 * ((((1+gc)**(f0-1) - (1+gc))/gc)) + (2*n0*cc)
    else : 
        cc_i= f0*n0*cc
    return cc_i
         

class Garage:
    def __init__(self, discount_rate, demand = ''):
        self.reset()
        self.time = 0
        self.S = [0,200,400,600,800,1000,1200,1400,1600,1800]
        self.A = [0,1,2,3]
        self.nS = 10
        self.nA =4
        self.discount_rate = discount_rate
        self.demand = demand
        return

    def step(self, state, action):
         time = self.time
         current_capacity=state
         if time ==0:
             possible_next_capacity = action*200
             E_cost = cc_start(action) # set starting cost equal to expansion here to group construction costs into 1
         else:
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
         
         if demand =='stochastic':
             current_demand = demand_static(time)
         
         
         new_state = possible_next_capacity    
         revenue = np.minimum(current_capacity, current_demand)*p
         reward = (revenue - opex(current_capacity) - fixed_cost -E_cost) /((1+self.discount_rate)**time)
         prob = 1
        
         self.time += 1
         if self.time ==20:
             done = True
             self.reset
         else :
            done = False
         return prob, new_state, reward, done
    
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.state = 0
        self.time = 0
        self.state_values = np.zeros(10)
        self.policy =np.ones((10,4))/4

        return 
    def get_transition_probability(self, state, new_state):
        """
        Args:
            state (Tuple[int, int]): A tuple storing the number of available locations, respectively at A and B
            new_state (Tuple[int, int]): A possible future state of the environment

        Returns:
            (float): The probability that the system transitions from a state `s` to a state `s'`.
        """
        prob =1 #startinf fully determinstic
        return prob
    
    def get_valid_action(self, state):
        """
        Return an action that is compatible with the current state of the system, mainly making sure the max floors is not surpassed

        Args:
            state (Tuple[int, int]): 
            action (int): 
        Returns:
            (int): a feasible number of loors to expand by
        """
        current_capacity = state
        if current_capacity ==1800:
            action = np.array([0])
        elif current_capacity ==1600:
            action = np.array([0,1])
        elif current_capacity == 1400:
            action = np.array([0,1,2])
        elif current_capacity <= 1200:
            action = np.array([0,1,2,3])
        return action 
    
    def transitions(self, state, action):
         time = self.time
         current_capacity=state
         if time ==0:
             possible_next_capacity = action*200
             E_cost = cc_start(action) # set starting cost equal to expansion here
         else: 
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
                    
             current_demand = demand_static(time)      
             new_state = possible_next_capacity    
             revenue = np.minimum(current_capacity, current_demand)*p
             reward = (revenue - opex(current_capacity) - fixed_cost -E_cost)
         return np.array([reward, 1 ])
    
    def terminal_value(self, state):
         current_capacity=state
         current_demand = demand_static(20)
         revenue = np.minimum(current_capacity, current_demand)*p
         reward = (revenue - opex(current_capacity)) 
        
         return reward

class Garage_Complete:
    def __init__(self, discount_rate, demand = ''): # the discount rate here refers to the financial one, or r in the 1/(1+r)^n formula. NOT discount factor used for rewards. Respectively, they are .12 and .89 in our case. 
        self.reset()
        self.time = 0
        self.S = [0,200,400,600,800,1000,1200,1400,1600,1800]
        self.A = [0,1,2,3]
        self.nS = 10
        self.nA =4
        self.discount_rate = discount_rate
        self.demand = demand
        #set random demand curve profile that resets on each episode
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        return

    def step(self, state, action, time):
         current_capacity=state
         if time ==0:
             possible_next_capacity = action*200
             E_cost = cc_start(action) # set starting cost equal to expansion here to group construction costs into 1
         else:
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
         
         if self.demand =='stochastic':
             current_demand = demand_stochastic_less(time,self.rD0s, self.rD10s, self.rDfs)
         else: current_demand = demand_static(time)
         
         new_state = possible_next_capacity    
         revenue = np.minimum(current_capacity, current_demand)*p
         reward = (revenue - opex(current_capacity) - fixed_cost -E_cost) /((1+self.discount_rate)**time)
         prob = 1
        
         self.time += 1
         if self.time ==20:
             done = True
             self.reset
         else :
            done = False
         return prob, new_state, reward, done
    
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.state = 0
        self.time = 0
        self.policy =np.ones((10,4))/4
        #reinitialize random demand curve profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()

        return 
    
    def get_valid_action(self, state):
        """
        Return an action that is compatible with the current state of the system, mainly making sure the max floors is not surpassed

        Args:
            state (Tuple[int, int]): 
            action (int): 
        Returns:
            (int): a feasible number of loors to expand by
        """
        current_capacity = state
        if self.time == 0: 
            action = np.array([0,1,2,3,4])
        else: 
            if current_capacity ==1800:
                action = np.array([0])
            elif current_capacity ==1600:
                action = np.array([0])
            elif current_capacity == 1400:
                action = np.array([0,1])
            elif current_capacity <= 1200:
                action = np.array([0,1,2])
        return action 

    def terminal_value(self, state):
         if self.demand =='stochastic':
             current_demand = demand_stochastic_less(T,self.rD0s, self.rD10s, self.rDfs)
         else: current_demand = demand_static(T)
         current_capacity=state
         revenue = np.minimum(current_capacity, current_demand)*p
         value = (revenue - opex(current_capacity)) /((1 + r)**T)
        
         return value


class Garage_Complete_demand_scenario:
    def __init__(self, discount_rate, scenario_df, demand_series_index): # the discount rate here refers to the financial one, or r in the 1/(1+r)^n formula. NOT discount factor used for rewards. Respectively, they are .12 and .89 in our case. 
        self.reset()
        self.time = 0
        self.S = [0,200,400,600,800,1000,1200,1400,1600,1800]
        self.A = [0,1,2,3]
        self.nS = 10
        self.nA =4
        self.discount_rate = discount_rate
        self.demand_series = scenario_df[demand_series_index]
        return

    def step(self, state, action, time):
         current_capacity=state
         demand_series = self.demand_series
         if time ==0:
             possible_next_capacity = action*200
             E_cost = cc_start(action) # set starting cost equal to expansion here to group construction costs into 1
         else:
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
         
         current_demand = demand_series[time-1]
         new_state = possible_next_capacity    
         revenue = np.minimum(current_capacity, current_demand)*p
         reward = (revenue - opex(current_capacity) - fixed_cost -E_cost) /((1+self.discount_rate)**time)
         prob = 1
        
         self.time += 1
         if self.time ==20:
             done = True
             self.reset
         else :
            done = False
         return prob, new_state, reward, done
    
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.state = 0
        self.time = 0

        return 
    
    def get_valid_action(self, state):
        """
        Return an action that is compatible with the current state of the system, mainly making sure the max floors is not surpassed

        Args:
            state (Tuple[int, int]): 
            action (int): 
        Returns:
            (int): a feasible number of loors to expand by
        """
        current_capacity = state
        if self.time == 0: 
            action = np.array([0,1,2,3,4])
        else: 
            if current_capacity ==1800:
                action = np.array([0])
            elif current_capacity ==1600:
                action = np.array([0])
            elif current_capacity == 1400:
                action = np.array([0,1])
            elif current_capacity <= 1200:
                action = np.array([0,1,2])
        return action 

    def terminal_value(self, state):
         current_demand = self.demand_series[T]   
         current_capacity=state
         revenue = np.minimum(current_capacity, current_demand)*p
         value = (revenue - opex(current_capacity)) /((1 + r)**T)
        
         return value






class Garage_discounted:#trying now to set starting floors and see  what difference that makes
    def __init__(self):
        self.reset()
        self.time = 0
        self.S = [0,200,400,600,800,1000,1200,1400,1600,1800]
        self.A = [0,1,2,3]
        self.nS = 10
        self.nA =4
        return

    def step(self, state, action, time):
         current_capacity=state
         if time ==0:
             possible_next_capacity = action*200
             E_cost = cc_start(action) # set starting cost equal to expansion here to group construction costs into 1
         else:
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
             
         current_demand = demand_static(time)
         new_state = possible_next_capacity    
         revenue = np.minimum(current_capacity, current_demand)*p
         reward = (revenue - opex(current_capacity) - fixed_cost -E_cost) 
         disc_reward = reward / ((1+r)**time)
         prob = 1
        
         self.time += 1
         if self.time ==20:
             done = True
             self.reset
         else :
            done = False
         return prob, new_state, disc_reward, done
    
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.state = 0
        self.time = 0
        self.policy =np.ones((10,4))/4

        return 
    
    def get_valid_action(self, state):
        """
        Return an action that is compatible with the current state of the system, mainly making sure the max floors is not surpassed

        Args:
            state (Tuple[int, int]): 
            action (int): 
        Returns:
            (int): a feasible number of loors to expand by
        """
        current_capacity = state
        if current_capacity ==1800:
            action = np.array([0])
        elif current_capacity ==1600:
            action = np.array([0,1])
        elif current_capacity == 1400:
            action = np.array([0,1,2])
        elif current_capacity <= 1200:
            action = np.array([0,1,2,3])
        return action 

    def terminal_value(self, state):
         current_capacity=state
         current_demand = demand_static(20)
         revenue = np.minimum(current_capacity, current_demand)*p
         value = (revenue - opex(current_capacity)) /((1+r)**20) 
        
         return value

class Garage_discounted_stoch:#trying now to set starting floors and see  what difference that makes
    def __init__(self):
        self.reset()
        self.time = 0
        self.S = [0,200,400,600,800,1000,1200,1400,1600,1800]
        self.A = [0,1,2,3]
        self.nS = 10
        self.nA =4
        #set random demand curve profile that resets on each episode
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()
        return

    def step(self, state, action, time):
         current_capacity=state
         if time ==0:
             possible_next_capacity = action*200
             E_cost = cc_start(action) # set starting cost equal to expansion here to group construction costs into 1
         else:
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
             
         current_demand = demand_stochastic_less(time,self.rD0s, self.rD10s, self.rDfs)
         new_state = possible_next_capacity    
         revenue = np.minimum(current_capacity, current_demand)*p
         reward = (revenue - opex(current_capacity) - fixed_cost -E_cost) 
         disc_reward = reward / ((1+r)**time)
         prob = 1
        
         self.time += 1
         if self.time ==20:
             done = True
             self.reset
         else :
            done = False
         return prob, new_state, disc_reward, done
    
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.state = 0
        self.time = 0
        self.policy =np.ones((10,4))/4
        #reinitialize random demand curve profile
        self.rD0s = np.random.random_sample()
        self.rD10s = np.random.random_sample()
        self.rDfs = np.random.random_sample()

        return 
    
    def get_valid_action(self, state):
        """
        Return an action that is compatible with the current state of the system, mainly making sure the max floors is not surpassed

        Args:
            state (Tuple[int, int]): 
            action (int): 
        Returns:
            (int): a feasible number of loors to expand by
        """
        current_capacity = state
        if current_capacity ==1800:
            action = np.array([0])
        elif current_capacity ==1600:
            action = np.array([0,1])
        elif current_capacity == 1400:
            action = np.array([0,1,2])
        elif current_capacity <= 1200:
            action = np.array([0,1,2,3])
        return action 

    def terminal_value(self, state):
         current_capacity=state
         current_demand = demand_static(20)
         revenue = np.minimum(current_capacity, current_demand)*p
         value = (revenue - opex(current_capacity)) /(1.12**20) 
        
         return value