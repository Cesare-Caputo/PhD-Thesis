# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:12 2020

@author: cesa_
""" 
#SIMPLE SCRIPT VERSION FOR NOW TO EXPLORE PROBLEM


import gym
from gym import spaces
import numpy as np  # type: ignore
from realoptions import (
    OrnsteinUhlenbeckProcess, DeterministicProcess, MarkedPoissonPointProcess,
    NormalProcess, WaterPrivate, WaterLeakage, WaterService, WaterIndustry,
    WaterExternal, WaterStorage, OptionSupply, OptionLeakage, OptionPrivate,
    OptionMisc, OptionCapacity, WaterSupply, plot_d_and_s, plot_costs,
    plot_risk, plot_input, plot_time, plot_risk_comp, cost_benefit,
    plot_nb_comp
)

import pandas as pd
import numpy as np
from gym.spaces import Discrete, Box
from gym.utils import seeding
import matplotlib.pyplot as plt
from models_rl import *

# General variables
# =================
SIMULATIONS = 20  # number of simulations per option NOT NEEDED FOR DRL
C_ENV = 20  # environmental costs by exceeding the fish pass level [m£/year]
C_SUP = 100  # environmental costs by exceeding the supply level [m£/year]
C_WR = 100  # costs if demand can not be fulfilled [m£/year]
RATE = 0.035  # discount rate [-]
DISCOUNT = True  # discounting is enabled [True / False]
DYNAMIC = True  # consider dynamic allocation i.e. option 4 [True / False]
     
     
#Actions
NOTHING = 0
ACTION_1 = 20 #ML/d raw water source
ACTION_2 = 10 #ml/d increase capacity of curreent WTW plant
ACTION_3 = 10 #ml/d increase capacity by linking to assynt planty


def generate_input_models():
    """Function to generate the models for the inputs"""
    # population
    # - - - - -
    # low
    pop_model = OrnsteinUhlenbeckProcess(100000, 3000, .03, 64350)
    # medium
    # pop_model = OrnsteinUhlenbeckProcess(110000, 3000, .05, 64350)
    # high
    # pop_model = OrnsteinUhlenbeckProcess(120000, 10000, .07, 64350)

    # industry
    ind_model = OrnsteinUhlenbeckProcess(1.2, .05, .03, .64)

    # per capital consumption
    pcc_model = DeterministicProcess(175)

    # people per household
    pph_model = DeterministicProcess(2.152)

    # external events
    eev_model = MarkedPoissonPointProcess(
        1/20, np.random.poisson, parameters={'lam': 30})

    # external demand
    edm_model = DeterministicProcess(18)  # [Ml/d]

    # rainfall data
    pre_model = NormalProcess(1430, 140, mu_shift=-.001, sigma_shift=.005)

    # dict with input models
    return {
        'pop': pop_model,
        'pcc': pcc_model,
        'pph': pph_model,
        'ind': ind_model,
        'eev': eev_model,
        'edm': edm_model,
        'pre': pre_model
    }


def generate_consumers(processes=None):
    """Function to generate all consumers in Inverness"""

    private = WaterPrivate(
        name='private water consumption',
        priority=100,
        scale_factor=1.1,
        compensation=C_WR,
        processes=processes
    )

    leakage = WaterLeakage(
        name='water leakage',
        scale_factor=0.3,
        processes=processes
    )

    service = WaterService(
        name='service water consumption',
        priority=500,
        scale_factor=0.025,
        compensation=C_WR,
        processes=processes
    )

    industry = WaterIndustry(
        name='industrial water consumption',
        priority=400,
        compensation=C_WR,
        base_demand=0.5,
        scale_factor=1.,
        processes=processes
    )

    external = WaterExternal(
        name='external demand',
        priority=50,
        compensation=C_WR,
        processes=processes
    )

    return [private, leakage, service, industry, external]


def generate_producers(processes=None):
    """Function to generate all consumers in Inverness"""

    duntelchaig = WaterStorage(
        name='Reservoir A',  # 'Loch Duntelchaig',
        storage=144402,  # [Ml]
        operation=7373,  # [Ml]
        intake=13646,  # [Ml]
        area=5546219,  # [m2]
        dynamic=True,
        processes=processes,
    )

    ashie = WaterStorage(
        name='Reservoir B',  # 'Loch Ashie',
        operation=3.7 * 365,  # [Ml/d * d]
        active=False,
    )

    return [duntelchaig, ashie]


# State space could be just total demand and total supply for now, 
## THIS VERSION OF ENVIRONMENT ADDS A COUPLE OTHER OPTIONS TOO JUST TO CHECK

class WtwEnvSimple_v2(gym.Env):

    def __init__(self):
        # Models from previous paper
        self.discount_rate=0.0354
        # discount vector
        self.discount = np.ones((1, 61)) / (1+self.discount_rate) ** np.arange(0, 61, step=1)
        self.discounted = True

        # random input models

        self.models = generate_input_models()
        self.processes = {key: model.simulate()for key, model in self.models.items()}
        # generate consumers
        self.consumers = generate_consumers()

        # generate producers
        self.producers = generate_producers()
        self.wtw = WaterSupplyRL(name='Inverness WTW', capacity=38.5)
        
        # combine all models
        all_models = self.consumers + self.producers + [self.wtw]

        # reset all models
        for model in all_models:
            model.reset(processes=self.processes)
            
            
        self.base = OptionSupply(
            name='Base costs',
            increase=0,
            enhancement=0,
            replacement=1.475,
            opex=.145,
            duration=0,
            start=0,
            discount_rate=self.discount_rate) # this describes base case scenario and equivalent to doing nothing
        
        self.options = [self.base]

        supply, demand = self.wtw.balance_rl(self.producers, self.consumers, self.options, 0) #first timestep
        
        self.total_demand = np.sum(demand.get("data"), keepdims=True)
        self.total_supply = np.sum(supply.get("data"), keepdims=True)
        
        self.max_capacity = 100 #approx normalization for now
        self.min_capacity = 10 

        self.max_demand = 100
        self.min_demand = 10
        
        self.max_time_remaining = 61
        self.min_time_remaining = 0


        self.low = np.array([self.min_capacity, self.min_demand, self.min_time_remaining], dtype=np.float32)
        self.high = np.array([self.max_capacity, self.max_demand, self.max_time_remaining], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(4) 
        self.time_steps = 0
        self.mem = []
        self.current_capacity = self.total_supply[0]
        self.current_demand = self.total_demand[0]

        self.state = np.array([self.current_capacity, self.current_demand, self.max_time_remaining])
        self.constraint_penalty = 1 ### for underpumping or other undesierble behavuour TBD


    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        #current_capacity, current_demand , time= self.state
        max_capacity, min_capacity, max_demand, min_demand = self.max_capacity, self.min_capacity , self.max_demand, self.min_demand 
        
        
        #Retrieve demand and capacity values from state
        self.current_capacity = self.state[0]
        self.current_demand = self.state[1]
        self.time_left = self.state[2]
      
        
      # Capacity is restricted to max level even if agent chooses expansion
      #NOTE to retrieve option class and cost from other code
        if action == 0:
            options = self.options
            supply, demand =  self.wtw.balance_rl(self.producers, self.consumers, self.options , self.time_steps)
            self.total_demand = np.sum(demand.get("data"), keepdims=True)
            self.total_supply = np.sum(supply.get("data"), keepdims=True)
        elif action == 1:
            raw_20 = OptionSupply(
                name='New 20Ml/d raw water source',
                dynamic=False,
                increase=20,
                start=self.time_steps,
                duration=2,
                enhancement=34.2,
                replacement=0.0,
                opex=0.5,
                discount_rate=self.discount_rate,
            )
            
            self.options.append(raw_20)
            
            #balance supply and demand in this timestep    
            supply, demand =  self.wtw.balance_rl(self.producers, self.consumers, self.options, self.time_steps)
            self.total_demand = np.sum(demand.get("data"),  keepdims=True)
            self.total_supply = np.sum(supply.get("data"), keepdims=True)
        elif action ==2:
            cap_dyn = OptionCapacity(
                name='10Ml/d WTW capacity increase',
                dynamic=False,
                increase=10,
                start=self.time_steps,
                duration=1,
                enhancement=15,
                replacement=0,
                opex=0,
                discount_rate=self.discount_rate,
            )          

            self.options.append(cap_dyn)
            
            #balance supply and demand in this timestep    
            supply, demand =  self.wtw.balance_rl(self.producers, self.consumers, self.options, self.time_steps)
            self.total_demand = np.sum(demand.get("data"),  keepdims=True)
            self.total_supply = np.sum(supply.get("data"), keepdims=True)
            
        elif action ==3: ## linking capacity to assynt
            link_capacity = OptionCapacity(
                name='Assynt WTW capacity',
                increase=5,
                start=self.time_steps,
                duration=2,
                discount_rate=self.discount_rate,
            )

            self.options.append(link_capacity)
            
            #balance supply and demand in this timestep    
            supply, demand =  self.wtw.balance_rl(self.producers, self.consumers, self.options, self.time_steps)
            self.total_demand = np.sum(demand.get("data"),  keepdims=True)
            self.total_supply = np.sum(supply.get("data"), keepdims=True)            
            
            
            
        
        
        # variables to store the outputs in the info dict, will complete later
        topics = ['input', 'cost', 'compensation', 'investment',
                  'demand', 'supply', 'level', 'capacity', 'timing']
        result = {topic: {'data': [], 'labels': []} for topic in topics}    
        
        # Costs
        # =====
        # cost of inadequate water supply
        discount = self.discount
        options = self.options
        
        c_iws = {'data': [], 'labels': []}
        for consumer in self.consumers:
            c_iws['data'].append(consumer.costs(discount_rate=self.discount_rate,
                                                discounted=self.discounted))
            c_iws['labels'].append('Costs of %s' % consumer.name)

        # cost of interventions
        c_int = {'data': [], 'labels': []}
        for option in options:
            c_int['data'].append(option.costs(discounted=self.discounted))
            c_int['labels'].append(option.name)

        # cost of environmental impact
        c_env = {'data': [], 'labels': []}
        for lake in self.producers:
            if lake.dynamic:
                c_env['data'].append(lake.failure('operation') * C_ENV *
                                     (discount.T if self.discounted else 1))
                c_env['labels'].append('Costs of fish passing')

        # cost for under-pumping
        c_sup = {'data': [], 'labels': []}
        for lake in self.producers:
            if lake.dynamic:
                c_sup['data'].append(lake.failure('intake') * C_SUP *
                                     (discount.T if self.discounted else 1))
                c_sup['labels'].append('Costs of under-pumping')

        #_d,  _l = _combine(c_iws, c_env, c_sup)
        
        c_wr = {'data': [np.sum(np.hstack(c_iws['data']),
                                axis=1, keepdims=True)],
                'labels': ['Cost for inadequate water supply']}
        
        c_inv = {'data': [np.sum(np.hstack(c_int['data']),
                                 axis=1, keepdims=True)],
                 'labels': ['Cost for interventions']}
        
        
        total_costs = c_wr.get("data")[0].flatten()[self.time_steps] + c_inv.get("data")[0].flatten()[self.time_steps] + c_sup.get("data")[0].flatten()[self.time_steps]  + c_env.get("data")[0].flatten()[self.time_steps] 
        
        #NOTE not sure if cost already discounted, need to double check
        discounted_tot_costs = - (total_costs / (((1+self.discount_rate)**self.time_steps)))           
        reward = discounted_tot_costs 
        
        # Bring forward to next time step in episode and calculate reward
        # NOTE THAT BECAUSE WAY MODEL BUILT MAY NEED TO UPDATE LATER
        self.time_steps +=1       
        self.current_capacity = self.total_supply[0]
        self.current_demand = self.total_demand[0] # bring forward timestep
        self.state = (self.current_capacity, self.current_demand, self.max_time_remaining -self.time_steps)
         
        # Termination conditions
        if self.time_steps == self.max_time_remaining:
             done = True
        else :
             done = False

        return np.array(self.state), reward, done, {"Options exercised": self.options}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.time_steps = 0
        self.models = generate_input_models()
        self.processes = {key: model.simulate()for key, model in self.models.items()}
        # generate consumers
        self.consumers = generate_consumers()

        # generate producers
        self.producers = generate_producers()
        self.wtw = WaterSupplyRL(name='Inverness WTW', capacity=38.5)
        
        # combine all models
        all_models = self.consumers + self.producers + [self.wtw]

        # reset all models
        for model in all_models:
            model.reset(processes=self.processes)
        
        self.options = [self.base]
            
        supply, demand = self.wtw.balance_rl(self.producers, self.consumers, self.options, 0) #first timestep
        
        self.total_demand = np.sum(demand.get("data"), keepdims=True)
        self.total_supply = np.sum(supply.get("data"), keepdims=True)
        self.current_capacity = self.total_supply[0]
        self.current_demand = self.total_demand[0]

        self.state = np.array([self.current_capacity, self.current_demand, self.max_time_remaining])
        
        return self.state
    
    
def agent_policy(rand_generator):
    """
    Given random number generator and state, returns an action according to the agent's policy either 0 or 1.
    Returns:
        chosen action [int]
    """
    
    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0,1])
    
    return chosen_action 


# env = WtwEnvSimple()
# tensorboard_log="./sb3_wtw_test/"
# log_dir = "./sb3_wtw_ppo_test/"

# env = Monitor(env, log_dir)
# callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log= tensorboard_log)
# model.learn(total_timesteps=2000, callback=callback)

#model.save("garage_ppo_vgood_sb3_10_10_23")