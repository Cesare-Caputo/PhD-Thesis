# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:30:43 2020

@author: cesa_
"""

from garage_DP_class import Garage_Complete, cc_start
from backward_induction_dp import StochasticDP
from matplotlib import pyplot as plt
import numpy as np
from garage_cost import Exp_cost
from garage_demand import demand_static,  demand_stochastic_less
import pandas as pd
from garage_DP_helper import *
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= np.array([0,1,2,3, 4])
r = .12 # discount rate used when initializing environments


# Other policies for comparison
policy_inflex = np.array([ 6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
DV_D = np.array([1, 1, 1, 0, 2, 2, 4]) # Optimal determistic design vector with GA DR
f0 = 6 

# instantiating environment with discount rate and boundary values
env = Garage_Complete(discount_rate = 0 ) # undiscounted rewards
env_d = Garage_Complete(discount_rate = r ) # discounted rewards and determinstic demand
env_d_b = Garage_Complete(discount_rate = r ) # discounted rewards and determinstic demand WITH BOUNDARY CONDITIONS


#Setting up the various instances of the problem  for debugging
number_of_stages = 22
states = state_space
decisions = action_space
dp = StochasticDP(number_of_stages, states, decisions, minimize = False) # non discounted reward
dp_d = StochasticDP(number_of_stages, states, decisions, minimize = False) # discounted rewards
dp_d_b = StochasticDP(number_of_stages, states, decisions, minimize = False) # discounted rewards and boundary value




# This sets dp.probability[m, n, t, x] = p and dp.contribution[m, n, t, x] = c # Populating for all cases within this loop here
for t in range(0,22):
    for s in state_space:
        action_possible = env.get_valid_action(s)
        for a in action_possible:
             p, next_state, reward, done = env.step(s, a, t) # undiscounted, boundary is 0
             p_d, next_state_d, reward_d, done_d = env_d.step(s, a , t) # discounted, boundary is 0
             p_d_b, next_state_d_b, reward_d_b, done_d_b = env_d_b.step(s, a , t) # discounted, boundary set
             dp.add_transition(stage=t, from_state=s, decision=a, to_state=next_state, probability=p, contribution=reward)
             dp_d.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_d, probability=p, contribution=reward_d)
             dp_d_b.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_d_b, probability=p, contribution=reward_d_b)


# Set boundary conditions in last stage, not sure if should be 0 or actual rewrd for that stage since we can compute it for all states
for s in state_space:
    dp.boundary[s] = 0
    dp_d.boundary[s] = 0
    dp_d_b.boundary[s] = env_d_b.terminal_value(s)

# solving for optimal value and policy 
value_bi_d_b, policy_bi_d_b = dp_d_b.solve() # discounted determinstic with boundary values

# NPV calculations

# NPV_bi, model_bi = NPV_garage_DP_BI(policy_bi)
# print("NPV for optimal policy using backward induction is Million $", NPV_bi*(10**-6))
# NPV_BI_d, model_bi_d = NPV_garage_DP_BI(policy_bi_d)
# print("NPV for optimal policy with detetrminstic demand using discounted backward induction is Million $", NPV_BI_d*(10**-6))
NPV_BI_d_b, model_bi_d_b = NPV_garage_DP_BI(policy_bi_d_b)
print("NPV for optimal DP policy is Million $", NPV_BI_d_b*(10**-6))


# inflexible baseline comparison
NPV_inflex, model_inflex = NPV_garage_predefined_policy(policy_inflex)
print("NPV for inflexible policy is Million $", NPV_inflex*(10**-6))


NPV_ga_dr, model_ga_dr = NPV_garage_GA_DR(DV_D )
print("NPV for GA DR policy is Million $", NPV_ga_dr*(10**-6))


# plot the evolution of the various solutions

capacity_graph = capacity_evolution_plots(policy_bi_d_b , DV_D, 6)


