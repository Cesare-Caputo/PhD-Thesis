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
from garage_demand import demand_static,  demand_stochastic_less, demand_stochastic_series
import pandas as pd
from garage_DP_helper import *
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= np.array([0,1,2,3, 4])
r = .12 # discount rate used when initializing environments
T = 20 # years

# Other policies for comparison
policy_inflex = np.array([ 6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
DV_D = np.array([1, 1, 1, 0, 2, 2, 4]) # Optimal determistic design vector with GA DR
f0 = 6 


env = Garage_Complete(discount_rate = 0 ) # undiscounted rewards
env_s = Garage_Complete(discount_rate = 0, demand = 'stochastic' ) # undiscounted rewards
env_d = Garage_Complete(discount_rate = r ) # discounted rewards and determinstic demand
env_d_s = Garage_Complete(discount_rate = r , demand = 'stochastic' ) # discounted rewards and stochastich demand
env_d_b = Garage_Complete(discount_rate = r ) # discounted rewards and determinstic demand WITH BOUNDARY CONDITIONS
env_d_s_b = Garage_Complete(discount_rate = r , demand = 'stochastic' ) # discounted rewards and stochastich demand WITH BOUNDARY CONDITIONS


#Setting up the various instances of the problem  for debugging
number_of_stages = 22
states = state_space
decisions = action_space
dp = StochasticDP(number_of_stages, states, decisions, minimize = False) # non discounted rewards
dp_s = StochasticDP(number_of_stages, states, decisions, minimize = False) # non discounted rewards, stochastic demand
dp_d = StochasticDP(number_of_stages, states, decisions, minimize = False) # discounted rewards
dp_d_s = StochasticDP(number_of_stages, states, decisions, minimize = False) # discounted reward with stochastic demand
dp_d_b = StochasticDP(number_of_stages, states, decisions, minimize = False) # discounted rewards and boundary value
dp_d_s_b = StochasticDP(number_of_stages, states, decisions, minimize = False) # discounted reward with stochastic demand and boundary value

#starting off here with sample average approach
def action_average_reward(n_sim, env, t, s, a):
    r_stoch = []
    for i in range(0,n_sim):
        p_d_s, next_state_d_s, reward_d_s, done_d_s = env.step(s, a, t)
        r_stoch.append(reward_d_s)
    avg_reward = np.mean(r_stoch)
    return avg_reward


def average_boundary_value(n_sim, env, s):
    v_stoch = []
    for i in range(0,n_sim):
        terminal_value = env.terminal_value(s)
        v_stoch.append(terminal_value)
    avg_reward = np.mean(v_stoch)
    return avg_reward


# This sets dp.probability[m, n, t, x] = p and dp.contribution[m, n, t, x] = c # Populating for all cases within this loop here
n_sim = 1000 # this sets number of simulation considered in sample average approach to calculate avg reward and temrinal value below
for t in range(0,22):
    for s in state_space:
        action_possible = env.get_valid_action(s)
        for a in action_possible:
             p, next_state, reward, done = env.step(s, a, t) # undiscounted, boundary is 0 
             p_s, next_state_s, reward_s, done_s = env_s.step(s, a , t) # discounted, boundary is 0              
             p_d, next_state_d, reward_d, done_d = env_d.step(s, a , t) # discounted, boundary is 0 
             p_d_s, next_state_d_s, reward_d_s, done_d_s = env_d_s.step(s, a , t) # discounted ,stochastic returns, boundary 0 
             p_d_b, next_state_d_b, reward_d_b, done_d_b = env_d_b.step(s, a , t) # discounted, boundary set
             p_d_s_b, next_state_d_s_b, reward_d_s_b, done_d_s_b = env_d_s_b.step(s, a, t) # discounted, stochastic returns, boundary set
             avg_reward_stoch_d = action_average_reward(n_sim, env_d_s,  t, s, a) # with boundary
             avg_reward_stoch_b = action_average_reward(n_sim, env_d_s_b , t, s, a) # with no boundary
             dp.add_transition(stage=t, from_state=s, decision=a, to_state=next_state, probability=p, contribution=reward)
             dp_s.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_s, probability=p_s, contribution=reward_s)
             dp_d.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_d, probability=p, contribution=reward_d)
             dp_d_s.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_d_s, probability=p, contribution=avg_reward_stoch_d)
             dp_d_b.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_d_b, probability=p, contribution=reward_d_b)
             dp_d_s_b.add_transition(stage=t, from_state=s, decision=a, to_state=next_state_d_s_b, probability=p, contribution=avg_reward_stoch_b)


# Set boundary conditions in last stage, not sure if should be 0 or actual rewrd for that stage since we can compute it for all states
for s in state_space:
    dp.boundary[s] = 0
    dp_s.boundary[s] = 0
    dp_d.boundary[s] = 0
    dp_d_s.boundary[s] = 0
    dp_d_b.boundary[s] = env_d.terminal_value(s)
    dp_d_s_b.boundary[s] = average_boundary_value(n_sim, env_d_s_b, s)


value_bi, policy_bi = dp.solve() # undiscounted determinstic
value_bi_s, policy_bi_s = dp_s.solve() # undiscounted stochastic
value_bi_d, policy_bi_d = dp_d.solve() # discounted determinstic
value_bi_d_s, policy_bi_d_s = dp_d_s.solve() # discounted stochastic
value_bi_d_b, policy_bi_d_b = dp_d_b.solve() # discounted determinstic with boundary values
value_bi_d_s_b, policy_bi_d_s_b = dp_d_s_b.solve() # discounted stochastic with boundary values

# Scenario generation

def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df
        
     






# Expected NPV calculations
nsim = 1000 # number simulations used in ENPV calculation and CDF plotting
# ENPV_NI = ENPV_MC(nsim, policy_bi)
# print("ENPV for optimal policy with stochastic  demand using discounted backward induction is Million $", ENPV_NI*(10**-6))
# ENPV_NI_s = ENPV_MC(nsim, policy_bi_s)
# print("ENPV for optimal policy with stochastic  demand using undiscounted backward induction, no boundary is Million $", ENPV_NI_s*(10**-6))
ENPV_NI_d_s = ENPV_MC(nsim, policy_bi_d_s)
print("ENPV for optimal policy with stochastic  demand using discounted backward induction is Million $", ENPV_NI_d_s*(10**-6))
# ENPV_NI_DET = ENPV_MC(nsim, policy_bi_d)
# print("ENPV for optimal DETERMINSTIC policy under stochastic demand using discounted backward induction is Million $", ENPV_NI_DET*(10**-6))
# ENPV_NI_B = ENPV_MC(nsim, policy_bi_d_s_b)
# print("ENPV for optimal policy with stochastic  demand using discounted backward induction and boundaries is Million $", ENPV_NI_B*(10**-6))


# plotting CDF of stochastic performance comparison of DP, GA DR and inflexible designs

CDF_DP_DR_fixed(nsim, policy_bi_d, DV_D, f0)






