# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:30:43 2020

@author: cesa_
"""

from garage_DP_class import Garage_Complete, cc_start, Garage_Complete_demand_scenario
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


# env = Garage_Complete(discount_rate = 0 ) # undiscounted rewards
# env_s = Garage_Complete(discount_rate = 0, demand = 'stochastic' ) # undiscounted rewards
# env_d = Garage_Complete(discount_rate = r ) # discounted rewards and determinstic demand
# env_d_s = Garage_Complete(discount_rate = r , demand = 'stochastic' ) # discounted rewards and stochastich demand
# env_d_b = Garage_Complete(discount_rate = r ) # discounted rewards and determinstic demand WITH BOUNDARY CONDITIONS
# env_d_s_b = Garage_Complete(discount_rate = r , demand = 'stochastic' ) # discounted rewards and stochastich demand WITH BOUNDARY CONDITIONS


# Scenario generation

def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df
        
     
n_scenarios = 3 # this sets number of indipendantly scenarios generated for sample average approximation algorithm
scenario_df = scenario_generator(n_scenarios)

#creating environments
env1 = Garage_Complete_demand_scenario(r , scenario_df, demand_series_index= 0 )
env2 = Garage_Complete_demand_scenario(r ,scenario_df, demand_series_index= 1 )
env3 = Garage_Complete_demand_scenario( r , scenario_df, demand_series_index= 2 )

#Setting up as many instances of problem as there are scenarios 
number_of_stages = 22
states = state_space
decisions = action_space
dp_objs = list()
dp_envs = list()
# for i in range(n_scenarios):
#     dp_objs.append(StochasticDP(number_of_stages, states, decisions, minimize = False)) 

dp1 = StochasticDP(number_of_stages, states, decisions, minimize = False)
dp2= StochasticDP(number_of_stages, states, decisions, minimize = False)
dp3 =StochasticDP(number_of_stages, states, decisions, minimize = False)
    
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
        action_possible = env1.get_valid_action(s)
        for a in action_possible:
             p1, next_state1, reward1, done1 = env1.step(s, a, t)
             p2, next_state2, reward2, done2 = env2.step(s, a, t)
             p3, next_state3, reward3, done3 = env3.step(s, a, t)# undiscounted, boundary is 0 
             dp1.add_transition(stage=t, from_state=s, decision=a, to_state=next_state1, probability=p1, contribution=reward1)
             dp2.add_transition(stage=t, from_state=s, decision=a, to_state=next_state2, probability=p2, contribution=reward2)
             dp3.add_transition(stage=t, from_state=s, decision=a, to_state=next_state3, probability=p3, contribution=reward3)


# Set boundary conditions in last stage, not sure if should be 0 or actual rewrd for that stage since we can compute it for all states
for s in state_space:
    dp1.boundary[s] = 0
    dp2.boundary[s] = 0
    dp3.boundary[s] = 0


value_bi_1, policy_bi_1 = dp1.solve() 
value_bi_2, policy_bi_2 = dp2.solve() 
value_bi_3, policy_bi_3 = dp3.solve() 





# Expected NPV calculations
nsim = 1000 # number simulations used in ENPV calculation and CDF plotting
# ENPV_NI = ENPV_MC(nsim, policy_bi)
# print("ENPV for optimal policy with stochastic  demand using discounted backward induction is Million $", ENPV_NI*(10**-6))
# ENPV_NI_1 = ENPV_MC(nsim, policy_bi_1)
# print("ENPV for optimal policy from scenario 1 is Million $", ENPV_NI_1*(10**-6))
# ENPV_NI_2 = ENPV_MC(nsim, policy_bi_2)
# print("ENPV for optimal policy from scenario 2 is Million $", ENPV_NI_2*(10**-6))
# ENPV_NI_3 = ENPV_MC(nsim, policy_bi_3)
# print("ENPV for optimal policy from scenario 3 is Million $", ENPV_NI_3*(10**-6))
# ENPV_NI_B = ENPV_MC(nsim, policy_bi_d_s_b)
# print("ENPV for optimal policy with stochastic  demand using discounted backward induction and boundaries is Million $", ENPV_NI_B*(10**-6))


#deterministic NPV calculation for each scenario
NPV1, model1 = NPV_garage_DP_BI_scenario(policy_bi_1, demand_scenario = scenario_df[0])
NPV2, model2 = NPV_garage_DP_BI_scenario(policy_bi_2, demand_scenario = scenario_df[1])
NPV3, model3 = NPV_garage_DP_BI_scenario(policy_bi_3, demand_scenario = scenario_df[2])
ENPV_SAA = np.average([NPV1, NPV2, NPV3])

print("NPV for optimal policy from scenario 1 is Million $", NPV1*(10**-6))
print("NPV for optimal policy from scenario 2 is Million $", NPV2*(10**-6))
print("NPV for optimal policy from scenario 3 is Million $", NPV3*(10**-6))

print("ENPV for optimal policy from SAA is Million $", ENPV_SAA*(10**-6))

# plotting CDF of stochastic performance comparison of DP, GA DR and inflexible designs

#CDF_DP_DR_fixed(nsim, policy_bi_d, DV_D, f0)






