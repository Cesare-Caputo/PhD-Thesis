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
from Inflexible_baseline import *
from RL_SB_helper import *
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= np.array([0,1,2,3, 4])
r = .12 # discount rate used when initializing environments
T = 20 # years

# Other policies for comparison
DV_D = np.array([1, 1, 1, 0, 2, 2, 4]) # Optimal stochastich design vector obtained with GA DR
f0 = 6 # fully inflexible optimal number of starting floors

# This sets the number of indipendantly scenarios generated for sample average approximation algorithm
# the same number of scenarios is then used as an input into the CDF function below
#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 250 

# Scenario generation
def scenario_generator(scenarios):
    scenario_df = pd.DataFrame()
    for i in range(scenarios):
        scenario_df[i] = demand_stochastic_series(T)
    return scenario_df


scenario_df = scenario_generator(n_scenarios)

#creating environments
#Setting up as many instances of problem as there are scenarios 
number_of_stages = 22
states = state_space
decisions = action_space
dp_objs = list()
dp_envs = list()
for i in range (n_scenarios):
    dp_envs.append(Garage_Complete_demand_scenario(r , scenario_df, demand_series_index= i ))
    dp_objs.append(StochasticDP(number_of_stages, states, decisions, minimize = False))
                                     
    
# This sets dp.probability[m, n, t, x] = p and dp.contribution[m, n, t, x] = c # Populating for all cases within this loop here
# list containing dp envs objects and dp objects themselves referenced here
for i in range(n_scenarios):
    for t in range(0,22):
        for s in state_space:
            action_possible = dp_envs[i].get_valid_action(s)
            for a in action_possible:
                 p, next_state, reward, done = dp_envs[i].step(s, a, t) 
                 dp_objs[i].add_transition(stage=t, from_state=s, decision=a, to_state=next_state, probability=p, contribution=reward)

# Set boundary conditions in last stage, not sure if should be 0 or actual rewrd for that stage since we can compute it for all states
for i in range(n_scenarios):
    for s in state_space:
        dp_objs[i].boundary[s] = 0

value_l= []
policy_l= []
for i in range(n_scenarios):
    value, policy = dp_objs[i].solve()
    value_l.append(value)
    policy_l.append(policy)

# #deterministic NPV calculation for each scenario
def SAA_ENPV_outofsample(n_scenarios, policy):
    NPVs =[]
    models =[]
    scenario_df = scenario_generator(n_scenarios)
    for i in range(n_scenarios):
        NPV, model = NPV_garage_DP_BI_scenario(policy, demand_scenario = scenario_df[i])
        NPVs.append(NPV)
    ENPV_SAA= np.mean(NPVs)
    return ENPV_SAA
    return ENPV_SAA

def SAA_NPV_scenarios(n_scenarios):
    NPVs =[]
    models =[]
    for i in range(n_scenarios):
        NPV, model = NPV_garage_DP_BI_scenario(policy_l[i], demand_scenario = scenario_df[i])
        NPVs.append(NPV)
    return NPVs

def CDF_DP_SA_DR_fixed(nsim, policy_list,DV, f0):
    NPVd = [] # this holds results from DP policy
    NPVs =[] # this holds results for GA DR approach
    NPVf =[] # this holds results for inflexible baseline
    for i in range(nsim):
        NPV_d , model_d = NPV_garage_DP_BI_scenario(policy_list[i], demand_scenario = scenario_df[i]) 
        NPV_s, models  = NPV_garage_GA_DR(DV, demand = 'stochastic')
        NPV_f  = NPV_garage_inflex(f0, demand = 'stochastic') 
        NPVd.append(NPV_d)
        NPVs.append(NPV_s)
        NPVf.append(NPV_f)
    EPVd = np.array(NPVd)
    EPVs = np.array(NPVs)
    EPVf = np.array(NPVf)
    ENPVd = np.mean(EPVd)
    ENPVs = np.mean(EPVs)         
    ENPVf = np.mean(EPVf)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_d = bx.hist(EPVd, 100, density=True, histtype='step',
                        cumulative=True, label='DP policy')
    cdf_s = bx.hist(EPVs, 100, density=True, histtype='step',
                        cumulative=True, label='GA DR')
    cdf_f = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible baseline')    
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.25, 'ENPV SP SAA: {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVs, color='darkorange', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.6, 'ENPV GA DR: {:.2f} Million $'.format(ENPVs/1000000))
    plt.axvline(ENPVf, color='darkgreen', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.45, 'ENPV Inflexible: {:.2f} Million $'.format(ENPVf/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions : SP vs GA DR vs Inflexible  comparison')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d

def CDF_DP_SA_DR_fixed_scenarios(nsim, scenario_df , policy_d, DV, f0):
    NPVd =[] # this holds results from DP policy
    NPVs =[] # this holds results for GA DR approach
    NPVf =[] # this holds results for inflexible baseline
    for i in range(nsim - 1 ):
        NPV_d , model_d = NPV_garage_DP_BI_scenario(policy_d[i], demand_scenario = scenario_df[i])
        NPV_s  = NPV_garage_GA_DR_scenarios(DV, demand_scenario = scenario_df[i], demand = 'stochastic')
        NPV_f  = NPV_garage_inflex_scenarios(f0, demand_scenario = scenario_df[i], demand = 'stochastic') 
        NPVd.append(NPV_d)
        NPVs.append(NPV_s)
        NPVf.append(NPV_f)
    EPVd = np.array(NPVd)
    EPVs = np.array(NPVs)
    EPVf = np.array(NPVf)
    ENPVd = np.mean(EPVd)
    ENPVs = np.mean(EPVs)         
    ENPVf = np.mean(EPVf)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_d = bx.hist(EPVd, 100, density=True, histtype='step',
                        cumulative=True, label='DP policy')
    cdf_s = bx.hist(EPVs, 100, density=True, histtype='step',
                        cumulative=True, label='Flex DRs')
    cdf_f = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible baseline')    
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.25, 'ENPV DP SAA: {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVs, color='darkorange', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.6, 'ENPV Flex DRs: {:.2f} Million $'.format(ENPVs/1000000))
    plt.axvline(ENPVf, color='darkgreen', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.45, 'ENPV Inflexible: {:.2f} Million $'.format(ENPVf/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d


def SAA_optimality_gap(n_scenarios, scenario_df,  policy, policy_l):
    NPVs_lb =[]
    NPVs_ub =[]
    models =[]
    for i in range(n_scenarios):
        NPV_lb, model_lb = NPV_garage_DP_BI_constantpolicy(policy, demand_scenario = scenario_df[i])
        NPV_ub, model_ub = NPV_garage_DP_BI_scenario(policy_l[i], demand_scenario = scenario_df[i])
        NPVs_lb.append(NPV_lb)
        NPVs_ub.append(NPV_ub)
    ENPV_SAA_ub= np.mean(NPVs_ub)
    ENPV_SAA_lb= np.mean(NPVs_lb)
    gap = ENPV_SAA_ub - ENPV_SAA_lb
    return gap 



def SAA_optimal_policy(n_scenarios, scenario_df, policy_l):
    NPVs_lb =[]
    NPVs_ub =[]
    gaps =[]
    n_policies = len(policy_l)
    for j in range(n_policies):
        policy = policy_l[j]
        for i in range(n_scenarios):
            NPV_lb, model_lb = NPV_garage_DP_BI_constantpolicy(policy, demand_scenario = scenario_df[i])
            NPV_ub, model_ub = NPV_garage_DP_BI_scenario(policy_l[i], demand_scenario = scenario_df[i])
            NPVs_lb.append(NPV_lb)
            NPVs_ub.append(NPV_ub)
        ENPV_SAA_ub= np.mean(NPVs_ub)
        ENPV_SAA_lb= np.mean(NPVs_lb)
        gap = ENPV_SAA_ub - ENPV_SAA_lb
        gaps.append(gap)
    best_gap = min(gaps)
    best_gap_index = gaps.index(min(gaps))
    best_policy = policy_l[best_gap_index]
    return best_gap, best_policy

# ENPV_SAA = SAA_ENPV(n_scenarios)

# print("ENPV for optimal policy with stochastich programming SAA is Million $", ENPV_SAA*(10**-6))


# CDF_DP_SA_DR_fixed_scenarios(n_scenarios, scenario_df,  policy_l, DV_D, f0)

# CDF_RL_DP_SA_DR_fixed_scenarios(n_scenarios, model_3, test_env,  scenario_df,  policy_l, DV_D, f0)


best_gap, best_policy = SAA_optimal_policy(n_scenarios, scenario_df, policy_l)

policy_ENPV = SAA_ENPV_outofsample(n_scenarios, best_policy)

print(" Best SAA policy ENPV is", policy_ENPV)
