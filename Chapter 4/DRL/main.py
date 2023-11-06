# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:12:03 2021

@author: cesa_
"""
from matplotlib import pyplot as plt
import gym
from gym import envs
#import gymgarage
import os
import pandas as pd
from garage_RL_environment2 import agent_policy
import numpy as np
from stable_baselines.common.env_checker import check_env
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, CnnPolicy
from stable_baselines.common.policies import MlpPolicy as Mlp_a2c
from stable_baselines.common.policies import  MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf
from stable_baselines import DQN, A2C , ACKTR, ACER
from garage_ENPV_obj_arrayinput_RLsimplecomparison import NPV_garage, ENPV_MC, ENPV_MC_CDF
from garage_demand import cc_start
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
from RL_SB_helper import *
from garage_DP_helper import *
from Inflexible_baseline import *
from garage_DP_class import Garage_Complete, cc_start, Garage_Complete_demand_scenario
from backward_induction_dp import StochasticDP
from matplotlib import pyplot as plt
import numpy as np
from garage_cost import Exp_cost
from garage_demand import demand_static,  demand_stochastic_less, demand_stochastic_series
import pandas as pd



# Other policies for comparison
DV_D = np.array([1, 1, 1, 0, 2, 2, 4]) # Optimal stochastich design vector obtained with GA DR
f0 = 6 # fully inflexible optimal number of starting floors


# basic input parameters
state_space = [0,200,400,600,800,1000,1200,1400,1600,1800]
action_space= np.array([0,1,2,3, 4])
r = .12 # discount rate used when initializing environments
T = 20 # years

#importing required RL enviroments

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)
sess = tf.Session()
#delete if it's registered
env_name0 = 'garage-v0'
env_name = 'garage-v1'
env_name2 = 'garage-v2'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
if env_name0 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name0]
if env_name2 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name2]

train_env = gym.make('gym_garage:garage-v1')
test_env = gym.make('gym_garage:garage-v2')

#env = make_vec_env(lambda: genv, n_envs=1)
years = ['0' ,'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16','17','18','19','20']

# initiate agent and network
model_dqn = DQN(MlpPolicy, train_env, gamma =1,  verbose=0,exploration_fraction=0.9, exploration_final_eps=0.1, prioritized_replay=True ,  tensorboard_log="./garage_Chapter4/")
#Train the agent
model_dqn.learn(total_timesteps = 500000)
#save model
model_dqn.save("DQN_garage_good_ch4")



#THIS IS THE ONLY THING TO CHANGE, THEN CAN RUN CODE FROM HERE TO GENERATE CDF
n_scenarios = 2000 

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


def CDF_RL_DP_SA_DR_fixed_scenarios(nsim, model, env, scenario_df , policy_d, DV, f0):
    NPVd =[] # this holds results from DP policy
    NPVs =[] # this holds results for GA DR approach
    NPVf =[] # this holds results for inflexible baseline
    NPVr = [] # this holds results for RL based analysis
    for i in range(nsim - 1 ):
        NPV_d , model_d = NPV_garage_DP_BI_scenario(policy_d[i], demand_scenario = scenario_df[i])
        NPV_s  = NPV_garage_GA_DR_scenarios(DV, demand_scenario = scenario_df[i], demand = 'stochastic')
        NPV_f  = NPV_garage_inflex_scenarios(f0, demand_scenario = scenario_df[i], demand = 'stochastic') 
        NPVd.append(NPV_d)
        NPVs.append(NPV_s)
        NPVf.append(NPV_f)
    NPVr = NPVs_RL(nsim, model, env)
    EPVd = np.array(NPVd)
    EPVs = np.array(NPVs)
    EPVf = np.array(NPVf)
    EPVr = np.array(NPVr)
    ENPVd = np.mean(EPVd)
    ENPVs = np.mean(EPVs)         
    ENPVf = np.mean(EPVf)
    ENPVr = np.mean(EPVr)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_d = bx.hist(EPVd, 100, density=True, histtype='step',
                        cumulative=True, label='DP SAA')
    cdf_s = bx.hist(EPVs, 100, density=True, histtype='step',
                        cumulative=True, label='Flex DRs')
    cdf_f = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible ')    
    cdf_r = bx.hist(EPVr, 100, density=True, histtype='step',
                        cumulative=True, label='RL Flex') 
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.4, 'ENPV DP SAA: {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVs, color='darkorange', linestyle='dashed', linewidth=1)
    plt.text(-20000000, max_ylim*0.6, 'ENPV Flex DRs: {:.2f} Million $'.format(ENPVs/1000000))
    plt.axvline(ENPVf, color='darkgreen', linestyle='dashed', linewidth=1)
    plt.text(-20000000, max_ylim*0.5, 'ENPV Inflexible: {:.2f} Million $'.format(ENPVf/1000000))    
    plt.axvline(ENPVr, color='red', linestyle='dashed', linewidth=1)
    plt.text(-20000000, max_ylim*0.3, 'ENPV RL Flex: {:.2f} Million $'.format(ENPVr/1000000))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d


CDF_RL_DP_SA_DR_fixed_scenarios(n_scenarios, model_5, test_env,  scenario_df,  policy_l, DV_D, f0)
