# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:12:48 2021

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
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf
from stable_baselines import DQN
from garage_ENPV_obj_arrayinput_RLsimplecomparison import NPV_garage, ENPV_MC, ENPV_MC_CDF
from garage_demand import cc_start
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
cc_initial_fixed_2floor = 6400000

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    NPVs =[]
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
    all_episode_rewards.append(sum(episode_rewards))
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward before train:", mean_episode_reward)

#This returns the history of actions and demand for number of episodes, with inputs of environment, model and number of episodes respectively

def RL_history_2(env, model, episodes):
    all_episode_rewards = []
    env = env
    actions_dr = []
    states =[]
    for i in range(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            actions_dr.append(action)
            states.append(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
    return actions_dr, states


# this calculates decision rules used every time expansion in terms of years where demand>capacity
def RL_DR2(env, model, episodes):
    rl_act , rl_state = RL_history_2(env, model, episodes) 
    DRS = []
    n_steps = 20
    for a in range(episodes*n_steps):
        if rl_act[a] != 0 : 
            if rl_state[a-1][0] > rl_state[a-1][1] and rl_state[a-2][0] > rl_state[a-2][1] and rl_state[a-3][0] > rl_state[a-3][1]:
                DRS.append(3)
            elif rl_state[a-1][0] > rl_state[a-1][1] and rl_state[a-2][0] > rl_state[a-2][1]:
                DRS.append(2)
            elif rl_state[a-1][0] > rl_state[a-1][1]:
                DRS.append(1)
        #else: DRS.append(0)
    return DRS

def RL_DR_hist(env, model, episodes):
    DRS =RL_DR2(env, model, episodes)
    pct_wt = 1/len(DRS)
    DR_pct = [i *pct_wt for i in DRS]
    DR_avg = np.mean(DRS)
    fig, bx = plt.subplots(figsize=(8, 4))
    bins = [ .5, 1.5, 2.5, 3.5]
    ticks = [0,1,2,3]
    cdf_dr_rl = bx.hist(DRS,bins, weights=np.ones(len(DRS)) / len(DRS), label='RL Agent with stochastic demand')
    plt.axvline(DR_avg, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(DR_avg*.7, max_ylim*0.9, 'Average Rule: {:.2f}'.format(DR_avg))
    bx.grid(True)
    bx.set_xticks(ticks)
    #bx.legend(loc='center right')
    bx.set_title(' Decision Rules Histogram DQN Agent ')
    bx.set_xlabel('Decision Rule')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return cdf_dr_rl


def RL_FT_hist(env, model, episodes):
    rl_act , rl_state = RL_history_2(env, model, episodes) 
    pct_wt = 1/len(rl_act)
    FT_pct = [i *pct_wt for i in rl_act]
    FT_avg = np.mean(rl_act)
    fig, bx = plt.subplots(figsize=(8, 4))
    bins = [ .5, 1.5, 2.5, 3.5]
    ticks = [0,1,2,3]
    cdf_dr_rl = bx.hist(rl_act , bins, weights=np.ones(len(rl_act)) / len(rl_act), label='RL Agent with stochastic demand')
    plt.axvline(FT_avg, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(FT_avg*.7, max_ylim*0.9, 'Average Rule: {:.2f}'.format(FT_avg))
    bx.grid(True)
    bx.set_xticks(ticks)
    #bx.legend(loc='center right')
    bx.set_title(' Decision Rules Histogram DQN Agent ')
    bx.set_xlabel('Floors expanded by')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return cdf_dr_rl

# def CDF_RL_GA_simple(episodes, plan, genv):
#     env = genv
#     NPVs =[]
#     GA_NPVS =[]
#     for i in range(episodes):
#         NPV_GA = NPV_garage(plan, demand = 'stochastic') 
#         GA_NPVS.append(NPV_GA)
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#             episode_rewards.append(reward)
#         NPVs.append((sum(episode_rewards)))
#     GA_NPVAr = np.array(GA_NPVS)
#     NPVar = np.array(NPVs)
#     ENPV_rl = np.mean(NPVar)
#     ENPV_ga = np.mean(GA_NPVAr)
#     fig, bx = plt.subplots(figsize=(8, 4))
#     #NPV = np.array[]
#     cdf_r = bx.hist(NPVar, 100, density=True, histtype='step',
#                         cumulative=True, label='RL agent performance')
#     cdf_g = bx.hist(GA_NPVAr, 100, density=True, histtype='step',
#                         cumulative=True, label='GA agent performance')
#     plt.axvline(ENPV_rl, color='dodgerblue', linestyle='dashed', linewidth=1)
#     min_ylim, max_ylim = plt.ylim()
#     plt.text(-10000000, max_ylim*0.75, 'ENPV RL: {:.2f} Million $'.format(ENPV_rl/1000000))
#     plt.axvline(ENPV_ga, color='darkorange', linestyle='dashed', linewidth=1)
#     min_ylim, max_ylim = plt.ylim()
#     plt.text(-10000000, max_ylim*0.6, 'ENPV GA: {:.2f} Million $'.format(ENPV_ga/1000000))
#     bx.grid(True)
#     bx.legend(loc='upper left')
#     bx.set_title('CDF of design solutions : Simplified Case')
#     bx.set_xlabel('NPV of plan($)')
#     bx.set_ylabel('Probability')
#     return cdf_r

def CDF_RL_GA_simple_fixedRL(episodes, plan_ga , plan_rl, genv):
    env = genv
    RL_NPVS =[]
    GA_NPVS =[]
    for i in range(episodes):
        NPV_GA = NPV_garage(plan_ga, demand = 'stochastic') - 3200000 
        GA_NPVS.append(NPV_GA)
        NPV_RL = NPV_garage(plan_rl, demand = 'stochastic') - 3200000
        RL_NPVS.append(NPV_RL)
    GA_NPVAr = np.array(GA_NPVS)
    RL_NPVAr = np.array(RL_NPVS)
    ENPV_rl = np.mean(RL_NPVAr)
    ENPV_ga = np.mean(GA_NPVAr)
    fig, bx = plt.subplots(figsize=(8, 4))
    #NPV = np.array[]
    cdf_r = bx.hist(RL_NPVAr, 100, density=True, histtype='step',
                        cumulative=True, label='RL agent performance')
    cdf_g = bx.hist(GA_NPVAr, 100, density=True, histtype='step',
                        cumulative=True, label='GA agent performance')
    plt.axvline(ENPV_rl, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-10000000, max_ylim*0.75, 'ENPV RL: {:.2f} Million $'.format(ENPV_rl/1000000))
    plt.axvline(ENPV_ga, color='darkorange', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-10000000, max_ylim*0.6, 'ENPV GA: {:.2f} Million $'.format(ENPV_ga/1000000))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions : Simplified Case with fixed plan')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_r


def CDF_RL(episodes, model , test_env):
    env = test_env
    NPVs =[]
    fixed_cost_adj = 3600000
    for i in range(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            NPV_adj = (sum(episode_rewards)) + (fixed_cost_adj/(1.12**20))
        NPVs.append(NPV_adj)
    NPVar = np.array(NPVs )
    ENPV_rl = np.mean(NPVar)
    fig, bx = plt.subplots(figsize=(8, 4))
    #NPV = np.array[]
    cdf_r = bx.hist(NPVar, 100, density=True, histtype='step',
                        cumulative=True, label='RL agent performance')

    plt.axvline(ENPV_rl, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(0, max_ylim*0.75, 'ENPV RL: {:.2f} Million $'.format(ENPV_rl/1000000))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of  RL based design solutions ')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_r



def NPVs_RL(episodes, model , test_env):
    env = test_env
    NPVs =[]
    fixed_cost_adj = 3600000
    for i in range(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            NPV_adj = (sum(episode_rewards)) + (fixed_cost_adj/(1.12**20))
        NPVs.append(NPV_adj)
    return NPVs



def CDF_RL_simple(episodes, model):
    all_episode_rewards = []
    env = model.get_env()
    NPVs =[]
    for i in range(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            all_episode_rewards.append(sum(episode_rewards))
            NPVs.append((sum(episode_rewards)) - cc_start(1))
    NPVar = np.array(NPVs)
    ENPV = np.mean(NPVar)
    fig, bx = plt.subplots(figsize=(8, 4))
    cdf_rl = bx.hist(NPVar, 100, density=True, histtype='step',
                       cumulative=True, label='RL agent performance')
    plt.axvline(ENPV, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(ENPV*1.1, max_ylim*0.9, 'ENPV: {:.2f}'.format(ENPV))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solution')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_rl


def agent_test(model):
    genv = model.get_env()
    obs = genv.reset()
    n_steps = 20
    returns = 0
    actions =[]
    capacities = []
    for step in range(n_steps):
      action, _ = model.predict(obs)
      print("Step {}".format(step ))
      print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done)
      print('action probabilities', model.action_probability(obs))
      capacities.append(obs)
      returns += reward
      genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate NPV=", returns )
        break    




def agent_test_env(model, env):
    genv = env
    obs = genv.reset()
    n_steps = 20
    returns = 0
    actions =[]
    capacities = []
    for step in range(n_steps):
      action, _ = model.predict(obs)
      print("Step {}".format(step ))
      print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done)
      print('action probabilities', model.action_probability(obs))
      capacities.append(obs)
      returns += reward
      genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate NPV=", returns )
        break    


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
                        cumulative=True, label='DP policy')
    cdf_s = bx.hist(EPVs, 100, density=True, histtype='step',
                        cumulative=True, label='Flex DRs')
    cdf_f = bx.hist(EPVf, 100, density=True, histtype='step',
                        cumulative=True, label='Inflexible baseline')    
    cdf_r = bx.hist(EPVr, 100, density=True, histtype='step',
                        cumulative=True, label='RL Flex') 
    plt.axvline(ENPVd, color='dodgerblue', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(-20000000, max_ylim*0.35, 'ENPV DP SAA: {:.2f} Million $'.format(ENPVd/1000000))
    plt.axvline(ENPVs, color='darkorange', linestyle='dashed', linewidth=1)
    plt.text(-20000000, max_ylim*0.6, 'ENPV Flex DRs: {:.2f} Million $'.format(ENPVs/1000000))
    plt.axvline(ENPVf, color='darkgreen', linestyle='dashed', linewidth=1)
    plt.text(-20000000, max_ylim*0.45, 'ENPV Inflexible: {:.2f} Million $'.format(ENPVf/1000000))    
    plt.axvline(ENPVr, color='red', linestyle='dashed', linewidth=1)
    plt.text(-20000000, max_ylim*0.25, 'ENPV RL FLex: {:.2f} Million $'.format(ENPVf/1000000))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solutions')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_d


#THIS is original version of RL history function which was defined inside of main script
# def RL_history(episodes):
#     all_episode_rewards = []
#     env = genv
#     actions_dr = []
#     states =[]
#     for i in range(episodes):
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs)
#             actions_dr.append(action)
#             states.append(obs)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#     return actions_dr, states


#GA optimization

results = dict()

#res_g = differential_evolution(NPV_garage)


#resgastoch = CF_model_out(results['GA round2'])


# x = years
# y = cap
# y2 = dem
# plt.figure(figsize=(12,5))
# plt.xlabel('Demand vs capacity')
# plt.ylabel('Capacity/Demand')
# capr = plt.plot(x,y, label = 'RL Agent')
# demg = plt.plot(x, y2, label = 'Stochastic Demand')
# capg = plt.plot(x,y3, label = 'GA Agent')
# statd = plt.plot(x, y4, label = 'Static Demand')
# plt.legend(loc='upper left')
# plt.title('Demand and capacity evolution over project lifetime')
# plt.xlabel('Years')
# plt.ylabel('Parking spaces')
# plt.show()

