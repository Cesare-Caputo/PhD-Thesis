# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:12:48 2021

@author: cesa_
"""

from matplotlib import pyplot as plt
import gym
from gym import envs
import os
import pandas as pd
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
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
import itertools

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


def debug_rl(model, env , n_steps):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = env
    revenues =[]
    expansion_cost = []
    obs = env.reset()
    for step in range(n_steps):
      action, _ = model.predict(obs ,   deterministic = True)
      print("Step {}".format(step ))
      print("Action: ", action)
      obs, reward, done, info = env.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done)
      print(info)
      #revenues.append(info["Income $"])
      expansion_cost.append(info["Expansion Cost"])
      env.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!" )
        break    


def check_rl_model_dr_prob(model, observation):
    obs = observation
    action, _ = model.predict(obs, deterministic = False)
    print("Action: ", action)
    print('action probabilities', model.action_probability(obs))
    





def agent_test_env_debug(model, env):
    genv = env
    obs = genv.reset()
    n_steps = 20
    returns = 0
    actions =[]
    capacities = []
    initial_capex = 24335012
    for step in range(n_steps):
      action, _ = model.predict(obs)
      print("Step {}".format(step ))
      print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done)
      print('action probabilities', model.action_probability(obs))
      print(info)
      capacities.append(obs)
      returns += reward
      #genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate NPV=", returns 
 )
        break    

def agent_test_env_lstm (model, env):
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
      print(info)
      capacities.append(obs)
      returns += reward
      #genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate NPV=", returns 
 )
        break    



def RL_action_hist_from_list(actions_list):
    non_zero_act =[]
    for j in actions_list:
        act = j
        if j != 0 : non_zero_act.append(j)
    
    DRS = non_zero_act
    pct_wt = 1/len(DRS)
    DR_pct = [i *pct_wt for i in DRS]
    DR_avg = np.mean(DRS)
    fig, bx = plt.subplots(figsize=(8, 4))
    bins = [ .5, 1.5, 2.5, 3.5 , 4.5, 5.5, 6.5]
    ticks = [0,1,2,3 ,4, 5, 6]
    cdf_dr_rl = bx.hist(DRS,bins, weights=np.ones(len(DRS)) / len(DRS), label='DRL Low Flex ')
    plt.axvline(DR_avg, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(3, max_ylim*0.9, 'Average Expansion: {:.2f}'.format(DR_avg))
    bx.grid(True)
    bx.set_xticks(ticks)
    #bx.legend(loc='center right')
    bx.set_title(' Expansion Decision Histogram DR approach ')
    bx.set_xlabel('Decision Rule')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return cdf_dr_rl



def RL_action_hist_from_list_highflex(actions_list):
    non_zero_act =[]
    for j in actions_list:
        act = j
        if j != 0 : non_zero_act.append(j)
    
    DRS = non_zero_act
    pct_wt = 1/len(DRS)
    DR_pct = [i *pct_wt for i in DRS]
    DR_avg = np.mean(DRS)
    fig, bx = plt.subplots(figsize=(8, 4))
    bins = [ .5, 1.5, 2.5, 3.5 , 4.5, 5.5, 6.5 , 7.5 , 8.5 , 9.5, 10.5 , 11.5 , 12.5 , 13.5 ,14.5 , 15.5, 16.5 , 17.5, 18.5]
    ticks = [0,1,2,3 ,4, 5, 6 , 7 , 8 , 9, 10, 11, 12, 13, 14,  15, 16, 17 , 18]
    cdf_dr_rl = bx.hist(DRS,bins, weights=np.ones(len(DRS)) / len(DRS), label='DRL Low Flex ')
   # plt.axvline(DR_avg, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    #plt.text(3, max_ylim*0.9, 'Average Expansion: {:.2f}'.format(DR_avg))
    bx.grid(True)
    bx.set_xticks(ticks)
    #bx.legend(loc='center right')
    bx.set_title(' Expansion Decision Histogram High Flex DRL approach ')
    bx.set_xlabel('Decision Rule')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return cdf_dr_rl

def RL_history_1_det(env, model, episodes):
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
            action, _states = model.predict(obs , deterministic = True)
            actions_dr.append(action)
            states.append(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
    return actions_dr, states



def demand_capacity_obs_concat(capacity_obs, demand_history, time_step):
    time_left = 16 - time_step
    rl_obs = np.array([capacity_obs[0], demand_history[0][time_step] ,capacity_obs[1], demand_history[1][time_step] , capacity_obs[2],
                       demand_history[2][time_step] , capacity_obs[3], demand_history[3][time_step] , capacity_obs[4], demand_history[4][time_step] , 
                       capacity_obs[5], demand_history[5][time_step], time_left ])
    return rl_obs                  


def capacities_array_from_obs(obs):
    cap_ar = np.array([obs[0], obs[2] , obs[4] , obs[6] , obs[8] , obs[10]])
    return cap_ar
    

def RL_capacities_history_det(env, model, demand_history):
    all_episode_rewar = []
    env = env
    actions_dr = []
    states =[]
    starting_capacities = [0,0,0,0,0,0]
    capacities = []
    episode_steps = 16
    all_obs = []
    done = False
    obs = env.reset()
    while not done:
        for i in range(episode_steps):
            action, _states = model.predict(obs , deterministic = True)
            actions_dr.append(action)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs_new, reward, done, info = env.step(action)
            all_episode_rewar.append(reward)
            new_capacity_obs = capacities_array_from_obs(obs_new)
            obs = demand_capacity_obs_concat(new_capacity_obs, demand_history, i)
            all_obs.append(new_capacity_obs)
            #print('obs=', obs, 'reward=', reward, 'step=', i)
            #print('action probabilities', model.action_probability(obs))
            #print(obs , 'step' , i)
    #NPV = np.sum(all_episode_rewar)
    return all_obs



def RL_capacities_history_stoch(env, model, demand_history):
    all_episode_rewar = []
    env = env
    actions_dr = []
    states =[]
    starting_capacities = [0,0,0,0,0,0]
    capacities = []
    episode_steps = 16
    all_obs = []
    done = False
    obs = env.reset()
    while not done:
        for i in range(episode_steps):
            action, _states = model.predict(obs , deterministic = False)
            actions_dr.append(action)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs_new, reward, done, info = env.step(action)
            all_episode_rewar.append(reward)
            new_capacity_obs = capacities_array_from_obs(obs_new)
            obs = demand_capacity_obs_concat(new_capacity_obs, demand_history, i)
            all_obs.append(new_capacity_obs)
            #print('obs=', obs, 'reward=', reward, 'step=', i)
            #print('action probabilities', model.action_probability(obs))
            #print(obs , 'step' , i)
    #NPV = np.sum(all_episode_rewar)
    return all_obs




def RL_probs_history_stoch(env, model, demand_history):
    all_episode_rewar = []
    env = env
    actions_dr = []
    states =[]
    starting_capacities = [0,0,0,0,0,0]
    capacities = []
    episode_steps = 16
    action_probs = []
    done = False
    obs = env.reset()
    while not done:
        for i in range(episode_steps):
            action, _states = model.predict(obs , deterministic = False)
            actions_dr.append(action)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs_new, reward, done, info = env.step(action)
            all_episode_rewar.append(reward)
            new_capacity_obs = capacities_array_from_obs(obs_new)
            obs = demand_capacity_obs_concat(new_capacity_obs, demand_history, i)
            #all_obs.append(new_capacity_obs)
            #print('obs=', obs, 'reward=', reward, 'step=', i)
            #print('action probabilities', model.action_probability(obs))
            action_probs.append(model.action_probability(obs))
            #print(obs , 'step' , i)
    #NPV = np.sum(all_episode_rewar)
    return action_probs




def RL_capacities_history_bysector_adj(env, model, demand_history):
    rl_cap_hist = RL_capacities_history_bysector(env, model, demand_history)
    #rl_cap_hist[rl_cap_hist > 600] = 600
    rl_cap_hist_new = np.minimum(rl_cap_hist, 600)
    for i in range(6):
        rl_cap_hist_new[i][0] = np.minimum(rl_cap_hist_new[i][0] , 0)
    return rl_cap_hist_new


def RL_capacities_history_bysector_adj_det(env, model, demand_history):
    rl_cap_hist = RL_capacities_history_bysector_det(env, model, demand_history)
    #rl_cap_hist[rl_cap_hist > 600] = 600
    rl_cap_hist_new = np.minimum(rl_cap_hist, 600)
    for i in range(6):
        rl_cap_hist_new[i][0] = np.minimum(rl_cap_hist_new[i][0] , 0)
    return rl_cap_hist_new


def RL_capacities_history_bysector(env, model, demand_history):
    rl_hist = RL_capacities_history_stoch(env, model, demand_history)
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 =[]
    for i in range(16):
        s1.append(rl_hist[i][0])
        s2.append(rl_hist[i][1])
        s3.append(rl_hist[i][2])
        s4.append(rl_hist[i][3])
        s5.append(rl_hist[i][4])
        s6.append(rl_hist[i][5])
    return np.array([s1, s2, s3, s4, s5, s6])

def RL_capacities_history_bysector_det(env, model, demand_history):
    rl_hist = RL_capacities_history_det(env, model, demand_history)
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []
    s6 =[]
    for i in range(16):
        s1.append(rl_hist[i][0])
        s2.append(rl_hist[i][1])
        s3.append(rl_hist[i][2])
        s4.append(rl_hist[i][3])
        s5.append(rl_hist[i][4])
        s6.append(rl_hist[i][5])
    return np.array([s1, s2, s3, s4, s5, s6])


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
            action, _states = model.predict(obs , deterministic = True)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            NPV_adj = (sum(episode_rewards)) 
        NPVs.append(NPV_adj)
    return NPVs

def ENPVs_RL(episodes, model, test_env):
    NPV_list = NPVs_RL(episodes, model , test_env)
    ENPV = np.mean(NPV_list)
    return ENPV

def NPVs_RL_lstm( n_episodes, model):
    env = model.get_env()
    obs = env.reset()
    NPVs =[]
    n_steps = 17
    # Passing state=None to the predict function means
    # it is the initial state
    state = None
    # When using VecEnv, done is a vector
    for _ in range(n_episodes):
        done = [False for _ in range(env.num_envs)]
        episode_rewards = []
        for step in range (n_steps):
            # We need to pass the previous state and a mask for recurrent policies
            # to reset lstm state when a new episode begin
            action, state = model.predict(obs, state=state, mask=done)
            obs, reward , done, _ = env.step(action)
            episode_rewards.append(reward)
        NPV = np.sum(episode_rewards)
        NPVs.append(NPV/(env.num_envs)) # divide by number of environments to obtain true NPV per episode
    return NPVs

def RL_history_lstm(model, n_episodes):
    env = model.get_env()
    obs = env.reset()
    actions_dr = []
    states =[]
    n_steps = 17
    state = None
    # When using VecEnv, done is a vector
    for _ in range(n_episodes):
        done = [False for _ in range(env.num_envs)]
        episode_rewards = []
        for step in range (n_steps):
            # We need to pass the previous state and a mask for recurrent policies
            # to reset lstm state when a new episode begin
            action, state = model.predict(obs, state=state, mask=done)
            obs, reward , done, _ = env.step(action)
            actions_dr.append(action)
            states.append(state)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
    return actions_dr, states


def ENPVs_RL_lstm(episodes, model):
    NPV_list = NPVs_RL_lstm( episodes, model)
    ENPV = np.mean(NPV_list)
    return ENPV



def NPVs_RL_stoch(episodes, model , test_env):
    env = test_env
    NPVs =[]
    fixed_cost_adj = 0
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
            NPV_adj = (sum(episode_rewards) - fixed_cost_adj ) 
        NPVs.append(NPV_adj)
    return NPVs

def ENPVs_RL_stoch(episodes, model, test_env):
    NPV_list = NPVs_RL_stoch(episodes, model , test_env)
    ENPV = np.mean(NPV_list)
    return ENPV



def CDF_RL_simple(episodes, model, env):
    all_episode_rewards = []
    NPV_list = NPVs_RL(episodes, model, env)
    NPVar = np.array(NPV_list)
    ENPV = np.mean(NPVar)
    fig, bx = plt.subplots(figsize=(10, 5))
    cdf_rl = bx.hist(NPVar, 100, density=True, histtype='step',
                       cumulative=True, label='RL AKCTR agent ')
    plt.axvline(ENPV, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(ENPV*.75, max_ylim*0.9, 'ENPV: {:.2f}'.format(ENPV))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of design solution - Flexible Decentralized')
    bx.set_xlabel('NPV of plan($)')
    bx.set_ylabel('Probability')
    return cdf_rl


def CDF_RL_comparison_wte(nsim, rl_model , test_env, centralised_r_npvs, centralised_flex_npvs, decentralised_flex_npvs):
    NPVs_model1 = np.array(NPVs_RL(nsim, rl_model, test_env))
    NPVs_model2 = centralised_r_npvs
    NPVs_model3 = centralised_flex_npvs
    NPVs_model4 = decentralised_flex_npvs
    # calculate ENPV for each model
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_1 = bx.hist( NPVs_model1 , 100,  density=True, histtype='step',
                        cumulative=True, label='DRL Based Design')
    cdf_2 = bx.hist(NPVs_model2 , 100, density=True, histtype='step',
                        cumulative=True, label='Centralised Inflexible')    
    cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
                        cumulative=True, label='Centralised Flexible')
    cdf_4 = bx.hist(NPVs_model4 , 100, density=True, histtype='step',
                        cumulative=True, label='Decentralised Flexible')           
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.35, 'ENPV DQN MLP: {:.2f} Million $'.format(ENPV1/1000000))
    plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.6, 'ENPV Improved DQN MLP: {:.2f} Million $'.format(ENPV2/1000000))
    plt.axvline(ENPV3, color='darkgreen', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.45, 'ENPV DQN FF: {:.2f} Million $'.format(ENPV3/1000000))
    plt.axvline(ENPV4, color='red', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.25, 'ENPV A2C MLP: {:.2f} Million $'.format(ENPV4/1000000))
    #plt.text(-20000000, max_ylim*0.85, 'ENPV A2C LSTM small: {:.2f} Million $'.format(ENPV5/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different design alternatives')
    bx.set_xlabel('NPV of model($)')
    bx.set_ylabel('Probability')    
    return cdf_1


def CDF_RL_comparison_wte_2rl(nsim, rl_model , rl_model2, test_env, flex_env,  centralised_r_npvs, centralised_flex_npvs, decentralised_flex_npvs):
    NPVs_model0 = np.array(NPVs_RL_stoch(nsim, rl_model, test_env))
    NPVs_model1 = np.array(NPVs_RL(nsim, rl_model2, flex_env))
    NPVs_model2 = centralised_r_npvs
    NPVs_model3 = centralised_flex_npvs
    NPVs_model4 = decentralised_flex_npvs
    # calculate ENPV for each model
    ENPV0 = np.mean(NPVs_model0)
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    fig, bx = plt.subplots(figsize=(8, 4))
    bins_lf = sorted(NPVs_model0) + [np.inf]
    bins_hf = sorted(NPVs_model1) + [np.inf]
    bins_ci = sorted(NPVs_model2) + [np.inf]
    bins_cf = sorted(NPVs_model3) + [np.inf]
    bins_df = sorted(NPVs_model4) + [np.inf]
    cdf_0 = bx.hist( NPVs_model0 , bins_lf, marker = 'o' , density=True, histtype='step',
                        cumulative=True, label='DRL Based Design - Low Flex')     
    cdf_1 = bx.hist( NPVs_model1 , '^', bins_hf,  density=True, histtype='step',
                        cumulative=True, label='DRL Based Design - High Flex')
    cdf_2 = bx.hist(NPVs_model2 , 's', bins_ci, density=True, histtype='step',
                        cumulative=True, label='Centralised Inflexible')    
    cdf_3 = bx.hist(NPVs_model3 , 'X' , bins_cf, density=True, histtype='step',
                        cumulative=True, label='Centralised Flexible')
    cdf_4 = bx.hist(NPVs_model4 , 'P', bins_df, density=True, histtype='step',
                        cumulative=True, label='Decentralised Flexible')           
    min_ylim, max_ylim = plt.ylim()
    #plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.35, 'ENPV DQN MLP: {:.2f} Million $'.format(ENPV1/1000000))
    #plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.6, 'ENPV Improved DQN MLP: {:.2f} Million $'.format(ENPV2/1000000))
    #plt.axvline(ENPV3, color='darkgreen', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.45, 'ENPV DQN FF: {:.2f} Million $'.format(ENPV3/1000000))
    #plt.axvline(ENPV4, color='red', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.25, 'ENPV A2C MLP: {:.2f} Million $'.format(ENPV4/1000000))
    #plt.text(-20000000, max_ylim*0.85, 'ENPV A2C LSTM small: {:.2f} Million $'.format(ENPV5/1000000))    
    bx.grid(False)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different design alternatives')
    bx.set_xlabel('NPV of design($)')
    bx.set_ylabel('Probability')    
    return fig
    
def CDF_RL_comparison_wte_2rl_nohist(nsim, rl_model , rl_model2, test_env, flex_env,  centralised_r_npvs, centralised_flex_npvs, decentralised_flex_npvs):
    NPVs_model0 = np.array(NPVs_RL_stoch(nsim, rl_model, test_env))
    NPVs_model1 = np.array(NPVs_RL(nsim, rl_model2, flex_env))
    NPVs_model2 = centralised_r_npvs
    NPVs_model3 = centralised_flex_npvs
    NPVs_model4 = decentralised_flex_npvs
    # calculate ENPV for each model
    ENPV0 = np.mean(NPVs_model0)
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    
    
    
    fig, bx = plt.subplots(figsize=(8, 4))
    # build distributions so can add marker instead of using hist method
    bins_lf = np.sort(NPVs_model0) 
    n_lf = np.arange(1,len(bins_lf)+1) / np.float(len(bins_lf))
    bins_hf = np.sort(NPVs_model1) 
    n_hf = np.arange(1,len(bins_hf)+1) / np.float(len(bins_hf))    
    x_ci = np.sort(NPVs_model2) 
    n_ci = np.arange(len(x_ci)) / np.float(len(x_ci))    
    bins_cf = np.sort(NPVs_model3) 
    n_cf = np.arange(1,len(bins_cf)+1) / np.float(len(bins_cf))    
    x_df = np.sort(NPVs_model4) 
    n_df = np.arange(len(x_df)) / np.float(len(x_df))        

    plt.plot(bins_lf , n_lf , 'k' , marker = 'o' , markevery = 100,  label='DRL Based Design - High Flex')
    plt.plot(bins_hf , n_hf , 'k' ,marker = '^' , markevery = 100,label='DRL Based Design - Low Flex')
    plt.plot(x_ci , n_ci ,'k' ,marker = 's' , markevery = 25,label='Centralised Inflexible')
    plt.plot(bins_cf , n_cf ,'k' ,marker = 'X' , markevery = 100,label='Centralised Flexible')
    plt.plot(x_df , n_df ,'k' , marker = 'P' , markevery = 25, label='Decentralised Flexible')
    
          
    min_ylim, max_ylim = plt.ylim()
    # plt.axvline(ENPV0, linestyle='--' , marker = 'o' , markevery = [.2 ])
    # plt.axvline(ENPV1, linestyle='--' , marker = '^', markevery = .2)
    # plt.axvline(ENPV2, linestyle='--' , marker = 's' , markevery = .2)
    # plt.axvline(ENPV3, linestyle='--' , marker = 'X', markevery = .2)
    # plt.axvline(ENPV4, linestyle='--' , marker = 'P', markevery = .2)
    #plt.style.use('grayscale')
    bx.grid(False)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different design alternatives')
    bx.set_xlabel('Net Present Value of design($)')
    bx.set_ylabel('Probability')    
    return fig


def CDF_RL_comparison_wte_2rl_nohist_nocfnsim(nsim, rl_model , rl_model2, test_env, flex_env, x_ci, x_cf, x_df):
    mil_conv = 10**(-6)
    NPVs_model0 = np.array(NPVs_RL_stoch(nsim, rl_model, test_env)) * mil_conv
    NPVs_model1 = np.array(NPVs_RL(nsim, rl_model2, flex_env)) * mil_conv
    NPVs_model2 = x_ci * mil_conv
    NPVs_model3 = x_cf * mil_conv
    NPVs_model4 = x_df * mil_conv
    
    
    
    
    # calculate ENPV for each model
    ENPV0 = np.mean(NPVs_model0)
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    
    
    
    
    
    fig, bx = plt.subplots(figsize=(8, 4))
    # build distributions so can add marker instead of using hist method
    bins_lf = np.sort(NPVs_model0) 
    n_lf = np.arange(1,len(bins_lf)+1) / np.float(len(bins_lf))
    bins_hf = np.sort(NPVs_model1) 
    n_hf = np.arange(1,len(bins_hf)+1) / np.float(len(bins_hf))    
    x_ci = np.sort(NPVs_model2) 
    n_ci = np.arange(len(x_ci)) / np.float(len(x_ci))    
    bins_cf = np.sort(NPVs_model3) 
    n_cf = np.arange(1,len(bins_cf)+1) / np.float(len(bins_cf))    
    x_df = np.sort(NPVs_model4) 
    n_df = np.arange(len(x_df)) / np.float(len(x_df))        
    
    plt.plot(bins_lf , n_lf , 'g' , marker = 'o' , markevery = 100,  label='DRL-HF'  ,markersize= 5)
    plt.plot(bins_hf , n_hf , '--b' ,marker = '^' , markevery = 100,label='DRL-LF' , markersize= 5)
    plt.plot(x_ci , n_ci ,'-.r' ,marker = 's' , markevery = 25,label='Centralised Inflexible' , markersize= 5)
    #plt.plot(bins_cf , n_cf ,'k' ,marker = 'X' , markevery = 100,label='Centralised Flexible')
    plt.plot(x_df , n_df ,':c' , marker = 'P' , markevery = 25, label='Decentralised FDR', markersize= 5)
    
    
    #plo ENPV lines
    
    min_ylim, max_ylim = plt.ylim()
    
    
    enpv_tick_marks =np.array([min_ylim, .1, .3, .5 , .7, .9, max_ylim])
    plt.plot(np.repeat(ENPV0 , 7) , enpv_tick_marks ,  'g' , marker = 'o'  , markersize= 4 , linewidth= 1)
    plt.plot(np.repeat(ENPV1 , 7) , enpv_tick_marks ,  '--b' , marker = '^'  , markersize= 4 , linewidth= 1)
    plt.plot(np.repeat(ENPV2 , 7) , enpv_tick_marks ,  '-.r' , marker = 's' , markersize= 4 , linewidth= 1)
    plt.plot(np.repeat(ENPV4 , 7) , enpv_tick_marks ,  ':c' , marker = 'P'  , markersize= 5 , linewidth= 1)
    
          
    

    bx.grid(False)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different design alternatives')
    bx.set_xlabel('Net Present Value of Design (S$ Million)')
    bx.set_ylabel('Cumulative Probability') 
    
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    #plt.tight_layout()
    return fig

def RL_comparison_wte_2rl_npvlist(nsim, rl_model , rl_model2, test_env, flex_env):
    NPVs_model0 = np.array(NPVs_RL_stoch(nsim, rl_model, test_env))
    NPVs_model1 = np.array(NPVs_RL(nsim, rl_model2, flex_env))
    return NPVs_model0, NPVs_model1 



def CDF_RL_comparison_wte_2rl_1lstm(nsim, rl_model , rl_model2, flex_env,  centralised_r_npvs, centralised_flex_npvs, decentralised_flex_npvs):
    NPVs_model0 = np.array(NPVs_RL_lstm(nsim, rl_model))
    NPVs_model1 = np.array(NPVs_RL_stoch(nsim, rl_model2, flex_env))
    NPVs_model2 = centralised_r_npvs
    NPVs_model3 = centralised_flex_npvs
    NPVs_model4 = decentralised_flex_npvs
    # calculate ENPV for each model
    ENPV0 = np.mean(NPVs_model0)
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    cdf_0 = bx.hist( NPVs_model0 , 100,  density=True, histtype='step',
                        cumulative=True, label='DRL Based Design - Low Flex')     
    cdf_1 = bx.hist( NPVs_model1 , 100,  density=True, histtype='step',
                        cumulative=True, label='DRL Based Design - High Flex')
    cdf_2 = bx.hist(NPVs_model2 , 100, density=True, histtype='step',
                        cumulative=True, label='Centralised Inflexible')    
    cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
                        cumulative=True, label='Centralised Flexible')
    cdf_4 = bx.hist(NPVs_model4 , 100, density=True, histtype='step',
                        cumulative=True, label='Decentralised Flexible')           
    min_ylim, max_ylim = plt.ylim()
    #plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.35, 'ENPV DQN MLP: {:.2f} Million $'.format(ENPV1/1000000))
    #plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.6, 'ENPV Improved DQN MLP: {:.2f} Million $'.format(ENPV2/1000000))
    #plt.axvline(ENPV3, color='darkgreen', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.45, 'ENPV DQN FF: {:.2f} Million $'.format(ENPV3/1000000))
    #plt.axvline(ENPV4, color='red', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.25, 'ENPV A2C MLP: {:.2f} Million $'.format(ENPV4/1000000))
    #plt.text(-20000000, max_ylim*0.85, 'ENPV A2C LSTM small: {:.2f} Million $'.format(ENPV5/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different design alternatives')
    bx.set_xlabel('NPV of model($)')
    bx.set_ylabel('Probability')    
    return cdf_1

def CDF_RL_comparison_wte_lstm(nsim, rl_model , centralised_r_npvs, centralised_flex_npvs, decentralised_flex_npvs):
    NPVs_model1 = np.array(NPVs_RL_lstm(nsim, rl_model))
    NPVs_model2 = centralised_r_npvs
    NPVs_model3 = centralised_flex_npvs
    NPVs_model4 = decentralised_flex_npvs
    # calculate ENPV for each model
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    fig, bx = plt.subplots(figsize=(8, 4)) 
    
    cdf_1 = bx.hist( NPVs_model1 , 100,  density=True, histtype='step',
                        cumulative=True, label='DRL Based Design')
    cdf_2 = bx.hist(NPVs_model2 , 100, density=True, histtype='step',
                        cumulative=True, label='Centralised Inflexible')    
    cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
                        cumulative=True, label='Centralised Flexible')
    cdf_4 = bx.hist(NPVs_model4 , 100, density=True, histtype='step',
                        cumulative=True, label='Decentralised Flexible')           
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.35, 'ENPV DQN MLP: {:.2f} Million $'.format(ENPV1/1000000))
    plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.6, 'ENPV Improved DQN MLP: {:.2f} Million $'.format(ENPV2/1000000))
    plt.axvline(ENPV3, color='darkgreen', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.45, 'ENPV DQN FF: {:.2f} Million $'.format(ENPV3/1000000))
    plt.axvline(ENPV4, color='red', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.25, 'ENPV A2C MLP: {:.2f} Million $'.format(ENPV4/1000000))
    #plt.text(-20000000, max_ylim*0.85, 'ENPV A2C LSTM small: {:.2f} Million $'.format(ENPV5/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different design alternatives')
    bx.set_xlabel('NPV of model($)')
    bx.set_ylabel('Probability')    
    return cdf_1





# this works for 8 models, where last 4 implement recurrent policies
def CDF_RL_comparison_8models(nsim, test_env , lstm_test_env, model1, model2, model3, model4, model5_lstm, model6_lstm, model7_lstm, model8_lstm ):
    NPVs_model1 = np.array(NPVs_RL(nsim, model1, test_env))
    NPVs_model2 = np.array(NPVs_RL(nsim, model2, test_env))
    NPVs_model3 = np.array(NPVs_RL(nsim, model3, test_env))
    NPVs_model4 = np.array(NPVs_RL(nsim, model4, test_env))
    NPVs_model5 = np.array(NPVs_RL(nsim, model5_lstm, lstm_test_env))
    NPVs_model6 = np.array(NPVs_RL(nsim, model6_lstm, lstm_test_env))
    NPVs_model7 = np.array(NPVs_RL(nsim, model7_lstm, lstm_test_env))
    NPVs_model8 = np.array(NPVs_RL(nsim, model8_lstm, lstm_test_env))
    
    # calculate ENPV for each model
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    ENPV5 = np.mean(NPVs_model5)    
    ENPV6 = np.mean(NPVs_model6)    
    ENPV7 = np.mean(NPVs_model7)    
    ENPV8 = np.mean(NPVs_model8)    
    
    fig, bx = plt.subplots(figsize=(15, 10)) 
    
    cdf_1 = bx.hist( NPVs_model1 , 100,  density=True, histtype='step',
                        cumulative=True, label='DQN MLP')
    cdf_2 = bx.hist(NPVs_model2 , 100, density=True, histtype='step',
                        cumulative=True, label='Improved DQN MLP')    
    cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
                        cumulative=True, label='DQN FF')    
    cdf_4 = bx.hist(NPVs_model4 , 100, density=True, histtype='step',
                        cumulative=True, label='A2C MLP')    
    cdf_5 = bx.hist(NPVs_model5 , 100, density=True, histtype='step',
                        cumulative=True, label='A2C LSTM small')    
    cdf_6 = bx.hist(NPVs_model6 , 100, density=True, histtype='step',
                        cumulative=True, label='A2C LSTM large')   
    cdf_7 = bx.hist(NPVs_model7 , 100, density=True, histtype='step',
                        cumulative=True, label='A2C LSTM Default')   
    cdf_8 = bx.hist(NPVs_model8 , 100, density=True, histtype='step',
                        cumulative=True, label='A2C LSTM Default with Layer Norm')     
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.35, 'ENPV DQN MLP: {:.2f} Million $'.format(ENPV1/1000000))
    plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.6, 'ENPV Improved DQN MLP: {:.2f} Million $'.format(ENPV2/1000000))
    plt.axvline(ENPV3, color='darkgreen', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.45, 'ENPV DQN FF: {:.2f} Million $'.format(ENPV3/1000000))    
    plt.axvline(ENPV4, color='red', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.25, 'ENPV A2C MLP: {:.2f} Million $'.format(ENPV4/1000000))
    plt.axvline(ENPV5, color='blue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.85, 'ENPV A2C LSTM small: {:.2f} Million $'.format(ENPV5/1000000))
    plt.axvline(ENPV6, color='green', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.95, 'ENPV A2C LSTM large: {:.2f} Million $'.format(ENPV6/1000000))        
    plt.axvline(ENPV7, color='orange', linestyle='dashed', linewidth=1)
   # plt.text(-20000000, max_ylim*0.75, 'ENPV A2C LSTM Default: {:.2f} Million $'.format(ENPV7/1000000))    
    plt.axvline(ENPV8, color='orange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.15, 'ENPV A2C LSTM Defaultwith Layer Norm: {:.2f} Million $'.format(ENPV8/1000000))        
   
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different RL implementations - trained on 50k steps')
    bx.set_xlabel('NPV of model')
    bx.set_ylabel('Probability')    
    return cdf_1
    

    

def CDF_RL_comparison_5models_dqn(nsim, test_env , model1, model2, model3, model4 , model5):
    NPVs_model1 = np.array(NPVs_RL(nsim, model1, test_env))
    NPVs_model2 = np.array(NPVs_RL(nsim, model2, test_env))
    NPVs_model3 = np.array(NPVs_RL(nsim, model3, test_env))
    NPVs_model4 = np.array(NPVs_RL(nsim, model4, test_env))
    NPVs_model5 = np.array(NPVs_RL(nsim, model5, test_env))
    # calculate ENPV for each model
    ENPV1 = np.mean(NPVs_model1)
    ENPV2 = np.mean(NPVs_model2)
    ENPV3 = np.mean(NPVs_model3)
    ENPV4 = np.mean(NPVs_model4)
    ENPV5 = np.mean(NPVs_model5)
    fig, bx = plt.subplots(figsize=(15, 10)) 
    
    cdf_1 = bx.hist( NPVs_model1 , 100,  density=True, histtype='step',
                        cumulative=True, label='DQN MLP Manual increased exploration')
    cdf_2 = bx.hist(NPVs_model2 , 100, density=True, histtype='step',
                        cumulative=True, label='DQN MLP default')    
    cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
                        cumulative=True, label='DQN MLP small')
    cdf_4 = bx.hist(NPVs_model4 , 100, density=True, histtype='step',
                        cumulative=True, label='DQN MLP Optuna Trial 1')    
    cdf_5 = bx.hist(NPVs_model5 , 100, density=True, histtype='step',
                        cumulative=True, label='DQN MLP Optuna Trial 2')        
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.35, 'ENPV DQN MLP: {:.2f} Million $'.format(ENPV1/1000000))
    plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.6, 'ENPV Improved DQN MLP: {:.2f} Million $'.format(ENPV2/1000000))
    plt.axvline(ENPV3, color='darkgreen', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.45, 'ENPV DQN FF: {:.2f} Million $'.format(ENPV3/1000000))
    plt.axvline(ENPV4, color='red', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.25, 'ENPV A2C MLP: {:.2f} Million $'.format(ENPV4/1000000))
    plt.axvline(ENPV5, color='blue', linestyle='dashed', linewidth=1)
    #plt.text(-20000000, max_ylim*0.85, 'ENPV A2C LSTM small: {:.2f} Million $'.format(ENPV5/1000000))    
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of different RL implementations - trained on 50k steps')
    bx.set_xlabel('NPV of model')
    bx.set_ylabel('Probability')    
    return cdf_1
    
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
      action, _ = model.predict(obs , deterministic = True)
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





def plot_rl_prob_evolution(model_lf, env_lf, model_hf, env_hf, prob_plot_treshold, prob_plot_treshold_hf , df1):
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]

    demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])
    
    k = RL_probs_history_stoch(env_lf, model_lf, demand_df)
    k_hf = RL_probs_history_stoch(env_hf, model_hf, demand_df)
    years =[]
    for i in range(16):
        years.append(i)
    

    x = np.array(years)
    # obtain all one sector probabilities for low flex
    a0_pb_lf =[]
    s1_pb = []
    s2_pb = []
    s3_pb = []
    s4_pb = []
    s5_pb = []
    s6_pb = []
    
    
    
    a0_pb_hf = []
    a1_pb_hf = []
    a2_pb_hf = []
    a3_pb_hf = []
    a4_pb_hf = []
    a5_pb_hf = []
    a6_pb_hf = []
    a7_pb_hf = []
    a8_pb_hf = []
    a9_pb_hf = []
    a10_pb_hf = []
    a11_pb_hf = []
    a12_pb_hf = []
    a13_pb_hf = []
    a14_pb_hf = []
    a15_pb_hf = []
    a16_pb_hf = []
    a17_pb_hf = []
    a18_pb_hf = []
    
    
    
    for i in range(16):
        np.array(a0_pb_lf.append(k[i][0]))
        np.array(s1_pb.append(k[i][1]))
        np.array(s2_pb.append(k[i][2]))
        np.array(s3_pb.append(k[i][3]))
        np.array(s4_pb.append(k[i][4]))
        np.array(s5_pb.append(k[i][5]))
        np.array(s6_pb.append(k[i][6]))
        a0_pb_hf.append(k_hf[i][0])
        a1_pb_hf.append(k_hf[i][1])
        a2_pb_hf.append(k_hf[i][2])
        a3_pb_hf.append(k_hf[i][3])
        a4_pb_hf.append(k_hf[i][4])
        a5_pb_hf.append(k_hf[i][5])
        a6_pb_hf.append(k_hf[i][6])
        a7_pb_hf.append(k_hf[i][7])
        a8_pb_hf.append(k_hf[i][8])
        a9_pb_hf.append(k_hf[i][9])
        a10_pb_hf.append(k_hf[i][10])
        a11_pb_hf.append(k_hf[i][11])
        a12_pb_hf.append(k_hf[i][12])
        a13_pb_hf.append(k_hf[i][13])
        a14_pb_hf.append(k_hf[i][14])
        a15_pb_hf.append(k_hf[i][15])
        a16_pb_hf.append(k_hf[i][16])
        a17_pb_hf.append(k_hf[i][17])
        a18_pb_hf.append(k_hf[i][18])
        
        
        
    
    a = plt.figure(figsize=(12,5))
    plt.xlabel('Years')
    plt.ylabel('Action Probabilitity')
    plt.xticks([i for i in range (16)] )
    plt.title('Action Probability Distribution Over Time for 1 Stochastic Demand Scenario')
    #plot only actions with probability above treshold % at some point
    
    
    if  any(i > prob_plot_treshold for i in a0_pb_lf):
        plt.plot(x, a0_pb_lf, 'o:', label = 'A0 LF' )
    if  any(i > prob_plot_treshold for i in s1_pb):    
        plt.plot(x,s1_pb, 'o:', label = 'A1 LF')
    if  any(i > prob_plot_treshold for i in s2_pb):    
        plt.plot(x,s3_pb, 'o:', label = 'A1 LF')
    if  any(i > prob_plot_treshold for i in s3_pb): 
        plt.plot(x,s2_pb, 'o:', label = 'A3 LF')
    if  any(i > prob_plot_treshold for i in s4_pb): 
        plt.plot(x,s4_pb, 'o:', label = 'A4 LF')
    if  any(i > prob_plot_treshold for i in s5_pb): 
        plt.plot(x,s5_pb, 'o:', label = 'A5 LF')
    if  any(i > prob_plot_treshold for i in s6_pb): 
        plt.plot(x,s6_pb, 'o:', label = 'A6 LF')
    
    
    if  any(i > prob_plot_treshold_hf for i in a0_pb_hf):
        plt.plot(x, a0_pb_hf , 's-' ,  label = 'A0 HF')
    if  any(i > prob_plot_treshold_hf for i in a1_pb_hf):
        plt.plot(x, a1_pb_hf , 's-' ,  label = 'A1 HF')
    if  any(i > prob_plot_treshold_hf for i in a2_pb_hf):
        plt.plot(x, a2_pb_hf , 's-' , label = 'A2 HF')
    if  any(i > prob_plot_treshold_hf for i in a3_pb_hf):
        plt.plot(x, a3_pb_hf , 's-' ,  label = 'A3 HF')
    if  any(i > prob_plot_treshold_hf for i in a4_pb_hf):
        plt.plot(x, a4_pb_hf , 's-' ,  label = 'A4 HF')
    if  any(i > prob_plot_treshold_hf for i in a5_pb_hf):
        plt.plot(x, a5_pb_hf , 's-' ,  label = 'A5 HF')
    if  any(i > prob_plot_treshold_hf for i in a6_pb_hf):
        plt.plot(x, a6_pb_hf , 's-' ,  label = 'A6 HF')
    if  any(i > prob_plot_treshold_hf for i in a7_pb_hf):
        plt.plot(x, a7_pb_hf , 's-' ,  label = 'A7 HF')
    if  any(i > prob_plot_treshold_hf for i in a8_pb_hf):
        plt.plot(x, a8_pb_hf , 's-' ,  label = 'A8 HF')
    if  any(i > prob_plot_treshold_hf for i in a9_pb_hf):
        plt.plot(x, a9_pb_hf , 's-' ,  label = 'A9 HF')
    if  any(i > prob_plot_treshold_hf for i in a10_pb_hf):
        plt.plot(x, a10_pb_hf , 's-' ,  label = 'A10 HF')    
    if  any(i > prob_plot_treshold_hf for i in a11_pb_hf):
        plt.plot(x, a11_pb_hf , 's-' ,  label = 'A11 HF')   
    if  any(i > prob_plot_treshold_hf for i in a12_pb_hf):
        plt.plot(x, a12_pb_hf , 's-' ,  label = 'A12 HF')    
    if  any(i > prob_plot_treshold_hf for i in a13_pb_hf):
        plt.plot(x, a13_pb_hf , 's-' ,  label = 'A13 HF')    
    if  any(i > prob_plot_treshold_hf for i in a14_pb_hf):
        plt.plot(x, a14_pb_hf , 's-' ,  label = 'A14 HF')
    if  any(i > prob_plot_treshold_hf for i in a15_pb_hf):
        plt.plot(x, a18_pb_hf , 's-' ,  label = 'A15 HF')
    if  any(i > prob_plot_treshold_hf for i in a16_pb_hf):
        plt.plot(x, a16_pb_hf , 's-' ,  label = 'A16 HF')
    if  any(i > prob_plot_treshold_hf for i in a17_pb_hf):
        plt.plot(x, a17_pb_hf , 's-' ,  label = 'A17 HF')
    if  any(i > prob_plot_treshold_hf for i in a18_pb_hf):
        plt.plot(x, a15_pb_hf , 's-' ,  label = 'A18 HF')
    plt.legend(loc='lower center' , ncol = 6, bbox_to_anchor = (.5 , -.25))
    plt.show()
  
    return a , k, k_hf




def plot_rl_lowflex_prob_evolution(model_lf, env_lf, prob_plot_treshold, prob_plot_treshold_hf , df1):
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]

    demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])
    
    k = RL_probs_history_stoch(env_lf, model_lf, demand_df)
    years =[]
    for i in range(16):
        years.append(i)
    

    x = np.array(years)
    # obtain all one sector probabilities for low flex
    a0_pb_lf =[]
    s1_pb = []
    s2_pb = []
    s3_pb = []
    s4_pb = []
    s5_pb = []
    s6_pb = []
    
    
    
    a0_pb_hf = []
    a1_pb_hf = []
    a2_pb_hf = []
    a3_pb_hf = []
    a4_pb_hf = []
    a5_pb_hf = []
    a6_pb_hf = []
    a7_pb_hf = []
    a8_pb_hf = []
    a9_pb_hf = []
    a10_pb_hf = []
    a11_pb_hf = []
    a12_pb_hf = []
    a13_pb_hf = []
    a14_pb_hf = []
    a15_pb_hf = []
    a16_pb_hf = []
    a17_pb_hf = []
    a18_pb_hf = []
    
    
    
    for i in range(16):
        np.array(a0_pb_lf.append(k[i][0]))
        np.array(s1_pb.append(k[i][1]))
        np.array(s2_pb.append(k[i][2]))
        np.array(s3_pb.append(k[i][3]))
        np.array(s4_pb.append(k[i][4]))
        np.array(s5_pb.append(k[i][5]))
        np.array(s6_pb.append(k[i][6]))
        
        
        
    
    a = plt.figure(figsize=(12,5))
    plt.xlabel('Years')
    plt.ylabel('Action Probabilitity')
    plt.xticks([i for i in range (16)] )
    plt.title('Action Probability Distribution Over Time for 1 Stochastic Demand Scenario')
    #plot only actions with probability above treshold % at some point
    
    
    if  any(i > prob_plot_treshold for i in a0_pb_lf):
        plt.plot(x, a0_pb_lf, 'o:', label = 'A0 LF' )
    #if  any(i > prob_plot_treshold for i in s1_pb):    
        #plt.plot(x,s1_pb, 'o:', label = 'A1 LF')
    if  any(i > prob_plot_treshold for i in s2_pb):    
        plt.plot(x,s3_pb, 'o:', label = 'A1 LF')
    if  any(i > prob_plot_treshold for i in s3_pb): 
        plt.plot(x,s2_pb, 'o:', label = 'A3 LF')
    if  any(i > prob_plot_treshold for i in s4_pb): 
        plt.plot(x,s4_pb, 'o:', label = 'A4 LF')
    if  any(i > prob_plot_treshold for i in s5_pb): 
        plt.plot(x,s5_pb, 'o:', label = 'A5 LF')
    if  any(i > prob_plot_treshold for i in s6_pb): 
        plt.plot(x,s6_pb, 'o:', label = 'A6 LF')
    plt.legend(loc='lower center' , ncol = 6, bbox_to_anchor = (.5 , -.25))
    plt.show()
  
    return a 

def plot_rl_cap_evolution(model_lf, env_lf, model_hf, env_hf,  df1):
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]

    demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])
    
    
    rl_cap_history_bysec_highflex = RL_capacities_history_bysector_adj(env_hf, model_hf, demand_df) 
    #print(rl_cap_history_bysec_highflex)
    rl_cap_history_bysec_lowflex = RL_capacities_history_bysector_adj(env_lf, model_lf, demand_df) 
    #print(rl_cap_history_bysec_lowflex)
    #retrive capacities for each sector resulting from RL interaction
    years =[]
    for i in range(16):
        years.append(i)
    x = np.array(years)
    cap_s1_lf = rl_cap_history_bysec_lowflex[0]
    cap_s2_lf = rl_cap_history_bysec_lowflex[1]
    cap_s3_lf = rl_cap_history_bysec_lowflex[2]
    cap_s4_lf = rl_cap_history_bysec_lowflex[3]
    cap_s5_lf = rl_cap_history_bysec_lowflex[4]
    cap_s6_lf = rl_cap_history_bysec_lowflex[5]
    
    cap_s1_hf = rl_cap_history_bysec_highflex[0]
    cap_s2_hf = rl_cap_history_bysec_highflex[1]
    cap_s3_hf = rl_cap_history_bysec_highflex[2]
    cap_s4_hf = rl_cap_history_bysec_highflex[3]
    cap_s5_hf = rl_cap_history_bysec_highflex[4]
    cap_s6_hf = rl_cap_history_bysec_highflex[5]
    
    
    
    
    mark_f_h = .05
    
    marker = itertools.cycle((',', '+', '.', 'o', '*' , 's', 'p' ,'P' ,'h' , 'x' , 'X' , 'D' , '3' , '4' , '8' ,'v' , '^' ))    
    # plot only sectors where expansion decision made at some point
    a = plt.figure(figsize=(17,7))
    plt.xlabel('Years', fontsize = 16)
    plt.ylabel('Demand / Capacity (tpd)', fontsize = 16)
    if  any(i > 0 for i in cap_s1_lf):
        caps1_lf = plt.plot(x,cap_s1_lf, '-.' , marker = next(marker) , markevery = mark_f_h ,  label = 'DRL LF S1' )
    if  any(i > 0 for i in cap_s2_lf):
        caps2_lf = plt.plot(x ,cap_s2_lf, '-.', marker = next(marker),  label = 'DRL LF S2')
    if  any(i > 0 for i in cap_s3_lf):
        caps3_lf = plt.plot(x,cap_s3_lf, '-.' , marker = next(marker) , markevery = mark_f_h,label = 'DRL LF S3')
    if  any(i > 0 for i in cap_s4_lf):
        caps4_lf = plt.plot(x,cap_s4_lf, '-.' , marker = next(marker) , label = 'DRL LF S4')
    if  any(i > 0 for i in cap_s5_lf):
        caps5_lf = plt.plot(x,cap_s5_lf, '-.' , marker = next(marker) , label = 'DRL LF S5')
    if  any(i > 0 for i in cap_s6_lf):
        caps6_lf = plt.plot(x,cap_s6_lf, '-.' , marker = next(marker) , markevery = mark_f_h,label = 'DRL LF S6')
    

    if  any(i > 0 for i in cap_s1_hf):
        caps1_hf = plt.plot(x,cap_s1_hf,  '--', marker = next(marker) ,markevery = mark_f_h ,label = 'DRL HF S1')
    if  any(i > 0 for i in cap_s2_hf):                        
        caps2_hf = plt.plot(x,cap_s2_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S2')
    if  any(i > 0 for i in cap_s3_hf):
        caps3_hf = plt.plot(x,cap_s3_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S3')
    if  any(i > 0 for i in cap_s4_hf):
        caps4_hf = plt.plot(x,cap_s4_hf,  '--', marker = next(marker), markevery = mark_f_h ,label = 'DRL HF S4')
    if  any(i > 0 for i in cap_s5_hf):
        caps5_hf = plt.plot(x,cap_s5_hf,  '--', marker = next(marker) , markevery = mark_f_h,label = 'DRL HF S5')
    if  any(i > 0 for i in cap_s6_hf):
        caps6_hf = plt.plot(x,cap_s6_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S6')
    
    caps1_dr = plt.plot(x,cap_s1_dr, '1:', label = 'Flex DR S1',  markevery = mark_f_h,  alpha = .9)
    #caps2_dr = plt.plot(x,cap_s2_dr, label = 'Flex DR S1')
    #caps3_dr = plt.plot(x,cap_s3_dr, label = 'Flex DR S1')
    #caps4_dr = plt.plot(x,cap_s4_dr, label = 'Flex DR S1')
    #caps5_dr = plt.plot(x,cap_s5_dr, label = 'Flex DR S1')
    caps6_dr = plt.plot(x,cap_s6_dr, '2:', label = 'Flex DR S6', markevery = mark_f_h , alpha = .9)
    
    
    
    
    
    dems1 = plt.plot(x, s_1_demand_df, marker = next(marker) , markevery = mark_f_h , label = 'S1 FW'  )
    dems2 = plt.plot(x, s_2_demand_df, marker = next(marker) , markevery = mark_f_h , label = 'S2 FW' )
    dems3 = plt.plot(x, s_3_demand_df, marker = next(marker) , markevery = mark_f_h ,label = 'S3 FW' )
    dems4 = plt.plot(x, s_4_demand_df, marker = next(marker), markevery = mark_f_h ,label = 'S4 FW' )
    dems5 = plt.plot(x, s_5_demand_df, marker = next(marker) ,markevery = mark_f_h , label = 'S5 FW' )
    dems6 = plt.plot(x, s_6_demand_df, marker = next(marker), markevery = mark_f_h ,label = 'S6 FW' )
    #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
    plt.legend(loc='lower center' , ncol = 4 ,bbox_to_anchor = (.5 , -.35))
    plt.title('Demand and capacity evolution over project lifetime' , fontsize= 24)
    plt.xlabel('Years')
    #plt.setxlim(15)
    plt.xticks([i for i in range (16)] )
    plt.show()
    return a , rl_cap_history_bysec_lowflex  , rl_cap_history_bysec_highflex




def plot_rl_cap_evolution_markers_subplot(model_lf, env_lf, model_hf, env_hf,  df1):
    from matplotlib.lines import Line2D
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]

    demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])
    
    
    rl_cap_history_bysec_highflex = RL_capacities_history_bysector_adj(env_hf, model_hf, demand_df) 
    #print(rl_cap_history_bysec_highflex)
    rl_cap_history_bysec_lowflex = RL_capacities_history_bysector_adj(env_lf, model_lf, demand_df) 
    #print(rl_cap_history_bysec_lowflex)
    #retrive capacities for each sector resulting from RL interaction
    years =[]
    for i in range(16):
        years.append(i)
    x = np.array(years)
    cap_s1_lf = rl_cap_history_bysec_lowflex[0]
    cap_s2_lf = rl_cap_history_bysec_lowflex[1]
    cap_s3_lf = rl_cap_history_bysec_lowflex[2]
    cap_s4_lf = rl_cap_history_bysec_lowflex[3]
    cap_s5_lf = rl_cap_history_bysec_lowflex[4]
    cap_s6_lf = rl_cap_history_bysec_lowflex[5]
    
    cap_s1_hf = rl_cap_history_bysec_highflex[0]
    cap_s2_hf = rl_cap_history_bysec_highflex[1]
    cap_s3_hf = rl_cap_history_bysec_highflex[2]
    cap_s4_hf = rl_cap_history_bysec_highflex[3]
    cap_s5_hf = rl_cap_history_bysec_highflex[4]
    cap_s6_hf = rl_cap_history_bysec_highflex[5]
    
    
    
    
    mark_f_h = .05
    mark_loc_lf = [.5, 3.5 , 5.5 , 7.5 , 10.5 , 14.5 , 17.5]
    
    marker = itertools.cycle((',', '+', '.', 'o', '*' , 's', 'p' ,'P' ,'h' , 'x' , 'X' , 'D' , '3' , '4' , '8' ,'v' , '^' ))    
    # plot only sectors where expansion decision made at some point
    plt.subplot(3 ,1, 2)
    #plt.subplot(3 ,1, 2).set_title("Low Flex and Dr")
    #a = plt.figure(figsize=(17,7))
    #plt.xlabel('Years', fontsize = 16)
    plt.ylabel('Demand / Capacity (tpd)', fontsize = 12)
    if  any(i > 0 for i in cap_s1_lf):
        caps1_lf = plt.plot(x,cap_s1_lf, '-.' , marker = next(marker) , markevery = mark_f_h ,  label = 'DRL LF S1' )
    if  any(i > 0 for i in cap_s2_lf):
        caps2_lf = plt.plot(x ,cap_s2_lf, '-.', marker = next(marker),  label = 'DRL LF S2')
    if  any(i > 0 for i in cap_s3_lf):
        caps3_lf = plt.plot(x,cap_s3_lf, '-.' , marker = next(marker) , markevery = mark_f_h,label = 'DRL LF S3')
    if  any(i > 0 for i in cap_s4_lf):
        caps4_lf = plt.plot(x,cap_s4_lf, '-.' , marker = next(marker) , label = 'DRL LF S4')
    if  any(i > 0 for i in cap_s5_lf):
        caps5_lf = plt.plot(x,cap_s5_lf, '-.' , marker = next(marker) , label = 'DRL LF S5')
    if  any(i > 0 for i in cap_s6_lf):
        caps6_lf = plt.plot(x,cap_s6_lf, '-.' , marker = next(marker) , markevery = mark_f_h,label = 'DRL LF S6')
    #plt.subplot(3 ,1, 1).set_title("Demand")
    caps1_dr = plt.plot(x,cap_s1_dr, '1:', label = 'Flex DR S1',  markevery = mark_f_h,  alpha = .9)
    #caps2_dr = plt.plot(x,cap_s2_dr, label = 'Flex DR S1')
    #caps3_dr = plt.plot(x,cap_s3_dr, label = 'Flex DR S1')
    #caps4_dr = plt.plot(x,cap_s4_dr, label = 'Flex DR S1')
    #caps5_dr = plt.plot(x,cap_s5_dr, label = 'Flex DR S1')
    caps6_dr = plt.plot(x,cap_s6_dr, '2:', label = 'Flex DR S6', markevery = mark_f_h , alpha = .9)
    
    
    #plt.xticks([i for i in range (16)] ) 
    #plt.axis('off')
    plt.xticks([])
    plt.legend(loc='upper left' , ncol = 2 ,bbox_to_anchor = (1.05 , 1))
    plt.subplot(3,1,3)
    #plt.subplot(3 ,1, 3).set_title("High Flex")
    if  any(i > 0 for i in cap_s1_hf):
        caps1_hf = plt.plot(x,cap_s1_hf,  '--', marker = next(marker) ,markevery = mark_f_h ,label = 'DRL HF S1')
    if  any(i > 0 for i in cap_s2_hf):                        
        caps2_hf = plt.plot(x,cap_s2_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S2')
    if  any(i > 0 for i in cap_s3_hf):
        caps3_hf = plt.plot(x,cap_s3_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S3')
    if  any(i > 0 for i in cap_s4_hf):
        caps4_hf = plt.plot(x,cap_s4_hf,  '--', marker = next(marker), markevery = mark_f_h ,label = 'DRL HF S4')
    if  any(i > 0 for i in cap_s5_hf):
        caps5_hf = plt.plot(x,cap_s5_hf,  '--', marker = next(marker) , markevery = mark_f_h,label = 'DRL HF S5')
    if  any(i > 0 for i in cap_s6_hf):
        caps6_hf = plt.plot(x,cap_s6_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S6')
    plt.legend(loc='upper left' , ncol = 2 ,bbox_to_anchor = (1.05 , 1))
    plt.xticks([i for i in range (16)] )
    plt.xlabel('Years')
    
    plt.subplot(3,1,1)

    
    dems1 = plt.plot(x, s_1_demand_df, marker = next(marker) , markevery = mark_f_h , label = 'S1 FW'  )
    dems2 = plt.plot(x, s_2_demand_df, marker = next(marker) , markevery = mark_f_h , label = 'S2 FW' )
    dems3 = plt.plot(x, s_3_demand_df, marker = next(marker) , markevery = mark_f_h ,label = 'S3 FW' )
    dems4 = plt.plot(x, s_4_demand_df, marker = next(marker), markevery = mark_f_h ,label = 'S4 FW' )
    dems5 = plt.plot(x, s_5_demand_df, marker = next(marker) ,markevery = mark_f_h , label = 'S5 FW' )
    dems6 = plt.plot(x, s_6_demand_df, marker = next(marker), markevery = mark_f_h ,label = 'S6 FW' )
    plt.xticks([])
    #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
    plt.legend(loc='upper left' , ncol = 2 ,bbox_to_anchor = (1.05 , 1))
    plt.title('Demand and capacity evolution over project lifetime' , fontsize= 12)
    #plt.xlabel('Years')
    #plt.tight_layout()
    #plt.setxlim(15)
    #plt.xticks([i for i in range (16)] )
    plt.show()
    
    #plt.subplots_adjust(hspace = 3)
    return rl_cap_history_bysec_lowflex  , rl_cap_history_bysec_highflex


def plot_rl_cap_evolution_markers_subplot_ax(model_lf, env_lf, model_hf, env_hf,  df1):
    from matplotlib.lines import Line2D
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]

    demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])
    
    
    rl_cap_history_bysec_highflex = RL_capacities_history_bysector_adj(env_hf, model_hf, demand_df) 
    #print(rl_cap_history_bysec_highflex)
    rl_cap_history_bysec_lowflex = RL_capacities_history_bysector_adj(env_lf, model_lf, demand_df) 
    #print(rl_cap_history_bysec_lowflex)
    #retrive capacities for each sector resulting from RL interaction
    years =[]
    for i in range(16):
        years.append(i)
    x = np.array(years)
    cap_s1_lf = rl_cap_history_bysec_lowflex[0]
    cap_s2_lf = rl_cap_history_bysec_lowflex[1]
    cap_s3_lf = rl_cap_history_bysec_lowflex[2]
    cap_s4_lf = rl_cap_history_bysec_lowflex[3]
    cap_s5_lf = rl_cap_history_bysec_lowflex[4]
    cap_s6_lf = rl_cap_history_bysec_lowflex[5]
    
    cap_s1_hf = rl_cap_history_bysec_highflex[0]
    cap_s2_hf = rl_cap_history_bysec_highflex[1]
    cap_s3_hf = rl_cap_history_bysec_highflex[2]
    cap_s4_hf = rl_cap_history_bysec_highflex[3]
    cap_s5_hf = rl_cap_history_bysec_highflex[4]
    cap_s6_hf = rl_cap_history_bysec_highflex[5]
    
    
    
    
    mark_f_h = .05
    mark_loc_lf = [.5, 3.5 , 5.5 , 7.5 , 10.5 , 14.5 , 17.5]
    
    marker = itertools.cycle((',', '+', '.', 'o', '*' , 's', 'p' ,'P' ,'h' , 'x' , 'X' , 'D' , '3' , '4' , '8' ,'v' , '^' ))    
    # plot only sectors where expansion decision made at some point
    fig , axs = plt.subplots(3 ,figsize=(10, 7))
    #a = plt.figure(figsize=(17,7))
    plt.xlabel('Years', fontsize = 16)
    #plt.ylabel('Demand / Capacity (tpd)', fontsize = 16)
    if  any(i > 0 for i in cap_s1_lf):
        axs[1].plot(x,cap_s1_lf, '-.' , marker = next(marker) , markevery = mark_f_h ,  label = 'DRL LF S1' )
    if  any(i > 0 for i in cap_s2_lf):
        axs[1].plot(x ,cap_s2_lf, '-.', marker = next(marker),  label = 'DRL LF S2')
    if  any(i > 0 for i in cap_s3_lf):
        axs[1].plot(x,cap_s3_lf, '-.' , marker = next(marker) , markevery = mark_f_h,label = 'DRL LF S3')
    if  any(i > 0 for i in cap_s4_lf):
        axs[1].plot(x,cap_s4_lf, '-.' , marker = next(marker) , label = 'DRL LF S4')
    if  any(i > 0 for i in cap_s5_lf):
        axs[1].plot(x,cap_s5_lf, '-.' , marker = next(marker) , label = 'DRL LF S5')
    if  any(i > 0 for i in cap_s6_lf):
        axs[1].plot(x,cap_s6_lf, '-.' , marker = next(marker) , markevery = mark_f_h,label = 'DRL LF S6')
    #plt.xticks([i for i in range (16)] )   
    
    #plt.subplot(3,1,3)
    if  any(i > 0 for i in cap_s1_hf):
        axs[2].plot(x,cap_s1_hf,  '--', marker = next(marker) ,markevery = mark_f_h ,label = 'DRL HF S1')
    if  any(i > 0 for i in cap_s2_hf):                        
        axs[2].plot(x,cap_s2_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S2')
    if  any(i > 0 for i in cap_s3_hf):
        axs[2].plot(x,cap_s3_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S3')
    if  any(i > 0 for i in cap_s4_hf):
        axs[2].plot(x,cap_s4_hf,  '--', marker = next(marker), markevery = mark_f_h ,label = 'DRL HF S4')
    if  any(i > 0 for i in cap_s5_hf):
        axs[2].plot(x,cap_s5_hf,  '--', marker = next(marker) , markevery = mark_f_h,label = 'DRL HF S5')
    if  any(i > 0 for i in cap_s6_hf):
        axs[2].plot(x,cap_s6_hf,  '--', marker = next(marker) , markevery = mark_f_h ,label = 'DRL HF S6')
    axs[2].legend(loc='upper left' , ncol = 2 ,bbox_to_anchor = (1.05 , 1))
    #plt.xticks([i for i in range (16)] )
    
    #plt.subplot(3,1,1)
    caps1_dr = axs[1].plot(x,cap_s1_dr, '1:', label = 'Flex DR S1',  markevery = mark_f_h,  alpha = .9)
    #caps2_dr = plt.plot(x,cap_s2_dr, label = 'Flex DR S1')
    #caps3_dr = plt.plot(x,cap_s3_dr, label = 'Flex DR S1')
    #caps4_dr = plt.plot(x,cap_s4_dr, label = 'Flex DR S1')
    #caps5_dr = plt.plot(x,cap_s5_dr, label = 'Flex DR S1')
    caps6_dr = axs[1].plot(x,cap_s6_dr, '2:', label = 'Flex DR S6', markevery = mark_f_h , alpha = .9)
    axs[1].legend(loc='upper left' , ncol = 2 ,bbox_to_anchor = (1.05 , 1))
    
    
    
    
    dems1 = axs[0].plot(x, s_1_demand_df, marker = next(marker) , markevery = mark_f_h , label = 'S1 FW'  )
    dems2 = axs[0].plot(x, s_2_demand_df, marker = next(marker) , markevery = mark_f_h , label = 'S2 FW' )
    dems3 = axs[0].plot(x, s_3_demand_df, marker = next(marker) , markevery = mark_f_h ,label = 'S3 FW' )
    dems4 = axs[0].plot(x, s_4_demand_df, marker = next(marker), markevery = mark_f_h ,label = 'S4 FW' )
    dems5 = axs[0].plot(x, s_5_demand_df, marker = next(marker) ,markevery = mark_f_h , label = 'S5 FW' )
    dems6 = axs[0].plot(x, s_6_demand_df, marker = next(marker), markevery = mark_f_h ,label = 'S6 FW' )
    #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
    axs[0].legend(loc='upper left' , ncol = 2 ,bbox_to_anchor = (1.05 , 1))
    #fig.suptitle('Demand and capacity evolution over project lifetime' , fontsize= 16)
    
    axs[0].set_title("Stochastic Demand Evolution Scenario")
    axs[1].set_title("Flex DR and Low Flex DRL Designs Capacity Evolution")
    axs[2].set_title("High Flex DRL Design Capacity Evolution")
    
    axs[0].set_ylabel("Tpd")
    axs[1].set_ylabel("Tpd")
    axs[2].set_ylabel("Tpd")
    plt.subplots_adjust(hspace = .6)
    
    plt.xlabel('Years')
    #plt.tight_layout()
    #plt.setxlim(15)
    plt.xticks([i for i in range (16)] )
    plt.show()
    return rl_cap_history_bysec_lowflex  , rl_cap_history_bysec_highflex



def plot_rl_cap_evolution_markers_subplot_ax_presetmarker(model_lf, env_lf, model_hf, env_hf,  df1):
    from matplotlib.lines import Line2D
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]

    demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])
    
    
    rl_cap_history_bysec_highflex = RL_capacities_history_bysector_adj(env_hf, model_hf, demand_df) 
    #print(rl_cap_history_bysec_highflex)
    rl_cap_history_bysec_lowflex = RL_capacities_history_bysector_adj(env_lf, model_lf, demand_df) 
    #print(rl_cap_history_bysec_lowflex)
    #retrive capacities for each sector resulting from RL interaction
    years =[]
    for i in range(16):
        years.append(i)
    x = np.array(years)
    cap_s1_lf = rl_cap_history_bysec_lowflex[0]
    cap_s2_lf = rl_cap_history_bysec_lowflex[1]
    cap_s3_lf = rl_cap_history_bysec_lowflex[2]
    cap_s4_lf = rl_cap_history_bysec_lowflex[3]
    cap_s5_lf = rl_cap_history_bysec_lowflex[4]
    cap_s6_lf = rl_cap_history_bysec_lowflex[5]
    
    cap_s1_hf = rl_cap_history_bysec_highflex[0]
    cap_s2_hf = rl_cap_history_bysec_highflex[1]
    cap_s3_hf = rl_cap_history_bysec_highflex[2]
    cap_s4_hf = rl_cap_history_bysec_highflex[3]
    cap_s5_hf = rl_cap_history_bysec_highflex[4]
    cap_s6_hf = rl_cap_history_bysec_highflex[5]
    
    
    
    
    mark_f_h = .05
    mark_loc_lf = [.5, 3.5 , 5.5 , 7.5 , 10.5 , 14.5 , 17.5]
    
    marker = itertools.cycle((',', '+', '.', 'o', '*' , 's', 'p' ,'P' ,'h' , 'x' , 'X' , 'D' , '3' , '4' , '8' ,'v' , '^' ))    
    # plot only sectors where expansion decision made at some point
    fig , axs = plt.subplots(3 ,figsize=(10, 13))
    #a = plt.figure(figsize=(17,7))

    #plt.ylabel('Demand / Capacity (tpd)', fontsize = 16)
    if  any(i > 0 for i in cap_s1_lf):
        axs[1].plot(x,cap_s1_lf, '--D' , markevery = mark_f_h ,  label = 'DRL LF S1' )
    if  any(i > 0 for i in cap_s2_lf):
        axs[1].plot(x ,cap_s2_lf, '-.', marker = next(marker),  label = 'DRL LF S3')
    if  any(i > 0 for i in cap_s3_lf):
        axs[1].plot(x,cap_s3_lf, '-.+'  , markevery = mark_f_h,label = 'DRL LF S2')
    if  any(i > 0 for i in cap_s4_lf):
        axs[1].plot(x,cap_s4_lf, '-.' , marker = next(marker) , label = 'DRL LF S4')
    if  any(i > 0 for i in cap_s5_lf):
        axs[1].plot(x,cap_s5_lf, '-.' , marker = next(marker) , label = 'DRL LF S5')
    if  any(i > 0 for i in cap_s6_lf):
        axs[1].plot(x,cap_s6_lf, '-' , marker = 'o' , markevery = mark_f_h,label = 'DRL LF S6')
    axs[1].set_xticks([i for i in range (16)] )   
    
    #plt.subplot(3,1,3)
    if  any(i > 0 for i in cap_s1_hf):
        axs[2].plot(x,cap_s1_hf,  '--D' ,label = 'DRL HF S1')
    if  any(i > 0 for i in cap_s2_hf):                        
        axs[2].plot(x,cap_s2_hf,  '-.+',  markevery = mark_f_h ,label = 'DRL HF S3')
    if  any(i > 0 for i in cap_s3_hf):
        axs[2].plot(x,cap_s3_hf,  ':x' ,  markevery = mark_f_h ,label = 'DRL HF S6')
    if  any(i > 0 for i in cap_s4_hf):
        axs[2].plot(x,cap_s4_hf,  'o',  markevery = mark_f_h ,label = 'DRL HF S4')
    if  any(i > 0 for i in cap_s5_hf):
        axs[2].plot(x,cap_s5_hf, '-' ,   marker = '*' , markevery = mark_f_h,label = 'DRL HF S5')
    if  any(i > 0 for i in cap_s6_hf):
        axs[2].plot(x,cap_s6_hf, '-' , marker = 's' , markevery = mark_f_h ,label = 'DRL HF S2')
    axs[2].legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.35))
    axs[2].set_xticks([i for i in range (16)] )
    axs[0].set_xticks([i for i in range (16)] )
    #plt.subplot(3,1,1)
    caps1_dr = axs[1].plot(x,cap_s1_dr, ':x', label = 'Flex DR S1',  markevery = mark_f_h,  alpha = .9)
    #caps2_dr = plt.plot(x,cap_s2_dr, label = 'Flex DR S1')
    #caps3_dr = plt.plot(x,cap_s3_dr, label = 'Flex DR S1')
    #caps4_dr = plt.plot(x,cap_s4_dr, label = 'Flex DR S1')
    #caps5_dr = plt.plot(x,cap_s5_dr, label = 'Flex DR S1')
    caps6_dr = axs[1].plot(x,cap_s6_dr, marker = 's', label = 'Flex DR S6', markevery = mark_f_h , alpha = .9)
    axs[1].legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.3))
    
    
    
    
    dems1 = axs[0].plot(x, s_1_demand_df, '--D' ,  linewidth = .8 , markevery = mark_f_h , label = 'S1 FW'  )
    dems2 = axs[0].plot(x, s_2_demand_df, '-.+'  , markevery = mark_f_h , label = 'S2 FW' )
    dems3 = axs[0].plot(x, s_3_demand_df, ':x'  , markevery = mark_f_h ,label = 'S3 FW' )
    dems4 = axs[0].plot(x, s_4_demand_df, marker = 'o', markevery = mark_f_h ,label = 'S4 FW' )
    dems5 = axs[0].plot(x, s_5_demand_df, marker = '*' ,markevery = mark_f_h , label = 'S5 FW' )
    dems6 = axs[0].plot(x, s_6_demand_df, marker = 's', markevery = mark_f_h ,label = 'S6 FW' )
    #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
    axs[0].legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.3))
    #fig.suptitle('Demand and capacity evolution over project lifetime' , fontsize= 16)
    
    axs[0].set_title("Stochastic Demand Evolution Scenario" , fontsize = 14)
    axs[1].set_title("Flex DR and Low Flex DRL Designs Capacity Evolution" , fontsize = 14)
    axs[2].set_title("High Flex DRL Design Capacity Evolution" , fontsize = 14)
    
    axs[0].set_ylabel('Tonnes per day' , fontsize = 13)
    axs[1].set_ylabel('Tonnes per day' , fontsize = 13)
    axs[2].set_ylabel('Tonnes per day' , fontsize = 13)
    plt.subplots_adjust( hspace = .4)
    
    plt.xlabel('Years', fontsize = 13)
    #plt.ylabel('Tonnes per day' , fontsize = 16)
    #plt.tight_layout()
    #plt.setxlim(15)
    plt.xticks([i for i in range (16)] )
    plt.show()
    return rl_cap_history_bysec_lowflex  , rl_cap_history_bysec_highflex


def plot_cap_ev_horizontal_subplots(df1):
    s_1_demand_df = df1["Demand1" ]
    s_2_demand_df = df1["Demand2" ]
    s_3_demand_df = df1["Demand3" ]
    s_4_demand_df = df1["Demand4" ]
    s_5_demand_df = df1["Demand5" ]
    s_6_demand_df = df1["Demand6" ]
    
    cap_s1_dr = df1["Capacity1" ]
    cap_s2_dr = df1["Capacity2" ]
    cap_s3_dr = df1["Capacity3" ]
    cap_s4_dr = df1["Capacity4" ]
    cap_s5_dr = df1["Capacity5" ]
    cap_s6_dr = df1["Capacity6" ]
    
    cap_s1_hf = df1["Cap1HF"]
    cap_s2_hf = df1["Cap2HF"]
    cap_s3_hf = df1["Cap3HF"]
    cap_s4_hf = df1["Cap4HF"]
    cap_s5_hf = df1["Cap5HF"]
    cap_s6_hf = df1["Cap6HF"]
    
    
    cap_s1_lf = df1["Cap1LF"]
    cap_s2_lf = df1["Cap2LF"]
    cap_s3_lf = df1["Cap3LF"]
    cap_s4_lf = df1["Cap4LF"]
    cap_s5_lf = df1["Cap5LF"]
    cap_s6_lf = df1["Cap6LF"]
    years =[]
    for i in range(16):
        years.append(i)
    x = np.array(years)
    fig , (ax1, ax2, ax3) = plt.subplots(1,3 , sharex = True, sharey = True, figsize=(15, 5))
#a= plt.figure(figsize=(12,5))
    mark_f_h = 1
    marker = itertools.cycle((',', '+', '.', 'o', '*' , 's', 'p' ,'P' ,'h' , 'x' , 'X' , 'D' , '3' , '4' , '8' ,'v' , '^' )) 
    
    #plt.ylabel('Demand / Capacity (tpd)', fontsize = 16)
    if  any(i > 0 for i in cap_s1_lf):
        ax2.plot(x,cap_s1_lf, ':xb' , markevery = mark_f_h ,  label = 'S1' )
    if  any(i > 0 for i in cap_s2_lf):
        ax2.plot(x ,cap_s2_lf, '-.+',  label = 'S2')
    if  any(i > 0 for i in cap_s3_lf):
        ax2.plot(x,cap_s3_lf, '-.oy'  , markevery = mark_f_h,label = 'S3')
    if  any(i > 0 for i in cap_s4_lf):
        ax2.plot(x,cap_s4_lf, '-.' , marker = next(marker) , label = 'S4')
    if  any(i > 0 for i in cap_s5_lf):
        ax2.plot(x,cap_s5_lf, '-.' , marker = next(marker) , label = 'S5')
    if  any(i > 0 for i in cap_s6_lf):
        ax2.plot(x,cap_s6_lf, '-g' , marker = 's' , linewidth = 3, alpha = .5 , markevery = mark_f_h,label = 'S6')
    ax2.set_xticks([i for i in range (16)] )   
    #plt.subplot(3,1,3)
    if  any(i > 0 for i in cap_s1_hf):
        ax3.plot(x,cap_s1_hf,  ':x' ,label = 'S1')
    if  any(i > 0 for i in cap_s2_hf):                        
        ax3.plot(x,cap_s2_hf,  '-.', marker = 'P',  markevery = mark_f_h ,label = 'S2')
    if  any(i > 0 for i in cap_s3_hf):
        ax3.plot(x,cap_s3_hf,  '-.oy' ,  markevery = mark_f_h ,label = 'S3')
    if  any(i > 0 for i in cap_s4_hf):
        ax3.plot(x,cap_s4_hf, '-.r', marker = 'o',  markevery = mark_f_h ,label = 'S4')
    if  any(i > 0 for i in cap_s5_hf):
        ax3.plot(x,cap_s5_hf, '-' ,   marker = '*' , markevery = mark_f_h,label = 'DRL HF S5')
    if  any(i > 0 for i in cap_s6_hf):
        ax3.plot(x,cap_s6_hf, '-g' , marker = 's' , markevery = mark_f_h ,label = 'S6')
    
    #axs[0][2].set_xticks([i for i in range (16)] )
    #axs[0][0].set_xticks([i for i in range (16)] )
    #plt.subplot(3,1,1)
    caps1_dr = ax1.plot(x,cap_s1_dr, ':xb', label = 'S1',  markevery = mark_f_h,  alpha = .9)
    #caps2_dr = plt.plot(x,cap_s2_dr, label = 'Flex DR S1')
    #caps3_dr = plt.plot(x,cap_s3_dr, label = 'Flex DR S1')
    #caps4_dr = plt.plot(x,cap_s4_dr, label = 'Flex DR S1')
    #caps5_dr = plt.plot(x,cap_s5_dr, label = 'Flex DR S1')
    caps6_dr = ax1.plot(x,cap_s6_dr,'g', marker= 's', label = 'S6', markevery = mark_f_h , alpha = .5)
    
    #axs[0][0].legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.3))
    
    
    # dems1 = axs[0].plot(x, s_1_demand_df, '--D' ,  linewidth = .8 , markevery = mark_f_h , label = 'S1 FW'  )
    # dems2 = axs[0].plot(x, s_2_demand_df, '-.+'  , markevery = mark_f_h , label = 'S2 FW' )
    # dems3 = axs[0].plot(x, s_3_demand_df, ':x'  , markevery = mark_f_h ,label = 'S3 FW' )
    # dems4 = axs[0].plot(x, s_4_demand_df, marker = 'o', markevery = mark_f_h ,label = 'S4 FW' )
    # dems5 = axs[0].plot(x, s_5_demand_df, marker = '*' ,markevery = mark_f_h , label = 'S5 FW' )
    # dems6 = axs[0].plot(x, s_6_demand_df, marker = 's', markevery = mark_f_h ,label = 'S6 FW' )
    
    
    
    #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
    ax1.legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.23))
    ax2.legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.23))
    ax3.legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.23))
    #fig.suptitle('Demand and capacity evolution over project lifetime' , fontsize= 16)
    
    ax1.set_title("Decentralised FDR" , fontsize = 16)
    ax2.set_title("DRL-LF" , fontsize = 16)
    ax3.set_title("DRL-HF" , fontsize = 16)
    
    ax1.set_ylabel('Installed Capacity (tpd)' , fontsize = 14)
    # ax2.set_ylabel('Tonnes per day' , fontsize = 13)
    # ax3.set_ylabel('Tonnes per day' , fontsize = 13)
    plt.subplots_adjust( hspace = .4)
    
    ax1.set_xlabel('Years', fontsize = 13)
    ax2.set_xlabel('Years', fontsize = 13)
    ax3.set_xlabel('Years', fontsize = 13)
    #plt.ylabel('Tonnes per day' , fontsize = 16)
    plt.tight_layout()
    #plt.setxlim(15)
    return fig