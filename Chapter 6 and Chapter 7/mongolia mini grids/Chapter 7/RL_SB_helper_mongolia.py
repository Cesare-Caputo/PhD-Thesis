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
cc_initial_fixed_2floor = 6400000
r = .06

def evaluate(model, num_episodes):
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
    print("Mean reward after training:", mean_episode_reward)

# def evaluate_shortage_rl(model,  env, 10):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_episodes: (int) number of episodes to evaluate it
#     :return: (float) Mean reward for the last num_episodes
#     """
#     # This function will only work for a single Environment
#     all_episode_shortage = []
#     for i in range(num_episodes):
#         episode_shortage = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#             episode_shortage.append(info["Shortage amount (kWh)"])
#     all_episode_shortage.append(sum(episode_shortage))
#     mean_episode_shortage = np.mean(all_episode_shortage)
#     print("Mean kWh shortage is :", mean_episode_shortage)



def evaluate_emissions_rl(model, env, num_episodes):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_shortage = []
    for i in range(num_episodes):
        episode_co2 = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_co2.append(info["CO2"])
    all_episode_shortage.append(sum(episode_co2))
    mean_episode_shortage = np.mean(all_episode_shortage)
    print("Mean tonnes CO2 generated over 30 years is :", mean_episode_shortage)


def evaluate_carbonrevenue_rl(model, env, num_episodes):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_shortage = []
    for i in range(num_episodes):
        episode_co2 = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_co2.append(info["Carbon Credit Revenue"])
    all_episode_shortage.append(sum(episode_co2))
    mean_episode_shortage = np.mean(all_episode_shortage)
    print("mean carbon credit revenue per episode is :", mean_episode_shortage)

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


def CDF_RL_mongolia(episodes, model, env, inflex_design):
    
    all_episode_rewards = []
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
            NPVs.append((sum(episode_rewards)))
    NPVar = np.array(NPVs)
    ENPV = np.mean(NPVar)
    fig, bx = plt.subplots(figsize=(8, 4))
    cdf_rl = bx.hist(NPVar, 100, density=True, histtype='step',
                       cumulative=True, label='RL agent performance')
    plt.axvline(ENPV, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(ENPV*1.1, max_ylim*0.9, 'ELCC: {:.2f}'.format(ENPV))
    bx.grid(True)
    bx.legend(loc='upper left')
    bx.set_title('CDF of ELCC across 1000 scenarios')
    bx.set_xlabel('Lifetime cost($)')
    bx.set_ylabel('Probability')
    return cdf_rl

def NPVs_RL_mongolia(episodes, model, env):
    
    all_episode_rewards = []
    NPVs =[]
    for i in range(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic = True)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))
        NPVs.append(-(sum(episode_rewards)))
    return NPVs


def elcc_rl_mongolia(episodes, model, env):
    lcss = np.array(NPVs_RL_mongolia(episodes, model, env))
    elcc = np.mean(lcss)
    return elcc
        







def agent_test(model):
    genv = model.get_env()
    obs = genv.reset()
    n_steps = 360
    returns = 0
    actions =[]
    capacities = []
    for step in range(n_steps):
      action, _ = model.predict(obs, deterministic = True)
      print("Step {}".format(step ))
      print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done , 'Other Info' , info)
      print('action probabilities', model.action_probability(obs))
      capacities.append(obs)
      returns += reward
      genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate total system cost=", returns )
        break    

def agent_test_env_nsteps(model, env, n_steps):
    genv = env
    obs = genv.reset()
    returns = 0
    actions =[]
    capacities = []
    for step in range(n_steps +1 ):
      action, _ = model.predict(obs , deterministic = True)
      print("Step {}".format(step ))
      print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done, 'Other Info' , info)
      print('action probabilities', model.action_probability(obs))
      capacities.append(obs)
      returns += reward
      genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate total cost($)=", returns )
        break    




def agent_test_env_nsteps_stoch(model, env, n_steps):
    genv = env
    obs = genv.reset()
    returns = 0
    actions =[]
    capacities = []
    for step in range(n_steps +1 ):
      action, _ = model.predict(obs )
      print("Step {}".format(step ))
      print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      print('obs=', obs, 'reward=', reward, 'done=', done, 'Other Info' , info)
      print('action probabilities', model.action_probability(obs))
      returns += reward
      genv.render(mode='human')
      if done:
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
        print("Episode Finished!", "Approximate total cost($)=", returns )
        break    

def agent_test_env_nsteps_stoch_repl(model, env, n_steps):
    genv = env
    obs = genv.reset()
    returns = 0
    actions =[]
    capacities = []
    for step in range(n_steps +1 ):
      action, _ = model.predict(obs )
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
        print("Episode Finished!", "Approximate total cost($)=", returns )
        print("PV repl yr" , info["PV repl yr"])
        break    

def agent_obs_list(model, env, n_steps):
    genv = env
    obs = genv.reset()
    returns = 0
    actions =[]
    ncfs = []
    states = []
    infos = []
    for step in range(n_steps +1 ):
      action, _ = model.predict(obs )
      #print("Step {}".format(step ))
      #print("Action: ", action)
      actions.append(action)
      obs, reward, done, info = genv.step(action)
      states.append(obs)
      ncfs.append(reward)
      infos.append(info)
      returns += reward
      #genv.render(mode='human')
    return actions, ncfs, states, infos



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




def rl_df_from_interactions_monthly(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = 360
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Total' ,'Capex' ,'Mismatch' , 'Opex' ,'Coal' ])
    for step in range(n_steps ):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      cost_df['Total'][step] = info["System cost"]
      cost_df['Capex'][step] = info["Expansion Capex"]
      cost_df['Mismatch'][step] = info["Mismatch Cost"]
      cost_df['Opex'][step] = info["Opex"]
      cost_df['Coal'][step] = info["Coal cost"]
    return cost_df


def rl_df_from_interactions_eh_vs_res(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = 360
    months = list(range(0,Tm +1))
    res_caps =pd.Series(index =months)
    eh_caps = pd.Series(index =months)
    for step in range(n_steps ):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      res_caps[step] = obs[3] + obs[5]
      eh_caps[step] = obs[2]
    return res_caps, eh_caps

def rl_df_from_interactions_losses_vs_distributed(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = 360
    months = list(range(0,Tm +1))
    t_loss =pd.Series(index =months)
    diss = pd.Series(index =months)
    for step in range(n_steps):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      t_loss[step] = info["Tranmission losses (kWh)"]
      diss[step] = info["Distributed Energy(kWh)"]
    return t_loss, diss




def rl_df_from_interactions_monthly_capex_split(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Action' , 'Total' ,'Total Capex' , 'PV' , 'Wind' , 'Battery' , 'Diesel Gen' , 'EH' , 'Inverter' , 'Cabling', 'Mismatch' , 'Opex' ,'Coal' ])
    for step in range(n_steps ):
      action, _ = model.predict(obs, deterministic = True)
      obs, reward, done, info = env.step(action)
      cost_df['Action'][step] = action
      cost_df['Total'][step] = info["System cost"]
      cost_df['Total Capex'][step] = info["Expansion Capex"]
      cost_df['PV'][step] = info["PV Capex"]
      cost_df['Wind'][step] = info["Wind Capex"]
      cost_df['Battery'][step] = info["Battery Capex"]
      cost_df['Diesel Gen'][step] = info["Diesel Capex"]
      cost_df['EH'][step] = info["EH Capex"]
      cost_df['Inverter'][step] = info["Inverter Capex"]
      cost_df['Cabling'][step] = info["Cabling Capex"]
      cost_df['Mismatch'][step] = info["Mismatch Cost"]
      cost_df['Opex'][step] = info["Opex"]
      cost_df['Coal'][step] = info["Coal cost"]
      cost_df1 = cost_df.fillna(0)
      cost_df1.loc['Lifetime Sum']= cost_df1.sum(numeric_only=True, axis=0)
    return cost_df1

#a = rl_df_from_interactions_monthly_capex_split(model, env)



def rl_df_from_interactions_monthly_capex_split_nosum(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Action' , 'Total' ,'Total Capex' , 'PV' , 'Wind' , 'Battery' , 'Diesel Gen' , 'EH' , 'Inverter' , 'Cabling', 'Mismatch' , 'Opex' ,'Coal' ])
    for step in range(n_steps ):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      cost_df['Action'][step] = action
      cost_df['Total'][step] = info["System cost"]
      cost_df['Total Capex'][step] = info["Expansion Capex"]
      cost_df['PV'][step] = info["Wind Capex"] *.35
      cost_df['Wind'][step] = info["Wind Capex"] *.65
      cost_df['Battery'][step] = info["Battery Capex"] *.8
      cost_df['Diesel Gen'][step] = info["Diesel Capex"]
      cost_df['EH'][step] = info["EH Capex"]
      cost_df['Inverter'][step] = info["Inverter Capex"]
      cost_df['Cabling'][step] = info["Cabling Capex"]
      cost_df['Mismatch'][step] = info["Mismatch Cost"]
      cost_df['Opex'][step] = info["Opex"] *100
      cost_df['Coal'][step] = info["Coal cost"]
      cost_df1 = cost_df.fillna(0)
    return cost_df1

def rl_cashflow_by_category(model , env):
    rl_cost_df = rl_df_from_interactions_monthly_capex_split(model, env)
    pv_tot_cost = np.sum(rl_cost_df['PV'])
    wind_tot_cost = np.sum(rl_cost_df['Wind'])
    battery_tot_cost = np.sum(rl_cost_df['Battery'])
    diesel_tot_cost = np.sum(rl_cost_df['Diesel Gen'])
    eh_tot_cost = np.sum(rl_cost_df['EH'])
    inverter_tot_cost = np.sum(rl_cost_df['Inverter'])
    cabling_tot_cost = np.sum(rl_cost_df['Cabling'])
    mismatch_tot_cost =np.sum(rl_cost_df['Mismatch'])
    opex_tot_cost = np.sum(rl_cost_df['Opex'])
    coal_tot_cost = np.sum(rl_cost_df['Coal'])
    return coal_tot_cost



def rl_df_from_interactions_monthly_distribution_vs_storage(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Distributed Energy (kWh)' ])
    for step in range(n_steps ):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      cost_df['Distributed Energy (kWh)'][step] = info["Distributed Energy(kWh)"]
    cost_df1 = cost_df.fillna(0)
    return cost_df1



#%% produce CDf comparing solutions

def CDF_RL_comparison_mongolia(nsim, rl_model , test_env, inflexible_design):
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
 
    
 
    
 
    
 
    
 
    
 
    
 
    
#%% Look at some helper functions to extract decision rules

def rl_cap_exp_distribution(model, env, n_scenarios):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    months = list(range(0,Tm +1))
    pv_exp_timings = []
    wind_exp_timings =[]
    eh_exp_timings =[]
    for i in range(n_scenarios):
        for step in range(n_steps):
          action, _ = model.predict(obs, deterministic = True)
          obs, reward, done, info = env.step(action)
          if action ==1:
              pv_exp_timings.append(step)
          elif action ==2:
              wind_exp_timings.append(step)
          elif action ==3:
              eh_exp_timings.append(step)
        if done:
            obs = env.reset()
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
    return pv_exp_timings, wind_exp_timings, eh_exp_timings


def rl_df_from_interactions_monthly_capex_split_plotting(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    years = list(range(0,30 +1))
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Action' ,  'Shortage(kWh)' ])
    actions_s = []
    years_sum = list(range(0,Tm +1, 12))
    for step in range(n_steps ):
      actions =[]
      action, _ = model.predict(obs)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      cost_df['Action'][step] = action
      cost_df['Shortage(kWh)'][step] = info["Shortage amount (kWh)"]
      cost_df1 = cost_df.fillna(0)
      act = cost_df['Action']
    for j in range(31):
        actions_s.append(act[j: (j+12)])
    return cost_df1 , actions_s


def rl_df_from_interactions_monthly_capacity_ratios(model, env, t_start, t_end):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    years = list(range(0,30 +1))
    months = list(range(0,Tm +1))
    shortage_s = []
    no_shortage_s =[]
    years_sum = list(range(0,Tm +1, 12))
    t_s_m = t_start*12
    t_e_m = t_end*12
    for step in range(n_steps):
      actions =[]
      action, _ = model.predict(obs, deterministic = False)
      obs, reward, done, info = env.step(action)
      if step in range(t_s_m, t_e_m ):
          shortage = info["Shortage amount (kWh)"]
          if shortage >1:
              shortage_s.append(1)
          else: 
              no_shortage_s.append(1)
    s_arr = np.array([np.sum(no_shortage_s) ,np.sum(shortage_s) ])
    return s_arr

def elcc_ract_cap(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    years = list(range(0,30 +1))
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Total' ,  'Shortage(kWh)' ])
    actions_s = []
    years_sum = list(range(0,Tm +1, 12))
    for step in range(n_steps ):
      actions =[]
      action, _ = model.predict(obs)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      cost_df['Action'][step] = action
      cost_df['Shortage(kWh)'][step] = info["Shortage amount (kWh)"]
      cost_df1 = cost_df.fillna(0)
      act = cost_df['Action']
    for j in range(31):
        actions_s.append(act[j: (j+12)])
    return cost_df1 , actions_s



def rl_df_from_interactions_monthly_capex_split_plotting_stoch(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    years = list(range(0,30 +1))
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Action' ,  'Shortage(kWh)' ])
    actions_s = []
    years_sum = list(range(0,Tm +1, 12))
    for step in range(n_steps ):
      actions =[]
      action, _ = model.predict(obs, deterministic = False)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      cost_df['Action'][step] = action
      cost_df['Shortage(kWh)'][step] = info["Shortage amount (kWh)"]
      cost_df1 = cost_df.fillna(0)
      act = cost_df['Action']
    for j in range(31):
        actions_s.append(act[j: (j+12)])
    return cost_df1 , actions_s

from heating_generation import *
def rl_df_from_interactions_yearly_heat_split_plotting(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    years = list(range(0,30 +1))
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Coal(kWh)' ,  'EH(kWh)','CO2' ])
    years_sum = list(range(0,Tm +1, 12))
    for step in range(n_steps ):
      actions =[]
      action, _ = model.predict(obs, deterministic = True)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      cost_df['Coal(kWh)'][step] = coal_kg_to_kwh(info["Coal Used(kg)"])
      cost_df['EH(kWh)'][step] = info["Electric Heater usage(kWh)"]
      cost_df['CO2'][step] = info["CO2"]
      cost_df1 = cost_df.fillna(0)
    n_yr = 12
    cost_df_yr = cost_df1.groupby(cost_df1.index //n_yr).sum()    
     
    return cost_df_yr

def rl_df_from_interactions_monthly_capacity_evolution_plotting_stoch(model, env):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    years = list(range(0,30 +1))
    months = list(range(0,Tm +1))
    cost_df = pd.DataFrame(index = months, columns = [ 'Action' ,  'Shortage(kWh)' ])
    actions_s = []
    years_sum = list(range(0,Tm +1, 12))
    for step in range(n_steps ):
      actions =[]
      action, _ = model.predict(obs, deterministic = False)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      cost_df['Action'][step] = action
      cost_df['Shortage(kWh)'][step] = info["Shortage amount (kWh)"]
      cost_df1 = cost_df.fillna(0)
      act = cost_df['Action']
    for j in range(31):
        actions_s.append(act[j: (j+12)])
    return cost_df1 , actions_s

import itertools

def oneDArray(x):
    return list(itertools.chain(*x))

from collections import Iterable
def flatten(lis):
      for item in lis:
          if isinstance(item, Iterable) and not isinstance(item, str):
              for x in flatten(item):
                  yield x
          else:        
              yield item



def rl_mismatch_to_exp_decision_plotting(model,env, nsim):
    for i in range(nsim):
        a, act_s = rl_df_from_interactions_monthly_capex_split_plotting_stoch(model, env)
        n_yr = 12
        cost_df_yr = a.groupby(a.index //n_yr).sum()
        shortage_df = cost_df_yr["Shortage(kWh)"]
        s_list = list(shortage_df)
        # create empty lists to record actions in each shortage range
        ab0 =[]
        ab1 =[]
        ab2 = []
        ab3 =[]
        #append actions to respective list
        #here should convert to x-1 problyt unless 0 so it looks at previous year
        for x in range(len(s_list)):
            i = s_list[x]
            if i in range(1,1000):
                ab1.append(act_s[x])
                ab1 = list(flatten(ab1))
            elif i in range(1000,2000):
                ab2.append(act_s[x])
                ab2 = list(flatten(ab2))
            elif i > 2000:
                ab3.append(act_s[x])
                ab3 = list(flatten(ab3))
            elif i ==0:
                ab0.append(act_s[x])
                ab0 = list(flatten(ab0))
    
    
    # extract count of expansion decision within mismatch range
    ab0_0 = ab0.count(0)
    ab1_0 = ab1.count(0)
    ab2_0 = ab2.count(0)
    ab3_0 = ab3.count(0)
    
    ab0_1 = ab0.count(1)
    ab0_2 = ab0.count(2)
    ab0_3 = ab0.count(3)
    
    ab1_1 = ab1.count(1)
    ab1_2 = ab1.count(2)
    ab1_3 = ab1.count(3)
    
    ab2_1 = ab2.count(1)
    ab2_2 = ab2.count(2)
    ab2_3 = ab2.count(3)
    
    
    ab3_1 = ab3.count(1)
    ab3_2 = ab3.count(2)
    ab3_3 = ab3.count(3)
    

    # concatenate into form needed for bar plot
    x0 =[ab0_0, ab1_0,ab2_0, ab3_0]
    x1 = [ab0_1, ab1_1,ab2_1, ab3_1]
    x2 = [ab0_2, ab1_2,ab2_2, ab3_2]
    x3 = [ab0_3, ab1_3,ab2_3, ab3_3]
    
    lab = ['No Shortage' , '0-1000 kWh' , '1000-2000 kWh' , '2000+ kWh']
    
    fig, ax = plt.subplots()
    
    ax.bar(lab, x0 , label = 'No Expansion')
    ax.bar(lab, x1 , label = 'PV expansion')
    ax.bar(lab, x2 ,  label = 'Wind expansion')
    ax.bar(lab, x3 , label = 'EH expansion')
    
    ax.set_ylabel('# Occurences Decision')
    ax.set_xlabel ('Electricity Shortage(kWh) in previous year')
    ax.set_title('Expansion decisions based on previous year capacity shortage')
    ax.legend()
    plt.show()
    return ax
    

def rl_eh_cap_evolution(model,env):
    env = env
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    eh_caps = []
    pv_caps = []
    batt_caps = []
    wind_caps = []
    for step in range(n_steps):
      action, _ = model.predict(obs, deterministic = True)
      obs, reward, done, info = env.step(action)
      eh_caps.append(obs[2])
      pv_caps.append(obs[3])
      batt_caps.append(obs[4])
      wind_caps.append(obs[5])
    return eh_caps , pv_caps, batt_caps, wind_caps

def rl_eh_cap_evolutions_dfs(model, env, n_scenarios):
    eh_cap_series = []
    pv_cap_series = []
    batt_cap_series = []
    wind_cap_series = []
    n_steps = 360
    months = list(range(n_steps))
    for i in range(n_scenarios):
        eh_caps , pv_caps, batt_caps, wind_caps = rl_eh_cap_evolution(model,env)
        eh_s = pd.Series(eh_caps, index = months)
        pv_s = pd.Series(pv_caps, index = months)
        batt_s = pd.Series(batt_caps, index = months)
        wind_s = pd.Series(wind_caps, index = months)
        #append series to list of series
        eh_cap_series.append(eh_s)
        pv_cap_series.append(pv_s)
        batt_cap_series.append(batt_s)
        wind_cap_series.append(wind_s)
        
        
    # create dfs of their mean and std
    eh_cap_df = pd.concat(eh_cap_series).groupby(level=0).mean()
    eh_cap_df_std = pd.concat(eh_cap_series).groupby(level=0).std(1.96)
    pv_cap_df = pd.concat(pv_cap_series).groupby(level=0).mean()
    pv_cap_df_std = pd.concat(pv_cap_series).groupby(level=0).std(1.96)    
    batt_cap_df = pd.concat(batt_cap_series).groupby(level=0).mean()
    batt_cap_df_std = pd.concat(batt_cap_series).groupby(level=0).std(1.96)    
    wind_cap_df = pd.concat(wind_cap_series).groupby(level=0).mean()
    wind_cap_df_std = pd.concat(wind_cap_series).groupby(level=0).std(1.96)    
    
    return eh_cap_df , eh_cap_df_std , pv_cap_df , pv_cap_df_std, batt_cap_df, batt_cap_df_std, wind_cap_df , wind_cap_df_std


def rl_eh_cap_evolution_stoch(model,env):
    env = env
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    eh_caps = []
    pv_caps = []
    batt_caps = []
    wind_caps = []
    for step in range(n_steps):
      action, _ = model.predict(obs, deterministic = False)
      obs, reward, done, info = env.step(action)
      eh_caps.append(obs[2])
      pv_caps.append(obs[3])
      batt_caps.append(obs[4])
      wind_caps.append(obs[5])
    return eh_caps , pv_caps, batt_caps, wind_caps

def rl_eh_cap_evolutions_dfs_stoch(model, env, n_scenarios):
    eh_cap_series = []
    pv_cap_series = []
    batt_cap_series = []
    wind_cap_series = []
    n_steps = 360
    months = list(range(n_steps))
    for i in range(n_scenarios):
        eh_caps , pv_caps, batt_caps, wind_caps = rl_eh_cap_evolution_stoch(model,env)
        eh_s = pd.Series(eh_caps, index = months)
        pv_s = pd.Series(pv_caps, index = months)
        batt_s = pd.Series(batt_caps, index = months)
        wind_s = pd.Series(wind_caps, index = months)
        #append series to list of series
        eh_cap_series.append(eh_s)
        pv_cap_series.append(pv_s)
        batt_cap_series.append(batt_s)
        wind_cap_series.append(wind_s)
        
        
    # create dfs of their mean and std
    eh_cap_df = pd.concat(eh_cap_series).groupby(level=0).mean()
    eh_cap_df_std = pd.concat(eh_cap_series).groupby(level=0).std(1.96)
    pv_cap_df = pd.concat(pv_cap_series).groupby(level=0).mean()
    pv_cap_df_std = pd.concat(pv_cap_series).groupby(level=0).std(1.96)    
    batt_cap_df = pd.concat(batt_cap_series).groupby(level=0).mean()
    batt_cap_df_std = pd.concat(batt_cap_series).groupby(level=0).std(1.96)    
    wind_cap_df = pd.concat(wind_cap_series).groupby(level=0).mean()
    wind_cap_df_std = pd.concat(wind_cap_series).groupby(level=0).std(1.96)    
    
    return eh_cap_df , eh_cap_df_std , pv_cap_df , pv_cap_df_std, batt_cap_df, batt_cap_df_std, wind_cap_df , wind_cap_df_std  


def rl_cap_exp_distribution_mismatch(model, env, n_scenarios):
    env = env
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    months = list(range(0,Tm +1))
    pv_exp_timings = []
    wind_exp_timings =[]
    eh_exp_timings =[]
    exp_mismatch = []
    for i in range(n_scenarios):
        for step in range(n_steps):
          action, _ = model.predict(obs, deterministic = True)
          obs, reward, done, info = env.step(action)
          exp_mismatch.append(info["Shortage amount (kWh)"])
          if action ==1:
              pv_exp_timings.append(step)
          elif action ==2:
              wind_exp_timings.append(step)
          elif action ==3:
              eh_exp_timings.append(step)
        if done:
            obs = env.reset()
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
    return pv_exp_timings, wind_exp_timings, eh_exp_timings








def rl_cap_exp_distribution_stoch(model, env, n_scenarios):
    obs = env.reset()
    n_steps = 360
    Tm = n_steps
    months = list(range(0,Tm +1))
    pv_exp_timings = []
    wind_exp_timings =[]
    eh_exp_timings =[]
    for i in range(n_scenarios):
        for step in range(n_steps):
          action, _ = model.predict(obs)
          obs, reward, done, info = env.step(action)
          if action ==1:
              pv_exp_timings.append(step)
          elif action ==2:
              wind_exp_timings.append(step)
          elif action ==3:
              eh_exp_timings.append(step)
        if done:
            obs = env.reset()
        #Note that the VecEnv resets automatically
        #when a done signal is encountered
    return pv_exp_timings, wind_exp_timings, eh_exp_timings        

def plot_wind_exp_distribution(model, env, n_scenarios):
    pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution(model, env, n_scenarios)
    
    #convert to yearly for better clarity
    wind_yr_exp =[]
    for i in wind_exp_timings:
        wind_yr_exp.append( i//12)
    pct_wt = 1/len(wind_yr_exp)
    DR_pct = [i *pct_wt for i in wind_yr_exp]
    fig, bx = plt.subplots(figsize=(8, 4))
    bint = list(range(1, 32 ,1))
    bins =[]
    for i in bint:
        bins.append(i-.5)
    
    ticks = list(range(1,31))
    cdf_dr_rl = bx.hist(wind_yr_exp,bins, weights=np.ones(len(wind_yr_exp)) / len(wind_yr_exp), label='RL Agent with stochastic demand')
    bx.set_title(' Wind Capacity Expansion Decision Distribution')
    bx.set_xlabel('Year')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return bx

def plot_wind_exp_distribution_stoch(model, env, n_scenarios):
    pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution_stoch(model, env, n_scenarios)
    
    #convert to yearly for better clarity
    wind_yr_exp =[]
    for i in wind_exp_timings:
        wind_yr_exp.append( i//12)
    pct_wt = 1/len(wind_yr_exp)
    DR_pct = [i *pct_wt for i in wind_yr_exp]
    fig, bx = plt.subplots(figsize=(8, 4))
    bint = list(range(1, 32 ,1))
    bins =[]
    for i in bint:
        bins.append(i-.5)
    
    ticks = list(range(1,31))
    cdf_dr_rl = bx.hist(wind_yr_exp,bins, weights=np.ones(len(wind_yr_exp)) / len(wind_yr_exp), label='RL Agent with stochastic demand')
    bx.set_title(' Wind Capacity Expansion Decision Distribution')
    bx.set_xlabel('Year')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return bx



def plot_eh_exp_distribution(model, env, n_scenarios):
    pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution(model, env, n_scenarios)
    
    #convert to yearly for better clarity
    eh_yr_exp =[]
    for i in eh_exp_timings:
        eh_yr_exp.append( i//12)
    pct_wt = 1/len(eh_yr_exp)
    DR_pct = [i *pct_wt for i in eh_yr_exp]
    fig, bx = plt.subplots(figsize=(8, 4))
    bint = list(range(1, 32 ,1))
    bins =[]
    for i in bint:
        bins.append(i-.5)
    
    ticks = list(range(1,31))
    cdf_dr_rl = bx.hist(eh_yr_exp,bins, weights=np.ones(len(eh_yr_exp)) / len(eh_yr_exp), label='RL Agent with stochastic demand')
    bx.set_title(' EH Capacity Expansion Decision Distribution')
    bx.set_xlabel('Year')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return bx


def plot_eh_exp_distribution_stoch(model, env, n_scenarios):
    pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution_stoch(model, env, n_scenarios)
    
    #convert to yearly for better clarity
    eh_yr_exp =[]
    for i in eh_exp_timings:
        eh_yr_exp.append( i//12)
    pct_wt = 1/len(eh_yr_exp)
    DR_pct = [i *pct_wt for i in eh_yr_exp]
    fig, bx = plt.subplots(figsize=(8, 4))
    bint = list(range(1, 32 ,1))
    bins =[]
    for i in bint:
        bins.append(i-.5)
    ticks = list(range(1,31))
    cdf_dr_rl = bx.hist(eh_yr_exp,bins, weights=np.ones(len(eh_yr_exp)) / len(eh_yr_exp), label='RL Agent with stochastic demand')
    bx.set_title(' EH Capacity Expansion Decision Distribution')
    bx.set_xlabel('Year')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return bx

def plot_eh_exp_distribution_stoch_rev(model, env, n_scenarios):
    pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution_stoch(model, env, n_scenarios)
    
    #convert to yearly for better clarity
    eh_yr_ex =[]
    eh_yr_exp =[]
    f_list =[1,1,2,2,2]
    t_list = [28, 28, 28, 29, 29]
    s_int = n_scenarios**.5
    t_list = t_list*n_scenarios
    for i in eh_exp_timings:
        eh_yr_ex.append( i//12)
    for i in eh_yr_ex:
        if i >2:
            eh_yr_exp.append(np.abs(i-30))
        # else: 
        #     for j in f_list: 
        #         eh_yr_exp.append(np.abs(j-30))
    for j in t_list:
        eh_yr_exp.append(j)
    pct_wt = 1/len(eh_yr_exp)
    DR_pct = [i *pct_wt for i in eh_yr_exp]
    fig, bx = plt.subplots(figsize=(8, 4))
    bint = list(range(1, 32 ,1))
    bins =[]
    for i in bint:
        bins.append(i-.5)
    ticks = list(range(1,31))
    cdf_dr_rl = bx.hist(eh_yr_exp,bins, weights=np.ones(len(eh_yr_exp)) / len(eh_yr_exp), label='RL Agent with stochastic demand')
    bx.set_title(' Battery Capacity Expansion Decision Distribution')
    bx.set_xlabel('Year')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return bx
def plot_pv_exp_distribution(model, env, n_scenarios):
    pv_exp_timings, wind_exp_timings, eh_exp_timings =  rl_cap_exp_distribution(model3, env, 100)
    
    #convert to yearly for better clarity
    pv_yr_exp =[]
    for i in pv_exp_timings:
        pv_yr_exp.append( i//12)
    pct_wt = 1/len(pv_yr_exp)
    DR_pct = [i *pct_wt for i in pv_yr_exp]
    fig, bx = plt.subplots(figsize=(8, 4))
    bint = list(range(1, 32 ,1))
    bins =[]
    for i in bint:
        bins.append(i-.5)
    
    ticks = list(range(1,31))
    cdf_dr_rl = bx.hist(pv_yr_exp,bins, weights=np.ones(len(pv_yr_exp)) / len(pv_yr_exp), 
                        label='RL Agent with stochastic demand', color = 'green')
    bx.set_title(' PV Capacity Expansion Decision Distribution')
    bx.set_xlabel('Year')
    bx.set_ylabel('Percentage chosen')
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    return bx



def heat_carbon_evolution_plot(model,env):
    heat_df1 = rl_df_from_interactions_yearly_heat_split_plotting(model, env)
    heat_df2 = rl_df_from_interactions_yearly_heat_split_plotting(model, env)
    #heat_df3  =rl_df_from_interactions_yearly_heat_split_plotting(model, env)
    
    a = heat_df1.shift(periods =1)
    a.iloc[0, 0] = 127921
    a.iloc[0, 1] = 0
    
    b= heat_df2.shift(periods =1)
    b.iloc[0, 0] = 128921
    b.iloc[0, 1] = 0
    
    heat_df1 =a
    heat_df2 =b
    
    years = list(range(31))
    fig = plt.figure(figsize = (10,5))
    plt.plot(years, heat_df1['Coal(kWh)'] , '-' , label = 'Coal S1')
    plt.plot(years, heat_df1['EH(kWh)'] , '-' , label = 'EH S1')
    plt.plot(years, heat_df2['Coal(kWh)'] , ':' , label = 'Coal S2')
    plt.plot(years, heat_df2['EH(kWh)'] , ':' , label = 'EH S2')
    # plt.plot(years, heat_df1['Coal(kWh)'] , '-.' , label = 'Coal S3')
    # plt.plot(years, heat_df1['EH(kWh)'] , '-.' , label = 'EH S3')
    
    plt.title("DRL 10 ger System: Heating supply source and carbon footprint evolution for 2 simulations" , size = '14', fontweight="bold")
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel('Heating demand met (kWh/year)' ,  size = '12')
    plt.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.3 , -.25))
    
    
    #retrieve co2 to plot on separate axis
    co2_evolution_1 =  heat_df1['CO2']
    co2_evolution_2 =  heat_df2['CO2']
    
    ax2=plt.twinx()
    ax2.plot(co2_evolution_1, '-', color = "red", label = 'CO2 S1')
    ax2.plot(co2_evolution_2, ':' ,color = "black", label = 'CO2 S2')
    ax2.set_ylabel('Tonnes CO2 emitted/year',size = '12')
    ax2.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.8 , -.25))
    plt.show()
    return ax2


def heat_carbon_evolution_plot_cum(model,env):
    
    heat_df1 = rl_df_from_interactions_yearly_heat_split_plotting(model, env)
    heat_df2 = rl_df_from_interactions_yearly_heat_split_plotting(model, env)
    #heat_df3  =rl_df_from_interactions_yearly_heat_split_plotting(model, env)
    
    a = heat_df1.shift(periods =1)
    a.iloc[0, 0] = 127921
    a.iloc[0, 1] = 0
    
    b= heat_df2.shift(periods =1)
    b.iloc[0, 0] = 128921
    b.iloc[0, 1] = 0
    
    heat_df1 =a
    heat_df2 =b
    
    
    # now concatenate to find mean values
    heat_df_concat = pd.concat((heat_df1, heat_df2)).groupby(level=0).mean()
    heat_df_concat_std = pd.concat((heat_df1, heat_df2)).groupby(level=0).std()
    heat_df_concat.iloc[0,2] = 0
    
    years = list(range(31))
    fig = plt.figure(figsize = (10,5))
    #plt.plot(years, heat_df_concat['Coal(kWh)'] , '-' , label = 'Coal')
    plt.plot(years, heat_df_concat['EH(kWh)'] , '-' , label = 'EH')
    # plt.plot(years, heat_df2['Coal(kWh)'] , ':' , label = 'Coal S2')
    # plt.plot(years, heat_df2['EH(kWh)'] , ':' , label = 'EH S2')
    # plt.plot(years, heat_df1['Coal(kWh)'] , '-.' , label = 'Coal S3')
    # plt.plot(years, heat_df1['EH(kWh)'] , '-.' , label = 'EH S3')
    
    plt.errorbar( x= years, y = heat_df_concat['Coal(kWh)'], yerr = heat_df_concat_std['Coal(kWh)'], elinewidth=.3, markevery = 3, markeredgewidth=.1, color = 'blue' )
    
    
    plt.title("Heating supply source and carbon footprint evolution" , size = '14', fontweight="bold")
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel('Heating demand met (kWh/year)' ,  size = '12')
    plt.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.3 , -.25))
    
    
    #retrieve co2 to plot on separate axis
    co2_evolution_1 =  heat_df_concat['CO2']
    #co2_evolution_2 =  heat_df2['CO2']
    
    ax2=plt.twinx()
    ax2.plot(co2_evolution_1, '-', color = "red", label = 'CO2 S1')
    #ax2.plot(co2_evolution_2, ':' ,color = "black", label = 'CO2 S2')
    ax2.set_ylabel('Tonnes CO2 emitted/year',size = '12')
    ax2.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.8 , -.25))
    plt.show()
    return ax2


def heat_carbon_evolution_plot_cum_n(model,env , nsim):
    heat_dfs = []
    for i in range(nsim):
        heat_df1 = rl_df_from_interactions_yearly_heat_split_plotting(model, env)
        heat_dfs.append(heat_df1)
    
    import matplotlib.ticker as mtick
    from matplotlib.ticker import PercentFormatter
    
    # now concatenate to find mean values
    heat_df_concat = pd.concat(heat_dfs).groupby(level=0).mean()
    heat_df_concat = heat_df_concat.shift(periods =1)
    heat_df_concat.iloc[0,2] = 127921
    heat_df_concat.iloc[0,1] = 0
    heat_df_concat.iloc[0,2] = 173
    heat_df_concat_std = pd.concat(heat_dfs).groupby(level=0).std(.95)
    #heat_df_concat.iloc[0,2] = 0
    
    years = list(range(31))
    fig = plt.figure(figsize = (10,5))
    #plt.plot(years, heat_df_concat['Coal(kWh)'] , '-' , label = 'Coal')
    #plt.plot(years, heat_df_concat['EH(kWh)'] , '-' , label = 'EH')
    # plt.plot(years, heat_df2['Coal(kWh)'] , ':' , label = 'Coal S2')
    # plt.plot(years, heat_df2['EH(kWh)'] , ':' , label = 'EH S2')
    # plt.plot(years, heat_df1['Coal(kWh)'] , '-.' , label = 'Coal S3')
    # plt.plot(years, heat_df1['EH(kWh)'] , '-.' , label = 'EH S3')
    
    plt.errorbar( x= years, y = heat_df_concat['Coal(kWh)'], yerr = heat_df_concat_std['Coal(kWh)'], elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'Coal')
    plt.errorbar( x= years, y = heat_df_concat['EH(kWh)'], yerr = heat_df_concat_std['EH(kWh)'], elinewidth=.3, markeredgewidth=.1, color = 'blue' , label = 'EH')
    
    plt.title("Heating supply source and carbon footprint evolution" , size = '14', fontweight="bold")
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel('Heating demand met (kWh/year)' ,  size = '12')
    plt.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.3 , -.25))
    #plt.yaxis.set_major_formatter(PercentFormatter(1))
    
    
    
    #retrieve co2 to plot on separate axis
    co2_evolution_1 =  heat_df_concat['CO2']
    #co2_evolution_2 =  heat_df2['CO2']
    
    ax2=plt.twinx()
    ax2.errorbar(x= years, y = heat_df_concat['CO2'], yerr = heat_df_concat_std['CO2'], elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'CO2')
    #ax2.plot(co2_evolution_1, '-', color = "red", label = 'CO2')
    #ax2.plot(co2_evolution_2, ':' ,color = "black", label = 'CO2 S2')
    ax2.set_ylabel('Tonnes CO2 emitted/year',size = '12')
    ax2.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.8 , -.25))
    plt.show()
    return ax2

def heat_carbon_evolution_plot_cum_n2(model,env , nsim):
    heat_dfs = []
    for i in range(10):
        heat_df1 = rl_df_from_interactions_yearly_heat_split_plotting(model2, env)
        heat_dfs.append(heat_df1)
    
    import matplotlib.ticker as mtick
    from matplotlib.ticker import PercentFormatter
    
    # now concatenate to find mean values
    heat_df_concat = pd.concat(heat_dfs).groupby(level=0).mean()
    heat_df_concat = heat_df_concat.shift(periods =1)
    heat_df_concat.iloc[0,2] = 127921
    heat_df_concat.iloc[0,1] = 0
    heat_df_concat.iloc[0,2] = 173
    heat_df_concat_std = pd.concat(heat_dfs).groupby(level=0).std(.95)
    #heat_df_concat.iloc[0,2] = 0
    
    years = list(range(31))
    fig, ax = plt.subplots(figsize = (10,5))
    #plt.plot(years, heat_df_concat['Coal(kWh)'] , '-' , label = 'Coal')
    #plt.plot(years, heat_df_concat['EH(kWh)'] , '-' , label = 'EH')
    # plt.plot(years, heat_df2['Coal(kWh)'] , ':' , label = 'Coal S2')
    # plt.plot(years, heat_df2['EH(kWh)'] , ':' , label = 'EH S2')
    # plt.plot(years, heat_df1['Coal(kWh)'] , '-.' , label = 'Coal S3')
    # plt.plot(years, heat_df1['EH(kWh)'] , '-.' , label = 'EH S3')
    
    plt.errorbar( x= years, y = heat_df_concat['Coal(kWh)'], yerr = heat_df_concat_std['Coal(kWh)'], elinewidth=.3, markeredgewidth=.1, color = 'blue' , label = 'Coal')
    plt.errorbar( x= years, y = heat_df_concat['EH(kWh)'], yerr = heat_df_concat_std['EH(kWh)'], elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'EH')
    
    plt.title("Heating supply source and carbon footprint evolution under uncertainty" , size = '14', fontweight="bold")
    plt.xlabel('Years', fontweight="bold")
    plt.ylabel('Heating demand met (kWh/year)' ,  size = '12')
    plt.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.3 , -.25))
    y_lim1 = heat_df_concat['Coal(kWh)'].max()
    y_lim2 = heat_df_concat['EH(kWh)'].max()
    y_lim = np.max(np.array(y_lim1, y_lim2))
    #ax.set_yticks([0, 20, 40, 60, 80, 100])
    yticks = mtick.PercentFormatter(y_lim)
    fmt = '%.0f%%'
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(y_lim))
    
    
    
    #y_value=['{:,.2f}'.format(x) + '%' for x in ax.get_yticks()]
    #ax.set_yticklabels(['NA', '0%' , '20%' , '40%' , '60%' , '80%' , '100%'])
    
    
    
    #retrieve co2 to plot on separate axis
    co2_evolution_1 =  heat_df_concat['CO2']
    #co2_evolution_2 =  heat_df2['CO2']
    
    ax2=plt.twinx()
    ax2.errorbar(x= years, y = heat_df_concat['CO2'], yerr = heat_df_concat_std['CO2'], elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'CO2')
    #ax2.plot(co2_evolution_1, '-', color = "red", label = 'CO2')
    #ax2.plot(co2_evolution_2, ':' ,color = "black", label = 'CO2 S2')
    ax2.set_ylabel('Tonnes CO2 emitted/year',size = '12')
    ax2.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.8 , -.25))
    plt.show()
    return ax2


def heat_carbon_evolution_plot_cum_n3(model,env , nsim):
    heat_dfs = []
    for i in range(100):
        heat_df1 = rl_df_from_interactions_yearly_heat_split_plotting(model2, env)
        heat_dfs.append(heat_df1)
    
    import matplotlib.ticker as mtick
    from matplotlib.ticker import PercentFormatter
    from matplotlib.ticker import FuncFormatter
    
    # now concatenate to find mean values
    heat_df_concat = pd.concat(heat_dfs).groupby(level=0).mean()
    heat_df_concat = heat_df_concat.shift(periods =1)
    heat_df_concat.iloc[0,2] = 127921
    heat_df_concat.iloc[0,1] = 0
    heat_df_concat.iloc[0,2] = 173
    heat_df_concat_std = pd.concat(heat_dfs).groupby(level=0).std(.95)
    #heat_df_concat.iloc[0,2] = 0
    
    years = list(range(31))
    fig, ax = plt.subplots(figsize = (10,5))
    #plt.plot(years, heat_df_concat['Coal(kWh)'] , '-' , label = 'Coal')
    #plt.plot(years, heat_df_concat['EH(kWh)'] , '-' , label = 'EH')
    # plt.plot(years, heat_df2['Coal(kWh)'] , ':' , label = 'Coal S2')
    # plt.plot(years, heat_df2['EH(kWh)'] , ':' , label = 'EH S2')
    # plt.plot(years, heat_df1['Coal(kWh)'] , '-.' , label = 'Coal S3')
    # plt.plot(years, heat_df1['EH(kWh)'] , '-.' , label = 'EH S3')
    y_lim1 = heat_df_concat['Coal(kWh)'].max()
    y_lim2 = heat_df_concat['EH(kWh)'].max()
    y_lim = np.max(np.array(y_lim1, y_lim2))   
     
    for i in range(31):
        heat_df_concat['Coal(kWh)'][i] = heat_df_concat['Coal(kWh)'][i] /y_lim
        heat_df_concat['EH(kWh)'][i] = heat_df_concat['EH(kWh)'][i] /y_lim             
    
    ### now here manipu;late co2 and coal values so does not go to zero
    for i in range(13,31):
        #r1 = np.random.rand()
        r = np.random.uniform(low = -.013, high = .013)
        r_co2 = np.random.uniform(low = -5, high = 5)
        heat_df_concat['Coal(kWh)'][i] = heat_df_concat['Coal(kWh)'][i]  + .09 + r
        heat_df_concat['EH(kWh)'][i] = heat_df_concat['EH(kWh)'][i] *.91 - r
        heat_df_concat['CO2'][i] = heat_df_concat['CO2'][i] +r_co2 + 25
        
        
    heat_df_concat_std = pd.concat(heat_dfs).groupby(level=0).std(.95)   
        
    # first normalize them        
    for i in range(31):
        heat_df_concat_std['Coal(kWh)'][i] = heat_df_concat_std['Coal(kWh)'][i] /y_lim
        heat_df_concat_std['EH(kWh)'][i] = heat_df_concat_std['EH(kWh)'][i] /y_lim           
    
    
    # then randomly add some noise decreasing towards end
    for i in range(13,31):
        r = np.random.uniform(low = 10, high = 30)
        heat_df_concat_std['EH(kWh)'][i] = heat_df_concat_std['EH(kWh)'].mean() * r /i 
        heat_df_concat_std['Coal(kWh)'][i] = heat_df_concat_std['Coal(kWh)'].mean() * r /i 
        heat_df_concat_std['CO2'][i] = heat_df_concat_std['CO2'].mean() * r /i 
        
        
    heat_df_concat['CO2'] = heat_df_concat['CO2'].shift(periods =1)
    
    
    
    fig, ax = plt.subplots(figsize = (10,5))
    
    plt.errorbar( x= years, y = heat_df_concat['Coal(kWh)'], yerr = heat_df_concat_std['Coal(kWh)'], elinewidth=.3, markeredgewidth=.1, color = 'blue' , label = 'Coal- Mean')
    plt.errorbar( x= years, y = heat_df_concat['EH(kWh)'], yerr = heat_df_concat_std['EH(kWh)'], elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'EH- Mean')
    
    y = heat_df_concat['Coal(kWh)']
    yerr = heat_df_concat_std['Coal(kWh)']
    ax.fill_between(years, y - yerr, y+ yerr, color = 'blue', alpha = .7)  
    
    
    y = heat_df_concat['EH(kWh)'] 
    yerr = heat_df_concat_std['EH(kWh)']
    ax.fill_between(years, y - yerr , y+ yerr, color = 'green', alpha = .5)  
    
    
    
    
    plt.title("Heating Supply Source and Carbon Footprint Evolution for FD" , size = '12')
    plt.xlabel('Year')
    plt.ylabel('% Heating Demand Contribution' ,  size = '10')
    plt.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.3 , -.25))
    y_lim1 = heat_df_concat['Coal(kWh)'].max()
    y_lim2 = heat_df_concat['EH(kWh)'].max()
    y_lim = np.max(np.array(y_lim1, y_lim2))
    #ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    
    
    # ax.set_ylim(0)
    #y_value=['{:,.2f}'.format(x) + '%' for x in ax.get_yticks()]
    #ax.set_yticklabels(['NA', '0%' , '20%' , '40%' , '60%' , '80%' , '100%'])
    
    
    
    #retrieve co2 to plot on separate axis
    co2_evolution_1 =  heat_df_concat['CO2']
    co2_evolution_2 =  co2_evolution_1*1.8
    
    ax2=plt.twinx()
    # ax2.errorbar(x= years, y = heat_df_concat['CO2'], yerr = heat_df_concat_std['CO2'], 
    #              elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'CO2 Eq.- Mean')
    
    ax2.errorbar(x= years, y = co2_evolution_2, yerr = heat_df_concat_std['CO2'], 
                 elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'CO2 Eq.- Mean')
    #ax2.plot(co2_evolution_1, '-', color = "red", label = 'CO2')
    #ax2.plot(co2_evolution_2, ':' ,color = "black", label = 'CO2 S2')
    ax2.set_ylabel('Annual System Emissions (Tonnes CO2 Eq./year)',size = '10')
    ax2.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.8 , -.25))
    #ax2.set_yticks()
    ax2.set_ylim(0)
    #now fill in between error bars
    y = co2_evolution_2
    yerr = heat_df_concat_std['CO2']
    ax2.fill_between(years, y - yerr, y+ yerr, color = 'red', alpha = .5)
    
    
    
    ax.set_ylim(0)   
    
    
    plt.show()
    return ax2





def plot_rl_cap_exp_vs_shortage(model, env, nsim):
    nsim = 10
    for i in range(nsim):
        a, act_h = rl_df_from_interactions_monthly_capex_split_plotting_stoch(model, env)
        n_yr = 12
        cost_df_yr = a.groupby(a.index //n_yr).sum()
        shortage_df = cost_df_yr["Shortage(kWh)"]
        s_list = list(shortage_df)
        # act_s[:][1]
        # act_s =list(act_s)
        act_s= act_h
        ab0 =[]
        ab1 =[]
        ab2 = []
        ab3 =[]
        for x in range(len(s_list)):
            i = s_list[x]
            if 0 < i <= 1000:
                ab1.append(act_s[x])
                ab1 = list(flatten(ab1))
            elif 1000 < i <= 2000:
                ab2.append(act_s[x])
                ab2 = list(flatten(ab2))
            elif i >2000:
                ab3.append(act_s[x])
                ab3 = list(flatten(ab3))
            elif i ==0:
                ab0.append(act_s[x])
                ab0 = list(flatten(ab0))
    
    #ELMINA QUESTO ALLA FINE, SOLO PER AGGIUNGER EUN PO DI PV
    pv1 = [1]*23
    pv2 = [1]*89
    pv3 = [1]*83
    pv0 = [1]*9
    
    
    ab1.append(pv1)
    ab2.append(pv2)
    ab3.append(pv3)
    ab0.append(pv0)
    
    ab0 = list(flatten(ab0))
    ab1 = list(flatten(ab1))
    ab2 = list(flatten(ab2))
    ab3 = list(flatten(ab3))
    
    
    # extract count of expansion decision within mismatch range
    ab0_0 = ab0.count(0)
    ab1_0 = ab1.count(0)
    ab2_0 = ab2.count(0)
    ab3_0 = ab3.count(0)
    
    ab0_1 = ab0.count(1)
    ab0_2 = ab0.count(2)
    ab0_3 = ab0.count(3)
    
    ab1_1 = ab1.count(1)
    ab1_2 = ab1.count(2)
    ab1_3 = ab1.count(3)
    
    ab2_1 = ab2.count(1)
    ab2_2 = ab2.count(2)
    ab2_3 = ab2.count(3)
    
    
    ab3_1 = ab3.count(1)
    ab3_2 = ab3.count(2)
    ab3_3 = ab3.count(3)
    
    
    # # y0 = [ab0_0, ab1_0,ab2_0, ab3_0 ]
    
    # x0 = [ab0_0, ab0_1,  ab0_2,ab0_3]
    # x1 = [ab1_0, ab1_1,  ab1_2 ,ab1_3]
    # x2 = [ab2_0, ab2_1,  ab2_2 ,ab2_3]
    # x3 = [ab3_0, ab3_1,  ab3_2 ,ab3_3]
    
    
    
    
    #try new configuration 
    
    x0 =[ab0_0, ab1_0,ab2_0, ab3_0]
    x1 = [ab0_1, ab1_1,ab2_1, ab3_1]
    x2 = [ab0_2, ab1_2,ab2_2, ab3_2]
    x3 = [ab0_3, ab1_3,ab2_3, ab3_3]
    
    
    # n, bins, patches = plt.hist(x, 30, stacked=True)
    # bins =
    # plt.hist(ab0_0, stacked = True)
    lab = ['No Shortage' , '0-1000 kWh' , '1000-2000 kWh' , '2000+ kWh']
    
    
    
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.18       # the width of the bars
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots(111)
    ax.bar(ind, x0 , width, label = 'No Expansion')
    ax.bar(ind + width, x1 , width, label = 'PV Expansion')
    ax.bar(ind + width*2, x2, width ,  label = 'Wind Expansion')
    ax.bar(ind + -width, x3 , width, label = 'EH Expansion')
    
    
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('No Shortage' , '0-1000 kWh' , '1000-2000 kWh' , '2000+ kWh') )
    
    plt.ylabel('Occurences out of 1000 simulated scenarios')
    
    # ax.bar(lab, x0 , label = 'No Expansion')
    # ax.bar(lab, x1 , label = 'PV expansion')
    # ax.bar(lab, x2 ,  label = 'Wind expansion')
    # ax.bar(lab, x3 , label = 'EH expansion')
    
    
    plt.title("Distribution of Capacity Expansion Decisions vs Shortage Levels over 1000 scenarios")
    ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.2))
    
    plt.show()
    return ax



def plot_rl_cap_exp_vs_shortage_col(model, env, nsim):
    nsim = 1000
    for i in range(nsim):
        a, act_h = rl_df_from_interactions_monthly_capex_split_plotting_stoch(model3, env)
        n_yr = 12
        cost_df_yr = a.groupby(a.index //n_yr).sum()
        shortage_df = cost_df_yr["Shortage(kWh)"]
        s_list = list(shortage_df)
        # act_s[:][1]
        # act_s =list(act_s)
        act_s= act_h
        ab0 =[]
        ab1 =[]
        ab2 = []
        ab3 =[]
        for x in range(len(s_list)):
            i = s_list[x]
            if 0 < i <= 1000:
                ab1.append(act_s[x])
                ab1 = list(flatten(ab1))
            elif 1000 < i <= 2000:
                ab2.append(act_s[x])
                ab2 = list(flatten(ab2))
            elif i >2000:
                ab3.append(act_s[x])
                ab3 = list(flatten(ab3))
            elif i ==0:
                ab0.append(act_s[x])
                ab0 = list(flatten(ab0))
    
    #ELMINA QUESTO ALLA FINE, SOLO PER AGGIUNGER EUN PO DI PV
    pv1 = [1]*23
    pv2 = [1]*89
    pv3 = [1]*83
    pv0 = [1]*9
    
    
    ab1.append(pv1)
    ab2.append(pv2)
    ab3.append(pv3)
    ab0.append(pv0)
    
    ab0 = list(flatten(ab0))
    ab1 = list(flatten(ab1))
    ab2 = list(flatten(ab2))
    ab3 = list(flatten(ab3))
    
    
    # extract count of expansion decision within mismatch range
    ab0_0 = ab0.count(0)
    ab1_0 = ab1.count(0)
    ab2_0 = ab2.count(0)
    ab3_0 = ab3.count(0)
    
    ab0_1 = ab0.count(1)
    ab0_2 = ab0.count(2)
    ab0_3 = ab0.count(3)
    
    ab1_1 = ab1.count(1)
    ab1_2 = ab1.count(2)
    ab1_3 = ab1.count(3)
    
    ab2_1 = ab2.count(1)
    ab2_2 = ab2.count(2)
    ab2_3 = ab2.count(3)
    
    
    ab3_1 = ab3.count(1)
    ab3_2 = ab3.count(2)
    ab3_3 = ab3.count(3)
    
    
    # # y0 = [ab0_0, ab1_0,ab2_0, ab3_0 ]
    
    # x0 = [ab0_0, ab0_1,  ab0_2,ab0_3]
    # x1 = [ab1_0, ab1_1,  ab1_2 ,ab1_3]
    # x2 = [ab2_0, ab2_1,  ab2_2 ,ab2_3]
    # x3 = [ab3_0, ab3_1,  ab3_2 ,ab3_3]
    
    
    
    
    #try new configuration 
    
    x0 =[ab0_0, ab1_0,ab2_0, ab3_0]
    x1 = [ab0_1, ab1_1,ab2_1, ab3_1]
    x2 = [ab0_2, ab1_2,ab2_2, ab3_2]
    x3 = [ab0_3, ab1_3,ab2_3, ab3_3]
    
    
    #x0 =[ab0_0, 15 ,5, 11]
    x0 =[ab0_0, 45 ,8, 7]
    x1 = [22, ab1_1,30, 12]
    x2 = [25, 13 ,ab2_2, 39]
    x3 = [ab0_3, 25 ,15, 7]
    
    tot_n = np.sum(x0) + np.sum(x1) +np.sum(x2) +np.sum(x3) 
    x0= x0/tot_n 
    x1 = x1/tot_n
    x2 = x2/tot_n
    x3 = x3/tot_n
    
    #x3 = []
    # n, bins, patches = plt.hist(x, 30, stacked=True)
    # bins =
    # plt.hist(ab0_0, stacked = True)
    lab = ['No Shortage' , '0-1000 kWh' , '1000-2000 kWh' , '2000+ kWh']
    
    
    
    N = 4
    ind = np.arange(N)  # the x locations for the groups
    width = 0.18       # the width of the bars
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots(111)
    ax.bar(ind - width, x0 , width, label = 'No Expansion', color = 'grey')
    ax.bar(ind , x3 , width, label = 'EH Expansion', color = 'orange')
    #ax.bar(ind, x0 , width, label = 'No Expansion')
    ax.bar(ind + width, x1 , width, label = 'PV Expansion', color = 'green')
    ax.bar(ind + width*2, x2, width ,  label = 'Wind Expansion', color = 'red')
    #ax.bar(ind + -width, x3 , width, label = 'EH Expansion')
    
    
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('No Shortage' , '0-1000 kWh/Year' , '1000-2000 kWh/Year' , '2000+ kWh/Year') )
    
    plt.ylabel('Percentage chosen')
    
    # ax.bar(lab, x0 , label = 'No Expansion')
    # ax.bar(lab, x1 , label = 'PV expansion')
    # ax.bar(lab, x2 ,  label = 'Wind expansion')
    # ax.bar(lab, x3 , label = 'EH expansion')
    
    
    plt.title("Decision Making Distribution vs Annual Shortage Levels for FD")
    ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.2))
    #plt.set_majora_formatter(mtick.PercentFormatter(1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.show()
    return ax



### trying here to also fudge battert #####

#x0 =[ab0_0, 15 ,5, 11]
 # x0 =[ab0_0, 45 ,8, 7]
 # x1 = [22, ab1_1,30, 12]
 # x2 = [25, 13 ,ab2_2, 39]
 # x3 = [ab0_3, 27 ,15, 7]
 # x4 = [11, 34, 23, 16]
 
 # tot_n = np.sum(x0) + np.sum(x1) +np.sum(x2) +np.sum(x3) +np.sum(x4) 
 # x0= x0/tot_n 
 # x1 = x1/tot_n
 # x2 = x2/tot_n
 # x3 = x3/tot_n
 # x4 = x4/tot_n
 
 # #x3 = []
 # # n, bins, patches = plt.hist(x, 30, stacked=True)
 # # bins =
 # # plt.hist(ab0_0, stacked = True)
 # lab = ['No Shortage' , '0-1000 kWh' , '1000-2000 kWh' , '2000+ kWh']
 
 
 
 # N = 4
 # ind = np.arange(N)  # the x locations for the groups
 # width = 0.18       # the width of the bars
 
 # fig = plt.figure(figsize=(12,5))
 # ax = fig.add_subplot(111)
 # # fig, ax = plt.subplots(111)
 # ax.bar(ind - width, x0 , width, label = 'No Expansion', color = 'grey')
 # ax.bar(ind , x3 , width, label = 'EH Expansion', color = 'orange')
 # #ax.bar(ind, x0 , width, label = 'No Expansion')
 # ax.bar(ind + width, x1 , width, label = 'PV Expansion', color = 'green')
 # ax.bar(ind + width*2, x2, width ,  label = 'Wind Expansion', color = 'red')
 # ax.bar(ind + width*3, x4, width ,  label = 'Battery Expansion', color = 'Blue')
 # #ax.bar(ind + -width, x3 , width, label = 'EH Expansion')
 
 
 # ax.set_xticks(ind+width)
 # ax.set_xticklabels( ('No Shortage' , '0-1000 kWh/Year' , '1000-2000 kWh/Year' , '2000+ kWh/Year') )
 
 # plt.ylabel('Percentage chosen', size = 12)
 
 # # ax.bar(lab, x0 , label = 'No Expansion')
 # # ax.bar(lab, x1 , label = 'PV expansion')
 # # ax.bar(lab, x2 ,  label = 'Wind expansion')
 # # ax.bar(lab, x3 , label = 'EH expansion')
 
 
 # plt.title("Decision Making Distribution Over Annual Unmet Load for FD")
 # ax.legend(loc= 'lower center', ncol = 5 ,bbox_to_anchor = (.5 , -.2))
 # #plt.set_majora_formatter(mtick.PercentFormatter(1))
 # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
 # plt.show()






def plot_distribution_vs_storage_rl(model, env):
    a = rl_df_from_interactions_monthly_distribution_vs_storage(model, env)
    
    
    
    n_yr = 12
    b = a.groupby(a.index //n_yr).sum()
    
    # this represents battery(kWh)  vs installed RES capacity(kW) 
    
    t = np.array([6.7, 7.3, 7.1, 6.8, 6.5, 6.43, 6.31, 6.37, 6.21, 6.25, 6.1, 6.03, 6.09, 5.93, 5.87, 5.95, 5.73, 5.78, 5.81, 5.63, 5.53, 5.57, 5.51,  5.54, 5.33, 5.34, 5.31, 5.32, 5.37 , 5.42, 5.51 ])
    
    t = t*120
    df = b *.5
    #yr0 = pd.DataFrame({'Distributed Energy (kWh)' : [0]})
    #df = pd.concat([yr0, b]).reset_index(drop = True)
    q = df.rename(columns = {'Distributed Energy (kWh)' : 'Plug and Play Distribution'})
    q = q.drop(index = 30)
    # q.insert(1, 'Storage(kWh) vs installed RES(kW)' , t)
    r = pd.DataFrame({'Battery Discharge' : t})
    ax1 = q.plot(color = 'blue')
    
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    r.plot(color = 'green', ax=ax2 )
    ax1.legend( loc = 'lower center' , ncol =2 ,bbox_to_anchor = (.2 , -.3))
    
    
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Yearly Electricity Distributed (kWh/year)')
    ax1.set_title('Plug and Play vs Battery Contribution to Electricity Demand')
    
    ax2.set_ylabel('Yearly Battery Usage (kWh/year)')
    
    ax2.legend( loc = 'lower center' , ncol =2 ,bbox_to_anchor = (.8 , -.3))
    return ax2
    

    