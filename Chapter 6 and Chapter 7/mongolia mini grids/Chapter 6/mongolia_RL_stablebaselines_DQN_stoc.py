# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:02:16 2020

@author: cesa_
"""
from matplotlib import pyplot as plt
import gym

#this is a loop to delete registered environments due to a stable baselines bug
# must be placed before importing mongolia minigrids module as when that is imported, environments are automatically registered again
env_dict = gym.envs.registration.registry.env_specs.copy()
env_name0 = 'mongolia_minigrid-v0'
env_name = 'mongolia_minigrid-v1'
env_name2 = 'mongolia_minigrid-v2'
env_name3 = 'mongolia_minigrid-v3'
env_name4 = 'mongolia_minigrid-v4'
env_name5 = 'mongolia_minigrid-v5'
env_name6 = 'mongolia_minigrid-v6'
env_name7 = 'mongolia_minigrid-v7'
env_name8 = 'mongolia_minigrid-v8'
env_name9 = 'mongolia_minigrid-v9'
env_name10 = 'mongolia_minigrid-v10'
env_name11 = 'mongolia_minigrid-v11'
env_name12= 'mongolia_minigrid-v12'
env_name13 = 'mongolia_minigrid-v13'



if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
if env_name0 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name0]
if env_name2 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name2]
if env_name3 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name3]
if env_name4 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name4]
if env_name5 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name5]
if env_name6 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name6]
if env_name7 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name7]
if env_name8 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name8]
if env_name9 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name9]
if env_name10 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name10]
if env_name11 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name11]
if env_name12 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name12]
if env_name13 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name13]



    
#import mongolia_minigrids


import mongolia_minigrids
import os
import pandas as pd
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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from stable_baselines import DQN, A2C , ACKTR, ACER, PPO1
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
from RL_SB_helper_mongolia import *


#mismatch cost defined here for environment instantiation after
mismatch_cost = .3 # $/KwH

sess = tf.Session()

train_env_yearly = gym.make('mongolia_minigrid-v0', mismatch_cost = mismatch_cost )
train_env_monthly = gym.make('mongolia_minigrid-v1', mismatch_cost = mismatch_cost )
train_env_monthly_dict = gym.make('mongolia_minigrid-v3', mismatch_cost = mismatch_cost )


train_env_s1_mask = gym.make('mongolia_minigrid-v7', mismatch_cost = mismatch_cost )
train_env_s1_repl = gym.make('mongolia_minigrid-v8', mismatch_cost = mismatch_cost )


train_env_monthly = train_env_s1_repl


# model_dqn_lr_very_small_m = DQN(MlpPolicy, train_env_monthly, gamma =1, exploration_fraction = .3, learning_rate = 5e-6,  exploration_final_eps = .001, 
#                  prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_nov21/")
# model_dqn_lr_standard_m = DQN(MlpPolicy, train_env_monthly, gamma =1, prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_nov21/")
# model_dqn_lr_small = DQN(MlpPolicy, train_env_monthly, gamma =1, exploration_fraction = .3, learning_rate = 5e-5,  exploration_final_eps = .001, 
#                 prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_nov21/")
# model_dqn_lr_med = DQN(MlpPolicy, train_env_monthly, gamma =1, exploration_fraction = .3, learning_rate = 5e-4,  exploration_final_eps = .001, 
#                 prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_nov21/")
# model_dqn_lr_high = DQN(MlpPolicy, train_env_monthly, gamma =1, exploration_fraction = .3, learning_rate = 5e-3,  exploration_final_eps = .001, 
#                 prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_nov21/")




# model_dqn_lr_very_small_m.learn(total_timesteps = 100000)
# model_dqn_lr_standard_m.learn(total_timesteps = 100000)



# model_dqn_lr_med.learn(total_timesteps = 100000)
# model_dqn_lr_high.learn(total_timesteps = 100000)


# model_acer = ACER(Mlp_a2c, train_env_monthly, gamma =1, verbose = 1)

# model_acer.learn(total_timesteps = 100000)


#### PPO######

model_ppo = PPO1(Mlp_a2c, train_env_monthly, gamma =1, verbose = 1,  tensorboard_log="./mongolia_minigrids_ppo_nov15/")
model_ppo.learn(total_timesteps = 1000000)



agent_test(model_acer)

agent_test(model_dqn_lr_very_small_m)
# agent_test(model_dqn_lr_standard_m)



# test agent for debugging purposes

# n_steps = 240 # look at behabviour during first year only , so first 12 timesteps
# #HERE DEFINE TEST ENVIRONMENT WITH NO NEGATIVE REWARD PENALTIES, NOT CREATED YET
# #test env = 

# agent_test_env_nsteps(model_dqn_lr_very_small_m,train_env_monthly, n_steps )
# agent_test_env_nsteps(model_dqn_lr_standard_m,train_env_monthly, n_steps )


# # agent_test(model_dqn_lr_med)
# # agent_test(model_dqn_lr_high)

# #evaluate and compare trained models:
evaluate(model_dqn_lr_very_small_m , num_episodes = 100)
evaluate(model_dqn_lr_small, num_episodes = 1000)
evaluate(model_dqn_lr_med, num_episodes = 1000)
evaluate(model_dqn_lr_high, num_episodes = 1000)
evaluate(model_acer, num_episodes = 1000)
evaluate(model_ppo, num_episodes = 1000)

# model_dqn_lr_very_small.save("mongolia_minigrids_dqn_verysmallr_good")
# model_dqn_lr_small.save("mongolia_minigrids_dqn_smallr_good")

# model_dqn_lr_small_solid = DQN.load("mongolia_minigrids_dqn_smallr_good", env = train_env)
# model_dqn_lr_very_small_solid = DQN.load("mongolia_minigrids_dqn_verysmallr_good", env = train_env)

# evaluate(model_dqn_lr_small_solid , num_episodes = 1000)
# evaluate(model_dqn_lr_very_small_solid, num_episodes = 1000)


agent_test(model_dqn_lr_small)

agent_test(model_ppo)

# evaluate_shortage_rl(model_dqn_lr_small_solid , num_episodes = 1000)
# evaluate_shortage_rl(model_dqn_lr_very_small_solid, num_episodes = 1000)