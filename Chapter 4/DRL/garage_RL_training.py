# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:02:16 2020

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

#Train the agent
model_dqn = DQN(MlpPolicy, train_env, gamma =1,  verbose=0,exploration_fraction=0.3, exploration_final_eps=0.1, prioritized_replay=True ,  tensorboard_log="./garage_IISE_comparison_ccstart_included/")

model_dqn.learn(total_timesteps = 500000)



# Asses decision rules
RL_FT_hist(test_env, model_acer, 1000)
