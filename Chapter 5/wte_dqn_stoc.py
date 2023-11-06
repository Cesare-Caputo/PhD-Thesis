# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:10:18 2021

@author: cesa_
"""

from matplotlib import pyplot as plt
import gym
from gym import envs
#import gymgarage
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
import tensorflow as tf
from stable_baselines import DQN, A2C , ACKTR, ACER
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
from RL_SB_helper_wte import *
from custom_policy_networks_wte import *
#import gym_wte_full 
import joblib
from stable_baselines.common.callbacks import EvalCallback

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)
sess = tf.Session()
#delete if it's registered
env_name0 = 'wte-v1'
env_name = 'wte-v0'
env_name2 = 'wte-v2'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
if env_name0 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name0]
if env_name2 in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name2]

train_env = gym.make('gym_wte_full:wte-v0')
test_env = gym.make('gym_wte_full:wte-v1')

#specifcy callbacl to make sure best model is not missed
# Use deterministic actions for evaluation
eval_callback = EvalCallback(test_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)



# create different versions of the agent
model_dqn_mlp = DQN(MlpPolicy, train_env, gamma =1,  verbose=0,exploration_fraction=0.7, exploration_final_eps=0.01, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_mlp_default = DQN(MlpPolicy, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_ff_small = DQN(CustomDQN_FF_Policy_small, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_ff_df = DQN(DQN_FF_Policy, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_ff_norm = DQN(CustomDQN_FF_Policy_norm, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_mlp_small = DQN(CustomDQN_MLP_Policy_small, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_mlp_small_norm = DQN(CustomDQN_MLP_Policy_small_norm, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
model_dqn_low_exp = DQN(MlpPolicy, train_env, gamma =1,  verbose=0,exploration_fraction=0.3, exploration_final_eps=0.1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/" )


#try two optuna models at same time too for proper benchmarking
study1 = joblib.load("dqn_wte_study_1.pkl")
# study2 = joblib.load("dqn_garage_study_3.pkl")
# study3 = joblib.load("dqn_garage_study_2.pkl")


trial1 = study1.best_trial


# try with optuna hyperparameters too
model_dqn_optuna1 = DQN(MlpPolicy, train_env, gamma =1, exploration_fraction =trial1.params['exploration_fraction'] , exploration_final_eps = trial1.params['exploration_final_eps'] , 
                       exploration_initial_eps = trial1.params['exploration_initial_eps'] , learning_rate = trial1.params['learning_rate'] , learning_starts =  trial1.params['learning_starts'] , 
                       verbose=0, prioritized_replay=True , 
                       tensorboard_log="./wte_jmd_2_flex_dr_comp/")




# train the different agents on 50k timesteps
# train models to obtain performance comparison
model_dqn_mlp.learn(total_timesteps = 50000 , callback = eval_callback )
model_dqn_mlp_default.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_ff_small.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_ff_df.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_ff_norm.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_mlp_small.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_mlp_small_norm.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_low_exp.learn(total_timesteps = 50000, callback = eval_callback)
model_dqn_optuna1.learn(total_timesteps = 50000, callback = eval_callback)





n_test_episodes = 500

print("ENPV for exploration adjusted MLP DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp, test_env))
print("ENPV for default MLP DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp_default, test_env))
print("ENPV for small FF network DQN" , ENPVs_RL(n_test_episodes, model_dqn_ff_small, test_env))
print("ENPV for default FF network DQN" , ENPVs_RL(n_test_episodes, model_dqn_ff_df, test_env))
print("ENPV for norm FF network DQN" , ENPVs_RL(n_test_episodes, model_dqn_ff_norm, test_env))
print("ENPV for small MLP network DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp_small, test_env))
print("ENPV for small NORM MLP network DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp_small_norm, test_env))
print("ENPV for small exploration MLP DQN" , ENPVs_RL(n_test_episodes, model_dqn_low_exp, test_env))
print("ENPV for optuna DQN" , ENPVs_RL(n_test_episodes, model_dqn_optuna1, test_env))

agent_test_env(model_dqn_mlp, test_env)
agent_test_env(model_dqn_mlp_default, test_env)
agent_test_env(model_dqn_mlp_small, test_env)
agent_test_env(model_dqn_ff_small, test_env)
agent_test_env(model_dqn_ff_norm, test_env)
agent_test_env(model_dqn_mlp_small_norm, test_env)


#%% save successful models

model_dqn_mlp_small_norm.save("wte-decent-2")
# model_dqn_ff_small.save("nn_arch_test_mlp_default")
# model_dqn_mlp_small.save("nn_arch_test_mlp_default_small")
# model_dqn_optuna2.save("nn_arch_test_mlp_optuna_solid")

#%% Continue learning only on succesful models
model_dqn_mlp_small_norm.learn(total_timesteps = 100000, callback = eval_callback)
model_dqn_ff_norm.learn(total_timesteps = 100000, callback = eval_callback)

#%% Access succesful model callback
model_wte_dqn_cb = DQN.load("best_model" , env = test_env) # this one mostly does 3 floors, higher ENPV but risky

agent_test_env(model_wte_dqn_cb, train_env)
print("ENPV for callbackdqn" , ENPVs_RL(n_test_episodes, model_wte_dqn_cb, train_env))
