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
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv 
from stable_baselines import PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow as tf
from stable_baselines import DQN, A2C , ACKTR, ACER,  PPO1 , TRPO , PPO2
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
from RL_SB_helper_wte import *
from custom_policy_networks_wte import *
#import gym_wte_full 
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import EvalCallback , StopTrainingOnRewardThreshold


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


callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=5e7, verbose=1)
eval_callback = EvalCallback(test_env, callback_on_new_best=callback_on_best,
                             best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)




# model_dqn_mlp_small_norm = DQN(CustomDQN_MLP_Policy_small_norm, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_3_flex_dr_comp/")

# model_dqn_mlp_small_norm.learn(total_timesteps = 500000, callback = eval_callback)

if __name__ == '__main__':
    env_id = "gym_wte_full:wte-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = make_vec_env(env_id, num_cpu, seed =0)


    
    
    model_a2c_wte_vec = A2C(Custom_FF_Policy_small,   env, gamma =1,   tensorboard_log="./wte_jmd_3_flex_dr_comp/")
    model_akctr_wte_vec= ACKTR(Custom_FF_Policy_small, env, gamma =1, n_steps = 16,  tensorboard_log="./wte_jmd_3_flex_dr_comp/")
    model_ppo2_wte_vec = PPO2(Custom_FF_Policy_small, env, gamma =1,  tensorboard_log="./wte_jmd_3_flex_dr_comp/")
    model_acer_wte_vec = PPO2(Mlp_a2c, env, gamma =1,  tensorboard_log="./wte_jmd_3_flex_dr_vcomp/")


    
    model_a2c_wte_vec.learn(total_timesteps = 500000, callback = eval_callback)
    model_akctr_wte_vec.learn(total_timesteps = 500000, callback = eval_callback)
    model_ppo2_wte_vec.learn(total_timesteps = 500000, callback = eval_callback)
    model_acer_wte_vec.learn(total_timesteps = 500000, callback = eval_callback)

# #%% Access succesful model callback
# model_wte_akctr_cb = ACKTR.load("AKCTR_model_GREAT" , env = env) # this one mostly does 3 floors, higher ENPV but risky


agent_test_env(model_akctr_wte_vec, test_env)

# n_test_episodes = 1000

print("ENPV for AKCTR model is" , ENPVs_RL(1000, model_akctr_wte_vec, test_env))

# # CDF_RL_simple(n_test_episodes, model_wte_akctr_cb, test_env )

# # agent_test_env(model_wte_dqn_cb, train_env)
# # print("ENPV for callbackdqn" , ENPVs_RL(n_test_episodes, model_wte_dqn_cb, train_env))

#model_akctr_wte_vec.save("akctr_act4_goodperf")



#%% Access succesful model callback
# model_wte_a2ctr_cb = ACKTR.load("best_model" , env = env) # this one mostly does 3 floors, higher ENPV but risky
# agent_test_env(model_wte_a2ctr_cb, test_env)
# print("ENPV for callbackdqn" , ENPVs_RL(n_test_episodes, model_wte_dqn_cb, test_env))

# model_wte_a2ctr_cb .save("model_wte_a2ctr_cb _act4_goodperf")

# #%% Produce CDF comparing RL and and other solutions
# from openpyxl import load_workbook
# import os
# wb = load_workbook(filename = r'C:\Users\cesa_\WTE_comp_xl.xlsx')


# df1 = pd.read_excel(
#      os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
#      engine='openpyxl', sheet_name='Simulation',
# )


# df1.head()

# centralized_rigid_npvs = df1["centralizedR" ]#, "DecentralizedF" , "CentralizedF"]
# centralized_flex_npvs = df1["CentralizedF" ]
# decentralized_flex_npvs = df1["DecentralizedF" ]



# centralized_rigid_npvs.head()

# centralized_flex_npvs.head()

# decentralized_flex_npvs.head()





# CDF_RL_comparison_wte(n_test_episodes, model_wte_akctr_cb, test_env, centralized_rigid_npvs, centralized_flex_npvs , decentralized_flex_npvs)
 

# #%% Look at agent actions

# actions_list , states_list = RL_history_2(test_env, model_wte_akctr_cb, n_test_episodes)






#%% save successful models
# model_trpo_wte_vec.save("AKCTR_model_GREAT")


# agent_test_env(model_trpo_wte_vec, test_env)

# n_test_episodes = 1000

# print("ENPV for AKCTR model is" , ENPVs_RL(n_test_episodes, model_trpo_wte_vec, test_env))



#%% Proceed with more training for successful models

# model_trpo_wte_vec.learn(total_timesteps = 500000, callback = eval_callback)


# agent_test_env(model_trpo_wte_vec, test_env)

# print("ENPV for AKCTR model is" , ENPVs_RL(n_test_episodes, model_trpo_wte_vec, test_env))

# # 
# train_env = gym.make('gym_wte_full:wte-v0')
# test_env = gym.make('gym_wte_full:wte-v1')

# # create different versions of the agent
# model_dqn_mlp = DQN(MlpPolicy, train_env, gamma =1,  verbose=0,exploration_fraction=0.7, exploration_final_eps=0.01, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_mlp_default = DQN(MlpPolicy, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_ff_small = DQN(CustomDQN_FF_Policy_small, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_ff_df = DQN(DQN_FF_Policy, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_ff_norm = DQN(CustomDQN_FF_Policy_norm, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_mlp_small = DQN(CustomDQN_MLP_Policy_small, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_mlp_small_norm = DQN(CustomDQN_MLP_Policy_small_norm, train_env, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/")
# model_dqn_low_exp = DQN(MlpPolicy, train_env, gamma =1,  verbose=0,exploration_fraction=0.3, exploration_final_eps=0.1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_2_flex_dr_comp/" )

# # train the different agents on 50k timesteps
# # train models to obtain performance comparison
# model_dqn_mlp.learn(total_timesteps = 50000)
# model_dqn_mlp_default.learn(total_timesteps = 50000)
# model_dqn_ff_small.learn(total_timesteps = 50000)
# model_dqn_ff_df.learn(total_timesteps = 50000)
# model_dqn_ff_norm.learn(total_timesteps = 50000)
# model_dqn_mlp_small.learn(total_timesteps = 50000)
# model_dqn_mlp_small_norm.learn(total_timesteps = 50000)
# model_dqn_low_exp.learn(total_timesteps = 50000)


# n_test_episodes = 500

# print("ENPV for exploration adjusted MLP DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp, test_env))
# print("ENPV for default MLP DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp_default, test_env))
# print("ENPV for small FF network DQN" , ENPVs_RL(n_test_episodes, model_dqn_ff_small, test_env))
# print("ENPV for default FF network DQN" , ENPVs_RL(n_test_episodes, model_dqn_ff_df, test_env))
# print("ENPV for norm FF network DQN" , ENPVs_RL(n_test_episodes, model_dqn_ff_norm, test_env))
# print("ENPV for small MLP network DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp_small, test_env))
# print("ENPV for small NORM MLP network DQN" , ENPVs_RL(n_test_episodes, model_dqn_mlp_small_norm, test_env))
# print("ENPV for small exploration MLP DQN" , ENPVs_RL(n_test_episodes, model_dqn_low_exp, test_env))


# agent_test_env(model_dqn_mlp, test_env)
# agent_test_env(model_dqn_mlp_default, test_env)
# agent_test_env(model_dqn_mlp_small, test_env)
# agent_test_env(model_dqn_ff_small, test_env)
# agent_test_env(model_dqn_ff_norm, test_env)
# agent_test_env(model_dqn_mlp_small_norm, test_env)
