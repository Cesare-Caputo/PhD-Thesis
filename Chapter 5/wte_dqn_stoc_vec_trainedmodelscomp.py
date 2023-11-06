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
import gym_wte_full 
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import EvalCallback , StopTrainingOnRewardThreshold
# from wte_env_flex_dr_debug import WTE_EnvFull_debug

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)
#print(env_ids)
sess = tf.Session()
#delete if it's registered
# env_name0 = 'wte-v1'
# env_name = 'wte-v0'
# env_name2 = 'wte-v2'
# env_name3 = 'wte-v3'
# env_name4 = 'wte-v4'
# if env_name in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name]
# if env_name0 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name0]
# if env_name2 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name2]
# if env_name3 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name3]
# if env_name4 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name4]


def make_env():
   def maker():
       env = gym.make("gym_wte_full:wte-v1")
       return env
   return maker
    
train_env = gym.make('gym_wte_full:wte-v0')
test_env = gym.make('gym_wte_full:wte-v1')
debug_env = gym.make('gym_wte_full:wte-v3')
train_env_capmax = gym.make('gym_wte_full:wte-v4')

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=6e7, verbose=1)
eval_callback_dqn = EvalCallback(test_env, callback_on_new_best=callback_on_best,
                             best_model_save_path='./logs/dqn/',
                             log_path='./logs/', eval_freq=2000,
                             deterministic=True, render=False)

eval_callback_akctr = EvalCallback(test_env, callback_on_new_best=callback_on_best,
                             best_model_save_path='./logs/akctr/',
                             log_path='./logs/', eval_freq=2000,
                             deterministic=True, render=False)

eval_callback_acer = EvalCallback(test_env, callback_on_new_best=callback_on_best,
                             best_model_save_path='./logs/acer/',
                             log_path='./logs/', eval_freq=2000,
                             deterministic=True, render=False)

eval_callback_trpo = EvalCallback(test_env, callback_on_new_best=callback_on_best,
                             best_model_save_path='./logs/trpo/',
                             log_path='./logs/', eval_freq=2000,
                             deterministic=False, render=False)



# Create one env for testing
# test_env_lstm = DummyVecEnv([make_env() for _ in range(1)])
# test_obs = test_env_lstm.reset()
model_trpo_wte_df_highflex = TRPO(Mlp_a2c ,debug_env, gamma =1,   tensorboard_log="./wte_jmd_capmax_highflex/" )
#model_trpo_wte_df_highflex = TRPO(Mlp_a2c ,debug_env, gamma =1,   tensorboard_log="./wte_jmd_capmax_lowflex/" )
# model_trpo_wte_df.learn(total_timesteps = 500000, callback = eval_callback_trpo)

#model_dqn_mlp_df_highflex = DQN(MlpPolicy, train_env, exploration_fraction = .3, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_capmax_highflex/")
model_trpo_wte_df_highflex.learn(total_timesteps = 100000)
#model_dqn_mlp_df_highflex.learn(total_timesteps = 100000  ,callback = eval_callback_dqn)
#

#
model_trpo_wte_df_highflex.save("trpo_wte_highflex_lowtraining")


#model_trpo_wte_df_highflex.learn(total_timesteps = 1000000)


# if __name__ == '__main__':
#     env_id = "gym_wte_full:wte-v3"
#     num_cpu = 4  # Number of processes to use
#     # Create the vectorized environment
#     env = make_vec_env(env_id, num_cpu, seed =0)
#     model_akctr_highflex = ACKTR(Mlp_a2c, env, gamma =1,  tensorboard_log=  "./wte_jmd_capmax_highflex/" )
#     model_akctr_highflex.learn(total_timesteps = 500000)
# #     #model_ppo2_wte_vec.learn(total_timesteps = 500000, callback = eval_callback_akctr)
# #     #model_acer_wte_vec.learn(total_timesteps = 50000, callback = eval_callback_akctr)



# model_akctr_highflex.save("akctr_wte_hf_ok")
# #%% Access succesful model callback
#model_wte_akctr_cb = ACKTR.load("AKCTR_model_GREAT" , env = env) 
# model_wte_trpo_200_1 = TRPO.load("wte-trpo-t0-200exp_correct_ok")
# model_wte_trpo_200_2 = TRPO.load("wte-trpo-t0-200exp_better1")
# model_wte_trpo_200_3 = c


# print("ENPV for AKCTR capmax model is" , ENPVs_RL(1000, model_akctr_wte_vec_ff_capmax, train_env_capmax) )
# print("ENPV for TRPO capmax model1 is" , ENPVs_RL(1000, model_trpo_wte_df_capmax, train_env_capmax))


# print("ENPV for TRPO model1 is" , ENPVs_RL_stoch(1000, model_trpo_wte_df_capmax, test_env))



# print("ENPV for TRPO model2 is" , ENPVs_RL(1000, model_wte_trpo_200_1, train_env_capmax))
# print("ENPV for TRPO model3 is" , ENPVs_RL(1000, model_wte_trpo_200_1, train_env_capmax))



# model1 = A2C.load("best_model_a2c1" , env = env)
# model2 = DQN.load("best_model_dqn2" )
# model2 = ACKTR.load("best_model" , env = env)
#model3 = A2C.load("best_model_a2c1" , env = env)

#model_wte_trpo_200_3.learn(total_timesteps = 50000)

#model_wte_akctr_cb_db = ACKTR.load("AKCTR_model_GREAT" , env = env) 

agent_test_env_debug(model_akctr_highflex, debug_env)
#agent_test_env(model_trpo_wte_df_capmax, test_env)
# # agent_test_env(model_dqn_mlp_small_norm, test_env)
# agent_test_env(model_wte_trpo_200_3 , test_env)
#agent_test_env(model_a2c_wte_vec, test_env)
#agent_test_env_debug(model_trpo_wte, test_env)

#print("ENPV for TRPO model is" , ENPVs_RL_stoch(2000, model_trpo_wte_df, test_env))
# print("ENPV for dqn callback" , ENPVs_RL(1000, model2, test_env))

#model_trpo_wte.save("wte_trpo_good_df_mlp")
#debug_rl(model_akctr_wte_vec, debug_env, 16)

# n_test_episodes = 1000

#print("ENPV for AKCTR CB model is" , ENPVs_RL(1000, model_wte_akctr_cb_db, test_env) - 24335012)
# print("ENPV for AKCTR model1 is" , ENPVs_RL(1000, model1, test_env))
# print("ENPV for AKCTR model2 is" , ENPVs_RL(1000, model2, test_env))
# print("ENPV for AKCTR model3 is" , ENPVs_RL(1000, model3, test_env))

# # n_test_episodes = 1000

# print("ENPV for AKCTR model is" , ENPVs_RL(1000, model_akctr_wte_vec, test_env))

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
#       os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
#       engine='openpyxl', sheet_name='Simulation',
# )


# df1.head()

# centralized_rigid_npvs = df1["centralizedR" ]#, "DecentralizedF" , "CentralizedF"]
# centralized_flex_npvs = df1["CentralizedF" ]
# decentralized_flex_npvs = df1["DecentralizedF" ]



# centralized_rigid_npvs.head()

# centralized_flex_npvs.head()

# decentralized_flex_npvs.head()





# CDF_RL_comparison_wte_2rl(2000, model_trpo_wte_df_capmax, model_wte_trpo_200_3, test_env, test_env,  centralized_rigid_npvs, centralized_flex_npvs , decentralized_flex_npvs)
 

# # #%% Look at agent actions

# actions_list1 , states_list = RL_history_2(test_env, model_wte_trpo_200_3, 1000)
# actions_list2 , states_list = RL_history_2(test_env, model_trpo_wte_df_capmax, 1000)

# actions_plot_list = actions_list1 + actions_list2



#%% save successful models
# model_trpo_wte_vec.save("AKCTR_model_GREAT")


# agent_test_env(model_trpo_wte_vec, test_env)

# n_test_episodes = 1000

# print("ENPV for AKCTR model is" , ENPVs_RL(n_test_episodes, model_trpo_wte_vec, test_env))



#%% Proceed with more training for successful models


