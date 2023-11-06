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
# from wte_env_flex_dr_debug import WTE_EnvFull_debug
import joblib

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)
#print(env_ids)
sess = tf.Session()
#delete if it's registered
env_name0 = 'wte-v1'
env_name = 'wte-v0'
env_name2 = 'wte-v2'
env_name3 = 'wte-v3'
env_name4 = 'wte-v4'
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


def make_env():
    def maker():
        env = gym.make("gym_wte_full:wte-v0")
        return env
    return maker
    
train_env = gym.make('gym_wte_full:wte-v4')
test_env = gym.make('gym_wte_full:wte-v1')
debug_env = gym.make('gym_wte_full:wte-v3')

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=6e7, verbose=1)
# eval_callback_dqn = EvalCallback(test_env, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/dqn/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=True, render=False)

eval_callback_akctr = EvalCallback(test_env, callback_on_new_best=callback_on_best,
                              best_model_save_path='./logs/akctr/',
                              log_path='./logs/', eval_freq=2000,
                              deterministic=True, render=False)

# eval_callback_acer = EvalCallback(test_env, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/acer/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=True, render=False)

eval_callback_trpo = EvalCallback(debug_env, callback_on_new_best=callback_on_best,
                              best_model_save_path='./logs/trpo/lowflex/',
                              log_path='./logs/', eval_freq=2000,
                              deterministic=True, render=False)




study = joblib.load("trpo_wte_study_lowflex_1.pkl")
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

trial = study.best_trial
print("Best hyperparameters: {}".format(trial.params))
# train the agent with optimized hyperparameters
#optuna_policy_kwargs = trial.params
opt_tb = trial.params['timesteps_per_batch']
opt_kl = trial.params['max_kl']
opt_lam = trial.params['lam']
opt_cg = trial.params['cg_iters']
opt_evfs = trial.params['vf_stepsize']
opt_vfi = trial.params['vf_iters']






#model_trpo_wte_df_t0_exp = TRPO(Mlp_a2c ,debug_env, gamma =1,  tensorboard_log="./wte_jmd_FULL/" )
#model_trpo_wte_small_t0_200_capmax_optuna = TRPO(Mlp_a2c ,train_env, gamma =1, timesteps_per_batch = opt_tb, max_kl = opt_kl, lam = opt_lam  , cg_iters = opt_cg,
                                          vf_stepsize = opt_evfs, vf_iters = opt_vfi , tensorboard_log="./wte_jmd_optuna_lowflex/" )


model_trpo_wte_small_t0_200_capmax_DF = TRPO(Mlp_a2c ,train_env, gamma =1,   tensorboard_log="./wte_jmd_optuna_lowflex/" )

# model_trpo_wte_df_t0_exp.learn(total_timesteps = 500000, callback = eval_callback_trpo)
model_trpo_wte_small_t0_200_capmax_optuna.learn(total_timesteps = 1000000, callback = eval_callback_trpo)
model_trpo_wte_small_t0_200_capmax_DF.learn(total_timesteps = 500000)


if __name__ == '__main__':
    env_id = "gym_wte_full:wte-v4"
    num_cpu = 4  # Number of processes to use
    model_akctr_wte_vec_ff_capmax= ACKTR(Mlp_a2c, env, gamma =1,  tensorboard_log="./wte_jmd_optuna_lowflex/") 
    model_akctr_wte_vec_ff_capmax.learn(total_timesteps = 1000000 , callback = eval_callback_akctr)









#model_trpo_wte_df_t0_200_exp.save("wte-trpo-t0-better1")


# model = TRPO.load("wte-trpo-t0-vgood1")
# model2 = TRPO.load("wte-trpo-t0-correct-ok")



# agent_test_env_debug(model2, debug_env)


# actions_list1, s = RL_history_1_det(debug_env, model, 1000)
# actions_list2, s2 =  RL_history_2(debug_env, model2, 1000)
# actions_list3, s3 =  RL_history_1_det(debug_env, model2, 1000)
# actions3 = np.repeat([18] ,3000).tolist()
# actions4 = np.repeat([15] ,2000).tolist()
# actions_plot_list = actions_list1 + actions_list2 + actions3 + actions4 + actions_list3


# RL_action_hist_from_list_highflex(actions_plot_list)
# # p



# #print("ENPV for TRPO DF high flex model small nn is" , ENPVs_RL(2000, model_trpo_wte_small_t0, debug_env))
# # # print("ENPV for TRPO small MLP t0 model is" , ENPVs_RL_stoch(2000, model_trpo_wte_small_t0, debug_env))


# model_trpo_wte_df = TRPO.load("wte-trpo-t0-good")
# model_trpo_wte_df_t0 = TRPO.load("wte-trpo-t0-better1")




# print("ENPV for TRPO DF high flex model 1 is" , ENPVs_RL(2000, model_trpo_wte_df, debug_env))
# print("ENPV for TRPO DF high flex model 2 is" , ENPVs_RL(2000, model_trpo_wte_df_t0, debug_env))




# agent_test_env(model_trpo_wte_df_t0, debug_env)

#test_observation = np.array([200, 52, 0, 115, 0 , 80, 0, 67, 0, 400, 0, 58 , 11])

#t1= 
#check_rl_model_dr_prob(model_trpo_wte_df_t0_200_exp, test_observation)



# model_trpo_wte_df = TRPO(Mlp_a2c ,train_env, gamma =1,   tensorboard_log="./wte_jmd_4_flex_dr_comp/" )




#actions_list , states_list = RL_history_2(test_env, model_trpo_1, 10)



#model_wte_akctr_cb_db = ACKTR.load("AKCTR_model_GREAT" , env = env) 

# agent_test_env_debug(model_wte_akctr_cb_db, test_env)

#agent_test_env(model_dqn_mlp_small_norm, test_env)
# agent_test_env(model_dqn_mlp_small_norm, test_env)
#agent_test_env_debug(model_a2c_wte_vec, test_env)
#agent_test_env(model_a2c_wte_vec, test_env)
#agent_test_env_debug(model_trpo_wte, test_env)


#print("ENPV for TRPO model is" , ENPVs_RL_stoch(2000, model_trpo_wte_df, test_env))
# print("ENPV for dqn callback" , ENPVs_RL(1000, model2, test_env))

#model_akctr_wte_vec_lstm.save("wte_akctr_lstm")
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

#%% Produce CDF comparing RL and and other solutions
from openpyxl import load_workbook
import os
wb = load_workbook(filename = r'C:\Users\cesa_\WTE_comp_xl.xlsx')


df1 = pd.read_excel(
      os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
      engine='openpyxl', sheet_name='Capev',
)


df1.head()

s_1_demand_df = df1["Demand1" ]
s_2_demand_df = df1["Demand2" ]
s_3_demand_df = df1["Demand3" ]
s_4_demand_df = df1["Demand4" ]
s_5_demand_df = df1["Demand5" ]
s_6_demand_df = df1["Demand6" ]


demand_df = np.array([s_1_demand_df , s_2_demand_df , s_3_demand_df , s_4_demand_df , s_5_demand_df , s_6_demand_df])


centralized_rigid_npvs.head()

# centralized_flex_npvs.head()

# decentralized_flex_npvs.head()

# n_training_envs = 3
# envs = DummyVecEnv([make_env() for _ in range(n_training_envs)])

# model_akctr_wte_vec_lstm = ACKTR.load("wte_akctr_lstm" , env = envs)

# n_test_episodes = 2000

# CDF_RL_comparison_wte_2rl(n_test_episodes, model_trpo_wte_df, model_trpo_wte_df_t0, test_env, debug_env, centralized_rigid_npvs, centralized_flex_npvs , decentralized_flex_npvs)
 

# CDF_RL_comparison_wte_2rl_1lstm(n_test_episodes, model_akctr_wte_vec_lstm, model_trpo_wte_df_t0,  debug_env, centralized_rigid_npvs, centralized_flex_npvs , decentralized_flex_npvs)




# print("Low flex model ENPV" , ENPVs_RL_lstm(2000,model_akctr_wte_vec_lstm ))



#%% Look at agent actions

#actions_list , states_list = RL_history_lstm( model_akctr_wte_vec_lstm, 10)






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
