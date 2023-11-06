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
# if env_name in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name]
# if env_name0 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name0]
# if env_name2 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name2]
# if env_name3 in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name3]

def make_env():
   def maker():
       env = gym.make("gym_wte_full:wte-v0")
       return env
   return maker
    
train_env = gym.make('gym_wte_full:wte-v0')
test_env = gym.make('gym_wte_full:wte-v1')
debug_env = gym.make('gym_wte_full:wte-v3')

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=6e7, verbose=1)
# eval_callback_dqn = EvalCallback(test_env, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/dqn/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=True, render=False)

# eval_callback_akctr = EvalCallback(test_env, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/akctr/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=True, render=False)

# eval_callback_acer = EvalCallback(test_env, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/acer/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=True, render=False)

# eval_callback_trpo = EvalCallback(test_env, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/trpo/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=False, render=False)





n_training_envs = 3
envs = DummyVecEnv([make_env() for _ in range(n_training_envs)])


# Create one env for testing
test_env_lstm = DummyVecEnv([make_env() for _ in range(1)])
test_obs = test_env_lstm.reset()

eval_callback_akctr_lstm = EvalCallback(test_env_lstm, callback_on_new_best=callback_on_best,
                         best_model_save_path='./logs/akctr_lstm/',
                         log_path='./logs/', eval_freq=2000,
                         deterministic=True, render=False)





model_trpo_df = TRPO.load("wte_trpo_good_df_mlp")
model_trpo_1 = TRPO.load("wte_trpo_good1")




model_akctr_wte_vec_lstm= ACKTR.load("wte_akctr_lstm" , envs)












# model.predict(test_obs) would through an error
# because the number of test env is different from the number of training env
# so we need to complete the observation with zeroes
# zero_completed_obs = np.zeros((n_training_envs,) + envs.observation_space.shape)
# zero_completed_obs[0, :] = test_obs



# #model_akctr_wte_vec_lstm.learn(total_timesteps = 500000)

# done = False
# for _ in range(10):
#     state = None
#     action, state = model_akctr_wte_vec_lstm.predict(zero_completed_obs, state=state)
#     # The test env is expecting only one action
#     new_obs, reward, done, info = test_env_lstm.step([action[0]])
#     # Update the obs
#     zero_completed_obs[0, :] = new_obs
#     print(reward)
#     print(action[0])


# model = model_akctr_wte_vec_lstm
# env = envs
# obs = env.reset()
# # Passing state=None to the predict function means
# # it is the initial state
# state = None
# # When using VecEnv, done is a vector
# done = [False for _ in range(env.num_envs)]
# actions = []
# for _ in range(100):
#     # We need to pass the previous state and a mask for recurrent policies
#     # to reset lstm state when a new episode begin
#     print("Step {}".format(_ ))
#     action, state = model.predict(obs, state=state, mask=done)
#     print("Action: ", action)
#     obs, reward , done, _ = env.step(action)
#     actions.append(action)
    #Note: with VecEnv, env.reset() is automatically called

    # Show the env
    #env.render(mode = 'human')

#NPV_RL_lstm( model_akctr_wte_vec_lstm )


# def NPVs_RL_lstm(episodes, model , test_env):
#     env = test_env
#     NPVs =[]
#     n_steps = 16
#     for i in range(episodes):
#         episode_rewards = []
#         done = False
#         obs = env.reset()
#         zero_completed_obs = np.zeros((n_training_envs,) + envs.observation_space.shape)
#         for step in range(n_steps):
#             # model.predict(test_obs) would through an error
#             # because the number of test env is different from the number of training env
#             # so we need to complete the observation with zeroes
#             state = None
#             #zero_completed_obs[0, :] = test_obs
#             # _states are only useful when using LSTM policies
#             action, state = model.predict(zero_completed_obs, state = state, mask = done ,  deterministic = True)
#             # The test env is expecting only one action
#             new_obs, reward, done, info = env.step(action[0])
#             # Update the obs
#             zero_completed_obs[0, :] = new_obs
#             episode_rewards.append(reward)
#             NPV_adj = (sum(episode_rewards)) 
#         NPVs.append(NPV_adj)
#     return NPVs



#




# model_trpo_wte_df = TRPO(Mlp_a2c ,train_env, gamma =1,   tensorboard_log="./wte_jmd_4_flex_dr_comp/" )
# model_trpo_wte_df.learn(total_timesteps = 500000, callback = eval_callback_trpo)

# model_dqn_mlp_small_norm = DQN(CustomDQN_MLP_Policy_small_norm, train_env, exploration_fraction = .7, gamma =1, prioritized_replay=True ,  tensorboard_log="./wte_jmd_4_flex_dr_comp/")
# model_dqn_mlp_small_norm.learn(total_timesteps = 500000, callback = eval_callback_dqn)

if __name__ == '__main__':
    env_id = "gym_wte_full:wte-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = make_vec_env(env_id, num_cpu, seed =0)


    
    
#     #model_a2c_wte_vec = A2C(Mlp_a2c,   env, gamma =1,   tensorboard_log="./wte_jmd_4_flex_dr_comp/")
#     #model_akctr_wte_vec_ff= ACKTR(Custom_FF_Policy_small, env, gamma =1, n_steps = 16,  tensorboard_log="./wte_jmd_4_flex_dr_comp/")
#     model_akctr_wte_vec_lstm= ACKTR(MlpLstmPolicy, env, gamma =1, n_steps = 16,  tensorboard_log="./wte_jmd_4_flex_dr_comp/")
# # #     model_ppo2_wte_vec = PPO2(Custom_FF_Policy_small, env, gamma =1,  tensorboard_log="./wte_jmd_3_flex_dr_comp/")
# #      #model_acer_wte_vec = PPO2(Mlp_a2c, env, gamma =1,  tensorboard_log="./wte_jmd_4_flex_dr_vcomp/")


    
#     model_akctr_wte_vec_lstm.learn(total_timesteps = 500000, callback = eval_callback_akctr_lstm)
    #model_akctr_wte_vec.learn(total_timesteps = 500000, callback = eval_callback_akctr)
#     #model_ppo2_wte_vec.learn(total_timesteps = 500000, callback = eval_callback_akctr)
#     #model_acer_wte_vec.learn(total_timesteps = 50000, callback = eval_callback_akctr)

# #%% Access succesful model callback
#model_wte_akctr_cb = ACKTR.load("AKCTR_model_GREAT" , env = env) 
# model1 = A2C.load("best_model_a2c1" , env = env)
model_dqn1 = DQN.load("best_model_dqn_1" )
model_dqn2 = DQN.load("best_model_dqn2" )
model_a2c_1 = A2C.load("best_model_a2c1" , env = env)
model_akctr_wte1 =ACKTR.load("AKCTR_model_GREAT", env = env)                                
model_akctr_wte2 =ACKTR.load("akctr_act4_goodperf", env = env) 
model_akctr_wte3 = ACKTR.load("best_model", env = env)



# ASSES ENPV OF ALL PRODUCED MODELS

initial_2_floor_cost = 24335012

print("ENPV for AKCTR  model 1 is" , ENPVs_RL(1000, model_akctr_wte1, test_env) )
print("ENPV for AKCTR model2 is" , ENPVs_RL(1000, model_akctr_wte2, test_env))
print("ENPV for AKCTR model3 is" , ENPVs_RL(1000, model_akctr_wte3, test_env))

print("ENPV for TRPO model 1 is" , ENPVs_RL_stoch(1000, model_trpo_1 , test_env))
print("ENPV for TRPo default" , ENPVs_RL_stoch(1000, model_trpo_df, test_env))

print("ENPV for AKCTR LSTMmodel is" , ENPVs_RL_lstm(1000, model_akctr_wte_vec_lstm) )


print("ENPV for dqn 1 is" , ENPVs_RL(1000, model_dqn1, test_env))
print("ENPV for dqn 2 is" , ENPVs_RL(1000, model_dqn2, test_env))



actions_list , states_list = RL_history_2(test_env, model_trpo_1, 10)



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

# model_wte_a2ctr_cb.save("model_wte_a2ctr_cb _act4_goodperf")

#%% Produce CDF comparing RL and and other solutions
from openpyxl import load_workbook
import os
import pandas as pd
wb = load_workbook(filename = r'C:\Users\cesa_\WTE_comp_xl.xlsx')


df1 = pd.read_excel(
      os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
      engine='openpyxl', sheet_name='Simulation',
)


df1.head()


dr_hist = df1["DeFleDR"][1:70]


RL_action_hist_from_list(dr_hist)

# centralized_rigid_npvs = df1["centralizedR" ]#, "DecentralizedF" , "CentralizedF"]
# centralized_flex_npvs = df1["CentralizedF" ]
# decentralized_flex_npvs = df1["DecentralizedF" ]



# centralized_rigid_npvs.head()

# centralized_flex_npvs.head()

# decentralized_flex_npvs.head()



# n_test_episodes = 2000

# CDF_RL_comparison_wte_lstm(n_test_episodes, model_akctr_wte_vec_lstm,  centralized_rigid_npvs, centralized_flex_npvs , decentralized_flex_npvs)
 

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
