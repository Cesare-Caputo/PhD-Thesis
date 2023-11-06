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
env_name12 = 'mongolia_minigrid-v12'
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


    
import mongolia_minigrids



import os
import pandas as pd
import numpy as np
from stable_baselines.common.env_checker import check_env
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, CnnPolicy
from stable_baselines.common.policies import MlpPolicy as Mlp_a2c
from stable_baselines.common.policies import  MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN, A2C , ACKTR, ACER,  TRPO , PPO2
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from stable_baselines import DQN, A2C , ACKTR, ACER
from scipy import optimize
from scipy.optimize import differential_evolution
from matplotlib.ticker import PercentFormatter
from RL_SB_helper_mongolia import *
from mongolia_plotting import *
from custom_policy_networks_mongolia import *
from stable_baselines.common import set_global_seeds, make_vec_env
import joblib
#NOW FORMULATE INFLEXIBLE BASELINBE
#mismatch cost defined here for environment instantiation after
mismatch_cost = .3417 # $/KwH

sess = tf.Session()

# train_env_s1 = gym.make('mongolia_minigrid-v6', mismatch_cost = mismatch_cost )
# train_env_s1_mask = gym.make('mongolia_minigrid-v7', mismatch_cost = mismatch_cost )
train_env_s1_repl = gym.make('mongolia_minigrid-v8', mismatch_cost = mismatch_cost )
train_env_s2_repl = gym.make('mongolia_minigrid-v9', mismatch_cost = mismatch_cost )
train_env_s3_repl = gym.make('mongolia_minigrid-v10', mismatch_cost = mismatch_cost )
train_env_s4_repl = gym.make('mongolia_minigrid-v11', mismatch_cost = mismatch_cost )
train_env_s5_repl = gym.make('mongolia_minigrid-v12', mismatch_cost = mismatch_cost )
train_env_s6_repl = gym.make('mongolia_minigrid-v13', mismatch_cost = mismatch_cost )


model1 = DQN.load("dqn_optuna_mongolia_repl_1")
# model2 = DQN.load("dqn_df_mongolia_repl_1")
# model3 = TRPO.load("trpo_df_mongolia_repl_1")

env_s1 = train_env_s1_repl
env_s2 = train_env_s2_repl
env_s3 = train_env_s3_repl
env_s4 = train_env_s4_repl
env_s5 = train_env_s5_repl
env_s6 = train_env_s6_repl


model = model1
episodes =2000

s1_elccs = elcc_rl_mongolia(episodes, model, env_s1)
print("PC1 ELCC RL optimal solution" , s1_elccs)
s2_elccs = elcc_rl_mongolia(episodes, model, env_s2)
print("PC2 ELCC RL optimal solution" , s2_elccs)
s3_elccs = elcc_rl_mongolia(episodes, model, env_s3)
print("PC3 ELCC RL optimal solution" , s3_elccs)
s4_elccs = elcc_rl_mongolia(episodes, model, env_s4)
print("PC4 ELCC RL optimal solution" , s4_elccs)
s5_elccs = elcc_rl_mongolia(episodes, model, env_s5)
print("PC5 ELCC RL optimal solution" , s5_elccs)
s6_elccs = elcc_rl_mongolia(episodes, model, env_s6)
print("PC6 ELCC RL optimal solution" , s6_elccs)



# model2 = DQN.load("dqn_optuna_mongolia_repl_1" , env = env_s6)
# agent_test_env_nsteps(model, env_s1, 36)
# agent_test_env_nsteps(model, env_s2, 10)
#agent_test_env_nsteps(model, env_s6, 100)


#EH DISCOUNT NOT PROPERLY COMPUTED IN S3
# # agent_test_env_nsteps(model, env_s5, 36)
#agent_test_env_nsteps(model, env_s6, 36)


# evaluate_carbonrevenue_rl(model, env_s4, episodes)
# evaluate_carbonrevenue_rl(model, env_s5, episodes)

#TEST EMISSIONS

# evaluate_emissions_rl(model_dqn_df_mask, 10)

# agent_test_env_nsteps_stoch(model_trpo_mongolia_df , train_env_s1, 36 )
# # # # model_dqn_small_mlp.save("mongolia_dqn_det_small_mlp")
# # # # model_dqn_small_mlp_norm.save("mongolia_dqn_det_small_mlp_norm")
# model_trpo_mongolia_df.save("mongolia_trpo_stoch_s1")




# rl_cost_df = rl_df_from_interactions_monthly_capex_split_nosum(model_trpo_mongolia_df, train_env_det_split)
# n_yr = 12
# rl_cost_df_monthly = rl_cost_df.groupby(rl_cost_df.index //n_yr).sum()
# rl_cost_df_monthly.drop('Action' , inplace = True, axis = 1)
# rl_cost_df_monthly.drop('Total' , inplace = True, axis = 1)
# rl_cost_df_monthly.drop('Total Capex' , inplace = True, axis = 1)





# rl_cost_df_monthly["Opex"] = rl_cost_df_monthly["Opex"] - rl_cost_df_monthly["Coal"]
# #rl_cost_df_monthly.drop('Opex' , inplace = True, axis = 1)
# a = rl_cost_df_monthly.plot.bar(stacked = True)
# a.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.4))
# a.set_xlabel('Years')
# a.set_ylabel('Cost ($)')
# a.set_title('Cumulative Cost Breakdown- Determinstic DRL')



# from openpyxl import load_workbook
# import os
# import pandas as pd
# wb = load_workbook(filename = r'C:\Users\cesa_\drl_2_x.xlsx')


# df1 = pd.read_excel(
#       os.path.join(  r'C:\Users\cesa_\drl_2_x.xlsx' ),
#       engine='openpyxl', sheet_name='drl_2_s5',
# )


# a = df1.plot.bar(stacked = True)
# a.legend( loc = 'lower center' , ncol = 5 ,bbox_to_anchor = (.5 , -.4))
# a.set_xlabel('Years')
# a.set_ylabel('Cost ($)')
# a.set_title('Scenario 5: Cumulative Cost Breakdown- Determinstic DRL')





#rl_cost_df_monthly.to_csv('drl_2.csv')







#drl_yearly_cf_bar_det(model_trpo_mongolia_df ,train_env_det_split)

# agent_test_env_nsteps_stoch(model_trpo_mongolia_df, train_env_det_split, 240)

# model1_trpo = TRPO.load("mongolia_trpo_det_df" , env = train_env_det_split)
# model2_trpo = TRPO.load("trpo_dF_1_det_ok" , env = train_env_det_split)
# model1_dqn = DQN.load("optuna_dqn_diesel_pref" , env = train_env_det_split )





# study1 = joblib.load("mongolia_deterministic_dqn_1.pkl")

# trial1 = study1.best_trial


# # try with optuna hyperparameters too
# model_dqn_optuna2 = DQN(MlpPolicy, train_env_det_split, gamma =1, exploration_fraction =trial1.params['exploration_fraction'] , exploration_final_eps = trial1.params['exploration_final_eps'] , 
#                        exploration_initial_eps = trial1.params['exploration_initial_eps'] , learning_rate = trial1.params['learning_rate'] , learning_starts =  trial1.params['learning_starts'] , 
#                        verbose=0, prioritized_replay=True , 
#                        tensorboard_log="./mongolia_minigrids_10gers_determinstic_optuna/")


# model_dqn_optuna2.learn(total_timesteps = 500000)






# if __name__ == '__main__':
#     env_id = "mongolia_minigrids:mongolia_minigrid-v4"
#     num_cpu = 4  # Number of processes to use
#     # Create the vectorized environment
#     mismatch_cost = mismatch_cost
#     env = make_vec_env(env_id, num_cpu, seed =0)
#     model_akctr_mongolia = ACKTR(Mlp_a2c, train_env_det_split, gamma =1,  tensorboard_log=  "./mongolia_minigrids_10gers_determinstic/" )
#     model_akctr_mongolia.learn(total_timesteps = 500000)



# model_akctr_mongolia.save("mongolia_akctr_det_df")


# # model_dqn_lr_standard_m_det = DQN(MlpPolicy, train_env_determinstic, gamma =1, prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_10gers_determinstic/")
# # # model_dqn_lr_small = DQN(MlpPolicy, train_env, gamma =1, exploration_fraction = .3, learning_rate = 5e-5,  exploration_final_eps = .001, 
# # #                 prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_basic_start363W_salvage/")
# # # model_dqn_lr_med = DQN(MlpPolicy, train_env, gamma =1, exploration_fraction = .3, learning_rate = 5e-4,  exploration_final_eps = .001, 
# # #                 prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_basic_start363W_salvage/")
# # # model_dqn_lr_high = DQN(MlpPolicy, train_env, gamma =1, exploration_fraction = .3, learning_rate = 5e-3,  exploration_final_eps = .001, 
# # #                 prioritized_replay=True ,  tensorboard_log="./mongolia_minigrids_basic_start363W_salvage/")

# model_dqn_lr_very_small_m_det.learn(total_timesteps = 500000)
# model_dqn_df_det.learn(total_timesteps = 500000)
# model_dqn_adj_det.learn(total_timesteps = 500000)



# model_dqn_lr_very_small_m_det.save("high_eps_mongolia_det_1")
# model_dqn_df_det.save("adj_eps_mongolia_det_1")
# model_dqn_adj_det.save("df_eps_mongolia_det_1")


# model1 = DQN.load("high_eps_mongolia_det_1" , env = train_env_det_split)
# model2 = DQN.load("adj_eps_mongolia_det_1", env = train_env_det_split)
# model3 = DQN.load("df_eps_mongolia_det_1", env = train_env_det_split)




# #drl_monthly_cf_plot_det(model1 ,train_env_det_split )

# drl_yearly_cf_bar_det(model1 ,train_env_det_split)




# # model_dqn_lr_standard_m.learn(total_timesteps = 100000)


# # model_dqn_lr_very_small_m_det.save("mongolia~_ok_quick")


# # model1 =  DQN.load(""mongolia~_ok_quick")    
# #model_dqn_lr_very_small_m_det = DQN.load("mongolia_minigrids_dqn_verysmallr_good" )

# n_steps = 24

# agent_test(model_trpo_mongolia_df)
# model_trpo_mongolia_df.save("trpo_df_1_det_ok")
# # # agent_test(model2)
# # # agent_test(model3)
# # agent_test_env_nsteps(model_dqn_optuna1, train_env_det_split, 100)


# agent_test_env_nsteps_stoch(model_trpo_mongolia_df, train_env_det_split, 24)

# # model_dqn_optuna1.save("optuna_dqn_diesel_pref")
# # # # # agent_test(model_dqn_lr_standard_m)

# # # # model2 = DQN.load("mongolia_minigrids_dqn_2_basic_variedaction" , env = train_env_determinstic)

# # # # agent_test(model2)

# # rl_cost_df = rl_df_from_interactions_monthly_capex_split_nosum(model_dqn_lr_very_small_m_det ,train_env_det_split)






# #drl_lifetime_cf_bar_det(model_dqn_optuna1 ,train_env_det_split)

# drl_yearly_cf_bar_det(model_trpo_mongolia_df ,train_env_det_split)




# a = rl_cashflow_by_category(model_dqn_optuna1 ,train_env_det_split)






# drl_monthly_cf_plot_det(model_dqn_optuna1 ,train_env_determinstic ) 


#ab = evaluate_shortage_rl(model_dqn_lr_very_small_m_det, 1)
# test agent for debugging purposes

# n_steps = 240 # look at behabviour during first year only , so first 12 timesteps
# #HERE DEFINE TEST ENVIRONMENT WITH NO NEGATIVE REWARD PENALTIES, NOT CREATED YET
# #test env = 

# agent_test_env_nsteps(model_dqn_lr_very_small_m,train_env_monthly, n_steps )
# agent_test_env_nsteps(model_dqn_lr_standard_m,train_env_monthly, n_steps )


# # agent_test(model_dqn_lr_med)
# # agent_test(model_dqn_lr_high)

# #evaluate and compare trained models:
#evaluate(model_dqn_lr_very_small_m , num_episodes = 100)
# evaluate(model_dqn_lr_small, num_episodes = 1000)
# evaluate(model_dqn_lr_med, num_episodes = 1000)
# evaluate(model_dqn_lr_high, num_episodes = 1000)

# model_dqn_lr_very_small.save("mongolia_minigrids_dqn_verysmallr_good")
# model_dqn_lr_small.save("mongolia_minigrids_dqn_smallr_good")

# model_dqn_lr_small_solid = DQN.load("mongolia_minigrids_dqn_smallr_good", env = train_env)
# model_dqn_lr_very_small_solid = DQN.load("mongolia_minigrids_dqn_verysmallr_good", env = train_env)

# evaluate(model_dqn_lr_small_solid , num_episodes = 1000)
# evaluate(model_dqn_lr_very_small_solid, num_episodes = 1000)


#agent_test(model_dqn_lr_small_solid)

# evaluate_shortage_rl(model_dqn_lr_small_solid , num_episodes = 1000)
# evaluate_shortage_rl(model_dqn_lr_very_small_solid, num_episodes = 1000)