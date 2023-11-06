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
        env = gym.make("gym_wte_full:wte-v4")
        return env
    return maker
    
train_env = gym.make('gym_wte_full:wte-v4')
test_env = gym.make('gym_wte_full:wte-v1')
debug_env = gym.make('gym_wte_full:wte-v3')




from openpyxl import load_workbook
import os
wb = load_workbook(filename = r'C:\Users\cesa_\WTE_comp_xl.xlsx')


df1 = pd.read_excel(
      os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
      engine='openpyxl', sheet_name='Capev',
)


df1.head()

df3 = pd.read_excel(
      os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
      engine='openpyxl', sheet_name='Probev',
)







df2 = pd.read_excel(
      os.path.join(  r'C:\Users\cesa_\WTE_comp_xl.xlsx' ),
      engine='openpyxl', sheet_name='Simulation',
)

df2.dropna()

centralized_rigid_npvs = df2["centralizedR" ]#, "DecentralizedF" , "CentralizedF"]
centralized_flex_npvs = df2["CentralizedF" ]
decentralized_flex_npvs = df2["DecentralizedF" ]



x_ci = centralized_rigid_npvs.dropna()

x_cf = centralized_flex_npvs.dropna()

x_df = decentralized_flex_npvs.dropna()

len(decentralized_flex_npvs)



# # load modesl to use for plot
model1 = ACKTR.load("AKCTR_model_GREAT" )
# model2 =  TRPO.load("wte-trpo-t0-good" ) # THIS ONE IS EXCELLENT FOR LOW FLEX
# model3 = TRPO.load("wte-trpo-t0-vgood")
model4 = TRPO.load("trpo_wte_highflex_1")
# model5 = ACKTR.load("akctr_wte_lowflex_df")
# #model6 = ACKTR.load("model_wte_a2ctr_cb _act4_goodperf")
# model7 = TRPO.load("trpo_defaulthp_lowflex")
# model8 = TRPO.load("best_model_trpo_wte_lowflex")
# model9 = TRPO.load("wte_trpo_good_df_mlp")
# model10 = TRPO.load("wte-trpo-t0-200exp_better1")
# model11 = TRPO.load("wte_trpo_good3")
# model12 = TRPO.load("wte-trpo-t0-200exp_correct_ok")
# model13 = TRPO.load("trpo_optuna_lowflex")
# model14 = TRPO.load("wte_trpo_good2" , env = test_env)
# model15 = TRPO.load("trpo_wte_highflex_lowtraining")
# model16 = TRPO.load("wte-trpo-t0-vgood1")




# s_1_demand_df = df1["Demand1" ]
# s_2_demand_df = df1["Demand2" ]
# s_3_demand_df = df1["Demand3" ]
# s_4_demand_df = df1["Demand4" ]
# s_5_demand_df = df1["Demand5" ]
# s_6_demand_df = df1["Demand6" ]

# cap_s1_dr = df1["Capacity1" ]
# cap_s2_dr = df1["Capacity2" ]
# cap_s3_dr = df1["Capacity3" ]
# cap_s4_dr = df1["Capacity4" ]
# cap_s5_dr = df1["Capacity5" ]
# cap_s6_dr = df1["Capacity6" ]

# cap_s1_hf = df1["Cap1HF"]
# cap_s2_hf = df1["Cap2HF"]
# cap_s3_hf = df1["Cap3HF"]
# cap_s4_hf = df1["Cap4HF"]
# cap_s5_hf = df1["Cap5HF"]
# cap_s6_hf = df1["Cap6HF"]


# cap_s1_lf = df1["Cap1LF"]
# cap_s2_lf = df1["Cap2LF"]
# cap_s3_lf = df1["Cap3LF"]
# cap_s4_lf = df1["Cap4LF"]
# cap_s5_lf = df1["Cap5LF"]
# cap_s6_lf = df1["Cap6LF"]

nsim = 2000


#ax = CDF_RL_comparison_wte_2rl_nohist(nsim, model1 , model4, test_env, debug_env,  x_ci, x_cf, x_df)


# #extract p5 and p95 for each distribution

npvs_hf, npvs_lf  = RL_comparison_wte_2rl_npvlist(nsim, model1 , model4, test_env, debug_env )







# npvs_hf = NPVs_model0
# npvs_lf = NPVs_model1
print("P5 HF" , np.percentile(npvs_hf, 5))
print("P5 LF" , np.percentile(npvs_lf, 5))
# print("P5 CI" , np.percentile(x_ci, 5))
# print("P5 CF" , np.percentile(x_cf, 5))
# print("P5 DF" , np.percentile(x_df, 5))


print("P95 HF" , np.percentile(npvs_hf, 95))
print("P95 LF" , np.percentile(npvs_lf, 95))
# print("P95 CI" , np.percentile(x_ci, 95))
# print("P95 CF" , np.percentile(x_cf, 95))
# print("P95 DF" , np.percentile(x_df, 95))


print("ENPV HF " , np.mean(npvs_hf))
print("ENPV LF " , np.mean(npvs_lf))
#print("ENPV CI " , np.mean(x_ci))



# prob_plot_treshold = .05
# prob_plot_treshold_hf = .05

 

#plot_rl_cap_evolution(model12, test_env, model4, debug_env, df1)
# rl_hist_lf, rl_hist_hf = plot_rl_cap_evolution_markers_subplot(model12, test_env, model4, debug_env, df1)
# rl_hist_lf, rl_hist_hf = plot_rl_cap_evolution_markers_subplot_ax_presetmarker(model12, test_env, model4, debug_env, df1)
# plot_rl_cap_evolution(model12, test_env, model4, debug_env, demand_df)
#plot_rl_prob_evolution(model8, test_env, model4, debug_env, demand_df, prob_plot_treshold, prob_plot_treshold_hf)
#a, klf, khf = plot_rl_prob_evolution(model12, test_env, model4, debug_env, prob_plot_treshold, prob_plot_treshold_hf, df1)
 

#a = plot_rl_lowflex_prob_evolution(model14, test_env, prob_plot_treshold, prob_plot_treshold_hf, df1)


#plot_rl_cap_evolution(model14, test_env, model4, debug_env, df1)

#kt = RL_probs_history_stoch(test_env, model11, demand_df)
#kth = RL_probs_history_stoch(debug_env, model4, demand_df)





# p0lf = df3["P0LF"]
# p1lf = df3["P1LF"]
# p3lf = df3["P3LF"]
# p6lf = df3["P6LF"]



# a = plt.figure(figsize=(10,5))
# plt.xlabel('Years')
# plt.ylabel('Action Probabilitity')
# plt.xticks([i for i in range (16)] )
# plt.title('Action Probability Distribution Over Time for 1 Stochastic Demand Scenario')
# plt.plot(x, p0lf , '--' , marker = 's' , label = 'A0 LF' )
# plt.plot(x, p1lf , '-.' , marker = 'o' , label = 'A1 LF' )
# plt.plot(x, p3lf , ':' , marker = '*' , label = 'A3 LF' )
# plt.plot(x, p6lf , '-' , marker = '+' , label = 'A6 LF' )



# plt.legend(loc='lower center' , ncol = 6, bbox_to_anchor = (.5 , -.25))
# plt.show()



# rl_cap_history_bysec_highflex = RL_capacities_history_bysector_adj(debug_env, model4, demand_df) 
# print(rl_cap_history_bysec_highflex)
# rl_cap_history_bysec_lowflex = RL_capacities_history_bysector_adj(test_env, model12, demand_df) 
# #print(rl_cap_history_bysec_lowflex)
# #retrive capacities for each sector resulting from RL interaction

# years =[]
# for i in range(16):
#     years.append(i)
# x = np.array(years)

# # caps3_hf_a = np.insert(caps3_hf_a, 0 ,0)
# # caps6_hf_a = np.insert(cap_s6_hf_a, 0 ,0)

# # caps3_hf_a = np.delete(caps3_hf_a, [16])

# # caps6_hf_a = np.delete(caps6_hf_a, [16])

# #caps3_hf_a2 = np.delete(caps5_hf_a, [16])

# mark_f_h = .05
# # mark_loc_lf = [.5, 3.5 , 5.5 , 7.5 , 10.5 , 14.5 , 17.5]

# # marker = itertools.cycle((',', '+', '.', 'o', '*' , 's', 'p' ,'P' ,'h' , 'x' , 'X' , 'D' , '3' , '4' , '8' ,'v' , '^' ))    
# # # plot only sectors where expansion decision made at some point

# fig = plt.figure(figsize = (12,5))
    
# plt.plot(x, s_1_demand_df, ':xb' , markevery = mark_f_h , label = 'S1'  )
# plt.plot(x, s_2_demand_df , 'y', markevery = mark_f_h , label = 'S2 ' )
# plt.plot(x, s_3_demand_df, '-.sg' , markevery = mark_f_h ,label = 'S3' )
# plt.plot(x, s_4_demand_df,  '--r', marker = 'd',markevery = mark_f_h ,label = 'S4' )
# plt.plot(x, s_5_demand_df,':' ,  color = 'orange' , marker = '*'  ,markevery = mark_f_h , label = 'S5' )
# plt.plot(x, s_6_demand_df, '-' , color = 'sienna', marker = 'o', markevery = mark_f_h ,label = 'S6' )
# #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
# plt.legend(loc='upper left' , ncol = 1 , fontsize= 14)
# plt.title('Stochastic Scenario Evolution' , fontsize= 16 , weight = 'bold')
# plt.xlabel('Year' , fontsize= 14)
# plt.ylabel('FW Demand (tpd)' , fontsize= 14)
# plt.xticks([i for i in range (16)] )







# fig , (ax1, ax2, ax3) = plt.subplots(1,3 , sharex = True, sharey = True, figsize=(12, 4))
# #a= plt.figure(figsize=(12,5))


# #plt.ylabel('Demand / Capacity (tpd)', fontsize = 16)
# if  any(i > 0 for i in cap_s1_lf):
#     ax2.plot(x,cap_s1_lf, ':xb' , markevery = mark_f_h ,  label = 'S1' )
# if  any(i > 0 for i in cap_s2_lf):
#     ax2.plot(x ,cap_s2_lf, '-.+',  label = 'S2')
# if  any(i > 0 for i in cap_s3_lf):
#     ax2.plot(x,cap_s3_lf, '-.sg'  , markevery = mark_f_h,label = 'S3')
# if  any(i > 0 for i in cap_s4_lf):
#     ax2.plot(x,cap_s4_lf, '-.' , marker = next(marker) , label = 'S4')
# if  any(i > 0 for i in cap_s5_lf):
#     ax2.plot(x,cap_s5_lf, '-.' , marker = next(marker) , label = 'S5')
# if  any(i > 0 for i in cap_s6_lf):
#     ax2.plot(x,cap_s6_lf, '-' , color = 'sienna', marker = 'o' , linewidth = 3, alpha = .7 , markevery = mark_f_h,label = 'S6')
# ax2.set_xticks([0,5,10,15])   
# ax1.set_xticks([0,5,10,15])
# ax3.set_xticks([0,5,10,15])
# #plt.subplot(3,1,3)
# if  any(i > 0 for i in cap_s1_hf):
#     ax3.plot(x,cap_s1_hf,  ':xb' ,label = 'S1')
# if  any(i > 0 for i in cap_s2_hf):                        
#     ax3.plot(x,cap_s2_hf,  '-.', marker = 'P',  markevery = mark_f_h ,label = 'S2')
# if  any(i > 0 for i in cap_s3_hf):
#     ax3.plot(x,cap_s3_hf,  '-.sg' ,  markevery = mark_f_h ,label = 'S3')
# if  any(i > 0 for i in cap_s4_hf):
#     ax3.plot(x,cap_s4_hf, '--r', marker = 'd',  markevery = mark_f_h ,label = 'S4')
# if  any(i > 0 for i in cap_s5_hf):
#     ax3.plot(x,cap_s5_hf, '-' ,   marker = '*' , markevery = mark_f_h,label = 'DRL HF S5')
# if  any(i > 0 for i in cap_s6_hf):
#     ax3.plot(x,cap_s6_hf, '-' , color = 'sienna', marker = 'o' , alpha = .7, markevery = mark_f_h ,label = 'S6')

# #axs[0][2].set_xticks([i for i in range (16)] )
# #axs[0][0].set_xticks([i for i in range (16)] )
# #plt.subplot(3,1,1)
# caps1_dr = ax1.plot(x,cap_s1_dr, ':xb', label = 'S1',  markevery = mark_f_h)
# #caps2_dr = plt.plot(x,cap_s2_dr, label = 'Flex DR S1')
# #caps3_dr = plt.plot(x,cap_s3_dr, label = 'Flex DR S1')
# #caps4_dr = plt.plot(x,cap_s4_dr, label = 'Flex DR S1')
# #caps5_dr = plt.plot(x,cap_s5_dr, label = 'Flex DR S1')
# caps6_dr = ax1.plot(x,cap_s6_dr,'-', color = 'sienna' ,alpha = .7,marker= 'o', label = 'S6', markevery = mark_f_h )

# # handles, labels = ax3.get_legend_handles_labels()
# # ax2.legend(handles, labels, loc='lower center' , ncol = 4 ,bbox_to_anchor = (.5 , -.3), 
# #            fontsize = 13)


# #capr_highflex = plt.plot(x, y4, label = 'DRL low flex design')
# # ax1.legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.3))
# # ax2.legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.3))
# # ax3.legend(loc='lower center' , ncol = 6 ,bbox_to_anchor = (.5 , -.3))
# #fig.suptitle('Demand and capacity evolution over project lifetime' , fontsize= 16)

# ax1.set_title("Decentralised FDR" , fontsize = 16, weight = 'bold')
# ax2.set_title("DRL-LF" , fontsize = 16, weight = 'bold')
# ax3.set_title("DRL-HF" , fontsize = 16, weight = 'bold')

# ax1.set_ylabel('Installed Capacity (tpd)' , fontsize = 14)
# # ax2.set_ylabel('Tonnes per day' , fontsize = 13)
# # ax3.set_ylabel('Tonnes per day' , fontsize = 13)
# plt.subplots_adjust( wspace = .08, hspace = .4)

# ax1.set_xlabel('Year', fontsize = 13)
# ax2.set_xlabel('Year', fontsize = 13)
# ax3.set_xlabel('Year', fontsize = 13)
#plt.ylabel('Tonnes per day' , fontsize = 16)
#plt.tight_layout()
#plt.setxlim(15)
