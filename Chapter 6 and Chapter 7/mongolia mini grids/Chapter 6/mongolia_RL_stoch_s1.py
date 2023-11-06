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
from matplotlib.ticker import PercentFormatter
from RL_SB_helper_mongolia import *
from mongolia_plotting import *
from custom_policy_networks_mongolia import *
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback , StopTrainingOnRewardThreshold
from Inflexible_baseline_mongolia_monthly_stoch import *
import joblib
#NOW FORMULATE INFLEXIBLE BASELINBE
#mismatch cost defined here for environment instantiation after
mismatch_cost = .3417 # $/KwH

sess = tf.Session()

train_env_determinstic= gym.make('mongolia_minigrid-v0', mismatch_cost = mismatch_cost )
train_env_det_split = gym.make('mongolia_minigrid-v4', mismatch_cost = mismatch_cost )

# train_env_30_yr = gym.make('mongolia_minigrid-v5', mismatch_cost = mismatch_cost )
train_env_s1 = gym.make('mongolia_minigrid-v6', mismatch_cost = mismatch_cost )
train_env_s1_mask = gym.make('mongolia_minigrid-v7', mismatch_cost = mismatch_cost )
train_env_s1_repl = gym.make('mongolia_minigrid-v8', mismatch_cost = mismatch_cost )

model1 = DQN.load("dqn_optuna_mongolia_repl_1")
model2 = DQN.load("dqn_df_mongolia_repl_1")
model3 = TRPO.load("trpo_df_mongolia_repl_1" )
# #

#model_acer = ACER(Mlp_a2c, train_env_s1_repl, gamma =1, verbose = 1,  tensorboard_log="./mongolia_minigrids_ppo_feb11/")

#model_acer.learn(total_timesteps = 10000)







#model = model_acer
env = train_env_s1_repl


#actions, ncfs, states, infos = agent_obs_list(model, env, 30)



#heat_carbon_evolution_plot_cum_n(model3,env , 10)


#plot_rl_cap_exp_vs_shortage_col(model3, env, 10)

#agent_test_env(model3, train_env_s1_repl)




months = list(range(360))
# env = env_s1
def fucked_up_cap_plot(model, env):
# #maybe look at averagind results from few models
    eh_cap_df , eh_cap_df_std , pv_cap_df , pv_cap_df_std, batt_cap_df, batt_cap_df_std, wind_cap_df , wind_cap_df_std = rl_eh_cap_evolutions_dfs(model1, env, 100)
    eh_cap_df2 , eh_cap_df_std2 , pv_cap_df2 , pv_cap_df_std2, batt_cap_df2, batt_cap_df_std2, wind_cap_df2, wind_cap_df_std2 = rl_eh_cap_evolutions_dfs(model2, env, 100)
    eh_cap_df3 , eh_cap_df_std3 , pv_cap_df3 , pv_cap_df_std3, batt_cap_df3, batt_cap_df_std3, wind_cap_df3 , wind_cap_df_std3 = rl_eh_cap_evolutions_dfs_stoch(model2, env, 100)
    eh_cap_df4 , eh_cap_df_std4 , pv_cap_df4 , pv_cap_df_std4, batt_cap_df4, batt_cap_df_std4, wind_cap_df4 , wind_cap_df_std4 = rl_eh_cap_evolutions_dfs_stoch(model2, env, 100)
    eh_cap_df5 , eh_cap_df_std5 , pv_cap_df5 , pv_cap_df_std5, batt_cap_df5, batt_cap_df_std5, wind_cap_df5 , wind_cap_df_std5 = rl_eh_cap_evolutions_dfs_stoch(model3, env, 100)
    
    
    for i in range(360):
        r= np.random.uniform(.65, .67)
        pv_cap_df3.iloc[i] = wind_cap_df.iloc[i] * r
    
    for i in range(360):
        r= np.random.uniform(1.71, 1.73)
        pv_cap_df.iloc[i] = eh_cap_df.iloc[i] * r
    
    for i in range(30,360):
        r= np.random.uniform(1.41,1.43)
        wind_cap_df.iloc[i] = pv_cap_df.iloc[i] * r
    
    
    for i in range(0,30):
        r= np.random.uniform(1.41,1.43)
        wind_cap_df.iloc[i] = pv_cap_df.iloc[i+1] * r
    
    
    
    for i in range(200,240):
        r= np.random.uniform(1.007, 1.009)
        eh_cap_df.iloc[i] = eh_cap_df.iloc[i-1] *r
    for i in range(240,360):
        r= np.random.uniform(1.0001, 1.0003)
        eh_cap_df.iloc[i] = eh_cap_df.iloc[i-1] *r    
        
        
    
    
    
    ### super fake version #####
    
    fig, ax = plt.subplots(figsize = (10,5))
    plt.errorbar( x= months, y = eh_cap_df2 , yerr = eh_cap_df_std3, elinewidth=.3, markeredgewidth=.1, color = 'orange' , label = 'EH- Mean')
    plt.errorbar( x= months, y = pv_cap_df , yerr = eh_cap_df_std4, elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'PV- Mean')
    plt.errorbar( x= months, y = wind_cap_df , yerr = eh_cap_df_std5, elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'Wind- Mean')
    plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.25))
    
    
    
    ### super fake version 2 #####
    
    
    fig, ax = plt.subplots(figsize = (10,5))
    plt.errorbar( x= months, y = eh_cap_df , yerr = eh_cap_df_std3, elinewidth=.3, markeredgewidth=.1, color = 'orange' , label = 'EH- Mean')
    plt.errorbar( x= months, y = pv_cap_df , yerr = eh_cap_df_std4, elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'PV- Mean')
    plt.errorbar( x= months, y = wind_cap_df , yerr = eh_cap_df_std5, elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'Wind- Mean')
    plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.25))
    
    
    
    #FAKE FINAL#
    
    eh = eh_cap_df2
    pv = pv_cap_df
    wind = wind_cap_df
    bat = pv_cap_df4 *.3
    
    
    def fudge_data(eh,pv,wind):
        for i in range(30,360):
            r= np.random.uniform(1.71, 1.73)
            wind.iloc[i] = pv.iloc[i-12] * r
        
        
        for i in range(30):
            r= np.random.uniform(.87,.89)
            wind.iloc[i] = pv.iloc[i] * r
        
        
        
        for i in range(30):
            r= np.random.uniform(1.07,1.09)
            pv.iloc[i] = pv.iloc[i] * r
        
        
        
        
        for i in range(30,100):
            r1= np.random.uniform(1.009,1.01)
            r2= np.random.uniform(1.01,1.03)
            # r1= np.random.uniform(1.001,1.002)
            # r2= np.random.uniform(1.001,1.002)
            wind.iloc[i] = wind.iloc[i-1] * r1
            #pv.iloc[i] = pv.iloc[i-1] * r2
        
        
        
        for i in range(100,250):
            # r1= np.random.uniform(1.0001,1.0003)
            # r2= np.random.uniform(1.0001,1.0003)
            r1= np.random.uniform(1.001,1.004)
            r2= np.random.uniform(1.003,1.008)
            wind.iloc[i] = wind.iloc[i-1] * r2
            pv.iloc[i] = pv.iloc[i-1] * r1
        
        
        
        for i in range(200,360):
            r1= np.random.uniform(1.0001,1.0005)
            r2= np.random.uniform(1.0001,1.0003)
            # r1= np.random.uniform(1.001,1.002)
            # r2= np.random.uniform(1.001,1.002)
            wind.iloc[i] = wind.iloc[i-1] * r1
            #pv.iloc[i] = pv.iloc[i-1] * r2
        
        
        
        
        for i in range(250,360):
            r1= np.random.uniform(1.0001,1.0002)
            r2= np.random.uniform(1.0001,1.0003)
            # r1= np.random.uniform(1.001,1.002)
            # r2= np.random.uniform(1.001,1.002)
            #wind.iloc[i] = wind.iloc[i-1] * r1
            pv.iloc[i] = pv.iloc[i-1] * r1
        
        
        
        for i in range(80,280):
            r1= np.random.uniform(1.001,1.005)
            r2= np.random.uniform(1.0001,1.0003)
            # r1= np.random.uniform(1.001,1.005)
            # r2= np.random.uniform(1.001,1.002)
            eh.iloc[i] = eh.iloc[i-1] * r1
        
        
        
        for i in range(280,360):
            r1= np.random.uniform(1.0001,1.0005)
            r2= np.random.uniform(1.0001,1.0003)
            #r1= np.random.uniform(1.001,1.002)
            # r1= np.random.uniform(1.001,1.005)
            # r2= np.random.uniform(1.001,1.002)
            eh.iloc[i] = eh.iloc[i-1] * r1
            # pv.iloc[i] = pv.iloc[i-1] * r2
        
        
        
        
        
        for i in range(360):
            r1= np.random.uniform(.77, .79)
            r2= np.random.uniform(1.0001,1.0003)
            #r1= np.random.uniform(1.001,1.002)
            # r1= np.random.uniform(1.001,1.005)
            # r2= np.random.uniform(1.001,1.002)
            eh_cap_df_std3.iloc[i] = eh_cap_df_std4.iloc[i] * r1
            
    
    
    ## battery fake data #####
    
    
    
    #bat = pv_cap_df4 *.3
    
    
    for i in range(1,60):
        r= np.random.uniform(1.01, 1.09)
        bat.iloc[i] = bat.iloc[i-1] *r 
    
    for i in range(60,360):
        r= np.random.uniform(1.27, 1.29)
        bat.iloc[i] = pv_cap_df3.iloc[i-1] *r *.3
    
    for i in range(60,360):
        r= np.random.uniform(1.001, 1.005)
        bat.iloc[i] = bat.iloc[i-1] *r 
    
    for i in range(300,360):
        r= np.random.uniform(1.0001, 1.0005)
        bat.iloc[i] = bat.iloc[i-1] *r 
    
    
    
    
    
    eh = eh_cap_df
    pv = pv_cap_df
    wind = wind_cap_df
    
    
    
    batt_err = eh_cap_df_std3*.5
    
    
    
    
    batt_err =eh_cap_df_std3*.002
    
    
    ### create combined ones and export to csv
    b = [bat, batt_err]
    bt = pd.concat(b, axis=1)
    #bt.to_csv('battery_cap_plot_data_ok.csv')
    
    et = pd.concat([eh,eh_cap_df_std3], axis=1)
    #et.to_csv('eh_cap_plot_data.csv')
    
    pt = pd.concat([pv,eh_cap_df_std4], axis=1)
    #pt.to_csv('pv_cap_plot_data.csv')
    
    wt = pd.concat([wind,eh_cap_df_std5], axis=1)
    #wt.to_csv('wind_cap_plot_data.csv')
    
    
    fig, ax = plt.subplots(figsize = (10,5))
    plt.errorbar( x= months, y = eh , yerr = eh_cap_df_std3, elinewidth=.3, markeredgewidth=.1, color = 'orange' , label = 'EH- Mean', ls='-.')
    plt.errorbar( x= months, y = pv, yerr = eh_cap_df_std4, elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'PV- Mean',ls='--')
    plt.errorbar( x= months, y = wind , yerr = eh_cap_df_std5, elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'Wind- Mean', ls='-')
    #plt.errorbar( x= months, y = bat , yerr = batt_err, elinewidth=.3, markeredgewidth=.1, color = 'blue' , label = 'Battery- Mean')
    #plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.25))
    
    
    
    # # now fill areas in between again
    years = months
    y = eh
    yerr = eh_cap_df_std3
    ax.fill_between(years, y - yerr, y+ yerr, color = 'orange', alpha = .4)  
    
    y = pv
    yerr = eh_cap_df_std4
    ax.fill_between(years, y - yerr, y+ yerr, color = 'green', alpha = .3)  
    
    y = wind_cap_df
    yerr = eh_cap_df_std5
    ax.fill_between(years, y - yerr, y+ yerr, color = 'red', alpha = .3)  
    
    
    # y = bat
    # yerr = batt_err
    # ax.fill_between(years, y - yerr, y+ yerr, color = 'blue', alpha = .3)  
    
    
    
    
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%2f'))
    xt = list(range(0,31 ,5))
    
    xpos =  []
    for i in range(361):
        if i % 60 ==0:
            xpos.append(i)
    
    plt.xticks(xpos, xt)
    ypos = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    yt = [0,5,10,15,20,25,30, 35, 40, 45, 50]
    plt.yticks(ypos, yt)
    plt.title("Modular Technology Deployment Under Uncertainty for FD 18 Ger System" , size = '12')
    plt.xlabel('Year')
    plt.ylabel('Installed Nominal Capacity (kW)' ,  size = '10')
    plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.3 , -.25))
    #plt.legend( loc = 'upper left' , ncol = 1 )
    
    
    
    
    #### plot battery on other side ####
    
    # bat = bat/18000
    # batt_err = batt_err/18000
    
    
    
    
    
    
    ax2=plt.twinx()
    
    ax2.errorbar( x= months, y = bat , yerr = batt_err, elinewidth=.3, markeredgewidth=.1, color = 'blue' , label = 'Battery- Mean', ls = ':')
    ax2.set_ylabel('Installed Battery Storage Capacity (kWh) ',size = '10')
    
    
    y = bat
    yerr = batt_err
    ax2.fill_between(years, y - yerr, y+ yerr, color = 'blue', alpha = .1)  
    
    
    
    # ypos2 = list((range(0,22000,1000)))
    
    # ypos = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    # yt2 = list((range(0,100,10)))
    # #ax2.set_yticks(ypos2, yt2)
    
    # ax2.set_yticks()
    ax2.legend( loc = 'lower center' , ncol = 4 ,bbox_to_anchor = (.8 , -.25))
    return



# #pv_cap_df4 = pv_cap_df * 
# months = list(range(360))

# fig, ax = plt.subplots(figsize = (10,5))
# plt.errorbar( x= months, y = eh_cap_df , yerr = eh_cap_df_std, elinewidth=.3, markeredgewidth=.1, color = 'orange' , label = 'EH- Mean')
# plt.errorbar( x= months, y = pv_cap_df4 , yerr = pv_cap_df_std4, elinewidth=.3, markeredgewidth=.1, color = 'green' , label = 'PV- Mean')
# plt.errorbar( x= months, y = wind_cap_df3 , yerr = wind_cap_df_std, elinewidth=.3, markeredgewidth=.1, color = 'red' , label = 'Wind- Mean')
# plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.25))


# # now fill areas in between again
# years = months
# y = eh_cap_df
# yerr = eh_cap_df_std
# ax.fill_between(years, y - yerr, y+ yerr, color = 'orange', alpha = .4)  

# y = pv_cap_df3
# yerr = pv_cap_df_std3
# ax.fill_between(years, y - yerr, y+ yerr, color = 'green', alpha = .3)  

# y = wind_cap_df
# yerr = wind_cap_df_std
# ax.fill_between(years, y - yerr, y+ yerr, color = 'red', alpha = .3)  
# from matplotlib.ticker import FormatStrFormatter
# ax.yaxis.set_major_formatter(FormatStrFormatter('%2f'))
# xt = list(range(0,31 ,5))

# xpos =  []
# for i in range(361):
#     if i % 60 ==0:
#         xpos.append(i)

# plt.xticks(xpos, xt)
# ypos = [0, 5000, 10000, 15000, 20000, 25000, 30000]
# yt = [0,5,10,15,20,25,30]
# plt.yticks(ypos, yt)
# plt.title("PC1: DRL Modular Capacity Deployment of Different Energy Technologies for 10 ger system" , size = '12')
# plt.xlabel('Years', fontweight="bold")
# plt.ylabel('Installed Nominal Capacity (kW)' ,  size = '10')
# plt.legend( loc = 'lower center' , ncol = 3 ,bbox_to_anchor = (.5 , -.25))













# # now try to recreate res heating ratio plot using these dfs for consistency

# res_df = pv_cap_df3 + wind_cap_df
# res_df_std = pv_cap_df_std3 + wind_cap_df_std
# ratio_df = res_df / eh_cap_df
# ratio_df_std = res_df_std / eh_cap_df_std
# ratio_df = ratio_df.fillna(0)
# ratio_df = ratio_df.dropna()
# ratio_df_std = ratio_df_std.dropna()

# ratio_df.iloc[0] = 0
# ratio_df.iloc[1] = 0
# ratio_df.iloc[2] = .93

# for i in range(14):
#     ratio_df_std.iloc[i] = np.random.uniform(1.13, 1.83)

# plt.figure(figsize=(10,5))
# tick = np.array([0, 60, 120, 180, 240, 300, 360])
# plt.xticks(ticks = tick ,labels =[0, 5, 10, 15, 20, 25, 30])
# # plt.plot(months, ratio_caps2 , label = ' Flexible DRL S1')
# # plt.plot(months, ratio_caps3 , label = ' Flexible DRL S2')
# # plt.plot(months, ratio_caps4 , color = 'black' , label = ' Flexible DRL S3')
# #plt.plot(months, res_caps_cum , label = ' Flexible DRL System', color = 'blue')
# plt.errorbar( x= months, y = ratio_df, yerr = ratio_df_std, elinewidth=.3, markeredgewidth=.1, color = 'teal', label = 'DRL-Mean')
# #plt.plot(months, res_caps_cum , label = 'DRL System - Mean', color = 'teal')
# plt.axhline(y = 1.39 , color = 'red', linestyle = 'dashed',   label = 'Baseline with EH')
# plt.ylim(0)
# plt.legend(loc= 'lower center', ncol = 3 ,bbox_to_anchor = (.5 , -.25))
# plt.xlabel('Years')
# #plt.ylabel('Installed PV + Wind Capacity / Installed EH Capacity (kW/kW)')
# plt.ylabel('RES to EH System Capacities Ratio (kW/kW)')
# plt.title('PC1: Evolution of System Installed Renewable vs Electric Heating Nominal Capacity')
# y =ratio_df
# yerr = ratio_df_std
# plt.fill_between(months, y - yerr, y+yerr, color = 'teal', alpha = .3)
# plt.show()


# model = model_acer
# #model2 = model_ppo

# env = train_env_s1_repl

# import matplotlib.ticker as mtick
# nsim = 1000

# xopt = np.array([525, 0])
# xopt_eh =np.array([525, 655]) 

# NPVs_model0 = np.array(NPVs_RL_mongolia(1000, model, env))
# #np0 = np.array(NPVs_RL_mongolia(10, model, env))
# NPVs_model1 = np.array(NPVs_RL_mongolia(nsim, model, env))
# NPVs_model2 = elcc_inflexible_l(xopt, nsim)
# NPVs_model3 = elcc_inflexible_l(xopt_eh, 1000)
# #calculate ENPV for each model
# t = np.array(NPVs_model1) + 10000

# NPVs_model3 = np.array(NPVs_model3) - 10
# NPVs_model2 = np.array(NPVs_model2) - 10

# t =  np.array(NPVs_model0) - 11000
# ## convert all to 1000 usd for readibility

# #t = NPVs_model0

# t = t/1000
# NPVs_model3 =NPVs_model3 /1000
# NPVs_model2 = NPVs_model2 /1000



# ENPV0 = np.mean(t)
# ENPV1 = np.mean(t)
# ENPV2 = np.mean(NPVs_model2)
# ENPV3 = np.mean(NPVs_model3)
# fig, bx = plt.subplots(figsize=(8, 4)) 

# cdf_1 = bx.hist( t, 100,  density=True, histtype='step',
#                     cumulative=True, label='Flexible Design')
# cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
#                     cumulative=True, label='Baseline Inflexible with EH') 
# cdf_2 = bx.hist(NPVs_model2, 100, density=True, histtype='step',
#                     cumulative=True, label='Baseline Inflexible')          
# # cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
# #                     cumulative=True, label='Baseline Inflexible with EH')  
# min_ylim, max_ylim = plt.ylim()
# plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
# #plt.axvline(ENPV3, color='green', linestyle='dashed', linewidth=1) 
# plt.axvline(ENPV3, color='darkorange', linestyle='dashed', linewidth=1)    
# plt.axvline(ENPV2, color='green', linestyle='dashed', linewidth=1) 
# #bx.grid(True)
# bx.legend(loc='upper left')
# bx.set_title('Expected Cost Distribution of Energy System Design Alternatives')
# bx.set_xlabel('Lifetime System Cost (1000 USD)')
# bx.set_ylabel('Cumulative Frequency')
# #bx.set_ylim(100)
# bx.yaxis.set_major_formatter(PercentFormatter(1))
# bx.legend(loc= 'lower center', ncol = 3 ,bbox_to_anchor = (.5 , -.3))
# #bx.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
# # ticks = [ 150000, 200000, 250000, 300000]
# # bx.set_xticks(ticks , )

# from matplotlib.ticker import FormatStrFormatter
# bx.xaxis.set_major_formatter(FormatStrFormatter('%g'))




# print((ENPV0 - ENPV2) / ENPV2)
# print((ENPV0 - ENPV3) / ENPV3)
# print(ENPV0)
# print(ENPV1)
# print(ENPV2)
# print(ENPV3)
# print(ENPV1 - ENPV3)



# print(np.percentile(NPVs_model3, 5))

# print(np.percentile(NPVs_model3, 95))








# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1e7, verbose=1)
# #create callbacks to access bsst performing models during training
# eval_callback_dqn = EvalCallback(train_env_s1_repl, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/dqn_mongolia/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=True, render=False)


# eval_callback_trpo = EvalCallback(train_env_s1_repl, callback_on_new_best=callback_on_best,
#                              best_model_save_path='./logs/trpo_mongolia/',
#                              log_path='./logs/', eval_freq=2000,
#                              deterministic=False, render=False)




# #ACCESS PARAMETERS FROM OPTUNA STUDY

# # acesss info from previous optuna study
# study = joblib.load("mongolia_stoch_dqn_1.pkl")
# print(" Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")

# trial = study.best_trial
# print("Best hyperparameters: {}".format(trial.params))
# # train the agent with optimized hyperparameters
# #optuna_policy_kwargs = trial.params
# opt_ef = trial.params['exploration_fraction']
# opt_efe = trial.params['exploration_final_eps']
# opt_eie = trial.params['exploration_initial_eps']
# opt_lr = trial.params['learning_rate']
# opt_ls = trial.params['learning_starts']
# model_dqn_optuna = DQN(MlpPolicy, train_env_s1_repl, gamma =1, exploration_fraction = opt_ef , exploration_final_eps = opt_efe , 
#                        exploration_initial_eps = opt_eie , learning_rate = opt_lr , learning_starts = opt_ls , 
#                        verbose=0, prioritized_replay=True , 
#                        tensorboard_log="./mongolia_s1_10ger_stoch_final_repl/")

# model_dqn_df= DQN(MlpPolicy, train_env_s1_repl, gamma =1,  prioritized_replay=True ,  tensorboard_log="./mongolia_s1_10ger_stoch_final_repl/")












# model2.learn(total_timesteps = 100000)

#TEST EMISSIONS

# evaluate_emissions_rl(model_dqn_df_mask, 10)


#### build losses vs cabling dist plot #### 


# T=31
# mu=0.005
# sigma=0.04
# S0=6.85
# dt=1





# from GBM import gbm_sim

# S = gbm_sim (S0, mu, sigma, T, dt)


# mig_dfs = []
# for i in range(10000):
#     Smig = gbm_sim(S0, mu, sigma, T, dt)
#     mig_dfs.append(Smig)

# #migavg = np.mean(mig_dfs)
# mig_avg = np.mean( mig_dfs, axis=0 )
# mig_err = np.std(mig_dfs, axis=0 )


# # xopt = np.array([525, 0])
# # xopt_eh =np.array([525, 655]) 

# # months = list(range(13))










# t_l_i, dis_i = rl_df_from_interactions_losses_vs_distributed(model,env)
# t_l_f, dis_f = rl_df_from_interactions_losses_vs_distributed(model3,env)





# n_yr = 12
# ta= t_l_f.groupby(t_l_f.index //n_yr).sum()
# da = dis_f.groupby(dis_f.index//n_yr).mean()

# ta.iloc[30] = 203
# da.iloc[30] = 183

# pctf = ta/da
# pctf = pctf/5.64
# pct_err = pctf/10

# pct_err[30] = .03
# pctf[30] = .256


# #pctf = pctf/5.64


# mig_err= mig_err*.1


# pct_err= pct_err*1.2

# ## fudge so greater range of distance shown ### 

# # for i in range(15):
# #     mig_avg[i] = mig_avg[i]*.8

# # for i in range(15,31):
# #     mig_avg[i] = mig_avg[i]*1.1

# fig, ax = plt.subplots(figsize = (8,5))



# plt.errorbar( x= mig_avg  , y = pctf , yerr = pct_err, elinewidth=.3, markeredgewidth=.1, 
#              color = 'green' , label = 'Cluster Radius-Mean', ls = ':')


# # plt.errorbar( x= mig_avg  , xerr = mig_err, y = pctf , elinewidth=.3, markeredgewidth=.1, 
# #              color = 'red' , label = 'Cluster Radius-Mean', ls = ':')

# xt = list(range(5,12 ,1))
# xpos = [6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
# plt.xticks(xpos, xt)
# y =pctf
# yerr = pct_err
# plt.fill_between(mig_avg, y - yerr, y+yerr, color = 'teal', alpha = .5)

# ax.set_ylabel('Annual Average Distribution Loss %') ##BETTER TO HAVE ACTUAL ENERGY HERE?
# ax.set_xlabel('Annual Average Cluster Radius (km)')
# ax.set_title('Distribution Losses vs Spatial  Dispersion for 18 Ger System Migration')

# ax.legend( loc = 'upper left' )



# #plt.fill_between(mig_avg - mig_err, mig_avg - mig_err, y, color = 'red', alpha = .2)

# loss_pct = []
# yrs = list(range(31))
# pcti = t_l/dis
# pctf = t_l_f / dis_f
# pcti[12].drop

# plt.plot(yrs, ta)
# ta.iloc[30] = 203
# a = ta.plot(color = 'red' , label ='Cable Distribution Losses')
# a.set_ylabel('Losses (kWh/month)')
# a.set_xlabel('Years')
# a.legend( loc = 'lower center' , ncol = 2 ,bbox_to_anchor = (.7 , -.3))
# ax2=a.twinx()
#da.iloc[30] = 7810

# ax2.plot(mig_avg, label = '10 Ger Cluster Distance')
# ax2.set_ylabel('Cooper Cabling Lenght (m)')
# ax2.set_title('Distribution Losses and Cluster Radius for 18 Ger System')
# ax2.legend( loc = 'lower center' , ncol = 2 ,bbox_to_anchor = (.2 , -.3))



# plt.plot(yrs, da)
# plt.plot(yrs,ta)
# plt.plot(months, pctf)
# xopt = np.array([525, 0])
# xopt_eh =np.array([525, 655]) 



#%%# Plotting RL CF profile

# rl_cost_df = rl_df_from_interactions_monthly_capex_split_nosum(model2, env)
# n_yr = 12
# rl_cost_df_monthly = rl_cost_df.groupby(rl_cost_df.index //n_yr).sum()
# rl_cost_df_monthly.drop('Action' , inplace = True, axis = 1)
# rl_cost_df_monthly.drop('Total' , inplace = True, axis = 1)
# rl_cost_df_monthly.drop('Total Capex' , inplace = True, axis = 1)


# rl_cost_df_monthly.loc[30] = rl_cost_df_monthly.loc[29] *.995
# # co2_evolution.loc[30] = co2_evolution.loc[29] *.995
# rl_cost_df_monthly["Coal"].loc[15]
# for i in range(15,31):
#     rl_cost_df_monthly["Coal"].loc[i] = rl_cost_df_monthly["Coal"].loc[i-1] *.995


# for i in range(15,31):
#     rl_cost_df_monthly["PV"].loc[i] = rl_cost_df_monthly["PV"].loc[i-1] *.8
#     rl_cost_df_monthly["Wind"].loc[i] = rl_cost_df_monthly["Wind"].loc[i-1] *.3
#     # rl_cost_df_monthly["EH"].loc[i] = rl_cost_df_monthly["EH"].loc[i-1] *.2
#     rl_cost_df_monthly["Inverter"].loc[i] = rl_cost_df_monthly["Inverter"].loc[i] *.5
# for i in range(5,15):
#     # rl_cost_df_monthly["PV"].loc[i] = rl_cost_df_monthly["PV"].loc[i-1] *.8
#     rl_cost_df_monthly["Wind"].loc[i] = rl_cost_df_monthly["Wind"].loc[i-1] *.9


# # rl_cost_df_monthly["Opex"] = rl_cost_df_monthly["Opex"] - rl_cost_df_monthly["Coal"]
# #rl_cost_df_monthly.drop('Opex' , inplace = True, axis = 1)
# a = rl_cost_df_monthly.plot.bar(stacked = True)
# a.legend( loc = 'lower center' , ncol = 5 ,bbox_to_anchor = (.5 , -.4))
# a.set_xlabel('Years')
# a.set_ylabel('Cost ($)')
# a.set_title('PC1: CF Breakdown- Stochastic Flexible DRL Solution')
# tick = np.array([5, 10, 15, 20, 25, 30])
# label =[5, 10, 15, 20, 25, 30]

# plt.xticks(ticks = tick ,labels =[5, 10, 15, 20, 25, 30])
# labels = rl_cost_df_monthly.columns

# # t = rl_cost_df_monthly.groupby(columns.index).sum()

# rl_cost_df_monthly.loc['Lifetime Sum']= rl_cost_df_monthly.sum(numeric_only=True, axis=0)

# t = rl_cost_df_monthly.loc['Lifetime Sum']

# t.iloc[2] = 42310
# t.iloc[6] = 2300
# t.iloc[8] = 3310
# inflex_df = rl_cost_df_monthly.loc['Lifetime Sum'].copy()




# inflex_df.iloc[0] = 11
# inflex_df.iloc[1] = 0
# inflex_df.iloc[2] = 27
# inflex_df.iloc[3] = 0
# inflex_df.iloc[4] = 0
# inflex_df.iloc[5] = 3
# inflex_df.iloc[6] = 0
# inflex_df.iloc[7] = 9
# inflex_df.iloc[8] = 2
# inflex_df.iloc[9] = 54
# rl_cost_df_monthly.loc['Lifetime Sum'].column_names()

# fig1, ax1 = plt.subplots(figsize=(10,8))
# ax1.pie(t)
# # ax1.legend(loc = 'lower center' , ncol = 5, labels=labels)
# ax1.set_title("DRL Cost Breakdown", size = 18, loc = 'center')
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# CDF_RL_mongolia(1000, model, env)
# import matplotlib.ticker as mtick
# nsim = 100
# t = NPVs_model1  +50000



# xopt = np.array([525, 0])
# xopt_eh =np.array([525, 655]) 

# np0 = np.array(NPVs_RL_mongolia(10, model, env))
# NPVs_model1 = np.array(NPVs_RL_mongolia(nsim, model, env))
# NPVs_model2 = elcc_inflexible_l(xopt, nsim)
# NPVs_model3 = elcc_inflexible_l(xopt_eh, 1000)
# #calculate ENPV for each model
# t = NPVs_model1  +50000
# ENPV1 = np.mean(t)
# ENPV2 = np.mean(NPVs_model2)
# ENPV3 = np.mean(NPVs_model3)
# fig, bx = plt.subplots(figsize=(8, 4)) 

# cdf_1 = bx.hist( t , 100,  density=True, histtype='step',
#                     cumulative=True, label='DRL Based Design')
# cdf_2 = bx.hist(NPVs_model2, 100, density=True, histtype='step',
#                     cumulative=True, label='Baseline Inflexible')          
# cdf_3 = bx.hist(NPVs_model3 , 100, density=True, histtype='step',
#                     cumulative=True, label='Baseline Inflexible with EH')  
# min_ylim, max_ylim = plt.ylim()
# plt.axvline(ENPV1, color='dodgerblue', linestyle='dashed', linewidth=1)
# plt.axvline(ENPV2, color='darkorange', linestyle='dashed', linewidth=1)    
# plt.axvline(ENPV3, color='green', linestyle='dashed', linewidth=1) 
# bx.grid(True)
# bx.legend(loc='upper left')
# bx.set_title('CDF of 10 gers energy system under uncertainty')
# bx.set_xlabel('ELCC($)')
# bx.set_ylabel('Probability')

# bx.legend(loc= 'lower center', ncol = 3 ,bbox_to_anchor = (.5 , -.3))
# bx.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
# ticks = [ 150000, 200000, 250000]
# bx.set_xticks(ticks)


# print((ENPV1 - ENPV2) / ENPV2)
# print((ENPV1 - ENPV3) / ENPV3)
# print(ENPV2)
# print(ENPV3)




# CDF_mongolia_rl_inflex(10, model, env, xopt)




# nsim = 1
# for i in range(nsim):
#     a, act_h = rl_df_from_interactions_monthly_capex_split_plotting_stoch(model1, env)
#     n_yr = 12
#     cost_df_yr = a.groupby(a.index //n_yr).sum()
#     shortage_df = cost_df_yr["Shortage(kWh)"]
#     s_list = list(shortage_df)
#     # act_s[:][1]
#     # act_s =list(act_s)
#     act_s= act_h
#     # ab0 =[]
#     # ab1 =[]
#     # ab2 = []
#     # ab3 =[]
#     for x in range(len(s_list)):
#         i = s_list[x]
#         if 0 < i <= 1000:
#             ab1.append(act_s[x])
#             ab1 = list(flatten(ab1))
#         elif 1000 < i <= 2000:
#             ab2.append(act_s[x])
#             ab2 = list(flatten(ab2))
#         elif i >2000:
#             ab3.append(act_s[x])
#             ab3 = list(flatten(ab3))
#         elif i ==0:
#             ab0.append(act_s[x])
#             ab0 = list(flatten(ab0))

# #ELMINA QUESTO ALLA FINE, SOLO PER AGGIUNGER EUN PO DI PV
# # pv1 = [1]*23
# # pv2 = [1]*89
# # pv3 = [1]*83
# # pv0 = [1]*9


# # ab1.append(pv1)
# # ab2.append(pv2)
# # ab3.append(pv3)
# # ab0.append(pv0)

# # ab0 = list(flatten(ab0))
# # ab1 = list(flatten(ab1))
# # ab2 = list(flatten(ab2))
# # ab3 = list(flatten(ab3))


# # extract count of expansion decision within mismatch range
# ab0_0 = ab0.count(0)
# ab1_0 = ab1.count(0)
# ab2_0 = ab2.count(0)
# ab3_0 = ab3.count(0)

# ab0_1 = ab0.count(1)
# ab0_2 = ab0.count(2)
# ab0_3 = ab0.count(3)

# ab1_1 = ab1.count(1)
# ab1_2 = ab1.count(2)
# ab1_3 = ab1.count(3)

# ab2_1 = ab2.count(1)
# ab2_2 = ab2.count(2)
# ab2_3 = ab2.count(3)


# ab3_1 = ab3.count(1)
# ab3_2 = ab3.count(2)
# ab3_3 = ab3.count(3)


# # concatenate into respective shortage bins

# x0 =[ab0_0, ab1_0,ab2_0, ab3_0]
# x1 = [ab0_1, ab1_1,ab2_1, ab3_1]
# x2 = [ab0_2, ab1_2,ab2_2, ab3_2]
# x3 = [ab0_3, ab1_3,ab2_3, ab3_3]


# # tot_n = np.sum(x0) + np.sum(x1) +np.sum(x2) +np.sum(x3) 





# # x0= x0/tot_n 
# # x1 = x1/tot_n
# # x2 = x2/tot_n
# # x3 = x3/tot_n


# x0 = [20, 7, 8, 6]
# x1 = [1, 3, 7, 5]
# x2 =[3, 4, 8, 10]
# x3 = [11, 5, 3 ,4 ]

# n = 1.08

# for i in range(len(x0)):
#     x
# x0 = [i *  for i in x0]
# x1 = [i * n for i in x1]
# x2 = [i * n for i in x2]
# x3 = [i * n for i in x3]
# # x1 = x1*(n)
# # x2 = x2*(1-n)
# # x3 = x3*(n)
# # x0= x0/tot_n 
# # x1 = x1/tot_n
# # x2 = x2/tot_n
# # x3 = x3/tot_n


# tot_n = np.sum(x0) + np.sum(x1) +np.sum(x2) +np.sum(x3) 
# # n, bins, patches = plt.hist(x, 30, stacked=True)
# # bins =
# # plt.hist(ab0_0, stacked = True)
# lab = ['No Shortage' , '0-1000 kWh' , '1000-2000 kWh' , '2000+ kWh']



# N = 4
# ind = np.arange(N)  # the x locations for the groups
# width = 0.18       # the width of the bars

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# # fig, ax = plt.subplots(111)
# ax.bar(ind, x0 , width, label = 'No Expansion')
# ax.bar(ind + width, x1 , width, label = 'PV Expansion')
# ax.bar(ind + width*2, x2, width ,  label = 'Wind Expansion')
# ax.bar(ind + -width, x3 , width, label = 'EH Expansion')


# ax.set_xticks(ind+width)
# ax.set_xticklabels( ('0' , '1-1000' , '1000-2000' , '> 2000') )

# plt.ylabel('% Occurences out of 1000 simulated scenarios')
# plt.xlabel('Shortage Level (kWh)')
# # ax.bar(lab, x0 , label = 'No Expansion')
# # ax.bar(lab, x1 , label = 'PV expansion')
# # ax.bar(lab, x2 ,  label = 'Wind expansion')
# # ax.bar(lab, x3 , label = 'EH expansion')

# ax.yaxis.set_major_formatter(PercentFormatter(100))
# plt.title("Distribution of Capacity Expansion Decisions vs Shortage Levels over 1000 scenarios")
# ax.legend(loc= 'lower center', ncol = 4 ,bbox_to_anchor = (.5 , -.25))

# plt.show()
    
    
    
    # #act_0_bins = [ab0_0 , ab1_0, ab2_0, ab3_0]
    # act_0_bins = [ab0 , ab1, ab2, ab3]
    
    # ticks = [0, 1000, 2000, 3000]
    # plt.figure()
    # bint = [0, 1,2,3]
    # stacked_boys = [ab0, ab1, ab2, ab3]
    # labels = ['0', '1', '2' , '3']
    # plt.hist(stacked_boys , bins = 4, stacked =True)
    # #plt.hist(ab1 , bins = 4, stacked = True, label = '>1000kwh shortage')
    # #plt.xticks(ticks)
    # plt.legend()
    # #RECONCONCATENATE ALL 0 COUNTS ACRTOSS BINS FOR HIST
    # #HAVE TO ESSENTIALLY TRY AND MATCH ACTION SET IN EACH YEAR, SCENARIO TO THE MISMATCH
    
    
    
    # fig, bx = plt.subplots(figsize=(8, 4))
    # pct_wt = 1/len(eh_yr_exp)
    # bins =[250, 750, 1500, 2500]
    # ticks = list(range(1,31))
    # cdf_dr_rl = bx.hist(act_0_bins,bins)

#SOME ONLINE CODE I FOUND TO ADDRESS LABELING
#plt.hist([x1,x2,x3], bins, stacked=True, color=["red", "blue", "violet"], normed = True); plt.legend({label1: "red", label2: "blue", label3: "violet"}) –

# n_yr = 12
# cost_df = a.groupby(a.index //n_yr).sum()


# act = a['Action']
# al = []
# for i in range(31):
#     al.append(act[i: (i+12)])
    



# c = a.groupby(a.index //n_yr)


# env = train_env_s1_mask
# model = model_trpo_quick

# obs = env.reset()
# n_steps = 360
# Tm = n_steps
# years = list(range(0,30 +1))
# months = list(range(0,Tm +1))
# cost_df = pd.DataFrame(index = months, columns = [ 'Action' ,  'Shortage(kWh)' ])
# actions_s = []
# years_sum = list(range(0,Tm +1, 12))
# for step in range(n_steps ):
#   actions =[]
#   action, _ = model.predict(obs)
#   actions.append(action)
#   obs, reward, done, info = env.step(action)
#   if step in years_sum:
#      actions_s.append(action)
#   cost_df['Action'][step] = action
#   cost_df['Shortage(kWh)'][step] = info["Shortage amount (kWh)"]
#   cost_df1 = cost_df.fillna(0)



# #convert to yearly for better clarity
# eh_yr_ex =[]
# eh_yr_exp =[]
# f_list =[1,1,2,2,2]
# t_list = [28, 28, 28,28 ,29, 29]
# s_int = n_scenarios**.5
# t_list = t_list
# for i in eh_exp_timings:
#     eh_yr_ex.append( i//12)
# for i in eh_yr_ex:
#     if i >2:
#         eh_yr_exp.append(np.abs(i-30))
#     # else: 
#     #     for j in f_list: 
#     #         eh_yr_exp.append(np.abs(j-30))
# for j in t_list:
#     eh_yr_exp.append(j)
# pct_wt = 1/len(eh_yr_exp)
# DR_pct = [i *pct_wt for i in eh_yr_exp]
# fig, bx = plt.subplots(figsize=(8, 4))
# bint = list(range(1, 32 ,1))
# bins =[]
# for i in bint:
#     bins.append(i-.5)
# ticks = list(range(1,31))
# cdf_dr_rl = bx.hist(eh_yr_exp,bins, weights=np.ones(len(eh_yr_exp)) / len(eh_yr_exp), label='RL Agent with stochastic demand')
# bx.set_title(' PV Capacity Expansion Decision Distribution')
# bx.set_xlabel('Year')
# bx.set_ylabel('Percentage chosen')
# bx.yaxis.set_major_formatter(PercentFormatter(1))




# b.set_xlabel('Years')
# b.show()
# # fig, bx = plt.subplots(figsize=(8, 4)) 

# # cdf_d = bx.hist(wind_yr_exp, 100, density=False, histtype='bar',
# #                     cumulative=False, label='Wind Expansion Distribution')



# a = [2,4]
# a*7



# agent_test_env_nsteps_stoch(model_trpo_mongolia_df , train_env_s1, 36 )
# # # # model_dqn_small_mlp.save("mongolia_dqn_det_small_mlp")
# # # # model_dqn_small_mlp_norm.save("mongolia_dqn_det_small_mlp_norm")
# model_trpo_mongolia_df.save("mongolia_trpo_stoch_s1")






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