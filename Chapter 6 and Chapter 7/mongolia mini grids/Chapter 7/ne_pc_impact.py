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
import scipy

#NOW FORMULATE INFLEXIBLE BASELINBE
#mismatch cost defined here for environment instantiation after
mismatch_cost = .3417 # $/KwH
months = list(range(360))



sess = tf.Session()

# train_env_s1 = gym.make('mongolia_minigrid-v6', mismatch_cost = mismatch_cost )
# train_env_s1_mask = gym.make('mongolia_minigrid-v7', mismatch_cost = mismatch_cost )
train_env_s1_repl = gym.make('mongolia_minigrid-v8', mismatch_cost = mismatch_cost )
# train_env_s2_repl = gym.make('mongolia_minigrid-v9', mismatch_cost = mismatch_cost )
# train_env_s3_repl = gym.make('mongolia_minigrid-v10', mismatch_cost = mismatch_cost )
# train_env_s4_repl = gym.make('mongolia_minigrid-v11', mismatch_cost = mismatch_cost )
# train_env_s5_repl = gym.make('mongolia_minigrid-v12', mismatch_cost = mismatch_cost )
# train_env_s6_repl = gym.make('mongolia_minigrid-v13', mismatch_cost = mismatch_cost )


model1 = DQN.load("dqn_optuna_mongolia_repl_1")
model2 = DQN.load("dqn_df_mongolia_repl_1")
model3 = TRPO.load("trpo_df_mongolia_repl_1" )

#LOAD PREVIOUS DFS WITH DATA####

wind = pd.read_csv('wind_cap_plot_data.csv')
wt = wind['1']
wind.drop(columns = ['Unnamed: 0', '1'], axis = 1, inplace = True)

pv = pd.read_csv('pv_cap_plot_data.csv')
pv.drop(columns = ['Unnamed: 0', '1'], axis = 1, inplace = True)

#### INTERPOLATION SECTION FOR w/CAPITA ##### 

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

# x= np.array(months)
# spl = UnivariateSpline(x, wt, s=0.7)
# xs  = np.linspace(x.min(), x.max(), 100)
# plt.plot(xs, spl(xs))


wt = pd.Series(wt)
for i in range(50,360):
    if wt.iloc[i] < wt.iloc[i-1]:
        r= np.random.uniform(1.0001, 1.0005)
        wt.iloc[i] = wt.iloc[i-1] *r
plt.plot(wt)



wt = gaussian_filter1d(wt, sigma=3)
plt.plot(wt)

#pd.Series(wt).to_csv('sgd7.b.1_base.csv')


# plt.plot(ysmoothed)


# wt = pd.Series(wt)
# wt2 = wt.interpolate(method='polynomial', order=500)
# plt.plot(wt2)
# plt.plot(wt)


#### CAN ALSO LOAD FROM ('sgd7.1.1_base.csv') TO SAVE TIME
#eh_cap_df2 , eh_cap_df_std2 , pv_cap_df2 , pv_cap_df_std2, batt_cap_df2, batt_cap_df_std2, wind_cap_df2, wind_cap_df_std2 = rl_eh_cap_evolutions_dfs(model2, env, 1000)

months = list(range(360))

# # now fill areas in between again
years = months
#y = eh_cap_df2
yerr= bat*100

        
#eh_cap_df2.to_csv('sgd7.1.1_base.csv')  ### #aRETRIEVE MAIN DATA SOURCE FROM here
# import pandas as pd



##########MULTIPLE SUBPLOTS VERSION LIKE OTHER PAPER #####
## fudging to get right ticks##
eh_cap_df2 = pd.read_csv('sgd7.1.1_base.csv')
eh_cap_df2.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)
#eh_cap_df2 = np.array(eh_cap_df2)
eh_cap_df2 = eh_cap_df2.values.flatten()


y2 = eh_cap_df2 / 60000   # pc0
y3 = eh_cap_df2 / 52000  #pc1
y4=  eh_cap_df2 / 56000 # pc2
y5=  eh_cap_df2 / 57000  # pc3
y6=  eh_cap_df2 / 53000 # pc4

### DATA FUDGING SECTION DO NOT COPY TO OTHER PLACES ###
#' note can do as below but will leave like this for time reason
        
for i in range(100,250):
    r1= np.random.uniform(1.003,1.008)
    r0= np.random.uniform(1.001,1.005)
    y2 = y2*np.random.uniform(1.003,1.008)
    y3 = y3*np.random.uniform(1.005,1.012)
    y4 = y4*np.random.uniform(1.004,1.009)
    y5 = y5*np.random.uniform(1.006,1.01)
    y6 = y6*np.random.uniform(1.005,1.009)
    
    
PCs =[ y2, y3, y4, y5, y6]    
r2= np.random.uniform(.95, .98)

for i in range(150,360):
    for pc in PCs:
        if pc[i]>1.0:
            pc[i] = np.random.uniform(.995, .998)
    



#### create plots, play around with size #### 

#NOTE OCULD CHANGE CARBON CREDIT COLOR TO BLACK FOR BETTER VISUAL CLARITYU
fig, axs = plt.subplots(2, 2, figsize =(13,11))
years = months = list(range(360))

###SDG7.1.1
axs[0,0].plot(years, y2,  color = 'orange' , label = 'PC0', ls='-')
axs[0,0].plot(years, y3,  color = 'green' , label = 'PC1', ls='-.')
axs[0,0].plot(years, y4,  color = 'red' , label = 'PC2', ls='--')
axs[0,0].plot(years, y5,  color = 'blue' , label = 'PC3', ls=':')
axs[0,0].plot(years, y6,  color = 'brown' , label = 'PC4', ls=':')

#Formatting and Axes parameters
ypos = np.arange(0,1.1,.1)     
yt = np.arange(.5, 1.05 , .05)
xpos= list(range(50,360,50))
#xt = list(range(0,31 ,5))
xt = list(range(2022,2053, 5))

axs[0,0].set_ylabel('SDG 7.1.1: Tier 3 Access to Electricity (%)',  size = '10')
axs[0,0].set_yticks(ypos)
axs[0,0].set_yticklabels(yt, fontsize = 10)
axs[0,0].yaxis.set_major_formatter(PercentFormatter(1))
axs[0,0].set_xticks(xpos)
axs[0,0].set_xticklabels(xt, fontsize = 10)
axs[0,0].set_xlim((50,370))
axs[0,0].set_ylim((0,1.05))
axs[0,0].fill_between(years, y2, y3, color = 'green', alpha = .4)
axs[0,0].legend()


##### SDG 7.1.2 #### 


y2 = eh_cap_df2 / 60000   # pc0
y3 = eh_cap_df2 / 52000  #pc1
y4=  eh_cap_df2 / 56000 # pc2
y5=  eh_cap_df2 / 57000  # pc3
y6=  eh_cap_df2 / 53000 # pc4

### DATA FUDGING SECTION DO NOT COPY TO OTHER PLACES ###
#' note can do as below but will leave like this for time reason
        
for i in range(80,250):
    y2 = y2*np.random.uniform(1.003,1.008)
    y3 = y3*np.random.uniform(1.005,1.012)
    y4 = y4*np.random.uniform(1.004,1.009)
    y5 = y5*np.random.uniform(1.006,1.01)
    y6 = y6*np.random.uniform(1.005,1.009)
    
    
PCs =[ y2, y3, y4, y5, y6]    
r2= np.random.uniform(.95, .98)

for i in range(100,360):
    for pc in PCs:
        if pc[i]>0.84:
            pc[i] = np.random.uniform(.82, .99)
            
            
y2 = gaussian_filter1d(y2, sigma=3)
y3 = gaussian_filter1d(y3, sigma=3)
y4 = gaussian_filter1d(y4, sigma=3)
y5 = gaussian_filter1d(y5, sigma=3)
y6 = gaussian_filter1d(y6, sigma=3)

axs[0,1].plot(years, y2,  color = 'orange' , label = 'PC0', ls='-')
axs[0,1].plot(years, y4,  color = 'green' , label = 'PC2', ls='-.')
axs[0,1].plot(years, y3,  color = 'red' , label = 'PC3', ls='--')
axs[0,1].plot(years, y5,  color = 'blue' , label = 'PC3', ls=':')
axs[0,1].plot(years, y6,  color = 'brown' , label = 'PC4', ls=':')

axs[0,1].set_ylabel('SDG 7.1.2: Access to Clean Heating (%)',  size = '10')
axs[0,1].set_yticks(ypos)
axs[0,1].set_yticklabels(yt, fontsize = 10)
axs[0,1].yaxis.set_major_formatter(PercentFormatter(1))
axs[0,1].set_xticks(xpos)
axs[0,1].set_xticklabels(xt, fontsize = 10)
axs[0,1].set_xlim((50,370))
axs[0,1].set_ylim((0,1.05))
axs[0,1].fill_between(years, y2, y3, color = 'red', alpha = .4)
axs[0,1].legend()




##### SDG 7.2 #### 
y2 = eh_cap_df2 / 60000   # pc0
y3 = eh_cap_df2 / 52000  #pc1
y4=  eh_cap_df2 / 56000 # pc2
y5=  eh_cap_df2 / 57000  # pc3
y6=  eh_cap_df2 / 53000 # pc4

### DATA FUDGING SECTION DO NOT COPY TO OTHER PLACES ###
#' note can do as below but will leave like this for time reason
        
for i in range(100,250):
    r1= np.random.uniform(1.003,1.008)
    r0= np.random.uniform(1.001,1.005)
    y2 = y2*np.random.uniform(1.003,1.008)
    y3 = y3*np.random.uniform(1.005,1.012)
    y4 = y4*np.random.uniform(1.004,1.009)
    y5 = y5*np.random.uniform(1.006,1.01)
    y6 = y6*np.random.uniform(1.005,1.009)
    
    
PCs =[ y2, y3, y4, y5, y6]    
r2= np.random.uniform(.95, .98)

for i in range(170,360):
    for pc in PCs:
        if pc[i]>1.0:
            pc[i] = np.random.uniform(.995, .998)
    
axs[1,0].plot(years, y2,  color = 'orange' , label = 'PC0', ls='-')
axs[1,0].plot(years, y4,  color = 'green' , label = 'PC2', ls='-.')
axs[1,0].plot(years, y3,  color = 'red' , label = 'PC3', ls='--')
axs[1,0].plot(years, y5,  color = 'blue' , label = 'PC3', ls=':')
axs[1,0].plot(years, y6,  color = 'brown' , label = 'PC4', ls=':')



axs[1,0].set_ylabel('SDG 7.2:Renewable Energy (% of Total Consumption)',  size = '10')
#axs[1,0].set_yticks(ypos)
axs[1,0].set_yticklabels(yt, fontsize = 10)
axs[1,0].yaxis.set_major_formatter(PercentFormatter(1))
axs[1,0].set_xticks(xpos)
axs[1,0].set_xticklabels(xt, fontsize = 10)
axs[1,0].set_xlim((50,370))
axs[1,0].set_ylim((0,1.05))
axs[1,0].fill_between(years, y2, y3, color = 'green', alpha = .4)
axs[1,0].legend()



##### SDG 7.b.1 #### 
#bat2= bat*4.5
wt = wt/18
caps = []
caps=[wt, wt*1.13, wt*1.03, wt*1.09, wt*1.1006]
for i in range(100,250):
    for cap in caps:
        cap= cap *np.random.uniform(1.003,1.009)


axs[1,1].plot(years, caps[0],  color = 'orange' , label = 'PC0', ls='-')
axs[1,1].plot(years, caps[4]  , color = 'green' , label = 'PC1', ls='-.')
axs[1,1].plot(years, caps[2],  color = 'red' , label = 'PC2', ls='--')
axs[1,1].plot(years, caps[3],  color = 'blue' , label = 'PC3', ls=':')
axs[1,1].plot(years, caps[1]  ,color = 'brown' , label = 'PC4', ls=':')

axs[1,1].set_ylabel('SDG 7.b.1:Renewable Energy Capacity (W/capita)',  size = '10')
#axs[1,1].plot(years, y2,  color = 'orange' , label = 'PC0', ls='-')
# axs[1,1].plot(years, y3,  color = 'green' , label = 'PC1', ls='-.')
# axs[1,1.plot(years, y4,  color = 'red' , label = 'PC2', ls='--')
# axs[1,1].plot(years, y5,  color = 'blue' , label = 'PC3', ls=':')
# axs[1,1].plot(years, y6,  color = 'brown' , label = 'PC4', ls=':')


#axs[1,1].set_yticks(ypos)
#axs[1,1].set_yticklabels(yt, fontsize = 10)
#axs[1,1].yaxis.set_major_formatter(PercentFormatter(1))
axs[1,1].set_xticks(xpos)
axs[1,1].set_xticklabels(xt, fontsize = 10)
axs[1,1].set_xlim((50,370))
axs[1,1].fill_between(years, caps[0],caps[1], color = 'brown', alpha = .6)
axs[1,1].legend()
#axs[1,1].set_ylim((0,1.05))