# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:08:26 2023

@author: ccaputo
"""
import os
import numpy as np
import gym
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.ppo import MlpPolicy as MlpPPO
from stable_baselines3.dqn import MlpPolicy as MlpDQN
from stable_baselines3.a2c import MlpPolicy as MlpA2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from wtw_env_draft import WtwEnvSimple
from wtw_env_draft_v1  import WtwEnvSimple_v1
from wtw_env_draft_v3 import WtwEnvSimple_v3
#from realoptions import 
from sim import generate_options, multiple_scenarios_res
from realoptions import (
    OrnsteinUhlenbeckProcess, DeterministicProcess, MarkedPoissonPointProcess,
    NormalProcess, WaterPrivate, WaterLeakage, WaterService, WaterIndustry,
    WaterExternal, WaterStorage, OptionSupply, OptionLeakage, OptionPrivate,
    OptionMisc, OptionCapacity, WaterSupply, plot_d_and_s, plot_costs,
    plot_risk, plot_input, plot_time, plot_prob, plot_risk_comp, cost_benefit,
    plot_nb_comp
)




roa_options = generate_options()
results_roa = multiple_scenarios_res(roa_options)


#a = single_scenario(roa_options["4"])
results = results_roa
for i, (option, result) in enumerate(results.items()):
    #print(result["cost"])
    costs = result["cost"]
    data = costs['data']
    labels = costs['labels']
    _len = data.shape[2]

    values = np.sum(data, axis=(0, 1))


ax = plot_risk_comp2(results_roa)
################ RL Results #################


env = WtwEnvSimple_v1()
tensorboard_log="./sb3_wtw_test_v1" # we leave on same tensorboard to see improvements in performance visually
log_dir = "./sb3_wtw_ppo_test_v1/"


path = os.path.join(log_dir, 'best_model') #weird bug here saying permission denied
model = PPO.load("best_model", env = env)

model_ppo_v3 = PPO.load("ppo_wtw_decent_v3.3")

env3 = WtwEnvSimple_v3()
env3 = Monitor(env3)



def NPCs_RL(episodes, model , test_env):
    env = test_env
    NPVs =[]
    for i in range(episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic= False)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            NPV_adj = (sum(episode_rewards))
        NPVs.append(-NPV_adj)
    NPVar = np.array(NPVs )
    return NPVar

rl_res1 = NPCs_RL(1000, model , env)
rl_res = NPCs_RL(1000, model , env)
rl_res2 = NPCs_RL(1000, model , env)
rl_res3 = NPCs_RL(1000, model_ppo_v3 , env3)

def plot_risk_comp2(results, name='', filename='risk_comp', fformat='pdf'):
    fig, ax = plt.subplots(1, sharex=False, figsize=(16/2, 8/2))
    # fig.suptitle(
    #     '{}'.format(name))
    colors = ['C{}'.format(i) for i in range(11)]

    colors.extend(['magenta', 'red'])
    print(colors)
    for i, (option, result) in enumerate(results.items()):
        ax = plot_prob(result['cost'], ax=ax, color=colors[i],
                       cum=True, mean=True, label=option, log=True)
    # ax2 = plot_prob(result['cost'], ax=ax2, title='PDF', cdf=False,
    #                 xlabel='[m£]', cum=True, mean=True, ci=.98)

    # fig.subplots_adjust(hspace=0.15, wspace=0.15, left=.1,
    #                     right=.95, top=0.90, bottom=0.07)

    ax.set_xlabel(r'Costs [m£]')
    ax.set_ylabel('Cumulative probability')
    return ax

def plot_risk_comp_rl(results, rl_results, name='', filename='risk_comp', fformat='pdf'):
    fig, ax = plt.subplots(1, sharex=False, figsize=(16/2, 8/2))
    colors = ['C{}'.format(i) for i in range(11)]

    colors.extend(['magenta', 'red'])
    print(colors)
    for i, (option, result) in enumerate(results.items()):
        ax = plot_prob(result['cost'], ax=ax, color=colors[i],
                       cum=True, mean=True, label=option, log=True)
    bins_rl = np.sort(rl_results) 
    _len = len(rl_results)
    #n_lf = np.arange(1,len(bins_lf)+1) / float(len(bins_lf))
    n_rl =np.array(range(_len))/float(_len)
    ax.plot(bins_rl, n_rl, color = "black", label = "DRL-LF")
    ax.axvline(np.mean(rl_results), color='black', linestyle='dashed', linewidth=1)
    ax.set_xlabel(r'Costs [m£]')
    ax.set_ylabel('Cumulative probability')
    ax.legend()
    return ax



### NOTE THAT RESULTS ARE OBTAINED FROM SIM FILE FOR REPRODUCIBILITY TOMORROW
#### NEXT TASK IS TO FIND OUT OF SAMPLE PERFORMANCE OF OTHER SOLUTIONS

plot_risk_comp_rl(results_roa, rl_res)
plot_risk_comp_rl(results_roa, rl_res2)


# bins_lf = np.sort(rl_res) 
# _len = len(rl_res)
# #n_lf = np.arange(1,len(bins_lf)+1) / float(len(bins_lf))
# n_lf =np.array(range(_len))/float(_len)


# ax = plot_risk_comp2(results, rl_res)
# ax.plot(bins_lf, n_lf, color = "black", label = "DRL")

# # ax.hist(rl_res ,100, density=True, histtype='step',
# #                     cumulative=True, color = "black", label='RL agent performance')
# ax.axvline(np.mean(rl_res), color='black', linestyle='dashed', linewidth=1)
# ax.legend()