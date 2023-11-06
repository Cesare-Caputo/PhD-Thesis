# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:26:04 2021

@author: cesa_
"""

# hide all deprecation warnings from tensorflow
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import optuna
import gym
from gym import envs
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy as MlpDQN
from stable_baselines import PPO2 , DQN, ACKTR, TRPO, PPO1
from stable_baselines.common.policies import MlpPolicy as Mlp_a2c
from stable_baselines.common.evaluation import evaluate_policy
#from custom_policy_networks_wte import *
from stable_baselines.common.cmd_util import make_vec_env
import joblib

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)
sess = tf.Session()
#delete if it's registered
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
#mismatch cost defined here for environment instantiation after
mismatch_cost = .3417 # $/KwH
train_env = gym.make('mongolia_minigrid-v7', mismatch_cost = mismatch_cost )
test_env = train_env

# if __name__ == '__main__':
#     env_id = "gym_wte_full:wte-v3"
#     num_cpu = 4  # Number of processes to use
#     # Create the vectorized environment
#     env = make_vec_env(env_id, num_cpu, seed =0)

def optimize_acktr(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'ent_coef': trial.suggest_loguniform('ent_coef', .001, .1),
        'vf_coef': trial.suggest_loguniform('vf_coef', .1, .5),
        'learning_rate': trial.suggest_loguniform('learning_rate', .1, .5),
        'max_grad_norm': trial.suggest_loguniform('max_grad_norm', .2, .7),

    }

def optimize_trpo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'timesteps_per_batch': trial.suggest_categorical("timesteps_per_batch",[30, 300, 900, 700, 1000 ,2000, 4000, 7000, 8000]),
        'max_kl': trial.suggest_loguniform('max_kl', .001, .5),
        'lam': trial.suggest_loguniform('lam', .95, .99),
        'cg_iters': trial.suggest_categorical('cg_iters', [3,7 , 10, 13, 16, 20, 30]),
        'vf_stepsize': trial.suggest_loguniform('vf_stepsize', 2e-9, .05),
        'vf_iters': trial.suggest_categorical('vf_iters',[2, 5, 6, 8 ,9, 10, 3])  }


def optimize_dqn(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'exploration_fraction': trial.suggest_loguniform('exploration_fraction', .1, .9),
        'exploration_final_eps': trial.suggest_loguniform('exploration_final_eps', .01, .25),
        'exploration_initial_eps': trial.suggest_loguniform('exploration_initial_eps', .2, .99),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, .1),
        'learning_starts' : trial.suggest_loguniform('learning_starts', 100, 10000),

    }



def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'timesteps_per_actorbatch': trial.suggest_categorical("timesteps_per_actorbatch",[ 300, 900, 700, 1000 ,2000, 4000, 7000, 8000, 15000]),
        'clip_param': trial.suggest_loguniform('clip_param', 1e-3, 9e-1),
        'optim_batchsize': trial.suggest_categorical("optim_batchsize",[32,64,128,256]),
        'optim_epochs': trial.suggest_categorical('optim_epochs', [3,7 , 10, 13, 16, 20, 30]), }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo(trial)
    model = PPO1(Mlp_a2c,train_env,  verbose=0, **model_params)
    model.learn(100000)
    mean_reward, _ = evaluate_policy(model, train_env, n_eval_episodes=2)

    return -1 * mean_reward







if __name__ == '__main__':
    study = optuna.create_study(study_name='mongolia_stoch_ppo1')
    try:
        study.optimize(optimize_agent, n_trials=100  , n_jobs=1)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')

trial = study.best_trial

joblib.dump(study, "mongolia_stoch_ppo1.pkl")
print("Best hyperparameters: {}".format(trial.params))
