# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:07:08 2021

@author: cesa_
"""

from stable_baselines.common.policies import FeedForwardPolicy, register_policy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy as ff_dqn
import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)
        
        
        
 # Custom MLP policy of three layers of size 128 each
class Custom_FF_Policy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Custom_FF_Policy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")       
    




class CustomDQN_FF_Policy_small(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_FF_Policy_small, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")
        
        
        
class CustomDQN_FF_Policy_small_norm(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_FF_Policy_small, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=True,
                                           feature_extraction="mlp")