# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:07:08 2021

@author: cesa_
"""

from stable_baselines.common.policies import  register_policy, LstmPolicy , FeedForwardPolicy
from stable_baselines.deepq.policies import FeedForwardPolicy as ff_dqn
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


class CustomLSTMPolicy_large(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)
        

class CustomLSTMPolicy_small(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=16, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

        
        
 # Custom MLP policy of three layers of size 128 each
class Custom_FF_Policy_large(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Custom_FF_Policy_large, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[32, 128, 32],
                                                          vf=[32, 128, 32])],
                                           feature_extraction="mlp")       
    

class Custom_FF_Policy_small(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Custom_FF_Policy_small, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[8, 64, 8],
                                                          vf=[8, 64, 8])],
                                           feature_extraction="mlp")   

class Custom_FF_Policy_wte_md(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(Custom_FF_Policy_wte_md, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[13, 64, 64, 19],
                                                          vf=[13, 64, 64,  19])],
                                           feature_extraction="mlp")   


class DQN_FF_Policy(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(DQN_FF_Policy, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")


class CustomDQN_FF_Policy_small(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_FF_Policy_small, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")



class CustomDQN_FF_Policy_large(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_FF_Policy_large, self).__init__(*args, **kwargs,
                                           layers=[64,64],
                                           layer_norm=False,
                                           feature_extraction="mlp")


class CustomDQN_FF_Policy_norm(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_FF_Policy_norm, self).__init__(*args, **kwargs,
                                           layers=[32,32],
                                           layer_norm=True,
                                           feature_extraction="mlp")






class CustomDQN_MLP_Policy_small(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_MLP_Policy_small, self).__init__(*args, **kwargs,
                                           layers=[8, 32, 8 ],
                                           layer_norm=False,
                                           feature_extraction="mlp")



class CustomDQN_MLP_Policy_small_norm(ff_dqn):
    def __init__(self, *args, **kwargs):
        super(CustomDQN_MLP_Policy_small_norm, self).__init__(*args, **kwargs,
                                           layers=[8, 32, 8 ],
                                           layer_norm=True,
                                           feature_extraction="mlp")












# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class Custom_A2C_Policy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Custom_A2C_Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            extracted_features = nature_cnn(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})