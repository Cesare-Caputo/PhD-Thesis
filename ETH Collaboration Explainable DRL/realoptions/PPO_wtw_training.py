# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:42:27 2023

@author: ccaputo
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from wtw_FULL_env_draft_v3 import WtwEnvFull_v3


# create the custom environment
env = WtwEnvFull_v3()

# define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# initialize the policy network
policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)

# define the optimizer
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# PPO hyperparameters
clip_epsilon = 0.2
num_steps = 128
num_episodes = 1000

# initialize the observation and reward normalization
obs_norm = np.zeros(env.observation_space.shape[0])
rew_norm = 0
num_actions = env.action_space.n


# training loop
for episode in range(num_episodes):
    state = env.reset()
    state = (state - obs_norm) / np.sqrt(obs_norm + 1e-8)
    done = False
    episode_rewards = []
    while not done:
        # get action probabilities from policy network
        action_probs = policy(torch.from_numpy(state).float()) + 1e-10
        #print(action_probs)
        action_probs = action_probs.view(-1, num_actions)
        action_probs = torch.softmax(action_probs, dim=1)

        # sample action from action probabilities
        m = Categorical(action_probs)
        action = m.sample()

        # take the action in the environment
        next_state, reward, done, _ = env.step(action.item())

        # update the reward normalization
        rew_norm = 0.99 * rew_norm + 0.01 * reward
        reward = reward / np.sqrt(rew_norm + 1e-8)

        episode_rewards.append(reward)

        # update the observation normalization
        obs_norm = 0.99 * obs_norm + 0.01 * next_state
        next_state = (next_state - obs_norm) / np.sqrt(obs_norm + 1e-8)

        # calculate the advantage
        advantage = reward - (policy(torch.from_numpy(next_state).float()) - policy(torch.from_numpy(state).float()))

        # calculate the old action probability
        old_action_probs = action_probs.gather(1, action.unsqueeze(-1))
        #old_action_probs = action_probs.gather(1, action.view(-1,1))

        # calculate the ratio (pi_theta / pi_theta_old)
        ratio = torch.exp(torch.log(old_action_probs) - torch.log(old_action_probs))

        # calculate the surrogate loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage

        loss = -torch.min(surr1, surr2).mean()

        # update the policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Episode: {}, Average Reward: {}".format(episode+1, sum(episode_rewards)/len(episode_rewards)))