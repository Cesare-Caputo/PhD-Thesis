# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:08:26 2023

@author: ccaputo
"""
import os
import numpy as np
import gym

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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wtw_env_draft import WtwEnvSimple
from wtw_env_draft_v3 import WtwEnvSimple_v3


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                # if self.verbose > 0:
                #print(f"Num timesteps: {self.num_timesteps}")
                #print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    # if self.verbose > 0:
                    #print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True



################# FULL ENV VERSION with increased observation space but reduced action space ############
from wtw_FULL_env_draft_v1 import WtwEnvFull_v1


# env = WtwEnvFull_v1()
# # we leave on same tensorboard to see improvements in performance visually
# tensorboard_log = "./sb3_wtw_FULL_v1"
# log_dir = "./sb3_wtw_ppo_FULL_v1/"

# env = Monitor(env)
# callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
# model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-5, clip_range= .1,
#             tensorboard_log=tensorboard_log)
# model.learn(total_timesteps=700000, callback=callback)


# model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-5,
#             tensorboard_log=tensorboard_log)
# model.learn(total_timesteps=700000, callback=callback)

# model.save("ppo_full_env_v1.3_df")


######################### NEXT ITERATION ###################
# from wtw_FULL_env_draft_v2 import WtwEnvFull_v2

# env = WtwEnvFull_v2()

# tensorboard_log = "./sb3_wtw_FULL_v2"
# log_dir = "./sb3_wtw_ppo_FULL_v2/"

# env = Monitor(env, log_dir)
# callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

# model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-5,
#             tensorboard_log=tensorboard_log)
# model.learn(total_timesteps=500000, callback=callback)
# model.save("ppo_full_env_v2.1_df")




######################### NEXT ITERATION ###################
from wtw_FULL_env_draft_v3 import WtwEnvFull_v3

env = WtwEnvFull_v3()

tensorboard_log = "./sb3_wtw_FULL_v2"
log_dir = "./sb3_wtw_ppo_FULL_v2/"

# env = Monitor(env, log_dir)

#env = DummyVecEnv(env)

# wrap it
env = make_vec_env(lambda: env, n_envs=1)
env = VecNormalize(env, norm_obs = True, norm_reward = True)

#callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)

model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-6,
            tensorboard_log=tensorboard_log)
model.learn(total_timesteps=700000)
model.save("ppo_full_env_v3.6_norm_df")

# mean_reward, std_reward = evaluate_policy(
#     model, env, deterministic=True, n_eval_episodes=1000)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# mean_reward, std_reward = evaluate_policy(
#     model, env, deterministic=False, n_eval_episodes=1000)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


#best_model = PPO.load("best_model")
# visualizing agent intuition
#model = best_model
obs = env.reset()
for i in range(61):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(action)
    print(obs)
    if action != 0:
        print('action taken', action)
        print("in year", i)
    # print(info)
    if done:
        print("episode finished!")
        obs = env.reset()
