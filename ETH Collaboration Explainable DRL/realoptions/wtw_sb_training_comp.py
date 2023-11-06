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
from wtw_env_draft import WtwEnvSimple
from wtw_env_draft_v1  import WtwEnvSimple_v1

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
              #if self.verbose > 0:
                #print(f"Num timesteps: {self.num_timesteps}")
                #print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  #if self.verbose > 0:
                    #print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True
    
    
    
    

# env = WtwEnvSimple()
# tensorboard_log="./sb3_wtw_test_v0/"
# log_dir = "./sb3_wtw_ppo_test_v0/"

# env = Monitor(env, log_dir)
# callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
# model = PPO('MlpPolicy', env, verbose=1, learning_rate = 3e-4, tensorboard_log= tensorboard_log)
# model.learn(total_timesteps=200000, callback=callback)


# ### double check env works with appending options taken
# ### IT WORKS MUCH BETTER, WILL UPDATE OTHERS ACCORDINGLY

# env = WtwEnvSimple_v0()
# tensorboard_log="./sb3_wtw_test_v1/"
# log_dir = "./sb3_wtw_ppo_test_v0/"

# env = Monitor(env, log_dir)
# callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
# model = PPO('MlpPolicy', env, verbose=1, learning_rate = 3e-4, tensorboard_log= tensorboard_log)
# model.learn(total_timesteps=100000, callback=callback)





# HERE TRY ENV WITH ALSO WTW CAPACITY INCREASE OPTION

env = WtwEnvSimple_v1()
tensorboard_log="./sb3_wtw_agents_test_v1" # we leave on same tensorboard to see improvements in performance visually
log_dir = "./sb3_wtw_ppo_test_v1/"

env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)


model_ppo = PPO('MlpPolicy', env, verbose=1, learning_rate = 3e-5, tensorboard_log= tensorboard_log)
model_ppo.learn(total_timesteps=300000, callback=callback)
model_ppo.save("ppo_wtw_v1")


model_a2c = A2C("MlpPolicy", env, verbose=1, tensorboard_log= tensorboard_log)
model_a2c.learn(total_timesteps=300000, callback=callback)
model_a2c.save("a2c_wtw_v1")

model_dqn = DQN("MlpPolicy", env, verbose=1, tensorboard_log= tensorboard_log)
model_dqn.learn(total_timesteps=300000, callback=callback)
model_dqn.save("dqn_wtw_v1")



model = model_ppo

mean_reward, std_reward = evaluate_policy(model, env, deterministic= True, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

mean_reward, std_reward = evaluate_policy(model, env, deterministic= False, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model = model_a2c

mean_reward, std_reward = evaluate_policy(model, env, deterministic= True, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

mean_reward, std_reward = evaluate_policy(model, env, deterministic= False, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model = model_dqn

mean_reward, std_reward = evaluate_policy(model, env, deterministic= True, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

mean_reward, std_reward = evaluate_policy(model, env, deterministic= False, n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")




# mean_reward, std_reward = evaluate_policy(model, env, deterministic= True, n_eval_episodes=100)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# mean_reward, std_reward = evaluate_policy(model, env, deterministic= False, n_eval_episodes=100)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# #### visualizing agent intuition
# obs = env.reset()
# for i in range(61):
#     action, _states = model.predict(obs, deterministic= False)
#     obs, rewards, done, info = env.step(action)
#     print(obs)
#     if action != 0:
#         print('action taken' , action)
#     #print(info)
#     if done:
#         print("episode finished!")
#         obs = env.reset()


