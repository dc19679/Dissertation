from gts_env import GeneticToggleEnv
from stable_baselines3 import PPO
import os
from gts_env import GeneticToggleEnv
import numpy as np
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import time
from gts_env_diff_reward import GeneticToggleEnvs

import matplotlib.pyplot as plt


env = GeneticToggleEnvs()

#
#
# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"
#
# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)
#
# if not os.path.exists(logdir):
# 	os.makedirs(logdir)
#
# env = GeneticToggleEnv()
env.reset()
#
model = PPO(MlpPolicy, env, verbose=0)
#
# TIMESTEPS = 100
# iters = 0
# while iters < 3:
# 	iters += 1
# 	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
# 	model.save(f"{models_dir}/{TIMESTEPS*iters}")
#
def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# Random Agent, before training
# mean_reward_before_train = evaluate(model, num_episodes=100)



# Train the agent for 1000 steps
model.learn(total_timesteps=1000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# plt.scatter(env.lacI_values, env.tetR_values)
# plt.xlabel('LacI')
# plt.ylabel('TetR')
# plt.show()