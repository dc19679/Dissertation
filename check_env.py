import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.env_checker import check_env
from gts_env import GeneticToggleEnv

env = GeneticToggleEnv()

check_env(env)

print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
# Discrete(2) means that there is two discrete actions
print("Action space:", env.action_space)

# The reset method is called at the beginning of an episode
obs = env.reset()
# Sample a random action
action = env.action_space.sample()
print("Sampled action:", action)
obs, reward, done, info = env.step(action)

print(obs.shape, reward, done, info)