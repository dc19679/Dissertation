from gts_env import GeneticToggleEnv
from gts_env_diff_reward import GeneticToggleEnvs
from gts_env_ode import GeneticToggleEnviro
from stable_baselines3 import PPO
import numpy as np

from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt


env = GeneticToggleEnviro()
model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4)

num_episodes = 1
episode_length = 1000
total_timesteps = num_episodes * episode_length

model.learn(total_timesteps=total_timesteps)

time_steps = range(len(env.lacI_values))
print("time steps",time_steps)
print("range of the length", range(len(env.lacI_values)))

plt.plot(time_steps, env.lacI_values, label='LacI')
plt.plot(time_steps, env.tetR_values, label='TetR')
plt.xlabel('Time Step')
plt.ylabel('Expression Level')
plt.title('LacI and TetR Trajectories')
plt.legend()
plt.show()