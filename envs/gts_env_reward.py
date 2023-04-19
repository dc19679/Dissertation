import numpy as np
import gym
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.cm as cm



class GeneticToggle(gym.Env):
    """
    Custom gym environment for a genetic toggle switch

      ### Action Space ###

  The action is a ndarray with shape () which can take the values (), indicating the
  concentration of aTc and IPTG
    | Num |               Action               |
    |-----|------------------------------------|
    |  0  | Increase aTc and IPTG              |
    |  1  | Increase aTc and decrease IPTG     |
    |  2  | Decrease IPTG and increase aTc     |
    |  3  | Decrease IPTG and IPTG             |



  ### Observation Space ###

  The observation space is a ndarray with shape (), with the values corressponding to the
  concentrations of aTc and IPTG and the levels if LacI and TetR
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | mRNa LacI             |          0          |        10000      |
    | 1   | mRNA TetR             |          0          |        10000      |
    | 2   | Level of LacI         |          0          |        3000       |
    | 3   | Level of TetR         |          0          |        2000       |


    ### Rewards ###

    Since the goal is to keep one cell about the unstable equilibrium state for as long as
    possible, a reward of +1 will be given for every step that is towards the unstable
    region, and a reward of +5 for being in that unstable region. Then there will be a 0
    reward for going away from the unstable region.
    Calculate the error as the absolute distance between the target level and the current
    level of the LacI and TetR


    ### Episode End ###

    The episode will end if any of the following occurs:
    1. Termination: If the cell is not around the unstable reigon for a long period of time
    2. Termination: If the cell maintains around the untable reigon for a good amount of time
    """

    def __init__(self,aTc = 20,IPTG = 0.25, klm0=3.20e-2, klm=8.30, thetaAtc=11.65, etaAtc=2.00, thetaTet=30.00,
                 etaTet=2.00, glm=1.386e-1, ktm0=1.19e-1, ktm=2.06, thetaIptg=9.06e-2,
                 etaIptg=2.00, thetaLac=31.94, etaLac=2.00, gtm=1.386e-1, klp=9.726e-1, glp=1.65e-2, ktp=1.170,
                 gtp=1.65e-2, aTc_range=[0, 100], IPTG_range=[0, 1], LacI_target_state=520, TetR_target_state=280, episode_length=1000):


        self.action_space = spaces.Discrete(4)

        low = np.array([0, 0, 0, 0], dtype=np.float64)
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)

        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.float64)


        # Set parameters
        self.aTc = aTc
        self.IPTG = IPTG
        self.klm0 = klm0
        self.klm = klm
        self.thetaAtc = thetaAtc
        self.etaAtc = etaAtc
        self.thetaTet = thetaTet
        self.etaTet = etaTet
        self.glm = glm
        self.ktm0 = ktm0
        self.ktm = ktm
        self.thetaIptg = thetaIptg
        self.etaIptg = etaIptg
        self.thetaLac = thetaLac
        self.etaLac = etaLac
        self.gtm = gtm
        self.klp = klp
        self.glp = glp
        self.ktp = ktp
        self.gtp = gtp

        # Define the inducers range
        self.aTc_range = [0,100]
        self.IPTG_range = [0,1]

        # Target variable or the unstable reigon
        self.LacI_target_state = LacI_target_state
        self.TetR_target_state = TetR_target_state

        self.step_counter = 0

        # Length of an episode
        self.episode_length = episode_length

        self.lacI_values = []
        self.tetR_values = []
        self.aTc_values = []
        self.IPTG_values = []
        self.euclidean_distances = []

        self.step_size = 1
        self.odeint_steps = 5

        # Store parameters in a tuple
        self.params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet,
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm,
                       klp, glp, ktp, gtp)

        # Initialise state
        self.state = None


    def step(self, action):
        """
        Execute a single time step in the environment
        """

        assert self.state is not None, "Call reset before using step method."

        # The actions the agent can perform (0,1,2,3)
        if action == 0:
            # Increase aTc and IPTG, but only if they have not reached their maximum value
            self.aTc += 10
            self.IPTG += 0.1
        elif action == 1:
            # Increase aTc but decrease IPTG, but only if they have not reached their maximum value
            self.aTc += 10
            self.IPTG -= 0.1
        elif action == 2:
            # Decrease aTc but increase IPTG, but only if they have not reached their maximum value
            self.aTc -= 10
            self.IPTG += 0.1
        else:
            # Decrease aTc and IPTG, but only if they have not reached their maximum value
            self.aTc -= 10
            self.IPTG -= 0.1

        # Check if aTc and IPTG are within their valid range
        if self.aTc > self.aTc_range[1]:
            self.aTc = self.aTc_range[1]
        elif self.aTc < self.aTc_range[0]:
            self.aTc = self.aTc_range[0]

        if self.IPTG > self.IPTG_range[1]:
            self.IPTG = self.IPTG_range[1]
        elif self.IPTG < self.IPTG_range[0]:
            self.IPTG = self.IPTG_range[0]

        self.lacI_values.append(self.state[2])
        self.tetR_values.append(self.state[3])

        self.aTc_values.append(self.aTc)
        self.IPTG_values.append(self.IPTG)




        def deterministic(u, t, aTc, IPTG, args):
            """
            Determinsitic ODE system of the Genetic Toggle Switch
            """
            mRNAl, mRNAt, LacI, TetR = u

            klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

            dmRNAl_dt = klm0 + (
                    klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
            dmRNAt_dt = ktm0 + (
                    ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
            # print("dmRNAt_dt",dmRNAt_dt)
            dLacI_dt = klp * mRNAl - glp * LacI
            dTetR_dt = ktp * mRNAt - gtp * TetR

            return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


        # Returns the current state of the environment using the previous environment state as the initial condition
        for t_step in range(1):
            t = np.linspace(0, self.step_size, self.odeint_steps)
            y0 = self.state
            sol = odeint(deterministic, y0, t, args=(self.aTc,self.IPTG,self.params,))
            self.step_counter += 1

        self.state = sol[-1]

        # Initialise reward to 0
        reward = 0

        # Calculate reward
        lacI_diff = abs(self.LacI_target_state - self.state[2])
        tetR_diff = abs(self.TetR_target_state - self.state[3])

        euclidean = np.sqrt(lacI_diff**2 + tetR_diff**2)
        self.euclidean_distances.append(euclidean)

        reward = -euclidean

        done = False
        # print(self.episode_length)
        if self.episode_length is not None:
            self.episode_length -= 1
            if self.episode_length == 0:
                # print("episode done")
                done = True

        info = {}
        observation = self.state


        # Return observation, reward, and info
        return observation, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """

        # Define initial state
        self.state = np.random.uniform(low=0, high=1000, size=(4,))
        # Update environment variables
        self.aTc = 20
        self.IPTG = 0.25

        self.LacI_target_state = 520
        self.TetR_target_state = 280

        self.aTc_range = [0,100]
        self.IPTG_range = [0,1]

        # reset LacI and TetR value lists
        self.lacI_values = []
        self.tetR_values = []

        self.aTc_values = []
        self.IPTG_values = []
        self.euclidean_distances = []

        self.step_counter = 0


        # Reset episode length counter
        self.episode_length = 1000

        self.step_size = 1
        self.odeint_steps = 5

        # Check if the observation space contains the current state
        if self.observation_space.contains(self.state):
            # Return the current state as a NumPy array
            return np.array(self.state)
        else:
            # If the state is not within the observation space, return an array of zeros
            return np.zeros_like(self.state)



    def render(self, mode='human'):
        if self.state is None:
            return

        if mode == 'human':
            # plot trajectory in LacI-TetR space

            num_time_steps = len(self.lacI_values)

            # Set up the plot
            fig, ax = plt.subplots()

            # Choose a colormap for the segments
            colormap = cm.get_cmap('viridis', num_time_steps)

            # Plot each line segment with a different color
            for i in range(1, num_time_steps):
                ax.plot(
                    self.lacI_values[i - 1:i + 1],
                    self.tetR_values[i - 1:i + 1],
                    color=colormap(i),
                    lw=1,
                    # marker='o'
                )
            circle = plt.Circle((520, 280), 8, fill=True)
            ax.add_artist(circle)

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=num_time_steps - 1))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label("Time step")

            # Set axis labels
            ax.set_xlabel("LacI")
            ax.set_ylabel("TetR")

            # Set title
            ax.set_title("Cell Path in State Space")

            # Display the plot
            plt.show()

        elif mode == 'rgb_array':
            # return an empty image
            return np.zeros((300, 300, 3), dtype=np.uint8)





env = GeneticToggle()
model = PPO(ActorCriticPolicy, env, verbose=2,gamma=1.0)

num_episodes = 1000
episode_length = 1000
total_timesteps = num_episodes * episode_length

model.learn(total_timesteps=total_timesteps)


mean_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0

    for _ in range(episode_length):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            break

    mean_rewards.append(episode_reward / episode_length)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Different initial conditions
# 2D graph od ED over time
# 2D plot of the cell path
# Error plots
# How generalising it is/ overfitting


window_size = 10  # Adjust the window size according to your needs
moving_avg_rewards = moving_average(mean_rewards, window_size)

print("moving average rewards:", moving_avg_rewards)
print("aTc values:", env.aTc_values)
print("Length of aTc values",len(env.aTc_values))
print("IPTG values:",env.IPTG_values)
print("Length of IPTG values:", len(env.IPTG_values))
print("Euclidan distance:", env.euclidean_distances)
print("Length of ED:", len(env.euclidean_distances))


plt.figure()
# plt.plot(mean_rewards, label='Mean Reward')
plt.plot(moving_avg_rewards, label='Moving Average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.linspace(0,len(env.aTc_values), len(env.aTc_values)), env.aTc_values)
plt.xlabel("time")
plt.ylabel("aTc")
plt.title("aTc over time")
plt.show()

plt.figure()
plt.plot(np.linspace(0,len(env.IPTG_values),len(env.IPTG_values)), env.IPTG_values)
plt.xlabel("time")
plt.ylabel("IPTG")
plt.title("IPTG over time")
plt.show()

plt.figure()
plt.plot(np.linspace(0,len(env.euclidean_distances), len(env.euclidean_distances)), env.euclidean_distances)
plt.xlabel("time")
plt.ylabel("ED")
plt.title("ED over time")
plt.show()

env.render()