import numpy as np
import gym
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.cm as cm
# from stable_baselines3.common.policies import MultiInputPolicy


class GeneticToggle(gym.Env):


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

        print("LacI",self.state[2])
        print("TeR",self.state[3])


        # Initialise reward to 0
        reward = 0

        # Calculate reward
        lacI_diff = abs(self.LacI_target_state - self.state[2])
        tetR_diff = abs(self.TetR_target_state - self.state[3])

        if lacI_diff < 20:
            reward = 1000
            print("LacI in 20")
        elif 20 <= lacI_diff < 50:
            reward = 100
            print("LacI in 50")
        elif 50 <= lacI_diff < 100:
            reward = 10
            print("LacI in 100")
        elif lacI_diff > 100:
            reward = -100
        elif tetR_diff < 20:
            reward = 1000
            print("TetR in 20")
        elif 20 <= tetR_diff < 50:
            reward = 100
            print("TetR in 50")
        elif 50 <= tetR_diff < 100:
            reward = 10
            print("TetR in 50")
        elif tetR_diff > 100:
            reward = -100


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
        self.state = np.random.uniform(low=0, high=3000, size=(4,))
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
                    lw=2,
                    marker='o'
                )
            circle = plt.Circle((520, 280), 4, fill=False)
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
model = PPO(ActorCriticPolicy, env, verbose=2)

num_episodes = 10000
episode_length = 3000
total_timesteps = num_episodes * episode_length

model.learn(total_timesteps=total_timesteps)

print("Length of lacI_values:", len(env.lacI_values))
print("Length of tetR_values:", len(env.tetR_values))
env.render()