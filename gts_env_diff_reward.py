import numpy as np
import gym
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class GeneticToggleEnvs(gym.Env):
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
    reigon, and a reward of +5 for being in that unstable reigon. Then there will be a -1
    reward for going away from the unstable reigon.

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
        """
        Initialise the GeneticToggleEnv environment
        """

        ### Define action and observation spaces ###

        # There are 4 possible actions the agent can take
        # --------> Increase both inducers
        # --------> Increase one of the inducers and decrease the other inducer
        # --------> Increase the other inducer and decrease the "other" one
        # --------> Decrease both inducers

        self.action_space = spaces.Discrete(4)

        # 4-Dimensional observation space representing the current state of the system,
        # In this case it is the concentration of mRNAl, mRNAt, LacI and TetR
        # The values range from [0, infinty]

        low = np.array([0, 0, 0, 0], dtype=np.float64)
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)

        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.float64)

        #

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
        self.aTc_range = aTc_range
        self.IPTG_range = IPTG_range

        # Target variable or the unstable reigon
        self.LacI_target_state = 520
        self.TetR_target_state = 280

        # Length of an episode
        self.episode_length = episode_length

        self.step_size = 1
        self.odeint_steps = 5

        self.lacI_values = []
        self.tetR_values = []

        # Store parameters in a tuple
        self.params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet,
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm,
                       klp, glp, ktp, gtp)

        # Initialise state
        # Indicating that the state has not been defined
        self.state = None
        self.prev_state = None

        # self.reset()

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

        print("state before:",self.state)
        # print("state before the ode", self.state)

        def deterministic(u, t, aTc, IPTG, args):
            """
            Determinsitic ODE system of the Genetic Toggle Switch
            """
            mRNAl, mRNAt, LacI, TetR = u

            klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

            dmRNAl_dt = klm0 + (
                    klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
            print("dmRNAl_dt",dmRNAl_dt)
            dmRNAt_dt = ktm0 + (
                    ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
            print("dmRNAt_dt",dmRNAt_dt)
            dLacI_dt = klp * mRNAl - glp * LacI
            dTetR_dt = ktp * mRNAt - gtp * TetR

            return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]

        # print("aTc level before rk4:", self.aTc)
        # print("IPTG level before rk4:", self.IPTG)
        for t_step in range(1):
            y0 = self.state
            def rk4(state, t, h, args):
                """
                Fourth Order Runge-Kutta method
                This function updates a single RK4 step

                :param args: arguments
                :param state: The current state of the environment
                :param t: Current time
                :param h: Step size
                """
                print("aTc in rk4", self.aTc)
                print("IPTG in rk4", self.IPTG)

                k1 = deterministic(state, t, self.aTc, self.IPTG, self.params)
                # print("k1", k1)
                k2 = deterministic(state + np.array(k1) * (h / 2), t + h / 2, self.aTc, self.IPTG, self.params)
                k3 = deterministic(state + np.array(k2) * (h / 2), t + h / 2, self.aTc, self.IPTG, self.params)
                k4 = deterministic(state + np.array(k3) * h, t + h, self.aTc, self.IPTG, self.params)

                return state + ((np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) / 6) * h
            print("y0:", y0)
            print("h:", self.h)
            self.state = rk4(y0, self.time, self.h, self.params)
            print("state after:", self.state)

        # Returns the current state of the environment using the previous environment state
        # as the initial condition

        # Update the state using the RK4 integration method
        # h = 1 so it iterates one step at a time
        # print("LacI after the ode:",self.state[2])
        # print("TetR after the ode:", self.state[3])
        # print("LacI",self.state[2])
        # print("TetR",self.state[3])

        # Log the trajectory of the single cell in the LacI and TetR space
        self.lacI_values.append(self.state[2])
        self.tetR_values.append(self.state[3])

        # Initialise reward to 0
        # print("aTc:", self.aTc)
        # print("IPTG:", self.IPTG)

        # At each step, make sure the solver is taking number of steps in each step

        # Calculate reward
        lacI_diff = abs(self.LacI_target_state - self.state[2])
        tetR_diff = abs(self.TetR_target_state - self.state[3])

        # distance_from_target = np.sqrt(lacI_diff ** 2 + tetR_diff ** 2)

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

        reward = 0.0
        # Check if episode is over
        done = False
        # print("episode length",self.episode_length)
        if self.episode_length is not None:
            self.episode_length -= 1
            if self.episode_length == 0:
                done = True

        # observation = self.state
        # Have more observation for the learning aspects - aTc and IPTG, other state variables

        # Calculate additional information to return
        lacI = self.state[2]
        tetR = self.state[3]

        lacI_target = self.LacI_target_state
        tetR_target = self.TetR_target_state

        error_distance_LacI = abs(lacI - lacI_target)
        error_distance_TetR = abs(tetR - tetR_target)

        info = {
            'aTc concentration': self.aTc,
            'IPTG concentration': self.IPTG,
            'lacI level': lacI,
            'tetR level': tetR,
            'Abs distance of lacI and lacI target': error_distance_LacI,
            'Abs distance of tetR and tetR target': error_distance_TetR,
        }

        observation = self.state


        # Return observation, reward, and info
        return observation, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """

        # Define initial state
        self.state = np.random.uniform(low=0, high=3000, size=(4,))

        # define intial time
        self.time = 0

        # Define step size
        self.h = 1

        # Update environment variables
        self.aTc = 20
        self.IPTG = 0.25

        self.LacI_target_state = 520
        self.TetR_target_state = 280

        # reset LacI and TetR value lists
        self.lacI_values = []
        self.tetR_values = []


        # Reset episode length counter
        self.episode_length = 2000

        # Check if the observation space contains the current state
        if self.observation_space.contains(self.state):
            # Return the current state as a NumPy array
            return np.array(self.state)

    def render(self, mode='human'):
        if self.state is None:
            return

        if mode == 'human':
            # plot trajectory in LacI-TetR space
            plt.plot(self.lacI_values, self.tetR_values)
            plt.xlabel('LacI')
            plt.ylabel('TetR')
            plt.title('Genetic Toggle Switch Trajectory')
            plt.show()

        elif mode == 'rgb_array':
            # return an empty image
            return np.zeros((300, 300, 3), dtype=np.uint8)

env = GeneticToggleEnvs()
model = PPO(ActorCriticPolicy, env, verbose=2)

num_episodes = 1
episode_length = 2000
total_timesteps = num_episodes * episode_length

model.learn(total_timesteps=total_timesteps)

# time_steps = range(len(env.lacI_values))
# print("time steps",len(env.lacI_values))
# print("range of the length", range(len(env.lacI_values)))
#
# plt.scatter(time_steps, env.lacI_values, label='LacI')
# plt.scatter(time_steps, env.tetR_values, label='TetR')
# plt.xlabel('Time Step')
# plt.ylabel('Expression Level')
# plt.title('LacI and TetR Trajectories')
# plt.legend()
# plt.show()
print("Length of lacI_values:", len(env.lacI_values))
print("Length of tetR_values:", len(env.tetR_values))
env.render()