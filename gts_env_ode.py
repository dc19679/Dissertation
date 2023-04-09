import numpy as np
import gym
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.policies import MultiInputPolicy


class GeneticToggleEnviro(gym.Env):

    def __init__(self, aTc=20.0, IPTG=0.25, klm0=3.20e-2, klm=8.30, thetaAtc=11.65, etaAtc=2.00, thetaTet=30.00,
                 etaTet=2.00, glm=1.386e-1, ktm0=1.19e-1, ktm=2.06, thetaIptg=9.06e-2,
                 etaIptg=2.00, thetaLac=31.94, etaLac=2.00, gtm=0.1, klp=0.1, glp=0.1, ktp=0.1,
                 gtp=0.1, aTc_range=[0, 100], IPTG_range=[0, 1], LacI_target_state=520, TetR_target_state=280, episode_length=5000):

        # self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([100, 1, 100, 1]), dtype=np.float64)

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
        self.aTc_range = aTc_range
        self.IPTG_range = IPTG_range

        # Target variable or the unstable reigon
        self.LacI_target_state = LacI_target_state
        self.TetR_target_state = TetR_target_state

        # Length of an episode
        self.episode_length = episode_length

        self.lacI_values = []
        self.tetR_values = []

        self.step_size = 1
        self.odeint_steps = 2

        # Store parameters in a tuple
        self.params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet,
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm,
                       klp, glp, ktp, gtp)

        # Initialise state
        # Indicating that the state has not been defined
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
            self.IPTG -= 0.1
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

        if self.aTc > self.aTc_range[1]:
            self.aTc = 100
        elif self.aTc < self.aTc_range[0]:
            self.aTc = 0
        elif self.IPTG > self.IPTG_range[1]:
            self.IPTG = 1
        elif self.IPTG < self.IPTG_range[0]:
            self.IPTG = 0

        # self.aTc += action[0]
        # self.IPTG += action[1]
        # self.aTc -= action[2]
        # self.IPTG -= action[3]

        # print("aTc before the function:", self.aTc)
        # print("IPTG before the function",self.IPTG)


        def deterministic(u, t, aTc, IPTG, args):
            """
            Determinsitic ODE system of the Genetic Toggle Switch
            """
            mRNAl, mRNAt, LacI, TetR = u

            klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args
            # print("aTc in the function:", self.aTc)
            # print("IPTG in the function", self.IPTG)
            dmRNAl_dt = klm0 + (
                    klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
            dmRNAt_dt = ktm0 + (
                    ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
            dLacI_dt = klp * mRNAl - glp * LacI
            dTetR_dt = ktp * mRNAt - gtp * TetR

            return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


        # Returns the current state of the environment using the previous environment state as the initial condition
        for t_step in range(1):
            t = np.linspace(0, self.step_size, self.odeint_steps)
            args = (
            self.klm0, self.klm, self.thetaAtc, self.etaAtc, self.thetaTet, self.etaTet, self.glm, self.ktm0,
            self.ktm, self.thetaIptg, self.etaIptg, self.thetaLac, self.etaLac, self.gtm, self.klp, self.glp,
            self.ktp, self.gtp)
            y0 = self.state
            sol = odeint(deterministic, y0, t, args=(self.aTc,self.IPTG,self.params,))

        # print("aTc",self.aTc)
        # print("IPTG",self.IPTG)

        self.state = sol[-1]

        print("LacI",self.state[2])
        print("TeR",self.state[3])

        # Log the trajectory of the single cell in the LacI and TetR space
        self.lacI_values.append(self.state[2])
        self.tetR_values.append(self.state[3])
        # print(self.lacI_values)
        # print(self.tetR_values)

        # Initialise reward to 0
        reward = 0

        # Calculate reward
        lacI_diff = abs(self.LacI_target_state - self.state[2])
        tetR_diff = abs(self.TetR_target_state - self.state[3])

        # distance_from_target = np.sqrt(lacI_diff ** 2 + tetR_diff ** 2)

        if lacI_diff < 20:
            reward = 1000
        elif 20 <= lacI_diff < 50:
            reward = 100
        elif 50 <= lacI_diff < 100:
            reward = 10
        elif lacI_diff > 100:
            reward = -100
        elif tetR_diff < 20:
            reward = 1000
        elif 20 <= tetR_diff < 50:
            reward = 100
        elif 50 <= tetR_diff < 100:
            reward = 10
        elif tetR_diff > 100:
            reward = -100

        # if distance_from_target < 10:
        #     # Apply a smaller penalty for being close to the target state
        #     reward = 1000 - 10 * distance_from_target
        # else:
        #     # Apply a larger penalty for being far from the target state
        #     reward =  -distance_from_target

        # print("reward:",reward)
        # reward = -distance_from_target
        # if distance_from_target < 10:
        #     # Apply a penalty for moving too far from the target state
        #     reward = 1000
        # elif 10 <= distance_from_target < 40:
        #     reward = 10
        # elif 40 <= distance_from_target < 60:
        #     reward = 1
        # elif 60 <= distance_from_target < 90:
        #     reward = -1
        # elif 90 <= distance_from_target < 150:
        #     reward = -10
        # elif distance_from_target >= 150:
        #     reward = -100
        # Check if episode is over
        done = False

        if self.episode_length is not None:
            self.episode_length -= 1
            if self.episode_length == 0:
                done = True

        # Calculate additional information to return
        lacI = self.state[2]
        tetR = self.state[3]

        # lacI_target = self.target_state[0]
        # tetR_target = self.target_state[1]
        #
        # error_distance_LacI = abs(lacI - lacI_target)
        # error_distance_TetR = abs(tetR - tetR_target)

        # info = {
        #     'aTc concentration': self.aTc,
        #     'IPTG concentration': self.IPTG,
        #     'lacI level': lacI,
        #     'tetR level': tetR,
        #     'Abs distance of lacI and lacI target': error_distance_LacI,
        #     'Abs distance of tetR and tetR target': error_distance_TetR,
        # }
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

        # reset LacI and TetR value lists
        self.lacI_values = []
        self.tetR_values = []

        # Reset episode length counter
        self.episode_length = 5000

        self.step_size = 1
        self.odeint_steps = 2

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
            plt.plot(self.lacI_values, self.tetR_values)
            plt.xlabel('LacI')
            plt.ylabel('TetR')
            plt.title('Genetic Toggle Switch Trajectory')
            plt.show()

        elif mode == 'rgb_array':
            # return an empty image
            return np.zeros((300, 300, 3), dtype=np.uint8)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs, net_arch=[64, 64])



env = GeneticToggleEnviro()
model = PPO(ActorCriticPolicy, env, verbose=1, ent_coef=0.35,clip_range=0.2)

num_episodes = 10
episode_length = 3000
total_timesteps = num_episodes * episode_length

model.learn(total_timesteps=total_timesteps)

time_steps = range(len(env.lacI_values))
print("time steps",len(env.lacI_values))
print("range of the length", range(len(env.lacI_values)))

plt.scatter(time_steps, env.lacI_values, label='LacI')
plt.scatter(time_steps, env.tetR_values, label='TetR')
plt.xlabel('Time Step')
plt.ylabel('Expression Level')
plt.title('LacI and TetR Trajectories')
plt.legend()
plt.show()