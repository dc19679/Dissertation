import numpy as np
import gym
from gym import spaces
from scipy.integrate import odeint

class GeneticToggleEnv(gym.Env):
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
    reigon, and a reward of +5 for being in that unstable reigon. Then there will be a 0
    reward for going away from the unstable reigon.

    Calculate the error as the absolute distance between the target level and the current
    level of the LacI and TetR

    ### Episode End ###

    The episode will end if any of the following occurs:

    1. Termination: If the cell is not around the unstable reigon for a long period of time
    2. Termination: If the cell maintains around the untable reigon for a good amount of time

    """

    def __init__(self, aTc = 20.0, IPTG = 0.25, klm0=3.20e-2, klm=8.30, thetaAtc=11.65, etaAtc=2.00, thetaTet=30.00, 
                 etaTet=2.00, glm=1.386e-1, ktm0=1.19e-1, ktm=2.06, thetaIptg=9.06e-2, 
                 etaIptg=2.00, thetaLac=31.94, etaLac=2.00, gtm=0.1, klp=0.1, glp=0.1, ktp=0.1,
                 gtp=0.1, aTc_range=[0, 100], IPTG_range=[0, 1], target_state=[750, 280], episode_length=100):
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

        low = np.array([0, 0, 0,0], dtype=np.float64)
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
        self.target_state = target_state

        # Length of an episode
        self.episode_length = episode_length

        # Store parameters in a tuple
        self.params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, 
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, 
                       klp, glp, ktp, gtp)
        
        # Initialise state
        # Holds information about the current state of the environment and is updated
        # with each step taken by the agent
        self.state = None

        # self.reset()

    def step(self, action):
        """
        Execute a single time step in the environment
        """

        assert self.state is not None, "Call reset before using step method." 

        # The actions the agent can perform (0,1,2,3)
        if action == 0:
            
            # Increase aTc and IPTG
            self.aTc += 4
            self.IPTG += 0.05
        elif action == 1:
            
            # Increase aTc but decrease IPTG
            self.aTc += 4
            self.IPTG -= 0.05
        elif action == 2:
            
            # Decrease aTc but increase IPTG
            self.aTc -= 4
            self.IPTG += 0.05
        else:
            
            # Decrease aTc and IPTG
            self.aTc -= 4
            self.IPTG -= 0.05


        def deterministic (u, t, aTc, IPTG, args):
            """
            Determinsitic ODE system of the Genetic Toggle Switch
            """
            mRNAl, mRNAt, LacI, TetR = u

            klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

            dmRNAl_dt = klm0 + (klm/(1 + ((TetR/thetaTet) / (1 + (aTc / thetaAtc )**etaAtc))**etaTet)) - glm * mRNAl
            dmRNAt_dt = ktm0 + (ktm/(1 + ((LacI/thetaLac) / (1 + (IPTG / thetaIptg )**etaIptg))**etaLac)) - gtm * mRNAt
            dLacI_dt = klp*mRNAl - glp*LacI
            dTetR_dt = ktp*mRNAt - gtp*TetR

            return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]
        
        
        
        def rk4(state, t, h, args):
            """
            Fourth Order Runge-Kutta method
            This function updates a single RK4 step

            :param state: The current state of the environment
            :param t: Current time
            :param h: Step size
            """


            k1 = deterministic(state, t, self.aTc, self.IPTG, self.params)
            k2 = deterministic(state + np.array(k1) * (h / 2), t + h / 2, self.aTc, self.IPTG, self.params)
            k3 = deterministic(state + np.array(k2) * (h / 2), t + h / 2, self.aTc, self.IPTG, self.params)
            k4 = deterministic(state + np.array(k3) * h, t + h, self.aTc, self.IPTG, self.params)

            return (state + ((np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) / 6) * h)
        # Returns the current state of the environment using the previous environment state
        # as the initial condition
        
        # Update the state using the RK4 integration method
        self.state = rk4(self.state, self.time, self.h, self.params)

        # Initialise reward to 0
        reward = 0

        # Calculate reward
        if self.target_state is not None:

            # If the LacI and TetR values are in the unstable state then give a reward
            if (abs(self.state[2] - self.target_state[0])) < 5 and (abs(self.state[3] - self.target_state[1]) < 5):
                reward += -(abs(self.state[2] - self.target_state[0]) + abs(self.state[3] - self.target_state[1]))
                # print("Reward")
            # If the LacI and TetR are not in the unstable state then take away a reward
            else:
                reward -= -(abs(self.state[2] - self.target_state[0]) + abs(self.state[3] - self.target_state[1]))
                # print("Penalty")
                
        else:
            reward = 0

        # Check if episode is over
        done = False
        if self.episode_length is not None:
            self.episode_length -= 1
            if self.episode_length == 0:
                done = True

        # observation = self.state
        # Have more observation for the learning aspects - aTc and IPTG, other state variables

        # Calculate additional information to return
        lacI = self.state[2]
        tetR = self.state[3]

        lacI_target = self.target_state[0]
        tetR_target = self.target_state[1]

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
        # Define initial state values
        mRNAl_0 = 0.0
        mRNAt_0 = 0.0
        LacI_0 = 0.0
        TetR_0 = 0.0

        # Define initial state
        self.state = np.random.uniform(low=0, high=1000, size=(4,))
        
        # define intial time
        self.time = 0

        # Define step size
        self.h = 0.01

        # Update environment variables
        self.aTc = 20
        self.IPTG = 0.25

        # Reset episode length counter
        self.current_step = 0

        # Check if the observation space contains the current state
        if self.observation_space.contains(self.state):
            # Return the current state as a NumPy array
            return np.array(self.state)





