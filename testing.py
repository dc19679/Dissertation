
import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint

# Parameters
klm0 = 3.20e-2
klm = 8.30
thetaAtc = 11.65
etaAtc = 2.00
thetaTet = 30.00
etaTet = 2.00
glm = 1.386e-1
ktm0 = 1.19e-1
ktm = 2.06
thetaIptg = 9.06e-2
etaIptg = 2.00
thetaLac = 31.94
etaLac = 2.00
gtm = 1.386e-1
klp = 9.726e-1
glp = 1.65e-2
ktp = 1.170
gtp = 1.65e-2
aTc = 22
IPTG = 0.2

params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet,
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm,
                       klp, glp, ktp, gtp)

state = [0,0,0,0]
def deterministic(u, t, aTc, IPTG, args):
    """
    Determinsitic ODE system of the Genetic Toggle Switch
    """
    mRNAl, mRNAt, LacI, TetR = u

    aTc = 22
    IPTG = 0.2

    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (
            klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (
            ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


# def rk4(state, h, t, args):
#     """
#     Fourth Order Runge-Kutta method
#     This function updates a single RK4 step
#
#     :param args: arguments
#     :param state: The current state of the environment
#     :param t: Current time
#     :param h: Step size
#     """
#
#     k1 = deterministic(state, t, aTc, IPTG, params)
#     k2 = deterministic(state + np.array(k1) * (h / 2), t + h / 2, aTc, IPTG, params)
#     k3 = deterministic(state + np.array(k2) * (h / 2), t + h / 2, aTc, IPTG, params)
#     k4 = deterministic(state + np.array(k3) * h, t + h, aTc, IPTG, params)
#
#     return state + ((np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) / 6) * h
#

initial = [342, 224, 456, 40]
# time = np.linspace(0,10000,2)
# hello = rk4(initial, h=1, t=time, args=params)
# # print(hello)
# solution = odeint(deterministic, initial, time, args=(22,0.2,params,))
# # print(solution)
# print(solution[1])
#

for t_step in range(1):
    t = np.linspace(0, 1, 5)
    # args = (
    #     self.klm0, self.klm, self.thetaAtc, self.etaAtc, self.thetaTet, self.etaTet, self.glm, self.ktm0,
    #     self.ktm, self.thetaIptg, self.etaIptg, self.thetaLac, self.etaLac, self.gtm, self.klp, self.glp,
    #     self.ktp, self.gtp)
    y0 = initial
    sol = odeint(deterministic, y0, t, args=(90, 40, params,))
print(sol)
print(sol[-1])