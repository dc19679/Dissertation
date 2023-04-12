
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


params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet,
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm,
                       klp, glp, ktp, gtp)

def deterministic(u, t, aTc, IPTG, args):
    """
    Determinsitic ODE system of the Genetic Toggle Switch
    """
    mRNAl, mRNAt, LacI, TetR = u
    # print("aTc",aTc)
    # print("IPTG",IPTG)

    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (
            klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (
            ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]



initial = [0, 0, 0,0 ]
# time = np.linspace(0,10000,2)
# hello = rk4(initial, h=1, t=time, args=params)
# # print(hello)
# solution = odeint(deterministic, initial, time, args=(22,0.2,params,))
# # print(solution)
# print(solution[1])
#

for t_step in range(1):
    t = np.linspace(0, 1, 5)

    y0 = initial
    sol = odeint(deterministic, y0, t, args=(30, 0.15, params,))

# print(sol)

hi = np.random.randint(low=0, high=1001, size=(1, 4))

# print(hi)

a = [2, 3]
b = [77, 5]

c = [abs(b_i - a_i) for a_i, b_i in zip(a, b)]
print(c)




# state before: [0 0 0 0]
# aTc 30
# IPTG 0.15
# state after: [[0.         0.         0.         0.        ]
#  [2.04732521 0.53541983 0.24999609 0.0786489 ]
#  [4.02492413 1.05259196 0.98723077 0.31058207]
#  [5.93516452 1.55207452 2.19306858 0.68992585]
#  [7.78032318 2.03431179 3.84951428 1.210978  ]]