from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import PyDSTool as dst
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
# Define parameters
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
# Define parameters
params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp)

def deterministic(u, t, IPTG, args):
    """
    Determinsitic ODE system of the Genetic Toggle Switch
    """
    mRNAl, mRNAt, LacI, TetR = u

    aTc = 10

    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]

# Function to find steady state
def find_steady_state(IPTG, args):
    t = np.linspace(0, 1000, 1001)
    u0 = [2.55166047363230, 38.7108543679906, 102.155003051775, 1196.05604522200]
    u = odeint(deterministic, u0, t, args=(IPTG, args))
    return u[-1]

# Bifurcation plot
def bifurcation_plot(IPTG_range, args):
    steady_states = []

    for IPTG in IPTG_range:
        steady_state = find_steady_state(IPTG, args)
        steady_states.append(steady_state[-2:])

    steady_states = np.array(steady_states)
    # plt.plot(IPTG_range, steady_states[:, 0], 'r', label='LacI')
    plt.plot(IPTG_range, steady_states[:, 1], 'b', label='TetR')
    plt.xlabel('IPTG')
    plt.ylabel('Steady state')
    plt.legend()
    plt.title('Bifurcation Plot')
    plt.show()

IPTG_range = np.linspace(0, 1, 201)
bifurcation_plot(IPTG_range, params)
# # Define parameters
# klm0 = 3.20e-2
# klm = 8.30
# thetaAtc = 11.65
# etaAtc = 2.00
# thetaTet = 30.00
# etaTet = 2.00
# glm = 1.386e-1
# ktm0 = 1.19e-1
# ktm = 2.06
# thetaIptg = 9.06e-2
# etaIptg = 2.00
# thetaLac = 31.94
# etaLac = 2.00
# gtm = 1.386e-1
# klp = 9.726e-1
# glp = 1.65e-2
# ktp = 1.170
# gtp = 1.65e-2
#
#
#
#
# def deterministic(u, t, IPTG, args):
#     """
#     Determinsitic ODE system of the Genetic Toggle Switch
#     """
#     mRNAl, mRNAt, LacI, TetR = u
#
#     aTc = 10
#
#     klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args
#
#     dmRNAl_dt = klm0 + (klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
#     dmRNAt_dt = ktm0 + (ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
#     dLacI_dt = klp * mRNAl - glp * LacI
#     dTetR_dt = ktp * mRNAt - gtp * TetR
#
#     return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]
#
#
# # Create grid of IPTG values
# iptg_vals = np.linspace(0, 1, 201)
#
# params = [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp,
#           glp, ktp, gtp]
#
# time = np.linspace(0, 10000, num=10001)
# tetR_vals = []
#
# for i in iptg_vals:
#     initial_condition = np.random.uniform(low=1000, high=2000, size=(4,))
#     print(initial_condition)
#     sol = odeint(deterministic, initial_condition, time, args=(i,params,))
#     tetR_vals.append(sol[-1, 3])
#     # Iterates 21 times
#     # Each iteration there is 201 arrays of 1x4 array
#
# print(tetR_vals)
# plt.plot(iptg_vals, tetR_vals)
# plt.xlabel("IPTG values")
# plt.ylabel("TetR")
# plt.title("Bifurcation plots of tetR for varying values of IPTG")
# plt.show()
#
