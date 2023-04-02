from scipy.integrate import odeint
from matplotlib import pyplot as plt
from math import nan
import numpy as np

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


def deterministic(u, t, args):
    """
    Determinsitic ODE system of the Genetic Toggle Switch
    """
    mRNAl, mRNAt, LacI, TetR = u

    aTc = 100
    IPTG = 0.25

    klm0, klm, theta_atc, etaAtc, theta_Tet, eta_tet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (klm / (1 + ((TetR / thetaTet) / (1 + (aTc / theta_atc) ** etaAtc)) ** eta_tet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


# Different initial conditions to see how the system behaves
initial1 = [60.11025325,20.85153934,60.11025325,20.85153934]
initial2 = [550, 360, 404, 530]
initial3 = [440, 540, 0, 1]
initial4 = [9, 233, 180, 4]

params = [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp,
          glp, ktp, gtp]

def rk4(state, t, h,aTc,IPTG, args):
    """
    Fourth Order Runge-Kutta method
    This function updates a single RK4 step

    :param args: arguments
    :param state: The current state of the environment
    :param t: Current time
    :param h: Step size
    """
    # aTc = 100
    # IPTG = 0.25

    k1 = deterministic(state, t, aTc, IPTG, params)
    k2 = deterministic(state + np.array(k1) * (h / 2), t + h / 2, aTc, IPTG, params)
    k3 = deterministic(state + np.array(k2) * (h / 2), t + h / 2, aTc, IPTG, params)
    k4 = deterministic(state + np.array(k3) * h, t + h, aTc, IPTG, params)

    return state + ((np.array(k1) + 2 * np.array(k2) + 2 * np.array(k3) + np.array(k4)) / 6) * h


time = np.linspace(0, 10000, 1000001)
timesss = np.linspace(0, 20, 2)

solution = []
time = np.linspace(0,1000,1001)
for i in time:
    ans = rk4(initial2,i,h=1,args=(100,0.25,params,))
    solution.append(ans)



print(solution)


sol = odeint(deterministic, initial1, time, args=(params,))
sol2 = odeint(deterministic, [1222,3333,3333,5555], time, args=(params,))

# print("sol", sol)
print("sol", sol)

# sol3 = odeint(deterministic, initial3, time, args=(params,))
# sol4 = odeint(deterministic, initial4, time, args=(params,))


# plt.plot(time, sol[:, 2], 'g', label='Initial Cond 1')
# plt.plot(time, sol2[:, 2], 'r', label='Initial Cond 2')
# # plt.plot(time, sol3[:, 2], 'b', label='Initial Cond 3')
# # plt.plot(time, sol4[:, 2], 'y', label='Initial Cond 4')
# plt.xlabel('Time')
# plt.ylabel('LacI')
# plt.xlim([0,1000])
# plt.title('Cells stabilising to stable points ')
# plt.legend()
# plt.show()

###### STABLE POINTS FOR LACI = 56.11 (56) and 2575.87 (2575.87) #########

# plt.plot(time, sol[:, 3], 'g', label='Initial Cond 1')
# plt.plot(time, sol2[:, 3], 'r', label='Initial Cond 2')
# plt.plot(time, sol3[:, 3], 'b', label='Initial Cond 3')
# plt.plot(time, sol4[:, 3], 'y', label='Initial Cond 4')
# plt.xlabel('Time')
# plt.ylabel('TetR')
# plt.xlim([0,1000])
# plt.title('Cells stabilising to stable points ')
# plt.legend()
# plt.show()

###### STABLE POINTS FOR TETR = 72.8 (73) and 1072.62 (1073) #########