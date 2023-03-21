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

    aTc = 20
    IPTG = 0.25

    klm0, klm, theta_atc, etaAtc, theta_Tet, eta_tet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (klm / (1 + ((TetR / thetaTet) / (1 + (aTc / theta_atc) ** etaAtc)) ** eta_tet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


# Different initial conditions to see how the system behaves
initial1 = [0, 0, 0, 0]
initial2 = [20, 30, 40, 50]
initial3 = [50, 50, 0, 100]
initial4 = [90, 23, 180, 4]

params = [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp,
          glp, ktp, gtp]
time = np.linspace(0, 1500, 1501)

sol = odeint(deterministic, initial1, time, args=(params,))
sol2 = odeint(deterministic, initial2, time, args=(params,))
sol3 = odeint(deterministic, initial3, time, args=(params,))
sol4 = odeint(deterministic, initial4, time, args=(params,))

plt.plot(time, sol[:, 2], 'g', label='Initial Cond 1')
plt.plot(time, sol2[:, 2], 'r', label='Initial Cond 2')
plt.plot(time, sol3[:, 2], 'b', label='Initial Cond 3')
plt.plot(time, sol4[:, 2], 'y', label='Initial Cond 4')
plt.xlabel('Time')
plt.ylabel('LacI')
plt.title('LacI and TetR nucllines curves')
plt.legend()
plt.show()
