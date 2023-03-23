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

    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


params = [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp,
          glp, ktp, gtp]
time = np.linspace(0, 2000, 2001)

solution = odeint(deterministic, [0, 0, 0, 0], time, args=(params,))


def nullclines(args, aTc=25, IPTG=0.34):
    """
    Nullclines of the Genetic Toggle Switch
    """
    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args
    n = 201
    LacI_vector = np.linspace(0, 4000, 501)
    print("LacI Vector:", LacI_vector)
    TetR_vector = np.linspace(0, 1500, 151)
    # print("TetR Vector:", TetR_vector)

    mRNAL_vector = (klm0 + (klm / (1 + ((TetR_vector / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet))) / glm

    mRNAT_vector = (ktm0 + (
                ktm / (1 + ((LacI_vector / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac))) / gtm


    n_lacI_vector = (klp * mRNAL_vector) / glp
    n_tetR_vector = (ktp * mRNAT_vector) / gtp

    # Plotting the nullclines
    plt.plot(n_lacI_vector, TetR_vector, 'g', label='LacI nullcline')
    plt.plot(LacI_vector, n_tetR_vector, 'c', label='TetR nullcline')
    plt.xlabel('LacI')
    plt.ylabel('TetR')
    plt.title('LacI and TetR nucllines curves')
    plt.legend()
    plt.show()


nullclines(params)
