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


def hill(x, theta, eta):
    """
    Decreasing Hill function
    """
    return pow(1 + pow((x / theta), eta), -1)


def deterministic(u, t, args, aTc = 20, IPTG = 0.25):
    """
    Deterministic model of the Genetic Toggle Switch
    """

    mRNAL, mRNAT, lacI, tetR = u

    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAL_dt = klm0 + klm * hill(tetR*hill(aTc, thetaAtc, etaAtc), thetaTet, etaTet)-glm*mRNAL

    dmRNAT_dt = ktm0 + ktm * hill(lacI*hill(IPTG, thetaIptg, etaIptg), thetaLac, etaLac)-gtm*mRNAT

    dLacI_dt = klp*mRNAL - glp*lacI

    dTetR_dt = ktp*mRNAT - gtp*tetR

    return [dmRNAL_dt,dmRNAT_dt,dLacI_dt,dTetR_dt]

params = [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp]
time = np.linspace(0,2000,2001)

solution = odeint(deterministic,[0,0,0,0], time, args=(params,))

def nullclines(args, aTc = 20, IPTG = 0.25):
    """
    Nullclines of the Genetic Toggle Switch
    """
    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args
    n = 10
    LacI_vector = np.linspace(0, 2000, n)
    print("LacI Vector:",LacI_vector)
    TetR_vector = np.linspace(0, 2000, n)
    print("TetR Vector:", TetR_vector)

    mRNAL_vector =(ktm0 + ktm * hill(LacI_vector * hill(IPTG,thetaIptg,etaIptg),thetaLac,etaLac)) / glm
    print("mRNAL vector:", mRNAL_vector )
    mRNAT_vector = (klm0 + klm * hill(TetR_vector * hill(aTc,thetaAtc,etaAtc),thetaTet,etaTet)) / gtm
    print("mRNAT Vector:", mRNAT_vector)

    n_lacI_vector = (klp * mRNAL_vector)/ glp
    n_tetR_vector = (ktp * mRNAT_vector)/ gtp

    # Plotting the nullclines
    plt.plot(n_lacI_vector, TetR_vector,'g', label='LacI nullcline')
    plt.plot(LacI_vector, n_tetR_vector,'c', label='TetR nullcline')
    plt.xlabel('LacI')
    plt.ylabel('TetR')
    plt.title('LacI and TetR nucllines curves')
    plt.legend()
    plt.show()

nullclines(params)