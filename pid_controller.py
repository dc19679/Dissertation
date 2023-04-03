import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def deterministic(u, t, aTc, IPTG, args):
    """
    Determinsitic ODE system of the Genetic Toggle Switch
    """
    mRNAl, mRNAt, LacI, TetR = u

    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (
            klm / (1 + ((TetR / thetaTet) / (1 + (aTc / thetaAtc) ** etaAtc)) ** etaTet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (
            ktm / (1 + ((LacI / thetaLac) / (1 + (IPTG / thetaIptg) ** etaIptg)) ** etaLac)) - gtm * mRNAt
    dLacI_dt = klp * mRNAl - glp * LacI
    dTetR_dt = ktp * mRNAt - gtp * TetR

    return [dmRNAl_dt, dmRNAt_dt, dLacI_dt, dTetR_dt]


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


# Initial conditions
y0 = [0, 0,0, 0]
aTc0 = 20
IPTG0 = 0.25

# PID control parameters for LacI
LacI_Kc = 0.0001 # Gain
LacI_TauI = 1.0
LacI_tauD = 1.0
LacI_integral = 0.0
derivative_LacI = 0.0

# PID control parameters for LacI
TetR_Kc = 0.0001 # Gain
TetR_TauI = 0.05
TetR_tauD = 1.0
TetR_integral = 0.0
derivative_TetR = 0.0

# Time space
time = np.linspace(0,2000,2001)


# Storage for plotting
y = np.zeros((4, len(time)))


# Assigning the first value of y to the initial condition
y[0,0] = y0[0]
y[1,0] = y0[1]
y[2,0] = y0[2]
y[3,0] = y0[3]



# Setting the target setpoints
LacI_target = np.zeros(len(time))
TetR_target = np.zeros(len(time))

LacI_target[0:] = 520
TetR_target[0:] = 280
print(LacI_target)
print(TetR_target)

# Parameters
params = (klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet,
                       glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm,
                       klp, glp, ktp, gtp)
# Control Variables
aTc = np.zeros(len(time))
IPTG = np.zeros(len(time))

# Error between the Setpoint and the system output
error_LacI = np.zeros(len(time))
error_TetR = np.zeros(len(time))

for i in range(len(time)-1):

    ### Proportional ###
    error_LacI[i] = LacI_target[i] - y0[2]
    error_TetR[i] = TetR_target[i] - y0[3]

    ### Integral ###
    LacI_integral = LacI_integral + error_LacI[i] * (time[i + 1] - time[i])
    TetR_integral = TetR_integral + error_TetR[i] * (time[i + 1] - time[i])

    ### Derivative ###
    if i >= 1:
        derivative_LacI = (y[2,i] - y[2, i-1]) /  (time[i + 1] - time[i])
        derivative_TetR = (y[3,i] - y[3, i-1]) /  (time[i + 1] - time[i])

    ### Dual PI Controller ##
    aTc[i] = aTc0 + LacI_Kc * error_LacI[i] + LacI_Kc/LacI_TauI * LacI_integral + LacI_Kc * LacI_tauD * derivative_LacI
    IPTG[i] = IPTG0 + TetR_Kc * error_TetR[i] + TetR_Kc/TetR_TauI * TetR_integral + TetR_Kc * TetR_tauD * derivative_TetR

    # Update the system
    solution = odeint(deterministic, y0, [time[i], time[i+1]], args=(aTc[i], IPTG[i], params))

    # Use the current solution as the input for the next time step
    y0 = solution[1]

    # Storing the values for plotting
    y[0, i + 1] = y0[0]
    y[1, i + 1] = y0[1]
    y[2, i + 1] = y0[2]
    y[3, i + 1] = y0[3]
    # print(y)


# PLot the graph
plt.plot(time,LacI_target, linewidth = 1, label = "LacI Target")
plt.plot(time,TetR_target, linewidth = 1, label = "TetR Target")
plt.plot(time,y[2], linewidth = 2, label = "LacI")
plt.plot(time,y[3], linewidth = 2, label = "TetR")
plt.ylabel("LacI and TetR")
plt.xlabel("Time")
plt.legend(loc = "best")
plt.show()
