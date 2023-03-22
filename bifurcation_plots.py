from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

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
aTc = 25
IPTG = 0.25

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


# Create grid of IPTG values
iptg_vals = np.linspace(0, 1, 100)

# Create empty lists to store equilibria
tetr_eqs = []
iptg_eqs = []

# Loop over IPTG values
for iptg in iptg_vals:
    # Define function to solve for equilibria
    def find_equilibria(x, IPTG):
        mRNAl, mRNAt, LacI, TetR = x
        params = [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac,
                  gtm, klp,
                  glp, ktp, gtp]
        # print(deterministic(x, 0, [klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp, aTc, IPTG]))
        return deterministic(x, 0, params)
    
    # Find equilibria
    sol = solve_ivp(lambda t, x: find_equilibria(x, iptg), [0, 1], [0.0, 0.0, 0.0, 0.0], rtol=1e-8, atol=1e-8)
    equilibria = sol.y.T[-1]
    print(equilibria)
    
    # Append tetR and IPTG values to lists
    tetr_eqs.append(equilibria[3])
    iptg_eqs.append(iptg)


# Plot bifurcation diagram
fig, ax = plt.subplots()
ax.plot(iptg_vals, tetr_eqs, 'b-')
ax.set_xlabel('IPTG (mM)')
ax.set_ylabel('TetR (nM)')
ax.set_title('Bifurcation plots - IPTG vs TetR')
plt.show()
 