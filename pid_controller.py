import time

# Define PID controller constants
Kp = 0.1   # Proportional gain
Ki = 0.01  # Integral gain
Kd = 0.01  # Derivative gain

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



# Define initial values
mRNAl = 0
mRNAt = 0
LacI = 0
TetR = 0

# Define setpoint and error variables
setpoint = 0.5
error = 0
last_error = 0
error_sum = 0

dt = 0.1

# Define loop function
def loop(dt):
    global error, last_error, error_sum, mRNAl, mRNAt, LacI, TetR

    # Calculate error
    error = setpoint - (LacI / (LacI + TetR + 1e-10))
    
    # Calculate PID output
    p_term = Kp * error
    i_term = Ki * error_sum
    d_term = Kd * (error - last_error)
    pid_output = p_term + i_term + d_term
    
    # Update error sum and last error
    error_sum += error
    last_error = error
    
    # Update system state using ODEs
    klm0, klm, thetaAtc, etaAtc, thetaTet, etaTet, glm, ktm0, ktm, thetaIptg, etaIptg, thetaLac, etaLac, gtm, klp, glp, ktp, gtp = args

    dmRNAl_dt = klm0 + (klm/(1 + ((TetR/thetaTet) / (1 + (aTc / thetaAtc )**etaAtc))**etaTet)) - glm * mRNAl
    dmRNAt_dt = ktm0 + (ktm/(1 + ((LacI/thetaLac) / (1 + (IPTG / thetaIptg )**etaIptg))**etaLac)) - gtm * mRNAt
    dLacI_dt = klp*mRNAl - glp*LacI
    dTetR_dt = ktp*mRNAt - gtp*TetR
    
    mRNAl += dmRNAl_dt
    mRNAt += dmRNAt_dt
    LacI += dLacI_dt + pid_output
    TetR += dTetR_dt - pid_output
    
    # Print current system state and error
    print(LacI,TetR)


# Main loop
num_iterations = 1000
for _ in range(num_iterations):
    loop(dt)
    time.sleep(dt)