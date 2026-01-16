import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Sports car Example
# State: x = [rho, rho_dot, rho_ddot]
# Initial estimate: m_0 = [0 0 7]T, engine should accelerate at 7 m/s^2
# Adding some inital uncertainty. P_0 = diag([14^2 5^2 1^2])
# Constant acceleration of the vehicle tels us that F(t) = F = [0 1 0; 0 0 1; 0 0 0]
# We are only taking range measurements, so H_k = [1 0 0]
# Additionally, to account for noise in our dynamic model, we define the process noise power spectral density to be Q_s(t) = diag([1 0.1 0.01])

def car_eoms(t, x, F, Q, M):
    dxdt = F @ x[:3]

    if len(x) > 3:
        P = np.reshape(x[3:], (3, 3))
        dP = F @ P + P @ F.T + M @ Q @ M.T
        # Add a vectorized rate of change to the rate of change vector
        dxdt = np.concatenate([dxdt, dP.flatten()])
        return dxdt
    else:
        return dxdt

# Set random number seed
rng = np.random.default_rng(100)

# Relevent system parameters
dt = 0.5
timeVector = np.arange(0, 10, dt)
# Measurement noise covariance
Rk = 10**2
# Process noise PSD
Qs = np.diag([1, 0.1, 0.01])
# Dynamics
F = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
# Measurement model
H = np.array([[1, 0, 0]])
# Process noise mapping
M = np.eye(3)

# Define our intial estimates for mean and covariance and the truth at the starting time
# Truth and estimates at t=0
m0 = np.array([0, 0, 7]) # initial estimate [m, m/s, m/s^2]
P0 = np.diag([14^2, 5^2, 1^2]) # initial covariance

# Initialize kalman filter
# generating an initial truth
xt0 = rng.multivariate_normal(m0, P0)
xk = xt0 # truth at time 0
mk = m0
Pk = P0

nSteps = len(timeVector)
zkp = np.zeros(nSteps)
xtp = np.zeros((3, nSteps))  # Store truth once per step
mpt_prior = np.zeros((3, nSteps))  # A priori estimates
mpt_posterior = np.zeros((3, nSteps))  # A posteriori estimates
Ppt_prior = np.zeros((3, 3, nSteps))
Ppt_posterior = np.zeros((3, 3, nSteps))

# Store initial values
xtp[:, 0] = xt0
mpt_prior[:, 0] = m0
Ppt_prior[:, :, 0] = P0

# Kalman Filter
for i,tk in enumerate(timeVector[1:], start=1): # Start from index 1
    # Propagate our truth
    X = solve_ivp(car_eoms, [tk-dt, tk], xk, args=(F, Qs, M), rtol=1e-6, atol=1e-6)
    xk = X.y[:, -1]

    # Generate a measurement
    zk = (H @ xk)[0] + rng.normal(0, np.sqrt(Rk)) 

    # Predict (Propagate our estimate)
    X = solve_ivp(car_eoms, [tk, tk+dt], np.concatenate([mk, Pk.flatten()]), args=(F, Qs, M), rtol=1e-6, atol=1e-6)
    mkm = X.y[:3, -1] # propagated mean
    Pkm_flat = X.y[3:, -1] # propagated covariance of measurement
    Pkm = np.reshape(Pkm_flat, (3, 3))

    # Store a priori (before measurement update)
    zkp[i] = zk
    xtp[:, i] = xk
    mpt_prior[:, i] = mkm
    Ppt_prior[:, :, i] = Pkm

    # Correct/Update our estimate
    zht = (H @ mkm).item() # maps propated mean into measurement space, zht = expected measurement
    Wk = (H @ Pkm @ H.T + Rk).item()  # innovation matrix
    Ck = Pkm @ H.T 
    Kk = Ck / Wk # Kalman gain
    mkp = mkm + Kk.flatten() * (zk - zht) # updated mean
    Pkp = Pkm - Ck @ Kk.T - Kk @ Ck.T + Kk * Wk * Kk.T # updated covariance

    # Store a posteriori (after measurement update)
    mpt_posterior[:, i] = mkp
    Ppt_posterior[:, :, i] = Pkp

    # Update for next iteration
    mk = mkp
    Pk = Pkp

# Plots
plt.figure(figsize=(12, 6))
plt.plot(timeVector, zkp, 'ko', markersize=4, label='Measurements', alpha=0.6)
plt.plot(timeVector, xtp[0, :], 'r-', linewidth=2, label='Truth')
plt.plot(timeVector, mpt_posterior[0, :], 'b-', linewidth=2, label='Estimate (a posteriori)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.title('Kalman Filter: Position Estimation')
plt.show()

# Plot error with 3 sigma bounds
error = xtp[0, :] - mpt_posterior[0, :]
plt.figure(figsize=(12, 6))
plt.plot(timeVector, error, 'r-', label='Estimation Error')
plt.plot(timeVector, 3 * np.sqrt(Ppt_posterior[0, 0, :]), 'b--', label='+3σ bound')
plt.plot(timeVector, -3 * np.sqrt(Ppt_posterior[0, 0, :]), 'b--', label='-3σ bound')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)
plt.title('Estimation Error with 3σ Bounds')
plt.show()

# Plot error with 3 sigma bounds (both a priori and a posteriori)
error_prior = xtp[0, :] - mpt_prior[0, :]
error_posterior = xtp[0, :] - mpt_posterior[0, :]

plt.figure(figsize=(12, 6))
# A priori error and bounds
plt.plot(timeVector, error_prior, 'r-', linewidth=2, label='Error (a priori)')
plt.plot(timeVector, 3 * np.sqrt(Ppt_prior[0, 0, :]), 'r--', alpha=0.6, label='+3σ (a priori)')
plt.plot(timeVector, -3 * np.sqrt(Ppt_prior[0, 0, :]), 'r--', alpha=0.6, label='-3σ (a priori)')

# A posteriori error and bounds
plt.plot(timeVector, error_posterior, 'b-', linewidth=2, label='Error (a posteriori)')
plt.plot(timeVector, 3 * np.sqrt(Ppt_posterior[0, 0, :]), 'b--', alpha=0.6, label='+3σ (a posteriori)')
plt.plot(timeVector, -3 * np.sqrt(Ppt_posterior[0, 0, :]), 'b--', alpha=0.6, label='-3σ (a posteriori)')

plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)
plt.title('Estimation Error: A Priori vs A Posteriori with 3σ Bounds')
plt.show()
