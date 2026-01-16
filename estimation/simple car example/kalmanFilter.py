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
P0 = np.diag([14**2, 5**2, 1**2]) # initial covariance

# Initialize kalman filter
# generating an initial truth
xt0 = rng.multivariate_normal(m0, P0)
xk = xt0 # truth at time 0
mk = m0
Pk = P0

# Change storage arrays to interleave a priori and a posteriori
nSteps = len(timeVector)
zkp = np.zeros(nSteps)
xtp = np.zeros((3, nSteps))
# Single array that will hold interleaved a priori and a posteriori
mpt = np.zeros((3, 2 * nSteps - 1))  # 2x size to hold both
Ppt = np.zeros((3, 3, 2 * nSteps - 1))
time_interleaved = np.zeros(2 * nSteps - 1)  # Time vector for interleaved data

# Store initial values (a priori at t=0)
mpt[:, 0] = m0
Ppt[:, :, 0] = P0
time_interleaved[0] = timeVector[0]
xtp[:, 0] = xt0

ind = 1  # Index for interleaved arrays

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
    mpt[:, ind] = mkm
    Ppt[:, :, ind] = Pkm
    time_interleaved[ind] = tk
    ind += 1

    # Correct/Update our estimate
    zht = (H @ mkm).item() # maps propated mean into measurement space, zht = expected measurement
    Wk = (H @ Pkm @ H.T + Rk).item()  # innovation matrix
    Ck = Pkm @ H.T 
    Kk = Ck / Wk # Kalman gain
    mkp = mkm + Kk.flatten() * (zk - zht) # updated mean
    Pkp = Pkm - Ck @ Kk.T - Kk @ Ck.T + Kk * Wk * Kk.T # updated covariance

    # Store a posteriori (after measurement update)
    mpt[:, ind] = mkp
    Ppt[:, :, ind] = Pkp
    time_interleaved[ind] = tk  # Same time, but a posteriori
    ind += 1

    # Update for next iteration
    mk = mkp
    Pk = Pkp

# Position estimation plot (with interleaved a priori/a posteriori)
# Create interleaved truth array to match mpt
xtp_interleaved = np.zeros(2 * nSteps - 1)
xtp_interleaved[0] = xtp[0, 0]  # Initial truth
for i in range(1, nSteps):
    # Both a priori and a posteriori use the same truth at time tk
    xtp_interleaved[2*i - 1] = xtp[0, i]  # A priori truth
    xtp_interleaved[2*i] = xtp[0, i]      # A posteriori truth (same time)

plt.figure(figsize=(12, 6))
plt.plot(time_interleaved, xtp_interleaved, 'r-', linewidth=2, label='Truth')
plt.plot(time_interleaved, mpt[0, :], 'b-', linewidth=2, label='Estimate (interleaved)')
# Plot measurements at their actual times
plt.plot(timeVector, zkp, 'ko', markersize=5, label='Measurements', alpha=0.7, zorder=3)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.title('Kalman Filter: Position Estimation (A Priori/A Posteriori Interleaved)')
plt.show()

# Plot error with 3 sigma bounds (interleaved - creates jagged/sawtooth pattern)
error = np.zeros(2 * nSteps - 1)
error[0] = xtp[0, 0] - mpt[0, 0]  # Initial error

# Calculate error for interleaved array
for i in range(1, nSteps):
    # A priori error
    error[2*i - 1] = xtp[0, i] - mpt[0, 2*i - 1]
    # A posteriori error
    error[2*i] = xtp[0, i] - mpt[0, 2*i]

plt.figure(figsize=(12, 6))
plt.plot(time_interleaved, error, 'r-', linewidth=2, label='Estimation Error')
plt.plot(time_interleaved, 3 * np.sqrt(Ppt[0, 0, :]), 'b--', label='+3σ bound')
plt.plot(time_interleaved, -3 * np.sqrt(Ppt[0, 0, :]), 'b--', label='-3σ bound')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)
plt.title('Estimation Error with 3σ Bounds (Interleaved A Priori/A Posteriori)')
plt.show()
