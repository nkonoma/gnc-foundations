import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Falling body example with extended Kalman filter

def falling_body_eoms(t, x, g, rho_0, k_p):
    x = x.ravel()
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    # d and rho
    rho = rho_0 * np.exp(-x1 / k_p)
    d = rho * x2**2 / x3

    dxdt = np.hstack((x2, d - g, 0))

    # Jacobian of the dynamics
    if len(x) > 3:
        F = np.array([[0, 1, 0], 
                      [-(rho_0*x2**2*np.exp(-x1/k_p))/(2*k_p*x3), (rho_0*x2*np.exp(-x1/k_p))/x3, -(rho_0*x2**2*np.exp(-x1/k_p))/(2*x3**2)],
                      [0, 0, 0]])
        P = np.reshape(x[3:], (3, 3))
        dP = F @ P + P @ F.T
        dxdt = np.concatenate([dxdt, dP.flatten()])
        return dxdt
    else:
        return dxdt

# Set random number seed
rng = np.random.default_rng(100)

# initial mean and covariance
m0 = np.array([1e5, -6000, 2000]) # ft, ft/s, lb/ft^2
P0 = np.diag([500, 2e4, 2.5e5]) # ft^2, ft^2/s^2, lb^2/ft^4
# initial truth
xt0 = rng.multivariate_normal(m0, P0)
xk = xt0 # truth at time 0
mk = m0
Pk = P0

# system parameters
rho_0 = 3.4e-3 # lb s^2/ft^4
k_p = 22000 # ft
g = 32.2 # ft/s^2
Rk = 100 # ft^2
Lk = 1

# measurement mapping matrix Hk
Hk = np.array([[1, 0, 0]])

# time vector
tv = np.arange(0, 18, 0.1)

# Change storage arrays to interleave a priori and a posteriori
nSteps = len(tv)
zkp = np.zeros(nSteps)
xtp = np.zeros((3, nSteps))
# Single array that will hold interleaved a priori and a posteriori
mpt = np.zeros((3, 2 * nSteps - 1))  # 2x size to hold both
Ppt = np.zeros((3, 3, 2 * nSteps - 1))
time_interleaved = np.zeros(2 * nSteps - 1)  # Time vector for interleaved data

# Store initial values (a priori at t=0)
mpt[:, 0] = m0 # 
Ppt[:, :, 0] = P0 
time_interleaved[0] = tv[0]
xtp[:, 0] = xt0

ind = 1  # Index for interleaved arrays

# Extended Kalman Filter
for i in range(1, nSteps): # Start from index 1
    # Propagate our truth
    X = solve_ivp(falling_body_eoms, [tv[i-1], tv[i]], xk, args=(g, rho_0, k_p), rtol=1e-6, atol=1e-6)
    xtk = X.y[:, -1]

    # Generate a measurement
    zk = (Hk @ xtk)[0] + rng.normal(0, np.sqrt(Rk))
    
    # Predict (Propogate our estimate)
    X = solve_ivp(falling_body_eoms, [tv[i-1], tv[i]], np.concatenate([mk, Pk.flatten()]), args=(g, rho_0, k_p), rtol=1e-6, atol=1e-6)
    mkm = X.y[:3, -1] # propagated mean
    Pkm_flat = X.y[3:, -1] # propagated covariance of measurement
    Pkm = np.reshape(Pkm_flat, (3, 3))

    # Store a priori (before measurement update)
    zkp[i] = zk
    xtp[:, i] = xtk
    mpt[:, ind] = mkm
    Ppt[:, :, ind] = Pkm
    time_interleaved[ind] = tv[i]
    ind += 1

    # Correct/Update our estimate
    zht = (Hk @ mkm).item() # maps propated mean into measurement space, zht = expected measurement
    Wk = (Hk @ Pkm @ Hk.T + Rk).item()  # innovation covariance
    Ck = Pkm @ Hk.T 
    Kk = Ck / Wk # Kalman gain
    mkp = mkm + Kk.flatten() * (zk - zht) # updated mean
    # Pkp = Pkm - Ck @ Kk.T - Kk @ Ck.T + Kk * Wk * Kk.T # updated covariance (full form)
    Pkp = Pkm - Kk @ Ck.T # updated covariance (simplified form)

    # Store a posteriori (after measurement update)
    mpt[:, ind] = mkp
    Ppt[:, :, ind] = Pkp
    time_interleaved[ind] = tv[i]  # Same time, but a posteriori
    ind += 1

    # Update for next iteration
    xk = xtk  # Update truth state
    mk = mkp  # Update estimate
    Pk = Pkp  # Update covariance

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time_interleaved, mpt[0, :], 'b-', linewidth=2, label='Estimate (interleaved)')
plt.plot(tv[1:], zkp[1:], 'ko', markersize=3, label='Measurements', alpha=0.7, zorder=3)
plt.xlabel('Time (s)')
plt.ylabel('Height (ft)')
plt.legend()
plt.grid(True)
plt.title('Extended Kalman Filter: Height Estimation (A Priori/A Posteriori Interleaved)')
plt.show()

# Plot error with 1 sigma bounds
error = np.zeros(2 * nSteps - 1)
error[0] = xtp[0, 0] - mpt[0, 0]  # Initial error

# Calculate position error for interleaved array
for i in range(1, nSteps):
    # A priori error
    error[2*i - 1] = xtp[0, i] - mpt[0, 2*i - 1]
    # A posteriori error
    error[2*i] = xtp[0, i] - mpt[0, 2*i]

plt.figure(figsize=(12, 6))
plt.plot(time_interleaved, error, 'r-', linewidth=2, label='Estimation Error')
plt.plot(time_interleaved, np.sqrt(Ppt[0, 0, :]), 'b--', label='+1σ bound')
plt.plot(time_interleaved, -np.sqrt(Ppt[0, 0, :]), 'b--', label='-1σ bound')
plt.xlabel('Time (s)')
plt.ylabel('Error (ft)')
plt.legend()
plt.grid(True)
plt.title('Estimation Error with 1σ Bounds (Interleaved A Priori/A Posteriori)')
plt.show()

# Plot velocity error with 1 sigma bounds
error = np.zeros(2 * nSteps - 1)
error[0] = xtp[1, 0] - mpt[1, 0]  # Initial error

# Calculate velocity error for interleaved array
for i in range(1, nSteps):
    # A priori error
    error[2*i - 1] = xtp[1, i] - mpt[1, 2*i - 1]
    # A posteriori error
    error[2*i] = xtp[1, i] - mpt[1, 2*i]

plt.figure(figsize=(12, 6))
plt.plot(time_interleaved, error, 'r-', linewidth=2, label='Estimation Error')
plt.plot(time_interleaved, np.sqrt(Ppt[1, 1, :]), 'b--', label='+1σ bound')
plt.plot(time_interleaved, -np.sqrt(Ppt[1, 1, :]), 'b--', label='-1σ bound')
plt.xlabel('Time (s)')
plt.ylabel('Error (ft/s)')
plt.legend()
plt.grid(True)
plt.title('Estimation Error with 1σ Bounds (Interleaved A Priori/A Posteriori)')
plt.show()

# Plot ballistic coefficient error with 1 sigma bounds
error = np.zeros(2 * nSteps - 1)
error[0] = xtp[2, 0] - mpt[2, 0]  # Initial error

# Calculate ballistic coefficient error for interleaved array
for i in range(1, nSteps):
    # A priori error
    error[2*i - 1] = xtp[2, i] - mpt[2, 2*i - 1]
    # A posteriori error
    error[2*i] = xtp[2, i] - mpt[2, 2*i]

plt.figure(figsize=(12, 6))
plt.plot(time_interleaved, error, 'r-', linewidth=2, label='Estimation Error')
plt.plot(time_interleaved, np.sqrt(Ppt[2, 2, :]), 'b--', label='+1σ bound')
plt.plot(time_interleaved, -np.sqrt(Ppt[2, 2, :]), 'b--', label='-1σ bound')
plt.xlabel('Time (s)')
plt.ylabel('Error (lb/ft^2)')
plt.legend()
plt.grid(True)
plt.title('Estimation Error with 1σ Bounds (Interleaved A Priori/A Posteriori)')
plt.show()
