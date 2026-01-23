import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

norm = np.linalg.norm

# 2-body orbit example with extended Kalman filter

def dynamics2body(t, x, mu):
    x = x.ravel()
    r = x[:1]
    v = x[1:3]

    f0 = np.hstack((v, -mu*r/norm(r)**3))
    dxdt = f0
    if len(x) > 3:
        # Jacobian of the dynamics
        P = np.reshape(x[3:], (2, 2))
        dP = -mu*np.eye(2) / norm(r)**3 + 3*mu*r.T @ r / norm(r)**5 @ P
        dxdt = np.concatenate([f0, dP.flatten()])
        return dxdt
    else:
        return dxdt

# Set a random number seed
rng = np.random.default_rng(200)

# constants
mu = 3.986004415e5
Re = 6378.136 # km

# timing parameters
t0 = 0
dt = 30
tf = 3600
tv = np.arange(t0, tf, dt)

# Specify initial mean/covariance and generate random truth
nx = 4
h0 = 700 # circular orbit altitude km
m0 = np.array([Re+h0, 0, 0, np.sqrt(mu/(Re+h0))])
P0 = np.diag([0.1, 0.1, 1e-3, 1e-3])
x0 = m0 + np.linalg.cholesky(P0) @ rng.standard_normal(nx)

# Specify the power spectral density of the process noise
nq = 2
Qs = (1e-9)**2 @ np.eye(nq)

# Specify the measurement noise covariance
nr = 1
Rk = (10e-3)**2 * np.eye(nr)

# storage arrays for interleaved a priori and a posteriori
nSteps = len(tv)

