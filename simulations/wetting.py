import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# ===============================
# Laplaciano 1D
# ===============================

@njit(parallel=True)
def laplacian_1d(h):

    L = len(h)
    lap = np.zeros(L)

    for i in prange(L):

        ip = (i+1) % L
        im = (i-1) % L

        lap[i] = h[ip] + h[im] - 2*h[i]

    return lap


# ===============================
# Simulación 1D Wetting
# ===============================

@njit(parallel=True)
def simulate_wetting_1d(L, dt, tmax, p, F, D, eps, times, nruns):

    nm = len(times)

    hmean = np.zeros(nm)

    sqrt_dt = np.sqrt(dt)


    for r in prange(nruns):

        h = np.ones(L)

        idx = 0


        for t in range(1, tmax+1):

            lap = laplacian_1d(h)


            for i in range(L):

                noise = np.random.randn()

                rep = 1.0 / ((h[i]+eps)**(p+1))

                h[i] += (
                    lap[i]
                    + rep
                    + F
                )*dt + np.sqrt(2*D)*sqrt_dt*noise


                if h[i] < 1e-4:
                    h[i] = 1e-4


            if idx < nm and t == times[idx]:

                hmean[idx] += np.mean(h)

                idx += 1


    hmean /= nruns

    return hmean


# ===============================
# Ajuste del exponente θ
# ===============================

def fit_theta(t, h):

    mask = (t>200) & (t<8000)

    x = np.log(t[mask])
    y = np.log(h[mask])

    a,_ = np.polyfit(x,y,1)

    return a


# ===============================
# Log times
# ===============================

def log_times(tmin, tmax, nm):
    times = np.logspace(
        np.log10(tmin),
        np.log10(tmax),
        nm
    ).astype(int)
    return times
