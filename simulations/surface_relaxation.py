import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# =====================================================
# Simulación RDSR (Random Deposition with Surface Relaxation)
# =====================================================

@njit
def simulate_rdsr(L, T, times):

    h = np.zeros(L, dtype=np.int64)

    nt = len(times)

    w = np.zeros(nt)
    hmean = np.zeros(nt)

    idx = 0
    tmax = times[-1]


    for t in range(1, tmax + 1):

        # L deposiciones = 1 unidad de tiempo
        for _ in range(L):

            i = np.random.randint(0, L)

            left = h[(i-1) % L]
            mid = h[i]
            right = h[(i+1) % L]

            if left <= mid and left <= right:
                j = (i-1) % L
            elif right <= mid and right <= left:
                j = (i+1) % L
            else:
                j = i

            h[j] += 1


        # Medición
        if t == times[idx]:

            mean = h.mean()

            var = ((h - mean)**2).mean()

            w[idx] = np.sqrt(var)
            hmean[idx] = mean

            idx += 1

            if idx >= nt:
                break


    return w, hmean



# =====================================================
# Promedio paralelo
# =====================================================

@njit(parallel=True)
def average_rdsr(L, times, runs):

    nt = len(times)

    w_avg = np.zeros(nt)
    h_avg = np.zeros(nt)


    for r in prange(runs):

        w, h = simulate_rdsr(L, times[-1], times)

        for i in range(nt):
            w_avg[i] += w[i]
            h_avg[i] += h[i]


    w_avg /= runs
    h_avg /= runs


    return w_avg, h_avg



# =====================================================
# Tiempos log
# =====================================================

def log_times(tmin, tmax, n):

    t = np.logspace(
        np.log10(tmin),
        np.log10(tmax),
        n
    ).astype(np.int64)

    return np.unique(t)



# =====================================================
# Ajuste potencia
# =====================================================

def fit_power(t, y, tmin=None, tmax=None):

    mask = np.ones_like(t, dtype=bool)

    if tmin is not None:
        mask &= (t >= tmin)

    if tmax is not None:
        mask &= (t <= tmax)


    logt = np.log(t[mask])
    logy = np.log(y[mask])

    p = np.polyfit(logt, logy, 1)

    return p[0], p[1]
