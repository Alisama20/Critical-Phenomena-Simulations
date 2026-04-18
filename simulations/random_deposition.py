import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# =====================================================
# Simulación RD (Random Deposition)
# =====================================================

@njit
def simulate_rd(L, T, times):
    """
    Simula una realización de Random Deposition
    Devuelve w(t) y h_mean(t) en los tiempos dados
    """

    h = np.zeros(L, dtype=np.int64)

    nt = len(times)

    w = np.zeros(nt)
    hmean = np.zeros(nt)

    tmax = times[-1]

    current_index = 0


    for t in range(1, tmax + 1):

        # Deposición
        i = np.random.randint(0, L)
        h[i] += 1


        # Medición
        if t == times[current_index]:

            mean = 0.0
            for k in range(L):
                mean += h[k]
            mean /= L

            var = 0.0
            for k in range(L):
                diff = h[k] - mean
                var += diff * diff
            var /= L


            w[current_index] = np.sqrt(var)
            hmean[current_index] = mean

            current_index += 1

            if current_index >= nt:
                break


    return w, hmean



# =====================================================
# Promedio paralelo
# =====================================================

@njit(parallel=True)
def average_rd(L, times, runs):

    nt = len(times)

    w_avg = np.zeros(nt)
    h_avg = np.zeros(nt)


    for r in prange(runs):

        w, h = simulate_rd(L, times[-1], times)

        for i in range(nt):
            w_avg[i] += w[i]
            h_avg[i] += h[i]


    for i in range(nt):
        w_avg[i] /= runs
        h_avg[i] /= runs


    return w_avg, h_avg



# =====================================================
# Generador de tiempos logarítmicos
# =====================================================

def log_times(tmin, tmax, n):

    times = np.logspace(
        np.log10(tmin),
        np.log10(tmax),
        n
    ).astype(np.int64)

    # Eliminar duplicados
    times = np.unique(times)

    return times



# =====================================================
# Ajuste de beta
# =====================================================

def fit_beta(t, w, tmin=None, tmax=None):

    mask = np.ones_like(t, dtype=bool)

    if tmin is not None:
        mask &= (t >= tmin)

    if tmax is not None:
        mask &= (t <= tmax)


    logt = np.log(t[mask])
    logw = np.log(w[mask])

    p = np.polyfit(logt, logw, 1)

    return p[0], p[1]
