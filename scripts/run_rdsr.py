"""
Run Random Deposition with Surface Relaxation (RDSR) simulations.

RDSR combines random deposition with local relaxation where particles prefer
lower sites. This leads to continuous roughening with scaling exponent
β ≈ 0.25 (Edwards-Wilkinson universality class), smoother than pure RD.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulations.surface_relaxation import average_rdsr, log_times, fit_power


def main():

    # Parameters
    L_list = [100, 200, 400, 800]
    runs = 400

    tmin = 10
    tmax = 50000
    nt = 60

    # Theoretical EW exponents
    beta_th = 0.25
    alpha_th = 0.5
    z_th = 2.0

    # Generate logarithmic times
    times = log_times(tmin, tmax, nt)

    # Run simulations
    results = {}

    for L in L_list:
        print(f"Simulating L = {L}")
        w, h = average_rdsr(L, times, runs)
        results[L] = (w, h)

    # Plot 1: Interface width (roughness)
    plt.figure(figsize=(7, 6))

    for L in L_list:
        w, _ = results[L]
        plt.loglog(times, w, 'o-', label=f"L={L}")

    plt.xlabel("t")
    plt.ylabel("w(t)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("figures/varianzasRDSR.png", dpi=300)
    print("Saved: figures/varianzasRDSR.png")
    plt.close()

    # Plot 2: Mean height
    plt.figure(figsize=(7, 6))

    for L in L_list:
        _, h = results[L]
        plt.loglog(times, h, 'o-', label=f"L={L}")

    plt.xlabel("t")
    plt.ylabel(r"$\bar h(t)$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("figures/mediasRDSR.png", dpi=300)
    print("Saved: figures/mediasRDSR.png")
    plt.close()

    # Fit exponent β
    Lfit = L_list[-1]
    wfit, _ = results[Lfit]

    beta, c = fit_power(times, wfit, tmin=200, tmax=6000)
    print(f"\nBeta = {beta:.4f}")

    # Plot 3: Fit comparison with theory
    plt.figure(figsize=(7, 6))

    plt.loglog(times, wfit, 'o', label="Datos")

    fit_num = np.exp(c) * times**beta
    plt.loglog(times, fit_num, '--', label=rf"Ajuste: $\beta={beta:.3f}$")

    fit_th = times**beta_th
    plt.loglog(times, fit_th, ':', label=r"Teoría EW: $\beta=1/4$")

    plt.xlabel("t")
    plt.ylabel("w(t)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("figures/ajusteRDSR.png", dpi=300)
    print("Saved: figures/ajusteRDSR.png")
    plt.close()

    # Plot 4: Collapse (Family-Vicsek scaling)
    plt.figure(figsize=(7, 6))

    for L in L_list:
        w, _ = results[L]
        wsat = w[-1]
        tx = L**z_th

        plt.loglog(times/tx, w/wsat, 'o-', label=f"L={L}")

    # Theoretical collapse
    x = np.logspace(-5, 2, 200)
    f = np.where(x < 1, x**beta_th, 1.0)
    plt.loglog(x, f, 'k--', label="Teoría EW")

    plt.xlabel(r"$t/t_x$")
    plt.ylabel(r"$w/w_{sat}$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("figures/colapsoRDSR.png", dpi=300)
    print("Saved: figures/colapsoRDSR.png")
    plt.close()

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
