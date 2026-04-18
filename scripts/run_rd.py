"""
Run Random Deposition (RD) simulations for interface growth.

Random Deposition is a non-equilibrium growth model where particles are
randomly deposited on a 1D substrate. The interface width w(t) exhibits
scaling behavior with exponent β ≈ 0.5 (KPZ universality class).
"""

import numpy as np
import matplotlib.pyplot as plt
from simulations.random_deposition import average_rd, log_times, fit_beta


def main():

    # Parameters
    L_list = [100, 200, 400, 800]
    runs = 500

    tmin = 10
    tmax = 50000
    nt = 60

    # Generate logarithmic times
    times = log_times(tmin, tmax, nt)

    # Run simulations
    results = {}

    for L in L_list:
        print(f"Simulating L = {L}")
        w, h = average_rd(L, times, runs)
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
    plt.savefig("figures/varianzasRD.png", dpi=300)
    print("Saved: figures/varianzasRD.png")
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
    plt.savefig("figures/mediasRD.png", dpi=300)
    print("Saved: figures/mediasRD.png")
    plt.close()

    # Fit exponent β
    Lfit = L_list[-1]
    wfit, _ = results[Lfit]

    beta, c = fit_beta(times, wfit, tmin=100, tmax=20000)
    print(f"\nBeta (L={Lfit}) = {beta:.4f}")

    # Plot 3: Fit comparison
    plt.figure(figsize=(7, 6))

    plt.loglog(times, wfit, 'o', label="Datos")

    fit = np.exp(c) * times**beta
    plt.loglog(times, fit, '--', label=rf"Ajuste: $\beta={beta:.3f}$")

    plt.xlabel("t")
    plt.ylabel("w(t)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("figures/ajusteRD.png", dpi=300)
    print("Saved: figures/ajusteRD.png")
    plt.close()

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
