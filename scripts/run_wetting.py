"""
Run Wetting simulations with coupled equation.

Models surface wetting using a nonlinear PDE with repulsive interaction.
The dynamics exhibits critical behavior with exponent θ = 1/(p+2) that
depends on the interaction power law p.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulations.wetting import simulate_wetting_1d, fit_theta, log_times


def main():

    # Simulation parameters
    L = 100
    dt = 0.01
    tmax = 1000
    nruns = 100

    # Physics parameters
    D = 1.0           # Diffusion
    eps = 0.1         # Regularization
    p = 4             # Repulsion power law
    F = 0.0           # Force (near critical)

    # Time points for measurement
    tmin = 20
    nm = 10
    times = log_times(tmin, tmax, nm)

    # Single run
    print(f"Simulating single run with p={p}")
    h = simulate_wetting_1d(L, dt, tmax, p, F, D, eps, times, nruns)

    theta = fit_theta(times, h)
    theta_th = 1.0 / (p + 2)

    print(f"θ_numerical = {theta:.3f}")
    print(f"θ_theoretical = {theta_th:.3f}")

    # Scan over interaction strength
    print("\n" + "="*50)
    print("Scanning interaction strength p...")
    print("="*50)

    p_values = np.linspace(0, 5, 20)
    theta_num = []
    theta_teo = []

    for p in p_values:
        h = simulate_wetting_1d(L, dt, tmax, p, F, D, eps, times, nruns)
        theta = fit_theta(times, h)
        theta_num.append(theta)
        theta_teo.append(1.0 / (p + 2))
        print(f"p={p:.2f}  θ_num={theta:.3f}  θ_teo={1/(p+2):.3f}")

    # Plot θ(p)
    plt.figure(figsize=(7, 5))
    plt.plot(p_values, theta_num, "o-", label=r"$\theta_{\rm num}$", linewidth=2)
    plt.plot(p_values, theta_teo, "s--", label=r"$\theta_{\rm teo}=1/(p+2)$", linewidth=2)

    plt.xlabel("p", fontsize=12)
    plt.ylabel(r"$\theta$", fontsize=12)
    plt.title("Exponente crítico vs potencia de interacción", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/thetaEnFuncioDep.png", dpi=300)
    print("\nSaved: figures/thetaEnFuncioDep.png")
    plt.close()

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
