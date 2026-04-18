# Critical Phenomena and Cooperative Phenomena

Python implementations of three core phenomena from the Critical Phenomena and Cooperative Phenomena course.

## Repository layout

```
├── simulations/
│   ├── random_deposition.py       # Pure random deposition (KPZ class)
│   ├── surface_relaxation.py      # RD with surface relaxation (EW class)
│   └── wetting.py                 # Wetting with repulsive interaction
├── scripts/
│   ├── run_rd.py                  # Simulate RD: roughness evolution
│   ├── run_rdsr.py                # Simulate RDSR: scaling collapse
│   └── run_wetting.py             # Wetting: θ(p) critical exponent
└── figures/                        # Output PNG figures
```

## Physics

### 1 — Random Deposition (RD)

Pure random deposition on a 1D substrate: particles land at random positions without any relaxation.

**Interface width scaling:**
$$w(t) \sim t^\beta$$

with exponent $\beta \approx 0.5$ (KPZ universality class).

**Key features:**
- Continuous roughening over time
- Non-smooth interfaces
- Width grows as square root of time
- Belongs to Kardar-Parisi-Zhang (KPZ) universality class

### 2 — Random Deposition with Surface Relaxation (RDSR)

Combines random deposition with local relaxation: deposited particles preferentially occupy lower sites among their neighbors.

**Interface width scaling:**
$$w(t) \sim t^\beta$$

with exponent $\beta \approx 0.25$ (Edwards-Wilkinson universality class).

**Family-Vicsek scaling collapse:**
$$w(t, L) / w_{\text{sat}} \approx f(t / L^z)$$

where $z = 2$ and the scaling function exhibits growth regime ($t \ll L^z$) and saturation regime ($t \gg L^z$).

**Key features:**
- Smoother interfaces than pure RD due to relaxation
- Different universality class (EW vs KPZ)
- Demonstrates scaling collapse
- Exponents match theoretical predictions

### 3 — Wetting with Repulsive Interaction

Nonlinear PDE model for surface wetting with a power-law repulsive interaction:

$$\frac{\partial h}{\partial t} = \nabla^2 h + \frac{1}{(h + \epsilon)^{p+1}} + F + \sqrt{2D}\xi$$

where $p$ controls the interaction strength and $\epsilon$ regularizes singularities.

**Critical exponent:**
$$\theta = \frac{1}{p+2}$$

depends on the interaction power law $p$.

**Key features:**
- Demonstrates how microscopic interactions determine critical exponents
- Linear scaling collapse with respect to $p$
- Regularization needed for small $h$

## Results

### RD: KPZ scaling

- Non-smooth interfaces grow according to KPZ exponent
- Roughness scaling shows $w(t) \sim t^{0.5}$ behavior
- Demonstrates generic properties of growth processes

### RDSR: Edwards-Wilkinson scaling

- Surface relaxation suppresses roughness growth
- Scaling collapse validates Family-Vicsek form
- Critical exponents match theoretical predictions to $\sim$1% accuracy

### Wetting: Critical exponent dependence

- Exponent $\theta$ varies smoothly with interaction strength $p$
- Numerical values agree with theoretical predictions $\theta = 1/(p+2)$
- Demonstrates universality of critical behavior

## Usage

```bash
pip install numpy scipy matplotlib numba

python scripts/run_rd.py
python scripts/run_rdsr.py
python scripts/run_wetting.py
```

All scripts write PNG figures to `figures/` and print results to stdout.

## Dependencies

- `numpy` — Numerical computing
- `matplotlib` — Visualization
- `numba` — JIT compilation for fast simulation

## References

This repository implements models from the course on Critical Phenomena and Cooperative Phenomena,
demonstrating universal scaling behavior and how different microscopic mechanisms lead to different
universality classes in non-equilibrium phenomena.
