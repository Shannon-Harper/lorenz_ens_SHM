# run_lorenz_ensemble.py — SOLUTION
"""
Driver script: Lorenz63 Ensemble Predictability Experiment
==========================================================

Produces a 3-panel figure showing how predictability
depends on where you start on the Lorenz attractor.

Each panel shows:
  - The full attractor (light blue background)
  - An ensemble of trajectories (red) started from a small cloud
    of initial conditions near a chosen point

The three starting regions are:
  (a) Deep left lobe  — ring barely grows (highly predictable)
  (b) High left lobe  — ring distorts into banana/boomerang (transition zone)
  (c) Saddle region   — ring explodes, members go left and right (no predictability)

Output: saves 'lorenz_ensemble_predictability.png'

Usage
-----
    cd solutions/ && python run_lorenz_ensemble.py
"""
import numpy as np
try:
    from solutions.lorenz63 import Lorenz63          # from outside: python -m solutions.run_lorenz_ensemble
    from solutions.plotting import plot_ensemble_panels
except ImportError:
    from lorenz63 import Lorenz63                    # from inside:  cd solutions && python run_lorenz_ensemble.py
    from plotting import plot_ensemble_panels

# --- Configuration ---
DT = 0.01
SPINUP_STEPS = 500000       # let transients die out
REFERENCE_STEPS = 20000   # long trajectory for background attractor
ENSEMBLE_STEPS = 50     # how far to integrate each ensemble
N_MEMBERS = 30            # ensemble size
PERTURBATION_SCALE = 0.3  # std dev of initial perturbations
SAVE_PATH = "lorenz_ensemble_predictability.png"


def main():
    """Run the full ensemble experiment."""
    # Step 1 — Create model
    model = Lorenz63()

    # Step 2 — Generate reference trajectory
    # Spin up from (1, 1, 1) for SPINUP_STEPS to let transients die out
    spinup = model.run(np.array([1.0, 1.0, 1.0]), DT, SPINUP_STEPS)
    # Run the reference from the end of the spinup
    reference = model.run(spinup[-1], DT, REFERENCE_STEPS)

    # Step 4 — Create initial condition clouds
    np.random.seed(42)  # reproducibility
    deep_left_state = np.array([-15,1,45])
    high_left_state = np.array([-8,-3,34])
    saddle_state = np.array([0,0,18])
    ics_deep = deep_left_state + np.random.randn(N_MEMBERS, 3) * PERTURBATION_SCALE
    ics_high = high_left_state + np.random.randn(N_MEMBERS, 3) * PERTURBATION_SCALE
    ics_saddle = saddle_state + np.random.randn(N_MEMBERS, 3) * PERTURBATION_SCALE

    # Step 5 — Run ensembles
    ensemble_deep = model.run_ensemble(ics_deep, DT, ENSEMBLE_STEPS)
    ensemble_high = model.run_ensemble(ics_high, DT, ENSEMBLE_STEPS)
    ensemble_saddle = model.run_ensemble(ics_saddle, DT, ENSEMBLE_STEPS)

    # Step 6 — Plot
    fig, axes = plot_ensemble_panels(
        [ensemble_deep, ensemble_high, ensemble_saddle],
        reference,
        ["(a) Deep left lobe", "(b) High left lobe", "(c) Saddle region"],
        save_path=SAVE_PATH,
    )

    print(f"Figure saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
