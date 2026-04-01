# integrators.py — SOLUTION
"""Numerical integration methods for ODE systems."""
import numpy as np


def euler_step(state, tendency_fn, dt):
    """

    Parameters
    ----------
    state : np.ndarray
        Current state vector (e.g., [x, y, z] for Lorenz63).
    tendency_fn : callable
        Function that takes state and returns dstate/dt.
    dt : float
        Time step size.

    Returns
    -------
    np.ndarray
        State at the next time step.
    """
    return state + tendency_fn(state) * dt


def integrate(state0, tendency_fn, dt, n_steps):
    """

    Parameters
    ----------
    state0 : np.ndarray
        Initial state vector, shape (n_vars,).
    tendency_fn : callable
        Function that takes state and returns tendency, shape (n_vars,).
    dt : float
        Time step size.
    n_steps : int
        Number of time steps to take.

    Returns
    -------
    np.ndarray
        Trajectory array, shape (n_steps + 1, n_vars).
        Row 0 is the initial condition, row -1 is the final state.
    """
    trajectory = np.zeros((n_steps + 1, len(state0)))
    trajectory[0] = state0
    for i in range(n_steps):
        trajectory[i + 1] = euler_step(trajectory[i], tendency_fn, dt)
    return trajectory


if __name__ == "__main__":
    # Self-test: exponential decay dy/dt = -y, y(0) = 1
    result = integrate(np.array([1.0]), lambda y: -y, dt=0.01, n_steps=100)
    final = result[-1, 0]
    exact = np.exp(-1.0)
    print(f"Euler result: {final:.4f}, Exact: {exact:.4f}, Error: {abs(final - exact):.4f}")
    assert abs(final - exact) < 0.01, f"Error too large: {abs(final - exact)}"
    print("integrators.py: all checks passed!")
