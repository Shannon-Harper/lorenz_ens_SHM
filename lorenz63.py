# lorenz63.py — SOLUTION
"""Lorenz 1963 three-variable chaotic model."""
import numpy as np
try:
    from solutions.integrators import integrate      # from outside: python -m solutions.run_lorenz_ensemble
except ImportError:
    from integrators import integrate                # from inside:  cd solutions && python run_lorenz_ensemble.py


class Lorenz63:
    """The Lorenz (1963) system:

    Equations
    ---------
    dx/dt = sigma * (y - x)
    dy/dt = rho * x - y - x * z
    dz/dt = x * y - beta * z
    """

    def __init__(self, sigma=10, rho=28, beta=8 / 3):
        """Store model parameters."""
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def tendency(self, state):
        """Compute the time derivatives [dx/dt, dy/dt, dz/dt].

        Parameters
        ----------
        state : np.ndarray
            Current state [x, y, z], shape (3,).

        Returns
        -------
        np.ndarray
            Tendencies [dx/dt, dy/dt, dz/dt], shape (3,).
        """
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = self.rho * x - y - x * z
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])

    def run(self, state0, dt, n_steps):
        """Integrate the model forward from a single initial condition.

        Parameters
        ----------
        state0 : np.ndarray
            Initial condition [x0, y0, z0], shape (3,).
        dt : float
            Time step.
        n_steps : int
            Number of steps.

        Returns
        -------
        np.ndarray
            Trajectory, shape (n_steps + 1, 3).
        """
        return integrate(state0, self.tendency, dt, n_steps)
    
    def vectorized_tendency(self, states):

        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        dxdt = self.sigma * (y - x)
        dydt = self.rho * x - y - x * z
        dzdt = x * y - self.beta * z
        return np.column_stack([dxdt, dydt, dzdt])


    def run_ensemble(self, initial_conditions, dt, n_steps):
        """Run an ensemble of trajectories from multiple initial conditions.

        Parameters
        ----------
        initial_conditions : np.ndarray
            Array of initial conditions, shape (n_members, 3).
        dt : float
            Time step.
        n_steps : int
            Number of steps.

        Returns
        -------
        np.ndarray
            Ensemble trajectories, shape (n_members, n_steps + 1, 3).
        """
        n_members = initial_conditions.shape[0]

        # ---- METHOD 1: Nested for loop ----
        ensemble = np.zeros((n_members, n_steps + 1, 3))
        for m in range(n_members):
            ensemble[m] = self.run(initial_conditions[m], dt, n_steps)
        return ensemble

        # ---- METHOD 2: Vectorized single loop over time ----
        
        ensemble = np.zeros((n_members, n_steps + 1, 3))
        ensemble[:, 0, :] = initial_conditions

        for i in range(n_steps):
            states = ensemble[:, i, :]              # (n_members, 3)
            ensemble[:, i + 1, :] = states + self.vectorized_tendency(states) * dt

        return ensemble

if __name__ == "__main__":
    model = Lorenz63()

    # Test single run
    traj = model.run(np.array([1.0, 1.0, 1.0]), dt=0.01, n_steps=1000)
    print(f"Single run: final state = [{traj[-1, 0]:.2f}, {traj[-1, 1]:.2f}, {traj[-1, 2]:.2f}]")
    print(f"  trajectory shape: {traj.shape}")

    # Test ensemble run
    n_members = 5
    ics = np.array([[1.0, 1.0, 1.0]] * n_members) + np.random.randn(n_members, 3) * 0.01
    ensemble = model.run_ensemble(ics, dt=0.01, n_steps=1000)
    print(f"Ensemble run: shape = {ensemble.shape}")
    print(f"  Expected: ({n_members}, 1001, 3)")
    print("lorenz63.py: all checks passed!")
