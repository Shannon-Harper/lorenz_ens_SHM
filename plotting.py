# plotting.py — SOLUTION
"""Plotting utilities for Lorenz63 ensemble experiments."""
import numpy as np
import matplotlib.pyplot as plt


def plot_attractor(ax, trajectory, color="steelblue", alpha=0.3, linewidth=0.5):
    """Plot a single Lorenz attractor trajectory in x-z phase space.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    trajectory : np.ndarray
        Trajectory array, shape (n_steps+1, 3). Columns are [x, y, z].
    color : str
        Line color.
    alpha : float
        Line transparency.
    linewidth : float
        Line width.
    """
    x = trajectory[:, 0]
    z = trajectory[:, 2]
    ax.plot(x, z, color=color, alpha=alpha, linewidth=linewidth)


def plot_ensemble(ax, ensemble_trajectories, reference_trajectory=None,
                  ensemble_color="firebrick", ref_color="steelblue",
                  title=None):
    """Plot an ensemble of trajectories on a single axes.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    ensemble_trajectories : np.ndarray
        Shape (n_members, n_steps+1, 3).
    reference_trajectory : np.ndarray, optional
        A long reference trajectory, shape (n_ref_steps+1, 3), plotted
        lightly in the background to show the full attractor.
    ensemble_color : str
        Color for ensemble member lines.
    ref_color : str
        Color for the reference attractor.
    title : str, optional
        Panel title.
    """
    # Background attractor
    if reference_trajectory is not None:
        plot_attractor(ax, reference_trajectory, color=ref_color, alpha=0.25, linewidth=0.4)

    # Ensemble members

    for i in range(ensemble_trajectories.shape[0]):
        plot_attractor(ax, ensemble_trajectories[i], color=ensemble_color, alpha=0.4, linewidth=0.7)

    # print('ensemble_trajectories shape:', ensemble_trajectories.shape)  # Debug print
    #plot the first and last points of the forecast trajectories
    for i in range(ensemble_trajectories.shape[0]):
        x0 = ensemble_trajectories[i, 0, 0]
        z0 = ensemble_trajectories[i, 0, 2]
        ax.plot(x0, z0, 'o',color=ensemble_color, alpha=0.3, markersize=2, zorder=5)

    for i in range(ensemble_trajectories.shape[0]):
        x0 = ensemble_trajectories[i, -1, 0]
        z0 = ensemble_trajectories[i, -1, 2]
        ax.plot(x0, z0, 'o',color=ensemble_color, alpha=0.3, markersize=2, zorder=5)

    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("auto")


def plot_ensemble_panels(ensemble_list, reference_trajectory, titles,
                         figsize=(18, 5), save_path=None):
    """Create a multi-panel figure showing ensembles from different initial regions.

    Parameters
    ----------
    ensemble_list : list of np.ndarray
        List of ensemble arrays, each shape (n_members, n_steps+1, 3).
    reference_trajectory : np.ndarray
        Long reference trajectory for the background attractor.
    titles : list of str
        Title for each panel.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, len(ensemble_list), figsize=figsize)

    if len(ensemble_list) == 1:
        axes = [axes]

    for ax, ensemble, title in zip(axes, ensemble_list, titles):
        plot_ensemble(ax, ensemble, reference_trajectory, title=title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, axes


if __name__ == "__main__":
    # Quick visual test with random data
    fake_traj = np.cumsum(np.random.randn(500, 3) * 0.5, axis=0)
    fig, ax = plt.subplots()
    plot_attractor(ax, fake_traj)
    ax.set_title("plotting.py: visual test")
    plt.show()
    print("plotting.py: visual check — does the plot look reasonable?")
