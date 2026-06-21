"""Matplotlib visualization of the court and ball trajectories.

Matplotlib is imported lazily and nothing runs at import time, so importing this
module never opens a window.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from padelvision.config import CourtConfig
from padelvision.geometry.court import court_segments


def _new_ax(ax):
    if ax is not None:
        return ax, False
    import matplotlib.pyplot as plt  # lazy

    fig = plt.figure(figsize=(10, 8))
    return fig.add_subplot(111, projection="3d"), True


def plot_court(config: CourtConfig | None = None, ax=None, show: bool = True):
    """Draw the padel court wireframe. Returns the matplotlib 3D axis."""
    ax, owns = _new_ax(ax)
    for start, end in court_segments(config):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="black",
        )
    ax.set_xlabel("X (length, m)")
    ax.set_ylabel("Y (width, m)")
    ax.set_zlabel("Z (height, m)")
    _set_equal_aspect(ax)
    if show and owns:
        import matplotlib.pyplot as plt

        plt.show()
    return ax


def plot_trajectory(points_3d: np.ndarray, ax=None, show: bool = True, color: str = "red"):
    """Plot a 3D ball trajectory. Returns the matplotlib 3D axis."""
    points_3d = np.asarray(points_3d).reshape(-1, 3)
    ax, owns = _new_ax(ax)
    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker="o", color=color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if show and owns:
        import matplotlib.pyplot as plt

        plt.show()
    return ax


def plot_court_with_trajectory(
    points_3d: np.ndarray, config: CourtConfig | None = None, show: bool = True
):
    """Overlay a ball trajectory on the court wireframe."""
    ax = plot_court(config, show=False)
    plot_trajectory(points_3d, ax=ax, show=False)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return ax


def _set_equal_aspect(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    radius = spans.max() / 2
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
