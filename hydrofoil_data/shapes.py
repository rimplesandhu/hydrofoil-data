"""Airfoil coordinate loaders using AeroSandbox."""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


def load_all_shapes(config: dict) -> dict[str, np.ndarray]:
    """Load all airfoil coordinates specified in the config shapes block."""
    import aerosandbox as asb

    shapes: dict[str, np.ndarray] = {}

    # Load coordinates for every airfoil designation
    for desig in config["shapes"]:
        af = asb.Airfoil(desig).repanel(n_points_per_side=100)
        shapes[desig] = af.coordinates

    return shapes


def load_raw_shapes(config: dict) -> dict[str, np.ndarray]:
    """Load airfoil coordinates as published, without repaneling."""
    import aerosandbox as asb

    shapes: dict[str, np.ndarray] = {}

    # Load each airfoil's coordinates straight from its source table
    for desig in config["shapes"]:
        af = asb.Airfoil(desig)
        shapes[desig] = af.coordinates

    return shapes


def save_raw_shapes(
    shapes: dict[str, np.ndarray], out_dir: str = "output/data/raw_shapes"
) -> None:
    """Write each airfoil's raw coordinates to its own x/c, y/c txt file."""
    os.makedirs(out_dir, exist_ok=True)

    # Write one file per airfoil, named after its designation
    for desig, coords in shapes.items():
        path = os.path.join(out_dir, f"{desig}_raw.txt")
        np.savetxt(path, coords, fmt="%.6f", header="x/c y/c")
        print(f"Saved {path}")


def plot_shapes(
    shapes: dict[str, np.ndarray],
    save_path: str | None = None,
    title: str = "Airfoil profiles",
) -> None:
    """Overlay all airfoil profiles on one axes, normalised to unit chord."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for desig, coords in shapes.items():
        ax.plot(
            coords[:, 0], coords[:, 1], label=desig, linewidth=0.8,
            marker="o", markersize=1.5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()

    # Save to file if requested, otherwise show interactively
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()
