"""Airfoil coordinate loaders using AeroSandbox."""

from __future__ import annotations

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

FIXES_DIR = os.path.join(os.path.dirname(__file__), "airfoil_fixes")


def _load_airfoil(desig: str):
    """Build an aerosandbox Airfoil, preferring a local coordinate fix."""
    import aerosandbox as asb

    # Some UIUC entries (e.g. naca633218) are known to be wrong; a
    # corrected coordinate file overrides the bundled database if present
    fix_path = os.path.join(FIXES_DIR, f"{desig}.dat")
    if os.path.exists(fix_path):
        coordinates = np.loadtxt(fix_path)
        return asb.Airfoil(name=desig, coordinates=coordinates)

    return asb.Airfoil(desig)


def load_all_shapes(config: dict) -> dict[str, np.ndarray]:
    """Load all airfoil coordinates, repanelling per the config's settings."""
    repanel_cfg = config.get("repanel", {})
    enabled = repanel_cfg.get("enabled", True)
    n_points_per_side = repanel_cfg.get("n_points_per_side", 100)

    shapes: dict[str, np.ndarray] = {}

    # Load coordinates for every airfoil designation
    for desig in config["shapes"]:
        af = _load_airfoil(desig)
        if enabled:
            af = af.repanel(n_points_per_side=n_points_per_side)
        shapes[desig] = af.coordinates

    return shapes


def load_raw_shapes(config: dict) -> dict[str, np.ndarray]:
    """Load airfoil coordinates as published, without repaneling."""
    shapes: dict[str, np.ndarray] = {}

    # Load each airfoil's coordinates straight from its source table
    for desig in config["shapes"]:
        af = _load_airfoil(desig)
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


def save_shapes(
    shapes: dict[str, np.ndarray], out_path: str = "output/data/hydrofoil_hkt5"
) -> None:
    """Write all foils' coordinates to one txt and one nc file.

    Foils with fewer points than the longest one are padded with NaN,
    since raw (unrepanelled) coordinates can differ in point count.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    foil_ids = list(shapes.keys())
    n_points = max(shapes[foil_id].shape[0] for foil_id in foil_ids)

    # Pad every foil's coordinates to the same length with NaN
    coords = np.full((len(foil_ids), n_points, 2), np.nan)
    for i, foil_id in enumerate(foil_ids):
        n = shapes[foil_id].shape[0]
        coords[i, :n, :] = shapes[foil_id]

    # Write one foil-name header followed by its x/c, y/c rows per block
    txt_path = f"{out_path}_shapes.txt"
    with open(txt_path, "w") as f:
        for foil_id in foil_ids:
            f.write(f"# {foil_id}\n")
            np.savetxt(f, shapes[foil_id], fmt="%.6f")
            f.write("\n")
    print(f"Saved {txt_path}")

    # Combine every foil's coordinates into one nc file along foil_id
    nc_path = f"{out_path}_shapes.nc"
    ds = xr.Dataset(
        {
            "x": (("foil_id", "point"), coords[:, :, 0]),
            "y": (("foil_id", "point"), coords[:, :, 1]),
        },
        coords={"foil_id": foil_ids},
    )
    ds.to_netcdf(nc_path)
    print(f"Saved {nc_path}")


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
