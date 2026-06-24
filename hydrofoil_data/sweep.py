"""Full multi-fidelity parameter sweep orchestration."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from hydrofoil_data.shapes import load_all_shapes
from hydrofoil_data.solvers.neuralfoil_solver import run_neuralfoil
from hydrofoil_data.solvers.xfoil_solver import run_xfoil

logger = logging.getLogger(__name__)


def run_full_sweep(config: dict) -> xr.Dataset:
    """Run the full multi-fidelity sweep over all shapes and conditions."""
    shapes = load_all_shapes(config)
    foil_ids = list(shapes.keys())

    sweep_cfg = config["sweep"]
    alphas = list(
        np.arange(
            sweep_cfg["alpha_start"],
            sweep_cfg["alpha_end"] + 0.5 * sweep_cfg["alpha_step"],
            sweep_cfg["alpha_step"],
        )
    )
    re_values: list[float] = [float(r) for r in sweep_cfg["reynolds"]]
    fidelities: list[str] = _active_fidelities(config["solvers"])

    # Pre-allocate NaN arrays: dims = (foil, alpha, Re, fidelity)
    shape4 = (len(foil_ids), len(alphas), len(re_values), len(fidelities))
    nan4 = lambda: np.full(shape4, np.nan, dtype=np.float64)

    cl_arr = nan4()
    cd_arr = nan4()
    cm_arr = nan4()
    cpmin_arr = nan4()
    conv_arr = nan4()        # 1.0 / 0.0 / NaN
    conf_arr = nan4()        # NeuralFoil only

    # Fill arrays
    for fi, foil_id in enumerate(foil_ids):
        coords = shapes[foil_id]
        for ri, re in enumerate(re_values):
            for fidi, fidelity in enumerate(fidelities):
                print(
                    f"  Running: foil={foil_id}  Re={re:.2e}  "
                    f"fidelity={fidelity}"
                )
                df = _run_solver(
                    fidelity, coords, alphas, re, config["solvers"]
                )
                if df is None:
                    continue

                for ai, alpha in enumerate(alphas):
                    row = df[np.isclose(df["alpha"], alpha)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    cl_arr[fi, ai, ri, fidi] = row.get("Cl", np.nan)
                    cd_arr[fi, ai, ri, fidi] = row.get("Cd", np.nan)
                    cm_arr[fi, ai, ri, fidi] = row.get("Cm", np.nan)
                    cpmin_arr[fi, ai, ri, fidi] = row.get("Cp_min", np.nan)

                    if fidelity == "xfoil":
                        conv_arr[fi, ai, ri, fidi] = row.get(
                            "converged", np.nan
                        )
                    if fidelity == "neuralfoil":
                        conf_arr[fi, ai, ri, fidi] = row.get(
                            "analysis_confidence", np.nan
                        )

    # Build xarray Dataset
    coords_ds = {
        "foil_id": foil_ids,
        "alpha": alphas,
        "Re": re_values,
        "fidelity": fidelities,
    }
    dims = ("foil_id", "alpha", "Re", "fidelity")

    ds = xr.Dataset(
        {
            "Cl": (dims, cl_arr),
            "Cd": (dims, cd_arr),
            "Cm": (dims, cm_arr),
            "Cp_min": (dims, cpmin_arr),
            "converged": (dims, conv_arr),
            "analysis_confidence": (dims, conf_arr),
        },
        coords=coords_ds,
    )
    ds["alpha"].attrs["units"] = "degrees"
    ds["Re"].attrs["long_name"] = "Reynolds number"
    return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_fidelities(solvers_cfg: dict) -> list[str]:
    """Return list of enabled fidelity labels from config."""
    order = ["neuralfoil", "xfoil"]
    return [f for f in order if solvers_cfg.get(f, {}).get("enabled", False)]


def _run_solver(
    fidelity: str,
    coords: np.ndarray,
    alphas: list[float],
    re: float,
    solvers_cfg: dict,
) -> pd.DataFrame | None:
    """Dispatch to the appropriate solver and return its DataFrame."""
    if fidelity == "neuralfoil":
        model_size = solvers_cfg["neuralfoil"].get("model_size", "large")
        try:
            return run_neuralfoil(coords, alphas, re, model_size=model_size)
        except Exception as exc:
            logger.error(
                "NeuralFoil sweep failed for Re=%.0f: %s", re, exc
            )
            return None

    if fidelity == "xfoil":
        n_crit = solvers_cfg["xfoil"].get("n_crit", 9.0)
        max_iter = solvers_cfg["xfoil"].get("max_iter", 100)
        try:
            return run_xfoil(
                coords, alphas, re, n_crit=n_crit, max_iter=max_iter
            )
        except Exception as exc:
            logger.error(
                "XFoil sweep failed for Re=%.0f: %s", re, exc
            )
            return None

    raise ValueError(f"Unknown fidelity '{fidelity}'")
