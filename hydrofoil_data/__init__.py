"""Multifidelity hydrofoil aerodynamic data generation."""

from hydrofoil_data.shapes import load_all_shapes
from hydrofoil_data.sweep import run_full_sweep
from hydrofoil_data.postprocess import (
    compute_cavitation_proxy,
    compute_cavitation_sigma,
    summarize_convergence,
    plot_confidence_map,
    plot_foil_re_comparison,
)

__all__ = [
    "load_all_shapes",
    "run_full_sweep",
    "compute_cavitation_proxy",
    "compute_cavitation_sigma",
    "summarize_convergence",
    "plot_confidence_map",
    "plot_foil_re_comparison",
]
