"""Multifidelity hydrofoil aerodynamic data generation."""

from hydrofoil_data.postprocess import (
    compute_cavitation_proxy,
    compute_cavitation_sigma,
    plot_confidence_map,
    plot_foil_re_comparison,
    summarize_convergence,
)
from hydrofoil_data.shapes import load_all_shapes
from hydrofoil_data.sweep import run_full_sweep

__all__ = [
    "compute_cavitation_proxy",
    "compute_cavitation_sigma",
    "plot_confidence_map",
    "plot_foil_re_comparison",
    "summarize_convergence",
    "load_all_shapes",
    "run_full_sweep",
]
