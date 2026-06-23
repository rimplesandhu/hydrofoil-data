"""Aerodynamic solvers: NeuralFoil and XFoil wrappers."""

from hydrofoil_data.solvers.neuralfoil_solver import run_neuralfoil
from hydrofoil_data.solvers.xfoil_solver import run_xfoil

__all__ = ["run_neuralfoil", "run_xfoil"]
