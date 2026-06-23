# hydrofoil-data

Multifidelity aerodynamic data generation for hydrofoil/tidal-turbine
airfoil sections, combining XFoil (panel method) and NeuralFoil
(neural-network surrogate) solvers.

## What it does

For each airfoil, angle of attack, and Reynolds number in the sweep
config, the pipeline runs both solvers and assembles the results into a
single `xarray` dataset (`Cl`, `Cd`, `Cm`, `Cp_min`, convergence/confidence
flags), then derives a cavitation-inception proxy and writes comparison
plots per airfoil/Re.

## Layout

- `hydrofoil_data/shapes.py` — loads airfoil coordinates via AeroSandbox.
- `hydrofoil_data/solvers/xfoil_solver.py` — drives the compiled
  `bin/xfoil` binary through a batch script and parses its polar output.
- `hydrofoil_data/solvers/neuralfoil_solver.py` — wraps the NeuralFoil
  surrogate model.
- `hydrofoil_data/sweep.py` — orchestrates the sweep over
  (foil, alpha, Re, fidelity) and builds the combined `xarray.Dataset`.
- `hydrofoil_data/postprocess.py` — cavitation proxy, convergence
  summary, and plotting (lift/drag polars, Cp_min, confidence maps,
  cavitation bucket).
- `hydrofoil_data/get_nf_airfoils.py` — lists airfoils in the
  AeroSandbox/NeuralFoil training database.
- `hydrofoil_data/cli.py` — `hfdata` command-line entry point
  (`run`, `list-airfoils`).
- `configs/sweep_config.yaml` — airfoils, alpha/Re ranges, solver
  settings, and operating conditions (chord, depth, temperature) for a
  sweep.

## Usage

```bash
hfdata run --config configs/sweep_config.yaml
hfdata list-airfoils
```

Output dataset and figures are written under `output/`.

## Requirements

Needs a local XFoil binary at `bin/xfoil` (not tracked in this repo —
build it from `xfoil_src/` or your own XFoil install).
