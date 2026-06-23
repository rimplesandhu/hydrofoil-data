"""Command-line interface for hydrofoil_data, installed as `hfdata`."""

from __future__ import annotations

import argparse
import os

import yaml

from hydrofoil_data.get_nf_airfoils import main as list_airfoils
from hydrofoil_data.postprocess import (
    compute_cavitation_proxy,
    compute_cavitation_sigma,
    plot_foil_re_comparison,
    summarize_convergence,
)
from hydrofoil_data.shapes import load_all_shapes, plot_shapes
from hydrofoil_data.sweep import run_full_sweep


def run(config_path: str) -> None:
    """Run the XFoil + NeuralFoil sweep and plot one figure per foil/Re."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run both solvers over every (foil, alpha, Re) combination
    ds = run_full_sweep(config)
    ds = compute_cavitation_proxy(ds)

    # Persist the combined dataset and a convergence/confidence summary
    out_path = config["output"]["path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")
    summary = summarize_convergence(ds)
    summary.to_csv("output/data/convergence_summary.csv")
    print(summary)

    # Plot the airfoil shapes being swept
    plot_shapes(
        load_all_shapes(config),
        save_path="output/figures/airfoil_shapes.png",
    )

    # One comparison figure per (foil, Re) pair
    op = config["operating"]
    re_values = [float(r) for r in config["sweep"]["reynolds"]]
    sigma_by_re = compute_cavitation_sigma(
        re_values, float(op["chord"]), float(op["depth"]),
        float(op["temperature"]),
    )
    for foil_id in ds["foil_id"].values:
        for re in re_values:
            plot_foil_re_comparison(ds, foil_id, re, sigma_by_re[re])


def main() -> None:
    """Entry point for the `hfdata` command."""
    parser = argparse.ArgumentParser(prog="hfdata")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run the XFoil + NeuralFoil sweep and plot figures"
    )
    run_parser.add_argument("--config", default="configs/sweep_config.yaml")

    subparsers.add_parser(
        "list-airfoils", help="List the AeroSandbox airfoil database"
    )

    args = parser.parse_args()
    if args.command == "run":
        run(args.config)
    elif args.command == "list-airfoils":
        list_airfoils()


if __name__ == "__main__":
    main()
