"""Post-processing utilities: cavitation proxy, convergence summary, plots."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_cavitation_sigma(
    re_values: list[float], chord: float, depth: float, temperature: float,
) -> dict[float, float]:
    """Cavitation index sigma per Re from operating depth/temperature."""
    t_c = temperature
    rho_w = 999.842 - 0.0708 * t_c - 0.00379 * t_c ** 2
    mu_w = 1.787e-3 * np.exp(
        -0.03144 * t_c + 1.69e-4 * t_c ** 2 - 2.02e-6 * t_c ** 3
    )
    pv_w = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    p_inf = 101_325.0 + rho_w * 9.81 * depth

    sigma = {}
    for re in re_values:
        v = re * mu_w / (rho_w * chord)
        sigma[re] = (p_inf - pv_w) / (0.5 * rho_w * v ** 2)
    return sigma


def compute_cavitation_proxy(ds: xr.Dataset) -> xr.Dataset:
    """Add cavitation inception proxy variable sigma_inception = -Cp_min.

    Under the fully-wetted assumption, cavitation inception occurs when
    the local pressure equals vapour pressure, giving sigma_i = -Cp_min.

    Args:
        ds: xarray Dataset containing the Cp_min variable.

    Returns:
        Dataset with new data variable sigma_inception added.
    """
    ds = ds.copy()
    ds["sigma_inception"] = -ds["Cp_min"]
    ds["sigma_inception"].attrs["long_name"] = (
        "Cavitation inception number (fully-wetted)"
    )
    ds["sigma_inception"].attrs["units"] = "dimensionless"
    return ds


def summarize_convergence(ds: xr.Dataset) -> pd.DataFrame:
    """Summarise XFoil convergence fraction and NeuralFoil confidence per foil.

    Args:
        ds: xarray Dataset with converged and analysis_confidence variables.

    Returns:
        DataFrame indexed by (foil_id, Re) with columns:
        xfoil_converge_fraction, neuralfoil_mean_confidence.
    """
    records = []

    for foil_id in ds["foil_id"].values:
        for re in ds["Re"].values:
            row: dict = {"foil_id": foil_id, "Re": float(re)}

            # XFoil convergence fraction
            if "xfoil" in ds["fidelity"].values:
                conv = ds["converged"].sel(
                    foil_id=foil_id, Re=re, fidelity="xfoil"
                ).values
                valid = conv[np.isfinite(conv)]
                row["xfoil_converge_fraction"] = (
                    float(valid.mean()) if len(valid) > 0 else float("nan")
                )
            else:
                row["xfoil_converge_fraction"] = float("nan")

            # NeuralFoil mean analysis confidence
            if "neuralfoil" in ds["fidelity"].values:
                conf = ds["analysis_confidence"].sel(
                    foil_id=foil_id, Re=re, fidelity="neuralfoil"
                ).values
                valid_c = conf[np.isfinite(conf)]
                row["neuralfoil_mean_confidence"] = (
                    float(valid_c.mean()) if len(valid_c) > 0 else float("nan")
                )
            else:
                row["neuralfoil_mean_confidence"] = float("nan")

            records.append(row)

    df = pd.DataFrame(records).set_index(["foil_id", "Re"])
    return df


def save_full_results(ds: xr.Dataset, out_path: str) -> None:
    """Write every (foil_id, alpha, Re, fidelity) row and variable, no summary."""
    df = ds.to_dataframe().reset_index()
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def plot_polars(
    ds: xr.Dataset,
    foil_id: str,
    re: float,
    sigma: float | list[float] | None = None,
) -> None:
    """Plot Cl, Cl/Cd, Cp_min with sigma lines, and analysis confidence."""
    os.makedirs("output/figures", exist_ok=True)

    # Decide which optional panels are available
    has_conf = (
        "neuralfoil" in ds["fidelity"].values
        and "analysis_confidence" in ds.data_vars
    )
    has_cpmin = "Cp_min" in ds.data_vars
    ncols = 2 + int(has_cpmin) + int(has_conf)
    figw = ncols * 6

    alphas = ds["alpha"].values
    fig, axes = plt.subplots(1, ncols, figsize=(figw, 5))

    style = {
        "neuralfoil": ("C0", "--", "NeuralFoil"),
        "xfoil":      ("C1", "-",  "XFoil"),
    }

    # Cl and L/D for all fidelities
    for fid in ds["fidelity"].values:
        color, ls, label = style.get(fid, ("C2", ":", fid))
        subset = ds.sel(foil_id=foil_id, Re=re, fidelity=fid)
        cl = subset["Cl"].values
        cd = subset["Cd"].values

        axes[0].plot(alphas, cl, color=color, ls=ls, label=label, lw=1.5)

        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(np.abs(cd) > 1e-10, cl / cd, np.nan)
        axes[1].plot(alphas, ld, color=color, ls=ls, label=label, lw=1.5)

    for ax, ylabel, title in zip(
        axes[:2],
        ["$C_l$", "$C_l / C_d$"],
        ["Lift polar", "Lift-to-drag polar"],
    ):
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{foil_id}  Re={re:.0e}  —  {title}")
        ax.legend()
        ax.grid(True, linewidth=0.4)

    # Cp_min panel for all fidelities with optional sigma lines
    ax_idx = 2
    if has_cpmin:
        ax = axes[ax_idx]
        for fid in ds["fidelity"].values:
            color, ls, label = style.get(fid, ("C2", ":", fid))
            cp = ds["Cp_min"].sel(
                foil_id=foil_id, Re=re, fidelity=fid
            ).values
            ax.plot(alphas, cp, color=color, ls=ls, label=label, lw=1.5)

        # Vapor-pressure lines: Cp_v = -sigma
        sigmas = (
            [sigma] if isinstance(sigma, (int, float))
            else (sigma if sigma is not None else [])
        )
        for s in sigmas:
            ax.axhline(
                -float(s),
                color="red",
                ls=":",
                lw=1.2,
                label=f"$C_{{p,v}}$ (σ={s:.2g})",
            )

        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel("$C_{p,\\min}$")
        ax.set_title(f"{foil_id}  Re={re:.0e}  —  $C_{{p,\\min}}$")
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.4)
        ax_idx += 1

    # Analysis confidence panel (NeuralFoil only)
    if has_conf:
        ax = axes[ax_idx]
        conf = ds["analysis_confidence"].sel(
            foil_id=foil_id, Re=re, fidelity="neuralfoil"
        ).values
        color, ls, label = style["neuralfoil"]
        ax.plot(alphas, conf, color=color, ls=ls, label=label, lw=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel("Analysis confidence")
        ax.set_title(
            f"{foil_id}  Re={re:.0e}  —  Analysis confidence"
        )
        ax.legend()
        ax.grid(True, linewidth=0.4)

    fig.tight_layout()
    fname = f"output/figures/{foil_id}_Re{re:.0e}_polars.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def plot_cp_min_map(
    ds: xr.Dataset,
    fidelity: str,
    sigma: float | list[float] | None = None,
) -> None:
    """Heatmap of Cp_min with optional vapor-pressure contour lines."""
    os.makedirs("output/figures", exist_ok=True)

    re_values = ds["Re"].values
    re_mid = re_values[len(re_values) // 2]

    data = (
        ds["Cp_min"]
        .sel(Re=re_mid, fidelity=fidelity)
        .values
    )  # shape: (n_foils, n_alpha)

    foil_ids = ds["foil_id"].values
    alphas = ds["alpha"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        alphas,
        np.arange(len(foil_ids)),
        data,
        cmap="viridis_r",
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="$C_{p,\\min}$")

    # Vapor-pressure contour: Cp_v = -sigma (cavitation onset when Cp < Cp_v)
    sigmas = (
        [sigma] if isinstance(sigma, (int, float))
        else (sigma if sigma is not None else [])
    )
    for s in sigmas:
        cp_v = -float(s)
        cs = ax.contour(
            alphas,
            np.arange(len(foil_ids)),
            data,
            levels=[cp_v],
            colors=["red"],
            linewidths=1.5,
            linestyles="--",
        )
        ax.clabel(
            cs,
            fmt=f"$C_{{p,v}}$ (σ={s:.2g})",
            fontsize=7,
            inline=True,
        )

    ax.set_yticks(np.arange(len(foil_ids)))
    ax.set_yticklabels(foil_ids, fontsize=8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_title(
        f"$C_{{p,\\min}}$ map — {fidelity}  Re={re_mid:.0e}"
    )
    fig.tight_layout()

    fname = f"output/figures/cpmin_map_{fidelity}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def plot_confidence_map(ds: xr.Dataset) -> None:
    """Heatmap of NeuralFoil analysis_confidence over (alpha, foil_id)."""
    os.makedirs("output/figures", exist_ok=True)

    if "analysis_confidence" not in ds.data_vars:
        print("analysis_confidence not in dataset; skipping.")
        return
    if "neuralfoil" not in ds["fidelity"].values:
        print("neuralfoil fidelity not in dataset; skipping.")
        return

    # Use middle Re value
    re_values = ds["Re"].values
    re_mid = re_values[len(re_values) // 2]

    data = (
        ds["analysis_confidence"]
        .sel(Re=re_mid, fidelity="neuralfoil")
        .values
    )  # shape: (n_foils, n_alpha)

    foil_ids = ds["foil_id"].values
    alphas = ds["alpha"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        alphas,
        np.arange(len(foil_ids)),
        data,
        cmap="RdYlGn",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Analysis confidence")
    ax.set_yticks(np.arange(len(foil_ids)))
    ax.set_yticklabels(foil_ids, fontsize=8)
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_title(
        f"NeuralFoil analysis confidence map  Re={re_mid:.0e}"
    )
    fig.tight_layout()

    fname = "output/figures/confidence_map_neuralfoil.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def plot_foil_re_comparison(
    ds: xr.Dataset, foil_id: str, re: float, sigma: float,
) -> None:
    """One (foil, Re) figure: XFoil points vs NeuralFoil line, 3x2 grid."""
    os.makedirs(f"output/figures/{foil_id}", exist_ok=True)
    sub = ds.sel(foil_id=foil_id, Re=re)
    alpha = sub["alpha"].values
    has_xfoil = "xfoil" in ds["fidelity"].values
    has_nf = "neuralfoil" in ds["fidelity"].values

    xf = sub.sel(fidelity="xfoil") if has_xfoil else None
    nf = sub.sel(fidelity="neuralfoil") if has_nf else None

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    ax_cl, ax_cd, ax_cm, ax_cpmin, ax_conf, ax_bkt = axes.flatten()

    # Quantities shared by both solvers: XFoil points, NeuralFoil line
    shared = [
        (ax_cl, "Cl", "$C_l$"), (ax_cd, "Cd", "$C_d$"),
        (ax_cm, "Cm", "$C_m$"),
    ]
    for ax, var, ylabel in shared:
        if has_xfoil:
            ax.plot(
                alpha, xf[var].values, "o", color="black", markersize=4,
                label="XFoil",
            )
        if has_nf:
            ax.plot(
                alpha, nf[var].values, color="tab:blue", lw=1.5,
                label="NeuralFoil",
            )
        ax.set_xlabel(r"$\alpha$ (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.4)
        ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
        ax.legend(fontsize=8)

    # Cp_min panel: plotted as -Cp_min (positive = stronger suction), same
    # sign convention as the cavitation bucket and the sigma threshold line
    if has_xfoil:
        ax_cpmin.plot(
            alpha, -xf["Cp_min"].values, "o", color="black", markersize=4,
            label="XFoil",
        )
    if has_nf:
        ax_cpmin.plot(
            alpha, -nf["Cp_min"].values, color="tab:blue", lw=1.5,
            label="NeuralFoil",
        )
    ax_cpmin.set_xlabel(r"$\alpha$ (deg)")
    ax_cpmin.set_ylabel("$-C_{p,\\min}$")
    ax_cpmin.grid(True, linewidth=0.4)
    ax_cpmin.axhline(0, color="k", linewidth=0.4, linestyle="--")
    ax_cpmin.legend(fontsize=8)

    # NeuralFoil analysis confidence, with XFoil convergence (1/0) overlaid
    if has_nf:
        ax_conf.plot(
            alpha, nf["analysis_confidence"].values, color="tab:blue",
            lw=1.5, label="NeuralFoil confidence",
        )
    if has_xfoil:
        ax_conf.plot(
            alpha, xf["converged"].values, "x", color="black",
            markersize=6, label="XFoil converged",
        )
    ax_conf.set_ylim(0, 1)
    ax_conf.set_xlabel(r"$\alpha$ (deg)")
    ax_conf.set_ylabel("Analysis confidence")
    ax_conf.grid(True, linewidth=0.4)
    ax_conf.axhline(0, color="k", linewidth=0.4, linestyle="--")
    ax_conf.legend(fontsize=8)

    # Cavitation bucket: Cl vs sigma_i = -Cp_min, both solvers as points/line
    if has_nf:
        nf_order = np.argsort(nf["Cl"].values)
        ax_bkt.plot(
            nf["Cl"].values[nf_order], -nf["Cp_min"].values[nf_order],
            color="tab:blue", lw=1.5, label="NeuralFoil",
        )
    if has_xfoil:
        ax_bkt.plot(
            xf["Cl"].values, -xf["Cp_min"].values, "o", color="black",
            markersize=4, label="XFoil",
        )
    ax_bkt.axhline(sigma, color="red", ls=":", lw=2.0)
    ax_bkt.set_xlabel("$C_l$")
    ax_bkt.set_ylabel("$\\sigma_i = -C_{p,\\min}$")
    ax_bkt.set_title("Cavitation bucket")
    ax_bkt.grid(True, linewidth=0.4)
    ax_bkt.legend(fontsize=8)

    fig.suptitle(
        f"{foil_id}  —  XFoil vs NeuralFoil  Re={re:.0e}  σ={sigma:.2g}",
        fontsize=13,
    )
    fig.tight_layout()

    re_tag = f"Re{re:.0e}".replace("+", "")
    out_path = f"output/figures/{foil_id}/{foil_id}_{re_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
