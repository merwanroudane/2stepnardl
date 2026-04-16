"""
plotting.py — Matplotlib charts for TwoStepNARDL post-estimation
================================================================
Provides publication-quality figures for:
  - Cumulative dynamic multipliers
  - Persistence profile (half-life)
  - Impulse response functions
  - CUSUM stability test
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from .model import NARDLResults
from . import postestimation as pe


def _setup_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_multipliers(
    res: NARDLResults,
    horizon: int = 40,
    pos_color: str = "#1a6faf",
    neg_color: str = "#b22222",
    asym_color: str = "#2e8b57",
    title: str = "Cumulative Dynamic Multipliers",
    figsize: tuple = (9, 5),
    ax=None,
    show: bool = True,
):
    """
    Plot positive, negative and asymmetry (difference) multiplier paths.

    Parameters
    ----------
    res : NARDLResults
    horizon : int
    pos_color, neg_color, asym_color : str  hex or named colors
    title : str
    figsize : tuple
    ax : matplotlib Axes or None
    show : bool — call plt.show()

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    mult = pe.multipliers(res, horizon=horizon, show_table=False)
    h = np.arange(1, horizon + 1)

    fig, ax_ = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))

    if res.k == 1:
        mp = mult["pos"].values
        mn = mult["neg"].values
        md = mult["diff"].values
        lr_p = float(res.b_lr[0])
        lr_n = float(res.b_lr[1])

        ax_.fill_between(h, 0, md, alpha=0.12, color=asym_color, label="_nolegend_")
        ax_.axhline(0, color="#888888", linewidth=0.8, linestyle="-")
        ax_.plot(h, mp, color=pos_color, linewidth=2.2, label="Positive shock")
        ax_.plot(h, mn, color=neg_color, linewidth=2.2, label="Negative shock")
        ax_.plot(h, md, color=asym_color, linewidth=1.5, linestyle="--", label="Asymmetry (diff)")
        ax_.axhline(lr_p, color=pos_color, linewidth=1, linestyle=":", alpha=0.6)
        ax_.axhline(lr_n, color=neg_color, linewidth=1, linestyle=":", alpha=0.6)

    _setup_ax(ax_, title, "Horizon (periods)", "Cumulative effect")
    ax_.legend(frameon=True, framealpha=0.9, loc="best", fontsize=10)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_halflife(
    res: NARDLResults,
    horizon: int = 40,
    color: str = "#1a6faf",
    title: str = "Persistence Profile",
    figsize: tuple = (9, 5),
    ax=None,
    show: bool = True,
):
    """Plot the persistence profile (PP) of the ECM."""
    import matplotlib.pyplot as plt

    hl_res = pe.half_life(res, horizon=horizon, show_table=False)
    pp = hl_res["pp_series"].values
    h = np.arange(len(pp))

    fig, ax_ = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))

    ax_.plot(h, pp, color=color, linewidth=2.2, label="PP(h)")
    ax_.axhline(0.5, color="#b22222", linewidth=1.2, linestyle="--", alpha=0.7, label="50% threshold")
    ax_.axhline(0.0, color="#888888", linewidth=0.8)

    if "pp_halflife" in hl_res:
        hl_h = hl_res["pp_halflife"]
        ax_.axvline(hl_h, color="#b22222", linewidth=1, linestyle=":", alpha=0.7)
        ax_.annotate(
            f"HL = {hl_h}",
            xy=(hl_h, 0.5), xytext=(hl_h + 1, 0.6),
            fontsize=9, color="#b22222",
            arrowprops=dict(arrowstyle="->", color="#b22222"),
        )

    ax_.set_ylim(-0.1, 1.1)
    _setup_ax(ax_, title, "Horizon (periods)", "Proportion of disequilibrium remaining")
    ax_.legend(frameon=True, framealpha=0.9, fontsize=10)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_irf(
    res: NARDLResults,
    horizon: int = 20,
    pos_color: str = "#1a6faf",
    neg_color: str = "#b22222",
    title: str = "Impulse Response Functions",
    figsize: tuple = (10, 5),
    show: bool = True,
):
    """Plot period-by-period IRF for pos and neg shocks."""
    import matplotlib.pyplot as plt

    irf_df = pe.irf(res, horizon=horizon, show_table=False)
    h = np.arange(1, horizon + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax_, col, color, lbl in [
        (axes[0], "pos_period", pos_color, "Positive shock"),
        (axes[1], "neg_period", neg_color, "Negative shock"),
    ]:
        if col in irf_df.columns:
            vals = irf_df[col].values
            ax_.bar(h, vals, color=color, alpha=0.75, label=lbl, edgecolor="white")
            ax_.axhline(0, color="#888888", linewidth=0.8)
            _setup_ax(ax_, lbl, "Horizon (periods)", "Period response")
            ax_.legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_cusum(
    res: NARDLResults,
    color: str = "#1a6faf",
    band_color: str = "#b22222",
    title: str = "CUSUM Stability Test",
    figsize: tuple = (9, 5),
    show: bool = True,
):
    """CUSUM plot with 5% significance bands."""
    import matplotlib.pyplot as plt

    X = res._sr_X
    y = res._sr_endog
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    rmse = float(np.sqrt(np.var(resid, ddof=X.shape[1])))
    T = len(resid)

    cusum = np.cumsum(resid) / rmse
    obs = np.arange(1, T + 1)

    # 5% bands (approximation: ±0.948 * sqrt(T) * (t - t_min)/(t_max - t_min))
    band = 0.948 * np.sqrt(T) * (obs - 1) / (T - 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(obs, -band, band, alpha=0.08, color=band_color)
    ax.plot(obs, band, color=band_color, linewidth=1.2, linestyle="--", alpha=0.8, label="5% bands")
    ax.plot(obs, -band, color=band_color, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.plot(obs, cusum, color=color, linewidth=2.0, label="CUSUM")
    ax.axhline(0, color="#888888", linewidth=0.8)

    _setup_ax(ax, title, "Observation", "CUSUM")
    ax.legend(fontsize=10)
    ax.annotate(
        "5% significance boundaries shown as dashed lines",
        xy=(0.5, 0.02), xycoords="axes fraction",
        ha="center", fontsize=8.5, color="#555555",
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig
