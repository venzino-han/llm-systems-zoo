"""Matplotlib helpers so every figure in reports/ looks the same."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_lines(
    x: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    path: str | Path | None = None,
    logx: bool = False,
    logy: bool = False,
    marker: str = "o",
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(x, ys, marker=marker, label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend()
    if path:
        save_fig(fig, path)
    return fig


def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ylabel: str,
    title: str = "",
    path: str | Path | None = None,
    errors: Sequence[float] | None = None,
    annotate: bool = True,
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels)), 4))
    bars = ax.bar(labels, values, yerr=errors, capsize=3, color=PALETTE[: len(labels)])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if annotate:
        for b, v in zip(bars, values):
            ax.annotate(f"{v:,.3g}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    if path:
        save_fig(fig, path)
    return fig
