"""Plot helpers in the Singtel palette (red sampled from the Singtel logo in the brief)."""
import os

import matplotlib.pyplot as plt
import numpy as np

SINGTEL_RED = "#E40230"
DEEP_RED = "#C00000"
MAROON = "#943634"
GREY = "#6B6B6B"
LIGHT = "#F4C7CF"
INK = "#1A1A1A"

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "reports", "figures")


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BBBBBB", "axes.grid": True, "grid.color": "#EEEEEE",
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
    })


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight", dpi=150)


def churn_rate_bars(df, col, target="churn", ax=None, order=None, base=None):
    """Churn rate by category with the base size printed on each bar."""
    ax = ax or plt.gca()
    g = df.groupby(col, observed=True)[target].agg(["mean", "size"])
    if order is not None:
        g = g.reindex(order)
    base = df[target].mean() if base is None else base
    colors = [SINGTEL_RED if m > base * 1.1 else (GREY if m < base * 0.9 else "#B0B0B0") for m in g["mean"]]
    bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=colors)
    for b, (m, n) in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{m:.0%}\nn={int(n):,}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(base * 100, ls="--", color=INK, lw=0.8)
    ax.set_ylim(0, max(g["mean"].max() * 100 * 1.3, base * 100 * 1.5))
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=20)
    return ax


def gains_plot(curves, ax=None, capacity=0.2):
    """curves: {label: DataFrame from churn_model.gain_curve}."""
    ax = ax or plt.gca()
    palette = [SINGTEL_RED, INK, MAROON, GREY]
    for (label, c), col in zip(curves.items(), palette):
        ax.plot(c["share_contacted"] * 100, c["share_churners"] * 100, label=label, color=col, lw=2)
    ax.plot([0, 100], [0, 100], ls=":", color=GREY, label="Random list")
    ax.axvline(capacity * 100, color=DEEP_RED, lw=1, ls="--")
    ax.text(capacity * 100 + 1, 5, "20% outreach capacity", color=DEEP_RED, fontsize=9)
    ax.set_xlabel("% of customers contacted (highest score first)")
    ax.set_ylabel("% of churners reached")
    ax.legend(loc="lower right", frameon=False)
    return ax
