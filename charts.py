"""Matplotlib figures for the Streamlit page and the training notebook.

Kept apart from app.py so each figure can be drawn without starting
Streamlit. Every function returns a Figure; the caller shows and closes it.
No backend is forced here, so the notebook keeps its inline plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scoring import METRICS

INK, PAPER, SAND = "#1f2933", "#faf8f5", "#efe9df"
TEAL, BRICK = "#0f766e", "#b4443c"
MODEL_COLOURS = {
    "Logistic Regression": "#2f6f73",
    "Decision Tree": "#a8742a",
    "KNN": "#5b7aa3",
    "Naive Bayes": "#7d6b8f",
    "Support Vector Machine": "#b4543c",
}
SAND_TEAL = LinearSegmentedColormap.from_list("sand_teal", [SAND, TEAL])


def style_axes(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9aa5b1")
    ax.tick_params(colors=INK, labelsize=9)


def dot_plot(scores: pd.DataFrame):
    """One row per metric, one dot per model."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    ypos = np.arange(len(METRICS))[::-1]
    for name, row in scores.iterrows():
        ax.scatter(
            row[METRICS].to_numpy(float),
            ypos,
            s=72,
            color=MODEL_COLOURS.get(name, INK),
            edgecolor=PAPER,
            linewidth=0.9,
            zorder=3,
            label=name,
        )
    ax.set_yticks(ypos, METRICS)
    ax.grid(axis="x", color="#ddd5c6", linewidth=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    values = scores[METRICS].to_numpy(float)
    if np.isfinite(values).any():
        ax.set_xlim(max(0.0, np.nanmin(values) - 0.05), min(1.0, np.nanmax(values) + 0.04))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def class_f1_heat(class_f1: pd.DataFrame):
    """Models down the side, outcomes across the top, F1 in each cell."""
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    fig.patch.set_facecolor(PAPER)
    values = class_f1.to_numpy(float)
    ax.imshow(values, cmap=SAND_TEAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(values.shape[1]), list(class_f1.columns))
    ax.set_yticks(range(values.shape[0]), list(class_f1.index))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(
                j, i, "n/a" if np.isnan(v) else f"{v:.2f}",
                ha="center", va="center", fontsize=10,
                color="white" if v > 0.6 else INK,
            )
    ax.tick_params(colors=INK, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def confusion_figure(counts: pd.DataFrame, as_share: bool):
    raw = counts.to_numpy(float)
    totals = raw.sum(axis=1, keepdims=True)
    shown = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0) if as_share else raw
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor(PAPER)
    ax.imshow(shown, cmap=SAND_TEAL, vmin=0, vmax=max(shown.max(), 1e-9))
    n = len(counts)
    ax.set_xticks(range(n), list(counts.columns))
    ax.set_yticks(range(n), list(counts.index))
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Actual", color=INK)
    for i in range(n):
        for j in range(n):
            text = f"{shown[i, j]:.0%}" if as_share else f"{int(raw[i, j])}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=11,
                color="white" if shown[i, j] > 0.55 * shown.max() else INK,
            )
    ax.tick_params(colors=INK, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def weight_bars(items, signed: bool):
    """Horizontal bars for a model's top features (largest at the top)."""
    # sklearn's name for the pooled rare codes is not something to show a reader
    names = [item["feature"].replace("_infrequent_sklearn", "_rare_codes") for item in items][::-1]
    weights = [item["weight"] for item in items][::-1]
    colours = [BRICK if w > 0 else TEAL for w in weights] if signed else [TEAL] * len(weights)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    ax.barh(names, weights, color=colours)
    if signed:
        ax.axvline(0, color=INK, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    return fig


def risk_histogram(p_dropout, is_dropout, threshold):
    """Predicted dropout chance, split by what really happened."""
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    bins = np.linspace(0, 1, 21)
    ax.hist(p_dropout[~is_dropout], bins=bins, color=TEAL, alpha=0.8, label="did not drop out")
    ax.hist(p_dropout[is_dropout], bins=bins, color=BRICK, alpha=0.8, label="dropped out")
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.1)
    ax.set_xlabel("Predicted chance of dropping out", color=INK)
    ax.set_ylabel("Students", color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig
