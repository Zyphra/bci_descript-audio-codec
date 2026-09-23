"""Figures for the EEGBCI classifier eval — styled to match the terminal chart.

Horizontal bars, cyan = computed from the original signal, orange = through the codec.
Reads accuracy.csv / bands.csv from classify.py.

Usage: python scripts/eval/plots.py --results scripts/eval/results/<name>
"""
import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/groups/bci/jonas/workspace/style_guide")
from zyphra_finalized_plot_helpers import (  # noqa: E402
    bottom_legend, new_figure, save_figure, style_axis,
)

# (key, label, arm) — arm: orig = cyan, codec = orange, transfer = outlined
ROWS = [
    ("orig_wave", "input", "orig"),
    ("orig_spec", "input spectrum", "orig"),
    ("orig_wave_nogain", "input (no gain)", "orig_alt"),
    ("codes", "codes", "codec"),
    ("latents", "tokens", "codec"),
    ("recon_wave", "recon", "codec"),
    ("recon_spec", "recon spectrum", "codec"),
    ("recon_wave_nogain", "recon (no gain)", "codec_alt"),
    ("transfer_orig->recon", "transfer o->r", "transfer"),
]
CHANCE = {"eyes": 50.0, "fist_lr": 50.0, "fists_feet": 50.0}


def panels(acc):
    out = []
    for (task, subset), g in acc.groupby(["task", "subset"], sort=False):
        title = task if subset == "all" else f"{task} ({subset})"
        vals = {r["representation"]: (100 * r["acc"], 100 * r["std"]) for _, r in g.iterrows()}
        chance = CHANCE.get(task)
        if task.startswith("subject_id"):
            chance = 100.0 / 109
            title = "subject identification (109-way)"
        out.append((title.replace("_", " "), vals, chance))
    return out


def accuracy_figure(acc, mode, model_name, out_path):
    ps = panels(acc)
    ncol = min(3, len(ps))
    nrow = int(np.ceil(len(ps) / ncol))
    fig, theme = new_figure(
        "Does the codec keep task information?",
        f"EEGBCI classification accuracy — {model_name}",
        mode=mode, figsize=(6.2 * ncol, 2.0 + 3.6 * nrow), title_y=0.975,
    )
    axes = np.atleast_1d(fig.subplots(nrow, ncol)).ravel()
    top = 1 - (1.15 / (2.0 + 3.6 * nrow))
    fig.subplots_adjust(top=top, bottom=0.10 + 0.10 / nrow,
                        left=0.11, right=0.98, hspace=0.6, wspace=0.42)
    colors = {"orig": theme.blue, "orig_alt": theme.blue,
              "codec": theme.orange, "codec_alt": theme.orange}
    for ax, (title, vals, chance) in zip(axes, ps):
        present = [(k, lab, arm) for k, lab, arm in ROWS if k in vals]
        style_axis(ax, mode=mode, title=title, grid="x", title_size=12.5)
        ys = np.arange(len(present))[::-1]
        for yi, (key, label, arm) in zip(ys, present):
            v, sd = vals[key]
            alt = arm.endswith("_alt")
            ax.barh(yi, v, height=0.68,
                    color="none" if arm == "transfer" else colors[arm],
                    edgecolor=theme.orange if arm == "transfer" else colors[arm],
                    linewidth=1.3, alpha=0.45 if alt else 1.0, zorder=2)
            ax.errorbar(v, yi, xerr=sd, color=theme.muted, linewidth=1.0, capsize=2, zorder=3)
            ax.text(v + sd + 1.0, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=9.5, color=theme.text_2)
        if chance:
            ax.axvline(chance, color=theme.muted, linestyle="--", linewidth=1.1, zorder=1)
            ax.text(chance, len(present) - 0.35, " chance", fontsize=8.5, color=theme.muted)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab, _ in present], fontsize=10)
        hi = max(v for v, _ in (vals[k] for k, _, _ in present))
        ax.set_xlim(0, min(100, hi * 1.28))
        ax.set_xlabel("accuracy (%)", fontsize=11, color=theme.text_2)
        ax.set_ylim(-0.7, len(present) - 0.2)
    for ax in axes[len(ps):]:
        ax.set_visible(False)
    handles = [
        mpatches.Patch(color=theme.blue, label="from original signal"),
        mpatches.Patch(color=theme.orange, label="through codec (tokens / recon)"),
        mpatches.Patch(color=theme.orange, alpha=0.45, label="gain not restored"),
        mpatches.Patch(facecolor="none", edgecolor=theme.orange, linewidth=1.3,
                       label="trained on orig, tested on recon"),
    ]
    bottom_legend(fig, handles, [h.get_label() for h in handles], mode=mode, y=0.005)
    save_figure(fig, out_path, mode=mode, also_svg=False)


def bands_figure(bands, mode, model_name, out_path):
    tasks = list(bands.task.unique())
    fig, theme = new_figure(
        "Which frequency bands survive the codec?",
        f"Log-power correlation, reconstruction vs original — {model_name}",
        mode=mode, figsize=(4.8 * len(tasks) + 1, 5.6),
    )
    axes = np.atleast_1d(fig.subplots(1, len(tasks)))
    fig.subplots_adjust(top=0.74, bottom=0.17, left=0.08, right=0.98, wspace=0.18)
    for ax, task in zip(axes, tasks):
        g = bands[bands.task == task]
        style_axis(ax, mode=mode, title=task.replace("_", " "), grid="y", title_size=12.5)
        ax.bar(range(len(g)), g.log_power_corr, width=0.62, color=theme.orange)
        for x, v in enumerate(g.log_power_corr):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5,
                    color=theme.text_2)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.band, fontsize=10.5)
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("log-power correlation", fontsize=11.5, color=theme.text_2)
    save_figure(fig, out_path, mode=mode, also_svg=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--modes", default="light", help="comma list: light,dark")
    args = ap.parse_args()
    res = Path(args.results)
    model_name = res.name.replace("_", " ")
    acc = pd.read_csv(res / "accuracy.csv")
    bands = pd.read_csv(res / "bands.csv")
    plots = res / "plots"
    for mode in args.modes.split(","):
        accuracy_figure(acc, mode, model_name, plots / f"accuracy_{mode}.png")
        bands_figure(bands, mode, model_name, plots / f"bands_{mode}.png")
    print(f"wrote figures to {plots}/")


if __name__ == "__main__":
    main()
