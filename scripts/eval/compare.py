"""Cross-checkpoint comparison figure: accuracy and band preservation vs bitrate.

Usage: python scripts/eval/compare.py --glob 'scripts/eval/results/*_f3'
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/groups/bci/jonas/workspace/style_guide")
from zyphra_finalized_plot_helpers import (  # noqa: E402
    bottom_legend, new_figure, save_figure, style_axis,
)
import matplotlib.lines as mlines  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

TOKENS_PER_S = 8  # 125 ms frames
PANELS = [  # (task, subset, input representation, codec representation, title)
    ("eyes", "all", "orig_spec", "latents", "eyes open vs closed"),
    ("fist_lr", "real", "orig_wave", "latents", "left vs right fist (real)"),
    ("fists_feet", "real", "orig_wave", "latents", "fists vs feet (real)"),
]


def collect(pattern):
    rows = []
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        if not (d / "accuracy.csv").exists():
            continue
        k = json.loads((d / "model_kwargs.json").read_text())
        acc = pd.read_csv(d / "accuracy.csv")
        bands = pd.read_csv(d / "bands.csv")
        bits = k["n_codebooks"] * math.log2(k["codebook_size"]) * TOKENS_PER_S
        rows.append(dict(
            name=f"{k['n_codebooks']}x{k['codebook_size']}", bits=bits,
            compression=256 * 16 / bits, acc=acc, bands=bands))
    return sorted(rows, key=lambda r: r["bits"])


def get(acc, task, subset, rep):
    m = acc[(acc.task == task) & (acc.subset == subset) & (acc.representation == rep)]
    return (100 * m.acc.iloc[0], 100 * m["std"].iloc[0]) if len(m) else (np.nan, np.nan)


def figure(runs, mode, out_path):
    fig, theme = new_figure(
        "Does more bitrate buy more task information?",
        "EEGBCI accuracy vs codec bitrate — solid: tokens, dashed: original signal (subject-wise CV)",
        mode=mode, figsize=(16, 6.4),
    )
    axes = fig.subplots(1, len(PANELS))
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.06, right=0.99, wspace=0.18)
    x = [r["bits"] for r in runs]
    for ax, (task, subset, in_rep, codec_rep, title) in zip(axes, PANELS):
        style_axis(ax, mode=mode, title=title, grid="y", title_size=13,
                   xlabel="bitrate (bits/s per channel)")
        ys = [get(r["acc"], task, subset, codec_rep)[0] for r in runs]
        es = [get(r["acc"], task, subset, codec_rep)[1] for r in runs]
        base = np.nanmean([get(r["acc"], task, subset, in_rep)[0] for r in runs])
        ax.errorbar(x, ys, yerr=es, color=theme.orange, marker="o", markersize=7,
                    linewidth=2, capsize=3, zorder=3)
        ax.axhline(base, color=theme.blue, linestyle="--", linewidth=1.8)
        ax.text(x[-1], base + 0.8, f"original {base:.0f}", ha="right", fontsize=10,
                color=theme.blue)
        ax.axhline(50, color=theme.muted, linestyle=":", linewidth=1.2)
        ax.text(x[0], 45.8, "chance", fontsize=9, color=theme.muted)
        for r, xi, yi in zip(runs, x, ys):
            ax.annotate(r["name"], (xi, yi), textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=9, color=theme.text_2)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=10)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_ylim(45, 90)
    axes[0].set_ylabel("accuracy (%)", fontsize=12, color=theme.text_2)
    handles = [
        mlines.Line2D([], [], color=theme.orange, marker="o", linewidth=2, label="codec tokens"),
        mlines.Line2D([], [], color=theme.blue, linestyle="--", linewidth=1.8,
                      label="original signal (same classifier)"),
    ]
    bottom_legend(fig, handles, [h.get_label() for h in handles], mode=mode, y=0.02)
    save_figure(fig, out_path, mode=mode, also_svg=False)


def bands_figure(runs, mode, out_path):
    fig, theme = new_figure(
        "Which frequencies survive, by bitrate?",
        "Log-power correlation between reconstruction and original (eyes task)",
        mode=mode, figsize=(12, 6.2),
    )
    ax = fig.subplots()
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.08, right=0.98)
    style_axis(ax, mode=mode, grid="y", xlabel="bitrate (bits/s per channel)")
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    colors = [theme.orange, theme.blue, theme.family_colors.get("nvidia", theme.orange),
              theme.family_colors.get("qwen", theme.blue), theme.muted]
    x = [r["bits"] for r in runs]
    for b, c in zip(bands, colors):
        ys = [r["bands"][(r["bands"].task == "eyes") & (r["bands"].band == b)]
              .log_power_corr.iloc[0] for r in runs]
        ax.plot(x, ys, marker="o", markersize=6, linewidth=2, color=c, label=b)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("log-power correlation", fontsize=12, color=theme.text_2)
    h, l = ax.get_legend_handles_labels()
    bottom_legend(fig, h, l, mode=mode, y=0.02)
    save_figure(fig, out_path, mode=mode, also_svg=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="scripts/eval/results/*_f3")
    ap.add_argument("--out", default="scripts/eval/results/_comparison")
    args = ap.parse_args()
    runs = collect(args.glob)
    if not runs:
        sys.exit(f"no completed results matching {args.glob}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure(runs, "light", out / "accuracy_vs_bitrate.png")
    bands_figure(runs, "light", out / "bands_vs_bitrate.png")

    tbl = []
    for r in runs:
        row = {"model": r["name"], "bits/s": int(r["bits"]), "compression": f"{r['compression']:.0f}x"}
        for task, subset, in_rep, codec_rep, _ in PANELS:
            row[f"{task}_orig"] = round(get(r["acc"], task, subset, in_rep)[0], 1)
            row[f"{task}_tokens"] = round(get(r["acc"], task, subset, codec_rep)[0], 1)
        tbl.append(row)
    df = pd.DataFrame(tbl)
    df.to_csv(out / "comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    print(f"\nwrote {out}/accuracy_vs_bitrate.png, bands_vs_bitrate.png, comparison.csv")


if __name__ == "__main__":
    main()
