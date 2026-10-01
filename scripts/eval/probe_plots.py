"""Probe overview figures: classifier accuracy pre vs post codec (recon focus).

The question each figure answers: how much of the input-waveform classifier
accuracy survives the codec round trip? (Latent/token evaluation lives with the
downstream transformer work, not here; classify.py can still produce those rows
with EVAL_FULL_REPS=1.)

Reads all battery-tier results under scripts/eval/results/<run>/ and renders:
  results/_overview/probe_summary.png   per-task panels (input vs recon) +
                                        chance-corrected retention average
  results/<run>/card.png                fixed-layout card next to each run's cache
  results/_cards/<run>.png              the same cards side by side

Tasks appear automatically once their results exist (core probe: fist_lr +
ssvep; TASK_P300 / TASK_MOTOR4 in run_eval.sh add panels). The overall score
averages only tasks available for EVERY checkpoint.

Usage: python scripts/eval/probe_plots.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/groups/bci/jonas/workspace/style_guide")
from zyphra_finalized_plot_helpers import (  # noqa: E402
    bottom_legend, new_figure, save_figure, style_axis,
)
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

RES = Path(__file__).parent / "results"
OUT = RES / "_overview"

# (task, subset, chance %, panel title) — order fixes panel order
CANDIDATES = [
    ("fist_lr", "real", 50.0, "left vs right fist (real, 2-class)"),
    ("ssvep", "all", 100.0 / 12, "SSVEP frequency (12-class)"),
    ("p300", "all", 50.0, "P300 target vs nontarget (balanced, 2-class)"),
    ("motor_4class", "all", 25.0, "motor imagery (4-class)"),
]


# ---------------------------------------------------------------- loading

def merged_runs():
    """One merged accuracy frame per run folder (results/<run>/<dataset dir>/...)."""
    runs = {}
    for d in sorted(RES.glob("*/*hop32*")):
        if d.parent.name.startswith("_") or not (d / "accuracy.csv").exists():
            continue
        # battery-tier filter: eegbci _s9, erpbci/seed _s2, bciciv2a/ssvep unsuffixed
        name = d.name
        keep = (name.endswith("_s9") or name.endswith("_s2")
                or (("bciciv2a_" in name or "ssvep_" in name) and "_s" not in name.split("hop32_")[1][9:]))
        if not keep:
            continue
        run = d.parent.name
        k = json.loads((d / "model_kwargs.json").read_text())
        e = runs.setdefault(run, {"run": run, "name": f"{k['n_codebooks']}x{k['codebook_size']}",
                                  "group": run.split("_")[0], "acc": [],
                                  "bits": k["n_codebooks"] * np.log2(k["codebook_size"]) * 8})
        e["acc"].append(pd.read_csv(d / "accuracy.csv"))
    out = []
    for run, e in runs.items():
        e["acc"] = pd.concat(e["acc"], ignore_index=True).drop_duplicates(
            subset=["task", "subset", "representation"], keep="first")
        out.append(e)
    return sorted(out, key=lambda r: (r["bits"], len(r["group"]), r["group"]))


def get(acc, task, subset, rep):
    m = acc[(acc.task == task) & (acc.subset == subset) & (acc.representation == rep)]
    return (100 * m.acc.iloc[0], 100 * m["std"].iloc[0]) if len(m) else (np.nan, np.nan)


def _has(acc, task, subset):
    return np.isfinite(get(acc, task, subset, "recon_wave")[0])


def load_probe_runs():
    """Merged battery-tier runs, restricted to the probe tasks that exist anywhere."""
    runs = merged_runs()
    spec = [c for c in CANDIDATES if any(_has(r["acc"], c[0], c[1]) for r in runs)]
    tasks = {c[0] for c in spec}
    for r in runs:
        a = r["acc"]
        r["acc"] = a[a.task.isin(tasks)
                     & ((a.task != "fist_lr") | (a.subset == "real"))]
    runs = [r for r in runs if len(r["acc"])]
    return runs, spec


# ---------------------------------------------------------------- scores

def probe_rows(runs, spec):
    """Per-checkpoint input/recon accuracies + chance-corrected retention.

    Retention = (recon - chance) / (input - chance); the overall score averages
    only tasks present for every run so mixed-coverage batches rank fairly.
    """
    common = [c for c in spec if all(_has(r["acc"], c[0], c[1]) for r in runs)]
    rows = []
    for r in runs:
        d = {"run": r["run"]}
        for task, subset, chance, _ in spec:
            inp, _ = get(r["acc"], task, subset, "orig_wave")
            acc, sd = get(r["acc"], task, subset, "recon_wave")
            d[(task, "recon")] = (acc, sd)
            d[(task, "input")] = inp
            d[(task, "ret")] = 100 * (acc - chance) / (inp - chance)
        d["overall"] = np.mean([d[(t, "ret")] for t, *_ in common])
        rows.append(d)
    rows.sort(key=lambda d: -d["overall"])
    return rows, common


# ---------------------------------------------------------------- figures

def probe_summary_figure(runs, spec, mode, out_path):
    """One accuracy panel per probe task (input line vs recon bars) + a
    chance-corrected retention panel; checkpoints sorted by overall retention."""
    rows, common = probe_rows(runs, spec)
    labels = [d["run"] for d in rows]
    x = np.arange(len(rows))
    n_pan = len(spec) + 1
    fig, theme = new_figure(
        "Codec probe: classifier accuracy pre vs post",
        "Bars: classifier on the reconstruction - dashed line: same classifier on the input - "
        "bottom: retention (input = 100%), sorted by it",
        mode=mode, figsize=(15, 3.6 * n_pan + 1.6), title_y=0.985)
    axes = np.atleast_1d(fig.subplots(n_pan, 1))
    fig.subplots_adjust(top=1 - 2.0 / (3.6 * n_pan + 1.6), bottom=0.09,
                        left=0.07, right=0.99, hspace=0.6)

    for ax, (task, subset, chance, title) in zip(axes, spec):
        style_axis(ax, mode=mode, title=title, grid="y", title_size=12.5)
        ys = [d[(task, "recon")][0] for d in rows]
        es = [d[(task, "recon")][1] for d in rows]
        ax.bar(x, ys, 0.62, color=theme.orange)
        ax.errorbar(x, ys, yerr=es, fmt="none",
                    ecolor=theme.muted, elinewidth=0.8, capsize=1.5)
        inp = np.nanmean([d[(task, "input")] for d in rows])
        ax.axhline(inp, color=theme.blue, linestyle="--", linewidth=1.6)
        ax.axhline(chance, color=theme.muted, linestyle="-", linewidth=0.8, alpha=0.6)
        ax.set_ylim(40 if chance >= 45 else 0, 100)
        ax.set_ylabel("accuracy (%)", fontsize=11, color=theme.text_2)

    ax = axes[-1]
    tag = " + ".join(t for t, *_ in common)
    style_axis(ax, mode=mode, grid="y", title_size=12.5,
               title=f"retention, averaged over {tag} (input = 100%)")
    ax.bar(x, [d["overall"] for d in rows], 0.62, color=theme.orange)
    ax.axhline(100, color=theme.blue, linestyle="--", linewidth=1.6)
    ax.axhline(0, color=theme.muted, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("retention (%)", fontsize=11, color=theme.text_2)

    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(labels, fontsize=9.5, rotation=25, ha="right")
    handles = [mpatches.Patch(color=theme.orange, label="reconstruction"),
               mlines.Line2D([], [], color=theme.blue, linestyle="--", label="input (raw wave)")]
    bottom_legend(fig, handles, [h.get_label() for h in handles], mode=mode, y=0.008,
                  fontsize=10.5)
    save_figure(fig, out_path, mode=mode, also_svg=False)
    return rows


def checkpoint_cards(rows, spec, mode):
    """One fixed-layout card per checkpoint (identical dimensions and 0-100 axes),
    written as results/<run>/card.png plus a copy in results/_cards/."""
    import shutil
    cards = RES / "_cards"
    cards.mkdir(exist_ok=True)
    width = 0.32
    for d in rows:
        have = [c for c in spec if np.isfinite(d[(c[0], "recon")][0])]
        x = np.arange(len(have))
        fig, theme = new_figure(
            d["run"], "classifier accuracy: input vs reconstruction - probe tasks",
            mode=mode, figsize=(7.5, 6.2), title_y=0.97)
        ax = fig.subplots()
        fig.subplots_adjust(top=0.8, bottom=0.2, left=0.11, right=0.97)
        style_axis(ax, mode=mode, grid="y")
        ax.bar(x - width / 2, [d[(t, "input")] for t, *_ in have], width * 0.9,
               color=theme.blue, alpha=0.8)
        ax.bar(x + width / 2, [d[(t, "recon")][0] for t, *_ in have], width * 0.9,
               color=theme.orange)
        ax.errorbar(x + width / 2, [d[(t, "recon")][0] for t, *_ in have],
                    yerr=[d[(t, "recon")][1] for t, *_ in have], fmt="none",
                    ecolor=theme.muted, elinewidth=0.8, capsize=2)
        for xi, (task, _, chance, _) in zip(x, have):
            ax.plot([xi - 1.4 * width, xi + 1.4 * width], [chance, chance],
                    color=theme.muted, linewidth=1.0)
        ax.text(0.02, 0.95, f"retention {d['overall']:.0f}% (input = 100)",
                transform=ax.transAxes, fontsize=11, color=theme.text_2, va="top")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t} ({round(100 / chance)}-class)"
                            for t, _, chance, _ in have], fontsize=11)
        ax.set_ylabel("accuracy (%)", fontsize=11, color=theme.text_2)
        handles = [mpatches.Patch(color=theme.blue, alpha=0.8, label="input (raw wave)"),
                   mpatches.Patch(color=theme.orange, label="reconstruction"),
                   mlines.Line2D([], [], color=theme.muted, label="chance")]
        bottom_legend(fig, handles, [h.get_label() for h in handles], mode=mode,
                      y=0.02, fontsize=10)
        save_figure(fig, RES / d["run"] / "card.png", mode=mode, also_svg=False)
        shutil.copyfile(RES / d["run"] / "card.png", cards / f"{d['run']}.png")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("only", nargs="*",
                    help="run-folder names to restrict the overview to (e.g. "
                         "'probe_plots.py 1k_1X16 crt_cw_25'); the small overview "
                         "is written to probe_summary_<name>.png next to the full one")
    ap.add_argument("--name", default="small",
                    help="suffix for a restricted overview's filename (default: small; "
                         "run_eval.sh uses 'latest' for the just-processed batch)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    runs, spec = load_probe_runs()
    if args.only:
        known = {r["run"] for r in runs}
        for miss in set(args.only) - known:
            print(f"!! no results for '{miss}' (have: {sorted(known)})")
        runs = [r for r in runs if r["run"] in args.only]
        if not runs:
            sys.exit("nothing to plot")
        spec = [c for c in spec if any(_has(r["acc"], c[0], c[1]) for r in runs)]
    out_png = OUT / (f"probe_summary_{args.name}.png" if args.only else "probe_summary.png")
    print(f"{len(runs)} checkpoints, tasks: {[c[0] for c in spec]}")
    rows = probe_summary_figure(runs, spec, "light", out_png)
    if not args.only:   # cards always cover the full set; a subset run just re-plots
        checkpoint_cards(rows, spec, "light")
    print("\n== ranking: retention of input accuracy through the codec ==")
    for i, d in enumerate(rows, 1):
        per_task = "   ".join(f"{t} {d[(t, 'ret')]:5.1f}" for t, *_ in spec
                              if np.isfinite(d[(t, "ret")]))
        print(f"{i:2d}. {d['run']:<12} overall {d['overall']:5.1f}   {per_task}")
    print("\noverview figure:", out_png,
          "- per-checkpoint cards in results/<run>/card.png and results/_cards/")


if __name__ == "__main__":
    main()
