"""Generate one compact comparison figure from EEG-DAC sweep summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]


def load_rows(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "complete" or not row.get("final_wave_l1"):
                    continue
                numeric = [
                    "compression_float32", "final_wave_l1", "final_stft_logpower",
                    "final_token_agreement", "final_entropy_efficiency",
                ]
                for key in numeric:
                    row[key] = float(row[key])
                row["root"] = root
                rows.append(row)
    return rows


def by_name(rows: list[dict], name: str) -> dict:
    return next(row for row in rows if row["run"] == name)


def nonfinite_fraction(row: dict) -> float:
    history = json.loads((row["root"] / row["run"] / "history.json").read_text())
    gradients = [epoch["train"]["other/grad_norm"] for epoch in history]
    return sum(not math.isfinite(value) for value in gradients) / len(gradients)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", nargs="+",
        default=["runs/overnight_20260902", "runs/overnight_20260902_followup"],
    )
    parser.add_argument(
        "--output", default="runs/overnight_20260902/figures/sweep_overview.png"
    )
    args = parser.parse_args()
    roots = [(REPO / root).resolve() for root in args.roots]
    rows = load_rows(roots)
    output = (REPO / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # All-run rate-distortion view, with unhealthy codebooks visually separated.
    ax = axes[0, 0]
    for row in rows:
        healthy = row["final_entropy_efficiency"] >= 0.8
        ax.scatter(
            row["compression_float32"], row["final_wave_l1"],
            c=row["final_token_agreement"], cmap="viridis", vmin=0.2, vmax=0.85,
            marker="o" if healthy else "x", s=55, alpha=0.8,
        )
    ax.axhline(0.1256, color="0.35", linestyle="--", label="zero-output baseline")
    for name, label in [
        ("final_2x64_ampoff_clip100", "2×64 FP32"),
        ("final_1x32_ampoff_clip100", "1×32 FP32"),
        ("final_1x16_ampoff_clip100", "1×16 FP32"),
        ("capacity_1x8_clip100", "1×8"),
    ]:
        if any(row["run"] == name for row in rows):
            row = by_name(rows, name)
            ax.annotate(label, (row["compression_float32"], row["final_wave_l1"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Nominal float32 compression (×, per electrode)")
    ax.set_ylabel("Final waveform L1 (lower is better)")
    ax.set_title("Rate–distortion frontier (color = token agreement)")
    ax.legend(frameon=False)

    # Fidelity and invariance across every healthy run.
    ax = axes[0, 1]
    healthy_rows = [row for row in rows if row["final_entropy_efficiency"] >= 0.8]
    points = ax.scatter(
        [row["final_wave_l1"] for row in healthy_rows],
        [row["final_token_agreement"] for row in healthy_rows],
        c=[row["final_entropy_efficiency"] for row in healthy_rows],
        cmap="plasma", vmin=0.8, vmax=1.0, s=60, alpha=0.8,
    )
    fig.colorbar(points, ax=ax, label="Codebook entropy efficiency")
    ax.set_xlabel("Final waveform L1 (lower is better)")
    ax.set_ylabel("Clean/noisy hard-token agreement")
    ax.set_title("Fidelity–robustness trade-off (healthy runs)")

    # Teacher-student objective ablation.
    ax = axes[1, 0]
    teacher_names = [
        "plain_noaug_clip100", "denoise_cons0_clip100", "teacher_cons03_clip100",
        "clip100_seed0", "teacher_cons3_clip100", "amplitude_scale_clip100",
    ]
    labels = ["plain", "noise only", "cons. 0.3", "cons. 1", "cons. 3", "+ scale"]
    teacher = [by_name(rows, name) for name in teacher_names]
    x = range(len(teacher))
    ax.bar(x, [row["final_token_agreement"] for row in teacher], color="#4c78a8", alpha=0.8, label="agreement")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Token agreement")
    ax2 = ax.twinx()
    ax2.plot(x, [row["final_wave_l1"] for row in teacher], color="#e45756", marker="o", label="Wave L1")
    ax2.set_ylabel("Waveform L1")
    ax.set_title("Why the teacher–student loss helps")

    # Precision stability comparison.
    ax = axes[1, 1]
    precision_names = ["clip100_seed0", "amp_off_clip100_seed0", "amp_bfloat16_clip100_seed0"]
    precision_labels = ["float16", "float32", "bfloat16"]
    precision = [by_name(rows, name) for name in precision_names]
    x = range(len(precision))
    ax.bar(x, [row["final_wave_l1"] for row in precision], color="#72b7b2", label="final Wave L1")
    ax.axhline(0.1256, color="0.35", linestyle="--")
    ax.set_xticks(list(x), precision_labels)
    ax.set_ylabel("Final waveform L1")
    ax2 = ax.twinx()
    ax2.plot(x, [nonfinite_fraction(row) for row in precision], color="#f58518", marker="o", linewidth=2)
    ax2.set_ylabel("Fraction of epochs with non-finite grad norm")
    ax2.set_ylim(0, 0.4)
    ax.set_title("Numerical precision stability")

    fig.suptitle("EEG-DAC systematic sweep", fontsize=17)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
