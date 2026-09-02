"""Small, consistent plot set used directly by ``scripts/train_eeg.py``."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch


STUDENT_CORRUPTIONS = [
    ("Slow drift", "slow_drift_probability"),
    ("Baseline shift", "baseline_shift_probability"),
    ("Line noise", "line_noise_probability"),
    ("Gaussian noise", "gaussian_noise_probability"),
    ("Signal mixing", "signal_mix_probability"),
]
PHASE_TRANSFORM = ("Phase shift (teacher + student)", "phase_shift_probability")
PROBABILITY_KEYS = [key for _, key in STUDENT_CORRUPTIONS] + [PHASE_TRANSFORM[1]]


def _examples(dataset, count: int) -> Tuple[list, torch.Tensor]:
    examples = [dataset[index] for index in range(min(count, len(dataset)))]
    return examples, torch.stack([example["eeg"] for example in examples])


def _short_recording(path: str) -> str:
    return Path(path).name.removesuffix("_cleaned_noref_raw.fif")


def _isolated_augmentation(config: dict, selected: str) -> dict:
    isolated = copy.deepcopy(config)
    isolated["enabled"] = True
    for key in PROBABILITY_KEYS:
        isolated[key] = 0.0
    isolated[selected] = 1.0
    return isolated


def _seeded_augment(
    clean: torch.Tensor,
    sample_rate: int,
    augment_config: dict,
    augment_fn: Callable,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Augment on CPU so a fixed seed gives the same diagnostic on every GPU.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return augment_fn(clean, sample_rate, augment_config)


def save_example_manifest(dataset, count: int, sample_rate: int, output: Path) -> None:
    examples, _ = _examples(dataset, count)
    payload = [
        {
            "index": index,
            "recording": example["recording"],
            "channel": example["channel"],
            "start_sample": int(example["start_sample"]),
            "start_seconds": float(example["start_sample"] / sample_rate),
        }
        for index, example in enumerate(examples)
    ]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_augmentations(
    dataset,
    augment_fn: Callable,
    augment_config: dict,
    sample_rate: int,
    output: Path,
    columns: int = 5,
    seconds: float = 5.0,
    seed: int = 7,
) -> Path:
    """Plot isolated transforms plus the combined distribution used in training."""
    examples, clean = _examples(dataset, columns)
    rows = STUDENT_CORRUPTIONS + [PHASE_TRANSFORM, ("Combined training draw", "combined")]
    transformed: Dict[str, torch.Tensor] = {}
    for row_index, (label, key) in enumerate(rows):
        config = augment_config if key == "combined" else _isolated_augmentation(augment_config, key)
        corrupted, target = _seeded_augment(
            clean, sample_rate, config, augment_fn, seed + row_index
        )
        transformed[label] = target if key == PHASE_TRANSFORM[1] else corrupted

    shown = min(clean.shape[-1], int(round(seconds * sample_rate)))
    time_axis = np.arange(shown) / sample_rate
    clean_np = clean[:, 0, :shown].numpy()
    figure, axes = plt.subplots(
        len(rows), columns, figsize=(3.8 * columns, 2.05 * len(rows)),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    for row_index, (label, _) in enumerate(rows):
        augmented_np = transformed[label][:, 0, :shown].numpy()
        for column_index, example in enumerate(examples):
            axis = axes[row_index, column_index]
            axis.plot(time_axis, augmented_np[column_index], color="#d95f02", lw=0.7, alpha=0.7)
            axis.plot(time_axis, clean_np[column_index], color="#1f4e79", lw=0.95, zorder=3)
            axis.axhline(0, color="0.8", lw=0.4, zorder=0)
            if row_index == 0:
                axis.set_title(
                    f"{_short_recording(example['recording'])} · {example['channel']}\n"
                    f"{example['start_sample'] / sample_rate:.1f}s",
                    fontsize=8,
                )
            if column_index == 0:
                axis.set_ylabel(label, fontsize=8)
            if row_index == len(rows) - 1:
                axis.set_xlabel("Time (s)")
            axis.tick_params(labelsize=7)
    axes[0, 0].plot([], [], color="#1f4e79", label="clean")
    axes[0, 0].plot([], [], color="#d95f02", label="augmented")
    axes[0, 0].legend(fontsize=7, frameon=False)
    figure.suptitle("Configured EEG augmentations", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


@torch.no_grad()
def token_agreement(
    model,
    dataset,
    augment_fn: Callable,
    augment_config: dict,
    sample_rate: int,
    device: torch.device,
    count: int = 16,
    seed: int = 7,
) -> Dict[str, float]:
    """Compare hard codes for a fixed clean/corrupted validation batch."""
    _, clean = _examples(dataset, count)
    corrupted, target = _seeded_augment(clean, sample_rate, augment_config, augment_fn, seed)
    was_training = model.training
    model.eval()
    clean_codes = model(target.to(device), sample_rate=sample_rate)["codes"]
    noisy_codes = model(corrupted.to(device), sample_rate=sample_rate)["codes"]
    if was_training:
        model.train()
    values = (clean_codes == noisy_codes).float().mean(dim=(0, 2)).cpu().tolist()
    metrics = {f"codebook_{index + 1}": float(value) for index, value in enumerate(values)}
    metrics["mean"] = float(np.mean(values))
    return metrics


def _grid_shape(count: int) -> Tuple[int, int]:
    columns = 4
    return math.ceil(count / columns), columns


@torch.no_grad()
def plot_reconstructions(
    model,
    dataset,
    sample_rate: int,
    device: torch.device,
    epoch: int,
    output_dir: Path,
    count: int = 16,
    seconds: float = 5.0,
    psd_fmin: float = 1.0,
    psd_fmax: float = 45.0,
    psd_window_seconds: float = 1.0,
) -> Tuple[Path, Path]:
    """Create matching 4-column waveform and PSD reconstruction galleries."""
    examples, clean = _examples(dataset, count)
    was_training = model.training
    model.eval()
    reconstruction = model(clean.to(device), sample_rate=sample_rate)["audio"].cpu()
    if was_training:
        model.train()

    count = len(examples)
    rows, columns = _grid_shape(count)
    shown = min(clean.shape[-1], int(round(seconds * sample_rate)))
    time_axis = np.arange(shown) / sample_rate
    clean_np = clean[:, 0].numpy()
    reconstruction_np = reconstruction[:, 0].numpy()

    waveform_path = output_dir / f"reconstruction_epoch_{epoch:04d}.png"
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 2.35 * rows),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    for index, axis in enumerate(axes.ravel()):
        if index >= count:
            axis.axis("off")
            continue
        error = float(
            np.mean(np.abs(clean_np[index, :shown] - reconstruction_np[index, :shown]))
        )
        axis.plot(time_axis, reconstruction_np[index, :shown], color="#d95f02", lw=0.75, alpha=0.75)
        axis.plot(time_axis, clean_np[index, :shown], color="#1f4e79", lw=0.95, zorder=3)
        axis.set_title(
            f"{_short_recording(examples[index]['recording'])} · {examples[index]['channel']} · "
            f"L1 {error:.4f}", fontsize=8,
        )
        axis.axhline(0, color="0.8", lw=0.4, zorder=0)
        if index % columns == 0:
            axis.set_ylabel("Normalized amplitude")
        if index // columns == rows - 1:
            axis.set_xlabel("Time (s)")
        axis.tick_params(labelsize=7)
    axes[0, 0].plot([], [], color="#1f4e79", label="target")
    axes[0, 0].plot([], [], color="#d95f02", label="reconstruction")
    axes[0, 0].legend(fontsize=7, frameon=False)
    figure.suptitle(f"Held-out EEG waveform reconstruction · epoch {epoch}", fontsize=14)
    figure.savefig(waveform_path, dpi=180)
    plt.close(figure)

    psd_path = output_dir / f"reconstruction_psd_epoch_{epoch:04d}.png"
    nperseg = min(shown, max(8, int(round(psd_window_seconds * sample_rate))))
    frequency, target_psd = welch(
        clean_np[:, :shown], fs=sample_rate, nperseg=nperseg,
        noverlap=nperseg // 2, axis=-1,
    )
    _, reconstruction_psd = welch(
        reconstruction_np[:, :shown], fs=sample_rate, nperseg=nperseg,
        noverlap=nperseg // 2, axis=-1,
    )
    target_db = 10 * np.log10(target_psd + 1e-12)
    reconstruction_db = 10 * np.log10(reconstruction_psd + 1e-12)
    selected = (frequency >= psd_fmin) & (frequency <= psd_fmax)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 2.35 * rows),
        sharex=True, squeeze=False, constrained_layout=True,
    )
    for index, axis in enumerate(axes.ravel()):
        if index >= count:
            axis.axis("off")
            continue
        error = float(np.mean(np.abs(target_db[index, selected] - reconstruction_db[index, selected])))
        axis.plot(frequency[selected], reconstruction_db[index, selected], color="#d95f02", lw=0.85)
        axis.plot(frequency[selected], target_db[index, selected], color="#1f4e79", lw=1.05, zorder=3)
        axis.set_title(
            f"{_short_recording(examples[index]['recording'])} · {examples[index]['channel']} · "
            f"PSD L1 {error:.2f} dB", fontsize=8,
        )
        if index % columns == 0:
            axis.set_ylabel("PSD (dB)")
        if index // columns == rows - 1:
            axis.set_xlabel("Frequency (Hz)")
        axis.grid(alpha=0.2, lw=0.4)
        axis.tick_params(labelsize=7)
    axes[0, 0].plot([], [], color="#1f4e79", label="target")
    axes[0, 0].plot([], [], color="#d95f02", label="reconstruction")
    axes[0, 0].legend(fontsize=7, frameon=False)
    figure.suptitle(f"Held-out EEG PSD reconstruction · epoch {epoch}", fontsize=14)
    figure.savefig(psd_path, dpi=180)
    plt.close(figure)
    return waveform_path, psd_path


def plot_codebook_usage(
    counts: np.ndarray,
    epoch: int,
    agreement: Dict[str, float],
    output_dir: Path,
) -> Path:
    """Plot current and longitudinal validation code usage for every RVQ level."""
    fractions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    history_path = output_dir / "codebook_usage_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history = [row for row in history if int(row["epoch"]) != int(epoch)]
    history.append({"epoch": int(epoch), "fractions": fractions.tolist()})
    history.sort(key=lambda row: row["epoch"])
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    n_codebooks, codebook_size = counts.shape
    figure, axes = plt.subplots(
        n_codebooks, 2, figsize=(15, 3.0 * n_codebooks),
        squeeze=False, constrained_layout=True,
    )
    epochs = [int(row["epoch"]) for row in history]
    for level in range(n_codebooks):
        snapshot = axes[level, 0].imshow(
            fractions[level][None, :], aspect="auto", interpolation="nearest", cmap="viridis"
        )
        axes[level, 0].set_yticks([])
        axes[level, 0].set_xlabel("Code ID")
        axes[level, 0].set_title(
            f"Codebook {level + 1} · epoch {epoch} · "
            f"agreement {agreement[f'codebook_{level + 1}']:.1%}"
        )
        figure.colorbar(snapshot, ax=axes[level, 0], label="Assignment fraction")

        longitudinal = np.asarray([row["fractions"][level] for row in history])
        image = axes[level, 1].imshow(
            longitudinal, aspect="auto", interpolation="nearest", cmap="viridis"
        )
        axes[level, 1].set_yticks(np.arange(len(epochs)), labels=epochs)
        axes[level, 1].set_xlabel("Code ID")
        axes[level, 1].set_ylabel("Epoch")
        axes[level, 1].set_title(f"Codebook {level + 1} usage over training")
        figure.colorbar(image, ax=axes[level, 1], label="Assignment fraction")
    figure.suptitle("Validation codebook utilization", fontsize=14)
    path = output_dir / f"codebook_usage_epoch_{epoch:04d}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
