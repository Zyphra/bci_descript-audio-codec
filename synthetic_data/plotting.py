"""Plot realistic source-simulated EEG samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from synthetic_data.generative_model import (  # noqa: E402
    SyntheticSourceDataset,
    configure_runtime_cache,
)

configure_runtime_cache()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_SAMPLES_DIR = Path("synthetic_data/samples")
DEFAULT_WAVEFORM_OUTPUT = (
    DEFAULT_SAMPLES_DIR / "waveform_space" / "source_samples_waveform.png"
)
DEFAULT_WAVEFORM_GRID_OUTPUT = (
    DEFAULT_SAMPLES_DIR / "waveform_space" / "source_samples_waveform_grid.png"
)
DEFAULT_PSD_OUTPUT = (
    DEFAULT_SAMPLES_DIR / "power_spectral_density" / "source_samples_psd.png"
)
DEFAULT_PSD_GRID_OUTPUT = (
    DEFAULT_SAMPLES_DIR / "power_spectral_density" / "source_samples_psd_grid.png"
)


def build_sample_parameter_grid(n_repeats: int = 3) -> List[Dict[str, Any]]:
    settings = [
        {
            "source_freq_hz": 6.0,
            "amplitude_nam": 2.0,
            "phase_rad": 0.0,
            "noise_std_multiplier": 5.0,
        },
        {
            "source_freq_hz": 10.0,
            "amplitude_nam": 2.0,
            "phase_rad": 0.5 * np.pi,
            "noise_std_multiplier": 5.0,
        },
        {
            "source_freq_hz": 18.0,
            "amplitude_nam": 1.0,
            "phase_rad": np.pi,
            "noise_std_multiplier": 5.0,
        },
        {
            "source_freq_hz": 10.0,
            "amplitude_nam": 6.0,
            "phase_rad": 0.0,
            "noise_std_multiplier": 5.0,
        },
        {
            "source_freq_hz": 10.0,
            "amplitude_nam": 2.0,
            "phase_rad": 0.0,
            "noise_std_multiplier": 15.0,
        },
    ]

    rows: List[Dict[str, Any]] = []
    for setting_id, setting in enumerate(settings):
        for sample_id in range(n_repeats):
            row = dict(setting)
            row["setting_id"] = setting_id
            row["sample_id"] = sample_id
            rows.append(row)
    return rows


def build_parameter_grid() -> List[Dict[str, Any]]:
    return build_sample_parameter_grid()


def _parse_optional_float_list(value: str):
    if value.lower() in {"none", "null", ""}:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _format_phase(phase_rad: float) -> str:
    value = phase_rad / np.pi
    return f"{value:g} pi"


def _format_setting(row: Dict[str, Any]) -> str:
    return (
        f"source_freq={row['source_freq_hz']:g} Hz, "
        f"amplitude={row['amplitude_nam']:g} nAm\n"
        f"phase={_format_phase(float(row['phase_rad']))}, "
        f"noise_std x{row['noise_std_multiplier']:g}"
    )


def _choose_plot_channels(
    dataset: SyntheticSourceDataset, n_plot_channels: int
) -> np.ndarray:
    clean_rms_by_channel = np.sqrt(np.mean(dataset.X_clean ** 2, axis=(0, 2)))
    n_channels = min(int(n_plot_channels), len(dataset.ch_names))
    return np.argsort(clean_rms_by_channel)[::-1][:n_channels]


def _panel_title(dataset: SyntheticSourceDataset, montage: str) -> str:
    title = f"MNE source EEG ({montage}, {dataset.noise_mode} noise)"
    if dataset.noise_mode == "spatial_distance":
        title += (
            f", Gaussian length={dataset.spatial_noise_length_scale_cm:g} cm"
        )
    if dataset.noise_temporal_mode == "powerlaw":
        title += f", temporal 1/f^{dataset.noise_powerlaw_beta:g}"
    elif dataset.noise_temporal_mode == "white":
        title += ", temporal white"
    else:
        title += ", temporal MNE IIR"
    return title


def generate_sample_dataset(
    montage: str = "biosemi32",
    duration_sec: float = 2.0,
    sfreq: float = 256.0,
    source_label: str = "caudalmiddlefrontal-lh",
    source_extent_mm: float = 10.0,
    subjects_dir: Optional[str] = None,
    fetch_fsaverage: bool = True,
    random_state: int = 0,
    n_repeats: int = 3,
    n_plot_channels: int = 8,
    noise_mode: str = "mne_diagonal",
    noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
    spatial_noise_length_scale_cm: float = 6.0,
    noise_temporal_mode: str = "mne_iir",
    noise_powerlaw_beta: float = 1.0,
) -> SyntheticSourceDataset:
    trial_parameters = build_sample_parameter_grid(n_repeats=n_repeats)
    return SyntheticSourceDataset(
        n_trials=len(trial_parameters),
        montage=montage,
        duration_sec=duration_sec,
        sfreq=sfreq,
        source_label=source_label,
        source_extent_mm=source_extent_mm,
        trial_parameters=trial_parameters,
        random_state=random_state,
        subjects_dir=subjects_dir,
        fetch_fsaverage=fetch_fsaverage,
        noise_mode=noise_mode,
        noise_iir_filter=noise_iir_filter,
        spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
        noise_temporal_mode=noise_temporal_mode,
        noise_powerlaw_beta=noise_powerlaw_beta,
    ).generate(save=False)


def _write_waveform_panel(
    dataset: SyntheticSourceDataset,
    output: str,
    n_repeats: int,
    n_plot_channels: int,
    save_dataset: bool = False,
    show: bool = False,
) -> None:
    plot_channel_indices = _choose_plot_channels(dataset, n_plot_channels)
    plot_channel_names = [dataset.ch_names[idx] for idx in plot_channel_indices]
    setting_ids = sorted({int(row["setting_id"]) for row in dataset.metadata})
    visible = dataset.X[:, plot_channel_indices, :] * 1e6
    trace_scale = max(float(np.percentile(np.abs(visible), 95)), 1.0)
    offset = trace_scale * 4.0
    y_offsets = offset * np.arange(len(plot_channel_indices))[::-1]

    fig, axes = plt.subplots(
        len(setting_ids),
        n_repeats,
        figsize=(4.2 * n_repeats, 2.35 * len(setting_ids)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    color = "#0f6b78"

    for row_idx, setting_id in enumerate(setting_ids):
        row_trials = [
            (trial_idx, row)
            for trial_idx, row in enumerate(dataset.metadata)
            if int(row["setting_id"]) == setting_id
        ]
        row_trials = sorted(row_trials, key=lambda item: int(item[1]["sample_id"]))
        for col_idx, (trial_idx, row) in enumerate(row_trials):
            ax = axes[row_idx, col_idx]
            sample_uv = dataset.X[trial_idx, plot_channel_indices, :] * 1e6
            for channel_row, trace in enumerate(sample_uv):
                ax.plot(
                    dataset.times,
                    trace + y_offsets[channel_row],
                    color=color,
                    linewidth=0.8,
                )
            if col_idx == 0:
                ax.set_yticks(y_offsets)
                ax.set_yticklabels(plot_channel_names, fontsize=8)
                ax.set_ylabel(
                    _format_setting(row),
                    fontsize=8,
                    rotation=0,
                    labelpad=64,
                    ha="right",
                    va="center",
                )
            else:
                ax.set_yticks(y_offsets)
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(f"sample {col_idx + 1}", fontsize=10)
            if row_idx == len(setting_ids) - 1:
                ax.set_xlabel("Time (s)")
            ax.set_ylim(y_offsets[-1] - 2.0 * offset, y_offsets[0] + 2.0 * offset)
            ax.grid(axis="x", color="0.9", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(_panel_title(dataset, dataset.montage), fontsize=13)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if save_dataset:
        dataset.save(output_dir=str(output_path.parent), prefix=output_path.stem)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sample_panel(
    output: str = str(DEFAULT_WAVEFORM_OUTPUT),
    montage: str = "biosemi32",
    duration_sec: float = 2.0,
    sfreq: float = 256.0,
    source_label: str = "caudalmiddlefrontal-lh",
    source_extent_mm: float = 10.0,
    subjects_dir: Optional[str] = None,
    fetch_fsaverage: bool = True,
    random_state: int = 0,
    n_repeats: int = 3,
    n_plot_channels: int = 8,
    noise_mode: str = "mne_diagonal",
    noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
    spatial_noise_length_scale_cm: float = 6.0,
    noise_temporal_mode: str = "mne_iir",
    noise_powerlaw_beta: float = 1.0,
    save_dataset: bool = False,
    show: bool = False,
) -> SyntheticSourceDataset:
    dataset = generate_sample_dataset(
        montage=montage,
        duration_sec=duration_sec,
        sfreq=sfreq,
        source_label=source_label,
        source_extent_mm=source_extent_mm,
        subjects_dir=subjects_dir,
        fetch_fsaverage=fetch_fsaverage,
        random_state=random_state,
        n_repeats=n_repeats,
        noise_mode=noise_mode,
        noise_iir_filter=noise_iir_filter,
        spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
        noise_temporal_mode=noise_temporal_mode,
        noise_powerlaw_beta=noise_powerlaw_beta,
    )
    _write_waveform_panel(
        dataset=dataset,
        output=output,
        n_repeats=n_repeats,
        n_plot_channels=n_plot_channels,
        save_dataset=save_dataset,
        show=show,
    )
    return dataset


def _compute_psd_uv2_per_hz(
    data_uv: np.ndarray, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    centered = data_uv - data_uv.mean(axis=-1, keepdims=True)
    n_times = centered.shape[-1]
    window = np.hanning(n_times).astype(np.float64)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spectrum = np.fft.rfft(centered * window, axis=-1)
    psd = (np.abs(spectrum) ** 2) / (sfreq * np.sum(window**2))
    if n_times % 2 == 0:
        psd[..., 1:-1] *= 2.0
    else:
        psd[..., 1:] *= 2.0
    return freqs, psd


def _one_over_f_reference(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_mask: np.ndarray,
    source_freq_hz: float,
) -> np.ndarray:
    candidates = freq_mask & (freqs >= 20.0)
    candidates &= np.abs(freqs - source_freq_hz) > 2.0
    if not np.any(candidates):
        candidates = freq_mask & (np.abs(freqs - source_freq_hz) > 2.0)
    if not np.any(candidates):
        candidates = freq_mask

    candidate_indices = np.flatnonzero(candidates)
    ref_idx = candidate_indices[np.argmin(np.abs(freqs[candidate_indices] - 30.0))]
    ref_anchor = float(psd[ref_idx])
    return ref_anchor * (freqs[freq_mask] / freqs[ref_idx]) ** -1.0


def _write_psd_panel(
    dataset: SyntheticSourceDataset,
    output: str,
    n_plot_channels: int,
    psd_fmin: float = 1.0,
    psd_fmax: float = 80.0,
    show: bool = False,
) -> None:
    plot_channel_indices = _choose_plot_channels(dataset, n_plot_channels)
    setting_ids = sorted({int(row["setting_id"]) for row in dataset.metadata})
    freqs, psd = _compute_psd_uv2_per_hz(
        dataset.X[:, plot_channel_indices, :] * 1e6,
        dataset.sfreq,
    )
    nyquist = dataset.sfreq / 2.0
    fmax = min(float(psd_fmax), nyquist)
    fmin = max(float(psd_fmin), float(freqs[1]) if len(freqs) > 1 else 0.0)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if int(freq_mask.sum()) < 3:
        raise ValueError("PSD frequency range must contain at least 3 bins")

    fig, axes = plt.subplots(
        len(setting_ids),
        1,
        figsize=(8.5, 2.1 * len(setting_ids)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    sample_color = "#78aeb7"
    mean_color = "#0f6b78"
    reference_color = "0.35"

    for row_idx, setting_id in enumerate(setting_ids):
        ax = axes[row_idx, 0]
        row_trials = [
            (trial_idx, row)
            for trial_idx, row in enumerate(dataset.metadata)
            if int(row["setting_id"]) == setting_id
        ]
        row_trials = sorted(row_trials, key=lambda item: int(item[1]["sample_id"]))
        trial_indices = [trial_idx for trial_idx, _ in row_trials]
        row = row_trials[0][1]
        sample_psd = psd[trial_indices].mean(axis=1)
        mean_psd = sample_psd.mean(axis=0)

        for sample in sample_psd:
            ax.plot(
                freqs[freq_mask],
                sample[freq_mask],
                color=sample_color,
                alpha=0.55,
                linewidth=0.8,
            )
        ax.plot(
            freqs[freq_mask],
            mean_psd[freq_mask],
            color=mean_color,
            linewidth=1.5,
        )

        one_over_f = _one_over_f_reference(
            freqs,
            mean_psd,
            freq_mask,
            source_freq_hz=float(row["source_freq_hz"]),
        )
        ax.plot(
            freqs[freq_mask],
            one_over_f,
            color=reference_color,
            linestyle="--",
            linewidth=0.9,
        )

        ax.axvline(
            float(row["source_freq_hz"]),
            color="0.25",
            alpha=0.25,
            linewidth=0.8,
        )
        ax.set_ylabel(
            _format_setting(row),
            fontsize=8,
            rotation=0,
            labelpad=86,
            ha="right",
            va="center",
        )
        ax.set_xlim(fmin, fmax)
        ax.set_ylim(bottom=0.0)
        ax.grid(which="both", color="0.9", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].set_title(
        "linear PSDs (uV^2/Hz) with mean and 1/f reference",
        fontsize=10,
    )
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"{_panel_title(dataset, dataset.montage)} PSD", fontsize=13)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_psd_panel(
    output: str = str(DEFAULT_PSD_OUTPUT),
    montage: str = "biosemi32",
    duration_sec: float = 2.0,
    sfreq: float = 256.0,
    source_label: str = "caudalmiddlefrontal-lh",
    source_extent_mm: float = 10.0,
    subjects_dir: Optional[str] = None,
    fetch_fsaverage: bool = True,
    random_state: int = 0,
    n_repeats: int = 3,
    n_plot_channels: int = 8,
    noise_mode: str = "mne_diagonal",
    noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
    spatial_noise_length_scale_cm: float = 6.0,
    noise_temporal_mode: str = "mne_iir",
    noise_powerlaw_beta: float = 1.0,
    psd_fmin: float = 1.0,
    psd_fmax: float = 80.0,
    show: bool = False,
) -> SyntheticSourceDataset:
    dataset = generate_sample_dataset(
        montage=montage,
        duration_sec=duration_sec,
        sfreq=sfreq,
        source_label=source_label,
        source_extent_mm=source_extent_mm,
        subjects_dir=subjects_dir,
        fetch_fsaverage=fetch_fsaverage,
        random_state=random_state,
        n_repeats=n_repeats,
        noise_mode=noise_mode,
        noise_iir_filter=noise_iir_filter,
        spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
        noise_temporal_mode=noise_temporal_mode,
        noise_powerlaw_beta=noise_powerlaw_beta,
    )
    _write_psd_panel(
        dataset=dataset,
        output=output,
        n_plot_channels=n_plot_channels,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        show=show,
    )
    return dataset


def _build_amplitude_noise_grid_trials(
    amplitudes_nam: Sequence[float],
    noise_std_multipliers: Sequence[float],
    source_freq_hz: float,
    n_repeats: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for noise_id, noise_std_multiplier in enumerate(noise_std_multipliers):
        for amplitude_id, amplitude_nam in enumerate(amplitudes_nam):
            setting_id = noise_id * len(amplitudes_nam) + amplitude_id
            for sample_id in range(n_repeats):
                rows.append(
                    {
                        "setting_id": setting_id,
                        "noise_id": noise_id,
                        "amplitude_id": amplitude_id,
                        "sample_id": sample_id,
                        "source_freq_hz": float(source_freq_hz),
                        "amplitude_nam": float(amplitude_nam),
                        "phase_rad": 0.0,
                        "noise_std_multiplier": float(noise_std_multiplier),
                    }
                )
    return rows


def _format_powerlaw_label(beta: float) -> str:
    return f"PSD~f^-{float(beta):g}"


def _write_psd_parameter_grid(
    dataset: SyntheticSourceDataset,
    output: str,
    amplitudes_nam: Sequence[float],
    noise_std_multipliers: Sequence[float],
    noise_mode: str,
    noise_powerlaw_beta: float,
    random_state: int,
    n_plot_channels: int,
    psd_fmin: float,
    psd_fmax: float,
    show: bool = False,
) -> None:
    plot_channel_indices = _choose_plot_channels(dataset, n_plot_channels)
    freqs, _ = _compute_psd_uv2_per_hz(
        dataset.X_clean[:, plot_channel_indices, :] * 1e6,
        dataset.sfreq,
    )
    nyquist = dataset.sfreq / 2.0
    fmax = min(float(psd_fmax), nyquist)
    fmin = max(float(psd_fmin), float(freqs[1]) if len(freqs) > 1 else 0.0)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if int(freq_mask.sum()) < 3:
        raise ValueError("PSD frequency range must contain at least 3 bins")

    fig, axes = plt.subplots(
        len(noise_std_multipliers),
        len(amplitudes_nam),
        figsize=(3.35 * len(amplitudes_nam), 2.35 * len(noise_std_multipliers)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    sample_color = "#78aeb7"
    mean_color = "#0f6b78"
    reference_color = "0.35"

    dataset.noise_mode = dataset._normalize_noise_mode(noise_mode)
    dataset.noise_temporal_mode = "powerlaw"
    dataset.noise_powerlaw_beta = float(noise_powerlaw_beta)
    dataset.apply_noise(random_state=random_state + 101)
    freqs, psd = _compute_psd_uv2_per_hz(
        dataset.X[:, plot_channel_indices, :] * 1e6,
        dataset.sfreq,
    )

    for row_idx, noise_std_multiplier in enumerate(noise_std_multipliers):
        for col_idx, amplitude_nam in enumerate(amplitudes_nam):
            ax = axes[row_idx, col_idx]
            trial_indices = [
                idx
                for idx, row in enumerate(dataset.metadata)
                if int(row["noise_id"]) == row_idx
                and int(row["amplitude_id"]) == col_idx
            ]
            sample_psd = psd[trial_indices].mean(axis=1)
            mean_psd = sample_psd.mean(axis=0)
            for sample in sample_psd:
                ax.plot(
                    freqs[freq_mask],
                    sample[freq_mask],
                    color=sample_color,
                    alpha=0.45,
                    linewidth=0.7,
                )
            ax.plot(
                freqs[freq_mask],
                mean_psd[freq_mask],
                color=mean_color,
                linewidth=1.4,
            )

            source_freq_hz = float(dataset.metadata[trial_indices[0]]["source_freq_hz"])
            one_over_f = _one_over_f_reference(
                freqs,
                mean_psd,
                freq_mask,
                source_freq_hz=source_freq_hz,
            )
            ax.plot(
                freqs[freq_mask],
                one_over_f,
                color=reference_color,
                linestyle="--",
                linewidth=0.8,
            )
            ax.axvline(source_freq_hz, color="0.25", alpha=0.2, linewidth=0.7)
            ax.set_xlim(fmin, fmax)
            ax.set_ylim(bottom=0.0)
            ax.grid(color="0.9", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(f"amplitude={float(amplitude_nam):g} nAm", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(
                    f"noise_std x{float(noise_std_multiplier):g}\n(uV^2/Hz)",
                    fontsize=8,
                )
            if row_idx == len(noise_std_multipliers) - 1:
                ax.set_xlabel("Frequency (Hz)")

    title = (
        f"Linear PSD grid ({dataset.montage}, {dataset.noise_mode} covariance, "
        f"temporal {_format_powerlaw_label(noise_powerlaw_beta)}, "
        f"source_freq={dataset.metadata[0]['source_freq_hz']:g} Hz)"
    )
    fig.suptitle(title, fontsize=12)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_psd_parameter_grid(
    output: str = str(DEFAULT_PSD_GRID_OUTPUT),
    montage: str = "biosemi32",
    duration_sec: float = 2.0,
    sfreq: float = 256.0,
    source_label: str = "caudalmiddlefrontal-lh",
    source_extent_mm: float = 10.0,
    subjects_dir: Optional[str] = None,
    fetch_fsaverage: bool = True,
    random_state: int = 0,
    n_repeats: int = 4,
    n_plot_channels: int = 8,
    noise_mode: str = "spatial_distance",
    noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
    spatial_noise_length_scale_cm: float = 6.0,
    grid_source_freq_hz: float = 10.0,
    grid_amplitudes_nam: Sequence[float] = (
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.5,
    ),
    grid_noise_std_multipliers: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 15.0),
    noise_powerlaw_beta: float = 1.0,
    psd_fmin: float = 1.0,
    psd_fmax: float = 80.0,
    show: bool = False,
) -> SyntheticSourceDataset:
    normalized_noise_mode = SyntheticSourceDataset._normalize_noise_mode(noise_mode)
    if normalized_noise_mode not in {"mne_diagonal", "spatial_distance"}:
        raise ValueError(
            "PSD parameter grid requires noise_mode='mne_diagonal' "
            "or noise_mode='spatial_distance'"
        )

    trial_parameters = _build_amplitude_noise_grid_trials(
        amplitudes_nam=grid_amplitudes_nam,
        noise_std_multipliers=grid_noise_std_multipliers,
        source_freq_hz=grid_source_freq_hz,
        n_repeats=n_repeats,
    )
    dataset = SyntheticSourceDataset(
        n_trials=len(trial_parameters),
        montage=montage,
        duration_sec=duration_sec,
        sfreq=sfreq,
        source_label=source_label,
        source_extent_mm=source_extent_mm,
        trial_parameters=trial_parameters,
        random_state=random_state,
        subjects_dir=subjects_dir,
        fetch_fsaverage=fetch_fsaverage,
        noise_mode="none",
        noise_iir_filter=noise_iir_filter,
        spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
        noise_temporal_mode="powerlaw",
        noise_powerlaw_beta=noise_powerlaw_beta,
    ).generate(save=False)
    _write_psd_parameter_grid(
        dataset=dataset,
        output=output,
        amplitudes_nam=grid_amplitudes_nam,
        noise_std_multipliers=grid_noise_std_multipliers,
        noise_mode=normalized_noise_mode,
        noise_powerlaw_beta=noise_powerlaw_beta,
        random_state=random_state,
        n_plot_channels=n_plot_channels,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        show=show,
    )
    return dataset


def _write_waveform_parameter_grid(
    dataset: SyntheticSourceDataset,
    output: str,
    amplitudes_nam: Sequence[float],
    noise_std_multipliers: Sequence[float],
    noise_mode: str,
    noise_powerlaw_beta: float,
    random_state: int,
    n_plot_channels: int,
    show: bool = False,
) -> None:
    plot_channel_indices = _choose_plot_channels(dataset, n_plot_channels)
    plot_channel_indices = plot_channel_indices[: min(len(plot_channel_indices), 6)]

    fig, axes = plt.subplots(
        len(noise_std_multipliers),
        len(amplitudes_nam),
        figsize=(3.35 * len(amplitudes_nam), 2.35 * len(noise_std_multipliers)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    color = "#0f6b78"

    dataset.noise_mode = dataset._normalize_noise_mode(noise_mode)
    dataset.noise_temporal_mode = "powerlaw"
    dataset.noise_powerlaw_beta = float(noise_powerlaw_beta)
    dataset.apply_noise(random_state=random_state + 101)

    visible = dataset.X[:, plot_channel_indices, :] * 1e6
    trace_scale = max(float(np.percentile(np.abs(visible), 95)), 0.05)
    offset = trace_scale * 3.8
    y_offsets = offset * np.arange(len(plot_channel_indices))[::-1]

    for row_idx, noise_std_multiplier in enumerate(noise_std_multipliers):
        for col_idx, amplitude_nam in enumerate(amplitudes_nam):
            ax = axes[row_idx, col_idx]
            trial_indices = [
                idx
                for idx, row in enumerate(dataset.metadata)
                if int(row["noise_id"]) == row_idx
                and int(row["amplitude_id"]) == col_idx
                and int(row["sample_id"]) == 0
            ]
            if not trial_indices:
                raise ValueError("Waveform grid could not find sample_id=0")

            sample_uv = dataset.X[trial_indices[0], plot_channel_indices, :] * 1e6
            for channel_row, trace in enumerate(sample_uv):
                ax.plot(
                    dataset.times,
                    trace + y_offsets[channel_row],
                    color=color,
                    linewidth=0.8,
                )

            ax.set_yticks([])
            ax.set_ylim(y_offsets[-1] - 2.0 * offset, y_offsets[0] + 2.0 * offset)
            ax.grid(axis="x", color="0.9", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(f"amplitude={float(amplitude_nam):g} nAm", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(
                    f"noise_std x{float(noise_std_multiplier):g}",
                    fontsize=8,
                )
            if row_idx == len(noise_std_multipliers) - 1:
                ax.set_xlabel("Time (s)")

    title = (
        f"Waveform grid ({dataset.montage}, {dataset.noise_mode} covariance, "
        f"temporal {_format_powerlaw_label(noise_powerlaw_beta)}, "
        f"source_freq={dataset.metadata[0]['source_freq_hz']:g} Hz)"
    )
    fig.suptitle(title, fontsize=12)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_waveform_parameter_grid(
    output: str = str(DEFAULT_WAVEFORM_GRID_OUTPUT),
    montage: str = "biosemi32",
    duration_sec: float = 2.0,
    sfreq: float = 256.0,
    source_label: str = "caudalmiddlefrontal-lh",
    source_extent_mm: float = 10.0,
    subjects_dir: Optional[str] = None,
    fetch_fsaverage: bool = True,
    random_state: int = 0,
    n_repeats: int = 4,
    n_plot_channels: int = 8,
    noise_mode: str = "spatial_distance",
    noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
    spatial_noise_length_scale_cm: float = 6.0,
    grid_source_freq_hz: float = 10.0,
    grid_amplitudes_nam: Sequence[float] = (
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.5,
    ),
    grid_noise_std_multipliers: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 15.0),
    noise_powerlaw_beta: float = 1.0,
    show: bool = False,
) -> SyntheticSourceDataset:
    normalized_noise_mode = SyntheticSourceDataset._normalize_noise_mode(noise_mode)
    if normalized_noise_mode not in {"mne_diagonal", "spatial_distance"}:
        raise ValueError(
            "Waveform parameter grid requires noise_mode='mne_diagonal' "
            "or noise_mode='spatial_distance'"
        )

    trial_parameters = _build_amplitude_noise_grid_trials(
        amplitudes_nam=grid_amplitudes_nam,
        noise_std_multipliers=grid_noise_std_multipliers,
        source_freq_hz=grid_source_freq_hz,
        n_repeats=n_repeats,
    )
    dataset = SyntheticSourceDataset(
        n_trials=len(trial_parameters),
        montage=montage,
        duration_sec=duration_sec,
        sfreq=sfreq,
        source_label=source_label,
        source_extent_mm=source_extent_mm,
        trial_parameters=trial_parameters,
        random_state=random_state,
        subjects_dir=subjects_dir,
        fetch_fsaverage=fetch_fsaverage,
        noise_mode="none",
        noise_iir_filter=noise_iir_filter,
        spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
        noise_temporal_mode="powerlaw",
        noise_powerlaw_beta=noise_powerlaw_beta,
    ).generate(save=False)
    _write_waveform_parameter_grid(
        dataset=dataset,
        output=output,
        amplitudes_nam=grid_amplitudes_nam,
        noise_std_multipliers=grid_noise_std_multipliers,
        noise_mode=normalized_noise_mode,
        noise_powerlaw_beta=noise_powerlaw_beta,
        random_state=random_state,
        n_plot_channels=n_plot_channels,
        show=show,
    )
    return dataset


def plot_sample_outputs(
    waveform_output: str = str(DEFAULT_WAVEFORM_OUTPUT),
    waveform_grid_output: str = str(DEFAULT_WAVEFORM_GRID_OUTPUT),
    psd_output: str = str(DEFAULT_PSD_OUTPUT),
    psd_grid_output: str = str(DEFAULT_PSD_GRID_OUTPUT),
    plot_kind: str = "both",
    montage: str = "biosemi32",
    duration_sec: float = 2.0,
    sfreq: float = 256.0,
    source_label: str = "caudalmiddlefrontal-lh",
    source_extent_mm: float = 10.0,
    subjects_dir: Optional[str] = None,
    fetch_fsaverage: bool = True,
    random_state: int = 0,
    n_repeats: int = 3,
    n_plot_channels: int = 8,
    noise_mode: str = "mne_diagonal",
    noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
    spatial_noise_length_scale_cm: float = 6.0,
    noise_temporal_mode: str = "mne_iir",
    noise_powerlaw_beta: float = 1.0,
    psd_fmin: float = 1.0,
    psd_fmax: float = 80.0,
    grid_source_freq_hz: float = 10.0,
    grid_amplitudes_nam: Sequence[float] = (
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.5,
    ),
    grid_noise_std_multipliers: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 15.0),
    grid_noise_powerlaw_beta: float = 1.0,
    save_dataset: bool = False,
    show: bool = False,
) -> SyntheticSourceDataset:
    valid_plot_kinds = {
        "waveform",
        "waveform_grid",
        "psd",
        "both",
        "psd_grid",
        "all",
    }
    if plot_kind not in valid_plot_kinds:
        raise ValueError(
            "plot_kind must be one of: 'waveform', 'waveform_grid', "
            "'psd', 'both', 'psd_grid', 'all'"
        )

    if plot_kind == "waveform_grid":
        return plot_waveform_parameter_grid(
            output=waveform_grid_output,
            montage=montage,
            duration_sec=duration_sec,
            sfreq=sfreq,
            source_label=source_label,
            source_extent_mm=source_extent_mm,
            subjects_dir=subjects_dir,
            fetch_fsaverage=fetch_fsaverage,
            random_state=random_state,
            n_repeats=n_repeats,
            n_plot_channels=n_plot_channels,
            noise_mode=noise_mode,
            noise_iir_filter=noise_iir_filter,
            spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
            grid_source_freq_hz=grid_source_freq_hz,
            grid_amplitudes_nam=grid_amplitudes_nam,
            grid_noise_std_multipliers=grid_noise_std_multipliers,
            noise_powerlaw_beta=grid_noise_powerlaw_beta,
            show=show,
        )

    if plot_kind == "psd_grid":
        return plot_psd_parameter_grid(
            output=psd_grid_output,
            montage=montage,
            duration_sec=duration_sec,
            sfreq=sfreq,
            source_label=source_label,
            source_extent_mm=source_extent_mm,
            subjects_dir=subjects_dir,
            fetch_fsaverage=fetch_fsaverage,
            random_state=random_state,
            n_repeats=n_repeats,
            n_plot_channels=n_plot_channels,
            noise_mode=noise_mode,
            noise_iir_filter=noise_iir_filter,
            spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
            grid_source_freq_hz=grid_source_freq_hz,
            grid_amplitudes_nam=grid_amplitudes_nam,
            grid_noise_std_multipliers=grid_noise_std_multipliers,
            noise_powerlaw_beta=grid_noise_powerlaw_beta,
            psd_fmin=psd_fmin,
            psd_fmax=psd_fmax,
            show=show,
        )

    dataset = generate_sample_dataset(
        montage=montage,
        duration_sec=duration_sec,
        sfreq=sfreq,
        source_label=source_label,
        source_extent_mm=source_extent_mm,
        subjects_dir=subjects_dir,
        fetch_fsaverage=fetch_fsaverage,
        random_state=random_state,
        n_repeats=n_repeats,
        noise_mode=noise_mode,
        noise_iir_filter=noise_iir_filter,
        spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
        noise_temporal_mode=noise_temporal_mode,
        noise_powerlaw_beta=noise_powerlaw_beta,
    )
    if plot_kind in {"waveform", "both", "all"}:
        _write_waveform_panel(
            dataset=dataset,
            output=waveform_output,
            n_repeats=n_repeats,
            n_plot_channels=n_plot_channels,
            save_dataset=save_dataset,
            show=show,
        )
    if plot_kind in {"psd", "both", "all"}:
        _write_psd_panel(
            dataset=dataset,
            output=psd_output,
            n_plot_channels=n_plot_channels,
            psd_fmin=psd_fmin,
            psd_fmax=psd_fmax,
            show=show,
        )
    if plot_kind == "all":
        plot_waveform_parameter_grid(
            output=waveform_grid_output,
            montage=montage,
            duration_sec=duration_sec,
            sfreq=sfreq,
            source_label=source_label,
            source_extent_mm=source_extent_mm,
            subjects_dir=subjects_dir,
            fetch_fsaverage=fetch_fsaverage,
            random_state=random_state,
            n_repeats=n_repeats,
            n_plot_channels=n_plot_channels,
            noise_mode=noise_mode,
            noise_iir_filter=noise_iir_filter,
            spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
            grid_source_freq_hz=grid_source_freq_hz,
            grid_amplitudes_nam=grid_amplitudes_nam,
            grid_noise_std_multipliers=grid_noise_std_multipliers,
            noise_powerlaw_beta=grid_noise_powerlaw_beta,
            show=show,
        )
        plot_psd_parameter_grid(
            output=psd_grid_output,
            montage=montage,
            duration_sec=duration_sec,
            sfreq=sfreq,
            source_label=source_label,
            source_extent_mm=source_extent_mm,
            subjects_dir=subjects_dir,
            fetch_fsaverage=fetch_fsaverage,
            random_state=random_state,
            n_repeats=n_repeats,
            n_plot_channels=n_plot_channels,
            noise_mode=noise_mode,
            noise_iir_filter=noise_iir_filter,
            spatial_noise_length_scale_cm=spatial_noise_length_scale_cm,
            grid_source_freq_hz=grid_source_freq_hz,
            grid_amplitudes_nam=grid_amplitudes_nam,
            grid_noise_std_multipliers=grid_noise_std_multipliers,
            noise_powerlaw_beta=grid_noise_powerlaw_beta,
            psd_fmin=psd_fmin,
            psd_fmax=psd_fmax,
            show=show,
        )
    return dataset


def plot_parameter_panel(*args, **kwargs) -> SyntheticSourceDataset:
    return plot_sample_panel(*args, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_WAVEFORM_OUTPUT),
        help="Waveform-space output path.",
    )
    parser.add_argument(
        "--waveform-grid-output",
        default=str(DEFAULT_WAVEFORM_GRID_OUTPUT),
        help="Waveform parameter-grid output path.",
    )
    parser.add_argument(
        "--psd-output",
        default=str(DEFAULT_PSD_OUTPUT),
        help="Power spectral density output path.",
    )
    parser.add_argument(
        "--psd-grid-output",
        default=str(DEFAULT_PSD_GRID_OUTPUT),
        help="PSD parameter-grid output path.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=["waveform", "waveform_grid", "psd", "both", "psd_grid", "all"],
        default="both",
    )
    parser.add_argument("--montage", default="biosemi32")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--source-label", default="caudalmiddlefrontal-lh")
    parser.add_argument("--source-extent-mm", type=float, default=10.0)
    parser.add_argument("--subjects-dir", default=None)
    parser.add_argument("--no-fetch-fsaverage", action="store_true")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-plot-channels", type=int, default=8)
    parser.add_argument(
        "--noise-mode",
        choices=["mne_diagonal", "spatial_distance", "iid", "none", "mne_cov"],
        default="mne_diagonal",
    )
    parser.add_argument("--noise-iir-filter", default="0.2,-0.2,0.04")
    parser.add_argument("--spatial-noise-length-scale-cm", type=float, default=6.0)
    parser.add_argument(
        "--noise-temporal-mode",
        choices=["mne_iir", "white", "powerlaw", "iir"],
        default="mne_iir",
    )
    parser.add_argument("--noise-powerlaw-beta", type=float, default=1.0)
    parser.add_argument("--psd-fmin", type=float, default=1.0)
    parser.add_argument("--psd-fmax", type=float, default=80.0)
    parser.add_argument("--grid-source-freq-hz", type=float, default=10.0)
    parser.add_argument("--grid-amplitudes-nam", default="0.01,0.05,0.1,0.15,0.2,0.5")
    parser.add_argument(
        "--grid-noise-std-multipliers",
        "--grid-noise-std-multiplier",
        dest="grid_noise_std_multipliers",
        default="1,2,5,10,15",
    )
    parser.add_argument("--grid-powerlaw-beta", type=float, default=1.0)
    parser.add_argument(
        "--grid-powerlaw-betas",
        dest="legacy_grid_powerlaw_betas",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        grid_noise_powerlaw_beta = args.grid_powerlaw_beta
        if args.legacy_grid_powerlaw_betas is not None:
            legacy_betas = _parse_float_list(args.legacy_grid_powerlaw_betas)
            if legacy_betas:
                grid_noise_powerlaw_beta = legacy_betas[0]

        dataset = plot_sample_outputs(
            waveform_output=args.output,
            waveform_grid_output=args.waveform_grid_output,
            psd_output=args.psd_output,
            psd_grid_output=args.psd_grid_output,
            plot_kind=args.plot_kind,
            montage=args.montage,
            duration_sec=args.duration_sec,
            sfreq=args.sfreq,
            source_label=args.source_label,
            source_extent_mm=args.source_extent_mm,
            subjects_dir=args.subjects_dir,
            fetch_fsaverage=not args.no_fetch_fsaverage,
            random_state=args.random_state,
            n_repeats=args.n_repeats,
            n_plot_channels=args.n_plot_channels,
            noise_mode=args.noise_mode,
            noise_iir_filter=_parse_optional_float_list(args.noise_iir_filter),
            spatial_noise_length_scale_cm=args.spatial_noise_length_scale_cm,
            noise_temporal_mode=args.noise_temporal_mode,
            noise_powerlaw_beta=args.noise_powerlaw_beta,
            psd_fmin=args.psd_fmin,
            psd_fmax=args.psd_fmax,
            grid_source_freq_hz=args.grid_source_freq_hz,
            grid_amplitudes_nam=_parse_float_list(args.grid_amplitudes_nam),
            grid_noise_std_multipliers=_parse_float_list(
                args.grid_noise_std_multipliers
            ),
            grid_noise_powerlaw_beta=grid_noise_powerlaw_beta,
            save_dataset=args.save_dataset,
            show=args.show,
        )
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.plot_kind in {"waveform", "both", "all"}:
        print(f"Wrote {args.output}")
    if args.plot_kind in {"waveform_grid", "all"}:
        print(f"Wrote {args.waveform_grid_output}")
    if args.plot_kind in {"psd", "both", "all"}:
        print(f"Wrote {args.psd_output}")
    if args.plot_kind in {"psd_grid", "all"}:
        print(f"Wrote {args.psd_grid_output}")
    print(f"Generated panel data: X.shape={dataset.X.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
