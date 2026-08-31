"""Train and inspect a channel-independent DAC tokenizer for EEG.

VENV: bci/jonas/venv_dac (it's currenlty only a venv, need to move it to docker eventually)

* all DAC modules untouched 
* scripts/train_eeg.py is a wrapper that imports .fif files [channels, time]
* samples one channel at a time (so the model continues to receive its native [batch, 1, time] input). 
* currently running on the Alice toy dataset

TODO: 
* add support for multiple channels in the input (currently samples one channel at a time)
    * add support for different EEG channel montages


RUN
--------
Train the Alice toy model::

    python3 scripts/train_eeg.py train --config conf/eeg.yml

Create reconstruction, token-use, and noise-stability diagnostics::

    python3 scripts/train_eeg.py visualize --config conf/eeg.yml
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_cache_root = Path(tempfile.gettempdir()) / f"eeg-dac-{os.getuid()}"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import mne
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

import dac


@dataclass
class EEGRecording:
    path: str
    raw: object
    channel_names: List[str]
    median: np.ndarray
    scale: np.ndarray


def _merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = Path(path).resolve().parent / parent_path
    return _merge_config(load_config(str(parent_path)), config)


def resolve_files(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = glob.glob(str(Path(pattern).expanduser()))
        files.extend(matches)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No FIF files matched: {list(patterns)}")
    return files


def load_recordings(data_cfg: dict) -> List[EEGRecording]:
    target_sfreq = float(data_cfg["sample_rate"])
    train_fraction = float(data_cfg["train_fraction"])
    normalization_seconds = float(data_cfg.get("normalization_seconds", 30.0))
    calibration_chunks = int(data_cfg.get("normalization_chunks", 3))
    recordings = []
    for path in resolve_files(data_cfg["files"]):
        # Keep the FIF lazy. MNE will seek directly to the requested 10-second
        # window in __getitem__, rather than holding every subject in RAM.
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        raw.pick(picks="eeg")
        if not raw.ch_names:
            raise ValueError(f"No EEG channels found in {path}")
        if not math.isclose(raw.info["sfreq"], target_sfreq):
            raise ValueError(
                f"{path} is {raw.info['sfreq']} Hz, expected {target_sfreq} Hz. "
                "Resample it during preprocessing to preserve lazy FIF reads."
            )

        # Estimate robust per-channel statistics from small calibration chunks
        # spread across the training portion. This avoids loading the full file
        # and avoids leaking validation samples into normalization statistics.
        train_stop = int(raw.n_times * train_fraction)
        total_calibration = min(
            int(round(normalization_seconds * target_sfreq)), train_stop
        )
        chunk_samples = max(1, total_calibration // calibration_chunks)
        max_start = max(0, train_stop - chunk_samples)
        starts = np.linspace(0, max_start, calibration_chunks, dtype=int)
        calibration = np.concatenate(
            [
                raw.get_data(start=int(start), stop=int(start) + chunk_samples)
                for start in starts
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        median = np.median(calibration, axis=1, keepdims=True).astype(np.float32)
        mad = np.median(np.abs(calibration - median), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * mad, np.finfo(np.float32).eps).astype(np.float32)
        recordings.append(
            EEGRecording(path, raw, list(raw.ch_names), median, scale)
        )
        print(
            f"Indexed {Path(path).name}: {len(raw.ch_names)} EEG channels, "
            f"{raw.n_times / target_sfreq:.1f} s at {target_sfreq:g} Hz"
        )
    return recordings


class EEGWindowDataset(Dataset):
    """Random channel/window samples from a temporal train or validation split."""

    def __init__(
        self,
        recordings: Sequence[EEGRecording],
        sample_rate: int,
        window_seconds: float,
        n_examples: int,
        train_fraction: float,
        clip_mad: float,
        split: str,
        seed: int,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.recordings = recordings
        self.window_samples = int(round(sample_rate * window_seconds))
        self.n_examples = int(n_examples)
        self.train_fraction = float(train_fraction)
        self.clip_mad = float(clip_mad)
        self.split = split
        self.seed = int(seed)

        for rec in recordings:
            split_sample = int(rec.raw.n_times * self.train_fraction)
            available = split_sample if split == "train" else rec.raw.n_times - split_sample
            if available < self.window_samples:
                raise ValueError(
                    f"{split} portion of {rec.path} has {available} samples, "
                    f"shorter than the {self.window_samples}-sample window"
                )

    def __len__(self) -> int:
        return self.n_examples

    def _rng(self, index: int) -> np.random.Generator:
        # Validation must be exactly repeatable. Training is re-randomized by
        # DataLoader worker RNG, while index still prevents duplicated batches.
        if self.split == "val":
            return np.random.default_rng(self.seed + index)
        return np.random.default_rng(np.random.randint(0, 2**31 - 1) + index)

    def __getitem__(self, index: int) -> Dict[str, object]:
        rng = self._rng(index)
        rec_index = int(rng.integers(len(self.recordings)))
        rec = self.recordings[rec_index]
        channel = int(rng.integers(len(rec.channel_names)))
        split_sample = int(rec.raw.n_times * self.train_fraction)
        lo, hi = (0, split_sample) if self.split == "train" else (split_sample, rec.raw.n_times)
        start = int(rng.integers(lo, hi - self.window_samples + 1))
        stop = start + self.window_samples

        x = rec.raw.get_data(picks=[channel], start=start, stop=stop).astype(
            np.float32, copy=False
        )
        x = x - rec.median[channel]
        x = x / rec.scale[channel]
        x = np.clip(x, -self.clip_mad, self.clip_mad) / self.clip_mad
        return {
            "eeg": torch.from_numpy(x.copy()),
            "recording": rec.path,
            "channel": rec.channel_names[channel],
            "start_sample": start,
        }

    def fixed_item(self, recording_index: int = 0, channel_index: int = 0) -> Dict[str, object]:
        """Return the first validation window for repeatable visual comparisons."""
        rec = self.recordings[recording_index]
        if self.split == "train":
            start = 0
        else:
            start = int(rec.raw.n_times * self.train_fraction)
        channel = channel_index % len(rec.channel_names)
        stop = start + self.window_samples
        x = rec.raw.get_data(picks=[channel], start=start, stop=stop).astype(
            np.float32, copy=False
        )
        x = x - rec.median[channel]
        x = x / rec.scale[channel]
        x = np.clip(x, -self.clip_mad, self.clip_mad) / self.clip_mad
        return {
            "eeg": torch.from_numpy(x.copy()),
            "recording": rec.path,
            "channel": rec.channel_names[channel],
            "start_sample": start,
        }


def make_datasets(cfg: dict) -> Tuple[EEGWindowDataset, EEGWindowDataset]:
    data_cfg = cfg["data"]
    recordings = load_recordings(data_cfg)
    common = dict(
        recordings=recordings,
        sample_rate=data_cfg["sample_rate"],
        window_seconds=data_cfg["window_seconds"],
        train_fraction=data_cfg["train_fraction"],
        clip_mad=data_cfg["clip_mad"],
        seed=cfg["training"]["seed"],
    )
    train = EEGWindowDataset(
        n_examples=data_cfg["examples_per_epoch"], split="train", **common
    )
    val = EEGWindowDataset(n_examples=data_cfg["val_examples"], split="val", **common)
    return train, val


def make_model(cfg: dict) -> dac.model.DAC:
    model_cfg = dict(cfg["model"])
    model_cfg["sample_rate"] = cfg["data"]["sample_rate"]
    return dac.model.DAC(**model_cfg)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _uniform(
    bounds: Sequence[float], batch_size: int, device: torch.device
) -> torch.Tensor:
    low, high = (float(value) for value in bounds)
    return torch.empty(batch_size, 1, 1, device=device).uniform_(low, high)


def _global_phase_shift(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate positive/negative Fourier phases while preserving a real waveform."""
    n_samples = x.shape[-1]
    frequencies = torch.fft.fftfreq(n_samples, device=x.device)
    direction = torch.sign(frequencies)
    # The Nyquist bin is self-conjugate for even lengths and must stay real.
    if n_samples % 2 == 0:
        direction[n_samples // 2] = 0
    rotation = torch.exp(1j * angles * direction.view(1, 1, -1))
    return torch.fft.ifft(torch.fft.fft(x.float(), dim=-1) * rotation, dim=-1).real


@torch.no_grad()
def augment_eeg(
    clean: torch.Tensor, sample_rate: float, augment_cfg: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a corrupted input and its clean waveform reconstruction target.

    Additive EEG nuisances affect only the encoder input, making training
    denoising. Global Fourier phase rotation is applied to both input and target:
    an unknown phase rotation is not uniquely invertible from the rotated signal.
    """
    if not augment_cfg or not augment_cfg.get("enabled", False):
        return clean, clean

    batch_size = clean.shape[0]
    device = clean.device
    target = clean.clone()

    phase_cfg = augment_cfg.get("phase_shift", {})
    if phase_cfg.get("probability", 0) > 0:
        apply = (
            torch.rand(batch_size, 1, 1, device=device)
            < float(phase_cfg["probability"])
        )
        max_radians = float(phase_cfg.get("max_radians", math.pi / 4))
        angles = torch.empty(batch_size, 1, 1, device=device).uniform_(
            -max_radians, max_radians
        )
        angles = angles * apply
        target = _global_phase_shift(target, angles)

    corrupted = target.clone()
    time = torch.arange(clean.shape[-1], device=device, dtype=torch.float32)
    time = time.view(1, 1, -1) / float(sample_rate)

    drift_cfg = augment_cfg.get("slow_drift", {})
    if drift_cfg.get("probability", 0) > 0:
        apply = (
            torch.rand(batch_size, 1, 1, device=device)
            < float(drift_cfg["probability"])
        )
        amplitude = _uniform(drift_cfg["amplitude"], batch_size, device)
        frequency = _uniform(drift_cfg["frequency_hz"], batch_size, device)
        phase = _uniform([0, 2 * math.pi], batch_size, device)
        corrupted = corrupted + apply * amplitude * torch.sin(
            2 * math.pi * frequency * time + phase
        )

    baseline_cfg = augment_cfg.get("baseline_shift", {})
    if baseline_cfg.get("probability", 0) > 0:
        apply = (
            torch.rand(batch_size, 1, 1, device=device)
            < float(baseline_cfg["probability"])
        )
        offset = _uniform(baseline_cfg["offset"], batch_size, device)
        corrupted = corrupted + apply * offset

    line_cfg = augment_cfg.get("line_noise", {})
    if line_cfg.get("probability", 0) > 0:
        apply = (
            torch.rand(batch_size, 1, 1, device=device)
            < float(line_cfg["probability"])
        )
        choices = torch.as_tensor(
            line_cfg.get("frequencies_hz", [50, 60]), device=device
        )
        choice_index = torch.randint(len(choices), (batch_size,), device=device)
        frequency = choices[choice_index].view(batch_size, 1, 1)
        amplitude = _uniform(line_cfg["amplitude"], batch_size, device)
        phase = _uniform([0, 2 * math.pi], batch_size, device)
        corrupted = corrupted + apply * amplitude * torch.sin(
            2 * math.pi * frequency * time + phase
        )

    clip = float(augment_cfg.get("clip", 1.0))
    return corrupted.clamp(-clip, clip), target.clamp(-clip, clip)


def multiscale_stft_loss(
    estimate: torch.Tensor, target: torch.Tensor, windows: Iterable[int]
) -> torch.Tensor:
    """Linear-frequency magnitude and log-magnitude loss for EEG."""
    total = estimate.new_zeros(())
    x = estimate.squeeze(1).float()
    y = target.squeeze(1).float()
    windows = list(windows)
    for n_fft in windows:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        x_mag = torch.stft(
            x, n_fft, hop_length=n_fft // 4, window=window, return_complex=True
        ).abs()
        y_mag = torch.stft(
            y, n_fft, hop_length=n_fft // 4, window=window, return_complex=True
        ).abs()
        total = total + F.l1_loss(x_mag, y_mag)
        total = total + F.l1_loss(torch.log(x_mag + 1e-5), torch.log(y_mag + 1e-5))
    return total / len(windows)


def compute_losses(out: dict, target: torch.Tensor, loss_cfg: dict) -> Dict[str, torch.Tensor]:
    values = {
        "waveform": F.l1_loss(out["audio"], target),
        "stft": multiscale_stft_loss(out["audio"], target, loss_cfg["stft_windows"]),
        "commitment": out["vq/commitment_loss"],
        "codebook": out["vq/codebook_loss"],
    }
    values["total"] = sum(values[name] * float(loss_cfg[name]) for name in values)
    return values


def mean_metrics(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


@torch.no_grad()
def validate(model, loader, device, loss_cfg) -> Dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        x = batch["eeg"].to(device)
        out = model(x, sample_rate=model.sample_rate)
        losses = compute_losses(out, x, loss_cfg)
        rows.append({key: float(value.detach()) for key, value in losses.items()})
    return mean_metrics(rows)


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: dict, val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": cfg,
        },
        path,
    )


def train(cfg: dict) -> None:
    train_cfg = cfg["training"]
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = choose_device(train_cfg["device"])
    train_data, val_data = make_datasets(cfg)
    loader_args = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_data, shuffle=False, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)

    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
    )
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    hop_ms = 1000 * model.hop_length / model.sample_rate
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Training on {device}; {n_params / 1e6:.1f}M parameters; "
        f"hop={model.hop_length} samples ({hop_ms:g} ms)"
    )
    best = float("inf")
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        rows = []
        for batch in train_loader:
            clean = batch["eeg"].to(device, non_blocking=True)
            x, target = augment_eeg(
                clean, model.sample_rate, cfg.get("augmentations", {})
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x, sample_rate=model.sample_rate)
                losses = compute_losses(out, target, cfg["loss"])
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            rows.append({key: float(value.detach()) for key, value in losses.items()})

        train_metrics = mean_metrics(rows)
        val_metrics = validate(model, val_loader, device, cfg["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | train {train_metrics['total']:.4f} | "
            f"val {val_metrics['total']:.4f}"
        )
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, cfg, val_metrics["total"])
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, best)
        with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    print(f"Done. Best validation loss: {best:.4f}; checkpoint: {output_dir / 'best.pt'}")


def load_trained_model(cfg: dict, device: torch.device):
    checkpoint_path = Path(cfg["training"]["output_dir"]) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Train the model first; missing {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(checkpoint.get("config", cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint_path


def code_statistics(codes: np.ndarray, codebook_size: int) -> Tuple[np.ndarray, np.ndarray]:
    usage = np.zeros((codes.shape[0], codebook_size), dtype=np.int64)
    perplexity = np.zeros(codes.shape[0], dtype=np.float64)
    for level in range(codes.shape[0]):
        usage[level] = np.bincount(codes[level].ravel(), minlength=codebook_size)
        probability = usage[level] / max(usage[level].sum(), 1)
        nonzero = probability > 0
        perplexity[level] = np.exp(-np.sum(probability[nonzero] * np.log(probability[nonzero])))
    return usage, perplexity


@torch.no_grad()
def evaluate_codebooks(cfg: dict) -> None:
    """Measure code utilization across the deterministic validation sample."""
    device = choose_device(cfg["training"]["device"])
    _, val_data = make_datasets(cfg)
    model, checkpoint_path = load_trained_model(cfg, device)
    n_examples = min(
        len(val_data), int(cfg.get("evaluation", {}).get("code_examples", len(val_data)))
    )
    loader = DataLoader(
        val_data,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    usage = np.zeros(
        (model.n_codebooks, int(cfg["model"]["codebook_size"])), dtype=np.int64
    )
    seen = 0
    for batch in loader:
        if seen >= n_examples:
            break
        x = batch["eeg"].to(device)
        if seen + x.shape[0] > n_examples:
            x = x[: n_examples - seen]
        codes = model(x, sample_rate=model.sample_rate)["codes"].cpu().numpy()
        for level in range(codes.shape[1]):
            usage[level] += np.bincount(
                codes[:, level].ravel(), minlength=usage.shape[1]
            )
        seen += x.shape[0]

    rows = []
    for level, counts in enumerate(usage):
        probability = counts / max(counts.sum(), 1)
        nonzero = probability > 0
        perplexity = float(
            np.exp(-np.sum(probability[nonzero] * np.log(probability[nonzero])))
        )
        rows.append(
            {
                "level": level + 1,
                "codes_used": int(nonzero.sum()),
                "usage_fraction": float(nonzero.mean()),
                "perplexity": perplexity,
                "dominant_code_fraction": float(probability.max()),
            }
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "validation_examples": seen,
        "tokens_per_level": int(usage[0].sum()),
        "codebooks": rows,
    }
    output_path = Path(cfg["training"]["output_dir"]) / "codebook_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Evaluated {seen} validation windows ({usage[0].sum()} tokens/level)")
    for row in rows:
        print(
            f"level {row['level']:02d}: used {row['codes_used']:4d} | "
            f"perplexity {row['perplexity']:8.2f} | "
            f"dominant {100 * row['dominant_code_fraction']:6.2f}%"
        )
    print(f"Saved {output_path}")


@torch.no_grad()
def visualize(cfg: dict) -> None:
    output_dir = Path(cfg["training"]["output_dir"])
    import matplotlib.pyplot as plt

    device = choose_device(cfg["training"]["device"])
    _, val_data = make_datasets(cfg)
    model, checkpoint_path = load_trained_model(cfg, device)
    item = val_data.fixed_item(
        channel_index=int(cfg["visualize"].get("channel_index", 0))
    )
    x = item["eeg"].unsqueeze(0).to(device)
    out = model(x, sample_rate=model.sample_rate)
    reconstruction = out["audio"]
    codes = out["codes"][0].cpu().numpy()

    noise_levels = [0.0] + [float(value) for value in cfg["visualize"]["noise_std"]]
    agreement = []
    noisy_codes = []
    for noise_std in noise_levels:
        corrupted = torch.clamp(x + noise_std * torch.randn_like(x), -1.0, 1.0)
        c = model(corrupted, sample_rate=model.sample_rate)["codes"][0].cpu().numpy()
        noisy_codes.append(c)
        agreement.append((c == codes).mean(axis=1))
    agreement = np.asarray(agreement)

    usage, perplexity = code_statistics(codes, int(cfg["model"]["codebook_size"]))
    sfreq = float(cfg["data"]["sample_rate"])
    time = np.arange(x.shape[-1]) / sfreq
    token_time = np.arange(codes.shape[-1]) * model.hop_length / sfreq
    x_np = x[0, 0].cpu().numpy()
    y_np = reconstruction[0, 0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].plot(time, x_np, label="original", linewidth=0.9)
    axes[0, 0].plot(time, y_np, label="reconstruction", linewidth=0.9, alpha=0.8)
    axes[0, 0].set(title=f"EEG reconstruction — {item['channel']}", xlabel="Time (s)")
    axes[0, 0].legend()

    for ax, signal, title in [
        (axes[0, 1], x_np, "Original spectrum"),
        (axes[0, 2], y_np, "Reconstructed spectrum"),
    ]:
        ax.specgram(signal, NFFT=256, Fs=sfreq, noverlap=192, cmap="magma")
        ax.set_ylim(0, min(45, sfreq / 2))
        ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (Hz)")

    axes[1, 0].imshow(
        codes,
        aspect="auto",
        interpolation="nearest",
        extent=[token_time[0], token_time[-1] if len(token_time) > 1 else 0, codes.shape[0], 0],
        cmap="viridis",
    )
    axes[1, 0].set(title="RVQ token IDs", xlabel="Time (s)", ylabel="Codebook level")

    used_fraction = (usage > 0).mean(axis=1)
    levels = np.arange(1, codes.shape[0] + 1)
    axes[1, 1].bar(levels, used_fraction * 100)
    axes[1, 1].set(
        title="Codebook use in this window",
        xlabel="Codebook level",
        ylabel="Codes used (%)",
        ylim=(0, 100),
    )
    for level, value in zip(levels, perplexity):
        axes[1, 1].text(level, used_fraction[level - 1] * 100 + 2, f"P={value:.1f}", ha="center", fontsize=7)

    for level in range(codes.shape[0]):
        axes[1, 2].plot(noise_levels, agreement[:, level] * 100, marker="o", alpha=0.6)
    axes[1, 2].plot(
        noise_levels,
        agreement.mean(axis=1) * 100,
        color="black",
        marker="o",
        linewidth=3,
        label="mean",
    )
    axes[1, 2].set(
        title="Token stability under added noise",
        xlabel="Noise SD (normalized units)",
        ylabel="Unchanged tokens (%)",
        ylim=(0, 105),
    )
    axes[1, 2].legend()

    output_file = output_dir / cfg["visualize"]["output_file"]
    fig.suptitle(f"EEG DAC diagnostics — {checkpoint_path.name}")
    fig.savefig(output_file, dpi=160)
    plt.close(fig)
    np.savez_compressed(
        output_file.with_suffix(".npz"),
        original=x_np,
        reconstruction=y_np,
        codes=codes,
        noisy_codes=np.stack(noisy_codes),
        noise_levels=np.asarray(noise_levels),
        agreement=agreement,
        usage=usage,
        perplexity=perplexity,
        sample_rate=sfreq,
        hop_length=model.hop_length,
    )
    print(f"Saved {output_file}")
    print(f"Saved {output_file.with_suffix('.npz')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "visualize"])
    parser.add_argument("--config", default="conf/eeg.yml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.command == "train":
        train(cfg)
    elif args.command == "evaluate":
        evaluate_codebooks(cfg)
    else:
        visualize(cfg)


if __name__ == "__main__":
    main()
