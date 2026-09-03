"""Train EEG with the original DAC training script's structure and config style.

Run on the reserved physical GPU 3::
    CUDA_VISIBLE_DEVICES=3 python scripts/train_eeg.py --args.load conf/base_eeg.yml
"""


import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# ADDED FOR EEG
import glob
import json
import math
import random
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Sequence, Tuple

# ADDED FOR EEG — MNE needs writable cache locations.
_cache_root = Path(tempfile.gettempdir()) / f"eeg-dac-{os.getuid()}"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))

import argbind
import torch
# ORIGINAL AUDIO, not needed here.
# from audiotools import AudioSignal
# from audiotools import ml
# from audiotools.core import util
# from audiotools.data import transforms
# from audiotools.data.datasets import AudioDataset
# from audiotools.data.datasets import AudioLoader
# from audiotools.data.datasets import ConcatDataset
# from audiotools.ml.decorators import timer
# from audiotools.ml.decorators import Tracker
# from audiotools.ml.decorators import when
# from torch.utils.tensorboard import SummaryWriter

# ADDED FOR EEG (basically replacing the audio specific tools)
import mne
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import dac
from eeg_plotting import training_plots

warnings.filterwarnings("ignore", category=UserWarning)  # KEPT from original DAC.

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
# AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
# Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)
AdamW = argbind.bind(torch.optim.AdamW, "generator")

# @argbind.bind("generator", "discriminator")
@argbind.bind("generator")

def ExponentialLR(optimizer, gamma: float = 1.0):
    """KEPT: same scheduler wrapper as original DAC."""
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DAC = argbind.bind(dac.model.DAC)  # KEPT from original DAC.
# Discriminator = argbind.bind(dac.model.Discriminator)

# Data
# AudioDataset = argbind.bind(AudioDataset, "train", "val")
# AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
# filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
#     "BaseTransform",
#     "Compose",
#     "Choose",
# ]
# tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
# filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
# losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


# ADDED FOR EEG EXPERIMENT TRACKING: optional Weights & Biases configuration.
@argbind.bind()
def WandB(
    enabled: bool = False,
    project: str = "eeg-dac",
    entity: str = "",
    name: str = "eeg-alice-teacher-student",
    group: str = "",
    tags: list = ["eeg", "dac", "teacher-student"],
    mode: str = "online",
    log_freq: int = 1,
    log_reconstruction: bool = True,
    watch_model: bool = False,
) -> dict:
    return {
        "enabled": enabled,
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "tags": tags,
        "mode": mode,
        "log_freq": log_freq,
        "log_reconstruction": log_reconstruction,
        "watch_model": watch_model,
    }


# ADDED FOR EEG: one consistent, training-integrated diagnostic plot schedule.
@argbind.bind()
def EEGPlots(
    enabled: bool = True,
    every_epochs: int = 10,
    n_examples: int = 16,
    augmentation_examples: int = 5,
    seconds: float = 5.0,
    seed: int = 7,
    psd_fmin: float = 1.0,
    psd_fmax: float = 45.0,
    psd_window_seconds: float = 1.0,
) -> dict:
    return locals()


def initialize_wandb(config: dict, args, output: Path, model: torch.nn.Module):
    """Start an optional W&B run without making wandb a required dependency."""
    if not config["enabled"]:
        return None, None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled, but wandb is not installed. Run "
            "`/data/groups/bci/jonas/venv_dac/bin/pip install wandb`, then `wandb login`."
        ) from error

    init_kwargs = {
        "project": config["project"],
        "name": config["name"],
        "tags": config["tags"],
        "mode": config["mode"],
        "dir": str(output),
        "config": dict(args),
    }
    if config["entity"]:
        init_kwargs["entity"] = config["entity"]
    if config["group"]:
        init_kwargs["group"] = config["group"]
    run = wandb.init(**init_kwargs)
    # Per-batch training curves use training_step; epoch aggregates and
    # validation curves use epoch.
    wandb.define_metric("training_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="training_step")
    for namespace in (
        "train (avg per epoch)/*",
        "validation (per epoch)/*",
        "robustness/*",
        "timing/*",
        "best/*",
    ):
        wandb.define_metric(namespace, step_metric="epoch")
    run.config.update(
        {
            "model/parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model/hop_length": int(model.hop_length),
            "model/token_rate_hz": float(model.sample_rate / model.hop_length),
        },
        allow_val_change=True,
    )
    if config["watch_model"]:
        wandb.watch(model, log="gradients", log_freq=config["log_freq"])
    return wandb, run


def save_run_configuration(args, output: Path) -> None:
    """Save both the authored input and fully resolved run configuration."""
    resolved = dict(args)
    argbind.dump_args(resolved, output / "config_resolved.yml")
    source_name = resolved.get("args.load")
    if source_name:
        source = Path(source_name).expanduser().resolve()
        if source.is_file():
            shutil.copy2(source, output / "config_input.yml")


def get_infinite_loader(dataloader):
    """KEPT from original DAC: cycle over a finite loader indefinitely."""
    while True:
        for batch in dataloader:
            yield batch


# ============================================================================
# Added: MNE fif data loader (may wanna use our .pt files or so later)
# treats channels independently
# normalized with MAD (median absolute deviation) per channel not across channels
# normalization is based on 3 X 30s segments TODO: we may want to do this for each 10s chunk
# loads 10s segments, if sam-ple size = 12, we repeat this 12 times independetly (pretty slow)
# TODO: for more efficient loading, we may want to switch to the .pt files
# ============================================================================

@dataclass
class EEGRecording:
    path: str
    raw: object
    channel_names: List[str]
    median: np.ndarray
    scale: np.ndarray


class EEGWindowDataset(Dataset):
    """Random single-electrode windows from a temporal recording split."""

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
        self.recordings = list(recordings)
        self.window_samples = int(round(sample_rate * window_seconds))
        self.n_examples = int(n_examples)
        self.train_fraction = float(train_fraction)
        self.clip_mad = float(clip_mad)
        self.split = split
        self.seed = int(seed)
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        for recording in self.recordings:
            split_sample = int(recording.raw.n_times * self.train_fraction)
            available = split_sample if split == "train" else recording.raw.n_times - split_sample
            if available < self.window_samples:
                raise ValueError(f"{recording.path}: {split} interval is shorter than one window")

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.split == "val":
            rng = np.random.default_rng(self.seed + index)
        else:
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1) + index)
        recording = self.recordings[int(rng.integers(len(self.recordings)))]
        channel = int(rng.integers(len(recording.channel_names)))
        split_sample = int(recording.raw.n_times * self.train_fraction)
        lo, hi = (0, split_sample) if self.split == "train" else (split_sample, recording.raw.n_times)
        start = int(rng.integers(lo, hi - self.window_samples + 1))
        values = recording.raw.get_data(
            picks=[channel], start=start, stop=start + self.window_samples
        ).astype(np.float32, copy=False)
        values = (values - recording.median[channel]) / recording.scale[channel]
        values = np.clip(values, -self.clip_mad, self.clip_mad) / self.clip_mad
        return {
            "eeg": torch.from_numpy(values.copy()),
            "recording": recording.path,
            "channel": recording.channel_names[channel],
            "start_sample": start,
        }


def resolve_files(patterns: Sequence[str]) -> List[str]:
    """ADDED FOR EEG: expand FIF globs."""
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(str(Path(pattern).expanduser())))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No FIF files matched {list(patterns)}")
    return paths


def load_recordings(
    files: Sequence[str],
    sample_rate: int,
    train_fraction: float,
    normalization_seconds: float,
    normalization_chunks: int,
) -> List[EEGRecording]:
    """ADDED FOR EEG: open FIF lazily and estimate train-only robust scales."""
    recordings = []
    for path in resolve_files(files):
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        raw.pick(picks="eeg")
        if not math.isclose(float(raw.info["sfreq"]), float(sample_rate)):
            raise ValueError(f"{path}: expected {sample_rate} Hz, got {raw.info['sfreq']} Hz")
        train_stop = int(raw.n_times * train_fraction)
        total = min(int(round(normalization_seconds * sample_rate)), train_stop)
        chunk_samples = max(1, total // normalization_chunks)
        starts = np.linspace(0, max(0, train_stop - chunk_samples), normalization_chunks, dtype=int)
        calibration = np.concatenate(
            [raw.get_data(start=int(start), stop=int(start) + chunk_samples) for start in starts],
            axis=1,
        ).astype(np.float32, copy=False)
        median = np.median(calibration, axis=1, keepdims=True).astype(np.float32)
        mad = np.median(np.abs(calibration - median), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * mad, np.finfo(np.float32).eps).astype(np.float32)
        recordings.append(EEGRecording(path, raw, list(raw.ch_names), median, scale))
        print(
            f"Indexed {Path(path).name}: {len(raw.ch_names)} EEG channels, "
            f"{raw.n_times / sample_rate:.1f} s at {sample_rate} Hz"
        )
    return recordings


# ============================================================================
# DAC's original data transformation

# @argbind.bind("train", "val")
# def build_transform(
#     augment_prob: float = 1.0,
#     preprocess: list = ["Identity"],
#     augment: list = ["Identity"],
#     postprocess: list = ["Identity"],
# ):
#     to_tfm = lambda l: [getattr(tfm, x)() for x in l]
#     preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
#     augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
#     postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
#     transform = transforms.Compose(preprocess, augment, postprocess)
#     return transform
#
# EEG REPLACEMENT: return explicit corruption parameters. The transformation is
# applied in train_loop so it can return both noisy student and clean target.
# ============================================================================

# ALL VALUES BELOW ARE OVERRIDEN WITH YML
@argbind.bind() #dac originally had different yml file configs for train & eval. We only call this in the train loop, so it's never called in eval
def EEGAugment(
    enabled: bool = True,
    slow_drift_probability: float = 0.5,
    slow_drift_frequency_hz: list = [0.05, 0.5],
    slow_drift_amplitude: list = [0.0, 0.15],
    baseline_shift_probability: float = 0.5,
    baseline_shift_offset: list = [-0.15, 0.15],
    phase_shift_probability: float = 0.5,
    phase_shift_max_radians: float = math.pi / 4,
    line_noise_probability: float = 0.5,
    line_noise_frequencies_hz: list = [50.0, 60.0],
    line_noise_amplitude: list = [0.0, 0.08],
    gaussian_noise_probability: float = 0.5,
    gaussian_noise_std: list = [0.0, 0.1],
    signal_mix_probability: float = 0.5,
    signal_mix_fraction: list = [0.0, 0.1],
    clip: float = 1.0,
) -> dict:
    """ADDED FOR EEG: materialize augmentation arguments from argbind."""
    return locals()


def _uniform(bounds: Sequence[float], batch_size: int, device: torch.device) -> torch.Tensor:
    """ADDED FOR EEG: draw one corruption magnitude per example."""
    low, high = (float(value) for value in bounds)
    return torch.empty(batch_size, 1, 1, device=device).uniform_(low, high)


def _global_phase_shift(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """ADDED FOR EEG: rotate Fourier phase while preserving a real signal."""
    n_samples = values.shape[-1]
    frequencies = torch.fft.fftfreq(n_samples, device=values.device)
    direction = torch.sign(frequencies)
    if n_samples % 2 == 0:
        direction[n_samples // 2] = 0
    rotation = torch.exp(1j * angles * direction.view(1, 1, -1))
    return torch.fft.ifft(torch.fft.fft(values.float(), dim=-1) * rotation, dim=-1).real


@torch.no_grad()
def augment_eeg(clean: torch.Tensor, sample_rate: float, cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """ADDED FOR EEG: produce corrupted student input and clean target."""
    if not cfg["enabled"]:
        return clean, clean
    batch_size, device = clean.shape[0], clean.device
    target = clean.clone()

    apply = (torch.rand(batch_size, 1, 1, device=device) < cfg["phase_shift_probability"])
    angles = torch.empty(batch_size, 1, 1, device=device).uniform_(
        -cfg["phase_shift_max_radians"], cfg["phase_shift_max_radians"]
    ) * apply
    target = _global_phase_shift(target, angles)
    corrupted = target.clone()
    time = torch.arange(clean.shape[-1], device=device, dtype=torch.float32).view(1, 1, -1) / sample_rate

    apply = (torch.rand(batch_size, 1, 1, device=device) < cfg["slow_drift_probability"])
    amplitude = _uniform(cfg["slow_drift_amplitude"], batch_size, device)
    frequency = _uniform(cfg["slow_drift_frequency_hz"], batch_size, device)
    phase = _uniform([0.0, 2 * math.pi], batch_size, device)
    corrupted += apply * amplitude * torch.sin(2 * math.pi * frequency * time + phase)

    apply = (torch.rand(batch_size, 1, 1, device=device) < cfg["baseline_shift_probability"])
    corrupted += apply * _uniform(cfg["baseline_shift_offset"], batch_size, device)

    apply = (torch.rand(batch_size, 1, 1, device=device) < cfg["line_noise_probability"])
    choices = torch.tensor(cfg["line_noise_frequencies_hz"], device=device)
    frequency = choices[torch.randint(len(choices), (batch_size,), device=device)].view(-1, 1, 1)
    amplitude = _uniform(cfg["line_noise_amplitude"], batch_size, device)
    phase = _uniform([0.0, 2 * math.pi], batch_size, device)
    corrupted += apply * amplitude * torch.sin(2 * math.pi * frequency * time + phase)

    apply = (torch.rand(batch_size, 1, 1, device=device) < cfg["gaussian_noise_probability"])
    corrupted += apply * _uniform(cfg["gaussian_noise_std"], batch_size, device) * torch.randn_like(corrupted)

    if batch_size > 1:
        apply = (torch.rand(batch_size, 1, 1, device=device) < cfg["signal_mix_probability"])
        corrupted += apply * _uniform(cfg["signal_mix_fraction"], batch_size, device) * target.roll(1, dims=0)
    return corrupted.clamp(-cfg["clip"], cfg["clip"]), target.clamp(-cfg["clip"], cfg["clip"])


# ============================================================================
# original DAC build dataset

# @argbind.bind("train", "val", "test")
# def build_dataset(
#     sample_rate: int,
#     folders: dict = None,
# ):
#     # Give one loader per key/value of dictionary, where
#     # value is a list of folders. Create a dataset for each one.
#     # Concatenate the datasets with ConcatDataset, which
#     # cycles through them.
#     datasets = []
#     for _, v in folders.items():
#         loader = AudioLoader(sources=v)
#         transform = build_transform()
#         dataset = AudioDataset(loader, sample_rate, transform=transform)
#         datasets.append(dataset)
#
#     dataset = ConcatDataset(datasets)
#     dataset.transform = transform
#     return dataset
#
# EEG REPLACEMENT: MNE FIF recordings and deterministic temporal splits.
# ============================================================================


@argbind.bind("train", "val")
def build_dataset(
    sample_rate: int,
    files: list = None,
    window_seconds: float = 10.0,
    n_examples: int = 8192,
    split: str = "train",
    train_fraction: float = 0.8,
    clip_mad: float = 8.0,
    normalization_seconds: float = 30.0,
    normalization_chunks: int = 3,
    seed: int = 0,
) -> EEGWindowDataset:
    if files is None:
        raise ValueError("build_dataset.files must list at least one FIF pattern")
    recordings = load_recordings(
        files, sample_rate, train_fraction, normalization_seconds, normalization_chunks
    )
    return EEGWindowDataset(
        recordings, sample_rate, window_seconds, n_examples, train_fraction,
        clip_mad, split, seed
    )


# ============================================================================
# losses
# Original MelSpectrogramLoss and GANLoss construction is commented out below.
# ============================================================================
# waveform_loss = losses.L1Loss()
# stft_loss = losses.MultiScaleSTFTLoss()
# mel_loss = losses.MelSpectrogramLoss()
# gan_loss = losses.GANLoss(discriminator)


@argbind.bind()
def EEGLoss(stft_windows: list = [64, 128, 256], stft_mode: str = "log_power") -> dict:
    """ADDED FOR EEG: configure linear-frequency spectral reconstruction."""
    return {"stft_windows": stft_windows, "stft_mode": stft_mode}


def multiscale_stft_loss(
    estimate: torch.Tensor, target: torch.Tensor, windows: Iterable[int], mode: str
) -> torch.Tensor:
    """CHANGED: linear-frequency EEG STFT replaces the audio mel loss."""
    total = estimate.new_zeros(())
    x, y = estimate.squeeze(1).float(), target.squeeze(1).float()
    windows = list(windows)
    for n_fft in windows:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        x_mag = torch.stft(x, n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
        y_mag = torch.stft(y, n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
        if mode == "log_power":
            total += F.l1_loss(torch.log(x_mag.square() + 1e-8), torch.log(y_mag.square() + 1e-8))
        elif mode == "magnitude":
            total += F.l1_loss(x_mag, y_mag)
            total += F.l1_loss(torch.log(x_mag + 1e-5), torch.log(y_mag + 1e-5))
        else:
            raise ValueError(f"Unknown STFT mode {mode}")
    return total / len(windows)

# this loss is a new addition here. Original DA seems to have avoided code book collapse without such explicit loss
# given that, we may not want to use this loss and figure out how to tune the training code to prevent collapse
# however, with this loss, we prevent collapse!
def assignment_diversity_loss(model, latents: torch.Tensor, temperature: float = 0.1, confidence: float = 0.1):
    """ADDED FOR EEG: discourage global codebook collapse."""
    losses, offset, eps = [], 0, 1e-8
    for quantizer in model.quantizer.quantizers:
        width = quantizer.codebook_dim
        projected = F.normalize(latents[:, offset : offset + width].float().transpose(1, 2), dim=-1)
        offset += width
        codebook = F.normalize(quantizer.codebook.weight.float(), dim=-1)
        probability = (projected @ codebook.t() / temperature).softmax(dim=-1)
        sample_entropy = -(probability * (probability + eps).log()).sum(dim=-1).mean()
        average = probability.mean(dim=(0, 1))
        batch_entropy = -(average * (average + eps).log()).sum()
        losses.append(confidence * sample_entropy - batch_entropy)
    return torch.stack(losses).mean()


def latent_consistency_loss(model, student: torch.Tensor, teacher: torch.Tensor):
    """ADDED FOR EEG: align noisy-student and detached clean-teacher latents."""
    # teacher & student versions of the signal get derived here: student_input, target = augment_eeg
    losses, offset = [], 0
    for quantizer in model.quantizer.quantizers:
        width = quantizer.codebook_dim
        losses.append(
            1 - F.cosine_similarity(
                student[:, offset : offset + width].float(),
                teacher[:, offset : offset + width].detach().float(),
                dim=1,
            ).mean()
        )
        offset += width
    return torch.stack(losses).mean()


def compute_losses(
    model,
    output: dict,
    target: torch.Tensor,
    loss_cfg: dict,
    lambdas: dict,
    student_latents: torch.Tensor = None,
    teacher_latents: torch.Tensor = None,
):
    """CHANGED: replace mel/GAN terms with the current EEG objective."""
    values = {
        "waveform/loss": F.l1_loss(output["audio"], target),
        "stft/loss": multiscale_stft_loss(
            output["audio"], target, loss_cfg["stft_windows"], loss_cfg["stft_mode"]
        ),
        "vq/commitment_loss": output["vq/commitment_loss"],
        "vq/codebook_loss": output["vq/codebook_loss"],
        "eeg/diversity_loss": assignment_diversity_loss(model, output["latents"]),
    }
    if student_latents is not None and teacher_latents is not None:
        values["eeg/latent_consistency_loss"] = latent_consistency_loss(model, student_latents, teacher_latents)
    values["loss"] = sum(values[key] * float(weight) for key, weight in lambdas.items() if key in values)
    return values


# ============================================================================
# State
# ============================================================================


@dataclass
class State:
    generator: torch.nn.Module
    optimizer_g: torch.optim.Optimizer
    scheduler_g: object
    train_data: EEGWindowDataset
    val_data: EEGWindowDataset
    augment: dict
    loss_cfg: dict
    device: torch.device

    # ORIGINAL AUDIO FIELDS — COMMENTED OUT:
    # discriminator: Discriminator
    # optimizer_d: AdamW
    # scheduler_d: ExponentialLR
    # mel_loss: losses.MelSpectrogramLoss
    # gan_loss: losses.GANLoss
    # tracker: Tracker


# ============================================================================
# load
# ============================================================================


def load(args, device: torch.device, resume: bool = False, save_path: str = "runs/eeg_dac_style") -> State:
    # generator = DAC(); discriminator = Discriminator()
    # optimizer_d = AdamW(discriminator.parameters())
    generator = DAC().to(device)
    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters())
        scheduler_g = ExponentialLR(optimizer_g)
    if resume and (Path(save_path) / "latest.pt").exists():
        checkpoint = torch.load(Path(save_path) / "latest.pt", map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["model"])
        optimizer_g.load_state_dict(checkpoint["optimizer"])
        scheduler_g.load_state_dict(checkpoint["scheduler"])

    with argbind.scope(args, "train"):
        train_data = build_dataset(generator.sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(generator.sample_rate)
    return State(
        generator, optimizer_g, scheduler_g, train_data, val_data,
        EEGAugment(), EEGLoss(), device
    )

def mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """ADDED FOR EEG: average ordinary metric dictionaries."""
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


# ============================================================================
# val_loop
# ============================================================================


@torch.no_grad()
def val_loop(batch, state: State, lambdas: dict):
    state.generator.eval()
    target = batch["eeg"].to(state.device)
    output = state.generator(target, sample_rate=state.generator.sample_rate)
    losses = compute_losses(state.generator, output, target, state.loss_cfg, lambdas)
    return losses, output["codes"]


# ============================================================================
# train_loop
#
# ORIGINAL DISCRIMINATOR UPDATE — COMMENTED OUT:
# output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)
# state.optimizer_d.zero_grad(); backward(...); state.optimizer_d.step()
#
# EEG replacement keeps the generator forward/backward and adds clean teacher.
# ============================================================================


def train_loop(state: State, batch, lambdas: dict, scaler, amp: bool, grad_clip: float):
    state.generator.train()
    clean = batch["eeg"].to(state.device)
    student_input, target = augment_eeg(clean, state.generator.sample_rate, state.augment)
    with torch.autocast(device_type=state.device.type, enabled=amp and state.device.type == "cuda"):
        output = state.generator(student_input, sample_rate=state.generator.sample_rate)
        losses = compute_losses(state.generator, output, target, state.loss_cfg, lambdas)
        consistency_weight = float(lambdas.get("eeg/latent_consistency_loss", 0.0))
        if consistency_weight:
            with torch.no_grad():
                teacher = state.generator(target, sample_rate=state.generator.sample_rate)
            losses["eeg/latent_consistency_loss"] = latent_consistency_loss(
                state.generator, output["latents"], teacher["latents"]
            )
            losses["loss"] += consistency_weight * losses["eeg/latent_consistency_loss"]

    state.optimizer_g.zero_grad(set_to_none=True)
    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(state.optimizer_g)
    grad_norm = torch.nn.utils.clip_grad_norm_(state.generator.parameters(), grad_clip)
    scale_before = scaler.get_scale()
    scaler.step(state.optimizer_g)
    scaler.update()
    # GradScaler skips optimizer.step() when it detects non-finite gradients.
    # Only advance the scheduler after a real optimizer update.
    if scaler.get_scale() >= scale_before:
        state.scheduler_g.step()
    metrics = {key: float(value.detach()) for key, value in losses.items()}
    metrics["other/grad_norm"] = float(grad_norm.detach())
    metrics["other/learning_rate"] = float(state.optimizer_g.param_groups[0]["lr"])
    return metrics


# ============================================================================
# CHANGED FROM ORIGINAL DAC: checkpoint
# Original saves generator and discriminator folders. EEG saves one .pt package.
# ============================================================================


def checkpoint(state: State, path: Path, iteration: int, validation: dict, args) -> None:
    """CHANGED: save one PyTorch package rather than generator/discriminator folders."""
    torch.save(
        {
            "model": state.generator.state_dict(),
            "optimizer": state.optimizer_g.state_dict(),
            "scheduler": state.scheduler_g.state_dict(),
            "iteration": iteration,
            "validation": validation,
            "args": dict(args),
        },
        path,
    )


# ============================================================================
# CHANGED FROM ORIGINAL DAC: save_samples
# Original writes playable audio to TensorBoard. EEG writes exact arrays for
# subsequent plotting, avoiding an audio interpretation of EEG.
# ============================================================================


@torch.no_grad()
def save_samples(state: State, save_path: Path) -> None:
    state.generator.eval()
    batch = state.val_data[0]["eeg"].unsqueeze(0).to(state.device)
    output = state.generator(batch, sample_rate=state.generator.sample_rate)
    np.savez_compressed(
        save_path / "latest_reconstruction.npz",
        original=batch[0, 0].cpu().numpy(),
        reconstruction=output["audio"][0, 0].cpu().numpy(),
        codes=output["codes"][0].cpu().numpy(),
        sample_rate=state.generator.sample_rate,
    )


# ============================================================================
# KEPT IN STRUCTURE, CHANGED IN METRICS: validate
# ============================================================================


@torch.no_grad()
def validate(state: State, loader, lambdas: dict) -> Tuple[Dict[str, float], np.ndarray]:
    """CHANGED: validate tensor reconstruction and discrete-code health."""
    rows = []
    usage = np.zeros((state.generator.n_codebooks, state.generator.codebook_size), dtype=np.int64)
    for batch in loader:
        losses, codes = val_loop(batch, state, lambdas)
        rows.append({key: float(value.detach()) for key, value in losses.items()})
        codes = codes.cpu().numpy()
        for level in range(codes.shape[1]):
            usage[level] += np.bincount(codes[:, level].ravel(), minlength=state.generator.codebook_size)
    metrics = mean_metrics(rows)
    perplexities, used, dominant = [], [], []
    for counts in usage:
        probability = counts / max(counts.sum(), 1)
        nonzero = probability > 0
        perplexities.append(float(np.exp(-np.sum(probability[nonzero] * np.log(probability[nonzero])))))
        used.append(int(nonzero.sum()))
        dominant.append(float(probability.max()))
    for level, (perplexity, n_used, dominant_fraction) in enumerate(
        zip(perplexities, used, dominant), start=1
    ):
        metrics[f"codebook_{level}/perplexity"] = perplexity
        metrics[f"codebook_{level}/codes_used"] = n_used
        metrics[f"codebook_{level}/dominant_fraction"] = dominant_fraction
    metrics.update(
        code_perplexity_min=min(perplexities),
        codes_used_min=min(used),
        dominant_code_fraction_max=max(dominant),
    )
    return metrics, usage


# ============================================================================
# CHANGED FROM ORIGINAL DAC: train
# Keeps infinite loader, periodic validation, scheduling, samples, checkpoints;
# replaces Accelerator/Tracker with explicit PyTorch while retaining the proven
# EEG numerical path.
# ============================================================================


@argbind.bind(without_prefix=True)
def train(
    args,
    save_path: str = "runs/eeg_dac_style",
    num_iters: int = 20490,
    valid_freq: int = 683,
    sample_freq: int = 683,
    batch_size: int = 12,
    val_batch_size: int = 12,
    num_workers: int = 0,
    seed: int = 0,
    amp: bool = True,
    grad_clip: float = 10.0,
    lambdas: dict = None,
    resume: bool = False,
):
    """CHANGED: retain DAC loop structure with the proven EEG optimization path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    save_run_configuration(args, output)
    state = load(args, device, resume=resume, save_path=save_path)
    wandb_config = WandB()
    plot_config = EEGPlots()
    wandb, wandb_run = initialize_wandb(wandb_config, args, output, state.generator)
    train_loader = get_infinite_loader(
        DataLoader(
            state.train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    )
    val_loader = DataLoader(
        state.val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    history_path = output / "history.json"
    history = json.loads(history_path.read_text()) if resume and history_path.exists() else []
    best = min((row["validation"]["loss"] for row in history), default=float("inf"))
    interval_rows = []
    epoch_started_at = time.perf_counter()

    # These are fixed held-out examples, so visual changes across epochs reflect
    # model learning rather than a changing selection of EEG windows.
    plots_dir = output / "plots"
    startup_wandb_metrics = {}
    if plot_config["enabled"]:
        plots_dir.mkdir(parents=True, exist_ok=True)
        training_plots.save_example_manifest(
            state.val_data, plot_config["n_examples"], state.generator.sample_rate,
            plots_dir / "examples.json",
        )
        augmentation_path = training_plots.plot_augmentations(
            state.val_data,
            augment_eeg,
            state.augment,
            state.generator.sample_rate,
            plots_dir / "augmentations.png",
            columns=plot_config["augmentation_examples"],
            seconds=plot_config["seconds"],
            seed=plot_config["seed"],
        )
        if wandb_run is not None:
            # Commit this with the first scalar row. If media alone is the first
            # W&B row, a new automatic workspace may omit the scalar panels.
            startup_wandb_metrics["plots/augmentation_overview"] = wandb.Image(
                str(augmentation_path)
            )

    for iteration in range(num_iters):
        completed_epoch = False
        training = train_loop(state, next(train_loader), lambdas, scaler, amp, grad_clip)
        interval_rows.append(training)
        validation_due = (iteration + 1) % valid_freq == 0 or iteration + 1 == num_iters
        if wandb_run is not None:
            step_metrics = {
                f"train/{key}": value
                for key, value in training.items()
                if key != "loss"
            }
            step_metrics["train/total_loss"] = training["loss"]
            wandb_run.log(
                {
                    **step_metrics,
                    **startup_wandb_metrics,
                    "training_step": iteration + 1,
                },
                step=iteration + 1,
                # Validation metrics for this exact step are added below.
                commit=not validation_due,
            )
            startup_wandb_metrics = {}
        if validation_due:
            training = mean_metrics(interval_rows)
            interval_rows = []
            validation, codebook_usage = validate(state, val_loader, lambdas)
            epoch = math.ceil((iteration + 1) / valid_freq)
            epoch_seconds = time.perf_counter() - epoch_started_at
            agreement = {}
            if plot_config["enabled"]:
                agreement = training_plots.token_agreement(
                    state.generator,
                    state.val_data,
                    augment_eeg,
                    state.augment,
                    state.generator.sample_rate,
                    state.device,
                    count=plot_config["n_examples"],
                    seed=plot_config["seed"],
                )
            row = {
                "epoch": epoch,
                "iteration": iteration + 1,
                "train": training,
                "validation": validation,
                "token_agreement": agreement,
            }
            row["timing"] = {"epoch_seconds": epoch_seconds}
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2) + "\n")
            checkpoint(state, output / "latest.pt", iteration + 1, validation, args)
            if validation["loss"] < best:
                best = validation["loss"]
                checkpoint(state, output / "best.pt", iteration + 1, validation, args)
            if wandb_run is not None:
                epoch_train_metrics = {
                    f"train (avg per epoch)/{key}": value
                    for key, value in training.items()
                    if key != "loss"
                }
                epoch_train_metrics["train (avg per epoch)/total_loss"] = training["loss"]
                validation_metrics = {
                    f"validation (per epoch)/{key}": value
                    for key, value in validation.items()
                    if key != "loss"
                }
                validation_metrics["validation (per epoch)/total_loss"] = validation["loss"]
                robustness_metrics = {
                    f"robustness/code_agreement/{key}": value
                    for key, value in agreement.items()
                }
                wandb_run.log(
                    {
                        **epoch_train_metrics,
                        **validation_metrics,
                        **robustness_metrics,
                        "epoch": epoch,
                        "timing/epoch_seconds": epoch_seconds,
                        "best/validation_loss": best,
                    },
                    step=iteration + 1,
                    # Keep this step open when images will be added immediately below.
                    commit=not (
                        plot_config["enabled"]
                        and (
                            epoch % max(1, plot_config["every_epochs"]) == 0
                            or iteration + 1 == num_iters
                        )
                    ),
                )
            print(
                f"epoch {epoch:03d} | train {training['loss']:.4f} | "
                f"val {validation['loss']:.4f} | ppl {validation['code_perplexity_min']:.1f} | "
                f"time {epoch_seconds:.1f}s"
            )
            completed_epoch = True
            plot_due = (
                plot_config["enabled"]
                and (
                    epoch % max(1, plot_config["every_epochs"]) == 0
                    or iteration + 1 == num_iters
                )
            )
            if plot_due:
                waveform_path, psd_path = training_plots.plot_reconstructions(
                    state.generator,
                    state.val_data,
                    state.generator.sample_rate,
                    state.device,
                    epoch,
                    plots_dir,
                    count=plot_config["n_examples"],
                    seconds=plot_config["seconds"],
                    psd_fmin=plot_config["psd_fmin"],
                    psd_fmax=plot_config["psd_fmax"],
                    psd_window_seconds=plot_config["psd_window_seconds"],
                )
                codebook_path = training_plots.plot_codebook_usage(
                    codebook_usage, epoch, agreement, plots_dir
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "plots/waveform_reconstruction": wandb.Image(str(waveform_path)),
                            "plots/psd_reconstruction": wandb.Image(str(psd_path)),
                            "plots/codebook_usage": wandb.Image(str(codebook_path)),
                            "epoch": epoch,
                        },
                        step=iteration + 1,
                    )
        if completed_epoch:
            # Start the next epoch after checkpoint/sample logging so those I/O
            # costs are not incorrectly charged to the following epoch.
            epoch_started_at = time.perf_counter()
    print(f"Done. Best validation loss {best:.4f}; checkpoint {output / 'best.pt'}")
    if wandb_run is not None:
        wandb_run.summary["best_validation_loss"] = best
        wandb_run.finish()


# ============================================================================
# KEPT FROM ORIGINAL DAC: argbind CLI and scope
# Original wraps train in audiotools Accelerator; EEG calls the annotated train.
# ============================================================================


if __name__ == "__main__":
    parsed = argbind.parse_args()
    with argbind.scope(parsed):
        train(parsed)
