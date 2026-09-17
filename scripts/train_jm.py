"""
# Run this script with: 

CUDA_VISIBLE_DEVICES=4 python scripts/train_jm.py --args.load conf/base_jm_eeg.yml



CUDA_VISIBLE_DEVICES=4 python scripts/train_jm.py \
   --args.load conf/base_jm_eeg.yml \
   --batch_size 1 \
   --val_batch_size 1 \
   --val_batch_size 1 \
   --num_workers 0

CUDA_VISIBLE_DEVICES=5 python scripts/train_jm.py \
   --args.load conf/base_jm_audio.yml \
   --batch_size 1 \
   --val_batch_size 1 \
   --val_batch_size 1 \
   --num_workers 0


# VENV installation instructions (it's mostly the DAC requirements file, but with fixed versioning for some packages)
  python3.10 -m venv /data/groups/bci/jonas/venv_dac_audio                                                                                                                                                           
  source /data/groups/bci/jonas/venv_dac_audio/bin/activate                                                                                                                                                          
                                                                                                                                                                                                                     
  python -m pip install --upgrade pip setuptools wheel

  python -m pip install \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

  python -m pip install \
    numpy==1.24.4 \
    numba==0.57.1 \
    argbind==0.3.9 \
    einops tqdm \
    tensorboard==2.13.0 \
    protobuf==3.19.6

  # Install local audiotools fork
  python -m pip install -e \
    /data/groups/bci/jonas/workspace/bci_audiotools

  # Install local DAC fork
  python -m pip install -e \
    /data/groups/bci/jonas/workspace/bci_descript-audio-codec \
    --no-deps

    # NEED TO ADD PIP INSTALL 
    # MNE
    # wandb

"""

import os
import shutil
from contextlib import ExitStack
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets_eeg import AudioDataset
from audiotools.data.datasets_eeg import AudioLoader
from audiotools.data.datasets_eeg import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

import dac


# ADDED FOR EEG EXPERIMENT TRACKING: optional Weights & Biases configuration.
@argbind.bind()
def WandB(
    enabled: bool = False,
    project: str = "eeg-dac",
    entity: str = "",
    name: str = "eeg-alice",
    group: str = "",
    tags: list = ["eeg", "dac"],
    mode: str = "online",
    log_freq: int = 1,
    log_reconstruction: bool = True,
    stft_window_length: int = 256,
    watch_model: bool = False,
) -> dict:
    if log_freq < 1:
        raise ValueError("WandB.log_freq must be at least 1")
    if stft_window_length < 4:
        raise ValueError("WandB.stft_window_length must be at least 4")
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
        "stft_window_length": stft_window_length,
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
            "`python -m pip install wandb`, then `wandb login`."
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
    # This trainer is iteration-based; validation uses the same step axis.
    run.define_metric("training_step")
    for namespace in ("train/*", "val/*"):
        run.define_metric(namespace, step_metric="training_step")
    run.define_metric("reconstruction_step")
    run.define_metric("reconstruction/*", step_metric="reconstruction_step", summary="none")
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


warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DAC = argbind.bind(dac.model.DAC)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, g_extra = DAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    if False: #jm
        original = batch["signal"].clone()
        normalized = batch["signal"].clone()
    with torch.no_grad():
        if False: #jm
            with state.train_data.transform.filter("preprocess"):
                normalized = state.train_data.transform(
                    normalized, **batch["transform_args"]
                )
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
    if False: #jm
        import matplotlib.pyplot as plt
        import numpy as np

        transform_name = "Identity"
        for stage in state.train_data.transform.transforms:
            for transform in stage.transforms:
                if transform.__class__.__name__ != "Identity":
                    transform_name = transform.__class__.__name__
        plot_dir = Path("runs/transform_plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        def plot_original_vs_normalized_vs_transformed(
            original, normalized, transformed, path
        ):
            original_data = original.audio_data[0, 0].detach().cpu().numpy()
            normalized_data = normalized.audio_data[0, 0].detach().cpu().numpy()
            transformed_data = transformed.audio_data[0, 0].detach().cpu().numpy()
            original_time = np.arange(len(original_data)) / original.sample_rate
            normalized_time = np.arange(len(normalized_data)) / normalized.sample_rate
            transformed_time = np.arange(len(transformed_data)) / transformed.sample_rate

            plt.figure(figsize=(15, 5))
            plt.plot(original_time, original_data, label="Original", color="green", alpha=0.7)
            plt.plot(normalized_time, normalized_data, label="Normalized", color="tab:blue", alpha=0.7)
            plt.plot(transformed_time, transformed_data, label="Transformed", color="tab:orange", alpha=0.7)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()


        plot_original_vs_normalized_vs_transformed(
            original,
            normalized,
            signal,
            plot_dir / f"{transform_name}.png",
        )
        import pdb; pdb.set_trace()

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


def reconstruction_figures(target, reconstruction, sample_rate, window_length, title):
    """Plot full waveforms and linear-frequency STFTs with a shared dB scale."""
    from matplotlib.figure import Figure

    target = target.detach().float().cpu().flatten()
    reconstruction = reconstruction.detach().float().cpu().flatten()
    time = torch.arange(target.numel()).numpy() / sample_rate
    waveform = Figure(figsize=(12, 3), layout="constrained")
    ax = waveform.subplots()
    ax.plot(time, target.numpy(), color="#0072B2", linewidth=0.6, label="Target")
    ax.plot(time, reconstruction.numpy(), color="#D55E00", linewidth=0.6,
            alpha=0.85, label="Reconstruction")
    ax.set(xlabel="Time (s)", ylabel="Amplitude (model input units)", title=title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    # Constant padding also supports clips shorter than the FFT window.
    hop = max(1, window_length // 4)
    window = torch.hann_window(window_length)
    signals = torch.stack([target, reconstruction])
    magnitude = torch.stft(
        signals, n_fft=window_length, hop_length=hop, window=window,
        center=True, pad_mode="constant", return_complex=True,
    ).abs() / window.sum()
    db = 20 * magnitude.clamp_min(1e-10).log10()
    vmax = float(db.max())
    vmin = vmax - 80.0
    frequencies = torch.fft.rfftfreq(window_length, d=1.0 / sample_rate).numpy()
    times = torch.arange(db.shape[-1]).numpy() * hop / sample_rate
    spectrogram = Figure(figsize=(12, 4), layout="constrained")
    axes = spectrogram.subplots(1, 2, sharex=True, sharey=True)
    for ax, values, label in zip(axes, db, ("Target", "Reconstruction")):
        mesh = ax.pcolormesh(times, frequencies, values.numpy(), shading="auto",
                             cmap="magma", vmin=vmin, vmax=vmax)
        ax.set(title=label, xlabel="Time (s)", xlim=(0, target.numel() / sample_rate),
               ylim=(0, sample_rate / 2))
    axes[0].set_ylabel("Frequency (Hz)")
    spectrogram.colorbar(mesh, ax=list(axes), label="STFT magnitude (dB re 1 input unit)")
    spectrogram.suptitle(f"{title} — STFT ({window_length} samples, hop {hop})")
    return waveform, spectrogram


def reconstruction_due(completed_steps, save_iters, sample_freq, last_iter):
    """Use completed iteration counts, including num_iters on the last update."""
    return (completed_steps == 1 or completed_steps in save_iters or last_iter
            or (sample_freq > 0 and completed_steps % sample_freq == 0))


@torch.no_grad()
def save_samples(state, val_idx, writer, wandb=None, wandb_run=None,
                 save_path=None, stft_window_length=256):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )

    if wandb_run is not None:
        completed_steps = state.tracker.step + 1
        plots = {"training_step": state.tracker.step,
                 "reconstruction_step": completed_steps}
        output = Path(save_path) / "plots" / f"step_{completed_steps:06d}"
        output.mkdir(parents=True, exist_ok=True)
        for nb, idx in enumerate(val_idx):
            figures = reconstruction_figures(
                signal.audio_data[nb, 0], recons.audio_data[nb, 0],
                signal.sample_rate, stft_window_length,
                f"Sample {idx} · iteration {completed_steps}",
            )
            for kind, figure in zip(("waveform", "stft"), figures):
                try:
                    path = output / f"sample_{idx}_{kind}.png"
                    figure.savefig(path, dpi=160)
                    plots[f"reconstruction/{kind}_{idx}"] = wandb.Image(
                        str(path), caption=f"Sample {idx}, iteration {completed_steps}"
                    )
                    if writer is not None:
                        writer.add_figure(f"reconstruction/{kind}_{idx}", figure,
                                          global_step=completed_steps, close=False)
                finally:
                    figure.clear()
        wandb_run.log(plots)


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    wandb_config = WandB()
    # Use the global rank so multi-process launches create only one W&B run.
    is_primary = int(os.environ.get("RANK", accel.local_rank)) == 0
    with ExitStack() as stack:
        if writer is not None:
            stack.callback(writer.close)
        wandb, wandb_run = None, None
        if is_primary:
            save_run_configuration(args, Path(save_path))
            wandb, wandb_run = initialize_wandb(
                wandb_config, args, Path(save_path), accel.unwrap(state.generator)
            )
            if wandb_run is not None:
                stack.enter_context(wandb_run)

        with tracker.live:
            for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
                output = train_loop(state, batch, accel, lambdas)

                last_iter = (
                    tracker.step == num_iters - 1 if num_iters is not None else False
                )
                if wandb_run is not None and (
                    tracker.step % wandb_config["log_freq"] == 0 or last_iter
                ):
                    wandb_run.log({
                        "training_step": tracker.step,
                        **{f"train/{key}": value for key, value in output.items()
                           if isinstance(value, (int, float))},
                    })
                if reconstruction_due(tracker.step + 1, save_iters, sample_freq, last_iter):
                    save_samples(
                        state, val_idx, writer, wandb,
                        wandb_run if wandb_config["log_reconstruction"] else None,
                        save_path, wandb_config["stft_window_length"],
                    )

                if tracker.step % valid_freq == 0 or last_iter:
                    validate(state, val_dataloader, accel)
                    if wandb_run is not None:
                        # Tracker stores the full validation pass means, whereas
                        # validate() returns only the final batch's output.
                        metrics = {
                            key: values[-1]
                            for key, values in tracker.history["val"].items()
                            if key != "step"
                        }
                        wandb_run.log({
                            "training_step": tracker.step,
                            **{f"val/{key}": value for key, value in metrics.items()},
                        })
                    checkpoint(state, save_iters, save_path)
                    tracker.done("val", f"Iteration {tracker.step}")

                if last_iter:
                    break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
