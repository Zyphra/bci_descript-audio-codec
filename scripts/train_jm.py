"""Train the EEG DAC codec (jm variant; synced with train_cw.py on 2026-10-09,
the pre-sync version is kept as train_jm_10_09.py).

Run directly on a reserved GPU:
    CUDA_VISIBLE_DEVICES=7 NUMPY_MADVISE_HUGEPAGE=0 python scripts/train_jm.py \
        --args.load conf/base_jm_eeg.yml

or via slurm:
    sbatch scripts/train_jm_slurm.sh conf/base_jm_eeg.yml
"""

import os
import math
import json
import shutil
import signal
import threading
from contextlib import ExitStack, contextmanager
import sys
import warnings
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Optional

import argbind
import torch
import numpy as np
import torch.nn.functional as F
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
from dac.utils.provenance import create_launch_record, link_wandb_launch
from dac.utils.wandb_resume import prepare_wandb_run
from dac.utils.codebook_charts import wandb_summary_charts
from dac.utils.gradient_diagnostics import (
    GENERATOR_CLIP_NORM,
    GradientDiagnostics as GradientMonitor,
    GradientDiagnosticLogger,
    component_losses,
    define_wandb_metrics as define_gradient_metrics,
    log_training_metrics,
    validate_schedule,
)


# ADDED FOR EEG EXPERIMENT TRACKING: optional Weights & Biases configuration.
@argbind.bind()
def WandB(
    enabled: bool = False,
    project: str = "eeg-dac",
    entity: str = "",
    name: str = "eeg-alice",
    group: str = "",
    notes: str = "",
    tags: list = ["eeg", "dac"],
    mode: str = "online",
    log_freq: int = 1,
    log_reconstruction: bool = True,
    stft_window_length: int = 256,
    watch_model: bool = False,
    interrupt_timeout: float = 30.0,
    codebook_chart_presets: dict = {},
) -> dict:
    if log_freq < 1:
        raise ValueError("WandB.log_freq must be at least 1")
    if stft_window_length < 4:
        raise ValueError("WandB.stft_window_length must be at least 4")
    if not math.isfinite(interrupt_timeout) or interrupt_timeout <= 0:
        raise ValueError("WandB.interrupt_timeout must be finite and positive")
    if codebook_chart_presets and (
        set(codebook_chart_presets) != {"linear", "log"}
        or any(not isinstance(value, str) or not value for value in codebook_chart_presets.values())
    ):
        raise ValueError("WandB.codebook_chart_presets needs both linear and log preset IDs")
    return {
        "enabled": enabled,
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "notes": notes,
        "tags": tags,
        "mode": mode,
        "log_freq": log_freq,
        "log_reconstruction": log_reconstruction,
        "stft_window_length": stft_window_length,
        "watch_model": watch_model,
        "interrupt_timeout": interrupt_timeout,
        "codebook_chart_presets": dict(codebook_chart_presets),
    }


@argbind.bind()
def GradientDiagnostics(
    enabled: bool = False,
    every_steps: int = 1000,
    early_steps: list = [1, 2, 5, 10],
) -> dict:
    validate_schedule(every_steps, early_steps)
    return {"enabled": enabled, "every_steps": every_steps, "early_steps": list(early_steps)}


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
    if config["notes"]:
        init_kwargs["notes"] = config["notes"]
    if config["mode"] == "online":
        init_kwargs["id"] = prepare_wandb_run(
            output, project=config["project"], entity=config["entity"],
            resume=args.get("resume", False),
        )
        init_kwargs["resume"] = "allow"
        print(f"W&B run ID: {init_kwargs['id']} (resume=allow)", flush=True)
    run = wandb.init(**init_kwargs)
    # This trainer is iteration-based; validation uses the same step axis.
    run.define_metric("training_step")
    if args.get("GradientDiagnostics.enabled", False):
        define_gradient_metrics(run)
    for namespace in ("train/*", "val/*", "train_codebooks/*", "val_codebooks/*"):
        run.define_metric(namespace, step_metric="training_step")
    # Native mean curves support run overlays without custom chart presets.
    for split in ("train", "val"):
        for metric in ("utilization", "perplexity", "dominant_fraction"):
            for statistic in ("mean", "sem"):
                name = f"mean_{metric}" if statistic == "mean" else f"{metric}_sem"
                run.define_metric(f"{split}_codebooks/{name}",
                                  step_metric="training_step", hidden=statistic != "mean")
        run.define_metric(f"{split}_codebooks/observed_layers",
                          step_metric="training_step", hidden=True)
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


def finish_interrupted_wandb(wandb, run, timeout):
    """Bound both run finalization and service teardown on older W&B SDKs.

    A hard exit is reserved for an already interrupted process: raising from
    finish alone can leave W&B's atexit hook waiting on the same stuck service.
    """
    run_dir = str(Path(run.dir).parent)
    recovery = f"Local W&B data: {run_dir}\nRecover with: wandb sync {run_dir}\n"

    def force_exit(reason):
        try:
            os.write(2, (f"\n{reason}; exiting with code 130.\n" + recovery).encode())
        finally:
            os._exit(130)

    previous_handler = signal.signal(
        signal.SIGINT, lambda *_: force_exit("Second Ctrl-C during W&B shutdown")
    )
    watchdog = threading.Timer(
        timeout, force_exit, args=(f"W&B shutdown exceeded {timeout:g} seconds",)
    )
    watchdog.daemon = True
    watchdog.start()
    try:
        print(
            f"\nTraining interrupted. Allowing W&B {timeout:g} seconds to sync; "
            "press Ctrl-C again to exit immediately.\n" + recovery,
            file=sys.stderr, flush=True,
        )
        run.finish(exit_code=130)
        # Include service shutdown in the deadline, rather than leaving it to
        # an unbounded atexit callback after run.finish() returns.
        wandb.teardown(exit_code=130)
    except BaseException:
        force_exit("W&B interrupted-run cleanup failed")
    finally:
        watchdog.cancel()
        signal.signal(signal.SIGINT, previous_handler)


@contextmanager
def managed_wandb_run(wandb, run, interrupt_timeout):
    try:
        yield run
    except KeyboardInterrupt:
        finish_interrupted_wandb(wandb, run, interrupt_timeout)
        raise
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        try:
            run.finish(exit_code=0)
        except KeyboardInterrupt:
            # Ctrl-C may arrive during final upload after training completes.
            finish_interrupted_wandb(wandb, run, interrupt_timeout)
            raise


def save_run_configuration(args, output: Path) -> Path:
    """Save immutable per-launch configs/code, plus the legacy latest configs."""
    launch = create_launch_record(output, {
        "dac": Path(dac.__file__).resolve().parent,
        "audiotools": Path(sys.modules["audiotools"].__file__).resolve().parent,
    })
    resolved = dict(args)
    argbind.dump_args(resolved, launch / "config_resolved.yml")
    shutil.copy2(launch / "config_resolved.yml", output / "config_resolved.yml")
    source_name = resolved.get("args.load")
    if source_name:
        source = Path(source_name).expanduser().resolve()
        if source.is_file():
            shutil.copy2(source, launch / "config_input.yml")
            shutil.copy2(launch / "config_input.yml", output / "config_input.yml")
    return launch


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
# EEG variants of the STFT-domain band transforms (FrequencyMask / FrequencyNoise).
# Three fixes in one (all verified 2026-10-19):
#  1. The stock transforms only edit `stft_data` (mask_frequencies documents "apply
#     .istft() to get the masked audio") and never call istft -> the WAVEFORM the
#     model trains on is untouched: a silent no-op. These variants istft back.
#  2. audiotools gives 256 Hz signals a 16-sample STFT (16 Hz bins); a few-Hz band
#     at 50/60 Hz would contain no bin. These variants use a 256-sample STFT
#     (1 Hz bins) and restore the signal's own params afterwards.
#  3. argbind configures transforms per CLASS, so two notches at different line
#     frequencies need two class names, each with its OWN __init__ (a subclass
#     that merely inherits __init__ silently reads the parent's args).
from audiotools import STFTParams
_EEG_STFT = STFTParams(window_length=256, hop_length=64, window_type="hann")


def _eeg_band_class(name, base):
    def __init__(self, f_center: tuple = ("uniform", 0.0, 1.0),
                 f_width: tuple = ("const", 0.1), name: str = None, prob: float = 1.0):
        base.__init__(self, f_center=f_center, f_width=f_width, name=name, prob=prob)

    def transform(self, signal, **kwargs):
        # SpectralTransform.transform does stft -> _transform -> istft with the
        # signal's own params; swap in the EEG params for the whole round trip.
        old = signal.stft_params
        signal.stft_params = _EEG_STFT
        signal.stft_data = None
        try:
            return base.transform(self, signal, **kwargs)
        finally:
            signal.stft_params = old
            signal.stft_data = None

    def _transform(self, signal, **kwargs):
        signal.stft_params = _EEG_STFT               # the masked slice must match
        return base._transform(self, signal, **kwargs)

    cls = type(name, (base,), {"__init__": __init__, "transform": transform, "_transform": _transform})
    cls.__init__.__qualname__ = f"{name}.__init__"
    return cls


transforms.FrequencyMask50 = _eeg_band_class("FrequencyMask50", transforms.FrequencyMask)
transforms.FrequencyMask60 = _eeg_band_class("FrequencyMask60", transforms.FrequencyMask)
transforms.FrequencyNoise60 = _eeg_band_class("FrequencyNoise60", transforms.FrequencyNoise)


class LineNoise(transforms.BaseTransform):
    """ADD mains hum: a sinusoid at `freq_hz` with random phase, at `level_db`
    relative to the signal's RMS (0 dB = hum as loud as the EEG). Time-domain,
    so it works at any sample rate. (audiotools' FrequencyNoise *replaces* a band
    with unit-scale STFT noise, which is far below real hum levels.)"""

    def __init__(self, level_db: tuple = ("uniform", -30.0, -10.0), freq_hz: float = 60.0,
                 freq_jitter_hz: float = 0.0, name: str = None, prob: float = 1.0):
        super().__init__(name=name, prob=prob)
        self.level_db = level_db
        self.freq_hz = freq_hz
        self.freq_jitter_hz = freq_jitter_hz   # jm 2026-10-20: hum frequency drawn from freq_hz +- jitter per window

    def _instantiate(self, state):
        return {"level_db": util.sample_from_dist(self.level_db, state),
                "phase": float(state.uniform(0.0, 2 * np.pi)),
                "freq_hz": float(self.freq_hz + state.uniform(-self.freq_jitter_hz, self.freq_jitter_hz))}

    def _transform(self, signal, level_db, phase, freq_hz=None):
        x = signal.audio_data                                        # [B, C, T]
        t = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) / signal.sample_rate
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        level_db = torch.as_tensor(level_db, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        phase = torch.as_tensor(phase, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        freq = (torch.as_tensor(freq_hz, device=x.device, dtype=x.dtype).view(-1, 1, 1)
                if freq_hz is not None else torch.tensor(self.freq_hz, device=x.device, dtype=x.dtype))
        amp = rms * (2 ** 0.5) * 10 ** (level_db / 20)
        signal.audio_data = x + amp * torch.sin(2 * np.pi * freq * t + phase)
        return signal


class GaussianNoise(transforms.BaseTransform):
    """ADD white gaussian noise at a given SNR (dB) relative to each window's power
    (0 dB = noise as strong as the EEG). Defined by SNR so that a training level is
    directly comparable with the stress-test levels of code_consistency_jm (30 .. 0 dB);
    audiotools' NoiseFloor is specified in loudness dB and its SNR depends on the window.
    (jm, 2026-10-20, noise sweep)"""

    def __init__(self, snr_db: tuple = ("uniform", 10.0, 30.0), name: str = None, prob: float = 1.0):
        super().__init__(name=name, prob=prob)
        self.snr_db = snr_db

    def _instantiate(self, state, signal=None):
        return {"snr_db": util.sample_from_dist(self.snr_db, state)}

    def _transform(self, signal, snr_db):
        x = signal.audio_data                                        # [B, C, T]
        power = x.pow(2).mean(-1, keepdim=True)
        snr_db = torch.as_tensor(snr_db, device=x.device, dtype=x.dtype).view(-1, 1, 1)
        noise = torch.randn_like(x) * (power * 10 ** (-snr_db / 10)).sqrt()
        signal.audio_data = x + noise
        return signal


transforms.GaussianNoise = GaussianNoise
transforms.LineNoise60 = type("LineNoise60", (LineNoise,), {})
transforms.LineNoise60.__init__ = LineNoise.__init__  # own __init__ for argbind (see above)


CONSISTENCY_TAU = 0.1   # softmax temperature over code cosine similarities


@torch.no_grad()
def _clean_branch(model, target):
    """Teacher: encode the CLEAN target (no gradient) -> its codes and pre-quantization
    latents, the same quantities `out` carries for the corrupted input."""
    z, codes, latents, _, _ = model.encode(target.audio_data)
    return codes, latents


def consistency_losses(model, out, target, corrupted=None):
    """Key-consistency objective (jm, 2026-10-19): make the corrupted input's tokens
    match the clean input's tokens, not just its reconstruction.
      consistency/code_loss   cross-entropy of the corrupted frame's soft code assignment
                              (softmax of cosine-to-codebook / tau) against the CLEAN key
      consistency/latent_loss MSE between corrupted and clean pre-quantization latents
      consistency/usage_loss  KL(mean soft assignment over the batch || uniform)
                              = log V - H(mean softmax): keeps the codebook in use.
    Both teacher terms use the clean branch as a stop-gradient teacher and are averaged
    over the CORRUPTED windows only (`corrupted`: [B] bool, input != target). Without the
    usage term and the mask the CE collapses every frame onto one code (sweep-5
    consist_lo/hi/code: codebook usage 1e-2..1e-4, val stft 0.47..0.80)."""
    codes_clean, lat_clean = _clean_branch(model, target)
    lat = out["latents"]                              # [B, Q*D, T], with gradient
    B = lat.shape[0]
    if corrupted is None:
        corrupted = torch.ones(B, dtype=torch.bool, device=lat.device)
    w = corrupted.float() / corrupted.float().sum().clamp(min=1.0)   # [B], sums to 1 (or 0)
    latent_loss = (F.mse_loss(lat, lat_clean, reduction="none").mean((1, 2)) * w).sum()
    code_loss, usage_loss, D = 0.0, 0.0, model.quantizer.quantizers[0].codebook_dim
    for q, vq in enumerate(model.quantizer.quantizers[: out["codes"].shape[1]]):
        e = F.normalize(lat[:, q * D:(q + 1) * D], dim=1)             # [B, D, T]
        c = F.normalize(vq.codebook.weight, dim=1)                   # [V, D]
        logits = torch.einsum("bdt,vd->btv", e, c) / CONSISTENCY_TAU  # cosine / tau, [B, T, V]
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V), codes_clean[:, q].reshape(-1),
                             reduction="none").view(B, -1).mean(1)   # [B]
        code_loss = code_loss + (ce * w).sum()
        p_mean = logits.float().softmax(-1).reshape(-1, V).mean(0)   # marginal assignment, all windows
        usage_loss = usage_loss + math.log(V) + (p_mean * (p_mean + 1e-12).log()).sum()
        with torch.no_grad():
            same = (out["codes"][:, q] == codes_clean[:, q]).float().mean()
    return {"consistency/code_loss": code_loss, "consistency/latent_loss": latent_loss,
            "consistency/usage_loss": usage_loss, "consistency/key_agreement": same,
            "consistency/corrupted_frac": corrupted.float().mean()}
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
    corrupt_prob: float = 1.0,
    corrupt_choose: bool = False,  # True: pick exactly one corruption per window
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    corrupt: list = ["Identity"], #JM: augment that is only applied to the input, not the target
    postprocess: list = ["Identity"],
):
    """preprocess -> augment -> corrupt -> postprocess.

    All stages feed the model INPUT; the loss TARGET is the same chain with
    `corrupt` skipped (see train_loop). So: preprocess/postprocess = always-on
    conditioning on both arms (e.g. EEGNormalize), augment = random variation the
    codec must REPRESENT (both arms), corrupt = input-only damage the codec must
    REMOVE (invariance: noise, line hum, clipping).
    """
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    # Compose: every listed corruption fires independently at its own .prob;
    # Choose: exactly one of them is picked per window.
    corrupt_cls = transforms.Choose if corrupt_choose else transforms.Compose
    corrupt = corrupt_cls(*to_tfm(corrupt), name="corrupt", prob=corrupt_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, corrupt, postprocess)
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

        # print("Inside build_dataset")
        # import ipdb; ipdb.set_trace()

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    use_gan: bool
    discriminator: Optional[Discriminator]
    optimizer_d: Optional[AdamW]
    scheduler_d: Optional[ExponentialLR]

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: Optional[losses.GANLoss]
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
    use_gan: bool = True,
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
        # A reused output folder may still contain an older discriminator.
        checkpoint_used_gan = getattr(generator, "metadata", {}).get("use_gan", True)
        if use_gan and checkpoint_used_gan and (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DAC() if generator is None else generator
    if use_gan:
        discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(f"GAN training: {'enabled' if use_gan else 'disabled (both adversarial weights are zero)'}")
    if use_gan:
        tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    if use_gan:
        discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    optimizer_d, scheduler_d = None, None
    if use_gan:
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
    gan_loss = losses.GANLoss(discriminator) if use_gan else None

    return State(
        use_gan=use_gan,
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
def val_loop(batch, state, accel, codebook_counts=None):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    with state.val_data.transform.filter("preprocess", "augment", "postprocess"):
        target = state.val_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    if codebook_counts is not None:
        accumulate_codebook_counts(codebook_counts, out["codes"])

    stft_loss, stft_components = state.stft_loss(recons, target, return_components=True)
    waveform_loss = state.waveform_loss(recons, target)
    return {
        "loss": waveform_loss + stft_loss,
        "mel/loss": state.mel_loss(recons, target),
        "stft/loss": stft_loss,
        **{f"stft/{name}_loss": value for name, value in stft_components.items()},
        "waveform/loss": waveform_loss,
    }


@timer()
def train_loop(state, batch, accel, lambdas, codebook_counts=None, gradient_diagnostics=None):
    state.generator.train()
    if state.use_gan:
        state.discriminator.train()
    output = {}

    # print("Inside train_loop. What is batch?")
    # import pdb; state.tracker.live.stop(); pdb.set_trace()

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        # INPUT = full transform chain; TARGET = the same chain with the `corrupt`
        # stage skipped. Transforms under `corrupt` therefore damage only the input,
        # training the codec to be INVARIANT to them (noise, line hum, clipping);
        # transforms under `augment` land in input AND target, so the codec must
        # REPRESENT them (filter diversity); `postprocess` (e.g. EEGNormalize) runs
        # on both arms AFTER corruption, so inputs always arrive in canonical scale.
        # Identical transform_args -> shared stages draw identical random parameters
        # in both calls. With corrupt=[Identity], target == signal (plain DAC).
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        with state.train_data.transform.filter("preprocess", "augment", "postprocess"): #JM: corrupt is skipped for the target
            target = state.train_data.transform(
                batch["signal"].clone(), **batch["transform_args"]
            )

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    if codebook_counts is not None:
        batch_codebook_counts = torch.zeros_like(codebook_counts)
    else:
        model = accel.unwrap(state.generator)
        batch_codebook_counts = torch.zeros(
            model.n_codebooks, model.codebook_size, dtype=torch.long, device=out["codes"].device
        )
    accumulate_codebook_counts(batch_codebook_counts, out["codes"], out["codebook_mask"])
    if codebook_counts is not None:
        # Keep interval counts local: they are reduced separately at validation time.
        codebook_counts.add_(batch_codebook_counts)


    if False:
        print(f"Inside train_loop w/ {state.generator.quantizer_dropout=:.2f}. What is the codebook utilization?")
        import ipdb; state.tracker.live.stop(); ipdb.set_trace()


    diagnostic_due = gradient_diagnostics is not None and gradient_diagnostics.due(state.tracker.step)
    discriminator_sample = None
    if state.use_gan:
        with accel.autocast():
            output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, target)

        state.optimizer_d.zero_grad()
        accel.backward(output["adv/disc_loss"])
        accel.scaler.unscale_(state.optimizer_d)
        if diagnostic_due:
            discriminator_sample = gradient_diagnostics.measure_discriminator(state.discriminator)
        output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
            state.discriminator.parameters(), 10.0
        )
        accel.step(state.optimizer_d)
        state.scheduler_d.step()

    with accel.autocast():
        output["stft/loss"], stft_components = state.stft_loss(
            recons, target, return_components=True
        )
        output.update({f"stft/{name}_loss": value for name, value in stft_components.items()})
        output["mel/loss"] = state.mel_loss(recons, target)
        output["waveform/loss"] = state.waveform_loss(recons, target)
        if state.use_gan:
            (
                output["adv/gen_loss"],
                output["adv/feat_loss"],
            ) = state.gan_loss.generator_loss(recons, target)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        if any(lambdas.get(k, 0) for k in ("consistency/code_loss", "consistency/latent_loss", "consistency/usage_loss")):
            corrupted = (signal.audio_data != target.audio_data).flatten(1).any(1)   # [B]: did a corruption fire?
            output.update(consistency_losses(accel.unwrap(state.generator), out, target, corrupted))
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

    state.optimizer_g.zero_grad()
    gradient_sample = None
    if diagnostic_due:
        gradient_sample = gradient_diagnostics.measure_components(
            component_losses(output, lambdas, state.stft_loss)
        )
        if discriminator_sample is not None:
            gradient_sample["diagnostic_seconds"] += discriminator_sample.pop("diagnostic_seconds")
            gradient_sample.update(discriminator_sample)
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    if gradient_sample is not None:
        gradient_diagnostics.measure_total(gradient_sample)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), GENERATOR_CLIP_NORM
    )
    if gradient_sample is not None:
        # Tracker ignores nested dictionaries, so sampled metrics never become stale.
        output["_gradient_diagnostics"] = gradient_diagnostics.finish(
            gradient_sample, output["other/grad_norm"]
        )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size
    # Pool assignments across ranks, rather than averaging nonlinear usage scores.
    reduce_codebook_counts(batch_codebook_counts)
    output["codebook_usage_score"] = codebook_usage_score(batch_codebook_counts)

    return {k: v for k, v in sorted(output.items())}


@timer()
def train_step(state, dataloader, accel, lambdas, **kwargs):
    """Time batch retrieval and training, including completion of CUDA work."""
    batch = next(dataloader)
    output = train_loop(state, batch, accel, lambdas, **kwargs)
    if torch.device(accel.device).type == "cuda":
        torch.cuda.synchronize(accel.device)
    return output


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history, "use_gan": state.use_gan}
    if getattr(state, "launch_id", None) is not None:
        metadata["launch_id"] = state.launch_id

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "loss"): # no longer mel/loss (CW).
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
        if state.use_gan:
            discriminator_extra = {
                "optimizer.pth": state.optimizer_d.state_dict(),
                "scheduler.pth": state.scheduler_d.state_dict(),
            }
            accel.unwrap(state.discriminator).save_to_folder(
                f"{save_path}/{tag}", discriminator_extra
            )


def reconstruction_figures(target, reconstruction, sample_rate, window_length, title,
                           stft_mae=None, phase_circ_err=None):
    """Plot waveforms, STFT magnitudes, and phases with shared comparison scales."""
    from matplotlib.figure import Figure

    target = target.detach().float().cpu().flatten()
    reconstruction = reconstruction.detach().float().cpu().flatten()
    time = torch.arange(target.numel()).numpy() / sample_rate
    waveform = Figure(figsize=(12, 3), layout="constrained")
    ax = waveform.subplots()
    ax.plot(time, target.numpy(), color="#0072B2", linewidth=0.6, label="Target")
    ax.plot(time, reconstruction.numpy(), color="#D55E00", linewidth=0.6,
            alpha=0.85, label="Reconstruction")
    waveform_mae = float((target - reconstruction).abs().mean())
    ax.set(xlabel="Time (s)", ylabel="Amplitude (model input units)",
           title=f"{title} (MAE={waveform_mae:.4g})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    # Constant padding also supports clips shorter than the FFT window.
    hop = max(1, window_length // 4)
    window = torch.hann_window(window_length)
    signals = torch.stack([target, reconstruction])
    spectrum = torch.stft(
        signals, n_fft=window_length, hop_length=hop, window=window,
        center=True, pad_mode="constant", return_complex=True,
    )
    magnitude = spectrum.abs() / window.sum()
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
    stft_title = f"{title} — STFT ({window_length} samples, hop {hop})"
    if stft_mae is not None:
        stft_title += f"\nMAE_(mag,log)={stft_mae[0]:.4g},{stft_mae[1]:.4g}"
    spectrogram.suptitle(stft_title)

    phase = spectrum.angle()
    phase_figure = Figure(figsize=(12, 4), layout="constrained")
    phase_axes = phase_figure.subplots(1, 2, sharex=True, sharey=True)
    for ax, values, label in zip(phase_axes, phase, ("Target", "Reconstruction")):
        mesh = ax.pcolormesh(times, frequencies, values.numpy(), shading="auto",
                             cmap="twilight", vmin=-math.pi, vmax=math.pi)
        ax.set(title=label, xlabel="Time (s)", xlim=(0, target.numel() / sample_rate),
               ylim=(0, sample_rate / 2))
    phase_axes[0].set_ylabel("Frequency (Hz)")
    colorbar = phase_figure.colorbar(mesh, ax=list(phase_axes), label="STFT phase (radians)")
    colorbar.set_ticks([-math.pi, 0, math.pi], labels=["−π", "0", "π"])
    if phase_circ_err is None:
        phase_circ_err = float((1 - torch.cos(phase[1] - phase[0])).mean())
        error_scope = "displayed scale"
    else:
        error_scope = "sum across loss scales"
    phase_figure.suptitle(
        f"{title} — STFT phase ({window_length} samples, hop {hop})"
        f"\ncirc_err = {phase_circ_err:.4g} ({error_scope})"
    )
    return waveform, spectrogram, phase_figure


def reconstruction_due(completed_steps, save_iters, sample_freq, last_iter):
    """Use completed iteration counts, including num_iters on the last update."""
    return (completed_steps == 1 or completed_steps in save_iters or last_iter
            or (sample_freq > 0 and completed_steps % sample_freq == 0))


STAGES = ("preprocess", "augment", "corrupt", "postprocess")


@torch.no_grad()
def transform_stage_figure(state, idx, seconds=2.0):
    """One TRAINING window replayed stage by stage: raw -> +preprocess -> +augment
    -> +corrupt -> +postprocess (= model input). Left: waveform (first `seconds`),
    right: log power spectrum. The transform_args carry the random draw, so every
    stage is the exact signal the model saw for that draw (corrupt is what the
    loss target skips)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    batch = state.train_data.collate([state.train_data[idx]])
    batch = util.prepare_batch(batch, accel.device)
    sr = batch["signal"].sample_rate
    stages = [("raw", batch["signal"].audio_data[0, 0].float().cpu().numpy())]
    for k in range(1, len(STAGES) + 1):
        with state.train_data.transform.filter(*STAGES[:k]):
            sig = state.train_data.transform(batch["signal"].clone(), **batch["transform_args"])
        stages.append((f"+{STAGES[k - 1]}", sig.audio_data[0, 0].float().cpu().numpy()))

    # which transforms fired in this draw: audiotools keys the args as
    # "<i>.<stage>" -> {"<j>.<Transform>": {..., "mask"}, "mask": stage gate}
    def _on(d):
        m = d.get("mask") if isinstance(d, dict) else None
        return m is not None and bool(torch.as_tensor(m).flatten()[0])
    fired = {}

    def _walk(d):  # stage dicts sit under the outer Compose; depth may vary
        for key, val in d.items():
            if not isinstance(val, dict):
                continue
            name = key.split(".", 1)[-1]
            if name in STAGES:
                fired[name] = [tk.split(".", 1)[-1] for tk, ta in val.items()
                               if isinstance(ta, dict) and tk.split(".", 1)[-1] != "Identity"
                               and _on(ta)] if _on(val) else []
            else:
                _walk(val)
    _walk(batch["transform_args"])

    n = int(seconds * sr)
    fig, axes = plt.subplots(len(stages), 2, figsize=(14, 2.1 * len(stages)),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    t = np.arange(n) / sr
    for (name, x), (ax_t, ax_f) in zip(stages, axes):
        ax_t.plot(t, x[:n], linewidth=0.6, color="#0072B2")
        label = name
        if name.startswith("+"):
            f = fired.get(name[1:], [])
            label += "\n(" + (", ".join(f) if f else "nothing fired") + ")"
        ax_t.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax_t.set_xlim(0, seconds)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        ax_f.semilogy(f, spec + 1e-12, linewidth=0.6, color="#D55E00")
        ax_f.set_xlim(0, sr / 2)
        for ax in (ax_t, ax_f):
            ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time (s)", fontsize=9)
    axes[-1, 1].set_xlabel("frequency (Hz)", fontsize=9)
    fig.suptitle(f"Transform stages, training sample {idx} (left: waveform, right: power spectrum)",
                 fontsize=11)
    fig.tight_layout()
    return fig


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
            # Match the plotted channel, using the trainer's FFT scales and
            # log transform. Components are unweighted sums across scales.
            _, components = state.stft_loss(
                AudioSignal(recons.audio_data[nb:nb + 1, :1], signal.sample_rate),
                AudioSignal(signal.audio_data[nb:nb + 1, :1], signal.sample_rate),
                return_components=True,
            )
            figures = reconstruction_figures(
                signal.audio_data[nb, 0], recons.audio_data[nb, 0],
                signal.sample_rate, stft_window_length,
                f"Sample {idx} · iteration {completed_steps}",
                stft_mae=(float(components["mag"]), float(components["log_mag"])),
                phase_circ_err=float(components["phase"]),
            )
            for kind, figure in zip(("waveform", "stft", "stft_phase"), figures):
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
        # the augmentation pipeline itself, stage by stage — logged ONCE (first sample
        # step; it does not depend on the model), four windows to show the variety of draws
        for idx in (range(4) if state.tracker.step == 0 else ()):
            figure = transform_stage_figure(state, idx)
            try:
                path = output / f"transform_stages_{idx}.png"
                figure.savefig(path, dpi=130)
                plots[f"transform_stages/sample_{idx}"] = wandb.Image(
                    str(path), caption=f"raw -> preprocess -> augment -> corrupt -> postprocess, "
                                       f"iteration {completed_steps}")
            finally:
                figure.clear()
                import matplotlib.pyplot as plt
                plt.close(figure)
        wandb_run.log(plots)


@torch.no_grad()
def accumulate_codebook_counts(counts, codes, active_mask=None):
    """Accumulate assignments; dropout-inactive stages must not contribute."""
    codes = codes.detach()
    for q in range(codes.shape[1]):
        selected = codes[:, q, :]
        if active_mask is not None:
            selected = selected[active_mask[:, q]]
        counts[q].add_(torch.bincount(selected.reshape(-1), minlength=counts.shape[1]))


def reduce_codebook_counts(counts):
    """Sum histograms across ranks before calculating usage statistics."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)


@torch.no_grad()
def codebook_usage_score(counts):
    """Geometric mean of per-layer perplexity / size; NaN if a layer is unobserved."""
    counts = counts.float()
    totals = counts.sum(dim=1, keepdim=True)
    probabilities = counts / totals.clamp_min(1)
    # Zero-probability entries contribute zero entropy without evaluating log(0).
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(counts.dtype).tiny).log()).sum(dim=1)
    score = (entropy.mean() - math.log(counts.shape[1])).exp().clamp(max=1)
    return torch.where((totals > 0).all(), score, score.new_full((), float("nan")))


def codebook_usage_metrics(counts, split="val"):
    """Compute usage metrics and sample budgets from aggregated assignments."""
    metrics = {}
    for q, row in enumerate(counts.double()):
        total = row.sum().item()
        used = (row > 0).sum().item()
        p = row[row > 0] / total if total else row[:0]
        prefix = f"{split}/codebook_{q}"
        metrics.update({
            f"{prefix}/assignments": int(total),
            f"{prefix}/utilization": used / row.numel(),
            f"{prefix}/perplexity": (-(p * p.log()).sum()).exp().item() if total else 0.0,
            f"{prefix}/dominant_fraction": row.max().item() / total if total else 0.0,
        })
    return metrics


def codebook_layer_summary(counts, metrics, split):
    """Equal-weight layer means and sample SEM, excluding unobserved layers."""
    observed = (counts.sum(dim=1) > 0).nonzero().flatten().tolist()
    summary = {"observed_layers": len(observed)}
    for metric in ("utilization", "perplexity", "dominant_fraction"):
        values = torch.tensor([metrics[f"{split}/codebook_{q}/{metric}"]
                               for q in observed], dtype=torch.float64)
        summary[f"{metric}_mean"] = values.mean().item() if observed else None
        summary[f"{metric}_sem"] = (
            (values.std(unbiased=True) / math.sqrt(len(observed))).item()
            if len(observed) > 1 else None
        )
    return summary


def update_codebook_summary_history(path, step, summary, shape):
    """Persist histories across resume; replace repeated steps and discard future ones."""
    shape = list(shape)
    history = json.loads(path.read_text()) if path.exists() else {}
    rows = history.get("rows", []) if history.get("shape") == shape else []
    rows = [row for row in rows if row["step"] < step]
    rows.append({"step": step, **summary})
    history = {"shape": shape, "rows": rows}
    path.write_text(json.dumps(history, allow_nan=False, indent=2) + "\n")
    return history


def codebook_summary_figures(history, split):
    """Plot the full history of layer means with actual standard-error bars."""
    from matplotlib.figure import Figure

    figures = {}
    size = history["shape"][1]
    rows = history["rows"]
    for metric, title in (("utilization", "Utilization"), ("perplexity", "Perplexity"),
                          ("dominant_fraction", "Dominant-code fraction")):
        figure = Figure(figsize=(7, 4), layout="constrained")
        ax = figure.subplots()
        means = [row[f"{metric}_mean"] for row in rows]
        ax.plot([row["step"] for row in rows],
                [value if value is not None else float("nan") for value in means],
                marker="o", markersize=3, label="Layer mean", color="C0")
        with_sem = [row for row in rows if row[f"{metric}_sem"] is not None]
        if with_sem:
            ax.errorbar([row["step"] for row in with_sem],
                        [row[f"{metric}_mean"] for row in with_sem],
                        yerr=[row[f"{metric}_sem"] for row in with_sem],
                        fmt="none", ecolor="C0", capsize=3, label="±1 SEM across layers")
        ax.set(title=f"{split.capitalize()} {title.lower()} · layer mean ± SEM",
               xlabel="Training step", ylabel=title,
               ylim=(0, size if metric == "perplexity" else 1))
        if metric == "dominant_fraction":
            ax.set_ylim(0.5 / size, 1)
            ax.set_yscale("log")
            ax.axhline(1 / size, color="gray", linestyle="--", label=f"Uniform: 1/{size}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small")
        figures[metric] = figure
    return figures


def log_codebook_usage(counts, step, save_path, writer=None, wandb=None, wandb_run=None,
                       split="val", chart_presets=None):
    """Log usage over a validation pass or recent training interval."""
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    counts = counts.detach().cpu()
    metrics = codebook_usage_metrics(counts, split)
    summary = codebook_layer_summary(counts, metrics, split)
    if split == "val":
        # Training logs this name per batch; do not overwrite it with interval usage.
        metrics["val/codebook_usage_score"] = codebook_usage_score(counts).item()
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 3, gridspec_kw={"height_ratios": [1, 1.3]})
    books = list(range(counts.shape[0]))
    for ax, metric, title in zip(axes[0],
            ("utilization", "perplexity", "dominant_fraction"),
            ("Utilization (fraction)", "Perplexity", "Dominant-code fraction")):
        ax.bar(books, [metrics[f"{split}/codebook_{q}/{metric}"] for q in books])
        ax.set(title=title, xlabel="Codebook", xticks=books,
               ylim=(0, counts.shape[1] if metric == "perplexity" else 1))
        if metric == "dominant_fraction":
            ax.set_ylim(0.5 / counts.shape[1], 1)
            ax.set_yscale("log")
            ax.axhline(1 / counts.shape[1], color="gray", linestyle="--",
                       label=f"Uniform: 1/{counts.shape[1]}")
            ax.legend(fontsize="small")
    for ax in axes[1]:
        ax.remove()
    ax = figure.add_subplot(figure.axes[0].get_subplotspec().get_gridspec()[1, :])
    probabilities = counts.float() / counts.sum(dim=1, keepdim=True).clamp_min(1)
    positive = probabilities[probabilities > 0]
    # LogNorm masks unused codes; keep valid limits even for empty or one-hot counts.
    vmin = min(positive.min().item(), 0.1) if positive.numel() else 0.1
    mesh = ax.imshow(probabilities.numpy(), aspect="auto", origin="lower",
                     interpolation="nearest", norm=LogNorm(vmin=vmin, vmax=1))
    scope = "full validation pass" if split == "val" else "training interval since last log"
    ax.set(title=f"Assignment frequency across the {scope}",
           xlabel="Code ID", ylabel="Codebook", yticks=books)
    figure.colorbar(mesh, ax=ax, label="Fraction of assignments")
    figure.suptitle(f"{split.capitalize()} codebook usage · {step + 1} completed updates")
    output = Path(save_path) / "plots" / f"step_{step + 1:06d}"
    output.mkdir(parents=True, exist_ok=True)
    summary_figures = {}
    try:
        # Preserve existing validation filenames; training gets its own files.
        prefix = "" if split == "val" else f"{split}_"
        path = output / f"{prefix}codebook_usage.png"
        figure.savefig(path, dpi=160)
        # Retain exact counts so metrics can be recomputed without inference.
        torch.save(counts, output / f"{prefix}codebook_counts.pt")
        history = update_codebook_summary_history(
            output.parent / f"{split}_codebook_summary_history.json",
            step, summary, counts.shape,
        )
        summary_figures = codebook_summary_figures(history, split)
        for metric, summary_figure in summary_figures.items():
            summary_path = output / f"{prefix}{metric}_mean_sem.png"
            summary_figure.savefig(summary_path, dpi=160)
            if writer is not None:
                writer.add_figure(f"{split}_codebooks/{metric}", summary_figure,
                                  global_step=step, close=False)
        summary_scalars = {f"{split}_codebooks/{key}": value if value is not None else float("nan")
                           for key, value in summary.items()}
        if writer is not None:
            for key, value in {**metrics, **summary_scalars}.items():
                writer.add_scalar(key, value, step)
            writer.add_figure(f"{split}/codebook_usage", figure, global_step=step, close=False)
        if wandb_run is not None:
            wandb_metrics = {}
            for key, value in metrics.items():
                if key.endswith("/assignments"):
                    continue
                if key.startswith(f"{split}/codebook_") and key.count("/") == 2:
                    book, metric = key.split("/")[1:]
                    key = f"{split}_codebooks/{book}_{metric}"
                wandb_metrics[key] = value
            summary_charts = (
                wandb_summary_charts(history, split, wandb, wandb_run, chart_presets)
                if chart_presets else {}
            )
            # Keep JSON/TensorBoard history compatible; W&B means sort together.
            wandb_summary = {
                (f"{split}_codebooks/mean_{key.rsplit('/', 1)[1][:-5]}"
                 if key.endswith("_mean") else key): value
                for key, value in summary_scalars.items()
            }
            wandb_run.log({"training_step": step, **wandb_metrics,
                           **wandb_summary,
                           **summary_charts,
                           f"{split}/codebook_usage": wandb.Image(str(path))})
    finally:
        figure.clear()
        for summary_figure in summary_figures.values():
            summary_figure.clear()


def validate(state, val_dataloader, accel):
    model = accel.unwrap(state.generator)
    counts = torch.zeros(
        model.n_codebooks, model.codebook_size, dtype=torch.long, device=accel.device
    )
    output = {}
    for batch in val_dataloader:
        output = val_loop(batch, state, accel, codebook_counts=counts)
    # Sum raw counts, never per-batch or per-rank utilization/perplexity.
    reduce_codebook_counts(counts)
    output["codebook_counts"] = counts.cpu()
    # Print one compact table after aggregation, using Tracker's console so it
    # coexists with the live progress display and is also written to log.txt.
    if state.tracker.rank == 0:
        from rich.table import Table

        metrics = codebook_usage_metrics(output["codebook_counts"])
        table = Table(title="Validation codebook usage (full pass)")
        for column in ("Book", "Utilization", "Perplexity", "Dominant code"):
            table.add_column(column, justify="right")
        for q in range(counts.shape[0]):
            prefix = f"val/codebook_{q}"
            table.add_row(
                str(q),
                f"{metrics[f'{prefix}/utilization']:.1%}",
                f"{metrics[f'{prefix}/perplexity']:.1f}",
                f"{metrics[f'{prefix}/dominant_fraction']:.1%}",
            )
        state.tracker.print(table)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
    if state.optimizer_d is not None and hasattr(state.optimizer_d, "consolidate_state_dict"):
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
    allowed_loss_keys = {
        "consistency/code_loss", "consistency/latent_loss", "consistency/usage_loss",   # jm key-consistency terms
        "waveform/loss",
        "mel/loss",
        "stft/loss",
        "stft/mag_loss",
        "stft/log_mag_loss",
        "stft/phase_loss",
        "adv/gen_loss",
        "adv/feat_loss",
        "vq/commitment_loss",
        "vq/codebook_loss",
    }
    # Reject unknown keys even when their weights are zero.
    unknown = set(lambdas) - allowed_loss_keys
    if unknown:
        raise ValueError(
            f"Unknown loss keys: {sorted(unknown)}. "
            f"Keys are case-sensitive. Allowed: {sorted(allowed_loss_keys)}"
        )

    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    use_gan = any(
        lambdas.get(key, 0.0) != 0.0
        for key in ("adv/gen_loss", "adv/feat_loss")
    )
    # Save/log the derived value rather than load()'s compatibility default.
    args["use_gan"] = use_gan
    gradient_config = GradientDiagnostics()
    args.update({f"GradientDiagnostics.{key}": value for key, value in gradient_config.items()})
    is_primary = int(os.environ.get("RANK", accel.local_rank)) == 0
    launch = save_run_configuration(args, Path(save_path)) if is_primary else None
    state = load(args, accel, tracker, save_path, use_gan=use_gan)
    gradient_diagnostics = None
    diagnostic_logger = None
    if gradient_config["enabled"]:
        gradient_diagnostics = GradientMonitor(
            accel.unwrap(state.generator), accel,
            every_steps=gradient_config["every_steps"], early_steps=gradient_config["early_steps"],
        )
        diagnostic_logger = GradientDiagnosticLogger(save_path, tracker.step)
    if launch is not None:
        state.launch_id = launch.name
        tracker.print(f"Launch provenance: {launch}")
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
        persistent_workers=(num_workers > 0),
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=(num_workers > 0),
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global val_loop, validate, save_samples, checkpoint
    tracked_train_step = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_step)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    wandb_config = WandB()
    with ExitStack() as stack:
        if writer is not None:
            stack.callback(writer.close)
        wandb, wandb_run = None, None
        if is_primary:
            # Record values actually consumed by argbind after initialization too.
            argbind.dump_args(dict(argbind.get_used_args()), launch / "config_used.yml")
            wandb, wandb_run = initialize_wandb(
                wandb_config, args, Path(save_path), accel.unwrap(state.generator)
            )
            if wandb_run is not None:
                stack.enter_context(managed_wandb_run(
                    wandb, wandb_run, wandb_config["interrupt_timeout"]
                ))
                link_wandb_launch(launch, wandb_run)
                if writer is not None:
                    # Flush TensorBoard before a possible forced W&B exit.
                    stack.callback(writer.flush)

        # Reset on each log (and on resume): these are recent, not lifetime counts.
        model = accel.unwrap(state.generator)
        train_codebook_counts = torch.zeros(
            model.n_codebooks, model.codebook_size, dtype=torch.long, device=accel.device
        )
        with tracker.live:
            for tracker.step in count(start=tracker.step):
                output = tracked_train_step(
                    state, train_dataloader, accel, lambdas,
                    codebook_counts=train_codebook_counts,
                    gradient_diagnostics=gradient_diagnostics,
                )

                last_iter = (
                    tracker.step == num_iters - 1 if num_iters is not None else False
                )
                log_training_metrics(
                    output, tracker.step,
                    normal_due=tracker.step % wandb_config["log_freq"] == 0 or last_iter,
                    diagnostic_logger=diagnostic_logger, writer=writer,
                    wandb=wandb, wandb_run=wandb_run,
                )
                if reconstruction_due(tracker.step + 1, save_iters, sample_freq, last_iter):
                    save_samples(
                        state, val_idx, writer, wandb,
                        wandb_run if wandb_config["log_reconstruction"] else None,
                        save_path, wandb_config["stft_window_length"],
                    )

                if tracker.step % valid_freq == 0 or last_iter:
                    reduce_codebook_counts(train_codebook_counts)
                    if is_primary:
                        log_codebook_usage(
                            train_codebook_counts, tracker.step, save_path,
                            writer, wandb, wandb_run, split="train",
                            chart_presets=wandb_config["codebook_chart_presets"],
                        )
                    train_codebook_counts.zero_()
                    validation_output = validate(state, val_dataloader, accel)
                    if is_primary:
                        log_codebook_usage(
                            validation_output["codebook_counts"], tracker.step,
                            save_path, writer, wandb, wandb_run,
                            chart_presets=wandb_config["codebook_chart_presets"],
                        )
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
