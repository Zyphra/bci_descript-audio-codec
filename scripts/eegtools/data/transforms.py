import math
import torch
from typing import Sequence, Tuple
import argbind

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
