import torch
import torch.nn.functional as F
from typing import Iterable
import argbind

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

# this loss is a new addition here. Original DAC seems to have avoided code book collapse without such explicit loss
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