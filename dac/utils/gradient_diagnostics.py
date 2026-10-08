"""Occasional loss-by-parameter gradient measurements, independent of optimizers."""

import json
import math
import time
import warnings
from pathlib import Path

import torch


GROUPS = ("encoder", "vq_projections", "codebooks", "decoder")
MODEL_PARTS = (*GROUPS, "discriminator")
LOSSES = ("mel", "stft_mag", "stft_log_mag", "stft_phase", "waveform",
          "adv_feat", "adv_gen", "adv_disc", "vq_commitment", "vq_codebook")
PANELS = (*(f"{loss}_{group}" for loss in LOSSES for group in MODEL_PARTS),
          "z_clip_scale", "z_diagnostic_seconds")
WANDB_SOURCES = {
    f"train_gradients_{section}/{loss}_{group}": f"train_gradients/{kind}_{group}_{loss}"
    for section, kind in (("weighted", "weighted"), ("unweighted", "raw"))
    for loss in LOSSES for group in MODEL_PARTS
}
WANDB_SOURCES.update({
    f"train_gradients_{section}/z_{monitor}": f"train_gradients/{monitor}"
    for section in ("weighted", "unweighted")
    for monitor in ("clip_scale", "diagnostic_seconds")
})
# (first, second, parameter groups). Ratios are second / first.
GRADIENT_PAIRS = (
    ("reconstruction", "commitment", ("encoder", "vq_input")),
    ("waveform", "mag", ("encoder", "decoder")),
    ("waveform", "log_mag", ("encoder", "decoder")),
    ("mag", "log_mag", ("encoder", "decoder")),
)
GEOMETRY_SOURCES = {
    f"train_gradient_cosines/{a}_vs_{b}_{group}": f"train_gradients/cosine_{a}_vs_{b}_{group}"
    for a, b, groups in GRADIENT_PAIRS for group in groups
}
GEOMETRY_SOURCES.update({
    f"train_gradient_ratios/{b}_over_{a}_{group}": f"train_gradients/ratio_{b}_over_{a}_{group}"
    for a, b, groups in GRADIENT_PAIRS for group in groups
})
RECONSTRUCTION_LOSSES = frozenset(("waveform", "mel", "stft_mag", "stft_log_mag", "stft_phase"))
GEOMETRY_EPS = 1e-12
# Keep clipping and its diagnostic multiplier tied to the same threshold.
GENERATOR_CLIP_NORM = 1e3


def validate_schedule(every_steps, early_steps):
    if type(every_steps) is not int or every_steps < 1:
        raise ValueError("GradientDiagnostics.every_steps must be a positive integer")
    if any(type(step) is not int or step < 1 for step in early_steps):
        raise ValueError("GradientDiagnostics.early_steps must contain positive integers")


def component_losses(output, lambdas, stft_loss):
    """Match train_cw's objective, including direct STFT-component lambdas."""
    keys = {
        "waveform": "waveform/loss", "mel": "mel/loss",
        "vq_commitment": "vq/commitment_loss", "vq_codebook": "vq/codebook_loss",
        "adv_gen": "adv/gen_loss", "adv_feat": "adv/feat_loss",
    }
    result = {name: (output[key], float(lambdas.get(key, 0)))
              for name, key in keys.items() if key in output}
    for name, attribute in (("mag", "mag_weight"), ("log_mag", "log_weight"),
                            ("phase", "phase_weight")):
        key = f"stft/{name}_loss"
        if key in output:
            # MultiScaleSTFTLoss.forward does not multiply by self.weight.
            weight = (float(lambdas.get("stft/loss", 0)) * getattr(stft_loss, attribute)
                      + float(lambdas.get(key, 0)))
            result[f"stft_{name}"] = (output[key], weight)
    supported = {*keys.values(), "stft/loss", "stft/mag_loss",
                 "stft/log_mag_loss", "stft/phase_loss"}
    unknown = [key for key, weight in lambdas.items()
               if weight != 0 and key in output and key not in supported]
    if unknown:
        raise ValueError(f"Gradient diagnostics cannot attribute these objective terms: {unknown}")
    if any(not math.isfinite(weight) for _, weight in result.values()):
        raise ValueError("Gradient diagnostics require finite loss weights")
    return result


class GradientDiagnostics:
    """Single-device FP32 diagnostics; never write to parameters or .grad buffers."""

    def __init__(self, model, accel, every_steps=1000, early_steps=(1, 2, 5, 10)):
        validate_schedule(every_steps, early_steps)
        if (accel.amp or accel.world_size > 1 or accel.use_ddp or accel.use_dp
                or (torch.distributed.is_initialized())):
            raise ValueError("GradientDiagnostics supports single-device FP32 only; disable AMP/DDP/DP")
        self.every_steps = every_steps
        self.early_steps = frozenset(early_steps)
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.parameters or any(p.dtype != torch.float32 for p in self.parameters):
            raise ValueError("GradientDiagnostics requires trainable FP32 parameters")
        self.device = self.parameters[0].device
        if any(p.device != self.device for p in self.parameters):
            raise ValueError("GradientDiagnostics requires all parameters on one device")
        indices = {id(p): i for i, p in enumerate(self.parameters)}
        self.groups = {name: [] for name in GROUPS}
        self.layer_groups = {}
        self.input_projection_indices = []

        def add(module, group, layer=None):
            for parameter in module.parameters():
                if parameter.requires_grad:
                    index = indices[id(parameter)]
                    self.groups[group].append(index)
                    if layer is not None:
                        self.layer_groups.setdefault(layer, []).append(index)

        add(model.encoder, "encoder")
        add(model.decoder, "decoder")
        for layer, quantizer in enumerate(model.quantizer.quantizers):
            add(quantizer.in_proj, "vq_projections", f"codebook_{layer}_projections")
            self.input_projection_indices.extend(
                indices[id(p)] for p in quantizer.in_proj.parameters() if p.requires_grad
            )
            add(quantizer.out_proj, "vq_projections", f"codebook_{layer}_projections")
            add(quantizer.codebook, "codebooks", f"codebook_{layer}_embeddings")
        assigned = [index for group in self.groups.values() for index in group]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(indices.values()):
            raise ValueError("Gradient diagnostic groups must cover each trainable parameter exactly once")

    def _gradient_geometry(self, vectors):
        """Whole-group geometry; never average layer cosines or scalar norms.

        Vectors are detached weighted gradients, with None denoting exact zero.
        Undefined cosines/ratios are absent, not zero and not carried forward.
        """
        values = {}
        for a, b, groups in GRADIENT_PAIRS:
            for group in groups:
                x, y = vectors.get((a, group)), vectors.get((b, group))
                if x is None or y is None:
                    continue
                zero = torch.zeros((), device=self.device)
                nx = torch.sqrt(sum((v.square().sum() for v in x if v is not None), zero))
                ny = torch.sqrt(sum((v.square().sum() for v in y if v is not None), zero))
                if not bool(torch.isfinite(nx) & torch.isfinite(ny)):
                    continue
                values[f"pair_norm_{a}_vs_{b}_{group}_first"] = nx
                values[f"pair_norm_{a}_vs_{b}_{group}_second"] = ny
                if nx <= GEOMETRY_EPS:
                    continue
                values[f"ratio_{b}_over_{a}_{group}"] = ny / nx
                if ny <= GEOMETRY_EPS:
                    continue
                dot = sum(((u * v).sum() for u, v in zip(x, y)
                           if u is not None and v is not None), zero)
                values[f"cosine_{a}_vs_{b}_{group}"] = (dot / nx / ny).clamp(-1, 1)
        return values

    def due(self, step):
        completed = step + 1
        return completed in self.early_steps or completed % self.every_steps == 0

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _norms(self, gradients):
        # Only one scalar per parameter is retained; never concatenate parameters.
        norms = torch.stack([
            torch.linalg.vector_norm(gradient.detach()) if gradient is not None
            else torch.zeros((), device=self.device)
            for gradient in gradients
        ])
        result = {name: torch.linalg.vector_norm(norms[indices])
                  for name, indices in {**self.groups, **self.layer_groups}.items()}
        for group, suffix in (("vq_projections", "projections"), ("codebooks", "embeddings")):
            layers = [value for name, value in result.items()
                      if name.startswith("codebook_") and name.endswith(f"_{suffix}")]
            if layers:
                result[group] = torch.stack(layers).mean()
        return result

    def measure_discriminator(self, discriminator):
        """Read the real discriminator backward's unscaled, pre-clipping gradients.

        Its objective has coefficient one. Other losses do not update this
        optimizer, even when they are differentiable through the discriminator.
        """
        self._synchronize()
        start = time.perf_counter()
        parameters = [p for p in discriminator.parameters() if p.requires_grad]
        if any(p.device != self.device or p.dtype != torch.float32 for p in parameters):
            raise ValueError("GradientDiagnostics requires discriminator parameters on the same device in FP32")
        norms = [torch.linalg.vector_norm(p.grad.detach()) for p in parameters if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.zeros((), device=self.device)
        values = {f"weighted_discriminator_{loss}": torch.zeros((), device=self.device)
                  for loss in LOSSES}
        values.update({f"{kind}_{group}_adv_disc": torch.zeros((), device=self.device)
                       for kind in ("raw", "weighted") for group in GROUPS})
        values.update(raw_discriminator_adv_disc=norm, weighted_discriminator_adv_disc=norm,
                      weight_adv_disc=torch.ones((), device=self.device))
        self._synchronize()
        values["diagnostic_seconds"] = torch.tensor(time.perf_counter() - start, device=self.device)
        return values

    def measure_components(self, components):
        self._synchronize()
        start = time.perf_counter()
        values = {}
        geometry_groups = {"encoder": self.groups["encoder"],
                           "decoder": self.groups["decoder"],
                           "vq_input": self.input_projection_indices}
        vectors = {}
        names = {"waveform": "waveform", "stft_mag": "mag",
                 "stft_log_mag": "log_mag", "vq_commitment": "commitment"}
        for name, (loss, weight) in components.items():
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError(f"Gradient diagnostic loss {name} must be a scalar tensor")
            gradients = (torch.autograd.grad(
                loss, self.parameters, retain_graph=True, create_graph=False, allow_unused=True
            ) if loss.requires_grad else (None,) * len(self.parameters))
            norms = self._norms(gradients)
            # Reuse these traversals; no second backward or model forward.
            # Keep only groups needed by the eight pairs, on diagnostic steps.
            for group, indices in geometry_groups.items():
                keep = (name in names and
                        (group in ("encoder", "decoder") if name != "vq_commitment"
                         else group in ("encoder", "vq_input")))
                reconstruct = name in RECONSTRUCTION_LOSSES and group in ("encoder", "vq_input")
                if not (keep or reconstruct):
                    continue
                vector = tuple(gradients[i].detach() * weight
                               if gradients[i] is not None and weight != 0 else None
                               for i in indices)
                if keep:
                    vectors[names[name], group] = vector
                if reconstruct:
                    old = vectors.get(("reconstruction", group), (None,) * len(indices))
                    vectors["reconstruction", group] = tuple(
                        v if u is None else u if v is None else u + v
                        for u, v in zip(old, vector)
                    )
            del gradients
            for group, norm in norms.items():
                values[f"raw_{group}_{name}"] = norm
                # Preserve nonfinite raw norms while avoiding 0 * inf/nan.
                values[f"weighted_{group}_{name}"] = norm * abs(weight) if weight else norm.new_zeros(())
            values[f"weight_{name}"] = torch.tensor(weight, device=self.device)
        values.update(self._gradient_geometry(vectors))
        del vectors
        self._synchronize()
        values["diagnostic_seconds"] = torch.tensor(time.perf_counter() - start, device=self.device)
        return values

    def measure_total(self, values):
        """Call after the real backward pass and unscaling, before clipping."""
        self._synchronize()
        start = time.perf_counter()
        values.update({f"total_{group}": norm
                       for group, norm in self._norms(p.grad for p in self.parameters).items()})
        self._synchronize()
        values["diagnostic_seconds"] += time.perf_counter() - start

    def finish(self, values, global_norm, clip_norm=GENERATOR_CLIP_NORM):
        """Detach and transfer scalars once; report nonfinite readings explicitly."""
        start = time.perf_counter()
        values["global_norm"] = global_norm.detach()
        # Matches torch.nn.utils.clip_grad_norm_, including its epsilon.
        values["clip_scale"] = (clip_norm / (global_norm.detach() + 1e-6)).clamp(max=1)
        result = dict(zip(values, torch.stack(list(values.values())).cpu().tolist()))
        result["diagnostic_seconds"] += time.perf_counter() - start
        nonfinite = [key for key, value in result.items() if not math.isfinite(value)]
        if nonfinite:
            warnings.warn(f"Nonfinite gradient diagnostics: {', '.join(nonfinite)}", RuntimeWarning)
        return {f"train_gradients/{key}": value for key, value in result.items()}


def define_wandb_metrics(run):
    # Two 5 x 10 grids, each with the same two monitoring metrics.
    for key in (*WANDB_SOURCES, *GEOMETRY_SOURCES):
        run.define_metric(key, step_metric="training_step", hidden=False)


class GradientDiagnosticLogger:
    """Sparse histories bypass Tracker's persistent metric dictionary."""

    def __init__(self, save_path, start_step):
        self.path = Path(save_path) / "gradient_diagnostics.json"
        history = json.loads(self.path.read_text()) if self.path.exists() else []
        self.history = [row for row in history if row["training_step"] < start_step]

    def record(self, step, metrics, writer=None, wandb=None):
        self.history = [row for row in self.history if row["training_step"] < step]
        self.history.append({"training_step": step, **{
            key: value if math.isfinite(value) else None for key, value in metrics.items()
        }})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.history, allow_nan=False) + "\n")
        temporary.replace(self.path)
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(key, value, step)
        # Native scalar panels support run overlays. Preserve true zeros; absent
        # losses/groups (e.g. GAN disabled) are unavailable, not measured zeros.
        payload = {key: metrics.get(source, float("nan"))
                   for key, source in WANDB_SOURCES.items()}
        payload.update({key: metrics[source] for key, source in GEOMETRY_SOURCES.items()
                        if source in metrics and math.isfinite(metrics[source])})
        return payload


def log_training_metrics(output, step, normal_due, diagnostic_logger=None,
                         writer=None, wandb=None, wandb_run=None):
    """Merge sampled diagnostics and regular scalar logging into one W&B write."""
    diagnostics = output.pop("_gradient_diagnostics", None)
    payload = {"training_step": step}
    if normal_due:
        payload.update({f"train/{key}": value for key, value in output.items()
                        if isinstance(value, (int, float))})
    if diagnostics is not None:
        payload.update(diagnostic_logger.record(
            step, diagnostics, writer, wandb if wandb_run is not None else None
        ))
    if wandb_run is not None and (normal_due or diagnostics is not None):
        wandb_run.log(payload)
        if diagnostics is not None:
            # Keep all raw/per-layer measurements accessible in W&B Files without
            # exposing each measurement as another chartable scalar or table.
            wandb_run.save(str(diagnostic_logger.path),
                           base_path=str(diagnostic_logger.path.parent), policy="now")
