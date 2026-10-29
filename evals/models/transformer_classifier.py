import torch
from torch import nn
from torch.nn import functional as F
from linear_attention_transformer import LinearAttentionTransformer


def sinusoidal_encoding(n_positions, dim):
    """Return (n_positions, dim) sine/cosine encodings; dim must be even."""
    positions = torch.arange(n_positions, dtype=torch.float32)[:, None]
    frequencies = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
    angles = positions * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TransformerClassifier(nn.Module):
    """Map (batch_size, channels, coarse_time, codebook_dim) to class logits.

    normalize=True gives additive positional vectors unit L2 norm.
    RoPE rotations and projected input embeddings are left unchanged.
    """

    def __init__(
        self,
        n_classes,
        channels=16,
        coarse_time=32,
        codebook_dim=8, #check this guy out {8, 16, ..., }
        model_dim=64,
        heads=2,
        depth=4, #4 seems to be at sweet spot of small + good performance
        temporal_pos="sinusoidal",
        channel_pos="learned",
        aggregation_mode="mean_pool_final",
        normalize=True,
        channel_coords=None,
    ):
        super().__init__()
        if temporal_pos not in ("sinusoidal", "rotary"):
            raise ValueError("temporal_pos must be 'sinusoidal' or 'rotary'")
        if channel_pos not in (None, "learned", "spatial"):
            raise ValueError("channel_pos must be None, 'learned', or 'spatial'")
        if aggregation_mode not in (
            "CLS_token", "mean_pool_final", "attn_pool_final", "flatten_final"
        ):
            raise ValueError(
                "aggregation_mode must be 'CLS_token', 'mean_pool_final', "
                "'attn_pool_final', or 'flatten_final'"
            )
        if heads < 1 or model_dim % heads:
            raise ValueError("model_dim must be divisible by a positive heads count")
        pos_dim = model_dim if temporal_pos == "sinusoidal" else model_dim // heads
        if pos_dim < 2 or pos_dim % 2:
            raise ValueError("The positional embedding dimension must be positive and even")

        self.input_shape = (channels, coarse_time, codebook_dim)
        self.temporal_pos = temporal_pos
        self.channel_pos = channel_pos
        self.aggregation_mode = aggregation_mode
        self.normalize = normalize

        #  we will make linear embeddings out of the codebooks
        # standardizes model_dim for fair comparison across models
        self.input_projection = nn.Linear(codebook_dim, model_dim)

        # self.register_buffer: not a trainable parameter, saved w/ models state_dict
        self.register_buffer("time_encoding", sinusoidal_encoding(coarse_time, pos_dim))

        if channel_pos == "learned":
            self.channel_embedding = nn.Parameter(torch.randn(channels, model_dim) * 0.02)
        elif channel_pos == "spatial":
            # take (C,3) coordinates --> project to model_dim space
            if channel_coords is None or tuple(channel_coords.shape) != (channels, 3):
                raise ValueError("spatial encoding requires (channels, 3) channel_coords")
            self.register_buffer("channel_coords", channel_coords.detach().float().clone())
            self.channel_projection = nn.Linear(3, model_dim)

        num_tokens = channels * coarse_time
        classifier_dim = model_dim
        if aggregation_mode == "CLS_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            num_tokens += 1
        elif aggregation_mode == "attn_pool_final":
            self.pool_score = nn.Linear(model_dim, 1)
        elif aggregation_mode == "flatten_final":
            classifier_dim = num_tokens * model_dim

        self.transformer = LinearAttentionTransformer(
            dim=model_dim, depth=depth, heads=heads,
            max_seq_len=num_tokens,
            causal=False,
        )
        self.classification_head = nn.Linear(classifier_dim, n_classes)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected a batch of tensors with shape {self.input_shape}")
        batch_size, channels, coarse_time, codebook_dim = x.shape
        x = self.input_projection(x)  # (batch_size, channels, coarse_time, model_dim)

        rotary_pos = None
        temporal = self.time_encoding.to(x)
        if self.temporal_pos == "sinusoidal":
            if self.normalize:
                temporal = F.normalize(temporal, dim=-1, eps=1e-6)
            x = x + temporal[None, None, :, :]
        else:
            # Repeat time positions for each channel; RoPE acts inside attention.
            rotary_pos = temporal.repeat(channels, 1)[None, :, :]

        channel = None
        if self.channel_pos == "learned":
            channel = self.channel_embedding
        elif self.channel_pos == "spatial":
            channel = self.channel_projection(self.channel_coords)
        if channel is not None:
            if self.normalize:
                channel = F.normalize(channel, dim=-1, eps=1e-6)
            x = x + channel.to(x)[None, :, None, :]

        x = x.flatten(1, 2)  # (batch_size, channels * coarse_time, model_dim)
        if self.aggregation_mode == "CLS_token":
            cls = self.cls_token.to(x).expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)
            if rotary_pos is not None:
                # Position zero has sin=0, cos=1, so CLS is not rotated.
                rotary_pos = torch.cat((temporal[None, :1, :], rotary_pos), dim=1)

        x = self.transformer(x, pos_emb=rotary_pos)
        if self.aggregation_mode == "CLS_token":
            pooled = x[:, 0]
        elif self.aggregation_mode == "mean_pool_final":
            pooled = x.mean(dim=1)
        elif self.aggregation_mode == "attn_pool_final":
            weights = self.pool_score(x).softmax(dim=1)
            pooled = (weights * x).sum(dim=1)
        else:  # flatten_final
            pooled = x.flatten(1)
        return self.classification_head(pooled)


@torch.no_grad()
def classification_metrics(logits, labels):
    """Summarize a whole split, rather than averaging per-batch accuracies.

    Cross-entropy measures error against the labels. Prediction entropy measures
    uncertainty: zero means confident; log(number of classes) means uniform.
    Both use natural logarithms (nats).
    """
    n_classes = logits.shape[1]
    predictions = logits.argmax(dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()

    # Rows are true classes; columns are predicted classes.
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for true_class in range(n_classes):
        for predicted_class in range(n_classes):
            confusion[true_class, predicted_class] = (
                (labels == true_class) & (predictions == predicted_class)
            ).sum()
    support = confusion.sum(dim=1)
    present = support > 0
    recall = confusion.diag()[present].float() / support[present]

    return {
        "cross_entropy": F.cross_entropy(logits, labels).item(),
        "balanced_accuracy": recall.mean().item(),
        "prediction_entropy": -(probabilities * log_probabilities).sum(dim=1).mean().item(),
        "confusion_matrix": confusion.tolist(),
        "n_examples": len(labels),
    }


def train(model, dataloader, optimizer, max_grad_norm=1.0, loss_function=F.cross_entropy):
    """Train for ONE epoch; return metrics and gradient diagnostics.

    The caller controls the epoch loop. Reported gradients are measured before
    clipping. Training predictions are collected while weights are changing;
    use test(model, train_loader) for a fixed-weight training-set measurement.
    """
    model.train()
    device = next(model.parameters()).device
    all_logits, all_labels = [], []
    gradient_norms = []
    parameter_gradient_sums = {name: 0.0 for name, p in model.named_parameters() if p.requires_grad}
    missing_gradients = set()

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()

        # Collect diagnostics on the GPU, then transfer them together.
        gradients = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing_gradients.add(name)
            else:
                gradients[name] = parameter.grad
        squared_norm = 0.0
        if gradients:
            finite = torch.stack([torch.isfinite(g).all() for g in gradients.values()]).cpu().tolist()
            norms = torch.stack([g.norm() for g in gradients.values()]).cpu().tolist()
            for name, is_finite, norm in zip(gradients, finite, norms):
                if not is_finite:
                    raise FloatingPointError(f"Non-finite gradient in {name}")
                parameter_gradient_sums[name] += norm
                squared_norm += norm ** 2
        gradient_norms.append(squared_norm ** 0.5)

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Detach so the metrics do not keep the backward graph in memory.
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_labels:
        raise ValueError("Training dataloader is empty")
    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["gradient_norms"] = gradient_norms
    metrics["parameter_gradient_norms"] = {
        name: total / len(gradient_norms) for name, total in parameter_gradient_sums.items()
    }
    metrics["missing_gradients"] = sorted(missing_gradients)
    return metrics


@torch.no_grad()
def test(model, dataloader):
    """Evaluate a split without updating weights or calculating gradients."""
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_labels = [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    if not all_labels:
        raise ValueError("Evaluation dataloader is empty")
    return classification_metrics(torch.cat(all_logits), torch.cat(all_labels))
