from typing import List, Optional, Union

import torch
from audiotools.ml import BaseModel
from torch import nn
from torch.nn import functional as F

from dac.nn.quantize import ResidualVectorQuantize


class k8MOSS(BaseModel):
    """
    Patch-merging transformer quantizer inspired by MOSS Tokenizer
    https://arxiv.org/pdf/2602.10934

    Key ingredients:
    (1) Coarse-time: defines the initial patch size in seconds
    (2) Projection stages: defines the down-projection schedule in a series of proportions in (0, 1]
        (a) to get number of token IDs per second, do 
        int(1 / coarse_time / 2^{len(projection_stages)})
    (3) embed_dim: defines the initial projection size to transformer model dimension
        (a) important because embed_dim will define what projection stage values give valid integers downstream
        (b) defaults to 4*patch_length to keep same proportionality as original MOSS, but still much smaller for EEG

    General idea: 
    (1) Right now, no channel dimension (singleton dimension) so no channel coordinates
    (2) Transformer process each channel-sample as an indepedent stream of tokens
    (3) Loop:
        (a) MERGE: halve tokens, double token dimensionality
            (i) No loss of information, just merges temporally adjacent positions
        (b) TRANSFORMER: mix time information into merged tokens
        (c) PROJECT: 
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        coarse_time: float = 1 / 8, # seconds
        projection_stages: Optional[List[float]] = None,
        embed_dim: Optional[int] = None,
        num_heads: int = 4,
        theta: float = 10000.0,
        n_codebooks: int = 4,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if sampling_rate <= 0 or coarse_time <= 0:
            raise ValueError("sampling_rate and coarse_time must be positive")
        self.sampling_rate = self.sample_rate = sampling_rate
        self.coarse_time = coarse_time
        self.patch_length = int(sampling_rate * coarse_time)
        if self.patch_length < 1:
            raise ValueError("Each patch must contain at least one sample")
        self.embed_dim = 4 * self.patch_length if embed_dim is None else embed_dim
        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        self.projection_stages = (
            [0.5, 0.5, 0.75, 0.75]
            if projection_stages is None else list(projection_stages)
        )
        if not self.projection_stages:
            raise ValueError("At least one projection stage is required")
        self.num_merges = len(self.projection_stages) - 1
        self.hop_length = self.patch_length * 2 ** self.num_merges
        self.num_heads = num_heads
        self.theta = theta

        self.stage_widths = []
        width = self.embed_dim
        for stage, ratio in enumerate(self.projection_stages):
            input_width = width * (2 if stage else 1)
            width = input_width * ratio
            if width <= 0 or width != int(width):
                raise ValueError(f"Invalid width at stage {stage}: {width}")
            width = int(width)
            self.stage_widths.append((input_width, width))
        self.latent_dim = width

        if n_codebooks < 1 or codebook_size < 1:
            raise ValueError("n_codebooks and codebook_size must be positive")
        dims = [codebook_dim] * n_codebooks if isinstance(codebook_dim, int) else codebook_dim
        if len(dims) != n_codebooks or any(d < 1 for d in dims):
            raise ValueError("Provide one positive codebook dimension per codebook")
        if not 0 <= quantizer_dropout <= 1:
            raise ValueError("quantizer_dropout must be between 0 and 1")
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout

        self.encoder = Encoder(self.patch_length, self.stage_widths, num_heads, theta)
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.decoder = Decoder(self.patch_length, self.stage_widths, num_heads, theta)

    def check_input_length(self, data):
        length = data.shape[-1]
        if length < self.hop_length or length % self.hop_length:
            raise ValueError(
                f"Input length must be a positive multiple of {self.hop_length} samples; "
                f"got {length}"
            )

    def preprocess(self, data, sample_rate=None):
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sample_rate}")
        if data.ndim == 2:
            data = data.unsqueeze(1)
        if data.ndim != 3 or data.shape[1] != 1:
            raise ValueError("Expected single-channel segments: [B, T] or [B, 1, T]")
        self.check_input_length(data)
        return data

    def encode(self, data, n_quantizers=None):
        """Return DAC's (z, codes, latents, commitment, codebook) tuple."""
        data = self.preprocess(data)
        x = self.encoder(data).transpose(1, 2).contiguous()
        return self.quantizer(x, n_quantizers)

    def decode(self, z):
        """Decode DAC-layout quantized latents [B, D, W] to [B, 1, T]."""
        return self.decoder(z.transpose(1, 2).unsqueeze(1))

    def forward(self, data, sample_rate=None, n_quantizers=None):
        squeeze_channel = data.ndim == 2
        data = self.preprocess(data, sample_rate)
        z, codes, latents, commit, codebook = self.encode(data, n_quantizers)
        reconstruction = self.decode(z)
        if squeeze_channel:
            reconstruction = reconstruction.squeeze(1)
        return {
            "audio": reconstruction,
            "z": z,
            "codes": codes,
            "latents": latents,
            "codebook_mask": self.quantizer.last_active_mask,
            "vq/commitment_loss": commit,
            "vq/codebook_loss": codebook,
        }


class Encoder(nn.Module):
    def __init__(self, patch_length, stage_widths, num_heads, theta):
        super().__init__()
        self.patch_length = patch_length
        self.patch_projection = nn.Linear(patch_length, stage_widths[0][0])
        self.projections = nn.ModuleList([
            nn.Linear(input_width, output_width)
            for input_width, output_width in stage_widths
        ])
        self.attention = nn.ModuleList([
            Attention(width, num_heads, theta) for width, _ in stage_widths
        ])

    def forward(self, data):
        # W: current token count; D_in/D_out: encoder stage input/output widths.
        B, _, T = data.shape
        x = data.reshape(B, 1, T // self.patch_length, self.patch_length)  # [B, 1, W, patch_length]
        x = self.patch_projection(x)  # [B, 1, W, embed_dim]
        for stage, (attention, projection) in enumerate(zip(self.attention, self.projections)):
            if stage > 0:
                x = self.merge(x)  # [B, 1, W, D] -> [B, 1, W // 2, 2 * D]
            x = attention(x, stage)  # [B, 1, W, D_in]
            x = projection(x)  # [B, 1, W, D_out]
        return x.squeeze(1)  # [B, W, latent_dim]

    @staticmethod
    def merge(x):
        B, C, W, D = x.shape
        return x.reshape(B, C, W // 2, 2 * D)


class Decoder(nn.Module):
    def __init__(self, patch_length, stage_widths, num_heads, theta):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(output_width, input_width)
            for input_width, output_width in stage_widths
        ])
        self.attention = nn.ModuleList([
            Attention(width, num_heads, theta) for width, _ in stage_widths
        ])
        self.patch_projection = nn.Linear(stage_widths[0][0], patch_length)

    def forward(self, x):
        # W: current token count; D_in/D_out: encoder stage input/output widths.
        B = x.shape[0]
        for stage in reversed(range(len(self.projections))):
            x = self.projections[stage](x)  # [B, 1, W, D_out] -> [B, 1, W, D_in]
            x = self.attention[stage](x, stage)  # [B, 1, W, D_in]
            if stage > 0:
                x = self.unmerge(x)  # [B, 1, W, D] -> [B, 1, 2 * W, D // 2]
        x = self.patch_projection(x)  # [B, 1, W, patch_length]
        return x.reshape(B, 1, -1)  # [B, 1, T]

    @staticmethod
    def unmerge(x):
        B, C, W, D = x.shape
        return x.reshape(B, C, W * 2, D // 2)


class Attention(nn.Module):
    """Pre-norm transformer block with temporal RoPE on the full head."""

    def __init__(self, dim, num_heads=4, theta=10000.0):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % (2 * num_heads):
            raise ValueError("Each attention head needs an even feature width for RoPE")
        if theta <= 0:
            raise ValueError("RoPE theta must be positive")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.theta = theta
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, stage):
        B, C, W, D = x.shape
        if C != 1:
            raise ValueError("The transformer expects a singleton channel dimension")
        positions = (torch.arange(W, device=x.device, dtype=torch.float32) + 0.5) * 2 ** stage
        x = x.squeeze(1)
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(B, W, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4).unbind(0)
        )
        q, k = self.apply_rope(q, k, positions)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + self.out_proj(y.transpose(1, 2).reshape(B, W, D))
        x = x + self.mlp(self.norm2(x))
        return x.unsqueeze(1)

    def apply_rope(self, q, k, positions):
        freq = self.theta ** (
            -torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32)
            / self.head_dim
        )
        angles = positions[:, None] * freq[None, :]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]

        def rotate(v):
            a, b = v.float().reshape(*v.shape[:-1], -1, 2).unbind(-1)
            return torch.stack(
                (a * cos - b * sin, a * sin + b * cos), dim=-1
            ).flatten(-2).to(v.dtype)

        return rotate(q), rotate(k)
