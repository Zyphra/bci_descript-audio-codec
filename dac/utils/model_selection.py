"""Select the EEG codec while sharing existing DAC.* RVQ configuration keys."""

import inspect
from pathlib import Path

import argbind

from dac.model import DAC, k8MOSS

DAC = argbind.bind(DAC)


SHARED_SETTINGS = (
    "sample_rate", "n_codebooks", "codebook_size", "codebook_dim", "quantizer_dropout",
)


@argbind.bind(without_prefix=True)
def generator_class(generator_type: str = "dac"):
    if generator_type == "dac":
        return DAC
    if generator_type == "k8moss":
        return k8MOSS
    raise ValueError(f"Unknown generator_type {generator_type!r}; choose 'dac' or 'k8moss'")


@argbind.bind()
def K8MOSS(
    shared_settings,
    coarse_time: float = 1 / 8,
    projection_stages: list = [0.5, 0.5, 0.75, 0.75],
    embed_dim: int = None,
    zero_init_biases: bool = False,
    num_heads: int = 4,
    theta: float = 10000.0,
):
    """Architecture-only settings; sample rate and RVQ always come from DAC.*."""
    return k8MOSS(
        sampling_rate=shared_settings["sample_rate"],
        **{key: shared_settings[key] for key in SHARED_SETTINGS if key != "sample_rate"},
        coarse_time=coarse_time,
        projection_stages=projection_stages,
        embed_dim=embed_dim,
        zero_init_biases=zero_init_biases,
        num_heads=num_heads,
        theta=theta,
    )


def shared_settings(args):
    defaults = inspect.signature(DAC).parameters
    return {key: args.get(f"DAC.{key}", defaults[key].default) for key in SHARED_SETTINGS}


def build_generator(args):
    if generator_class() is DAC:
        return DAC()
    return K8MOSS(shared_settings(args))


def load_generator(folder, *, package=True, map_location="cpu"):
    """Load the selected architecture, rejecting a different codec's checkpoint."""
    cls = generator_class()
    folder = Path(folder)
    for other in (DAC, k8MOSS):
        if other is not cls and (folder / other.__name__.lower()).exists():
            raise ValueError(
                f"{folder} contains a {other.__name__} checkpoint, but the selected "
                f"generator is {cls.__name__}. Use a separate save_path for each architecture."
            )
    if not (folder / cls.__name__.lower()).exists():
        return None, {}
    return cls.load_from_folder(folder, package=package, map_location=map_location, strict=True)
