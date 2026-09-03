# EEG adaptation of Descript Audio Codec

This fork trains the DAC encoder, residual vector quantizer, and decoder as an
EEG tokenizer. The original `dac/` package and audio training path remain
unchanged. EEG-specific data handling, objectives, augmentations, diagnostics,
and training live in separate files.

## Current files

| File | Status | Purpose |
|---|---|---|
| `dac/` | Original, untouched | DAC encoder, decoder, and residual vector quantizer |
| `scripts/train.py` | Original, untouched | Descript's audio GAN trainer |
| `conf/base.yml` | Original, untouched | Descript's audio training configuration |
| `scripts/train_eeg.py` | EEG addition | Current annotated EEG training program |
| `conf/base_eeg.yml` | EEG addition | Current EEG configuration in DAC's argbind format |
| `scripts/eeg_plotting/training_plots.py` | EEG addition | Integrated augmentation, reconstruction, PSD, and codebook plots |
| `scripts/eeg_dac_models.py` | Optional experiment | Alternative EEG decoders; not used by the current trainer |
| `conf/old/` and `runs/old/` | Archive | Earlier configurations, results, figures, and checkpoints |

The EEG trainer marks its sections as **KEPT**, **CHANGED**, **COMMENTED OUT**,
or **ADDED FOR EEG**. Displaced audio operations remain as comments beside their
EEG replacements; the executable original remains in `scripts/train.py`.

## Execution paths

Original audio DAC:

```text
conf/base.yml + conf/1gpu.yml
    -> argbind
    -> scripts/train.py
    -> AudioDataset + audio transforms
    -> DAC generator + discriminator
    -> waveform + STFT + mel + adversarial + VQ losses
```

Current EEG tokenizer:

```text
conf/base_eeg.yml
    -> argbind
    -> scripts/train_eeg.py
    -> lazy MNE FIF windows + robust per-electrode normalization
    -> noisy student and detached clean teacher
    -> original DAC encoder + RVQ + decoder
    -> waveform + log-power STFT + VQ + diversity + consistency losses
```

Each EEG channel is currently sampled and encoded independently as a mono
signal. At 256 Hz with encoder strides `[2, 2, 2, 4]`, the model emits 8 latent
frames per second, or one frame every 125 ms. Each frame contains two residual
code IDs, each selected from a 128-entry codebook.

## What is retained, removed, and added

Retained from DAC:

- convolutional encoder and decoder, Snake activations, and residual units;
- residual vector quantization, codebook loss, and commitment loss;
- AdamW, the learning-rate scheduler, AMP, gradient clipping, and checkpoint
  state.

Not used by the current EEG trainer:

- the audio discriminator and adversarial/feature-matching objectives;
- mel-spectrogram loss and audio-specific transforms;
- the multi-GPU accelerator/tracker wrapper and playable-audio sample export.

Added for EEG:

- lazy `.fif` loading through MNE and robust median/MAD normalization;
- slow drift, baseline shift, phase shift, line noise, Gaussian noise, and
  signal-mixing augmentations;
- detached clean-teacher versus noisy-student latent consistency;
- linear-frequency log-power STFT reconstruction and assignment diversity;
- clean/noisy hard-token agreement and codebook-collapse diagnostics;
- W&B step-level losses, epoch summaries, fixed reconstruction galleries, PSD
  galleries, and codebook-usage heatmaps.

The exact settings and loss weights in `conf/base_eeg.yml` are the source of
truth for each run.

## Run the EEG trainer

Create a Python 3.10 environment with the pinned EEG dependencies. The optional
argument chooses where the environment is stored:

```bash
bash scripts/create_eeg_venv.sh ~/venv_dac
```

This installs the current checkout in editable mode; do not run `pip install
dac`, which installs an unrelated PyPI package. Each colleague should then log
in to W&B once with `wandb login` if online experiment tracking is enabled.

```bash
source ~/venv_dac/bin/activate
cd /data/groups/bci/jonas/workspace/bci_descript-audio-codec

CUDA_VISIBLE_DEVICES=3 python scripts/train_eeg.py \
  --args.load conf/base_eeg.yml
```

Each run saves `latest.pt`, the lowest-validation-loss `best.pt`, an epoch-level
`history.json`, and a `plots/` directory. It also saves `config_input.yml` (the
exact supplied file) and `config_resolved.yml` (defaults plus command-line
overrides actually used). The resolved configuration is also stored in W&B and
inside each checkpoint. Earlier outputs remain in `runs/old/`.

---

The remainder of this README is the original Descript Audio Codec documentation.

# Descript Audio Codec (.dac): High-Fidelity Audio Compression with Improved RVQGAN

This repository contains training and inference scripts
for the Descript Audio Codec (.dac), a high fidelity general
neural audio codec, introduced in the paper titled **High-Fidelity Audio Compression with Improved RVQGAN**.

![](https://static.arxiv.org/static/browse/0.3.4/images/icons/favicon-16x16.png) [arXiv Paper: High-Fidelity Audio Compression with Improved RVQGAN
](http://arxiv.org/abs/2306.06546) <br>
📈 [Demo Site](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5)<br>
⚙ [Model Weights](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth)

👉 With Descript Audio Codec, you can compress **44.1 KHz audio** into discrete codes at a **low 8 kbps bitrate**.  <br>
🤌 That's approximately **90x compression** while maintaining exceptional fidelity and minimizing artifacts.  <br>
💪 Our universal model works on all domains (speech, environment, music, etc.), making it widely applicable to generative modeling of all audio.  <br>
👌 It can be used as a drop-in replacement for EnCodec for all audio language modeling applications (such as AudioLMs, MusicLMs, MusicGen, etc.) <br>

<p align="center">
<img src="./assets/comparsion_stats.png" alt="Comparison of compressions approaches. Our model achieves a higher compression factor compared to all baseline methods. Our model has a ~90x compression factor compared to 32x compression factor of EnCodec and 64x of SoundStream. Note that we operate at a target bitrate of 8 kbps, whereas EnCodec operates at 24 kbps and SoundStream at 6 kbps. We also operate at 44.1 kHz, whereas EnCodec operates at 48 kHz and SoundStream operates at 24 kHz." width=35%></p>


## Usage

### Installation
```
pip install descript-audio-codec
```
OR

```
pip install git+https://github.com/descriptinc/descript-audio-codec
```

### Weights
Weights are released as part of this repo under MIT license.
We release weights for models that can natively support 16 kHz, 24kHz, and 44.1kHz sampling rates.
Weights are automatically downloaded when you first run `encode` or `decode` command. You can cache them using one of the following commands
```bash
python3 -m dac download # downloads the default 44kHz variant
python3 -m dac download --model_type 44khz # downloads the 44kHz variant
python3 -m dac download --model_type 24khz # downloads the 24kHz variant
python3 -m dac download --model_type 16khz # downloads the 16kHz variant
```
We provide a Dockerfile that installs all required dependencies for encoding and decoding. The build process caches the default model weights inside the image. This allows the image to be used without an internet connection. [Please refer to instructions below.](#docker-image)


### Compress audio
```
python3 -m dac encode /path/to/input --output /path/to/output/codes
```

This command will create `.dac` files with the same name as the input files.
It will also preserve the directory structure relative to input root and
re-create it in the output directory. Please use `python -m dac encode --help`
for more options.

### Reconstruct audio from compressed codes
```
python3 -m dac decode /path/to/output/codes --output /path/to/reconstructed_input
```

This command will create `.wav` files with the same name as the input files.
It will also preserve the directory structure relative to input root and
re-create it in the output directory. Please use `python -m dac decode --help`
for more options.

### Programmatic Usage
```py
import dac
from audiotools import AudioSignal

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

model.to('cuda')

# Load audio signal file
signal = AudioSignal('input.wav')

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)

x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# Decode audio signal
y = model.decode(z)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
y.write('output.wav')
```

### Docker image
We provide a dockerfile to build a docker image with all the necessary
dependencies.
1. Building the image.
    ```
    docker build -t dac .
    ```
2. Using the image.

    Usage on CPU:
    ```
    docker run dac <command>
    ```

    Usage on GPU:
    ```
    docker run --gpus=all dac <command>
    ```

    `<command>` can be one of the compression and reconstruction commands listed
    above. For example, if you want to run compression,

    ```
    docker run --gpus=all dac python3 -m dac encode ...
    ```


## Training
The baseline model configuration can be trained using the following commands.

### Pre-requisites
Please install the correct dependencies
```
pip install -e ".[dev]"
```

## Environment setup

We have provided a Dockerfile and docker compose setup that makes running experiments easy.

To build the docker image do:

```
docker compose build
```

Then, to launch a container, do:

```
docker compose run -p 8888:8888 -p 6006:6006 dev
```

The port arguments (`-p`) are optional, but useful if you want to launch a Jupyter and Tensorboard instances within the container. The
default password for Jupyter is `password`, and the current directory
is mounted to `/u/home/src`, which also becomes the working directory.

Then, run your training command.


### Single GPU training
```
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --args.load conf/ablations/baseline.yml --save_path runs/baseline/
```

### Multi GPU training
```
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node gpu scripts/train.py --args.load conf/ablations/baseline.yml --save_path runs/baseline/
```

## Testing
We provide two test scripts to test CLI + training functionality. Please
make sure that the trainig pre-requisites are satisfied before launching these
tests. To launch these tests please run
```
python -m pytest tests
```

## Results

<p align="left">
<img src="./assets/objective_comparisons.png" width=75%></p>
