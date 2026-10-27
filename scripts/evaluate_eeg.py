"""Paired EEG checkpoint evaluation; see --help and README.md.

Spectral diagnostics are single-resolution, physically scaled estimates, not
the trainer's multi-scale STFT loss. All checkpoints see frozen model inputs.
"""

# python scripts/evaluate_eeg.py \
#   runs/vq_wt_012/latest runs/vq_wt_015/latest \
#   --labels commit_1 commit_2 \
#   --windows 2500 --device cuda:0 \
#   --output analysis/commitment_eval

#
# #
#

# # Seed 0: create the shared set of 2,500 validation windows.
# python scripts/evaluate_eeg.py \
#   runs/vq_wt_015/latest runs/vq_wt_030/latest \
#   --labels codebook_1_seed_0 codebook_4_seed_0 \
#   --config runs/vq_wt_015/config_resolved.yml \
#   --windows 2500 \
#   --device cuda:0 --batch-size 8 \
#   --output analysis/codebook_1_vs_4_seed_0

# # Seed 1: reuse exactly those inputs.
# python scripts/evaluate_eeg.py \
#   runs/vq_wt_016/latest runs/vq_wt_031/latest \
#   --labels codebook_1_seed_1 codebook_4_seed_1 \
#   --inputs analysis/codebook_1_vs_4_seed_0/inputs.npz \
#   --device cuda:0 --batch-size 8 \
#   --output analysis/codebook_1_vs_4_seed_1

# # Seed 2: reuse the same inputs again.
# python scripts/evaluate_eeg.py \
#   runs/vq_wt_017/latest runs/vq_wt_032/latest \
#   --labels codebook_1_seed_2 codebook_4_seed_2 \
#   --inputs analysis/codebook_1_vs_4_seed_0/inputs.npz \
#   --device cuda:0 --batch-size 8 \
#   --output analysis/codebook_1_vs_4_seed_2



import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BANDS = "delta:1:4,theta:4:8,alpha:8:13,beta:13:30,low_gamma:30:45,mid_gamma:45:80,hi_gamma:80:128"


def parse_bands(value):
    bands = []
    for part in value.split(","):
        name, low, high = part.split(":")
        low, high = float(low), float(high)
        if not name or not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            raise ValueError(f"Invalid band: {part}")
        bands.append((name, low, high))
    if len({b[0] for b in bands}) != len(bands):
        raise ValueError("Band names must be unique")
    return bands


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def resolve_checkpoint(path):
    """Accept a run, tag directory, dac directory, explicit model file, or a
    precomputed-reconstruction .npz (key 'reconstruction', [N,T] in input
    units, same windows as --inputs) produced by a non-DAC coder such as
    scripts/mdct_coder.py."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    for relative in ("dac/weights.pth", "weights.pth", "latest/dac/weights.pth",
                     "dac/package.pth", "package.pth", "latest/dac/package.pth"):
        candidate = path / relative
        if candidate.is_file():
            return candidate
    raise ValueError(f"No checkpoint found at {path}")


def find_config(checkpoint):
    for parent in checkpoint.parents:
        candidate = parent / "config_resolved.yml"
        if candidate.is_file():
            return candidate
    raise ValueError("Cannot infer validation config; pass --config or --inputs")


def freeze_inputs(config_path, count, seconds, seed, *, raw=False, split="val"):
    """Use the training dataset and transform, once, before loading any model."""
    import argbind
    import torch
    import yaml
    import train_cw as trainer

    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    config = yaml.load(Path(config_path).read_text(), Loader=yaml.FullLoader)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = dict(config)
    if raw:
        # Neutralize global and scoped training transforms before dataset
        # construction. Source selection/cropping still uses this dataset config.
        for scope in ("", f"{split}/"):
            for stage in ("preprocess", "augment", "postprocess"):
                config[f"{scope}build_transform.{stage}"] = ["Identity"]
            config[f"{scope}build_transform.augment_prob"] = 0.0
    config[f"{split}/AudioDataset.n_examples"] = count
    if seconds is not None:
        config[f"{split}/AudioDataset.duration"] = seconds
    rate = int(config["DAC.sample_rate"])
    arrays, provenance = [], []
    with argbind.scope(config, split):
        dataset = trainer.build_dataset(rate)
        for index in range(count):
            item = dataset[index]
            # The training loader pads short recordings. Such padding would
            # contaminate spectral estimates, so fail instead of scoring it.
            source_dataset = dataset.datasets[index % len(dataset.datasets)]
            for loader in source_dataset.loaders.values():
                entry = loader.fif_index.get(str(item.get("path", "")))
                if entry and entry["n_times"] / entry["sfreq"] < source_dataset.duration:
                    raise ValueError(f"Recording is shorter than the requested excerpt: {item['path']}")
            batch = dataset.collate([item])
            transformed = dataset.transform(batch["signal"].clone(), **batch["transform_args"])
            array = transformed.audio_data.detach().cpu().numpy()[0]
            if array.shape[0] != 1:
                raise ValueError("EEG DAC evaluation requires single-channel windows")
            arrays.append(array[0])
            provenance.append({"index": index, "path": str(item.get("path", "")),
                               "metadata": item["signal"].metadata})
            if (index + 1) % 100 == 0:
                print(f"Frozen {index + 1}/{count} validation windows", flush=True)
    return np.stack(arrays).astype(np.float32), rate, {
        "config_path": str(Path(config_path).resolve()), "config": config,
        "seed": seed, "windows": provenance, "split": split,
        "units": ("loader output before evaluation transforms" if raw else
                  "model input units after configured validation transforms"),
    }


def prepare_condition(raw, rate, *, notched, frequencies, bandwidth, pad_seconds,
                      target_std, clip):
    """Apply explicit evaluation transforms, independent of checkpoint config."""
    import torch
    from audiotools import AudioSignal
    from audiotools.data.transforms import Compose, EEGNormalize, NotchFilter

    stages = []
    if notched:
        stages.append(NotchFilter(frequencies=frequencies, bandwidth_hz=bandwidth,
                                  pad_seconds=pad_seconds, prob=1.0))
    stages.append(EEGNormalize(target_std=target_std, clip=clip, prob=1.0))
    transform = Compose(*stages)
    result = np.empty_like(raw, dtype=np.float32)
    # One window at a time matches freeze_inputs and bounds CPU memory usage.
    with torch.inference_mode():
        for index, window in enumerate(raw):
            signal = AudioSignal(torch.from_numpy(window.copy())[None, None], rate)
            kwargs = transform.instantiate(np.random.RandomState(0), signal=signal)
            result[index] = transform(signal, **kwargs).audio_data[0, 0].cpu().numpy()
    return result


def prepare_paired_inputs(args, bands):
    """Freeze source windows once, then branch before normalization/clipping."""
    if not args.config or args.inputs or args.checkpoints or args.checkpoint_list or args.labels:
        raise ValueError("--prepare-paired-inputs requires --config and no checkpoints, --labels, or --inputs")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    settings = dict(frequencies=args.notch_frequencies, bandwidth=args.notch_bandwidth,
                    pad_seconds=args.notch_pad_seconds, target_std=args.target_std, clip=args.clip)
    if (not math.isfinite(args.target_std) or args.target_std <= 0 or
            not math.isfinite(args.clip) or args.clip <= 0):
        raise ValueError("--target-std and --clip must be positive and finite")
    # Validate notch settings before reading the dataset.
    from audiotools.data.transforms import NotchFilter
    NotchFilter(frequencies=args.notch_frequencies, bandwidth_hz=args.notch_bandwidth,
                pad_seconds=args.notch_pad_seconds)
    raw, rate, provenance = freeze_inputs(args.config, args.windows, args.duration, args.seed,
                                        raw=True, split=args.split or "val")
    nperseg, trim = round(args.stft_seconds * rate), round(args.trim_seconds * rate)
    validate_inputs(raw, rate, nperseg, trim, bands)
    if max(args.notch_frequencies) >= rate / 2 or args.notch_bandwidth >= rate / 2:
        raise ValueError("Notch frequencies and bandwidth must be below Nyquist")
    output.mkdir(parents=True, exist_ok=True)
    raw_hash = fingerprint(raw)
    np.savez_compressed(output / "raw_windows.npz", raw_audio=raw, sample_rate=rate,
                        provenance_json=json.dumps(provenance, default=json_default))
    manifest = {"status": "preparing", "arguments": vars(args), "raw_input_sha256": raw_hash,
                "windows": len(raw), "sample_rate": rate, "settings": settings,
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "conditions": {}}
    save_json(output / "manifest.json", manifest)
    for condition in ("unnotched", "notched"):
        print(f"Preparing {condition}: {len(raw)} shared source windows", flush=True)
        audio = prepare_condition(raw, rate, notched=condition == "notched", **settings)
        validate_inputs(audio, rate, nperseg, trim, bands)
        pipeline = (["NotchFilter"] if condition == "notched" else []) + ["EEGNormalize"]
        condition_provenance = dict(provenance, condition=condition, pipeline=pipeline,
                                    evaluation_settings=settings, raw_input_sha256=raw_hash,
                                    units="model input units after explicit evaluation transforms")
        filename = f"{condition}.npz"
        np.savez_compressed(output / filename, audio=audio, sample_rate=rate,
                            provenance_json=json.dumps(condition_provenance, default=json_default))
        manifest["conditions"][condition] = {"file": filename, "input_sha256": fingerprint(audio),
                                               "pipeline": pipeline}
    manifest["status"] = "complete"
    save_json(output / "manifest.json", manifest)
    print(f"Prepared paired inputs in {output}; evaluate with --inputs unnotched.npz or notched.npz", flush=True)


def validate_inputs(x, rate, nperseg, trim, bands):
    if x.ndim != 2 or not x.shape[0] or not np.isfinite(x).all():
        raise ValueError("Inputs must be finite nonempty [windows, samples] arrays")
    if rate <= 0 or nperseg < 4 or trim < 0 or x.shape[1] - 2 * trim < nperseg:
        raise ValueError("Scored window is shorter than one STFT segment; reduce trimming or use longer inputs")
    for name, low, high in bands:
        if high > rate / 2:
            raise ValueError(f"Band {name} extends past Nyquist ({rate / 2:g} Hz)")
        if (high - low) < rate / nperseg or low * nperseg / rate < 2:
            raise ValueError(f"Band {name} needs a longer STFT segment (at least two cycles at {low:g} Hz)")


def spectrum(x, rate, nperseg):
    # No synthetic boundary frames or zero-padding. Spectrum magnitude is in
    # input units; PSD uses the exact Hann window energy and one-sided factors.
    f, t, z = signal.stft(x, fs=rate, window="hann", nperseg=nperseg,
                          noverlap=nperseg // 2, nfft=nperseg, detrend=False,
                          boundary=None, padded=False, axis=-1, scaling="spectrum")
    window = signal.get_window("hann", nperseg)
    psd_frames = abs(z) ** 2 * window.sum() ** 2 / (rate * (window ** 2).sum())
    factors = np.ones(len(f))
    factors[1:-1 if nperseg % 2 == 0 else None] = 2
    psd_frames *= factors[:, None]
    return f, t, z, psd_frames.mean(axis=-1)


def integrate_band(psd, frequencies, low, high):
    """Piecewise-linear PSD integration with exact shared band boundaries."""
    inside = (frequencies > low) & (frequencies < high)
    grid = np.r_[low, frequencies[inside], high]
    values = np.stack([np.interp(grid, frequencies, row) for row in np.atleast_2d(psd)])
    return np.trapz(values, grid, axis=-1)


def coherence_gain(zx, zy, rate, nperseg, density_floor=1e-10):
    """Pool cross/auto spectra over windows AND frames before taking ratios.

    Do not average single-frame coherences (identically one). This is an
    energy-weighted ensemble diagnostic, not a per-window confidence estimate.
    Cross spectrum is conj(target) * reconstruction, so phase is output minus input.
    """
    window = signal.get_window("hann", nperseg)
    scale = np.full(zx.shape[1], window.sum()**2 / (rate*(window**2).sum()))
    scale[1:-1 if nperseg % 2 == 0 else None] *= 2
    pxx = (abs(zx)**2).mean(axis=(0,2))*scale
    pyy = (abs(zy)**2).mean(axis=(0,2))*scale
    pxy = (zx.conj()*zy).mean(axis=(0,2))*scale
    count = zx.shape[0]*zx.shape[2]
    target_valid = (pxx > density_floor) & (count >= 2)
    coherent_valid = target_valid & (pyy > density_floor)
    coherence = np.full(pxx.shape, np.nan)
    gain = np.full(pxx.shape, np.nan)
    rms = np.full(pxx.shape, np.nan)
    phase = np.full(pxx.shape, np.nan)
    coherence[coherent_valid] = np.clip(abs(pxy[coherent_valid])**2 /
                                       (pxx[coherent_valid]*pyy[coherent_valid]), 0, 1)
    gain[target_valid] = abs(pxy[target_valid])/pxx[target_valid]
    rms[target_valid] = np.sqrt(pyy[target_valid]/pxx[target_valid])
    phase_valid = coherent_valid & (abs(pxy) > density_floor)
    phase[phase_valid] = np.angle(pxy[phase_valid], deg=True)
    return {"coherence": coherence, "coherent_gain": gain, "rms_gain": rms,
            "transfer_phase_deg": phase, "coherence_valid": coherent_valid,
            "gain_valid": target_valid, "spectral_observations": np.asarray(count)}


def analyze(x, y, rate, nperseg, trim, bands, amplitude_floor, power_floor, coherence_floor=1e-10):
    if x.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("Reconstruction must be finite and match the input shape")
    region = slice(trim, -trim if trim else None)
    x, y = x[:, region].astype(np.float64), y[:, region].astype(np.float64)
    f, _, zx, px = spectrum(x, rate, nperseg)
    _, _, zy, py = spectrum(y, rate, nperseg)
    _, _, _, pe = spectrum(y - x, rate, nperseg)
    mag_error = np.mean(abs(abs(zy) - abs(zx)), axis=-1)
    log_error = np.mean(abs(np.log10(np.maximum(abs(zy), amplitude_floor)) -
                            np.log10(np.maximum(abs(zx), amplitude_floor))), axis=-1)
    wave = np.mean(abs(y - x), axis=-1)
    rmse = np.sqrt(np.mean((y - x) ** 2, axis=-1))
    band_data = {}
    for name, low, high in bands:
        target = integrate_band(px, f, low, high)
        reconstructed = integrate_band(py, f, low, high)
        db = 10 * np.log10((reconstructed + power_floor) / (target + power_floor))
        band_data[name] = {"target_power": target, "reconstructed_power": reconstructed,
                           "signed_db": db, "absolute_db": abs(db),
                           "absolute_power_error": abs(reconstructed - target),
                           "target_below_floor": target < power_floor}
    return {"frequency": f, "target_psd": px, "reconstructed_psd": py,
            "residual_psd": pe, "magnitude_error": mag_error, "log_magnitude_error": log_error,
            "waveform_mae": wave, "waveform_rmse": rmse, "bands": band_data,
            **coherence_gain(zx, zy, rate, nperseg, coherence_floor)}


def select_examples(results, count):
    """Shared windows: typical, p90, worst, and largest paired MAE difference."""
    scores = np.stack([r["waveform_mae"] for r in results])
    pooled = scores.mean(axis=0)
    ordered = np.argsort(pooled, kind="stable")
    choices = [(ordered[int(round((len(ordered) - 1) * q))], title)
               for q, title in ((.5, "typical"), (.9, "p90 error"), (1., "worst error"))]
    if len(results) > 1:
        difference = np.max(abs(scores[1:] - scores[0]), axis=0)
        choices.append((int(np.argmax(difference)), "largest MAE difference from baseline"))
    choices += [(i, "error quantile") for i in ordered[np.linspace(0, len(ordered) - 1,
                                                                 min(count, len(ordered)), dtype=int)]]
    selected = []
    for index, reason in choices:
        previous = next((j for j, (i, _) in enumerate(selected) if i == int(index)), None)
        if previous is None:
            selected.append((int(index), reason))
        elif reason != "error quantile":
            selected[previous] = (int(index), selected[previous][1] + "; " + reason)
        if len(selected) == min(count, len(ordered)):
            break
    return selected


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9))}


def export_metrics(output, labels, results, bands):
    summaries, windows, frequencies, paired = [], [], [], []
    transfer = []
    for label, result in zip(labels, results):
        for key in ("waveform_mae", "waveform_rmse", "magnitude_error", "log_magnitude_error"):
            values = result[key]
            if values.ndim == 2:
                values = values.mean(axis=-1)
            summaries.append({"checkpoint": label, "metric": key, **statistics(values)})
        for band, _, _ in bands:
            for key, values in result["bands"][band].items():
                summaries.append({"checkpoint": label, "metric": f"{band}/{key}", **statistics(values)})
        for i in range(len(result["waveform_mae"])):
            row = {"checkpoint": label, "window": i, "waveform_mae": result["waveform_mae"][i],
                   "waveform_rmse": result["waveform_rmse"][i],
                   "magnitude_error": result["magnitude_error"][i].mean(),
                   "log_magnitude_error": result["log_magnitude_error"][i].mean()}
            for band, _, _ in bands:
                row.update({f"{band}/{k}": v[i] for k, v in result["bands"][band].items()})
            windows.append(row)
        keys = ("target_psd", "reconstructed_psd", "residual_psd", "magnitude_error", "log_magnitude_error")
        for j, frequency in enumerate(result["frequency"]):
            transfer.append({"checkpoint": label, "frequency_hz": frequency,
                             **{k: result[k][j] for k in ("coherence", "coherent_gain", "rms_gain",
                                 "transfer_phase_deg", "coherence_valid", "gain_valid")},
                             "spectral_observations": int(result["spectral_observations"])})
            frequencies.append({"checkpoint": label, "frequency_hz": frequency,
                                **{f"{k}_{stat}": value for k in keys
                                   for stat, value in statistics(result[k][:, j]).items()}})
        if result is not results[0]:
            for key in ("waveform_mae", "magnitude_error", "log_magnitude_error"):
                delta = result[key] - results[0][key]
                if delta.ndim == 2:
                    delta = delta.mean(axis=-1)
                paired.append({"checkpoint": label, "baseline": labels[0], "metric": key,
                               "fraction_windows_improved": float(np.mean(delta < 0)), **statistics(delta)})
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per_window.csv", windows)
    write_csv(output / "frequency.csv", frequencies)
    write_csv(output / "paired_differences.csv", paired)
    write_csv(output / "coherence_gain.csv", transfer)


def plot_results(output, labels, results, bands, x, reconstructions, rate, nperseg,
                 trim, examples, amplitude_floor, power_floor, fmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(labels))]
    def finish(fig, filename):
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, key, title in zip(axes.flat,
                             ("waveform_mae", "waveform_rmse", "magnitude_error", "log_magnitude_error"),
                             ("Waveform MAE", "Waveform RMSE", "STFT magnitude error", "STFT log10-magnitude error")):
        values = [r[key].mean(-1) if r[key].ndim == 2 else r[key] for r in results]
        ax.boxplot(values, showfliers=False)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)
        ax.tick_params(axis="x", labelrotation=25)
        ax.set_title(title + " ↓")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Window distributions (boxes: IQR; line: median; whiskers: 1.5 IQR)\nSingle-resolution spectral diagnostics; not training loss values")
    finish(fig, "summary.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for label, color, result in zip(labels, colors, results):
        f = result["frequency"]
        axes[0].plot(f, result["coherence"], label=label, color=color)
        axes[1].plot(f, result["coherent_gain"], label=label + " coherent", color=color)
        axes[1].plot(f, result["rms_gain"], label=label + " RMS", color=color, ls="--", alpha=.7)
        axes[2].plot(f, result["transfer_phase_deg"], label=label, color=color)
    axes[0].set(ylabel="Magnitude-squared coherence", ylim=(0,1.02))
    axes[1].set_ylabel("Amplitude gain")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].axhline(1, color="black", lw=.8)
    axes[2].set(ylabel="Output − input phase (degrees)", ylim=(-180,180), xlabel="Frequency (Hz)")
    axes[2].axhline(0, color="black", lw=.8)
    for ax in axes:
        ax.set_xlim(0, fmax)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Pooled cross-spectral diagnostics across windows and STFT frames\n"
                 "Undefined low-power bins omitted; phase is useful primarily where coherence is high")
    finish(fig, "coherence_gain.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    f = results[0]["frequency"]
    axes[0, 0].semilogy(f, np.maximum(results[0]["target_psd"].mean(0), 1e-20), color="black", label="Target")
    for label, color, result in zip(labels, colors, results):
        for ax, key in ((axes[0, 0], "reconstructed_psd"), (axes[0, 1], "residual_psd")):
            ax.semilogy(f, np.maximum(result[key].mean(0), 1e-20), color=color, label=label)
        for ax, key in ((axes[0, 2], "magnitude_error"), (axes[1, 0], "log_magnitude_error")):
            ax.plot(f, result[key].mean(0), color=color, label=label)
        if result is not results[0]:
            for ax, key in ((axes[1, 1], "magnitude_error"), (axes[1, 2], "log_magnitude_error")):
                ax.plot(f, (result[key] - results[0][key]).mean(0), color=color, label=label)
    titles = ("Target / reconstruction PSD [input²/Hz]", "Residual PSD [input²/Hz]",
              "Mean magnitude error [input units]", "Mean absolute log10-magnitude error",
              "Magnitude error change vs first checkpoint", "Log-magnitude error change vs first checkpoint")
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel="Frequency (Hz)", xlim=(0, fmax))
        ax.grid(alpha=.2)
    for ax in axes[1, 1:]:
        ax.axhline(0, color="black", lw=.8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Means over identical windows; negative paired changes indicate improvement")
    finish(fig, "frequency_errors.png")

    fig, axes = plt.subplots(len(bands), 3, figsize=(15, 3 * len(bands)), squeeze=False, layout="constrained")
    for row, (band, low, high) in enumerate(bands):
        for ax, key in zip(axes[row, :2], ("signed_db", "absolute_db")):
            ax.boxplot([r["bands"][band][key] for r in results], showfliers=False)
            ax.set_xticks(np.arange(1, len(labels) + 1), labels)
            ax.set_title(f"{band} {low:g}–{high:g} Hz: {key.replace('_', ' ')}")
            ax.tick_params(axis="x", labelrotation=25)
            ax.axhline(0, color="gray", lw=.8)
        ax = axes[row, 2]
        for label, color, result in zip(labels, colors, results):
            data = result["bands"][band]
            ax.scatter(10 * np.log10(data["target_power"] + power_floor),
                       10 * np.log10(data["reconstructed_power"] + power_floor),
                       s=5, alpha=.25, color=color, label=label)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=.8)
        ax.set(xlabel="Target power (dB re 1 input²)", ylabel="Reconstructed power (dB)")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Band-power fidelity in normalized model-input units; positive signed error = excess power")
    finish(fig, "band_power.png")

    for index, reason in examples:
        series = [x[index]] + [y[index] for y in reconstructions]
        spectra = [spectrum(v[None], rate, nperseg) for v in series]
        db = [20 * np.log10(np.maximum(abs(z[0]), amplitude_floor)) for _, _, z, _ in spectra]
        vmax = max(float(a.max()) for a in db)
        vmin = max(20 * np.log10(amplitude_floor), vmax - 80)
        fig, axes = plt.subplots(len(series), 2, figsize=(14, 2.6 * len(series)), squeeze=False, layout="constrained")
        times = np.arange(x.shape[1]) / rate
        ylim = max(float(np.max(abs(v))) for v in series) * 1.05 + 1e-8
        for row, (label, values, spec, magnitude) in enumerate(zip(["Target"] + labels, series, spectra, db)):
            ax = axes[row, 0]
            if row:
                ax.plot(times, x[index], color="black", alpha=.4, lw=.6, label="Target")
            ax.plot(times, values, lw=.7, label=label)
            if trim:
                ax.axvspan(0, trim / rate, color="gray", alpha=.15)
                ax.axvspan((x.shape[1] - trim) / rate, x.shape[1] / rate, color="gray", alpha=.15)
            title = label if row == 0 else f"{label}: scored MAE={results[row - 1]['waveform_mae'][index]:.5f}"
            ax.set(title=title, xlabel="Time (s)", ylabel="Input units", ylim=(-ylim, ylim))
            ax.legend(fontsize=7, loc="upper right")
            f, t, _, _ = spec
            mesh = axes[row, 1].pcolormesh(t, f, magnitude, shading="auto", vmin=vmin, vmax=vmax, cmap="magma")
            axes[row, 1].set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, fmax))
        fig.colorbar(mesh, ax=list(axes[:, 1]), label="STFT magnitude (dB re 1 input unit)")
        fig.suptitle(f"Window {index}: {reason}; shaded waveform edges excluded from metrics")
        finish(fig, f"example_{index:05d}.png")


def reconstruct(model, audio, sample_rate, bottleneck_mode="quantized"):
    """Continuous diagnostic preserves projections; only discrete VQ is bypassed.

    Caller supplies eval mode and inference_mode. No model state or weights are
    changed, so the same checkpoint can be evaluated normally afterward.
    """
    if bottleneck_mode == "quantized":
        return model(audio, sample_rate)["audio"]
    if bottleneck_mode != "continuous":
        raise ValueError(f"Unknown bottleneck mode: {bottleneck_mode}")
    if len(model.quantizer.quantizers) != 1:
        raise ValueError("Continuous mode requires exactly one codebook")
    length = audio.shape[-1]
    z = model.encoder(model.preprocess(audio, sample_rate))
    quantizer = model.quantizer.quantizers[0]
    z_continuous = quantizer.out_proj(quantizer.in_proj(z))
    return model.decode(z_continuous)[..., :length]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="*", help="Run directories, checkpoint tags, dac folders, or model .pth files")
    p.add_argument("--checkpoint-list", type=Path, help="YAML list of paths or {path: ..., label: ...} entries; paths relative to YAML")
    p.add_argument("--labels", nargs="+", help="Unique labels in checkpoint order; first checkpoint is paired baseline")
    p.add_argument("--output", type=Path, required=True, help="New/empty output directory")
    p.add_argument("--config", type=Path, help="Resolved training config for shared validation data; default: first checkpoint's config")
    p.add_argument("--split", choices=("train", "val"), default=None, help="Dataset/transform scope when loading config (default: val); disallowed with --inputs")
    p.add_argument("--inputs", type=Path, help="Reuse frozen NPZ with audio [N,T], sample_rate, optional provenance_json; no re-normalization")
    p.add_argument("--prepare-paired-inputs", action="store_true", help="Prepare unnotched/notched NPZs from the same raw windows, then exit; requires --config, no checkpoints")
    p.add_argument("--notch-frequencies", type=float, nargs="+", default=[50., 60., 100., 120.], help="Preparation only: notch centers in Hz")
    p.add_argument("--notch-bandwidth", type=float, default=2., help="Preparation only: single-pass notch bandwidth in Hz")
    p.add_argument("--notch-pad-seconds", type=float, default=1., help="Preparation only: reflection padding")
    p.add_argument("--target-std", type=float, default=.3, help="Preparation only: per-window normalization target before clipping")
    p.add_argument("--clip", type=float, default=1., help="Preparation only: symmetric clipping limit after normalization")
    p.add_argument("--windows", type=int, default=2500, help="Windows to freeze from config (default: 2500; ignored with --inputs)")
    p.add_argument("--duration", type=float, help="Override validation excerpt duration in seconds")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", help="cpu or cuda:N (default: cpu)")
    p.add_argument("--bottleneck-mode", choices=("quantized", "continuous"), default="quantized",
                   help="Inference only: continuous skips codebook lookup but preserves learned projections; requires one codebook")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--stft-seconds", type=float, default=2.0)
    p.add_argument("--trim-seconds", type=float, default=.5, help="Exclude this much at each end from every metric")
    p.add_argument("--amplitude-floor", type=float, default=1e-5, help="Log10-STFT magnitude clamp in model input units")
    p.add_argument("--power-floor", type=float, default=1e-10, help="Additive band-power floor in squared model input units")
    p.add_argument("--coherence-floor", type=float, default=1e-10, help="Minimum pooled PSD in input units squared/Hz for coherence/gain; undefined bins are NaN")
    p.add_argument("--bands", default=DEFAULT_BANDS, help="Comma-separated name:low:high Hz")
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--fmax", type=float, default=128., help="Plot frequency limit only; CSV includes all frequencies")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.prepare_paired_inputs and args.bottleneck_mode != "quantized":
        raise ValueError("--bottleneck-mode continuous requires model evaluation, not input preparation")
    if args.inputs and (args.config or args.duration is not None or args.split is not None):
        raise ValueError("--inputs cannot be combined with --config, --duration, or --split; split comes from frozen provenance")
    for name in ("windows", "batch_size", "threads", "examples", "stft_seconds", "amplitude_floor", "power_floor", "coherence_floor", "fmax"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.trim_seconds) or args.trim_seconds < 0:
        raise ValueError("--trim-seconds must be finite and nonnegative")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("--duration must be positive and finite")
    import torch
    import yaml
    from dac import DAC
    torch.set_num_threads(args.threads)
    if args.prepare_paired_inputs:
        prepare_paired_inputs(args, bands=parse_bands(args.bands))
        return
    entries = [{"path": path} for path in args.checkpoints]
    if args.checkpoint_list:
        entries_from_file = yaml.safe_load(args.checkpoint_list.read_text())
        if not isinstance(entries_from_file, list):
            raise ValueError("Checkpoint YAML must be a list")
        for entry in entries_from_file:
            entry = {"path": entry} if isinstance(entry, str) else dict(entry)
            entry["path"] = str(args.checkpoint_list.resolve().parent / entry["path"])
            entries.append(entry)
    if not entries:
        raise ValueError("Provide at least one checkpoint")
    paths = [resolve_checkpoint(entry["path"]) for entry in entries]
    labels = args.labels or [entry.get("label", f"{i:02d}_{Path(entry['path']).name}") for i, entry in enumerate(entries)]
    if len(labels) != len(paths) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique label per checkpoint")
    bands = parse_bands(args.bands)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as data:
            x = np.asarray(data["audio"], dtype=np.float32)
            rate = float(data["sample_rate"])
            if not math.isfinite(rate) or rate <= 0 or rate != int(rate):
                raise ValueError("sample_rate must be a positive integer")
            rate = int(rate)
            provenance = json.loads(str(data["provenance_json"])) if "provenance_json" in data else {"source": str(args.inputs.resolve())}
    else:
        config_path = args.config or find_config(paths[0])
        print(f"Freezing {args.split or 'val'} inputs using {config_path}", flush=True)
        x, rate, provenance = freeze_inputs(config_path, args.windows, args.duration, args.seed,
                                          split=args.split or "val")
    nperseg, trim = round(args.stft_seconds * rate), round(args.trim_seconds * rate)
    validate_inputs(x, rate, nperseg, trim, bands)
    np.savez_compressed(output / "inputs.npz", audio=x, sample_rate=rate,
                        provenance_json=json.dumps(provenance, default=json_default))
    save_json(output / "manifest.json", {
        "status": "running", "arguments": vars(args), "input_sha256": fingerprint(x),
        "sample_rate": rate, "windows": len(x), "samples_per_window": x.shape[1],
        "scored_samples": x.shape[1] - 2 * trim, "stft_samples": nperseg,
        "frequency_spacing_hz": rate / nperseg, "bands": bands,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
        "checkpoints": [{"label": label, "path": str(path), "size": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns} for label, path in zip(labels, paths)],
        "provenance": provenance,
        "coherence_estimator": "Pool auto/cross spectra over windows and Hann STFT frames before ratios; 50% overlap, no detrending. Energy-weighted ensemble, not mean per-window coherence. NaN denotes undefined bins; observations are not independent replicates.",
        "interpretation": "Descriptive window statistics, not independent-subject confidence intervals. Band powers are in transformed input units. Spectral metrics differ from training losses.",
    })
    results, reconstructions = [], []
    for i, (label, path) in enumerate(zip(labels, paths)):
        if path.suffix == ".npz":
            # Precomputed reconstruction from a non-DAC coder (e.g. scripts/mdct_coder.py).
            # It must have been produced from exactly these input windows.
            print(f"Evaluating {label} (precomputed reconstruction): {path}", flush=True)
            with np.load(path, allow_pickle=False) as data:
                y = np.asarray(data["reconstruction"], dtype=np.float32)
                if "input_sha256" in data and str(data["input_sha256"]) != fingerprint(x):
                    raise ValueError(f"{label}: reconstruction was made from different input windows")
            if y.shape != x.shape:
                raise ValueError(f"{label}: reconstruction shape {y.shape} != input shape {x.shape}")
        else:
            print(f"Evaluating {label} ({args.bottleneck_mode} bottleneck): {path}", flush=True)
            model = DAC.load(str(path), strict=True).to(args.device).eval()
            if model.sample_rate != rate:
                raise ValueError(f"{label}: checkpoint sample rate {model.sample_rate} != input rate {rate}")
            y = np.empty_like(x)
            with torch.inference_mode():
                for start in range(0, len(x), args.batch_size):
                    batch = torch.from_numpy(x[start:start + args.batch_size, None]).to(args.device)
                    prediction = reconstruct(model, batch, rate, args.bottleneck_mode)
                    if prediction.shape != batch.shape:
                        raise ValueError(f"{label}: unexpected reconstruction shape {prediction.shape}")
                    y[start:start + len(batch)] = prediction[:, 0].cpu().numpy()
                    if (start // args.batch_size + 1) % 25 == 0:
                        print(f"  Reconstructed {min(start + len(batch), len(x))}/{len(x)} windows", flush=True)
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        result = analyze(x, y, rate, nperseg, trim, bands, args.amplitude_floor, args.power_floor, args.coherence_floor)
        results.append(result)
        reconstructions.append(y)
        np.savez_compressed(output / f"checkpoint_{i:02d}.npz", reconstruction=y,
                            **{k: v for k, v in result.items() if k != "bands"},
                            **{f"band_{band}_{key}": value for band, data in result["bands"].items() for key, value in data.items()})
        print(f"  waveform MAE={result['waveform_mae'].mean():.6f}, log-magnitude error={result['log_magnitude_error'].mean():.6f}", flush=True)
    export_metrics(output, labels, results, bands)
    examples = select_examples(results, args.examples)
    save_json(output / "examples.json", [{"window": i, "reason": reason} for i, reason in examples])
    plot_results(output, labels, results, bands, x, reconstructions, rate, nperseg, trim,
                 examples, args.amplitude_floor, args.power_floor, min(args.fmax, rate / 2))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["status"] = "complete"
    save_json(output / "manifest.json", manifest)
    print(f"Evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
