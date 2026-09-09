from dataclasses import dataclass
from pathlib import Path
import glob
import json
import math
import random
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

# ============================================================================
# Added: MNE fif data loader (may wanna use our .pt files or so later)
# treats channels independently
# normalized with MAD (median absolute deviation) per channel not across channels
# normalization is based on 3 X 30s segments TODO: we may want to do this for each 10s chunk
# loads 10s segments, if sam-ple size = 12, we repeat this 12 times independetly (pretty slow)
# TODO: for more efficient loading, we may want to switch to the .pt files
# ============================================================================

@dataclass
class EEGRecording:
    path: str
    raw: object
    channel_names: List[str]
    median: np.ndarray
    scale: np.ndarray


class EEGWindowDataset(Dataset):
    """Random single-electrode windows from a temporal recording split."""

    def __init__(
        self,
        recordings: Sequence[EEGRecording],
        sample_rate: int,
        window_seconds: float,
        n_examples: int,
        train_fraction: float,
        clip_mad: float,
        split: str,
        seed: int,
    ) -> None:
        self.recordings = list(recordings)
        self.window_samples = int(round(sample_rate * window_seconds))
        self.n_examples = int(n_examples)
        self.train_fraction = float(train_fraction)
        self.clip_mad = float(clip_mad)
        self.split = split
        self.seed = int(seed)
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        for recording in self.recordings:
            split_sample = int(recording.raw.n_times * self.train_fraction)
            available = split_sample if split == "train" else recording.raw.n_times - split_sample
            if available < self.window_samples:
                raise ValueError(f"{recording.path}: {split} interval is shorter than one window")

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.split == "val":
            rng = np.random.default_rng(self.seed + index)
        else:
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1) + index)
        recording = self.recordings[int(rng.integers(len(self.recordings)))]
        channel = int(rng.integers(len(recording.channel_names)))
        split_sample = int(recording.raw.n_times * self.train_fraction)
        lo, hi = (0, split_sample) if self.split == "train" else (split_sample, recording.raw.n_times)
        start = int(rng.integers(lo, hi - self.window_samples + 1))
        values = recording.raw.get_data(
            picks=[channel], start=start, stop=start + self.window_samples
        ).astype(np.float32, copy=False)
        values = (values - recording.median[channel]) / recording.scale[channel]
        values = np.clip(values, -self.clip_mad, self.clip_mad) / self.clip_mad
        return {
            "eeg": torch.from_numpy(values.copy()),
            "recording": recording.path,
            "channel": recording.channel_names[channel],
            "start_sample": start,
        }


def resolve_files(patterns: Sequence[str]) -> List[str]:
    """ADDED FOR EEG: expand FIF globs."""
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(str(Path(pattern).expanduser())))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No FIF files matched {list(patterns)}")
    return paths


def load_recordings(
    files: Sequence[str],
    sample_rate: int,
    train_fraction: float,
    normalization_seconds: float,
    normalization_chunks: int,
) -> List[EEGRecording]:
    """ADDED FOR EEG: open FIF lazily and estimate train-only robust scales."""
    recordings = []
    for path in resolve_files(files):
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        raw.pick(picks="eeg")
        if not math.isclose(float(raw.info["sfreq"]), float(sample_rate)):
            raise ValueError(f"{path}: expected {sample_rate} Hz, got {raw.info['sfreq']} Hz")
        train_stop = int(raw.n_times * train_fraction)
        total = min(int(round(normalization_seconds * sample_rate)), train_stop)
        chunk_samples = max(1, total // normalization_chunks)
        starts = np.linspace(0, max(0, train_stop - chunk_samples), normalization_chunks, dtype=int)
        calibration = np.concatenate(
            [raw.get_data(start=int(start), stop=int(start) + chunk_samples) for start in starts],
            axis=1,
        ).astype(np.float32, copy=False)
        median = np.median(calibration, axis=1, keepdims=True).astype(np.float32)
        mad = np.median(np.abs(calibration - median), axis=1, keepdims=True)
        scale = np.maximum(1.4826 * mad, np.finfo(np.float32).eps).astype(np.float32)
        recordings.append(EEGRecording(path, raw, list(raw.ch_names), median, scale))
        print(
            f"Indexed {Path(path).name}: {len(raw.ch_names)} EEG channels, "
            f"{raw.n_times / sample_rate:.1f} s at {sample_rate} Hz"
        )
    return recordings