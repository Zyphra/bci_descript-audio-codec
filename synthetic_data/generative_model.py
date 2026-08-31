"""Generate source-controlled synthetic EEG datasets with MNE.

The dataset is controlled in source space and projected to EEG channels
through an MNE forward model. The resulting arrays are shaped as
``(n_trials, n_channels, n_timepoints)``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def configure_runtime_cache() -> Path:
    """Keep MNE, matplotlib, and numba cache writes out of the user home."""
    cache_root = Path(tempfile.gettempdir()) / f"eeg-source-sim-{os.getuid()}"
    for name in ("home", "matplotlib", "numba", "xdg", "subjects"):
        (cache_root / name).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(cache_root / "home"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("SUBJECTS_DIR", str(cache_root / "subjects"))
    return cache_root


def require_dependency(module_name: str, install_name: Optional[str] = None) -> None:
    """Raise a concise error for optional packages used by the simulator."""
    if importlib.util.find_spec(module_name) is not None:
        return
    install_name = install_name or module_name
    raise ModuleNotFoundError(
        f"{install_name} is required for MNE fsaverage source simulations. "
        f"Install it in the active environment with: "
        f"python3 -m pip install {install_name}. "
        "If system Python blocks pip, create and activate a virtual environment first."
    )


configure_runtime_cache()

import mne  # noqa: E402
from mne.proj import make_eeg_average_ref_proj  # noqa: E402


class SyntheticSourceDataset:
    """Controlled latent-source EEG simulation dataset.

    Parameters are defined on the latent source waveform:

    ``amplitude_nam * 1e-9 * sin(2 * pi * source_freq_hz * t + phase_rad)``

    By default, noise follows the MNE simulation example: diagonal sensor-space
    covariance noise from ``mne.make_ad_hoc_cov`` is added to the simulated
    ``Raw`` object with an IIR filter. The ``spatial_distance`` mode keeps the
    same per-channel MNE noise scale, but uses a Gaussian kernel over electrode
    distances to add spatial covariance. ``noise_temporal_mode`` controls the
    temporal spectrum of the sensor noise. ``noise_std_multiplier`` scales
    covariance standard deviation: ``1.0`` is MNE's ad-hoc EEG level, ``2.0``
    doubles the standard deviation, and ``0.0`` leaves the trial clean.
    """

    DEFAULT_MONTAGES = ("biosemi32", "biosemi64", "biosemi128")

    def __init__(
        self,
        n_trials: int = 24,
        montage: str = "biosemi32",
        duration_sec: float = 2.0,
        sfreq: float = 256.0,
        frequencies_hz: Sequence[float] = (8.0, 10.0, 12.0),
        amplitudes_nam: Sequence[float] = (10.0,),
        phases_rad: Sequence[float] = (0.0,),
        noise_levels: Sequence[float] = (1.0,),
        source_label: str = "caudalmiddlefrontal-lh",
        source_location: str = "center",
        source_extent_mm: float = 10.0,
        trial_parameters: Optional[Sequence[Mapping[str, Any]]] = None,
        random_state: int = 0,
        shuffle_trials: bool = False,
        inter_trial_gap_sec: float = 0.25,
        subjects_dir: Optional[str] = None,
        fetch_fsaverage: bool = True,
        src_spacing: str = "ico-5",
        noise_mode: str = "mne_diagonal",
        noise_iir_filter: Optional[Sequence[float]] = (0.2, -0.2, 0.04),
        spatial_noise_length_scale_cm: float = 6.0,
        noise_temporal_mode: str = "mne_iir",
        noise_powerlaw_beta: float = 1.0,
        n_jobs: Optional[int] = None,
        verbose: str = "ERROR",
    ) -> None:
        self.n_trials = int(n_trials)
        self.montage = str(montage)
        self.duration_sec = float(duration_sec)
        self.sfreq = float(sfreq)
        self.frequencies_hz = tuple(float(v) for v in frequencies_hz)
        self.amplitudes_nam = tuple(float(v) for v in amplitudes_nam)
        self.phases_rad = tuple(float(v) for v in phases_rad)
        self.noise_levels = tuple(float(v) for v in noise_levels)
        self.source_label = str(source_label)
        self.source_location = str(source_location)
        self.source_extent_mm = float(source_extent_mm)
        self.trial_parameters = (
            None if trial_parameters is None else list(trial_parameters)
        )
        self.random_state = int(random_state)
        self.shuffle_trials = bool(shuffle_trials)
        self.inter_trial_gap_sec = float(inter_trial_gap_sec)
        self.subjects_dir = subjects_dir
        self.fetch_fsaverage = bool(fetch_fsaverage)
        self.src_spacing = str(src_spacing)
        self.noise_mode = self._normalize_noise_mode(noise_mode)
        self.noise_iir_filter = (
            None
            if noise_iir_filter is None
            else tuple(float(v) for v in noise_iir_filter)
        )
        self.spatial_noise_length_scale_cm = float(spatial_noise_length_scale_cm)
        self.noise_temporal_mode = self._normalize_noise_temporal_mode(
            noise_temporal_mode
        )
        self.noise_powerlaw_beta = float(noise_powerlaw_beta)
        self.n_jobs = n_jobs
        self.verbose = verbose

        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if self.montage not in mne.channels.get_builtin_montages():
            raise ValueError(
                f"Unknown montage {self.montage!r}. "
                "Use mne.channels.get_builtin_montages() to list valid names."
            )
        if self.noise_mode not in {"mne_diagonal", "spatial_distance", "iid", "none"}:
            raise ValueError(
                "noise_mode must be one of: "
                "'mne_diagonal', 'spatial_distance', 'iid', 'none'"
            )
        if self.spatial_noise_length_scale_cm <= 0:
            raise ValueError("spatial_noise_length_scale_cm must be positive")
        if self.noise_temporal_mode not in {"mne_iir", "white", "powerlaw"}:
            raise ValueError(
                "noise_temporal_mode must be one of: "
                "'mne_iir', 'white', 'powerlaw'"
            )
        if self.noise_powerlaw_beta < 0:
            raise ValueError("noise_powerlaw_beta must be non-negative")

        self.n_timepoints = int(round(self.duration_sec * self.sfreq))
        if self.n_timepoints < 2:
            raise ValueError("duration_sec * sfreq must produce at least 2 samples")
        self.times = np.arange(self.n_timepoints, dtype=np.float64) / self.sfreq

        self.info = None
        self.forward = None
        self.source_simulator = None
        self.raw_clean = None
        self._fs_dir = None
        self.events = None
        self.metadata: List[Dict[str, Any]] = []
        self.ch_names: List[str] = []
        self.X_clean = None
        self.X = None
        self.source_waveforms = None
        self.epochs = None

    @staticmethod
    def _normalize_noise_mode(noise_mode: str) -> str:
        if noise_mode == "mne_cov":
            return "mne_diagonal"
        return str(noise_mode)

    @staticmethod
    def _normalize_noise_temporal_mode(noise_temporal_mode: str) -> str:
        if noise_temporal_mode == "iir":
            return "mne_iir"
        return str(noise_temporal_mode)

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.X is None:
            raise RuntimeError("Call generate() before indexing the dataset")
        return {
            "X": self.X[index],
            "X_clean": self.X_clean[index],
            "source_waveform": self.source_waveforms[index],
            "metadata": self.metadata[index],
        }

    @property
    def shape(self) -> Optional[tuple]:
        if self.X is None:
            return None
        return self.X.shape

    def generate(
        self,
        save: bool = False,
        output_dir: str = "synthetic_data/generated",
        include_epochs: bool = False,
    ) -> "SyntheticSourceDataset":
        """Generate the dataset, optionally saving it to disk."""
        require_dependency("nibabel")

        rng = np.random.RandomState(self.random_state)
        self.metadata = self._build_trial_table(rng)
        self.source_waveforms = self._build_source_waveforms(self.metadata)
        self.events = self._build_events()

        self.info, self.ch_names = self._build_info()
        self.forward = self._build_forward(self.info)

        label = self._build_source_label()
        source_simulator = mne.simulation.SourceSimulator(
            self.forward["src"],
            tstep=1.0 / self.sfreq,
            duration=self._raw_duration_sec(),
        )
        source_simulator.add_data(label, self.source_waveforms, self.events)
        self.source_simulator = source_simulator

        raw = mne.simulation.simulate_raw(
            self.info,
            source_simulator,
            forward=self.forward,
            verbose=self.verbose,
        )
        self.raw_clean = raw
        epochs_clean = self._epoch_raw(raw)
        self.X_clean = epochs_clean.get_data(copy=True).astype(np.float32)
        self.apply_noise(random_state=self.random_state)

        if save:
            self.save(output_dir=output_dir, include_epochs=include_epochs)
        return self

    def apply_noise(
        self,
        random_state: Optional[int] = None,
    ) -> "SyntheticSourceDataset":
        """Regenerate sensor noise for an already simulated clean dataset."""
        if self.X_clean is None or self.info is None or self.events is None:
            raise RuntimeError("Call generate() before apply_noise()")

        seed = self.random_state if random_state is None else int(random_state)
        rng = np.random.RandomState(seed)
        self._update_noise_metadata()
        if self.noise_mode in {"mne_diagonal", "spatial_distance"}:
            if self.noise_temporal_mode == "powerlaw":
                self.X = self._add_powerlaw_covariance_noise(self.metadata, rng)
            else:
                if self.raw_clean is None:
                    raise RuntimeError(
                        "raw_clean is required for MNE covariance noise. "
                        "Call generate() before apply_noise()."
                    )
                self.X = self._add_mne_covariance_noise(
                    self.raw_clean,
                    self.metadata,
                    rng,
                )
        elif self.noise_mode == "iid":
            self.X = self._add_iid_trial_noise(self.X_clean, self.metadata, rng)
        else:
            self.X = self.X_clean.copy()
            for row in self.metadata:
                row["noise_rms_volts"] = 0.0

        self.epochs = mne.EpochsArray(
            self.X,
            self.info.copy(),
            events=self.events.copy(),
            event_id={"source": 1},
            tmin=0.0,
            baseline=None,
            verbose=self.verbose,
        )
        return self

    def save(
        self,
        output_dir: str = "synthetic_data/generated",
        prefix: Optional[str] = None,
        include_epochs: bool = False,
        overwrite: bool = True,
    ) -> Dict[str, str]:
        """Save generated arrays and metadata.

        The method is intentionally explicit; generation does not save unless
        ``generate(save=True)`` is used or this method is called directly.
        """
        if self.X is None:
            raise RuntimeError("Call generate() before save()")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"source_sim_{self.montage}_{self.n_trials}trials"

        paths = {
            "npz": str(output / f"{prefix}.npz"),
            "metadata_csv": str(output / f"{prefix}_metadata.csv"),
            "config_json": str(output / f"{prefix}_config.json"),
        }
        if not overwrite:
            for path in paths.values():
                if Path(path).exists():
                    raise FileExistsError(path)

        np.savez_compressed(
            paths["npz"],
            X=self.X,
            X_clean=self.X_clean,
            source_waveforms=self.source_waveforms.astype(np.float32),
            times=self.times.astype(np.float32),
            sfreq=np.array(self.sfreq, dtype=np.float32),
            ch_names=np.array(self.ch_names, dtype=object),
            montage=np.array(self.montage),
            metadata_json=np.array(json.dumps(self.metadata)),
        )
        self._write_metadata_csv(paths["metadata_csv"])
        with open(paths["config_json"], "w", encoding="utf-8") as handle:
            json.dump(self._config_dict(), handle, indent=2, sort_keys=True)

        if include_epochs:
            epochs_path = output / f"{prefix}-epo.fif"
            self.epochs.save(epochs_path, overwrite=overwrite)
            paths["epochs_fif"] = str(epochs_path)
        return paths

    def _build_trial_table(self, rng: np.random.RandomState) -> List[Dict[str, Any]]:
        if self.trial_parameters is None:
            base_rows = []
            product = itertools.product(
                self.frequencies_hz,
                self.amplitudes_nam,
                self.phases_rad,
                self.noise_levels,
            )
            for frequency_hz, amplitude_nam, phase_rad, noise_level in product:
                base_rows.append(
                    {
                        "source_freq_hz": frequency_hz,
                        "amplitude_nam": amplitude_nam,
                        "phase_rad": phase_rad,
                        "noise_std_multiplier": noise_level,
                    }
                )
        else:
            base_rows = [dict(row) for row in self.trial_parameters]

        if not base_rows:
            raise ValueError("No trial parameters were provided")

        repeats = int(math.ceil(self.n_trials / float(len(base_rows))))
        rows = [dict(row) for row in (base_rows * repeats)[: self.n_trials]]
        if self.shuffle_trials:
            rng.shuffle(rows)

        table = []
        for trial_id, row in enumerate(rows):
            source_freq_hz = float(
                row.get(
                    "source_freq_hz",
                    row.get("frequency_hz", row.get("frequency", 10.0)),
                )
            )
            amplitude_nam = float(row.get("amplitude_nam", row.get("amplitude", 10.0)))
            phase_rad = float(row.get("phase_rad", row.get("phase", 0.0)))
            noise_std_multiplier = float(
                row.get(
                    "noise_std_multiplier",
                    row.get("noise_level", row.get("noise", 0.0)),
                )
            )
            if source_freq_hz <= 0:
                raise ValueError("source_freq_hz values must be positive")
            if noise_std_multiplier < 0:
                raise ValueError("noise_std_multiplier values must be non-negative")

            metadata = dict(row)
            metadata.update(
                {
                    "trial_id": trial_id,
                    "source_freq_hz": source_freq_hz,
                    "frequency_hz": source_freq_hz,
                    "amplitude_nam": amplitude_nam,
                    "amplitude_am": amplitude_nam * 1e-9,
                    "phase_rad": phase_rad,
                    "noise_std_multiplier": noise_std_multiplier,
                    "noise_level": noise_std_multiplier,
                    "noise_covariance_multiplier": noise_std_multiplier**2,
                    "noise_mode": self.noise_mode,
                    "noise_temporal_mode": self.noise_temporal_mode,
                    "noise_powerlaw_beta": (
                        self.noise_powerlaw_beta
                        if self.noise_temporal_mode == "powerlaw"
                        else None
                    ),
                    "spatial_noise_length_scale_cm": (
                        self.spatial_noise_length_scale_cm
                        if self.noise_mode == "spatial_distance"
                        else None
                    ),
                    "duration_sec": self.duration_sec,
                    "sfreq": self.sfreq,
                    "n_timepoints": self.n_timepoints,
                    "montage": self.montage,
                    "source_label": self.source_label,
                    "source_location": self.source_location,
                    "source_extent_mm": self.source_extent_mm,
                }
            )
            table.append(metadata)
        return table

    def _update_noise_metadata(self) -> None:
        for row in self.metadata:
            row["noise_mode"] = self.noise_mode
            row["noise_temporal_mode"] = self.noise_temporal_mode
            row["noise_powerlaw_beta"] = (
                self.noise_powerlaw_beta
                if self.noise_temporal_mode == "powerlaw"
                else None
            )
            row["spatial_noise_length_scale_cm"] = (
                self.spatial_noise_length_scale_cm
                if self.noise_mode == "spatial_distance"
                else None
            )

    def _build_source_waveforms(
        self, metadata: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        waveforms = np.empty((len(metadata), self.n_timepoints), dtype=np.float64)
        for idx, row in enumerate(metadata):
            amplitude_am = float(row["amplitude_am"])
            source_freq_hz = float(row["source_freq_hz"])
            phase_rad = float(row["phase_rad"])
            waveforms[idx] = amplitude_am * np.sin(
                2.0 * np.pi * source_freq_hz * self.times + phase_rad
            )
        return waveforms

    def _build_events(self) -> np.ndarray:
        gap_samples = int(round(self.inter_trial_gap_sec * self.sfreq))
        stride = self.n_timepoints + max(0, gap_samples)
        events = np.zeros((self.n_trials, 3), dtype=int)
        events[:, 0] = stride * np.arange(self.n_trials, dtype=int)
        events[:, 2] = 1
        return events

    def _raw_duration_sec(self) -> float:
        last_sample = int(self.events[-1, 0]) + self.n_timepoints
        return last_sample / self.sfreq

    def _build_info(self):
        montage = mne.channels.make_standard_montage(self.montage)
        ch_names = list(montage.ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="eeg")
        info.set_montage(montage)
        info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
        info["projs"].append(make_eeg_average_ref_proj(info, verbose=self.verbose))
        return info, ch_names

    def _epoch_raw(self, raw):
        return mne.Epochs(
            raw,
            self.events,
            event_id={"source": 1},
            tmin=0.0,
            tmax=(self.n_timepoints - 1) / self.sfreq,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            verbose=self.verbose,
        )

    def _subjects_dir_path(self) -> Path:
        if self.subjects_dir is not None:
            return Path(self.subjects_dir).expanduser().resolve()
        return Path(os.environ["SUBJECTS_DIR"]).expanduser().resolve()

    def _fsaverage_dir(self) -> Path:
        if self._fs_dir is not None:
            return self._fs_dir

        subjects_dir = self._subjects_dir_path()
        fs_dir = subjects_dir / "fsaverage"
        if self.fetch_fsaverage:
            fs_dir = Path(
                mne.datasets.fetch_fsaverage(
                    subjects_dir=subjects_dir,
                    verbose=self.verbose,
                )
            )

        required = [
            fs_dir / "bem" / f"fsaverage-{self.src_spacing}-src.fif",
            fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif",
            fs_dir / "label" / "lh.aparc.annot",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing fsaverage simulation files. "
                "Set fetch_fsaverage=True or pass subjects_dir containing fsaverage. "
                f"Missing: {missing}"
            )
        self._fs_dir = fs_dir
        return fs_dir

    def _build_forward(self, info):
        fs_dir = self._fsaverage_dir()
        src = fs_dir / "bem" / f"fsaverage-{self.src_spacing}-src.fif"
        bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
        return mne.make_forward_solution(
            info,
            trans="fsaverage",
            src=src,
            bem=bem,
            meg=False,
            eeg=True,
            mindist=5.0,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    def _build_source_label(self):
        fs_dir = self._fsaverage_dir()
        labels = mne.read_labels_from_annot(
            "fsaverage",
            regexp=self.source_label,
            subjects_dir=fs_dir.parent,
            verbose=self.verbose,
        )
        if not labels:
            raise ValueError(
                f"No fsaverage label matched source_label={self.source_label!r}"
            )
        return mne.label.select_sources(
            "fsaverage",
            labels[0],
            location=self.source_location,
            extent=self.source_extent_mm,
            subjects_dir=fs_dir.parent,
        )

    def _add_mne_covariance_noise(
        self,
        raw_clean,
        metadata: Sequence[Mapping[str, Any]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        X = np.empty_like(self.X_clean)
        noise_std_multipliers = sorted(
            {float(row["noise_std_multiplier"]) for row in metadata}
        )

        for noise_std_multiplier in noise_std_multipliers:
            trial_indices = [
                idx
                for idx, row in enumerate(metadata)
                if float(row["noise_std_multiplier"]) == noise_std_multiplier
            ]
            if noise_std_multiplier == 0.0:
                X[trial_indices] = self.X_clean[trial_indices]
                for trial_id in trial_indices:
                    metadata[trial_id]["noise_rms_volts"] = 0.0
                continue

            raw_noisy = raw_clean.copy()
            cov = self._make_noise_covariance(raw_noisy.info, noise_std_multiplier)
            iir_filter = (
                self.noise_iir_filter
                if self.noise_temporal_mode == "mne_iir"
                else None
            )
            mne.simulation.add_noise(
                raw_noisy,
                cov,
                iir_filter=iir_filter,
                random_state=int(rng.randint(0, np.iinfo(np.int32).max)),
                verbose=self.verbose,
            )
            noisy_epochs = self._epoch_raw(raw_noisy)
            X_for_level = noisy_epochs.get_data(copy=True).astype(np.float32)
            X[trial_indices] = X_for_level[trial_indices]
            noise = X_for_level - self.X_clean
            for trial_id in trial_indices:
                metadata[trial_id]["noise_rms_volts"] = float(
                    np.sqrt(np.mean(noise[trial_id] ** 2))
                )

        return X.astype(np.float32, copy=False)

    def _add_powerlaw_covariance_noise(
        self,
        metadata: Sequence[Mapping[str, Any]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        X = self.X_clean.copy()
        noise_std_multipliers = sorted(
            {float(row["noise_std_multiplier"]) for row in metadata}
        )

        for noise_std_multiplier in noise_std_multipliers:
            trial_indices = [
                idx
                for idx, row in enumerate(metadata)
                if float(row["noise_std_multiplier"]) == noise_std_multiplier
            ]
            if noise_std_multiplier == 0.0:
                for trial_id in trial_indices:
                    metadata[trial_id]["noise_rms_volts"] = 0.0
                continue

            cov = self._make_noise_covariance(self.info, noise_std_multiplier)
            colorer = self._covariance_colorer(cov)
            for trial_id in trial_indices:
                latent = self._powerlaw_latent_noise(colorer.shape[1], rng)
                noise = colorer @ latent
                X[trial_id] += noise.astype(np.float32)
                metadata[trial_id]["noise_rms_volts"] = float(
                    np.sqrt(np.mean(noise**2))
                )

        return X.astype(np.float32, copy=False)

    def _covariance_colorer(self, cov) -> np.ndarray:
        data = np.asarray(cov["data"], dtype=np.float64)
        if bool(cov["diag"]):
            return np.diag(np.sqrt(data))

        data = 0.5 * (data + data.T)
        eigvals, eigvecs = np.linalg.eigh(data)
        tolerance = float(np.max(np.abs(eigvals))) * 1e-12
        keep = eigvals > tolerance
        if not np.any(keep):
            raise ValueError("Noise covariance has no positive eigenvalues")
        return eigvecs[:, keep] * np.sqrt(eigvals[keep])[np.newaxis, :]

    def _powerlaw_latent_noise(
        self,
        n_latents: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        freqs = np.fft.rfftfreq(self.n_timepoints, d=1.0 / self.sfreq)
        scale = np.zeros_like(freqs)
        nonzero = freqs > 0
        scale[nonzero] = freqs[nonzero] ** (-0.5 * self.noise_powerlaw_beta)

        coefficients = (
            rng.standard_normal((n_latents, len(freqs)))
            + 1j * rng.standard_normal((n_latents, len(freqs)))
        )
        coefficients *= scale[np.newaxis, :]
        coefficients[:, 0] = 0.0
        if self.n_timepoints % 2 == 0:
            coefficients[:, -1] = coefficients[:, -1].real

        noise = np.fft.irfft(coefficients, n=self.n_timepoints, axis=-1)
        noise -= noise.mean(axis=-1, keepdims=True)
        std = noise.std(axis=-1, keepdims=True)
        noise /= np.maximum(std, np.finfo(np.float64).eps)
        return noise

    def _make_noise_covariance(self, info, noise_std_multiplier: float):
        if self.noise_mode == "mne_diagonal":
            cov = mne.make_ad_hoc_cov(info)
            cov["data"] *= noise_std_multiplier**2
            return cov
        if self.noise_mode == "spatial_distance":
            return self._make_spatial_distance_covariance(info, noise_std_multiplier)
        raise RuntimeError(f"Unsupported covariance noise mode: {self.noise_mode}")

    def _make_spatial_distance_covariance(
        self, info, noise_std_multiplier: float
    ):
        base_cov = mne.make_ad_hoc_cov(info)
        names = list(base_cov["names"])
        name_to_pick = {name: idx for idx, name in enumerate(info["ch_names"])}
        try:
            picks = [name_to_pick[name] for name in names]
        except KeyError as exc:
            raise ValueError(f"Covariance channel is missing from info: {exc}") from exc

        positions = np.array(
            [info["chs"][pick]["loc"][:3] for pick in picks],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positions)):
            raise ValueError(
                "spatial_distance noise requires finite electrode positions. "
                "Use a standard montage or noise_mode='mne_diagonal'."
            )

        length_scale_m = self.spatial_noise_length_scale_cm / 100.0
        deltas = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        correlation = np.exp(-0.5 * (distances / length_scale_m) ** 2)

        # The noise knob scales standard deviation, so covariance scales by x**2.
        variances = (
            np.asarray(base_cov["data"], dtype=np.float64)
            * noise_std_multiplier**2
        )
        stds = np.sqrt(variances)
        data = correlation * np.outer(stds, stds)
        np.fill_diagonal(data, variances)
        return mne.Covariance(
            data,
            names,
            base_cov["bads"],
            base_cov["projs"],
            nfree=0,
            method="spatial_distance",
            verbose=self.verbose,
        )

    def _add_iid_trial_noise(
        self,
        X_clean: np.ndarray,
        metadata: Sequence[Mapping[str, Any]],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        X = X_clean.copy()
        eps = np.finfo(np.float32).eps
        for trial_id, row in enumerate(metadata):
            noise_std_multiplier = float(row["noise_std_multiplier"])
            if noise_std_multiplier == 0.0:
                row["noise_rms_volts"] = 0.0
                continue

            base_noise = rng.standard_normal(X[trial_id].shape).astype(np.float32)
            base_rms = float(np.sqrt(np.mean(base_noise ** 2)))
            clean_rms = float(np.sqrt(np.mean(X_clean[trial_id] ** 2)))
            noise_rms = noise_std_multiplier * max(clean_rms, eps)
            X[trial_id] += base_noise * (noise_rms / max(base_rms, eps))
            row["noise_rms_volts"] = noise_rms
        return X.astype(np.float32, copy=False)

    def _write_metadata_csv(self, path: str) -> None:
        keys = sorted({key for row in self.metadata for key in row.keys()})
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metadata)

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "montage": self.montage,
            "duration_sec": self.duration_sec,
            "sfreq": self.sfreq,
            "source_frequencies_hz": self.frequencies_hz,
            "frequencies_hz": self.frequencies_hz,
            "amplitudes_nam": self.amplitudes_nam,
            "phases_rad": self.phases_rad,
            "noise_std_multipliers": self.noise_levels,
            "noise_levels": self.noise_levels,
            "source_label": self.source_label,
            "source_location": self.source_location,
            "source_extent_mm": self.source_extent_mm,
            "random_state": self.random_state,
            "shuffle_trials": self.shuffle_trials,
            "inter_trial_gap_sec": self.inter_trial_gap_sec,
            "subjects_dir": str(self._subjects_dir_path()),
            "fetch_fsaverage": self.fetch_fsaverage,
            "src_spacing": self.src_spacing,
            "noise_mode": self.noise_mode,
            "noise_iir_filter": self.noise_iir_filter,
            "spatial_noise_length_scale_cm": self.spatial_noise_length_scale_cm,
            "noise_temporal_mode": self.noise_temporal_mode,
            "noise_powerlaw_beta": self.noise_powerlaw_beta,
        }


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_optional_float_list(value: str) -> Optional[List[float]]:
    if value.lower() in {"none", "null", ""}:
        return None
    return _parse_float_list(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--montage", default="biosemi32")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument(
        "--source-frequencies-hz",
        "--frequencies-hz",
        dest="source_frequencies_hz",
        default="8,10,12",
    )
    parser.add_argument("--amplitudes-nam", default="10")
    parser.add_argument("--phases-rad", default="0")
    parser.add_argument(
        "--noise-std-multipliers",
        "--noise-levels",
        dest="noise_std_multipliers",
        default="1",
    )
    parser.add_argument("--source-label", default="caudalmiddlefrontal-lh")
    parser.add_argument("--source-location", default="center")
    parser.add_argument("--source-extent-mm", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--shuffle-trials", action="store_true")
    parser.add_argument("--inter-trial-gap-sec", type=float, default=0.25)
    parser.add_argument("--subjects-dir", default=None)
    parser.add_argument("--no-fetch-fsaverage", action="store_true")
    parser.add_argument("--src-spacing", default="ico-5")
    parser.add_argument(
        "--noise-mode",
        choices=["mne_diagonal", "spatial_distance", "iid", "none", "mne_cov"],
        default="mne_diagonal",
    )
    parser.add_argument("--noise-iir-filter", default="0.2,-0.2,0.04")
    parser.add_argument("--spatial-noise-length-scale-cm", type=float, default=6.0)
    parser.add_argument(
        "--noise-temporal-mode",
        choices=["mne_iir", "white", "powerlaw", "iir"],
        default="mne_iir",
    )
    parser.add_argument("--noise-powerlaw-beta", type=float, default=1.0)
    parser.add_argument("--output-dir", default="synthetic_data/generated")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--include-epochs", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        dataset = SyntheticSourceDataset(
            n_trials=args.n_trials,
            montage=args.montage,
            duration_sec=args.duration_sec,
            sfreq=args.sfreq,
            frequencies_hz=_parse_float_list(args.source_frequencies_hz),
            amplitudes_nam=_parse_float_list(args.amplitudes_nam),
            phases_rad=_parse_float_list(args.phases_rad),
            noise_levels=_parse_float_list(args.noise_std_multipliers),
            source_label=args.source_label,
            source_location=args.source_location,
            source_extent_mm=args.source_extent_mm,
            random_state=args.random_state,
            shuffle_trials=args.shuffle_trials,
            inter_trial_gap_sec=args.inter_trial_gap_sec,
            subjects_dir=args.subjects_dir,
            fetch_fsaverage=not args.no_fetch_fsaverage,
            src_spacing=args.src_spacing,
            noise_mode=args.noise_mode,
            noise_iir_filter=_parse_optional_float_list(args.noise_iir_filter),
            spatial_noise_length_scale_cm=args.spatial_noise_length_scale_cm,
            noise_temporal_mode=args.noise_temporal_mode,
            noise_powerlaw_beta=args.noise_powerlaw_beta,
        ).generate(
            save=args.save,
            output_dir=args.output_dir,
            include_epochs=args.include_epochs,
        )
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Generated {dataset.montage}: X.shape={dataset.X.shape}, "
        f"source_waveforms.shape={dataset.source_waveforms.shape}"
    )
    if args.save:
        print(f"Saved dataset files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
