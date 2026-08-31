# Synthetic EEG Data

This folder contains a small MNE-based synthetic EEG dataset generator and
plotting utilities for probing robustness to source frequency, amplitude,
phase, and noise.

- `generative_model.py` builds source-level sinusoidal activity, projects it to
  EEG sensors with MNE/fsaverage, and adds sensor noise.
- `plotting.py` generates waveform-space and power spectral density (PSD)
  panels under `samples/`.
- Current grids use BioSemi 32 channels, 256 Hz sampling, spatial-distance
  noise covariance w/ a Gaussian kernel, and temporal noise with `PSD ~ f^-1`.

The latest focused grid fixes `amplitude=0.001 nAm`, `source_freq=5 Hz`, and
`noise_std x15`, then varies phase from `0` to `1.75 pi`.

NEXT STEPS: should we be adding cluster logic? 
