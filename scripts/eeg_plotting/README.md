# EEG training plots

`training_plots.py` is the single active plotting module. It is called directly
by `scripts/train_eeg.py` and produces a startup augmentation gallery plus fixed
waveform, PSD, and codebook diagnostics at the configured epoch interval.

Earlier standalone plotting scripts have been removed. Previous plot output,
experiment summaries, probe results, and checkpoints remain under `runs/old/`.
