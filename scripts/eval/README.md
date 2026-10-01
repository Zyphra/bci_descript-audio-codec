# Codec classifier eval

Answers: does a DAC checkpoint preserve task-relevant EEG information — in its
tokens, and in its reconstructions? Labeled windows for 5 datasets live under
`/data/groups/bci/datasets/processed/v8_sets/classifier_eval/` (eegbci, erpbci,
bciciv2a, seed, ssvep).

## One command

```
sbatch scripts/eval/run_eval.sh <run-name or path/to/weights.pth> ...
DEVICE=cuda:5 bash scripts/eval/run_eval.sh <ckpt> ...        # reserved GPU
```

Per checkpoint it checks the cache, codecs the labeled windows if needed
(dump.py, GPU), trains the classifiers (classify.py, CPU), then rebuilds every
figure and prints the ranking over ALL cached checkpoints. Tasks and subject
tiers are set in the EDIT-ME block (or `TASKS=...` env); the default is the
bread-and-butter probe (fist_lr + ssvep, ~9 min/checkpoint) — flip on
TASK_P300 / TASK_MOTOR4 for milestone depth and the figures grow panels
automatically.

`python scripts/eval/probe_plots.py` re-renders the figures alone (~30 s, no GPU).

## Features per task (tailored: each task gets the detector its signal needs)

Every task compares the SAME feature pipeline + classifier on the input waveform
vs the codec reconstruction; the input-vs-recon gap is the measurement. Retention
= (recon - chance) / (input - chance). All pipelines end in StandardScaler +
logistic regression (C=0.1) unless noted. Defined in classify.py TASK_CFG.

| task | signal axis | features from the waveform | dims |
|---|---|---|---|
| fist_lr, fists_feet | spatial band power | bandpass 8-30 Hz -> channel covariance -> CSP (6 filters, fit per fold) -> log-variance -> LDA | 6 |
| ssvep | spectral precision | FFT amplitude, 0.25 Hz bins, 7-32 Hz, per channel | ~100/ch |
| p300 | temporal shape | anti-aliased resample to 32 Hz -> raw time-course per channel (balanced accuracy) | 32/s/ch |
| motor_4class | spatial band power (4-class) | 8-30 Hz covariance -> log-Euclidean tangent space | ch*(ch+1)/2 |
| eyes, emotion | broadband power | log-variance per channel | 1/ch |

Also produced per run: a transfer row (train on input, test on recon — separates
distribution shift from information loss) and bands.csv (classifier-free per-band
log-power correlation + relative error between recon and input).

Alternatives (tested 2026-09-29, kept out of the default probe): a generic 5-band
power set and a ~50-per-channel "universal" set (wave="universal" in TASK_CFG)
both lower the input ceiling and badly under-report motor damage (retention reads
~80% where CSP shows 35%) — motor information lives BETWEEN channels, which
per-channel features cannot express. Universal is a reasonable extra lens for a
new task whose informative structure is unknown; EVAL_FULL_REPS=1 additionally
restores the codes/latents/spectra/no-gain rows.

## Layout

```
run_eval.sh      the entry point (Slurm header + driver)
dump.py          stage 1, GPU: codec every window, standard .dac/wav + array caches
classify.py      stage 2, CPU: classifiers on input/tokens/recon, subject-aware CV
probe_plots.py   figures + ranking (also called by run_eval.sh at the end)

results/<run>/                    one folder per checkpoint
    card.png                        the run's summary plot (fixed layout, tileable)
    <dataset>_<arch>_<hash>[_sN]/   cached data + accuracy.csv per dataset
results/_overview/                probe_summary / probe_by_task / probe_scatter
results/_analysis/                archived probe-selection study (REPORT.md + figures)
logs/                             slurm logs (gitignored, like results/)
```

Finished checkpoints are cached by weight hash and skipped on re-runs, so old
results never need recomputing and new checkpoints simply join the figures.
