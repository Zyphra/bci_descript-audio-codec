# EEGBCI classifier eval

Answers: does the DAC codec preserve task-relevant EEG information?
Dataset: PhysioNet EEG Motor Movement/Imagery (109 subjects, 64 ch), v8-preprocessed,
labeled windows at `/data/groups/bci/datasets/processed/v8_sets/classifier_eval/eegbci/`.

This folder is gitignored (scripts + cached outputs live together).

## Pipeline

```
# Stage 1 (GPU): codec every labeled window, cache orig/recon/codes/latents per task
python scripts/eval/dump.py --ckpt runs/<run>/latest/dac/weights.pth --device cuda:0

# Stage 2 (CPU): classifiers on the cached representations
python scripts/eval/classify.py --results scripts/eval/results/<name>
```

`dump.py` snapshots the checkpoint weights into the results folder (training keeps
overwriting `latest`, results stay reproducible). Preprocessing per window, applied
identically to both arms: interpolate bads, bandpass 1-40 Hz, EEGNormalize
(per-channel std -> 0.3, clip +-1 — exactly the training transform).

## What gets scored (GroupKFold(5) over subjects — never split by window)

| representation | features / classifier |
|---|---|
| orig_wave / recon_wave | eyes: log-variance + logreg; motor: CSP(6)+LDA on 8-30 Hz |
| orig_spec / recon_spec | 5-band log power per channel + logreg |
| codes | per (channel, book) code histogram + logreg |
| latents | per (channel, dim) mean+std over frames + logreg |
| alpha_occ_* (eyes) | occipital alpha baseline — corpus reference: 76.8% |
| transfer_* | fit on orig, test on recon (distribution shift vs information loss) |

Motor tasks scored on real / imagined / all separately. `bands.csv` reports per-band
reconstruction quality (median relative power error, log-power correlation).

## Results layout

Storage mirrors stock DAC (`python -m dac encode` / `decode`): `.dac` files for codes,
`.wav` for waveforms, one file per window at `<task>/<subject>/<rec>_s<start>`:

```
results/<n_books>x<vocab>_hop<hop>/
  weights_snapshot.pth  model_kwargs.json
  input/<task>/S001/S001R01_s0.wav       preprocessed input window (64-ch wav)
  encoded/<task>/S001/S001R01_s0.dac     DACFile: uint16 codes [C, Q, frames]
  decoded/<task>/S001/S001R01_s0.wav     codec reconstruction (64-ch wav)
  {task}_windows.parquet                 subject, label, imagined, split, stem, ...
  {task}_ch_names.json  accuracy.csv  bands.csv  plots/
```

Continuous latents are not stored (stock DAC doesn't); classify.py rebuilds the
quantized latents from codes + the snapshot codebooks.
(`convert_h5_to_dac.py` migrated the original h5 caches to this layout.)
