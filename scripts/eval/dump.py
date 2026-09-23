"""Stage 1: run labeled EEGBCI windows through a DAC checkpoint, saving in the
standard DAC formats (mirroring `python -m dac encode` / `decode`):

  results/<name>/
    input/<task>/<subject>/<rec>_s<start>.wav    preprocessed input window (64-ch wav)
    encoded/<task>/<subject>/<rec>_s<start>.dac  DACFile (uint16 codes + metadata)
    decoded/<task>/<subject>/<rec>_s<start>.wav  codec reconstruction (64-ch wav)
    <task>_windows.parquet                       labels + relative file stem per window
    weights_snapshot.pth / model_kwargs.json     frozen copy of the checkpoint

A `.dac` file holds codes shaped [channels, n_codebooks, frames] and loads with
`dac.DACFile.load`. Continuous latents are not stored (stock DAC doesn't); they are
recoverable from the codes + snapshot, which is what classify.py does.

Usage:
  python scripts/eval/dump.py --ckpt runs/dac-eeg_alice_jm_3/latest/dac/weights.pth \
      --tasks eyes,fist_lr,fists_feet --device cuda:0
"""
import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
try:  # line-buffer stdout so progress is visible live in slurm logs, not in bursts
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from pathlib import Path

# Cap BLAS/OpenMP threads BEFORE numpy/torch import. This stage is GPU work; left
# uncapped, MNE's filtering and numpy fan out over every core on the box.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
# See classify.py: numpy huge-page requests stall in compaction on this node.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import mne
import numpy as np
import pandas as pd
import torch
from audiotools import AudioSignal

torch.set_num_threads(4)
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dac.model import DAC, DACFile  # noqa: E402

EEGBCI = Path("/data/groups/bci/datasets/processed/v8_sets/classifier_eval/eegbci")
SR = 256
TARGET_STD = 0.3  # EEGNormalize defaults used in every training config
CLIP = 1.0
BANDPASS = (1.0, 40.0)


def load_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kwargs = blob["metadata"]["kwargs"]
    model = DAC(**kwargs)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, kwargs


def preprocess_file(path):
    """Read one .fif, interpolate bads, bandpass -> (data[C, T] float32, ch_names)."""
    raw = mne.io.read_raw_fif(path, preload=True, verbose="error")
    raw.pick("eeg", verbose="error")
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    raw.filter(*BANDPASS, verbose="error")
    names = [n.rstrip(".").upper() for n in raw.ch_names]
    return raw.get_data().astype(np.float32), names


def normalize(x):
    """EEGNormalize, replicated exactly: per-channel per-window std -> 0.3, clip +-1.

    Returns the normalized window and the per-channel gain in dB. The codec must see
    normalized input (that is how it was trained), but forcing every channel to the same
    variance erases the relative channel amplitudes that CSP decodes, so the gain is kept
    as side information and restored after decoding -- exactly what stock DAC uses
    DACFile.input_db for.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True).clip(min=1e-8)
    norm = np.clip((x - mean) * (TARGET_STD / std), -CLIP, CLIP)
    gain_db = 20.0 * np.log10(std[:, 0] / TARGET_STD)  # [C], undo with 10**(db/20)
    return norm, gain_db.astype(np.float32)


def window_stem(row):
    rec = Path(row.path).stem.replace("_raw", "")
    return f"{row.task}/{row.subject}/{rec}_s{row.start_sample}"


def array_cache_paths(out_dir, task):
    d = out_dir / f"{task}_arrays"
    return d, {k: d / f"{k}.npy" for k in ("orig", "recon", "codes", "gains")}


@torch.no_grad()
def dump_task(df, task, model, kwargs, out_dir, device, limit, fraction, subject_fraction):
    rows = df[df.task == task].reset_index(drop=True)
    if subject_fraction > 1:
        # every k-th SUBJECT, all their windows: preserves per-subject trial counts,
        # which within-subject CSP needs; preferred downsample for quick passes.
        keep = sorted(rows.subject.unique())[::subject_fraction]
        rows = rows[rows.subject.isin(keep)].reset_index(drop=True)
    if fraction > 1:
        rows = rows.iloc[::fraction].reset_index(drop=True)  # every k-th: keeps all subjects
    if limit:
        rows = rows.iloc[:limit].reset_index(drop=True)
    # Dump in path order so one .fif is opened once; the parquet is written in the same
    # order, which keeps row i aligned with row i of the consolidated arrays below.
    rows = rows.sort_values("path").reset_index(drop=True)
    parquet = out_dir / f"{task}_windows.parquet"
    cache_dir, cache_files = array_cache_paths(out_dir, task)
    if parquet.exists() and all(p.exists() for p in cache_files.values()):
        n_done = sum(1 for _ in (out_dir / "encoded" / task).rglob("*.dac"))
        if n_done == len(pd.read_parquet(parquet)) == len(rows):
            print(f"{task}: {n_done} windows already dumped — skipping (use --force to redo)")
            return
    n_q = kwargs["n_codebooks"]

    # Consolidated arrays alongside the per-window DAC/wav files. The per-window files are
    # the portable artefact; these exist because reading 3 x N small files back off Lustre
    # is orders of magnitude slower than three big sequential reads.
    hop = int(np.prod(kwargs["encoder_rates"]))
    W = len(rows)
    lengths = (rows.duration_s * SR).round().astype(int).to_numpy()
    Tmin, C = int(lengths.min()), 64
    F = Tmin // hop
    cache_dir.mkdir(parents=True, exist_ok=True)
    mm = {
        "orig": np.lib.format.open_memmap(cache_files["orig"], mode="w+",
                                          dtype=np.float16, shape=(W, C, Tmin)),
        "recon": np.lib.format.open_memmap(cache_files["recon"], mode="w+",
                                           dtype=np.float16, shape=(W, C, Tmin)),
        "codes": np.lib.format.open_memmap(cache_files["codes"], mode="w+",
                                           dtype=np.int16, shape=(W, C, n_q, F)),
        "gains": np.lib.format.open_memmap(cache_files["gains"], mode="w+",
                                           dtype=np.float32, shape=(W, C)),
    }

    cache_path, cache, ch_ref = None, None, None
    for i, r in enumerate(tqdm(list(rows.itertuples()), desc=f"codec {task}", unit="win")):
        if r.path != cache_path:
            cache, names = preprocess_file(EEGBCI / r.path)
            cache_path = r.path
            if ch_ref is None:
                ch_ref = names
            assert names == ch_ref, f"channel order differs in {r.path}"
        n = int(round(r.duration_s * SR))
        seg, gain_db = normalize(cache[:, r.start_sample : r.start_sample + n])  # [64, n], [64]

        x = torch.from_numpy(seg).unsqueeze(1).to(device)  # channels as batch [64, 1, n]
        x = model.preprocess(x, SR)
        z, codes, _, _, _ = model.encode(x)
        recon = model.decode(z)[..., :n].squeeze(1).cpu()  # [64, n]

        stem = window_stem(r)
        for sub, wave in (("input", torch.from_numpy(seg)), ("decoded", recon)):
            p = out_dir / sub / f"{stem}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            AudioSignal(wave.unsqueeze(0), SR).write(p)
        p = out_dir / "encoded" / f"{stem}.dac"
        p.parent.mkdir(parents=True, exist_ok=True)
        DACFile(
            codes=codes.cpu(),
            chunk_length=codes.shape[-1],
            original_length=n,
            input_db=torch.from_numpy(gain_db),  # per-channel gain, restored at eval time
            channels=seg.shape[0],
            sample_rate=SR,
            padding=True,
            dac_version="1.0.0",
        ).save(p)

        mm["orig"][i] = seg[:, :Tmin].astype(np.float16)
        mm["recon"][i] = recon.numpy()[:, :Tmin].astype(np.float16)
        mm["codes"][i] = codes.cpu().numpy()[..., :F].astype(np.int16)
        mm["gains"][i] = 10.0 ** (gain_db / 20.0)

    for v in mm.values():
        v.flush()
    del mm
    rows["stem"] = [window_stem(r) for r in rows.itertuples()]
    rows.to_parquet(out_dir / f"{task}_windows.parquet")
    (out_dir / f"{task}_ch_names.json").write_text(json.dumps(ch_ref))
    print(f"{task}: wrote {len(rows)} windows under {out_dir}/{{input,encoded,decoded}}/{task}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/dac-eeg_alice_jm_3/latest/dac/weights.pth")
    ap.add_argument("--tasks", default="eyes,fist_lr,fists_feet")
    ap.add_argument("--name", default=None, help="results subfolder; default derived from ckpt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N windows per task")
    ap.add_argument("--force", action="store_true",
                    help="redo everything, replacing existing results for this model name")
    ap.add_argument("--fraction", type=int, default=1,
                    help="keep every k-th window (quick checks); results get a _f<k> suffix")
    ap.add_argument("--subject-fraction", type=int, default=1,
                    help="keep every k-th subject with ALL their windows; _s<k> suffix")
    args = ap.parse_args()

    model, kwargs = load_model(args.ckpt, args.device)
    ckpt_id = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()[:8]
    name = args.name or (
        f"{kwargs['n_codebooks']}x{kwargs['codebook_size']}"
        f"_hop{int(np.prod(kwargs['encoder_rates']))}_{ckpt_id}"
        + (f"_f{args.fraction}" if args.fraction > 1 else "")
        + (f"_s{args.subject_fraction}" if args.subject_fraction > 1 else "")
    )
    # Working root: node-local /scratch when the runner provides it (Lustre-weather-proof);
    # the runner rsyncs each finished checkpoint back to scripts/eval/results/.
    root = Path(os.environ.get("EVAL_RESULTS_ROOT", Path(__file__).parent / "results"))
    out_dir = root / name
    snap = out_dir / "weights_snapshot.pth"
    if snap.exists() and not filecmp.cmp(args.ckpt, snap, shallow=False):
        if args.force:
            print(f"--force: replacing existing results for a DIFFERENT checkpoint in {out_dir}")
            shutil.rmtree(out_dir)
        else:
            sys.exit(
                f"{out_dir} holds results for a DIFFERENT checkpoint of the same architecture\n"
                f"(training probably overwrote 'latest' since). Either keep both with\n"
                f"  --name {name}_<something>   or redo with --force."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    if not snap.exists():
        shutil.copy2(args.ckpt, snap)
    # Tiny codebook tables saved separately so classify need not reload the 230 MB
    # snapshot per checkpoint just to rebuild latents from codes.
    np.savez(out_dir / "codebooks.npz",
             **{str(q): model.quantizer.quantizers[q].codebook.weight.detach().cpu().numpy()
                for q in range(kwargs["n_codebooks"])})
    (out_dir / "model_kwargs.json").write_text(json.dumps({"ckpt": str(args.ckpt), **kwargs}, indent=2))
    latest = out_dir.parent / "_latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out_dir.name)
    print(f"model {name}  ->  {out_dir}")

    df = pd.read_parquet(EEGBCI / "labels.parquet")
    for task in args.tasks.split(","):
        dump_task(df, task.strip(), model, kwargs, out_dir, args.device,
                  args.limit, args.fraction, args.subject_fraction)


if __name__ == "__main__":
    main()
