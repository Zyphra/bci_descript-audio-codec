"""Build a window-matched evaluation bank on a second dataset root.

Replays the frozen raw windows of an existing bank (raw_windows.npz from
evaluate_eeg.py --prepare-paired-inputs): for every window it finds the source
channel in the original FIF by exact match, reads the same channel and offset
from the counterpart FIF under --new-root, applies EEGNormalize (target_std,
clip) and writes two NPZs usable with evaluate_eeg.py --inputs:

  <output>/<new_label>.npz       windows from the new root
  <output>/<old_label>.npz       the same windows from the original root
                                 (subset: windows whose file is missing under
                                 the new root are dropped from BOTH)

Usage:
  python scripts/make_matched_bank.py \
    --raw analysis/compare_notched_vs_unnotched_2026-10-13/notch_paired_val_inputs/raw_windows.npz \
    --old-root /data/groups/bci/datasets/processed/v8_sets \
    --new-root /data/groups/bci/datasets/processed/v8_0.5Hz_sets \
    --old-label v8 --new-label v8_0.5Hz \
    --output analysis/hpf_matched_val_bank
"""
import argparse, hashlib, json, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

def normalize(x, target_std, clip):
    x = x - x.mean(-1, keepdims=True)
    sd = np.maximum(x.std(-1, keepdims=True), 1e-8)      # EEGNormalize: std.clamp_min(1e-8)
    return np.clip(x * (target_std / sd), -clip, clip).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--old-root", required=True); ap.add_argument("--new-root", required=True)
    ap.add_argument("--old-label", default="old"); ap.add_argument("--new-label", default="new")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-std", type=float, default=0.3); ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args()
    import mne
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.raw, allow_pickle=False) as d:
        raw = np.asarray(d["raw_audio"], dtype=np.float64); rate = int(d["sample_rate"])
        prov = json.loads(str(d["provenance_json"]))
    windows = prov["windows"]; T = raw.shape[1]
    kept, new_arr, old_arr, records, missing, unmatched = [], [], [], [], [], []
    cache = {}
    def open_raw(p):
        if p not in cache:
            if len(cache) > 8: cache.clear()
            cache[p] = mne.io.read_raw_fif(p, preload=False, verbose=False)
        return cache[p]
    for i, w in enumerate(windows):
        old_path = w["path"]; off = float(w["metadata"]["offset"]); start = int(round(off * rate))
        new_path = old_path.replace(a.old_root.rstrip("/"), a.new_root.rstrip("/"), 1)
        if not os.path.exists(new_path):
            missing.append(i); continue
        r_old = open_raw(old_path); picks = mne.pick_types(r_old.info, eeg=True, exclude=[])
        x = r_old.get_data(picks=picks, start=start, stop=start + T)
        if x.shape[1] != T:
            unmatched.append(i); continue
        err = np.abs(x - raw[i][None]).max(1)
        c = int(np.argmin(err))
        if err[c] > 1e-9 * max(1.0, np.abs(raw[i]).max() / 1e-6) * 1e-6:   # exact up to float32 storage
            unmatched.append(i); continue
        ch_name = r_old.ch_names[picks[c]]
        r_new = open_raw(new_path)
        if ch_name not in r_new.ch_names:
            unmatched.append(i); continue
        y = r_new.get_data(picks=[r_new.ch_names.index(ch_name)], start=start, stop=start + T)[0]
        if len(y) != T:
            unmatched.append(i); continue
        kept.append(i); old_arr.append(x[c]); new_arr.append(y)
        records.append({"index": i, "old_path": old_path, "new_path": new_path, "offset": off, "channel": ch_name})
        if (i + 1) % 250 == 0: print(f"{i+1}/{len(windows)} windows, kept {len(kept)}", flush=True)
    old_arr = np.stack(old_arr); new_arr = np.stack(new_arr)
    base = {"source_raw": str(a.raw.resolve()), "source_provenance": {k: v for k, v in prov.items() if k != "windows"},
            "n_source_windows": len(windows), "n_kept": len(kept), "dropped_missing_file": missing,
            "dropped_unmatched": unmatched, "windows": records,
            "evaluation_settings": {"target_std": a.target_std, "clip": a.clip}, "pipeline": ["EEGNormalize"],
            "units": "model input units after explicit evaluation transforms"}
    for label, arr, root in [(a.old_label, old_arr, a.old_root), (a.new_label, new_arr, a.new_root)]:
        audio = normalize(arr, a.target_std, a.clip)
        p = dict(base, root=root, label=label)
        np.savez_compressed(a.output / f"{label}.npz", audio=audio, sample_rate=rate, provenance_json=json.dumps(p))
        print(f"wrote {a.output / (label + '.npz')}: {audio.shape}, sha256 {hashlib.sha256(audio.tobytes()).hexdigest()[:16]}", flush=True)
    json.dump(base, open(a.output / "manifest.json", "w"), indent=1)
    print(f"kept {len(kept)} of {len(windows)}; missing file {len(missing)}; unmatched {len(unmatched)}", flush=True)
    if len(kept) < len(windows):
        print("NOTE: the", a.old_label, "NPZ is a subset of the original bank; use it, not the original, for the 2x2 comparison.")

if __name__ == "__main__":
    main()
