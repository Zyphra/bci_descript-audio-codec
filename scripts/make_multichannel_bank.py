"""Build a multi-channel frozen bank from an existing single-channel bank.

Replays the (file, offset) of every window in a frozen bank (raw_windows.npz from
evaluate_eeg.py --prepare-paired-inputs) and reads ALL EEG channels at that offset,
so that windows from different electrodes are time-aligned. Each channel is normalized
on its own with the loader rule (EEGNormalize: zero-mean, scale to target_std, clip),
exactly as a single-channel window would be. No notch.

Output <output>/multichannel.npz, flat so that evaluate_eeg.py --inputs,
scripts/mdct_coder.py and scripts/code_entropy.py can read 'audio' unchanged:
  audio          [n_channel_windows, T] float32    one row per (window, channel)
  window_id      [n]  int32   index into the source bank's window list
  channel_idx    [n]  int16   position of this channel within its window
  channel_name   [n]  str
  pos            [n, 3] float32  electrode location from the FIF (metres; zeros if absent)
  is_source      [n]  bool    True for the channel the single-channel bank used
  n_channels     [n_windows] int16   channels kept per source window
  sample_rate, provenance_json
Windows with fewer than --min-channels usable channels are dropped.

Usage:
  python scripts/make_multichannel_bank.py \
    --raw analysis/compare_notched_vs_unnotched_2026-10-13/notch_paired_val_inputs/raw_windows.npz \
    --output analysis/cross_channel_2026-10-23/bank
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")


def normalize(x, target_std, clip):
    x = x - x.mean(-1, keepdims=True)
    sd = np.maximum(x.std(-1, keepdims=True), 1e-8)
    return np.clip(x * (target_std / sd), -clip, clip).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-std", type=float, default=0.3); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--min-channels", type=int, default=2)
    a = ap.parse_args()
    import mne
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.raw, allow_pickle=False) as d:
        raw = np.asarray(d["raw_audio"], dtype=np.float64); rate = int(d["sample_rate"])
        prov = json.loads(str(d["provenance_json"]))
    windows = prov["windows"]; T = raw.shape[1]
    audio, wid, cidx, names, pos, is_src, nch, records, dropped = [], [], [], [], [], [], [], [], []
    cache = {}
    def open_raw(p):
        if p not in cache:
            if len(cache) > 8: cache.clear()
            cache[p] = mne.io.read_raw_fif(p, preload=False, verbose=False)
        return cache[p]
    for i, w in enumerate(windows):
        path = w["path"]; off = float(w["metadata"]["offset"]); start = int(round(off * rate))
        r = open_raw(path)
        picks = mne.pick_types(r.info, meg=False, eeg=True, exclude="bads")   # same picks as the training loader
        x = r.get_data(picks=picks, start=start, stop=start + T)
        if x.shape[1] != T or len(picks) < a.min_channels:
            dropped.append((i, "short" if x.shape[1] != T else "few_channels")); continue
        flat = ~np.isfinite(x).all(1) | (x.std(1) < 1e-12)                   # dead channels: all-constant or NaN
        err = np.abs(x - raw[i][None]).max(1); src = int(np.argmin(err))
        if err[src] > 1e-6 * max(1.0, np.abs(raw[i]).max()):
            dropped.append((i, "source_channel_not_found")); continue
        keep = np.where(~flat)[0]
        if len(keep) < a.min_channels:
            dropped.append((i, "few_channels")); continue
        xn = normalize(x[keep], a.target_std, a.clip)
        audio.append(xn); wid.extend([i] * len(keep)); cidx.extend(range(len(keep)))
        names.extend(r.ch_names[picks[k]] for k in keep)
        pos.append(np.stack([r.info["chs"][picks[k]]["loc"][:3] for k in keep]).astype(np.float32))
        is_src.extend(bool(k == src) for k in keep); nch.append(len(keep))
        records.append({"window": i, "path": path, "offset": off, "n_channels": int(len(keep)),
                        "n_dead": int(flat.sum()), "source_channel": r.ch_names[picks[src]]})
        if (i + 1) % 250 == 0: print(f"{i+1}/{len(windows)} windows, {sum(nch)} channel-windows", flush=True)
    audio = np.concatenate(audio); pos = np.concatenate(pos)
    out = a.output / "multichannel.npz"
    provenance = {"source_raw": str(a.raw.resolve()), "source_provenance": {k: v for k, v in prov.items() if k != "windows"},
                  "n_source_windows": len(windows), "n_kept_windows": len(nch), "dropped": dropped,
                  "pipeline": ["EEGNormalize"], "evaluation_settings": {"target_std": a.target_std, "clip": a.clip},
                  "windows": records}
    np.savez_compressed(out, audio=audio, window_id=np.array(wid, np.int32), channel_idx=np.array(cidx, np.int16),
                        channel_name=np.array(names), pos=pos, is_source=np.array(is_src), n_channels=np.array(nch, np.int16),
                        sample_rate=rate, provenance_json=json.dumps(provenance))
    print(f"wrote {out}: {audio.shape[0]} channel-windows from {len(nch)} windows "
          f"(channels per window min {min(nch)} median {int(np.median(nch))} max {max(nch)}); dropped {len(dropped)}", flush=True)


if __name__ == "__main__":
    main()
