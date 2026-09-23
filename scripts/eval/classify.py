"""Stage 2: train simple classifiers on the cached representations from dump.py.

Representations, all with subject-wise GroupKFold(5):
  orig_wave / recon_wave : eyes -> log-variance per channel + logistic regression
                           motor -> CSP(6) + LDA on 8-30 Hz
  orig_spec / recon_spec : 5-band log power per channel + logistic regression
  codes                  : per (channel, book) code histogram + logistic regression
  latents                : per (channel, dim) mean+std over frames + logistic regression
  alpha_occ (eyes only)  : occipital log alpha + broadband power + logistic regression
                           (matches the corpus reference baseline, 76.8%)
  transfer               : primary classifier fit on orig, tested on recon
Motor tasks are scored on real / imagined / all trials separately.

Usage:
  python scripts/eval/classify.py --results scripts/eval/results/5x16_hop32
"""
import argparse
import json
import os
import sys
try:  # line-buffer stdout so progress is visible live in slurm logs, not in bursts
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from pathlib import Path

# Keep this eval a polite tenant on a shared 224-core box: cap threads BEFORE numpy
# imports, otherwise every filtfilt/welch/covariance silently fans out over all cores.
N_JOBS = 8        # parallel CV folds — modest: Lustre chokes on too many workers
BLAS_THREADS = 2  # threads per fold worker  -> N_JOBS * BLAS_THREADS cores at peak
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(BLAS_THREADS))
# NumPy's huge-page requests trigger near-always-failing sync compaction on this
# fragmented node (8 MiB touch: 2.18 s vs 4-58 ms). Must be set before numpy import.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
import scipy.linalg
from scipy.signal import butter, filtfilt
from threadpoolctl import threadpool_limits

# torch, mne and the DAC package are deliberately NOT imported here: the normal path
# (array cache + codebooks.npz) doesn't need them, and their import cost dominated
# process startup. Legacy fallback paths import them lazily.

DECIM = 3        # time decimation for CSP input (256 Hz -> 85 Hz; signal is 8-30 Hz)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (13, 30), "gamma": (30, 40)}
OCCIPITAL = ["O1", "OZ", "O2", "PO3", "POZ", "PO4", "PO7", "PO8"]
SR = 256


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))


# ---------------- feature extractors ----------------
# All heavy array math runs CHUNKED over windows with preallocated outputs. This node
# (35 days up) has badly fragmented memory: any multi-MB fresh allocation can stall for
# minutes in kernel memory compaction (3.3M compact_stalls, 98% failing), while small
# allocations are instant. Chunking sidesteps the pathology; it is also cache-friendlier.

CHUNK = 16  # windows per block: keeps temporaries ~1-2 MB


def psd(x):
    """Welch PSD via chunked numpy rfft: x [W, C, T] float32 -> (freqs[129], p [W, C, 129]).

    Replaces scipy.signal.welch, whose internal allocations hit the compaction stalls
    even for small inputs. 256-sample Hann segments, 50% overlap, 1 Hz resolution.
    Constant scale factors are dropped: every consumer takes logs / ratios.
    """
    W, C, T = x.shape
    nper, step = 256, 128
    win = np.hanning(nper).astype(np.float32)
    nseg = max(1, (T - nper) // step + 1)
    out = np.empty((W, C, nper // 2 + 1), np.float32)
    for a in range(0, W, CHUNK):
        b = min(a + CHUNK, W)
        acc = np.zeros((b - a, C, nper // 2 + 1), np.float32)
        for s in range(nseg):
            seg = x[a:b, :, s * step : s * step + nper] * win
            F = np.fft.rfft(seg, axis=-1)
            acc += F.real ** 2 + F.imag ** 2
        out[a:b] = acc / nseg
    return np.fft.rfftfreq(nper, 1 / SR), out


def bandpower(x):
    """x [W, C, T] float32 -> log power in canonical bands [W, C*5]."""
    f, p = psd(x)
    feats = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        feats.append(np.log(p[..., m].mean(-1) + 1e-20))
    return np.stack(feats, -1).reshape(len(x), -1)


def logvar(x):
    out = np.empty(x.shape[:2], np.float32)
    for a in range(0, len(x), CHUNK):
        out[a : a + CHUNK] = np.log(x[a : a + CHUNK].var(-1) + 1e-12)
    return out


def alpha_occ(x, ch_names):
    idx = [ch_names.index(c) for c in OCCIPITAL if c in ch_names]
    f, p = psd(np.ascontiguousarray(x[:, idx]))
    a = np.log(p[..., (f >= 8) & (f < 12)].mean(-1) + 1e-20)
    b = np.log(p[..., (f >= 1) & (f < 40)].mean(-1) + 1e-20)
    return np.concatenate([a, b], -1)


def code_hist(codes, n_frames, vocab):
    """codes [W, C, Q, F] -> normalized histogram per (channel, book) [W, C*Q*vocab]."""
    W, C, Q, _ = codes.shape
    c = codes[..., :n_frames]
    out = np.zeros((W, C, Q, vocab), np.float32)
    for v in range(vocab):
        out[..., v] = (c == v).mean(-1)
    return out.reshape(W, -1)


def latent_stats(lat, n_frames):
    W = len(lat)
    out = np.empty((W, lat.shape[1] * lat.shape[2] * 2), np.float32)
    for a in range(0, W, CHUNK):
        l = lat[a : a + CHUNK, ..., :n_frames].astype(np.float32)
        out[a : a + CHUNK] = np.concatenate([l.mean(-1), l.std(-1)], -1).reshape(len(l), -1)
    return out


# ---------------- classifiers ----------------

def _fold(clf_fn, Xfit, Xtest, y, tr, te):
    """One CV fold. Xfit and Xtest differ only for the transfer test."""
    with threadpool_limits(limits=BLAS_THREADS):
        clf = clf_fn()
        clf.fit(Xfit[tr], y[tr])
        return (clf.predict(Xtest[te]) == y[te]).mean()


def _within_folds(y, groups, n_splits=4, min_trials=16):
    """(fit_idx, test_idx) pairs that stay inside one subject.

    Motor decoding does not transfer across subjects: CSP spatial filters are
    subject-specific, so cross-subject CSP sits near chance and leaves no headroom
    to detect codec degradation. Scoring within subject restores a usable baseline.
    The codec never saw any of this data, so within-subject CV leaks nothing about it.
    """
    folds = []
    for s in np.unique(groups):
        idx = np.flatnonzero(groups == s)
        if len(idx) < min_trials or len(np.unique(y[idx])) < 2:
            continue
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(idx, y[idx]):
            folds.append((idx[tr], idx[te]))
    return folds


def _cv(Xfit, Xtest, y, groups, clf_fn, within):
    n_groups = len(np.unique(groups))
    folds = (_within_folds(y, groups) if within
             else list(GroupKFold(min(5, n_groups)).split(Xfit, y, groups)))
    accs = Parallel(n_jobs=min(len(folds), N_JOBS))(
        delayed(_fold)(clf_fn, Xfit, Xtest, y, tr, te) for tr, te in folds)
    if within:  # average per subject first, then report spread across subjects
        per_sub = np.mean(np.asarray(accs).reshape(-1, 4), axis=1)
        return np.mean(per_sub), np.std(per_sub)
    return np.mean(accs), np.std(accs)


def cv_score(X, y, groups, clf_fn, within=False):
    return _cv(X, X, y, groups, clf_fn, within)


def cv_transfer(Xo, Xr, y, groups, clf_fn, within=False):
    return _cv(Xo, Xr, y, groups, clf_fn, within)


def csp_lda():
    """CSP(6)+LDA operating on PRECOMPUTED per-trial covariances (from mu_beta).

    Classic CSP recomputes trial covariances inside every fit; across folds x arms x
    subsets that dominated the whole eval. But CSP only ever uses the data through the
    covariances: class covariance = mean of trial covariances, and the log-variance
    feature of a spatial filter w on trial i is log(w' C_i w). So covariances are
    computed once per task and each fold reduces to a 64x64 generalized eigenproblem —
    same model, orders of magnitude faster.
    """
    class CovCspLda:
        n_components = 6
        shrink = 0.1  # shrinkage toward scaled identity (ledoit-wolf stand-in)

        def __init__(self):
            self.lda = LinearDiscriminantAnalysis()

        def _class_cov(self, covs):
            m = covs.mean(0)
            return (1 - self.shrink) * m + self.shrink * (np.trace(m) / m.shape[0]) * np.eye(m.shape[0])

        def fit(self, covs, y):
            c0, c1 = self._class_cov(covs[y == 0]), self._class_cov(covs[y == 1])
            w, v = scipy.linalg.eigh(c0, c0 + c1)
            order = np.argsort(np.abs(w - 0.5))[::-1]  # most discriminative eigenvectors
            self.filters = v[:, order[: self.n_components]].T  # [k, C]
            self.lda.fit(self._features(covs), y)
            return self

        def _features(self, covs):
            var = np.einsum("kc,ncd,kd->nk", self.filters, covs, self.filters)
            return np.log(np.maximum(var, 1e-20))

        def predict(self, covs):
            return self.lda.predict(self._features(covs))

    return CovCspLda()


def mu_beta(x):
    """Per-trial covariances of the 8-30 Hz band — the sufficient statistics for CSP.

    Bandpass, decimate (256 Hz is oversampled for a 30 Hz-limited signal), demean,
    covariance per trial. Chunked: filtfilt on the full array would allocate GBs.
    Output [W, C, C] feeds csp_lda directly.
    """
    b, a = butter(4, [8 / (SR / 2), 30 / (SR / 2)], "bandpass")
    W, C, _ = x.shape
    covs = np.empty((W, C, C), np.float64)
    for i in range(0, W, CHUNK):
        xf = np.ascontiguousarray(filtfilt(b, a, x[i : i + CHUNK], axis=-1)[..., ::DECIM],
                                  dtype=np.float64)
        xf -= xf.mean(-1, keepdims=True)
        covs[i : i + CHUNK] = np.einsum("wct,wdt->wcd", xf, xf) / (xf.shape[-1] - 1)
    return covs


# ---------------- per-task evaluation ----------------

def load_task(res_dir, task):
    """Read the standard DAC layout back into arrays cropped to the shortest window."""
    meta = pd.read_parquet(res_dir / f"{task}_windows.parquet")
    ch_names = json.loads((res_dir / f"{task}_ch_names.json").read_text())
    kwargs = json.loads((res_dir / "model_kwargs.json").read_text())
    hop = int(np.prod(kwargs["encoder_rates"]))
    lengths = (meta.duration_s * SR).round().astype(int).to_numpy()
    Tmin = int(lengths.min())
    W = len(meta)
    F = Tmin // hop  # frames fully covered by every window (per-window padding varies)
    # Fast path: consolidated arrays written by dump.py. Reading them back as three
    # sequential files takes seconds; walking the 3 x N per-window files off Lustre takes
    # tens of minutes, and collapses further when the box is busy.
    cache_dir = res_dir / f"{task}_arrays"
    if all((cache_dir / f"{k}.npy").exists() for k in ("orig", "recon", "codes", "gains")):
        def load_chunked(name, crop, dtype):
            """Stream a .npy sequentially in ~MB reads. Avoids BOTH failure modes here:
            np.load materialization = one giant allocation (kernel compaction stalls);
            mmap_mode = pagewise random reads (crawls on a loaded Lustre)."""
            import numpy.lib.format as fmt
            with open(cache_dir / f"{name}.npy", "rb") as fh:
                version = fmt.read_magic(fh)
                shape, fortran, dt = fmt._read_array_header(fh, version)
                assert not fortran
                out = np.empty(shape[:-1] + (crop,), dtype)  # lazy alloc; faulted per chunk
                buf = np.empty((CHUNK,) + shape[1:], dt)
                for a in range(0, shape[0], CHUNK):
                    n = min(CHUNK, shape[0] - a)
                    view = memoryview(buf).cast("B")[: n * buf[0].nbytes]
                    fh.readinto(view)
                    out[a : a + n] = buf[:n, ..., :crop]
            return out
        orig = load_chunked("orig", Tmin, np.float32)
        recon = load_chunked("recon", Tmin, np.float32)
        codes = load_chunked("codes", F, np.int64)
        gains = np.load(cache_dir / "gains.npy")[:, :, None].astype(np.float32)
        print(f"  loaded {task}: {len(meta)} windows from array cache")
        return meta, kwargs, ch_names, hop, Tmin, orig, recon, codes, gains

    # Legacy slow path: per-window files. Heavy imports deferred to here on purpose.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dac.model import DACFile
    first = DACFile.load(res_dir / "encoded" / f"{meta.stem.iloc[0]}.dac")
    C, Q, _ = first.codes.shape
    orig = np.empty((W, C, Tmin), np.float32)
    recon = np.empty((W, C, Tmin), np.float32)
    codes = np.empty((W, C, Q, F), np.int64)
    gains = np.ones((W, C, 1), np.float32)
    from tqdm import tqdm
    print(f"  no array cache for {task}; reading per-window files (slow) and building one")
    for i, stem in enumerate(tqdm(meta.stem, desc=f"load {task}", unit="win")):
        orig[i] = sf.read(res_dir / "input" / f"{stem}.wav", frames=Tmin, dtype="float32")[0].T
        recon[i] = sf.read(res_dir / "decoded" / f"{stem}.wav", frames=Tmin, dtype="float32")[0].T
        dac = DACFile.load(res_dir / "encoded" / f"{stem}.dac")
        codes[i] = dac.codes.numpy()[..., :F]
        # Per-channel gain removed by the codec's input normalization, kept as side
        # information (stock DAC's input_db). Restoring it gives CSP back the relative
        # channel amplitudes it decodes. Older dumps stored a scalar 0.0.
        gain = np.asarray(dac.input_db, dtype=np.float32).reshape(-1)
        if gain.size == C:
            gains[i] = (10.0 ** (gain / 20.0))[:, None]
    cache_dir.mkdir(parents=True, exist_ok=True)   # so the next run takes the fast path
    np.save(cache_dir / "orig.npy", orig.astype(np.float16))
    np.save(cache_dir / "recon.npy", recon.astype(np.float16))
    np.save(cache_dir / "codes.npy", codes.astype(np.int16))
    np.save(cache_dir / "gains.npy", gains[:, :, 0])
    return meta, kwargs, ch_names, hop, Tmin, orig, recon, codes, gains


def latents_from_codes(res_dir, codes, kwargs, n_frames):
    """Quantized low-dim latents via codebook lookup — what tokens alone carry."""
    nq = kwargs["n_codebooks"]
    if (res_dir / "codebooks.npz").exists():  # tiny file written by dump.py
        z = np.load(res_dir / "codebooks.npz")
        books = [z[str(q)] for q in range(nq)]
    else:  # older dumps: pull them out of the full snapshot (heavy import, legacy only)
        import torch
        sd = torch.load(res_dir / "weights_snapshot.pth", map_location="cpu",
                        weights_only=False)["state_dict"]
        books = [sd[f"quantizer.quantizers.{q}.codebook.weight"].numpy() for q in range(nq)]
    per_book = [b[codes[:, :, q, :n_frames]] for q, b in enumerate(books)]  # [W,C,F,8] each
    return np.concatenate(per_book, -1).transpose(0, 1, 3, 2)  # [W, C, Q*8, F]


def eval_subject_id(task, meta, orig, recon, codes, lat, Fmin, kwargs):
    """Who is this? 109-way subject identification (chance 0.9%).

    Split by RUN, never by window: every subject appears in train and test, but the test
    recording is one the classifier never saw, so it cannot win by memorising one
    recording's noise floor. Tests whether the codec keeps individual traits -- the
    privacy-relevant flip side of keeping task information.
    """
    runs = meta.run.to_numpy()
    if len(np.unique(runs)) < 2:
        return []
    subj = pd.factorize(meta.subject)[0]
    feats = {
        "orig_spec": bandpower(orig),
        "recon_spec": bandpower(recon),
        "codes": code_hist(codes, Fmin, kwargs["codebook_size"]),
        "latents": latent_stats(lat, Fmin),
        "orig_wave": logvar(orig),
        "recon_wave": logvar(recon),
    }
    uruns = np.unique(runs)
    folds = [(np.flatnonzero(runs != r), np.flatnonzero(runs == r)) for r in uruns]
    out = []
    for name, X in feats.items():
        accs = Parallel(n_jobs=min(len(folds), N_JOBS))(
            delayed(_fold)(logreg, X, X, subj, tr, te) for tr, te in folds)
        out.append((f"subject_id[{task}]", "all", name, float(np.mean(accs)), float(np.std(accs))))
    print(f"  subject_id[{task}] done ({len(np.unique(subj))}-way, leave-one-run-out, "
          f"chance {100/len(np.unique(subj)):.1f}%)")
    return out


import time as _time
_T0 = _time.time()


def _mark(msg):
    print(f"  [t+{_time.time() - _T0:7.1f}s] {msg}", flush=True)


def eval_task(res_dir, task):
    meta, kwargs, ch_names, hop, Tmin, orig_n, recon_n, codes, gains = load_task(res_dir, task)
    _mark(f"{task}: loaded")
    Fmin = Tmin // hop
    lat = latents_from_codes(res_dir, codes, kwargs, Fmin)
    _mark(f"{task}: latents")
    y = (meta.label == sorted(meta.label.unique())[1]).to_numpy().astype(int)
    groups = meta.subject.to_numpy()
    orig = np.empty_like(orig_n)
    recon = np.empty_like(recon_n)
    for a in range(0, len(orig_n), CHUNK):  # gain restored (primary), chunked multiply
        orig[a : a + CHUNK] = orig_n[a : a + CHUNK] * gains[a : a + CHUNK]
        recon[a : a + CHUNK] = recon_n[a : a + CHUNK] * gains[a : a + CHUNK]

    subsets = {"all": np.ones(len(meta), bool)}
    if task != "eyes":
        # real / imagined only: the pooled "all" mixes two different effect sizes and
        # costs another full set of CSP fits without telling us anything new.
        subsets = {
            "real": ~meta.imagined.to_numpy(),
            "imagined": meta.imagined.to_numpy(),
        }

    feats = {
        "orig_spec": bandpower(orig),
        "recon_spec": bandpower(recon),
        "codes": code_hist(codes, Fmin, kwargs["codebook_size"]),
        "latents": latent_stats(lat, Fmin),
    }
    if task == "eyes":
        feats["orig_wave"] = logvar(orig)
        feats["recon_wave"] = logvar(recon)
        feats["alpha_occ_orig"] = alpha_occ(orig, ch_names)
        feats["alpha_occ_recon"] = alpha_occ(recon, ch_names)
        feats["orig_wave_nogain"] = logvar(orig_n)
        feats["recon_wave_nogain"] = logvar(recon_n)
    else:
        wave_o, wave_r = mu_beta(orig), mu_beta(recon)
        wave_o_ng, wave_r_ng = mu_beta(orig_n), mu_beta(recon_n)  # gain NOT restored

    # Motor decoding is subject-specific (CSP filters do not transfer), so it is scored
    # within subject; eyes has a strong cross-subject effect and stays cross-subject.
    within = task != "eyes"
    _mark(f"{task}: features built")
    rows = []
    for sub, mask in subsets.items():
        g, yy = groups[mask], y[mask]
        for name, X in feats.items():
            m, s = cv_score(X[mask], yy, g, logreg, within)
            _mark(f"{task}/{sub}: {name}")
            rows.append((task, sub, name, m, s))
        if task == "eyes":
            m, s = cv_transfer(feats["orig_wave"][mask], feats["recon_wave"][mask], yy, g, logreg)
        else:
            for name, X in (("orig_wave", wave_o), ("recon_wave", wave_r),
                            ("orig_wave_nogain", wave_o_ng), ("recon_wave_nogain", wave_r_ng)):
                m, s = cv_score(X[mask], yy, g, csp_lda, within)
                rows.append((task, sub, name, m, s))
            m, s = cv_transfer(wave_o[mask], wave_r[mask], yy, g, csp_lda, within)
        rows.append((task, sub, "transfer_orig->recon", m, s))
        m, s = cv_transfer(feats["orig_spec"][mask], feats["recon_spec"][mask], yy, g, logreg, within)
        rows.append((task, sub, "transfer_spec", m, s))
        cv = "within-subject" if within else "cross-subject"
        print(f"  {task}/{sub} done ({mask.sum()} windows, {len(np.unique(g))} subjects, {cv} CV)")

    if task == "eyes":  # 2 runs -> clean leave-one-run-out; motor tasks would be 109-way
        _mark(f"{task}: subject_id start")
        rows += eval_subject_id(task, meta, orig, recon, codes, lat, Fmin, kwargs)

    # band-power preservation
    _mark(f"{task}: band-psd start")
    band_rows = []
    f, po = psd(orig)
    _, pr = psd(recon)
    _mark(f"{task}: band-psd done")
    for bname, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        o, r = po[..., m].mean(-1), pr[..., m].mean(-1)
        rel_err = np.median(np.abs(r - o) / (o + 1e-20))
        corr = np.corrcoef(np.log(o + 1e-20).ravel(), np.log(r + 1e-20).ravel())[0, 1]
        band_rows.append((task, bname, rel_err, corr))
    return rows, band_rows


CHART_ROWS = [  # (representation, label, color) — cyan = original arm, yellow = codec arm
    ("orig_wave", "input", "\033[36m"),
    ("orig_spec", "input spectrum", "\033[36m"),
    ("orig_wave_nogain", "input (no gain)", "\033[36m"),
    ("codes", "codes", "\033[33m"),
    ("latents", "tokens", "\033[33m"),
    ("recon_wave", "recon", "\033[33m"),
    ("recon_spec", "recon spectrum", "\033[33m"),
    ("recon_wave_nogain", "recon (no gain)", "\033[33m"),
]


def terminal_chart(acc, width=50, color=True, echo=True):
    """Unicode bar chart: accuracy per task, one bar per representation. 100% = full width.
    Returns the rendered text; prints it when echo. color=False for a plain file copy."""
    C = {"cyan": "\033[36m", "yellow": "\033[33m", "reset": "\033[0m", "dim": "\033[2m"}
    if not color:
        C = {k: "" for k in C}
    lines = ["== Accuracy per task (GroupKFold over subjects) =="]
    for (task, subset), g in acc.groupby(["task", "subset"], sort=False):
        vals = {r["representation"]: 100 * r["acc"] for _, r in g.iterrows()}
        lines.append(f"\n{task} / {subset}")
        for rep, label, code in CHART_ROWS:
            if rep not in vals:
                continue
            n = int(round(vals[rep] / 100 * width))
            col = C["cyan"] if code == "\033[36m" else C["yellow"]
            lines.append(f"  {label:15s} {col}{'█' * n}{C['reset']} {vals[rep]:.1f}")
        ch = 100.0 / 109 if str(task).startswith("subject_id") else 50.0
        lines.append(f"  {'':15s} {C['dim']}{'─' * max(1, int(round(ch / 100 * width)))}"
                     f"┤ {ch:.1f}% chance{C['reset']}")
    text = "\n".join(lines)
    if echo:
        print(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", default="eyes,fist_lr,fists_feet")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if accuracy.csv already exists")
    args = ap.parse_args()
    res_dir = Path(args.results)

    acc_csv, bands_csv = res_dir / "accuracy.csv", res_dir / "bands.csv"
    if acc_csv.exists() and not args.force:
        print(f"cached results found in {res_dir} — reprinting (use --force to recompute)")
        acc = pd.read_csv(acc_csv)
    else:
        all_rows, all_bands = [], []
        for task in args.tasks.split(","):
            if not (res_dir / f"{task}_windows.parquet").exists():
                print(f"skipping {task}: no dump")
                continue
            r, b = eval_task(res_dir, task.strip())
            all_rows += r
            all_bands += b
        acc = pd.DataFrame(all_rows, columns=["task", "subset", "representation", "acc", "std"])
        bands = pd.DataFrame(all_bands, columns=["task", "band", "median_rel_err", "log_power_corr"])
        acc.to_csv(acc_csv, index=False)
        bands.to_csv(bands_csv, index=False)

    terminal_chart(acc)
    (res_dir / "summary.txt").write_text(terminal_chart(acc, color=False, echo=False) + "\n")
    print(f"\nsaved: {res_dir}/summary.txt (chart), accuracy.csv, bands.csv, plots/")


if __name__ == "__main__":
    main()
