"""Cross-channel redundancy of code streams on a multi-channel frozen bank.

Question: at the same instant t, how much does the code on electrode A tell you about
the code on electrode B?  This is the redundancy a multi-channel codec could remove and
an independent-per-channel codec (the current DAC setup) cannot.

Input: multichannel.npz from scripts/make_multichannel_bank.py (one row per
(window, channel), with window_id, pos, is_source).

Coders:
  DAC checkpoints  RUN:LABEL specs, as in scripts/code_entropy.py. Codes per codebook i.
  MDCT             --mdct N:RATE specs, e.g. 128:102. scripts/mdct_coder.py's global
                   water-filling quantizer tuned to RATE bits/s on this bank; the
                   "codes" are the quantized integers of each coded coefficient k.

For each pair set P (ordered pairs (A,B) of channels within one window, frames aligned):
  H_B        marginal entropy of B's code (per codebook / coefficient), bits/frame
  Hc(B|A)    H(B,A) - H(A) on pairs
  Hc_shuf    the same with B's codes permuted over all (window, frame) positions:
             same marginals, true MI zero -> finite-sample bias
  MI         Hc_shuf - Hc  (bias-corrected; a LOWER bound when the table is sparse)
  redundancy sum MI / sum H_B
Pair sets:
  nearest   each channel with its nearest neighbour by electrode position
  random    each channel with --partners random other channels in the window
  farthest  each channel with its farthest electrode
  temporal  (A,B) = (same channel t-1, t): the Result-1 measure on this bank, for reference
Output: one CSV row per coder x pair set.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="*", help="DAC RUN:LABEL specs")
p.add_argument("--mdct", nargs="*", default=[], help="MDCT N:RATE specs, e.g. 128:102")
p.add_argument("--inputs", required=True); p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--trim", type=int, default=128); p.add_argument("--partners", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
rng = np.random.default_rng(a.seed)


def entropy_bits(ids):
    _, cnt = np.unique(ids, return_counts=True); pr = cnt / cnt.sum()
    return float(-(pr * np.log2(pr)).sum())


def cond_entropy(cond, target, K):
    return entropy_bits(cond.astype(np.int64) * K + target) - entropy_bits(cond)


# ---------------------------------------------------------------- bank + pair sets
with np.load(a.inputs, allow_pickle=False) as z:
    x = z["audio"].astype(np.float32); wid = z["window_id"]; pos = z["pos"]
rate = 256
rows_of = {}
for r, w in enumerate(wid): rows_of.setdefault(int(w), []).append(r)
pair_sets = {"nearest": [], "random": [], "farthest": []}
for w, rows in rows_of.items():
    rows = np.array(rows); P = pos[rows]
    d = np.linalg.norm(P[:, None] - P[None], axis=-1); np.fill_diagonal(d, np.inf)
    has_pos = np.isfinite(d).any() and (P != 0).any()
    for i, r in enumerate(rows):
        others = np.delete(np.arange(len(rows)), i)
        if has_pos:
            pair_sets["nearest"].append((r, rows[np.argmin(d[i])]))
            dd = d[i].copy(); dd[i] = -1; pair_sets["farthest"].append((r, rows[np.argmax(dd)]))
        for j in rng.choice(others, size=min(a.partners, len(others)), replace=False):
            pair_sets["random"].append((r, rows[j]))
pair_sets = {k: np.array(v) for k, v in pair_sets.items() if v}
print({k: len(v) for k, v in pair_sets.items()}, "channel pairs;", len(x), "channel-windows", flush=True)


def measure(codes, K, label, extra):
    """codes: [n_rows, n_streams, T] ints < K (streams = codebooks or MDCT coefficients)."""
    n_rows, S, T = codes.shape
    out = []
    H = [entropy_bits(codes[:, s].ravel()) for s in range(S)]
    for name, pairs in list(pair_sets.items()) + [("temporal", None)]:
        MI, Hc_l = [], []
        for s in range(S):
            if H[s] == 0: MI.append(0.0); Hc_l.append(0.0); continue
            if name == "temporal":
                A, B = codes[:, s, :-1].ravel(), codes[:, s, 1:].ravel()
            else:
                A, B = codes[pairs[:, 0], s].ravel(), codes[pairs[:, 1], s].ravel()
            Hc = cond_entropy(A, B, K)
            Hc_s = cond_entropy(A, B[rng.permutation(len(B))], K)
            MI.append(Hc_s - Hc); Hc_l.append(Hc)
        out.append(dict(label=label, pair_set=name, pairs=len(A), streams=S, H_sum=sum(H), Hc_sum=sum(Hc_l),
                        MI_sum=sum(MI), redundancy=sum(MI) / sum(H), **extra))
        print(f"  {label:14s} {name:9s} pairs={len(A):>9,d}  H={sum(H):6.2f}  MI={sum(MI):5.3f} bits/frame  "
              f"redundancy={100 * sum(MI) / sum(H):5.1f}%", flush=True)
    return out


rows = []
# ---------------------------------------------------------------- DAC
if a.runs:
    from dac.model import DAC
    from evaluate_eeg import resolve_checkpoint
    for spec in a.runs:
        run, label = spec.split(":") if ":" in spec else (spec, Path(spec).name)
        model = DAC.load(str(resolve_checkpoint(run)), strict=True).to(a.device).eval()
        hop = int(np.prod(model.encoder_rates)); K = model.codebook_size
        codes = []
        with torch.inference_mode():
            for s in range(0, len(x), a.batch_size):
                b = torch.from_numpy(x[s:s + a.batch_size, None]).to(a.device)
                _, c, _, _, _ = model.encode(model.preprocess(b, rate)); codes.append(c.cpu().numpy())
        codes = np.concatenate(codes)[:, :, a.trim // hop: -(a.trim // hop)].astype(np.int64)
        rows += measure(codes, K, label, dict(coder="dac", hop=hop, K=K, bits_per_s=rate / hop * np.log2(K) * codes.shape[1]))
        del model; torch.cuda.empty_cache()
# ---------------------------------------------------------------- MDCT
if a.mdct:
    from mdct_coder import MDCT, code
    for spec in a.mdct:
        N, target = int(spec.split(":")[0]), float(spec.split(":")[1])
        m = MDCT(N); X = m.forward(x.astype(np.float64)); var = X.reshape(-1, N).var(0); fps = rate / N
        lo, hi = np.log(var.min()) - 10, np.log(var.max()) + 1
        for _ in range(50):
            mid = 0.5 * (lo + hi); R = fps * code(X, var, np.exp(mid))[1]
            lo, hi = (mid, hi) if R > target else (lo, mid)
        theta = np.exp(0.5 * (lo + hi)); _, bits, _ = code(X, var, theta)
        coded = np.where(var > theta)[0]; step = np.sqrt(12 * theta)
        q = np.round(X[:, :, coded] / step).astype(np.int64)                 # rows x frames x coded coeffs
        q = q - q.min(); K = int(q.max()) + 1
        q = np.transpose(q, (0, 2, 1))[:, :, max(1, a.trim // N): -max(1, a.trim // N)]   # rows x coeffs x frames
        label = f"mdct{N}_{int(target)}"
        print(f"{label}: theta={theta:.3e} {len(coded)}/{N} coefficients coded, rate {fps * bits:.1f} bits/s, alphabet {K}", flush=True)
        rows += measure(q, K, label, dict(coder="mdct", hop=N, K=K, bits_per_s=fps * bits))

keys = ["coder", "label", "pair_set", "hop", "K", "bits_per_s", "pairs", "streams", "H_sum", "Hc_sum", "MI_sum", "redundancy"]
with open(a.output, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("wrote", a.output)
