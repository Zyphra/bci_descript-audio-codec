"""Classical MDCT transform coder as a baseline for the DAC-EEG codec.

Pipeline (textbook, no learning):
  1. MDCT with a sine (Princen-Bradley) window, N coefficients per N new samples,
     50 % overlap, time-domain aliasing cancellation on inverse -> perfect
     reconstruction when nothing is quantized.
  2. Global reverse water-filling bit allocation: the variance of each MDCT
     coefficient index k is measured over the whole bank (all windows, all frames).
     Coefficients with variance > theta get a uniform scalar quantizer with step
     sqrt(12 theta) (error variance theta); the rest are zeroed. One theta for
     all coefficients = equal error in every coded bin, the water-filling condition.
  3. Rate = empirical entropy of the quantized integer symbols per coefficient
     index, summed, times frames per second (rate/N). This is what an ideal
     adaptive arithmetic coder would spend; no entropy coder is written.
  4. theta is bisected until the rate hits each --targets bits/s.

Outputs, per target rate, a reconstruction .npz that scripts/evaluate_eeg.py accepts
in place of a checkpoint (key 'reconstruction' plus 'input_sha256' for safety), and a
sweep.csv with theta, rate, MSE and the per-coefficient entropies.

Usage:
  python scripts/mdct_coder.py --inputs BANK.npz --output DIR --N 128 --targets 50 102 204 405

Limitations (deliberate, so this is the simplest honest baseline):
  * allocation is global, not per window: no side information, no adaptation to the
    1.2-2.3x per-window spectral variation. A per-window scale-factor variant would do
    better and cost side-info bits.
  * plain rounding, no dead zone, no noise filling.
  * window edges: the bank windows are 5 s; N zeros of padding on each side, frames
    are (T/N + 1) per window; rate is quoted steady-state as rate/N frames per second.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "scripts"))
from evaluate_eeg import fingerprint


# ----------------------------------------------------------------------------
# MDCT
# ----------------------------------------------------------------------------
class MDCT:
    def __init__(self, N):
        self.N = N
        n = np.arange(2 * N)
        k = np.arange(N)
        self.window = np.sin(np.pi * (n + 0.5) / (2 * N))                      # Princen-Bradley
        self.C = np.cos(np.pi / N * (n[None, :] + 0.5 + N / 2) * (k[:, None] + 0.5))  # N x 2N

    def forward(self, x):
        """x: [B, T] with T a multiple of N -> coefficients [B, F, N], F = T/N + 1."""
        B, T = x.shape; N = self.N
        xp = np.pad(x, ((0, 0), (N, N)))
        F = T // N + 1
        blocks = np.stack([xp[:, f * N:f * N + 2 * N] for f in range(F)], axis=1)  # B x F x 2N
        return (blocks * self.window) @ self.C.T                                   # B x F x N

    def inverse(self, X, T):
        B, F, N = X.shape
        y = np.zeros((B, T + 2 * N))
        blocks = (X @ self.C) * self.window * (2.0 / N)                            # B x F x 2N
        for f in range(F):
            y[:, f * N:f * N + 2 * N] += blocks[:, f]
        return y[:, N:N + T]


# ----------------------------------------------------------------------------
# Quantize + rate
# ----------------------------------------------------------------------------
def entropy_bits(q):
    _, cnt = np.unique(q, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def code(X, var, theta):
    """Global water-filling: returns (dequantized X, rate bits/frame, per-k entropies)."""
    coded = var > theta
    step = np.sqrt(12 * theta)
    q = np.where(coded, np.round(X / step), 0.0)
    Xh = q * step
    H = np.array([entropy_bits(q[:, :, k].ravel()) if coded[k] else 0.0 for k in range(X.shape[-1])])
    return Xh, float(H.sum()), H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True); p.add_argument("--output", required=True)
    p.add_argument("--N", type=int, default=128, help="MDCT coefficients per frame (frame hop in samples)")
    p.add_argument("--targets", type=float, nargs="+", default=[50, 102, 204, 405], help="target bits/s")
    p.add_argument("--rate", type=int, default=256)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    with np.load(a.inputs, allow_pickle=False) as z:
        x = np.asarray(z["audio"], dtype=np.float64)
    sha = fingerprint(x.astype(np.float32))
    B, T = x.shape
    assert T % a.N == 0, "window length must be a multiple of N"
    mdct = MDCT(a.N)
    X = mdct.forward(x)
    err = np.abs(mdct.inverse(X, T) - x).max()
    assert err < 1e-9, f"MDCT not perfectly reconstructing: max err {err}"
    var = X.reshape(-1, a.N).var(axis=0)
    frames_per_s = a.rate / a.N
    print(f"N={a.N}: {X.shape[1]} frames/window, {frames_per_s:g} frames/s steady state, "
          f"coefficient variance range {var.min():.2e}..{var.max():.2e}", flush=True)

    def rate_at(theta):
        Xh, bits, H = code(X, var, theta)
        return frames_per_s * bits, Xh, H

    rows = []
    for target in a.targets:
        lo, hi = np.log(var.min()) - 10, np.log(var.max()) + 1      # rate decreases with theta
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            R, _, _ = rate_at(np.exp(mid))
            if R > target: lo = mid
            else: hi = mid
        theta = np.exp(0.5 * (lo + hi))
        R, Xh, H = rate_at(theta)
        y = mdct.inverse(Xh, T)
        mse = float(((y - x) ** 2).mean())
        coded = int((var > theta).sum())
        label = f"mdct{a.N}_{int(round(target))}bps"
        np.savez_compressed(out / f"{label}.npz", reconstruction=y.astype(np.float32), input_sha256=sha,
                            bits_per_s=R, theta=theta, N=a.N, coefficient_entropy=H)
        rows.append(dict(label=label, N=a.N, target_bits_per_s=target, bits_per_s=R, theta=theta,
                         coded_coefficients=coded, of=a.N, mse=mse, snr_db=10 * np.log10(x.var() / mse)))
        print(f"{label}: rate {R:.1f} bits/s, theta {theta:.3e}, {coded}/{a.N} coefficients coded "
              f"(up to {coded * a.rate / (2 * a.N):.0f} Hz), MSE {mse:.5f}, SNR {rows[-1]['snr_db']:.1f} dB", flush=True)
    with open(out / f"sweep_N{a.N}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
